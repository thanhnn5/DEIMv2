"""
DEIMv2-L single-file reimplementation (inference-only).

Changes from original engine/ source:
  1. SelfAttention.compute_attention — 5D reshape replaced with 4D ops
     (B,N,3,heads,head_dim) -> (B,N,3,C) then slice, then per-tensor reshape
     Numerical output is identical; eliminates RESHAPE/5D GPU delegate errors.
  2. SyncBatchNorm -> BatchNorm2d in SpatialPriorModulev2
     Identical in eval mode; SyncBN requires distributed setup not present on device.
  3. No imports from engine.* — fully standalone.

Usage:
    python deimv2_single.py \
        -r weights/deimv2_dinov3_l_coco_custom_best_stg2.pth \
        --num-classes 2
"""

import argparse
import copy
import functools
import math
from collections import OrderedDict
from functools import partial
from typing import Callable, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


# =============================================================================
# Utility helpers
# =============================================================================


def _with_pos_embed(tensor, pos_embed):
    return tensor if pos_embed is None else tensor + pos_embed


def box_xyxy_to_cxcywh(x: torch.Tensor) -> torch.Tensor:
    x0, y0, x1, y1 = x.unbind(-1)
    return torch.stack([(x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)], dim=-1)


def inverse_sigmoid(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    x = x.clip(min=0.0, max=1.0)
    return torch.log(x.clip(min=eps) / (1 - x).clip(min=eps))


def bias_init_with_prob(prior_prob: float = 0.01) -> float:
    return float(-math.log((1 - prior_prob) / prior_prob))


def get_activation(act: str, inplace: bool = True) -> nn.Module:
    if act is None:
        return nn.Identity()
    act = act.lower()
    if act in ("silu", "swish"):
        m = nn.SiLU()
    elif act == "relu":
        m = nn.ReLU()
    elif act == "leaky_relu":
        m = nn.LeakyReLU()
    elif act == "gelu":
        m = nn.GELU()
    elif act == "hardsigmoid":
        m = nn.Hardsigmoid()
    else:
        raise RuntimeError(f"Unknown activation: {act}")
    if hasattr(m, "inplace"):
        m.inplace = inplace
    return m


# --- DINOv3 tensor helpers ---
def cat_keep_shapes(x_list: List[torch.Tensor]):
    shapes = [x.shape for x in x_list]
    num_tokens = [x.select(dim=-1, index=0).numel() for x in x_list]
    flattened = torch.cat([x.flatten(0, -2) for x in x_list])
    return flattened, shapes, num_tokens


def uncat_with_shapes(flattened, shapes, num_tokens):
    splits = torch.split_with_sizes(flattened, num_tokens, dim=0)
    adjusted = [s[:-1] + torch.Size([flattened.shape[-1]]) for s in shapes]
    return [o.reshape(sh) for o, sh in zip(splits, adjusted)]


def named_apply(fn, module, name="", depth_first=True, include_root=False):
    if not depth_first and include_root:
        fn(module=module, name=name)
    for child_name, child in module.named_children():
        child_name = f"{name}.{child_name}" if name else child_name
        named_apply(fn, child, child_name, depth_first, include_root=True)
    if depth_first and include_root:
        fn(module=module, name=name)
    return module


# =============================================================================
# DINOv3 backbone layers
# =============================================================================


class RMSNormDino(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def reset_parameters(self):
        nn.init.constant_(self.weight, 1)

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self._norm(x.float()).type_as(x) * self.weight


class LayerScale(nn.Module):
    def __init__(
        self, dim: int, init_values: float = 1e-5, inplace: bool = False, device=None
    ):
        super().__init__()
        self.inplace = inplace
        self.init_values = init_values
        self.gamma = nn.Parameter(torch.empty(dim, device=device))

    def reset_parameters(self):
        nn.init.constant_(self.gamma, self.init_values)

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class PatchEmbed(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        norm_layer=None,
        flatten_embedding=True,
    ):
        super().__init__()
        ih, iw = (img_size, img_size) if isinstance(img_size, int) else img_size
        ph, pw = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        self.img_size = (ih, iw)
        self.patch_size = (ph, pw)
        self.patches_resolution = (ih // ph, iw // pw)
        self.num_patches = self.patches_resolution[0] * self.patches_resolution[1]
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.flatten_embedding = flatten_embedding
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=(ph, pw), stride=(ph, pw)
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        H, W = x.size(2), x.size(3)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        if not self.flatten_embedding:
            x = x.reshape(-1, H, W, self.embed_dim)
        return x

    def reset_parameters(self):
        k = 1 / (self.in_chans * (self.patch_size[0] ** 2))
        nn.init.uniform_(self.proj.weight, -math.sqrt(k), math.sqrt(k))
        if self.proj.bias is not None:
            nn.init.uniform_(self.proj.bias, -math.sqrt(k), math.sqrt(k))


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
        bias=True,
        device=None,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias, device=device)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias, device=device)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

    def forward_list(self, x_list):
        x_flat, shapes, num_tokens = cat_keep_shapes(x_list)
        x_flat = self.forward(x_flat)
        return uncat_with_shapes(x_flat, shapes, num_tokens)


class SwiGLUFFN(nn.Module):
    """Decoder SwiGLU FFN — matches deim_utils.SwiGLUFFN exactly.
    Uses combined w12 projection (w1+w2 fused) and w3 output projection.
    Called as SwiGLUFFN(d_model, dim_feedforward // 2, d_model).
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        bias: bool = True,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.w12 = nn.Linear(in_features, 2 * hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)

    def forward(self, x):
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        return self.w3(F.silu(x1) * x2)

    def forward_list(self, x_list):
        x_flat, shapes, num_tokens = cat_keep_shapes(x_list)
        x_flat = self.forward(x_flat)
        return uncat_with_shapes(x_flat, shapes, num_tokens)


class RopePositionEmbedding(nn.Module):
    def __init__(
        self,
        embed_dim,
        *,
        num_heads,
        base=100.0,
        min_period=None,
        max_period=None,
        normalize_coords="separate",
        shift_coords=None,
        jitter_coords=None,
        rescale_coords=None,
        dtype=None,
        device=None,
    ):
        super().__init__()
        D_head = embed_dim // num_heads
        self.base = base
        self.min_period = min_period
        self.max_period = max_period
        self.D_head = D_head
        self.normalize_coords = normalize_coords
        self.shift_coords = shift_coords
        self.jitter_coords = jitter_coords
        self.rescale_coords = rescale_coords
        self.dtype = dtype
        self.register_buffer(
            "periods",
            torch.empty(D_head // 4, device=device, dtype=dtype),
            persistent=True,
        )
        self._init_weights()

    def _init_weights(self):
        device = self.periods.device
        dtype = self.dtype
        if self.base is not None:
            periods = self.base ** (
                2
                * torch.arange(self.D_head // 4, device=device, dtype=dtype)
                / (self.D_head // 2)
            )
        else:
            base = self.max_period / self.min_period
            exponents = torch.linspace(
                0, 1, self.D_head // 4, device=device, dtype=dtype
            )
            periods = base**exponents / base * self.max_period
        self.periods.data = periods

    def forward(self, *, H: int, W: int):
        device = self.periods.device
        dtype = self.dtype
        dd = {"device": device, "dtype": dtype}
        if self.normalize_coords == "separate":
            coords_h = torch.arange(0.5, H, **dd) / H
            coords_w = torch.arange(0.5, W, **dd) / W
        elif self.normalize_coords == "max":
            m = max(H, W)
            coords_h = torch.arange(0.5, H, **dd) / m
            coords_w = torch.arange(0.5, W, **dd) / m
        else:
            m = min(H, W)
            coords_h = torch.arange(0.5, H, **dd) / m
            coords_w = torch.arange(0.5, W, **dd) / m
        coords = torch.stack(
            torch.meshgrid(coords_h, coords_w, indexing="ij"), dim=-1
        ).flatten(0, 1)
        coords = 2.0 * coords - 1.0
        angles = 2 * math.pi * coords[:, :, None] / self.periods[None, None, :]
        angles = angles.flatten(1, 2).tile(2)
        return torch.sin(angles), torch.cos(angles)


class LinearKMaskedBias(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        o = self.out_features
        assert o % 3 == 0
        if self.bias is not None:
            self.register_buffer(
                "bias_mask", torch.full_like(self.bias, fill_value=math.nan)
            )

    def forward(self, input):
        masked_bias = (
            self.bias * self.bias_mask.to(self.bias.dtype)
            if self.bias is not None
            else None
        )
        return F.linear(input, self.weight, masked_bias)


class SelfAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        proj_bias=True,
        attn_drop=0.0,
        proj_drop=0.0,
        mask_k_bias=False,
        device=None,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        linear_class = LinearKMaskedBias if mask_k_bias else nn.Linear
        self.qkv = linear_class(dim, dim * 3, bias=qkv_bias, device=device)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias, device=device)
        self.proj_drop = nn.Dropout(proj_drop)

    def _apply_rope(self, q, k, rope):
        q_dtype, k_dtype = q.dtype, k.dtype
        sin, cos = rope
        q, k = q.to(sin.dtype), k.to(sin.dtype)
        prefix = q.shape[-2] - sin.shape[-2]

        def rope_apply(x):
            x1, x2 = x.chunk(2, dim=-1)
            return (x * cos) + (torch.cat([-x2, x1], dim=-1) * sin)

        q = torch.cat((q[:, :, :prefix, :], rope_apply(q[:, :, prefix:, :])), dim=-2)
        k = torch.cat((k[:, :, :prefix, :], rope_apply(k[:, :, prefix:, :])), dim=-2)
        return q.to(q_dtype), k.to(k_dtype)

    def compute_attention(self, qkv, rope=None):
        B, N, _ = qkv.shape
        C = self.qkv.in_features
        head_dim = C // self.num_heads

        # FIX: avoid 5D reshape — stay ≤4D throughout
        # Original was: qkv.reshape(B, N, 3, self.num_heads, head_dim)  ← 5D, illegal on GPU delegate
        qkv3 = qkv.reshape(B, N, 3, C)  # (B, N, 3, C) — 4D
        q = (
            qkv3[:, :, 0, :].reshape(B, N, self.num_heads, head_dim).transpose(1, 2)
        )  # (B, H, N, D)
        k = qkv3[:, :, 1, :].reshape(B, N, self.num_heads, head_dim).transpose(1, 2)
        v = qkv3[:, :, 2, :].reshape(B, N, self.num_heads, head_dim).transpose(1, 2)

        if rope is not None:
            q, k = self._apply_rope(q, k, rope)
        x = F.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2).reshape(B, N, C)
        return x

    def forward(self, x, rope=None):
        qkv = self.qkv(x)
        attn_v = self.compute_attention(qkv=qkv, rope=rope)
        x = self.proj(attn_v)
        return self.proj_drop(x)

    def forward_list(self, x_list, rope_list=None):
        assert len(x_list) == len(rope_list)
        x_flat, shapes, num_tokens = cat_keep_shapes(x_list)
        qkv_flat = self.qkv(x_flat)
        qkv_list = uncat_with_shapes(qkv_flat, shapes, num_tokens)
        att_out = [
            self.compute_attention(qkv, rope=rope)
            for qkv, rope in zip(qkv_list, rope_list)
        ]
        x_flat, shapes, num_tokens = cat_keep_shapes(att_out)
        x_flat = self.proj(x_flat)
        return uncat_with_shapes(x_flat, shapes, num_tokens)


class SelfAttentionBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        ffn_ratio=4.0,
        qkv_bias=False,
        proj_bias=True,
        ffn_bias=True,
        drop=0.0,
        attn_drop=0.0,
        init_values=None,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        attn_class=SelfAttention,
        ffn_layer=Mlp,
        mask_k_bias=False,
        device=None,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = attn_class(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            mask_k_bias=mask_k_bias,
            device=device,
        )
        self.ls1 = (
            LayerScale(dim, init_values=init_values, device=device)
            if init_values
            else nn.Identity()
        )
        self.norm2 = norm_layer(dim)
        self.mlp = ffn_layer(
            in_features=dim,
            hidden_features=int(dim * ffn_ratio),
            act_layer=act_layer,
            drop=drop,
            bias=ffn_bias,
            device=device,
        )
        self.ls2 = (
            LayerScale(dim, init_values=init_values, device=device)
            if init_values
            else nn.Identity()
        )
        self.sample_drop_ratio = drop_path

    @staticmethod
    def _maybe_index_rope(rope, indices):
        if rope is None:
            return None
        sin, cos = rope
        if sin.ndim == 4:
            return sin[indices], cos[indices]
        return sin, cos

    def _forward_list(self, x_list, rope_list=None):
        if self.training and self.sample_drop_ratio > 0.0:
            # stochastic depth path — inference only, skip for simplicity
            raise RuntimeError("stochastic depth only during training")
        x_out = []
        for x, rope in zip(x_list, rope_list):
            x_attn = x + self.ls1(self.attn(self.norm1(x), rope=rope))
            x_out.append(x_attn + self.ls2(self.mlp(self.norm2(x_attn))))
        return x_out

    def forward(self, x_or_list, rope_or_list=None):
        if isinstance(x_or_list, torch.Tensor):
            return self._forward_list([x_or_list], rope_list=[rope_or_list])[0]
        if rope_or_list is None:
            rope_or_list = [None] * len(x_or_list)
        return self._forward_list(x_or_list, rope_list=rope_or_list)


# DINOv3 ViT configs (vits16 used by DEIMv2-L)
_DINOV3_CONFIGS = {
    "dinov3_vits16": dict(
        img_size=224,
        patch_size=16,
        in_chans=3,
        pos_embed_rope_base=100,
        pos_embed_rope_normalize_coords="separate",
        pos_embed_rope_rescale_coords=2,
        pos_embed_rope_dtype="fp32",
        embed_dim=384,
        depth=12,
        num_heads=6,
        ffn_ratio=4,
        qkv_bias=True,
        drop_path_rate=0.0,
        layerscale_init=1e-5,
        norm_layer="layernormbf16",
        ffn_layer="mlp",
        ffn_bias=True,
        proj_bias=True,
        n_storage_tokens=4,
        mask_k_bias=True,
    ),
}
_NORM_LAYERS = {
    "layernorm": partial(nn.LayerNorm, eps=1e-6),
    "layernormbf16": partial(nn.LayerNorm, eps=1e-5),
    "rmsnorm": RMSNormDino,
}
_FFN_LAYERS = {
    "mlp": Mlp,
    "swiglu": SwiGLUFFN,
}
_DTYPE_MAP = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


def _init_weights_vit(module, name=""):
    if isinstance(module, nn.Linear):
        torch.nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    if isinstance(module, nn.LayerNorm):
        module.reset_parameters()
    if isinstance(module, LayerScale):
        module.reset_parameters()
    if isinstance(module, PatchEmbed):
        module.reset_parameters()
    if isinstance(module, RMSNormDino):
        module.reset_parameters()


class DinoVisionTransformer(nn.Module):
    def __init__(self, name):
        super().__init__()
        cfg = _DINOV3_CONFIGS[name]
        embed_dim = cfg["embed_dim"]
        depth = cfg["depth"]
        num_heads = cfg["num_heads"]
        norm_layer_cls = _NORM_LAYERS[cfg["norm_layer"]]
        ffn_layer_cls = _FFN_LAYERS[cfg["ffn_layer"]]

        self.num_features = self.embed_dim = embed_dim
        self.n_blocks = depth
        self.num_heads = num_heads
        self.patch_size = cfg["patch_size"]

        self.patch_embed = PatchEmbed(
            img_size=cfg["img_size"],
            patch_size=cfg["patch_size"],
            in_chans=cfg["in_chans"],
            embed_dim=embed_dim,
            flatten_embedding=False,
        )
        self.cls_token = nn.Parameter(torch.empty(1, 1, embed_dim))
        self.n_storage_tokens = cfg["n_storage_tokens"]
        if self.n_storage_tokens > 0:
            self.storage_tokens = nn.Parameter(
                torch.empty(1, self.n_storage_tokens, embed_dim)
            )
        self.rope_embed = RopePositionEmbedding(
            embed_dim=embed_dim,
            num_heads=num_heads,
            base=cfg["pos_embed_rope_base"],
            normalize_coords=cfg["pos_embed_rope_normalize_coords"],
            rescale_coords=cfg["pos_embed_rope_rescale_coords"],
            dtype=_DTYPE_MAP[cfg["pos_embed_rope_dtype"]],
        )
        self.blocks = nn.ModuleList(
            [
                SelfAttentionBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    ffn_ratio=cfg["ffn_ratio"],
                    qkv_bias=cfg["qkv_bias"],
                    proj_bias=cfg["proj_bias"],
                    ffn_bias=cfg["ffn_bias"],
                    norm_layer=norm_layer_cls,
                    act_layer=nn.GELU,
                    ffn_layer=ffn_layer_cls,
                    init_values=cfg["layerscale_init"],
                    mask_k_bias=cfg["mask_k_bias"],
                )
                for _ in range(depth)
            ]
        )
        self.norm = norm_layer_cls(embed_dim)
        self.mask_token = nn.Parameter(torch.empty(1, embed_dim))
        self.init_weights()

    def init_weights(self):
        self.rope_embed._init_weights()
        nn.init.normal_(self.cls_token, std=0.02)
        if self.n_storage_tokens > 0:
            nn.init.normal_(self.storage_tokens, std=0.02)
        nn.init.zeros_(self.mask_token)
        named_apply(_init_weights_vit, self)

    def _prepare_tokens(self, x):
        x = self.patch_embed(x)
        B, H, W, _ = x.shape
        x = x.flatten(1, 2)
        cls_token = self.cls_token + 0 * self.mask_token
        storage = (
            self.storage_tokens
            if self.n_storage_tokens > 0
            else torch.empty(
                1,
                0,
                cls_token.shape[-1],
                dtype=cls_token.dtype,
                device=cls_token.device,
            )
        )
        x = torch.cat(
            [cls_token.expand(B, -1, -1), storage.expand(B, -1, -1), x], dim=1
        )
        return x, (H, W)

    def _get_intermediate_layers_impl(self, x, n=1):
        x, (H, W) = self._prepare_tokens(x)
        total = len(self.blocks)
        blocks_to_take = range(total - n, total) if isinstance(n, int) else n
        rope = self.rope_embed(H=H, W=W)  # same H,W for all blocks
        output = []
        for i, blk in enumerate(self.blocks):
            x = blk(x, rope)
            if i in blocks_to_take:
                output.append(x)
        return output

    def get_intermediate_layers(
        self,
        x,
        *,
        n=1,
        reshape=False,
        return_class_token=False,
        return_extra_tokens=False,
        norm=True,
    ):
        outputs = self._get_intermediate_layers_impl(x, n)
        if norm:
            outputs = [self.norm(o) for o in outputs]
        class_tokens = [o[:, 0] for o in outputs]
        extra_tokens = [o[:, 1 : self.n_storage_tokens + 1] for o in outputs]
        outputs = [o[:, self.n_storage_tokens + 1 :] for o in outputs]
        if return_class_token and return_extra_tokens:
            return tuple(zip(outputs, class_tokens, extra_tokens))
        elif return_class_token:
            return tuple(zip(outputs, class_tokens))
        elif return_extra_tokens:
            return tuple(zip(outputs, extra_tokens))
        return tuple(outputs)


# =============================================================================
# Backbone: SpatialPriorModule + DINOv3STAs
# =============================================================================


class SpatialPriorModulev2(nn.Module):
    """Lite spatial prior CNN.
    FIX: SyncBatchNorm -> BatchNorm2d (identical in eval; avoids distributed requirement).
    """

    def __init__(self, inplanes=16):
        super().__init__()
        BN = nn.BatchNorm2d  # was SyncBatchNorm
        self.stem = nn.Sequential(
            nn.Conv2d(3, inplanes, 3, stride=2, padding=1, bias=False),
            BN(inplanes),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(inplanes, 2 * inplanes, 3, stride=2, padding=1, bias=False),
            BN(2 * inplanes),
        )
        self.conv3 = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(2 * inplanes, 4 * inplanes, 3, stride=2, padding=1, bias=False),
            BN(4 * inplanes),
        )
        self.conv4 = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(4 * inplanes, 4 * inplanes, 3, stride=2, padding=1, bias=False),
            BN(4 * inplanes),
        )

    def forward(self, x):
        c2 = self.conv2(self.stem(x))
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)
        return c2, c3, c4


class DINOv3STAs(nn.Module):
    def __init__(
        self,
        name="dinov3_vits16",
        weights_path=None,
        interaction_indexes=None,
        finetune=True,
        conv_inplane=16,
        hidden_dim=None,
        patch_size=16,
        use_sta=True,
    ):
        super().__init__()
        interaction_indexes = interaction_indexes or []
        self.dinov3 = DinoVisionTransformer(name=name)
        if weights_path:
            import os

            if os.path.exists(weights_path):
                print(f"Loading DINOv3 weights from {weights_path}")
                self.dinov3.load_state_dict(
                    torch.load(weights_path, map_location="cpu")
                )
            else:
                print(f"DINOv3 weights not found at {weights_path}, using random init")
        embed_dim = self.dinov3.embed_dim
        self.interaction_indexes = interaction_indexes
        self.patch_size = patch_size
        if not finetune:
            self.dinov3.eval()
            self.dinov3.requires_grad_(False)
        self.use_sta = use_sta
        if use_sta:
            self.sta = SpatialPriorModulev2(inplanes=conv_inplane)
        else:
            conv_inplane = 0
        hidden_dim = hidden_dim if hidden_dim is not None else embed_dim
        # FIX: SyncBatchNorm -> BatchNorm2d
        BN = nn.BatchNorm2d
        self.convs = nn.ModuleList(
            [
                nn.Conv2d(embed_dim + conv_inplane * 2, hidden_dim, 1, bias=False),
                nn.Conv2d(embed_dim + conv_inplane * 4, hidden_dim, 1, bias=False),
                nn.Conv2d(embed_dim + conv_inplane * 4, hidden_dim, 1, bias=False),
            ]
        )
        self.norms = nn.ModuleList([BN(hidden_dim), BN(hidden_dim), BN(hidden_dim)])

    def forward(self, x):
        H_c, W_c = x.shape[2] // 16, x.shape[3] // 16
        bs = x.shape[0]
        all_layers = self.dinov3.get_intermediate_layers(
            x, n=self.interaction_indexes, return_class_token=True
        )
        sem_feats = []
        num_scales = len(all_layers) - 2
        for i, sem_feat in enumerate(all_layers):
            feat, _ = sem_feat
            sem_feat = feat.transpose(1, 2).view(bs, -1, H_c, W_c).contiguous()
            resize_H = int(H_c * 2 ** (num_scales - i))
            resize_W = int(W_c * 2 ** (num_scales - i))
            sem_feat = F.interpolate(
                sem_feat,
                size=[resize_H, resize_W],
                mode="bilinear",
                align_corners=False,
            )
            sem_feats.append(sem_feat)
        if self.use_sta:
            detail_feats = self.sta(x)
            fused = [torch.cat([s, d], dim=1) for s, d in zip(sem_feats, detail_feats)]
        else:
            fused = sem_feats
        c2 = self.norms[0](self.convs[0](fused[0]))
        c3 = self.norms[1](self.convs[1](fused[1]))
        c4 = self.norms[2](self.convs[2](fused[2]))
        return c2, c3, c4


# =============================================================================
# Encoder: HybridEncoder
# =============================================================================


class ConvNormLayer_fuse(nn.Module):
    def __init__(
        self,
        ch_in,
        ch_out,
        kernel_size,
        stride,
        g=1,
        padding=None,
        bias=False,
        act=None,
    ):
        super().__init__()
        padding = (kernel_size - 1) // 2 if padding is None else padding
        self.conv = nn.Conv2d(
            ch_in, ch_out, kernel_size, stride, groups=g, padding=padding, bias=bias
        )
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = nn.Identity() if act is None else get_activation(act)

    def forward(self, x):
        if hasattr(self, "conv_bn_fused"):
            return self.act(self.conv_bn_fused(x))
        return self.act(self.norm(self.conv(x)))

    def convert_to_deploy(self):
        if not hasattr(self, "conv_bn_fused"):
            c = self.conv
            self.conv_bn_fused = nn.Conv2d(
                c.in_channels,
                c.out_channels,
                c.kernel_size,
                c.stride,
                groups=c.groups,
                padding=c.padding,
                bias=True,
            )
        k = self.conv.weight
        rm, rv = self.norm.running_mean, self.norm.running_var
        g, b, eps = self.norm.weight, self.norm.bias, self.norm.eps
        std = (rv + eps).sqrt()
        t = (g / std).reshape(-1, 1, 1, 1)
        self.conv_bn_fused.weight.data = k * t
        self.conv_bn_fused.bias.data = b - rm * g / std
        del self.conv, self.norm


class ConvNormLayer(nn.Module):
    def __init__(
        self,
        ch_in,
        ch_out,
        kernel_size,
        stride,
        g=1,
        padding=None,
        bias=False,
        act=None,
    ):
        super().__init__()
        padding = (kernel_size - 1) // 2 if padding is None else padding
        self.conv = nn.Conv2d(
            ch_in, ch_out, kernel_size, stride, groups=g, padding=padding, bias=bias
        )
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = nn.Identity() if act is None else get_activation(act)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class VGGBlock(nn.Module):
    def __init__(self, ch_in, ch_out, act="relu"):
        super().__init__()
        self.ch_in, self.ch_out = ch_in, ch_out
        self.conv1 = ConvNormLayer(ch_in, ch_out, 3, 1, padding=1, act=None)
        self.conv2 = ConvNormLayer(ch_in, ch_out, 1, 1, padding=0, act=None)
        self.act = nn.Identity() if act is None else get_activation(act)

    def forward(self, x):
        if hasattr(self, "conv"):
            return self.act(self.conv(x))
        return self.act(self.conv1(x) + self.conv2(x))

    def convert_to_deploy(self):
        if not hasattr(self, "conv"):
            self.conv = nn.Conv2d(self.ch_in, self.ch_out, 3, 1, padding=1)
        k3, b3 = self._fuse(self.conv1)
        k1, b1 = self._fuse(self.conv2)
        self.conv.weight.data = k3 + F.pad(k1, [1, 1, 1, 1])
        self.conv.bias.data = b3 + b1
        del self.conv1, self.conv2

    def _fuse(self, branch):
        k = branch.conv.weight
        rm, rv, g, b, eps = (
            branch.norm.running_mean,
            branch.norm.running_var,
            branch.norm.weight,
            branch.norm.bias,
            branch.norm.eps,
        )
        std = (rv + eps).sqrt()
        t = (g / std).reshape(-1, 1, 1, 1)
        return k * t, b - rm * g / std


class CSPLayer2(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        num_blocks=3,
        expansion=1.0,
        bias=False,
        act="silu",
        bottletype=VGGBlock,
    ):
        super().__init__()
        hidden = int(out_channels * expansion)
        self.conv1 = ConvNormLayer_fuse(
            in_channels, hidden * 2, 1, 1, bias=bias, act=act
        )
        self.bottlenecks = nn.Sequential(
            *[bottletype(hidden, hidden, act=act) for _ in range(num_blocks)]
        )
        self.conv3 = (
            ConvNormLayer_fuse(hidden, out_channels, 1, 1, bias=bias, act=act)
            if hidden != out_channels
            else nn.Identity()
        )

    def forward(self, x):
        y = list(self.conv1(x).chunk(2, 1))
        return self.conv3(y[0] + self.bottlenecks(y[1]))


class RepNCSPELAN5(nn.Module):
    def __init__(self, c1, c2, c3, c4, n=3, bias=False, act="silu"):
        super().__init__()
        self.c = c3 // 2
        self.cv1 = ConvNormLayer_fuse(c1, c3, 1, 1, bias=bias, act=act)
        self.cv2 = nn.Sequential(
            CSPLayer2(c3 // 2, c4, n, 1, bias=bias, act=act, bottletype=VGGBlock)
        )
        self.cv3 = nn.Sequential(
            CSPLayer2(c4, c4, n, 1, bias=bias, act=act, bottletype=VGGBlock)
        )
        self.cv4 = ConvNormLayer_fuse(c3 + 2 * c4, c2, 1, 1, bias=bias, act=act)

    def forward(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))


class SCDown(nn.Module):
    def __init__(self, c1, c2, k, s, act=None):
        super().__init__()
        self.cv1 = ConvNormLayer_fuse(c1, c2, 1, 1)
        self.cv2 = ConvNormLayer_fuse(c2, c2, k, s, c2)

    def forward(self, x):
        return self.cv2(self.cv1(x))


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout, batch_first=True
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = get_activation(activation)

    def forward(self, src, src_mask=None, pos_embed=None):
        q = k = _with_pos_embed(src, pos_embed)
        src2, _ = self.self_attn(q, k, value=src, attn_mask=src_mask)
        src = self.norm1(src + self.dropout1(src2))
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        return self.norm2(src + self.dropout2(src2))


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = nn.ModuleList(
            [copy.deepcopy(encoder_layer) for _ in range(num_layers)]
        )
        self.norm = norm

    def forward(self, src, src_mask=None, pos_embed=None):
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=src_mask, pos_embed=pos_embed)
        if self.norm is not None:
            output = self.norm(output)
        return output


class HybridEncoder(nn.Module):
    def __init__(
        self,
        in_channels=(512, 1024, 2048),
        feat_strides=(8, 16, 32),
        hidden_dim=256,
        nhead=8,
        dim_feedforward=1024,
        dropout=0.0,
        enc_act="gelu",
        use_encoder_idx=(2,),
        num_encoder_layers=1,
        pe_temperature=10000,
        expansion=1.0,
        depth_mult=1.0,
        act="silu",
        eval_spatial_size=None,
        version="deim",
        csp_type="csp2",
        fuse_op="sum",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.feat_strides = feat_strides
        self.hidden_dim = hidden_dim
        self.use_encoder_idx = use_encoder_idx
        self.num_encoder_layers = num_encoder_layers
        self.pe_temperature = pe_temperature
        self.eval_spatial_size = eval_spatial_size
        self.out_channels = [hidden_dim] * len(in_channels)
        self.out_strides = feat_strides
        self.fuse_op = fuse_op
        self.input_proj = nn.ModuleList()
        for c in in_channels:
            if c != hidden_dim:
                self.input_proj.append(
                    nn.Sequential(
                        OrderedDict(
                            [
                                ("conv", nn.Conv2d(c, hidden_dim, 1, bias=False)),
                                ("norm", nn.BatchNorm2d(hidden_dim)),
                            ]
                        )
                    )
                )
            else:
                self.input_proj.append(nn.Identity())
        encoder_layer = TransformerEncoderLayer(
            hidden_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=enc_act,
        )
        self.encoder = nn.ModuleList(
            [
                TransformerEncoder(copy.deepcopy(encoder_layer), num_encoder_layers)
                for _ in range(len(use_encoder_idx))
            ]
        )
        input_dim = hidden_dim if fuse_op == "sum" else hidden_dim * 2
        c1 = input_dim
        c2 = hidden_dim
        c3 = hidden_dim * 2
        c4 = round(expansion * hidden_dim // 2)
        num_blocks = round(3 * depth_mult)
        Fuse_Block = RepNCSPELAN5(c1=c1, c2=c2, c3=c3, c4=c4, n=num_blocks, act=act)
        Lateral_Conv = ConvNormLayer_fuse(hidden_dim, hidden_dim, 1, 1)
        SCDown_Conv = nn.Sequential(SCDown(hidden_dim, hidden_dim, 3, 2))
        self.lateral_convs = nn.ModuleList(
            [copy.deepcopy(Lateral_Conv) for _ in range(len(in_channels) - 1)]
        )
        self.fpn_blocks = nn.ModuleList(
            [copy.deepcopy(Fuse_Block) for _ in range(len(in_channels) - 1)]
        )
        self.downsample_convs = nn.ModuleList(
            [copy.deepcopy(SCDown_Conv) for _ in range(len(in_channels) - 1)]
        )
        self.pan_blocks = nn.ModuleList(
            [copy.deepcopy(Fuse_Block) for _ in range(len(in_channels) - 1)]
        )
        self._reset_parameters()

    def _reset_parameters(self):
        if self.eval_spatial_size:
            for idx in self.use_encoder_idx:
                stride = self.feat_strides[idx]
                pos_embed = self._build_2d_sincos_pos_embed(
                    self.eval_spatial_size[1] // stride,
                    self.eval_spatial_size[0] // stride,
                    self.hidden_dim,
                    self.pe_temperature,
                )
                setattr(self, f"pos_embed{idx}", pos_embed)

    @staticmethod
    def _build_2d_sincos_pos_embed(w, h, embed_dim=256, temperature=10000.0):
        grid_w = torch.arange(int(w), dtype=torch.float32)
        grid_h = torch.arange(int(h), dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing="ij")
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1.0 / (temperature**omega)
        out_w = grid_w.flatten()[..., None] @ omega[None]
        out_h = grid_h.flatten()[..., None] @ omega[None]
        return torch.concat(
            [out_w.sin(), out_w.cos(), out_h.sin(), out_h.cos()], dim=1
        )[None]

    def forward(self, feats):
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
        if self.num_encoder_layers > 0:
            for i, enc_ind in enumerate(self.use_encoder_idx):
                h, w = proj_feats[enc_ind].shape[2:]
                src = proj_feats[enc_ind].flatten(2).permute(0, 2, 1)
                if self.training or self.eval_spatial_size is None:
                    pos_embed = self._build_2d_sincos_pos_embed(
                        w, h, self.hidden_dim, self.pe_temperature
                    ).to(src.device)
                else:
                    pos_embed = getattr(self, f"pos_embed{enc_ind}").to(src.device)
                memory = self.encoder[i](src, pos_embed=pos_embed)
                proj_feats[enc_ind] = (
                    memory.permute(0, 2, 1)
                    .reshape(-1, self.hidden_dim, h, w)
                    .contiguous()
                )
        inner_outs = [proj_feats[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_high = inner_outs[0]
            feat_low = proj_feats[idx - 1]
            feat_high = self.lateral_convs[len(self.in_channels) - 1 - idx](feat_high)
            inner_outs[0] = feat_high
            up = F.interpolate(feat_high, scale_factor=2.0, mode="nearest")
            fused = (
                (up + feat_low)
                if self.fuse_op == "sum"
                else torch.cat([up, feat_low], dim=1)
            )
            inner_outs.insert(
                0, self.fpn_blocks[len(self.in_channels) - 1 - idx](fused)
            )
        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_high = inner_outs[idx + 1]
            down = self.downsample_convs[idx](feat_low)
            fused = (
                (down + feat_high)
                if self.fuse_op == "sum"
                else torch.cat([down, feat_high], dim=1)
            )
            outs.append(self.pan_blocks[idx](fused))
        return outs


# =============================================================================
# Decoder utilities
# =============================================================================


def weighting_function(reg_max, up, reg_scale, deploy=False):
    if deploy:
        upper_bound1 = (abs(up[0]) * abs(reg_scale)).item()
        upper_bound2 = (abs(up[0]) * abs(reg_scale) * 2).item()
        step = (upper_bound1 + 1) ** (2 / (reg_max - 2))
        left_values = [-((step) ** i) + 1 for i in range(reg_max // 2 - 1, 0, -1)]
        right_values = [(step) ** i - 1 for i in range(1, reg_max // 2)]
        values = (
            [-upper_bound2]
            + left_values
            + [torch.zeros_like(up[0][None])]
            + right_values
            + [upper_bound2]
        )
        return torch.tensor(values, dtype=up.dtype, device=up.device)
    else:
        upper_bound1 = abs(up[0]) * abs(reg_scale)
        upper_bound2 = abs(up[0]) * abs(reg_scale) * 2
        step = (upper_bound1 + 1) ** (2 / (reg_max - 2))
        left_values = [-((step) ** i) + 1 for i in range(reg_max // 2 - 1, 0, -1)]
        right_values = [(step) ** i - 1 for i in range(1, reg_max // 2)]
        values = (
            [-upper_bound2]
            + left_values
            + [torch.zeros_like(up[0][None])]
            + right_values
            + [upper_bound2]
        )
        return torch.cat(values, 0)


def distance2bbox(points, distance, reg_scale):
    """Original (training-accurate). Export script applies a TFLite-safe patch."""
    reg_scale = abs(reg_scale)
    x1 = points[..., 0] - (0.5 * reg_scale + distance[..., 0]) * (
        points[..., 2] / reg_scale
    )
    y1 = points[..., 1] - (0.5 * reg_scale + distance[..., 1]) * (
        points[..., 3] / reg_scale
    )
    x2 = points[..., 0] + (0.5 * reg_scale + distance[..., 2]) * (
        points[..., 2] / reg_scale
    )
    y2 = points[..., 1] + (0.5 * reg_scale + distance[..., 3]) * (
        points[..., 3] / reg_scale
    )
    return box_xyxy_to_cxcywh(torch.stack([x1, y1, x2, y2], -1))


def deformable_attention_core_func_v2(
    value,
    value_spatial_shapes,
    sampling_locations,
    attention_weights,
    num_points_list,
    method="default",
    value_shape="default",
):
    if value_shape == "default":
        bs, n_head, c, _ = value[0].shape
    elif value_shape == "reshape":
        bs, _, n_head, c = value.shape
        split_shape = [h * w for h, w in value_spatial_shapes]
        value = value.permute(0, 2, 3, 1).flatten(0, 1).split(split_shape, dim=-1)
    _, Len_q, _, _, _ = sampling_locations.shape
    sampling_grids = (
        2 * sampling_locations - 1 if method == "default" else sampling_locations
    )
    sampling_grids = sampling_grids.permute(0, 2, 1, 3, 4).flatten(0, 1)
    sampling_locations_list = sampling_grids.split(num_points_list, dim=-2)
    sampling_value_list = []
    for level, (h, w) in enumerate(value_spatial_shapes):
        value_l = value[level].reshape(bs * n_head, c, h, w)
        grid_l = sampling_locations_list[level]
        if method == "default":
            sv = F.grid_sample(
                value_l,
                grid_l,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )
        elif method == "discrete":
            coord = (grid_l * torch.tensor([[w, h]], device=value_l.device) + 0.5).to(
                torch.int32
            )
            coord = coord.clamp(0, h - 1).reshape(
                bs * n_head, Len_q * num_points_list[level], 2
            )
            s_idx = (
                torch.arange(coord.shape[0], device=value_l.device)
                .unsqueeze(-1)
                .repeat(1, coord.shape[1])
            )
            sv = value_l[s_idx, :, coord[..., 1], coord[..., 0]]
            sv = sv.permute(0, 2, 1).reshape(
                bs * n_head, c, Len_q, num_points_list[level]
            )
        sampling_value_list.append(sv)
    attn = attention_weights.permute(0, 2, 1, 3).reshape(
        bs * n_head, 1, Len_q, sum(num_points_list)
    )
    output = (
        (torch.concat(sampling_value_list, dim=-1) * attn)
        .sum(-1)
        .reshape(bs, n_head * c, Len_q)
    )
    return output.permute(0, 2, 1)


# =============================================================================
# Decoder modules
# =============================================================================


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3, act="relu"):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.act = get_activation(act)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self._norm(x.float()).type_as(x) * self.scale


class Gate(nn.Module):
    def __init__(self, d_model, use_rmsnorm=False):
        super().__init__()
        self.gate = nn.Linear(2 * d_model, 2 * d_model)
        b = bias_init_with_prob(0.5)
        init.constant_(self.gate.bias, b)
        init.constant_(self.gate.weight, 0)
        self.norm = RMSNorm(d_model) if use_rmsnorm else nn.LayerNorm(d_model)

    def forward(self, x1, x2):
        gates = torch.sigmoid(self.gate(torch.cat([x1, x2], dim=-1)))
        g1, g2 = gates.chunk(2, dim=-1)
        return self.norm(g1 * x1 + g2 * x2)


class Integral(nn.Module):
    def __init__(self, reg_max=32):
        super().__init__()
        self.reg_max = reg_max

    def forward(self, x, project):
        shape = x.shape
        x = F.softmax(x.reshape(-1, self.reg_max + 1), dim=1)
        x = F.linear(x, project.to(x.device)).reshape(-1, 4)
        return x.reshape(list(shape[:-1]) + [-1])


class LQE(nn.Module):
    def __init__(self, k, hidden_dim, num_layers, reg_max, act="relu"):
        super().__init__()
        self.k = k
        self.reg_max = reg_max
        self.reg_conf = MLP(4 * (k + 1), hidden_dim, 1, num_layers, act=act)
        init.constant_(self.reg_conf.layers[-1].bias, 0)
        init.constant_(self.reg_conf.layers[-1].weight, 0)

    def forward(self, scores, pred_corners):
        B, L, _ = pred_corners.size()
        prob = F.softmax(pred_corners.reshape(B, L, 4, self.reg_max + 1), dim=-1)
        prob_topk, _ = prob.topk(self.k, dim=-1)  # exact — TFLite export patches this
        # prob_topk = prob[..., :self.k]  # TFLite-friendly patch
        stat = torch.cat([prob_topk, prob_topk.mean(dim=-1, keepdim=True)], dim=-1)
        return scores + self.reg_conf(stat.reshape(B, L, -1))


class MSDeformableAttention(nn.Module):
    def __init__(
        self,
        embed_dim=256,
        num_heads=8,
        num_levels=4,
        num_points=4,
        method="default",
        offset_scale=0.5,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.offset_scale = offset_scale
        num_points_list = (
            num_points if isinstance(num_points, list) else [num_points] * num_levels
        )
        self.num_points_list = num_points_list
        num_points_scale = [1 / n for n in num_points_list for _ in range(n)]
        self.register_buffer(
            "num_points_scale", torch.tensor(num_points_scale, dtype=torch.float32)
        )
        self.total_points = num_heads * sum(num_points_list)
        self.method = method
        self.head_dim = embed_dim // num_heads
        self.sampling_offsets = nn.Linear(embed_dim, self.total_points * 2)
        self.attention_weights = nn.Linear(embed_dim, self.total_points)
        self.ms_deformable_attn_core = functools.partial(
            deformable_attention_core_func_v2, method=method
        )
        self._reset_parameters()
        if method == "discrete":
            for p in self.sampling_offsets.parameters():
                p.requires_grad = False

    def _reset_parameters(self):
        init.constant_(self.sampling_offsets.weight, 0)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (
            2 * math.pi / self.num_heads
        )
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = grid_init / grid_init.abs().max(-1, keepdim=True).values
        grid_init = grid_init.reshape(self.num_heads, 1, 2).tile(
            [1, sum(self.num_points_list), 1]
        )
        scaling = torch.concat(
            [torch.arange(1, n + 1) for n in self.num_points_list]
        ).reshape(1, -1, 1)
        grid_init *= scaling
        self.sampling_offsets.bias.data[...] = grid_init.flatten()
        init.constant_(self.attention_weights.weight, 0)
        init.constant_(self.attention_weights.bias, 0)

    def forward(self, query, reference_points, value, value_spatial_shapes):
        bs, Len_q = query.shape[:2]
        sampling_offsets = self.sampling_offsets(query).reshape(
            bs, Len_q, self.num_heads, sum(self.num_points_list), 2
        )
        attention_weights = F.softmax(
            self.attention_weights(query).reshape(
                bs, Len_q, self.num_heads, sum(self.num_points_list)
            ),
            dim=-1,
        )
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.tensor(value_spatial_shapes, device=query.device)
            offset_normalizer = offset_normalizer.flip([1]).reshape(
                1, 1, 1, self.num_levels, 1, 2
            )
            sampling_locations = (
                reference_points.reshape(bs, Len_q, 1, self.num_levels, 1, 2)
                + sampling_offsets / offset_normalizer
            )
        else:
            nps = self.num_points_scale.to(dtype=query.dtype).unsqueeze(-1)
            offset = (
                sampling_offsets
                * nps
                * reference_points[:, :, None, :, 2:]
                * self.offset_scale
            )
            sampling_locations = reference_points[:, :, None, :, :2] + offset
        return self.ms_deformable_attn_core(
            value,
            value_spatial_shapes,
            sampling_locations,
            attention_weights,
            self.num_points_list,
        )


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model=256,
        n_head=8,
        dim_feedforward=1024,
        dropout=0.0,
        activation="relu",
        n_levels=4,
        n_points=4,
        cross_attn_method="default",
        layer_scale=None,
        use_gateway=False,
    ):
        super().__init__()
        if layer_scale is not None:
            dim_feedforward = round(layer_scale * dim_feedforward)
            d_model = round(layer_scale * d_model)
        self.self_attn = nn.MultiheadAttention(
            d_model, n_head, dropout=dropout, batch_first=True
        )
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = RMSNorm(d_model)
        self.cross_attn = MSDeformableAttention(
            d_model, n_head, n_levels, n_points, method=cross_attn_method
        )
        self.dropout2 = nn.Dropout(dropout)
        self.use_gateway = use_gateway
        if use_gateway:
            self.gateway = Gate(d_model, use_rmsnorm=True)
        else:
            self.norm2 = RMSNorm(d_model)
        self.swish_ffn = SwiGLUFFN(d_model, dim_feedforward // 2, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = RMSNorm(d_model)

    def forward(
        self,
        target,
        reference_points,
        value,
        spatial_shapes,
        attn_mask=None,
        query_pos_embed=None,
    ):
        q = k = _with_pos_embed(target, query_pos_embed)
        target2, _ = self.self_attn(q, k, value=target, attn_mask=attn_mask)
        target = self.norm1(target + self.dropout1(target2))
        target2 = self.cross_attn(
            _with_pos_embed(target, query_pos_embed),
            reference_points,
            value,
            spatial_shapes,
        )
        if self.use_gateway:
            target = self.gateway(target, self.dropout2(target2))
        else:
            target = self.norm2(target + self.dropout2(target2))
        target2 = self.swish_ffn(target)
        return self.norm3(
            (target + self.dropout4(target2)).clamp(min=-65504, max=65504)
        )


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        hidden_dim,
        decoder_layer,
        decoder_layer_wide,
        num_layers,
        num_head,
        reg_max,
        reg_scale,
        up,
        eval_idx=-1,
        layer_scale=2,
        act="relu",
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.layer_scale = layer_scale
        self.num_head = num_head
        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx
        self.up, self.reg_scale, self.reg_max = up, reg_scale, reg_max
        self.layers = nn.ModuleList(
            [copy.deepcopy(decoder_layer) for _ in range(self.eval_idx + 1)]
            + [
                copy.deepcopy(decoder_layer_wide)
                for _ in range(num_layers - self.eval_idx - 1)
            ]
        )
        self.lqe_layers = nn.ModuleList(
            [copy.deepcopy(LQE(4, 64, 2, reg_max, act=act)) for _ in range(num_layers)]
        )
        self.register_buffer(
            "project", weighting_function(reg_max, up, reg_scale), persistent=False
        )

    def value_op(self, memory, value_proj, value_scale, memory_mask, spatial_shapes):
        value = value_proj(memory) if value_proj is not None else memory
        value = (
            F.interpolate(memory, size=value_scale)
            if value_scale is not None
            else value
        )
        if memory_mask is not None:
            value = value * memory_mask.to(value.dtype).unsqueeze(-1)
        value = value.reshape(value.shape[0], value.shape[1], self.num_head, -1)
        split_shape = [h * w for h, w in spatial_shapes]
        return value.permute(0, 2, 3, 1).split(split_shape, dim=-1)

    def convert_to_deploy(self):
        self.project = weighting_function(
            self.reg_max, self.up, self.reg_scale, deploy=True
        )
        self.layers = self.layers[: self.eval_idx + 1]
        self.lqe_layers = nn.ModuleList(
            [nn.Identity()] * self.eval_idx + [self.lqe_layers[self.eval_idx]]
        )

    def forward(
        self,
        target,
        ref_points_unact,
        memory,
        spatial_shapes,
        bbox_head,
        score_head,
        query_pos_head,
        pre_bbox_head,
        integral,
        attn_mask=None,
        memory_mask=None,
    ):
        output = target
        output_detach = pred_corners_undetach = 0
        value = self.value_op(memory, None, None, memory_mask, spatial_shapes)
        project = self.project
        ref_points_detach = F.sigmoid(ref_points_unact)
        query_pos_embed = query_pos_head(ref_points_detach).clamp(min=-10, max=10)
        dec_out_bboxes, dec_out_logits, dec_out_corners, dec_out_refs = [], [], [], []
        for i, layer in enumerate(self.layers):
            ref_points_input = ref_points_detach.unsqueeze(2)
            if i >= self.eval_idx + 1 and self.layer_scale > 1:
                query_pos_embed = F.interpolate(
                    query_pos_embed, scale_factor=self.layer_scale
                )
                value = self.value_op(
                    memory, None, query_pos_embed.shape[-1], memory_mask, spatial_shapes
                )
                output = F.interpolate(output, size=query_pos_embed.shape[-1])
                output_detach = output.detach()
            output = layer(
                output,
                ref_points_input,
                value,
                spatial_shapes,
                attn_mask,
                query_pos_embed,
            )
            if i == 0:
                pre_bboxes = F.sigmoid(
                    pre_bbox_head(output) + inverse_sigmoid(ref_points_detach)
                )
                pre_scores = score_head[0](output)
                ref_points_initial = pre_bboxes.detach()
            pred_corners = bbox_head[i](output + output_detach) + pred_corners_undetach
            inter_ref_bbox = distance2bbox(
                ref_points_initial, integral(pred_corners, project), self.reg_scale
            )
            if self.training or i == self.eval_idx:
                scores = score_head[i](output)
                scores = self.lqe_layers[i](scores, pred_corners)
                dec_out_logits.append(scores)
                dec_out_bboxes.append(inter_ref_bbox)
                dec_out_corners.append(pred_corners)
                dec_out_refs.append(ref_points_initial)
                if not self.training:
                    break
            pred_corners_undetach = pred_corners
            ref_points_detach = inter_ref_bbox.detach()
            output_detach = output.detach()
        return (
            torch.stack(dec_out_bboxes),
            torch.stack(dec_out_logits),
            torch.stack(dec_out_corners),
            torch.stack(dec_out_refs),
            pre_bboxes,
            pre_scores,
        )


class DEIMTransformer(nn.Module):
    def __init__(
        self,
        num_classes=80,
        hidden_dim=256,
        num_queries=300,
        feat_channels=(256, 256, 256),
        feat_strides=(8, 16, 32),
        num_levels=3,
        num_points=4,
        nhead=8,
        num_layers=6,
        dim_feedforward=1024,
        dropout=0.0,
        activation="relu",
        num_denoising=100,
        learn_query_content=False,
        eval_spatial_size=None,
        eval_idx=-1,
        eps=1e-2,
        cross_attn_method="default",
        query_select_method="default",
        reg_max=32,
        reg_scale=4.0,
        layer_scale=1,
        mlp_act="relu",
        use_gateway=True,
        share_bbox_head=False,
        share_score_head=False,
    ):
        super().__init__()
        feat_strides = list(feat_strides)
        for _ in range(num_levels - len(feat_strides)):
            feat_strides.append(feat_strides[-1] * 2)
        self.hidden_dim = hidden_dim
        scaled_dim = round(layer_scale * hidden_dim)
        self.nhead = nhead
        self.feat_strides = feat_strides
        self.num_levels = num_levels
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.eps = eps
        self.num_layers = num_layers
        self.eval_spatial_size = eval_spatial_size

        self.reg_max = reg_max
        self.cross_attn_method = cross_attn_method
        self.query_select_method = query_select_method
        self._build_input_proj_layer(feat_channels)
        self.up = nn.Parameter(torch.tensor([0.5]), requires_grad=False)
        self.reg_scale = nn.Parameter(torch.tensor([reg_scale]), requires_grad=False)
        dec_layer = TransformerDecoderLayer(
            hidden_dim,
            nhead,
            dim_feedforward,
            dropout,
            activation,
            num_levels,
            num_points,
            cross_attn_method=cross_attn_method,
            use_gateway=use_gateway,
        )
        dec_layer_wide = TransformerDecoderLayer(
            hidden_dim,
            nhead,
            dim_feedforward,
            dropout,
            activation,
            num_levels,
            num_points,
            cross_attn_method=cross_attn_method,
            layer_scale=layer_scale,
            use_gateway=use_gateway,
        )
        self.decoder = TransformerDecoder(
            hidden_dim,
            dec_layer,
            dec_layer_wide,
            num_layers,
            nhead,
            reg_max,
            self.reg_scale,
            self.up,
            eval_idx,
            layer_scale,
            act=activation,
        )
        self.num_denoising = num_denoising
        if num_denoising > 0:
            self.denoising_class_embed = nn.Embedding(
                num_classes + 1, hidden_dim, padding_idx=num_classes
            )
            init.normal_(self.denoising_class_embed.weight[:-1])
        self.learn_query_content = learn_query_content
        if learn_query_content:
            self.tgt_embed = nn.Embedding(num_queries, hidden_dim)
        self.enc_score_head = nn.Linear(
            hidden_dim, 1 if query_select_method == "agnostic" else num_classes
        )
        self.enc_bbox_head = MLP(hidden_dim, hidden_dim, 4, 3, act=mlp_act)
        self.query_pos_head = MLP(4, hidden_dim, hidden_dim, 3, act=mlp_act)
        self.pre_bbox_head = MLP(hidden_dim, hidden_dim, 4, 3, act=mlp_act)
        self.integral = Integral(self.reg_max)
        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx
        dec_score = nn.Linear(hidden_dim, num_classes)
        self.dec_score_head = nn.ModuleList(
            [
                dec_score if share_score_head else copy.deepcopy(dec_score)
                for _ in range(self.eval_idx + 1)
            ]
            + [copy.deepcopy(dec_score) for _ in range(num_layers - self.eval_idx - 1)]
        )
        dec_bbox = MLP(hidden_dim, hidden_dim, 4 * (self.reg_max + 1), 3, act=mlp_act)
        self.dec_bbox_head = nn.ModuleList(
            [
                dec_bbox if share_bbox_head else copy.deepcopy(dec_bbox)
                for _ in range(self.eval_idx + 1)
            ]
            + [
                MLP(scaled_dim, scaled_dim, 4 * (self.reg_max + 1), 3, act=mlp_act)
                for _ in range(num_layers - self.eval_idx - 1)
            ]
        )
        if eval_spatial_size:
            anchors, valid_mask = self._generate_anchors()
            self.register_buffer("anchors", anchors)
            self.register_buffer("valid_mask", valid_mask)
        self._reset_parameters(feat_channels)

    def convert_to_deploy(self):
        self.dec_score_head = nn.ModuleList(
            [nn.Identity()] * self.eval_idx + [self.dec_score_head[self.eval_idx]]
        )
        self.dec_bbox_head = nn.ModuleList(
            [
                self.dec_bbox_head[i] if i <= self.eval_idx else nn.Identity()
                for i in range(len(self.dec_bbox_head))
            ]
        )

    def _reset_parameters(self, feat_channels):
        bias = bias_init_with_prob(0.01)
        init.constant_(self.enc_score_head.bias, bias)
        init.constant_(self.enc_bbox_head.layers[-1].weight, 0)
        init.constant_(self.enc_bbox_head.layers[-1].bias, 0)
        init.constant_(self.pre_bbox_head.layers[-1].weight, 0)
        init.constant_(self.pre_bbox_head.layers[-1].bias, 0)
        for cls_, reg_ in zip(self.dec_score_head, self.dec_bbox_head):
            init.constant_(cls_.bias, bias)
            if hasattr(reg_, "layers"):
                init.constant_(reg_.layers[-1].weight, 0)
                init.constant_(reg_.layers[-1].bias, 0)
        if self.learn_query_content:
            init.xavier_uniform_(self.tgt_embed.weight)
        for w in [
            self.query_pos_head.layers[0].weight,
            self.query_pos_head.layers[1].weight,
            self.query_pos_head.layers[-1].weight,
        ]:
            init.xavier_uniform_(w)
        for m, c in zip(self.input_proj, feat_channels):
            if c != self.hidden_dim:
                init.xavier_uniform_(m[0].weight)

    def _build_input_proj_layer(self, feat_channels):
        self.input_proj = nn.ModuleList()
        for c in feat_channels:
            if c == self.hidden_dim:
                self.input_proj.append(nn.Identity())
            else:
                self.input_proj.append(
                    nn.Sequential(
                        OrderedDict(
                            [
                                ("conv", nn.Conv2d(c, self.hidden_dim, 1, bias=False)),
                                ("norm", nn.BatchNorm2d(self.hidden_dim)),
                            ]
                        )
                    )
                )
        c = feat_channels[-1]
        for _ in range(self.num_levels - len(feat_channels)):
            if c == self.hidden_dim:
                self.input_proj.append(nn.Identity())
            else:
                self.input_proj.append(
                    nn.Sequential(
                        OrderedDict(
                            [
                                (
                                    "conv",
                                    nn.Conv2d(
                                        c, self.hidden_dim, 3, 2, padding=1, bias=False
                                    ),
                                ),
                                ("norm", nn.BatchNorm2d(self.hidden_dim)),
                            ]
                        )
                    )
                )
                c = self.hidden_dim

    def _get_encoder_input(self, feats):
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
        if self.num_levels > len(proj_feats):
            for i in range(len(proj_feats), self.num_levels):
                proj_feats.append(
                    self.input_proj[i](feats[-1] if i == len(feats) else proj_feats[-1])
                )
        feat_flatten, spatial_shapes = [], []
        for feat in proj_feats:
            _, _, h, w = feat.shape
            feat_flatten.append(feat.flatten(2).permute(0, 2, 1))
            spatial_shapes.append([h, w])
        return torch.concat(feat_flatten, 1), spatial_shapes

    def _generate_anchors(
        self, spatial_shapes=None, grid_size=0.05, dtype=torch.float32, device="cpu"
    ):
        if spatial_shapes is None:
            spatial_shapes = [
                [int(self.eval_spatial_size[0] / s), int(self.eval_spatial_size[1] / s)]
                for s in self.feat_strides
            ]
        anchors = []
        for lvl, (h, w) in enumerate(spatial_shapes):
            grid_y, grid_x = torch.meshgrid(
                torch.arange(h), torch.arange(w), indexing="ij"
            )
            grid_xy = (
                torch.stack([grid_x, grid_y], -1).unsqueeze(0) + 0.5
            ) / torch.tensor([w, h], dtype=dtype)
            wh = torch.ones_like(grid_xy) * grid_size * (2.0**lvl)
            anchors.append(torch.concat([grid_xy, wh], -1).reshape(-1, h * w, 4))
        anchors = torch.concat(anchors, 1).to(device)
        valid_mask = ((anchors > self.eps) * (anchors < 1 - self.eps)).all(
            -1, keepdim=True
        )
        anchors = torch.log(anchors / (1 - anchors))
        anchors = torch.where(valid_mask, anchors, torch.inf)
        return anchors, valid_mask

    def _select_topk(self, memory, outputs_logits, anchors, topk):
        if self.query_select_method == "default":
            _, topk_ind = torch.topk(outputs_logits.max(-1).values, topk, dim=-1)
        elif self.query_select_method == "one2many":
            _, topk_ind = torch.topk(outputs_logits.flatten(1), topk, dim=-1)
            topk_ind = topk_ind // self.num_classes
        else:
            _, topk_ind = torch.topk(outputs_logits.squeeze(-1), topk, dim=-1)
        topk_anchors = anchors.gather(
            1, topk_ind.unsqueeze(-1).repeat(1, 1, anchors.shape[-1])
        )
        topk_logits = (
            outputs_logits.gather(
                1, topk_ind.unsqueeze(-1).repeat(1, 1, outputs_logits.shape[-1])
            )
            if self.training
            else None
        )
        topk_memory = memory.gather(
            1, topk_ind.unsqueeze(-1).repeat(1, 1, memory.shape[-1])
        )
        return topk_memory, topk_logits, topk_anchors

    def _get_decoder_input(
        self, memory, spatial_shapes, denoising_logits=None, denoising_bbox_unact=None
    ):
        if self.training or self.eval_spatial_size is None:
            anchors, valid_mask = self._generate_anchors(
                spatial_shapes, device=memory.device
            )
        else:
            anchors, valid_mask = self.anchors, self.valid_mask
        if memory.shape[0] > 1:
            anchors = anchors.repeat(memory.shape[0], 1, 1)
        memory = valid_mask.to(memory.dtype) * memory
        enc_logits = self.enc_score_head(memory)
        topk_mem, topk_logits, topk_anchors = self._select_topk(
            memory, enc_logits, anchors, self.num_queries
        )
        topk_bbox_unact = self.enc_bbox_head(topk_mem) + topk_anchors
        enc_topk_bboxes_list, enc_topk_logits_list = [], []
        if self.training:
            enc_topk_bboxes_list.append(F.sigmoid(topk_bbox_unact))
            enc_topk_logits_list.append(topk_logits)
        content = (
            self.tgt_embed.weight.unsqueeze(0).tile([memory.shape[0], 1, 1])
            if self.learn_query_content
            else topk_mem.detach()
        )
        topk_bbox_unact = topk_bbox_unact.detach()
        if denoising_bbox_unact is not None:
            topk_bbox_unact = torch.concat(
                [denoising_bbox_unact, topk_bbox_unact], dim=1
            )
            content = torch.concat([denoising_logits, content], dim=1)
        return content, topk_bbox_unact, enc_topk_bboxes_list, enc_topk_logits_list

    def forward(self, feats, targets=None):
        memory, spatial_shapes = self._get_encoder_input(feats)
        # denoising only during training — skipped here
        denoising_logits = denoising_bbox_unact = attn_mask = dn_meta = None
        content, ref_unact, _, _ = self._get_decoder_input(memory, spatial_shapes)
        out_bboxes, out_logits, out_corners, out_refs, pre_bboxes, pre_logits = (
            self.decoder(
                content,
                ref_unact,
                memory,
                spatial_shapes,
                self.dec_bbox_head,
                self.dec_score_head,
                self.query_pos_head,
                self.pre_bbox_head,
                self.integral,
                attn_mask=attn_mask,
            )
        )
        return {"pred_logits": out_logits[-1], "pred_boxes": out_bboxes[-1]}


# =============================================================================
# Top-level DEIM model
# =============================================================================


class DEIM(nn.Module):
    def __init__(self, backbone, encoder, decoder):
        super().__init__()
        self.backbone = backbone
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x, targets=None):
        x = self.backbone(x)
        x = self.encoder(x)
        return self.decoder(x, targets)

    def deploy(self):
        self.eval()
        for m in self.modules():
            if hasattr(m, "convert_to_deploy"):
                m.convert_to_deploy()
        return self


class DEIMv2Wrapper(nn.Module):
    """Export wrapper: single input tensor, raw decoder outputs."""

    def __init__(self, model: DEIM):
        super().__init__()
        self.model = model

    def forward(self, images: torch.Tensor):
        out = self.model(images)
        return out["pred_logits"], out["pred_boxes"]


# =============================================================================
# Model factory
# =============================================================================


def build_deimv2_l(num_classes: int = 2) -> DEIM:
    """Build DEIMv2-L with custom config (DINOv3-ViT-S backbone, hidden_dim=224)."""
    backbone = DINOv3STAs(
        name="dinov3_vits16",
        interaction_indexes=[5, 8, 11],
        finetune=True,
        conv_inplane=32,
        hidden_dim=224,
        patch_size=16,
        use_sta=True,
    )
    encoder = HybridEncoder(
        in_channels=[224, 224, 224],
        feat_strides=[8, 16, 32],
        hidden_dim=224,
        nhead=8,
        dim_feedforward=896,
        dropout=0.0,
        enc_act="gelu",
        use_encoder_idx=[2],
        num_encoder_layers=1,
        expansion=1.0,
        depth_mult=1.0,
        act="silu",
        eval_spatial_size=[640, 640],
        version="deim",
        csp_type="csp2",
        fuse_op="sum",
    )
    decoder = DEIMTransformer(
        num_classes=num_classes,
        hidden_dim=224,
        num_queries=300,
        feat_channels=[224, 224, 224],
        feat_strides=[8, 16, 32],
        num_levels=3,
        num_points=[3, 6, 3],
        nhead=8,
        num_layers=4,
        dim_feedforward=1792,
        dropout=0.0,
        activation="silu",
        num_denoising=100,
        eval_spatial_size=[640, 640],
        eval_idx=-1,
        reg_max=32,
        reg_scale=4.0,
        layer_scale=1,
        mlp_act="silu",
        use_gateway=True,
        cross_attn_method="default",
        query_select_method="default",
    )
    return DEIM(backbone, encoder, decoder)


# =============================================================================
# Weight loading & verification
# =============================================================================


def load_weights(model: DEIM, checkpoint_path: str):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = ckpt["ema"]["module"] if "ema" in ckpt else ckpt["model"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(
            f"  Missing keys  ({len(missing)}): {missing[:5]}{'...' if len(missing) > 5 else ''}"
        )
    if unexpected:
        print(
            f"  Unexpected keys ({len(unexpected)}): {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}"
        )
    return missing, unexpected


def verify_against_original(
    checkpoint_path: str,
    config_path: str,
    num_classes: int,
    input_size=(640, 640),
    atol=1e-4,
):
    """
    Compare single-file model outputs against the original engine/ model.
    Requires engine/ to be importable (run from project root).
    """
    import sys, os

    sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
    from engine.core import YAMLConfig

    print("Loading original model via YAMLConfig ...")
    cfg = YAMLConfig(config_path, resume=checkpoint_path)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = ckpt["ema"]["module"] if "ema" in ckpt else ckpt["model"]
    cfg.model.load_state_dict(state)

    class OrigWrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.deploy()

        def forward(self, x):
            out = self.model(x)
            return out["pred_logits"], out["pred_boxes"]

    orig = OrigWrapper().eval()

    print("Loading single-file model ...")
    new_model = build_deimv2_l(num_classes=num_classes)
    load_weights(new_model, checkpoint_path)
    new_model.deploy()
    wrapper = DEIMv2Wrapper(new_model).eval()

    h, w = input_size
    sample = torch.randn(1, 3, h, w)

    print("Running inference ...")
    with torch.no_grad():
        orig_logits, orig_boxes = orig(sample)
        new_logits, new_boxes = wrapper(sample)

    for name, t_orig, t_new in [
        ("pred_logits", orig_logits, new_logits),
        ("pred_boxes", orig_boxes, new_boxes),
    ]:
        diff = (t_orig - t_new).abs()
        ok = diff.max().item() <= atol
        print(
            f"  [{'PASS' if ok else 'FAIL'}] {name:12s}  max_diff={diff.max().item():.2e}  atol={atol}"
        )

    return orig_logits, orig_boxes, new_logits, new_boxes


# =============================================================================
# CLI
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="DEIMv2-L single-file model — load & verify"
    )
    parser.add_argument(
        "-r", "--resume", type=str, required=True, help="Checkpoint path"
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default=None,
        help="YAMLConfig path for cross-verification (optional)",
    )
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--input-size", type=int, nargs=2, default=[640, 640])
    parser.add_argument("--atol", type=float, default=1e-4)
    args = parser.parse_args()

    if args.config:
        print("=== Cross-verification against original engine/ model ===")
        verify_against_original(
            args.resume,
            args.config,
            args.num_classes,
            tuple(args.input_size),
            args.atol,
        )
    else:
        print("=== Standalone load & forward pass ===")
        model = build_deimv2_l(num_classes=args.num_classes)
        print("Loading weights ...")
        missing, unexpected = load_weights(model, args.resume)
        model.deploy()
        wrapper = DEIMv2Wrapper(model).eval()
        h, w = args.input_size
        sample = torch.randn(1, 3, h, w)
        print("Running forward pass ...")
        with torch.no_grad():
            logits, boxes = wrapper(sample)
        print(f"  pred_logits: {tuple(logits.shape)}  dtype={logits.dtype}")
        print(f"  pred_boxes:  {tuple(boxes.shape)}   dtype={boxes.dtype}")
        if not missing and not unexpected:
            print("Weight loading: all keys matched.")
        print("Done.")


if __name__ == "__main__":
    main()
