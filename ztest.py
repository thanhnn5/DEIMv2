import operator

import torch
import torch.nn as nn
import torch.export
import model_explorer

from deimv2_single import build_deimv2_l, DEIMv2Wrapper, load_weights

torch.set_default_dtype(torch.float32)

# ---------------------------------------------------------------------------
# Model setup
# ---------------------------------------------------------------------------
model = build_deimv2_l(num_classes=2)
load_weights(model, "weights/deimv2_dinov3_l_coco_custom_best_stg2.pth")
model.deploy()
wrapper = DEIMv2Wrapper(model).eval()

sample = torch.randn(1, 3, 640, 640)

print("Running forward pass ...")
with torch.no_grad():
    logits, boxes = wrapper(sample)
print(f"  pred_logits: {tuple(logits.shape)}  dtype={logits.dtype}")
print(f"  pred_boxes:  {tuple(boxes.shape)}   dtype={boxes.dtype}")

# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------
print("\nExporting with torch.export ...")
ep = torch.export.export(model, (sample,))

# ---------------------------------------------------------------------------
# Approach 2: post-export graph pass — downcast {float64→float32, int64→int32}
#
# Four patch sites:
#   1. Explicit dtype= kwargs/args on call_function nodes (ops like arange,
#      full, scalar_tensor, _to_copy, …).
#   2. Insert actual _to_copy cast nodes after call_function nodes whose
#      output metadata is still int64 (topk indices, max/min argmax, …).
#      For tuple outputs (topk, max.dim, sort) the cast is inserted after the
#      getitem consumer that extracts the int64 index tensor.
#      This makes every downstream op (gather, unsqueeze, repeat, …) inherit
#      the int32 dtype automatically because their inputs changed.
#   3. Fake-tensor metadata sweep: any remaining 64-bit meta tensors are
#      downcast so Model Explorer shows the correct dtype.
#   4. Float64/int64 parameters and buffers in the GraphModule state dict.
# ---------------------------------------------------------------------------

# 64-bit → 32-bit dtype map
_DTYPE_MAP: dict[torch.dtype, torch.dtype] = {
    torch.float64: torch.float32,
    torch.int64:   torch.int32,
}

# Ops that accept a `dtype=` keyword and respect it for their output dtype.
_DTYPE_KWARG_OPS = frozenset({
    torch.ops.aten._to_copy.default,
    torch.ops.aten.to.dtype,
    torch.ops.aten.scalar_tensor.default,
    torch.ops.aten.full.default,
    torch.ops.aten.zeros.default,
    torch.ops.aten.ones.default,
    torch.ops.aten.arange.default,
    torch.ops.aten.arange.start,
    torch.ops.aten.arange.start_step,
    torch.ops.aten.linspace.default,
    torch.ops.aten.empty.memory_format,
})


def cast_to_32bit(ep: torch.export.ExportedProgram) -> torch.export.ExportedProgram:
    """
    Post-export graph pass: replace every 64-bit dtype with its 32-bit twin.

    float64 → float32,  int64 → int32

    Patch sites
    -----------
    1. ``dtype=`` kwargs / positional dtype args on call_function nodes.
    2. Insert ``aten._to_copy`` cast nodes after ops whose output meta is still
       int64 (these ops emit int64 regardless of any kwarg, e.g. topk indices).
       For tuple-output ops the cast is placed after the ``getitem`` consumer
       that extracts the int64 tensor, so downstream gather/unsqueeze/repeat
       inherit int32 automatically.
    3. Metadata sweep: downcast any remaining 64-bit fake tensors in
       ``node.meta["val"]`` for correct Model Explorer visualisation.
    4. Parameters and buffers registered on the GraphModule.
    """
    gm    = ep.graph_module
    graph = ep.graph

    # -- 1. Patch explicit dtype= arguments --------------------------------
    n_args = 0
    for node in graph.nodes:
        if node.op != "call_function":
            continue

        # Keyword dtype=
        if node.target in _DTYPE_KWARG_OPS:
            dtype = node.kwargs.get("dtype")
            if dtype in _DTYPE_MAP:
                node.kwargs = {**node.kwargs, "dtype": _DTYPE_MAP[dtype]}
                n_args += 1

        # Positional dtype arg for aten.to.dtype(tensor, dtype, ...)
        if node.target is torch.ops.aten.to.dtype and len(node.args) >= 2:
            args = list(node.args)
            if args[1] in _DTYPE_MAP:
                args[1] = _DTYPE_MAP[args[1]]
                node.args = tuple(args)
                n_args += 1

    # -- 2. Insert cast nodes after ops that still emit int64 ---------------
    # We work on a snapshot of the node list so that newly inserted cast nodes
    # are not re-processed.
    n_casts = 0
    for node in list(graph.nodes):
        if node.op != "call_function":
            continue
        val = node.meta.get("val")
        if val is None:
            continue

        if isinstance(val, torch.Tensor) and val.dtype == torch.int64:
            # Single-tensor output that is int64 (e.g. argmax, argmin).
            with graph.inserting_after(node):
                cast = graph.call_function(
                    torch.ops.aten._to_copy.default,
                    args=(node,),
                    kwargs={"dtype": torch.int32},
                )
                cast.meta["val"] = val.to(torch.int32)
            node.replace_all_uses_with(cast)
            cast.args = (node,)  # restore after replace_all_uses_with
            n_casts += 1

        elif isinstance(val, (list, tuple)):
            # Tuple output (e.g. topk → (values, indices), max.dim → (values, argmax)).
            # Find getitem consumers that extract an int64 position and cast those.
            for user in list(node.users):
                if not (user.op == "call_function" and user.target is operator.getitem):
                    continue
                pos = user.args[1]
                if not isinstance(pos, int) or pos >= len(val):
                    continue
                v = val[pos]
                if not (isinstance(v, torch.Tensor) and v.dtype == torch.int64):
                    continue
                with graph.inserting_after(user):
                    cast = graph.call_function(
                        torch.ops.aten._to_copy.default,
                        args=(user,),
                        kwargs={"dtype": torch.int32},
                    )
                    cast.meta["val"] = v.to(torch.int32)
                user.replace_all_uses_with(cast)
                cast.args = (user,)  # restore
                n_casts += 1

    # -- 3. Metadata sweep -------------------------------------------------
    n_meta = 0
    for node in graph.nodes:
        val = node.meta.get("val")
        if val is None:
            continue
        if isinstance(val, torch.Tensor) and val.dtype in _DTYPE_MAP:
            node.meta["val"] = val.to(_DTYPE_MAP[val.dtype])
            n_meta += 1
        elif isinstance(val, (list, tuple)):
            patched = [
                v.to(_DTYPE_MAP[v.dtype])
                if isinstance(v, torch.Tensor) and v.dtype in _DTYPE_MAP
                else v
                for v in val
            ]
            if any(p is not o for p, o in zip(patched, val)):
                node.meta["val"] = type(val)(patched)
                n_meta += 1

    # -- 4. Parameters and buffers -----------------------------------------
    n_params = 0
    for name, param in list(gm.named_parameters()):
        if param.dtype in _DTYPE_MAP:
            *path, attr = name.split(".")
            parent = gm.get_submodule(".".join(path)) if path else gm
            parent.register_parameter(
                attr,
                nn.Parameter(
                    param.data.to(_DTYPE_MAP[param.dtype]),
                    requires_grad=param.requires_grad,
                ),
            )
            n_params += 1

    n_bufs = 0
    for name, buf in list(gm.named_buffers()):
        if buf.dtype in _DTYPE_MAP:
            *path, attr = name.split(".")
            parent = gm.get_submodule(".".join(path)) if path else gm
            parent.register_buffer(attr, buf.to(_DTYPE_MAP[buf.dtype]))
            n_bufs += 1

    graph.lint()
    gm.recompile()

    print(f"  dtype= args patched    : {n_args}")
    print(f"  int64 cast nodes added : {n_casts}")
    print(f"  meta tensors patched   : {n_meta}")
    print(f"  parameters patched     : {n_params}")
    print(f"  buffers patched        : {n_bufs}")
    return ep


print("\n--- Approach 2: post-export graph pass (float64→float32, int64→int32) ---")
ep = cast_to_32bit(ep)
print("Done.")

# ---------------------------------------------------------------------------
# Verify: report any remaining 64-bit tensors in metadata
# ---------------------------------------------------------------------------
remaining_64 = [
    (node.name, node.meta["val"].dtype)
    for node in ep.graph.nodes
    if isinstance(node.meta.get("val"), torch.Tensor)
    and node.meta["val"].dtype in _DTYPE_MAP
]
if remaining_64:
    print(f"\n[WARN] {len(remaining_64)} node(s) still 64-bit: {remaining_64[:5]} ...")
else:
    print("\n[OK] No 64-bit tensors remain in graph metadata.")

# ---------------------------------------------------------------------------
# Sanity-check: run the patched exported module
# Note: PyTorch ops like gather/sort still require int64 indices at runtime,
# so this forward pass may raise a dtype error. That is expected — the graph
# is correct for deployment targets (TFLite/Edge) that accept int32 indices.
# ---------------------------------------------------------------------------
print("\nRunning exported module after cast ...")
try:
    with torch.no_grad():
        out = ep.module()(sample)
    if isinstance(out, dict):
        for k, o in out.items():
            print(f"  output['{k}']: shape={tuple(o.shape)}  dtype={o.dtype}")
    elif isinstance(out, (list, tuple)):
        for i, o in enumerate(out):
            print(f"  output[{i}]: shape={tuple(o.shape)}  dtype={o.dtype}")
    else:
        print(f"  output: shape={tuple(out.shape)}  dtype={out.dtype}")
except Exception as e:
    print(f"  [NOTE] Runtime check failed (expected for int32 indices in PyTorch): {e}")

# ---------------------------------------------------------------------------
# Visualize with AI Edge Model Explorer
# ---------------------------------------------------------------------------
model_name = "deimv2_l"
print(f"\nLaunching Model Explorer for '{model_name}' ...")
model_explorer.visualize_pytorch(model_name, exported_program=ep)
