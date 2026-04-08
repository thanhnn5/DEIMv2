import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from tabulate import tabulate
from executorch.devtools.backend_debug.delegation_info import get_delegation_info

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from engine.core import YAMLConfig


BACKENDS = ('xnnpack', 'vulkan', 'coreml', 'metal', 'mps', 'none')


class DEIMv2Wrapper(nn.Module):
    """
    Thin wrapper around the core DEIM model (backbone + encoder + decoder).
    No preprocessing or postprocessing is included.

    Input:
        images  (B, 3, H, W) — already-normalised image tensor

    Outputs:
        pred_logits  (B, num_queries, num_classes)
        pred_boxes   (B, num_queries, 4)  — normalised [cx, cy, w, h]
    """

    def __init__(self, cfg):
        super().__init__()
        self.model = cfg.model.deploy()

    def forward(self, images: torch.Tensor):
        outputs = self.model(images)
        return outputs['pred_logits'], outputs['pred_boxes']


def load_model(config_path: str, checkpoint_path: str) -> DEIMv2Wrapper:
    cfg = YAMLConfig(config_path, resume=checkpoint_path)

    if 'HGNetv2' in cfg.yaml_cfg:
        cfg.yaml_cfg['HGNetv2']['pretrained'] = False

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state = checkpoint['ema']['module'] if 'ema' in checkpoint else checkpoint['model']
    cfg.model.load_state_dict(state)

    return DEIMv2Wrapper(cfg).eval()


def get_partitioner(backend: str):
    if backend == 'xnnpack':
        from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
        return XnnpackPartitioner()
    elif backend == 'vulkan':
        from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner
        return VulkanPartitioner(
            compile_options={
                "force_fp16": False,
                "small_texture_limits": True,
                "downcast_64_bit": True,
                "require_dynamic_shapes": False,
                "skip_bool_tensors": True,
            }
        )
    elif backend == 'coreml':
        import coremltools as ct
        from executorch.backends.apple.coreml.compiler import CoreMLBackend
        from executorch.backends.apple.coreml.partition.coreml_partitioner import CoreMLPartitioner
        return CoreMLPartitioner(
            lower_full_graph=True,
            compile_specs=CoreMLBackend.generate_compile_specs(
                compute_precision=ct.precision.FLOAT32  # try FP32 first to isolate the issue
            )
        )
    elif backend == 'metal':
        # WIP: does not work for now, rely on torchao dylib, which currently not exist
        from executorch.backends.apple.metal.metal_partitioner import MetalPartitioner
        return MetalPartitioner([])
    elif backend == 'mps':
        raise NotImplementedError("MPS backend is not supported yet.")
    elif backend == 'none':
        return None
    else:
        raise ValueError(f"Unknown backend: {backend!r}. Choose from {BACKENDS}.")


def verify_outputs(torch_out: tuple, et_out, atol: float = 1e-4) -> bool:
    names = ['pred_logits', 'pred_boxes']
    all_close = True
    for i, (t_tensor, name) in enumerate(zip(torch_out, names)):
        t_np = t_tensor.detach().cpu().numpy()
        e_np = et_out[i].detach().cpu().numpy() if isinstance(et_out[i], torch.Tensor) else np.array(et_out[i])
        close = np.allclose(t_np, e_np, atol=atol)
        max_diff = float(np.max(np.abs(t_np - e_np)))
        status = "PASS" if close else "FAIL"
        print(f"  [{status}] {name:10s}  shape={t_np.shape}  max_diff={max_diff:.6f}  atol={atol}")
        if not close:
            all_close = False
    return all_close


def main(args):
    from executorch.exir import to_edge_transform_and_lower

    device = torch.device(args.device)
    h, w = args.input_size

    print(f"Loading model from config: {args.config}, checkpoint: {args.resume} ...")
    model = load_model(args.config, args.resume).to(device)

    sample_inputs = (torch.randn(1, 3, h, w, device=device),)
    print(f"Sample input: {tuple(sample_inputs[0].shape)}")

    print("\nRunning PyTorch reference inference ...")
    with torch.no_grad():
        torch_out = model(*sample_inputs)
    for name, t in zip(['pred_logits', 'pred_boxes'], torch_out):
        print(f"  {name:10s}  shape={tuple(t.shape)}  dtype={t.dtype}")

    print("\nExporting with torch.export.export ...")
    exported_program = torch.export.export(model, sample_inputs)
    print("Export successful.")

    partitioner = get_partitioner(args.backend)
    partitioner_list = [partitioner] if partitioner is not None else []

    backend_label = args.backend if args.backend != 'none' else 'no-delegate'
    print(f"\nLowering to ExecuTorch (backend: {backend_label}) ...")
    et_program = to_edge_transform_and_lower(
        exported_program,
        partitioner=partitioner_list,
    ).to_executorch()
    print("Lowering successful.")

    if args.show_delegation_info:
        graph_module = et_program.exported_program().graph_module
        delegation_info = get_delegation_info(graph_module)
        print(delegation_info.get_summary())
        df = delegation_info.get_operator_delegation_dataframe()
        print(tabulate(df, headers="keys", tablefmt="fancy_grid"))

    if args.verify:
        print("\nVerifying output tolerance ...")
        et_out = et_program.exported_program().module()(*sample_inputs)
        passed = verify_outputs(torch_out, et_out, atol=args.atol)
        if passed:
            print("Verification PASSED.")
        else:
            print(
                f"Some outputs exceeded tolerance (atol={args.atol}). "
                "Consider raising --atol or checking for unsupported ops."
            )

    with open(args.output, 'wb') as f:
        et_program.write_to_file(f)

    size_mb = os.path.getsize(args.output) / (1024 ** 2)
    print(f"\nSaved to: {args.output}  ({size_mb:.1f} MB)")
    print("Done.")

    # Round-trip: reload and run once more to confirm the file is valid
    print("\nVerifying saved .pte file (round-trip load) ...")
    from executorch.runtime import Runtime

    runtime = Runtime.get()
    program = runtime.load_program(args.output)
    method  = program.load_method('forward')
    outputs = method.execute([sample_inputs[0]])
    verify_outputs(torch_out, outputs, atol=args.atol)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Export DEIMv2 to ExecuTorch (.pte) format'
    )
    parser.add_argument('-c', '--config', type=str, required=True,
                        help='Path to YAML config (e.g. configs/deimv2/deimv2_dinov3_l_coco.yml)')
    parser.add_argument('-r', '--resume', type=str, required=True,
                        help='Path to model checkpoint (.pth)')
    parser.add_argument('-o', '--output', type=str, default='deimv2.pte',
                        help='Output .pte file path (default: deimv2.pte)')
    parser.add_argument('--input-size', type=int, nargs=2, default=[640, 640],
                        metavar=('H', 'W'),
                        help='Input spatial size H W (default: 640 640)')
    parser.add_argument('--backend', type=str, default='xnnpack', choices=BACKENDS,
                        help='ExecuTorch delegate backend (default: xnnpack)')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Torch device for export (default: cpu)')
    parser.add_argument('--verify', action='store_true',
                        help='Run output verification after lowering')
    parser.add_argument('--show-delegation-info', action='store_true',
                        help='Show delegation info after lowering')
    parser.add_argument('--atol', type=float, default=1e-4,
                        help='Absolute tolerance for output verification (default: 1e-4)')
    args = parser.parse_args()
    main(args)
