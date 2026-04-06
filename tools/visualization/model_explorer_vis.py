"""
Model Explorer Visualization Tool

Visualize PyTorch or TFLite models using the AI Edge Model Explorer.

Usage (TFLite):
    python tools/visualization/model_explorer_vis.py \
        --model weights/deimv2_l.tflite

Usage (PyTorch):
    python tools/visualization/model_explorer_vis.py \
        --pytorch \
        -c configs/deimv2/deimv2_dinov3_l_coco.yml \
        -r path/to/checkpoint.pth \
        --input-size 640 640

Requirements:
    pip install ai-edge-model-explorer
    pip install ai-edge-model-explorer-adapter  # for PyTorch support
"""

import argparse
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))


def visualize_tflite(model_path: str) -> None:
    import model_explorer

    if not os.path.isfile(model_path):
        print(f"[ERROR] TFLite model not found: {model_path}")
        sys.exit(1)

    print(f"[INFO] Visualizing TFLite model: {model_path}")
    model_explorer.visualize(model_path)


def visualize_pytorch(
    config: str,
    checkpoint: str,
    input_size: list[int],
    model_name: str,
    device: str,
) -> None:
    import torch
    import model_explorer

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

    from engine.core import YAMLConfig  # type: ignore

    print(f"[INFO] Loading model from config: {config}")
    cfg = YAMLConfig(config, resume=checkpoint)

    if checkpoint:
        state = torch.load(checkpoint, map_location="cpu")
        state_dict = state.get("ema", state).get("module", state)
        cfg.model.load_state_dict(state_dict)

    model = cfg.model.eval().to(device)

    h, w = input_size
    dummy_input = (torch.rand(1, 3, h, w).to(device),)

    print(f"[INFO] Exporting model with torch.export (input size: {h}x{w})")
    ep = torch.export.export(model, dummy_input)

    print(f"[INFO] Launching Model Explorer for '{model_name}'")
    model_explorer.visualize_pytorch(model_name, exported_program=ep)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize a TFLite or PyTorch model with AI Edge Model Explorer"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to a .tflite model file (TFLite mode)",
    )
    parser.add_argument(
        "--pytorch",
        action="store_true",
        help="Visualize a PyTorch model (requires -c and -r)",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default=None,
        help="Path to the YAML config file (PyTorch mode)",
    )
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        default=None,
        help="Path to the checkpoint .pth file (PyTorch mode)",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        nargs=2,
        default=[640, 640],
        metavar=("H", "W"),
        help="Input image size as H W (default: 640 640)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="deimv2",
        help="Name shown in Model Explorer (PyTorch mode, default: deimv2)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for model loading in PyTorch mode (default: cpu)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.pytorch:
        if not args.config:
            print("[ERROR] --config is required in PyTorch mode.")
            sys.exit(1)
        visualize_pytorch(
            config=args.config,
            checkpoint=args.resume,
            input_size=args.input_size,
            model_name=args.model_name,
            device=args.device,
        )
    elif args.model:
        visualize_tflite(args.model)
    else:
        print("[ERROR] Provide --model <path.tflite> or --pytorch with --config.")
        sys.exit(1)


if __name__ == "__main__":
    main()
