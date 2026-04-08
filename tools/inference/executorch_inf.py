"""
DEIMv2-L ExecuTorch (.pte) inference with OpenCV pre/postprocessing.

All operations are intentionally written to mirror what you would do on Android:
  Python cv2           →  Android OpenCV (Imgproc / Core)
  numpy / torch ops    →  float[] / FloatBuffer / EValue in Java/Kotlin
  executorch.runtime   →  com.facebook.executorch.Module

Usage:
    python tools/inference/executorch_inf.py \
        -m deimv2_l_coreml.pte \
        -i image.jpg \
        --score-thresh 0.45

Requirements:
    pip install executorch opencv-python numpy torch
"""

import argparse

import cv2
import numpy as np
import torch
from executorch.runtime import Runtime


# ---------------------------------------------------------------------------
# Constants  (must match training config)
# ---------------------------------------------------------------------------
INPUT_SIZE  = (640, 640)          # (H, W)
MEAN        = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD         = np.array([0.229, 0.224, 0.225], dtype=np.float32)
NUM_CLASSES = 2
NUM_QUERIES = 300


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
def preprocess(bgr_image: np.ndarray, input_size=INPUT_SIZE) -> torch.Tensor:
    """
    BGR uint8 HWC  →  float32 NCHW torch.Tensor

    Android equivalent:
        Imgproc.resize(mat, resized, new Size(W, H))
        Imgproc.cvtColor(resized, rgb, Imgproc.COLOR_BGR2RGB)
        normalize each channel: pixel = (pixel/255 - mean) / std
        wrap in EValue(Tensor.fromBlob(floatArray, new long[]{1,3,H,W}))
    """
    h, w = input_size

    # 1. Resize  (Android: Imgproc.resize)
    resized = cv2.resize(bgr_image, (w, h), interpolation=cv2.INTER_LINEAR)

    # 2. BGR → RGB  (Android: Imgproc.cvtColor COLOR_BGR2RGB)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    # 3. uint8 → float32 in [0, 1]
    rgb = rgb.astype(np.float32) / 255.0

    # 4. Normalise  (Android: per-pixel: (v - mean) / std)
    rgb = (rgb - MEAN) / STD           # HWC, float32

    # 5. HWC → CHW, add batch dim  →  shape (1, 3, H, W)
    #    Android: fill FloatBuffer in CHW order, wrap as Tensor
    chw = rgb.transpose(2, 0, 1)      # (3, H, W)
    return torch.from_numpy(chw[np.newaxis].copy())   # (1, 3, H, W)


# ---------------------------------------------------------------------------
# Postprocessing
# ---------------------------------------------------------------------------
def postprocess(
    pred_logits: np.ndarray,
    pred_boxes: np.ndarray,
    orig_hw: tuple[int, int],
    score_thresh: float = 0.45,
    num_top_queries: int = NUM_QUERIES,
) -> dict:
    """
    Mirror of PostProcessor.forward() in pure numpy.

    pred_logits  (1, 300, 2)  — raw logits (before sigmoid)
    pred_boxes   (1, 300, 4)  — normalised [cx, cy, w, h]
    orig_hw      (H, W)       — original image dimensions

    Android equivalent:
        sigmoid: 1f / (1f + Math.exp(-x))
        topk: sort flat scores array, take top-k indices
        box decode: xyxy = [(cx-w/2)*W, (cy-h/2)*H, (cx+w/2)*W, (cy+h/2)*H]
    """
    orig_h, orig_w = orig_hw

    # Remove batch dim
    logits = pred_logits[0]   # (300, 2)
    boxes  = pred_boxes[0]    # (300, 4)

    # 1. Sigmoid  (Android: 1f / (1f + exp(-v)))
    scores = 1.0 / (1.0 + np.exp(-logits.astype(np.float64))).astype(np.float32)
                               # (300, 2)

    # 2. Flatten and top-k across all (query, class) pairs
    scores_flat = scores.flatten()                     # (300*2,)
    topk_idx    = np.argpartition(scores_flat, -num_top_queries)[-num_top_queries:]
    topk_idx    = topk_idx[np.argsort(scores_flat[topk_idx])[::-1]]

    topk_scores = scores_flat[topk_idx]                # (300,)
    labels      = topk_idx % NUM_CLASSES               # which class
    query_idx   = topk_idx // NUM_CLASSES              # which query

    # 3. Decode boxes: cxcywh (normalised) → xyxy (pixels)
    sel_boxes = boxes[query_idx]                       # (300, 4)
    cx, cy, bw, bh = sel_boxes[:, 0], sel_boxes[:, 1], sel_boxes[:, 2], sel_boxes[:, 3]
    x1 = (cx - bw / 2) * orig_w
    y1 = (cy - bh / 2) * orig_h
    x2 = (cx + bw / 2) * orig_w
    y2 = (cy + bh / 2) * orig_h
    xyxy = np.stack([x1, y1, x2, y2], axis=1)         # (300, 4)

    # 4. Threshold filter
    mask = topk_scores >= score_thresh
    return {
        'labels': labels[mask],
        'boxes':  xyxy[mask],
        'scores': topk_scores[mask],
    }


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def inspect_model(method):
    """Print input/output metadata — useful for confirming shapes and dtypes."""
    meta = method.metadata
    print("── Method metadata ──")
    print(f"  {meta}")


def draw_detections(image: np.ndarray, detections: dict) -> np.ndarray:
    vis = image.copy()
    for label, box, score in zip(detections['labels'], detections['boxes'], detections['scores']):
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(vis, f"{label} {score:.2f}", (x1, max(y1 - 4, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return vis


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(args):
    # Load ExecuTorch model
    # Android equivalent:
    #   Module module = Module.load("/data/.../deimv2_l.pte");
    runtime = Runtime.get()
    program = runtime.load_program(args.model)
    method  = program.load_method('forward')
    print(f"Loaded: {args.model}")
    print(f"  Available methods: {program.method_names}")
    inspect_model(method)

    # Load image
    bgr = cv2.imread(args.input)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {args.input}")
    orig_h, orig_w = bgr.shape[:2]

    # Preprocess
    # Android: build EValue from CHW FloatBuffer
    input_tensor = preprocess(bgr, INPUT_SIZE)    # torch.Tensor (1, 3, H, W)
    print(f"Input: {tuple(input_tensor.shape)}  dtype={input_tensor.dtype}")

    # Run inference
    # Android: EValue[] outputs = module.forward(new EValue[]{EValue.from(tensor)});
    outputs = method.execute([input_tensor])

    pred_logits = outputs[0].numpy()   # (1, 300, 2)
    pred_boxes  = outputs[1].numpy()   # (1, 300, 4)
    print(f"pred_logits: {pred_logits.shape}  pred_boxes: {pred_boxes.shape}")

    # Postprocess
    detections = postprocess(pred_logits, pred_boxes, (orig_h, orig_w), args.score_thresh)
    print(f"Detected {len(detections['labels'])} objects (thresh={args.score_thresh})")
    for label, box, score in zip(detections['labels'], detections['boxes'], detections['scores']):
        print(f"  class={label:3d}  score={score:.3f}  box={box.astype(int).tolist()}")

    # Visualise
    vis      = draw_detections(bgr, detections)
    out_path = args.output or "executorch_results.jpg"
    cv2.imwrite(out_path, vis)
    print(f"Saved: {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DEIMv2-L ExecuTorch inference')
    parser.add_argument('-m', '--model',       type=str, required=True,
                        help='Path to .pte model')
    parser.add_argument('-i', '--input',       type=str, required=True,
                        help='Input image path')
    parser.add_argument('-o', '--output',      type=str, default=None,
                        help='Output image path (default: executorch_results.jpg)')
    parser.add_argument('--score-thresh',      type=float, default=0.45,
                        help='Score threshold (default: 0.45)')
    args = parser.parse_args()
    main(args)
