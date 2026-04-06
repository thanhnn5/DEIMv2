"""
DEIMv2-L TFLite inference with OpenCV pre/postprocessing.

All operations are intentionally written to mirror what you would do on Android:
  Python cv2          →  Android OpenCV (Imgproc / Core)
  numpy float ops     →  float[] / FloatBuffer in Java/Kotlin
  TFLite Interpreter  →  org.tensorflow.lite.Interpreter

Usage:
    python tools/inference/tflite_inf.py \
        -m deimv2_l.tflite \
        -i image.jpg \
        --score-thresh 0.45

Requirements:
    pip install tflite-runtime opencv-python numpy
    # or: pip install tensorflow (includes tflite)
"""

import argparse

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# TFLite runtime — prefer lightweight tflite-runtime, fall back to tensorflow
# ---------------------------------------------------------------------------
try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter


# ---------------------------------------------------------------------------
# Constants  (must match training config)
# ---------------------------------------------------------------------------
INPUT_SIZE   = (640, 640)          # (H, W)
MEAN         = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD          = np.array([0.229, 0.224, 0.225], dtype=np.float32)
NUM_CLASSES  = 2
NUM_QUERIES  = 300


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
def preprocess(bgr_image: np.ndarray, input_size=INPUT_SIZE):
    """
    BGR uint8 HWC  →  float32 NCHW (or NHWC, see note below)

    Android equivalent:
        Imgproc.resize(mat, resized, new Size(W, H))
        Imgproc.cvtColor(resized, rgb, Imgproc.COLOR_BGR2RGB)
        normalize each channel: pixel = (pixel/255 - mean) / std
        copy into FloatBuffer row-major

    NOTE: litert-torch preserves PyTorch's NCHW layout by default.
          Check your model's actual input shape with inspect_model() below
          and transpose to NHWC if needed.
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
    #    Android: manually fill FloatBuffer in CHW order
    chw = rgb.transpose(2, 0, 1)       # (3, H, W)
    return chw[np.newaxis].copy()      # (1, 3, H, W), contiguous


# ---------------------------------------------------------------------------
# Postprocessing
# ---------------------------------------------------------------------------
def postprocess(
    pred_logits: np.ndarray,
    pred_boxes: np.ndarray,
    orig_hw: tuple[int, int],
    score_thresh: float = 0.45,
    num_top_queries: int = NUM_QUERIES,
):
    """
    Mirror of PostProcessor.forward() in pure numpy.

    pred_logits  (1, 300, 2)  — raw logits (before sigmoid)
    pred_boxes   (1, 300, 4)   — normalised [cx, cy, w, h]
    orig_hw      (H, W)        — original image dimensions

    Returns list of dicts: [{labels, boxes, scores}] one per image.

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
    scores_flat = scores.flatten()                    # (300*2,)
    topk_idx    = np.argpartition(scores_flat, -num_top_queries)[-num_top_queries:]
    topk_idx    = topk_idx[np.argsort(scores_flat[topk_idx])[::-1]]

    topk_scores = scores_flat[topk_idx]               # (300,)
    labels      = topk_idx % NUM_CLASSES              # which class
    query_idx   = topk_idx // NUM_CLASSES             # which query

    # 3. Decode boxes: cxcywh (normalised) → xyxy (pixels)
    sel_boxes = boxes[query_idx]                      # (300, 4)
    cx, cy, bw, bh = sel_boxes[:, 0], sel_boxes[:, 1], sel_boxes[:, 2], sel_boxes[:, 3]
    x1 = (cx - bw / 2) * orig_w
    y1 = (cy - bh / 2) * orig_h
    x2 = (cx + bw / 2) * orig_w
    y2 = (cy + bh / 2) * orig_h
    xyxy = np.stack([x1, y1, x2, y2], axis=1)        # (300, 4)

    # 4. Threshold filter
    mask        = topk_scores >= score_thresh
    return {
        'labels': labels[mask],
        'boxes':  xyxy[mask],
        'scores': topk_scores[mask],
    }


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def inspect_model(interpreter: Interpreter):
    """Print input/output tensor details — useful to confirm NCHW vs NHWC."""
    print("── Inputs ──")
    for d in interpreter.get_input_details():
        print(f"  idx={d['index']}  name={d['name']}  shape={d['shape']}  dtype={d['dtype']}")
    print("── Outputs ──")
    for d in interpreter.get_output_details():
        print(f"  idx={d['index']}  name={d['name']}  shape={d['shape']}  dtype={d['dtype']}")


def draw_detections(image: np.ndarray, detections: dict, score_thresh: float):
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
    # Load TFLite model
    interpreter = Interpreter(model_path=args.model, num_threads=args.threads)
    interpreter.allocate_tensors()
    inspect_model(interpreter)

    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Check if model expects NHWC (TFLite convention) or NCHW (litert-torch default)
    input_shape = input_details[0]['shape']   # e.g. [1, 3, 640, 640] or [1, 640, 640, 3]
    use_nhwc = (input_shape[-1] == 3)

    # Load image
    bgr = cv2.imread(args.input)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {args.input}")
    orig_h, orig_w = bgr.shape[:2]

    # Preprocess
    nchw = preprocess(bgr, INPUT_SIZE)        # (1, 3, H, W)
    if use_nhwc:
        model_input = nchw.transpose(0, 2, 3, 1)   # → (1, H, W, 3)
    else:
        model_input = nchw

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], model_input)
    interpreter.invoke()

    pred_boxes  = interpreter.get_tensor(output_details[0]['index'])  # (1, 300, 4)
    pred_logits = interpreter.get_tensor(output_details[1]['index'])  # (1, 300, 2)

    # Postprocess
    detections = postprocess(pred_logits, pred_boxes, (orig_h, orig_w), args.score_thresh)
    print(f"Detected {len(detections['labels'])} objects")
    for label, box, score in zip(detections['labels'], detections['boxes'], detections['scores']):
        print(f"  class={label:3d}  score={score:.3f}  box={box.astype(int).tolist()}")

    # Visualise
    vis = draw_detections(bgr, detections, args.score_thresh)
    out_path = "tflite_results.jpg"
    cv2.imwrite(out_path, vis)
    print(f"Saved: {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DEIMv2-L TFLite inference')
    parser.add_argument('-m', '--model',  type=str, required=True,  help='Path to .tflite model')
    parser.add_argument('-i', '--input',  type=str, required=True,  help='Input image path')
    parser.add_argument('--score-thresh', type=float, default=0.45, help='Score threshold (default: 0.45)')
    parser.add_argument('--threads',      type=int,   default=4,    help='TFLite interpreter threads')
    args = parser.parse_args()
    main(args)
