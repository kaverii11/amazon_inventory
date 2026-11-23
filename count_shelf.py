"""Count products on a shelf photo.

    python count_shelf.py photo.jpg [--model weights/sku110k_yolo11n.onnx] [--out photo_boxes.jpg]

Prints the count and saves a copy of the photo with a box around every detected product.
The confidence threshold defaults to the one picked on the SKU-110K validation set (results/sku110k/threshold_val.json).
"""
import argparse
import json
from pathlib import Path

import cv2

from invcount.shelf.onnx_infer import detect, session

ROOT = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--model", default=str(ROOT / "weights" / "sku110k_yolo11n.onnx"))
    ap.add_argument("--conf", type=float, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    conf = args.conf
    if conf is None:
        conf = json.load(open(ROOT / "results" / "sku110k" / "threshold_val.json"))["threshold"]
    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f"could not read {args.image}")
    boxes, scores = detect(session(args.model), img, conf)
    t = max(1, round(max(img.shape[:2]) / 600))
    for x1, y1, x2, y2 in boxes.astype(int):
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 0), t)
    label = f"{len(boxes)} products"
    cv2.putText(img, label, (10, 40 * t), cv2.FONT_HERSHEY_SIMPLEX, 1.2 * t, (0, 0, 0), 4 * t)
    cv2.putText(img, label, (10, 40 * t), cv2.FONT_HERSHEY_SIMPLEX, 1.2 * t, (255, 255, 255), 2 * t)
    out = args.out or str(Path(args.image).with_name(Path(args.image).stem + "_boxes.jpg"))
    cv2.imwrite(out, img)
    print(f"{len(boxes)} products (confidence >= {conf:.2f}); saved {out}")


if __name__ == "__main__":
    main()
