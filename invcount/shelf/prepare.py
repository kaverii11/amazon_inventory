"""Convert SKU-110K (official train/val/test CSVs) to YOLO format.

- Images are resized so the long side is --max-side px (labels are normalised, so boxes are unaffected);
  this makes training-time decoding ~10x cheaper than the original ~2448x3264 photos.
- Boxes are clipped to the image (about 1.2% of boxes poke outside it in the CSVs).
- Checked beforehand: every CSV width/height equals the raw pixel size and no image has an EXIF rotation.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from PIL import Image, ImageFile

from invcount.shelf.common import CSV_COLS, RAW, YOLO


def box_to_yolo(df: pd.DataFrame) -> np.ndarray:
    """(x1,y1,x2,y2,W,H) pixel boxes -> clipped, normalised (cx, cy, w, h); drops boxes with no area left."""
    W, H = df.W.to_numpy(float), df.H.to_numpy(float)
    x1 = np.clip(df.x1.to_numpy(float), 0, W)
    x2 = np.clip(df.x2.to_numpy(float), 0, W)
    y1 = np.clip(df.y1.to_numpy(float), 0, H)
    y2 = np.clip(df.y2.to_numpy(float), 0, H)
    keep = (x2 > x1) & (y2 > y1)
    out = np.stack([(x1 + x2) / 2 / W, (y1 + y2) / 2 / H, (x2 - x1) / W, (y2 - y1) / H], 1)
    return out[keep]


def convert(job):
    """Returns (n_boxes, truncated). A few SKU-110K JPEGs are truncated; they are decoded with the missing
    tail filled in (PIL LOAD_TRUNCATED_IMAGES) and listed in data/sku110k/yolo/truncated.txt."""
    name, boxes, split, max_side = job
    out_img = YOLO / "images" / split / name
    out_lbl = YOLO / "labels" / split / name.replace(".jpg", ".txt")
    truncated = False
    if not (out_img.exists() and out_lbl.exists()):
        try:
            ImageFile.LOAD_TRUNCATED_IMAGES = False
            im = Image.open(RAW / "images" / name).convert("RGB")
        except OSError:
            ImageFile.LOAD_TRUNCATED_IMAGES = True
            im = Image.open(RAW / "images" / name).convert("RGB")
            truncated = True
        _save(im, boxes, out_img, out_lbl, max_side)
    return len(boxes), truncated


def _save(im, boxes, out_img, out_lbl, max_side):
    s = max_side / max(im.size)
    if s < 1:
        im = im.resize((round(im.width * s), round(im.height * s)), Image.BILINEAR)
    im.save(out_img, quality=95)
    with open(out_lbl, "w") as f:
        f.writelines(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n" for cx, cy, w, h in boxes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-side", type=int, default=1280)
    args = ap.parse_args()
    for split in ("train", "val", "test"):
        (YOLO / "images" / split).mkdir(parents=True, exist_ok=True)
        (YOLO / "labels" / split).mkdir(parents=True, exist_ok=True)
        d = pd.read_csv(RAW / "annotations" / f"annotations_{split}.csv", header=None, names=CSV_COLS)
        jobs = [(n, box_to_yolo(g), split, args.max_side) for n, g in d.groupby("image_name", sort=True)]
        with ProcessPoolExecutor(8) as ex:
            res = list(ex.map(convert, jobs, chunksize=16))
        bad = [j[0] for j, (_, t) in zip(jobs, res) if t]
        if bad:
            with open(YOLO / "truncated.txt", "a") as f:
                f.writelines(f"{split}/{b}\n" for b in bad)
        print(split, len(jobs), "images", sum(n for n, _ in res), "boxes", len(bad), "truncated (this run)", flush=True)
    (YOLO / "sku110k.yaml").write_text(
        f"path: {YOLO}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: object\n")


if __name__ == "__main__":
    main()
