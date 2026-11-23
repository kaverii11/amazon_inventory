"""Matching, precision/recall and counting helpers for detection results (pure numpy, no dataset needed)."""
import numpy as np

from invcount.metrics import count_metrics


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between boxes a (N,4) and b (M,4) in x1,y1,x2,y2."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-12)


def match(pred: np.ndarray, scores: np.ndarray, gt: np.ndarray, iou_thr=0.5) -> int:
    """Greedy match in descending score order (COCO style). Returns number of true positives."""
    if len(pred) == 0 or len(gt) == 0:
        return 0
    order = np.argsort(-scores)
    ious = iou_matrix(pred[order], gt)
    taken = np.zeros(len(gt), bool)
    tp = 0
    for i in range(len(order)):
        cand = np.where(~taken & (ious[i] >= iou_thr))[0]
        if len(cand):
            taken[cand[np.argmax(ious[i, cand])]] = True
            tp += 1
    return tp


def pr_at_threshold(preds, gts, thr, iou_thr=0.5) -> dict:
    """Precision/recall over a dataset at one confidence threshold. preds: list of (boxes, scores)."""
    tp = fp = fn = 0
    for (b, s), g in zip(preds, gts):
        k = s >= thr
        t = match(b[k], s[k], g, iou_thr)
        tp += t
        fp += int(k.sum()) - t
        fn += len(g) - t
    return {"precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1), "tp": tp, "fp": fp, "fn": fn}


def counts_at(preds, thr) -> np.ndarray:
    return np.array([int((s >= thr).sum()) for _, s in preds])


def pick_count_threshold(preds, true_counts, grid=None) -> tuple[float, list]:
    """Confidence threshold minimising per-image count MAE (use on VAL only). Returns (best, table)."""
    grid = np.round(np.arange(0.05, 0.951, 0.01), 2) if grid is None else grid
    table = [(float(t), count_metrics(true_counts, counts_at(preds, t))["mae"]) for t in grid]
    best = min(table, key=lambda r: r[1])[0]
    return best, table
