"""Counting metrics shared by the classification (ABID) and detection (SKU-110K) parts."""
import numpy as np


def count_metrics(y_true, y_pred, n_classes=None) -> dict:
    """Per-image counting errors.

    accuracy: exact count; within1: |error| <= 1; mae / rmse in items;
    mae_pct: total absolute error / total true items (MAE as % of items);
    mean_signed: average (pred - true), > 0 means over-counting.
    If n_classes is given, also a confusion matrix (rows = true, cols = predicted).
    """
    t = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    if t.shape != p.shape or t.ndim != 1 or len(t) == 0:
        raise ValueError("y_true and y_pred must be equal-length non-empty 1-D arrays")
    err = p - t
    out = {
        "n": int(len(t)),
        "accuracy": float(np.mean(err == 0)),
        "within1": float(np.mean(np.abs(err) <= 1)),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mean_signed": float(np.mean(err)),
        "mae_pct": float(np.sum(np.abs(err)) / np.sum(t)) if np.sum(t) > 0 else float("nan"),
    }
    if n_classes is not None:
        cm = np.zeros((n_classes, n_classes), dtype=int)
        for a, b in zip(t.astype(int), np.clip(p.astype(int), 0, n_classes - 1)):
            cm[a, b] += 1
        out["confusion"] = cm.tolist()
    return out


def accuracy_ci95(acc: float, n: int) -> float:
    """Half-width of a normal-approximation 95% interval for an accuracy measured on n items."""
    return float(1.96 * np.sqrt(acc * (1 - acc) / n))
