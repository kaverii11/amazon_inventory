"""ONE-TIME evaluation of the selected Phase 1 models on the held-out test sets.

Models: constant baselines, the three SVMs (configs picked on val by svm.py) and the CNN run named in
results/abid/cnn_selected.json (picked on val). Writes results/abid/final_test.json and confusion-matrix
PNGs. Refuses to run twice unless --force, so test numbers can't be quietly re-tuned.
"""
import argparse
import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from invcount.abid.cache import load_split
from invcount.abid.cnn import build, device, predict
from invcount.abid.common import DATA, RESULTS
from invcount.abid.svm import cached_features
from invcount.metrics import accuracy_ci95, count_metrics

MODELS = DATA / "models"


def plot_cm(cm, title, path):
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    ax.imshow(cm, cmap="Blues")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xlabel("predicted count")
    ax.set_ylabel("true count")
    ax.set_title(title, fontsize=9)
    ax.set_xticks(range(6))
    ax.set_yticks(range(6))
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    out_path = RESULTS / "final_test.json"
    if out_path.exists() and not args.force:
        raise SystemExit(f"{out_path} exists: test set already used once. Use --force only to regenerate identical numbers.")

    _, ytr, _ = load_split("train")
    _, yva, _ = load_split("val")
    sel = json.load(open(RESULTS / "cnn_selected.json"))
    # constant baselines: most frequent train class (ties -> smallest), and the constant with lowest val MAE
    vals, counts = np.unique(ytr, return_counts=True)
    majority = int(vals[np.argmax(counts)])
    best_const = int(min(range(6), key=lambda c: np.mean(np.abs(yva - c))))

    dev = device()
    net = build(sel["arch"], pretrained=False).to(dev)
    net.load_state_dict(torch.load(MODELS / f"{sel['tag']}.pt", map_location=dev))

    res = {"selection": {"majority_class": majority, "best_constant_on_val": best_const, "cnn": sel}}
    fig_dir = RESULTS / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    for split in ("test", "test_natural"):
        X, y, _ = load_split(split)
        preds = {
            f"majority class ({majority})": np.full_like(y, majority),
            f"constant {best_const} (best on val)": np.full_like(y, best_const),
        }
        for feat in ("blob", "hog", "pixels"):
            preds[f"SVM {feat}"] = joblib.load(MODELS / f"svm_{feat}.joblib").predict(cached_features(feat, split))
        preds[f"CNN {sel['arch']}"] = predict(net, X, dev)
        res[split] = {}
        for name, p in preds.items():
            m = count_metrics(y, p, n_classes=6)
            m["accuracy_ci95"] = accuracy_ci95(m["accuracy"], m["n"])
            res[split][name] = m
            print(f"{split:13s} {name:32s} acc {m['accuracy']:.3f} ±{m['accuracy_ci95']:.3f}  "
                  f"±1 {m['within1']:.3f}  MAE {m['mae']:.3f}  RMSE {m['rmse']:.3f}")
            if name.startswith(("SVM", "CNN")):
                plot_cm(m["confusion"], f"{name} - {split} (n={m['n']})",
                        fig_dir / f"cm_{split}_{name.split()[0].lower()}_{name.split()[1]}.png")
    json.dump(res, open(out_path, "w"), indent=1)


if __name__ == "__main__":
    main()
