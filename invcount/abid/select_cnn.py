"""Pick the CNN run with the highest val accuracy across results/abid/cnn_*.json -> cnn_selected.json."""
import json

from invcount.abid.common import RESULTS


def main():
    runs = []
    for p in sorted(RESULTS.glob("cnn_resnet*.json")):
        d = json.load(open(p))
        runs.append({"tag": p.stem, "arch": d["args"]["arch"], "lr": d["args"]["lr"], "epochs": d["args"]["epochs"],
                     "best_epoch": d["best_epoch"], "val": d["history"][d["best_epoch"]]})
    for r in runs:
        print(f"{r['tag']:45s} best epoch {r['best_epoch']:2d}  val acc {r['val']['val_acc']:.3f}  "
              f"±1 {r['val']['val_within1']:.3f}  MAE {r['val']['val_mae']:.3f}")
    best = max(runs, key=lambda r: r["val"]["val_acc"])
    json.dump({**best, "all_runs": runs}, open(RESULTS / "cnn_selected.json", "w"), indent=1)
    print("selected:", best["tag"])


if __name__ == "__main__":
    main()
