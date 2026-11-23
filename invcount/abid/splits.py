"""Build fixed, class-balanced train/val/test splits (+ a natural-distribution test set).

Input: data/abid/metadata_sample.csv from scan_metadata.py.
Output: splits/abid/{train,val,test,test_natural}.csv with columns id,quantity (committed to git).
"""
import argparse

import numpy as np
import pandas as pd

from invcount.abid.common import CLASSES, DATA, SPLITS


def make_splits(meta: pd.DataFrame, per_class: int, fracs=(0.70, 0.15, 0.15),
                n_natural: int = 0, seed: int = 0) -> dict:
    """Return {"train","val","test"[,"test_natural"]} DataFrames of (id, quantity).

    Balanced part: `per_class` images of each count 0..5, split per class by `fracs`, so each split
    is balanced too. Natural part: `n_natural` images drawn at random from the whole pool before the balanced part,
    so it keeps the dataset's own mix of counts (used only to compare with the paper, which used the natural mix).
    All splits are disjoint. Same input + seed -> same output.
    """
    assert abs(sum(fracs) - 1) < 1e-9
    rng = np.random.default_rng(seed)
    meta = meta.dropna(subset=["quantity"]).astype({"id": int, "quantity": int})
    meta = meta[meta.quantity.isin(CLASSES)].drop_duplicates("id").sort_values("id").reset_index(drop=True)

    # Natural-mix test set is drawn FIRST from the whole pool, so its mix of counts is the dataset's own.
    natural = None
    if n_natural:
        idx = np.sort(rng.permutation(len(meta))[:n_natural])
        natural = meta.iloc[idx].reset_index(drop=True)[["id", "quantity"]]
        meta = meta[~meta.id.isin(natural.id)]

    n_tr = round(per_class * fracs[0])
    n_va = round(per_class * fracs[1])
    out = {"train": [], "val": [], "test": []}
    for c in CLASSES:
        ids = meta.loc[meta.quantity == c, "id"].to_numpy()
        if len(ids) < per_class:
            raise ValueError(f"class {c}: only {len(ids)} images, need {per_class}")
        pick = rng.permutation(ids)[:per_class]
        for name, part in zip(("train", "val", "test"), np.split(pick, [n_tr, n_tr + n_va])):
            out[name].append(pd.DataFrame({"id": part, "quantity": c}))
    res = {k: pd.concat(v).sort_values("id").reset_index(drop=True) for k, v in out.items()}

    if natural is not None:
        res["test_natural"] = natural
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=1500)
    ap.add_argument("--n-natural", type=int, default=1350)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    meta = pd.read_csv(DATA / "metadata_sample.csv")
    res = make_splits(meta, args.per_class, n_natural=args.n_natural, seed=args.seed)
    SPLITS.mkdir(parents=True, exist_ok=True)
    for k, df in res.items():
        df.to_csv(SPLITS / f"{k}.csv", index=False)
        print(k, len(df), df.quantity.value_counts().sort_index().to_dict())


if __name__ == "__main__":
    main()
