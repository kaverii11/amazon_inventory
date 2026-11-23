"""Decode every split image once, squash to 224x224 RGB (as v1 did) and cache as uint8 arrays.

Output: data/abid/cache224_<split>.npz with X (N,224,224,3) uint8, y (N,), ids (N,).
"""
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from PIL import Image

from invcount.abid.common import DATA, SPLITS

SIZE = 224
SPLIT_NAMES = ("train", "val", "test", "test_natural")


def load(i):
    with Image.open(DATA / "images" / f"{i:05d}.jpg") as im:
        return np.asarray(im.convert("RGB").resize((SIZE, SIZE), Image.BILINEAR))


def load_split(name):
    d = np.load(DATA / f"cache{SIZE}_{name}.npz")
    return d["X"], d["y"], d["ids"]


def main():
    for name in SPLIT_NAMES:
        df = pd.read_csv(SPLITS / f"{name}.csv")
        with ProcessPoolExecutor() as ex:
            X = np.stack(list(ex.map(load, df.id.tolist(), chunksize=64)))
        np.savez(DATA / f"cache{SIZE}_{name}.npz", X=X, y=df.quantity.to_numpy(), ids=df.id.to_numpy())
        print(name, X.shape)


if __name__ == "__main__":
    main()
