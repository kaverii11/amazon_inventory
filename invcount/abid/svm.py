"""v1's hand-crafted features + RBF SVM, redone on the balanced splits. Tuned on val only.

Features (same recipes as v1/SVM_miniproj): 
  blob   - Laplacian-of-Gaussian blobs on the 224px grey image -> per-row sum/max/count of blob radii (672-d)
  hog    - HOG (8 orientations, 16px cells, 4x4 blocks) on the Laplacian of the grey image (15,488-d)
  pixels - raw 224x224x3 pixels (150,528-d)
Pipeline: standardise -> PCA -> RBF SVC. Grid over PCA size, C and gamma; best by val accuracy.
Fitted pipelines are saved for the one-time test run in final_test.py. Test data is not read here.
"""
import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor

import cv2
import joblib
import numpy as np
from skimage.color import rgb2gray
from skimage.feature import blob_log, hog
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from invcount.abid.cache import load_split
from invcount.abid.common import DATA, RESULTS
from invcount.metrics import count_metrics

MODELS = DATA / "models"


def blob_feat(img):
    g = rgb2gray(img)
    blobs = blob_log(g, max_sigma=30, num_sigma=10, threshold=0.1)
    m = np.zeros(g.shape, dtype=np.int16)
    for y, x, r in blobs:
        m[int(y), int(x)] = int(r)
    return np.concatenate([m.sum(1), m.max(1), np.count_nonzero(m, axis=1)]).astype(np.float32)


def hog_feat(img):
    lap = cv2.Laplacian(rgb2gray(img), cv2.CV_64F)
    return hog(lap, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(4, 4),
               block_norm="L2", visualize=False).astype(np.float32)


FEATS = {"blob": blob_feat, "hog": hog_feat}


def features(name, X):
    if name == "pixels":
        return X.reshape(len(X), -1).astype(np.float32)
    with ProcessPoolExecutor() as ex:
        return np.stack(list(ex.map(FEATS[name], X, chunksize=32)))


def cached_features(name, split):
    p = DATA / "features" / f"{name}_{split}.npy"
    if name != "pixels" and p.exists():
        return np.load(p)
    X, _, _ = load_split(split)
    F = features(name, X)
    if name != "pixels":
        p.parent.mkdir(parents=True, exist_ok=True)
        np.save(p, F)
    return F


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat", choices=["blob", "hog", "pixels"], required=True)
    args = ap.parse_args()
    _, ytr, _ = load_split("train")
    _, yva, _ = load_split("val")
    t0 = time.time()
    Ftr, Fva = cached_features(args.feat, "train"), cached_features(args.feat, "val")
    print(f"features {Ftr.shape} in {time.time() - t0:.0f}s", flush=True)

    pca_sizes = [20, 50, 200] if args.feat == "blob" else [30, 100, 300]
    rows, best = [], None
    for k in pca_sizes:
        scaler = StandardScaler().fit(Ftr)
        pca = PCA(k, svd_solver="randomized", random_state=0).fit(scaler.transform(Ftr))
        Ztr, Zva = pca.transform(scaler.transform(Ftr)), pca.transform(scaler.transform(Fva))
        g0 = 1.0 / (k * Ztr.var())  # sklearn's gamma="scale"
        for C in [0.3, 1, 10, 100]:
            for gf in [0.1, 0.3, 1, 3]:
                t = time.time()
                svc = SVC(C=C, gamma=g0 * gf, kernel="rbf").fit(Ztr, ytr)
                m = count_metrics(yva, svc.predict(Zva))
                row = {"pca": k, "C": C, "gamma_x_scale": gf, "val_acc": m["accuracy"], "val_mae": m["mae"],
                       "fit_s": round(time.time() - t, 1)}
                rows.append(row)
                print(row, flush=True)
                if best is None or row["val_acc"] > best[0]["val_acc"]:
                    best = (row, make_pipeline(scaler, pca, svc))
    RESULTS.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    joblib.dump(best[1], MODELS / f"svm_{args.feat}.joblib")
    json.dump({"grid": rows, "best": best[0]}, open(RESULTS / f"svm_{args.feat}_val.json", "w"), indent=1)
    print("BEST", best[0])


if __name__ == "__main__":
    main()
