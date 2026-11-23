"""Download the bin images listed in splits/abid/*.csv into data/abid/images/ (resumable, checks JPEGs decode)."""
import argparse
import io
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests
from PIL import Image

from invcount.abid.common import BUCKET_URL, DATA, SPLITS, image_key

IMG_DIR = DATA / "images"


def image_path(i: int):
    return IMG_DIR / f"{i:05d}.jpg"


def fetch(session, i):
    p = image_path(i)
    if p.exists():
        return i, True
    r = session.get(f"{BUCKET_URL}/{image_key(i)}", timeout=60)
    if r.status_code != 200:
        return i, False
    try:
        Image.open(io.BytesIO(r.content)).verify()
    except Exception:
        return i, False
    p.write_bytes(r.content)
    return i, True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    ids = sorted({int(i) for f in SPLITS.glob("*.csv") for i in pd.read_csv(f).id})
    s = requests.Session()
    s.mount("https://", requests.adapters.HTTPAdapter(pool_maxsize=args.workers, max_retries=3))
    with ThreadPoolExecutor(args.workers) as ex:
        res = list(ex.map(lambda i: fetch(s, i), ids))
    bad = [i for i, ok in res if not ok]
    print(f"{len(ids) - len(bad)}/{len(ids)} images ok; failed: {bad[:20]}")


if __name__ == "__main__":
    main()
