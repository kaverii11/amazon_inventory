"""Download metadata for a random sample of ABID images and record EXPECTED_QUANTITY.

Public bucket, no account needed. Output: data/abid/metadata_sample.csv (id, quantity).
Resumable: ids already in the CSV are skipped.
"""
import argparse
import csv
import json
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

from invcount.abid.common import BUCKET_URL, DATA, N_IMAGES, meta_key


def fetch(session, i):
    r = session.get(f"{BUCKET_URL}/{meta_key(i)}", timeout=30)
    if r.status_code != 200:
        return i, None
    try:
        return i, int(json.loads(r.content)["EXPECTED_QUANTITY"])
    except (ValueError, KeyError, TypeError):
        return i, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=64)
    args = ap.parse_args()

    DATA.mkdir(parents=True, exist_ok=True)
    out = DATA / "metadata_sample.csv"
    done = set()
    if out.exists():
        with open(out) as f:
            done = {int(r["id"]) for r in csv.DictReader(f)}

    ids = random.Random(args.seed).sample(range(1, N_IMAGES + 1), args.n)
    todo = [i for i in ids if i not in done]
    print(f"{len(done)} already scanned, {len(todo)} to fetch")

    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(pool_connections=args.workers, pool_maxsize=args.workers, max_retries=3)
    session.mount("https://", adapter)
    new_file = not out.exists()
    with open(out, "a", newline="") as f, ThreadPoolExecutor(args.workers) as ex:
        w = csv.writer(f)
        if new_file:
            w.writerow(["id", "quantity"])
        futs = [ex.submit(fetch, session, i) for i in todo]
        for k, fut in enumerate(as_completed(futs), 1):
            i, q = fut.result()
            w.writerow([i, "" if q is None else q])
            if k % 5000 == 0:
                f.flush()
                print(f"{k}/{len(todo)}", flush=True)


if __name__ == "__main__":
    main()
