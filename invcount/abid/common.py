"""Shared paths and helpers for the Amazon Bin Image Dataset (ABID) part."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "abid"
SPLITS = ROOT / "splits" / "abid"
RESULTS = ROOT / "results" / "abid"

BUCKET_URL = "https://aft-vbi-pds.s3.amazonaws.com"
N_IMAGES = 535_234  # keys metadata/00001.json .. metadata/535234.json
CLASSES = list(range(6))  # 0..5 items


def image_key(i: int) -> str:
    return f"bin-images/{i:05d}.jpg"


def meta_key(i: int) -> str:
    return f"metadata/{i:05d}.json"
