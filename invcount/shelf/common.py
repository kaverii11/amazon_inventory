"""Paths for the SKU-110K (retail shelf detection) part."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "sku110k" / "SKU110K_fixed"
YOLO = ROOT / "data" / "sku110k" / "yolo"
RESULTS = ROOT / "results" / "sku110k"
RUNS = ROOT / "runs"
LOCAL_FIGS = ROOT / "data" / "sku110k" / "figures_local"  # SKU-110K images may not be redistributed: kept out of git
CSV_COLS = ["image_name", "x1", "y1", "x2", "y2", "cls", "W", "H"]
MAX_DET = 1000  # measured max is 718 boxes in one image (val); Ultralytics' default cap of 300 would truncate counts
