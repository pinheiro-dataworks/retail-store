"""Download the raw CSVs into data/raw/ and verify them.

The pipeline expects three files (sales.csv, stores.csv, features.csv) from
the adapted version of the Walmart "Store Sales Forecasting" data published on
Kaggle by Noopur Bhatt. The original competition data (2010-2012, 45 stores)
has a different layout and will not work here.

Usage
-----
    # needs a Kaggle API token in ~/.kaggle/kaggle.json or KAGGLE_USERNAME/KAGGLE_KEY
    python scripts/download_data.py --dataset <owner>/<dataset-slug>

    # files already downloaded by hand? just verify them
    python scripts/download_data.py --verify-only

The dataset slug can also be set through the RETAIL_DATASET environment
variable. Checksums below are the SHA-256 of the files used to produce the
published metrics; a mismatch means the numbers in the README will differ.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"

EXPECTED_SHA256 = {
    "sales.csv": "176afb430e63fa04a72bb327d9c93430eb65f1bd588a8dfcf9a4a0402f43ea18",
    "stores.csv": "26d0566c67a20c2ad5ba12aa5a59741a6140600cd7a120c14aa609cc0e7853d8",
    "features.csv": "7120ece56301465339db42ba961e8748b4419af0cf1bfe622aba1d2692778bd8",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(raw_dir: Path) -> bool:
    ok = True
    for name, expected in EXPECTED_SHA256.items():
        path = raw_dir / name
        if not path.exists():
            print(f"[missing]  {path}")
            ok = False
            continue
        actual = sha256(path)
        if actual == expected:
            print(f"[ok]       {name}")
        else:
            print(f"[mismatch] {name}: {actual}")
            ok = False
    return ok


def download(dataset: str, raw_dir: Path) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    cmd = ["kaggle", "datasets", "download", "-d", dataset, "-p", str(raw_dir), "--unzip"]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", default=os.environ.get("RETAIL_DATASET"),
                        help="Kaggle dataset slug, e.g. owner/dataset-name")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    if not args.verify_only:
        if not args.dataset:
            parser.error("pass --dataset or set RETAIL_DATASET (see the docstring)")
        download(args.dataset, args.raw_dir)

    return 0 if verify(args.raw_dir) else 1


if __name__ == "__main__":
    sys.exit(main())
