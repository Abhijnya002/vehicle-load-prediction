"""Generate synthetic CAN + accelerometer drives and write parquet files."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import DATA_PROCESSED, DATA_RAW, RANDOM_SEED
from src.data_generation import generate_dataset
from src.features import featurize_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-drives", type=int, default=40,
                        help="Number of synthetic drives to simulate")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    print(f"[gen] Simulating {args.n_drives} drives (seed={args.seed})...")
    raw = generate_dataset(args.n_drives, seed=args.seed)
    raw_path = DATA_RAW / "drives.parquet"
    raw.to_parquet(raw_path, index=False)
    print(f"[gen] Raw rows: {len(raw):,} -> {raw_path}")

    print("[gen] Building per-window features...")
    feats = featurize_dataset(raw)
    feat_path = DATA_PROCESSED / "features.parquet"
    feats.to_parquet(feat_path, index=False)
    print(f"[gen] Feature rows: {len(feats):,} | cols: {feats.shape[1]} -> {feat_path}")


if __name__ == "__main__":
    main()
