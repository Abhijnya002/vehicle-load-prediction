"""Train regression models with grouped CV and persist the best one."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import DATA_PROCESSED, MODELS_DIR
from src.features import split_xy
from src.models import cross_validate_grouped, make_models, save_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features",
                        default=str(DATA_PROCESSED / "features.parquet"))
    parser.add_argument("--cv-folds", type=int, default=5)
    args = parser.parse_args()

    feats = pd.read_parquet(args.features)
    X, y, groups = split_xy(feats)
    print(f"[train] X={X.shape} y={y.shape} drives={len(np.unique(groups))}")

    cv_rows = []
    best_name, best_score, best_model = None, -np.inf, None
    for name, pipe in make_models().items():
        print(f"[train] CV fitting {name}...")
        report = cross_validate_grouped(pipe, X, y, groups, n_splits=args.cv_folds)
        print(f"[train]   {name}: mean R2={report.mean_r2:.4f} (±{report.std_r2:.4f})")
        cv_rows.append({"model": name, "mean_r2": report.mean_r2, "std_r2": report.std_r2})

        # Refit on full data and save
        pipe.fit(X, y)
        save_model(name, pipe)

        if report.mean_r2 > best_score:
            best_name, best_score, best_model = name, report.mean_r2, pipe

    cv_df = pd.DataFrame(cv_rows).sort_values("mean_r2", ascending=False)
    cv_path = MODELS_DIR / "cv_report.csv"
    cv_df.to_csv(cv_path, index=False)
    print(f"[train] CV report -> {cv_path}")

    assert best_model is not None
    save_model("best", best_model)
    (MODELS_DIR / "best_model.txt").write_text(best_name + "\n")
    print(f"[train] Best model: {best_name} (R2={best_score:.4f}) -> models/best.joblib")


if __name__ == "__main__":
    main()
