"""Score saved models on a held-out drive split and write plots + metrics."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import DATA_PROCESSED, MODELS_DIR, OUTPUTS_DIR, RANDOM_SEED, TARGETS
from src.evaluation import (
    per_target_metrics,
    plot_drive_overlay,
    plot_feature_importance,
    plot_prediction_scatter,
    plot_residuals,
    write_metrics_csv,
)
from src.features import split_xy
from src.models import load_model, make_models


def held_out_split(features: pd.DataFrame, test_frac: float = 0.25):
    X, y, groups = split_xy(features)
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_frac, random_state=RANDOM_SEED)
    train_idx, test_idx = next(splitter.split(X, y, groups))
    return (
        X.iloc[train_idx], X.iloc[test_idx],
        y.iloc[train_idx], y.iloc[test_idx],
        features.iloc[test_idx],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features",
                        default=str(DATA_PROCESSED / "features.parquet"))
    args = parser.parse_args()

    feats = pd.read_parquet(args.features)
    X_train, X_test, y_train, y_test, test_meta = held_out_split(feats)
    print(f"[eval] Train={len(X_train)} Test={len(X_test)}")

    metrics_by_model = {}
    for name in make_models().keys():
        path = MODELS_DIR / f"{name}.joblib"
        if not path.exists():
            print(f"[eval] Skipping {name} (no saved model)")
            continue
        model = load_model(name)
        y_pred = model.predict(X_test)
        metrics = per_target_metrics(y_test, y_pred)
        metrics_by_model[name] = metrics
        print(f"\n[eval] {name}")
        print(metrics.to_string(index=False))

    csv_path = write_metrics_csv(metrics_by_model)
    print(f"\n[eval] Metrics CSV -> {csv_path}")

    # For the best model, generate plots
    best_name = (MODELS_DIR / "best_model.txt").read_text().strip()
    best = load_model("best")
    y_pred_best = best.predict(X_test)

    plot_prediction_scatter(y_test, y_pred_best, OUTPUTS_DIR / "pred_vs_actual.png")
    plot_residuals(y_test, y_pred_best, OUTPUTS_DIR / "residuals.png")
    plot_feature_importance(best, list(X_test.columns),
                            OUTPUTS_DIR / "feature_importance.png")
    print(f"[eval] Scatter / residual / importance plots -> {OUTPUTS_DIR}")

    # Pick the drive with the most test windows and overlay a time-series
    drive_counts = test_meta.groupby("drive_id").size().sort_values(ascending=False)
    if len(drive_counts) > 0:
        focal_drive = int(drive_counts.index[0])
        mask = test_meta["drive_id"].to_numpy() == focal_drive
        order = np.argsort(test_meta.loc[mask, "t_end"].to_numpy())
        actual = test_meta[mask].iloc[order][["drive_id", "t_end"] + TARGETS].reset_index(drop=True)
        pred_arr = y_pred_best[mask][order]
        pred_df = actual[["drive_id", "t_end"]].copy()
        for i, name in enumerate(TARGETS):
            pred_df[name] = pred_arr[:, i]
        plot_drive_overlay(actual, pred_df, OUTPUTS_DIR / f"drive_{focal_drive}_overlay.png")
        print(f"[eval] Drive overlay (drive_id={focal_drive}) -> outputs/")

    print(f"[eval] Best model = {best_name}")


if __name__ == "__main__":
    main()
