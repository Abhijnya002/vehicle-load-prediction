"""Metrics, residual analysis, and plotting helpers."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.config import OUTPUTS_DIR, TARGETS


def per_target_metrics(y_true: pd.DataFrame, y_pred: np.ndarray) -> pd.DataFrame:
    rows = []
    y_true_arr = y_true.to_numpy()
    for i, name in enumerate(TARGETS):
        yt = y_true_arr[:, i]
        yp = y_pred[:, i]
        rmse = float(np.sqrt(mean_squared_error(yt, yp)))
        mae = float(mean_absolute_error(yt, yp))
        r2 = float(r2_score(yt, yp))
        # mean absolute percentage error, guarding tiny y
        denom = np.maximum(np.abs(yt), 1.0)
        mape = float(np.mean(np.abs((yt - yp) / denom)) * 100.0)
        rows.append({"target": name, "rmse": rmse, "mae": mae, "r2": r2, "mape_pct": mape})
    return pd.DataFrame(rows)


def plot_prediction_scatter(y_true: pd.DataFrame, y_pred: np.ndarray, out_path: Path) -> None:
    fig, axes = plt.subplots(1, len(TARGETS), figsize=(4 * len(TARGETS), 4))
    if len(TARGETS) == 1:
        axes = [axes]
    for ax, name, i in zip(axes, TARGETS, range(len(TARGETS))):
        yt = y_true.iloc[:, i]
        yp = y_pred[:, i]
        ax.scatter(yt, yp, s=4, alpha=0.4)
        lo = float(min(yt.min(), yp.min()))
        hi = float(max(yt.max(), yp.max()))
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        ax.set_xlabel("actual")
        ax.set_ylabel("predicted")
        ax.set_title(name)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_residuals(y_true: pd.DataFrame, y_pred: np.ndarray, out_path: Path) -> None:
    fig, axes = plt.subplots(1, len(TARGETS), figsize=(4 * len(TARGETS), 4))
    if len(TARGETS) == 1:
        axes = [axes]
    for ax, name, i in zip(axes, TARGETS, range(len(TARGETS))):
        resid = y_true.iloc[:, i].to_numpy() - y_pred[:, i]
        ax.hist(resid, bins=40)
        ax.axvline(0, color="k", linestyle="--", lw=1)
        ax.set_xlabel("residual")
        ax.set_title(f"{name}\nμ={resid.mean():.1f} σ={resid.std():.1f}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_feature_importance(model, feature_names: list[str], out_path: Path, top_k: int = 25) -> None:
    """Best-effort: averages importances across multi-output estimators."""
    reg = model.named_steps["reg"]
    importances = None

    if hasattr(reg, "feature_importances_"):
        importances = reg.feature_importances_
    elif hasattr(reg, "estimators_"):
        per_est = []
        for est in reg.estimators_:
            if hasattr(est, "feature_importances_"):
                per_est.append(est.feature_importances_)
        if per_est:
            importances = np.mean(per_est, axis=0)
    elif hasattr(reg, "coef_"):
        coef = np.atleast_2d(reg.coef_)
        importances = np.mean(np.abs(coef), axis=0)

    if importances is None:
        return

    order = np.argsort(importances)[::-1][:top_k]
    fig, ax = plt.subplots(figsize=(8, 0.3 * top_k + 1))
    ax.barh(range(len(order)), importances[order][::-1])
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([feature_names[i] for i in order][::-1], fontsize=8)
    ax.set_xlabel("importance")
    ax.set_title("Top features")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_drive_overlay(drive_actual: pd.DataFrame, drive_pred: pd.DataFrame, out_path: Path) -> None:
    """Overlay actual vs predicted load traces over time for one drive."""
    fig, axes = plt.subplots(len(TARGETS), 1, figsize=(10, 2.2 * len(TARGETS)), sharex=True)
    if len(TARGETS) == 1:
        axes = [axes]
    t = drive_actual["t_end"].to_numpy()
    for ax, name in zip(axes, TARGETS):
        ax.plot(t, drive_actual[name], label="actual", lw=1.2)
        ax.plot(t, drive_pred[name], label="predicted", lw=1.0, alpha=0.85)
        ax.set_ylabel(name)
        ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("t (s)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def write_metrics_csv(metrics_by_model: dict[str, pd.DataFrame]) -> Path:
    out = []
    for model_name, df in metrics_by_model.items():
        df = df.copy()
        df.insert(0, "model", model_name)
        out.append(df)
    combined = pd.concat(out, ignore_index=True)
    path = OUTPUTS_DIR / "metrics.csv"
    combined.to_csv(path, index=False)
    return path
