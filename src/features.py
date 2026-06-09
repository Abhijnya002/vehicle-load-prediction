"""Time-series feature engineering.

Raw CAN + accelerometer signals are aggregated over a sliding window into
per-window statistical, spectral, and lag features. Design parameters
(constant per drive) are carried through as-is. Targets for each window
are taken as the value at the window's end time.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew

from src.config import (
    CAN_SIGNALS,
    DESIGN_PARAMS,
    SAMPLE_HZ,
    TARGETS,
    WINDOW_SECONDS,
    WINDOW_STRIDE_SECONDS,
)


# Spectral bands (Hz) used for accelerometer band-power features
ACCEL_BANDS = [(0.5, 4.0), (4.0, 12.0), (12.0, 30.0)]

# Signals that get spectral / vibration features applied
ACCEL_SIGNALS = ["accel_x_mps2", "accel_y_mps2", "accel_z_mps2"]

# Lag taps (in seconds) for each CAN signal
LAG_SECONDS = [0.1, 0.5, 1.0]


def _time_features(arr: np.ndarray, prefix: str) -> dict:
    out = {
        f"{prefix}_mean": float(np.mean(arr)),
        f"{prefix}_std":  float(np.std(arr)),
        f"{prefix}_min":  float(np.min(arr)),
        f"{prefix}_max":  float(np.max(arr)),
        f"{prefix}_rms":  float(np.sqrt(np.mean(arr ** 2))),
        f"{prefix}_p2p":  float(np.ptp(arr)),
    }
    if len(arr) > 3 and np.std(arr) > 1e-9:
        out[f"{prefix}_kurt"] = float(kurtosis(arr))
        out[f"{prefix}_skew"] = float(skew(arr))
    else:
        out[f"{prefix}_kurt"] = 0.0
        out[f"{prefix}_skew"] = 0.0
    return out


def _band_power(arr: np.ndarray, fs: float) -> dict:
    """Power in fixed frequency bands plus dominant frequency."""
    arr = arr - arr.mean()
    n = len(arr)
    if n < 8:
        return {}
    freqs = np.fft.rfftfreq(n, 1.0 / fs)
    spec = np.abs(np.fft.rfft(arr)) ** 2
    out = {}
    for lo, hi in ACCEL_BANDS:
        mask = (freqs >= lo) & (freqs < hi)
        out[f"band_{lo:g}_{hi:g}"] = float(spec[mask].sum())
    out["dom_freq"] = float(freqs[np.argmax(spec)])
    return out


def _window_row(window: pd.DataFrame, fs: float) -> dict:
    feats: dict = {}
    for sig in CAN_SIGNALS:
        feats.update(_time_features(window[sig].to_numpy(), sig))
    for sig in ACCEL_SIGNALS:
        bp = _band_power(window[sig].to_numpy(), fs)
        for k, v in bp.items():
            feats[f"{sig}_{k}"] = v
    # Cross-signal interaction: combined planar accel magnitude
    ax = window["accel_x_mps2"].to_numpy()
    ay = window["accel_y_mps2"].to_numpy()
    feats["planar_accel_rms"] = float(np.sqrt(np.mean(ax ** 2 + ay ** 2)))
    return feats


def _lag_values(drive: pd.DataFrame, end_idx: int, fs: float) -> dict:
    feats = {}
    for sig in CAN_SIGNALS:
        for lag_s in LAG_SECONDS:
            lag_i = max(0, end_idx - int(lag_s * fs))
            feats[f"{sig}_lag_{lag_s:g}s"] = float(drive[sig].iloc[lag_i])
    return feats


def featurize_drive(drive: pd.DataFrame, fs: float = SAMPLE_HZ) -> pd.DataFrame:
    """Slide a window across one drive and emit one row of features per stride."""
    win = int(WINDOW_SECONDS * fs)
    stride = int(WINDOW_STRIDE_SECONDS * fs)
    n = len(drive)
    if n < win:
        return pd.DataFrame()

    static_cols = {name: float(drive[name].iloc[0]) for name in DESIGN_PARAMS}
    drive_id = int(drive["drive_id"].iloc[0])

    rows = []
    for end in range(win, n + 1, stride):
        start = end - win
        window = drive.iloc[start:end]
        feats = _window_row(window, fs)
        feats.update(_lag_values(drive, end - 1, fs))
        feats.update(static_cols)
        feats["drive_id"] = drive_id
        feats["t_end"] = float(drive["t"].iloc[end - 1])
        for tgt in TARGETS:
            feats[tgt] = float(drive[tgt].iloc[end - 1])
        rows.append(feats)
    return pd.DataFrame(rows)


def featurize_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Apply featurize_drive across every drive_id."""
    out_frames: list[pd.DataFrame] = []
    for drive_id, sub in df.groupby("drive_id", sort=True):
        out_frames.append(featurize_drive(sub))
    out = pd.concat(out_frames, ignore_index=True)
    # Reorder for readability: identifiers first, targets last
    id_cols = ["drive_id", "t_end"]
    feat_cols = [c for c in out.columns if c not in id_cols + TARGETS]
    return out[id_cols + feat_cols + TARGETS]


def split_xy(features_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Return (X, y, groups) where groups is the per-row drive_id."""
    y = features_df[TARGETS].copy()
    drop_cols = ["drive_id", "t_end"] + TARGETS
    X = features_df.drop(columns=drop_cols)
    groups = features_df["drive_id"].to_numpy()
    return X, y, groups
