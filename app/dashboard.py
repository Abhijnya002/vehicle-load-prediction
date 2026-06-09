"""Streamlit dashboard for interactive load prediction.

Lets the user sweep design parameters (mass, suspension stiffness, etc.),
simulates a fresh drive at that operating point, runs the trained model,
and overlays predicted vs ground-truth loads — surfacing how design
variations propagate through the same drive cycle without re-running CAE.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import (
    DESIGN_PARAM_RANGES,
    DESIGN_PARAMS,
    MODELS_DIR,
    SAMPLE_HZ,
    TARGETS,
)
from src.data_generation import DesignPoint, generate_drive
from src.features import featurize_drive, split_xy
from src.models import load_model


st.set_page_config(page_title="Vehicle Load Prediction", layout="wide")
st.title("Vehicle Load Prediction from CAN + Accelerometer Data")
st.caption(
    "ML regression models trained on synthetic CAN and accelerometer signals "
    "predict corner loads and chassis vibration across design parameter sweeps."
)

# ----- Sidebar: model selection + design parameter sliders -------------------
available_models = sorted(p.stem for p in MODELS_DIR.glob("*.joblib"))
if not available_models:
    st.error("No trained models found. Run `python scripts/train.py` first.")
    st.stop()

default_idx = available_models.index("best") if "best" in available_models else 0
model_name = st.sidebar.selectbox("Model", available_models, index=default_idx)
model = load_model(model_name)

st.sidebar.markdown("### Design parameters")
design_values: dict[str, float] = {}
slider_steps = {
    "mass_kg": 25.0,
    "wheelbase_m": 0.01,
    "track_m": 0.01,
    "cg_height_m": 0.01,
    "front_weight_ratio": 0.01,
    "suspension_stiffness_N_per_m": 500.0,
    "damping_ratio": 0.01,
    "tire_pressure_kPa": 1.0,
}
for name in DESIGN_PARAMS:
    lo, hi = DESIGN_PARAM_RANGES[name]
    default = (lo + hi) / 2.0
    design_values[name] = float(st.sidebar.slider(
        name, float(lo), float(hi), float(default), step=slider_steps.get(name, (hi - lo) / 50.0)
    ))

duration_s = st.sidebar.slider("Drive duration (s)", 20, 120, 60, step=10)
seed = st.sidebar.number_input("Random seed", 0, 10_000, value=7)

run = st.sidebar.button("Simulate drive & predict", type="primary")


def _simulate_with_overrides(dp: DesignPoint, rng_seed: int, duration: int) -> pd.DataFrame:
    """Wrap generate_drive but force the design point and duration."""
    import src.config as cfg
    original_duration = cfg.DRIVE_DURATION_S
    cfg.DRIVE_DURATION_S = int(duration)
    try:
        rng = np.random.default_rng(rng_seed)
        # Monkey-patch DesignPoint.sample to return our fixed point this call
        original_sample = DesignPoint.sample
        DesignPoint.sample = staticmethod(lambda _rng: dp)
        try:
            drive = generate_drive(drive_id=0, rng=rng)
        finally:
            DesignPoint.sample = original_sample
    finally:
        cfg.DRIVE_DURATION_S = original_duration
    return drive


if not run:
    st.info("Choose design parameters in the sidebar, then click "
            "**Simulate drive & predict**.")
    st.stop()

# ----- Simulate + predict ---------------------------------------------------
dp = DesignPoint(**design_values)
with st.spinner("Simulating drive cycle..."):
    drive = _simulate_with_overrides(dp, int(seed), int(duration_s))

with st.spinner("Computing features..."):
    feats = featurize_drive(drive)
    if feats.empty:
        st.error("Drive too short to produce a window. Increase duration.")
        st.stop()
    X, y_true, _ = split_xy(feats)
    # Ensure column alignment with the trained model
    fitted_cols = getattr(model.named_steps["scaler"], "feature_names_in_", None)
    if fitted_cols is not None:
        X = X[list(fitted_cols)]

with st.spinner("Predicting..."):
    y_pred = model.predict(X)

# ----- Summary metrics ------------------------------------------------------
from src.evaluation import per_target_metrics

metrics = per_target_metrics(y_true, y_pred)

col1, col2 = st.columns([2, 3])
with col1:
    st.subheader("Test metrics (this drive)")
    st.dataframe(metrics.style.format({"rmse": "{:.1f}", "mae": "{:.1f}",
                                       "r2": "{:.3f}", "mape_pct": "{:.2f}"}),
                 use_container_width=True)

with col2:
    st.subheader("Design point")
    st.json({k: round(v, 3) for k, v in design_values.items()})

# ----- Overlay plots --------------------------------------------------------
import matplotlib.pyplot as plt

t_end = feats["t_end"].to_numpy()
fig, axes = plt.subplots(len(TARGETS), 1, figsize=(10, 2.2 * len(TARGETS)), sharex=True)
for ax, name, i in zip(axes, TARGETS, range(len(TARGETS))):
    ax.plot(t_end, y_true.iloc[:, i].to_numpy(), label="actual (physics)", lw=1.4)
    ax.plot(t_end, y_pred[:, i], label="predicted (ML)", lw=1.1, alpha=0.85)
    ax.set_ylabel(name)
    ax.legend(loc="upper right", fontsize=8)
axes[-1].set_xlabel("t (s)")
fig.tight_layout()
st.pyplot(fig)

# ----- Raw signal preview ---------------------------------------------------
with st.expander("Raw CAN + accelerometer signals (first 10 s)"):
    show = drive[drive["t"] <= 10.0]
    sig_cols = [
        "vehicle_speed_mps", "engine_rpm", "throttle_pct", "brake_pct",
        "accel_x_mps2", "accel_y_mps2", "accel_z_mps2",
    ]
    fig2, axes2 = plt.subplots(len(sig_cols), 1, figsize=(10, 1.6 * len(sig_cols)),
                               sharex=True)
    for ax, col in zip(axes2, sig_cols):
        ax.plot(show["t"], show[col], lw=0.9)
        ax.set_ylabel(col, fontsize=8)
    axes2[-1].set_xlabel("t (s)")
    fig2.tight_layout()
    st.pyplot(fig2)
