# Vehicle Load Prediction from CAN Data

Machine-learning regression models that estimate vehicle corner loads and
chassis vibration from synthetic CAN bus and accelerometer signals.
Trained across sweeps of vehicle design parameters (mass, suspension
stiffness, CG height, etc.) so a single fit can stand in for many
parameter-variation CAE runs.

## Stack
Python · Pandas · NumPy · SciPy · Scikit-learn · Streamlit

## What's in here

```
src/
  config.py            # paths, sample rate, channel / target lists
  data_generation.py   # physics-based synthetic CAN + IMU drive simulator
  features.py          # sliding-window stats, FFT band power, lag features
  models.py            # Linear / Ridge / RF / GBM pipelines + grouped CV
  evaluation.py        # per-target metrics + diagnostic plots
scripts/
  generate_data.py     # simulate drives -> raw + featurised parquet
  train.py             # GroupKFold CV across models, persist best
  evaluate.py          # held-out drive split, metrics + plots
app/
  dashboard.py         # Streamlit demo: sweep design params, predict live
```

## Quick start

```bash
pip install -r requirements.txt

python scripts/generate_data.py --n-drives 40
python scripts/train.py
python scripts/evaluate.py

streamlit run app/dashboard.py
```

Outputs (`metrics.csv`, scatter / residual / feature-importance / drive
overlay PNGs) land in `outputs/`. Saved models live in `models/`.

## Modelling approach

- **Inputs**: 9 CAN/IMU channels (speed, RPM, throttle, brake, steering,
  gear, 3-axis acceleration) plus 8 static design parameters per drive.
- **Window**: 1.0 s windows, 0.5 s stride at 100 Hz sampling.
- **Per-window features**: mean / std / min / max / RMS / peak-to-peak /
  kurtosis / skew on every channel; FFT band power and dominant frequency
  on accelerometer channels; lag taps at 0.1 / 0.5 / 1.0 s.
- **Targets**: 4 corner vertical loads (FL/FR/RL/RR, Newtons) + 1 s
  chassis vibration RMS (m/s²).
- **CV**: `GroupKFold` on `drive_id` so windows from the same drive can't
  leak between folds.

## How the synthetic data is built

Each drive is a 60 s record at 100 Hz. A piecewise speed profile drives
longitudinal acceleration; sinusoidal steering events drive lateral
acceleration via a bicycle model; a band-limited road excitation feeds a
1-DoF mass-spring-damper sprung-mass model for vertical dynamics.
Ground-truth corner loads come from quasi-static weight-transfer
equations plus the dynamic vertical force. Realistic sensor noise is
added on top of every logged channel.
