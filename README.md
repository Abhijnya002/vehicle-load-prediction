# Vehicle Load Prediction from CAN Data

End-to-end ML regression pipeline that estimates per-corner vertical loads
and chassis vibration on a passenger vehicle from synthesised CAN bus and
inertial-measurement (accelerometer / gyro) signals. Trained across
parameterised sweeps of vehicle design variables (mass, suspension
stiffness, CG height, tire pressure …) so that a single model can stand
in for a large family of CAE configurations.

**Stack:** Python 3.9+ · NumPy · pandas · SciPy · scikit-learn · Streamlit · joblib

---

## 1. Problem statement

Full-fidelity vehicle dynamics CAE (Adams / VI-CarRealTime / IPG CarMaker)
is the gold standard for predicting suspension corner loads and ride
vibration, but every parameter sweep requires a new solve, which is slow
and expensive. This project asks: given high-rate CAN + IMU signals (the
things you can actually log from a real vehicle), can a tabular ML model
learn the mapping from `signals × design parameters → loads` accurately
enough to replace a chunk of those CAE runs?

The answer here is **yes** for the synthetic test bed — the best models
achieve **R² ≈ 0.99 on corner loads** and **R² ≈ 1.00 on chassis vibration
RMS** on held-out drives

---

## 2. Repository layout

```
src/
  config.py            Sample rate, channel lists, design-parameter ranges
  data_generation.py   Physics-based synthetic drive simulator
  features.py          Sliding-window feature engineering
  models.py            sklearn pipelines + grouped CV wrapper
  evaluation.py        Per-target metrics + diagnostic plots
scripts/
  generate_data.py     CLI: simulate N drives -> raw + feature parquet
  train.py             CLI: GroupKFold CV across all models, persist best
  evaluate.py          CLI: held-out drive split, metrics + plots
app/
  dashboard.py         Streamlit demo: sweep design params, predict live
outputs/               Generated plots + metrics.csv (committed for preview)
models/                Saved .joblib model artefacts (gitignored)
data/                  Raw + featurised parquet (gitignored)
```

---

## 3. Synthetic data generation (`src/data_generation.py`)

A "drive" is a 60 s recording sampled at 100 Hz (6 000 samples). For each
drive the simulator:

1. **Samples a design point** uniformly from physically-plausible ranges:

   | Parameter                     | Range            |
   |-------------------------------|------------------|
   | `mass_kg`                     | 1400 – 2200      |
   | `wheelbase_m`                 | 2.55 – 2.95      |
   | `track_m`                     | 1.50 – 1.65      |
   | `cg_height_m`                 | 0.50 – 0.70      |
   | `front_weight_ratio`          | 0.50 – 0.62      |
   | `suspension_stiffness_N_per_m`| 25 000 – 60 000  |
   | `damping_ratio`               | 0.20 – 0.45      |
   | `tire_pressure_kPa`           | 200 – 260        |

2. **Generates a longitudinal velocity profile** as a piecewise sequence
   of accel / cruise / brake segments, low-pass-filtered at 0.8 Hz to
   smooth the segment boundaries (Butterworth, order 3).

3. **Derives longitudinal acceleration** `ax = dv/dt`, filtered at 3 Hz.

4. **Generates steering events** — random half-sine impulses on the road
   wheel angle δ — then computes lateral acceleration via the bicycle
   model:

   ```
   ay = v² · tan(δ) / wheelbase
   ```

5. **Excites the road** with band-limited white noise (0–15 Hz) plus 2–8
   discrete bumps (Hanning-windowed pulses), then passes it through a
   per-corner 1-DoF sprung-mass model integrated with explicit Euler:

   ```
   m·ẍ + c·(ẋ − ẋ_road) + k·(x − x_road) = 0
   c = 2·ζ·√(k·m),  m = mass / 4
   ```

   The output is the AC component of the body-mounted vertical
   accelerometer; gravity (9.81 m/s²) is added back to form the logged
   signal. Tire pressure scales the road-to-body transmission via a
   linear interpolant on the wheel input.

6. **Computes ground-truth corner loads** from quasi-static
   weight-transfer equations plus the dynamic vertical force from the
   suspension model:

   ```
   F_static_front = m·g·r_front,     F_static_rear = m·g·(1 − r_front)
   ΔW_long = m·ax·h_cg / L      (longitudinal weight transfer)
   ΔW_lat  = m·ay·h_cg / track  (lateral weight transfer)
   F_z_dyn = m·a_z_AC / 4       (dynamic vertical, per corner)

   F_FL = (F_static_front − ΔW_long)/2 + ΔW_lat/2 + F_z_dyn
   F_FR = (F_static_front − ΔW_long)/2 − ΔW_lat/2 + F_z_dyn
   F_RL = (F_static_rear  + ΔW_long)/2 + ΔW_lat/2 + F_z_dyn
   F_RR = (F_static_rear  + ΔW_long)/2 − ΔW_lat/2 + F_z_dyn
   ```

7. **Computes chassis vibration RMS** as a causal 1 s rolling RMS of the
   AC vertical acceleration (cumulative-sum trick → O(N)).

8. **Adds realistic sensor noise** to every logged channel (e.g. ±0.05
   m/s² on accelerometers, ±8 rpm on the tach, ±0.5 % on pedal positions,
   ±3 mrad on steering).

### Logged channels (CAN + IMU)

`vehicle_speed_mps`, `engine_rpm`, `throttle_pct`, `brake_pct`,
`steering_angle_rad`, `gear`, `accel_x_mps2`, `accel_y_mps2`, `accel_z_mps2`

### Prediction targets

`load_FL_N`, `load_FR_N`, `load_RL_N`, `load_RR_N`, `vibration_rms_mps2`

---

## 4. Feature engineering (`src/features.py`)

Each drive is sliced into overlapping windows of **1.0 s with 0.5 s
stride** (50 % overlap). For each window the following are emitted:

- **Time-domain stats** on every CAN/IMU channel (9 channels × 8 stats =
  72 features): mean, std, min, max, RMS, peak-to-peak, kurtosis, skewness.
- **Spectral features** on the three accelerometer channels (3 × 4 = 12
  features): FFT-based band power in 0.5–4 / 4–12 / 12–30 Hz bands plus
  the dominant frequency.
- **Lag taps** at 0.1 / 0.5 / 1.0 s into the past on every CAN channel
  (9 × 3 = 27 features).
- **Cross-signal interaction**: `planar_accel_rms = √mean(ax² + ay²)`.
- **Static design parameters** appended unchanged (8 features).

Total: **120 features per window**. Targets for each window are taken at
the window's last sample (causal — no future leakage).

A 40-drive simulation yields **4 760 feature rows × 120 features**.

---

## 5. Models and cross-validation (`src/models.py`)

Four pipelines, each `StandardScaler → regressor`:

| Pipeline            | Estimator                                      | Multi-output |
|---------------------|------------------------------------------------|--------------|
| `linear`            | `LinearRegression`                             | native       |
| `ridge`             | `Ridge(alpha=1.0)`                             | native       |
| `random_forest`     | `RandomForestRegressor(n=200, leaf=2)`         | native       |
| `gradient_boosting` | `MultiOutputRegressor(GradientBoosting(n=200))`| wrapped      |

### GroupKFold by drive

Naive random splits leak heavily because windows from the same drive
share the same design parameters and overlap in time. We use
`GroupKFold(n_splits=5)` with `groups = drive_id` so an entire drive is
either fully in train or fully in test for every fold. The held-out
evaluation in `scripts/evaluate.py` uses `GroupShuffleSplit(test_size=0.25)`
for the same reason.

### CV results (R² on 5 GroupKFold folds, mean ± std)

| Model               | Mean R²   | Std R²   |
|---------------------|-----------|----------|
| `ridge`             | **0.9520**| 0.0127   |
| `gradient_boosting` | 0.9517    | 0.0126   |
| `linear`            | 0.9513    | 0.0129   |
| `random_forest`     | 0.7669    | 0.0130   |

CV scoring is averaged across all 5 outputs, so the score is dominated by
shared variance across targets. Ridge wins narrowly because the
underlying physics (weight transfer) is almost exactly linear in `m`,
`ax`, `ay`, `h_cg`. Random Forest under-performs in this aggregated metric
because it cannot extrapolate to design points outside its training
quantiles.

### Held-out per-target results (25 % drive holdout)

| Model               | Target               | RMSE (N or m/s²) | MAE  | R²       | MAPE % |
|---------------------|----------------------|------------------|------|----------|--------|
| `random_forest`     | `load_FL_N`          | 182              | 113  | **0.991**| 3.6    |
| `random_forest`     | `load_FR_N`          | 162              | 101  | **0.992**| 2.3    |
| `random_forest`     | `load_RL_N`          | 190              | 114  | **0.990**| 5.4    |
| `random_forest`     | `load_RR_N`          | 186              | 112  | **0.989**| 4.8    |
| `gradient_boosting` | `vibration_rms_mps2` | 0.0068           | 0.005| **0.999**| 0.51   |
| `ridge`             | (averages)           | 293 N / 0.042    | 213  | 0.974    | 6.7    |

Take-away: **tree ensembles win on the individual held-out drive split**
because they capture the small non-linearities the quasi-static
weight-transfer model adds via the vertical-acceleration term. Random
Forest is the best load predictor; Gradient Boosting is the best
vibration predictor. The CV aggregate vs held-out divergence is itself a
useful teaching point — always look at per-target metrics.

---

## 6. Diagnostic plots (`outputs/`)

| File                              | Content                                              |
|-----------------------------------|------------------------------------------------------|
| `pred_vs_actual.png`              | Scatter for each target, dashed `y = x` overlay     |
| `residuals.png`                   | Per-target residual histograms with μ and σ         |
| `feature_importance.png`          | Top-25 features (mean importance across estimators) |
| `drive_<id>_overlay.png`          | Time-series overlay of actual vs predicted on one held-out drive |
| `metrics.csv`                     | Full per-model × per-target metrics table           |

---

## 7. Interactive dashboard (`app/dashboard.py`)

```bash
streamlit run app/dashboard.py
```

Sidebar sliders for every design parameter + drive duration + RNG seed.
Clicking **Simulate drive & predict** runs the simulator at the chosen
operating point, computes features, runs the saved model, and overlays
predicted vs actual load traces and the underlying CAN signals. This is
the "what-if" tool — change `suspension_stiffness_N_per_m` from 30k to
50k and watch how the predicted corner loads sharpen up under the same
drive cycle, without re-running CAE.

---

## 8. Reproducibility

- Deterministic given `--seed` (default 42). Each drive draws its own
  RNG from a parent generator so per-drive simulations are stable.
- All random estimators take `random_state=42`.
- Generated data and trained models are intentionally gitignored — they
  are regenerated by the three CLI scripts:

```bash
pip install -r requirements.txt
python scripts/generate_data.py --n-drives 40    # ~5 s
python scripts/train.py                          # ~25 s on a laptop
python scripts/evaluate.py                       # ~3 s
streamlit run app/dashboard.py
```

Tested versions: Python 3.9.6, NumPy 1.26.4, pandas 2.3.3,
scikit-learn 1.6.1, SciPy 1.13.1, Streamlit 1.50.0.

---

## 9. Known limitations & extensions

- **Synthetic ground truth.** The quasi-static weight-transfer model
  misses tire elasticity, anti-roll-bar coupling, and roll/pitch
  inertias. Replacing the simulator with logged CAE traces would lift
  the project from "demo" to "deployable surrogate".
- **No hyperparameter tuning.** A `GridSearchCV` over RF `max_depth`,
  GBM `n_estimators` / `learning_rate` would likely close the
  CV-vs-holdout gap.
- **Tabular model.** A 1D-CNN or LSTM on raw windows would let the model
  learn its own spectral features instead of relying on FFT band power.
- **Feature attribution.** Adding SHAP values per prediction would make
  the dashboard far more compelling for design engineers.
