"""Project-wide constants and paths."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"

for _p in (DATA_RAW, DATA_PROCESSED, MODELS_DIR, OUTPUTS_DIR):
    _p.mkdir(parents=True, exist_ok=True)

# Sampling
SAMPLE_HZ = 100              # CAN + accelerometer sample rate
DRIVE_DURATION_S = 60        # seconds per synthetic drive

# Physical constants
G = 9.81  # m/s^2

# Signal channels logged from the (synthetic) CAN bus + IMU
CAN_SIGNALS = [
    "vehicle_speed_mps",
    "engine_rpm",
    "throttle_pct",
    "brake_pct",
    "steering_angle_rad",
    "gear",
    "accel_x_mps2",
    "accel_y_mps2",
    "accel_z_mps2",
]

# Targets the ML model predicts (corner vertical loads + chassis RMS vibration)
TARGETS = [
    "load_FL_N",
    "load_FR_N",
    "load_RL_N",
    "load_RR_N",
    "vibration_rms_mps2",
]

# Design parameters that vary across drives. These are appended as static
# columns so the model can learn how loads change with vehicle setup.
DESIGN_PARAMS = [
    "mass_kg",
    "wheelbase_m",
    "track_m",
    "cg_height_m",
    "front_weight_ratio",
    "suspension_stiffness_N_per_m",
    "damping_ratio",
    "tire_pressure_kPa",
]

DESIGN_PARAM_RANGES = {
    "mass_kg":                     (1400.0, 2200.0),
    "wheelbase_m":                 (2.55,   2.95),
    "track_m":                     (1.50,   1.65),
    "cg_height_m":                 (0.50,   0.70),
    "front_weight_ratio":          (0.50,   0.62),
    "suspension_stiffness_N_per_m":(25_000, 60_000),
    "damping_ratio":               (0.20,   0.45),
    "tire_pressure_kPa":           (200.0,  260.0),
}

# Feature engineering windows (in seconds)
WINDOW_SECONDS = 1.0
WINDOW_STRIDE_SECONDS = 0.5

RANDOM_SEED = 42
