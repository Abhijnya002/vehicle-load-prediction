"""Synthetic CAN + accelerometer dataset generator.

Each "drive" is a fixed-duration recording sampled at SAMPLE_HZ. Vehicle
dynamics are simulated from a parameterised design point, producing both
the signals that a real CAN bus + IMU would log and the ground-truth
corner loads / chassis vibration that we want the ML model to predict.

The physics is deliberately simple but mechanistic: longitudinal and
lateral weight transfer driven by quasi-static dynamics, plus a vertical
road-roughness component filtered through a single-DoF suspension model.
This gives the regressors a realistic, learnable signal structure.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

from src.config import (
    CAN_SIGNALS,
    DESIGN_PARAM_RANGES,
    DESIGN_PARAMS,
    DRIVE_DURATION_S,
    G,
    SAMPLE_HZ,
    TARGETS,
)


@dataclass
class DesignPoint:
    mass_kg: float
    wheelbase_m: float
    track_m: float
    cg_height_m: float
    front_weight_ratio: float
    suspension_stiffness_N_per_m: float
    damping_ratio: float
    tire_pressure_kPa: float

    @classmethod
    def sample(cls, rng: np.random.Generator) -> "DesignPoint":
        kwargs = {}
        for name in DESIGN_PARAMS:
            lo, hi = DESIGN_PARAM_RANGES[name]
            kwargs[name] = float(rng.uniform(lo, hi))
        return cls(**kwargs)


def _lowpass(signal: np.ndarray, cutoff_hz: float, fs: float, order: int = 3) -> np.ndarray:
    nyq = 0.5 * fs
    b, a = butter(order, cutoff_hz / nyq, btype="low")
    return filtfilt(b, a, signal)


def _velocity_profile(n: int, fs: float, rng: np.random.Generator) -> np.ndarray:
    """Piecewise target speed (m/s), smoothed to remove discontinuities."""
    target = np.zeros(n)
    i = 0
    speed = float(rng.uniform(0.0, 8.0))
    while i < n:
        seg_dur = rng.uniform(3.0, 10.0)
        seg_n = min(int(seg_dur * fs), n - i)
        mode = rng.choice(["accel", "cruise", "brake"], p=[0.4, 0.3, 0.3])
        if mode == "accel":
            next_speed = min(speed + rng.uniform(4.0, 14.0), 33.0)
        elif mode == "brake":
            next_speed = max(speed - rng.uniform(4.0, 14.0), 0.0)
        else:
            next_speed = speed + rng.normal(0.0, 0.5)
        target[i:i + seg_n] = np.linspace(speed, next_speed, seg_n)
        speed = next_speed
        i += seg_n
    smoothed = _lowpass(target, cutoff_hz=0.8, fs=fs)
    return np.clip(smoothed, 0.0, None)


def _steering_profile(n: int, fs: float, rng: np.random.Generator) -> np.ndarray:
    """Series of sinusoidal corner events, in radians at the road wheel."""
    angle = np.zeros(n)
    i = 0
    while i < n:
        gap = int(rng.uniform(1.5, 4.0) * fs)
        i += gap
        if i >= n:
            break
        dur = int(rng.uniform(1.0, 3.0) * fs)
        end = min(i + dur, n)
        amplitude = rng.uniform(-0.20, 0.20)
        phase = np.linspace(0, np.pi, end - i)
        angle[i:end] += amplitude * np.sin(phase)
        i = end
    return _lowpass(angle, cutoff_hz=2.0, fs=fs)


def _road_excitation(n: int, fs: float, rng: np.random.Generator) -> np.ndarray:
    """Random road profile (m/s^2 input to the wheel) with broadband content."""
    noise = rng.standard_normal(n)
    rough = _lowpass(noise, cutoff_hz=15.0, fs=fs) * 1.8
    # Inject occasional bumps
    n_bumps = rng.integers(2, 8)
    for _ in range(n_bumps):
        center = int(rng.uniform(0, n))
        width = int(rng.uniform(0.05, 0.25) * fs)
        amp = rng.uniform(-3.0, 3.0)
        lo, hi = max(0, center - width), min(n, center + width)
        rough[lo:hi] += amp * np.hanning(hi - lo)
    return rough


def _suspension_response(road_accel: np.ndarray, dp: DesignPoint, fs: float) -> np.ndarray:
    """Sprung-mass vertical acceleration via a 1-DoF mass-spring-damper.

    Solved with explicit Euler at sample rate. Output is the body-mounted
    vertical accelerometer reading minus gravity bias (i.e. the AC part).
    """
    m = dp.mass_kg / 4.0  # per-corner sprung mass (rough)
    k = dp.suspension_stiffness_N_per_m
    c = 2.0 * dp.damping_ratio * np.sqrt(k * m)

    dt = 1.0 / fs
    n = len(road_accel)
    x = np.zeros(n)
    v = np.zeros(n)
    a = np.zeros(n)
    # Integrate road acceleration to get road displacement input
    road_vel = np.cumsum(road_accel) * dt
    road_disp = np.cumsum(road_vel) * dt
    for i in range(1, n):
        # F = k*(road_disp - x) + c*(road_vel - v)
        a[i] = (k * (road_disp[i] - x[i - 1]) + c * (road_vel[i] - v[i - 1])) / m
        v[i] = v[i - 1] + a[i] * dt
        x[i] = x[i - 1] + v[i] * dt
    return a


def _engine_rpm(speed_mps: np.ndarray, gear: np.ndarray) -> np.ndarray:
    """RPM = wheel rpm * final drive * gear ratio. Simple lookup."""
    gear_ratio = np.array([3.5, 2.1, 1.4, 1.0, 0.8, 0.65])
    final_drive = 3.7
    wheel_radius = 0.32  # m
    wheel_rpm = (speed_mps / (2.0 * np.pi * wheel_radius)) * 60.0
    ratios = gear_ratio[np.clip(gear.astype(int) - 1, 0, len(gear_ratio) - 1)]
    rpm = wheel_rpm * final_drive * ratios
    # Idle minimum
    return np.clip(rpm, 750.0, 7000.0)


def _choose_gear(speed_mps: np.ndarray) -> np.ndarray:
    bins = np.array([0.0, 5.0, 10.0, 16.0, 22.0, 28.0])  # upshift thresholds
    return np.clip(np.searchsorted(bins, speed_mps), 1, 6).astype(np.int8)


def generate_drive(drive_id: int, rng: np.random.Generator) -> pd.DataFrame:
    """Simulate one drive cycle and return a long-form DataFrame."""
    fs = SAMPLE_HZ
    n = int(DRIVE_DURATION_S * fs)
    t = np.arange(n) / fs

    dp = DesignPoint.sample(rng)

    speed = _velocity_profile(n, fs, rng)
    ax = np.gradient(speed, 1.0 / fs)
    ax_filt = _lowpass(ax, cutoff_hz=3.0, fs=fs)

    steering = _steering_profile(n, fs, rng)
    # Bicycle-model lateral accel: ay ~ v^2 * tan(delta) / wheelbase
    ay = (speed ** 2) * np.tan(steering) / dp.wheelbase_m
    ay = _lowpass(ay, cutoff_hz=3.0, fs=fs)

    road_input = _road_excitation(n, fs, rng)
    # Tire pressure modulates road->body transmission (lower psi = softer = less HF)
    pressure_gain = np.interp(dp.tire_pressure_kPa, [200.0, 260.0], [0.7, 1.0])
    az_ac = _suspension_response(road_input * pressure_gain, dp, fs)
    az = G + az_ac + rng.normal(0.0, 0.05, size=n)  # measured (gravity + dynamic)

    gear = _choose_gear(speed)
    rpm = _engine_rpm(speed, gear)

    # Pedal positions derived from desired ax (simple driver model)
    throttle = np.clip(ax_filt / 4.0, 0.0, 1.0) * 100.0
    brake = np.clip(-ax_filt / 6.0, 0.0, 1.0) * 100.0

    # --- Ground-truth corner loads (quasi-static weight transfer) ----------
    static_front = dp.mass_kg * G * dp.front_weight_ratio
    static_rear = dp.mass_kg * G * (1.0 - dp.front_weight_ratio)
    long_transfer = dp.mass_kg * ax_filt * dp.cg_height_m / dp.wheelbase_m
    lat_transfer = dp.mass_kg * ay * dp.cg_height_m / dp.track_m

    # Vertical road dynamics modulate the per-corner load
    vertical_force = dp.mass_kg * az_ac  # total dynamic vertical force on body
    vert_per_corner = vertical_force / 4.0

    load_FL = (static_front - long_transfer) / 2.0 + lat_transfer / 2.0 + vert_per_corner
    load_FR = (static_front - long_transfer) / 2.0 - lat_transfer / 2.0 + vert_per_corner
    load_RL = (static_rear + long_transfer) / 2.0 + lat_transfer / 2.0 + vert_per_corner
    load_RR = (static_rear + long_transfer) / 2.0 - lat_transfer / 2.0 + vert_per_corner

    # Vibration RMS computed in a 1s causal window (sample-level target).
    win = int(1.0 * fs)
    az_sq = az_ac ** 2
    csum = np.concatenate(([0.0], np.cumsum(az_sq)))
    sums = csum[np.arange(n) + 1] - csum[np.maximum(0, np.arange(n) + 1 - win)]
    counts = np.minimum(np.arange(n) + 1, win)
    vibration_rms = np.sqrt(sums / counts)

    # Sensor noise on the logged CAN signals
    speed_meas = speed + rng.normal(0.0, 0.05, n)
    rpm_meas = rpm + rng.normal(0.0, 8.0, n)
    throttle_meas = np.clip(throttle + rng.normal(0.0, 0.5, n), 0.0, 100.0)
    brake_meas = np.clip(brake + rng.normal(0.0, 0.5, n), 0.0, 100.0)
    steering_meas = steering + rng.normal(0.0, 0.003, n)
    ax_meas = ax_filt + rng.normal(0.0, 0.05, n)
    ay_meas = ay + rng.normal(0.0, 0.05, n)

    df = pd.DataFrame({
        "drive_id": drive_id,
        "t": t,
        "vehicle_speed_mps": speed_meas,
        "engine_rpm": rpm_meas,
        "throttle_pct": throttle_meas,
        "brake_pct": brake_meas,
        "steering_angle_rad": steering_meas,
        "gear": gear,
        "accel_x_mps2": ax_meas,
        "accel_y_mps2": ay_meas,
        "accel_z_mps2": az,
        "load_FL_N": load_FL,
        "load_FR_N": load_FR,
        "load_RL_N": load_RL,
        "load_RR_N": load_RR,
        "vibration_rms_mps2": vibration_rms,
    })

    # Append design parameters as constant columns (one row per drive sample)
    for name, val in asdict(dp).items():
        df[name] = val
    return df


def generate_dataset(n_drives: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frames = []
    for drive_id in range(n_drives):
        # Reseed per-drive so individual drives are reproducible
        sub_rng = np.random.default_rng(rng.integers(0, 2**32 - 1))
        frames.append(generate_drive(drive_id, sub_rng))
    out = pd.concat(frames, ignore_index=True)
    # Sanity-check column set
    expected = ["drive_id", "t"] + CAN_SIGNALS + TARGETS + DESIGN_PARAMS
    missing = [c for c in expected if c not in out.columns]
    if missing:
        raise RuntimeError(f"Missing expected columns: {missing}")
    return out
