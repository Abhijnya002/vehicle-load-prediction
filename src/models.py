"""Regression model factory + training utilities.

All models predict the full multi-output target vector
(load_FL, load_FR, load_RL, load_RR, vibration_rms). Linear and Ridge
naturally support multi-output regression; tree ensembles are wrapped in
MultiOutputRegressor so each target gets a dedicated forest/booster.
"""
from __future__ import annotations

from dataclasses import dataclass

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import MODELS_DIR, RANDOM_SEED


def make_models() -> dict[str, Pipeline]:
    """Return name -> sklearn Pipeline for every candidate model."""
    return {
        "linear": Pipeline([
            ("scaler", StandardScaler()),
            ("reg",    LinearRegression()),
        ]),
        "ridge": Pipeline([
            ("scaler", StandardScaler()),
            ("reg",    Ridge(alpha=1.0, random_state=RANDOM_SEED)),
        ]),
        "random_forest": Pipeline([
            ("scaler", StandardScaler(with_mean=False)),
            ("reg",    RandomForestRegressor(
                n_estimators=200,
                max_depth=None,
                min_samples_leaf=2,
                n_jobs=-1,
                random_state=RANDOM_SEED,
            )),
        ]),
        "gradient_boosting": Pipeline([
            ("scaler", StandardScaler()),
            ("reg",    MultiOutputRegressor(GradientBoostingRegressor(
                n_estimators=200,
                max_depth=3,
                learning_rate=0.05,
                random_state=RANDOM_SEED,
            ))),
        ]),
    }


@dataclass
class CVReport:
    model_name: str
    mean_r2: float
    std_r2: float
    fold_r2: list[float]


def cross_validate_grouped(model: Pipeline, X, y, groups, *, n_splits: int = 5) -> CVReport:
    """Group-aware CV: drives are kept together across folds.

    Prevents leakage from neighbouring window samples in the same drive.
    """
    n_splits = min(n_splits, len(np.unique(groups)))
    cv = GroupKFold(n_splits=n_splits)
    scores = cross_val_score(model, X, y, groups=groups, cv=cv, scoring="r2", n_jobs=-1)
    return CVReport(
        model_name=type(model.named_steps["reg"]).__name__,
        mean_r2=float(np.mean(scores)),
        std_r2=float(np.std(scores)),
        fold_r2=[float(s) for s in scores],
    )


def save_model(name: str, model: Pipeline) -> str:
    path = MODELS_DIR / f"{name}.joblib"
    joblib.dump(model, path)
    return str(path)


def load_model(name: str) -> Pipeline:
    path = MODELS_DIR / f"{name}.joblib"
    return joblib.load(path)
