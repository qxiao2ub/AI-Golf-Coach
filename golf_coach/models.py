"""Synthetic-data ML and neural-network demonstrations for the MVP."""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .geometry import clamp

ML_FEATURE_COLUMNS = [
    "address_torso_tilt_deg",
    "address_mean_knee_angle_deg",
    "max_shoulder_hip_separation_deg",
    "max_hip_sway_shoulder_widths",
    "max_head_motion_shoulder_widths",
    "impact_mean_elbow_angle_deg",
    "tempo_ratio",
    "smoothness_index",
    "detection_coverage",
]


def _band_score(value: float, ideal_low: float, ideal_high: float, outer_low: float, outer_high: float) -> float:
    if ideal_low <= value <= ideal_high:
        return 100.0
    if value < ideal_low:
        return 100.0 * clamp((value - outer_low) / max(ideal_low - outer_low, 1e-9), 0.0, 1.0)
    return 100.0 * clamp((outer_high - value) / max(outer_high - ideal_high, 1e-9), 0.0, 1.0)


def _low_is_good(value: float, ideal_high: float, outer_high: float) -> float:
    if value <= ideal_high:
        return 100.0
    return 100.0 * clamp((outer_high - value) / max(outer_high - ideal_high, 1e-9), 0.0, 1.0)


def transparent_target(row: Dict[str, float]) -> float:
    """A visible scoring rule used only to synthesize architecture-demo labels."""
    posture = 0.55 * _band_score(row["address_torso_tilt_deg"], 8.0, 38.0, 0.0, 62.0)
    posture += 0.45 * _band_score(row["address_mean_knee_angle_deg"], 142.0, 174.0, 115.0, 180.0)
    rotation = _band_score(row["max_shoulder_hip_separation_deg"], 12.0, 52.0, 0.0, 78.0)
    balance = 0.55 * _low_is_good(row["max_hip_sway_shoulder_widths"], 0.30, 0.90)
    balance += 0.45 * _low_is_good(row["max_head_motion_shoulder_widths"], 0.38, 1.05)
    extension = _band_score(row["impact_mean_elbow_angle_deg"], 138.0, 178.0, 95.0, 180.0)
    tempo = _band_score(row["tempo_ratio"], 1.8, 4.0, 0.65, 6.5)
    smoothness = 100.0 * clamp(row["smoothness_index"] / 0.50, 0.0, 1.0)
    coverage = 100.0 * clamp(row["detection_coverage"], 0.0, 1.0)
    target = (
        0.18 * posture
        + 0.17 * rotation
        + 0.17 * balance
        + 0.14 * extension
        + 0.14 * tempo
        + 0.10 * smoothness
        + 0.10 * coverage
    )
    return float(clamp(target, 0.0, 100.0))


def make_synthetic_training_data(size: int = 900, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dataframe = pd.DataFrame({
        "address_torso_tilt_deg": rng.uniform(0.0, 70.0, size),
        "address_mean_knee_angle_deg": rng.uniform(105.0, 180.0, size),
        "max_shoulder_hip_separation_deg": rng.uniform(0.0, 90.0, size),
        "max_hip_sway_shoulder_widths": rng.uniform(0.0, 1.20, size),
        "max_head_motion_shoulder_widths": rng.uniform(0.0, 1.35, size),
        "impact_mean_elbow_angle_deg": rng.uniform(85.0, 180.0, size),
        "tempo_ratio": rng.uniform(0.50, 7.0, size),
        "smoothness_index": rng.uniform(0.05, 0.80, size),
        "detection_coverage": rng.uniform(0.45, 1.0, size),
    })
    labels = []
    for row in dataframe.to_dict(orient="records"):
        noise = rng.normal(0.0, 2.2)
        labels.append(clamp(transparent_target(row) + noise, 0.0, 100.0))
    dataframe["coach_score"] = labels
    return dataframe


@lru_cache(maxsize=1)
def train_demo_models() -> Tuple[Any, Any, Dict[str, float]]:
    """Train cached Random Forest and three-hidden-layer MLP architecture demonstrations."""
    data = make_synthetic_training_data()
    x_train, x_test, y_train, y_test = train_test_split(
        data[ML_FEATURE_COLUMNS],
        data["coach_score"],
        test_size=0.22,
        random_state=42,
    )
    random_forest = RandomForestRegressor(
        n_estimators=160,
        max_depth=10,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    neural_network = make_pipeline(
        StandardScaler(),
        MLPRegressor(
            hidden_layer_sizes=(64, 32, 16),
            activation="relu",
            solver="adam",
            alpha=0.001,
            learning_rate_init=0.002,
            max_iter=700,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=30,
            random_state=42,
        ),
    )
    random_forest.fit(x_train, y_train)
    neural_network.fit(x_train, y_train)
    forest_prediction = random_forest.predict(x_test)
    neural_prediction = neural_network.predict(x_test)
    metrics = {
        "random_forest_mae_synthetic": float(mean_absolute_error(y_test, forest_prediction)),
        "random_forest_r2_synthetic": float(r2_score(y_test, forest_prediction)),
        "neural_network_mae_synthetic": float(mean_absolute_error(y_test, neural_prediction)),
        "neural_network_r2_synthetic": float(r2_score(y_test, neural_prediction)),
        "synthetic_training_rows": float(len(data)),
    }
    return random_forest, neural_network, metrics


def predict_demo_models(summary: Dict[str, float]) -> Tuple[Dict[str, float], Dict[str, float]]:
    random_forest, neural_network, metrics = train_demo_models()
    row = pd.DataFrame([{name: float(summary[name]) for name in ML_FEATURE_COLUMNS}])
    predictions = {
        "Random Forest prototype": float(clamp(random_forest.predict(row)[0], 0.0, 100.0)),
        "Deep neural network prototype": float(clamp(neural_network.predict(row)[0], 0.0, 100.0)),
    }
    return predictions, metrics
