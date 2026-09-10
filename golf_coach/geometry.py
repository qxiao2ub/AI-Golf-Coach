"""Small, testable geometry helpers."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def clamp(value: float, low: float, high: float) -> float:
    return float(max(low, min(high, value)))


def smoothstep(value: float) -> float:
    value = clamp(value, 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


def lerp(a: np.ndarray, b: np.ndarray, amount: float) -> np.ndarray:
    return np.asarray(a, dtype=float) + (
        np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
    ) * smoothstep(amount)


def point(landmarks: np.ndarray, index: int) -> np.ndarray:
    return np.asarray(landmarks[index, :2], dtype=float)


def midpoint(a: Iterable[float], b: Iterable[float]) -> np.ndarray:
    return (np.asarray(a, dtype=float) + np.asarray(b, dtype=float)) / 2.0


def euclidean(a: Iterable[float], b: Iterable[float]) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)))


def angle_three_points(a: Iterable[float], b: Iterable[float], c: Iterable[float]) -> float:
    """Return angle ABC in degrees from 0 to 180."""
    ba = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    bc = np.asarray(c, dtype=float) - np.asarray(b, dtype=float)
    denominator = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denominator < 1e-9:
        return float("nan")
    cosine = float(np.dot(ba, bc) / denominator)
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def line_angle_degrees(a: Iterable[float], b: Iterable[float]) -> float:
    delta = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
    return float(np.degrees(np.arctan2(delta[1], delta[0])))


def line_orientation_difference(a_degrees: float, b_degrees: float) -> float:
    """Smallest unsigned difference between two undirected lines, from 0 to 90."""
    difference = abs((a_degrees - b_degrees + 90.0) % 180.0 - 90.0)
    return float(difference)


def torso_tilt_from_vertical(shoulder_center: np.ndarray, hip_center: np.ndarray) -> float:
    vector = np.asarray(shoulder_center, dtype=float) - np.asarray(hip_center, dtype=float)
    if np.linalg.norm(vector) < 1e-9:
        return float("nan")
    return float(abs(math.degrees(math.atan2(vector[0], -vector[1]))))
