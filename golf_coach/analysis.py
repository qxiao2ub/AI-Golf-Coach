"""Pose feature engineering, swing segmentation, and transparent scoring."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from .constants import (
    IMPORTANT_LANDMARKS,
    LEFT_ANKLE,
    LEFT_ELBOW,
    LEFT_HIP,
    LEFT_KNEE,
    LEFT_SHOULDER,
    LEFT_WRIST,
    NOSE,
    RIGHT_ANKLE,
    RIGHT_ELBOW,
    RIGHT_HIP,
    RIGHT_KNEE,
    RIGHT_SHOULDER,
    RIGHT_WRIST,
)
from .geometry import (
    angle_three_points,
    clamp,
    euclidean,
    line_angle_degrees,
    line_orientation_difference,
    midpoint,
    point,
    torso_tilt_from_vertical,
)


def _safe_mean(values: pd.Series, fallback: float = 0.0) -> float:
    cleaned = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return float(cleaned.mean()) if not cleaned.empty else float(fallback)


def _safe_max(values: pd.Series, fallback: float = 0.0) -> float:
    cleaned = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return float(cleaned.max()) if not cleaned.empty else float(fallback)


def _band_score(value: float, ideal_low: float, ideal_high: float, outer_low: float, outer_high: float) -> float:
    if not np.isfinite(value):
        return 0.0
    if ideal_low <= value <= ideal_high:
        return 100.0
    if value < ideal_low:
        return 100.0 * clamp((value - outer_low) / max(ideal_low - outer_low, 1e-9), 0.0, 1.0)
    return 100.0 * clamp((outer_high - value) / max(outer_high - ideal_high, 1e-9), 0.0, 1.0)


def _low_is_good_score(value: float, ideal_high: float, outer_high: float) -> float:
    if not np.isfinite(value):
        return 0.0
    if value <= ideal_high:
        return 100.0
    return 100.0 * clamp((outer_high - value) / max(outer_high - ideal_high, 1e-9), 0.0, 1.0)


def build_feature_dataframe(
    records: List[Dict[str, Any]],
    expected_samples: int,
) -> pd.DataFrame:
    """Convert MediaPipe landmarks into interpretable, frame-level golf indicators."""
    valid_records = [record for record in records if record.get("landmarks") is not None]
    if len(valid_records) < 4:
        raise RuntimeError("Not enough valid pose records were available for swing analysis.")

    first = valid_records[0]["landmarks"]
    shoulder_center_0 = midpoint(point(first, LEFT_SHOULDER), point(first, RIGHT_SHOULDER))
    hip_center_0 = midpoint(point(first, LEFT_HIP), point(first, RIGHT_HIP))
    shoulder_width_0 = max(euclidean(point(first, LEFT_SHOULDER), point(first, RIGHT_SHOULDER)), 1e-4)
    head_0 = point(first, NOSE)

    rows: List[Dict[str, Any]] = []
    for record in valid_records:
        landmarks = np.asarray(record["landmarks"], dtype=float)
        left_shoulder = point(landmarks, LEFT_SHOULDER)
        right_shoulder = point(landmarks, RIGHT_SHOULDER)
        left_hip = point(landmarks, LEFT_HIP)
        right_hip = point(landmarks, RIGHT_HIP)
        shoulder_center = midpoint(left_shoulder, right_shoulder)
        hip_center = midpoint(left_hip, right_hip)
        shoulder_width = max(euclidean(left_shoulder, right_shoulder), 1e-4)
        hand_center = midpoint(point(landmarks, LEFT_WRIST), point(landmarks, RIGHT_WRIST))
        shoulder_angle = line_angle_degrees(left_shoulder, right_shoulder)
        hip_angle = line_angle_degrees(left_hip, right_hip)
        visibility = float(np.mean(landmarks[IMPORTANT_LANDMARKS, 3]))

        rows.append({
            "frame_index": int(record["frame_index"]),
            "timestamp_s": float(record["timestamp_s"]),
            "left_elbow_angle": angle_three_points(
                left_shoulder, point(landmarks, LEFT_ELBOW), point(landmarks, LEFT_WRIST)
            ),
            "right_elbow_angle": angle_three_points(
                right_shoulder, point(landmarks, RIGHT_ELBOW), point(landmarks, RIGHT_WRIST)
            ),
            "left_knee_angle": angle_three_points(
                left_hip, point(landmarks, LEFT_KNEE), point(landmarks, LEFT_ANKLE)
            ),
            "right_knee_angle": angle_three_points(
                right_hip, point(landmarks, RIGHT_KNEE), point(landmarks, RIGHT_ANKLE)
            ),
            "torso_tilt_deg": torso_tilt_from_vertical(shoulder_center, hip_center),
            "shoulder_line_angle_deg": shoulder_angle,
            "hip_line_angle_deg": hip_angle,
            "shoulder_hip_separation_deg": line_orientation_difference(shoulder_angle, hip_angle),
            "hip_sway_shoulder_widths": euclidean(hip_center, hip_center_0) / shoulder_width_0,
            "head_motion_shoulder_widths": euclidean(point(landmarks, NOSE), head_0) / shoulder_width_0,
            "hand_center_x": float(hand_center[0]),
            "hand_center_y": float(hand_center[1]),
            "shoulder_width": shoulder_width,
            "pose_visibility": visibility,
            "shoulder_center_x": float(shoulder_center[0]),
            "shoulder_center_y": float(shoulder_center[1]),
            "hip_center_x": float(hip_center[0]),
            "hip_center_y": float(hip_center[1]),
            "address_center_shift": euclidean(shoulder_center, shoulder_center_0) / shoulder_width_0,
        })

    dataframe = pd.DataFrame(rows).sort_values("frame_index").reset_index(drop=True)
    dt = dataframe["timestamp_s"].diff().replace(0, np.nan)
    dx = dataframe["hand_center_x"].diff()
    dy = dataframe["hand_center_y"].diff()
    speed = np.sqrt(dx.pow(2) + dy.pow(2)) / dt / dataframe["shoulder_width"].clip(lower=1e-4)
    dataframe["hand_speed_shoulder_widths_per_s"] = speed.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    dataframe["hand_speed_smoothed"] = (
        dataframe["hand_speed_shoulder_widths_per_s"].rolling(5, center=True, min_periods=1).median()
    )
    acceleration = dataframe["hand_speed_smoothed"].diff() / dt
    dataframe["hand_acceleration"] = acceleration.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    dataframe["pose_detection_coverage"] = len(valid_records) / max(int(expected_samples), 1)
    dataframe = assign_swing_phases(dataframe)
    return dataframe


def assign_swing_phases(dataframe: pd.DataFrame) -> pd.DataFrame:
    result = dataframe.copy()
    count = len(result)
    if count < 4:
        result["phase"] = "Swing"
        return result

    address_end = max(1, int(round(count * 0.12)))
    top_search_end = max(address_end + 2, int(round(count * 0.70)))
    top_search_end = min(top_search_end, count)
    top_index = int(result.loc[address_end:top_search_end - 1, "hand_center_y"].idxmin())
    if top_index >= count - 2:
        top_index = max(address_end + 1, int(round(count * 0.45)))

    after_top = result.loc[top_index + 1:]
    if after_top.empty:
        impact_index = min(count - 1, top_index + 1)
    else:
        impact_index = int(after_top["hand_speed_smoothed"].idxmax())
        impact_index = max(impact_index, top_index + 1)

    phases: List[str] = []
    for row_index in range(count):
        if row_index <= address_end:
            phases.append("Address")
        elif row_index < top_index:
            phases.append("Backswing")
        elif row_index == top_index:
            phases.append("Top")
        elif row_index < impact_index:
            phases.append("Downswing")
        elif row_index == impact_index:
            phases.append("Impact")
        else:
            phases.append("Follow-through")
    result["phase"] = phases
    result.attrs["address_end_index"] = address_end
    result.attrs["top_index"] = top_index
    result.attrs["impact_index"] = impact_index
    return result


def summarize_swing(dataframe: pd.DataFrame, expected_samples: int) -> Tuple[Dict[str, float], Dict[str, float]]:
    if dataframe.empty:
        raise RuntimeError("The feature table is empty.")

    address_rows = dataframe[dataframe["phase"] == "Address"]
    if address_rows.empty:
        address_rows = dataframe.head(max(2, len(dataframe) // 10))
    impact_rows = dataframe[dataframe["phase"] == "Impact"]
    if impact_rows.empty:
        impact_rows = dataframe.iloc[[int(dataframe["hand_speed_smoothed"].idxmax())]]

    top_rows = dataframe[dataframe["phase"] == "Top"]
    top_time = float(top_rows["timestamp_s"].iloc[0]) if not top_rows.empty else float(dataframe["timestamp_s"].quantile(0.45))
    impact_time = float(impact_rows["timestamp_s"].iloc[0])
    address_time = float(address_rows["timestamp_s"].iloc[-1])
    backswing_duration = max(top_time - address_time, 1e-3)
    downswing_duration = max(impact_time - top_time, 1e-3)
    tempo_ratio = backswing_duration / downswing_duration

    acceleration_std = float(dataframe["hand_acceleration"].replace([np.inf, -np.inf], np.nan).fillna(0.0).std())
    smoothness_index = float(1.0 / (1.0 + acceleration_std))
    detection_coverage = len(dataframe) / max(int(expected_samples), 1)

    summary = {
        "address_torso_tilt_deg": _safe_mean(address_rows["torso_tilt_deg"]),
        "address_mean_knee_angle_deg": _safe_mean(
            pd.concat([address_rows["left_knee_angle"], address_rows["right_knee_angle"]])
        ),
        "max_shoulder_hip_separation_deg": _safe_max(dataframe["shoulder_hip_separation_deg"]),
        "max_hip_sway_shoulder_widths": _safe_max(dataframe["hip_sway_shoulder_widths"]),
        "max_head_motion_shoulder_widths": _safe_max(dataframe["head_motion_shoulder_widths"]),
        "impact_mean_elbow_angle_deg": _safe_mean(
            pd.concat([impact_rows["left_elbow_angle"], impact_rows["right_elbow_angle"]])
        ),
        "tempo_ratio": float(tempo_ratio),
        "backswing_duration_s": float(backswing_duration),
        "downswing_duration_s": float(downswing_duration),
        "peak_hand_speed_shoulder_widths_per_s": _safe_max(dataframe["hand_speed_smoothed"]),
        "smoothness_index": smoothness_index,
        "detection_coverage": float(clamp(detection_coverage, 0.0, 1.0)),
        "mean_pose_visibility": _safe_mean(dataframe["pose_visibility"]),
        "analyzed_frames": float(len(dataframe)),
        "analyzed_duration_s": float(dataframe["timestamp_s"].iloc[-1] - dataframe["timestamp_s"].iloc[0]),
    }

    posture = 0.60 * _band_score(summary["address_torso_tilt_deg"], 8.0, 38.0, 0.0, 62.0)
    posture += 0.40 * _band_score(summary["address_mean_knee_angle_deg"], 142.0, 174.0, 115.0, 180.0)
    balance = 0.55 * _low_is_good_score(summary["max_hip_sway_shoulder_widths"], 0.30, 0.90)
    balance += 0.45 * _low_is_good_score(summary["max_head_motion_shoulder_widths"], 0.38, 1.05)
    rotation = _band_score(summary["max_shoulder_hip_separation_deg"], 12.0, 52.0, 0.0, 78.0)
    tempo = _band_score(summary["tempo_ratio"], 1.8, 4.0, 0.65, 6.5)
    arm_extension = _band_score(summary["impact_mean_elbow_angle_deg"], 138.0, 178.0, 95.0, 180.0)
    consistency = 100.0 * clamp(summary["smoothness_index"] / 0.50, 0.0, 1.0)
    pose_confidence = 100.0 * clamp(
        0.65 * summary["detection_coverage"] + 0.35 * summary["mean_pose_visibility"], 0.0, 1.0
    )

    components = {
        "Posture": float(posture),
        "Balance": float(balance),
        "Rotation": float(rotation),
        "Tempo": float(tempo),
        "Arm extension": float(arm_extension),
        "Consistency": float(consistency),
        "Pose confidence": float(pose_confidence),
    }
    weights = {
        "Posture": 0.16,
        "Balance": 0.16,
        "Rotation": 0.18,
        "Tempo": 0.16,
        "Arm extension": 0.14,
        "Consistency": 0.10,
        "Pose confidence": 0.10,
    }
    overall = sum(components[name] * weights[name] for name in components)
    summary["heuristic_overall_score"] = float(clamp(overall, 0.0, 100.0))
    return summary, components


def frame_feature_map(dataframe: pd.DataFrame) -> Dict[int, Dict[str, Any]]:
    return {
        int(row["frame_index"]): row.to_dict()
        for _, row in dataframe.iterrows()
    }
