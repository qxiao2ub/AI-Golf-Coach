from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np

from golf_coach.analysis import build_feature_dataframe, frame_feature_map, summarize_swing
from golf_coach.coaching import ContextualBanditCoach, build_advice, build_practice_plan, weakest_state
from golf_coach.geometry import angle_three_points, line_orientation_difference
from golf_coach.models import predict_demo_models
from golf_coach.reports import build_download_package, build_report_html
from golf_coach.video import create_annotated_video, create_demo_video


def test_geometry_helpers() -> None:
    angle = angle_three_points(np.array([1.0, 0.0]), np.array([0.0, 0.0]), np.array([0.0, 1.0]))
    assert abs(angle - 90.0) < 1e-6
    assert abs(line_orientation_difference(10.0, 170.0) - 20.0) < 1e-6


def test_contextual_bandit_update() -> None:
    bandit = ContextualBanditCoach()
    state = "Tempo"
    action = bandit.recommend(state, epsilon=0.0, seed=42)
    reward = bandit.update(state, action, 5)
    assert reward == 1.0
    assert bandit.counts[state][action] == 1
    assert bandit.q_values[state][action] == 1.0


def test_complete_demo_pipeline(tmp_path: Path) -> None:
    source_path, records, metadata = create_demo_video(
        tmp_path / "demo.mp4", seconds=2.0, fps=18, width=320, height=240
    )
    features = build_feature_dataframe(records, expected_samples=metadata["expected_samples"])
    summary, components = summarize_swing(features, expected_samples=metadata["expected_samples"])
    predictions, model_metrics = predict_demo_models(summary)
    advice = build_advice(components, summary)
    profile = {
        "name": "Test Golfer",
        "age": 18,
        "height_cm": 175.0,
        "weight_kg": 70.0,
        "bmi": 22.86,
        "experience_level": "Beginner",
        "gym_activity_days_per_week": 3,
        "health_notes": "",
        "consent_to_process_video": True,
    }
    plan = build_practice_plan(advice, profile)
    state, _ = weakest_state(components)
    assert state in components
    assert 0.0 <= summary["heuristic_overall_score"] <= 100.0
    assert set(predictions) == {"Random Forest prototype", "Deep neural network prototype"}
    assert len(advice) == 4
    assert len(plan) == 4

    annotated_path = create_annotated_video(
        source_path,
        tmp_path / "annotated.mp4",
        records,
        frame_feature_map(features),
        summary["heuristic_overall_score"],
        frame_stride=1,
    )
    assert annotated_path.exists()
    assert annotated_path.stat().st_size > 0

    report = build_report_html(profile, metadata, summary, components, predictions, advice, plan)
    assert "Isaiah Goh" in report
    assert "Dr. Qingyang Xiao" in report

    package = build_download_package(
        profile,
        metadata,
        summary,
        components,
        predictions,
        model_metrics,
        advice,
        plan,
        features,
        ContextualBanditCoach().payload(),
        annotated_path,
    )
    with zipfile.ZipFile(io.BytesIO(package)) as archive:
        names = set(archive.namelist())
        assert "golf_coach_report.html" in names
        assert "swing_frame_features.csv" in names
        assert "annotated_golf_swing.mp4" in names
        assert "MODEL_CARD.md" in names
