"""Self-contained report and ZIP-package generation."""

from __future__ import annotations

import html
import io
import json
import zipfile
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

AUTHOR = "Isaiah Goh"
MENTOR = "Dr. Qingyang Xiao"


def _json_scalar(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _clean_dict(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {key: _json_scalar(value) for key, value in payload.items()}


def _table(dataframe: pd.DataFrame) -> str:
    return dataframe.to_html(index=False, border=0, classes="dataframe", escape=True)


def build_report_html(
    profile: Dict[str, Any],
    metadata: Dict[str, Any],
    summary: Dict[str, float],
    component_scores: Dict[str, float],
    model_predictions: Dict[str, float],
    advice: pd.DataFrame,
    practice_plan: pd.DataFrame,
    include_health_notes: bool = False,
) -> str:
    visible_profile = {
        key: value
        for key, value in profile.items()
        if key not in {"consent_to_process_video"}
        and (include_health_notes or key != "health_notes")
    }
    profile_table = pd.DataFrame(
        [(key.replace("_", " ").title(), value) for key, value in visible_profile.items()],
        columns=["Field", "Value"],
    )
    summary_table = pd.DataFrame(
        [
            (key.replace("_", " ").title(), round(float(value), 3))
            for key, value in summary.items()
        ],
        columns=["Metric", "Value"],
    )
    components_table = pd.DataFrame(
        [(name, round(float(score), 1)) for name, score in component_scores.items()],
        columns=["Component", "Score"],
    )
    models_table = pd.DataFrame(
        [(name, round(float(score), 1)) for name, score in model_predictions.items()],
        columns=["Architecture", "Prototype score"],
    )
    limitations = [
        "This is an educational prototype, not a certified coaching, medical, rehabilitation, or injury-risk system.",
        "Two-dimensional pose estimates depend on camera position, clothing, lighting, occlusion, and frame rate.",
        "The Random Forest and neural-network demonstrations are trained on synthetic labels, not coach-validated outcomes.",
        "The clubhead, ball flight, ground forces, and three-dimensional joint motion are outside this prototype's scope.",
        "Feedback learning can reflect preference bias and should not replace objective outcome evaluation.",
    ]
    safe_name = html.escape(str(profile.get("name", "Golfer")))
    overall = float(summary["heuristic_overall_score"])
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>AI Golf Coach Report - {safe_name}</title>
<style>
body {{ font-family: Arial, sans-serif; max-width: 1080px; margin: 32px auto; padding: 0 22px; line-height: 1.5; color: #17202a; }}
h1, h2 {{ color: #123c33; }}
.score {{ font-size: 38px; font-weight: 700; margin: 8px 0; }}
.notice {{ background: #fff6d8; border-left: 5px solid #c18c00; padding: 12px 16px; }}
.meta {{ color: #4b5563; }}
table {{ border-collapse: collapse; width: 100%; margin: 12px 0 26px; }}
th, td {{ border-bottom: 1px solid #d1d5db; text-align: left; padding: 8px; vertical-align: top; }}
th {{ background: #edf4f1; }}
footer {{ margin-top: 36px; padding-top: 18px; border-top: 1px solid #d1d5db; color: #4b5563; }}
</style>
</head>
<body>
<h1>AI-Based Golf Coach Report</h1>
<p class="meta"><strong>Author:</strong> {AUTHOR}<br><strong>Mentor:</strong> {MENTOR}</p>
<p class="score">{overall:.1f}/100</p>
<p class="notice"><strong>Educational prototype:</strong> Treat this score as a camera-dependent practice aid, not a professional diagnosis or final coaching judgment.</p>
<h2>Golfer Profile</h2>
{_table(profile_table)}
<h2>Video Analysis Metadata</h2>
{_table(pd.DataFrame([(key.replace('_', ' ').title(), _json_scalar(value)) for key, value in metadata.items()], columns=['Field', 'Value']))}
<h2>Swing Summary</h2>
{_table(summary_table)}
<h2>Component Scores</h2>
{_table(components_table)}
<h2>ML and Deep Neural Network Demonstrations</h2>
<p>These predictions demonstrate pipeline architecture only. Their training labels are synthetic.</p>
{_table(models_table)}
<h2>Priority Coaching Suggestions</h2>
{_table(advice)}
<h2>Four-Week Practice Plan</h2>
{_table(practice_plan)}
<h2>Limitations</h2>
<ul>{''.join(f'<li>{html.escape(item)}</li>' for item in limitations)}</ul>
<footer>Created by {AUTHOR} under the mentorship of {MENTOR}.</footer>
</body>
</html>"""


def build_download_package(
    profile: Dict[str, Any],
    metadata: Dict[str, Any],
    summary: Dict[str, float],
    component_scores: Dict[str, float],
    model_predictions: Dict[str, float],
    model_metrics: Dict[str, float],
    advice: pd.DataFrame,
    practice_plan: pd.DataFrame,
    features: pd.DataFrame,
    policy_payload: Dict[str, Any],
    annotated_video_path: Optional[Path],
    include_health_notes: bool = False,
) -> bytes:
    report_html = build_report_html(
        profile,
        metadata,
        summary,
        component_scores,
        model_predictions,
        advice,
        practice_plan,
        include_health_notes=include_health_notes,
    )
    profile_export = {
        key: value
        for key, value in profile.items()
        if key != "consent_to_process_video"
        and (include_health_notes or key != "health_notes")
    }
    summary_payload = {
        "project": "AI-Based Golf Coach",
        "author": AUTHOR,
        "mentor": MENTOR,
        "profile": _clean_dict(profile_export),
        "video_metadata": _clean_dict(metadata),
        "swing_summary": _clean_dict(summary),
        "component_scores": _clean_dict(component_scores),
        "model_predictions": _clean_dict(model_predictions),
        "synthetic_model_evaluation": _clean_dict(model_metrics),
        "limitations": [
            "2D, camera-dependent pose analysis",
            "Synthetic ML and neural-network labels",
            "No clubhead or ball-flight tracking",
            "No medical or injury-risk use",
        ],
    }
    model_card = f"""# AI Golf Coach Model Card

**Author:** {AUTHOR}  
**Mentor:** {MENTOR}

## Intended use
Educational demonstration of pose estimation, biomechanics feature engineering, supervised machine learning, a multi-layer neural network, feedback-based recommendation, and downloadable reporting.

## Not intended for
Medical diagnosis, injury prediction, rehabilitation decisions, certified coaching decisions, talent selection, or autonomous high-stakes recommendations.

## Training data
The default Random Forest and neural network use synthetic features and transparent synthetic labels. Replace them with consented, de-identified, coach-labeled data before reporting real predictive accuracy.

## Main limitations
- 2D camera perspective and occlusion distort angles.
- Clothing, lighting, frame rate, and camera motion affect pose quality.
- Clubhead and ball flight are not tracked.
- Fairness across demographic and ability groups has not been evaluated.
- User feedback can encode preference bias.
"""
    readme = f"""AI Golf Coach analysis package

Author: {AUTHOR}
Mentor: {MENTOR}

Files:
- golf_coach_report.html: self-contained report
- golf_coach_summary.json: machine-readable summary
- swing_frame_features.csv: sampled frame measurements
- coaching_recommendations.csv: prioritized feedback
- four_week_practice_plan.csv: practice plan
- rl_policy.json: feedback-learning state
- annotated_golf_swing.mp4: pose-annotated output, when available
- MODEL_CARD.md: intended use and limitations

Synthetic-data warning:
The bundled ML and neural-network predictions demonstrate architecture only. They are not validated measures of golf-coaching accuracy.
"""

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("golf_coach_report.html", report_html)
        archive.writestr("golf_coach_summary.json", json.dumps(summary_payload, indent=2))
        archive.writestr("swing_frame_features.csv", features.to_csv(index=False))
        archive.writestr("coaching_recommendations.csv", advice.to_csv(index=False))
        archive.writestr("four_week_practice_plan.csv", practice_plan.to_csv(index=False))
        archive.writestr("rl_policy.json", json.dumps(policy_payload, indent=2))
        archive.writestr("MODEL_CARD.md", model_card)
        archive.writestr("README_OUTPUTS.txt", readme)
        if annotated_video_path and annotated_video_path.exists():
            archive.write(annotated_video_path, arcname="annotated_golf_swing.mp4")
    return buffer.getvalue()
