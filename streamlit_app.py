"""Streamlit Community Cloud entry point for Isaiah Goh's AI Golf Coach."""

from __future__ import annotations

import json
import re
import shutil
import tempfile
import traceback
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import streamlit as st

from golf_coach.analysis import build_feature_dataframe, frame_feature_map, summarize_swing
from golf_coach.coaching import (
    ContextualBanditCoach,
    build_advice,
    build_practice_plan,
    weakest_state,
)
from golf_coach.models import predict_demo_models
from golf_coach.reports import build_download_package, build_report_html
from golf_coach.video import (
    PoseBackendUnavailable,
    create_annotated_video,
    create_demo_video,
    extract_pose_records,
    standardize_video,
)

AUTHOR = "Isaiah Goh"
MENTOR = "Dr. Qingyang Xiao"
APP_VERSION = "1.0.0"

st.set_page_config(
    page_title="AI Golf Coach | Isaiah Goh",
    page_icon="🏌️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
.block-container { padding-top: 1.7rem; padding-bottom: 3rem; }
.hero { padding: 1.2rem 1.4rem; border-radius: 18px; background: linear-gradient(135deg,#e9f6ee,#f7fbf8); border: 1px solid #cae4d3; margin-bottom: 1rem; }
.hero h1 { margin: 0; color: #123c33; }
.hero p { margin: .35rem 0 0; color: #35554d; }
.credit { font-size: .95rem; color: #52635f; }
.notice { padding: .8rem 1rem; border-left: 5px solid #b88400; background: #fff8df; border-radius: 6px; }
.good { padding: .8rem 1rem; border-left: 5px solid #278a56; background: #edf9f1; border-radius: 6px; }
.small { font-size: .9rem; color: #5b6663; }
footer { visibility: hidden; }
</style>
""",
    unsafe_allow_html=True,
)


def _safe_filename(name: str) -> str:
    base = Path(name).name
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", base)
    return cleaned or "uploaded_video.mp4"


def _new_workspace() -> Path:
    previous = st.session_state.get("workspace")
    if previous:
        shutil.rmtree(previous, ignore_errors=True)
    workspace = Path(tempfile.mkdtemp(prefix="isaiah_goh_golf_coach_"))
    (workspace / "models").mkdir(parents=True, exist_ok=True)
    st.session_state["workspace"] = str(workspace)
    return workspace


def _progress_callback(progress_bar: Any, status_box: Any):
    def callback(fraction: float, message: str) -> None:
        progress_bar.progress(float(fraction))
        status_box.info(message)
    return callback


def _profile_inputs() -> Dict[str, Any]:
    st.subheader("Golfer profile")
    col1, col2, col3 = st.columns(3)
    with col1:
        name = st.text_input("Name", value="Demo Golfer")
        age = st.number_input("Age", min_value=8, max_value=100, value=18, step=1)
        gender = st.selectbox(
            "Gender (optional)",
            ["Prefer not to say", "Female", "Male", "Non-binary", "Self-described"],
        )
    with col2:
        height_cm = st.number_input("Height (cm)", min_value=100.0, max_value=230.0, value=175.0, step=0.5)
        weight_kg = st.number_input("Weight (kg)", min_value=25.0, max_value=250.0, value=70.0, step=0.5)
        handedness = st.selectbox("Golf handedness", ["Right-handed", "Left-handed"])
    with col3:
        experience = st.selectbox("Experience level", ["Beginner", "Intermediate", "Advanced"])
        gym_days = st.slider("Gym/activity days per week", 0, 7, 3)
        primary_goal = st.selectbox(
            "Primary goal",
            ["Improve swing consistency", "Improve balance", "Improve tempo", "Improve rotation", "General practice"],
        )
    health_notes = st.text_area(
        "Health or movement considerations (optional)",
        placeholder="Only share information needed to adapt practice. Do not enter sensitive medical records.",
        height=80,
    )
    bmi = weight_kg / ((height_cm / 100.0) ** 2)
    st.caption(f"Calculated BMI: {bmi:.1f}. BMI is collected only for basic profile context and is not used for diagnosis.")
    return {
        "name": name.strip() or "Golfer",
        "gender": gender,
        "age": int(age),
        "height_cm": float(height_cm),
        "weight_kg": float(weight_kg),
        "bmi": round(float(bmi), 2),
        "golf_handedness": handedness,
        "experience_level": experience,
        "gym_activity_days_per_week": int(gym_days),
        "primary_goal": primary_goal,
        "health_notes": health_notes.strip(),
    }


with st.sidebar:
    st.header("AI Golf Coach")
    st.write(f"**Author:** {AUTHOR}")
    st.write(f"**Mentor:** {MENTOR}")
    st.caption(f"Version {APP_VERSION}")
    st.divider()
    st.markdown(
        "This repository is designed for GitHub and Streamlit Community Cloud. "
        "It includes pose analysis, supervised ML, a multi-layer neural network, "
        "and feedback-based reinforcement learning."
    )
    st.divider()
    st.markdown("**Recording guidance**")
    st.markdown(
        "Use a stationary camera, keep the entire golfer visible, record in good light, "
        "and avoid other people in the frame. Face-on footage works best for this MVP."
    )

st.markdown(
    f"""
<div class="hero">
  <h1>🏌️ AI-Based Golf Coach</h1>
  <p>Video pose analysis, interpretable swing metrics, ML/DNN scoring, personalized practice planning, and feedback learning.</p>
  <p class="credit"><strong>Author:</strong> {AUTHOR} &nbsp; | &nbsp; <strong>Mentor:</strong> {MENTOR}</p>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="notice"><strong>Educational prototype.</strong> This app is not a medical device, injury-risk tool, or replacement for a qualified golf coach. The ML and neural-network scores use synthetic training labels and demonstrate architecture rather than validated coaching accuracy.</div>
""",
    unsafe_allow_html=True,
)

analyze_tab, feedback_tab, about_tab = st.tabs(["Analyze a swing", "Feedback learning", "About the project"])

with analyze_tab:
    profile = _profile_inputs()
    st.divider()
    st.subheader("Video input")
    source_mode = st.radio(
        "Choose a source",
        ["Synthetic demo", "Upload my golf video"],
        horizontal=True,
        help="The demo verifies the full app without uploading personal data.",
    )
    uploaded_file = None
    if source_mode == "Upload my golf video":
        uploaded_file = st.file_uploader(
            "Upload a practice or playing clip",
            type=["mp4", "mov", "m4v", "avi", "wmv", "mkv"],
            help="For cloud reliability, clips are analyzed for a limited duration and sampled every few frames.",
        )
    else:
        st.info("Demo mode uses a generated stick-figure swing. It contains no personal video or health data.")

    with st.expander("Analysis settings", expanded=False):
        max_seconds = st.slider("Maximum analyzed seconds", 5, 30, 20)
        frame_stride = st.slider(
            "Frame sampling stride",
            1,
            6,
            3,
            help="A higher value is faster and uses less cloud memory; a lower value is smoother.",
        )
        include_health_notes = st.checkbox("Include optional health notes in downloaded reports", value=False)

    consent = st.checkbox(
        "I consent to processing this video and profile in the current app session.",
        value=(source_mode == "Synthetic demo"),
    )
    profile["consent_to_process_video"] = bool(consent)

    analyze_clicked = st.button("Analyze swing", type="primary", use_container_width=True)
    if analyze_clicked:
        if not consent:
            st.error("Consent is required before processing a video.")
        elif source_mode == "Upload my golf video" and uploaded_file is None:
            st.error("Upload a video before starting the analysis.")
        else:
            progress_bar = st.progress(0.0)
            status_box = st.empty()
            callback = _progress_callback(progress_bar, status_box)
            try:
                workspace = _new_workspace()
                callback(0.01, "Preparing the analysis workspace")
                if source_mode == "Synthetic demo":
                    source_path, records, metadata = create_demo_video(workspace / "synthetic_demo.mp4")
                    analysis_path = standardize_video(source_path, workspace / "synthetic_demo_h264.mp4")
                    metadata["source_mode"] = "Synthetic demo"
                else:
                    upload_name = _safe_filename(uploaded_file.name)
                    uploaded_path = workspace / upload_name
                    uploaded_path.write_bytes(uploaded_file.getvalue())
                    callback(0.02, "Converting the uploaded video for browser playback")
                    analysis_path = standardize_video(uploaded_path, workspace / "input_standardized.mp4")
                    records, metadata = extract_pose_records(
                        analysis_path,
                        model_dir=workspace / "models",
                        max_seconds=float(max_seconds),
                        frame_stride=int(frame_stride),
                        progress_callback=callback,
                    )
                    metadata["source_mode"] = "Uploaded video"
                    metadata["original_filename"] = upload_name

                callback(0.68, "Calculating swing metrics and phases")
                features = build_feature_dataframe(records, expected_samples=int(metadata["expected_samples"]))
                summary, component_scores = summarize_swing(
                    features, expected_samples=int(metadata["expected_samples"])
                )
                callback(0.70, "Running ML and neural-network architecture demonstrations")
                model_predictions, model_metrics = predict_demo_models(summary)
                advice = build_advice(component_scores, summary)
                practice_plan = build_practice_plan(advice, profile)
                annotated_path = create_annotated_video(
                    analysis_path,
                    workspace / "annotated_golf_swing.mp4",
                    records,
                    frame_feature_map(features),
                    overall_score=float(summary["heuristic_overall_score"]),
                    frame_stride=int(metadata["frame_stride"]),
                    progress_callback=callback,
                )

                state, state_score = weakest_state(component_scores)
                bandit = st.session_state.get("bandit")
                if not isinstance(bandit, ContextualBanditCoach):
                    bandit = ContextualBanditCoach()
                    st.session_state["bandit"] = bandit
                recommendation = bandit.recommend(state, epsilon=0.0, seed=42)
                st.session_state["rl_state"] = state
                st.session_state["rl_state_score"] = state_score
                st.session_state["rl_recommendation"] = recommendation
                st.session_state.setdefault("feedback_log", [])

                st.session_state["analysis_result"] = {
                    "profile": profile,
                    "metadata": metadata,
                    "summary": summary,
                    "component_scores": component_scores,
                    "model_predictions": model_predictions,
                    "model_metrics": model_metrics,
                    "advice": advice,
                    "practice_plan": practice_plan,
                    "features": features,
                    "source_path": str(analysis_path),
                    "annotated_path": str(annotated_path),
                    "include_health_notes": include_health_notes,
                }
                callback(1.0, "Analysis complete")
                status_box.success("Analysis complete. Results and downloads are shown below.")
            except PoseBackendUnavailable as exc:
                progress_bar.empty()
                status_box.empty()
                st.error(str(exc))
                st.info("The app itself is running. Use Synthetic demo to verify the full pipeline while checking the Streamlit dependency build log.")
            except Exception as exc:
                progress_bar.empty()
                status_box.empty()
                st.error(f"Analysis failed: {exc}")
                with st.expander("Technical details"):
                    st.code(traceback.format_exc())

    result = st.session_state.get("analysis_result")
    if result:
        st.divider()
        st.header("Analysis results")
        summary = result["summary"]
        components = result["component_scores"]
        predictions = result["model_predictions"]
        metadata = result["metadata"]
        features = result["features"]

        score_col, rf_col, nn_col, coverage_col = st.columns(4)
        score_col.metric("Transparent heuristic", f"{summary['heuristic_overall_score']:.1f}/100")
        rf_col.metric("Random Forest", f"{predictions['Random Forest prototype']:.1f}/100")
        nn_col.metric("Deep neural network", f"{predictions['Deep neural network prototype']:.1f}/100")
        coverage_col.metric("Pose coverage", f"{summary['detection_coverage'] * 100:.0f}%")
        st.caption("The Random Forest and neural-network values are synthetic-label architecture demonstrations, not validated coaching accuracy.")

        original_col, annotated_col = st.columns(2)
        with original_col:
            st.subheader("Input video")
            st.video(result["source_path"])
        with annotated_col:
            st.subheader("Annotated video")
            st.video(result["annotated_path"])

        chart_col, summary_col = st.columns([1, 1.25])
        with chart_col:
            st.subheader("Component scores")
            component_frame = pd.DataFrame(
                {"Component": list(components.keys()), "Score": list(components.values())}
            ).set_index("Component")
            st.bar_chart(component_frame, y="Score", horizontal=True)
        with summary_col:
            st.subheader("Key measurements")
            display_metrics = pd.DataFrame([
                ["Address torso tilt", f"{summary['address_torso_tilt_deg']:.1f}°"],
                ["Address mean knee angle", f"{summary['address_mean_knee_angle_deg']:.1f}°"],
                ["Maximum shoulder-hip separation", f"{summary['max_shoulder_hip_separation_deg']:.1f}°"],
                ["Maximum hip sway", f"{summary['max_hip_sway_shoulder_widths']:.2f} shoulder widths"],
                ["Maximum head motion", f"{summary['max_head_motion_shoulder_widths']:.2f} shoulder widths"],
                ["Impact mean elbow angle", f"{summary['impact_mean_elbow_angle_deg']:.1f}°"],
                ["Tempo ratio", f"{summary['tempo_ratio']:.2f}:1"],
                ["Peak hand speed", f"{summary['peak_hand_speed_shoulder_widths_per_s']:.2f} shoulder widths/s"],
                ["Pose backend", metadata.get("pose_backend", "Unknown")],
            ], columns=["Metric", "Value"])
            st.dataframe(display_metrics, hide_index=True, use_container_width=True)

        st.subheader("Movement signals")
        signal_tab1, signal_tab2, signal_tab3 = st.tabs(["Joint angles", "Movement", "Phases"])
        with signal_tab1:
            st.line_chart(
                features.set_index("timestamp_s")[[
                    "left_elbow_angle", "right_elbow_angle", "left_knee_angle", "right_knee_angle"
                ]]
            )
        with signal_tab2:
            st.line_chart(
                features.set_index("timestamp_s")[[
                    "shoulder_hip_separation_deg",
                    "hip_sway_shoulder_widths",
                    "head_motion_shoulder_widths",
                    "hand_speed_smoothed",
                ]]
            )
        with signal_tab3:
            st.dataframe(
                features[["timestamp_s", "phase", "torso_tilt_deg", "hand_speed_smoothed", "pose_visibility"]],
                hide_index=True,
                use_container_width=True,
            )

        st.subheader("Priority coaching suggestions")
        st.dataframe(result["advice"], hide_index=True, use_container_width=True)
        st.subheader("Personalized four-week practice plan")
        st.dataframe(result["practice_plan"], hide_index=True, use_container_width=True)

        st.subheader("Download results")
        bandit = st.session_state.get("bandit") or ContextualBanditCoach()
        report_html = build_report_html(
            result["profile"],
            result["metadata"],
            result["summary"],
            result["component_scores"],
            result["model_predictions"],
            result["advice"],
            result["practice_plan"],
            include_health_notes=bool(result["include_health_notes"]),
        )
        package_bytes = build_download_package(
            result["profile"],
            result["metadata"],
            result["summary"],
            result["component_scores"],
            result["model_predictions"],
            result["model_metrics"],
            result["advice"],
            result["practice_plan"],
            result["features"],
            bandit.payload(),
            Path(result["annotated_path"]),
            include_health_notes=bool(result["include_health_notes"]),
        )
        dl1, dl2, dl3, dl4 = st.columns(4)
        dl1.download_button(
            "Download full ZIP",
            package_bytes,
            file_name="Isaiah_Goh_AI_Golf_Coach_Analysis.zip",
            mime="application/zip",
            use_container_width=True,
        )
        dl2.download_button(
            "Download HTML report",
            report_html.encode("utf-8"),
            file_name="golf_coach_report.html",
            mime="text/html",
            use_container_width=True,
        )
        dl3.download_button(
            "Download features CSV",
            result["features"].to_csv(index=False).encode("utf-8"),
            file_name="swing_frame_features.csv",
            mime="text/csv",
            use_container_width=True,
        )
        dl4.download_button(
            "Download summary JSON",
            json.dumps(
                {
                    "author": AUTHOR,
                    "mentor": MENTOR,
                    "summary": result["summary"],
                    "component_scores": result["component_scores"],
                    "model_predictions": result["model_predictions"],
                },
                indent=2,
            ).encode("utf-8"),
            file_name="golf_coach_summary.json",
            mime="application/json",
            use_container_width=True,
        )

with feedback_tab:
    st.header("Feedback-based reinforcement learning")
    st.write(
        "The MVP uses a contextual bandit. The current weak component is the context, a drill is the action, "
        "and the user's 1–5 rating becomes a reward. The update changes future drill selection for that context."
    )
    result = st.session_state.get("analysis_result")
    if not result:
        st.info("Run a swing analysis first so the app can select a weakness and a drill.")
    else:
        bandit = st.session_state.get("bandit")
        if not isinstance(bandit, ContextualBanditCoach):
            bandit = ContextualBanditCoach()
            st.session_state["bandit"] = bandit
        state = st.session_state["rl_state"]
        recommendation = st.session_state["rl_recommendation"]
        state_score = float(st.session_state["rl_state_score"])
        st.markdown(
            f"""
<div class="good"><strong>Current practice context:</strong> {state} ({state_score:.1f}/100)<br><strong>Recommended drill:</strong> {recommendation}</div>
""",
            unsafe_allow_html=True,
        )
        rating = st.slider("After trying the drill, rate its usefulness", 1, 5, 4)
        comment = st.text_input("Optional feedback note")
        if st.button("Submit feedback and update policy", type="primary"):
            reward = bandit.update(state, recommendation, rating)
            log = st.session_state.setdefault("feedback_log", [])
            log.append({
                "state": state,
                "action": recommendation,
                "rating": rating,
                "reward": reward,
                "comment": comment.strip(),
            })
            st.session_state["rl_recommendation"] = bandit.recommend(state, epsilon=0.12)
            st.success(f"Policy updated with reward {reward:+.1f}. The next recommendation is {st.session_state['rl_recommendation']}.")
        st.download_button(
            "Download RL policy JSON",
            bandit.json_bytes(),
            file_name="rl_contextual_bandit_policy.json",
            mime="application/json",
        )
        if st.session_state.get("feedback_log"):
            st.subheader("Session feedback log")
            st.dataframe(pd.DataFrame(st.session_state["feedback_log"]), hide_index=True, use_container_width=True)

with about_tab:
    st.header("Project architecture")
    st.markdown(
        f"""
**Author:** {AUTHOR}  
**Mentor:** {MENTOR}

The repository implements four connected layers:

1. **Computer vision:** MediaPipe estimates 33 body landmarks from sampled video frames and OpenCV creates an annotated video.
2. **Interpretable biomechanics:** Geometry functions calculate joint angles, torso tilt, relative shoulder/hip rotation, sway, head motion, hand speed, tempo, and swing phases.
3. **Machine learning and deep neural network:** A Random Forest and a three-hidden-layer MLP demonstrate predictive architecture using transparent synthetic labels. They are placeholders for future consented, coach-labeled data.
4. **Reinforcement learning:** A contextual bandit learns which practice drill users rate most positively for each detected weakness.

The repository also includes the original Colab notebook, a model card, tests, a GitHub Actions workflow, Streamlit configuration, and deployment instructions.
"""
    )
    st.subheader("Responsible-use boundaries")
    st.markdown(
        "- Do not use the score for medical diagnosis, injury prediction, rehabilitation, selection, or certification.\n"
        "- Do not commit uploaded user videos or sensitive health notes to GitHub.\n"
        "- Replace synthetic model labels with de-identified, consented, coach-labeled data before evaluating real accuracy.\n"
        "- Evaluate camera-view robustness, demographic fairness, ability-level performance, and coach agreement before production use."
    )

st.markdown(
    f"<p class='small' style='text-align:center;margin-top:2rem'>Created by {AUTHOR} under the mentorship of {MENTOR}.</p>",
    unsafe_allow_html=True,
)
