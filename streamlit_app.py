"""Fast Streamlit Cloud edition of Isaiah Goh's AI Golf Coach.

The server imports only Streamlit and Python's standard library. Video pose
inference runs inside the visitor's browser through an embedded MediaPipe Web
component, so Streamlit Community Cloud does not need to install MediaPipe,
OpenCV, FFmpeg, PyTorch, or scikit-learn before the app can start.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import streamlit as st
import streamlit.components.v1 as components

ROOT = Path(__file__).resolve().parent
AUTHOR = "Isaiah Goh"
MENTOR = "Dr. Qingyang Xiao"
APP_VERSION = "2.0.0-fast-cloud"

st.set_page_config(
    page_title="AI Golf Coach | Isaiah Goh",
    page_icon="🏌️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
:root {
  --golf-green: #0f5d42;
  --golf-light: #edf8f2;
  --golf-gold: #c7942f;
  --ink: #16322a;
}
.block-container {padding-top: 1.15rem; padding-bottom: 3rem; max-width: 1500px;}
.hero {
  padding: 1.35rem 1.55rem;
  border-radius: 20px;
  background: linear-gradient(125deg, #e8f7ef 0%, #fbfdfb 62%, #fff7e5 100%);
  border: 1px solid #c9e6d6;
  box-shadow: 0 12px 30px rgba(15, 93, 66, .08);
  margin-bottom: 1rem;
}
.hero h1 {margin: 0; color: var(--ink); font-size: clamp(2rem, 4vw, 3.2rem);}
.hero p {margin: .42rem 0 0; color: #46645b; font-size: 1.03rem;}
.credit {font-size: .96rem !important; color: #52675f !important;}
.fast-badge {
  display: inline-block; padding: .25rem .65rem; border-radius: 999px;
  background: #0f5d42; color: white; font-size: .78rem; font-weight: 700;
  letter-spacing: .03em; margin-bottom: .55rem;
}
.notice {
  padding: .85rem 1rem; border-left: 5px solid #b88400;
  background: #fff8df; border-radius: 8px; margin: .6rem 0 1rem;
}
.success-note {
  padding: .85rem 1rem; border-left: 5px solid #17824f;
  background: #edf9f1; border-radius: 8px; margin: .6rem 0 1rem;
}
.metric-panel {
  padding: 1rem; background: #f8fbf9; border: 1px solid #dfece5;
  border-radius: 14px; min-height: 120px;
}
.small-note {font-size: .88rem; color: #5b6964;}
footer {visibility: hidden;}
</style>
""",
    unsafe_allow_html=True,
)


def _build_practice_plan(profile: dict[str, Any]) -> list[dict[str, str]]:
    """Create a transparent rule-based four-week practice plan."""
    goal = profile["primary_goal"]
    experience = profile["experience"]
    days = int(profile["activity_days"])
    session_minutes = 25 if days <= 2 else 35 if days <= 4 else 45

    focus_map = {
        "Improve swing consistency": (
            "repeatable setup and contact",
            "Use alignment sticks, pause at address, and record three sets of five swings.",
        ),
        "Improve balance": (
            "centered pressure and stable finish",
            "Use feet-together swings, hold the finish for three seconds, and reduce speed before adding power.",
        ),
        "Improve tempo": (
            "backswing-to-downswing rhythm",
            "Practice with a 3:1 count, alternate slow-motion and normal swings, and compare rhythm across sets.",
        ),
        "Improve rotation": (
            "shoulder and hip sequencing",
            "Use club-across-chest turns, step-through drills, and controlled half swings before full swings.",
        ),
        "Improve posture": (
            "athletic setup and spine control",
            "Use mirror checkpoints, hip-hinge drills, and short swings while maintaining head and torso stability.",
        ),
        "General practice": (
            "balanced fundamentals",
            "Rotate through setup, tempo, balance, contact, and reflection drills.",
        ),
    }
    focus, drill = focus_map.get(goal, focus_map["General practice"])
    intensity = {
        "Beginner": "Keep effort near 60–70% and prioritize clean movement patterns.",
        "Intermediate": "Use 70–85% effort and compare repeatability across multiple sets.",
        "Advanced": "Alternate technical blocks with competition-like pressure sets.",
    }[experience]

    return [
        {
            "week": "Week 1 — Baseline",
            "focus": focus,
            "plan": f"{session_minutes} minutes per session. {drill} Record one face-on and one down-the-line clip.",
        },
        {
            "week": "Week 2 — Controlled Repetition",
            "focus": "low-speed pattern building",
            "plan": f"Complete 3–4 sets of five swings. {intensity} Keep notes on the most repeatable cue.",
        },
        {
            "week": "Week 3 — Transfer",
            "focus": "variable clubs and targets",
            "plan": "Alternate two clubs or targets while preserving the same pre-shot routine. Recheck video after each block.",
        },
        {
            "week": "Week 4 — Retest",
            "focus": "comparison with the original baseline",
            "plan": "Repeat the Week 1 camera setup, compare the same metrics, and retain only cues that improved both score and comfort.",
        },
    ]


def _initialize_feedback_state() -> None:
    if "bandit_values" not in st.session_state:
        st.session_state.bandit_values = {
            "Tempo drill": 0.50,
            "Balance drill": 0.50,
            "Rotation drill": 0.50,
            "Posture drill": 0.50,
            "Video checkpoint": 0.50,
        }
    if "bandit_counts" not in st.session_state:
        st.session_state.bandit_counts = {key: 0 for key in st.session_state.bandit_values}


def _update_feedback(action: str, reward: float) -> None:
    _initialize_feedback_state()
    count = st.session_state.bandit_counts[action] + 1
    old_value = st.session_state.bandit_values[action]
    new_value = old_value + (reward - old_value) / count
    st.session_state.bandit_counts[action] = count
    st.session_state.bandit_values[action] = max(0.0, min(1.0, new_value))


with st.sidebar:
    st.header("🏌️ AI Golf Coach")
    st.write(f"**Author:** {AUTHOR}")
    st.write(f"**Mentor:** {MENTOR}")
    st.caption(f"Version {APP_VERSION}")
    st.divider()
    st.markdown("**Why this edition launches faster**")
    st.caption(
        "The Streamlit server has no heavy computer-vision dependencies. "
        "Pose inference runs in the visitor's browser, with an automatic motion-analysis fallback."
    )
    st.divider()
    st.markdown("**Recording guidance**")
    st.caption(
        "Use a stationary camera, keep the entire golfer visible, record in good light, "
        "and avoid other people in the frame. Face-on or down-the-line footage works best."
    )

st.markdown(
    f"""
<div class="hero">
  <span class="fast-badge">FAST STREAMLIT CLOUD EDITION</span>
  <h1>🏌️ AI-Based Golf Coach</h1>
  <p>Browser-based pose landmarks, joint-angle visualization, swing-phase analysis, interpretable scoring, practice planning, and feedback learning.</p>
  <p class="credit"><strong>Author:</strong> {AUTHOR} &nbsp; | &nbsp; <strong>Mentor:</strong> {MENTOR}</p>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="notice"><strong>Educational prototype.</strong> The scores and recommendations are for learning and demonstration. They are not medical advice, injury diagnosis, or a replacement for a qualified golf professional. The neural-network and reinforcement-learning sections demonstrate an AI architecture and are not clinically or professionally validated.</div>
""",
    unsafe_allow_html=True,
)

studio_tab, profile_tab, feedback_tab, about_tab = st.tabs(
    ["AI Swing Studio", "Profile & Practice Plan", "Feedback Learning", "About & Deployment"]
)

with studio_tab:
    st.markdown(
        """
<div class="success-note"><strong>Privacy-first video processing:</strong> the video selected inside the studio remains in the browser. The Streamlit server does not receive or store the video. The first pose-analysis run downloads a browser AI model; if that model cannot load, the studio automatically switches to a native motion-analysis fallback.</div>
""",
        unsafe_allow_html=True,
    )
    html_path = ROOT / "web" / "pose_studio.html"
    if not html_path.exists():
        st.error("The browser studio file is missing from the repository: web/pose_studio.html")
    else:
        studio_html = html_path.read_text(encoding="utf-8")
        studio_html = studio_html.replace("{{AUTHOR}}", AUTHOR).replace("{{MENTOR}}", MENTOR)
        components.html(studio_html, height=1880, scrolling=True)

with profile_tab:
    st.subheader("Golfer profile and four-week practice plan")
    st.caption("Profile information is kept only in the current Streamlit session unless you download it.")

    with st.form("profile_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            golfer_name = st.text_input("Name", value="Demo Golfer")
            age = st.number_input("Age", min_value=8, max_value=100, value=18, step=1)
            gender = st.selectbox(
                "Gender (optional)",
                ["Prefer not to say", "Female", "Male", "Non-binary", "Self-described"],
            )
        with col2:
            height_cm = st.number_input("Height (cm)", 100.0, 230.0, 175.0, 0.5)
            weight_kg = st.number_input("Weight (kg)", 25.0, 250.0, 70.0, 0.5)
            handedness = st.selectbox("Golf handedness", ["Right-handed", "Left-handed"])
        with col3:
            experience = st.selectbox("Experience level", ["Beginner", "Intermediate", "Advanced"])
            activity_days = st.slider("Activity days per week", 0, 7, 3)
            primary_goal = st.selectbox(
                "Primary goal",
                [
                    "Improve swing consistency",
                    "Improve balance",
                    "Improve tempo",
                    "Improve rotation",
                    "Improve posture",
                    "General practice",
                ],
            )
        health_notes = st.text_area(
            "Movement considerations (optional)",
            placeholder="Only enter information needed to adapt practice. Do not enter private medical records.",
        )
        submitted = st.form_submit_button("Generate practice plan", type="primary", use_container_width=True)

    if submitted:
        bmi = float(weight_kg) / math.pow(float(height_cm) / 100.0, 2)
        profile = {
            "name": golfer_name.strip() or "Golfer",
            "age": int(age),
            "gender": gender,
            "height_cm": float(height_cm),
            "weight_kg": float(weight_kg),
            "bmi": round(bmi, 2),
            "handedness": handedness,
            "experience": experience,
            "activity_days": int(activity_days),
            "primary_goal": primary_goal,
            "movement_considerations": health_notes.strip(),
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "author": AUTHOR,
            "mentor": MENTOR,
        }
        st.session_state.profile_report = {
            "profile": profile,
            "practice_plan": _build_practice_plan(profile),
            "disclaimer": "Educational planning only; not medical or professional golf advice.",
        }

    report = st.session_state.get("profile_report")
    if report:
        profile = report["profile"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("BMI", f"{profile['bmi']:.1f}")
        c2.metric("Experience", profile["experience"])
        c3.metric("Weekly activity", f"{profile['activity_days']} days")
        c4.metric("Primary goal", profile["primary_goal"])
        st.caption("BMI is displayed only as general profile context and is not used for diagnosis.")

        for item in report["practice_plan"]:
            with st.expander(item["week"], expanded=True):
                st.markdown(f"**Focus:** {item['focus']}")
                st.write(item["plan"])

        report_bytes = json.dumps(report, indent=2, ensure_ascii=False).encode("utf-8")
        st.download_button(
            "Download profile and practice plan (JSON)",
            data=report_bytes,
            file_name="isaiah_goh_ai_golf_coach_practice_plan.json",
            mime="application/json",
            use_container_width=True,
        )

with feedback_tab:
    st.subheader("Feedback-driven recommendation learning")
    st.write(
        "This lightweight contextual-bandit demonstration updates the estimated usefulness of each drill "
        "from thumbs-up and thumbs-down feedback during the current app session."
    )
    _initialize_feedback_state()
    actions = sorted(
        st.session_state.bandit_values,
        key=lambda item: st.session_state.bandit_values[item],
        reverse=True,
    )
    for action in actions:
        col_text, col_up, col_down = st.columns([5, 1, 1])
        with col_text:
            value = st.session_state.bandit_values[action]
            count = st.session_state.bandit_counts[action]
            st.markdown(f"**{action}** — learned preference score: `{value:.2f}` from `{count}` rating(s)")
            st.progress(value)
        with col_up:
            if st.button("👍", key=f"up_{action}", help=f"Helpful: {action}"):
                _update_feedback(action, 1.0)
                st.rerun()
        with col_down:
            if st.button("👎", key=f"down_{action}", help=f"Not helpful: {action}"):
                _update_feedback(action, 0.0)
                st.rerun()

    st.divider()
    st.caption(
        "This is an interpretable learning demonstration, not autonomous retraining of a production model. "
        "Session values reset when the app session ends."
    )

with about_tab:
    st.subheader("Architecture and deployment")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Fast cloud architecture")
        st.markdown(
            """
- **Streamlit server:** UI, profile forms, practice-plan generation, downloads, and feedback demonstration.
- **Browser AI:** MediaPipe Pose Landmarker processes selected video frames locally in the visitor's browser.
- **Fallback engine:** native browser pixel-motion analysis runs when the external pose model cannot load.
- **No server video upload:** the embedded studio uses a browser file picker instead of a Python file uploader.
- **No heavy build:** the repository intentionally omits `mediapipe`, OpenCV, FFmpeg, PyTorch, and scikit-learn from the cloud environment.
"""
        )
    with col2:
        st.markdown("#### AI components represented")
        st.markdown(
            """
- 33-landmark body-pose extraction when the browser model is available
- Joint angles, head movement, hip sway, tempo, rotation, and movement smoothness
- Transparent rule-based scoring plus a small fixed-weight neural-network demonstration
- Swing-phase segmentation: address, backswing, top, downswing, impact, follow-through
- Feedback-ranked coaching suggestions using an incremental contextual-bandit update
- Four-week practice-plan generator and downloadable JSON/CSV reports
"""
        )

    st.markdown(
        """
<div class="notice"><strong>Browser compatibility:</strong> MP4/H.264 and WebM are the most reliable upload formats. Some browsers cannot decode WMV, AVI, or HEVC directly. For those files, convert to MP4/H.264 before using the fast cloud studio.</div>
""",
        unsafe_allow_html=True,
    )

    notebook_path = ROOT / "notebooks" / "Isaiah_Goh_AI_Golf_Coach_Colab.ipynb"
    if notebook_path.exists():
        st.download_button(
            "Download the original Colab notebook",
            data=notebook_path.read_bytes(),
            file_name=notebook_path.name,
            mime="application/x-ipynb+json",
            use_container_width=True,
        )

    st.markdown("#### Credits")
    st.write(f"**Author:** {AUTHOR}")
    st.write(f"**Mentor:** {MENTOR}")
    st.caption("MIT-licensed repository. Third-party browser AI libraries retain their own licenses.")
