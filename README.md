# AI-Based Golf Coach

**Author:** Isaiah Goh  
**Mentor:** Dr. Qingyang Xiao

A GitHub-ready and Streamlit Community Cloud-ready educational prototype that analyzes a golf swing video, estimates body pose, calculates interpretable movement indicators, produces an annotated video, demonstrates machine learning and a deep neural network, recommends practice drills, and learns from user feedback.

> **Important:** This is not a medical device, injury-risk system, or certified golf-coaching product. Its Random Forest and neural-network predictions use synthetic training labels and demonstrate software architecture rather than validated coaching accuracy.

## Live application capabilities

- Profile inputs: name, age, gender, height, weight, BMI, handedness, experience, activity frequency, goals, and optional movement considerations
- Consent control and optional exclusion of health notes from reports
- Synthetic demo that runs without personal data
- Uploaded MP4, MOV, M4V, AVI, WMV, and MKV video support
- FFmpeg conversion to a browser-friendly MP4 when available
- MediaPipe body-pose estimation and 33-landmark annotation
- Address, backswing, top, downswing, impact, and follow-through phases
- Joint angles, torso tilt, shoulder/hip separation, sway, head motion, hand speed, tempo, visibility, and coverage
- Transparent heuristic component scores
- Random Forest supervised-learning demonstration
- Three-hidden-layer MLP deep-neural-network demonstration
- Contextual-bandit reinforcement learning from 1–5 drill feedback
- Priority coaching suggestions and a personalized four-week plan
- Downloadable annotated video, CSV, JSON, HTML report, model card, RL policy, and full ZIP package

## Repository structure

```text
.
├── streamlit_app.py               # Streamlit Community Cloud entry point
├── requirements.txt               # Python dependencies
├── packages.txt                   # Debian/apt packages for video support
├── .python-version                # Python 3.12 target
├── .streamlit/config.toml         # Streamlit configuration
├── golf_coach/
│   ├── analysis.py                # Feature engineering, phases, scoring
│   ├── coaching.py                # Advice, plan, contextual bandit
│   ├── constants.py               # Pose landmark indices
│   ├── geometry.py                # Testable geometry helpers
│   ├── models.py                  # Random Forest and deep MLP prototypes
│   ├── reports.py                 # HTML and ZIP exports
│   └── video.py                   # Upload conversion, pose extraction, annotation
├── notebooks/
│   └── Isaiah_Goh_AI_Golf_Coach_Colab.ipynb
├── tests/
├── MODEL_CARD.md
├── DEPLOYMENT.md
├── SECURITY.md
├── AUTHORS.md
└── LICENSE
```

## Run locally

Python 3.12 is recommended so local behavior matches the intended Streamlit Cloud deployment.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run streamlit_app.py
```

FFmpeg should be installed on the operating system for the most compatible uploaded and annotated video playback.

## Deploy to Streamlit Community Cloud

1. Create a GitHub repository and upload this folder's contents.
2. In Streamlit Community Cloud, create an app from the repository.
3. Set the entry point to `streamlit_app.py`.
4. In Advanced settings, select **Python 3.12**.
5. Deploy.
6. Run **Synthetic demo** first, then test a short face-on golf video.

See [DEPLOYMENT.md](DEPLOYMENT.md) for troubleshooting.

## How the AI pipeline works

### 1. Pose estimation

The video module samples frames and uses MediaPipe Pose. It includes a compatibility wrapper for the legacy `mp.solutions.pose` API and the newer Tasks Pose Landmarker API. Detected landmarks are normalized by shoulder width where possible to reduce sensitivity to camera distance.

### 2. Interpretable swing analysis

The app derives geometry and timing features from sampled pose landmarks. A heuristic phase detector finds the top from hand height and impact from post-top hand speed. Transparent reference bands turn those indicators into component scores.

### 3. Supervised machine learning

A Random Forest demonstrates how summary features could map to a coach score. Because no licensed coach-labeled dataset is bundled, the default labels are produced from a visible rule plus controlled noise.

### 4. Deep neural network

A multi-layer perceptron with hidden layers `(64, 32, 16)` demonstrates a deep-neural-network alternative. It uses standardized summary features and the same synthetic-label dataset.

### 5. Reinforcement learning

A contextual bandit treats the weakest component as context and practice drills as actions. A 1–5 user rating becomes a reward from -1 to +1, updating the selected drill's estimated value.

## Validation status

The repository includes automated tests for geometry, the complete synthetic-demo pipeline, model predictions, practice-plan generation, and ZIP/report exports. These are software tests, not evidence of golf-coaching validity.

Production evaluation should include:

- Multiple qualified coaches and agreement measurement
- Subject-level train/validation/test separation
- Diverse ages, body types, ability levels, handedness, clubs, shots, clothing, lighting, and camera views
- Repeatability and calibration studies
- Fairness and subgroup analysis
- Prospective testing against ball-flight and coaching outcomes

## Privacy

Uploaded videos are processed in the active application session and are not intentionally committed to the repository. Do not use this prototype with regulated, controlled, or sensitive data. See [SECURITY.md](SECURITY.md).

## License

MIT License. See [LICENSE](LICENSE).

## Credits

Created by **Isaiah Goh** under the mentorship of **Dr. Qingyang Xiao**.
