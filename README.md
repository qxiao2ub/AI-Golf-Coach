# AI-Based Golf Coach — Fast Streamlit Cloud Edition

**Author:** Isaiah Goh  
**Mentor:** Dr. Qingyang Xiao

This repository is a Streamlit-ready AI Golf Coach prototype designed to **start quickly on Streamlit Community Cloud**. The previous server-side build installed MediaPipe, OpenCV, FFmpeg, scikit-learn, and many Linux libraries before the page could appear. This edition moves video pose inference into the visitor's browser and keeps the Streamlit server intentionally lightweight.

## Why this version loads faster

- The root intentionally has **no `requirements.txt`** and **no `packages.txt`**.
- Streamlit Community Cloud supplies Streamlit and its core dependencies automatically.
- The Python entry point imports only Streamlit and Python standard-library modules.
- MediaPipe Pose Landmarker runs in the browser through WebAssembly after the page is visible.
- If the browser pose model cannot load, the app automatically uses a native pixel-motion fallback.
- Uploaded video remains inside the browser component and is not sent to the Python server.

## Main functions

- Browser-local video selection and playback
- MediaPipe 33-landmark pose estimation when available
- Annotated body skeleton and 2D joint-angle visualization
- Address, backswing, top, downswing, impact, and follow-through segmentation
- Posture, rotation, balance, tempo, extension, smoothness, and pose-coverage metrics
- Transparent scoring and a small fixed-weight neural-network architecture demonstration
- Feedback-ranked coaching suggestions using a contextual-bandit update in browser storage
- Four-week practice planning
- JSON, CSV, annotated PNG, and annotated WebM downloads
- Synthetic demonstration that runs without a personal video
- Original Colab notebook included under `notebooks/`

## Deploy on Streamlit Community Cloud

1. Extract this ZIP file.
2. Create a **new GitHub repository**, or fully replace the contents of the old repository.
3. Upload the files so `streamlit_app.py` is at the repository root.
4. Confirm that the repository does **not** contain the old `requirements.txt` or `packages.txt`.
5. In Streamlit Community Cloud, select:
   - Repository: your new GitHub repository
   - Branch: `main`
   - Main file path: `streamlit_app.py`
6. Deploy. The page should appear without a long apt/pip build.

For an existing Streamlit app, push this repository as a complete replacement, then open **Manage app → Reboot app**. If Streamlit continues using the previous Python environment or cached dependency build, delete the app and redeploy it from the new repository.

## Recommended browser and video formats

Use a current version of Chrome, Edge, Firefox, or Safari. MP4/H.264 and WebM are the most reliable formats for browser decoding. Some browsers cannot decode WMV, AVI, or HEVC directly; convert those clips to MP4/H.264 first.

## Local run

A normal local environment only needs Streamlit:

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install streamlit
streamlit run streamlit_app.py
```

## Architecture

```text
Streamlit server
  ├─ page layout and project credits
  ├─ profile form and BMI context
  ├─ rule-based four-week practice plan
  ├─ session contextual-bandit demonstration
  └─ embeds web/pose_studio.html

Browser component
  ├─ local video file picker
  ├─ MediaPipe Tasks Vision + WebAssembly
  ├─ native motion-analysis fallback
  ├─ pose features and swing phase segmentation
  ├─ charts, coaching suggestions, and feedback storage
  └─ report/video/image downloads
```

## Important limitations

This is an educational prototype, not a medical device, injury-risk system, or replacement for a qualified golf instructor. Joint angles are 2D image-plane estimates and can change with camera placement, perspective, occlusion, clothing, lighting, and model confidence. The neural-network score uses fixed demonstration weights rather than professionally labeled golf-coaching data.

## Repository structure

```text
.
├── streamlit_app.py
├── web/
│   └── pose_studio.html
├── notebooks/
│   └── Isaiah_Goh_AI_Golf_Coach_Colab.ipynb
├── .streamlit/
│   └── config.toml
├── tests/
│   └── test_repo.py
├── AUTHORS.md
├── DEPLOYMENT.md
├── MODEL_CARD.md
├── PRIVACY.md
└── LICENSE
```

## License

MIT License for this repository's original code. Third-party frameworks, models, and CDN assets retain their respective licenses.
