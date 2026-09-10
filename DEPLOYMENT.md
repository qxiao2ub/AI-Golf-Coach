# Deploy on Streamlit Community Cloud

## 1. Create the GitHub repository

1. Extract the ZIP archive.
2. Create a new GitHub repository, for example `isaiah-goh-ai-golf-coach`.
3. Upload **the contents of the extracted folder** so `streamlit_app.py` and `requirements.txt` are at the repository root.
4. Commit and push to the `main` branch.

## 2. Deploy

1. Sign in to Streamlit Community Cloud.
2. Select **Create app**.
3. Choose the GitHub repository and `main` branch.
4. Set the entry point to `streamlit_app.py`.
5. Open **Advanced settings** and select **Python 3.12**.
6. Deploy.

The repository provides:

- `requirements.txt` for Python dependencies
- `packages.txt` for FFmpeg and Linux runtime libraries
- `.streamlit/config.toml` for upload and theme settings
- `.python-version` documenting the tested deployment target

## 3. First launch checks

Run **Synthetic demo** first. It verifies the complete scoring, reporting, ML/DNN, coaching, RL, and download pipeline without requiring MediaPipe inference on a personal video.

Then upload a short, face-on golf swing. Keep the full body visible and use a stationary camera. The app analyzes up to the configured duration and samples frames to stay within shared cloud resources.

## Troubleshooting

### MediaPipe build or import failure

- Confirm that Streamlit is using Python 3.12.
- Confirm `mediapipe==0.10.35` appears in the build log.
- Reboot the app after dependency changes.
- The synthetic demo remains available even when uploaded-video pose inference is unavailable.

### Video cannot be displayed

Check that `packages.txt` was detected and FFmpeg was installed. The app attempts H.264 conversion and falls back to the OpenCV-generated MP4 when necessary.

### App exceeds resource limits

Use a shorter clip, raise the frame stride, lower the maximum analyzed seconds, or use the synthetic demo.
