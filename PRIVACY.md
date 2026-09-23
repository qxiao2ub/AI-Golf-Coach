# Privacy Notes

## Browser video handling

The embedded AI Swing Studio uses its own browser file picker. A selected video is represented by a temporary local object URL and is processed inside the browser. The Python Streamlit server does not receive the video bytes through this workflow.

## External network requests

When pose mode is initialized, the browser downloads MediaPipe JavaScript/WebAssembly assets from jsDelivr and a pose model from Google-hosted model storage. The app does not intentionally attach the selected video to those requests.

## Local storage

Thumbs-up/down coaching preferences are stored in the browser's local storage under `golfCoachPreferences`. Users can clear this through browser site-data controls. Streamlit's separate feedback demonstration is held in session state and resets when the session ends.

## Downloads

JSON, CSV, PNG, and WebM files are created in the browser and downloaded directly by the user. The app does not provide persistent cloud storage.

## Sensitive information

Do not enter medical records, government identifiers, financial account information, or other sensitive personal data. Optional health or movement notes should be limited to the minimum information needed for practice planning.
