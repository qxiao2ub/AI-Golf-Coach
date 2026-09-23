# Streamlit Community Cloud Deployment

## Recommended clean deployment

The safest method is to create a new GitHub repository from this ZIP. This prevents the old heavy dependency files from remaining in Git history or the repository root.

1. Extract the ZIP.
2. Open the extracted folder.
3. Verify these two files are absent:
   - `requirements.txt`
   - `packages.txt`
4. Push all repository contents to the root of a GitHub repository.
5. Open Streamlit Community Cloud and create an app.
6. Select `streamlit_app.py` as the main file.
7. Deploy.

## Updating the existing `ai-golf-coach` repository

Delete the old repository contents before copying this edition. In particular, remove:

- `requirements.txt`
- `packages.txt`
- `.python-version`
- the old `golf_coach/` server-side package
- any server-side MediaPipe/OpenCV model cache

Commit the deletion and the new files together. Streamlit should trigger a clean redeploy. Use **Manage app → Reboot app** after the commit is visible on GitHub.

## When the existing app still shows the old build

Python itself is selected when the Community Cloud app is deployed. A repository file cannot reliably change the Python interpreter of an already-created app. This fast edition supports current Streamlit Cloud Python versions because it has no compiled Python AI dependencies. However, if the old environment remains stuck:

1. Record the current app URL and secrets.
2. Delete the Streamlit app from the workspace.
3. Create it again from the new repository.
4. Use `streamlit_app.py` as the entry point.
5. Reuse the prior custom subdomain if desired.

## Expected startup behavior

- There should be no long Debian/apt installation stage.
- There should be no MediaPipe or OpenCV wheel installation stage.
- The Streamlit page should appear first.
- The pose model downloads only after a visitor clicks **Initialize pose AI** or **Analyze selected video**.
- If CDN/model access fails, the page remains functional through motion fallback and synthetic demo mode.

## Troubleshooting

### Page loads, but pose initialization fails

This usually means the browser or network blocked the MediaPipe CDN/model. The app automatically switches to motion fallback. Try Chrome or Edge, disable strict content blocking for the app domain, and retry.

### Video cannot be decoded

Convert the file to MP4 using H.264 video. Browser support for WMV, AVI, and HEVC varies by operating system and browser.

### The embedded studio is too short

Increase the `height` argument in `components.html(...)` inside `streamlit_app.py`.

### GitHub still contains old dependency files

Delete them in GitHub and commit the deletion. Simply uploading new files does not remove old files.
