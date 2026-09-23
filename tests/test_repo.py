"""Dependency-free repository smoke tests."""

from __future__ import annotations

import ast
import py_compile
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    required = [
        ROOT / "streamlit_app.py",
        ROOT / "web" / "pose_studio.html",
        ROOT / ".streamlit" / "config.toml",
        ROOT / "README.md",
        ROOT / "AUTHORS.md",
        ROOT / "LICENSE",
    ]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.exists()]
    assert not missing, f"Missing required files: {missing}"

    # The fast cloud edition must not trigger heavy package or apt installation.
    assert not (ROOT / "requirements.txt").exists(), "requirements.txt should be absent in the fast edition"
    assert not (ROOT / "packages.txt").exists(), "packages.txt should be absent in the fast edition"

    py_compile.compile(str(ROOT / "streamlit_app.py"), doraise=True)

    app_text = (ROOT / "streamlit_app.py").read_text(encoding="utf-8")
    html_text = (ROOT / "web" / "pose_studio.html").read_text(encoding="utf-8")
    combined = app_text + "\n" + html_text
    assert "Isaiah Goh" in combined
    assert "Dr. Qingyang Xiao" in combined

    tree = ast.parse(app_text)
    imported_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".")[0])
    allowed_roots = {"__future__", "json", "math", "datetime", "pathlib", "typing", "streamlit"}
    unexpected = imported_roots - allowed_roots
    assert not unexpected, f"Unexpected server imports in fast edition: {sorted(unexpected)}"

    html_ids = set(re.findall(r'id="([^"]+)"', html_text))
    required_ids = {
        "videoFile",
        "analyzeButton",
        "demoButton",
        "sourceVideo",
        "overlayCanvas",
        "results",
        "overallScore",
        "metricChart",
    }
    assert required_ids.issubset(html_ids), f"Missing HTML IDs: {sorted(required_ids - html_ids)}"

    script_blocks = re.findall(r"<script>(.*?)</script>", html_text, flags=re.DOTALL)
    assert script_blocks, "No inline browser application script was found"
    script = script_blocks[-1]
    for marker in [
        "initializePoseEngine",
        "analyzeVideo",
        "generateSyntheticRecords",
        "recordAnnotatedVideo",
        "Native browser motion fallback",
    ]:
        assert marker in script, f"Missing browser function/marker: {marker}"

    print("Repository smoke test passed.")


if __name__ == "__main__":
    main()
