"""Video I/O, synthetic demo creation, pose extraction, and annotation."""

from __future__ import annotations

import math
import shutil
import subprocess
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .constants import (
    LEFT_ANKLE,
    LEFT_ELBOW,
    LEFT_FOOT,
    LEFT_HEEL,
    LEFT_HIP,
    LEFT_KNEE,
    LEFT_SHOULDER,
    LEFT_WRIST,
    NOSE,
    POSE_CONNECTIONS,
    RIGHT_ANKLE,
    RIGHT_ELBOW,
    RIGHT_FOOT,
    RIGHT_HEEL,
    RIGHT_HIP,
    RIGHT_KNEE,
    RIGHT_SHOULDER,
    RIGHT_WRIST,
)
from .geometry import clamp, lerp

ProgressCallback = Optional[Callable[[float, str], None]]

POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)


class PoseBackendUnavailable(RuntimeError):
    """Raised when MediaPipe cannot be initialized for uploaded-video analysis."""


def _notify(callback: ProgressCallback, fraction: float, message: str) -> None:
    if callback is not None:
        callback(clamp(fraction, 0.0, 1.0), message)


def run_ffmpeg(command: Sequence[str]) -> Tuple[bool, str]:
    if shutil.which("ffmpeg") is None:
        return False, "ffmpeg is not installed"
    result = subprocess.run(
        list(command),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    return result.returncode == 0, result.stderr[-1600:]


def standardize_video(input_path: Path, output_path: Path, max_width: int = 960) -> Path:
    """Convert common uploads to browser-friendly H.264 MP4 when FFmpeg is available."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg", "-y", "-i", str(input_path),
        "-vf", f"scale='min({int(max_width)},iw)':-2",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "24",
        "-pix_fmt", "yuv420p", "-c:a", "aac", "-movflags", "+faststart",
        str(output_path),
    ]
    success, _ = run_ffmpeg(command)
    if success and output_path.exists() and output_path.stat().st_size > 0:
        return output_path
    return input_path


def video_metadata(path: Path, max_seconds: float = 20.0, frame_stride: int = 3) -> Dict[str, Any]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {path.name}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    capture.release()
    maximum_frames = int(max_seconds * fps)
    processed_source_frames = min(frame_count, maximum_frames) if frame_count else maximum_frames
    expected_samples = max(1, math.ceil(processed_source_frames / max(frame_stride, 1)))
    return {
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration_s": frame_count / fps if frame_count else 0.0,
        "max_analyzed_duration_s": min(frame_count / fps, max_seconds) if frame_count else max_seconds,
        "frame_stride": int(frame_stride),
        "expected_samples": expected_samples,
    }


def _landmark_array(landmarks: Sequence[Any]) -> np.ndarray:
    array = np.zeros((33, 5), dtype=np.float32)
    for index, landmark in enumerate(landmarks[:33]):
        array[index] = [
            float(landmark.x),
            float(landmark.y),
            float(getattr(landmark, "z", 0.0)),
            float(getattr(landmark, "visibility", 1.0)),
            float(getattr(landmark, "presence", 1.0)),
        ]
    return array


class MediaPipePoseBackend:
    """Compatibility wrapper for legacy MediaPipe Pose and MediaPipe Tasks."""

    def __init__(self, model_dir: Path):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.backend_name = ""
        self.detector: Any = None
        self._mp: Any = None

    def __enter__(self) -> "MediaPipePoseBackend":
        try:
            import mediapipe as mp  # type: ignore
        except Exception as exc:
            raise PoseBackendUnavailable(
                "MediaPipe could not be imported. Check requirements.txt and the Streamlit build log."
            ) from exc
        self._mp = mp

        # MediaPipe 0.10.x legacy API has a bundled pose model and is the most deployment-friendly.
        solutions = getattr(mp, "solutions", None)
        if solutions is not None and hasattr(solutions, "pose"):
            self.detector = solutions.pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                smooth_landmarks=True,
                enable_segmentation=False,
                min_detection_confidence=0.50,
                min_tracking_confidence=0.50,
            )
            self.backend_name = "MediaPipe Pose"
            return self

        # Newer MediaPipe distributions expose the Tasks API and require a model asset.
        try:
            model_path = self._ensure_task_model()
            vision = mp.tasks.vision
            base_options = mp.tasks.BaseOptions(model_asset_path=str(model_path))
            options = vision.PoseLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.VIDEO,
                num_poses=1,
                min_pose_detection_confidence=0.50,
                min_pose_presence_confidence=0.50,
                min_tracking_confidence=0.50,
                output_segmentation_masks=False,
            )
            self.detector = vision.PoseLandmarker.create_from_options(options)
            self.backend_name = "MediaPipe Pose Landmarker"
            return self
        except Exception as exc:
            raise PoseBackendUnavailable(
                "MediaPipe is installed, but neither the legacy Pose API nor the Tasks Pose Landmarker "
                "could be initialized. Demo mode remains available."
            ) from exc

    def _ensure_task_model(self) -> Path:
        model_path = self.model_dir / "pose_landmarker_lite.task"
        if model_path.exists() and model_path.stat().st_size > 100_000:
            return model_path
        try:
            urllib.request.urlretrieve(POSE_MODEL_URL, model_path)
        except Exception as exc:
            raise PoseBackendUnavailable(
                "The Pose Landmarker model could not be downloaded. Use MediaPipe 0.10.35 or allow "
                "outbound access to storage.googleapis.com."
            ) from exc
        return model_path

    def process(self, bgr_frame: np.ndarray, timestamp_ms: int) -> Optional[np.ndarray]:
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        if self.backend_name == "MediaPipe Pose":
            result = self.detector.process(rgb)
            if result.pose_landmarks is None:
                return None
            return _landmark_array(result.pose_landmarks.landmark)

        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self.detector.detect_for_video(mp_image, int(timestamp_ms))
        if not result.pose_landmarks:
            return None
        return _landmark_array(result.pose_landmarks[0])

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self.detector is not None and hasattr(self.detector, "close"):
            self.detector.close()


def extract_pose_records(
    input_path: Path,
    model_dir: Path,
    max_seconds: float = 20.0,
    frame_stride: int = 3,
    progress_callback: ProgressCallback = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Extract up to 33 pose landmarks from sampled frames of an uploaded video."""
    metadata = video_metadata(input_path, max_seconds=max_seconds, frame_stride=frame_stride)
    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {input_path.name}")

    fps = float(metadata["fps"])
    maximum_frames = int(max_seconds * fps)
    source_total = int(metadata["frame_count"] or maximum_frames)
    source_limit = min(source_total, maximum_frames)
    records: List[Dict[str, Any]] = []

    _notify(progress_callback, 0.03, "Initializing pose model")
    with MediaPipePoseBackend(model_dir) as backend:
        metadata["pose_backend"] = backend.backend_name
        frame_index = 0
        sampled_index = 0
        while frame_index < source_limit:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % max(frame_stride, 1) == 0:
                timestamp_ms = int(round(frame_index / fps * 1000.0))
                landmarks = backend.process(frame, timestamp_ms)
                records.append({
                    "frame_index": frame_index,
                    "sample_index": sampled_index,
                    "timestamp_s": frame_index / fps,
                    "landmarks": landmarks,
                })
                sampled_index += 1
                fraction = 0.05 + 0.62 * (frame_index + 1) / max(source_limit, 1)
                _notify(progress_callback, fraction, f"Tracking body pose: frame {frame_index + 1}/{source_limit}")
            frame_index += 1
    capture.release()

    metadata["sampled_frames"] = len(records)
    metadata["detected_frames"] = sum(record["landmarks"] is not None for record in records)
    metadata["detection_coverage"] = (
        metadata["detected_frames"] / max(metadata["expected_samples"], 1)
    )
    if metadata["detected_frames"] < 4:
        raise RuntimeError(
            "A stable full-body pose was not detected in enough frames. Try a brighter video, keep the "
            "golfer's entire body visible, and use a stationary camera."
        )
    return records, metadata


def draw_pose(
    frame: np.ndarray,
    landmarks: Optional[np.ndarray],
    phase: str = "",
    overall_score: Optional[float] = None,
    extra_text: Optional[Sequence[str]] = None,
) -> np.ndarray:
    canvas = frame.copy()
    height, width = canvas.shape[:2]
    if landmarks is not None:
        for start, end in POSE_CONNECTIONS:
            if landmarks[start, 3] < 0.25 or landmarks[end, 3] < 0.25:
                continue
            p1 = (int(landmarks[start, 0] * width), int(landmarks[start, 1] * height))
            p2 = (int(landmarks[end, 0] * width), int(landmarks[end, 1] * height))
            cv2.line(canvas, p1, p2, (70, 220, 100), 2, cv2.LINE_AA)
        for index, landmark in enumerate(landmarks):
            if landmark[3] < 0.25:
                continue
            center = (int(landmark[0] * width), int(landmark[1] * height))
            radius = 5 if index in {LEFT_WRIST, RIGHT_WRIST, LEFT_HIP, RIGHT_HIP} else 3
            cv2.circle(canvas, center, radius, (30, 90, 240), -1, cv2.LINE_AA)

    lines: List[str] = []
    if phase:
        lines.append(f"Phase: {phase}")
    if overall_score is not None:
        lines.append(f"Prototype score: {overall_score:.1f}/100")
    if extra_text:
        lines.extend(str(item) for item in extra_text)
    if lines:
        box_height = 24 + 25 * len(lines)
        cv2.rectangle(canvas, (8, 8), (min(width - 8, 500), box_height), (0, 0, 0), -1)
        for line_index, text in enumerate(lines):
            cv2.putText(
                canvas,
                text,
                (18, 34 + line_index * 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
    return canvas


def create_annotated_video(
    input_path: Path,
    output_path: Path,
    records: List[Dict[str, Any]],
    feature_rows: Dict[int, Dict[str, Any]],
    overall_score: float,
    frame_stride: int = 3,
    progress_callback: ProgressCallback = None,
) -> Path:
    """Create a sampled, annotated MP4 and convert it to H.264 when possible."""
    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not reopen video for annotation: {input_path.name}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
    temp_path = output_path.with_name(output_path.stem + "_temp.mp4")
    writer = cv2.VideoWriter(
        str(temp_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        max(fps / max(frame_stride, 1), 1.0),
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError("OpenCV could not create the annotated video.")

    record_map = {int(record["frame_index"]): record for record in records}
    sampled_indices = sorted(record_map)
    maximum_index = sampled_indices[-1] if sampled_indices else 0
    frame_index = 0
    written = 0
    while frame_index <= maximum_index:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_index in record_map:
            record = record_map[frame_index]
            metrics = feature_rows.get(frame_index, {})
            extra = []
            if "left_elbow_angle" in metrics and "right_elbow_angle" in metrics:
                extra.append(
                    f"Elbows: {metrics['left_elbow_angle']:.0f} / {metrics['right_elbow_angle']:.0f} deg"
                )
            if "hand_speed_shoulder_widths_per_s" in metrics:
                extra.append(
                    f"Hand speed: {metrics['hand_speed_shoulder_widths_per_s']:.2f} shoulder-widths/s"
                )
            annotated = draw_pose(
                frame,
                record.get("landmarks"),
                phase=str(metrics.get("phase", "")),
                overall_score=overall_score,
                extra_text=extra,
            )
            writer.write(annotated)
            written += 1
            _notify(
                progress_callback,
                0.72 + 0.18 * written / max(len(sampled_indices), 1),
                "Rendering annotated video",
            )
        frame_index += 1
    capture.release()
    writer.release()

    command = [
        "ffmpeg", "-y", "-i", str(temp_path),
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "24",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an",
        str(output_path),
    ]
    success, _ = run_ffmpeg(command)
    if success and output_path.exists() and output_path.stat().st_size > 0:
        temp_path.unlink(missing_ok=True)
        return output_path
    temp_path.replace(output_path)
    return output_path


def demo_landmarks(frame_index: int, total_frames: int) -> np.ndarray:
    """Generate a deterministic, plausible front-view stick-figure golf swing."""
    t = frame_index / max(total_frames - 1, 1)
    landmarks = np.zeros((33, 5), dtype=np.float32)
    landmarks[:, 3:] = 1.0

    address = np.array([0.56, 0.57])
    top = np.array([0.31, 0.24])
    impact = np.array([0.58, 0.58])
    finish = np.array([0.77, 0.27])
    if t < 0.15:
        hand_center = address + np.array([0.004 * math.sin(t * 45.0), 0.0])
        rotation = 0.0
    elif t < 0.46:
        amount = (t - 0.15) / 0.31
        hand_center = lerp(address, top, amount)
        rotation = math.sin(amount * math.pi / 2.0)
    elif t < 0.63:
        amount = (t - 0.46) / 0.17
        hand_center = lerp(top, impact, amount)
        rotation = math.cos(amount * math.pi / 2.0)
    else:
        amount = (t - 0.63) / 0.37
        hand_center = lerp(impact, finish, amount)
        rotation = -math.sin(amount * math.pi / 2.0)

    sway = 0.045 * math.sin(math.pi * t)
    shoulder_tilt = 0.030 * rotation
    hip_tilt = 0.010 * rotation
    left_shoulder = np.array([0.42 + sway, 0.36 - shoulder_tilt])
    right_shoulder = np.array([0.58 + sway, 0.36 + shoulder_tilt])
    left_hip = np.array([0.45 + sway, 0.61 - hip_tilt])
    right_hip = np.array([0.55 + sway, 0.61 + hip_tilt])

    landmarks[NOSE, :2] = [0.50 + sway + 0.018 * math.sin(2 * math.pi * t), 0.20]
    landmarks[LEFT_SHOULDER, :2] = left_shoulder
    landmarks[RIGHT_SHOULDER, :2] = right_shoulder
    landmarks[LEFT_HIP, :2] = left_hip
    landmarks[RIGHT_HIP, :2] = right_hip

    wrist_gap = np.array([0.018, 0.008])
    left_wrist = hand_center - wrist_gap
    right_wrist = hand_center + wrist_gap
    landmarks[LEFT_WRIST, :2] = left_wrist
    landmarks[RIGHT_WRIST, :2] = right_wrist
    landmarks[LEFT_ELBOW, :2] = left_shoulder * 0.44 + left_wrist * 0.56 + np.array([-0.025, 0.015])
    landmarks[RIGHT_ELBOW, :2] = right_shoulder * 0.44 + right_wrist * 0.56 + np.array([0.020, 0.010])

    knee_bend = 0.015 + 0.012 * math.sin(math.pi * t)
    landmarks[LEFT_KNEE, :2] = [0.43 + sway * 0.35, 0.78 - knee_bend]
    landmarks[RIGHT_KNEE, :2] = [0.57 + sway * 0.35, 0.78 - knee_bend]
    landmarks[LEFT_ANKLE, :2] = [0.40, 0.94]
    landmarks[RIGHT_ANKLE, :2] = [0.60, 0.94]
    landmarks[LEFT_HEEL, :2] = [0.39, 0.96]
    landmarks[RIGHT_HEEL, :2] = [0.61, 0.96]
    landmarks[LEFT_FOOT, :2] = [0.36, 0.97]
    landmarks[RIGHT_FOOT, :2] = [0.64, 0.97]

    # Fill less important facial, hand, and foot landmarks close to their parents.
    for index in range(1, 11):
        landmarks[index, :2] = landmarks[NOSE, :2] + np.array([
            (index % 3 - 1) * 0.008,
            (index // 3 - 1) * 0.006,
        ])
    for index in (17, 19, 21):
        landmarks[index, :2] = left_wrist + np.array([-0.006, 0.003 * (index - 18)])
    for index in (18, 20, 22):
        landmarks[index, :2] = right_wrist + np.array([0.006, 0.003 * (index - 19)])
    return landmarks


def create_demo_video(
    output_path: Path,
    seconds: float = 5.5,
    fps: int = 24,
    width: int = 640,
    height: int = 480,
) -> Tuple[Path, List[Dict[str, Any]], Dict[str, Any]]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame_count = int(seconds * fps)
    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError("Could not create the synthetic demo video.")
    records: List[Dict[str, Any]] = []
    for frame_index in range(frame_count):
        landmarks = demo_landmarks(frame_index, frame_count)
        frame = np.full((height, width, 3), 242, dtype=np.uint8)
        cv2.line(frame, (0, int(height * 0.97)), (width, int(height * 0.97)), (150, 150, 150), 2)
        frame = draw_pose(frame, landmarks, phase="Synthetic demo")
        cv2.putText(
            frame,
            "Synthetic swing - not a real golfer",
            (20, height - 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.56,
            (60, 60, 60),
            2,
            cv2.LINE_AA,
        )
        writer.write(frame)
        records.append({
            "frame_index": frame_index,
            "sample_index": frame_index,
            "timestamp_s": frame_index / fps,
            "landmarks": landmarks,
        })
    writer.release()
    metadata = {
        "fps": float(fps),
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration_s": seconds,
        "max_analyzed_duration_s": seconds,
        "frame_stride": 1,
        "expected_samples": frame_count,
        "sampled_frames": frame_count,
        "detected_frames": frame_count,
        "detection_coverage": 1.0,
        "pose_backend": "Synthetic deterministic demo",
    }
    return output_path, records, metadata
