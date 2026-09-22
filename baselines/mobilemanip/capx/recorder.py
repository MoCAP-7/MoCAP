"""Continuous episode recording for the Cap-X baselines on YOR.

Cap-X keeps one image per generated program, and ``events.jsonl`` one frame
per API call, so neither shows how the robot actually moved between them. For
the length of an episode a background thread therefore samples the camera
stream at a fixed rate into ``video.mp4`` (elapsed time and base pose drawn on
each frame) and the base pose into ``trajectory.jsonl``; every velocity command
the controller sends is appended to the same file.

The recorder only reads the camera stream the episode already receives. It
never calls the base RPC, whose request socket belongs to the controller
thread, and a recording failure never changes the episode.
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from .events import jsonable

RECORDING_HZ = 5.0
VIDEO_FILE = "video.mp4"
TRAJECTORY_FILE = "trajectory.jsonl"
FRAME_MAX_AGE_S = 0.75
STOP_TIMEOUT_S = 5.0
ENCODER_CLOSE_TIMEOUT_S = 30.0


class EpisodeRecorder:
    """Record video and trajectory of one episode from a ``YorEnvironment``."""

    def __init__(
        self,
        directory: str | Path,
        environment: Any,
        *,
        hz: float = RECORDING_HZ,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not math.isfinite(float(hz)) or float(hz) <= 0.0:
            raise ValueError("recording rate must be positive")
        self.directory = Path(directory)
        self.environment = environment
        self.hz = float(hz)
        self._clock = clock
        navigation_frame = getattr(environment, "navigation_frame", None)
        preview = getattr(environment, "latest_camera_preview", None)
        self._navigation_frame = navigation_frame if callable(navigation_frame) else None
        self._preview = preview if callable(preview) else None
        self.enabled = self._navigation_frame is not None or self._preview is not None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._trajectory = None
        self._video: _VideoSink | None = None
        self._video_failed = False
        self._origin: float | None = None
        self._restore_submit: Callable[[], None] | None = None
        self._counts = {"frames": 0, "frames_missing": 0, "poses": 0, "commands": 0}
        self._errors: list[str] = []

    def attach(self) -> None:
        """Start the clock and record the controller's velocity commands."""

        if not self.enabled or self._origin is not None:
            return
        self._origin = self._clock()
        original = getattr(self.environment, "submit_base_velocity", None)
        if not callable(original):
            return
        had_instance_attribute = "submit_base_velocity" in vars(self.environment)

        def submit_base_velocity(velocity: Sequence[float]) -> Any:
            try:
                reply = original(velocity)
            except Exception as exc:
                self._record_command(velocity, None, exc)
                raise
            self._record_command(velocity, reply, None)
            return reply

        self.environment.submit_base_velocity = submit_base_velocity

        def restore() -> None:
            if had_instance_attribute:
                self.environment.submit_base_velocity = original
            else:
                del self.environment.submit_base_velocity

        self._restore_submit = restore

    def start(self) -> None:
        """Attach and sample the camera stream in a background thread."""

        if not self.enabled or self._thread is not None:
            return
        self.attach()
        self._thread = threading.Thread(
            target=self._run, name="capx-episode-recorder", daemon=True
        )
        self._thread.start()

    def capture(self) -> None:
        """Record one pose sample and one video frame."""

        if not self.enabled:
            return
        if self._origin is None:
            self.attach()
        try:
            self._capture()
        except Exception as exc:  # noqa: BLE001 - recording never changes the episode
            self._error(f"capture failed: {exc!r}")

    def stop(self) -> dict[str, Any]:
        """Stop sampling, restore the command path, finish the files, and summarise."""

        if not self.enabled:
            return {"enabled": False, "reason": "the environment has no camera stream"}
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=STOP_TIMEOUT_S)
        if self._restore_submit is not None:
            self._restore_submit()
            self._restore_submit = None
        with self._lock:
            video = self._video
            self._video = None
            if self._trajectory is not None:
                self._trajectory.close()
                self._trajectory = None
        codec = None
        if video is not None:
            codec = video.codec
            try:
                video.close()
            except Exception as exc:  # noqa: BLE001 - the trajectory is still valid
                self._error(f"closing the video failed: {exc!r}")
        video_path = self.directory / VIDEO_FILE
        trajectory_path = self.directory / TRAJECTORY_FILE
        return {
            "enabled": True,
            "hz": self.hz,
            "video": VIDEO_FILE if video_path.is_file() else None,
            "video_codec": codec,
            "trajectory": TRAJECTORY_FILE if trajectory_path.is_file() else None,
            **self._counts,
            "errors": list(self._errors),
        }

    def _run(self) -> None:
        period = 1.0 / self.hz
        deadline = self._clock()
        while not self._stop.is_set():
            self.capture()
            deadline += period
            wait = deadline - self._clock()
            if wait < 0.0:
                deadline = self._clock()
                wait = 0.0
            self._stop.wait(wait)

    def _capture(self) -> None:
        pose = None
        pose_valid = False
        rgb = None
        timestamp_ns = None
        frame = None
        if self._navigation_frame is not None:
            try:
                frame = self._navigation_frame(max_age_s=FRAME_MAX_AGE_S)
            except Exception:  # noqa: BLE001 - no fresh frame or no valid pose
                frame = None
        if frame is not None:
            planar = getattr(frame, "planar_pose", None)
            if planar is not None:
                pose = [float(planar.x_m), float(planar.y_m), float(planar.yaw_rad)]
                pose_valid = bool(getattr(planar, "valid", False))
            rgb = getattr(frame, "rgb", None)
            stamp = getattr(frame, "timestamp_ns", None)
            timestamp_ns = None if stamp is None else int(stamp)
        elif self._preview is not None:
            preview = self._preview()
            if preview is not None:
                rgb = preview.get("rgb")
                stamp = preview.get("timestamp_ns")
                timestamp_ns = None if stamp is None else int(stamp)
        elapsed = self._elapsed()
        self._write(
            {
                "kind": "pose",
                "t_s": elapsed,
                "time": _now(),
                "frame_timestamp_ns": timestamp_ns,
                "pose_xy_yaw": pose,
                "pose_valid": pose_valid,
            }
        )
        self._counts["poses"] += 1
        if rgb is None:
            self._counts["frames_missing"] += 1
            return
        self._write_frame(np.asarray(rgb), elapsed, pose, pose_valid)

    def _write_frame(
        self, rgb: np.ndarray, elapsed: float, pose: list[float] | None, pose_valid: bool
    ) -> None:
        if self._video_failed:
            return
        image = _even_crop(np.ascontiguousarray(rgb, dtype=np.uint8))
        image = _overlay(image, elapsed, pose, pose_valid)
        with self._lock:
            if self._video is None:
                self.directory.mkdir(parents=True, exist_ok=True)
                self._video = _open_video(
                    self.directory / VIDEO_FILE, image.shape[1], image.shape[0], self.hz
                )
                if self._video is None:
                    self._video_failed = True
                    self._error("no video encoder is available")
                    return
            video = self._video
        try:
            video.write(image)
        except Exception as exc:  # noqa: BLE001 - keep the trajectory going
            self._video_failed = True
            self._error(f"writing the video failed: {exc!r}")
            return
        self._counts["frames"] += 1

    def _record_command(
        self, velocity: Sequence[float], reply: Any, error: BaseException | None
    ) -> None:
        try:
            record: dict[str, Any] = {
                "kind": "command",
                "t_s": self._elapsed(),
                "time": _now(),
                "velocity": jsonable(list(velocity)),
            }
            if error is None:
                record["reply"] = jsonable(reply)
            else:
                record["error"] = {"type": type(error).__name__, "message": str(error)}
            self._write(record)
            self._counts["commands"] += 1
        except Exception as exc:  # noqa: BLE001 - recording never changes the episode
            self._error(f"recording a command failed: {exc!r}")

    def _write(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, ensure_ascii=False, allow_nan=False)
        with self._lock:
            if self._trajectory is None:
                self.directory.mkdir(parents=True, exist_ok=True)
                self._trajectory = (self.directory / TRAJECTORY_FILE).open(
                    "a", encoding="utf-8"
                )
            self._trajectory.write(line + "\n")
            self._trajectory.flush()

    def _elapsed(self) -> float:
        origin = self._origin if self._origin is not None else self._clock()
        return round(self._clock() - origin, 3)

    def _error(self, message: str) -> None:
        if message in self._errors:
            return
        self._errors.append(message)
        print(f"[capx] recorder: {message}", file=sys.stderr, flush=True)


class _VideoSink:
    codec: str

    def write(self, rgb: np.ndarray) -> None:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError


class _FfmpegSink(_VideoSink):
    """H.264 through an ffmpeg pipe, playable in browsers and QuickTime."""

    codec = "libx264"

    def __init__(self, executable: str, path: Path, width: int, height: int, fps: float):
        self._process = subprocess.Popen(
            [
                executable,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                f"{width}x{height}",
                "-r",
                f"{fps:g}",
                "-i",
                "-",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-pix_fmt",
                "yuv420p",
                # A keyframe every second, each starting a fragment: a file
                # whose encoder is killed still plays up to its last second.
                "-g",
                str(max(1, round(fps))),
                "-movflags",
                "+frag_keyframe+empty_moov+default_base_moof",
                str(path),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            # The operator's Ctrl+C stops the episode, not the encoder: in the
            # terminal's process group it would end the file mid-episode.
            start_new_session=True,
        )

    def write(self, rgb: np.ndarray) -> None:
        assert self._process.stdin is not None
        self._process.stdin.write(rgb.tobytes())

    def close(self) -> None:
        assert self._process.stdin is not None
        self._process.stdin.close()
        self._process.wait(timeout=ENCODER_CLOSE_TIMEOUT_S)


class _OpenCvSink(_VideoSink):
    """MPEG-4 Part 2 through OpenCV when no H.264 encoder is installed."""

    codec = "mp4v"

    def __init__(self, writer: Any, cv2: Any):
        self._writer = writer
        self._cv2 = cv2

    def write(self, rgb: np.ndarray) -> None:
        self._writer.write(self._cv2.cvtColor(rgb, self._cv2.COLOR_RGB2BGR))

    def close(self) -> None:
        self._writer.release()


def _open_video(path: Path, width: int, height: int, fps: float) -> _VideoSink | None:
    executable = _ffmpeg_with_libx264()
    if executable is not None:
        try:
            return _FfmpegSink(executable, path, width, height, fps)
        except OSError:
            pass
    try:
        import cv2
    except ImportError:
        return None
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        writer.release()
        return None
    return _OpenCvSink(writer, cv2)


def _ffmpeg_with_libx264() -> str | None:
    candidates = []
    found = shutil.which("ffmpeg")
    if found:
        candidates.append(found)
    try:
        import imageio_ffmpeg

        candidates.append(imageio_ffmpeg.get_ffmpeg_exe())
    except Exception:  # noqa: BLE001 - optional encoder source
        pass
    for executable in candidates:
        try:
            encoders = subprocess.run(
                [executable, "-hide_banner", "-encoders"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            ).stdout
        except (OSError, subprocess.SubprocessError):
            continue
        if "libx264" in encoders:
            return executable
    return None


def _even_crop(image: np.ndarray) -> np.ndarray:
    height = image.shape[0] - image.shape[0] % 2
    width = image.shape[1] - image.shape[1] % 2
    return image[:height, :width]


def _overlay(
    image: np.ndarray, elapsed: float, pose: list[float] | None, pose_valid: bool
) -> np.ndarray:
    try:
        import cv2
    except ImportError:
        return image
    if pose is None:
        text = f"t {elapsed:6.1f} s  pose unavailable"
    else:
        text = (
            f"t {elapsed:6.1f} s  x {pose[0]:+.2f} m  y {pose[1]:+.2f} m  "
            f"yaw {math.degrees(pose[2]):+.0f} deg"
        )
        if not pose_valid:
            text += "  (invalid)"
    image = image.copy()
    for color, thickness in (((0, 0, 0), 3), ((255, 255, 255), 1)):
        cv2.putText(
            image, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness, cv2.LINE_AA
        )
    return image


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="milliseconds")
