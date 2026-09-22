"""Offline manipulation-event labelling for the passive-video readiness prior.

A passive egocentric video of a person walking up to an object and picking it
up contains, one moment before the hand moves, the stance from which that
person found the object reachable.  The robot's docking controller later
matches that "ready" frame against its own camera image to recover the bearing
the human docked from (``yor_agent.robot.readiness_prior``).  This module finds
those moments offline:

1. ego-motion from ffmpeg's ``tblend=difference`` + ``signalstats`` filters
   (the mean absolute inter-frame difference, a cheap head-motion proxy);
2. low-motion windows (the wearer standing still) that prune the video before
   a VLM labels ``object / hand / t_ready / t_grasp``, or that are merely
   recorded when the operator labels the events by hand;
3. three JPEGs per event (``nav`` = t_ready - 2 s, ``ready``, ``grasp``) and
   one events JSON (schema ``yor-manipulation-events-v1``).

The module uses the standard library only: ffmpeg / ffprobe run through
``subprocess`` and the VLM SDKs are imported lazily by
:func:`nav_planner.memory.make_client`.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import re
import shutil
import statistics
import struct
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .json_utils import atomic_write_json, read_json
from .memory import _provider_for_model, make_client
from .prompts import MANIPULATION_EVENTS_PROMPT
from .reference_frames import _find_ffmpeg, extract_video_sample_frames

EVENTS_SCHEMA = "yor-manipulation-events-v1"
HANDS = ("left", "right", "both")
LABEL_MODES = ("vlm", "manual")
FRAME_ROLES = ("nav", "ready", "grasp")
NAV_FRAME_LEAD_S = 2.0

_PTS_TIME = re.compile(r"pts_time:(\S+)")
_YAVG = re.compile(r"lavfi\.signalstats\.YAVG=(\S+)")
_FENCE = re.compile(r"^```[A-Za-z0-9_-]*[ \t]*\r?\n(.*?)\r?\n?```\s*$", re.S)
# JPEG start-of-frame markers (baseline, progressive, lossless, ...), which all
# carry precision, height, width after the segment length.  C4/C8/CC are
# Huffman tables, JPEG extensions, and arithmetic tables, not frame headers.
_SOF_MARKERS = frozenset(
    {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}
)
_THINKING_LEVELS = {
    "openai": frozenset({"none", "low", "medium", "high", "xhigh", "max"}),
    "gemini": frozenset({"low", "medium", "high"}),
}


@dataclass(frozen=True)
class ReadinessConfig:
    """Tunables for the offline event labelling.

    ``motion_threshold`` is the per-second YAVG below which the wearer counts
    as standing still; ``None`` derives it as ``motion_threshold_factor`` x
    the median second of the video (1.0 by default: a person standing at a
    desk still moves head and hands, so the still seconds sit only a little
    below the median and a tighter factor prunes the very event we want; the
    labeler rejects the extra windows).  ``min_window_s`` and
    ``window_margin_s`` shape the low-motion windows (runs of still seconds,
    grown by the margin on both sides).  The ``sample_*`` values describe the
    frames a VLM sees per window; the ``frame_width`` is the width of the
    saved nav/ready/grasp JPEGs.
    """

    motion_threshold: float | None = None
    motion_threshold_factor: float = 1.0
    motion_scale_width: int = 160
    min_window_s: float = 1.5
    window_margin_s: float = 1.0
    sample_fps: float = 2.0
    sample_width: int = 768
    frame_width: int = 1280
    model: str = "gemini-3.7-flash"
    thinking_level: str = "high"
    request_timeout_s: float = 900.0
    image_detail: str = "high"

    def __post_init__(self) -> None:
        if self.motion_threshold is not None and not (
            math.isfinite(self.motion_threshold) and self.motion_threshold >= 0
        ):
            raise ValueError("motion_threshold must be a finite value >= 0 or None")
        if not (
            math.isfinite(self.motion_threshold_factor)
            and self.motion_threshold_factor > 0
        ):
            raise ValueError("motion_threshold_factor must be a finite value > 0")
        if self.motion_scale_width < 16:
            raise ValueError("motion_scale_width must be at least 16 px")
        if self.min_window_s < 0:
            raise ValueError("min_window_s must be >= 0")
        if self.window_margin_s < 0:
            raise ValueError("window_margin_s must be >= 0")
        if self.sample_fps <= 0:
            raise ValueError("sample_fps must be positive")
        if self.sample_width <= 0:
            raise ValueError("sample_width must be positive")
        if self.frame_width <= 0:
            raise ValueError("frame_width must be positive")
        if self.request_timeout_s <= 0:
            raise ValueError("request_timeout_s must be positive")
        if self.image_detail not in {"low", "high", "original", "auto"}:
            raise ValueError("image_detail must be low, high, original, or auto")


# ---------------------------------------------------------------------------
# Ego-motion


def parse_signalstats(text: str) -> list[tuple[float, float]]:
    """Parse ffmpeg ``metadata=print`` output into ``(pts_time, yavg)`` pairs.

    The filter prints two lines per frame::

        frame:0    pts:1024    pts_time:0.1
        lavfi.signalstats.YAVG=1.37552

    Lines that do not belong to that pattern are ignored, so ffmpeg warnings
    interleaved on the same stream cannot break the parse.
    """

    samples: list[tuple[float, float]] = []
    pending: float | None = None
    for line in text.splitlines():
        time_match = _PTS_TIME.search(line)
        if time_match:
            try:
                pending = float(time_match.group(1))
            except ValueError:
                pending = None
            continue
        value_match = _YAVG.search(line)
        if value_match and pending is not None:
            try:
                samples.append((pending, float(value_match.group(1))))
            except ValueError:
                pass
            pending = None
    return samples


def ego_motion_per_frame(
    video: str | Path, *, scale_width: int = 160
) -> list[tuple[float, float]]:
    """Mean absolute luma difference between consecutive frames.

    The video is shrunk to ``scale_width`` px and converted to gray, then
    ``tblend=all_mode=difference`` turns each frame into |frame - previous| and
    ``signalstats`` reports its average (YAVG, 0..255).  Head motion, walking,
    and turning all raise it; standing still drops it close to zero.  The first
    sample sits one frame after the start because ``tblend`` consumes a frame.
    """

    ffmpeg = _find_ffmpeg()
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video),
        "-vf",
        (
            f"scale={int(scale_width)}:-2,format=gray,"
            "tblend=all_mode=difference,signalstats,"
            "metadata=print:key=lavfi.signalstats.YAVG:file=-"
        ),
        "-an",
        "-f",
        "null",
        "-",
    ]
    completed = subprocess.run(
        command, check=False, capture_output=True, text=True
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or "ffmpeg failed"
        raise RuntimeError(f"failed to measure ego-motion of {video}: {detail}")
    samples = parse_signalstats(completed.stdout)
    if not samples:
        raise RuntimeError(
            f"ffmpeg reported no frame-difference statistics for {video}"
        )
    return samples


def per_second_means(samples: Sequence[tuple[float, float]]) -> list[float]:
    """Average the per-frame values over whole seconds.

    Index ``s`` of the result is the mean over frames with ``floor(pts_time)
    == s``.  The list runs from second 0 to the last second that has a frame;
    a second without frames (a dropped stretch) repeats the previous mean so
    the list stays index-aligned with the timeline.
    """

    sums: dict[int, float] = {}
    counts: dict[int, int] = {}
    for pts_time, value in samples:
        second = int(math.floor(pts_time + 1e-6))
        if second < 0:
            continue
        sums[second] = sums.get(second, 0.0) + float(value)
        counts[second] = counts.get(second, 0) + 1
    if not sums:
        return []
    first = min(sums)
    last_mean = sums[first] / counts[first]
    means: list[float] = []
    for second in range(max(sums) + 1):
        if second in sums:
            last_mean = sums[second] / counts[second]
        means.append(last_mean)
    return means


def low_motion_windows(
    per_second: Sequence[float],
    *,
    threshold: float,
    min_window_s: float,
    margin_s: float,
    duration_s: float,
) -> list[dict[str, float]]:
    """Runs of still seconds, grown by ``margin_s``, clamped, and merged.

    A second is still when its mean YAVG is below ``threshold``.  A run of
    consecutive still seconds is kept when it spans at least ``min_window_s``
    (1.5 s therefore needs two whole seconds), expanded by ``margin_s`` on
    both sides so the approach and the first hand motion stay inside the
    window, clamped to ``[0, duration_s]``, and overlapping windows merge.
    """

    if duration_s <= 0:
        raise ValueError("duration_s must be positive")
    raw: list[tuple[float, float]] = []
    run_start: int | None = None
    # The sentinel closes a run that reaches the end of the video.
    for second, value in enumerate([*per_second, math.inf]):
        still = value < threshold
        if still and run_start is None:
            run_start = second
        elif not still and run_start is not None:
            if second - run_start >= min_window_s:
                raw.append((run_start - margin_s, second + margin_s))
            run_start = None
    merged: list[list[float]] = []
    for start, end in raw:
        start = max(0.0, float(start))
        end = min(float(duration_s), float(end))
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [{"start_s": start, "end_s": end} for start, end in merged]


# ---------------------------------------------------------------------------
# Event labels


def parse_manual_event(text: str) -> dict[str, Any]:
    """Parse ``OBJECT:HAND:T_READY:T_GRASP`` (for example ``can:right:16.5:19.0``)."""

    parts = [part.strip() for part in str(text).split(":")]
    if len(parts) != 4:
        raise ValueError(
            "event must be OBJECT:HAND:T_READY:T_GRASP "
            f"(for example can:right:16.5:19.0), got {text!r}"
        )
    object_name, hand, t_ready, t_grasp = parts
    try:
        t_ready_s = float(t_ready)
        t_grasp_s = float(t_grasp)
    except ValueError:
        raise ValueError(
            f"event times must be seconds as numbers, got {text!r}"
        ) from None
    return normalize_event(
        {
            "object": object_name,
            "hand": hand,
            "t_ready_s": t_ready_s,
            "t_grasp_s": t_grasp_s,
            "confidence": None,
            "notes": "",
        }
    )


def normalize_event(
    event: Mapping[str, Any], *, duration_s: float | None = None
) -> dict[str, Any]:
    """Validate one event and return it with the locked key set and types."""

    if not isinstance(event, Mapping):
        raise ValueError("each event must be a JSON object")
    object_name = str(event.get("object") or "").strip()
    if not object_name:
        raise ValueError("event object must not be empty")
    hand = str(event.get("hand") or "").strip().lower()
    if hand not in HANDS:
        raise ValueError(
            f"event hand must be one of {', '.join(HANDS)}; got {hand!r}"
        )
    t_ready_s = _seconds(event.get("t_ready_s"), "t_ready_s")
    t_grasp_s = _seconds(event.get("t_grasp_s"), "t_grasp_s")
    if t_ready_s < 0:
        raise ValueError(f"t_ready_s must be >= 0, got {t_ready_s}")
    if t_grasp_s < t_ready_s:
        raise ValueError(
            f"t_grasp_s ({t_grasp_s}) must not precede t_ready_s ({t_ready_s})"
        )
    if duration_s is not None and t_grasp_s > duration_s + 1e-6:
        raise ValueError(
            f"t_grasp_s ({t_grasp_s}) exceeds the video duration ({duration_s:.3f} s)"
        )
    confidence = event.get("confidence")
    if confidence is not None:
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            raise ValueError("event confidence must be a number or null")
        confidence = float(confidence)
    notes = event.get("notes")
    return {
        "object": object_name,
        "hand": hand,
        "t_ready_s": t_ready_s,
        "t_grasp_s": t_grasp_s,
        "confidence": confidence,
        "notes": "" if notes is None else str(notes),
    }


def parse_events_response(text: str, *, duration_s: float) -> list[dict[str, Any]]:
    """Parse the VLM reply: strict JSON, tolerating fences and surrounding prose."""

    payload = _loads_tolerant(text)
    events = payload.get("events")
    if not isinstance(events, list):
        raise ValueError('VLM response must contain an "events" list')
    return [normalize_event(event, duration_s=duration_s) for event in events]


def _loads_tolerant(text: str) -> dict[str, Any]:
    stripped = str(text).strip()
    fenced = _FENCE.match(stripped)
    if fenced:
        stripped = fenced.group(1).strip()
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        if start < 0:
            raise ValueError("VLM response contains no JSON object") from None
        try:
            value, _ = json.JSONDecoder().raw_decode(stripped, start)
        except json.JSONDecodeError:
            end = stripped.rfind("}")
            try:
                value = json.loads(stripped[start : end + 1])
            except json.JSONDecodeError as exc:
                raise ValueError(f"VLM response is not valid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("VLM response must be a JSON object")
    return value


def _seconds(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be seconds as a number")
    if isinstance(value, (int, float)):
        seconds = float(value)
    elif isinstance(value, str):
        try:
            seconds = float(value.strip())
        except ValueError:
            raise ValueError(f"{name} must be seconds as a number, got {value!r}") from None
    else:
        raise ValueError(f"{name} must be seconds as a number, got {value!r}")
    if not math.isfinite(seconds):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return seconds


# ---------------------------------------------------------------------------
# Video metadata, intrinsics, frames


def probe_video(video: str | Path) -> dict[str, Any]:
    """Width, height, fps, and duration of the first video stream via ffprobe."""

    ffprobe = _find_ffprobe()
    command = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate:format=duration",
        "-of",
        "json",
        str(video),
    ]
    completed = subprocess.run(
        command, check=False, capture_output=True, text=True
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or "ffprobe failed"
        raise RuntimeError(f"failed to probe {video}: {detail}")
    try:
        payload = json.loads(completed.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"ffprobe returned invalid JSON for {video}: {exc}") from exc
    streams = payload.get("streams") or []
    if not streams:
        raise RuntimeError(f"{video} has no video stream")
    stream = streams[0]
    try:
        width = int(stream["width"])
        height = int(stream["height"])
        duration_s = float((payload.get("format") or {})["duration"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"ffprobe output for {video} lacks size or duration: {exc}") from exc
    if width <= 0 or height <= 0 or duration_s <= 0:
        raise RuntimeError(
            f"{video} reports a non-positive size or duration ({width}x{height}, {duration_s} s)"
        )
    return {
        "width": width,
        "height": height,
        "fps": _parse_rate(stream.get("avg_frame_rate")),
        "duration_s": duration_s,
    }


def load_intrinsics_sidecar(
    video: str | Path, explicit: str | Path | None = None
) -> dict[str, Any] | None:
    """Pinhole intrinsics from ``<video>.intrinsics.json`` or an explicit path.

    ``x_undistorted.mp4`` looks for ``x_undistorted.intrinsics.json``.  A
    missing default sidecar returns ``None`` (the runtime then sweeps the focal
    length); a missing explicit path is an error.
    """

    if explicit is not None:
        path = Path(explicit).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
    else:
        path = Path(video).expanduser().resolve().with_suffix(".intrinsics.json")
        if not path.is_file():
            return None
    payload = read_json(path)
    model = str(payload.get("model") or "pinhole")
    if model != "pinhole":
        raise ValueError(f"{path}: only pinhole intrinsics are supported, got {model!r}")
    try:
        intrinsics = {
            "model": "pinhole",
            "width": int(payload["width"]),
            "height": int(payload["height"]),
            "fx": float(payload["fx"]),
            "fy": float(payload["fy"]),
            "cx": float(payload["cx"]),
            "cy": float(payload["cy"]),
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"{path}: pinhole intrinsics need width, height, fx, fy, cx, cy ({exc})"
        ) from exc
    for key in ("width", "height", "fx", "fy"):
        if intrinsics[key] <= 0:
            raise ValueError(f"{path}: {key} must be positive")
    intrinsics["source_path"] = str(path)
    return intrinsics


def jpeg_dimensions(data: bytes) -> tuple[int, int]:
    """``(width, height)`` from a JPEG's start-of-frame header, no decoder needed."""

    if len(data) < 4 or data[:2] != b"\xff\xd8":
        raise ValueError("not a JPEG file")
    index = 2
    while index + 4 <= len(data):
        if data[index] != 0xFF:
            raise ValueError("corrupt JPEG marker stream")
        marker = data[index + 1]
        if marker == 0xFF:
            index += 1
            continue
        if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
            index += 2
            continue
        if marker in (0xD9, 0xDA):
            break
        (length,) = struct.unpack(">H", data[index + 2 : index + 4])
        if marker in _SOF_MARKERS:
            if index + 9 > len(data):
                raise ValueError("truncated JPEG frame header")
            height, width = struct.unpack(">HH", data[index + 5 : index + 9])
            return int(width), int(height)
        index += 2 + length
    raise ValueError("JPEG has no frame header")


def extract_frame(
    video: str | Path,
    time_s: float,
    destination: Path,
    *,
    max_width: int,
) -> tuple[int, int]:
    """Save one JPEG at ``time_s`` (at most ``max_width`` wide) and return its size."""

    ffmpeg = _find_ffmpeg()
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{time_s:.3f}",
        "-i",
        str(video),
        "-frames:v",
        "1",
        "-vf",
        f"scale={int(max_width)}:-2:force_original_aspect_ratio=decrease",
        "-q:v",
        "2",
        "-y",
        str(destination),
    ]
    completed = subprocess.run(
        command, check=False, capture_output=True, text=True
    )
    if completed.returncode != 0 or not destination.is_file():
        detail = completed.stderr.strip() or "ffmpeg produced no image"
        raise RuntimeError(f"failed to extract the frame at {time_s:.3f} s: {detail}")
    return jpeg_dimensions(destination.read_bytes())


# ---------------------------------------------------------------------------
# Builder


class ManipulationEventsBuilder:
    """Turn one passive video into ``manipulation_events_<version>.json``.

    ``label="manual"`` uses the operator's events and never touches a VLM
    client; ``label="vlm"`` sends each low-motion window's sampled frames to
    the configured Gemini or OpenAI model.  ``client`` injects a pre-built
    client (tests), ``on_progress`` receives one dict per progress event.
    """

    def __init__(
        self,
        config: ReadinessConfig | None = None,
        *,
        client: Any | None = None,
        on_progress: Any | None = None,
    ) -> None:
        self.config = config or ReadinessConfig()
        self._client = client
        self._on_progress = on_progress

    def build(
        self,
        video_path: str | Path,
        output_dir: str | Path,
        *,
        label: str = "vlm",
        manual_events: Sequence[Mapping[str, Any]] = (),
        intrinsics_path: str | Path | None = None,
    ) -> dict[str, Any]:
        if label not in LABEL_MODES:
            raise ValueError(f"label must be one of {', '.join(LABEL_MODES)}; got {label!r}")
        video = Path(video_path).expanduser().resolve()
        if not video.is_file():
            raise FileNotFoundError(video)
        destination_dir = Path(output_dir).expanduser().resolve()
        created_at = datetime.now(timezone.utc)
        version = created_at.strftime("%Y%m%dT%H%M%S.%fZ")
        output = destination_dir / f"manipulation_events_{version}.json"
        frame_dir = destination_dir / f"{output.stem}_frames"

        source = self._source_metadata(video)
        self._progress(
            "video_probed",
            path=str(video),
            width=source["width"],
            height=source["height"],
            fps=source["fps"],
            duration_s=source["duration_s"],
        )
        intrinsics = load_intrinsics_sidecar(video, intrinsics_path)
        if intrinsics is not None and (
            intrinsics["width"] != source["width"]
            or intrinsics["height"] != source["height"]
        ):
            raise ValueError(
                f"intrinsics {intrinsics['source_path']} describe a "
                f"{intrinsics['width']}x{intrinsics['height']} image but the video is "
                f"{source['width']}x{source['height']}"
            )
        self._progress(
            "intrinsics_loaded",
            path=None if intrinsics is None else intrinsics["source_path"],
        )

        motion = self._analyse_motion(video, source["duration_s"])
        self._progress(
            "motion_windows_found",
            threshold=motion["threshold"],
            median_yavg=motion["median_yavg"],
            windows=motion["windows"],
        )

        if label == "manual":
            if not manual_events:
                raise ValueError("manual labelling needs at least one event")
            events = [
                normalize_event(event, duration_s=source["duration_s"])
                for event in manual_events
            ]
            labeling: dict[str, Any] = {
                "mode": "manual",
                "model": None,
                "prompt_sha256": None,
                "raw_response": None,
            }
        else:
            events, labeling = self._label_with_vlm(video, source, motion["windows"])
        events.sort(key=lambda event: (event["t_ready_s"], event["t_grasp_s"]))

        for index, event in enumerate(events):
            event["frames"] = self._extract_event_frames(
                video, source, event, index, frame_dir, destination_dir
            )
            self._progress(
                "event_frames_saved",
                index=index,
                object=event["object"],
                hand=event["hand"],
                t_ready_s=event["t_ready_s"],
                t_grasp_s=event["t_grasp_s"],
            )

        result = {
            "schema_version": EVENTS_SCHEMA,
            "created_at": created_at.isoformat(),
            "artifact_path": str(output),
            "source_video": source,
            "intrinsics": intrinsics,
            "motion": motion,
            "labeling": labeling,
            "events": events,
        }
        atomic_write_json(output, result)
        self._progress(
            "events_saved",
            path=str(output),
            events=len(events),
            windows=len(motion["windows"]),
        )
        return result

    # -- steps ---------------------------------------------------------------

    def _analyse_motion(self, video: Path, duration_s: float) -> dict[str, Any]:
        self._progress("ego_motion_started", scale_width=self.config.motion_scale_width)
        samples = ego_motion_per_frame(
            video, scale_width=self.config.motion_scale_width
        )
        per_second = per_second_means(samples)
        median = statistics.median(per_second) if per_second else 0.0
        threshold = (
            self.config.motion_threshold_factor * median
            if self.config.motion_threshold is None
            else float(self.config.motion_threshold)
        )
        windows = low_motion_windows(
            per_second,
            threshold=threshold,
            min_window_s=self.config.min_window_s,
            margin_s=self.config.window_margin_s,
            duration_s=duration_s,
        )
        return {
            "threshold": round(threshold, 4),
            "median_yavg": round(median, 4),
            "per_second_yavg": [round(value, 4) for value in per_second],
            "windows": [
                {"start_s": round(w["start_s"], 3), "end_s": round(w["end_s"], 3)}
                for w in windows
            ],
        }

    def _label_with_vlm(
        self,
        video: Path,
        source: dict[str, Any],
        windows: Sequence[Mapping[str, float]],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        provider = _provider_for_model(self.config.model)
        allowed = _THINKING_LEVELS[provider]
        if self.config.thinking_level not in allowed:
            raise ValueError(
                f"thinking_level for {provider} must be one of {', '.join(sorted(allowed))}"
            )
        client = self._get_client(provider)
        prompt = MANIPULATION_EVENTS_PROMPT
        events: list[dict[str, Any]] = []
        raw_responses: list[str] = []
        for window in windows:
            start_s = float(window["start_s"])
            end_s = float(window["end_s"])
            with tempfile.TemporaryDirectory(prefix="yor-readiness-window-") as directory:
                frames = self._sample_window(video, start_s, end_s, Path(directory) / "frames")
                self._progress(
                    "window_labeling_started",
                    start_s=start_s,
                    end_s=end_s,
                    frames=len(frames),
                    model=self.config.model,
                )
                context = (
                    f"The frames below are sampled at {self.config.sample_fps:g} frames "
                    f"per second from {_format_seconds(start_s)} to {_format_seconds(end_s)} "
                    f"of a {source['duration_s']:.1f} s video, a stretch in which the camera "
                    "wearer was mostly standing still. Each image is preceded by its "
                    "absolute video timestamp (M:SS.mmm); report t_ready_s and t_grasp_s "
                    "in seconds on that same clock."
                )
                if provider == "openai":
                    text = _request_openai(
                        client,
                        model=self.config.model,
                        thinking_level=self.config.thinking_level,
                        timeout_s=self.config.request_timeout_s,
                        image_detail=self.config.image_detail,
                        prompt=prompt,
                        context=context,
                        frames=frames,
                    )
                else:
                    text = _request_gemini(
                        client,
                        model=self.config.model,
                        thinking_level=self.config.thinking_level,
                        timeout_s=self.config.request_timeout_s,
                        prompt=prompt,
                        context=context,
                        frames=frames,
                    )
            raw_responses.append(f"# window {start_s:.3f}-{end_s:.3f} s\n{text}")
            window_events = parse_events_response(text, duration_s=source["duration_s"])
            events.extend(window_events)
            self._progress(
                "window_labeled",
                start_s=start_s,
                end_s=end_s,
                events=len(window_events),
            )
        labeling = {
            "mode": "vlm",
            "model": self.config.model,
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "raw_response": "\n\n".join(raw_responses) if raw_responses else None,
        }
        return events, labeling

    def _sample_window(
        self, video: Path, start_s: float, end_s: float, directory: Path
    ) -> list[dict[str, Any]]:
        duration_s = max(end_s - start_s, 1.0 / self.config.sample_fps)
        limit = int(min(1500, max(1, math.ceil(duration_s * self.config.sample_fps) + 1)))
        sampled = extract_video_sample_frames(
            video,
            directory,
            fps=self.config.sample_fps,
            limit=limit,
            max_width=self.config.sample_width,
            start_s=start_s,
            duration_s=duration_s,
        )
        frames: list[dict[str, Any]] = []
        for frame in sampled:
            frames.append(
                {
                    "timestamp": str(frame["timestamp"]),
                    "time_s": float(frame["time_s"]),
                    "data": base64.b64encode(Path(frame["path"]).read_bytes()).decode("ascii"),
                }
            )
        return frames

    def _extract_event_frames(
        self,
        video: Path,
        source: dict[str, Any],
        event: Mapping[str, Any],
        index: int,
        frame_dir: Path,
        relative_to: Path,
    ) -> dict[str, dict[str, Any]]:
        times = {
            "nav": max(0.0, float(event["t_ready_s"]) - NAV_FRAME_LEAD_S),
            "ready": float(event["t_ready_s"]),
            "grasp": float(event["t_grasp_s"]),
        }
        frames: dict[str, dict[str, Any]] = {}
        for role in FRAME_ROLES:
            seek_s = _clamp_seek(times[role], source)
            destination = frame_dir / f"event{index:02d}_{role}.jpg"
            width, height = extract_frame(
                video, seek_s, destination, max_width=self.config.frame_width
            )
            frames[role] = {
                "path": destination.relative_to(relative_to).as_posix(),
                "time_s": seek_s,
                "width": width,
                "height": height,
                "scale": width / source["width"],
            }
        return frames

    # -- helpers -------------------------------------------------------------

    def _get_client(self, provider: str) -> Any:
        if self._client is None:
            self._client = make_client(
                provider, timeout_s=self.config.request_timeout_s
            )
        return self._client

    @staticmethod
    def _source_metadata(video: Path) -> dict[str, Any]:
        stat = video.stat()
        probed = probe_video(video)
        return {
            "path": str(video),
            "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "width": probed["width"],
            "height": probed["height"],
            "fps": probed["fps"],
            "duration_s": probed["duration_s"],
        }

    def _progress(self, event: str, **payload: Any) -> None:
        if self._on_progress is not None:
            self._on_progress({"event": event, **payload})


def _request_gemini(
    client: Any,
    *,
    model: str,
    thinking_level: str,
    timeout_s: float,
    prompt: str,
    context: str,
    frames: Sequence[Mapping[str, Any]],
) -> str:
    inputs: list[dict[str, Any]] = [
        {"type": "text", "text": prompt},
        {"type": "text", "text": context},
    ]
    for frame in frames:
        inputs.extend(
            [
                {"type": "text", "text": f"VIDEO FRAME at {frame['timestamp']}:"},
                {"type": "image", "mime_type": "image/jpeg", "data": frame["data"]},
            ]
        )
    interaction = client.interactions.create(
        model=model,
        input=inputs,
        generation_config={"thinking_level": thinking_level},
        timeout=timeout_s,
    )
    text = str(getattr(interaction, "output_text", "") or "").strip()
    if not text:
        raise RuntimeError("Gemini returned an empty manipulation-event response")
    return text


def _request_openai(
    client: Any,
    *,
    model: str,
    thinking_level: str,
    timeout_s: float,
    image_detail: str,
    prompt: str,
    context: str,
    frames: Sequence[Mapping[str, Any]],
) -> str:
    content: list[dict[str, Any]] = [
        {"type": "input_text", "text": prompt},
        {"type": "input_text", "text": context},
    ]
    for frame in frames:
        content.extend(
            [
                {"type": "input_text", "text": f"VIDEO FRAME at {frame['timestamp']}:"},
                {
                    "type": "input_image",
                    "image_url": f"data:image/jpeg;base64,{frame['data']}",
                    "detail": image_detail,
                },
            ]
        )
    response = client.responses.create(
        model=model,
        input=[{"role": "user", "content": content}],
        reasoning={"effort": thinking_level},
        timeout=timeout_s,
    )
    text = str(getattr(response, "output_text", "") or "").strip()
    if not text:
        raise RuntimeError("OpenAI returned an empty manipulation-event response")
    return text


def _clamp_seek(time_s: float, source: Mapping[str, Any]) -> float:
    """Keep a seek inside the decodable range (the last frame starts 1/fps before the end)."""

    fps = float(source.get("fps") or 0.0)
    duration_s = float(source["duration_s"])
    last_frame_s = duration_s - (1.0 / fps if fps > 0 else 0.0)
    return max(0.0, min(float(time_s), max(0.0, last_frame_s)))


def _format_seconds(seconds: float) -> str:
    minutes, remainder = divmod(max(0.0, seconds), 60.0)
    return f"{int(minutes):d}:{remainder:06.3f}"


def _parse_rate(value: Any) -> float:
    text = str(value or "").strip()
    if not text:
        return 0.0
    if "/" in text:
        numerator, _, denominator = text.partition("/")
        try:
            num = float(numerator)
            den = float(denominator)
        except ValueError:
            return 0.0
        return num / den if den else 0.0
    try:
        return float(text)
    except ValueError:
        return 0.0


def _find_ffprobe() -> str:
    executable = shutil.which("ffprobe")
    if executable is not None:
        return executable
    # imageio-ffmpeg ships only ffmpeg, but some installs place ffprobe next to it.
    try:
        import imageio_ffmpeg

        bundled = Path(imageio_ffmpeg.get_ffmpeg_exe())
    except (ImportError, RuntimeError):
        bundled = None
    if bundled is not None:
        for candidate in (
            bundled.with_name("ffprobe"),
            bundled.with_name(bundled.name.replace("ffmpeg", "ffprobe")),
        ):
            if candidate.is_file():
                return str(candidate)
    raise RuntimeError(
        "ffprobe is required to read the video size and duration; install ffmpeg "
        "(brew install ffmpeg / apt install ffmpeg) so ffprobe is on PATH"
    )
