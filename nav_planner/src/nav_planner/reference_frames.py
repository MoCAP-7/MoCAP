"""Extract ordered video samples and cited visual evidence."""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path
from typing import Any


_TIMESTAMP = re.compile(
    r"(?<![\d:])(?:(?P<hours>\d{1,2}):)?"
    r"(?P<minutes>[0-5]?\d):(?P<seconds>[0-5]\d)(?![\d:])"
)


def extract_video_sample_frames(
    video_path: Path,
    output_dir: Path,
    *,
    fps: float = 2.0,
    limit: int = 300,
    max_width: int = 2048,
    start_s: float = 0.0,
    duration_s: float | None = None,
) -> list[dict[str, Any]]:
    """Extract a chronological image sequence for image-only VLMs.

    ``start_s`` / ``duration_s`` restrict the sampling to one stretch of the
    video (an input-side seek and read limit); the returned ``time_s`` and
    ``timestamp`` are absolute video times, so they stay comparable across
    windows.  The defaults sample the whole video, exactly as before.
    """

    if fps <= 0:
        raise ValueError("video frame fps must be positive")
    if not 0 < limit <= 1500:
        raise ValueError("video input frame limit must be in [1, 1500]")
    if max_width <= 0:
        raise ValueError("video frame width must be positive")
    if start_s < 0:
        raise ValueError("video sample start must be >= 0 s")
    if duration_s is not None and duration_s <= 0:
        raise ValueError("video sample duration must be positive")
    ffmpeg = _find_ffmpeg()
    output_dir.mkdir(parents=True, exist_ok=False)
    destination_pattern = output_dir / "frame_%06d.jpg"
    command = [ffmpeg, "-hide_banner", "-loglevel", "error"]
    if start_s > 0:
        command.extend(["-ss", f"{start_s:.3f}"])
    if duration_s is not None:
        command.extend(["-t", f"{duration_s:.3f}"])
    command += [
        "-i",
        str(video_path),
        "-vf",
        (
            f"fps={fps:.8f},"
            f"scale={max_width}:-2:force_original_aspect_ratio=decrease"
        ),
        "-frames:v",
        str(limit),
        "-q:v",
        "2",
        "-start_number",
        "0",
        "-y",
        str(destination_pattern),
    ]
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    paths = sorted(output_dir.glob("frame_*.jpg"))
    if completed.returncode != 0 or not paths:
        detail = completed.stderr.strip() or "ffmpeg produced no sampled frames"
        raise RuntimeError(f"failed to sample video frames: {detail}")
    frames: list[dict[str, Any]] = []
    for index, path in enumerate(paths):
        seconds = start_s + index / fps
        frames.append(
            {
                "timestamp": _format_timestamp(seconds),
                "time_s": seconds,
                "path": path,
                "mime_type": "image/jpeg",
            }
        )
    return frames


def cited_timestamps(text: str, *, limit: int = 32) -> list[tuple[str, float]]:
    """Return unique cited timestamps, spread across the observed timeline."""

    by_seconds: dict[int, str] = {}
    for match in _TIMESTAMP.finditer(text):
        hours = int(match.group("hours") or 0)
        minutes = int(match.group("minutes"))
        seconds = int(match.group("seconds"))
        total = hours * 3600 + minutes * 60 + seconds
        by_seconds.setdefault(total, match.group(0))
    ordered = sorted(by_seconds.items())
    if limit <= 0:
        return []
    if len(ordered) > limit:
        # Do not discard the end of a long tour merely because early sections
        # contain many citations.
        indices = {
            round(index * (len(ordered) - 1) / (limit - 1))
            for index in range(limit)
        } if limit > 1 else {len(ordered) // 2}
        ordered = [ordered[index] for index in sorted(indices)]
    return [(label, float(seconds)) for seconds, label in ordered]


def extract_reference_frames(
    video_path: Path,
    memory_text: str,
    output_dir: Path,
    *,
    relative_to: Path,
    limit: int = 32,
    max_width: int = 768,
) -> list[dict[str, Any]]:
    """Extract JPEGs at cited timestamps and return artifact metadata."""

    timestamps = cited_timestamps(memory_text, limit=limit)
    if not timestamps:
        return []
    ffmpeg = _find_ffmpeg()
    output_dir.mkdir(parents=True, exist_ok=False)
    frames: list[dict[str, Any]] = []
    for label, seconds in timestamps:
        filename = f"frame_{int(seconds):06d}.jpg"
        destination = output_dir / filename
        command = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{seconds:.3f}",
            "-i",
            str(video_path),
            "-frames:v",
            "1",
            "-vf",
            f"scale={max_width}:-2:force_original_aspect_ratio=decrease",
            "-q:v",
            "2",
            "-y",
            str(destination),
        ]
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0 or not destination.is_file():
            detail = completed.stderr.strip() or "ffmpeg produced no image"
            raise RuntimeError(f"failed to extract {label}: {detail}")
        frames.append(
            {
                "timestamp": label,
                "time_s": seconds,
                "path": str(destination.relative_to(relative_to)),
                "mime_type": "image/jpeg",
            }
        )
    return frames


def _find_ffmpeg() -> str:
    executable = shutil.which("ffmpeg")
    if executable is not None:
        return executable
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError) as exc:
        raise RuntimeError(
            "ffmpeg is required to extract memory reference frames; reinstall "
            "the nav_planner environment to install imageio-ffmpeg"
        ) from exc


def _format_timestamp(seconds: float) -> str:
    minutes, remainder = divmod(seconds, 60.0)
    hours, minutes = divmod(int(minutes), 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{remainder:06.3f}"
    return f"{minutes:d}:{remainder:06.3f}"
