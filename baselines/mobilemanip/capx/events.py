"""Per-call event log for the Cap-X baselines on YOR.

Cap-X keeps each generated program, its console output and one image per
program. That does not say what a failed motion did: how far the base moved,
what blocked it, or where the robot ended up. Every injected API call is
therefore appended to ``events.jsonl`` in the episode directory with its
arguments, the full result dictionary (also the one carried by a
``PrimitiveFailed``), its duration, the base pose and lease state before and
after the call and, for calls that act on or sense the robot, the camera frame
after the call. Code block boundaries are logged as well, so each call can be
matched to the program that made it.

Logging never changes the episode: a snapshot or write that fails is skipped
and reported once on stderr.
"""

from __future__ import annotations

import json
import math
import sys
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

EVENTS_FILE = "events.jsonl"
FRAMES_DIRECTORY = "event_frames"
#: Calls that neither move nor sense the robot get no pose or frame.
CALLS_WITHOUT_SNAPSHOT = frozenset({"say_something"})
MAX_ITEMS = 64
MAX_TEXT_CHARS = 4000
MAX_DEPTH = 8
FRAME_MAX_AGE_S = 0.75
BASE_STATUS_KEYS = ("lease_active", "lease_remaining_s", "last_velocity", "estop_latched")


@dataclass(frozen=True)
class RobotSnapshot:
    """Base pose, lease state and camera frame at one instant."""

    pose_xy_yaw: list[float] | None
    frame_timestamp_ns: int | None
    rgb: np.ndarray | None
    base: dict[str, Any] | None

    def record(self) -> dict[str, Any]:
        return {
            "pose_xy_yaw": self.pose_xy_yaw,
            "frame_timestamp_ns": self.frame_timestamp_ns,
            "base": jsonable(self.base),
        }


Snapshot = Callable[[], RobotSnapshot]


def yor_snapshot(environment: Any) -> Snapshot | None:
    """Snapshot the production ``YorEnvironment``; ``None`` when there is none."""

    if environment is None or not callable(getattr(environment, "navigation_frame", None)):
        return None

    def snapshot() -> RobotSnapshot:
        pose = None
        timestamp = None
        rgb = None
        try:
            frame = environment.navigation_frame(max_age_s=FRAME_MAX_AGE_S)
        except Exception:  # noqa: BLE001 - a missing frame is recorded as null
            frame = None
        if frame is not None:
            planar = getattr(frame, "planar_pose", None)
            if planar is not None:
                pose = [float(planar.x_m), float(planar.y_m), float(planar.yaw_rad)]
            stamp = getattr(frame, "timestamp_ns", None)
            timestamp = None if stamp is None else int(stamp)
            rgb = getattr(frame, "rgb", None)
        base = None
        base_status = getattr(environment, "base_status", None)
        if callable(base_status):
            try:
                status = base_status()
            except Exception:  # noqa: BLE001 - a missing status is recorded as null
                status = None
            if isinstance(status, Mapping):
                base = {key: status.get(key) for key in BASE_STATUS_KEYS}
        return RobotSnapshot(pose, timestamp, rgb, base)

    return snapshot


class EpisodeEventLog:
    """Append-only ``events.jsonl`` for one episode directory.

    The directory is created on the first write, so an episode that fails
    before its first program leaves nothing behind.
    """

    def __init__(
        self,
        directory: str | Path,
        *,
        snapshot: Snapshot | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.directory = Path(directory)
        self._snapshot = snapshot
        self._clock = clock
        self._origin = clock()
        self._lock = threading.Lock()
        self._sequence = 0
        self._block: int | None = None
        self._blocks = 0
        self._warned = False

    @property
    def path(self) -> Path:
        return self.directory / EVENTS_FILE

    def code_block_started(self, code: str) -> None:
        with self._lock:
            self._blocks += 1
            self._block = self._blocks
        self._write(
            {
                "event": "code_block_started",
                "block": self._block,
                "t_s": self._elapsed(),
                "started_at": _now(),
                "code_chars": len(code),
            }
        )

    def code_block_finished(self, info: Mapping[str, Any]) -> None:
        self._write(
            {
                "event": "code_block_finished",
                "block": self._block,
                "t_s": self._elapsed(),
                "sandbox_rc": jsonable(info.get("sandbox_rc")),
                "stdout": _text(info.get("stdout") or "", tail=True),
                "stderr": _text(info.get("stderr") or "", tail=True),
            }
        )

    def call(
        self,
        name: str,
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
        run: Callable[[], Any],
    ) -> Any:
        """Run one API call and append its record, re-raising what it raised."""

        with self._lock:
            self._sequence += 1
            sequence = self._sequence
        snapshot = None if name in CALLS_WITHOUT_SNAPSHOT else self._snapshot
        record: dict[str, Any] = {
            "event": "api_call",
            "sequence": sequence,
            "block": self._block,
            "name": name,
            "args": jsonable(list(args)),
            "kwargs": jsonable(dict(kwargs)),
            "t_s": self._elapsed(),
            "started_at": _now(),
        }
        before = self._take(snapshot)
        started = self._clock()
        try:
            value = run()
        except BaseException as exc:
            record.update(_failure(exc))
            self._finish(record, started, snapshot, before)
            raise
        record["outcome"] = "ok"
        record["result"] = jsonable(value)
        self._finish(record, started, snapshot, before)
        return value

    def _finish(
        self,
        record: dict[str, Any],
        started: float,
        snapshot: Snapshot | None,
        before: RobotSnapshot | None,
    ) -> None:
        record["elapsed_s"] = round(self._clock() - started, 3)
        after = self._take(snapshot)
        record["before"] = None if before is None else before.record()
        record["after"] = None if after is None else after.record()
        if after is not None and after.rgb is not None:
            record["frame"] = self._save_frame(record["sequence"], record["name"], after.rgb)
        self._write(record)

    def _take(self, snapshot: Snapshot | None) -> RobotSnapshot | None:
        if snapshot is None:
            return None
        try:
            return snapshot()
        except Exception as exc:  # noqa: BLE001 - logging never changes the episode
            self._warn(f"robot snapshot failed: {exc!r}")
            return None

    def _save_frame(self, sequence: int, name: str, rgb: Any) -> str | None:
        try:
            from PIL import Image

            directory = self.directory / FRAMES_DIRECTORY
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / f"{sequence:04d}_{name}.jpg"
            Image.fromarray(np.ascontiguousarray(rgb, dtype=np.uint8)).save(path, quality=85)
        except Exception as exc:  # noqa: BLE001 - logging never changes the episode
            self._warn(f"event frame was not saved: {exc!r}")
            return None
        return f"{FRAMES_DIRECTORY}/{path.name}"

    def _write(self, record: Mapping[str, Any]) -> None:
        try:
            line = json.dumps(record, ensure_ascii=False, allow_nan=False)
            with self._lock:
                self.directory.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as stream:
                    stream.write(line + "\n")
        except Exception as exc:  # noqa: BLE001 - logging never changes the episode
            self._warn(f"event was not written: {exc!r}")

    def _elapsed(self) -> float:
        return round(self._clock() - self._origin, 3)

    def _warn(self, message: str) -> None:
        if self._warned:
            return
        self._warned = True
        print(f"[capx] event log: {message}", file=sys.stderr, flush=True)


def jsonable(value: Any, _depth: int = 0) -> Any:
    """A JSON-safe copy: arrays and long sequences are summarised, NaN is null."""

    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, str):
        return _text(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    if _depth >= MAX_DEPTH:
        return _text(repr(value))
    if isinstance(value, np.ndarray):
        if value.size <= MAX_ITEMS:
            return jsonable(value.tolist(), _depth + 1)
        return {"array_shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, Mapping):
        return {str(key): jsonable(item, _depth + 1) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        items = list(value)
        converted = [jsonable(item, _depth + 1) for item in items[:MAX_ITEMS]]
        if len(items) > MAX_ITEMS:
            converted.append({"omitted_items": len(items) - MAX_ITEMS})
        return converted
    if isinstance(value, BaseException):
        return {"type": type(value).__name__, "message": _text(str(value))}
    return _text(repr(value))


def _failure(exc: BaseException) -> dict[str, Any]:
    error = {"type": type(exc).__name__, "message": _text(str(exc))}
    result = getattr(exc, "result", None)
    if isinstance(result, Mapping):
        return {
            "outcome": "primitive_failed",
            "reason": _text(str(getattr(exc, "reason", result.get("reason")))),
            "result": jsonable(result),
            "error": error,
        }
    if type(exc).__name__ == "EpisodeStopped":
        outcome = "refused_after_stop"
    elif isinstance(exc, Exception):
        outcome = "error"
    else:
        outcome = "interrupted"
    return {"outcome": outcome, "error": error}


def _text(value: str, *, tail: bool = False) -> str:
    text = str(value)
    if len(text) <= MAX_TEXT_CHARS:
        return text
    if tail:
        return "..." + text[-MAX_TEXT_CHARS:]
    return text[:MAX_TEXT_CHARS] + "..."


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="milliseconds")
