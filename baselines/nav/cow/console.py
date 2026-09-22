"""Operator console for a CoW episode: live progress lines."""

from __future__ import annotations

import math
import sys
import time
from typing import Any, Callable, Mapping

# The console warns after this many actions in a row that did not complete, and
# again after every further run of the same length.
INCOMPLETE_ACTION_WARNING = 5


def stderr_log(message: str) -> None:
    """Print one timestamped progress line to stderr; stdout keeps the result."""

    print(f"[cow {time.strftime('%H:%M:%S')}] {message}", file=sys.stderr, flush=True)


class EpisodeNarrator:
    """One readable console line per trace event."""

    def __init__(self, log: Callable[[str], None] = stderr_log) -> None:
        self._log = log
        self._mode: str | None = None
        self._incomplete = 0

    def say(self, message: str) -> None:
        self._log(message)

    def event(self, record: Mapping[str, Any]) -> None:
        handler = getattr(self, f"_{record.get('type')}", None)
        if handler is not None:
            handler(record)

    def _episode_start(self, record: Mapping[str, Any]) -> None:
        self._log(f"episode start: goal {record.get('goal')!r}, camera pose {_pose_text(record.get('pose'))}")

    def _pose_invalid_frame(self, record: Mapping[str, Any]) -> None:
        attempt = int(record.get("attempt") or 0) + 1
        self._log(f"ZED tracking is not OK; waiting for a valid pose (frame {attempt})")

    def _decision(self, record: Mapping[str, Any]) -> None:
        mode = str(record.get("mode"))
        if self._mode is not None and mode != self._mode:
            self._log(f"CoW mode {self._mode} -> {mode}")
        self._mode = mode
        if record.get("map_reset_this_step"):
            self._log("CoW found no plan, reset its map and spins again")
        score = float(record.get("attention_max") or 0.0)
        target = f"target pixel {score:.2f}" if score > 0.0 else "no target pixel"
        in_map = "yes" if record.get("roi_exists") else "no"
        failed = ", previous action judged failed" if record.get("predicted_failed_previous_action") else ""
        self._log(
            f"step {int(record.get('step') or 0):3d} {float(record.get('elapsed_s') or 0.0):6.1f}s "
            f"{mode:<7} -> {str(record.get('action')):<11} | {target}, target in map {in_map}, "
            f"frontiers {record.get('exploration_targets')}, decide {float(record.get('decide_s') or 0.0):.2f}s"
            f"{failed}"
        )

    def _primitive(self, record: Mapping[str, Any]) -> None:
        result = record.get("result") or {}
        name = result.get("primitive") or record.get("action")
        motion = _motion_text(result)
        if result.get("success"):
            self._incomplete = 0
            self._log(f"         {name} done{motion}")
            return
        self._incomplete += 1
        status = result.get("status") or "failed"
        self._log(f"         {name} {status} ({_reason_code(result.get('reason'))}){motion}")
        if self._incomplete % INCOMPLETE_ACTION_WARNING == 0:
            self._log(
                f"warning: {self._incomplete} actions in a row did not complete; "
                "Ctrl-C ends the episode and keeps its record"
            )

    def _error(self, record: Mapping[str, Any]) -> None:
        self._log(f"error: {record.get('error')}")

    def _map_image_error(self, record: Mapping[str, Any]) -> None:
        self._log(f"final map image not saved: {record.get('error')}")

    def _episode_end(self, record: Mapping[str, Any]) -> None:
        stop = record.get("stop") or {}
        base = "base stop confirmed" if stop.get("success") else f"base stop NOT confirmed ({stop.get('reason')})"
        self._log(
            f"episode end: {record.get('termination')} after {record.get('steps')} steps, "
            f"{float(record.get('elapsed_s') or 0.0):.1f} s; {base}"
        )


def _pose_text(pose: Any) -> str:
    values = list(pose or [])[:3]
    if len(values) < 3 or any(value is None for value in values):
        return "unavailable"
    x, y, yaw = (float(value) for value in values)
    return f"({x:+.2f}, {y:+.2f}, {math.degrees(yaw):+.0f} deg)"


def _motion_text(result: Mapping[str, Any]) -> str:
    parts = []
    start = list(result.get("start_pose_xy_yaw") or [])[:3]
    final = list(result.get("final_pose_xy_yaw") or [])[:3]
    if len(start) == 3 and len(final) == 3 and None not in start + final:
        moved = math.hypot(final[0] - start[0], final[1] - start[1])
        turned = math.degrees(math.atan2(math.sin(final[2] - start[2]), math.cos(final[2] - start[2])))
        turned = 0.0 if abs(turned) < 0.05 else turned
        parts.append(f"camera moved {moved:.3f} m, turned {turned:+.1f} deg")
    if result.get("elapsed_s") is not None:
        parts.append(f"in {float(result['elapsed_s']):.1f} s")
    return ": " + " ".join(parts) if parts else ""


def _reason_code(reason: Any) -> str:
    return str(reason or "unknown").split(":", 1)[0].strip()
