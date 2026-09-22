"""Execute CoW's discrete actions on YOR with the production navigation primitives."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any, Mapping


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def configure_import_paths() -> None:
    """Make yor_agent and the ZED stream client importable from a repository checkout."""

    for path in (REPOSITORY_ROOT, REPOSITORY_ROOT / "yor_agent" / "src", REPOSITORY_ROOT / "navdp"):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)


def build_yor_environment(task_config: str | Path, task_id: str) -> tuple[Any, dict[str, Any]]:
    """Build the same ``YorEnvironment`` the YOR agent uses for ``task_id``."""

    configure_import_paths()
    from yor_agent.launch import build_environment, load_config

    resolved = load_config(Path(task_config).expanduser(), task_id=task_id)
    return build_environment(resolved), resolved


def camera_geometry_from_yor(resolved: Mapping[str, Any]) -> tuple[float, tuple[float, float, float]]:
    """Calibrated camera height and gravity direction from the YOR task configuration."""

    try:
        settings = resolved["primitive_config"]["primitives"]["dock_to_visible_object"]["settings"]
        height = float(settings["ground_camera_height_m"])
        down = tuple(float(value) for value in settings["ground_down_camera_xyz"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "the YOR task configuration does not define the camera ground geometry; "
            "set camera.camera_height_m and camera.down_camera_xyz"
        ) from exc
    if len(down) != 3:
        raise ValueError("ground_down_camera_xyz must have three values")
    return height, (down[0], down[1], down[2])


# Reasons with which drive_straight's clearance gate refuses or cuts short a
# motion: the swept footprint would enter an occupied cell, or the space it
# would enter returned too little depth to be known free.
SAFETY_STOP_REASONS = frozenset(
    {"obstacle_too_close", "depth_unknown_in_sweep", "depth_mostly_invalid", "depth_insufficient_valid_pixels"}
)


def safety_stop_reason(result: Any) -> str | None:
    """The clearance-gate reason when YOR stopped a motion for safety, otherwise None.

    Other failures, such as a turn that timed out, are execution problems
    rather than a would-be collision and return None.
    """

    if result.get("success"):
        return None
    code = str(result.get("reason") or "").strip()
    if code.startswith("RuntimeError:"):
        code = code[len("RuntimeError:"):]
    code = code.split(";", 1)[0].split(":", 1)[0].strip()
    return code if code in SAFETY_STOP_REASONS else None


class CowActuator:
    """Map CoW's action strings to closed-loop YOR primitives.

    RotateLeft/RotateRight turn by CoW's rotation step about the base and
    MoveAhead drives CoW's forward step. ``drive_straight`` keeps its swept
    forward clearance check, so a blocked or partial move returns
    ``success: false`` with a reason in ``SAFETY_STOP_REASONS``.
    ``turn_relative`` closes the loop on yaw only and never refuses a turn for
    an obstacle; whatever the arms sweep while turning is left to the operator.
    """

    def __init__(self, controller: Any, *, rotation_deg: float, forward_m: float) -> None:
        self.controller = controller
        self.rotation_rad = math.radians(rotation_deg)
        self.forward_m = forward_m

    def execute(self, action: str) -> dict[str, Any]:
        if action == "RotateLeft":
            return dict(self.controller.turn_relative(self.rotation_rad))
        if action == "RotateRight":
            return dict(self.controller.turn_relative(-self.rotation_rad))
        if action == "MoveAhead":
            return dict(self.controller.drive_straight(self.forward_m))
        raise ValueError(f"CoW action {action!r} has no YOR primitive")
