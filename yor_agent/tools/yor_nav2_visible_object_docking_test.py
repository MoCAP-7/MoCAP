#!/usr/bin/env python3
"""Supervised no-LLM test for the production Nav2 docking backend."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np


YOR_AGENT_ROOT = Path(__file__).resolve().parents[1]
YOR_ROOT = YOR_AGENT_ROOT.parent
for source_root in reversed((YOR_AGENT_ROOT / "src", YOR_ROOT / "navdp")):
    source = str(source_root)
    if source_root.is_dir():
        if source in sys.path:
            sys.path.remove(source)
        sys.path.insert(0, source)

from yor_agent.exceptions import PrimitiveFailed
from yor_agent.launch import build_environment, load_config
from yor_agent.primitive_config import primitive_settings
from yor_agent.primitives.registry import PrimitiveRegistry
from yor_agent.primitives.visible_object_navigation import (
    register_visible_object_navigation_primitives,
)
from yor_agent.robot.nav2_visible_object_navigation import (
    Nav2VisibleObjectDockingController,
)


DEFAULT_CONFIG = YOR_AGENT_ROOT / "configs" / "mobile_manipulation.yaml"


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _print_progress(event: dict[str, Any]) -> None:
    """Print one stable, machine-readable line for operators and supervisors."""

    feedback = dict(event.get("feedback") or {})
    status = {
        key: event[key]
        for key in (
            "primitive",
            "target",
            "navigation_id",
            "state",
            "terminal",
            "success",
            "reason",
            "elapsed_s",
            "nav2_terminal",
        )
        if key in event
    }
    for key in (
        "distance_remaining_m",
        "estimated_time_remaining_s",
        "navigation_time_s",
        "number_of_recoveries",
        "current_pose_xy_yaw",
    ):
        if key in feedback:
            status[key] = feedback[key]
    print(
        "NAV2_STATUS " + json.dumps(status, default=_json_default),
        flush=True,
    )


def _preview(
    controller: Nav2VisibleObjectDockingController, object_name: str
) -> dict[str, Any]:
    """Compute the production semantic goal without invoking Nav2 motion."""

    rgb, depth, intrinsics = controller._perception._camera_input()
    pose = controller._perception._current_pose()
    target = controller._perception._detect_target(
        object_name, rgb, depth, intrinsics
    )
    report: dict[str, Any] = {
        "mode": "no-motion-semantic-goal-preview",
        "success": True,
        "object_name": object_name,
        "current_pose_xy_yaw": pose.tolist(),
        "target": target.metrics(),
        "docking_distance_m": controller.config.docking_distance_m,
        "robot_radius_m": controller.config.robot_radius_m,
        "base_to_camera_forward_left_m": [
            controller.config.base_to_camera_forward_m,
            controller.config.base_to_camera_left_m,
        ],
        "ground_plane": dict(controller._perception.last_ground_plane_debug),
    }
    if target.distance_m <= controller.config.docking_distance_m:
        report["reason"] = "already_within_docking_distance"
        report["nav2_goal_xy_yaw"] = None
        return report

    target_x, target_y = controller._perception._target_world_xy(target, pose)
    dx = target_x - float(pose[0])
    dy = target_y - float(pose[1])
    distance = math.hypot(dx, dy)
    if distance <= 1e-6:
        raise RuntimeError("semantic target produced a degenerate goal")
    report["reason"] = "goal_ready"
    report["target_world_xy_m"] = [target_x, target_y]
    camera_goal_x = target_x - controller.config.docking_distance_m * dx / distance
    camera_goal_y = target_y - controller.config.docking_distance_m * dy / distance
    goal_yaw = math.atan2(dy, dx)
    report["desired_camera_goal_xy_yaw"] = [
        camera_goal_x,
        camera_goal_y,
        goal_yaw,
    ]
    report["nav2_goal_xy_yaw"] = list(
        controller._base_goal_from_camera_goal(
            camera_goal_x, camera_goal_y, goal_yaw
        )
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--object-name", default="blue trash bin")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="send the production NavigateToPose goal after exact confirmation",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    object_name = args.object_name.strip()
    if not object_name:
        raise SystemExit("--object-name must be non-empty")

    config = load_config(args.config.expanduser().resolve())
    settings = primitive_settings(
        config["primitive_config"], "dock_to_visible_object"
    )
    if settings is None or str(settings.get("backend", "")).lower() != "nav2":
        raise SystemExit("dock_to_visible_object must configure backend: nav2")

    environment = build_environment(config)
    controller = Nav2VisibleObjectDockingController(
        environment, docking_config=settings
    )
    try:
        status = environment.base_status()
        if status.get("estop_latched", False):
            raise RuntimeError("base emergency stop is latched")

        preview = _preview(controller, object_name)
        print(json.dumps(preview, indent=2, default=_json_default), flush=True)
        if not args.execute:
            print("NO-MOTION PREVIEW SUCCESS: no Nav2 goal was sent.", flush=True)
            return 0
        if preview["nav2_goal_xy_yaw"] is None:
            print("Target is already within docking distance; no motion needed.")
            return 0

        phrase = f"NAV2 DOCK TO {object_name.upper()}"
        print(
            "Clear a 0.60 m radius around the complete route, raise the wheels "
            "for the first run, and keep the physical emergency stop in hand.",
            flush=True,
        )
        typed = input(f"To enable real base motion, type {phrase!r}: ").strip()
        if typed != phrase:
            print("Confirmation mismatch; no Nav2 goal was sent.")
            return 2

        registry = PrimitiveRegistry()
        register_visible_object_navigation_primitives(
            registry,
            environment,
            docking_config=settings,
            progress_callback=_print_progress,
        )
        try:
            result = registry.functions()["dock_to_visible_object"](object_name)
        except PrimitiveFailed as exc:
            result = exc.result
        except KeyboardInterrupt:
            return 130
        print(json.dumps(result, indent=2, default=_json_default), flush=True)
        return 0 if result.get("success", False) else 1
    finally:
        environment.safe_shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
