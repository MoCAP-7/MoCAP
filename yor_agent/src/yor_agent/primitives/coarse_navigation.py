"""Coarse CaP-X-style navigation primitives for the coarse navigation baseline.

CaP-X's mobile manipulation setup moves the base only in fixed discrete steps
(one metre forward, 45-degree turns) plus one relative planar position move,
and leaves exploration and the approach to the loops and conditionals of the
generated program. These primitives reproduce that vocabulary on YOR over the
same controller calls as the fine primitives in ``navigation.py``, with their
safety checks: ``go_forward`` and the forward part of ``goto_planar_position``
stop before the swept footprint meets an obstacle the camera has seen, and a
failed motion stops the base and skips the rest of the program.

The launcher registers them only where a primitive configuration exposes them
(``configs/capx_coarse_navigation.yaml``); the shipped configuration hides them.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from .navigation import _finite_float, _require_success
from .registry import PrimitiveRegistry

COARSE_NAVIGATION_PRIMITIVES = (
    "go_forward",
    "turn_left_45_degrees",
    "turn_right_45_degrees",
    "goto_planar_position",
    "say_something",
)
GO_FORWARD_DISTANCE_M = 1.0
TURN_ANGLE_DEG = 45.0
#: Sideways speed of ``goto_planar_position`` as a fraction of its forward
#: speed; the sideways part of the motion has no side-facing clearance sensor.
LATERAL_SPEED_RATIO = 0.8


def register_coarse_navigation_primitives(
    registry: PrimitiveRegistry,
    env: Any,
    *,
    primitive_defaults: Mapping[str, Mapping[str, Any]] | None = None,
) -> None:
    """Register the coarse navigation vocabulary against one ``YorEnvironment``."""

    configured = primitive_defaults or {}
    registry.register("go_forward", _make_go_forward(env, configured.get("go_forward")))
    registry.register(
        "turn_left_45_degrees",
        _make_turn_left_45_degrees(env, configured.get("turn_left_45_degrees")),
    )
    registry.register(
        "turn_right_45_degrees",
        _make_turn_right_45_degrees(env, configured.get("turn_right_45_degrees")),
    )
    registry.register(
        "goto_planar_position",
        _make_goto_planar_position(env, configured.get("goto_planar_position")),
    )
    registry.register("say_something", _make_say_something())


def _make_go_forward(env: Any, defaults: Mapping[str, Any] | None = None):
    defaults = defaults or {}
    default_max_speed_mps = defaults.get("max_speed_mps")
    default_timeout_s = defaults.get("timeout_s")

    def go_forward(
        *,
        max_speed_mps: float | None = default_max_speed_mps,
        timeout_s: float | None = default_timeout_s,
    ) -> dict[str, Any]:
        """Drive the base forward by exactly 1 meter using ZED pose feedback.

        Every velocity command is checked against ZED depth across the full
        robot footprint (chassis and arms in the travel pose), including
        obstacles seen earlier in the same call that have since dropped below
        the camera view; the base stops with ``obstacle_too_close`` before the
        swept footprint would touch them, and fails closed when the area ahead
        has no valid depth. The camera looks down from the mast, so an object
        lower than about 0.7 m that is already within the gripper reach, or
        lower than about 0.45 m within half a metre of the camera, is
        invisible unless it was seen earlier in the same call; a new call
        starts without that memory.

        Args:
            max_speed_mps: Optional positive speed cap. The Pi's lower
                advertised hardware limit always wins.
            timeout_s: Optional positive timeout, at most 45 seconds.

        Returns:
            Result dictionary with ``success``, ``reason``, final pose,
            progress, and front clearance. When an obstacle stops the base
            short of 1 meter, or on any other failure, the base is stopped and
            the rest of this policy is skipped.
        """

        max_speed_mps = _positive_or_none("max_speed_mps", max_speed_mps)
        result = env.controller.drive_straight(
            GO_FORWARD_DISTANCE_M, max_speed_mps=max_speed_mps, timeout_s=timeout_s
        )
        _require_success(env, "go_forward", result)
        return result

    return go_forward


def _make_turn_left_45_degrees(env: Any, defaults: Mapping[str, Any] | None = None):
    defaults = defaults or {}
    default_max_yaw_deg_s = defaults.get("max_yaw_deg_s")
    default_timeout_s = defaults.get("timeout_s")

    def turn_left_45_degrees(
        *,
        max_yaw_deg_s: float | None = default_max_yaw_deg_s,
        timeout_s: float | None = default_timeout_s,
    ) -> dict[str, Any]:
        """Turn the base in place 45 degrees to the left (counter-clockwise).

        The turn has no obstacle check: the whole 0.87 m span of the robot
        sweeps around its center while it turns.

        Args:
            max_yaw_deg_s: Optional positive yaw-rate cap in degrees/second.
                The Pi's lower advertised hardware limit always wins.
            timeout_s: Optional positive timeout, at most 45 seconds.

        Returns:
            Result dictionary with ``success``, ``reason``, final pose, elapsed
            time, and error metrics. On failure the base is stopped and the
            rest of this policy is skipped.
        """

        return _turn(env, "turn_left_45_degrees", TURN_ANGLE_DEG, max_yaw_deg_s, timeout_s)

    return turn_left_45_degrees


def _make_turn_right_45_degrees(env: Any, defaults: Mapping[str, Any] | None = None):
    defaults = defaults or {}
    default_max_yaw_deg_s = defaults.get("max_yaw_deg_s")
    default_timeout_s = defaults.get("timeout_s")

    def turn_right_45_degrees(
        *,
        max_yaw_deg_s: float | None = default_max_yaw_deg_s,
        timeout_s: float | None = default_timeout_s,
    ) -> dict[str, Any]:
        """Turn the base in place 45 degrees to the right (clockwise).

        The turn has no obstacle check: the whole 0.87 m span of the robot
        sweeps around its center while it turns.

        Args:
            max_yaw_deg_s: Optional positive yaw-rate cap in degrees/second.
                The Pi's lower advertised hardware limit always wins.
            timeout_s: Optional positive timeout, at most 45 seconds.

        Returns:
            Result dictionary with ``success``, ``reason``, final pose, elapsed
            time, and error metrics. On failure the base is stopped and the
            rest of this policy is skipped.
        """

        return _turn(env, "turn_right_45_degrees", -TURN_ANGLE_DEG, max_yaw_deg_s, timeout_s)

    return turn_right_45_degrees


def _turn(
    env: Any,
    primitive: str,
    angle_deg: float,
    max_yaw_deg_s: float | None,
    timeout_s: float | None,
) -> dict[str, Any]:
    max_yaw_rad_s = None
    if max_yaw_deg_s is not None:
        max_yaw_deg_s = _finite_float("max_yaw_deg_s", max_yaw_deg_s)
        if max_yaw_deg_s <= 0:
            raise ValueError("max_yaw_deg_s must be positive")
        minimum_rad_s = getattr(env.controller.config, "min_yaw_rad_s", None)
        if minimum_rad_s is not None:
            minimum_deg_s = math.degrees(float(minimum_rad_s))
            if max_yaw_deg_s < minimum_deg_s:
                raise ValueError(f"max_yaw_deg_s must be >= {minimum_deg_s:.3f}")
        max_yaw_rad_s = math.radians(max_yaw_deg_s)

    result = env.controller.turn_relative(
        math.radians(angle_deg), max_yaw_rad_s=max_yaw_rad_s, timeout_s=timeout_s
    )
    metrics = result.setdefault("metrics", {})
    metrics["requested_angle_deg"] = angle_deg
    if "yaw_error_rad" in metrics:
        metrics["yaw_error_deg"] = math.degrees(metrics["yaw_error_rad"])
    _require_success(env, primitive, result)
    return result


def _make_goto_planar_position(env: Any, defaults: Mapping[str, Any] | None = None):
    defaults = defaults or {}
    default_max_speed_mps = defaults.get("max_speed_mps")
    default_timeout_s = defaults.get("timeout_s")

    def goto_planar_position(
        forward_m: float,
        left_m: float,
        *,
        max_speed_mps: float | None = default_max_speed_mps,
        timeout_s: float | None = default_timeout_s,
    ) -> dict[str, Any]:
        """Move the base to a planar position relative to the robot, keeping its heading.

        ``forward_m`` is measured along the current heading (positive forward)
        and ``left_m`` sideways (positive robot-left). The base translates
        toward the position in one holonomic motion with ZED pose feedback and
        holds its current heading, so no yaw is given. The target must be at
        most 2 meters away and not behind the robot. The forward part of the
        motion is checked against ZED depth across the full robot footprint
        and stops with ``obstacle_too_close``; the robot has no side-facing
        clearance sensor, so the sideways part is not checked for obstacles
        beside the robot.

        Args:
            forward_m: Forward offset in meters, at least 0.
            left_m: Sideways offset in meters, positive left.
            max_speed_mps: Optional positive translation speed cap. The Pi's
                lower advertised hardware limit always wins.
            timeout_s: Optional positive timeout, at most 45 seconds; by
                default it scales with the distance.

        Returns:
            Result dictionary with ``success``, ``reason``, final pose, and the
            remaining position error. On failure the base is stopped and the
            rest of this policy is skipped.
        """

        forward_m = _finite_float("forward_m", forward_m)
        left_m = _finite_float("left_m", left_m)
        config = env.controller.config
        if forward_m < 0.0:
            raise ValueError(
                "forward_m must be >= 0; turn toward a position behind the robot first"
            )
        distance = math.hypot(forward_m, left_m)
        if distance > config.max_distance_m:
            raise ValueError(
                f"the position must be within {config.max_distance_m:.2f} m of the robot"
            )
        speed = _positive_or_none("max_speed_mps", max_speed_mps)
        if speed is None:
            speed = float(config.default_linear_mps)
        if timeout_s is None:
            timeout_s = min(
                float(config.max_duration_s),
                distance / (speed * LATERAL_SPEED_RATIO) * float(config.auto_timeout_scale)
                + float(config.auto_timeout_overhead_s),
            )
        result = env.controller.move_planar_relative(
            forward_m,
            left_m,
            0.0,
            max_linear_mps=speed,
            max_lateral_mps=speed * LATERAL_SPEED_RATIO,
            max_yaw_rad_s=float(config.default_yaw_rad_s),
            position_tolerance_m=float(config.distance_tolerance_m),
            yaw_tolerance_rad=float(config.yaw_tolerance_rad),
            timeout_s=_finite_float("timeout_s", timeout_s),
            # Only a small overshoot correction can reverse: the target itself
            # is never behind the robot.
            allow_reverse=True,
        )
        metrics = result.setdefault("metrics", {})
        metrics["requested_forward_m"] = forward_m
        metrics["requested_left_m"] = left_m
        _require_success(env, "goto_planar_position", result)
        return result

    return goto_planar_position


def _make_say_something():
    def say_something(text: str) -> dict[str, Any]:
        """Tell the supervising operator what you intend to do.

        The message is printed to the run log and returned with the rest of
        the program's output on the next turn. It does not move the robot.

        Args:
            text: A short, non-empty message for the operator.

        Returns:
            ``{"success": True, "text": text}``.
        """

        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")
        message = text.strip()
        print(f"[say] {message}")
        return {"success": True, "text": message}

    return say_something


def _positive_or_none(name: str, value: Any) -> float | None:
    if value is None:
        return None
    value = _finite_float(name, value)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value
