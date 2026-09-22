"""Navigation primitives exposed to generated policies.

This module owns the public navigation names, units, argument validation, and
docstrings (moved from ``YOR/Agent/agents_yor/api.py``). Motion control itself
stays in ``robot/navigation_controller.py``; nothing here duplicates it.

Effectful calls also own *primitive-level* execution safety: when the controller
reports ``success=False`` the wrapper requests a stop and raises
:class:`PrimitiveFailed`, so later motion in the same generated program is not
executed. That is not a task verifier -- it makes no claim about the goal.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from ..exceptions import PrimitiveFailed
from ..primitives.registry import PrimitiveRegistry


def register_navigation_primitives(
    registry: PrimitiveRegistry,
    env: Any,
    *,
    primitive_defaults: Mapping[str, Mapping[str, Any]] | None = None,
) -> None:
    """Register the v1 navigation vocabulary against one ``YorEnvironment``."""

    configured = primitive_defaults or {}
    registry.register("observe", _make_observe(env))
    registry.register("stop", _make_stop(env))
    registry.register(
        "turn_relative", _make_turn_relative(env, configured.get("turn_relative"))
    )
    registry.register(
        "drive_straight", _make_drive_straight(env, configured.get("drive_straight"))
    )
    registry.register(
        "drive_lateral", _make_drive_lateral(env, configured.get("drive_lateral"))
    )


def _make_observe(env: Any):
    def observe() -> dict[str, Any]:
        """Capture and return the current unified observation.

        Call this between motions when the policy needs to branch on what the
        robot can currently see or where it currently is.

        Returns:
            Dictionary with ``robot0_robotview`` (synchronized ``rgb`` and
            ``depth`` arrays plus ``timestamp_ns``), ``base`` (pose
            ``[x_m, y_m, yaw_rad]``, last velocity, lease and e-stop state),
            and ``left_arm`` / ``right_arm`` / ``arms`` / ``lift`` slots.
        """

        return env.observe()

    return observe


def _make_stop(env: Any):
    def stop() -> dict[str, Any]:
        """Stop the mobile base and confirm the Pi reports a zero command.

        This is a normal stop; it does not clear or change the latched
        emergency-stop state.

        Returns:
            Result dictionary with ``success``, ``reason``, and timing. If the
            stop cannot be confirmed the policy is interrupted immediately.
        """

        result = env.controller.stop()
        _require_success(env, "stop", result, request_stop=False)
        return result

    return stop


def _make_turn_relative(
    env: Any, defaults: Mapping[str, Any] | None = None
):
    defaults = defaults or {}
    default_max_yaw_deg_s = defaults.get("max_yaw_deg_s")
    default_timeout_s = defaults.get("timeout_s")

    def turn_relative(
        angle_deg: float,
        *,
        max_yaw_deg_s: float | None = default_max_yaw_deg_s,
        timeout_s: float | None = default_timeout_s,
    ) -> dict[str, Any]:
        """Turn the base in place by a relative angle in degrees.

        Positive angles turn left/counter-clockwise, negative angles turn
        right/clockwise. One call is limited to 180 degrees. The controller
        stops on stale or invalid localization and always sends a final zero.

        Args:
            angle_deg: Relative turn in degrees, in ``[-180, 180]``. A tiny
                angle is treated as already-at-target.
            max_yaw_deg_s: Optional positive yaw-rate cap in degrees/second.
                The Pi's lower advertised hardware limit always wins.
            timeout_s: Optional positive timeout, at most 45 seconds.

        Returns:
            Result dictionary with ``success``, ``reason``, final pose, elapsed
            time, and error metrics. Pose yaw stays in radians because it
            follows the ZED observation schema. On failure the base is stopped
            and the rest of this policy is skipped.
        """

        angle_deg = _finite_float("angle_deg", angle_deg)
        if abs(angle_deg) > 180.0:
            raise ValueError("abs(angle_deg) must be <= 180")
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
            math.radians(angle_deg),
            max_yaw_rad_s=max_yaw_rad_s,
            timeout_s=timeout_s,
        )
        metrics = result.setdefault("metrics", {})
        metrics["requested_angle_deg"] = angle_deg
        if "target_yaw_rad" in metrics:
            metrics["target_yaw_deg"] = math.degrees(metrics["target_yaw_rad"])
        if "yaw_error_rad" in metrics:
            metrics["yaw_error_deg"] = math.degrees(metrics["yaw_error_rad"])
        _require_success(env, "turn_relative", result)
        return result

    return turn_relative


def _make_drive_straight(
    env: Any, defaults: Mapping[str, Any] | None = None
):
    defaults = defaults or {}
    default_max_speed_mps = defaults.get("max_speed_mps")
    default_timeout_s = defaults.get("timeout_s")

    def drive_straight(
        distance_m: float,
        *,
        max_speed_mps: float | None = default_max_speed_mps,
        timeout_s: float | None = default_timeout_s,
    ) -> dict[str, Any]:
        """Drive a signed relative distance using ZED pose feedback.

        Positive distance drives forward. Every velocity command is checked
        against ZED depth across the full robot footprint (chassis and arms in
        the travel pose), including obstacles seen earlier in the same call
        that have since dropped below the camera view; the base stops with
        ``obstacle_too_close`` before the swept footprint would touch them,
        and fails closed when the area ahead has no valid depth. The camera
        looks down from the mast, so an object lower than about 0.7 m that is
        already within the gripper reach, or lower than about 0.45 m within
        half a metre of the camera, is invisible unless it was seen earlier in
        the same call; a new call starts without that memory. Negative
        distance reverses at a lower default speed and a shorter distance
        limit; the robot has no rear-facing clearance sensor, so reverse
        motion is not checked against anything behind the robot.

        Args:
            distance_m: Signed distance in meters. Positive is forward,
                negative is reverse. Magnitude must exceed 0.025 m; forward and
                reverse limits come from the navigation configuration.
            max_speed_mps: Optional positive speed-magnitude cap. The Pi's
                lower advertised hardware limit always wins; the command sign
                is derived from ``distance_m``.
            timeout_s: Optional positive timeout, at most 45 seconds.

        Returns:
            Result dictionary with ``success``, ``reason``, final pose,
            progress, cross-track error, and front clearance. On failure the
            base is stopped and the rest of this policy is skipped.
        """

        distance_m = _finite_float("distance_m", distance_m)
        if max_speed_mps is not None:
            max_speed_mps = _finite_float("max_speed_mps", max_speed_mps)
            if max_speed_mps <= 0:
                raise ValueError("max_speed_mps must be positive")

        result = env.controller.drive_straight(
            distance_m, max_speed_mps=max_speed_mps, timeout_s=timeout_s
        )
        _require_success(env, "drive_straight", result)
        return result

    return drive_straight


def _make_drive_lateral(
    env: Any, defaults: Mapping[str, Any] | None = None
):
    defaults = defaults or {}
    default_max_speed_mps = defaults.get("max_speed_mps")
    default_timeout_s = defaults.get("timeout_s")

    def drive_lateral(
        distance_m: float,
        *,
        max_speed_mps: float | None = default_max_speed_mps,
        timeout_s: float | None = default_timeout_s,
    ) -> dict[str, Any]:
        """Drive a signed lateral distance using ZED pose feedback.

        Positive distance moves robot-left and negative distance moves
        robot-right while holding the current heading. Distance, speed, and
        timeout limits match ``drive_straight``. The robot has no side-facing
        clearance sensor, so neither direction checks for obstacles beside the
        robot. Use only short steps when the intended side path is visibly and
        reliably clear, and re-observe after every step.

        Args:
            distance_m: Signed lateral distance in meters. Positive is left and
                negative is right. Magnitude must exceed 0.025 m and cannot
                exceed the configured forward distance limit.
            max_speed_mps: Optional positive speed-magnitude cap. The Pi's
                lower advertised hardware limit always wins.
            timeout_s: Optional positive timeout, at most 45 seconds.

        Returns:
            Result dictionary with ``success``, ``reason``, final pose,
            lateral progress, forward cross-track error, and an explicit false
            ``lateral_clearance_checked`` metric. On failure the base is
            stopped and the rest of this policy is skipped.
        """

        distance_m = _finite_float("distance_m", distance_m)
        if max_speed_mps is not None:
            max_speed_mps = _finite_float("max_speed_mps", max_speed_mps)
            if max_speed_mps <= 0:
                raise ValueError("max_speed_mps must be positive")

        result = env.controller.drive_lateral(
            distance_m, max_speed_mps=max_speed_mps, timeout_s=timeout_s
        )
        _require_success(env, "drive_lateral", result)
        return result

    return drive_lateral


def _require_success(
    env: Any,
    primitive: str,
    result: dict[str, Any],
    *,
    request_stop: bool = True,
) -> None:
    """Interrupt the current policy when an effectful primitive failed."""

    if result.get("success", False):
        return
    if request_stop:
        try:
            result["stop_after_failure"] = env.controller.stop()
        except Exception as exc:  # noqa: BLE001 - reported, never masked
            result["stop_after_failure"] = {
                "success": False,
                "reason": f"{type(exc).__name__}:{exc}",
            }
    raise PrimitiveFailed(primitive, result)


def _finite_float(name: str, value: Any) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value
