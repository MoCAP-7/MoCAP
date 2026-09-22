"""The single physical-YOR boundary owned by ``yor_agent``.

``YorEnvironment`` owns the hardware bridge, the closed-loop navigation
controller, the unified observation, and the lifecycle (reset / observe /
safe shutdown). Navigation and future manipulation are *primitive modules*
registered against this one environment, not separate environments.

Extracted from ``YOR/Agent/agents_yor/low_level.py``. The Gym/render/trial
compatibility of the old ``BaseEnv`` subclass is deliberately not carried over:
no reward, no ``task_completed``, no ``step(action)``, no viser hooks.
"""

from __future__ import annotations

import atexit
from collections.abc import Callable, Mapping
import threading
from typing import Any

import numpy as np

from .robot.contracts import NavigationHardware
from .robot.geometry import validated_transform
from .robot.hardware import YorHardwareBridge
from .robot.navigation_controller import NavigationConfig, NavigationController


class YorEnvironment:
    """Real-robot connection, observation, and lifecycle for the agent loop.

    ``reset()`` is deliberately a non-motion reset: it verifies an idle,
    un-latched base and returns the first synchronized RGB-D/pose observation.
    """

    def __init__(
        self,
        *,
        base_rpc_host: str = "192.168.1.10",
        base_rpc_port: int = 5557,
        zed_host: str = "127.0.0.1",
        zed_port: int = 6000,
        zed_transport: str = "atomic",
        published_color_order: str = "auto",
        arm_rpc_host: str | None = None,
        arm_rpc_port: int = 5558,
        require_arms: bool = False,
        observation_timeout_s: float = 3.0,
        frame_max_age_s: float = 0.30,
        navigation: Mapping[str, Any] | None = None,
        manipulation: Mapping[str, Any] | None = None,
        hardware: NavigationHardware | None = None,
        controller: NavigationController | None = None,
    ) -> None:
        self.observation_timeout_s = float(observation_timeout_s)
        self.frame_max_age_s = float(frame_max_age_s)
        if self.observation_timeout_s <= 0 or self.frame_max_age_s <= 0:
            raise ValueError("observation timeouts must be positive")
        self.navigation_config = dict(navigation or {})
        self.manipulation_config = dict(manipulation or {})
        self._hardware = hardware or YorHardwareBridge(
            base_rpc_host=base_rpc_host,
            base_rpc_port=base_rpc_port,
            zed_host=zed_host,
            zed_port=zed_port,
            zed_transport=zed_transport,
            published_color_order=published_color_order,
            arm_rpc_host=arm_rpc_host,
            arm_rpc_port=arm_rpc_port,
            require_arms=require_arms,
        )
        self.controller = controller or NavigationController(
            self, config=NavigationConfig.from_mapping(self.navigation_config)
        )
        self._latest_frame: Any | None = None
        self._closed = False
        self._motion_stop_callbacks: set[Callable[[], Any]] = set()
        self._motion_stop_callbacks_lock = threading.Lock()
        self._robot_self_filter_attached_objects: dict[str, dict[str, Any]] = {}
        self._robot_self_filter_attachment_lock = threading.Lock()
        # Process exit must remain a safe shutdown boundary even if a caller
        # never reaches the agent's ``finally``.
        atexit.register(self.safe_shutdown)

    # ------------------------------------------------------------------
    # Hardware-facing surface. None of this is registered as a primitive.
    # ------------------------------------------------------------------

    def navigation_frame(self, *, max_age_s: float):
        frame = self._hardware.latest_frame(max_age_s=max_age_s)
        if frame is None:
            raise RuntimeError(
                f"no synchronized ZED RGB-D/pose frame newer than {max_age_s:.3f}s"
            )
        pose = getattr(frame, "planar_pose", None)
        if pose is None or not bool(getattr(pose, "valid", False)):
            raise RuntimeError("ZED planar tracking pose is unavailable or invalid")
        self._latest_frame = frame
        return frame

    def navigation_frame_age_s(self) -> float | None:
        """Arrival age of the newest ZED frame for clearance horizons."""

        getter = getattr(self._hardware, "latest_frame_age_s", None)
        if not callable(getter):
            return None
        try:
            age = getter()
        except Exception:  # noqa: BLE001 - the controller falls back to its bound
            return None
        return None if age is None else float(age)

    def base_status(self) -> dict[str, Any]:
        return self._hardware.get_base_status()

    def submit_base_velocity(self, velocity: list[float]) -> dict[str, Any]:
        """Send one leased velocity command. Controller-only; never a primitive."""

        return self._hardware.submit_base_velocity(velocity)

    @property
    def has_manipulation(self) -> bool:
        """Whether arm hardware is reachable, for a future manipulation provider."""

        get_arm_status = getattr(self._hardware, "get_arm_status", None)
        if not callable(get_arm_status):
            return False
        try:
            return get_arm_status() is not None
        except Exception:  # noqa: BLE001 - absence of arms is not a run failure
            return False

    @property
    def has_visible_object_docking(self) -> bool:
        """Whether ZED camera calibration is configured for SAM3-based docking.

        This only needs the same camera calibration used by manipulation, not
        arm hardware, so it is gated separately from ``has_manipulation``.
        """

        return self.manipulation_config.get("camera_intrinsics") is not None

    def arm_status(self) -> dict[str, Any]:
        status = self._hardware.get_arm_status()
        if not isinstance(status, dict):
            raise RuntimeError("YOR arm RPC status is unavailable")
        return status

    def set_robot_self_filter_attached_object(
        self, arm: int | str, bounds_local: Any
    ) -> None:
        """Register one TCP-local held-object box for navigation filtering."""

        if arm in (0, "0", "left"):
            name = "left"
        elif arm in (1, "1", "right"):
            name = "right"
        else:
            raise ValueError("arm must be 0/'left' or 1/'right'")
        bounds = np.asarray(bounds_local, dtype=np.float64)
        if (
            bounds.shape != (2, 3)
            or not np.all(np.isfinite(bounds))
            or np.any(bounds[1] <= bounds[0])
        ):
            raise ValueError(
                "attached-object bounds must be a positive finite 2x3 box"
            )
        with self._robot_self_filter_attachment_lock:
            self._robot_self_filter_attached_objects[name] = {
                "bounds_local": bounds.tolist()
            }

    def clear_robot_self_filter_attached_object(self, arm: int | str) -> None:
        if arm in (0, "0", "left"):
            name = "left"
        elif arm in (1, "1", "right"):
            name = "right"
        else:
            raise ValueError("arm must be 0/'left' or 1/'right'")
        with self._robot_self_filter_attachment_lock:
            self._robot_self_filter_attached_objects.pop(name, None)

    def robot_self_filter_attached_objects(self) -> dict[str, dict[str, Any]]:
        """Return a detached, JSON-safe snapshot for direct and Nav2 clients."""

        with self._robot_self_filter_attachment_lock:
            return {
                name: {"bounds_local": [list(row) for row in entry["bounds_local"]]}
                for name, entry in self._robot_self_filter_attached_objects.items()
            }

    def move_arm_pose(
        self,
        arm: int | str,
        pose_xyz_rpy: list[float],
        timeout_s: float = 60.0,
    ) -> dict[str, Any]:
        """Send one Pi arm-service pose command. Controller-only; never a primitive."""

        return self._hardware.move_arm_pose(arm, pose_xyz_rpy, timeout_s)

    def plan_arm_poses(
        self, arm: int | str, poses_xyz_rpy: list[list[float]]
    ) -> dict[str, Any]:
        """Ask the Pi Mink backend to solve poses without commanding motion."""

        return self._hardware.plan_arm_poses(arm, poses_xyz_rpy)

    def execute_arm_trajectory(
        self,
        arm: int | str,
        waypoints: list[list[float]],
        timeout_s: float = 60.0,
    ) -> dict[str, Any]:
        """Execute a collision-planner joint path through the guarded Pi RPC."""

        return self._hardware.execute_arm_trajectory(arm, waypoints, timeout_s)

    def set_gripper(
        self,
        arm: int | str,
        opened: bool,
        timeout_s: float = 3.0,
        force_n: float | None = None,
    ) -> dict[str, Any]:
        return self._hardware.set_gripper(arm, opened, timeout_s, force_n)

    def manipulation_calibration(self, arm: int | str) -> dict[str, np.ndarray]:
        """Return the eye-to-hand camera calibration for one Nero arm."""

        if arm in (0, "0", "left"):
            transform_key = "left_arm_from_camera"
        elif arm in (1, "1", "right"):
            transform_key = "right_arm_from_camera"
        else:
            raise ValueError("arm must be 0/'left' or 1/'right'")
        intrinsics = self.manipulation_config.get("camera_intrinsics")
        transform = self.manipulation_config.get(transform_key)
        if intrinsics is None:
            raise RuntimeError("camera_intrinsics calibration is not configured")
        if transform is None:
            raise RuntimeError(f"{transform_key} calibration is not configured")
        matrix = np.asarray(intrinsics, dtype=np.float64)
        if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
            raise ValueError("camera_intrinsics must be a finite 3x3 matrix")
        resolution = self.manipulation_config.get("camera_calibration_resolution")
        if resolution is None:
            raise RuntimeError("camera_calibration_resolution is not configured")
        resolution_array = np.asarray(resolution, dtype=np.int64).reshape(-1)
        if resolution_array.size != 2 or np.any(resolution_array <= 0):
            raise ValueError(
                "camera_calibration_resolution must contain [width, height]"
            )
        return {
            "camera_intrinsics": matrix,
            "arm_from_camera": validated_transform(transform, name=transform_key),
            "camera_calibration_resolution": resolution_array,
        }

    # ------------------------------------------------------------------
    # Agent-facing lifecycle
    # ------------------------------------------------------------------

    def reset(self) -> dict[str, Any]:
        """Verify an idle, un-latched base and return the first observation."""

        status = self.base_status()
        if status.get("estop_latched", False):
            raise RuntimeError("YOR emergency stop is latched")
        if status.get("lease_active", False):
            raise RuntimeError(
                "the base already has an active velocity lease; stop the other controller"
            )
        observation = self.observe()
        if observation["arms"]["estop_latched"]:
            raise RuntimeError("YOR arm emergency stop is latched")
        return observation

    def observe(self) -> dict[str, Any]:
        """Return the unified real-robot observation."""

        frame = self._wait_for_frame()
        return self._observation_from_frame(frame)

    def observe_next(self) -> dict[str, Any]:
        """Wait for the next synchronized frame and return one observation.

        Motion remains stopped while callers use this method.  It exists so
        temporal perception filters cannot accidentally reuse the same recent
        frame through :meth:`observe`'s low-latency cache path.
        """

        frame = self._hardware.next_frame(timeout_s=self.observation_timeout_s)
        self._latest_frame = frame
        return self._observation_from_frame(frame)

    def _observation_from_frame(self, frame: Any) -> dict[str, Any]:
        rgb = np.ascontiguousarray(frame.rgb, dtype=np.uint8)
        depth = np.ascontiguousarray(frame.depth_m, dtype=np.float32)
        status = self.base_status()
        telemetry = status.get("telemetry", {})
        lift_height = (
            telemetry.get("lift_height_m") if isinstance(telemetry, dict) else None
        )
        get_arm_status = getattr(self._hardware, "get_arm_status", None)
        arms = get_arm_status() if callable(get_arm_status) else None
        left_arm = (
            {"available": False}
            if arms is None
            else dict(arms.get("left", {"available": False}))
        )
        right_arm = (
            {"available": False}
            if arms is None
            else dict(arms.get("right", {"available": False}))
        )
        arm_system = {
            "available": arms is not None,
            "estop_latched": False
            if arms is None
            else bool(arms.get("estop_latched", False)),
            "firmware_driver": None if arms is None else arms.get("firmware_driver"),
            "speed_percent": None if arms is None else arms.get("speed_percent"),
            "tcp_offsets_xyz_rpy": None
            if arms is None
            else arms.get("tcp_offsets_xyz_rpy"),
            "cartesian_backend": None
            if arms is None
            else arms.get("cartesian_backend"),
            "end_effector_frame": None
            if arms is None
            else arms.get("end_effector_frame"),
            "ik_max_iterations": None
            if arms is None
            else arms.get("ik_max_iterations"),
            "ik_planning_rpc": None
            if arms is None
            else arms.get("ik_planning_rpc"),
        }
        camera: dict[str, Any] = {
            "images": {"rgb": rgb, "depth": depth[:, :, None]},
            "timestamp_ns": int(frame.timestamp_ns),
        }
        configured_intrinsics = self.manipulation_config.get("camera_intrinsics")
        if configured_intrinsics is not None:
            camera["intrinsics"] = np.asarray(
                configured_intrinsics, dtype=np.float64
            )
        ground_height = getattr(frame, "ground_camera_height_m", None)
        ground_down = getattr(frame, "ground_down_camera_xyz", None)
        ground_timestamp = getattr(frame, "ground_plane_timestamp_ns", None)
        camera["ground_plane"] = {
            "valid": (
                ground_height is not None
                and ground_down is not None
                and ground_timestamp is not None
            ),
            "camera_height_m": ground_height,
            "down_camera_xyz": ground_down,
            "timestamp_ns": ground_timestamp,
        }
        return {
            "robot0_robotview": camera,
            "base": {
                "pose_xy_yaw": self._pose_array(frame),
                "last_velocity": np.asarray(
                    status.get("last_velocity", [np.nan] * 3), dtype=np.float64
                ),
                "lease_active": bool(status.get("lease_active", False)),
                "lease_remaining_s": float(status.get("lease_remaining_s", 0.0)),
                "estop_latched": bool(status.get("estop_latched", False)),
                "telemetry": telemetry,
            },
            "left_arm": left_arm,
            "right_arm": right_arm,
            "arms": arm_system,
            "lift": {
                "available": lift_height is not None,
                "height_m": None if lift_height is None else float(lift_height),
            },
        }

    def safe_shutdown(self) -> None:
        """Request/confirm a stop and close transports. Idempotent."""

        if self._closed:
            return
        self._closed = True
        self._run_motion_stop_callbacks()
        try:
            self.controller.stop()
        except Exception:  # noqa: BLE001 - closing must not raise over a run result
            pass
        try:
            self._hardware.close()
        finally:
            atexit.unregister(self.safe_shutdown)

    def request_stop(self) -> dict[str, Any]:
        """Cancel higher-level motion, then request a zero-velocity stop."""

        if self._closed:
            return {"accepted": True, "already_closed": True}
        callback_errors = self._run_motion_stop_callbacks()
        result = dict(self.controller.request_stop())
        if callback_errors:
            result["higher_level_stop_errors"] = callback_errors
        return result

    def register_motion_stop_callback(self, callback: Callable[[], Any]) -> None:
        """Register cancellation that must run before the base zero command.

        Nav2 uses this hook to cancel its active action and latch the ROS base
        bridge before the existing direct-RPC controller confirms zero speed.
        """

        if not callable(callback):
            raise TypeError("motion stop callback must be callable")
        with self._motion_stop_callbacks_lock:
            self._motion_stop_callbacks.add(callback)

    def unregister_motion_stop_callback(self, callback: Callable[[], Any]) -> None:
        with self._motion_stop_callbacks_lock:
            self._motion_stop_callbacks.discard(callback)

    def _run_motion_stop_callbacks(self) -> list[str]:
        with self._motion_stop_callbacks_lock:
            callbacks = tuple(self._motion_stop_callbacks)
        errors: list[str] = []
        for callback in callbacks:
            try:
                callback()
            except Exception as exc:  # noqa: BLE001 - still issue direct zero
                errors.append(f"{type(exc).__name__}: {exc}")
        return errors

    def latest_camera_preview(self) -> dict[str, Any] | None:
        """Return a read-only recent RGB frame for an operator display.

        This is deliberately not a registered primitive and is not used to
        construct model messages.  The Web UI labels it separately from the
        exact LLM-input snapshot emitted by :class:`DefaultAgent`.
        """

        if self._closed:
            return None
        frame = self._hardware.latest_frame(
            max_age_s=max(self.frame_max_age_s, 0.75)
        )
        if frame is None:
            return None
        return {
            "rgb": np.ascontiguousarray(frame.rgb, dtype=np.uint8),
            "timestamp_ns": int(frame.timestamp_ns),
        }

    # ------------------------------------------------------------------

    def _wait_for_frame(self):
        frame = self._hardware.latest_frame(max_age_s=self.frame_max_age_s)
        if frame is None:
            frame = self._hardware.next_frame(timeout_s=self.observation_timeout_s)
        self._latest_frame = frame
        return frame

    @staticmethod
    def _pose_array(frame: Any) -> np.ndarray:
        pose = getattr(frame, "planar_pose", None)
        if pose is None or not bool(getattr(pose, "valid", False)):
            return np.full(3, np.nan, dtype=np.float64)
        return np.asarray([pose.x_m, pose.y_m, pose.yaw_rad], dtype=np.float64)


def summarize_observation(observation: Mapping[str, Any]) -> dict[str, Any]:
    """Compact, JSON-safe view of an observation for prompts and traces.

    Full RGB-D arrays and telemetry stay local; a generated policy retrieves
    them through ``observe()`` when it actually needs them.
    """

    camera = observation.get("robot0_robotview", {}) or {}
    images = camera.get("images", {}) or {}
    rgb = images.get("rgb")
    depth = images.get("depth")
    base = observation.get("base", {}) or {}
    lift = observation.get("lift", {}) or {}
    arms = observation.get("arms", {}) or {}
    return {
        "timestamp_ns": camera.get("timestamp_ns"),
        "rgb_shape": None if rgb is None else list(np.shape(rgb)),
        "depth_shape": None if depth is None else list(np.shape(depth)),
        "base": {
            "pose_xy_yaw": _finite_list(base.get("pose_xy_yaw")),
            "last_velocity": _finite_list(base.get("last_velocity")),
            "lease_active": bool(base.get("lease_active", False)),
            "lease_remaining_s": float(base.get("lease_remaining_s", 0.0) or 0.0),
            "estop_latched": bool(base.get("estop_latched", False)),
        },
        "lift": {
            "available": bool(lift.get("available", False)),
            "height_m": lift.get("height_m"),
        },
        "arms": {
            "available": bool(arms.get("available", False)),
            "estop_latched": bool(arms.get("estop_latched", False)),
        },
    }


def _finite_list(values: Any) -> list[float | None]:
    if values is None:
        return []
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    return [None if not np.isfinite(value) else round(float(value), 4) for value in array]
