"""Manipulation primitives exposed to generated policies.

This module owns the public manipulation names, units, argument validation,
and docstrings (moved from ``YOR/Agent/agents_yor/api.py``'s
``YorManipulationApi``). Perception and arm-motion sequencing stay in
``robot/manipulation.py``; nothing here duplicates it.

Effectful calls (``goto_pose``, ``goto_grasp_pose``, ``open_gripper``,
``close_gripper``, ``lift_grasped_object``) also own
*primitive-level* execution safety: when the controller reports
``success=False`` the wrapper raises :class:`PrimitiveFailed`, so later motion
in the same generated program is not executed. Unlike the base, there is no
generic "stop" RPC for the arms, so the wrapper does not attempt one; raising
already halts the rest of that turn's program, which is what matters here.
That is not a task verifier -- it makes no claim about the goal.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Literal

import numpy as np

from ..exceptions import PrimitiveFailed
from ..primitives.registry import PrimitiveRegistry
from ..robot.manipulation import ManipulationController


def register_manipulation_primitives(
    registry: PrimitiveRegistry,
    env: Any,
    *,
    primitive_defaults: Mapping[str, Mapping[str, Any]] | None = None,
    primitive_settings: Mapping[str, Mapping[str, Any]] | None = None,
    segment_client_factory: Callable[[], Callable[..., Any]] | None = None,
    grasp_backend_factory: Callable[[dict[str, Any]], Any] | None = None,
    motion_planner_factory: Callable[[dict[str, Any]], Any] | None = None,
    attached_lift_enabled: bool = False,
) -> ManipulationController:
    """Register the v1 manipulation vocabulary against one ``YorEnvironment``.

    ``segment_client_factory`` and ``grasp_backend_factory`` default to the
    real SAM3 HTTP client and the configured grasp backend; tests inject
    fakes here instead.
    """

    configured = primitive_defaults or {}
    settings = primitive_settings or {}
    controller = ManipulationController(
        env,
        segment_client_factory=segment_client_factory,
        grasp_backend_factory=grasp_backend_factory,
        motion_planner_factory=motion_planner_factory,
        attached_lift_enabled=attached_lift_enabled,
    )
    registry.register(
        "get_object_pose",
        _make_get_object_pose(
            controller,
            configured.get("get_object_pose"),
            depth_retry_count=int(
                settings.get("get_object_pose", {}).get("depth_retry_count", 0)
            ),
        ),
    )
    registry.register(
        "sample_grasp_pose",
        _make_sample_grasp_pose(
            controller,
            depth_retry_count=int(
                settings.get("sample_grasp_pose", {}).get("depth_retry_count", 0)
            ),
        ),
    )
    registry.register(
        "goto_pose", _make_goto_pose(controller, configured.get("goto_pose"))
    )
    registry.register(
        "goto_grasp_pose",
        _make_goto_grasp_pose(controller, configured.get("goto_grasp_pose")),
    )
    registry.register(
        "open_gripper",
        _make_set_gripper(
            env,
            controller,
            opened=True,
            primitive="open_gripper",
            defaults=configured.get("open_gripper"),
            simulated=bool(
                settings.get("open_gripper", {}).get("simulated", False)
            ),
            attached_lift_enabled=attached_lift_enabled,
        ),
    )
    registry.register(
        "close_gripper",
        _make_set_gripper(
            env,
            controller,
            opened=False,
            primitive="close_gripper",
            defaults=configured.get("close_gripper"),
            simulated=bool(
                settings.get("close_gripper", {}).get("simulated", False)
            ),
            attached_lift_enabled=attached_lift_enabled,
        ),
    )
    registry.register(
        "lift_grasped_object",
        _make_lift_grasped_object(
            controller, configured.get("lift_grasped_object")
        ),
    )
    return controller


def _required_arm(arm: Literal["left", "right"]) -> str:
    """Validate the explicit string-only arm API."""

    if not isinstance(arm, str) or arm not in {"left", "right"}:
        raise ValueError(
            "arm must be explicitly passed as the string 'left' or 'right'"
        )
    return arm


def _make_get_object_pose(
    controller: ManipulationController,
    defaults: Mapping[str, Any] | None = None,
    *,
    depth_retry_count: int = 0,
):
    defaults = defaults or {}
    default_return_bbox_extent = defaults.get("return_bbox_extent", False)
    default_return_zed_distance = defaults.get("return_zed_distance", False)

    def get_object_pose(
        object_name: str,
        *,
        arm: Literal["left", "right"],
        return_bbox_extent: bool = default_return_bbox_extent,
        return_zed_distance: bool = default_return_zed_distance,
    ) -> (
        tuple[np.ndarray, np.ndarray, np.ndarray | None]
        | tuple[np.ndarray, np.ndarray, np.ndarray | None, float]
    ):
        """Estimate an object's pose in the selected Nero arm-base frame.

        Args:
            object_name: SAM3 prompt: the object's short category noun, one
                or two words such as ``"can"``, not a sentence. A prompt that
                also describes colour, material or position usually finds no
                instance. Use the name that worked for
                ``dock_to_visible_object``.
            arm: Required Nero arm identifier.
            return_bbox_extent: Return PCA-aligned full XYZ extents when true.
            return_zed_distance: Append the metric 3D distance from the ZED
                optical center to the object's median depth point when true.

        Returns:
            ``(position_xyz, quaternion_wxyz, bbox_extent_or_none)``. Object
            orientation is perception-derived and may be less reliable than
            the orientation returned by ``sample_grasp_pose``. With
            ``return_zed_distance=True``, a fourth value ``zed_distance_m`` is
            appended.
        """

        _required_arm(arm)
        return controller.get_object_pose(
            object_name,
            arm,
            return_bbox_extent=return_bbox_extent,
            return_zed_distance=return_zed_distance,
            depth_retry_count=depth_retry_count,
        )

    return get_object_pose


def _make_sample_grasp_pose(
    controller: ManipulationController,
    *,
    depth_retry_count: int = 0,
):
    def sample_grasp_pose(
        object_name: str, *, arm: Literal["left", "right"]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sample a configured-backend grasp in the selected Nero arm-base frame.

        Args:
            object_name: SAM3 prompt: the object's short category noun, one
                or two words such as ``"can"``, not a sentence. Use the name
                that worked for ``get_object_pose`` or
                ``dock_to_visible_object``.
            arm: Required Nero arm identifier.

        Returns:
            ``(position_xyz, quaternion_wxyz)`` for the selected feasible grasp.
            The tuple-shaped API remains compatible with existing callers.
        """

        _required_arm(arm)
        return controller.sample_grasp_pose(
            object_name,
            arm,
            depth_retry_count=depth_retry_count,
        )

    return sample_grasp_pose


def _make_goto_pose(
    controller: ManipulationController, defaults: Mapping[str, Any] | None = None
):
    defaults = defaults or {}
    default_z_approach = defaults.get("z_approach", 0.0)
    default_timeout_s = defaults.get("timeout_s", 60.0)

    def goto_pose(
        position: np.ndarray | Literal["home", "current"],
        quaternion_wxyz: np.ndarray | None = None,
        *,
        arm: Literal["left", "right"],
        z_approach: float = default_z_approach,
        timeout_s: float = default_timeout_s,
        camera_offset_xyz: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Move one Nero TCP to an absolute pose, relative offset, or home.

        Do not use this primitive to calculate or execute a grasp pose. Use it
        for non-grasp Cartesian adjustments, measured-pose-relative retreats,
        or an optional return home.

        Args:
            position: Absolute target XYZ meters; ``"current"`` to move from
                fresh measured TCP feedback by ``camera_offset_xyz``; or
                ``"home"`` for the configured fixed home joint target.
            quaternion_wxyz: Target WXYZ unit quaternion for Cartesian mode.
                Omit it for ``"current"`` and ``"home"`` modes. Relative mode
                preserves the current measured TCP orientation.
            arm: Required Nero arm identifier.
            z_approach: Optional Cartesian approach distance. It must be zero
                in ``"current"`` and ``"home"`` modes.
            timeout_s: Per-command timeout, capped by the Pi service.
            camera_offset_xyz: Required only with ``position="current"``. XYZ
                meters in the leveled ZED view frame: +X horizontal right, +Y
                vertical down, +Z horizontal forward. Camera pitch and the
                selected arm calibration are applied internally. For example,
                ``[0.0, -0.10, 0.0]`` raises the TCP by 10 cm. Prefer this
                measured-pose-relative mode for small Cartesian adjustments.

        Returns:
            Result dictionary with ``success`` and diagnostics. After
            ``goto_grasp_pose`` and ``close_gripper``, the held object increases
            the collision volume. So make a small upward/clearance retreat
            with ``position="current"``; then use ``drive_straight(-0.15)``
            to move a little bit backward before returning home. Continue to
            do the task after returning home. Home mode otherwise
            reverses the last collision-checked grasp segment when available,
            then refreshes the complete RGB-D scene, retains every observed
            object as an obstacle, and uses cuRobo with no direct-home fallback.
            On failure the rest of this policy is skipped.
        """

        _required_arm(arm)
        z_approach = _finite_float("z_approach", z_approach)
        if not 0.0 <= z_approach <= 0.25:
            raise ValueError("z_approach must be in [0, 0.25] meters")
        timeout_s = _finite_float("timeout_s", timeout_s)
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        if isinstance(position, str):
            if position not in {"current", "home"}:
                raise ValueError(
                    "string goto_pose target must be 'current' or 'home'"
                )
            if quaternion_wxyz is not None:
                raise ValueError(
                    "quaternion_wxyz must be omitted for string goto_pose targets"
                )
            if z_approach != 0.0:
                raise ValueError("z_approach must be zero for string goto_pose targets")
            if position == "current" and camera_offset_xyz is None:
                raise ValueError(
                    "camera_offset_xyz is required for goto_pose('current')"
                )
            if position == "home" and camera_offset_xyz is not None:
                raise ValueError(
                    "camera_offset_xyz must be omitted for goto_pose('home')"
                )
        elif quaternion_wxyz is None:
            raise ValueError("quaternion_wxyz is required for a Cartesian goto_pose")
        elif camera_offset_xyz is not None:
            raise ValueError(
                "camera_offset_xyz is only valid for goto_pose('current')"
            )

        result = controller.goto_pose(
            position,
            quaternion_wxyz,
            arm=arm,
            z_approach=z_approach,
            timeout_s=timeout_s,
            camera_offset_xyz=camera_offset_xyz,
        )
        _require_success("goto_pose", result)
        return result

    return goto_pose


def _make_goto_grasp_pose(
    controller: ManipulationController, defaults: Mapping[str, Any] | None = None
):
    defaults = defaults or {}
    default_approach_m = defaults.get("approach_m", 0.10)
    default_timeout_s = defaults.get("timeout_s", 60.0)

    def goto_grasp_pose(
        object_name: str,
        position: np.ndarray,
        quaternion_wxyz: np.ndarray,
        *,
        arm: Literal["left", "right"],
        approach_m: float = default_approach_m,
        timeout_s: float = default_timeout_s,
    ) -> dict[str, Any]:
        """Plan and execute a whole-arm collision-free path to a grasp pose.

        Args:
            object_name: Target instance prompt used to refresh its SAM3 mask:
                the short category noun given to ``sample_grasp_pose``.
            position: Target grasp-volume-center TCP XYZ returned by
                sample_grasp_pose.
            quaternion_wxyz: Target WXYZ quaternion returned by sample_grasp_pose.
            arm: Required Nero arm identifier.
            approach_m: Collision-checked local -Z pre-grasp offset.
            timeout_s: Total guarded Pi trajectory-execution timeout.

        Returns:
            Planner and guarded trajectory-execution diagnostics. No ordinary
            goto_pose fallback is attempted when collision planning fails.
        """

        _required_arm(arm)
        approach = _finite_float("approach_m", approach_m)
        if not 0.02 <= approach <= 0.25:
            raise ValueError("approach_m must be in [0.02, 0.25] meters")
        timeout = _finite_float("timeout_s", timeout_s)
        if timeout <= 0:
            raise ValueError("timeout_s must be positive")
        result = controller.goto_grasp_pose(
            object_name,
            position,
            quaternion_wxyz,
            arm,
            approach_m=approach,
            timeout_s=timeout,
        )
        _require_success("goto_grasp_pose", result)
        return result

    return goto_grasp_pose


def _make_lift_grasped_object(
    controller: ManipulationController,
    defaults: Mapping[str, Any] | None = None,
):
    defaults = defaults or {}
    default_lift_m = defaults.get("lift_m", 0.15)
    default_timeout_s = defaults.get("timeout_s", 60.0)

    def lift_grasped_object(
        *,
        arm: Literal["left", "right"],
        lift_m: float = default_lift_m,
        timeout_s: float = default_timeout_s,
    ) -> dict[str, Any]:
        """Attach the grasped object geometry, then collision-plan a lift.

        Call this immediately after a successful ``close_gripper``. The lift
        starts from fresh joint feedback, removes the rigidly held object from
        the refreshed world cloud, attaches its conservative geometry to the
        TCP collision model, and plans a constrained local-Z retreat.

        Args:
            arm: Required Nero arm identifier.
            lift_m: Local -Z retreat distance in [0.02, 0.25] meters.
            timeout_s: Total guarded Pi trajectory-execution timeout.

        Returns:
            Planner and execution diagnostics. On failure the rest of this
            policy is skipped.
        """

        _required_arm(arm)
        lift = _finite_float("lift_m", lift_m)
        if not 0.02 <= lift <= 0.25:
            raise ValueError("lift_m must be in [0.02, 0.25] meters")
        timeout = _finite_float("timeout_s", timeout_s)
        if timeout <= 0:
            raise ValueError("timeout_s must be positive")
        result = controller.lift_grasped_object(
            arm, lift_m=lift, timeout_s=timeout
        )
        _require_success("lift_grasped_object", result)
        return result

    return lift_grasped_object


def _make_set_gripper(
    env: Any,
    controller: ManipulationController,
    *,
    opened: bool,
    primitive: str,
    defaults: Mapping[str, Any] | None = None,
    simulated: bool = False,
    attached_lift_enabled: bool = False,
):
    defaults = defaults or {}
    default_timeout_s = defaults.get("timeout_s", 3.0)
    default_force_n = defaults.get("force_n")

    def set_gripper(
        *,
        arm: Literal["left", "right"],
        timeout_s: float = default_timeout_s,
        force_n: float | None = default_force_n,
    ) -> dict[str, Any]:
        _required_arm(arm)
        timeout_s = _finite_float("timeout_s", timeout_s)
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        if force_n is not None:
            force_n = _finite_float("force_n", force_n)

        # Always go through the controller, whether or not the attached-lift
        # feature is on. It owns the bookkeeping that a bare env.set_gripper
        # skips: on close, the held object leaves the obstacle cloud and joins
        # the robot model (cuRobo and the depth self-filter); on open, it is
        # released again. Bypassing it with attached lift off left every
        # grasped object in the scene, so goto_pose('home') after a grasp
        # failed with "Start or End state in collision" (2026-09-05). The
        # controller handles ``simulated`` itself, and the lift-before-home
        # gate keys off state that only the attached-lift path creates.
        method = controller.open_gripper if opened else controller.close_gripper
        result = method(
            arm,
            timeout_s=timeout_s,
            force_n=force_n,
            simulated=simulated,
        )
        _require_success(primitive, result)
        return result

    set_gripper.__name__ = primitive
    if opened:
        set_gripper.__doc__ = """Open one native Nero CAN gripper and verify feedback.

        Args:
            arm: Required Nero arm identifier.
            timeout_s: Feedback timeout.
            force_n: Optional force in newtons, constrained to [0.1, 3.0].

        Returns:
            Result dictionary from the Pi arm service's gripper feedback.
        """
    else:
        set_gripper.__doc__ = """Close one native Nero CAN gripper and verify stable feedback.

        Args:
            arm: Required Nero arm identifier.
            timeout_s: Feedback timeout.
            force_n: Optional force in newtons, constrained to [0.1, 3.0].

        Returns:
            Result dictionary from the Pi arm service's gripper feedback. A
            nonzero final width is valid when an object is held.
        """
    return set_gripper


def _require_success(primitive: str, result: dict[str, Any]) -> None:
    """Interrupt the current policy when an effectful primitive failed.

    There is no generic arm-stop RPC to request here, unlike the navigation
    primitives' base-velocity stop: raising already prevents the rest of the
    generated program from running.
    """

    if result.get("success", False):
        return
    raise PrimitiveFailed(primitive, result)


def _finite_float(name: str, value: Any) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real number")
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value
