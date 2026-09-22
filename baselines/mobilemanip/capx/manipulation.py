"""Faithful Cap-X local manipulation semantics adapted to the YOR hardware.

The baseline deliberately keeps Cap-X's core choices: the highest-scoring SAM3
instance, the highest-scoring Contact-GraspNet proposal, and direct Cartesian
pose execution.  It does not call YOR's GraspGen-X backend, target-association
filters, collision filters, IK candidate precheck, cuRobo planner, or
manipulation-readiness primitive.  The YOR calibration and guarded Pi RPC are
unavoidable hardware substrate shared by every method on this robot.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np

from .bootstrap import configure_import_paths

configure_import_paths()

from capx.integrations.base_api import ApiBase  # noqa: E402
from yor_agent.robot.geometry import (  # noqa: E402
    matrix_to_quaternion_wxyz,
    pose_matrix,
    quaternion_wxyz_to_matrix,
    quaternion_wxyz_to_rpy,
    rpy_to_quaternion_wxyz,
)
from yor_agent.robot.perception import init_contact_graspnet, init_sam3  # noqa: E402


MANIPULATION_PRIMITIVES = (
    "get_object_pose",
    "sample_grasp_pose",
    "goto_pose",
    "open_gripper",
    "close_gripper",
)


class CapXYorManipulationApi(ApiBase):
    """The five official Cap-X manipulation calls on a dual-arm YOR robot."""

    def __init__(
        self,
        env: Any,
        *,
        segment_client_factory: Callable[[], Callable[..., Any]] = init_sam3,
        grasp_client_factory: Callable[
            [], Callable[..., Any]
        ] = init_contact_graspnet,
    ) -> None:
        super().__init__(env)
        self._segment = segment_client_factory()
        self._plan_grasps = grasp_client_factory()

    def functions(self) -> dict[str, Callable[..., Any]]:
        return {name: getattr(self, name) for name in MANIPULATION_PRIMITIVES}

    def get_object_pose(
        self,
        object_name: str,
        arm: int | str,
        return_bbox_extent: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """Estimate an object pose from the top SAM3 instance and ZED depth.

        The orientation is a simple point-cloud OBB estimate and may be
        unreliable. Use the quaternion returned by ``sample_grasp_pose`` for
        grasping.

        Args:
            object_name: Natural-language description of the visible object.
            arm: ``0``/``"left"`` or ``1``/``"right"``; selects calibration.
            return_bbox_extent: Return full OBB side lengths when true.

        Returns:
            ``(position_xyz_m, quaternion_wxyz, bbox_extent_or_none)`` in the
            selected Nero arm-base frame.
        """

        name, rgb, depth, intrinsics, arm_from_camera = self._perception_input(arm)
        mask, score = self._top_sam3_mask(object_name, rgb)
        points = _masked_metric_points(depth, intrinsics, mask)
        points_arm = points @ arm_from_camera[:3, :3].T + arm_from_camera[:3, 3]
        center, rotation, extent = _oriented_bounds(points_arm)
        quaternion = matrix_to_quaternion_wxyz(rotation)
        self._log_step(
            "get_object_pose",
            f"SAM3 score={score:.3f}; metric points={len(points_arm)}; arm={name}",
            images=rgb,
        )
        return center, quaternion, extent if return_bbox_extent else None

    def sample_grasp_pose(
        self,
        object_name: str,
        arm: int | str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return Contact-GraspNet's highest-scoring grasp for a visible object.

        This intentionally performs no YOR candidate association, collision
        filtering, IK candidate precheck, or alternate-candidate selection.

        Args:
            object_name: Natural-language description of the visible object.
            arm: ``0``/``"left"`` or ``1``/``"right"``; selects calibration.

        Returns:
            ``(position_xyz_m, quaternion_wxyz)`` for the controlled Nero
            endpoint in the selected arm-base frame.
        """

        name, rgb, depth, intrinsics, arm_from_camera = self._perception_input(arm)
        mask, sam_score = self._top_sam3_mask(object_name, rgb)
        grasps, scores, _ = self._plan_grasps(
            depth,
            intrinsics,
            mask.astype(np.uint8),
            1,
        )
        grasps = np.asarray(grasps, dtype=np.float64)
        scores = np.asarray(scores, dtype=np.float64).reshape(-1)
        if grasps.ndim != 3 or grasps.shape[1:] != (4, 4):
            raise RuntimeError("Contact-GraspNet returned malformed grasp transforms")
        if len(grasps) != len(scores) or not len(scores):
            raise RuntimeError("Contact-GraspNet returned no scored grasps")
        finite = np.isfinite(scores) & np.all(np.isfinite(grasps), axis=(1, 2))
        if not np.any(finite):
            raise RuntimeError("Contact-GraspNet returned no finite grasps")
        ranked_scores = np.where(finite, scores, -np.inf)
        best_index = int(np.argmax(ranked_scores))
        model_from_endpoint, endpoint = self._model_from_controlled_endpoint(name)
        arm_from_endpoint = (
            arm_from_camera @ grasps[best_index] @ model_from_endpoint
        )
        position = arm_from_endpoint[:3, 3].copy()
        quaternion = matrix_to_quaternion_wxyz(arm_from_endpoint[:3, :3])
        self._log_step(
            "sample_grasp_pose",
            (
                f"SAM3 score={sam_score:.3f}; selected top CGN score="
                f"{scores[best_index]:.3f}; candidates={len(scores)}; "
                f"arm={name}; endpoint={endpoint}"
            ),
            images=rgb,
        )
        return position, quaternion

    def goto_pose(
        self,
        position: np.ndarray,
        quaternion_wxyz: np.ndarray,
        arm: int | str,
        z_approach: float = 0.0,
        timeout_s: float = 60.0,
    ) -> dict[str, Any]:
        """Move directly to a Cartesian endpoint pose through the guarded Pi RPC.

        If ``z_approach`` is positive, first move that distance along negative
        local Z, then move to the requested pose. No collision motion planner
        or no-motion IK precheck is used by this Cap-X baseline.

        Args:
            position: Target XYZ in meters in the selected arm-base frame.
            quaternion_wxyz: Target WXYZ unit quaternion.
            arm: ``0``/``"left"`` or ``1``/``"right"``.
            z_approach: Nonnegative local-Z approach distance, at most 0.25 m.
            timeout_s: Positive timeout for each commanded pose.

        Returns:
            Successful final guarded-RPC result. Any failed motion raises and
            interrupts the generated policy before later motion can execute.
        """

        name = _arm_name(arm)
        target = pose_matrix(position, quaternion_wxyz)
        z_approach = _finite_float("z_approach", z_approach)
        timeout_s = _finite_float("timeout_s", timeout_s)
        if not 0.0 <= z_approach <= 0.25:
            raise ValueError("z_approach must be in [0, 0.25] meters")
        if timeout_s <= 0.0:
            raise ValueError("timeout_s must be positive")
        if z_approach:
            pregrasp = target.copy()
            pregrasp[:3, 3] += target[:3, :3] @ np.asarray([0.0, 0.0, -z_approach])
            result = self._env.move_arm_pose(
                name, _transform_to_xyz_rpy(pregrasp), timeout_s
            )
            _require_success("goto_pose(pregrasp)", result)
        result = self._env.move_arm_pose(name, _transform_to_xyz_rpy(target), timeout_s)
        _require_success("goto_pose", result)
        self._log_step("goto_pose", f"Direct Cartesian motion complete; arm={name}")
        return result

    def open_gripper(
        self,
        arm: int | str,
        timeout_s: float = 3.0,
    ) -> dict[str, Any]:
        """Open one YOR gripper through the guarded hardware RPC.

        Args:
            arm: ``0``/``"left"`` or ``1``/``"right"``.
            timeout_s: Positive command timeout in seconds.

        Returns:
            Successful RPC result. A failed command interrupts the policy.
        """

        return self._set_gripper(arm, opened=True, timeout_s=timeout_s)

    def close_gripper(
        self,
        arm: int | str,
        timeout_s: float = 3.0,
    ) -> dict[str, Any]:
        """Close one YOR gripper through the guarded hardware RPC.

        Args:
            arm: ``0``/``"left"`` or ``1``/``"right"``.
            timeout_s: Positive command timeout in seconds.

        Returns:
            Successful RPC result. A failed command interrupts the policy.
        """

        return self._set_gripper(arm, opened=False, timeout_s=timeout_s)

    def _set_gripper(
        self, arm: int | str, *, opened: bool, timeout_s: float
    ) -> dict[str, Any]:
        name = _arm_name(arm)
        timeout_s = _finite_float("timeout_s", timeout_s)
        if timeout_s <= 0.0:
            raise ValueError("timeout_s must be positive")
        primitive = "open_gripper" if opened else "close_gripper"
        baseline_settings = dict(
            getattr(self._env, "capx_manipulation_config", {}) or {}
        )
        if bool(baseline_settings.get("simulate_gripper", False)):
            result = {
                "success": True,
                "reason": "configured_simulation",
                "simulated": True,
                "arm": name,
                "opened": opened,
            }
        else:
            result = self._env.set_gripper(name, opened, timeout_s, None)
        _require_success(primitive, result)
        self._log_step(
            primitive,
            f"Completed; arm={name}; simulated={result.get('simulated', False)}",
        )
        return result

    def _perception_input(
        self, arm: int | str
    ) -> tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        name = _arm_name(arm)
        observation = self._env.get_observation()
        camera = observation["robot0_robotview"]
        rgb = np.asarray(camera["images"]["rgb"], dtype=np.uint8)
        depth = np.asarray(camera["images"]["depth"], dtype=np.float32)
        if depth.ndim == 3 and depth.shape[-1] == 1:
            depth = depth[..., 0]
        if depth.shape != rgb.shape[:2]:
            raise RuntimeError("ZED RGB and depth dimensions do not match")
        calibration = self._env.manipulation_calibration(name)
        intrinsics = np.asarray(calibration["camera_intrinsics"], dtype=np.float64).copy()
        calibration_width, calibration_height = calibration[
            "camera_calibration_resolution"
        ]
        height, width = depth.shape
        scale_x = width / float(calibration_width)
        scale_y = height / float(calibration_height)
        intrinsics[0, 0] *= scale_x
        intrinsics[0, 2] = scale_x * (intrinsics[0, 2] + 0.5) - 0.5
        intrinsics[1, 1] *= scale_y
        intrinsics[1, 2] = scale_y * (intrinsics[1, 2] + 0.5) - 0.5
        return (
            name,
            rgb,
            depth,
            intrinsics,
            np.asarray(calibration["arm_from_camera"], dtype=np.float64),
        )

    def _top_sam3_mask(
        self, object_name: str, rgb: np.ndarray
    ) -> tuple[np.ndarray, float]:
        if not isinstance(object_name, str) or not object_name.strip():
            raise ValueError("object_name must be a non-empty string")
        results = self._segment(rgb, text_prompt=object_name.strip())
        if not results:
            raise RuntimeError(f"SAM3 found no instance for {object_name!r}")
        best = max(results, key=lambda item: float(item.get("score", 0.0)))
        mask = np.asarray(best["mask"], dtype=bool)
        if mask.shape != rgb.shape[:2]:
            raise RuntimeError("SAM3 mask shape does not match the ZED image")
        return mask, float(best.get("score", 0.0))

    def _model_from_controlled_endpoint(self, arm_name: str) -> tuple[np.ndarray, str]:
        config = self._env.manipulation_config
        depth_m = float(config.get("contact_graspnet_origin_to_tcp_m", 0.1034))
        alignment = np.asarray(
            config.get(
                "contact_graspnet_to_nero_tcp_rpy_rad",
                [0.0, 0.0, -math.pi / 2.0],
            ),
            dtype=np.float64,
        )
        if not np.isfinite(depth_m) or depth_m < 0.0:
            raise ValueError("contact_graspnet_origin_to_tcp_m must be nonnegative")
        if alignment.shape != (3,) or not np.all(np.isfinite(alignment)):
            raise ValueError(
                "contact_graspnet_to_nero_tcp_rpy_rad must have three finite values"
            )
        model_from_tcp = np.eye(4, dtype=np.float64)
        model_from_tcp[:3, :3] = quaternion_wxyz_to_matrix(
            rpy_to_quaternion_wxyz(alignment)
        )
        model_from_tcp[2, 3] = depth_m

        status = self._env.arm_status()
        if not isinstance(status, dict):
            raise RuntimeError("arm_status returned a non-dictionary")
        endpoint = str(status.get("end_effector_frame", "tcp")).lower()
        if endpoint == "tcp":
            return model_from_tcp, endpoint
        if endpoint != "flange":
            raise RuntimeError(f"unsupported arm endpoint frame {endpoint!r}")
        offsets = status.get("tcp_offsets_xyz_rpy")
        if not isinstance(offsets, dict) or arm_name not in offsets:
            raise RuntimeError("flange control requires the Pi's TCP offset metadata")
        offset = np.asarray(offsets[arm_name], dtype=np.float64)
        if offset.shape != (6,) or not np.all(np.isfinite(offset)):
            raise RuntimeError(f"invalid {arm_name} flange-to-TCP offset")
        flange_from_tcp = pose_matrix(
            offset[:3], rpy_to_quaternion_wxyz(offset[3:])
        )
        return model_from_tcp @ np.linalg.inv(flange_from_tcp), endpoint


def _arm_name(arm: int | str) -> str:
    if arm in (0, "0", "left"):
        return "left"
    if arm in (1, "1", "right"):
        return "right"
    raise ValueError("arm must be 0/'left' or 1/'right'")


def _finite_float(name: str, value: Any) -> float:
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _require_success(primitive: str, result: Any) -> None:
    if not isinstance(result, dict) or not result.get("success", False):
        raise RuntimeError(f"{primitive} failed: {result!r}")


def _transform_to_xyz_rpy(transform: np.ndarray) -> list[float]:
    quaternion = matrix_to_quaternion_wxyz(transform[:3, :3])
    rpy = quaternion_wxyz_to_rpy(quaternion)
    return [float(value) for value in np.concatenate((transform[:3, 3], rpy))]


def _masked_metric_points(
    depth: np.ndarray, intrinsics: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    valid = mask & np.isfinite(depth) & (depth > 0.0)
    rows, cols = np.nonzero(valid)
    if len(rows) < 20:
        raise RuntimeError(
            f"Only {len(rows)} valid depth points were available for the SAM3 mask"
        )
    z = depth[rows, cols].astype(np.float64)
    x = (cols.astype(np.float64) - intrinsics[0, 2]) * z / intrinsics[0, 0]
    y = (rows.astype(np.float64) - intrinsics[1, 2]) * z / intrinsics[1, 1]
    return np.column_stack((x, y, z))


def _oriented_bounds(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Dependency-light PCA OBB equivalent to Cap-X's Open3D OBB step."""

    center = np.mean(points, axis=0)
    centered = points - center
    _, _, axes = np.linalg.svd(centered, full_matrices=False)
    rotation = axes.T
    if np.linalg.det(rotation) < 0.0:
        rotation[:, -1] *= -1.0
    local = centered @ rotation
    lower = np.min(local, axis=0)
    upper = np.max(local, axis=0)
    extent = upper - lower
    obb_center = center + ((lower + upper) * 0.5) @ rotation.T
    return obb_center, rotation, extent
