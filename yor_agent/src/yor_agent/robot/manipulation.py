"""Perception, grasp selection, and arm motion for YOR's dual Nero arms."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any, Literal

import numpy as np

from .grasp_backends import DEFAULT_GRASP_BACKEND, create_grasp_backend
from .motion_planning_client import GraspMotionPlanningClient
from .geometry import (
    matrix_to_quaternion_wxyz,
    pose_matrix,
    quaternion_wxyz_to_matrix,
    quaternion_wxyz_to_rpy,
    rpy_to_quaternion_wxyz,
)


# Contact-GraspNet trains with a Panda gripper frame whose origin is behind the
# jaw baseline.  Its own build_6d_grasp() uses this exact gripper_depth value.
# This is a perception-frame convention, independent of Nero's flange-to-TCP
# calibration below it in the transform chain.
CONTACT_GRASPNET_GRIPPER_DEPTH_M = 0.1034
DEFAULT_GRASP_IK_PARALLEL_JAW_SYMMETRY_ENABLED = False
# Rest pose. Taught by hand on 2026-09-10 (tools/nero_pose_teach.py), tucked
# through the collision sphere model, and on 2026-09-13 lifted so the forearm
# clears the direct-motion gate's lowest arm layer: joints 1-4 went from
# (-0.25, 1.60, -2.11, 2.10) to (-0.15, 1.10, -1.95, 2.14). With the earlier
# pose the forearm's lowest point sat at z 0.73 m, so a table top at working
# height fell into the 0.77-0.82 m gate layer (points join a layer within
# layer_z_tolerance_m) and the gate refused the 10 cm forward step that
# prepare_for_manipulation needed; whether it refused depended on a few cm of
# docking distance. Opening the shoulder (joint 2) and bending the elbow lift
# that point to z 0.84 m; joint 3 pulls the width back in.
#
# The elbow's mechanical stop is at 2.20 rad (bent by hand in calibration
# drag on 2026-09-13), but the motor firmware saturates commands at the
# vendor's 2.147 rad whatever the driver is configured with, so the target is
# 2.14. The URDF limit cuRobo plans within is 2.19 so that the measured pose
# stays inside the planning range with room to leave it. Only the left elbow
# was measured; the right is assumed to match.
#
# The width is what the base has to fit through: at 0.956 m across the
# modelled envelope filled an office aisle beside a bookshelf and the Nav2
# arms monitor held the base for 12.9 s until docking was cancelled
# (2026-09-08 22:16). This pose is 0.94 m across (0.471 m to each side) and
# reaches 0.441 m ahead of the swerve centre, inside the 0.51 / 0.49 m
# navigation footprint, which therefore stays as it is.
#
# The properties the pose has to keep, all measured from the sphere model
# (repro: .claude/daily-reports/repro/2026-09-13/rest_pose/pose_sweep.py):
#   - no arm layer below z 0.82 m extends past the body outline, so the
#     height-banded collision monitors and the direct-motion gate charge a
#     table top against the body outline and not against the arms; the Nav2
#     arm band starts at z 0.83 m.
#   - the moving links (link 2 onward) keep 0.20 m from the centreline, clear
#     of the lift column.
#   - the elbow stays inside the URDF limit by at least 0.03 rad, as above.
# Re-measure with the probe in docs/nav2_setup.md after any change, and
# re-derive robot_footprint_xy from it. Keep this in sync with LEFT_HOME /
# RIGHT_HOME in services/nero_arm/service.py, which homes the arms at start-up.
#
# The right arm mirrors the left by flipping joints 1, 3, 5 and 6 (all
# link-frame z axes in this URDF): FK puts every link within 0.04 m of the
# left arm's y-mirror. Flipping joint 3 alone, which the old pose happened to
# satisfy because its other odd joints were zero, sends the right gripper
# 0.58 m away from the mirror.
NERO_HOME_JOINTS_RAD = {
    "left": np.asarray([-0.1500, 1.1000, -1.9500, 2.1400, -0.8258, -0.2723, -0.9225]),
    "right": np.asarray([0.1500, 1.1000, 1.9500, 2.1400, 0.8258, 0.2723, -0.9225]),
}
NERO_MIRROR_SIGNS = np.asarray([-1.0, 1.0, -1.0, 1.0, -1.0, -1.0, 1.0])
# Matches the Pi service's TRAJECTORY_MAX_WAYPOINTS: one planner path is one
# streamed RPC, so the arm no longer settles at chunk boundaries.
PI_MAX_TRAJECTORY_WAYPOINTS = 256
PI_MAX_TCP_POSITION_NORM_M = 1.0
PLANNER_MAX_TRAJECTORY_WAYPOINTS = 1024
TRAJECTORY_MAX_JOINT_STEP_RAD = 0.10
# Attached-object box built from the grasped object's masked depth points:
# only points within this radius of the planned TCP can belong to the held
# object (gripper opening 0.091 m; the objects handled so far are <= 0.15 m),
# and the grasp-motion planner rejects boxes with any extent above 0.40 m.
ATTACHED_OBJECT_DEFAULT_RADIUS_M = 0.20
ATTACHED_OBJECT_MAX_EXTENT_M = 0.40
DEFAULT_GRASP_IK_ACCEPTABLE_POSITION_ERROR_M = 0.020
DEFAULT_GRASP_IK_ACCEPTABLE_ROTATION_ERROR_RAD = 0.10


def _default_segment_client_factory() -> Callable[..., Any]:
    from .perception.sam3_client import init_sam3

    return init_sam3()


def _default_motion_planner_factory(config: dict[str, Any]) -> Any:
    return GraspMotionPlanningClient(config)


class ManipulationController:
    """Robot-owned manipulation implementation shared by tools and primitives."""

    def __init__(
        self,
        env: Any,
        *,
        segment_client_factory: Callable[[], Callable[..., Any]] | None = None,
        grasp_backend_factory: Callable[[dict[str, Any]], Any] | None = None,
        motion_planner_factory: Callable[[dict[str, Any]], Any] | None = None,
        attached_lift_enabled: bool = False,
    ) -> None:
        required = (
            "observe",
            "arm_status",
            "manipulation_calibration",
            "move_arm_pose",
            "set_gripper",
        )
        missing = [name for name in required if not callable(getattr(env, name, None))]
        if missing:
            raise TypeError(f"ManipulationController environment is missing: {missing}")
        self._env = env
        self._segment_client_factory = (
            segment_client_factory or _default_segment_client_factory
        )
        self._grasp_backend_factory = grasp_backend_factory or create_grasp_backend
        self._motion_planner_factory = (
            motion_planner_factory or _default_motion_planner_factory
        )
        self._attached_lift_enabled = bool(attached_lift_enabled)
        self._segment = None
        self._plan_grasp = None
        self._motion_planner = None
        # Populated by sample_grasp_pose for optional operator visualization.
        # This is diagnostic state, not part of the generated-code API.
        self.last_grasp_debug: dict[str, Any] | None = None
        # Perception/tail split of the most recent sample_grasp_pose, read back
        # by certify_pi_grasp_readiness so the certification reports its stages.
        self.last_sample_timings_s: dict[str, float] = {}
        # Split of the most recent generate_grasp_candidates. Also returned in
        # its result as ``timings_s``; this attribute is the fallback for the
        # certification path, which does not see that dictionary.
        self.last_generation_timings_s: dict[str, float] = {}
        # A grasp finishes in intentional contact with the target, so treating
        # the target as an ordinary obstacle immediately makes the next
        # plan_cspace request start in collision.  Keep only the final,
        # straight grasp segment so goto_pose("home") can first reverse the
        # exact path that was already checked against every non-target object.
        # The cache is process-local and arm-specific; a start-state check
        # below prevents a stale segment from ever being executed.
        self._pending_grasp_retreats: dict[str, np.ndarray] = {}
        # sample_grasp_pose keeps the collision/IK-filtered alternatives behind
        # its tuple-shaped public API so goto_grasp_pose can still use cuRobo's
        # native goalset selection on the same controller instance.
        self._sampled_grasp_goalsets: dict[str, dict[str, Any]] = {}
        # prepare_for_manipulation stores a short-lived, collision-safe,
        # strictly Pi-converged goalset here. The immediately following
        # sample_grasp_pose performs a
        # fresh target-instance check, then exposes the same goalset to
        # goto_grasp_pose instead of asking the stochastic grasp backend for a
        # different set of poses.
        self._prepared_grasp_certificates: dict[str, dict[str, Any]] = {}
        # A successful goto ends at contact but does not close the gripper.
        # These contexts enforce the explicit close -> attach -> lift ordering.
        self._pending_grasps: dict[str, dict[str, Any]] = {}
        self._attached_objects: dict[str, dict[str, Any]] = {}
        # Navigation filtering also tracks simulated grasps when the optional
        # collision-planned lift primitive is not exposed. Its box follows the
        # measured TCP through FK until open_gripper releases it.
        self._pending_self_filter_grasps: dict[str, dict[str, Any]] = {}

    def functions(self) -> dict[str, Callable[..., Any]]:
        """Return the semantic surface used by supervised, no-LLM tools."""

        return {
            "get_object_pose": self.get_object_pose,
            "sample_grasp_pose": self.sample_grasp_pose,
            "goto_grasp_pose": self.goto_grasp_pose,
            "lift_grasped_object": self.lift_grasped_object,
            "goto_pose": self.goto_pose,
            "open_gripper": self.open_gripper,
            "close_gripper": self.close_gripper,
        }

    @staticmethod
    def _log_step(*args: Any, **kwargs: Any) -> None:
        """Compatibility hook retained for the migrated diagnostic code."""

    @staticmethod
    def _log_step_update(*args: Any, **kwargs: Any) -> None:
        """Compatibility hook retained for the migrated diagnostic code."""

    @staticmethod
    def _arm_name(arm: int | str) -> str:
        if arm in (0, "0", "left"):
            return "left"
        if arm in (1, "1", "right"):
            return "right"
        raise ValueError("arm must be 0/'left' or 1/'right'")

    def _segment_client(self):
        if self._segment is None:
            self._segment = self._segment_client_factory()
        return self._segment

    def _grasp_client(self):
        if self._plan_grasp is None:
            self._plan_grasp = self._grasp_backend_factory(
                self._env.manipulation_config
            )
        return self._plan_grasp

    def _motion_planner_client(self):
        if self._motion_planner is None:
            self._motion_planner = self._motion_planner_factory(
                self._env.manipulation_config
            )
        return self._motion_planner

    def plan_grasp_goalset(
        self,
        current_joints: np.ndarray,
        tcp_poses: np.ndarray,
        obstacle_points: np.ndarray,
        *,
        approach_m: float,
        goal_joint_seed: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Run the exact cuRobo goalset planner shared by prepare and goto."""

        joints = np.asarray(current_joints, dtype=np.float64)
        goals = np.asarray(tcp_poses, dtype=np.float64)
        obstacles = np.asarray(obstacle_points, dtype=np.float32)
        if joints.shape != (7,) or not np.all(np.isfinite(joints)):
            raise ValueError("current_joints must contain seven finite values")
        if (
            goals.ndim != 3
            or goals.shape[1:] != (4, 4)
            or not 1 <= len(goals) <= 16
            or not np.all(np.isfinite(goals))
        ):
            raise ValueError("tcp_poses must contain 1 to 16 finite 4x4 poses")
        if obstacles.ndim != 2 or obstacles.shape[1] != 3 or not np.all(
            np.isfinite(obstacles)
        ):
            raise ValueError("obstacle_points must have finite shape [N, 3]")
        approach = float(approach_m)
        if not np.isfinite(approach) or not 0.02 <= approach <= 0.25:
            raise ValueError("approach_m must be in [0.02, 0.25] meters")
        clearance_m = float(
            self._env.manipulation_config.get(
                "grasp_motion_collision_clearance_m", 0.008
            )
        )
        approach_samples = int(
            self._env.manipulation_config.get("grasp_motion_approach_samples", 8)
        )
        scene_voxel_m = float(
            self._env.manipulation_config.get("grasp_motion_scene_voxel_m", 0.012)
        )
        return self._motion_planner_client().plan_grasp(
            joints,
            goals,
            obstacles,
            clearance_m=clearance_m,
            approach_m=approach,
            approach_samples=approach_samples,
            scene_voxel_m=scene_voxel_m,
            goal_joint_seed=goal_joint_seed,
            planner_attempts=self._planner_attempt_options(),
        )

    def _planner_attempt_options(self) -> dict[str, Any]:
        """Optional planner-service options forwarded verbatim from configuration.

        ``grasp_motion_max_attempts`` and ``grasp_motion_enable_graph_attempt``
        default to unset, which keeps cuRobo's own defaults (five attempts with
        PRM graph seeding from the second attempt). Lower values bound planning
        latency on the Jetson at the cost of rare hard-scene successes; they
        never relax any collision or joint-limit check. The ``scene_*`` keys
        select the observed-world representation (mesh mode, coarse far pitch,
        fine radius around the tool centres, reach crop); unset keeps the
        service defaults (greedy mesh, single pitch, 1.6 m crop).
        """

        options: dict[str, Any] = {}
        for config_key, request_key, minimum, maximum in (
            ("grasp_motion_max_attempts", "max_attempts", 1, 10),
            ("grasp_motion_enable_graph_attempt", "enable_graph_attempt", 0, 10),
            # Cap on trajopt time-optimal refinement passes (cuRobo: home 3,
            # grasp/lift 1). The Pi paces waypoints at a fixed 0.1 s, so the
            # refinement only costs planning time here.
            ("grasp_motion_finetune_attempts", "finetune_attempts", 0, 3),
        ):
            value = self._env.manipulation_config.get(config_key)
            if value is None:
                continue
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or not minimum <= value <= maximum
            ):
                raise ValueError(
                    f"{config_key} must be an integer in [{minimum}, {maximum}]"
                )
            options[request_key] = int(value)
        # World representation on the planner service (see
        # services/grasp_motion/service.py _scene_options for the semantics).
        mesh_mode = self._env.manipulation_config.get("grasp_motion_scene_mesh")
        if mesh_mode is not None:
            if mesh_mode not in ("tiled", "greedy", "per_voxel"):
                raise ValueError(
                    "grasp_motion_scene_mesh must be 'tiled', 'greedy' or 'per_voxel'"
                )
            options["scene_mesh"] = str(mesh_mode)
        for config_key, request_key, low, high in (
            ("grasp_motion_scene_tile_m", "scene_tile_m", 0.10, 2.0),
            ("grasp_motion_scene_coarse_voxel_m", "scene_coarse_voxel_m", 0.003, 0.05),
            ("grasp_motion_scene_fine_radius_m", "scene_fine_radius_m", 0.10, 1.0),
            ("grasp_motion_scene_crop_radius_m", "scene_crop_radius_m", 1.5, 10.0),
        ):
            value = self._env.manipulation_config.get(config_key)
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{config_key} must be a number in [{low}, {high}]")
            value = float(value)
            if not np.isfinite(value) or not low <= value <= high:
                raise ValueError(f"{config_key} must be a number in [{low}, {high}]")
            options[request_key] = value
        return options

    @staticmethod
    def _planner_diagnostics(plan: dict[str, Any]) -> dict[str, Any]:
        """Copy the planner service's per-phase timing and world diagnostics."""

        return {
            "planning_time_s": plan.get("planning_time_s"),
            "planner_timings": plan.get("timings"),
            "planner_scene": plan.get("scene"),
            "planner_attempts": plan.get("planner_attempts"),
            "planner_solver": {
                key: plan.get(key)
                for key in ("solver", "pregrasp_solver", "grasp_solver")
                if plan.get(key) is not None
            },
        }

    def _base_pose_snapshot(self) -> np.ndarray:
        frame = self._env.navigation_frame(
            max_age_s=self._env.controller.config.pose_max_age_s
        )
        pose = frame.planar_pose
        values = np.asarray(
            [pose.x_m, pose.y_m, pose.yaw_rad], dtype=np.float64
        )
        if values.shape != (3,) or not np.all(np.isfinite(values)):
            raise RuntimeError("base has no valid planar pose for grasp certification")
        return values

    def _arm_joint_snapshot(self, name: str) -> np.ndarray:
        status = self._env.arm_status()
        if not isinstance(status, dict) or bool(status.get("estop_latched", False)):
            raise RuntimeError("Pi arm RPC is unavailable or emergency-stopped")
        snapshot = status.get(name)
        if not isinstance(snapshot, dict):
            raise RuntimeError(f"arm RPC has no {name} status snapshot")
        joints = np.asarray(snapshot.get("joint_pos"), dtype=np.float64)
        if joints.shape != (7,) or not np.all(np.isfinite(joints)):
            raise RuntimeError(f"{name} arm has no valid seven-joint feedback")
        return joints

    def _pi_seed_for_tcp(
        self, name: str, target: np.ndarray
    ) -> tuple[
        np.ndarray | None,
        dict[str, Any] | None,
        bool,
        dict[str, Any] | None,
    ]:
        quaternion = matrix_to_quaternion_wxyz(target[:3, :3])
        target_rpy = quaternion_wxyz_to_rpy(quaternion)
        result = self._env.plan_arm_poses(
            name,
            [np.concatenate([target[:3, 3], target_rpy]).tolist()],
        )
        plan = None
        finite_nonconverged = False
        seed = None
        if isinstance(result, dict):
            plans = result.get("plans")
            if isinstance(plans, list) and plans:
                candidate = plans[0]
                if isinstance(candidate, dict):
                    plan = candidate
                    joints = np.asarray(
                        candidate.get("ik_joint_target"), dtype=np.float64
                    )
                    if (
                        candidate.get("success", False)
                        and joints.shape == (7,)
                        and np.all(np.isfinite(joints))
                    ):
                        converged = candidate.get("ik_converged")
                        if isinstance(converged, (bool, np.bool_)):
                            if bool(converged):
                                seed = joints
                            else:
                                finite_nonconverged = True
        result_mapping = result if isinstance(result, dict) else None
        return seed, plan, finite_nonconverged, result_mapping

    def certify_pi_grasp_readiness(
        self,
        object_name: str,
        arm: int | str,
        *,
        depth_retry_count: int = 0,
    ) -> dict[str, Any]:
        """Freshly cache a collision-safe, strictly Pi-converged goalset.

        This is the actual-pose validation used by
        ``prepare_for_manipulation``. It intentionally does not call cuRobo;
        the later ``goto_grasp_pose`` remains the authority for whole-arm
        collision-aware trajectory planning.
        """

        name = self._arm_name(arm)
        self._prepared_grasp_certificates.pop(name, None)
        certification_started = time.monotonic()
        position, quaternion = self.sample_grasp_pose(
            object_name,
            name,
            depth_retry_count=depth_retry_count,
        )
        certification_elapsed = max(0.0, time.monotonic() - certification_started)
        generation_elapsed = float(
            self.last_sample_timings_s.get("grasp_candidate_generation", 0.0)
        )
        stage_timings_s = {
            "grasp_candidate_generation": generation_elapsed,
            # Swept-volume collision gate plus the Pi IK precheck.
            "collision_and_pi_ik": max(0.0, certification_elapsed - generation_elapsed),
            "generation_split_s": dict(self.last_generation_timings_s),
        }
        cached = self._sampled_grasp_goalsets.get(name)
        if not isinstance(cached, dict):
            raise RuntimeError("fresh grasp sampling did not cache a goalset")
        goals = np.asarray(cached.get("tcp_poses"), dtype=np.float64)
        if goals.ndim != 3 or goals.shape[1:] != (4, 4) or len(goals) == 0:
            raise RuntimeError("fresh grasp sampling cached an invalid goalset")
        debug = self.last_grasp_debug
        if not isinstance(debug, dict) or not bool(
            debug.get("collision_check_enabled", False)
        ):
            raise RuntimeError(
                "strict readiness requires grasp_collision_check_enabled"
            )
        current_joints = self._arm_joint_snapshot(name)
        selected = pose_matrix(position, quaternion)
        if not np.allclose(goals[0], selected, atol=1e-5):
            raise RuntimeError("strict Pi goalset is not selected-first")
        self._prepared_grasp_certificates[name] = {
            "object_name": str(object_name),
            "selected_tcp_pose": selected.copy(),
            "tcp_poses": goals.copy(),
            "base_pose": self._base_pose_snapshot(),
            "joint_pos": current_joints.copy(),
            "created_monotonic": time.monotonic(),
            "certification": "collision_safe_strict_pi",
        }
        # Prevent an accidental direct goto from using the pre-certification
        # ordering. The public sample call consumes the certificate and installs
        # its selected-first goalset explicitly.
        self._sampled_grasp_goalsets.pop(name, None)
        return {
            "success": True,
            "reason": "strict_pi_grasp_goalset_ready",
            "source": "fresh_sample",
            "arm": name,
            "goalset_candidate_count": int(len(goals)),
            "selected_tcp_pose": selected.tolist(),
            "pi_converged_goal_count": int(len(goals)),
            "collision_safe": True,
            "curobo_called": False,
            "stage_timings_s": stage_timings_s,
        }

    def install_prepared_grasp_certificate(
        self,
        object_name: str,
        arm: int | str,
        tcp_poses: np.ndarray,
        *,
        source: str,
    ) -> dict[str, Any]:
        """Cache an externally certified, selected-first goalset for this arm.

        The caller guarantees that every pose is swept-volume collision-safe on
        the current scene and has strict Pi ``ik_converged=True`` from the
        current base pose. The certificate carries the same base/joint/age
        guards as ``certify_pi_grasp_readiness`` and is revalidated against a
        fresh SAM3 target by the next ``sample_grasp_pose``.
        """

        name = self._arm_name(arm)
        self._prepared_grasp_certificates.pop(name, None)
        for key in ("grasp_ik_precheck_enabled", "grasp_collision_check_enabled"):
            enabled = self._env.manipulation_config.get(key, False)
            if not isinstance(enabled, bool) or not enabled:
                raise RuntimeError(f"strict readiness requires {key}")
        goals = np.asarray(tcp_poses, dtype=np.float64)
        if (
            goals.ndim != 3
            or goals.shape[1:] != (4, 4)
            or not 1 <= len(goals) <= 16
            or not np.all(np.isfinite(goals))
        ):
            raise ValueError("certificate goalset must contain 1 to 16 finite poses")
        current_joints = self._arm_joint_snapshot(name)
        selected = goals[0].copy()
        self._prepared_grasp_certificates[name] = {
            "object_name": str(object_name),
            "selected_tcp_pose": selected,
            "tcp_poses": goals.copy(),
            "base_pose": self._base_pose_snapshot(),
            "joint_pos": current_joints.copy(),
            "created_monotonic": time.monotonic(),
            "certification": "collision_safe_strict_pi",
            "source": str(source),
        }
        self._sampled_grasp_goalsets.pop(name, None)
        self.last_grasp_debug = {
            "grasp_backend": f"{source}_certificate",
            "arm": name,
            "collision_check_enabled": True,
            "goalset_candidate_count": int(len(goals)),
        }
        return {
            "success": True,
            "reason": "strict_pi_grasp_goalset_ready",
            "source": str(source),
            "arm": name,
            "goalset_candidate_count": int(len(goals)),
            "selected_tcp_pose": selected.tolist(),
            "pi_converged_goal_count": int(len(goals)),
            "collision_safe": True,
            "curobo_called": False,
        }

    @staticmethod
    def _decimate_for_execution(
        trajectory: np.ndarray, max_step_rad: float
    ) -> np.ndarray:
        """Keep a subsequence of planner samples with adjacent steps <= cap.

        No joint value is synthesised: every kept waypoint is a sample of the
        collision-checked planner path, and both endpoints are always kept.
        Greedy: from each kept sample, skip forward while the max-abs joint
        step to the next candidate stays within ``max_step_rad``.
        """

        count = len(trajectory)
        keep = [0]
        index = 0
        while index < count - 1:
            next_index = index + 1
            while (
                next_index + 1 < count
                and float(np.max(np.abs(trajectory[next_index + 1] - trajectory[index])))
                <= max_step_rad
            ):
                next_index += 1
            keep.append(next_index)
            index = next_index
        return trajectory[np.asarray(keep, dtype=int)]

    def _execution_step_cap(self) -> float | None:
        """Optional execution-time waypoint step cap from configuration.

        The Pi streams waypoints to the firmware at the planner's 0.1 s
        sample spacing and only waits for arrival at the last one, so
        execution time follows the planned duration rather than the waypoint
        count. Unset (the default) keeps the planner path untouched; a cap
        keeps fewer samples for free-space segments, which now only coarsens
        the stream. Contact, lift and retreat segments are never decimated.
        """

        value = self._env.manipulation_config.get("grasp_motion_execution_step_rad")
        if value is None:
            return None
        cap = float(value)
        if not np.isfinite(cap) or not 0.02 <= cap <= TRAJECTORY_MAX_JOINT_STEP_RAD:
            raise ValueError(
                "grasp_motion_execution_step_rad must be in "
                f"[0.02, {TRAJECTORY_MAX_JOINT_STEP_RAD:.2f}] rad"
            )
        return cap

    def _execute_planned_trajectory(
        self,
        arm: str,
        waypoints: np.ndarray,
        timeout_s: float,
        *,
        execution_step_rad: float | None = None,
        protected_tail_count: int = 0,
    ) -> dict[str, Any]:
        """Execute one validated planner path through bounded Pi RPC chunks.

        ``execution_step_rad`` optionally decimates the free-space head of the
        path (see ``_decimate_for_execution``); the last ``protected_tail_count``
        waypoints (a constrained contact segment) are always sent unchanged.
        """

        trajectory = np.asarray(waypoints, dtype=np.float64)
        if (
            trajectory.ndim != 2
            or trajectory.shape[1] != 7
            or not np.all(np.isfinite(trajectory))
            or not 2 <= len(trajectory) <= PLANNER_MAX_TRAJECTORY_WAYPOINTS
        ):
            raise RuntimeError(
                "grasp-motion planner returned an invalid bounded joint trajectory"
            )
        maximum_step = float(np.max(np.abs(np.diff(trajectory, axis=0))))
        if maximum_step > TRAJECTORY_MAX_JOINT_STEP_RAD:
            raise RuntimeError(
                f"grasp-motion planner waypoint step {maximum_step:.3f} rad "
                f"exceeds {TRAJECTORY_MAX_JOINT_STEP_RAD:.2f} rad"
            )
        execute = getattr(self._env, "execute_arm_trajectory", None)
        if not callable(execute):
            raise RuntimeError(
                "collision-aware motion requires execute_arm_trajectory() "
                "on the environment"
            )
        planned_count = len(trajectory)
        protected_tail_count = int(protected_tail_count)
        if not 0 <= protected_tail_count <= planned_count:
            raise ValueError("protected_tail_count must be within the trajectory")
        if execution_step_rad is not None:
            cap = float(execution_step_rad)
            if not np.isfinite(cap) or not 0.02 <= cap <= TRAJECTORY_MAX_JOINT_STEP_RAD:
                raise ValueError("execution_step_rad must be in [0.02, 0.10] rad")
            head_count = planned_count - protected_tail_count
            if head_count >= 3:
                # The head ends at the first protected waypoint so the contact
                # segment starts exactly where the planner started it.
                head = self._decimate_for_execution(
                    trajectory[: head_count + (1 if protected_tail_count else 0)],
                    cap,
                )
                tail = trajectory[head_count + 1 :] if protected_tail_count else trajectory[:0]
                trajectory = np.concatenate([head, tail], axis=0)

        execution_started = time.monotonic()
        deadline = execution_started + float(timeout_s)
        summaries: list[dict[str, Any]] = []
        start_index = 0
        last_result: dict[str, Any] | None = None
        common = {
            "trajectory_waypoint_count": planned_count,
            "executed_waypoint_count": len(trajectory),
            "execution_step_cap_rad": execution_step_rad,
            "protected_tail_count": protected_tail_count,
        }
        while start_index < len(trajectory) - 1:
            end_index = min(
                start_index + PI_MAX_TRAJECTORY_WAYPOINTS,
                len(trajectory),
            )
            chunk = trajectory[start_index:end_index]
            remaining_s = deadline - time.monotonic()
            if remaining_s <= 0.0:
                return {
                    "success": False,
                    "reason": "joint_trajectory_total_timeout_before_chunk",
                    "arm": arm,
                    "trajectory_chunk_count": len(summaries),
                    "trajectory_chunks": summaries,
                    "execution_elapsed_s": time.monotonic() - execution_started,
                    **common,
                }
            chunk_started = time.monotonic()
            result = dict(execute(arm, chunk.tolist(), remaining_s))
            executed_entries = result.get("executed")
            summaries.append(
                {
                    "chunk_index": len(summaries),
                    "start_waypoint_index": start_index,
                    "end_waypoint_index": end_index - 1,
                    "waypoint_count": len(chunk),
                    "success": bool(result.get("success", False)),
                    "reason": result.get("reason"),
                    "maximum_step_rad": result.get("maximum_step_rad"),
                    "rpc_elapsed_s": time.monotonic() - chunk_started,
                    "pi_executed_count": (
                        len(executed_entries)
                        if isinstance(executed_entries, list)
                        else None
                    ),
                }
            )
            last_result = result
            if not result.get("success", False):
                return {
                    **result,
                    "trajectory_chunk_count": len(summaries),
                    "trajectory_chunks": summaries,
                    "execution_elapsed_s": time.monotonic() - execution_started,
                    **common,
                }
            start_index = end_index - 1

        assert last_result is not None
        # ``waypoint_count`` / ``executed`` below describe the LAST chunk only;
        # the totals live in trajectory_waypoint_count / executed_waypoint_count.
        return {
            **last_result,
            "last_chunk_waypoint_count": last_result.get("waypoint_count"),
            "trajectory_chunk_count": len(summaries),
            "trajectory_chunks": summaries,
            "execution_elapsed_s": time.monotonic() - execution_started,
            **common,
            "planner_maximum_step_rad": maximum_step,
        }

    def _perception_clients(self):
        """Return both clients for legacy callers that explicitly need both."""

        return self._segment_client(), self._grasp_client()

    @staticmethod
    def _reverse_grasp_segment(plan: dict[str, Any]) -> np.ndarray | None:
        """Return grasp -> pre-grasp waypoints from a cuRoboV2 grasp result."""

        segments = plan.get("segments")
        if not isinstance(segments, list):
            return None
        phase_waypoints: dict[str, np.ndarray] = {}
        for segment in segments:
            if not isinstance(segment, dict):
                continue
            phase = segment.get("phase")
            candidate = np.asarray(segment.get("waypoints"), dtype=np.float64)
            if (
                phase in ("approach", "grasp")
                and candidate.ndim == 2
                and candidate.shape[1] == 7
                and len(candidate) > 0
                and np.all(np.isfinite(candidate))
            ):
                phase_waypoints[str(phase)] = candidate
        approach = phase_waypoints.get("approach")
        grasp = phase_waypoints.get("grasp")
        if approach is None or grasp is None:
            return None

        # The service removes the duplicated approach/grasp boundary before
        # serializing its segments. Restore that pre-grasp endpoint, then
        # reverse the complete constrained Cartesian segment.
        forward = np.vstack((approach[-1:], grasp))
        retreat = forward[::-1].copy()
        if len(retreat) > 1:
            keep = np.ones(len(retreat), dtype=bool)
            keep[1:] = np.max(np.abs(np.diff(retreat, axis=0)), axis=1) > 1e-9
            retreat = retreat[keep]
        if not 2 <= len(retreat) <= PLANNER_MAX_TRAJECTORY_WAYPOINTS:
            return None
        if float(np.max(np.abs(np.diff(retreat, axis=0)))) > (
            TRAJECTORY_MAX_JOINT_STEP_RAD
        ):
            return None
        return retreat

    def _contact_graspnet_to_controlled_endpoint(
        self, name: str
    ) -> tuple[np.ndarray, str, float]:
        """Map a CGN gripper-base pose to the endpoint controlled by the Pi.

        CGN's local +Z points from its gripper-base origin to the jaw baseline.
        In normal ``tcp`` mode the Nero TCP is defined at that jaw center.  In
        diagnostic ``flange`` mode, also remove the configured Nero
        flange-to-TCP transform so the returned pose still describes the same
        physical grasp.
        """

        return self._grasp_model_to_controlled_endpoint(
            name, backend_name="contact_graspnet"
        )

    def _grasp_model_to_controlled_endpoint(
        self, name: str, *, backend_name: str
    ) -> tuple[np.ndarray, str, float]:
        """Map a backend's gripper-base frame to the controlled Nero endpoint."""

        if backend_name == "graspgenx":
            depth_key = "graspgenx_origin_to_tcp_m"
            alignment_key = "graspgenx_to_nero_tcp_rpy_rad"
            default_depth_m = 0.105
            # GraspGen-X canonicalizes parallel jaws to +X closing/+Z
            # approach.  The published Piper asset's canonical mesh is the
            # native Nero/Piper gripper frame rotated +90 degrees around +Z.
            default_alignment_rpy = [0.0, 0.0, np.pi / 2.0]
        else:
            depth_key = "contact_graspnet_origin_to_tcp_m"
            alignment_key = "contact_graspnet_to_nero_tcp_rpy_rad"
            default_depth_m = CONTACT_GRASPNET_GRIPPER_DEPTH_M
            default_alignment_rpy = [0.0, 0.0, -np.pi / 2.0]
        depth_m = float(
            self._env.manipulation_config.get(depth_key, default_depth_m)
        )
        if not np.isfinite(depth_m) or depth_m < 0.0:
            raise ValueError(f"{depth_key} must be finite and nonnegative")
        model_from_tcp = np.eye(4, dtype=np.float64)
        alignment_rpy = np.asarray(
            self._env.manipulation_config.get(
                alignment_key,
                default_alignment_rpy,
            ),
            dtype=np.float64,
        )
        if alignment_rpy.shape != (3,) or not np.all(np.isfinite(alignment_rpy)):
            raise ValueError(
                f"{alignment_key} must contain three finite values"
            )
        model_from_tcp[:3, :3] = quaternion_wxyz_to_matrix(
            rpy_to_quaternion_wxyz(alignment_rpy)
        )
        model_from_tcp[2, 3] = depth_m

        status = self._env.arm_status()
        if not isinstance(status, dict):
            raise TypeError("arm_status() returned a non-dictionary")
        endpoint = str(status.get("end_effector_frame", "tcp")).lower()
        if endpoint == "tcp":
            return model_from_tcp, endpoint, depth_m
        if endpoint != "flange":
            raise RuntimeError(f"unsupported arm endpoint frame {endpoint!r}")

        offsets = status.get("tcp_offsets_xyz_rpy")
        if not isinstance(offsets, dict) or name not in offsets:
            raise RuntimeError(
                "flange endpoint requires tcp_offsets_xyz_rpy in arm RPC status"
            )
        flange_to_tcp_xyz_rpy = np.asarray(offsets[name], dtype=np.float64)
        if (
            flange_to_tcp_xyz_rpy.shape != (6,)
            or not np.all(np.isfinite(flange_to_tcp_xyz_rpy))
        ):
            raise RuntimeError(f"invalid {name} flange-to-TCP offset from arm RPC")
        flange_from_tcp = pose_matrix(
            flange_to_tcp_xyz_rpy[:3],
            rpy_to_quaternion_wxyz(flange_to_tcp_xyz_rpy[3:]),
        )
        # T_arm_flange = T_arm_CGN @ T_CGN_TCP @ inverse(T_flange_TCP)
        return model_from_tcp @ np.linalg.inv(flange_from_tcp), endpoint, depth_m

    def _perception_input(self, arm: int | str):
        name = self._arm_name(arm)
        obs = self._env.observe()
        camera = obs["robot0_robotview"]
        rgb = np.asarray(camera["images"]["rgb"], dtype=np.uint8)
        depth = np.asarray(camera["images"]["depth"], dtype=np.float32)
        if depth.ndim == 3 and depth.shape[-1] == 1:
            depth = depth[..., 0]
        if depth.shape != rgb.shape[:2]:
            raise RuntimeError("ZED RGB and depth shapes do not match")
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
        return name, rgb, depth, intrinsics, calibration["arm_from_camera"]

    def _object_mask(self, object_name: str, rgb: np.ndarray) -> tuple[np.ndarray, float]:
        if not isinstance(object_name, str) or not object_name.strip():
            raise ValueError("object_name must be a non-empty string")
        segment = self._segment_client()
        results = segment(rgb, text_prompt=object_name.strip())
        if not results:
            raise RuntimeError(f"SAM3 found no instance for {object_name!r}")
        best = max(results, key=lambda item: float(item.get("score", 0.0)))
        score = float(best.get("score", 0.0))
        threshold = float(
            self._env.manipulation_config.get("sam3_score_threshold", 0.05)
        )
        if score < threshold:
            raise RuntimeError(
                f"best SAM3 score {score:.3f} is below threshold {threshold:.3f}"
            )
        mask = np.asarray(best["mask"], dtype=bool)
        if mask.shape != rgb.shape[:2] or np.count_nonzero(mask) < 30:
            raise RuntimeError("SAM3 mask is invalid or too small")
        return mask, score

    def _minimum_metric_depth_points(self) -> int:
        raw = self._env.manipulation_config.get(
            "minimum_metric_depth_points", 50
        )
        if isinstance(raw, bool) or not isinstance(raw, (int, np.integer)):
            raise TypeError("minimum_metric_depth_points must be an integer")
        value = int(raw)
        if not 10 <= value <= 100_000:
            raise ValueError(
                "minimum_metric_depth_points must be in [10, 100000]"
            )
        return value

    @staticmethod
    def _valid_metric_depth(
        depth: np.ndarray, mask: np.ndarray
    ) -> np.ndarray:
        return mask & np.isfinite(depth) & (depth > 0.08) & (depth < 2.5)

    @staticmethod
    def _dilate_mask(mask: np.ndarray, radius_pixels: int) -> np.ndarray:
        if radius_pixels <= 0:
            return np.asarray(mask, dtype=bool).copy()
        source = np.asarray(mask, dtype=bool)
        padded = np.pad(source, radius_pixels, mode="constant", constant_values=False)
        result = np.zeros_like(source)
        diameter = 2 * radius_pixels + 1
        for row_offset in range(diameter):
            for column_offset in range(diameter):
                result |= padded[
                    row_offset : row_offset + source.shape[0],
                    column_offset : column_offset + source.shape[1],
                ]
        return result

    def _collision_scene_points(
        self,
        depth: np.ndarray,
        target_mask: np.ndarray,
        intrinsics: np.ndarray,
        arm_from_camera: np.ndarray,
    ) -> np.ndarray:
        """Build a generic observed-obstacle cloud with the target excluded."""

        dilation = int(
            self._env.manipulation_config.get(
                "grasp_collision_target_mask_dilation_pixels", 0
            )
        )
        stride = int(
            self._env.manipulation_config.get("grasp_collision_pixel_stride", 2)
        )
        voxel_m = float(
            self._env.manipulation_config.get("grasp_collision_point_voxel_m", 0.008)
        )
        maximum_points = int(
            self._env.manipulation_config.get("grasp_collision_max_scene_points", 30_000)
        )
        if not 0 <= dilation <= 20:
            raise ValueError("grasp collision mask dilation must be in [0, 20]")
        if not 1 <= stride <= 16:
            raise ValueError("grasp collision pixel stride must be in [1, 16]")
        if not 0.003 <= voxel_m <= 0.05:
            raise ValueError("grasp collision point voxel must be in [0.003, 0.05] m")
        if not 1_000 <= maximum_points <= 100_000:
            raise ValueError("grasp collision max scene points must be in [1000, 100000]")

        excluded = self._dilate_mask(target_mask, dilation)
        valid = self._valid_metric_depth(depth, ~excluded)
        sampling = np.zeros_like(valid)
        sampling[::stride, ::stride] = True
        rows, columns = np.nonzero(valid & sampling)
        if rows.size < 100:
            raise RuntimeError(
                "observed collision scene has fewer than 100 non-target depth points"
            )
        z = depth[rows, columns].astype(np.float64)
        points_camera = np.column_stack(
            [
                (columns - intrinsics[0, 2]) * z / intrinsics[0, 0],
                (rows - intrinsics[1, 2]) * z / intrinsics[1, 1],
                z,
            ]
        )
        points_arm = (
            points_camera @ arm_from_camera[:3, :3].T
            + arm_from_camera[:3, 3]
        )
        # Deterministic voxel downsampling keeps the RPC bounded without any
        # object- or task-specific geometry assumptions.
        keys = np.floor(points_arm / voxel_m).astype(np.int64)
        _, selected = np.unique(keys, axis=0, return_index=True)
        points_arm = points_arm[np.sort(selected)]
        if len(points_arm) > maximum_points:
            indices = np.linspace(0, len(points_arm) - 1, maximum_points, dtype=int)
            points_arm = points_arm[indices]
        return np.ascontiguousarray(points_arm, dtype=np.float32)

    @staticmethod
    def _masked_points(
        depth: np.ndarray,
        mask: np.ndarray,
        intrinsics: np.ndarray,
        *,
        minimum_points: int = 50,
    ) -> np.ndarray:
        valid = ManipulationController._valid_metric_depth(depth, mask)
        rows, columns = np.nonzero(valid)
        if rows.size < int(minimum_points):
            raise RuntimeError(
                "object mask has insufficient valid metric depth: "
                f"{rows.size}/{int(minimum_points)} valid pixels"
            )
        z = depth[rows, columns].astype(np.float64)
        x = (columns - intrinsics[0, 2]) * z / intrinsics[0, 0]
        y = (rows - intrinsics[1, 2]) * z / intrinsics[1, 1]
        return np.column_stack([x, y, z])

    def _metric_object_observation(
        self,
        object_name: str,
        arm: int | str,
        *,
        depth_retry_count: int,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """Select the best fresh SAM3 observation with enough metric depth."""

        if (
            isinstance(depth_retry_count, bool)
            or not isinstance(depth_retry_count, int)
            or not 0 <= depth_retry_count <= 5
        ):
            raise ValueError("depth_retry_count must be an integer in [0, 5]")
        minimum_points = self._minimum_metric_depth_points()
        observations: list[dict[str, Any]] = []
        summaries: list[dict[str, Any]] = []
        for attempt_index in range(depth_retry_count + 1):
            name, rgb, depth, intrinsics, arm_from_camera = self._perception_input(
                arm
            )
            try:
                mask, sam_score = self._object_mask(object_name, rgb)
                segmentation_error = None
            except RuntimeError as exc:
                mask = np.zeros(rgb.shape[:2], dtype=bool)
                sam_score = None
                segmentation_error = str(exc)
            valid_depth = self._valid_metric_depth(depth, mask)
            valid_count = int(np.count_nonzero(valid_depth))
            mask_count = int(np.count_nonzero(mask))
            accepted = segmentation_error is None and valid_count >= minimum_points
            summary = {
                "attempt_index": attempt_index,
                "mask_pixels": mask_count,
                "valid_depth_points": valid_count,
                "minimum_valid_depth_points": minimum_points,
                "sam_score": None if sam_score is None else float(sam_score),
                "accepted": accepted,
                "segmentation_error": segmentation_error,
            }
            summaries.append(summary)
            observation = {
                "name": name,
                "rgb": rgb,
                "depth_m": depth,
                "intrinsics": intrinsics,
                "arm_from_camera": arm_from_camera,
                "mask": mask,
                "sam_score": sam_score,
                "valid_depth_points": valid_count,
                "summary": summary,
            }
            observations.append(observation)
            if progress_callback is not None:
                progress_callback(
                    {
                        "phase": "sam3_depth_validation",
                        "rgb": rgb.copy(),
                        "depth_m": depth.copy(),
                        "intrinsics": intrinsics.copy(),
                        "mask": mask.copy(),
                        "sam_score": sam_score,
                        "attempt_index": attempt_index,
                        "maximum_attempts": depth_retry_count + 1,
                        "mask_pixels": mask_count,
                        "valid_depth_points": valid_count,
                        "minimum_valid_depth_points": minimum_points,
                        "accepted": accepted,
                        "segmentation_error": segmentation_error,
                    }
                )
            if accepted:
                break

        usable = [
            observation
            for observation in observations
            if observation["summary"]["segmentation_error"] is None
        ]
        if not usable:
            errors = [
                str(summary["segmentation_error"])
                for summary in summaries
                if summary["segmentation_error"]
            ]
            raise RuntimeError(
                f"SAM3 produced no usable mask after {len(summaries)} attempts: "
                + "; ".join(errors)
            )
        selected = max(
            usable,
            key=lambda item: (
                int(item["valid_depth_points"]),
                float(item["sam_score"]),
            ),
        )
        best_count = int(selected["valid_depth_points"])
        if best_count < minimum_points:
            counts = [int(summary["valid_depth_points"]) for summary in summaries]
            raise RuntimeError(
                "object mask has insufficient valid metric depth after "
                f"{len(summaries)} SAM3 attempts: best {best_count}/{minimum_points} "
                f"valid pixels; attempt counts={counts}"
            )
        return selected, summaries

    def get_object_pose(
        self,
        object_name: str,
        arm: int | str,
        return_bbox_extent: bool = False,
        return_zed_distance: bool = False,
        *,
        depth_retry_count: int = 0,
    ) -> (
        tuple[np.ndarray, np.ndarray, np.ndarray | None]
        | tuple[np.ndarray, np.ndarray, np.ndarray | None, float]
    ):
        """Estimate an object's pose in the selected Nero arm-base frame.

        Args:
            object_name: Natural-language SAM3 text prompt.
            arm: ``0``/``"left"`` or ``1``/``"right"``. The returned position
                and orientation use that arm's calibrated base frame.
            return_bbox_extent: Return PCA-aligned full XYZ extents when true.
            return_zed_distance: Append the metric 3D distance from the ZED
                optical center to the object's median depth point when true.

        Returns:
            ``(position_xyz, quaternion_wxyz, bbox_extent_or_none)``. As in
            upstream CaP-X, object orientation is perception-derived and may be
            less reliable than the orientation returned by ``sample_grasp_pose``.
            When requested, ``zed_distance_m`` is appended as a fourth value.
        """

        selected, _ = self._metric_object_observation(
            object_name,
            arm,
            depth_retry_count=depth_retry_count,
        )
        name = str(selected["name"])
        rgb = np.asarray(selected["rgb"])
        depth = np.asarray(selected["depth_m"])
        intrinsics = np.asarray(selected["intrinsics"])
        arm_from_camera = np.asarray(selected["arm_from_camera"])
        mask = np.asarray(selected["mask"], dtype=bool)
        score = float(selected["sam_score"])
        self._log_step(
            "get_object_pose", f"Finding {object_name!r} for {name} arm", rgb
        )
        points_camera = self._masked_points(
            depth,
            mask,
            intrinsics,
            minimum_points=self._minimum_metric_depth_points(),
        )
        center_camera = np.median(points_camera, axis=0)
        zed_distance_m = float(np.linalg.norm(center_camera))
        points_arm = (
            points_camera @ arm_from_camera[:3, :3].T
            + arm_from_camera[:3, 3]
        )
        center = np.median(points_arm, axis=0)
        centered = points_arm - center
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        rotation = vh.T
        if np.linalg.det(rotation) < 0:
            rotation[:, -1] *= -1
        projections = centered @ rotation
        extent = np.quantile(projections, 0.98, axis=0) - np.quantile(
            projections, 0.02, axis=0
        )
        quaternion = matrix_to_quaternion_wxyz(rotation)
        self._log_step_update(
            text=(
                f"SAM3 score={score:.3f}; {points_arm.shape[0]} depth points; "
                f"ZED distance={zed_distance_m:.3f} m"
            )
        )
        result = (center, quaternion, extent if return_bbox_extent else None)
        if return_zed_distance:
            return (*result, zed_distance_m)
        return result

    def generate_grasp_candidates(
        self,
        object_name: str,
        arm: int | str,
        *,
        depth_retry_count: int = 0,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """Generate target-filtered grasps without choosing or moving the arm.

        The returned transforms are still expressed in the current optical
        camera frame.  This is the intentional hand-off used by local base-pose
        optimization: one SAM3/GraspGen observation can be rigidly transformed
        into many hypothetical base poses before a bounded batch IK request.
        """

        # Perception is the largest single stage of prepare_for_manipulation in
        # both the docked and the moved case, so it reports its own split.
        self.last_generation_timings_s = {}
        generation_started = time.monotonic()
        selected_observation, summaries = self._metric_object_observation(
            object_name,
            arm,
            depth_retry_count=depth_retry_count,
            progress_callback=progress_callback,
        )
        self.last_generation_timings_s["observation_and_sam3"] = max(
            0.0, time.monotonic() - generation_started
        )
        minimum_points = self._minimum_metric_depth_points()
        name = str(selected_observation["name"])
        rgb = np.asarray(selected_observation["rgb"])
        depth = np.asarray(selected_observation["depth_m"])
        intrinsics = np.asarray(selected_observation["intrinsics"])
        arm_from_camera = np.asarray(selected_observation["arm_from_camera"])
        mask = np.asarray(selected_observation["mask"], dtype=bool)
        sam_score = float(selected_observation["sam_score"])
        best_count = int(selected_observation["valid_depth_points"])
        points_camera = self._masked_points(
            depth,
            mask,
            intrinsics,
            minimum_points=minimum_points,
        )
        if progress_callback is not None:
            selected_summary = selected_observation["summary"]
            progress_callback(
                {
                    "phase": "grasp_generation",
                    "rgb": rgb.copy(),
                    "depth_m": depth.copy(),
                    "intrinsics": intrinsics.copy(),
                    "mask": mask.copy(),
                    "sam_score": sam_score,
                    "attempt_index": int(selected_summary["attempt_index"]),
                    "maximum_attempts": depth_retry_count + 1,
                    "mask_pixels": int(selected_summary["mask_pixels"]),
                    "valid_depth_points": best_count,
                    "minimum_valid_depth_points": minimum_points,
                    "accepted": True,
                    "segmentation_error": None,
                }
            )
        plan_grasp = self._grasp_client()
        generation_started = time.monotonic()
        raw_result = plan_grasp(
            depth,
            intrinsics,
            mask.astype(np.uint8),
            1,
        )
        backend_elapsed = max(0.0, time.monotonic() - generation_started)
        self.last_generation_timings_s["grasp_backend"] = backend_elapsed
        filtering_started = time.monotonic()
        backend_name = str(
            getattr(
                plan_grasp,
                "name",
                self._env.manipulation_config.get(
                    "grasp_backend", DEFAULT_GRASP_BACKEND
                ),
            )
        ).lower()
        backend_metadata: dict[str, Any] = {}
        if hasattr(raw_result, "poses"):
            grasps = raw_result.poses
            scores = raw_result.scores
            contact_points = raw_result.anchor_points
            backend_metadata = dict(raw_result.metadata)
        else:
            grasps, scores, contact_points = raw_result
        grasps = np.asarray(grasps, dtype=np.float64)
        scores = np.asarray(scores, dtype=np.float64).reshape(-1)
        contact_points = np.asarray(contact_points, dtype=np.float64)
        if grasps.ndim != 3 or grasps.shape[1:] != (4, 4) or scores.size == 0:
            raise RuntimeError(f"{backend_name} returned no valid grasp transforms")
        count = min(grasps.shape[0], scores.size)
        finite = np.asarray(
            [
                np.all(np.isfinite(grasps[index]))
                and np.isfinite(scores[index])
                for index in range(count)
            ]
        )
        if not np.any(finite):
            raise RuntimeError(f"{backend_name} returned only non-finite grasps")

        max_object_distance_m = float(
            self._env.manipulation_config.get(
                "grasp_candidate_max_object_distance_m", 0.10
            )
        )
        if not np.isfinite(max_object_distance_m) or max_object_distance_m <= 0:
            raise ValueError(
                "grasp_candidate_max_object_distance_m must be positive and finite"
            )
        distance_reference = str(
            self._env.manipulation_config.get(
                "grasp_candidate_distance_reference", "contact"
            )
        ).lower()
        if distance_reference not in ("contact", "origin"):
            raise ValueError(
                "grasp_candidate_distance_reference must be 'contact' or 'origin'"
            )
        object_distances = np.full(count, np.inf, dtype=np.float64)
        for index in np.flatnonzero(finite):
            reference_point = grasps[index, :3, 3]
            if distance_reference == "contact":
                if not (
                    contact_points.ndim == 2
                    and contact_points.shape[1] == 3
                    and index < contact_points.shape[0]
                    and np.all(np.isfinite(contact_points[index]))
                ):
                    continue
                reference_point = contact_points[index]
            delta = points_camera - reference_point
            object_distances[index] = float(
                np.sqrt(np.min(np.einsum("ij,ij->i", delta, delta)))
            )
        geometrically_valid = finite & (object_distances <= max_object_distance_m)
        if not np.any(geometrically_valid):
            best_distance = float(np.min(object_distances[finite]))
            raise RuntimeError(
                f"{backend_name} returned no candidate on the target object: "
                f"nearest grasp {distance_reference} is {best_distance:.3f} m "
                "from the SAM3 point cloud "
                f"(limit {max_object_distance_m:.3f} m)"
            )
        cgn_to_endpoint, endpoint_name, cgn_tcp_depth_m = (
            self._grasp_model_to_controlled_endpoint(
                name, backend_name=backend_name
            )
        )
        return {
            "arm": name,
            "rgb": rgb,
            "depth_m": depth,
            "intrinsics": intrinsics,
            "arm_from_camera": arm_from_camera,
            "mask": mask,
            "sam_score": float(sam_score),
            "points_camera": points_camera,
            "depth_validation_attempts": summaries,
            "timings_s": {
                **self.last_generation_timings_s,
                # Target association, collision distance and score filtering.
                "candidate_filtering": max(
                    0.0, time.monotonic() - filtering_started
                ),
            },
            "backend_name": backend_name,
            "backend_metadata": backend_metadata,
            "grasps_camera_from_model": grasps,
            "scores": scores,
            "contact_points_camera": contact_points,
            "object_distances_m": object_distances,
            "candidate_indices": np.flatnonzero(geometrically_valid),
            "distance_reference": distance_reference,
            "model_from_controlled_endpoint": cgn_to_endpoint,
            "controlled_endpoint": endpoint_name,
            "model_origin_to_tcp_m": cgn_tcp_depth_m,
        }

    def sample_grasp_pose(
        self,
        object_name: str,
        arm: int | str,
        *,
        depth_retry_count: int = 0,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sample a configured-backend grasp in the selected Nero arm-base frame.

        Args:
            object_name: Natural-language SAM3 text prompt.
            arm: Nero arm identifier.

        Returns:
            ``(position_xyz, quaternion_wxyz)`` for the selected grasp. Success
            guarantees that the cached goalset contains at least one
            swept-volume collision-safe grasp with Pi ``ik_converged=True``.
            This preserves the upstream CaP-X tuple-shaped calling convention.
        """

        self.last_grasp_debug = None
        prepared = self._consume_prepared_grasp_certificate(object_name, arm)
        if prepared is not None:
            return prepared
        ik_precheck_enabled = self._env.manipulation_config.get(
            "grasp_ik_precheck_enabled", True
        )
        if not isinstance(ik_precheck_enabled, bool):
            raise ValueError("grasp_ik_precheck_enabled must be a boolean")
        if not ik_precheck_enabled:
            raise RuntimeError(
                "sample_grasp_pose requires grasp_ik_precheck_enabled so success "
                "always means at least one strictly Pi-converged grasp exists"
            )
        collision_check_enabled = self._env.manipulation_config.get(
            "grasp_collision_check_enabled", False
        )
        if not isinstance(collision_check_enabled, bool):
            raise ValueError("grasp_collision_check_enabled must be a boolean")
        if not collision_check_enabled:
            raise RuntimeError(
                "sample_grasp_pose requires grasp_collision_check_enabled so "
                "success always means the cached Pi-converged goalset is "
                "swept-volume collision-safe"
            )
        self._sampled_grasp_goalsets.pop(self._arm_name(arm), None)
        # Split perception from the collision/IK tail so the actual-pose
        # certification reports where its seconds go, the same way
        # prepare_for_manipulation reports its own stages.
        sample_started = time.monotonic()
        self.last_sample_timings_s = {}
        generated = self.generate_grasp_candidates(
            object_name,
            arm,
            depth_retry_count=depth_retry_count,
            progress_callback=progress_callback,
        )
        self.last_sample_timings_s["grasp_candidate_generation"] = max(
            0.0, time.monotonic() - sample_started
        )
        name = str(generated["arm"])
        rgb = np.asarray(generated["rgb"])
        depth = np.asarray(generated["depth_m"])
        intrinsics = np.asarray(generated["intrinsics"])
        arm_from_camera = np.asarray(generated["arm_from_camera"])
        self._log_step(
            "sample_grasp_pose", f"Planning grasp for {object_name!r} with {name} arm", rgb
        )
        mask = np.asarray(generated["mask"], dtype=bool)
        sam_score = float(generated["sam_score"])
        points_camera = np.asarray(generated["points_camera"])
        backend_name = str(generated["backend_name"])
        backend_metadata = dict(generated["backend_metadata"])
        grasps = np.asarray(generated["grasps_camera_from_model"])
        scores = np.asarray(generated["scores"])
        contact_points = np.asarray(generated["contact_points_camera"])
        object_distances = np.asarray(generated["object_distances_m"])
        candidates = np.asarray(generated["candidate_indices"], dtype=np.int64)
        count = min(grasps.shape[0], scores.size)
        distance_reference = str(generated["distance_reference"])
        cgn_to_endpoint = np.asarray(generated["model_from_controlled_endpoint"])
        endpoint_name = str(generated["controlled_endpoint"])
        cgn_tcp_depth_m = float(generated["model_origin_to_tcp_m"])
        selected_plan: dict[str, Any] | None = None
        goalset_targets: list[np.ndarray] = []
        evaluated_candidate_count = 0
        debug_candidates: list[dict[str, Any]] = []
        collision_check: dict[str, Any] | None = None
        if ik_precheck_enabled:
            candidate_limit = int(
                self._env.manipulation_config.get(
                    "grasp_ik_precheck_candidate_count", 32
                )
            )
            if not 1 <= candidate_limit <= 64:
                raise ValueError(
                    "grasp_ik_precheck_candidate_count must be in [1, 64]"
                )
            plan_arm_poses = getattr(self._env, "plan_arm_poses", None)
            if not callable(plan_arm_poses):
                raise RuntimeError(
                    "grasp IK precheck is enabled but the environment does not "
                    "provide plan_arm_poses()"
                )

            parallel_jaw_symmetry_enabled = self._env.manipulation_config.get(
                "grasp_ik_parallel_jaw_symmetry_enabled",
                DEFAULT_GRASP_IK_PARALLEL_JAW_SYMMETRY_ENABLED,
            )
            if not isinstance(parallel_jaw_symmetry_enabled, bool):
                raise ValueError(
                    "grasp_ik_parallel_jaw_symmetry_enabled must be a boolean"
                )
            if parallel_jaw_symmetry_enabled and candidate_limit > 32:
                raise ValueError(
                    "grasp_ik_precheck_candidate_count must be <= 32 when "
                    "parallel-jaw symmetry planning is enabled"
                )
            candidate_approach_m = float(
                self._env.manipulation_config.get(
                    "grasp_collision_approach_m", 0.10
                )
            )
            if (
                not np.isfinite(candidate_approach_m)
                or not 0.02 <= candidate_approach_m <= 0.25
            ):
                raise ValueError(
                    "grasp_collision_approach_m must be in [0.02, 0.25] meters"
                )

            # Preserve the backend's preference before spending Pi IK work,
            # then test only a bounded top-K. Each raw model gripper-base
            # frame is converted to the controlled Nero endpoint.  A parallel
            # jaw grasp is physically unchanged after a local 180-degree turn
            # around its approach axis, so optionally solve both wrist branches.
            score_order = candidates[
                np.argsort(-scores[candidates], kind="stable")
            ][:candidate_limit]
            variant_records: list[dict[str, Any]] = []
            target_poses_xyz_rpy: list[list[float]] = []
            symmetry_rotation = np.eye(4, dtype=np.float64)
            symmetry_rotation[0, 0] = -1.0
            symmetry_rotation[1, 1] = -1.0
            for local_index, candidate_index in enumerate(score_order):
                arm_from_cgn = arm_from_camera @ grasps[candidate_index]
                symmetry_options = (
                    (False, True)
                    if parallel_jaw_symmetry_enabled
                    else (False,)
                )
                for symmetry_flipped in symmetry_options:
                    variant_cgn = (
                        arm_from_cgn @ symmetry_rotation
                        if symmetry_flipped
                        else arm_from_cgn
                    )
                    arm_target = variant_cgn @ cgn_to_endpoint
                    variant_records.append(
                        {
                            "local_index": local_index,
                            "arm_from_cgn_origin": variant_cgn,
                            "arm_target": arm_target,
                            "symmetry_flipped": symmetry_flipped,
                        }
                    )
                    quaternion = matrix_to_quaternion_wxyz(arm_target[:3, :3])
                    rpy = quaternion_wxyz_to_rpy(quaternion)
                    target_poses_xyz_rpy.append(
                        np.concatenate([arm_target[:3, 3], rpy]).tolist()
                    )

            # The Pi RPC validates an entire batch before solving any target.
            # One out-of-contract pose would therefore discard every otherwise
            # usable candidate. Apply the same protocol bound per variant here
            # and continue with the remaining candidates.
            bounded_records: list[dict[str, Any]] = []
            bounded_poses: list[list[float]] = []
            pi_target_bound_rejected_variant_count = 0
            pi_pregrasp_bound_rejected_variant_count = 0
            for record, target_pose in zip(
                variant_records, target_poses_xyz_rpy
            ):
                arm_target = np.asarray(record["arm_target"])
                target_position = arm_target[:3, 3]
                position_norm_m = float(np.linalg.norm(target_position))
                pregrasp_position = (
                    target_position
                    - candidate_approach_m * arm_target[:3, 2]
                )
                pregrasp_position_norm_m = float(
                    np.linalg.norm(pregrasp_position)
                )
                record["position_norm_m"] = position_norm_m
                record["pregrasp_position"] = pregrasp_position
                record["pregrasp_position_norm_m"] = pregrasp_position_norm_m
                record["near_side_approach"] = (
                    pregrasp_position_norm_m < position_norm_m
                )
                record["within_pi_target_bound"] = (
                    position_norm_m <= PI_MAX_TCP_POSITION_NORM_M
                )
                record["within_pi_pregrasp_bound"] = (
                    pregrasp_position_norm_m <= PI_MAX_TCP_POSITION_NORM_M
                )
                if (
                    record["within_pi_target_bound"]
                    and record["within_pi_pregrasp_bound"]
                ):
                    bounded_records.append(record)
                    bounded_poses.append(target_pose)
                else:
                    if not record["within_pi_target_bound"]:
                        pi_target_bound_rejected_variant_count += 1
                    elif not record["within_pi_pregrasp_bound"]:
                        pi_pregrasp_bound_rejected_variant_count += 1
            pi_bound_rejected_variant_count = (
                pi_target_bound_rejected_variant_count
                + pi_pregrasp_bound_rejected_variant_count
            )
            if not bounded_records:
                raise RuntimeError(
                    "all top grasp candidates exceed the Pi 1.0 m arm-base "
                    "target or pre-grasp position bound"
                )
            variant_records = bounded_records
            target_poses_xyz_rpy = bounded_poses

            if collision_check_enabled:
                clearance_m = float(
                    self._env.manipulation_config.get(
                        "grasp_collision_clearance_m", 0.008
                    )
                )
                approach_samples = int(
                    self._env.manipulation_config.get(
                        "grasp_collision_approach_samples", 6
                    )
                )
                obstacle_points = self._collision_scene_points(
                    depth, mask, intrinsics, arm_from_camera
                )
                collision_check = self._motion_planner_client().check_grasps(
                    np.stack(
                        [record["arm_target"] for record in variant_records], axis=0
                    ),
                    obstacle_points,
                    clearance_m=clearance_m,
                    approach_m=candidate_approach_m,
                    approach_samples=approach_samples,
                )
                safe = collision_check.get("safe")
                minimum_clearance = collision_check.get("minimum_clearance_m")
                if not isinstance(safe, list) or len(safe) != len(variant_records):
                    raise RuntimeError(
                        f"grasp collision service returned invalid result: {collision_check!r}"
                    )
                kept_records: list[dict[str, Any]] = []
                kept_poses: list[list[float]] = []
                for index, (record, target_pose, is_safe) in enumerate(
                    zip(variant_records, target_poses_xyz_rpy, safe)
                ):
                    record["collision_safe"] = bool(is_safe)
                    record["minimum_clearance_m"] = (
                        None
                        if not isinstance(minimum_clearance, list)
                        or index >= len(minimum_clearance)
                        else float(minimum_clearance[index])
                    )
                    if is_safe:
                        kept_records.append(record)
                        kept_poses.append(target_pose)
                if not kept_records:
                    raise RuntimeError(
                        "all top grasp candidates collide with observed non-target geometry"
                    )
                collision_check = dict(collision_check)
                collision_check["input_scene_points"] = int(len(obstacle_points))
                collision_check["safe_variant_count"] = len(kept_records)
                variant_records = kept_records
                target_poses_xyz_rpy = kept_poses

            planning = plan_arm_poses(name, target_poses_xyz_rpy)
            if not isinstance(planning, dict):
                raise TypeError("plan_arm_poses() returned a non-dictionary")
            raw_plans = planning.get("plans")
            if not isinstance(raw_plans, list):
                raise RuntimeError(
                    f"Pi IK precheck returned no candidate plans: {planning!r}"
                )
            valid_plans: list[tuple[int, dict[str, Any]]] = []
            for plan in raw_plans:
                if not isinstance(plan, dict) or not plan.get("success", False):
                    continue
                local_index = plan.get("candidate_index")
                if not isinstance(local_index, int) or not 0 <= local_index < len(
                    variant_records
                ):
                    continue
                position_error = float(plan.get("ik_position_error_m", np.inf))
                rotation_error = float(plan.get("ik_rotation_error_rad", np.inf))
                if not np.isfinite(position_error) or not np.isfinite(
                    rotation_error
                ):
                    continue
                valid_plans.append((local_index, plan))
            evaluated_candidate_count = len(raw_plans)
            if not valid_plans:
                raise RuntimeError(
                    "Pi Mink IK could not produce a finite plan for any of the "
                    f"{len(target_poses_xyz_rpy)} collision-safe "
                    f"{backend_name} variants: {planning!r}"
                )

            plans_by_candidate: dict[int, list[tuple[int, dict[str, Any]]]] = {}
            for variant_index, plan in valid_plans:
                local_index = int(variant_records[variant_index]["local_index"])
                plans_by_candidate.setdefault(local_index, []).append(
                    (variant_index, plan)
                )

            # A sampled grasp is an execution promise, not a best-effort pose.
            # Historical rollouts show that finite/non-converged Mink results,
            # including results inside the old 20 mm / 0.10 rad residual band,
            # are overwhelmingly rejected by cuRobo. Keep only Pi's explicit
            # convergence signal here. Residuals remain diagnostics, not an
            # alternative success criterion.
            candidate_plans: list[tuple[int, int, dict[str, Any]]] = []
            for local_index, variants in plans_by_candidate.items():
                converged_variants = [
                    item
                    for item in variants
                    if bool(item[1].get("ik_converged"))
                ]
                if not converged_variants:
                    continue
                variant_index, plan = min(
                    converged_variants,
                    key=lambda item: (
                        float(item[1].get("joint_travel_l2_rad", np.inf)),
                        float(item[1].get("joint_travel_max_rad", np.inf)),
                        float(item[1].get("ik_position_error_m", np.inf)),
                        float(item[1].get("ik_rotation_error_rad", np.inf)),
                    ),
                )
                candidate_plans.append((local_index, variant_index, plan))

            if not candidate_plans:
                raise RuntimeError(
                    "Pi Mink IK did not converge for any collision-safe grasp "
                    f"variant: finite_plans={len(valid_plans)} "
                    f"evaluated_variants={len(target_poses_xyz_rpy)}"
                )
            # Prefer an approach whose pre-grasp lies on the arm side of the
            # target, then preserve joint margin before model confidence.
            selected_local_index, selected_variant_index, selected_plan = min(
                candidate_plans,
                key=lambda item: (
                    not bool(variant_records[item[1]]["near_side_approach"]),
                    float(
                        variant_records[item[1]]["pregrasp_position_norm_m"]
                    ),
                    float(item[2].get("joint_travel_l2_rad", np.inf)),
                    float(item[2].get("joint_travel_max_rad", np.inf)),
                    -float(scores[score_order[item[0]]]),
                ),
            )
            best_index = int(score_order[selected_local_index])
            selected_record = variant_records[selected_variant_index]
            arm_from_grasp = selected_record["arm_target"]
            goalset_limit = int(
                self._env.manipulation_config.get("grasp_motion_goalset_count", 16)
            )
            if not 1 <= goalset_limit <= 16:
                raise ValueError("grasp_motion_goalset_count must be in [1, 16]")
            ordered_goalset = [
                (selected_local_index, selected_variant_index, selected_plan)
            ] + sorted(
                [
                    item
                    for item in candidate_plans
                    if item[1] != selected_variant_index
                ],
                key=lambda item: (
                    not bool(
                        variant_records[item[1]]["near_side_approach"]
                    ),
                    float(
                        variant_records[item[1]][
                            "pregrasp_position_norm_m"
                        ]
                    ),
                    float(item[2].get("joint_travel_l2_rad", np.inf)),
                    float(item[2].get("joint_travel_max_rad", np.inf)),
                    -float(scores[score_order[item[0]]]),
                ),
            )
            goalset_targets = [
                np.asarray(variant_records[variant_index]["arm_target"]).copy()
                for _, variant_index, _ in ordered_goalset[:goalset_limit]
            ]
            selected_plan = dict(selected_plan)
            selected_plan["ik_quality_acceptable"] = True
            selected_plan["ik_quality_fallback"] = False
            chosen_by_local_index = {
                local_index: (
                    variant_index,
                    {
                        **plan,
                        "ik_quality_acceptable": True,
                        "ik_quality_fallback": False,
                    },
                )
                for local_index, variant_index, plan in candidate_plans
            }
            variants_by_local_index: dict[int, list[dict[str, Any]]] = {}
            for variant_index, plan in valid_plans:
                record = variant_records[variant_index]
                local_index = int(record["local_index"])
                chosen = chosen_by_local_index.get(local_index)
                variants_by_local_index.setdefault(local_index, []).append(
                    {
                        "symmetry_flipped": bool(record["symmetry_flipped"]),
                        "selected_for_candidate": bool(
                            chosen is not None and chosen[0] == variant_index
                        ),
                        "arm_from_grasp": record["arm_target"].copy(),
                        "ik_plan": {
                            **plan,
                            "ik_quality_acceptable": bool(
                                plan.get("ik_converged", False)
                            ),
                            "ik_quality_fallback": False,
                        },
                    }
                )
            for local_index, candidate_index in enumerate(score_order):
                chosen = chosen_by_local_index.get(local_index)
                chosen_record = (
                    None if chosen is None else variant_records[chosen[0]]
                )
                arm_from_model_origin = arm_from_camera @ grasps[candidate_index]
                arm_from_canonical_endpoint = (
                    arm_from_model_origin @ cgn_to_endpoint
                )
                arm_from_symmetric_endpoint = (
                    arm_from_model_origin @ symmetry_rotation @ cgn_to_endpoint
                )
                contact_point_arm = None
                if (
                    contact_points.ndim == 2
                    and contact_points.shape[1] == 3
                    and candidate_index < contact_points.shape[0]
                    and np.all(np.isfinite(contact_points[candidate_index]))
                ):
                    contact_point_arm = (
                        arm_from_camera[:3, :3]
                        @ contact_points[candidate_index]
                        + arm_from_camera[:3, 3]
                    )
                debug_candidates.append(
                    {
                        "candidate_index": int(candidate_index),
                        "score": float(scores[candidate_index]),
                        "object_distance_m": float(
                            object_distances[candidate_index]
                        ),
                        "distance_reference": distance_reference,
                        "arm_from_grasp": (
                            arm_from_canonical_endpoint
                            if chosen_record is None
                            else chosen_record["arm_target"].copy()
                        ),
                        "arm_from_cgn_origin": (
                            arm_from_model_origin
                            if chosen_record is None
                            else chosen_record["arm_from_cgn_origin"].copy()
                        ),
                        # Preserve the backend's unmodified prediction even
                        # when IK selected the 180-degree wrist branch.  Viser
                        # uses the explicit endpoint fields below to compare
                        # canonical and symmetric mappings without relabeling a
                        # transformed pose as "raw".
                        "arm_from_model_origin": arm_from_model_origin.copy(),
                        "arm_from_selected_model_origin": (
                            arm_from_model_origin.copy()
                            if chosen_record is None
                            else chosen_record["arm_from_cgn_origin"].copy()
                        ),
                        "arm_from_canonical_endpoint": (
                            arm_from_canonical_endpoint.copy()
                        ),
                        "arm_from_symmetric_endpoint": (
                            arm_from_symmetric_endpoint.copy()
                        ),
                        "symmetry_flipped": bool(
                            False
                            if chosen_record is None
                            else chosen_record["symmetry_flipped"]
                        ),
                        "position_norm_m": (
                            None
                            if chosen_record is None
                            else float(chosen_record["position_norm_m"])
                        ),
                        "pregrasp_position": (
                            None
                            if chosen_record is None
                            else chosen_record["pregrasp_position"].copy()
                        ),
                        "pregrasp_position_norm_m": (
                            None
                            if chosen_record is None
                            else float(
                                chosen_record["pregrasp_position_norm_m"]
                            )
                        ),
                        "near_side_approach": (
                            None
                            if chosen_record is None
                            else bool(chosen_record["near_side_approach"])
                        ),
                        "contact_point_arm": contact_point_arm,
                        "anchor_point_arm": contact_point_arm,
                        "ik_plan": None if chosen is None else chosen[1],
                        "ik_variants": variants_by_local_index.get(local_index, []),
                        "selected": local_index == selected_local_index,
                    }
                )
        else:
            best_index = int(candidates[np.argmax(scores[candidates])])
            arm_from_cgn_origin = arm_from_camera @ grasps[best_index]
            arm_from_grasp = arm_from_cgn_origin @ cgn_to_endpoint
            goalset_targets = [arm_from_grasp.copy()]
            symmetry_rotation = np.eye(4, dtype=np.float64)
            symmetry_rotation[0, 0] = -1.0
            symmetry_rotation[1, 1] = -1.0
            arm_from_symmetric_endpoint = (
                arm_from_cgn_origin @ symmetry_rotation @ cgn_to_endpoint
            )
            contact_point_arm = None
            if (
                contact_points.ndim == 2
                and contact_points.shape[1] == 3
                and best_index < contact_points.shape[0]
                and np.all(np.isfinite(contact_points[best_index]))
            ):
                contact_point_arm = (
                    arm_from_camera[:3, :3] @ contact_points[best_index]
                    + arm_from_camera[:3, 3]
                )
            debug_candidates.append(
                {
                    "candidate_index": best_index,
                    "score": float(scores[best_index]),
                    "object_distance_m": float(object_distances[best_index]),
                    "distance_reference": distance_reference,
                    "arm_from_grasp": arm_from_grasp.copy(),
                    "arm_from_cgn_origin": arm_from_cgn_origin.copy(),
                    "arm_from_model_origin": arm_from_cgn_origin.copy(),
                    "arm_from_selected_model_origin": arm_from_cgn_origin.copy(),
                    "arm_from_canonical_endpoint": arm_from_grasp.copy(),
                    "arm_from_symmetric_endpoint": (
                        arm_from_symmetric_endpoint.copy()
                    ),
                    "symmetry_flipped": False,
                    "contact_point_arm": contact_point_arm,
                    "anchor_point_arm": contact_point_arm,
                    "ik_plan": None,
                    "ik_variants": [],
                    "selected": True,
                }
            )

        # This is the controlled endpoint target, not CGN's gripper-base origin.
        # The 0.1034 m CGN-origin -> jaw-center convention is handled above;
        # Nero's physical flange -> TCP transform remains owned by the Pi.
        position = arm_from_grasp[:3, 3].copy()
        quaternion = matrix_to_quaternion_wxyz(arm_from_grasp[:3, :3])
        # Keep collision/IK-filtered alternatives behind the tuple-shaped
        # public API for the immediately following goto_grasp_pose. Goalset
        # planning is part of grasp execution and must not depend on whether
        # the optional post-grasp lift primitive is exposed.
        self._sampled_grasp_goalsets[name] = {
            "object_name": str(object_name),
            "selected_tcp_pose": arm_from_grasp.copy(),
            "tcp_poses": np.stack(goalset_targets, axis=0),
        }
        self.last_grasp_debug = {
            "grasp_backend": backend_name,
            "backend_metadata": backend_metadata,
            "arm": name,
            "rgb": rgb.copy(),
            "depth_m": depth.copy(),
            "mask": mask.copy(),
            "intrinsics": intrinsics.copy(),
            "arm_from_camera": np.asarray(arm_from_camera).copy(),
            "candidates": debug_candidates,
            "selected_candidate_index": best_index,
            "ik_precheck_enabled": ik_precheck_enabled,
            "pi_target_position_bound_m": PI_MAX_TCP_POSITION_NORM_M,
            "pi_bound_rejected_variant_count": (
                pi_bound_rejected_variant_count if ik_precheck_enabled else 0
            ),
            "pi_target_bound_rejected_variant_count": (
                pi_target_bound_rejected_variant_count
                if ik_precheck_enabled
                else 0
            ),
            "pi_pregrasp_bound_rejected_variant_count": (
                pi_pregrasp_bound_rejected_variant_count
                if ik_precheck_enabled
                else 0
            ),
            "collision_check_enabled": collision_check_enabled,
            "collision_check": collision_check,
            "controlled_endpoint": endpoint_name,
            "contact_graspnet_origin_to_tcp_m": cgn_tcp_depth_m,
            "grasp_model_origin_to_tcp_m": cgn_tcp_depth_m,
        }
        ik_summary = "IK precheck=disabled"
        if selected_plan is not None:
            ik_summary = (
                f"IK precheck={evaluated_candidate_count} candidates; "
                f"converged={bool(selected_plan.get('ik_converged'))}; "
                f"position residual="
                f"{float(selected_plan['ik_position_error_m']):.4f} m; "
                f"rotation residual="
                f"{float(selected_plan['ik_rotation_error_rad']):.4f} rad; "
                f"quality fallback="
                f"{bool(selected_plan.get('ik_quality_fallback'))}"
            )
        self._log_step_update(
            text=(
                f"SAM3={sam_score:.3f}; grasp score={scores[best_index]:.3f}; "
                f"{distance_reference} distance="
                f"{object_distances[best_index]:.3f} m; "
                f"valid candidates={candidates.size}/{count}; {ik_summary}; arm={name}"
            )
        )
        return position, quaternion

    def _consume_prepared_grasp_certificate(
        self, object_name: str, arm: int | str
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Freshly validate and consume a prepare-for-manipulation goalset."""

        name = self._arm_name(arm)
        certificate = self._prepared_grasp_certificates.get(name)
        if not isinstance(certificate, dict):
            return None
        if certificate.get("object_name") != str(object_name):
            self._prepared_grasp_certificates.pop(name, None)
            return None
        maximum_age_s = float(
            self._env.manipulation_config.get(
                "prepared_grasp_certificate_max_age_s", 60.0
            )
        )
        age_s = time.monotonic() - float(certificate.get("created_monotonic", -np.inf))
        if not np.isfinite(maximum_age_s) or maximum_age_s <= 0.0:
            raise ValueError("prepared_grasp_certificate_max_age_s must be positive")
        if age_s < 0.0 or age_s > maximum_age_s:
            self._prepared_grasp_certificates.pop(name, None)
            return None
        current_base = self._base_pose_snapshot()
        certified_base = np.asarray(certificate.get("base_pose"), dtype=np.float64)
        translation_tolerance_m = float(
            self._env.manipulation_config.get(
                "prepared_grasp_base_translation_tolerance_m", 0.02
            )
        )
        yaw_tolerance_rad = float(
            self._env.manipulation_config.get(
                "prepared_grasp_base_yaw_tolerance_rad", np.deg2rad(1.0)
            )
        )
        if (
            not np.isfinite(translation_tolerance_m)
            or translation_tolerance_m <= 0.0
            or not np.isfinite(yaw_tolerance_rad)
            or yaw_tolerance_rad <= 0.0
        ):
            raise ValueError(
                "prepared grasp base tolerances must be positive and finite"
            )
        if certified_base.shape != (3,) or not np.all(np.isfinite(certified_base)):
            self._prepared_grasp_certificates.pop(name, None)
            return None
        yaw_error = float(
            np.arctan2(
                np.sin(current_base[2] - certified_base[2]),
                np.cos(current_base[2] - certified_base[2]),
            )
        )
        if (
            np.linalg.norm(current_base[:2] - certified_base[:2])
            > translation_tolerance_m
            or abs(yaw_error) > yaw_tolerance_rad
        ):
            self._prepared_grasp_certificates.pop(name, None)
            return None
        current_joints = self._arm_joint_snapshot(name)
        certified_joints = np.asarray(certificate.get("joint_pos"), dtype=np.float64)
        joint_tolerance_rad = float(
            self._env.manipulation_config.get(
                "prepared_grasp_joint_tolerance_rad", 0.02
            )
        )
        if not np.isfinite(joint_tolerance_rad) or joint_tolerance_rad <= 0.0:
            raise ValueError(
                "prepared_grasp_joint_tolerance_rad must be positive and finite"
            )
        if certified_joints.shape != (7,) or not np.all(
            np.isfinite(certified_joints)
        ):
            self._prepared_grasp_certificates.pop(name, None)
            return None
        if np.max(np.abs(current_joints - certified_joints)) > joint_tolerance_rad:
            self._prepared_grasp_certificates.pop(name, None)
            return None

        observed_name, rgb, depth, intrinsics, arm_from_camera = (
            self._perception_input(name)
        )
        if observed_name != name:
            raise RuntimeError("perception returned the wrong arm calibration")
        mask, sam_score = self._object_mask(object_name, rgb)
        target_camera = self._masked_points(
            depth,
            mask,
            intrinsics,
            minimum_points=self._minimum_metric_depth_points(),
        )
        target_arm = (
            target_camera @ arm_from_camera[:3, :3].T
            + arm_from_camera[:3, 3]
        )
        goals = np.asarray(certificate.get("tcp_poses"), dtype=np.float64)
        selected = np.asarray(
            certificate.get("selected_tcp_pose"), dtype=np.float64
        )
        tolerance_m = float(
            self._env.manipulation_config.get(
                "grasp_execution_target_tolerance_m", 0.12
            )
        )
        if not np.isfinite(tolerance_m) or not 0.02 <= tolerance_m <= 0.30:
            raise ValueError(
                "grasp_execution_target_tolerance_m must be in [0.02, 0.30]"
            )
        if (
            goals.ndim != 3
            or goals.shape[1:] != (4, 4)
            or selected.shape != (4, 4)
        ):
            self._prepared_grasp_certificates.pop(name, None)
            return None
        distances = np.asarray(
            [
                np.min(np.linalg.norm(target_arm - goal[:3, 3], axis=1))
                for goal in goals
            ],
            dtype=np.float64,
        )
        goals = goals[distances <= tolerance_m]
        if len(goals) == 0 or not np.allclose(goals[0], selected, atol=1e-5):
            self._prepared_grasp_certificates.pop(name, None)
            return None
        self._sampled_grasp_goalsets[name] = {
            "object_name": str(object_name),
            "selected_tcp_pose": selected.copy(),
            "tcp_poses": goals.copy(),
        }
        self._prepared_grasp_certificates.pop(name, None)
        self.last_grasp_debug = {
            "grasp_backend": "prepared_certificate",
            "arm": name,
            "rgb": np.asarray(rgb).copy(),
            "depth_m": np.asarray(depth).copy(),
            "mask": np.asarray(mask).copy(),
            "intrinsics": np.asarray(intrinsics).copy(),
            "arm_from_camera": np.asarray(arm_from_camera).copy(),
            "sam_score": float(sam_score),
            "certificate_age_s": float(age_s),
            "goalset_candidate_count": int(len(goals)),
        }
        return (
            selected[:3, 3].copy(),
            matrix_to_quaternion_wxyz(selected[:3, :3]),
        )

    def goto_grasp_pose(
        self,
        object_name: str,
        position: np.ndarray,
        quaternion_wxyz: np.ndarray,
        arm: int | str,
        *,
        approach_m: float = 0.10,
        timeout_s: float = 60.0,
    ) -> dict[str, Any]:
        """Plan and execute a fresh-scene, whole-arm collision-free grasp path."""

        name = self._arm_name(arm)
        filter_context = self._pending_self_filter_grasps.get(name)
        if name in self._attached_objects or (
            filter_context is not None
            and bool(filter_context.get("gripper_closed", False))
        ):
            return {
                "success": False,
                "reason": "open_gripper_required_before_new_grasp",
                "primitive": "goto_grasp_pose",
                "arm": name,
            }
        # Any new grasp attempt supersedes the previous arm-local retreat. A
        # failed or partial attempt must never leave an older path executable.
        self._pending_grasp_retreats.pop(name, None)
        self._pending_grasps.pop(name, None)
        self._pending_self_filter_grasps.pop(name, None)
        target = pose_matrix(position, quaternion_wxyz)
        approach_m = float(approach_m)
        timeout_s = float(timeout_s)
        if not np.isfinite(approach_m) or not 0.02 <= approach_m <= 0.25:
            raise ValueError("approach_m must be in [0.02, 0.25] meters")
        if not np.isfinite(timeout_s) or timeout_s <= 0.0:
            raise ValueError("timeout_s must be positive and finite")

        # Reacquire both the target segmentation and the obstacle scene at
        # execution time. A stale sample-time map is not a motion-safety map.
        observed_name, rgb, depth, intrinsics, arm_from_camera = (
            self._perception_input(name)
        )
        if observed_name != name:
            raise RuntimeError("perception returned the wrong arm calibration")
        target_mask, sam_score = self._object_mask(object_name, rgb)
        target_camera = self._masked_points(
            depth,
            target_mask,
            intrinsics,
            minimum_points=self._minimum_metric_depth_points(),
        )
        target_arm = (
            target_camera @ arm_from_camera[:3, :3].T
            + arm_from_camera[:3, 3]
        )
        target_distance_m = float(
            np.min(np.linalg.norm(target_arm - target[:3, 3], axis=1))
        )
        target_tolerance_m = float(
            self._env.manipulation_config.get(
                "grasp_execution_target_tolerance_m", 0.12
            )
        )
        if (
            not np.isfinite(target_tolerance_m)
            or not 0.02 <= target_tolerance_m <= 0.30
        ):
            raise ValueError(
                "grasp_execution_target_tolerance_m must be in [0.02, 0.30]"
            )
        if target_distance_m > target_tolerance_m:
            return {
                "success": False,
                "reason": "requested_grasp_is_not_on_fresh_target_instance",
                "arm": name,
                "target_distance_m": target_distance_m,
                "target_tolerance_m": target_tolerance_m,
                "sam_score": sam_score,
            }
        goalset_targets = np.asarray(target[None], dtype=np.float64)
        cached_goalset = self._sampled_grasp_goalsets.get(name)
        if (
            isinstance(cached_goalset, dict)
            and cached_goalset.get("object_name") == str(object_name)
        ):
            cached_selected = np.asarray(
                cached_goalset.get("selected_tcp_pose"), dtype=np.float64
            )
            cached_targets = np.asarray(cached_goalset.get("tcp_poses"), dtype=np.float64)
            if (
                cached_selected.shape == (4, 4)
                and cached_targets.ndim == 3
                and cached_targets.shape[1:] == (4, 4)
                and np.allclose(cached_selected, target, atol=1e-5)
            ):
                fresh_distances = np.asarray(
                    [
                        np.min(
                            np.linalg.norm(
                                target_arm - candidate[:3, 3], axis=1
                            )
                        )
                        for candidate in cached_targets
                    ]
                )
                goalset_targets = cached_targets[
                    fresh_distances <= target_tolerance_m
                ]
                if len(goalset_targets) == 0:
                    goalset_targets = np.asarray(target[None], dtype=np.float64)
        obstacle_points = self._collision_scene_points(
            depth, target_mask, intrinsics, arm_from_camera
        )

        current_joints = self._arm_joint_snapshot(name)

        # sample_grasp_pose selected a collision-safe grasp target, but its
        # tuple-shaped public API intentionally does not expose joints. Re-solve
        # the exact target from fresh measured joints. A converged Pi solution
        # is sent for an independent cuRobo FK/frame-consistency check. A finite
        # best-effort Pi solution that did not converge is diagnostic only:
        # cuRoboV2 owns final grasp feasibility and its internal IK seeds.
        (
            goal_joint_seed,
            pi_seed_plan,
            pi_nonconverged_seed_available,
            pi_seed_result,
        ) = self._pi_seed_for_tcp(
            name, target
        )
        if goal_joint_seed is None and not pi_nonconverged_seed_available:
            pi_ik_reason = None
            if isinstance(pi_seed_plan, dict):
                pi_ik_reason = pi_seed_plan.get("reason")
            elif isinstance(pi_seed_result, dict):
                pi_ik_reason = pi_seed_result.get("reason")
            return {
                "success": False,
                "reason": "pi_grasp_goal_joint_seed_unavailable",
                "primitive": "goto_grasp_pose",
                "arm": name,
                "pi_ik_reason": pi_ik_reason,
                "sam_score": sam_score,
                "target_distance_m": target_distance_m,
            }
        pi_goal_joint_seed_forwarded = goal_joint_seed is not None
        grasp_ik_position_tolerance_m = float(
            self._env.manipulation_config.get(
                "grasp_ik_acceptable_position_error_m",
                DEFAULT_GRASP_IK_ACCEPTABLE_POSITION_ERROR_M,
            )
        )
        grasp_ik_rotation_tolerance_rad = float(
            self._env.manipulation_config.get(
                "grasp_ik_acceptable_rotation_error_rad",
                DEFAULT_GRASP_IK_ACCEPTABLE_ROTATION_ERROR_RAD,
            )
        )

        plan = self.plan_grasp_goalset(
            current_joints,
            goalset_targets,
            obstacle_points,
            approach_m=approach_m,
            goal_joint_seed=goal_joint_seed,
        )
        if not plan.get("success", False):
            return {
                **plan,
                "primitive": "goto_grasp_pose",
                "arm": name,
                "goalset_candidate_count": int(len(goalset_targets)),
                "pi_goal_ik_converged": bool(
                    pi_seed_plan.get("ik_converged", False)
                ),
                "pi_goal_ik_position_error_m": pi_seed_plan.get(
                    "ik_position_error_m"
                ),
                "pi_goal_ik_rotation_error_rad": pi_seed_plan.get(
                    "ik_rotation_error_rad"
                ),
                "grasp_ik_position_tolerance_m": grasp_ik_position_tolerance_m,
                "grasp_ik_rotation_tolerance_rad": grasp_ik_rotation_tolerance_rad,
                "pi_goal_joint_seed_forwarded": pi_goal_joint_seed_forwarded,
                "sam_score": sam_score,
                "target_distance_m": target_distance_m,
            }
        planned_target = np.asarray(
            plan.get("selected_tcp_pose", target), dtype=np.float64
        )
        if planned_target.shape != (4, 4) or not np.all(np.isfinite(planned_target)):
            raise RuntimeError("grasp-motion planner returned an invalid selected pose")
        waypoints = np.asarray(plan.get("waypoints"), dtype=np.float64)
        if (
            waypoints.ndim != 2
            or waypoints.shape[1] != 7
            or not np.all(np.isfinite(waypoints))
        ):
            raise RuntimeError("grasp-motion planner returned invalid waypoints")
        # Never decimate the constrained local-Z contact segment (nor the
        # pre-grasp sample that starts it); only the free-space approach may
        # be executed with fewer of its planner samples.
        protected_tail_count = 0
        segments = plan.get("segments")
        if isinstance(segments, list):
            for segment in segments:
                if isinstance(segment, dict) and segment.get("phase") == "grasp":
                    grasp_waypoints = segment.get("waypoints") or []
                    protected_tail_count = min(
                        len(waypoints), len(grasp_waypoints) + 1
                    )
        execution = self._execute_planned_trajectory(
            name,
            waypoints,
            timeout_s,
            execution_step_rad=self._execution_step_cap(),
            protected_tail_count=protected_tail_count,
        )
        retreat = None
        attached_diagnostics: dict[str, Any] = {}
        if execution.get("success", False):
            retreat = self._reverse_grasp_segment(plan)
            if retreat is not None:
                self._pending_grasp_retreats[name] = retreat
            object_padding_m = float(
                self._env.manipulation_config.get(
                    "grasp_attached_object_padding_m", 0.005
                )
            )
            if not 0.0 <= object_padding_m <= 0.03:
                raise ValueError(
                    "grasp_attached_object_padding_m must be in [0, 0.03]"
                )
            attach_radius_m = float(
                self._env.manipulation_config.get(
                    "grasp_attached_object_radius_m",
                    ATTACHED_OBJECT_DEFAULT_RADIUS_M,
                )
            )
            if not 0.05 <= attach_radius_m <= 0.50:
                raise ValueError(
                    "grasp_attached_object_radius_m must be in [0.05, 0.50]"
                )
            target_local = (
                target_arm - planned_target[:3, 3]
            ) @ planned_target[:3, :3]
            # Only points near the grasp can be part of the held object. The
            # SAM3 mask may include the support surface, background or
            # silhouette flying pixels; without this filter a stray point turns
            # the attached box into something the planner rejects (2026-09-07:
            # "attached_bounds_local must have positive extents no larger than
            # 0.40 m" while holding a 0.12 m can).
            distances_m = np.linalg.norm(target_local, axis=1)
            near = distances_m <= attach_radius_m
            attached_point_total = int(len(target_local))
            attached_point_count = int(np.count_nonzero(near))
            radius_fallback = attached_point_count == 0
            if not radius_fallback:
                target_local = target_local[near]
            bounds_local = np.stack(
                [
                    np.min(target_local, axis=0) - object_padding_m,
                    np.max(target_local, axis=0) + object_padding_m,
                ]
            )
            minimum_extent_m = 0.01
            clamped_axes: list[int] = []
            extent = bounds_local[1] - bounds_local[0]
            for axis in range(3):
                if extent[axis] < minimum_extent_m:
                    midpoint = 0.5 * (
                        bounds_local[0, axis] + bounds_local[1, axis]
                    )
                    bounds_local[0, axis] = midpoint - minimum_extent_m / 2.0
                    bounds_local[1, axis] = midpoint + minimum_extent_m / 2.0
                elif extent[axis] > ATTACHED_OBJECT_MAX_EXTENT_M:
                    # The grasp-motion planner rejects larger boxes outright;
                    # keep a planner-sized box centred on the observed points.
                    midpoint = 0.5 * (
                        bounds_local[0, axis] + bounds_local[1, axis]
                    )
                    bounds_local[0, axis] = midpoint - ATTACHED_OBJECT_MAX_EXTENT_M / 2.0
                    bounds_local[1, axis] = midpoint + ATTACHED_OBJECT_MAX_EXTENT_M / 2.0
                    clamped_axes.append(axis)
            attached_diagnostics = {
                "attached_extent_m": (bounds_local[1] - bounds_local[0]).tolist(),
                "attached_point_count": attached_point_count,
                "attached_point_total": attached_point_total,
                "attached_radius_m": attach_radius_m,
                "attached_radius_fallback": radius_fallback,
                "attached_clamped_axes": clamped_axes,
            }
            context = {
                "object_name": str(object_name),
                "tcp_pose": planned_target.copy(),
                "attached_bounds_local": bounds_local,
                "gripper_closed": False,
                "lift_completed": False,
            }
            self._pending_self_filter_grasps[name] = dict(context)
            if self._attached_lift_enabled:
                self._pending_grasps[name] = {
                    **context,
                }
        final_tcp = None
        final_flange = None
        final_joints = None
        final_position_error_m = None
        final_rotation_error_rad = None
        if execution.get("success", False):
            final_status = self._env.arm_status()
            final_snapshot = (
                final_status.get(name) if isinstance(final_status, dict) else None
            )
            if isinstance(final_snapshot, dict):
                candidate_joints = np.asarray(
                    final_snapshot.get("joint_pos"), dtype=np.float64
                )
                if candidate_joints.shape == (7,) and np.all(
                    np.isfinite(candidate_joints)
                ):
                    final_joints = candidate_joints
                candidate_flange = np.asarray(
                    final_snapshot.get("flange_pose_xyz_rpy"), dtype=np.float64
                )
                if candidate_flange.shape == (6,) and np.all(
                    np.isfinite(candidate_flange)
                ):
                    final_flange = candidate_flange
                candidate_tcp = np.asarray(
                    final_snapshot.get("tcp_pose_xyz_rpy"), dtype=np.float64
                )
                if candidate_tcp.shape == (6,) and np.all(np.isfinite(candidate_tcp)):
                    final_tcp = candidate_tcp
                    final_position_error_m = float(
                        np.linalg.norm(final_tcp[:3] - planned_target[:3, 3])
                    )
                    final_quaternion = rpy_to_quaternion_wxyz(final_tcp[3:])
                    target_quaternion = matrix_to_quaternion_wxyz(
                        planned_target[:3, :3]
                    )
                    quaternion_dot = float(
                        np.clip(
                            abs(np.dot(final_quaternion, target_quaternion)),
                            0.0,
                            1.0,
                        )
                    )
                    final_rotation_error_rad = float(
                        2.0 * np.arccos(quaternion_dot)
                    )
        result = {
            **execution,
            "primitive": "goto_grasp_pose",
            "arm": name,
            "planner_reason": plan.get("reason"),
            **self._planner_diagnostics(plan),
            "planning_scene_points": plan.get("planning_scene_points"),
            "removed_robot_points": plan.get("removed_robot_points"),
            "trajectory_dt_s": plan.get("trajectory_dt_s"),
            "trajectory_waypoint_count": len(waypoints),
            "terminal_check": plan.get("terminal_check"),
            "goal_joint_seed_used": plan.get("goal_joint_seed_used", False),
            "goal_joint_seed_verified": plan.get(
                "goal_joint_seed_verified", False
            ),
            "pi_goal_ik_converged": bool(pi_seed_plan.get("ik_converged", False)),
            "pi_goal_ik_position_error_m": pi_seed_plan.get(
                "ik_position_error_m"
            ),
            "pi_goal_ik_rotation_error_rad": pi_seed_plan.get(
                "ik_rotation_error_rad"
            ),
            "pi_goal_joint_seed_forwarded": pi_goal_joint_seed_forwarded,
            "goal_joint_seed_fk_position_error_m": plan.get(
                "goal_joint_seed_fk_position_error_m"
            ),
            "goal_joint_seed_fk_rotation_error_rad": plan.get(
                "goal_joint_seed_fk_rotation_error_rad"
            ),
            "planned_endpoint_position_error_m": plan.get(
                "planned_endpoint_position_error_m"
            ),
            "planned_endpoint_rotation_error_rad": plan.get(
                "planned_endpoint_rotation_error_rad"
            ),
            "final_tcp_xyz_rpy": None if final_tcp is None else final_tcp.tolist(),
            "final_flange_xyz_rpy": (
                None if final_flange is None else final_flange.tolist()
            ),
            "final_joint_pos": (
                None if final_joints is None else final_joints.tolist()
            ),
            "final_tcp_position_error_m": final_position_error_m,
            "final_tcp_rotation_error_rad": final_rotation_error_rad,
            "home_retreat_available": retreat is not None,
            "home_retreat_waypoint_count": (
                None if retreat is None else int(len(retreat))
            ),
            "sam_score": sam_score,
            "target_distance_m": target_distance_m,
            **attached_diagnostics,
            "goalset_candidate_count": int(len(goalset_targets)),
            "selected_goalset_index": plan.get("selected_goalset_index", 0),
            "selected_tcp_pose": planned_target.tolist(),
        }
        if self._attached_lift_enabled:
            result["attached_lift_required_after_close"] = bool(
                execution.get("success", False)
            )
        return result

    def _attached_object_exclusion_mask(
        self,
        depth: np.ndarray,
        intrinsics: np.ndarray,
        arm_from_camera: np.ndarray,
        arm_from_tcp: np.ndarray,
        bounds_local: np.ndarray,
    ) -> np.ndarray:
        """Mask depth belonging to the rigidly carried object prediction."""

        clearance_m = float(
            self._env.manipulation_config.get(
                "grasp_attached_scene_clearing_padding_m", 0.015
            )
        )
        if not 0.0 <= clearance_m <= 0.05:
            raise ValueError(
                "grasp_attached_scene_clearing_padding_m must be in [0, 0.05]"
            )
        valid = self._valid_metric_depth(depth, np.ones(depth.shape, dtype=bool))
        rows, columns = np.nonzero(valid)
        z = depth[rows, columns].astype(np.float64)
        points_camera = np.column_stack(
            [
                (columns - intrinsics[0, 2]) * z / intrinsics[0, 0],
                (rows - intrinsics[1, 2]) * z / intrinsics[1, 1],
                z,
            ]
        )
        points_arm = (
            points_camera @ arm_from_camera[:3, :3].T
            + arm_from_camera[:3, 3]
        )
        points_local = (
            points_arm - arm_from_tcp[:3, 3]
        ) @ arm_from_tcp[:3, :3]
        lower = bounds_local[0] - clearance_m
        upper = bounds_local[1] + clearance_m
        carried = np.all((points_local >= lower) & (points_local <= upper), axis=1)
        mask = np.zeros(depth.shape, dtype=bool)
        mask[rows[carried], columns[carried]] = True
        return mask

    def lift_grasped_object(
        self,
        arm: int | str,
        *,
        lift_m: float = 0.15,
        timeout_s: float = 60.0,
    ) -> dict[str, Any]:
        """Attach the closed-gripper target model, then plan and execute lift."""

        name = self._arm_name(arm)
        if not self._attached_lift_enabled:
            return {
                "success": False,
                "reason": "attached_lift_feature_disabled",
                "primitive": "lift_grasped_object",
                "arm": name,
            }
        lift_m = float(lift_m)
        timeout_s = float(timeout_s)
        if not np.isfinite(lift_m) or not 0.02 <= lift_m <= 0.25:
            raise ValueError("lift_m must be in [0.02, 0.25] meters")
        if not np.isfinite(timeout_s) or timeout_s <= 0.0:
            raise ValueError("timeout_s must be positive and finite")

        context = self._pending_grasps.get(name)
        if context is None:
            return {
                "success": False,
                "reason": "no_pending_grasp_to_lift",
                "primitive": "lift_grasped_object",
                "arm": name,
            }
        if not bool(context.get("gripper_closed", False)):
            return {
                "success": False,
                "reason": "close_gripper_required_before_attached_lift",
                "primitive": "lift_grasped_object",
                "arm": name,
            }

        observed_name, _, depth, intrinsics, arm_from_camera = (
            self._perception_input(name)
        )
        if observed_name != name:
            raise RuntimeError("perception returned the wrong arm calibration")
        status = self._env.arm_status()
        if not isinstance(status, dict) or bool(status.get("estop_latched", False)):
            raise RuntimeError("Pi arm RPC is unavailable or emergency-stopped")
        snapshot = status.get(name)
        if not isinstance(snapshot, dict):
            raise RuntimeError(f"arm RPC has no {name} status snapshot")
        current_joints = np.asarray(snapshot.get("joint_pos"), dtype=np.float64)
        tcp_xyz_rpy = np.asarray(snapshot.get("tcp_pose_xyz_rpy"), dtype=np.float64)
        if current_joints.shape != (7,) or not np.all(np.isfinite(current_joints)):
            raise RuntimeError(f"{name} arm has no valid seven-joint feedback")
        if tcp_xyz_rpy.shape != (6,) or not np.all(np.isfinite(tcp_xyz_rpy)):
            # Test transports and older Pi snapshots may omit TCP feedback. The
            # planned grasp TCP remains a bounded fallback, while cuRobo still
            # starts from measured joints and validates its own FK.
            arm_from_tcp = np.asarray(context["tcp_pose"], dtype=np.float64)
        else:
            arm_from_tcp = pose_matrix(
                tcp_xyz_rpy[:3], rpy_to_quaternion_wxyz(tcp_xyz_rpy[3:])
            )
        bounds_local = np.asarray(
            context["attached_bounds_local"], dtype=np.float64
        )
        exclusion_mask = self._attached_object_exclusion_mask(
            depth,
            intrinsics,
            arm_from_camera,
            arm_from_tcp,
            bounds_local,
        )
        obstacle_points = self._collision_scene_points(
            depth, exclusion_mask, intrinsics, arm_from_camera
        )
        scene_voxel_m = float(
            self._env.manipulation_config.get("grasp_motion_scene_voxel_m", 0.012)
        )
        plan = self._motion_planner_client().plan_attached_lift(
            current_joints,
            obstacle_points,
            bounds_local,
            lift_m=lift_m,
            scene_voxel_m=scene_voxel_m,
            planner_attempts=self._planner_attempt_options(),
        )
        if not plan.get("success", False):
            return {
                **plan,
                "primitive": "lift_grasped_object",
                "arm": name,
                "object_name": context.get("object_name"),
            }
        waypoints = np.asarray(plan.get("waypoints"), dtype=np.float64)
        if (
            waypoints.ndim != 2
            or waypoints.shape[1] != 7
            or not np.all(np.isfinite(waypoints))
        ):
            raise RuntimeError("grasp-motion planner returned invalid lift waypoints")
        execution = self._execute_planned_trajectory(name, waypoints, timeout_s)
        if execution.get("success", False):
            context["lift_completed"] = True
            context["lift_m"] = lift_m
            self._attached_objects[name] = dict(context)
            self._pending_grasp_retreats.pop(name, None)
        result = {
            **execution,
            "primitive": "lift_grasped_object",
            "arm": name,
            "object_name": context.get("object_name"),
            "planner_reason": plan.get("reason"),
            **self._planner_diagnostics(plan),
            "planning_scene_points": plan.get("planning_scene_points"),
            "removed_robot_points": plan.get("removed_robot_points"),
            "trajectory_dt_s": plan.get("trajectory_dt_s"),
            "trajectory_waypoint_count": len(waypoints),
            "attached_object_sphere_count": plan.get(
                "attached_object_sphere_count"
            ),
            "lift_m": lift_m,
            "attached_object_collision_enabled": True,
        }
        return result

    def _goto_home(
        self,
        arm: int | str,
        *,
        timeout_s: float = 60.0,
    ) -> dict[str, Any]:
        """Plan and execute a fresh-scene collision-free path to Nero home."""

        name = self._arm_name(arm)
        timeout_s = float(timeout_s)
        if not np.isfinite(timeout_s) or timeout_s <= 0.0:
            raise ValueError("timeout_s must be positive and finite")

        pending = self._pending_grasps.get(name)
        if (
            pending is not None
            and bool(pending.get("gripper_closed", False))
            and not bool(pending.get("lift_completed", False))
        ):
            return {
                "success": False,
                "reason": "attached_lift_required_before_home",
                "primitive": "goto_pose",
                "target": "home",
                "arm": name,
            }
        if pending is not None:
            # An open-gripper home request abandons the contact pose. Never let
            # a later close attach geometry from that stale grasp endpoint.
            self._pending_grasps.pop(name, None)
        attached = self._attached_objects.get(name)

        retreat_report: dict[str, Any] = {
            "available": name in self._pending_grasp_retreats,
            "executed": False,
        }
        cached_retreat = (
            None
            if attached is not None
            else self._pending_grasp_retreats.get(name)
        )
        if cached_retreat is not None:
            status = self._env.arm_status()
            if not isinstance(status, dict) or bool(
                status.get("estop_latched", False)
            ):
                raise RuntimeError("Pi arm RPC is unavailable or emergency-stopped")
            snapshot = status.get(name)
            if not isinstance(snapshot, dict):
                raise RuntimeError(f"arm RPC has no {name} status snapshot")
            measured = np.asarray(snapshot.get("joint_pos"), dtype=np.float64)
            if measured.shape != (7,) or not np.all(np.isfinite(measured)):
                raise RuntimeError(f"{name} arm has no valid seven-joint feedback")
            retreat_start_error = float(
                np.max(np.abs(cached_retreat[0] - measured))
            )
            retreat_report["start_error_max_rad"] = retreat_start_error
            retreat_report["waypoint_count"] = int(len(cached_retreat))
            # Consume the path before motion. If execution stops part-way, its
            # original start state is no longer valid and it must not be retried.
            self._pending_grasp_retreats.pop(name, None)
            if retreat_start_error <= 0.02:
                retreat_execution = self._execute_planned_trajectory(
                    name, cached_retreat, timeout_s
                )
                retreat_report.update(retreat_execution)
                retreat_report["executed"] = bool(
                    retreat_execution.get("success", False)
                )
                if not retreat_execution.get("success", False):
                    return {
                        "success": False,
                        "reason": "grasp_retreat_execution_failed",
                        "primitive": "goto_pose",
                        "target": "home",
                        "arm": name,
                        "grasp_retreat": retreat_report,
                    }
            else:
                retreat_report["skipped_reason"] = (
                    "cached_grasp_retreat_start_mismatch"
                )

        # Only after leaving intentional target contact do we refresh the
        # complete scene. The former target is now preserved as an obstacle for
        # the ordinary collision-free plan from pre-grasp to home.
        observed_name, _, depth, intrinsics, arm_from_camera = (
            self._perception_input(name)
        )
        if observed_name != name:
            raise RuntimeError("perception returned the wrong arm calibration")
        status = self._env.arm_status()
        if not isinstance(status, dict) or bool(status.get("estop_latched", False)):
            raise RuntimeError("Pi arm RPC is unavailable or emergency-stopped")
        snapshot = status.get(name)
        if not isinstance(snapshot, dict):
            raise RuntimeError(f"arm RPC has no {name} status snapshot")
        current_joints = np.asarray(snapshot.get("joint_pos"), dtype=np.float64)
        if current_joints.shape != (7,) or not np.all(np.isfinite(current_joints)):
            raise RuntimeError(f"{name} arm has no valid seven-joint feedback")
        attached_bounds = None
        exclusion_mask = np.zeros(depth.shape, dtype=bool)
        if attached is not None:
            attached_bounds = np.asarray(
                attached.get("attached_bounds_local"), dtype=np.float64
            )
            tcp_xyz_rpy = np.asarray(
                snapshot.get("tcp_pose_xyz_rpy"), dtype=np.float64
            )
            if tcp_xyz_rpy.shape == (6,) and np.all(np.isfinite(tcp_xyz_rpy)):
                arm_from_tcp = pose_matrix(
                    tcp_xyz_rpy[:3], rpy_to_quaternion_wxyz(tcp_xyz_rpy[3:])
                )
            else:
                arm_from_tcp = np.asarray(attached["tcp_pose"], dtype=np.float64)
            exclusion_mask = self._attached_object_exclusion_mask(
                depth,
                intrinsics,
                arm_from_camera,
                arm_from_tcp,
                attached_bounds,
            )
        obstacle_points = self._collision_scene_points(
            depth,
            exclusion_mask,
            intrinsics,
            arm_from_camera,
        )
        target_joints = NERO_HOME_JOINTS_RAD[name]
        scene_voxel_m = float(
            self._env.manipulation_config.get("grasp_motion_scene_voxel_m", 0.012)
        )
        plan = self._motion_planner_client().plan_joint_target(
            current_joints,
            target_joints,
            obstacle_points,
            scene_voxel_m=scene_voxel_m,
            attached_bounds_local=attached_bounds,
            planner_attempts=self._planner_attempt_options(),
        )
        if not plan.get("success", False):
            return {
                **plan,
                "primitive": "goto_pose",
                "target": "home",
                "arm": name,
                "target_joints": target_joints.tolist(),
                "grasp_retreat": retreat_report,
            }
        waypoints = np.asarray(plan.get("waypoints"), dtype=np.float64)
        if (
            waypoints.ndim != 2
            or waypoints.shape[1] != 7
            or not np.all(np.isfinite(waypoints))
        ):
            raise RuntimeError("grasp-motion planner returned invalid home waypoints")
        if float(np.max(np.abs(waypoints[-1] - target_joints))) > 0.02:
            raise RuntimeError("grasp-motion planner did not terminate at Nero home")
        execution = self._execute_planned_trajectory(
            name,
            waypoints,
            timeout_s,
            execution_step_rad=self._execution_step_cap(),
        )
        result = {
            **execution,
            "primitive": "goto_pose",
            "target": "home",
            "arm": name,
            "planner_reason": plan.get("reason"),
            **self._planner_diagnostics(plan),
            "planning_scene_points": plan.get("planning_scene_points"),
            "removed_robot_points": plan.get("removed_robot_points"),
            "trajectory_dt_s": plan.get("trajectory_dt_s"),
            "trajectory_waypoint_count": len(waypoints),
            "maximum_step_rad": plan.get("maximum_step_rad"),
            "goal_error_rad": plan.get("goal_error_rad"),
            "target_joints": target_joints.tolist(),
            "grasp_retreat": retreat_report,
        }
        if attached is not None:
            result.update(
                {
                    "attached_object_collision_enabled": True,
                    "object_name": attached.get("object_name"),
                }
            )
        return result

    def goto_pose(
        self,
        position: np.ndarray | Literal["home", "current"],
        quaternion_wxyz: np.ndarray | None = None,
        *,
        arm: int | str,
        z_approach: float = 0.0,
        timeout_s: float = 60.0,
        camera_offset_xyz: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Move one Nero TCP to an absolute pose, relative offset, or home.

        Args:
            position: Target XYZ meters, ``"current"`` for a leveled ZED-frame
                offset from measured TCP feedback, or ``"home"``.
            quaternion_wxyz: Target WXYZ unit quaternion for Cartesian mode;
                omit it when position is ``"home"``.
            arm: Nero arm identifier.
            z_approach: Optional Cartesian approach distance. It must remain
                zero in ``"home"`` mode.
            timeout_s: Per-command timeout, capped by the Pi service.
            camera_offset_xyz: Leveled ZED [right, down, forward] offset used
                only by ``"current"`` mode.

        Returns:
            Monitored primitive-result dictionary. After ``goto_grasp_pose``,
            ``"home"`` mode first reverses its already checked final approach,
            then refreshes the complete RGB-D scene and uses cuRobo without a
            direct-home fallback.
        """

        name = self._arm_name(arm)
        z_approach = float(z_approach)
        if not np.isfinite(z_approach) or not 0.0 <= z_approach <= 0.25:
            raise ValueError("z_approach must be in [0, 0.25] meters")
        relative_details: dict[str, Any] = {}
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
            if position == "home":
                if camera_offset_xyz is not None:
                    raise ValueError(
                        "camera_offset_xyz must be omitted for goto_pose('home')"
                    )
                return self._goto_home(name, timeout_s=timeout_s)

            offset = np.asarray(camera_offset_xyz, dtype=np.float64)
            if offset.shape != (3,) or not np.all(np.isfinite(offset)):
                raise ValueError(
                    "camera_offset_xyz must contain three finite meters"
                )
            offset_norm = float(np.linalg.norm(offset))
            if not 0.0 < offset_norm <= 0.25:
                raise ValueError(
                    "camera_offset_xyz norm must be in (0, 0.25] meters"
                )

            observation = self._env.observe()
            camera = observation.get("robot0_robotview")
            if not isinstance(camera, dict):
                raise RuntimeError("fresh ZED observation is unavailable")
            ground = camera.get("ground_plane")
            if not isinstance(ground, dict) or not bool(ground.get("valid", False)):
                raise RuntimeError(
                    "fresh ZED ground plane is required for camera-relative motion"
                )
            down_camera = np.asarray(
                ground.get("down_camera_xyz"), dtype=np.float64
            )
            down_norm = float(np.linalg.norm(down_camera))
            if (
                down_camera.shape != (3,)
                or not np.all(np.isfinite(down_camera))
                or not down_norm > 1e-6
            ):
                raise RuntimeError("ZED ground-plane down axis is invalid")
            down_camera /= down_norm
            frame_timestamp = camera.get("timestamp_ns")
            ground_timestamp = ground.get("timestamp_ns")
            if frame_timestamp is None or ground_timestamp is None:
                raise RuntimeError("ZED ground-plane timestamp is unavailable")
            ground_age_s = max(
                0.0, (int(frame_timestamp) - int(ground_timestamp)) * 1e-9
            )
            if ground_age_s > 2.0:
                raise RuntimeError(
                    f"ZED ground plane is stale ({ground_age_s:.3f} s)"
                )

            optical_forward = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
            forward_camera = optical_forward - down_camera * float(
                np.dot(optical_forward, down_camera)
            )
            forward_norm = float(np.linalg.norm(forward_camera))
            if forward_norm <= 1e-6:
                raise RuntimeError(
                    "ZED optical axis is parallel to the ground-plane down axis"
                )
            forward_camera /= forward_norm
            right_camera = np.cross(down_camera, forward_camera)
            right_camera /= float(np.linalg.norm(right_camera))
            leveled_camera_axes = np.column_stack(
                [right_camera, down_camera, forward_camera]
            )
            delta_camera = leveled_camera_axes @ offset

            calibration = self._env.manipulation_calibration(name)
            arm_from_camera = np.asarray(
                calibration["arm_from_camera"], dtype=np.float64
            )
            delta_arm = arm_from_camera[:3, :3] @ delta_camera
            status = self._env.arm_status()
            if not isinstance(status, dict) or bool(
                status.get("estop_latched", False)
            ):
                raise RuntimeError("Pi arm RPC is unavailable or emergency-stopped")
            snapshot = status.get(name)
            if not isinstance(snapshot, dict):
                raise RuntimeError(f"arm RPC has no {name} status snapshot")
            current_tcp = np.asarray(
                snapshot.get("tcp_pose_xyz_rpy"), dtype=np.float64
            )
            if current_tcp.shape != (6,) or not np.all(np.isfinite(current_tcp)):
                raise RuntimeError(
                    f"{name} arm has no valid measured TCP pose for relative motion"
                )
            position = current_tcp[:3] + delta_arm
            quaternion_wxyz = rpy_to_quaternion_wxyz(current_tcp[3:])
            relative_details = {
                "relative_to_measured_tcp": True,
                "camera_offset_xyz_m": offset.tolist(),
                "leveled_camera_axes": leveled_camera_axes.tolist(),
                "delta_camera_xyz_m": delta_camera.tolist(),
                "delta_arm_xyz_m": delta_arm.tolist(),
                "start_tcp_xyz_rpy": current_tcp.tolist(),
                "ground_plane_age_s": ground_age_s,
            }
        elif camera_offset_xyz is not None:
            raise ValueError(
                "camera_offset_xyz is only valid for goto_pose('current')"
            )
        if quaternion_wxyz is None:
            raise ValueError("quaternion_wxyz is required for a Cartesian goto_pose")
        if name in self._attached_objects:
            return {
                "success": False,
                "reason": "freeform_cartesian_motion_forbidden_while_carrying_object",
                "primitive": "goto_pose",
                "arm": name,
            }
        # A free-form Cartesian command invalidates any cached assumption that
        # this arm is still at the endpoint of its last grasp segment.
        self._pending_grasp_retreats.pop(name, None)
        self._pending_grasps.pop(name, None)
        # Prevent execution in an unspecified frame even when a caller manually
        # invents coordinates instead of using a perception result.
        self._env.manipulation_calibration(name)
        target = pose_matrix(position, quaternion_wxyz)
        target_quaternion = matrix_to_quaternion_wxyz(target[:3, :3])
        self._log_step("goto_pose", f"Moving {name} Nero to {target[:3, 3].tolist()}")
        status = self._env.arm_status()
        if not isinstance(status, dict) or bool(status.get("estop_latched", False)):
            raise RuntimeError("Pi arm RPC is unavailable or emergency-stopped")
        snapshot = status.get(name)
        if not isinstance(snapshot, dict):
            raise RuntimeError(f"arm RPC has no {name} status snapshot")
        destinations: list[tuple[np.ndarray, np.ndarray, str]] = []
        if z_approach > 0:
            approach_position = target[:3, 3] + target[:3, :3] @ np.asarray(
                [0.0, 0.0, -z_approach]
            )
            destinations.append(
                (approach_position, target_quaternion, "approach")
            )
        destinations.append(
            (target[:3, 3].copy(), target_quaternion, "target")
        )

        results: list[dict[str, Any]] = []
        for command_index, (
            destination_position,
            destination_quaternion,
            phase,
        ) in enumerate(destinations, start=1):
            destination_rpy = quaternion_wxyz_to_rpy(destination_quaternion)
            command_result = dict(
                self._env.move_arm_pose(
                    name,
                    np.concatenate([destination_position, destination_rpy]).tolist(),
                    timeout_s,
                )
            )
            command_result.update(
                {
                    "phase": phase,
                    "command_index": command_index,
                    "command_count": len(destinations),
                    "command_position_m": destination_position.tolist(),
                }
            )
            results.append(command_result)
            if not command_result.get("success", False):
                result = dict(command_result)
                result.update(
                    {
                        "primitive": "goto_pose",
                        "arm": name,
                        "approach_executed": z_approach > 0,
                        "segmented": False,
                        "submotions": results,
                        **relative_details,
                    }
                )
                self._log_step_update(text=str(result))
                return result

        result = dict(results[-1])
        result.update(
            {
                "primitive": "goto_pose",
                "arm": name,
                "approach_executed": z_approach > 0,
                "segmented": False,
                "submotions": results,
                **relative_details,
            }
        )
        self._log_step_update(text=str(result))
        return result

    def open_gripper(
        self,
        arm: int | str,
        *,
        timeout_s: float = 3.0,
        force_n: float | None = None,
        simulated: bool = False,
    ) -> dict[str, Any]:
        """Open one native Nero CAN gripper and verify feedback.

        Args:
            arm: Nero arm identifier.
            timeout_s: Feedback timeout.
            force_n: Optional force in newtons, constrained to [0.1, 3.0].

        Returns:
            Monitored gripper primitive-result dictionary.
        """

        name = self._arm_name(arm)
        self._log_step("open_gripper", f"Opening {name} Nero gripper")
        if simulated:
            result = {
                "success": True,
                "status": "succeeded",
                "primitive": "open_gripper",
                "reason": "simulated_gripper_hardware_disconnected",
                "arm": name,
                "opened": True,
                "simulated": True,
                "hardware_control_sent": False,
            }
        else:
            result = self._env.set_gripper(name, True, timeout_s, force_n)
        if result.get("success", False):
            self._pending_grasps.pop(name, None)
            self._attached_objects.pop(name, None)
            self._pending_grasp_retreats.pop(name, None)
            self._pending_self_filter_grasps.pop(name, None)
            clear_attachment = getattr(
                self._env, "clear_robot_self_filter_attached_object", None
            )
            if callable(clear_attachment):
                clear_attachment(name)
        self._log_step_update(text=str(result))
        return result

    def close_gripper(
        self,
        arm: int | str,
        *,
        timeout_s: float = 3.0,
        force_n: float | None = None,
        simulated: bool = False,
    ) -> dict[str, Any]:
        """Close one native Nero CAN gripper and verify stable feedback.

        Args:
            arm: Nero arm identifier.
            timeout_s: Feedback timeout.
            force_n: Optional force in newtons, constrained to [0.1, 3.0].

        Returns:
            Monitored gripper primitive-result dictionary. A nonzero final width
            is valid when an object is held.
        """

        name = self._arm_name(arm)
        self._log_step("close_gripper", f"Closing {name} Nero gripper")
        if simulated:
            result = {
                "success": True,
                "status": "succeeded",
                "primitive": "close_gripper",
                "reason": "simulated_gripper_hardware_disconnected",
                "arm": name,
                "opened": False,
                "simulated": True,
                "hardware_control_sent": False,
            }
        else:
            result = self._env.set_gripper(name, False, timeout_s, force_n)
        if result.get("success", False):
            context = self._pending_grasps.get(name)
            if context is not None:
                context["gripper_closed"] = True
                self._attached_objects[name] = dict(context)
            filter_context = self._pending_self_filter_grasps.get(name)
            if filter_context is not None:
                filter_context["gripper_closed"] = True
                set_attachment = getattr(
                    self._env, "set_robot_self_filter_attached_object", None
                )
                if callable(set_attachment):
                    set_attachment(
                        name, filter_context["attached_bounds_local"]
                    )
                if context is None:
                    # The attached-lift feature is off, so goto_grasp_pose
                    # never queued a cuRobo attachment context. The held
                    # object still has to leave the obstacle cloud and join
                    # the robot model, or every later plan starts "in
                    # collision" with the thing between the fingers: on
                    # 2026-09-05 goto_pose('home') failed that way after every
                    # grasp while the log said "Start or End state in
                    # collision". The self-filter context carries the same
                    # box, so attach from it; _pending_grasps stays empty and
                    # the lift-before-home gate does not engage.
                    self._attached_objects[name] = dict(filter_context)
        self._log_step_update(text=str(result))
        return result
