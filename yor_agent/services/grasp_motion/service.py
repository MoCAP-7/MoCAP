"""Serve collision-safe Nero grasp checks and cuRobo motion plans over ZMQ."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import time
import traceback
from typing import Any

import msgpack_numpy
import numpy as np
from scipy.spatial import cKDTree
import torch
import trimesh
from yourdfpy import URDF
import yaml
import zmq

from curobo._src.cost.tool_pose_criteria import ToolPoseCriteria
from curobo._src.geom.sphere_fit.types import SphereFitType
from curobo._src.state.state_joint_trajectory_ops import (
    get_joint_state_at_horizon_index,
)
from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.scene import Cuboid, Mesh, Scene
from curobo.types import GoalToolPose, JointState, Pose

try:  # package import (tests, tooling)
    from .pregrasp_selection import select_pregrasp
    from .robot_points import ROBOT_POINT_MARGIN_M, remove_points_near_spheres
    from .scene_cache import (
        evict_cached_meshes,
        mesh_cache_of,
        observed_scene_name,
        verify_active_meshes,
    )
    from .solver_options import install_finetune_cap, parse_finetune_cap
    from .voxel_surface import crop_points, greedy_voxel_surface, tiled_voxel_surfaces
except ImportError:  # `python services/grasp_motion/service.py`
    from pregrasp_selection import select_pregrasp
    from robot_points import ROBOT_POINT_MARGIN_M, remove_points_near_spheres
    from scene_cache import (
        evict_cached_meshes,
        mesh_cache_of,
        observed_scene_name,
        verify_active_meshes,
    )
    from solver_options import install_finetune_cap, parse_finetune_cap
    from voxel_surface import crop_points, greedy_voxel_surface, tiled_voxel_surfaces


DEFAULT_CONFIG = (
    Path(__file__).resolve().parents[2]
    / ".runtime"
    / "grasp_motion"
    / "nero_with_official_gripper.yml"
)
DEFAULT_URDF = DEFAULT_CONFIG.with_suffix(".urdf")
GRIPPER_LINKS = (
    "gripper_flange",
    "gripper_base",
    "gripper_link1",
    "gripper_link2",
)
# Surface-shell radius is pitch*sqrt(2)/2 (setup_assets.py): 5.7 mm at the
# original 8 mm pitch, 11.3 mm at 16 mm. Anything larger is a stale asset
# from the old sparse fitter, which padded the small gripper links heavily.
GRIPPER_COLLISION_SPHERE_MAX_RADIUS_M = 0.015
ARM_LINKS = ("base_link", *(f"link{index}" for index in range(1, 8)))
ARM_COLLISION_SPHERE_EXPECTED_COUNT = 124
ARM_COLLISION_SPHERE_MAX_RADIUS_M = 0.040
ATTACHED_OBJECT_LINK = "attached_object"
ATTACHED_OBJECT_SPHERES = 32
# How far past the open-gripper occupancy the swept-volume check still reports
# an exact distance. It must exceed the 0.05 m ceiling that ``check`` enforces
# on ``clearance_m``, so the safety verdict never depends on the saturation.
CLEARANCE_REPORT_BAND_M = 0.06
NERO_JOINT_POSITION_LIMITS_RAD = np.asarray(
    [
        [-2.70526, 2.70526],
        [-1.74, 1.74],
        [-2.75, 2.75],
        [-1.01, 2.14],
        [-2.75, 2.75],
        [-0.73, 0.95],
        [-math.pi / 2, math.pi / 2],
    ],
    dtype=np.float64,
)
TRAJECTORY_MAX_JOINT_STEP_RAD = 0.10
TRAJECTORY_START_TOLERANCE_RAD = 0.02
TRAJECTORY_GOAL_TOLERANCE_RAD = 0.02
# A grasp plan can contain one 512-sample approach and one 512-sample
# contact segment.  This transport bound is independent of the Pi's smaller
# per-RPC execution bound; the controller chunks the returned path exactly.
PLANNER_MAX_WAYPOINTS = 1024
PLANNER_IK_SEEDS = 32
PLANNER_WARMUP_ITERATIONS = 2
# Grasp-model targets near Nero's workspace boundary are useful even when the
# exact jaw-center pose lies a small distance outside the kinematic surface.
# Official joint limits and every collision check remain unchanged; attached
# lift retains its stricter independent endpoint verification below.
GRASP_IK_POSITION_TOLERANCE_M = 0.020
GRASP_IK_ORIENTATION_TOLERANCE_RAD = 0.10
SEED_FK_POSITION_TOLERANCE_M = 0.015
SEED_FK_ROTATION_TOLERANCE_RAD = 0.10
PLANNED_TCP_POSITION_TOLERANCE_M = 0.015
PLANNED_TCP_ROTATION_TOLERANCE_RAD = 0.10
# cuRoboV2 MotionPlanner.plan_pose/plan_cspace default to five attempts with
# PRM graph seeding from the second attempt; goalset planning defaults to the
# same five IK+TrajOpt rounds. Requests may lower these once measured on the
# Jetson; the upper bound only guards against runaway requests.
PLANNER_MAX_ATTEMPTS_LIMIT = 10
# After a pre-grasp goalset reaches no member, candidates are planned one at a
# time (only a single goal gets cuRobo's graph seeding) while less than this has
# passed since the goalset plan started. The client's grasp_motion_timeout_ms
# covers the whole request, including the single-goal plan still running when
# the budget ends and the contact segment, so the budget stays well below it.
PREGRASP_SINGLE_GOAL_BUDGET_S = 75.0
PREGRASP_SINGLE_GOAL_BUDGET_LIMIT_S = 150.0
# Observed points farther than this from the arm base cannot enter any
# sphere-obstacle query: TCP bound 1.0 m + link/sphere reach margin 0.05 m +
# attached-object half diagonal 0.35 m + mesh query distance 0.10 m.
SCENE_CROP_RADIUS_MINIMUM_M = 1.5
DEFAULT_SCENE_CROP_RADIUS_M = 1.6
DEFAULT_SCENE_FINE_RADIUS_M = 0.30
# cuRoboV2's mesh distance kernel searches each mesh out to half its bounding
# box diagonal (curobo/_src/geom/data/data_mesh.py compute_local_sdf), so the
# scene is split into tile meshes of about this size: same exposed voxel
# faces, but a query radius of ~0.26 m instead of ~0.65 m for a table scene.
DEFAULT_SCENE_TILE_M = 0.30
SCENE_MESH_MODES = ("tiled", "greedy", "per_voxel")
# cuRobo launches one collision thread per (sphere, mesh slot) on every
# optimizer step, so the slot count multiplies the kernel launch even when the
# slots are empty: on the Jetson an almost empty world planned in 7.3 s with 1
# slot, 8.9 s with 2 and 75.8 s with 64 (2026-09-04). Deploy with one slot
# and a single per-voxel mesh; the tiled mode needs --scene-mesh-slots >= its
# tile count and is kept only for experiments.
SCENE_MESH_SLOTS = 1


def _scene_options(request: dict[str, Any], scene_voxel_m: float) -> dict[str, Any]:
    """Per-request world-representation options with safe bounds.

    ``scene_mesh``: ``per_voxel`` (default; cuRobo's single
    ``Mesh.from_pointcloud``), ``greedy`` (single mesh, coplanar faces merged;
    measured slower) or ``tiled`` (per-voxel surface split into
    ``scene_tile_m`` tiles; needs as many mesh slots and measured slower).
    ``scene_crop_radius_m`` drops points no robot sphere can reach.
    ``scene_coarse_voxel_m`` (optional, >= scene_voxel_m) meshes points
    farther than ``scene_fine_radius_m`` from every tool centre at that
    coarser pitch; near the tool centres the fine pitch is kept.
    """

    mesh_mode = str(request.get("scene_mesh", "per_voxel"))
    if mesh_mode not in SCENE_MESH_MODES:
        raise ValueError(f"scene_mesh must be one of {SCENE_MESH_MODES}")
    tile_m = float(request.get("scene_tile_m", DEFAULT_SCENE_TILE_M))
    if not math.isfinite(tile_m) or not 0.10 <= tile_m <= 2.0:
        raise ValueError("scene_tile_m must be in [0.10, 2.0] m")
    crop_radius_m = float(request.get("scene_crop_radius_m", DEFAULT_SCENE_CROP_RADIUS_M))
    if not math.isfinite(crop_radius_m) or crop_radius_m < SCENE_CROP_RADIUS_MINIMUM_M:
        raise ValueError(
            f"scene_crop_radius_m must be at least {SCENE_CROP_RADIUS_MINIMUM_M} m"
        )
    coarse_voxel_m = request.get("scene_coarse_voxel_m")
    if coarse_voxel_m is not None:
        coarse_voxel_m = float(coarse_voxel_m)
        if not math.isfinite(coarse_voxel_m) or not scene_voxel_m <= coarse_voxel_m <= 0.05:
            raise ValueError("scene_coarse_voxel_m must be in [scene_voxel_m, 0.05]")
    fine_radius_m = float(request.get("scene_fine_radius_m", DEFAULT_SCENE_FINE_RADIUS_M))
    if not math.isfinite(fine_radius_m) or not 0.10 <= fine_radius_m <= 1.0:
        raise ValueError("scene_fine_radius_m must be in [0.10, 1.0] m")
    return {
        "mesh_mode": mesh_mode,
        "tile_m": tile_m,
        "crop_radius_m": crop_radius_m,
        "coarse_voxel_m": coarse_voxel_m,
        "fine_radius_m": fine_radius_m,
    }


class _Stopwatch:
    """Accumulate named wall-clock laps for per-request planner diagnostics."""

    def __init__(self) -> None:
        self._started = time.monotonic()
        self._last = self._started
        self.laps: dict[str, float] = {}

    def lap(self, name: str) -> float:
        now = time.monotonic()
        elapsed = now - self._last
        self.laps[name] = self.laps.get(name, 0.0) + elapsed
        self._last = now
        return elapsed

    def report(self) -> dict[str, float]:
        report = {name: round(value, 4) for name, value in self.laps.items()}
        report["total_s"] = round(time.monotonic() - self._started, 4)
        return report


def _planner_attempt_kwargs(request: dict[str, Any]) -> dict[str, int]:
    """Optional per-request retry control; cuRobo defaults apply when absent."""

    kwargs: dict[str, int] = {}
    max_attempts = request.get("max_attempts")
    if max_attempts is not None:
        max_attempts = int(max_attempts)
        if not 1 <= max_attempts <= PLANNER_MAX_ATTEMPTS_LIMIT:
            raise ValueError(
                f"max_attempts must be in [1, {PLANNER_MAX_ATTEMPTS_LIMIT}]"
            )
        kwargs["max_attempts"] = max_attempts
    enable_graph_attempt = request.get("enable_graph_attempt")
    if enable_graph_attempt is not None:
        enable_graph_attempt = int(enable_graph_attempt)
        if not 0 <= enable_graph_attempt <= PLANNER_MAX_ATTEMPTS_LIMIT:
            raise ValueError(
                f"enable_graph_attempt must be in [0, {PLANNER_MAX_ATTEMPTS_LIMIT}]"
            )
        kwargs["enable_graph_attempt"] = enable_graph_attempt
    return kwargs


def _curobo_mesh_query_patched() -> bool | None:
    """Whether the vendored cuRobo bounds its mesh query radius (patch_curobo_mesh_query.py).

    Without the patch every sphere query searches out to half the scene mesh
    bounding-box diagonal and planning is several times slower. Reported in
    ``health`` so a fresh cuRobo checkout is noticed before the first grasp.
    """

    try:
        import curobo  # type: ignore

        source_path = (
            Path(curobo.__file__).resolve().parent / "_src" / "geom" / "data" / "data_mesh.py"
        )
        return "# yor: bounded mesh query radius" in source_path.read_text(encoding="utf-8")
    except Exception:
        return None


def _solver_timing(result: Any) -> dict[str, Any]:
    """Extract cuRobo's own solver timers/status from a TrajOptSolverResult."""

    timing: dict[str, Any] = {}
    for key in ("total_time", "solve_time"):
        value = getattr(result, key, None)
        if isinstance(value, (int, float)):
            timing[f"{key}_s"] = float(value)
    status = getattr(result, "status", None)
    if status is not None:
        timing["status"] = str(status)
    return timing


def _finite_array(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must have finite shape {shape}")
    return array


def _validate_collision_asset(robot_config: Path) -> dict[str, Any]:
    """Reject stale runtime assets that retain either oversized robot model.

    Returns per-link sphere statistics for ``health``.
    """

    try:
        payload = yaml.safe_load(robot_config.read_text(encoding="utf-8"))
        kinematics = payload["robot_cfg"]["kinematics"]
        buffer_m = float(kinematics["collision_sphere_buffer"])
        collision_spheres = kinematics["collision_spheres"]
    except Exception as exc:
        raise RuntimeError(
            "invalid Nero cuRobo asset; rerun services/grasp_motion/setup_assets.py"
        ) from exc
    if abs(buffer_m) > 1e-9:
        raise RuntimeError(
            "stale Nero cuRobo asset has a nonzero global collision-sphere "
            "buffer; rerun services/grasp_motion/setup_assets.py"
        )
    arm_radii = [
        float(sphere["radius"])
        for link_name in ARM_LINKS
        for sphere in collision_spheres.get(link_name, [])
    ]
    if (
        len(arm_radii) != ARM_COLLISION_SPHERE_EXPECTED_COUNT
        or not arm_radii
        or max(arm_radii) > ARM_COLLISION_SPHERE_MAX_RADIUS_M
    ):
        raise RuntimeError(
            "stale Nero cuRobo asset has an oversized or unexpected arm "
            "collision model; rerun services/grasp_motion/setup_assets.py"
        )
    gripper_radii: list[float] = []
    for link_name in GRIPPER_LINKS:
        spheres = collision_spheres.get(link_name, [])
        radii = [float(sphere["radius"]) for sphere in spheres]
        if not radii or max(radii) > GRIPPER_COLLISION_SPHERE_MAX_RADIUS_M:
            raise RuntimeError(
                f"stale Nero cuRobo asset has oversized {link_name} spheres; "
                "rerun services/grasp_motion/setup_assets.py"
            )
        gripper_radii.extend(radii)
    return {
        "arm_sphere_count": len(arm_radii),
        "gripper_sphere_count": len(gripper_radii),
        "gripper_sphere_max_radius_m": max(gripper_radii),
        "spheres_per_link": {
            link: len(collision_spheres.get(link, [])) for link in (*ARM_LINKS, *GRIPPER_LINKS)
        },
    }


def _pose_matrices(value: Any, name: str) -> np.ndarray:
    poses = np.asarray(value, dtype=np.float64)
    if poses.ndim == 2:
        poses = poses[None]
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError(f"{name} must have shape [K, 4, 4]")
    if not 1 <= len(poses) <= 64 or not np.all(np.isfinite(poses)):
        raise ValueError(f"{name} must contain 1-64 finite poses")
    if not np.allclose(poses[:, 3, :], [0.0, 0.0, 0.0, 1.0], atol=1e-5):
        raise ValueError(f"{name} contains an invalid homogeneous transform")
    rotations = poses[:, :3, :3]
    identity = np.eye(3, dtype=np.float64)[None]
    if not np.allclose(
        np.swapaxes(rotations, 1, 2) @ rotations, identity, atol=1e-4
    ) or not np.allclose(np.linalg.det(rotations), 1.0, atol=1e-4):
        raise ValueError(f"{name} contains a non-rigid rotation")
    return poses


def _point_cloud(value: Any, name: str, *, maximum: int = 100_000) -> np.ndarray:
    points = np.asarray(value, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"{name} must have shape [N, 3]")
    points = points[np.all(np.isfinite(points), axis=1)]
    if not 10 <= len(points) <= maximum:
        raise ValueError(f"{name} must contain between 10 and {maximum} finite points")
    return np.ascontiguousarray(points)



def _attached_bounds(value: Any) -> np.ndarray:
    bounds = _finite_array(value, (2, 3), "attached_bounds_local")
    extent = bounds[1] - bounds[0]
    if np.any(extent <= 0.0) or np.any(extent > 0.40):
        raise ValueError(
            "attached_bounds_local must have positive extents no larger than 0.40 m"
        )
    return bounds


class OpenGripperOccupancy:
    """Dense official-mesh occupancy expressed in the grasp-volume TCP frame."""

    def __init__(self, urdf_path: Path, *, pitch_m: float = 0.004) -> None:
        if not 0.002 <= pitch_m <= 0.01:
            raise ValueError("gripper occupancy pitch must be in [0.002, 0.01] m")
        robot = URDF.load(
            str(urdf_path),
            build_scene_graph=False,
            build_collision_scene_graph=True,
            load_meshes=False,
            load_collision_meshes=True,
        )
        robot.update_cfg({"gripper": 0.1})
        points: list[np.ndarray] = []
        for link_name in GRIPPER_LINKS:
            geometry_name = f"{link_name}.stl"
            mesh = robot.collision_scene.geometry[geometry_name].copy()
            tcp_from_geometry = robot.collision_scene.graph.get(
                frame_to=geometry_name, frame_from="grasp_tcp"
            )[0]
            mesh.apply_transform(tcp_from_geometry)
            points.append(mesh.voxelized(pitch_m).fill().points)
        occupied = np.concatenate(points, axis=0)
        quantized = np.round(occupied / pitch_m).astype(np.int32)
        _, unique = np.unique(quantized, axis=0, return_index=True)
        self.points = np.ascontiguousarray(occupied[np.sort(unique)])
        self.tree = cKDTree(self.points)
        self.pitch_m = float(pitch_m)
        self.voxel_half_diagonal_m = math.sqrt(3.0) * self.pitch_m / 2.0
        # Farthest occupied voxel from the grasp-volume TCP origin. It bounds
        # how far an obstacle point can sit and still be the nearest voxel.
        self.max_radius_m = float(np.linalg.norm(self.points, axis=1).max())

    def check(
        self,
        tcp_poses: np.ndarray,
        obstacle_points: np.ndarray,
        *,
        clearance_m: float,
        approach_m: float,
        approach_samples: int,
    ) -> dict[str, Any]:
        if not 0.0 <= clearance_m <= 0.05:
            raise ValueError("clearance_m must be in [0, 0.05]")
        if not 0.0 <= approach_m <= 0.25:
            raise ValueError("approach_m must be in [0, 0.25]")
        if not 1 <= approach_samples <= 16:
            raise ValueError("approach_samples must be in [1, 16]")
        safe = []
        minimum_clearances = []
        collision_steps = []
        offsets = np.linspace(-approach_m, 0.0, approach_samples)
        # A point farther from the step TCP than the gripper's own extent plus
        # the reported band cannot be the nearest occupied voxel inside that
        # band. A rotation preserves norms, so this radial test runs on the raw
        # arm-frame delta, before the transform and the tree query, and leaves
        # only the handful of points that can matter. ``clearance_m`` is capped
        # at 0.05 by the validation above and the band is wider, so the safety
        # verdict is unchanged; only the reported minimum saturates.
        cutoff = self.max_radius_m + CLEARANCE_REPORT_BAND_M
        squared_cutoff = cutoff * cutoff
        band_floor = CLEARANCE_REPORT_BAND_M - self.voxel_half_diagonal_m
        for pose in tcp_poses:
            rotation = pose[:3, :3]
            translation = pose[:3, 3]
            candidate_minimum = np.inf
            first_collision = None
            for step_index, offset in enumerate(offsets):
                step_translation = translation + rotation @ np.asarray([0.0, 0.0, offset])
                delta = obstacle_points - step_translation
                near = delta[np.einsum("ij,ij->i", delta, delta) <= squared_cutoff]
                if len(near) == 0:
                    minimum = band_floor
                else:
                    distances = self.tree.query(near @ rotation, k=1, workers=-1)[0]
                    minimum = min(
                        float(np.min(distances) - self.voxel_half_diagonal_m),
                        band_floor,
                    )
                candidate_minimum = min(candidate_minimum, minimum)
                if minimum < clearance_m and first_collision is None:
                    first_collision = step_index
            safe.append(first_collision is None)
            minimum_clearances.append(candidate_minimum)
            collision_steps.append(first_collision)
        return {
            "safe": safe,
            "minimum_clearance_m": minimum_clearances,
            "first_collision_step": collision_steps,
            "clearance_m": clearance_m,
            "approach_m": approach_m,
            "approach_samples": approach_samples,
            "occupancy_pitch_m": self.pitch_m,
            # ``minimum_clearance_m`` is exact below this and saturates at it.
            "clearance_report_band_m": band_floor,
        }


class NeroGraspMotionPlanner:
    def __init__(
        self,
        robot_config: Path,
        robot_urdf: Path,
        *,
        mesh_slots: int = SCENE_MESH_SLOTS,
    ) -> None:
        if not robot_config.is_file() or not robot_urdf.is_file():
            raise FileNotFoundError(
                "Nero planner assets are missing; run setup_assets.py first"
            )
        if not 1 <= int(mesh_slots) <= 1024:
            raise ValueError("mesh_slots must be in [1, 1024]")
        self.mesh_slots = int(mesh_slots)
        self.asset_stats = _validate_collision_asset(robot_config)
        self.robot_config_path = str(robot_config)
        self._finetune_cap: int | None = None
        self.default_finetune_cap: int | None = None
        self.device = torch.device("cuda:0")
        self.gripper = OpenGripperOccupancy(robot_urdf)
        config = MotionPlannerCfg.create(
            robot=str(robot_config.resolve()),
            collision_cache={"mesh": self.mesh_slots},
            # cuRoboV2's MotionPlanner is designed to run with its solver CUDA
            # graphs captured.  In particular, plan_grasp changes the pose
            # criteria between the approach and contact phases.  Running that
            # multi-stage pipeline without the normal graph lifecycle produced
            # repeatable failures at the final 10 cm contact phase on Orin.
            use_cuda_graph=True,
            # cuRobo's single-goal default. Sixteen seeds produced repeatable
            # false negatives for reachable Nero grasp targets on the Jetson.
            num_ik_seeds=PLANNER_IK_SEEDS,
            num_trajopt_seeds=4,
            max_goalset=16,
            position_tolerance=GRASP_IK_POSITION_TOLERANCE_M,
            orientation_tolerance=GRASP_IK_ORIENTATION_TOLERANCE_RAD,
            interpolation_dt=0.10,
            interpolation_buffer_size=512,
            optimizer_collision_activation_distance=0.01,
        )
        self.planner = MotionPlanner(config)
        # Per-request cap on trajopt finetune passes (see solver_options.py).
        # Installed before warmup so the wrapped methods are what the CUDA
        # graph lifecycle exercises; the cap is None during warmup.
        self._finetune_wrapped = install_finetune_cap(
            self.planner.trajopt_solver, lambda: self._finetune_cap
        )
        # Follow the cuRoboV2 lifecycle before serving the first real request.
        # Two iterations is the value used by upstream grasp-goalset tests and
        # keeps Jetson startup and shared-memory pressure bounded.
        self.planner.warmup(
            enable_graph=True,
            num_warmup_iterations=PLANNER_WARMUP_ITERATIONS,
        )
        try:
            attached_slots = self.planner.kinematics.config.kinematics_config.get_number_of_spheres(
                ATTACHED_OBJECT_LINK
            )
        except Exception as exc:
            raise RuntimeError(
                "Nero cuRobo assets do not reserve attached_object collision "
                "spheres; rerun services/grasp_motion/setup_assets.py"
            ) from exc
        if attached_slots < ATTACHED_OBJECT_SPHERES:
            raise RuntimeError(
                f"attached_object has {attached_slots} collision slots, expected at "
                f"least {ATTACHED_OBJECT_SPHERES}"
            )
        self.attached_object_slots = int(attached_slots)
        # Sphere fitting is the expensive part of AttachmentManager. Reuse the
        # exact TCP-local geometry across lift and later home/place plans, as
        # recommended by the official API, while bounding process memory.
        self._attachment_sphere_cache: dict[tuple[float, ...], torch.Tensor] = {}
        # cuRoboV2 caches Warp meshes by name and silently reuses old geometry
        # for a name it has already seen. Resolve the cache now so a cuRobo
        # build that hides it fails at startup rather than on the first grasp.
        self._mesh_cache = mesh_cache_of(self.planner)
        self._scene_refresh_count = 0
        self._last_scene: dict[str, Any] | None = None

    def health(self) -> dict[str, Any]:
        return {
            "success": True,
            "backend": "curobo",
            "cuda_device": torch.cuda.get_device_name(self.device),
            "joint_names": list(self.planner.joint_names),
            "tool_frames": list(self.planner.tool_frames),
            "robot_sphere_count": int(
                self.planner.kinematics.config.kinematics_config.total_spheres
            ),
            "gripper_occupancy_points": int(len(self.gripper.points)),
            "planner_maximum_waypoint_count": PLANNER_MAX_WAYPOINTS,
            "planner_ik_seed_count": PLANNER_IK_SEEDS,
            "planner_grasp_position_tolerance_m": GRASP_IK_POSITION_TOLERANCE_M,
            "planner_grasp_orientation_tolerance_rad": (
                GRASP_IK_ORIENTATION_TOLERANCE_RAD
            ),
            "planner_seed_reset_per_request": True,
            "planner_cuda_graph_enabled": True,
            "planner_warmup_iterations": PLANNER_WARMUP_ITERATIONS,
            "planner_verifies_external_goal_joint_solution": True,
            "planner_goalset_enabled": True,
            "attached_object_sphere_slots": self.attached_object_slots,
            "planner_max_attempts_limit": PLANNER_MAX_ATTEMPTS_LIMIT,
            "scene_refresh_count": self._scene_refresh_count,
            "cached_mesh_names": list(self._mesh_cache.wp_cache.keys()),
            "scene_mesh_modes": list(SCENE_MESH_MODES),
            "scene_mesh_slots": self.mesh_slots,
            "scene_crop_radius_minimum_m": SCENE_CROP_RADIUS_MINIMUM_M,
            "last_scene": self._last_scene,
            "robot_config_path": self.robot_config_path,
            "collision_asset": self.asset_stats,
            "curobo_mesh_query_patched": _curobo_mesh_query_patched(),
            "finetune_cap_default": self.default_finetune_cap,
            "finetune_cap_wrapped_methods": list(self._finetune_wrapped),
        }

    def set_finetune_cap(self, value: Any) -> int | None:
        """Cap trajopt finetune passes for the next plan (None = service default)."""

        cap = parse_finetune_cap(value)
        self._finetune_cap = self.default_finetune_cap if cap is None else cap
        return self._finetune_cap

    def check_grasps(self, request: dict[str, Any]) -> dict[str, Any]:
        poses = _pose_matrices(request.get("tcp_poses"), "tcp_poses")
        obstacles = _point_cloud(request.get("obstacle_points"), "obstacle_points")
        result = self.gripper.check(
            poses,
            obstacles,
            clearance_m=float(request.get("clearance_m", 0.008)),
            approach_m=float(request.get("approach_m", 0.10)),
            approach_samples=int(request.get("approach_samples", 6)),
        )
        result.update({"success": True, "candidate_count": len(poses)})
        return result

    def _remove_start_robot_points(
        self,
        points: np.ndarray,
        current_joints: np.ndarray,
        margin_m: float = ROBOT_POINT_MARGIN_M,
    ) -> tuple[np.ndarray, int]:
        """Remove the robot's own returns around the start-state sphere model.

        The sphere model is a thin surface shell (median radius 1.5-11 mm), so
        a point belongs to the robot when it lies within a sphere radius plus
        a margin for depth noise and calibration error. The former rule eroded
        each sphere by 12 mm and kept everything outside, which on this shell
        removed 0-2 of the arm's returns and made every plan start "in
        collision" with the arm itself. See robot_points.py.
        """

        q = torch.as_tensor(
            current_joints, dtype=torch.float32, device=self.device
        ).view(1, -1)
        spheres = self.planner.kinematics.get_robot_as_spheres(q)[0]
        spheres_xyzr = np.asarray(
            [[*sphere.pose[:3], float(sphere.radius)] for sphere in spheres],
            dtype=np.float32,
        ).reshape(-1, 4)
        return remove_points_near_spheres(points, spheres_xyzr, margin_m=margin_m)

    def _tcp_position(self, joints: np.ndarray) -> np.ndarray:
        """Tool-centre position (arm frame) for one joint vector via cuRobo FK."""

        q = JointState.from_position(
            torch.as_tensor(joints, dtype=torch.float32, device=self.device).view(1, 7),
            joint_names=self.planner.joint_names,
        )
        state = self.planner.kinematics.compute_kinematics(q)
        pose = state.tool_poses.get_link_pose(self.planner.tool_frames[0])
        return pose.position.detach().cpu().numpy().reshape(-1, 3)[0].astype(np.float64)

    @staticmethod
    def _mesh_from_arrays(name: str, vertices: np.ndarray, triangles: np.ndarray) -> Any:
        return Mesh(
            name,
            pose=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            vertices=vertices.tolist(),
            faces=triangles.reshape(-1).tolist(),
        )

    def _build_scene_meshes(
        self,
        name: str,
        points: np.ndarray,
        pitch_m: float,
        options: dict[str, Any],
        *,
        slots_available: int,
    ) -> tuple[list[Any], dict[str, Any]]:
        """Meshes for one point group (one mesh, or many tiles) plus statistics."""

        started = time.monotonic()
        mesh_mode = options["mesh_mode"]
        if mesh_mode == "per_voxel":
            mesh = Mesh.from_pointcloud(points, pitch=pitch_m, name=name)
            faces = getattr(mesh, "faces", None) or []
            meshes = [mesh]
            stats: dict[str, Any] = {
                "triangles": int(len(faces) // 3),
                "per_voxel_triangles": int(len(faces) // 3),
                "tiles": 1,
            }
        elif mesh_mode == "greedy":
            vertices, triangles, stats = greedy_voxel_surface(points, pitch_m)
            meshes = [self._mesh_from_arrays(name, vertices, triangles)]
            stats["tiles"] = 1
        else:
            tile_m = float(options["tile_m"])
            while True:
                tiles, stats = tiled_voxel_surfaces(points, pitch_m, tile_m)
                if len(tiles) <= slots_available or tile_m >= 4.0:
                    break
                # Too many tiles for the mesh cache: coarsen the tiling. The
                # surface is unchanged, only the per-mesh query radius grows.
                tile_m *= 2.0
            if len(tiles) > slots_available:
                raise ValueError(
                    f"scene needs {len(tiles)} tile meshes but only "
                    f"{slots_available} mesh slots remain"
                )
            meshes = [
                self._mesh_from_arrays(f"{name}_t{index:03d}", vertices, triangles)
                for index, (vertices, triangles) in enumerate(tiles)
            ]
        return meshes, {
            "name": name,
            "pitch_m": float(pitch_m),
            "points": int(len(points)),
            "mesh_mode": mesh_mode,
            "build_s": round(time.monotonic() - started, 4),
            **stats,
        }

    def _refresh_observed_scene(
        self,
        filtered: np.ndarray,
        scene_voxel_m: float,
        *,
        options: dict[str, Any],
        tool_centres: list[np.ndarray],
    ) -> dict[str, Any]:
        """Rebuild the observed-scene meshes and prove the planner now uses them.

        cuRoboV2 keeps Warp meshes in a name-keyed cache that neither
        ``update_world`` nor ``clear_scene_cache`` empties, and it reuses the
        old geometry for any name it has seen before (see ``scene_cache.py``).
        A long-lived service that reused one name planned every request after
        the first against the first observation. Evict first, load under fresh
        names, then verify that the active meshes are the fresh ones.

        Triangle count drives cuRobo's mesh query cost on the Jetson (7 s per
        plan with an almost empty world versus 63 s with a 38k-triangle
        per-voxel table scene, 2026-09-04 benchmark), hence the greedy mesh,
        the reach crop and the optional coarse pitch away from the tool centres.
        """

        started = time.monotonic()
        evicted = evict_cached_meshes(self._mesh_cache)
        base_name = observed_scene_name(self._scene_refresh_count)
        points, cropped = crop_points(filtered, options["crop_radius_m"])
        if len(points) < 10:
            # Nothing observed within reach; keep the full cloud rather than an
            # empty world so the planner still sees whatever was observed.
            points, cropped = filtered, 0
        coarse_voxel_m = options["coarse_voxel_m"]
        groups: list[tuple[str, np.ndarray, float]] = []
        if coarse_voxel_m is not None and coarse_voxel_m > scene_voxel_m and tool_centres:
            centres = np.stack(tool_centres, axis=0).astype(np.float64)
            deltas = points[:, None, :].astype(np.float64) - centres[None, :, :]
            nearest = np.min(np.einsum("ijk,ijk->ij", deltas, deltas), axis=1)
            near = nearest <= options["fine_radius_m"] ** 2
            if np.any(near):
                groups.append((f"{base_name}_fine", points[near], scene_voxel_m))
            if np.any(~near):
                groups.append((f"{base_name}_coarse", points[~near], coarse_voxel_m))
        else:
            groups.append((base_name, points, scene_voxel_m))
        meshes: list[Any] = []
        mesh_stats = []
        for name, group_points, pitch_m in groups:
            group_meshes, stats = self._build_scene_meshes(
                name,
                group_points,
                pitch_m,
                options,
                slots_available=self.mesh_slots - len(meshes),
            )
            meshes.extend(group_meshes)
            mesh_stats.append(stats)
        mesh_names = [str(mesh.name) for mesh in meshes]
        built = time.monotonic()
        self.planner.update_world(Scene(mesh=meshes))
        loaded = time.monotonic()
        warp_ids = verify_active_meshes(self._mesh_cache, mesh_names)
        self._scene_refresh_count += 1
        scene = {
            "mesh_name": mesh_names[0],
            "mesh_names": mesh_names,
            "mesh_count": len(mesh_names),
            "warp_mesh_ids": warp_ids,
            "warp_mesh_id": warp_ids[mesh_names[0]],
            "evicted_mesh_names": evicted,
            "meshes": mesh_stats,
            "triangle_count": int(sum(stats["triangles"] for stats in mesh_stats)),
            "per_voxel_triangle_count": int(
                sum(stats["per_voxel_triangles"] for stats in mesh_stats)
            ),
            "scene_voxel_m": float(scene_voxel_m),
            "coarse_voxel_m": coarse_voxel_m,
            "fine_radius_m": float(options["fine_radius_m"]),
            "tile_m": float(options["tile_m"]),
            "mesh_mode": options["mesh_mode"],
            "crop_radius_m": float(options["crop_radius_m"]),
            "cropped_points": int(cropped),
            "scene_points": int(len(points)),
            "tool_centres": [centre.tolist() for centre in tool_centres],
            "mesh_build_s": round(built - started, 4),
            "update_world_s": round(loaded - built, 4),
            "verify_s": round(time.monotonic() - loaded, 4),
        }
        self._last_scene = scene
        return scene

    def _tcp_fk_error(
        self, joints: np.ndarray, target_pose: np.ndarray
    ) -> tuple[float, float]:
        """Return cuRobo TCP position/orientation error for one joint state."""

        q = JointState.from_position(
            torch.as_tensor(joints, dtype=torch.float32, device=self.device).view(1, 7),
            joint_names=self.planner.joint_names,
        )
        state = self.planner.kinematics.compute_kinematics(q)
        actual = state.tool_poses.get_link_pose(self.planner.tool_frames[0])
        actual_position = actual.position.detach().cpu().numpy().reshape(-1, 3)[0]
        actual_quaternion = (
            actual.quaternion.detach().cpu().numpy().reshape(-1, 4)[0]
        )
        target = Pose.from_matrix(
            torch.as_tensor(
                target_pose, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
        )
        target_quaternion = (
            target.quaternion.detach().cpu().numpy().reshape(-1, 4)[0]
        )
        position_error = float(
            np.linalg.norm(actual_position - target_pose[:3, 3])
        )
        quaternion_dot = float(
            np.clip(abs(np.dot(actual_quaternion, target_quaternion)), 0.0, 1.0)
        )
        rotation_error = float(2.0 * math.acos(quaternion_dot))
        return position_error, rotation_error

    def _attach_local_bounds(
        self, q_start: JointState, bounds: np.ndarray
    ) -> int:
        """Populate the reserved TCP-fixed link with a conservative local box."""

        center = 0.5 * (bounds[0] + bounds[1])
        dims = bounds[1] - bounds[0]
        cuboid = Cuboid(
            name="attached_target",
            pose=[*center.tolist(), 1.0, 0.0, 0.0, 0.0],
            dims=dims.tolist(),
        )
        manager = self.planner.attachment_manager
        manager.detach(link_name=ATTACHED_OBJECT_LINK)
        cache_key = tuple(np.round(bounds.reshape(-1), decimals=6).tolist())
        spheres = self._attachment_sphere_cache.get(cache_key)
        if spheres is None:
            spheres = manager.fit_spheres(
                [cuboid],
                num_spheres=ATTACHED_OBJECT_SPHERES,
                surface_radius=0.004,
                sphere_fit_type=SphereFitType.VOXEL,
            )
            if len(self._attachment_sphere_cache) >= 32:
                self._attachment_sphere_cache.pop(
                    next(iter(self._attachment_sphere_cache))
                )
            self._attachment_sphere_cache[cache_key] = spheres
        manager.update(spheres, q_start, link_name=ATTACHED_OBJECT_LINK)
        return int(spheres.shape[0])

    def plan_attached_lift(self, request: dict[str, Any]) -> dict[str, Any]:
        """Plan a linear local-Z retreat after a verified gripper close."""

        started = time.monotonic()
        watch = _Stopwatch()
        attempt_kwargs = _planner_attempt_kwargs(request)
        current = _finite_array(request.get("current_joints"), (7,), "current_joints")
        obstacles = _point_cloud(request.get("obstacle_points"), "obstacle_points")
        bounds = _attached_bounds(request.get("attached_bounds_local"))
        lift_m = float(request.get("lift_m", 0.15))
        scene_voxel_m = float(request.get("scene_voxel_m", 0.012))
        if not 0.02 <= lift_m <= 0.25:
            raise ValueError("lift_m must be in [0.02, 0.25]")
        if not 0.003 <= scene_voxel_m <= 0.05:
            raise ValueError("scene_voxel_m must be in [0.003, 0.05]")

        filtered, removed = self._remove_start_robot_points(
            obstacles, current
        )
        if len(filtered) < 10:
            return {
                "success": False,
                "reason": "insufficient_scene_points_after_robot_self_filter",
                "input_scene_points": len(obstacles),
                "removed_robot_points": removed,
            }
        watch.lap("self_filter_s")
        scene = self._refresh_observed_scene(
            filtered,
            scene_voxel_m,
            options=_scene_options(request, scene_voxel_m),
            tool_centres=[self._tcp_position(current)],
        )
        watch.lap("scene_refresh_s")
        self.planner.reset_seed()
        q_start = JointState.from_position(
            torch.as_tensor(current, dtype=torch.float32, device=self.device).unsqueeze(0),
            joint_names=self.planner.joint_names,
        )
        standard_criteria = {
            frame: ToolPoseCriteria() for frame in self.planner.tool_frames
        }
        criteria_updated = False
        try:
            attached_spheres = self._attach_local_bounds(q_start, bounds)
            state = self.planner.kinematics.compute_kinematics(q_start)
            current_pose = state.tool_poses.get_link_pose(
                self.planner.tool_frames[0]
            )
            offset = Pose(
                position=torch.tensor(
                    [[0.0, 0.0, -lift_m]],
                    dtype=torch.float32,
                    device=self.device,
                ),
                quaternion=torch.tensor(
                    [[1.0, 0.0, 0.0, 0.0]],
                    dtype=torch.float32,
                    device=self.device,
                ),
            )
            goal_pose = current_pose.multiply(offset)
            goal = GoalToolPose.from_poses(
                {self.planner.tool_frames[0]: goal_pose},
                ordered_tool_frames=self.planner.tool_frames,
                num_goalset=1,
            )
            linear_motion = ToolPoseCriteria.linear_motion(
                axis="z",
                non_terminal_scale=1.0,
                project_distance_to_goal=True,
            )
            self.planner.update_tool_pose_criteria(
                {frame: linear_motion for frame in self.planner.tool_frames}
            )
            criteria_updated = True
            result = self.planner.plan_pose(goal, q_start, **attempt_kwargs)
        finally:
            if criteria_updated:
                self.planner.update_tool_pose_criteria(standard_criteria)
            self.planner.attachment_manager.detach(link_name=ATTACHED_OBJECT_LINK)
        watch.lap("plan_s")
        if result is None or result.success is None or not bool(result.success.any()):
            return {
                "success": False,
                "reason": "no_collision_free_attached_lift_path",
                "planning_time_s": time.monotonic() - started,
                "scene": scene,
                "timings": watch.report(),
                "solver": _solver_timing(result),
                "planner_attempts": attempt_kwargs,
                "input_scene_points": len(obstacles),
                "planning_scene_points": len(filtered),
                "removed_robot_points": removed,
                "attached_object_sphere_count": attached_spheres,
            }
        trajectory = result.get_interpolated_plan()
        if trajectory is None:
            return {
                "success": False,
                "reason": "attached_lift_plan_has_no_interpolated_trajectory",
            }
        active = self.planner.kinematics.get_active_js(trajectory)
        positions = active.position.detach().cpu().numpy().reshape(-1, 7)
        if not 2 <= len(positions) <= PLANNER_MAX_WAYPOINTS:
            return {
                "success": False,
                "reason": "planned_trajectory_waypoint_count_out_of_bounds",
                "waypoint_count": len(positions),
            }
        start_error = float(np.max(np.abs(positions[0] - current)))
        maximum_step = float(np.max(np.abs(np.diff(positions, axis=0))))
        target_matrix = goal_pose.get_matrix().detach().cpu().numpy().reshape(4, 4)
        endpoint_position_error, endpoint_rotation_error = self._tcp_fk_error(
            positions[-1], target_matrix
        )
        if start_error > TRAJECTORY_START_TOLERANCE_RAD:
            return {
                "success": False,
                "reason": "planned_trajectory_start_mismatch",
                "start_error_rad": start_error,
            }
        if maximum_step > TRAJECTORY_MAX_JOINT_STEP_RAD:
            return {
                "success": False,
                "reason": "planned_trajectory_joint_step_too_large",
                "maximum_step_rad": maximum_step,
            }
        if (
            endpoint_position_error > PLANNED_TCP_POSITION_TOLERANCE_M
            or endpoint_rotation_error > PLANNED_TCP_ROTATION_TOLERANCE_RAD
        ):
            return {
                "success": False,
                "reason": "planned_trajectory_tcp_goal_mismatch",
                "planned_endpoint_position_error_m": endpoint_position_error,
                "planned_endpoint_rotation_error_rad": endpoint_rotation_error,
            }
        watch.lap("postprocess_s")
        return {
            "success": True,
            "reason": "collision_free_attached_lift_planned",
            "scene": scene,
            "timings": watch.report(),
            "solver": _solver_timing(result),
            "planner_attempts": attempt_kwargs,
            "waypoints": positions.tolist(),
            "trajectory_dt_s": 0.10,
            "lift_m": lift_m,
            "attached_bounds_local": bounds.tolist(),
            "attached_object_sphere_count": attached_spheres,
            "input_scene_points": len(obstacles),
            "planning_scene_points": len(filtered),
            "removed_robot_points": removed,
            "planning_time_s": time.monotonic() - started,
            "start_error_rad": start_error,
            "maximum_step_rad": maximum_step,
            "planned_endpoint_position_error_m": endpoint_position_error,
            "planned_endpoint_rotation_error_rad": endpoint_rotation_error,
        }

    def plan_grasp(self, request: dict[str, Any]) -> dict[str, Any]:
        started = time.monotonic()
        watch = _Stopwatch()
        attempt_kwargs = _planner_attempt_kwargs(request)
        current = _finite_array(request.get("current_joints"), (7,), "current_joints")
        targets = _pose_matrices(
            request.get("tcp_poses", request.get("tcp_pose")), "tcp_poses"
        )
        if len(targets) > 16:
            raise ValueError("tcp_poses is limited to the configured 16-pose goalset")
        requested_target = targets[0]
        obstacles = _point_cloud(request.get("obstacle_points"), "obstacle_points")
        clearance_m = float(request.get("clearance_m", 0.008))
        approach_m = float(request.get("approach_m", 0.10))
        scene_voxel_m = float(request.get("scene_voxel_m", 0.012))
        if not 0.003 <= scene_voxel_m <= 0.05:
            raise ValueError("scene_voxel_m must be in [0.003, 0.05]")
        single_goal_budget_s = float(
            request.get("pregrasp_single_goal_budget_s", PREGRASP_SINGLE_GOAL_BUDGET_S)
        )
        if not 0.0 <= single_goal_budget_s <= PREGRASP_SINGLE_GOAL_BUDGET_LIMIT_S:
            raise ValueError(
                "pregrasp_single_goal_budget_s must be in "
                f"[0, {PREGRASP_SINGLE_GOAL_BUDGET_LIMIT_S:g}]"
            )

        goal_joint_seed = None
        seed_fk_position_error = None
        seed_fk_rotation_error = None
        if request.get("goal_joint_seed") is not None:
            goal_joint_seed = _finite_array(
                request.get("goal_joint_seed"), (7,), "goal_joint_seed"
            )
            if np.any(goal_joint_seed < NERO_JOINT_POSITION_LIMITS_RAD[:, 0]) or np.any(
                goal_joint_seed > NERO_JOINT_POSITION_LIMITS_RAD[:, 1]
            ):
                raise ValueError("goal_joint_seed exceeds official Nero joint limits")
            seed_fk_position_error, seed_fk_rotation_error = self._tcp_fk_error(
                goal_joint_seed, requested_target
            )
            if (
                seed_fk_position_error > SEED_FK_POSITION_TOLERANCE_M
                or seed_fk_rotation_error > SEED_FK_ROTATION_TOLERANCE_RAD
            ):
                return {
                    "success": False,
                    "reason": "goal_joint_seed_tcp_mismatch",
                    "goal_joint_seed_fk_position_error_m": seed_fk_position_error,
                    "goal_joint_seed_fk_rotation_error_rad": seed_fk_rotation_error,
                    "allowed_position_error_m": SEED_FK_POSITION_TOLERANCE_M,
                    "allowed_rotation_error_rad": SEED_FK_ROTATION_TOLERANCE_RAD,
                }

        watch.lap("validate_s")
        terminal = self.gripper.check(
            targets,
            obstacles,
            clearance_m=clearance_m,
            approach_m=approach_m,
            approach_samples=int(request.get("approach_samples", 8)),
        )
        watch.lap("terminal_check_s")
        safe_indices = np.flatnonzero(np.asarray(terminal["safe"], dtype=bool))
        if safe_indices.size == 0:
            return {
                "success": False,
                "reason": "all_goalset_gripper_swept_volumes_collide",
                "terminal_check": terminal,
                "timings": watch.report(),
            }
        safe_targets = targets[safe_indices]

        filtered, removed = self._remove_start_robot_points(
            obstacles, current
        )
        watch.lap("self_filter_s")
        if len(filtered) < 10:
            return {
                "success": False,
                "reason": "insufficient_scene_points_after_robot_self_filter",
                "input_scene_points": len(obstacles),
                "removed_robot_points": removed,
                "timings": watch.report(),
            }
        scene = self._refresh_observed_scene(
            filtered,
            scene_voxel_m,
            options=_scene_options(request, scene_voxel_m),
            tool_centres=[
                self._tcp_position(current),
                *[np.asarray(target[:3, 3], dtype=np.float64) for target in safe_targets],
            ],
        )
        watch.lap("scene_refresh_s")
        # cuRobo's samplers are stateful. A long-lived service otherwise gives
        # the same request a different (and sometimes exhausted) seed stream
        # depending on earlier plans. Make every request independent and
        # reproducible before entering the multi-stage grasp planner.
        self.planner.reset_seed()

        q_start = JointState.from_position(
            torch.as_tensor(current, dtype=torch.float32, device=self.device).unsqueeze(0),
            joint_names=self.planner.joint_names,
        )
        # A grasp candidate is useful only if the robot can first reach its
        # corresponding pre-grasp. cuRobo's stock plan_grasp first solves the
        # final-grasp goalset, commits to one member, and only then tries that
        # member's pre-grasp. That discarded otherwise viable candidates when
        # the selected final grasp had a blocked approach. Instead submit all
        # pre-grasps as one GPU goalset and let cuRobo select among the complete
        # candidate pool before planning the constrained contact segment.
        pregrasp_targets = safe_targets.copy()
        pregrasp_targets[:, :3, 3] -= (
            approach_m * safe_targets[:, :3, 2]
        )
        def pregrasp_goal(poses: np.ndarray) -> GoalToolPose:
            pose = Pose.from_matrix(
                torch.as_tensor(poses, dtype=torch.float32, device=self.device)
            )
            return GoalToolPose.from_poses(
                {self.planner.tool_frames[0]: pose},
                ordered_tool_frames=self.planner.tool_frames,
                num_goalset=len(poses),
            )

        # cuRobo plans a multi-member goalset without the graph seeding that
        # routes a trajectory around obstacles, so when the goalset reaches no
        # member the pre-grasps are planned one at a time, in request order,
        # within the single-goal budget.
        selection = select_pregrasp(
            len(safe_targets),
            lambda: self.planner.plan_pose(
                pregrasp_goal(pregrasp_targets), q_start, **attempt_kwargs
            ),
            lambda index: self.planner.plan_pose(
                pregrasp_goal(pregrasp_targets[index : index + 1]),
                q_start,
                **attempt_kwargs,
            ),
            single_goal_budget_s=single_goal_budget_s,
        )
        approach_result = selection.result
        pregrasp_planning = {
            "mode": selection.mode,
            "single_goal_budget_s": single_goal_budget_s,
            "attempts": selection.attempts,
        }
        watch.lap("pregrasp_plan_s")
        if not selection.success:
            return {
                "success": False,
                "reason": "no_collision_free_pregrasp_in_goalset",
                "status": str(getattr(approach_result, "status", None)),
                "pregrasp_planning": pregrasp_planning,
                "scene": scene,
                "timings": watch.report(),
                "pregrasp_solver": _solver_timing(approach_result),
                "planner_attempts": attempt_kwargs,
                "terminal_check": terminal,
                "pregrasp_goalset_candidate_count": int(len(safe_targets)),
                "input_scene_points": len(obstacles),
                "planning_scene_points": len(filtered),
                "removed_robot_points": removed,
                "goal_joint_seed_used": False,
                "goal_joint_seed_verified": goal_joint_seed is not None,
                "goal_joint_seed_fk_position_error_m": seed_fk_position_error,
                "goal_joint_seed_fk_rotation_error_rad": seed_fk_rotation_error,
                "planning_time_s": time.monotonic() - started,
            }

        if selection.mode == "single_goal":
            selected_safe_index = int(selection.candidate_index)
        elif approach_result.goalset_index is None:
            return {
                "success": False,
                "reason": "pregrasp_goalset_selection_missing",
                "pregrasp_goalset_candidate_count": int(len(safe_targets)),
            }
        else:
            selected_safe_index = int(
                approach_result.goalset_index.reshape(-1)[0].item()
            )
        if not 0 <= selected_safe_index < len(safe_targets):
            return {
                "success": False,
                "reason": "pregrasp_goalset_selection_out_of_bounds",
                "selected_pregrasp_goalset_index": selected_safe_index,
                "pregrasp_goalset_candidate_count": int(len(safe_targets)),
            }
        selected_request_index = int(safe_indices[selected_safe_index])
        selected_target = targets[selected_request_index]

        approach_end = get_joint_state_at_horizon_index(
            approach_result.js_solution, -1
        ).squeeze(0)
        approach_end = self.planner.kinematics.get_active_js(approach_end)
        selected_goal_pose = Pose.from_matrix(
            torch.as_tensor(
                selected_target[None], dtype=torch.float32, device=self.device
            )
        )
        selected_grasp_goal = GoalToolPose.from_poses(
            {self.planner.tool_frames[0]: selected_goal_pose},
            ordered_tool_frames=self.planner.tool_frames,
            num_goalset=1,
        )
        linear_motion = ToolPoseCriteria.linear_motion(
            axis="z",
            non_terminal_scale=1.0,
            project_distance_to_goal=True,
        )
        standard_criteria = {
            frame: ToolPoseCriteria() for frame in self.planner.tool_frames
        }
        self.planner.update_tool_pose_criteria(
            {frame: linear_motion for frame in self.planner.tool_frames}
        )
        try:
            # All gripper links remain collision-active. The target mask was
            # removed from the world before this point, so no contact-link
            # exception is needed for the final linear segment.
            grasp_result = self.planner.plan_pose(
                selected_grasp_goal, approach_end, **attempt_kwargs
            )
        finally:
            self.planner.update_tool_pose_criteria(standard_criteria)
        watch.lap("grasp_plan_s")
        if (
            grasp_result is None
            or grasp_result.success is None
            or not bool(grasp_result.success.any())
        ):
            return {
                "success": False,
                "reason": "no_collision_free_linear_grasp_path",
                "status": str(getattr(grasp_result, "status", None)),
                "pregrasp_planning": pregrasp_planning,
                "scene": scene,
                "timings": watch.report(),
                "pregrasp_solver": _solver_timing(approach_result),
                "grasp_solver": _solver_timing(grasp_result),
                "planner_attempts": attempt_kwargs,
                "terminal_check": terminal,
                "pregrasp_goalset_candidate_count": int(len(safe_targets)),
                "selected_pregrasp_goalset_index": selected_safe_index,
                "selected_goalset_index": selected_request_index,
                "input_scene_points": len(obstacles),
                "planning_scene_points": len(filtered),
                "removed_robot_points": removed,
                "goal_joint_seed_used": False,
                "goal_joint_seed_verified": goal_joint_seed is not None,
                "goal_joint_seed_fk_position_error_m": seed_fk_position_error,
                "goal_joint_seed_fk_rotation_error_rad": seed_fk_rotation_error,
                "planning_time_s": time.monotonic() - started,
            }

        segments = []
        for phase, trajectory, last_tstep in (
            (
                "approach",
                approach_result.interpolated_trajectory,
                approach_result.interpolated_last_tstep,
            ),
            (
                "grasp",
                grasp_result.interpolated_trajectory,
                grasp_result.interpolated_last_tstep,
            ),
        ):
            if trajectory is None:
                continue
            active_trajectory = self.planner.kinematics.get_active_js(trajectory)
            positions = active_trajectory.position.detach().cpu().numpy().reshape(-1, 7)
            if last_tstep is not None:
                end_index = int(last_tstep.reshape(-1)[0].item())
                if end_index > 0:
                    positions = positions[:end_index]
            if segments and np.allclose(segments[-1]["waypoints"][-1], positions[0]):
                positions = positions[1:]
            segments.append({"phase": phase, "waypoints": positions.tolist()})
        waypoints = [waypoint for segment in segments for waypoint in segment["waypoints"]]
        if not 2 <= len(waypoints) <= PLANNER_MAX_WAYPOINTS:
            return {
                "success": False,
                "reason": "planned_trajectory_waypoint_count_out_of_bounds",
                "waypoint_count": len(waypoints),
                "maximum_waypoint_count": PLANNER_MAX_WAYPOINTS,
            }
        waypoint_array = np.asarray(waypoints, dtype=np.float64)
        start_error = float(np.max(np.abs(waypoint_array[0] - current)))
        if start_error > TRAJECTORY_START_TOLERANCE_RAD:
            return {
                "success": False,
                "reason": "planned_trajectory_start_mismatch",
                "start_error_rad": start_error,
            }
        endpoint_position_error, endpoint_rotation_error = self._tcp_fk_error(
            waypoint_array[-1], selected_target
        )
        if (
            endpoint_position_error > GRASP_IK_POSITION_TOLERANCE_M
            or endpoint_rotation_error > GRASP_IK_ORIENTATION_TOLERANCE_RAD
        ):
            return {
                "success": False,
                "reason": "planned_trajectory_tcp_goal_mismatch",
                "planned_endpoint_position_error_m": endpoint_position_error,
                "planned_endpoint_rotation_error_rad": endpoint_rotation_error,
                "allowed_position_error_m": GRASP_IK_POSITION_TOLERANCE_M,
                "allowed_rotation_error_rad": GRASP_IK_ORIENTATION_TOLERANCE_RAD,
            }
        maximum_step = float(np.max(np.abs(np.diff(waypoint_array, axis=0))))
        if maximum_step > TRAJECTORY_MAX_JOINT_STEP_RAD:
            return {
                "success": False,
                "reason": "planned_trajectory_joint_step_too_large",
                "waypoint_count": len(waypoints),
                "maximum_step_rad": maximum_step,
            }
        watch.lap("postprocess_s")
        return {
            "success": True,
            "reason": "collision_free_path_planned",
            "pregrasp_planning": pregrasp_planning,
            "scene": scene,
            "timings": watch.report(),
            "pregrasp_solver": _solver_timing(approach_result),
            "grasp_solver": _solver_timing(grasp_result),
            "planner_attempts": attempt_kwargs,
            "waypoints": waypoints,
            "segments": segments,
            "goalset_candidate_count": int(len(targets)),
            "goalset_collision_safe_count": int(len(safe_targets)),
            "pregrasp_goalset_candidate_count": int(len(safe_targets)),
            "selected_pregrasp_goalset_index": selected_safe_index,
            "selected_goalset_index": selected_request_index,
            "selected_tcp_pose": selected_target.tolist(),
            "trajectory_dt_s": 0.10,
            "maximum_step_rad": maximum_step,
            "terminal_check": terminal,
            "input_scene_points": len(obstacles),
            "planning_scene_points": len(filtered),
            "removed_robot_points": removed,
            "start_error_rad": start_error,
            "goal_joint_seed_used": False,
            "goal_joint_seed_verified": goal_joint_seed is not None,
            "goal_joint_seed_fk_position_error_m": seed_fk_position_error,
            "goal_joint_seed_fk_rotation_error_rad": seed_fk_rotation_error,
            "planned_endpoint_position_error_m": endpoint_position_error,
            "planned_endpoint_rotation_error_rad": endpoint_rotation_error,
            "planning_time_s": time.monotonic() - started,
        }


    def plan_joint_target(self, request: dict[str, Any]) -> dict[str, Any]:
        """Plan through the full observed scene to an exact joint target."""

        started = time.monotonic()
        watch = _Stopwatch()
        attempt_kwargs = _planner_attempt_kwargs(request)
        current = _finite_array(request.get("current_joints"), (7,), "current_joints")
        target = _finite_array(request.get("target_joints"), (7,), "target_joints")
        for name, joints in (("current_joints", current), ("target_joints", target)):
            if np.any(joints < NERO_JOINT_POSITION_LIMITS_RAD[:, 0]) or np.any(
                joints > NERO_JOINT_POSITION_LIMITS_RAD[:, 1]
            ):
                raise ValueError(f"{name} exceeds official Nero joint limits")
        obstacles = _point_cloud(request.get("obstacle_points"), "obstacle_points")
        scene_voxel_m = float(request.get("scene_voxel_m", 0.012))
        if not 0.003 <= scene_voxel_m <= 0.05:
            raise ValueError("scene_voxel_m must be in [0.003, 0.05]")

        filtered, removed = self._remove_start_robot_points(
            obstacles, current
        )
        if len(filtered) < 10:
            return {
                "success": False,
                "reason": "insufficient_scene_points_after_robot_self_filter",
                "input_scene_points": len(obstacles),
                "removed_robot_points": removed,
            }
        watch.lap("self_filter_s")
        scene = self._refresh_observed_scene(
            filtered,
            scene_voxel_m,
            options=_scene_options(request, scene_voxel_m),
            tool_centres=[self._tcp_position(current), self._tcp_position(target)],
        )
        watch.lap("scene_refresh_s")
        self.planner.reset_seed()

        q_start = JointState.from_position(
            torch.as_tensor(current, dtype=torch.float32, device=self.device).unsqueeze(0),
            joint_names=self.planner.joint_names,
        )
        q_goal = JointState.from_position(
            torch.as_tensor(target, dtype=torch.float32, device=self.device).unsqueeze(0),
            joint_names=self.planner.joint_names,
        )
        attached_bounds = (
            None
            if request.get("attached_bounds_local") is None
            else _attached_bounds(request.get("attached_bounds_local"))
        )
        attached_spheres = 0
        try:
            if attached_bounds is not None:
                attached_spheres = self._attach_local_bounds(
                    q_start, attached_bounds
                )
            result = self.planner.plan_cspace(
                goal_state=q_goal,
                current_state=q_start,
                **attempt_kwargs,
            )
        finally:
            if attached_bounds is not None:
                self.planner.attachment_manager.detach(
                    link_name=ATTACHED_OBJECT_LINK
                )
        watch.lap("plan_s")
        if result is None or result.success is None or not bool(result.success.any()):
            return {
                "success": False,
                "reason": "no_collision_free_joint_target_path",
                "planning_time_s": time.monotonic() - started,
                "scene": scene,
                "timings": watch.report(),
                "solver": _solver_timing(result),
                "planner_attempts": attempt_kwargs,
                "input_scene_points": len(obstacles),
                "planning_scene_points": len(filtered),
                "removed_robot_points": removed,
            }
        trajectory = result.get_interpolated_plan()
        if trajectory is None:
            return {
                "success": False,
                "reason": "joint_target_plan_has_no_interpolated_trajectory",
                "planning_time_s": time.monotonic() - started,
            }
        active_trajectory = self.planner.kinematics.get_active_js(trajectory)
        positions = active_trajectory.position.detach().cpu().numpy().reshape(-1, 7)
        if not 2 <= len(positions) <= PLANNER_MAX_WAYPOINTS:
            return {
                "success": False,
                "reason": "planned_trajectory_waypoint_count_out_of_bounds",
                "waypoint_count": len(positions),
                "maximum_waypoint_count": PLANNER_MAX_WAYPOINTS,
                "planning_time_s": time.monotonic() - started,
            }
        start_error = float(np.max(np.abs(positions[0] - current)))
        goal_error = float(np.max(np.abs(positions[-1] - target)))
        maximum_step = float(np.max(np.abs(np.diff(positions, axis=0))))
        if start_error > TRAJECTORY_START_TOLERANCE_RAD:
            return {
                "success": False,
                "reason": "planned_trajectory_start_mismatch",
                "start_error_rad": start_error,
            }
        if goal_error > TRAJECTORY_GOAL_TOLERANCE_RAD:
            return {
                "success": False,
                "reason": "planned_trajectory_goal_mismatch",
                "goal_error_rad": goal_error,
            }
        if maximum_step > TRAJECTORY_MAX_JOINT_STEP_RAD:
            return {
                "success": False,
                "reason": "planned_trajectory_joint_step_too_large",
                "maximum_step_rad": maximum_step,
            }
        watch.lap("postprocess_s")
        return {
            "success": True,
            "reason": "collision_free_joint_target_path_planned",
            "scene": scene,
            "timings": watch.report(),
            "solver": _solver_timing(result),
            "planner_attempts": attempt_kwargs,
            "waypoints": positions.tolist(),
            "trajectory_dt_s": 0.10,
            "input_scene_points": len(obstacles),
            "planning_scene_points": len(filtered),
            "removed_robot_points": removed,
            "planning_time_s": time.monotonic() - started,
            "start_error_rad": start_error,
            "goal_error_rad": goal_error,
            "maximum_step_rad": maximum_step,
            "attached_object_collision_enabled": attached_bounds is not None,
            "attached_object_sphere_count": attached_spheres,
        }


def serve(planner: NeroGraspMotionPlanner, host: str, port: int) -> None:
    context = zmq.Context.instance()
    socket = context.socket(zmq.REP)
    socket.setsockopt(zmq.LINGER, 0)
    socket.bind(f"tcp://{host}:{port}")
    while True:
        request = msgpack_numpy.unpackb(socket.recv(), raw=False)
        try:
            if not isinstance(request, dict):
                raise TypeError("request must be a dictionary")
            action = request.get("action")
            finetune_cap = planner.set_finetune_cap(request.get("finetune_attempts"))
            if action == "health":
                response = planner.health()
            elif action == "check_grasps":
                response = planner.check_grasps(request)
            elif action == "plan_grasp":
                response = planner.plan_grasp(request)
            elif action == "plan_attached_lift":
                response = planner.plan_attached_lift(request)
            elif action == "plan_joint_target":
                response = planner.plan_joint_target(request)
            else:
                raise ValueError(f"unsupported action {action!r}")
            if isinstance(response, dict) and action != "health":
                response.setdefault("finetune_cap", finetune_cap)
        except Exception as exc:  # Keep the REP socket protocol synchronized.
            response = {
                "success": False,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(limit=12),
            }
        finally:
            planner.set_finetune_cap(None)
        socket.send(msgpack_numpy.packb(response, use_bin_type=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--robot-urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5559)
    parser.add_argument(
        "--scene-mesh-slots",
        type=int,
        default=SCENE_MESH_SLOTS,
        help=(
            "cuRobo mesh cache slots; tiled scenes need ~30-60, single-mesh "
            "deployments (scene_mesh per_voxel/greedy) should use 1"
        ),
    )
    parser.add_argument(
        "--finetune-attempts",
        type=int,
        default=None,
        help=(
            "service-wide cap on cuRobo trajopt finetune passes (0-3); unset "
            "keeps cuRobo's per-plan counts (home 3, grasp/lift 1). Requests "
            "may pass finetune_attempts to override per plan."
        ),
    )
    args = parser.parse_args()
    planner = NeroGraspMotionPlanner(
        args.robot_config.expanduser().resolve(),
        args.robot_urdf.expanduser().resolve(),
        mesh_slots=args.scene_mesh_slots,
    )
    planner.default_finetune_cap = parse_finetune_cap(args.finetune_attempts)
    serve(
        planner,
        args.host,
        args.port,
    )


if __name__ == "__main__":
    main()
