"""Local base alignment using strict Raspberry Pi IK grasp readiness."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import time
from collections.abc import Callable, Collection, Mapping, Sequence
from typing import Any

import numpy as np

from .geometry import matrix_to_quaternion_wxyz, quaternion_wxyz_to_rpy
from .manipulation import PI_MAX_TCP_POSITION_NORM_M, ManipulationController

# goto_grasp_pose submits at most this many goals to the cuRobo goalset.
MAX_CERTIFIED_GOALSET = 16
# How many strictly-certified base poses the trace keeps behind the selected
# one. The trace summariser replaces longer lists with a bare length, and the
# runner-up that matters after a refusal is always a near neighbour in rank.
RANKED_ALTERNATIVES_LOGGED = 8
# Movement-cost tiers summarised in the trace. The tiers that decide a prepare
# are the nearest ones, and the summariser collapses longer lists.
TIER_EVALUATION_LOGGED = 12


def _finite_or_none(value: Any, digits: int = 5) -> float | None:
    """A JSON-safe float for a trace: rounded when finite, otherwise None."""

    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return round(number, digits) if math.isfinite(number) else None


class _CandidatePlanningError(RuntimeError):
    """Candidate-planning failure carrying trace-safe planner evidence."""

    def __init__(self, message: str, diagnostics: Mapping[str, Any]) -> None:
        super().__init__(message)
        self.diagnostics = dict(diagnostics)


@dataclass(frozen=True)
class ManipulationReadinessConfig:
    """Closed-loop limits hidden behind ``prepare_for_manipulation``."""

    sam3_depth_retry_count: int = 2
    maximum_initial_target_distance_m: float = 0.95
    candidate_forward_offsets_m: tuple[float, ...] = (
        -0.10,
        -0.05,
        0.0,
        0.05,
        0.10,
        0.15,
        0.20,
        0.25,
        0.30,
    )
    candidate_lateral_offsets_m: tuple[float, ...] = (
        -0.20,
        -0.15,
        -0.10,
        -0.05,
        0.0,
        0.05,
        0.10,
        0.15,
        0.20,
    )
    candidate_yaw_offsets_rad: tuple[float, ...] = (
        math.radians(-20.0),
        math.radians(-15.0),
        math.radians(-10.0),
        math.radians(-5.0),
        0.0,
        math.radians(5.0),
        math.radians(10.0),
        math.radians(15.0),
        math.radians(20.0),
    )
    candidate_base_limit: int = 729
    grasp_candidate_limit: int = 64
    pi_ik_shortlist_base_limit: int = 128
    pi_ik_grasps_per_base: int = 4
    pi_ik_candidate_limit: int = 1024
    pi_ik_nominal_candidate_limit: int = 512
    pi_ik_preferred_tcp_radius_m: float = 0.42
    ik_batch_size: int = 32
    pi_ik_compute_budget_s: float = 15.0
    robustness_enabled: bool = True
    robustness_position_perturbation_m: float = 0.010
    robustness_yaw_perturbation_rad: float = math.radians(1.0)
    robustness_preferred_converged_variants: int = 3
    # Query the lowest motion tier's complete per-base grasp set first and stop
    # the nominal Pi wave once that tier is settled. The final ranking is
    # tier-first, so no farther base can displace a strict-ready nearer one;
    # only the robustness stage (soft ranking) still runs afterwards.
    pi_ik_best_tier_early_exit: bool = True
    # When the selected base is exactly the current pose, certify the
    # evaluation's own collision-safe, Pi-converged goalset instead of
    # re-running SAM3, the grasp model and Pi IK on the same observation.
    reuse_virtual_certificate_without_motion: bool = True
    minimum_feasible_grasps: int = 1
    candidate_translation_bucket_m: float = 0.020
    candidate_yaw_bucket_rad: float = math.radians(2.0)
    grasp_approach_m: float = 0.10
    base_to_camera_forward_m: float = 0.2143
    base_to_camera_left_m: float = 0.0603
    max_linear_mps: float = 0.10
    max_lateral_mps: float = 0.07
    max_yaw_rad_s: float = 0.25
    position_tolerance_m: float = 0.030
    yaw_tolerance_rad: float = math.radians(3.0)
    motion_timeout_s: float = 20.0
    # How many further strict-ready base poses, in the search's ranked order,
    # a refused motion may hand its turn to before prepare fails. 0 keeps the
    # single attempt at the selected pose.
    motion_alternative_limit: int = 0
    # When every strict-ready base pose's motion has been refused, resume the
    # Pi IK search into the shortlisted poses it never queried and try the
    # poses that certifies there, instead of failing. The best-tier early exit
    # stops at the nearest tier holding a strict-ready base, so when every such
    # base's motion is refused the farther tiers were never queried and the
    # call fails with most of its budget unspent. The resumed search shares
    # the call's Pi IK compute budget and query limit, and the attempts stay
    # bounded by motion_alternative_limit (0 leaves nothing to resume for).
    continue_search_after_refused_motions: bool = False
    ground_plane_max_age_s: float = 2.0

    @classmethod
    def from_mapping(
        cls, values: Mapping[str, Any] | None
    ) -> "ManipulationReadinessConfig":
        payload = dict(values or {})
        legacy_robustness_key = "robustness_min_converged_variants"
        if legacy_robustness_key in payload:
            if "robustness_preferred_converged_variants" in payload:
                raise ValueError(
                    "specify only one of robustness_min_converged_variants and "
                    "robustness_preferred_converged_variants"
                )
            payload["robustness_preferred_converged_variants"] = payload.pop(
                legacy_robustness_key
            )
        aliases = {
            "candidate_yaw_offsets_deg": (
                "candidate_yaw_offsets_rad",
                lambda items: tuple(math.radians(float(item)) for item in items),
            ),
            "yaw_tolerance_deg": ("yaw_tolerance_rad", math.radians),
            "robustness_yaw_perturbation_deg": (
                "robustness_yaw_perturbation_rad",
                math.radians,
            ),
            "candidate_yaw_bucket_deg": (
                "candidate_yaw_bucket_rad",
                math.radians,
            ),
        }
        for alias, (canonical, convert) in aliases.items():
            if alias not in payload:
                continue
            if canonical in payload:
                raise ValueError(f"specify only one of {alias} and {canonical}")
            payload[canonical] = convert(payload.pop(alias))
        tuple_fields = {
            "candidate_forward_offsets_m",
            "candidate_lateral_offsets_m",
            "candidate_yaw_offsets_rad",
        }
        for name in tuple_fields:
            if name in payload:
                raw = payload[name]
                if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
                    raise TypeError(f"{name} must be a numeric sequence")
                payload[name] = tuple(float(item) for item in raw)
        unknown = sorted(set(payload) - set(cls.__dataclass_fields__))
        if unknown:
            raise ValueError(f"unknown manipulation-readiness settings: {unknown}")
        if (
            "pi_ik_candidate_limit" in payload
            and "pi_ik_nominal_candidate_limit" not in payload
        ):
            payload["pi_ik_nominal_candidate_limit"] = min(
                cls.pi_ik_nominal_candidate_limit,
                int(payload["pi_ik_candidate_limit"]),
            )
        config = cls(**payload)
        config._validate()
        return config

    def _validate(self) -> None:
        finite_values = [
            value
            for name, value in asdict(self).items()
            if name
            not in {
                "candidate_forward_offsets_m",
                "candidate_lateral_offsets_m",
                "candidate_yaw_offsets_rad",
                "sam3_depth_retry_count",
                "candidate_base_limit",
                "grasp_candidate_limit",
                "pi_ik_shortlist_base_limit",
                "pi_ik_grasps_per_base",
                "pi_ik_candidate_limit",
                "pi_ik_nominal_candidate_limit",
                "ik_batch_size",
                "robustness_enabled",
                "robustness_preferred_converged_variants",
                "pi_ik_best_tier_early_exit",
                "reuse_virtual_certificate_without_motion",
                "minimum_feasible_grasps",
                "motion_alternative_limit",
                "continue_search_after_refused_motions",
            }
        ]
        sequences = (
            self.candidate_forward_offsets_m,
            self.candidate_lateral_offsets_m,
            self.candidate_yaw_offsets_rad,
        )
        if not all(math.isfinite(float(value)) for value in finite_values):
            raise ValueError("manipulation-readiness numeric settings must be finite")
        if not all(sequence for sequence in sequences) or not all(
            math.isfinite(float(value))
            for sequence in sequences
            for value in sequence
        ):
            raise ValueError("offset sequences must be non-empty and finite")
        if min(self.candidate_forward_offsets_m) < -0.10:
            raise ValueError("reverse candidate offsets must stay within -0.10 m")
        if max(self.candidate_forward_offsets_m) > 0.30:
            raise ValueError("candidate forward offsets must stay within 0.30 m")
        if max(abs(value) for value in self.candidate_lateral_offsets_m) > 0.20:
            raise ValueError("candidate lateral offsets must stay within +/-0.20 m")
        if max(abs(value) for value in self.candidate_yaw_offsets_rad) > math.radians(
            20
        ):
            raise ValueError("candidate yaw offsets must stay within +/-20 degrees")
        integer_bounds = {
            "sam3_depth_retry_count": (self.sam3_depth_retry_count, 0, 5),
            "candidate_base_limit": (self.candidate_base_limit, 1, 1024),
            "grasp_candidate_limit": (self.grasp_candidate_limit, 1, 64),
            "pi_ik_shortlist_base_limit": (
                self.pi_ik_shortlist_base_limit,
                1,
                1024,
            ),
            "pi_ik_grasps_per_base": (self.pi_ik_grasps_per_base, 1, 64),
            "pi_ik_candidate_limit": (self.pi_ik_candidate_limit, 1, 4096),
            "pi_ik_nominal_candidate_limit": (
                self.pi_ik_nominal_candidate_limit,
                1,
                4096,
            ),
            "ik_batch_size": (self.ik_batch_size, 1, 256),
            "robustness_preferred_converged_variants": (
                self.robustness_preferred_converged_variants,
                1,
                7,
            ),
            "minimum_feasible_grasps": (self.minimum_feasible_grasps, 1, 64),
            "motion_alternative_limit": (self.motion_alternative_limit, 0, 24),
        }
        for name, (value, low, high) in integer_bounds.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or not low <= value <= high
            ):
                raise ValueError(f"{name} must be in [{low}, {high}]")
        if self.minimum_feasible_grasps > self.grasp_candidate_limit:
            raise ValueError(
                "minimum_feasible_grasps cannot exceed grasp_candidate_limit"
            )
        positives = (
            self.maximum_initial_target_distance_m,
            self.pi_ik_preferred_tcp_radius_m,
            self.pi_ik_compute_budget_s,
            self.robustness_position_perturbation_m,
            self.robustness_yaw_perturbation_rad,
            self.candidate_translation_bucket_m,
            self.candidate_yaw_bucket_rad,
            self.grasp_approach_m,
            self.max_linear_mps,
            self.max_lateral_mps,
            self.max_yaw_rad_s,
            self.position_tolerance_m,
            self.yaw_tolerance_rad,
            self.motion_timeout_s,
            self.ground_plane_max_age_s,
        )
        if min(positives) <= 0.0:
            raise ValueError("manipulation-readiness limits must be positive")
        if self.maximum_initial_target_distance_m > 1.20:
            raise ValueError(
                "maximum_initial_target_distance_m must be at most 1.20 m"
            )
        if not 0.20 <= self.pi_ik_preferred_tcp_radius_m <= 0.80:
            raise ValueError(
                "pi_ik_preferred_tcp_radius_m must be in [0.20, 0.80] m"
            )
        if self.pi_ik_compute_budget_s > 30.0:
            raise ValueError("pi_ik_compute_budget_s must be at most 30 s")
        if not isinstance(self.robustness_enabled, bool):
            raise TypeError("robustness_enabled must be boolean")
        if not isinstance(self.pi_ik_best_tier_early_exit, bool):
            raise TypeError("pi_ik_best_tier_early_exit must be boolean")
        if not isinstance(self.reuse_virtual_certificate_without_motion, bool):
            raise TypeError(
                "reuse_virtual_certificate_without_motion must be boolean"
            )
        if not isinstance(self.continue_search_after_refused_motions, bool):
            raise TypeError(
                "continue_search_after_refused_motions must be boolean"
            )
        if self.pi_ik_nominal_candidate_limit > self.pi_ik_candidate_limit:
            raise ValueError(
                "pi_ik_nominal_candidate_limit cannot exceed pi_ik_candidate_limit"
            )
        if self.robustness_position_perturbation_m > 0.03:
            raise ValueError(
                "robustness_position_perturbation_m must be at most 0.03 m"
            )
        if self.robustness_yaw_perturbation_rad > math.radians(3.0):
            raise ValueError(
                "robustness_yaw_perturbation must be at most 3 degrees"
            )
        if math.hypot(
            self.base_to_camera_forward_m, self.base_to_camera_left_m
        ) > 0.50:
            raise ValueError("base-to-camera planar offset must be at most 0.50 m")
        if not 0.02 <= self.grasp_approach_m <= 0.25:
            raise ValueError("grasp_approach_m must be in [0.02, 0.25] m")
        if self.max_lateral_mps > self.max_linear_mps:
            raise ValueError("max_lateral_mps cannot exceed max_linear_mps")
        if self.candidate_translation_bucket_m > 0.10:
            raise ValueError(
                "candidate_translation_bucket_m must be at most 0.10 m"
            )
        if self.candidate_yaw_bucket_rad > math.radians(10.0):
            raise ValueError(
                "candidate_yaw_bucket must be at most 10 degrees"
            )


class ManipulationReadinessController:
    """Evaluate local base poses, move, then certify the actual grasp."""

    def __init__(
        self,
        env: Any,
        *,
        config: Mapping[str, Any] | ManipulationReadinessConfig | None = None,
        segment_client_factory: Callable[[], Callable[..., Any]] | None = None,
        grasp_backend_factory: Callable[[dict[str, Any]], Any] | None = None,
        motion_planner_factory: Callable[[dict[str, Any]], Any] | None = None,
        manipulation_controller: ManipulationController | None = None,
        debug_callback: Callable[[dict[str, Any]], None] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        required = (
            "navigation_frame",
            "plan_arm_poses",
            "manipulation_calibration",
            "observe",
        )
        missing = [name for name in required if not callable(getattr(env, name, None))]
        if missing or not hasattr(env, "controller"):
            raise TypeError(
                "ManipulationReadinessController environment is missing: "
                f"{missing + ([] if hasattr(env, 'controller') else ['controller'])}"
            )
        self.env = env
        self.config = (
            config
            if isinstance(config, ManipulationReadinessConfig)
            else ManipulationReadinessConfig.from_mapping(config)
        )
        self.manipulation = manipulation_controller or ManipulationController(
            env,
            segment_client_factory=segment_client_factory,
            grasp_backend_factory=grasp_backend_factory,
            motion_planner_factory=motion_planner_factory,
        )
        self.debug_callback = debug_callback
        self.clock = clock
        self.last_debug: dict[str, Any] | None = None

    @staticmethod
    def _navigation_from_camera(
        down_camera_xyz: np.ndarray,
    ) -> np.ndarray:
        down = np.asarray(down_camera_xyz, dtype=np.float64).reshape(-1)
        if down.shape != (3,) or not np.all(np.isfinite(down)):
            raise RuntimeError("dynamic_ground_plane_down_vector_invalid")
        norm = float(np.linalg.norm(down))
        if norm <= 1e-6:
            raise RuntimeError("dynamic_ground_plane_down_vector_invalid")
        down /= norm
        optical_forward = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
        forward = optical_forward - down * float(np.dot(optical_forward, down))
        forward_norm = float(np.linalg.norm(forward))
        if forward_norm <= 1e-6:
            raise RuntimeError("dynamic_ground_plane_forward_axis_invalid")
        forward /= forward_norm
        left = np.cross(forward, down)
        left /= np.linalg.norm(left)
        return np.vstack([forward, left, -down])

    def _ground_geometry(self, frame: Any) -> tuple[float, np.ndarray, np.ndarray]:
        height = getattr(frame, "ground_camera_height_m", None)
        down = getattr(frame, "ground_down_camera_xyz", None)
        ground_timestamp = getattr(frame, "ground_plane_timestamp_ns", None)
        frame_timestamp = getattr(frame, "timestamp_ns", None)
        if any(value is None for value in (height, down, ground_timestamp, frame_timestamp)):
            raise RuntimeError("dynamic_ground_plane_unavailable")
        age_s = max(0.0, (int(frame_timestamp) - int(ground_timestamp)) * 1e-9)
        if age_s > self.config.ground_plane_max_age_s:
            raise RuntimeError("dynamic_ground_plane_stale")
        camera_height = float(height)
        if not math.isfinite(camera_height) or camera_height <= 0.0:
            raise RuntimeError("dynamic_ground_plane_height_invalid")
        navigation_from_camera = self._navigation_from_camera(np.asarray(down))
        return camera_height, np.asarray(down, dtype=np.float64), navigation_from_camera

    def _camera_motion(
        self,
        forward_m: float,
        left_m: float,
        yaw_rad: float,
        navigation_from_camera: np.ndarray,
    ) -> np.ndarray:
        camera_from_navigation = navigation_from_camera.T
        cosine, sine = math.cos(yaw_rad), math.sin(yaw_rad)
        navigation_rotation = np.asarray(
            [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = (
            camera_from_navigation @ navigation_rotation @ navigation_from_camera
        )
        # The base command is defined at the swerve rotation center, while the
        # RGB-D camera is offset from it. A yaw therefore translates the camera
        # by (R - I) * base_from_camera in addition to the commanded base-center
        # translation. Omitting this lever arm makes every nonzero-yaw virtual
        # grasp target disagree with the pose reached by the real chassis.
        base_from_camera = np.asarray(
            [
                self.config.base_to_camera_forward_m,
                self.config.base_to_camera_left_m,
                0.0,
            ],
            dtype=np.float64,
        )
        camera_translation_navigation = (
            np.asarray([forward_m, left_m, 0.0], dtype=np.float64)
            + navigation_rotation @ base_from_camera
            - base_from_camera
        )
        transform[:3, 3] = (
            camera_from_navigation @ camera_translation_navigation
        )
        return transform

    def _movement_cost(
        self, forward_m: float, left_m: float, yaw_rad: float
    ) -> tuple[float, int]:
        """Return normalized chassis motion and its near-equivalent cost tier."""

        normalized = (
            math.hypot(forward_m, left_m)
            / self.config.candidate_translation_bucket_m
            + abs(yaw_rad) / self.config.candidate_yaw_bucket_rad
        )
        return float(normalized), int(math.floor(normalized + 1e-9))

    def _robustness_variants(self) -> tuple[tuple[str, float, float, float], ...]:
        """Return bounded base-pose errors covered by one Pi certificate."""

        if not self.config.robustness_enabled:
            return (("nominal", 0.0, 0.0, 0.0),)
        position = self.config.robustness_position_perturbation_m
        yaw = self.config.robustness_yaw_perturbation_rad
        return (
            ("nominal", 0.0, 0.0, 0.0),
            ("forward_plus", position, 0.0, 0.0),
            ("forward_minus", -position, 0.0, 0.0),
            ("left_plus", 0.0, position, 0.0),
            ("left_minus", 0.0, -position, 0.0),
            ("yaw_plus", 0.0, 0.0, yaw),
            ("yaw_minus", 0.0, 0.0, -yaw),
        )

    def _virtual_candidates(
        self,
        generated: Mapping[str, Any],
        navigation_from_camera: np.ndarray,
    ) -> tuple[list[dict[str, Any]], list[int]]:
        scores = np.asarray(generated["scores"], dtype=np.float64)
        indices = np.asarray(generated["candidate_indices"], dtype=np.int64)
        grasp_indices = indices[
            np.argsort(-scores[indices], kind="stable")
        ][: self.config.grasp_candidate_limit].tolist()
        grasps = np.asarray(generated["grasps_camera_from_model"], dtype=np.float64)
        model_from_endpoint = np.asarray(
            generated["model_from_controlled_endpoint"], dtype=np.float64
        )
        arm_from_camera = np.asarray(generated["arm_from_camera"], dtype=np.float64)
        current_arm_from_camera_inverse = np.linalg.inv(arm_from_camera)
        candidates: list[dict[str, Any]] = []
        for forward_m in self.config.candidate_forward_offsets_m:
            for left_m in self.config.candidate_lateral_offsets_m:
                for yaw_rad in self.config.candidate_yaw_offsets_rad:
                    movement_cost, movement_cost_tier = self._movement_cost(
                        forward_m, left_m, yaw_rad
                    )
                    motion = self._camera_motion(
                        forward_m, left_m, yaw_rad, navigation_from_camera
                    )
                    inverse_motion = np.linalg.inv(motion)
                    arm_targets = []
                    for grasp_index in grasp_indices:
                        camera_from_endpoint = (
                            grasps[grasp_index] @ model_from_endpoint
                        )
                        arm_target = (
                            arm_from_camera @ inverse_motion @ camera_from_endpoint
                        )
                        arm_targets.append(arm_target)
                    candidate_arm_from_camera = arm_from_camera @ inverse_motion
                    candidates.append(
                        {
                            "forward_m": float(forward_m),
                            "left_m": float(left_m),
                            "yaw_rad": float(yaw_rad),
                            "movement_cost": movement_cost,
                            "movement_cost_tier": movement_cost_tier,
                            "camera_motion": motion,
                            "candidate_arm_from_current_arm": (
                                candidate_arm_from_camera
                                @ current_arm_from_camera_inverse
                            ),
                            "arm_targets": arm_targets,
                            "proxy_score": float(
                                math.hypot(forward_m, left_m)
                                + 0.1 * abs(yaw_rad)
                            ),
                        }
                    )
        candidates.sort(key=lambda item: item["proxy_score"])
        if len(candidates) > self.config.candidate_base_limit:
            raise RuntimeError(
                "configured local base grid exceeds candidate_base_limit: "
                f"grid={len(candidates)} limit={self.config.candidate_base_limit}; "
                "increase the limit instead of silently dropping far candidates"
            )
        for index, candidate in enumerate(candidates):
            candidate["candidate_index"] = index
        return candidates, grasp_indices

    def _evaluate_candidates(
        self,
        generated: Mapping[str, Any],
        frame: Any,
    ) -> dict[str, Any]:
        # Sub-stage wall clock inside the evaluation. The Pi IK waves are
        # already reported as ``pi_ik_compute_elapsed_s``; these cover the local
        # expansion and ranking work that surrounds them, which the primitive's
        # ``virtual_candidate_evaluation`` bucket otherwise reports as one
        # opaque number.
        substage_timings_s: dict[str, float] = {}
        substage_started = self.clock()
        camera_height, down, navigation_from_camera = self._ground_geometry(frame)
        substage_timings_s["ground_geometry"] = max(
            0.0, self.clock() - substage_started
        )
        substage_started = self.clock()
        candidates, grasp_indices = self._virtual_candidates(
            generated, navigation_from_camera
        )
        substage_timings_s["virtual_candidates"] = max(
            0.0, self.clock() - substage_started
        )
        # The grasp/scene relationship is invariant under a rigid virtual base
        # transform, so run the official open-gripper swept-volume gate once
        # on the current scene before expanding the full base x grasp IK grid.
        grasps = np.asarray(generated["grasps_camera_from_model"], dtype=np.float64)
        model_from_endpoint = np.asarray(
            generated["model_from_controlled_endpoint"], dtype=np.float64
        )
        arm_from_camera = np.asarray(generated["arm_from_camera"], dtype=np.float64)
        current_targets = np.stack(
            [
                arm_from_camera @ grasps[index] @ model_from_endpoint
                for index in grasp_indices
            ],
            axis=0,
        )
        substage_started = self.clock()
        obstacle_points = self.manipulation._collision_scene_points(
            np.asarray(generated["depth_m"]),
            np.asarray(generated["mask"], dtype=bool),
            np.asarray(generated["intrinsics"]),
            arm_from_camera,
        )
        substage_timings_s["collision_scene_points"] = max(
            0.0, self.clock() - substage_started
        )
        manipulation_config = self.env.manipulation_config
        substage_started = self.clock()
        collision_check = self.manipulation._motion_planner_client().check_grasps(
            current_targets,
            obstacle_points,
            clearance_m=float(
                manipulation_config.get("grasp_collision_clearance_m", 0.008)
            ),
            approach_m=self.config.grasp_approach_m,
            approach_samples=int(
                manipulation_config.get("grasp_collision_approach_samples", 6)
            ),
        )
        substage_timings_s["collision_check_rpc"] = max(
            0.0, self.clock() - substage_started
        )
        safe = collision_check.get("safe")
        if not isinstance(safe, list) or len(safe) != len(grasp_indices):
            raise RuntimeError(
                f"grasp collision service returned invalid result: {collision_check!r}"
            )
        safe_local_indices = [index for index, value in enumerate(safe) if value]
        if not safe_local_indices:
            raise RuntimeError(
                "all grasp-pool candidates collide with observed non-target geometry"
            )
        grasp_pool_count = len(grasp_indices)
        grasp_indices = [grasp_indices[index] for index in safe_local_indices]
        for candidate in candidates:
            candidate["arm_targets"] = [
                candidate["arm_targets"][index] for index in safe_local_indices
            ]
        substage_started = self.clock()
        records: list[dict[str, Any]] = []
        pi_bound_rejected_count = 0
        for candidate in candidates:
            candidate["plans"] = []
            candidate["pi_bound_rejected_grasps"] = 0
            candidate["eligible_grasps"] = 0
            candidate["shortlisted_grasps"] = 0
            candidate["nearest_tcp_radius_m"] = math.inf
            for local_grasp_index, arm_target in enumerate(candidate["arm_targets"]):
                target_position = np.asarray(arm_target[:3, 3], dtype=np.float64)
                pregrasp_position = (
                    target_position
                    - self.config.grasp_approach_m * arm_target[:3, 2]
                )
                target_radius = float(np.linalg.norm(target_position))
                pregrasp_radius = float(np.linalg.norm(pregrasp_position))
                candidate["nearest_tcp_radius_m"] = min(
                    float(candidate["nearest_tcp_radius_m"]), target_radius
                )
                if (
                    target_radius > PI_MAX_TCP_POSITION_NORM_M
                    or pregrasp_radius > PI_MAX_TCP_POSITION_NORM_M
                ):
                    pi_bound_rejected_count += 1
                    candidate["pi_bound_rejected_grasps"] += 1
                    continue
                candidate["eligible_grasps"] += 1
                # ``pose_xyz_rpy`` and ``camera_from_endpoint`` are filled in
                # for the shortlist only. Ranking needs the two radii alone,
                # while the pose costs an SVD per pair (the SO(3) projection in
                # matrix_to_quaternion_wxyz) and the whole lattice is two orders
                # of magnitude larger than the shortlist that reaches Pi IK.
                records.append(
                    {
                        "candidate": candidate,
                        "local_grasp_index": local_grasp_index,
                        "grasp_index": int(grasp_indices[local_grasp_index]),
                        "arm_target": arm_target,
                        "camera_from_endpoint": None,
                        "pose_xyz_rpy": None,
                        "tcp_radius_m": target_radius,
                        "pregrasp_radius_m": pregrasp_radius,
                    }
                )
        if not records:
            raise RuntimeError("no virtual base/grasp candidates to evaluate")
        eligible_pair_count = len(records)
        scores = np.asarray(generated["scores"], dtype=np.float64)
        preferred_radius = self.config.pi_ik_preferred_tcp_radius_m
        for record in records:
            record["pi_prefilter_score"] = float(
                abs(record["tcp_radius_m"] - preferred_radius)
                + 0.5 * abs(record["pregrasp_radius_m"] - preferred_radius)
                + 0.1 * float(record["candidate"]["proxy_score"])
                - 0.02 * float(scores[record["grasp_index"]])
            )
        substage_timings_s["record_expansion"] = max(
            0.0, self.clock() - substage_started
        )

        substage_started = self.clock()
        # Keep the complete fine lattice above, but bound exact Pi work. Rank
        # base poses by their best cheap workspace proxy, then take grasps in
        # round-robin passes so the shortlist spans many bases before spending
        # a second query on any one base. This proxy never certifies success.
        by_base: dict[int, list[dict[str, Any]]] = {}
        for record in records:
            by_base.setdefault(
                int(record["candidate"]["candidate_index"]), []
            ).append(record)
        ranked_bases = sorted(
            by_base.values(),
            key=lambda items: (
                int(items[0]["candidate"]["movement_cost_tier"]),
                min(float(item["pi_prefilter_score"]) for item in items),
                float(items[0]["candidate"]["movement_cost"]),
                float(items[0]["candidate"]["proxy_score"]),
            ),
        )[: self.config.pi_ik_shortlist_base_limit]
        for items in ranked_bases:
            items.sort(
                key=lambda item: (
                    float(item["pi_prefilter_score"]),
                    -float(scores[item["grasp_index"]]),
                )
            )
        robustness_variants = self._robustness_variants()
        shortlisted: list[dict[str, Any]] = []
        query_limit = self.config.pi_ik_candidate_limit
        # The final ranking is tier-first, so no farther tier can win once a
        # tier holding a strict-ready base is fully evaluated. Emit the
        # shortlist one whole tier at a time (bases already arrive tier-sorted)
        # so the early-exit predicate can fire there. Prioritizing only the
        # single best tier and interleaving every remaining tier -- the previous
        # schedule -- pushed that completion to the end of the shortlist, which
        # exhausted the compute budget even when the winner sat at a near tier.
        if self.config.pi_ik_best_tier_early_exit:
            tier_groups: dict[int, list[list[dict[str, Any]]]] = {}
            for items in ranked_bases:
                tier_groups.setdefault(
                    int(items[0]["candidate"]["movement_cost_tier"]), []
                ).append(items)
            for tier in sorted(tier_groups):
                for items in tier_groups[tier]:
                    for record in items[: self.config.pi_ik_grasps_per_base]:
                        if len(shortlisted) >= query_limit:
                            break
                        shortlisted.append(record)
                    if len(shortlisted) >= query_limit:
                        break
                if len(shortlisted) >= query_limit:
                    break
        else:
            # Without the predicate there is nothing to exit early on, so keep
            # spreading the budget across bases before deepening any one of them.
            maximum_grasp_rank = min(
                self.config.pi_ik_grasps_per_base,
                max((len(items) for items in ranked_bases), default=0),
            )
            for grasp_rank in range(maximum_grasp_rank):
                for items in ranked_bases:
                    if grasp_rank >= len(items):
                        continue
                    if len(shortlisted) >= query_limit:
                        break
                    shortlisted.append(items[grasp_rank])
                if len(shortlisted) >= query_limit:
                    break
        for pair_index, record in enumerate(shortlisted):
            # Deferred from the lattice expansion above; only these records
            # reach Pi IK or a robustness variant.
            arm_target = record["arm_target"]
            quaternion = matrix_to_quaternion_wxyz(arm_target[:3, :3])
            record.update(
                {
                    "camera_from_endpoint": (
                        grasps[record["grasp_index"]] @ model_from_endpoint
                    ),
                    "pose_xyz_rpy": np.concatenate(
                        [arm_target[:3, 3], quaternion_wxyz_to_rpy(quaternion)]
                    ).tolist(),
                    "robustness_pair_index": pair_index,
                    "robustness_variant": "nominal",
                    "robustness_delta_forward_m": 0.0,
                    "robustness_delta_left_m": 0.0,
                    "robustness_delta_yaw_rad": 0.0,
                }
            )
        if not shortlisted:
            raise RuntimeError("Pi IK shortlist is unexpectedly empty")
        shortlisted_base_indices = {
            int(items[0]["candidate"]["candidate_index"]) for items in ranked_bases
        }
        for record in shortlisted:
            record["candidate"]["shortlisted_grasps"] += 1

        def perturbed_record(
            nominal: Mapping[str, Any],
            variant: tuple[str, float, float, float],
        ) -> dict[str, Any] | None:
            nonlocal pi_bound_rejected_count
            label, delta_forward, delta_left, delta_yaw = variant
            candidate = nominal["candidate"]
            perturbed_motion = self._camera_motion(
                float(candidate["forward_m"]) + delta_forward,
                float(candidate["left_m"]) + delta_left,
                float(candidate["yaw_rad"]) + delta_yaw,
                navigation_from_camera,
            )
            arm_target = (
                arm_from_camera
                @ np.linalg.inv(perturbed_motion)
                @ np.asarray(nominal["camera_from_endpoint"], dtype=np.float64)
            )
            target_position = np.asarray(arm_target[:3, 3], dtype=np.float64)
            pregrasp_position = (
                target_position
                - self.config.grasp_approach_m * arm_target[:3, 2]
            )
            if (
                np.linalg.norm(target_position) > PI_MAX_TCP_POSITION_NORM_M
                or np.linalg.norm(pregrasp_position) > PI_MAX_TCP_POSITION_NORM_M
            ):
                pi_bound_rejected_count += 1
                return None
            quaternion = matrix_to_quaternion_wxyz(arm_target[:3, :3])
            pose = np.concatenate(
                [arm_target[:3, 3], quaternion_wxyz_to_rpy(quaternion)]
            )
            return {
                **nominal,
                "arm_target": arm_target,
                "pose_xyz_rpy": pose.tolist(),
                "tcp_radius_m": float(np.linalg.norm(target_position)),
                "pregrasp_radius_m": float(np.linalg.norm(pregrasp_position)),
                "robustness_variant": label,
                "robustness_delta_forward_m": delta_forward,
                "robustness_delta_left_m": delta_left,
                "robustness_delta_yaw_rad": delta_yaw,
            }

        substage_timings_s["shortlist_ranking"] = max(
            0.0, self.clock() - substage_started
        )

        arm = str(generated["arm"])
        pi_batches: list[dict[str, Any]] = []
        # One row per Pi query in schedule order. A trace can then be replayed
        # at any smaller budget offline: "certified by query k" is a prefix
        # scan over this list, which the per-batch counts alone cannot give.
        pi_ik_query_log: list[dict[str, Any]] = []
        pi_started = self.clock()
        completed_record_count = 0
        budget_exhausted = False
        early_exit_stage: str | None = None
        terminal_batch_error: dict[str, Any] | None = None
        previous_batch_elapsed_s: float | None = None
        previous_batch_size = 0
        reported_worker_count = 4
        scheduled_query_count = 0

        def run_pi_records(
            stage_records: Sequence[dict[str, Any]],
            stage: str,
            *,
            stop_when: Callable[[], bool] | None = None,
            already_scheduled: int = 0,
        ) -> None:
            nonlocal budget_exhausted
            nonlocal completed_record_count
            nonlocal early_exit_stage
            nonlocal previous_batch_elapsed_s
            nonlocal previous_batch_size
            nonlocal reported_worker_count
            nonlocal scheduled_query_count
            nonlocal terminal_batch_error
            # ``already_scheduled`` of these records were counted by an
            # earlier pass that never reached them, so each pair is requested
            # once however many passes schedule it.
            scheduled_query_count += len(stage_records) - already_scheduled
            start = 0
            while (
                start < len(stage_records)
                and completed_record_count < self.config.pi_ik_candidate_limit
                and terminal_batch_error is None
            ):
                query_capacity = (
                    self.config.pi_ik_candidate_limit - completed_record_count
                )
                batch_size = min(
                    self.config.ik_batch_size,
                    len(stage_records) - start,
                    query_capacity,
                )
                budget_limited_batch = False
                if previous_batch_elapsed_s is not None and previous_batch_size > 0:
                    elapsed_s = max(0.0, self.clock() - pi_started)
                    remaining_s = self.config.pi_ik_compute_budget_s - elapsed_s
                    guarded_remaining_s = remaining_s - 0.05
                    estimated_candidate_s = (
                        previous_batch_elapsed_s / previous_batch_size * 1.20
                    )
                    if estimated_candidate_s > 0.0:
                        fitting_count = int(
                            math.floor(guarded_remaining_s / estimated_candidate_s)
                        )
                        budget_limited_batch = fitting_count < batch_size
                        batch_size = min(batch_size, fitting_count)
                    minimum_full_wave = min(
                        reported_worker_count,
                        self.config.ik_batch_size,
                        len(stage_records) - start,
                    )
                    if batch_size < minimum_full_wave:
                        budget_exhausted = True
                        break
                    if budget_limited_batch and batch_size < len(stage_records) - start:
                        batch_size -= batch_size % reported_worker_count
                        if batch_size <= 0:
                            budget_exhausted = True
                            break
                batch = stage_records[start : start + batch_size]
                batch_started = self.clock()
                try:
                    planning = self.env.plan_arm_poses(
                        arm, [record["pose_xyz_rpy"] for record in batch]
                    )
                except Exception as exc:
                    terminal_batch_error = {
                        "stage": stage,
                        "failed_batch_start": start,
                        "failed_batch_size": len(batch),
                        "error": f"{type(exc).__name__}: {str(exc)[:200]}",
                    }
                    if completed_record_count == 0:
                        raise _CandidatePlanningError(
                            "parallel Pi IK batch execution failed",
                            {
                                "stage": "parallel_pi_ik_execution",
                                "evaluated_base_count": len(candidates),
                                "pi_eligible_pair_count": eligible_pair_count,
                                "pi_shortlisted_base_count": len(ranked_bases),
                                "pi_ik_requested_count": scheduled_query_count,
                                "ik_query_count": completed_record_count,
                                "collision_safe_grasp_count": len(grasp_indices),
                                "bases_with_converged_grasp": 0,
                                "failed_batch_start": start,
                                "failed_batch_size": len(batch),
                                "completed_batches": len(pi_batches),
                                "error": terminal_batch_error["error"],
                                "substage_timings_s": dict(substage_timings_s),
                                "pi_ik_query_log": list(pi_ik_query_log),
                            },
                        ) from exc
                    break
                batch_elapsed_s = max(0.0, self.clock() - batch_started)
                plans = (
                    planning.get("plans")
                    if isinstance(planning, Mapping)
                    else None
                )
                if not isinstance(plans, list):
                    raise RuntimeError(f"Pi IK returned no plans: {planning!r}")
                pi_batches.append(
                    {
                        "stage": stage,
                        "candidate_count": len(batch),
                        "rpc_elapsed_s": float(batch_elapsed_s),
                        "server_ik_solve_elapsed_s": planning.get(
                            "ik_solve_elapsed_s"
                        ),
                        "server_ik_parallel_workers": planning.get(
                            "ik_parallel_workers"
                        ),
                        "valid_plan_count": planning.get("valid_plan_count"),
                    }
                )
                previous_batch_elapsed_s = float(batch_elapsed_s)
                previous_batch_size = len(batch)
                worker_count = planning.get("ik_parallel_workers")
                if isinstance(worker_count, int) and worker_count > 0:
                    reported_worker_count = worker_count
                completed_record_count += len(batch)
                by_index = {
                    int(plan["candidate_index"]): plan
                    for plan in plans
                    if isinstance(plan, Mapping)
                    and isinstance(plan.get("candidate_index"), int)
                }
                for local_index, record in enumerate(batch):
                    plan = dict(by_index.get(local_index, {"success": False}))
                    position_error = float(
                        plan.get("ik_position_error_m", np.inf)
                    )
                    rotation_error = float(
                        plan.get("ik_rotation_error_rad", np.inf)
                    )
                    finite = bool(
                        plan.get("success", False)
                        and np.isfinite(position_error)
                        and np.isfinite(rotation_error)
                    )
                    converged = bool(finite and plan.get("ik_converged", False))
                    # Mark the scheduled record itself so the early-exit
                    # predicate can see which shortlist entries are settled.
                    record["ik_evaluated"] = True
                    record["ik_converged"] = converged
                    enriched = {
                        **record,
                        "ik_plan": plan,
                        "ik_finite": finite,
                        "ik_acceptable": converged,
                        "ik_converged": converged,
                    }
                    record["candidate"]["plans"].append(enriched)
                    base = record["candidate"]
                    pi_ik_query_log.append(
                        {
                            "order": completed_record_count - len(batch) + local_index,
                            "stage": stage,
                            "batch_index": len(pi_batches) - 1,
                            "base_index": int(base["candidate_index"]),
                            "movement_cost_tier": int(base["movement_cost_tier"]),
                            "movement_cost": round(float(base["movement_cost"]), 4),
                            "forward_m": float(base["forward_m"]),
                            "left_m": float(base["left_m"]),
                            "yaw_rad": round(float(base["yaw_rad"]), 5),
                            "grasp_index": int(record["grasp_index"]),
                            "robustness_variant": str(record["robustness_variant"]),
                            "ik_converged": converged,
                        }
                    )
                start += len(batch)
                if stop_when is not None and stop_when():
                    early_exit_stage = stage
                    break

        nominal_limit = min(
            self.config.pi_ik_nominal_candidate_limit,
            len(shortlisted),
        )
        scheduled_nominal = shortlisted[:nominal_limit]
        # Shortlist positions some pass has scheduled. The early exit or the
        # budget can leave a scheduled pair unqueried, and a resumed pass that
        # schedules it again must not request it twice.
        scheduled_shortlist_positions: set[int] = set(range(nominal_limit))

        shortlisted_by_tier: dict[int, list[dict[str, Any]]] = {}
        for record in shortlisted:
            shortlisted_by_tier.setdefault(
                int(record["candidate"]["movement_cost_tier"]), []
            ).append(record)

        def best_tier_settled(
            scheduled: Sequence[Mapping[str, Any]] = scheduled_nominal,
            excluded_base_indices: Collection[int] = frozenset(),
        ) -> bool:
            """True once the nearest scheduled tier holding a strict-ready base is
            fully evaluated. Tiers are visited in ascending order: a fully
            evaluated tier without a ready base is skipped, an incompletely
            evaluated tier blocks the exit, because either could still change
            which tier wins the tier-first final ranking.

            A resumed search passes its own schedule and the bases whose motion
            was already attempted, which no longer count as ready. A tier's
            convergences are counted over its whole shortlist, so a base an
            earlier pass left part-evaluated keeps the grasps it converged
            there; records outside the schedule that were never queried add
            nothing."""

            by_tier: dict[int, list[Mapping[str, Any]]] = {}
            for record in scheduled:
                by_tier.setdefault(
                    int(record["candidate"]["movement_cost_tier"]), []
                ).append(record)
            for tier in sorted(by_tier):
                tier_records = by_tier[tier]
                if not all(
                    bool(record.get("ik_evaluated", False)) for record in tier_records
                ):
                    return False
                converged_per_base: dict[int, int] = {}
                for record in shortlisted_by_tier.get(tier, tier_records):
                    base_index = int(record["candidate"]["candidate_index"])
                    if base_index in excluded_base_indices:
                        continue
                    if bool(record.get("ik_converged", False)):
                        converged_per_base[base_index] = (
                            converged_per_base.get(base_index, 0) + 1
                        )
                if any(
                    count >= self.config.minimum_feasible_grasps
                    for count in converged_per_base.values()
                ):
                    return True
            return False

        def run_robustness(nominal_converged: list[dict[str, Any]]) -> None:
            """Query the perturbation bundle of converged nominal plans, nearest
            tier first, within the queries and budget left."""

            if not (
                self.config.robustness_enabled
                and nominal_converged
                and not budget_exhausted
                and terminal_batch_error is None
            ):
                return
            nominal_converged.sort(
                key=lambda item: (
                    int(item["candidate"]["movement_cost_tier"]),
                    float(item["ik_plan"].get("joint_travel_l2_rad", np.inf)),
                    float(item["candidate"]["movement_cost"]),
                    float(item["candidate"]["proxy_score"]),
                    float(item["pi_prefilter_score"]),
                )
            )
            perturbations = robustness_variants[1:]
            available_queries = max(
                0,
                self.config.pi_ik_candidate_limit - completed_record_count,
            )
            robust_pair_limit = available_queries // len(perturbations)
            robustness_records: list[dict[str, Any]] = []
            for nominal in nominal_converged[:robust_pair_limit]:
                variant_records = [
                    perturbed_record(nominal, variant)
                    for variant in perturbations
                ]
                if all(record is not None for record in variant_records):
                    robustness_records.extend(
                        record for record in variant_records if record is not None
                    )
            run_pi_records(robustness_records, "robustness")

        run_pi_records(
            scheduled_nominal,
            "nominal",
            stop_when=(
                best_tier_settled
                if self.config.pi_ik_best_tier_early_exit
                else None
            ),
        )
        nominal_converged = [
            plan
            for candidate in candidates
            for plan in candidate["plans"]
            if plan["robustness_variant"] == "nominal"
            and plan["ik_converged"]
        ]
        if not nominal_converged and not budget_exhausted:
            run_pi_records(shortlisted[nominal_limit:], "nominal_extended")
            scheduled_shortlist_positions.update(range(nominal_limit, len(shortlisted)))
            nominal_converged = [
                plan
                for candidate in candidates
                for plan in candidate["plans"]
                if plan["robustness_variant"] == "nominal"
                and plan["ik_converged"]
            ]

        run_robustness(nominal_converged)

        # The Pi window closes here. The qualification and ranking below are
        # local work, timed as their own substage rather than folded into the
        # Pi IK figure.
        pi_ik_compute_elapsed_s = max(0.0, self.clock() - pi_started)
        substage_started = self.clock()

        def rank_candidates() -> dict[str, Any]:
            """Qualify every base from all its plans so far and rank the ready ones.

            Plans only accumulate, so a resumed search that runs this again
            re-derives every base's readiness from all passes, and a base that
            was strict-ready before stays strict-ready.
            """

            for candidate in candidates:
                plans_by_pair: dict[int, list[dict[str, Any]]] = {}
                for plan in candidate["plans"]:
                    plans_by_pair.setdefault(
                        int(plan["robustness_pair_index"]), []
                    ).append(plan)
                qualified: list[dict[str, Any]] = []
                nominal_evaluated_count = 0
                nominal_finite_count = 0
                best_variant_count = 0
                for pair_plans in plans_by_pair.values():
                    nominal = next(
                        (
                            plan
                            for plan in pair_plans
                            if plan["robustness_variant"] == "nominal"
                        ),
                        None,
                    )
                    if nominal is None:
                        continue
                    nominal_evaluated_count += 1
                    nominal_finite_count += int(bool(nominal["ik_finite"]))
                    converged_variant_count = sum(
                        bool(plan["ik_converged"]) for plan in pair_plans
                    )
                    best_variant_count = max(
                        best_variant_count, converged_variant_count
                    )
                    # Nominal strict convergence is the only readiness gate.
                    # Small perturbations are deliberately a soft ranking
                    # signal because the actual arrival pose is sampled and
                    # strictly certified again after the base has stopped.
                    if nominal["ik_converged"]:
                        qualified.append(
                            {
                                **nominal,
                                "robustness_converged_variants": (
                                    converged_variant_count
                                ),
                                "robustness_evaluated_variants": len(pair_plans),
                            }
                        )
                candidate["feasible_grasp_count"] = len(qualified)
                candidate["converged_grasp_count"] = len(qualified)
                candidate["nominal_grasp_evaluated_count"] = nominal_evaluated_count
                candidate["finite_grasp_count"] = nominal_finite_count
                candidate["robustness_best_converged_variants"] = best_variant_count
                candidate["selected_grasp"] = None
                candidate["score"] = -math.inf
                candidate["strict_ready"] = False
                candidate["selection_quality"] = "unavailable"
                if len(qualified) < self.config.minimum_feasible_grasps:
                    continue

                def grasp_preference(item: Mapping[str, Any]) -> tuple[Any, ...]:
                    return (
                        -min(
                            self.config.robustness_preferred_converged_variants,
                            int(item["robustness_converged_variants"]),
                        ),
                        float(item["ik_plan"].get("joint_travel_l2_rad", np.inf)),
                        float(item["ik_plan"].get("joint_travel_max_rad", np.inf)),
                        float(item["ik_plan"].get("ik_position_error_m", np.inf)),
                        float(item["ik_plan"].get("ik_rotation_error_rad", np.inf)),
                        -float(scores[item["grasp_index"]]),
                    )

                qualified.sort(key=grasp_preference)
                selected_grasp = qualified[0]
                # Selected-first, strictly Pi-converged, swept-volume-safe
                # grasps for this base. A no-motion selection certifies this
                # list directly.
                candidate["qualified_grasps"] = list(qualified)
                candidate["selected_grasp"] = selected_grasp
                candidate["score"] = -float(candidate["movement_cost"])
                candidate["normalized_ik_residual"] = 0.0
                candidate["strict_ready"] = True
                candidate["selection_quality"] = (
                    "strict_pi_robustness_ranked"
                    if any(
                        item["robustness_evaluated_variants"] > 1
                        for item in qualified
                    )
                    else "strict_pi"
                )

            ranked = sorted(
                [
                    candidate
                    for candidate in candidates
                    if candidate["strict_ready"]
                ],
                key=lambda candidate: (
                    int(candidate["movement_cost_tier"]),
                    -int(candidate["converged_grasp_count"]),
                    -min(
                        self.config.robustness_preferred_converged_variants,
                        int(candidate["robustness_best_converged_variants"]),
                    ),
                    float(
                        candidate["selected_grasp"]["ik_plan"].get(
                            "joint_travel_l2_rad", np.inf
                        )
                    ),
                    float(candidate["movement_cost"]),
                    float(candidate["proxy_score"]),
                ),
            )
            return {
                "ranked": ranked,
                "nominal_converged_base_count": sum(
                    any(
                        plan["robustness_variant"] == "nominal"
                        and bool(plan["ik_converged"])
                        for plan in candidate["plans"]
                    )
                    for candidate in candidates
                ),
                "nominal_query_count": sum(
                    int(batch["candidate_count"])
                    for batch in pi_batches
                    if str(batch.get("stage", "")).startswith("nominal")
                ),
                "robustness_query_count": sum(
                    int(batch["candidate_count"])
                    for batch in pi_batches
                    if batch.get("stage") == "robustness"
                ),
                "nominal_converged_grasp_count": sum(
                    int(candidate.get("converged_grasp_count", 0))
                    for candidate in candidates
                ),
            }

        ranking = rank_candidates()
        substage_timings_s["post_ik_ranking"] = max(
            0.0, self.clock() - substage_started
        )
        evaluation_context = {
            "grasp_pool_count": grasp_pool_count,
            "collision_safe_count": len(grasp_indices),
            "shortlisted_base_indices": shortlisted_base_indices,
        }

        def search_accounts() -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
            """The current pose's stage-by-stage account and the per-tier one."""

            return (
                self._base_evaluation(
                    next(
                        (item for item in candidates if self._is_current_pose(item)),
                        None,
                    ),
                    **evaluation_context,
                ),
                self._tier_evaluation(candidates, shortlisted_base_indices),
            )

        current_pose_evaluation, tier_evaluation = search_accounts()
        # One entry per Pi search pass, this first one and every resumed one,
        # so a trace can say what each resume spent and what it certified.
        search_passes: list[dict[str, Any]] = [
            {
                "pass": "initial",
                "queries": len(pi_ik_query_log),
                "new_ready_bases": len(ranking["ranked"]),
                "tiers_queried": sorted(
                    {int(row["movement_cost_tier"]) for row in pi_ik_query_log}
                ),
                "budget_exhausted": budget_exhausted,
                "early_exit": early_exit_stage is not None,
                "terminal_batch_error": terminal_batch_error,
                "pi_ik_compute_elapsed_s": pi_ik_compute_elapsed_s,
            }
        ]
        if not ranking["ranked"]:
            if terminal_batch_error is not None:
                failure_stage = "parallel_pi_ik_execution"
                failure_message = (
                    "parallel Pi IK stopped after a later batch failed; "
                    "completed candidates had no strictly converged grasp"
                )
            elif budget_exhausted:
                failure_stage = "parallel_pi_virtual_certification_budget_exhausted"
                failure_message = (
                    "Pi IK compute budget ended before a strictly converged grasp "
                    "was found"
                )
            else:
                failure_stage = "parallel_pi_virtual_certification"
                failure_message = (
                    "no local base pose has a collision-safe, strictly "
                    "Pi-converged nominal grasp"
                )
            raise _CandidatePlanningError(
                f"{failure_message}: bases={len(candidates)} "
                f"queries={completed_record_count}/{scheduled_query_count}",
                {
                    "stage": failure_stage,
                    "evaluated_base_count": len(candidates),
                    "ik_query_count": completed_record_count,
                    "pi_ik_requested_count": scheduled_query_count,
                    "pi_eligible_pair_count": eligible_pair_count,
                    "pi_shortlisted_base_count": len(ranked_bases),
                    "collision_safe_grasp_count": len(grasp_indices),
                    "nominal_grasp_evaluated_count": ranking["nominal_query_count"],
                    "nominal_grasp_converged_count": (
                        ranking["nominal_converged_grasp_count"]
                    ),
                    "robustness_query_count": ranking["robustness_query_count"],
                    "bases_with_converged_grasp": 0,
                    "bases_with_nominal_converged_grasp": (
                        ranking["nominal_converged_base_count"]
                    ),
                    "robustness_enabled": self.config.robustness_enabled,
                    "robustness_variant_count": len(robustness_variants),
                    "robustness_preferred_converged_variants": (
                        self.config.robustness_preferred_converged_variants
                    ),
                    "pi_ik_compute_budget_s": self.config.pi_ik_compute_budget_s,
                    "pi_ik_budget_exhausted": budget_exhausted,
                    "pi_ik_early_exit": early_exit_stage is not None,
                    "pi_ik_scheduled_nominal_count": nominal_limit,
                    "terminal_batch_error": terminal_batch_error,
                    "pi_batches": pi_batches,
                    "pi_ik_compute_elapsed_s": pi_ik_compute_elapsed_s,
                    "substage_timings_s": dict(substage_timings_s),
                    "pi_ik_query_log": pi_ik_query_log,
                    "minimum_feasible_grasps": self.config.minimum_feasible_grasps,
                    "current_pose_evaluation": current_pose_evaluation,
                    "tier_evaluation": tier_evaluation,
                },
            )

        def search_can_continue() -> bool:
            """True while a resumed search still has something it may query: a
            shortlisted pair nobody queried, with no budget, query limit or
            failed batch having ended the Pi search."""

            return (
                not budget_exhausted
                and terminal_batch_error is None
                and completed_record_count < self.config.pi_ik_candidate_limit
                and pi_ik_compute_elapsed_s < self.config.pi_ik_compute_budget_s
                and any(
                    not record.get("ik_evaluated", False) for record in shortlisted
                )
            )

        def evaluation_from(
            ranking: Mapping[str, Any],
            current_pose_evaluation: dict[str, Any] | None,
            tier_evaluation: list[dict[str, Any]],
        ) -> dict[str, Any]:
            """The evaluation as the primitive reads it, over every pass so far."""

            ranked = ranking["ranked"]
            selected = ranked[0] if ranked else None
            return {
                "selected": selected,
                "candidates": candidates,
                "grasp_indices": grasp_indices,
                "ik_query_count": completed_record_count,
                "pi_ik_requested_count": scheduled_query_count,
                "pi_eligible_pair_count": eligible_pair_count,
                "pi_shortlisted_base_count": len(ranked_bases),
                "pi_batches": pi_batches,
                "pi_ik_query_log": pi_ik_query_log,
                "pi_ik_compute_elapsed_s": pi_ik_compute_elapsed_s,
                # The local work around the Pi window: expansion and ranking
                # before ``pi_started`` plus the qualification and final sort
                # after it. The primitive's virtual_candidate_evaluation bucket
                # reports these together with the Pi IK waves.
                "substage_timings_s": substage_timings_s,
                "minimum_feasible_grasps": self.config.minimum_feasible_grasps,
                "pi_ik_compute_budget_s": self.config.pi_ik_compute_budget_s,
                "pi_ik_budget_exhausted": budget_exhausted,
                "pi_ik_early_exit": early_exit_stage is not None,
                "pi_ik_early_exit_stage": early_exit_stage,
                "pi_ik_scheduled_nominal_count": nominal_limit,
                "pi_ik_terminal_batch_error": terminal_batch_error,
                "pi_bound_rejected_count": pi_bound_rejected_count,
                "collision_check": collision_check,
                "collision_safe_grasp_count": len(grasp_indices),
                "nominal_grasp_evaluated_count": ranking["nominal_query_count"],
                "nominal_grasp_converged_count": (
                    ranking["nominal_converged_grasp_count"]
                ),
                "robustness_query_count": ranking["robustness_query_count"],
                "strict_pi_candidate_count": len(ranked),
                "current_pose_evaluation": current_pose_evaluation,
                "selected_pose_evaluation": self._base_evaluation(
                    selected, **evaluation_context
                ),
                # What ``_base_evaluation`` needs to account for any other base
                # pose later, such as the one a refused motion hands its turn to.
                "base_evaluation_context": evaluation_context,
                "tier_evaluation": tier_evaluation,
                # Every strict-ready base pose in execution order, selected first.
                "ranked_candidates": list(ranked),
                # The runner-up base poses, in the order the primitive tries
                # them. Only ranked[0] is executed unless
                # motion_alternative_limit allows more, so when its motion is
                # refused the trace is the only place that can say whether a
                # cheaper or differently-directed candidate was sitting behind it.
                "ranked_alternatives": [
                    {
                        "rank": rank,
                        "candidate_index": int(candidate["candidate_index"]),
                        "forward_m": float(candidate["forward_m"]),
                        "left_m": float(candidate["left_m"]),
                        "yaw_deg": math.degrees(float(candidate["yaw_rad"])),
                        "movement_cost_tier": int(candidate["movement_cost_tier"]),
                        "movement_cost": float(candidate["movement_cost"]),
                        "converged_grasp_count": int(
                            candidate.get("converged_grasp_count", 0)
                        ),
                    }
                    for rank, candidate in enumerate(
                        ranked[:RANKED_ALTERNATIVES_LOGGED]
                    )
                ],
                "bases_with_nominal_converged_grasp": (
                    ranking["nominal_converged_base_count"]
                ),
                "robustness_enabled": self.config.robustness_enabled,
                "robustness_variant_count": len(robustness_variants),
                "robustness_preferred_converged_variants": (
                    self.config.robustness_preferred_converged_variants
                ),
                "camera_height_m": camera_height,
                "ground_down_camera_xyz": down,
                "navigation_from_camera": navigation_from_camera,
                "search_passes": [dict(item) for item in search_passes],
                # Resumes this search into the shortlisted pairs no pass has
                # queried yet; None once nothing is left to query.
                "continue_search": (
                    continue_search if search_can_continue() else None
                ),
            }

        def continue_search(attempted_base_indices: Collection[int]) -> dict[str, Any]:
            """Query the shortlisted pairs the earlier passes left unqueried.

            The best-tier early exit stops at the nearest tier holding a
            strict-ready base. When the motion to every such base is refused,
            the farther tiers are the only poses left, so this schedules every
            unqueried shortlisted pair in shortlist order and stops, as the
            first pass does, once the nearest resumed tier holding a ready base
            that is not in ``attempted_base_indices`` is fully evaluated.
            Robustness variants follow for the plans that converged here.

            The call's query limit and Pi IK compute budget are shared with the
            earlier passes: what they spent counts, the motions in between do
            not. Returns the evaluation rebuilt over every pass, whether or not
            this pass certified anything new; the caller decides what to do
            when it did not.
            """

            nonlocal early_exit_stage
            nonlocal pi_ik_compute_elapsed_s
            nonlocal pi_started
            attempted = frozenset(int(index) for index in attempted_base_indices)
            ready_before = {
                int(candidate["candidate_index"])
                for candidate in candidates
                if candidate.get("strict_ready", False)
            }
            plan_counts = [len(candidate["plans"]) for candidate in candidates]
            first_row = len(pi_ik_query_log)
            resumed_positions = [
                position
                for position, record in enumerate(shortlisted)
                if not record.get("ik_evaluated", False)
            ]
            resumed = [shortlisted[position] for position in resumed_positions]
            pass_started = self.clock()
            # run_pi_records measures its budget from ``pi_started``. Anchor it
            # so that only the Pi time the earlier passes spent lies behind the
            # clock, not the motions made since.
            pi_started = pass_started - pi_ik_compute_elapsed_s
            # The exit flag describes the latest nominal pass: a resumed pass
            # that reaches the end of its schedule has skipped nothing.
            early_exit_stage = None
            run_pi_records(
                resumed,
                "nominal_resumed",
                stop_when=(
                    (lambda: best_tier_settled(resumed, attempted))
                    if self.config.pi_ik_best_tier_early_exit
                    else None
                ),
                already_scheduled=len(
                    scheduled_shortlist_positions.intersection(resumed_positions)
                ),
            )
            scheduled_shortlist_positions.update(resumed_positions)
            resumed_early_exit = early_exit_stage is not None
            run_robustness(
                [
                    plan
                    for candidate, count in zip(candidates, plan_counts)
                    for plan in candidate["plans"][count:]
                    if plan["robustness_variant"] == "nominal"
                    and plan["ik_converged"]
                ]
            )
            pass_ended = self.clock()
            pi_ik_compute_elapsed_s = max(0.0, pass_ended - pi_started)
            resumed_ranking = rank_candidates()
            new_rows = pi_ik_query_log[first_row:]
            search_passes.append(
                {
                    "pass": "resumed",
                    "queries": len(new_rows),
                    "new_ready_bases": len(
                        {
                            int(candidate["candidate_index"])
                            for candidate in resumed_ranking["ranked"]
                        }
                        - ready_before
                    ),
                    "tiers_queried": sorted(
                        {int(row["movement_cost_tier"]) for row in new_rows}
                    ),
                    "budget_exhausted": budget_exhausted,
                    "early_exit": resumed_early_exit,
                    # Only this pass can have set it: an earlier batch error
                    # leaves nothing to resume.
                    "terminal_batch_error": terminal_batch_error,
                    "pi_ik_compute_elapsed_s": max(0.0, pass_ended - pass_started),
                    "post_ik_ranking_s": max(0.0, self.clock() - pass_ended),
                }
            )
            return evaluation_from(resumed_ranking, *search_accounts())

        return evaluation_from(ranking, current_pose_evaluation, tier_evaluation)

    def _base_evaluation(
        self,
        candidate: Mapping[str, Any] | None,
        *,
        grasp_pool_count: int,
        collision_safe_count: int,
        shortlisted_base_indices: set[int],
    ) -> dict[str, Any] | None:
        """Stage-by-stage account of one base pose in the certification search.

        The stages are counted in the order the search applies them, and
        ``rejected_at`` names the first one that left the pose without enough
        strictly converged grasps. The grasp collision gate runs once for the
        whole scene, so it treats every base pose alike; the stages that can
        tell one pose from another are the reach bound, the shortlist and Pi IK.
        """

        if candidate is None:
            return None
        nominal = [
            plan
            for plan in candidate.get("plans", [])
            if plan.get("robustness_variant") == "nominal"
        ]
        finite = [plan for plan in nominal if plan.get("ik_finite")]
        converged = [plan for plan in nominal if plan.get("ik_converged")]
        shortlisted = int(candidate["candidate_index"]) in shortlisted_base_indices
        eligible = int(candidate.get("eligible_grasps", 0))
        if eligible == 0:
            rejected_at = "reach_bound"
        elif not shortlisted:
            rejected_at = "shortlist"
        elif not nominal:
            rejected_at = "not_queried"
        elif not converged:
            rejected_at = "ik_not_converged"
        elif len(converged) < self.config.minimum_feasible_grasps:
            rejected_at = "too_few_converged_grasps"
        else:
            rejected_at = None

        best_ik = None
        pool = converged or finite or nominal
        if pool:
            best = min(
                pool,
                key=lambda plan: (
                    float(plan["ik_plan"].get("ik_position_error_m", math.inf)),
                    float(plan["ik_plan"].get("ik_rotation_error_rad", math.inf)),
                ),
            )
            plan = best["ik_plan"]
            best_ik = {
                "grasp_index": int(best["grasp_index"]),
                "ik_converged": bool(best.get("ik_converged")),
                "ik_position_error_m": _finite_or_none(plan.get("ik_position_error_m")),
                "ik_rotation_error_rad": _finite_or_none(plan.get("ik_rotation_error_rad")),
                "joint_travel_l2_rad": _finite_or_none(plan.get("joint_travel_l2_rad")),
                "tcp_radius_m": _finite_or_none(best.get("tcp_radius_m"), 4),
                "error": str(plan["error"])[:160] if plan.get("error") else None,
            }
        return {
            "forward_m": float(candidate["forward_m"]),
            "left_m": float(candidate["left_m"]),
            "yaw_deg": round(math.degrees(float(candidate["yaw_rad"])), 3),
            "movement_cost_tier": int(candidate["movement_cost_tier"]),
            "strict_ready": bool(candidate.get("strict_ready", False)),
            "rejected_at": rejected_at,
            "grasp_pool": int(grasp_pool_count),
            "collision_safe_grasps": int(collision_safe_count),
            "reach_bound_rejected_grasps": int(candidate.get("pi_bound_rejected_grasps", 0)),
            "reach_bound_m": float(PI_MAX_TCP_POSITION_NORM_M),
            "nearest_tcp_radius_m": _finite_or_none(candidate.get("nearest_tcp_radius_m"), 4),
            "preferred_tcp_radius_m": float(self.config.pi_ik_preferred_tcp_radius_m),
            "eligible_grasps": eligible,
            "shortlisted": shortlisted,
            "shortlisted_grasps": int(candidate.get("shortlisted_grasps", 0)),
            "ik_queried_grasps": len(nominal),
            "ik_finite_grasps": len(finite),
            "ik_converged_grasps": len(converged),
            "minimum_feasible_grasps": int(self.config.minimum_feasible_grasps),
            "best_ik": best_ik,
        }

    @staticmethod
    def _tier_evaluation(
        candidates: Sequence[Mapping[str, Any]],
        shortlisted_base_indices: set[int],
    ) -> list[dict[str, Any]]:
        """Per movement-cost tier: how far its base poses got in the search."""

        by_tier: dict[int, dict[str, Any]] = {}
        for candidate in candidates:
            tier = int(candidate["movement_cost_tier"])
            row = by_tier.setdefault(
                tier,
                {
                    "tier": tier,
                    "bases": 0,
                    "bases_all_reach_bound_rejected": 0,
                    "bases_shortlisted": 0,
                    "grasps_queried": 0,
                    "grasps_converged": 0,
                    "bases_ready": 0,
                },
            )
            nominal = [
                plan
                for plan in candidate.get("plans", [])
                if plan.get("robustness_variant") == "nominal"
            ]
            row["bases"] += 1
            row["bases_all_reach_bound_rejected"] += int(
                int(candidate.get("eligible_grasps", 0)) == 0
            )
            row["bases_shortlisted"] += int(
                int(candidate["candidate_index"]) in shortlisted_base_indices
            )
            row["grasps_queried"] += len(nominal)
            row["grasps_converged"] += sum(bool(plan.get("ik_converged")) for plan in nominal)
            row["bases_ready"] += int(bool(candidate.get("strict_ready", False)))
        return [by_tier[tier] for tier in sorted(by_tier)][:TIER_EVALUATION_LOGGED]

    @staticmethod
    def _is_current_pose(candidate: Mapping[str, Any]) -> bool:
        """True only for the exact identity lattice entry, never a near one."""

        return (
            float(candidate["forward_m"]) == 0.0
            and float(candidate["left_m"]) == 0.0
            and float(candidate["yaw_rad"]) == 0.0
        )

    def _certify_from_virtual_evaluation(
        self,
        object_name: str,
        arm: int | str,
        selected: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Install the current-pose evaluation goalset as the grasp certificate.

        Valid only for the identity candidate: its ``arm_targets`` are the
        observation's grasps in the current arm frame, every entry passed the
        open-gripper swept-volume gate on this scene before Pi IK ran, and
        ``qualified_grasps`` holds only strictly converged nominal plans in the
        same preference order used for selection.
        """

        if not self._is_current_pose(selected):
            raise RuntimeError("virtual certificate requires the current pose")
        qualified = selected.get("qualified_grasps") or []
        if not qualified:
            raise RuntimeError("current pose has no strictly converged grasp")
        goals = np.stack(
            [
                np.asarray(item["arm_target"], dtype=np.float64)
                for item in qualified[:MAX_CERTIFIED_GOALSET]
            ],
            axis=0,
        )
        selected_grasp = selected.get("selected_grasp")
        if not isinstance(selected_grasp, Mapping) or not np.allclose(
            goals[0], np.asarray(selected_grasp["arm_target"], dtype=np.float64)
        ):
            raise RuntimeError("virtual certificate goalset is not selected-first")
        return self.manipulation.install_prepared_grasp_certificate(
            object_name,
            arm,
            goals,
            source="virtual_evaluation",
        )

    def _live_pose_world(self) -> list[float]:
        """The base's current world-frame ``[x, y, yaw]`` from a fresh frame."""

        frame = self.env.navigation_frame(
            max_age_s=self.env.controller.config.pose_max_age_s
        )
        return list(self.env.controller._pose(frame))

    def _exceeds_motion_tolerance(
        self, distance_m: float, yaw_rad: float
    ) -> bool:
        """True when a remaining planar or heading error is worth a motion."""

        return (
            distance_m > self.config.position_tolerance_m
            or abs(yaw_rad) > self.config.yaw_tolerance_rad
        )

    @staticmethod
    def _base_pose_record(candidate: Mapping[str, Any]) -> dict[str, float | int]:
        """The relative base pose and movement cost of one candidate."""

        return {
            "forward_m": float(candidate["forward_m"]),
            "left_m": float(candidate["left_m"]),
            "yaw_rad": float(candidate["yaw_rad"]),
            "movement_cost": float(candidate["movement_cost"]),
            "movement_cost_tier": int(candidate["movement_cost_tier"]),
        }

    def _single_stage_motion_target(
        self,
        *,
        forward_m: float,
        left_m: float,
        yaw_rad: float,
        start_pose_world: Sequence[float] | None = None,
    ) -> dict[str, Any]:
        """Resolve one relative base candidate to an absolute world-frame goal.

        The offsets are relative to the pose the candidates were evaluated
        from. Pass that pose as ``start_pose_world`` when the base may have
        moved since; otherwise the live pose is read and used.
        """

        if start_pose_world is None:
            start_pose = self._live_pose_world()
        else:
            start_pose = [float(value) for value in start_pose_world]
        start_x, start_y, start_yaw = start_pose
        target_yaw = math.atan2(
            math.sin(start_yaw + yaw_rad), math.cos(start_yaw + yaw_rad)
        )
        start_cosine, start_sine = math.cos(start_yaw), math.sin(start_yaw)
        final_target = [
            start_x + start_cosine * forward_m - start_sine * left_m,
            start_y + start_sine * forward_m + start_cosine * left_m,
            target_yaw,
        ]
        return {
            "start_pose_world": list(start_pose),
            "target_pose_world": list(final_target),
            "target_relative_forward_m": float(forward_m),
            "target_relative_left_m": float(left_m),
            "target_relative_yaw_rad": float(yaw_rad),
        }

    def _execute_single_stage_motion(
        self, motion_target: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Track one absolute SE(2) goal without intermediate motion phases."""

        motion = dict(
            self.env.controller.move_planar_relative(
                0.0,
                0.0,
                0.0,
                max_linear_mps=self.config.max_linear_mps,
                max_lateral_mps=self.config.max_lateral_mps,
                max_yaw_rad_s=self.config.max_yaw_rad_s,
                position_tolerance_m=self.config.position_tolerance_m,
                yaw_tolerance_rad=self.config.yaw_tolerance_rad,
                timeout_s=self.config.motion_timeout_s,
                allow_reverse=True,
                target_pose_world=list(motion_target["target_pose_world"]),
            )
        )
        metrics = dict(motion.get("metrics") or {})
        metrics.update(
            {
                "holonomic": True,
                "execution_mode": "single_absolute_se2",
                **dict(motion_target),
                "position_tolerance_m": self.config.position_tolerance_m,
                "yaw_tolerance_rad": self.config.yaw_tolerance_rad,
                "reverse_correction_enabled": True,
                "rear_clearance_checked": False,
            }
        )
        motion["primitive"] = "move_planar_single_stage"
        motion["metrics"] = metrics
        return motion

    @staticmethod
    def _public_candidate(candidate: Mapping[str, Any]) -> dict[str, Any]:
        selected_grasp = candidate.get("selected_grasp")
        selected_summary = None
        if isinstance(selected_grasp, Mapping):
            plan = selected_grasp.get("ik_plan", {})
            selected_summary = {
                "grasp_index": int(selected_grasp["grasp_index"]),
                "arm_target": np.asarray(selected_grasp["arm_target"]).copy(),
                "backend_score": None,
                "ik_plan": dict(plan),
                "robustness_converged_variants": int(
                    selected_grasp.get("robustness_converged_variants", 1)
                ),
                "robustness_evaluated_variants": int(
                    selected_grasp.get("robustness_evaluated_variants", 1)
                ),
            }
        return {
            "candidate_index": int(candidate["candidate_index"]),
            "forward_m": float(candidate["forward_m"]),
            "left_m": float(candidate["left_m"]),
            "yaw_rad": float(candidate["yaw_rad"]),
            "movement_cost": float(candidate["movement_cost"]),
            "movement_cost_tier": int(candidate["movement_cost_tier"]),
            "proxy_score": float(candidate["proxy_score"]),
            "score": float(candidate.get("score", -math.inf)),
            "feasible_grasp_count": int(candidate.get("feasible_grasp_count", 0)),
            "nominal_grasp_evaluated_count": int(
                candidate.get("nominal_grasp_evaluated_count", 0)
            ),
            "converged_grasp_count": int(
                candidate.get("converged_grasp_count", 0)
            ),
            "finite_grasp_count": int(candidate.get("finite_grasp_count", 0)),
            "robustness_best_converged_variants": int(
                candidate.get("robustness_best_converged_variants", 0)
            ),
            "strict_ready": bool(candidate.get("strict_ready", False)),
            "selection_quality": str(
                candidate.get("selection_quality", "unavailable")
            ),
            "normalized_ik_residual": candidate.get("normalized_ik_residual"),
            "selected_grasp": selected_summary,
        }

    def _publish_debug(self, debug: dict[str, Any]) -> None:
        self.last_debug = debug
        if self.debug_callback is not None:
            self.debug_callback(debug)

    def prepare_for_manipulation(
        self, object_name: str, arm: int | str
    ) -> dict[str, Any]:
        """Move the base to a grasp-promising pose without returning a grasp."""

        if not isinstance(object_name, str) or not object_name.strip():
            raise ValueError("object_name must be a non-empty string")
        object_name = object_name.strip()
        start_time = self.clock()
        # Per-stage wall clock. The dict is installed in the result by
        # reference, so a run that fails part-way through still reports the
        # stages that already completed. ``pi_ik_compute_elapsed_s`` remains the
        # finer breakdown inside ``virtual_candidate_evaluation``.
        stage_timings_s: dict[str, float] = {}
        result: dict[str, Any] = {
            "success": False,
            "status": "failed",
            "primitive": "prepare_for_manipulation",
            "reason": "unknown",
            "object_name": object_name,
            "arm": ManipulationController._arm_name(arm),
            "stage_timings_s": stage_timings_s,
        }
        self.manipulation._prepared_grasp_certificates.pop(result["arm"], None)
        try:
            stopped = self.env.controller.stop()
            if not stopped.get("success", False):
                raise RuntimeError(f"initial stop failed: {stopped.get('reason')}")

            def publish_depth_validation(progress: dict[str, Any]) -> None:
                self._publish_debug(
                    {
                        **progress,
                        "object_name": object_name,
                        "arm": result["arm"],
                    }
                )

            stage_started = self.clock()
            generated = self.manipulation.generate_grasp_candidates(
                object_name,
                arm,
                depth_retry_count=self.config.sam3_depth_retry_count,
                progress_callback=publish_depth_validation,
            )
            stage_timings_s["grasp_candidate_generation"] = max(
                0.0, self.clock() - stage_started
            )
            # SAM3/observation vs grasp backend vs candidate filtering, for the
            # single largest stage of the docked case.
            result["generation_split_s"] = dict(generated.get("timings_s") or {})
            target_distance_m = float(
                np.median(
                    np.linalg.norm(
                        np.asarray(generated["points_camera"], dtype=np.float64),
                        axis=1,
                    )
                )
            )
            if target_distance_m > self.config.maximum_initial_target_distance_m:
                result.update(
                    {
                        "reason": "target_too_far_for_local_manipulation",
                        "target_distance_m": target_distance_m,
                        "maximum_distance_m": (
                            self.config.maximum_initial_target_distance_m
                        ),
                        "recovery": {
                            "action": "dock_to_visible_object",
                            "message": (
                                "prepare_for_manipulation only performs a small "
                                "local search. Move closer to the target, preferably "
                                "with dock_to_visible_object(object_name), then retry."
                            ),
                        },
                    }
                )
                self._publish_debug(
                    {
                        "phase": "target_too_far",
                        "object_name": object_name,
                        "arm": result["arm"],
                        "target_distance_m": target_distance_m,
                        "maximum_distance_m": (
                            self.config.maximum_initial_target_distance_m
                        ),
                        "rgb": np.asarray(generated["rgb"]).copy(),
                        "mask": np.asarray(generated["mask"]).copy(),
                    }
                )
                return result
            selected_depth_attempt = max(
                generated["depth_validation_attempts"],
                key=lambda item: int(item["valid_depth_points"]),
            )
            publish_depth_validation(
                {
                    "phase": "batched_ik_evaluation",
                    "rgb": np.asarray(generated["rgb"]).copy(),
                    "depth_m": np.asarray(generated["depth_m"]).copy(),
                    "intrinsics": np.asarray(generated["intrinsics"]).copy(),
                    "mask": np.asarray(generated["mask"]).copy(),
                    "sam_score": float(generated["sam_score"]),
                    "attempt_index": int(selected_depth_attempt["attempt_index"]),
                    "maximum_attempts": self.config.sam3_depth_retry_count + 1,
                    "mask_pixels": int(selected_depth_attempt["mask_pixels"]),
                    "valid_depth_points": int(
                        selected_depth_attempt["valid_depth_points"]
                    ),
                    "minimum_valid_depth_points": int(
                        selected_depth_attempt["minimum_valid_depth_points"]
                    ),
                    "accepted": True,
                    "segmentation_error": None,
                    "generated_grasp_count": int(
                        np.asarray(generated["candidate_indices"]).size
                    ),
                    "grasp_backend": str(generated["backend_name"]),
                }
            )
            stage_started = self.clock()
            frame = self.env.navigation_frame(
                max_age_s=self.env.controller.config.pose_max_age_s
            )
            stage_timings_s["navigation_frame_fetch"] = max(
                0.0, self.clock() - stage_started
            )
            stage_started = self.clock()
            evaluation = self._evaluate_candidates(generated, frame)
            stage_timings_s["virtual_candidate_evaluation"] = max(
                0.0, self.clock() - stage_started
            )
            selected = evaluation["selected"]
            debug = {
                "phase": "virtual_candidate_evaluation",
                "object_name": object_name,
                "arm": result["arm"],
                "stage_timings_s": stage_timings_s,
                "rgb": np.asarray(generated["rgb"]).copy(),
                "depth_m": np.asarray(generated["depth_m"]).copy(),
                "intrinsics": np.asarray(generated["intrinsics"]).copy(),
                "mask": np.asarray(generated["mask"]).copy(),
                "sam_score": float(generated["sam_score"]),
                "arm_from_camera": np.asarray(
                    generated["arm_from_camera"]
                ).copy(),
                "navigation_from_camera": np.asarray(
                    evaluation["navigation_from_camera"]
                ).copy(),
                "ground_camera_height_m": float(evaluation["camera_height_m"]),
                "ground_down_camera_xyz": np.asarray(
                    evaluation["ground_down_camera_xyz"]
                ).copy(),
                "candidates": [
                    self._public_candidate(candidate)
                    for candidate in evaluation["candidates"]
                ],
                "selected_candidate_index": int(selected["candidate_index"]),
                "ik_query_count": int(evaluation["ik_query_count"]),
                "pi_ik_requested_count": int(
                    evaluation["pi_ik_requested_count"]
                ),
                "pi_eligible_pair_count": int(
                    evaluation["pi_eligible_pair_count"]
                ),
                "pi_shortlisted_base_count": int(
                    evaluation["pi_shortlisted_base_count"]
                ),
                "pi_batches": list(evaluation["pi_batches"]),
                "pi_ik_query_log": list(evaluation["pi_ik_query_log"]),
                "pi_ik_compute_elapsed_s": float(
                    evaluation["pi_ik_compute_elapsed_s"]
                ),
                "pi_ik_compute_budget_s": float(
                    evaluation["pi_ik_compute_budget_s"]
                ),
                "pi_ik_budget_exhausted": bool(
                    evaluation["pi_ik_budget_exhausted"]
                ),
                "pi_ik_early_exit": bool(evaluation["pi_ik_early_exit"]),
                "pi_ik_scheduled_nominal_count": int(
                    evaluation["pi_ik_scheduled_nominal_count"]
                ),
                "pi_ik_terminal_batch_error": evaluation[
                    "pi_ik_terminal_batch_error"
                ],
                "pi_bound_rejected_count": int(
                    evaluation["pi_bound_rejected_count"]
                ),
                "collision_safe_grasp_count": int(
                    evaluation["collision_safe_grasp_count"]
                ),
                "nominal_grasp_evaluated_count": int(
                    evaluation["nominal_grasp_evaluated_count"]
                ),
                "nominal_grasp_converged_count": int(
                    evaluation["nominal_grasp_converged_count"]
                ),
                "robustness_query_count": int(
                    evaluation["robustness_query_count"]
                ),
                "strict_pi_candidate_count": int(
                    evaluation["strict_pi_candidate_count"]
                ),
                "bases_with_nominal_converged_grasp": int(
                    evaluation["bases_with_nominal_converged_grasp"]
                ),
                "robustness_enabled": bool(evaluation["robustness_enabled"]),
                "robustness_variant_count": int(
                    evaluation["robustness_variant_count"]
                ),
                "robustness_preferred_converged_variants": int(
                    evaluation["robustness_preferred_converged_variants"]
                ),
                "collision_check": dict(evaluation["collision_check"]),
                "sam3_depth_attempts": list(
                    generated["depth_validation_attempts"]
                ),
                "config": self.config,
            }
            self._publish_debug(debug)
            selected_base_pose = self._base_pose_record(selected)
            # The candidates are relative to the pose the search observed
            # from, and the base has stood still since, so the live pose read
            # here is the frame every candidate's absolute target is resolved
            # in. It is read once: a refused motion can leave the base
            # displaced, and a later candidate's target must not inherit that.
            observation_pose_world = self._live_pose_world()
            motion_target = self._single_stage_motion_target(
                forward_m=selected_base_pose["forward_m"],
                left_m=selected_base_pose["left_m"],
                yaw_rad=selected_base_pose["yaw_rad"],
                start_pose_world=observation_pose_world,
            )
            # These bounded records intentionally live in the primitive result so
            # both successful and interrupted executions retain the exact target
            # in trace.json. The LLM failure summarizer does not forward them.
            result["selected_base_pose"] = selected_base_pose
            result["motion_target"] = motion_target
            # Same reason, one step further: only this pose is ever executed,
            # so when its motion is refused the trace has to say what the
            # search had ranked behind it. The evaluation aggregates below are
            # assembled on the success path alone and never survive a refusal.
            result["ranked_alternatives"] = list(
                evaluation.get("ranked_alternatives") or []
            )
            # Why the pose the robot stands at did or did not certify, stage by
            # stage, beside the pose chosen instead and the per-tier picture.
            result["current_pose_evaluation"] = evaluation.get("current_pose_evaluation")
            result["selected_pose_evaluation"] = evaluation.get("selected_pose_evaluation")
            result["tier_evaluation"] = evaluation.get("tier_evaluation")
            # The per-query order is what the IK-budget curve is replayed from
            # (tools/prepare_ik_timing_report.py). A search that certified a
            # base but whose motion is then refused is still a measured search,
            # so the log and its batches are recorded before the motion too.
            result["pi_ik_query_log"] = list(evaluation.get("pi_ik_query_log") or [])
            result["pi_batches"] = list(evaluation.get("pi_batches") or [])
            result["minimum_feasible_grasps"] = int(
                evaluation.get("minimum_feasible_grasps", self.config.minimum_feasible_grasps)
            )
            # The base poses in the order the search ranked them, selected
            # first. A refused motion hands its turn to the next one while
            # motion_alternative_limit allows; 0 keeps the single attempt.
            ordered_candidates = list(
                evaluation.get("ranked_candidates") or [selected]
            )
            attempt_budget = 1 + self.config.motion_alternative_limit
            # With continue_search_after_refused_motions, a refusal of every
            # pose ranked so far resumes the search into the tiers its early
            # exit never queried, and the attempts go on with the poses
            # certified there, all within the same attempt budget.
            resume_search = (
                self.config.continue_search_after_refused_motions
                and self.config.motion_alternative_limit > 0
            )
            # The attempt records are installed by reference before the first
            # attempt, like stage_timings_s, so a run that raises part-way
            # through the attempts (a stale frame after a refusal, say) still
            # reports the ones that completed; ``motion`` and the flags below
            # follow each attempt as it ends.
            motion_attempts: list[dict[str, Any]] = []
            result["motion_attempts"] = motion_attempts
            search_resumes: list[dict[str, Any]] = []

            def record_search_totals() -> None:
                # The success path's aggregates never reach a failed call, and
                # a call that resumed may have spent most of its Pi IK budget
                # before failing, so the Pi time and the passes are recorded as
                # they change. ``pi_ik_early_exit`` describes the latest pass;
                # ``search_passes`` keeps each pass's own.
                result["pi_ik_compute_elapsed_s"] = float(
                    evaluation["pi_ik_compute_elapsed_s"]
                )
                result["pi_ik_budget_exhausted"] = bool(
                    evaluation["pi_ik_budget_exhausted"]
                )
                result["pi_ik_terminal_batch_error"] = evaluation[
                    "pi_ik_terminal_batch_error"
                ]
                result["search_passes"] = [
                    dict(item) for item in evaluation.get("search_passes") or []
                ]

            if resume_search:
                result["search_resumes"] = search_resumes
                record_search_totals()
            result["motion_executed"] = False
            result["executed_candidate_index"] = None
            debug["selected_base_pose"] = selected_base_pose
            debug["motion_target"] = motion_target
            debug["motion_attempts"] = motion_attempts
            self._publish_debug(debug)
            motion_executed = False
            executed: Mapping[str, Any] | None = None
            motion: dict[str, Any] = {}
            operator_stopped = False
            attempted_indices: set[int] = set()

            def next_unattempted() -> Mapping[str, Any] | None:
                return next(
                    (
                        item
                        for item in ordered_candidates
                        if int(item["candidate_index"]) not in attempted_indices
                    ),
                    None,
                )

            # Resumed searches are timed in their own bucket, so ``motion``
            # keeps covering the motions alone.
            resumed_search_s = 0.0
            stage_started = self.clock()
            attempt = 0
            while attempt < attempt_budget:
                candidate = next_unattempted()
                if candidate is None and resume_search:
                    # Every pose ranked so far was attempted and refused (a
                    # success, an operator stop or a failed stop has already
                    # left the loop), and attempts remain.
                    continue_search = evaluation.get("continue_search")
                    if continue_search is None:
                        break
                    resume_started = self.clock()
                    try:
                        evaluation = continue_search(frozenset(attempted_indices))
                    except Exception as exc:
                        # The call fails here, but its motions already ran:
                        # keep their timing and the failed resume beside the
                        # attempts before the exception reports the reason.
                        stage_timings_s["motion"] = max(
                            0.0, resume_started - stage_started - resumed_search_s
                        )
                        search_resumes.append(
                            {
                                "after_attempts": len(motion_attempts),
                                "error": f"{type(exc).__name__}: {str(exc)[:200]}",
                            }
                        )
                        raise
                    finally:
                        resumed_search_s += max(0.0, self.clock() - resume_started)
                        stage_timings_s["resumed_candidate_evaluation"] = (
                            resumed_search_s
                        )
                    search_pass = dict(
                        (evaluation.get("search_passes") or [{}])[-1]
                    )
                    search_resumes.append(
                        {
                            "after_attempts": len(motion_attempts),
                            "queries": int(search_pass.get("queries", 0)),
                            "new_ready_bases": int(
                                search_pass.get("new_ready_bases", 0)
                            ),
                            "tiers_queried": list(
                                search_pass.get("tiers_queried") or []
                            ),
                            "budget_exhausted": bool(
                                search_pass.get("budget_exhausted", False)
                            ),
                            "early_exit": bool(search_pass.get("early_exit", False)),
                            # A resume a Pi batch error cut short would
                            # otherwise read as one that found nothing.
                            "terminal_batch_error": search_pass.get(
                                "terminal_batch_error"
                            ),
                            "pi_ik_compute_elapsed_s": float(
                                search_pass.get("pi_ik_compute_elapsed_s", 0.0)
                            ),
                        }
                    )
                    record_search_totals()
                    # The recorded search follows the latest evaluation, so a
                    # call that still fails carries every query it measured.
                    # The candidates stay relative to observation_pose_world.
                    result["ranked_alternatives"] = list(
                        evaluation.get("ranked_alternatives") or []
                    )
                    result["current_pose_evaluation"] = evaluation.get(
                        "current_pose_evaluation"
                    )
                    result["tier_evaluation"] = evaluation.get("tier_evaluation")
                    result["pi_ik_query_log"] = list(
                        evaluation.get("pi_ik_query_log") or []
                    )
                    result["pi_batches"] = list(evaluation.get("pi_batches") or [])
                    debug.update(
                        {
                            "candidates": [
                                self._public_candidate(item)
                                for item in evaluation["candidates"]
                            ],
                            "ik_query_count": int(evaluation["ik_query_count"]),
                            "pi_ik_requested_count": int(
                                evaluation["pi_ik_requested_count"]
                            ),
                            "pi_batches": list(evaluation["pi_batches"]),
                            "pi_ik_query_log": list(evaluation["pi_ik_query_log"]),
                            "pi_ik_compute_elapsed_s": float(
                                evaluation["pi_ik_compute_elapsed_s"]
                            ),
                            "pi_ik_budget_exhausted": bool(
                                evaluation["pi_ik_budget_exhausted"]
                            ),
                            "pi_ik_early_exit": bool(evaluation["pi_ik_early_exit"]),
                            "pi_ik_terminal_batch_error": evaluation[
                                "pi_ik_terminal_batch_error"
                            ],
                            "pi_bound_rejected_count": int(
                                evaluation["pi_bound_rejected_count"]
                            ),
                            "nominal_grasp_evaluated_count": int(
                                evaluation["nominal_grasp_evaluated_count"]
                            ),
                            "nominal_grasp_converged_count": int(
                                evaluation["nominal_grasp_converged_count"]
                            ),
                            "robustness_query_count": int(
                                evaluation["robustness_query_count"]
                            ),
                            "strict_pi_candidate_count": int(
                                evaluation["strict_pi_candidate_count"]
                            ),
                            "bases_with_nominal_converged_grasp": int(
                                evaluation["bases_with_nominal_converged_grasp"]
                            ),
                            "search_resumes": search_resumes,
                        }
                    )
                    ordered_candidates = list(
                        evaluation.get("ranked_candidates") or []
                    )
                    # A resume that certified no pose not yet attempted ends
                    # the attempts, and the call fails as an exhausted search.
                    candidate = next_unattempted()
                if candidate is None:
                    break
                if attempt > 0:
                    motion_target = self._single_stage_motion_target(
                        forward_m=float(candidate["forward_m"]),
                        left_m=float(candidate["left_m"]),
                        yaw_rad=float(candidate["yaw_rad"]),
                        start_pose_world=observation_pose_world,
                    )
                # Until a motion has been commanded the base still stands at
                # the observation pose and the candidate offsets are the whole
                # remaining error. After one, the live pose decides: a refused
                # motion may have moved the base part of the way, so even the
                # observation pose itself can be worth driving back to.
                if motion_executed:
                    live_x, live_y, live_yaw = self._live_pose_world()
                    target_x, target_y, target_yaw = motion_target[
                        "target_pose_world"
                    ]
                    needs_motion = self._exceeds_motion_tolerance(
                        math.hypot(target_x - live_x, target_y - live_y),
                        math.atan2(
                            math.sin(target_yaw - live_yaw),
                            math.cos(target_yaw - live_yaw),
                        ),
                    )
                else:
                    needs_motion = self._exceeds_motion_tolerance(
                        math.hypot(
                            float(candidate["forward_m"]),
                            float(candidate["left_m"]),
                        ),
                        float(candidate["yaw_rad"]),
                    )
                if needs_motion:
                    motion_executed = True
                    attempted = self._execute_single_stage_motion(motion_target)
                else:
                    attempted = {
                        "success": True,
                        "reason": "selected_current_pose",
                        "primitive": "move_planar_single_stage",
                        "metrics": {
                            "holonomic": True,
                            "execution_mode": "single_absolute_se2",
                            **motion_target,
                        },
                    }
                if attempt > 0:
                    # This attempt's outcome replaces the previous refusal's
                    # in ``motion``; the first refusal keeps its full
                    # diagnostics beside the last attempt's. A run whose only
                    # attempt is refused records it once, in ``motion`` alone.
                    result.setdefault("first_refused_motion", motion)
                motion = attempted
                # Recorded explicitly: when no motion was needed the
                # actual-pose certification below re-observes from the same
                # pose that was already strictly certified as the virtual
                # nominal candidate.
                result["motion_executed"] = motion_executed
                result["motion"] = motion
                succeeded = bool(motion.get("success", False))
                reason = str(motion.get("reason", "unknown"))
                motion_attempts.append(
                    {
                        "attempt": attempt,
                        "candidate_index": int(candidate["candidate_index"]),
                        "forward_m": float(candidate["forward_m"]),
                        "left_m": float(candidate["left_m"]),
                        "yaw_deg": math.degrees(float(candidate["yaw_rad"])),
                        "movement_cost_tier": int(candidate["movement_cost_tier"]),
                        "movement_cost": float(candidate["movement_cost"]),
                        "needed_motion": bool(needs_motion),
                        "success": succeeded,
                        "reason": reason,
                    }
                )
                attempted_indices.add(int(candidate["candidate_index"]))
                attempt += 1
                if succeeded:
                    executed = candidate
                    result["executed_candidate_index"] = int(
                        candidate["candidate_index"]
                    )
                    break
                # An operator stop ends the run, not just this attempt: the
                # remaining candidates are not commanded and the result
                # carries this one refusal, as a single attempt would.
                if "operator_stop_requested" in reason:
                    operator_stopped = True
                    break
                # move_planar_relative stops the base on every exit path and
                # folds a failed stop into its reason; then the base state is
                # unknown and no further motion may be commanded.
                if "final_stop_failed" in reason:
                    break
            stage_timings_s["motion"] = max(
                0.0, self.clock() - stage_started - resumed_search_s
            )
            if executed is None:
                if self.config.motion_alternative_limit == 0 or operator_stopped:
                    raise RuntimeError(
                        "single-stage SE(2) motion failed: "
                        f"{motion.get('reason', 'unknown')}"
                    )
                result["motion_exhausted"] = True
                refusals: dict[str, int] = {}
                for entry in motion_attempts:
                    refusals[entry["reason"]] = refusals.get(entry["reason"], 0) + 1
                raise RuntimeError(
                    "no certified base reachable: "
                    f"{len(motion_attempts)} motions refused ("
                    + ", ".join(
                        f"{reason}x{count}" for reason, count in refusals.items()
                    )
                    + ")"
                )
            if executed is not selected:
                # The pose the robot now stands at is not the one the search
                # chose; the records describe the executed one from here on,
                # and its stage-by-stage account joins the selected pose's.
                selected_base_pose = self._base_pose_record(executed)
                result["selected_base_pose"] = selected_base_pose
                result["motion_target"] = motion_target
                result["executed_pose_evaluation"] = self._base_evaluation(
                    executed, **evaluation["base_evaluation_context"]
                )
                debug["selected_base_pose"] = selected_base_pose
                debug["motion_target"] = motion_target
                debug["executed_candidate_index"] = result[
                    "executed_candidate_index"
                ]
            selection_quality = str(executed["selection_quality"])
            stopped = self.env.controller.stop()
            if not stopped.get("success", False):
                raise RuntimeError(
                    f"post-motion planning stop failed: {stopped.get('reason')}"
                )
            stage_started = self.clock()
            certification: dict[str, Any] | None = None
            if (
                not motion_executed
                and self.config.reuse_virtual_certificate_without_motion
                and self._is_current_pose(executed)
            ):
                # The base did not move, so the evaluation's observation is the
                # actual-pose observation: its collision-safe (swept-volume
                # checked on this scene) and strictly Pi-converged nominal
                # grasps are exactly what a fresh certification would recompute.
                try:
                    certification = self._certify_from_virtual_evaluation(
                        object_name, arm, executed
                    )
                except Exception as exc:
                    certification = {
                        "success": False,
                        "reason": f"{type(exc).__name__}:{str(exc).strip()}",
                        "arm": result["arm"],
                        "source": "virtual_evaluation",
                        "curobo_called": False,
                    }
                debug["virtual_certificate_attempt"] = dict(certification)
            reused_virtual_certificate = True
            if certification is None or not certification.get("success", False):
                reused_virtual_certificate = False
                try:
                    certification = self.manipulation.certify_pi_grasp_readiness(
                        object_name,
                        arm,
                        depth_retry_count=self.config.sam3_depth_retry_count,
                    )
                except Exception as exc:
                    certification = {
                        "success": False,
                        "reason": f"{type(exc).__name__}:{str(exc).strip()}",
                        "arm": result["arm"],
                        "source": "fresh_sample",
                        "curobo_called": False,
                    }
            stage_timings_s["actual_pose_certification"] = max(
                0.0, self.clock() - stage_started
            )
            # Separates a reused virtual certificate from a second full
            # SAM3 + grasp + Pi IK round inside the same timing bucket.
            result["reused_virtual_certificate"] = reused_virtual_certificate
            if not certification.get("success", False):
                result.update(
                    {
                        "reason": "actual_pose_grasp_certification_failed",
                        "certification": certification,
                        "recovery": {
                            "action": "retry_prepare_for_manipulation",
                            "message": (
                                "The virtual pose had strict Pi IK, but no "
                                "collision-safe grasp converged in the fresh "
                                "actual-pose Pi check. Re-observe and retry "
                                "preparation; do not continue to goto_grasp_pose."
                            ),
                        },
                    }
                )
                debug["phase"] = "failed"
                debug["failed_after_phase"] = "actual_pose_strict_pi_certification"
                debug["failure_reason"] = result["reason"]
                debug["certification"] = certification
                debug["motion"] = motion
                self._publish_debug(debug)
                return result
            virtual_feasible_grasp_count = int(executed["feasible_grasp_count"])
            ready_reason = "grasp_execution_ready"
            debug["phase"] = "grasp_execution_ready"
            debug["selection_quality"] = selection_quality
            debug["motion"] = motion
            debug["certification"] = certification
            debug["requires_fresh_grasp_sampling"] = False
            debug["certified_goalset_available"] = True
            self._publish_debug(debug)
            result.update(
                {
                    "success": True,
                    "status": "succeeded",
                    "reason": ready_reason,
                    "selection_quality": selection_quality,
                    "ik_query_count": evaluation["ik_query_count"],
                    "pi_ik_requested_count": evaluation[
                        "pi_ik_requested_count"
                    ],
                    "pi_eligible_pair_count": evaluation[
                        "pi_eligible_pair_count"
                    ],
                    "pi_shortlisted_base_count": evaluation[
                        "pi_shortlisted_base_count"
                    ],
                    "pi_batches": list(evaluation["pi_batches"]),
                    "pi_ik_query_log": list(evaluation["pi_ik_query_log"]),
                    "pi_ik_compute_elapsed_s": evaluation[
                        "pi_ik_compute_elapsed_s"
                    ],
                    "evaluation_substage_timings_s": dict(
                        evaluation["substage_timings_s"]
                    ),
                    "minimum_feasible_grasps": int(
                        evaluation["minimum_feasible_grasps"]
                    ),
                    "pi_ik_compute_budget_s": evaluation[
                        "pi_ik_compute_budget_s"
                    ],
                    "pi_ik_budget_exhausted": evaluation[
                        "pi_ik_budget_exhausted"
                    ],
                    "pi_ik_early_exit": bool(evaluation["pi_ik_early_exit"]),
                    "pi_ik_scheduled_nominal_count": int(
                        evaluation["pi_ik_scheduled_nominal_count"]
                    ),
                    "pi_ik_terminal_batch_error": evaluation[
                        "pi_ik_terminal_batch_error"
                    ],
                    "pi_bound_rejected_count": evaluation[
                        "pi_bound_rejected_count"
                    ],
                    "collision_safe_grasp_count": int(
                        evaluation["collision_safe_grasp_count"]
                    ),
                    "nominal_grasp_evaluated_count": int(
                        evaluation["nominal_grasp_evaluated_count"]
                    ),
                    "nominal_grasp_converged_count": int(
                        evaluation["nominal_grasp_converged_count"]
                    ),
                    "robustness_query_count": int(
                        evaluation["robustness_query_count"]
                    ),
                    "strict_pi_candidate_count": int(
                        evaluation["strict_pi_candidate_count"]
                    ),
                    "bases_with_nominal_converged_grasp": int(
                        evaluation["bases_with_nominal_converged_grasp"]
                    ),
                    "robustness_enabled": bool(
                        evaluation["robustness_enabled"]
                    ),
                    "robustness_variant_count": int(
                        evaluation["robustness_variant_count"]
                    ),
                    "robustness_preferred_converged_variants": int(
                        evaluation["robustness_preferred_converged_variants"]
                    ),
                    "sam3_depth_attempt_count": len(
                        generated["depth_validation_attempts"]
                    ),
                    "target_distance_m": target_distance_m,
                    "selected_base_pose": selected_base_pose,
                    "motion_target": motion_target,
                    "virtual_feasible_grasp_count": (
                        virtual_feasible_grasp_count
                    ),
                    "virtual_evaluated_grasp_count": int(
                        executed.get("nominal_grasp_evaluated_count", 0)
                    ),
                    "requires_fresh_grasp_sampling": False,
                    "certified_goalset_available": True,
                    "curobo_called": False,
                    "certification": certification,
                    "motion": motion,
                }
            )
        except Exception as exc:
            result["reason"] = f"{type(exc).__name__}:{str(exc).strip()}"
            diagnostics = getattr(exc, "diagnostics", None)
            if isinstance(diagnostics, Mapping):
                # Keep bounded Pi search aggregates in the primitive failure;
                # the model-facing layer further compacts them.
                result["diagnostics"] = dict(diagnostics)
            previous = dict(self.last_debug or {})
            failed_after_phase = str(previous.get("phase", "unknown"))
            failure_debug = {
                **previous,
                "phase": "failed",
                "failed_after_phase": failed_after_phase,
                "failure_reason": result["reason"],
                "object_name": object_name,
                "arm": result["arm"],
            }
            if isinstance(diagnostics, Mapping):
                failure_debug["diagnostics"] = dict(diagnostics)
            self._publish_debug(failure_debug)
        finally:
            stop_result = self.env.controller.stop()
            result["final_stop"] = stop_result
            if not stop_result.get("success", False):
                result["success"] = False
                result["status"] = "failed"
                result["reason"] = (
                    f"{result['reason']}; final_stop_failed:{stop_result.get('reason')}"
                )
            result["elapsed_s"] = max(0.0, self.clock() - start_time)
        return result
