"""Target-conditioned, short-range docking for visible objects.

Ported from ``YOR/Agent/agents_yor/visible_object_navigation.py``'s
``YorVisibleObjectNavigationApi``. This module deliberately implements only
the first navigation stage: the requested object must initially be visible.
Straight motion remains the common case; a persistent, conservative RGB-D
occupancy map and A* supply bounded static-obstacle detours when necessary.

Reuses the same ``NavigationController`` instance already owned by
``YorEnvironment`` (``env.controller``) for turning/driving, and the same
``manipulation_config`` camera calibration (``camera_intrinsics``,
``camera_calibration_resolution``) used by ``robot/manipulation.py`` -- this
is a navigation feature, but it shares the ZED calibration block because both
need the same eye-in-hand camera intrinsics.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
import math
import time
from typing import Any

import numpy as np

from .ground_plane_gate import ACCEPTED, GroundPlaneGate
from .local_occupancy_planner import OccupancyPlan, StaticOccupancyMap
from .navigation_controller import wrap_angle


_DOCKING_CONFIG_ALIASES: dict[str, tuple[str, Callable[[float], float]]] = {
    # Human-friendly YAML names. Internally the controller keeps the original
    # canonical SI/radian fields so existing configurations remain valid.
    "forward_step_m": ("maximum_forward_step_m", float),
    "turn_step_deg": ("maximum_turn_step_rad", math.radians),
    "turn_speed_deg_s": ("turn_speed_rad_s", math.radians),
    "bearing_tolerance_deg": ("bearing_tolerance_rad", math.radians),
    "maximum_total_turn_deg": ("maximum_total_turn_rad", math.radians),
}


@dataclass(frozen=True)
class VisibleObjectDockingConfig:
    """Robot-owned parameters for visible-object docking."""

    docking_distance_m: float = 0.70
    target_min_depth_m: float = 0.10
    target_max_depth_m: float = 20.0
    target_min_depth_pixels: int = 20
    target_bbox_low_quantile: float = 0.02
    target_bbox_high_quantile: float = 0.98
    bearing_tolerance_rad: float = math.radians(10.0)
    maximum_turn_step_rad: float = math.radians(25.0)
    maximum_total_turn_rad: float = math.radians(360.0)
    turn_speed_rad_s: float = 0.35
    turn_timeout_s: float = 10.0
    maximum_forward_step_m: float = 0.50
    minimum_forward_command_m: float = 0.05
    maximum_total_forward_m: float = 20.0
    forward_speed_mps: float = 0.18
    corridor_half_width_m: float = 0.32
    # Inspect at most 0.60 m for a 0.50 m step. The drive controller separately
    # performs continuous near-field clearance checks while the base is moving.
    obstacle_path_margin_m: float = 0.10
    obstacle_min_depth_m: float = 0.10
    obstacle_max_depth_m: float = 20.0
    obstacle_top_row_fraction: float = 0.0
    obstacle_bottom_row_fraction: float = 1.0
    ground_camera_height_m: float = 1.122339129447937
    ground_down_camera_xyz: tuple[float, float, float] = (
        0.013630234132681617,
        0.9477663115351749,
        0.31867418382495005,
    )
    require_dynamic_ground_plane: bool = False
    ground_plane_max_age_s: float = 2.0
    # Plausibility gate on the SDK plane (robot/ground_plane_gate.py). The
    # SDK's fit is wrong in bursts whenever little floor is in view, so a
    # plane within one step of the plane in use is adopted at once, a larger
    # jump only after it has persisted for the settle time, and a plane
    # outside the error envelope around the configured fallback never. The
    # envelope therefore has to cover the lift's travel.
    ground_plane_max_height_step_m: float = 0.05
    ground_plane_max_tilt_step_deg: float = 3.0
    ground_plane_settle_s: float = 2.0
    ground_plane_max_height_error_m: float = 0.25
    ground_plane_max_tilt_error_deg: float = 15.0
    obstacle_min_height_m: float = 0.10
    obstacle_max_height_m: float = 1.45
    obstacle_temporal_frames: int = 3
    obstacle_cluster_width_m: float = 0.12
    obstacle_min_cluster_pixels: int = 30
    obstacle_min_sensor_pixels: int = 80
    avoidance_enabled: bool = True
    occupancy_resolution_m: float = 0.05
    occupancy_max_range_m: float = 5.0
    occupancy_pixel_stride: int = 3
    robot_radius_m: float = 0.30
    obstacle_clearance_margin_m: float = 0.0
    planner_lookahead_m: float = 0.80
    planner_max_expansions: int = 30_000
    target_lost_max_iterations: int = 8
    reacquisition_detector_model: str = "yoloe-26s-seg.pt"
    reacquisition_detector_device: str = "cuda:0"
    reacquisition_detector_image_size: int = 640
    reacquisition_detector_confidence: float = 0.15
    reacquisition_search_yaw_offsets_rad: tuple[float, ...] = tuple(
        math.radians(value)
        for value in (
            15.0,
            30.0,
            45.0,
            60.0,
            90.0,
            -15.0,
            -30.0,
            -45.0,
            -60.0,
            -90.0,
        )
    )
    avoidance_min_turn_clearance_m: float = 0.35
    avoidance_detection_range_m: float = 1.25
    avoidance_lateral_speed_mps: float = 0.10
    avoidance_motion_timeout_s: float = 18.0
    maximum_iterations: int = 100
    timeout_s: float = 300.0

    @property
    def effective_footprint_radius_m(self) -> float:
        """Robot radius plus an optional explicit obstacle clearance."""

        return self.robot_radius_m + self.obstacle_clearance_margin_m

    @classmethod
    def from_mapping(
        cls, values: Mapping[str, Any] | None
    ) -> "VisibleObjectDockingConfig":
        if not values:
            config = cls()
        else:
            normalized = dict(values)
            # The readiness-prior block belongs to the Nav2 backend; the legacy
            # backend ignores it but must accept the shared settings mapping.
            normalized.pop("readiness_prior", None)
            legacy_inflation = normalized.pop(
                "obstacle_inflation_radius_m", None
            )
            if legacy_inflation is not None:
                if (
                    "robot_radius_m" in normalized
                    or "obstacle_clearance_margin_m" in normalized
                ):
                    raise ValueError(
                        "configure either legacy 'obstacle_inflation_radius_m' "
                        "or explicit robot radius/clearance settings"
                    )
                # The legacy value represented the complete circular
                # footprint, not an extra obstacle dilation.
                normalized["robot_radius_m"] = legacy_inflation
                normalized["obstacle_clearance_margin_m"] = 0.0
            search_degrees = normalized.pop(
                "reacquisition_search_yaw_offsets_deg", None
            )
            if search_degrees is not None:
                if "reacquisition_search_yaw_offsets_rad" in normalized:
                    raise ValueError(
                        "configure only one of "
                        "'reacquisition_search_yaw_offsets_deg' or "
                        "'reacquisition_search_yaw_offsets_rad'"
                    )
                try:
                    normalized["reacquisition_search_yaw_offsets_rad"] = tuple(
                        math.radians(float(value)) for value in search_degrees
                    )
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        "reacquisition_search_yaw_offsets_deg must contain "
                        "finite numbers"
                    ) from exc
            for alias, (canonical, convert) in _DOCKING_CONFIG_ALIASES.items():
                if alias not in normalized:
                    continue
                if canonical in normalized:
                    raise ValueError(
                        f"configure only one of {alias!r} or {canonical!r}"
                    )
                try:
                    normalized[canonical] = convert(normalized.pop(alias))
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"{alias} must be a finite number") from exc
            known = set(cls.__dataclass_fields__)
            unknown = sorted(set(normalized) - known)
            if unknown:
                raise ValueError(f"unknown visible-object docking settings: {unknown}")
            config = cls(**normalized)
        config._validate()
        return config

    def _validate(self) -> None:
        values = asdict(self)
        integer_keys = {
            "target_min_depth_pixels",
            "obstacle_min_cluster_pixels",
            "obstacle_min_sensor_pixels",
            "obstacle_temporal_frames",
            "occupancy_pixel_stride",
            "planner_max_expansions",
            "target_lost_max_iterations",
            "reacquisition_detector_image_size",
            "maximum_iterations",
        }
        vector_keys = {
            "ground_down_camera_xyz",
            "reacquisition_search_yaw_offsets_rad",
        }
        string_keys = {
            "reacquisition_detector_model",
            "reacquisition_detector_device",
        }
        if not all(
            math.isfinite(float(value))
            for key, value in values.items()
            if key not in integer_keys | vector_keys | string_keys
        ):
            raise ValueError("visible-object docking settings must be finite")
        if not 0.1 <= self.docking_distance_m <= 1.5:
            raise ValueError("docking_distance_m must be in [0.1, 1.5]")
        if not 0.05 <= self.target_min_depth_m < self.target_max_depth_m <= 20.0:
            raise ValueError("invalid target depth range")
        if self.target_min_depth_pixels < 20:
            raise ValueError("target_min_depth_pixels must be at least 20")
        if not (
            0.0
            <= self.target_bbox_low_quantile
            < self.target_bbox_high_quantile
            <= 1.0
        ):
            raise ValueError("invalid target bbox quantiles")
        if not 0.0 < self.bearing_tolerance_rad < self.maximum_turn_step_rad:
            raise ValueError("invalid docking bearing tolerance")
        if not (
            self.maximum_turn_step_rad
            <= self.maximum_total_turn_rad
            <= 4.0 * math.pi
        ):
            raise ValueError("invalid docking turn limits")
        if not 0.05 <= self.turn_speed_rad_s <= 1.5:
            raise ValueError("turn_speed_rad_s must be in [0.05, 1.5]")
        if not 1.0 <= self.turn_timeout_s <= 30.0:
            raise ValueError("turn_timeout_s must be in [1, 30]")
        if not (
            0.025
            < self.minimum_forward_command_m
            <= self.maximum_forward_step_m
            <= self.maximum_total_forward_m
        ):
            raise ValueError("invalid docking forward-step limits")
        if not 0.01 <= self.forward_speed_mps <= 1.0:
            raise ValueError("forward_speed_mps must be in [0.01, 1.0]")
        if not 0.05 <= self.corridor_half_width_m <= 1.0:
            raise ValueError("corridor_half_width_m must be in [0.05, 1.0]")
        if self.obstacle_path_margin_m < 0.0:
            raise ValueError("obstacle_path_margin_m must be nonnegative")
        if not (
            0.05 <= self.obstacle_min_depth_m < self.obstacle_max_depth_m <= 20.0
        ):
            raise ValueError("invalid obstacle depth range")
        if not (
            0.0
            <= self.obstacle_top_row_fraction
            < self.obstacle_bottom_row_fraction
            <= 1.0
        ):
            raise ValueError("invalid obstacle image rows")
        if not 0.2 <= self.ground_camera_height_m <= 2.0:
            raise ValueError("ground_camera_height_m must be in [0.2, 2.0]")
        down = np.asarray(self.ground_down_camera_xyz, dtype=np.float64).reshape(-1)
        if down.shape != (3,) or not np.all(np.isfinite(down)):
            raise ValueError("ground_down_camera_xyz must contain three finite values")
        norm = float(np.linalg.norm(down))
        if norm <= 1e-6:
            raise ValueError("ground_down_camera_xyz must be nonzero")
        object.__setattr__(
            self, "ground_down_camera_xyz", tuple((down / norm).tolist())
        )
        if not isinstance(self.require_dynamic_ground_plane, bool):
            raise ValueError("require_dynamic_ground_plane must be boolean")
        if not 0.2 <= self.ground_plane_max_age_s <= 10.0:
            raise ValueError("ground_plane_max_age_s must be in [0.2, 10]")
        if not 0.005 <= self.ground_plane_max_height_step_m <= 0.5:
            raise ValueError(
                "ground_plane_max_height_step_m must be in [0.005, 0.5]"
            )
        if not 0.1 <= self.ground_plane_max_tilt_step_deg <= 45.0:
            raise ValueError("ground_plane_max_tilt_step_deg must be in [0.1, 45]")
        if not 0.1 <= self.ground_plane_settle_s <= 30.0:
            raise ValueError("ground_plane_settle_s must be in [0.1, 30]")
        if not (
            self.ground_plane_max_height_step_m
            <= self.ground_plane_max_height_error_m
            <= 2.0
        ):
            raise ValueError(
                "ground_plane_max_height_error_m must be in "
                "[ground_plane_max_height_step_m, 2.0]"
            )
        if not (
            self.ground_plane_max_tilt_step_deg
            <= self.ground_plane_max_tilt_error_deg
            <= 90.0
        ):
            raise ValueError(
                "ground_plane_max_tilt_error_deg must be in "
                "[ground_plane_max_tilt_step_deg, 90]"
            )
        if not (
            0.02
            <= self.obstacle_min_height_m
            < self.obstacle_max_height_m
            <= 2.5
        ):
            raise ValueError("invalid obstacle height range")
        if not 1 <= self.obstacle_temporal_frames <= 5:
            raise ValueError("obstacle_temporal_frames must be in [1, 5]")
        if not 0.02 <= self.obstacle_cluster_width_m <= 0.5:
            raise ValueError("invalid obstacle cluster width")
        if self.obstacle_min_cluster_pixels < 10:
            raise ValueError("obstacle_min_cluster_pixels must be at least 10")
        if self.obstacle_min_sensor_pixels < self.obstacle_min_cluster_pixels:
            raise ValueError(
                "obstacle_min_sensor_pixels must be >= obstacle_min_cluster_pixels"
            )
        if not isinstance(self.avoidance_enabled, bool):
            raise ValueError("avoidance_enabled must be boolean")
        if not 0.03 <= self.occupancy_resolution_m <= 0.25:
            raise ValueError("occupancy_resolution_m must be in [0.03, 0.25]")
        if not 1.0 <= self.occupancy_max_range_m <= self.obstacle_max_depth_m:
            raise ValueError(
                "occupancy_max_range_m must be in [1, obstacle_max_depth_m]"
            )
        if not 1 <= self.occupancy_pixel_stride <= 12:
            raise ValueError("occupancy_pixel_stride must be in [1, 12]")
        if not 0.15 <= self.robot_radius_m <= 1.0:
            raise ValueError("robot_radius_m must be in [0.15, 1.0]")
        if not 0.0 <= self.obstacle_clearance_margin_m <= 0.25:
            raise ValueError(
                "obstacle_clearance_margin_m must be in [0.0, 0.25]"
            )
        if self.effective_footprint_radius_m > 1.0:
            raise ValueError("effective footprint radius must be <= 1.0 m")
        if not self.minimum_forward_command_m <= self.planner_lookahead_m <= 2.0:
            raise ValueError("planner_lookahead_m must be in [minimum command, 2.0]")
        if not 100 <= self.planner_max_expansions <= 200_000:
            raise ValueError("planner_max_expansions must be in [100, 200000]")
        if not 1 <= self.target_lost_max_iterations <= 30:
            raise ValueError("target_lost_max_iterations must be in [1, 30]")
        if (
            not isinstance(self.reacquisition_detector_model, str)
            or not self.reacquisition_detector_model.strip()
        ):
            raise ValueError("reacquisition detector model must be non-empty")
        if (
            not isinstance(self.reacquisition_detector_device, str)
            or not self.reacquisition_detector_device.strip()
        ):
            raise ValueError("reacquisition detector device must be non-empty")
        if not 160 <= self.reacquisition_detector_image_size <= 1280:
            raise ValueError(
                "reacquisition_detector_image_size must be in [160, 1280]"
            )
        if not 0.0 < self.reacquisition_detector_confidence < 1.0:
            raise ValueError(
                "reacquisition_detector_confidence must be in (0, 1)"
            )
        search_offsets = tuple(self.reacquisition_search_yaw_offsets_rad)
        if not 1 <= len(search_offsets) <= 12 or not all(
            math.isfinite(value) and 0.0 < abs(value) <= math.pi / 2.0
            for value in search_offsets
        ):
            raise ValueError(
                "reacquisition search offsets must contain 1-12 finite, "
                "nonzero angles within +/-90 degrees"
            )
        object.__setattr__(
            self, "reacquisition_search_yaw_offsets_rad", search_offsets
        )
        if not (
            self.effective_footprint_radius_m
            <= self.avoidance_min_turn_clearance_m
            <= 2.0
        ):
            raise ValueError(
                "avoidance_min_turn_clearance_m must cover the inflated footprint"
            )
        if not (
            self.avoidance_min_turn_clearance_m
            < self.avoidance_detection_range_m
            <= self.occupancy_max_range_m
        ):
            raise ValueError(
                "avoidance_detection_range_m must exceed the turn clearance "
                "and fit occupancy_max_range_m"
            )
        if not 0.02 <= self.avoidance_lateral_speed_mps <= 1.0:
            raise ValueError("avoidance_lateral_speed_mps must be in [0.02, 1.0]")
        if not 2.0 <= self.avoidance_motion_timeout_s <= 45.0:
            raise ValueError("avoidance_motion_timeout_s must be in [2, 45]")
        if not 1 <= self.maximum_iterations <= 100:
            raise ValueError("maximum_iterations must be in [1, 100]")
        if not 1.0 <= self.timeout_s <= 300.0:
            raise ValueError("timeout_s must be in [1, 300]")


@dataclass(frozen=True)
class VisibleTarget:
    score: float
    mask: np.ndarray
    forward_min_m: float
    forward_max_m: float
    left_min_m: float
    left_max_m: float
    closest_forward_m: float
    closest_left_m: float
    distance_m: float
    bearing_rad: float
    depth_points: int

    def metrics(self) -> dict[str, Any]:
        return {
            "sam3_score": float(self.score),
            "target_bbox_forward_m": [
                float(self.forward_min_m),
                float(self.forward_max_m),
            ],
            "target_bbox_left_m": [float(self.left_min_m), float(self.left_max_m)],
            "target_closest_point_m": [
                float(self.closest_forward_m),
                float(self.closest_left_m),
            ],
            "target_distance_m": float(self.distance_m),
            "target_bearing_rad": float(self.bearing_rad),
            "target_bearing_deg": math.degrees(self.bearing_rad),
            "target_depth_points": int(self.depth_points),
        }


def _default_segment_client_factory() -> Callable[..., Any]:
    from .perception.sam3_client import init_sam3

    return init_sam3()


class VisibleObjectDockingController:
    """Perception-driven docking sequencing for one ``YorEnvironment``."""

    def __init__(
        self,
        env: Any,
        *,
        docking_config: Mapping[str, Any] | None = None,
        segment_client_factory: Callable[[], Callable[..., Any]] | None = None,
        detector_factory: Callable[[VisibleObjectDockingConfig], Any] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._env = env
        self._motion = env.controller
        manipulation = env.manipulation_config
        docking_values = (
            manipulation.get("visible_object_docking")
            if docking_config is None
            else docking_config
        )
        if docking_values is not None and not isinstance(docking_values, Mapping):
            raise TypeError("visible_object_docking configuration must be a mapping")
        self.config = VisibleObjectDockingConfig.from_mapping(docking_values)
        self._segment_client_factory = (
            segment_client_factory or _default_segment_client_factory
        )
        self._segment = None
        self._detector_factory = detector_factory or self._default_detector_factory
        self._detector: Any | None = None
        self._clock = clock
        self._active_ground_camera_height_m = self.config.ground_camera_height_m
        self._active_ground_down_camera_xyz = np.asarray(
            self.config.ground_down_camera_xyz, dtype=np.float64
        )
        # One gate for the controller's lifetime: a plane accepted on an
        # earlier observation is the plane in use on the next one.
        self._ground_gate = GroundPlaneGate(
            self.config.ground_camera_height_m,
            self.config.ground_down_camera_xyz,
            max_height_step_m=self.config.ground_plane_max_height_step_m,
            max_tilt_step_deg=self.config.ground_plane_max_tilt_step_deg,
            settle_s=self.config.ground_plane_settle_s,
            max_height_error_m=self.config.ground_plane_max_height_error_m,
            max_tilt_error_deg=self.config.ground_plane_max_tilt_error_deg,
        )
        self.last_ground_plane_debug: dict[str, Any] = {
            "dynamic": False,
            "camera_height_m": self.config.ground_camera_height_m,
            "down_camera_xyz": list(self.config.ground_down_camera_xyz),
            "age_s": None,
            "source": "configured_fallback",
        }
        self._docking_debug_callback: Callable[[dict[str, Any]], None] | None = None
        self._actual_pose_debug_callback: Callable[[list[float]], None] | None = None
        self.last_docking_debug: dict[str, Any] | None = None
        self.last_target_detection_debug: dict[str, Any] | None = None
        self._occupancy_map = self._new_occupancy_map()
        self._virtual_pose_xy_yaw = np.zeros(3, dtype=np.float64)
        self._last_observed_pose_xy_yaw: np.ndarray | None = None
        self._last_camera_timestamp_ns: int | None = None

    def _new_occupancy_map(self) -> StaticOccupancyMap:
        return StaticOccupancyMap(
            resolution_m=self.config.occupancy_resolution_m,
            inflation_radius_m=self.config.effective_footprint_radius_m,
        )

    def _segment_client(self):
        if self._segment is None:
            self._segment = self._segment_client_factory()
        return self._segment

    @staticmethod
    def _default_detector_factory(config: VisibleObjectDockingConfig) -> Any:
        from .perception.fast_detector import init_fast_detector

        return init_fast_detector(
            model=config.reacquisition_detector_model,
            device=config.reacquisition_detector_device,
            image_size=config.reacquisition_detector_image_size,
            confidence=config.reacquisition_detector_confidence,
        )

    def _fast_detector(self) -> Any:
        if self._detector is None:
            self._detector = self._detector_factory(self.config)
        return self._detector

    def _fast_target_detection(
        self,
        object_name: str,
        rgb: np.ndarray,
        intrinsics: np.ndarray,
    ) -> tuple[np.ndarray, float, float] | None:
        """Return the best YOLOE box, score, and planar bearing."""

        detections = self._fast_detector()(rgb, text_prompt=object_name) or ()
        best: tuple[np.ndarray, float] | None = None
        for item in detections:
            if isinstance(item, Mapping):
                box_value = item.get("box_xyxy", item.get("box"))
                score_value = item.get("score", 0.0)
            else:
                box_value = getattr(item, "box_xyxy", None)
                score_value = getattr(item, "score", 0.0)
            try:
                box = np.asarray(box_value, dtype=np.float64).reshape(-1)
                score = float(score_value)
            except (TypeError, ValueError):
                continue
            if (
                box.shape != (4,)
                or not np.all(np.isfinite(box))
                or not math.isfinite(score)
                or score < self.config.reacquisition_detector_confidence
                or box[2] <= box[0]
                or box[3] <= box[1]
            ):
                continue
            if best is None or score > best[1]:
                best = (box, score)
        if best is None:
            return None
        box, score = best
        center_x = 0.5 * float(box[0] + box[2])
        bearing = -math.atan2(
            center_x - float(intrinsics[0, 2]), float(intrinsics[0, 0])
        )
        return box, score, bearing

    def _reacquire_target_with_fast_detector(
        self,
        object_name: str,
        rgb: np.ndarray,
        intrinsics: np.ndarray,
        *,
        history: list[dict[str, Any]],
        metrics: dict[str, Any],
    ) -> tuple[bool, str]:
        """Stop translation and scan in yaw until YOLOE finds the target."""

        metrics["reacquisition_attempts"] = int(
            metrics.get("reacquisition_attempts", 0)
        ) + 1

        def detect(
            image: np.ndarray, camera: np.ndarray
        ) -> tuple[np.ndarray, float, float] | None:
            metrics["reacquisition_detector_calls"] = int(
                metrics.get("reacquisition_detector_calls", 0)
            ) + 1
            return self._fast_target_detection(object_name, image, camera)

        try:
            detection = detect(rgb, intrinsics)
        except Exception as exc:
            metrics["reacquisition_detector_error"] = (
                f"{type(exc).__name__}: {exc}"
            )
            return False, "target_reacquisition_detector_failed"
        current_offset = 0.0
        for desired_offset in self.config.reacquisition_search_yaw_offsets_rad:
            if detection is not None:
                break
            turn = wrap_angle(float(desired_offset) - current_offset)
            total_turn = float(metrics["total_absolute_turn_rad"]) + abs(turn)
            if total_turn > self.config.maximum_total_turn_rad:
                return False, "reacquisition_turn_budget_exceeded"
            motion = self._motion.turn_relative(
                turn,
                max_yaw_rad_s=self.config.turn_speed_rad_s,
                timeout_s=self.config.turn_timeout_s,
            )
            history.append(self._primitive_summary(motion))
            metrics["total_absolute_turn_rad"] = total_turn
            metrics["reacquisition_search_turns"] = int(
                metrics.get("reacquisition_search_turns", 0)
            ) + 1
            if not motion.get("success", False):
                return (
                    False,
                    "reacquisition_search_turn_failed:"
                    f"{motion.get('reason', 'unknown')}",
                )
            self._record_successful_motion(turn_rad=turn)
            current_offset = float(desired_offset)
            rgb, _, intrinsics = self._camera_input(require_new_frame=True)
            try:
                detection = detect(rgb, intrinsics)
            except Exception as exc:
                metrics["reacquisition_detector_error"] = (
                    f"{type(exc).__name__}: {exc}"
                )
                return False, "target_reacquisition_detector_failed"

        if detection is None:
            return False, "target_reacquisition_failed"

        box, score, bearing = detection
        metrics["reacquisition_detector_score"] = score
        metrics["reacquisition_box_xyxy"] = box.tolist()
        metrics["reacquisition_bearing_rad"] = bearing
        if abs(bearing) <= self.config.bearing_tolerance_rad:
            return True, "target_found_centered"

        turn = float(
            np.clip(
                bearing,
                -self.config.maximum_turn_step_rad,
                self.config.maximum_turn_step_rad,
            )
        )
        total_turn = float(metrics["total_absolute_turn_rad"]) + abs(turn)
        if total_turn > self.config.maximum_total_turn_rad:
            return False, "reacquisition_turn_budget_exceeded"
        motion = self._motion.turn_relative(
            turn,
            max_yaw_rad_s=self.config.turn_speed_rad_s,
            timeout_s=self.config.turn_timeout_s,
        )
        history.append(self._primitive_summary(motion))
        metrics["total_absolute_turn_rad"] = total_turn
        metrics["reacquisition_center_turns"] = int(
            metrics.get("reacquisition_center_turns", 0)
        ) + 1
        if not motion.get("success", False):
            return (
                False,
                f"reacquisition_center_turn_failed:{motion.get('reason', 'unknown')}",
            )
        self._record_successful_motion(turn_rad=turn)
        return True, "target_found_and_centered"

    def _camera_input(
        self, *, require_new_frame: bool = False
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        previous_timestamp_ns = self._last_camera_timestamp_ns
        next_observation = getattr(self._env, "observe_next", None)
        if require_new_frame and callable(next_observation):
            observation = next_observation()
        else:
            observation = self._env.observe()
        camera = observation["robot0_robotview"]
        rgb = np.asarray(camera["images"]["rgb"], dtype=np.uint8)
        depth = np.asarray(camera["images"]["depth"], dtype=np.float32)
        if depth.ndim == 3 and depth.shape[-1] == 1:
            depth = depth[..., 0]
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise RuntimeError("ZED RGB image must have shape HxWx3")
        if depth.shape != rgb.shape[:2]:
            raise RuntimeError("ZED RGB and depth shapes do not match")

        config = self._env.manipulation_config
        intrinsics = np.asarray(config.get("camera_intrinsics"), dtype=np.float64)
        resolution = np.asarray(
            config.get("camera_calibration_resolution"), dtype=np.int64
        ).reshape(-1)
        if intrinsics.shape != (3, 3) or not np.all(np.isfinite(intrinsics)):
            raise RuntimeError("camera_intrinsics must be a finite 3x3 matrix")
        if resolution.shape != (2,) or np.any(resolution <= 0):
            raise RuntimeError(
                "camera_calibration_resolution must contain [width, height]"
            )
        intrinsics = intrinsics.copy()
        height, width = depth.shape
        scale_x = width / float(resolution[0])
        scale_y = height / float(resolution[1])
        intrinsics[0, 0] *= scale_x
        intrinsics[0, 2] = scale_x * (intrinsics[0, 2] + 0.5) - 0.5
        intrinsics[1, 1] *= scale_y
        intrinsics[1, 2] = scale_y * (intrinsics[1, 2] + 0.5) - 0.5
        if intrinsics[0, 0] <= 0.0 or intrinsics[1, 1] <= 0.0:
            raise RuntimeError("camera focal lengths must be positive")
        timestamp = camera.get("timestamp_ns")
        if timestamp is not None:
            timestamp_ns = int(timestamp)
            if (
                require_new_frame
                and callable(next_observation)
                and previous_timestamp_ns is not None
                and timestamp_ns <= previous_timestamp_ns
            ):
                raise RuntimeError(
                    "ZED did not provide a newer frame during temporal sampling"
                )
            self._last_camera_timestamp_ns = timestamp_ns
        base = observation.get("base", {})
        pose = np.asarray(base.get("pose_xy_yaw", []), dtype=np.float64).reshape(-1)
        if pose.shape == (3,) and np.all(np.isfinite(pose)):
            self._last_observed_pose_xy_yaw = pose.copy()
            self._virtual_pose_xy_yaw = pose.copy()
        else:
            self._last_observed_pose_xy_yaw = None
        self._update_ground_plane(camera)
        return np.ascontiguousarray(rgb), np.ascontiguousarray(depth), intrinsics

    def _update_ground_plane(self, camera: Mapping[str, Any]) -> None:
        ground = camera.get("ground_plane")
        frame_timestamp = camera.get("timestamp_ns")
        failure = "ground plane is absent from the ZED frame"
        if isinstance(ground, Mapping) and bool(ground.get("valid", False)):
            try:
                height = float(ground.get("camera_height_m"))
                down = np.asarray(
                    ground.get("down_camera_xyz"), dtype=np.float64
                ).reshape(-1)
                plane_timestamp = int(ground.get("timestamp_ns"))
                image_timestamp = int(frame_timestamp)
                age_s = max(0.0, (image_timestamp - plane_timestamp) * 1e-9)
            except (TypeError, ValueError, OverflowError) as exc:
                failure = f"ground plane fields are invalid: {exc}"
            else:
                norm = float(np.linalg.norm(down))
                if (
                    math.isfinite(height)
                    and 0.1 < height < 3.5
                    and down.shape == (3,)
                    and np.all(np.isfinite(down))
                    and norm > 1e-6
                    and age_s <= self.config.ground_plane_max_age_s
                ):
                    down /= norm
                    # A held plane is still the SDK's: the last plane it
                    # convinced the gate of, so it satisfies the dynamic
                    # requirement and only the trace shows the refusal.
                    decision = self._ground_gate.offer(
                        height, down, image_timestamp * 1e-9
                    )
                    self._active_ground_camera_height_m = decision.camera_height_m
                    self._active_ground_down_camera_xyz = decision.down_camera_xyz
                    self.last_ground_plane_debug = {
                        "dynamic": True,
                        "camera_height_m": decision.camera_height_m,
                        "down_camera_xyz": decision.down_camera_xyz.tolist(),
                        "age_s": age_s,
                        "source": "zed_sdk_floor_plane",
                        "gate": decision.status,
                        "gate_reason": decision.reason,
                        "held_for_s": decision.held_for_s,
                    }
                    if decision.status != ACCEPTED:
                        self.last_ground_plane_debug[
                            "rejected_camera_height_m"
                        ] = height
                    return
                failure = (
                    f"ground plane is stale or invalid: height={height}, "
                    f"age_s={age_s:.3f}"
                )
        if self.config.require_dynamic_ground_plane:
            raise RuntimeError(f"dynamic_ground_plane_unavailable: {failure}")
        self._active_ground_camera_height_m = self.config.ground_camera_height_m
        self._active_ground_down_camera_xyz = np.asarray(
            self.config.ground_down_camera_xyz, dtype=np.float64
        )
        self.last_ground_plane_debug = {
            "dynamic": False,
            "camera_height_m": self.config.ground_camera_height_m,
            "down_camera_xyz": list(self.config.ground_down_camera_xyz),
            "age_s": None,
            "source": "configured_fallback",
            "fallback_reason": failure,
        }

    def _detect_target(
        self,
        object_name: str,
        rgb: np.ndarray,
        depth: np.ndarray,
        intrinsics: np.ndarray,
    ) -> VisibleTarget:
        self.last_target_detection_debug = None
        results = self._segment_client()(rgb, text_prompt=object_name)
        if not results:
            raise RuntimeError(f"SAM3 found no instance for {object_name!r}")
        best = max(results, key=lambda item: float(item.get("score", 0.0)))
        score = float(best.get("score", 0.0))
        score_threshold = float(
            self._env.manipulation_config.get("sam3_score_threshold", 0.05)
        )
        if not math.isfinite(score_threshold) or not 0.0 <= score_threshold <= 1.0:
            raise ValueError("sam3_score_threshold must be finite and in [0, 1]")
        if score < score_threshold:
            raise RuntimeError(
                f"best SAM3 score {score:.3f} is below threshold "
                f"{score_threshold:.3f}"
            )
        mask = np.asarray(best.get("mask"), dtype=bool)
        if mask.shape != depth.shape:
            raise RuntimeError(
                f"SAM3 mask shape {mask.shape} does not match depth {depth.shape}"
            )
        mask_pixels = int(np.count_nonzero(mask))
        self.last_target_detection_debug = {
            "object_name": object_name,
            "sam3_score": score,
            "mask": mask,
            "mask_pixels": mask_pixels,
            "finite_positive_depth_pixels": None,
            "in_range_depth_pixels": None,
            "configured_depth_range_m": [
                self.config.target_min_depth_m,
                self.config.target_max_depth_m,
            ],
            "observed_depth_p05_p50_p95_m": None,
        }
        if mask_pixels < self.config.target_min_depth_pixels:
            raise RuntimeError(
                f"SAM3 mask has only {mask_pixels} pixels; at least "
                f"{self.config.target_min_depth_pixels} are required"
            )

        finite_positive = mask & np.isfinite(depth) & (depth > 0.0)
        observed_depth = np.asarray(depth[finite_positive], dtype=np.float64)
        valid = (
            mask
            & np.isfinite(depth)
            & (depth >= self.config.target_min_depth_m)
            & (depth <= self.config.target_max_depth_m)
        )
        rows, columns = np.nonzero(valid)
        depth_percentiles = (
            None
            if observed_depth.size == 0
            else np.quantile(observed_depth, [0.05, 0.5, 0.95]).tolist()
        )
        self.last_target_detection_debug.update(
            {
                "finite_positive_depth_pixels": int(observed_depth.size),
                "in_range_depth_pixels": int(rows.size),
                "observed_depth_p05_p50_p95_m": depth_percentiles,
            }
        )
        if rows.size < self.config.target_min_depth_pixels:
            raise RuntimeError(
                "object mask has insufficient valid metric depth: "
                f"mask_pixels={mask_pixels}, "
                f"finite_positive={observed_depth.size}, in_range={rows.size}, "
                f"required={self.config.target_min_depth_pixels}, "
                f"range=[{self.config.target_min_depth_m:.2f}, "
                f"{self.config.target_max_depth_m:.2f}]m, "
                f"observed_p05_p50_p95_m={depth_percentiles}"
            )
        z = depth[rows, columns].astype(np.float64)
        points_camera = np.column_stack(
            [
                (columns - intrinsics[0, 2]) * z / intrinsics[0, 0],
                (rows - intrinsics[1, 2]) * z / intrinsics[1, 1],
                z,
            ]
        )
        _, planar_forward, planar_left = self._ground_axes()
        forward = points_camera @ planar_forward
        left = points_camera @ planar_left
        low = self.config.target_bbox_low_quantile
        high = self.config.target_bbox_high_quantile
        forward_min, forward_max = np.quantile(forward, [low, high])
        left_min, left_max = np.quantile(left, [low, high])
        closest_forward = float(np.clip(0.0, forward_min, forward_max))
        closest_left = float(np.clip(0.0, left_min, left_max))
        distance = math.hypot(closest_forward, closest_left)
        bearing = math.atan2(closest_left, closest_forward)
        if not all(
            math.isfinite(value)
            for value in (
                forward_min,
                forward_max,
                left_min,
                left_max,
                distance,
                bearing,
            )
        ):
            raise RuntimeError("target planar bounding box is non-finite")
        return VisibleTarget(
            score=score,
            mask=mask,
            forward_min_m=float(forward_min),
            forward_max_m=float(forward_max),
            left_min_m=float(left_min),
            left_max_m=float(left_max),
            closest_forward_m=closest_forward,
            closest_left_m=closest_left,
            distance_m=distance,
            bearing_rad=bearing,
            depth_points=int(rows.size),
        )

    def _ground_axes(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return calibrated camera down, floor-forward and floor-left axes."""

        down = self._active_ground_down_camera_xyz.copy()
        optical_forward = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
        planar_forward = optical_forward - down * float(
            np.dot(optical_forward, down)
        )
        norm = float(np.linalg.norm(planar_forward))
        if norm <= 1e-6:
            raise RuntimeError("camera optical axis is parallel to calibrated down")
        planar_forward /= norm
        planar_left = np.cross(planar_forward, down)
        planar_left /= np.linalg.norm(planar_left)
        return down, planar_forward, planar_left

    def _current_pose(self) -> np.ndarray:
        if self._last_observed_pose_xy_yaw is not None:
            return self._last_observed_pose_xy_yaw.copy()
        return self._virtual_pose_xy_yaw.copy()

    @staticmethod
    def _local_to_world(
        forward_m: np.ndarray | float,
        left_m: np.ndarray | float,
        pose_xy_yaw: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        cosine = math.cos(float(pose_xy_yaw[2]))
        sine = math.sin(float(pose_xy_yaw[2]))
        forward = np.asarray(forward_m, dtype=np.float64)
        left = np.asarray(left_m, dtype=np.float64)
        world_x = float(pose_xy_yaw[0]) + cosine * forward - sine * left
        world_y = float(pose_xy_yaw[1]) + sine * forward + cosine * left
        return world_x, world_y

    def _target_world_xy(
        self, target: VisibleTarget, pose_xy_yaw: np.ndarray
    ) -> tuple[float, float]:
        world_x, world_y = self._local_to_world(
            target.closest_forward_m, target.closest_left_m, pose_xy_yaw
        )
        return float(world_x), float(world_y)

    def _integrate_occupancy(
        self,
        depth: np.ndarray,
        intrinsics: np.ndarray,
        *,
        pose_xy_yaw: np.ndarray,
        excluded_mask: np.ndarray | None,
    ) -> dict[str, int]:
        """Fuse one stopped, temporally filtered depth view into the world map."""

        stride = self.config.occupancy_pixel_stride
        sampled_rows, sampled_columns = np.mgrid[
            0 : depth.shape[0] : stride, 0 : depth.shape[1] : stride
        ]
        rows = sampled_rows.reshape(-1)
        columns = sampled_columns.reshape(-1)
        z = np.asarray(depth[rows, columns], dtype=np.float64)
        valid = (
            np.isfinite(z)
            & (z >= self.config.obstacle_min_depth_m)
            & (z <= self.config.occupancy_max_range_m)
        )
        rows = rows[valid]
        columns = columns[valid]
        z = z[valid]
        if z.size == 0:
            return {"rays": 0, "structural_endpoints": 0}
        points_camera = np.column_stack(
            [
                (columns - intrinsics[0, 2]) * z / intrinsics[0, 0],
                (rows - intrinsics[1, 2]) * z / intrinsics[1, 1],
                z,
            ]
        )
        down, planar_forward, planar_left = self._ground_axes()
        forward = points_camera @ planar_forward
        left = points_camera @ planar_left
        height_above_floor = (
            self._active_ground_camera_height_m - points_camera @ down
        )
        in_front = forward > 0.0
        rows = rows[in_front]
        columns = columns[in_front]
        forward = forward[in_front]
        left = left[in_front]
        height_above_floor = height_above_floor[in_front]
        structural = (
            (height_above_floor >= self.config.obstacle_min_height_m)
            & (height_above_floor <= self.config.obstacle_max_height_m)
        )
        if excluded_mask is not None:
            mask = np.asarray(excluded_mask, dtype=bool)
            if mask.shape != depth.shape:
                raise ValueError("excluded occupancy mask does not match depth")
            structural &= ~mask[rows, columns]
        world_x, world_y = self._local_to_world(forward, left, pose_xy_yaw)
        endpoints = list(
            zip(world_x.tolist(), world_y.tolist(), strict=True)
        )
        self._occupancy_map.integrate_rays(
            (float(pose_xy_yaw[0]), float(pose_xy_yaw[1])),
            endpoints,
            structural.tolist(),
        )
        return {
            "rays": int(forward.size),
            "structural_endpoints": int(np.count_nonzero(structural)),
        }

    def _plan_avoidance(
        self,
        *,
        pose_xy_yaw: np.ndarray,
        target_world_xy: tuple[float, float],
    ) -> OccupancyPlan:
        return self._occupancy_map.plan_to_docking_ring(
            start_xy=(float(pose_xy_yaw[0]), float(pose_xy_yaw[1])),
            target_xy=target_world_xy,
            docking_distance_m=self.config.docking_distance_m,
            maximum_expansions=self.config.planner_max_expansions,
        )

    def _plan_avoidance_frontier(
        self,
        *,
        pose_xy_yaw: np.ndarray,
        target_world_xy: tuple[float, float],
        preferred_side: int | None,
    ) -> OccupancyPlan:
        """Find a known-free side step that can reveal an occluded route."""

        minimum_lateral_m = (
            max(
                self.config.corridor_half_width_m,
                self.config.effective_footprint_radius_m,
            )
            if preferred_side is None
            else self.config.occupancy_resolution_m
        )
        return self._occupancy_map.plan_to_lateral_frontier(
            start_xy=(float(pose_xy_yaw[0]), float(pose_xy_yaw[1])),
            target_xy=target_world_xy,
            minimum_lateral_m=minimum_lateral_m,
            minimum_target_progress_m=self.config.occupancy_resolution_m,
            maximum_expansions=self.config.planner_max_expansions,
            preferred_side=preferred_side,
        )

    @staticmethod
    def _avoidance_path_side(
        path_xy: tuple[tuple[float, float], ...],
        *,
        start_xy: tuple[float, float],
        target_xy: tuple[float, float],
    ) -> int | None:
        if len(path_xy) < 2:
            return None
        target_dx = float(target_xy[0]) - float(start_xy[0])
        target_dy = float(target_xy[1]) - float(start_xy[1])
        endpoint_dx = float(path_xy[-1][0]) - float(start_xy[0])
        endpoint_dy = float(path_xy[-1][1]) - float(start_xy[1])
        lateral = target_dx * endpoint_dy - target_dy * endpoint_dx
        if abs(lateral) <= 1e-9:
            return None
        return 1 if lateral > 0.0 else -1

    def _waypoint_body_delta(
        self, waypoint_xy: tuple[float, float], pose_xy_yaw: np.ndarray
    ) -> tuple[float, float]:
        dx = float(waypoint_xy[0]) - float(pose_xy_yaw[0])
        dy = float(waypoint_xy[1]) - float(pose_xy_yaw[1])
        cosine = math.cos(float(pose_xy_yaw[2]))
        sine = math.sin(float(pose_xy_yaw[2]))
        return cosine * dx + sine * dy, -sine * dx + cosine * dy

    @staticmethod
    def _precise_grid_step_tolerance(distance_m: float) -> float:
        """Use tighter ZED convergence for a sub-5-cm A* corner step."""

        return max(0.002, min(0.010, 0.25 * float(distance_m)))

    def _occupancy_motion_guard(
        self, waypoint_xy: tuple[float, float]
    ) -> Callable[[Any, dict[str, float]], dict[str, object]]:
        """Continuously validate a holonomic segment using live ZED pose."""

        def guard(frame: Any, _state: dict[str, float]) -> dict[str, object]:
            pose = getattr(frame, "planar_pose", None)
            if pose is None or not bool(getattr(pose, "valid", False)):
                return {"clear": False, "reason": "pose_invalid"}
            values = np.asarray(
                [pose.x_m, pose.y_m, pose.yaw_rad], dtype=np.float64
            )
            if not np.all(np.isfinite(values)):
                return {"clear": False, "reason": "pose_nonfinite"}
            return self._occupancy_map.segment_status(
                (float(values[0]), float(values[1])),
                waypoint_xy,
                allow_start_in_inflated=True,
            )

        return guard

    def _record_successful_motion(
        self,
        *,
        turn_rad: float = 0.0,
        forward_m: float = 0.0,
        left_m: float = 0.0,
    ) -> None:
        """Dead-reckon only for minimal test environments without ZED pose."""

        if self._last_observed_pose_xy_yaw is not None:
            return
        pose = self._virtual_pose_xy_yaw
        start_yaw = float(pose[2])
        cosine = math.cos(start_yaw)
        sine = math.sin(start_yaw)
        pose[0] += cosine * float(forward_m) - sine * float(left_m)
        pose[1] += sine * float(forward_m) + cosine * float(left_m)
        pose[2] = wrap_angle(start_yaw + float(turn_rad))

    def _notify_actual_pose(self, pose_xy_yaw: list[float]) -> None:
        callback = self._actual_pose_debug_callback
        if callback is None:
            return
        try:
            callback([float(value) for value in pose_xy_yaw])
        except Exception:
            # Visualization is diagnostic and must not stop the base.
            return

    def _turn_to_path_heading(
        self,
        bearing_rad: float,
        *,
        history: list[dict[str, Any]],
        metrics: dict[str, Any],
    ) -> str | None:
        """Execute a path-heading turn in bounded stopped increments."""

        remaining = wrap_angle(float(bearing_rad))
        while abs(remaining) > 1e-4:
            turn = float(
                np.clip(
                    remaining,
                    -self.config.maximum_turn_step_rad,
                    self.config.maximum_turn_step_rad,
                )
            )
            total_turn = float(metrics["total_absolute_turn_rad"]) + abs(turn)
            if total_turn > self.config.maximum_total_turn_rad:
                return "turn_budget_exceeded"
            motion = self._motion.turn_relative(
                turn,
                max_yaw_rad_s=self.config.turn_speed_rad_s,
                timeout_s=self.config.turn_timeout_s,
            )
            history.append(self._primitive_summary(motion))
            metrics["total_absolute_turn_rad"] = total_turn
            if not motion.get("success", False):
                return f"turn_failed: {motion.get('reason', 'unknown')}"
            self._record_successful_motion(turn_rad=turn)
            remaining = wrap_angle(remaining - turn)
        return None

    def _straight_path_status(
        self,
        depth: np.ndarray,
        intrinsics: np.ndarray,
        *,
        bearing_rad: float,
        travel_distance_m: float,
        excluded_mask: np.ndarray | None = None,
        minimum_path_limit_m: float | None = None,
    ) -> dict[str, Any]:
        """Check a base-width straight corridor and reject a coherent obstacle."""

        height, width = depth.shape
        row0 = int(round(height * self.config.obstacle_top_row_fraction))
        row1 = int(round(height * self.config.obstacle_bottom_row_fraction))
        roi = np.zeros(depth.shape, dtype=bool)
        roi[row0:row1] = True
        valid = (
            roi
            & np.isfinite(depth)
            & (depth >= self.config.obstacle_min_depth_m)
            & (depth <= self.config.obstacle_max_depth_m)
        )
        rows, columns = np.nonzero(valid)
        if excluded_mask is None:
            excluded = np.zeros(rows.shape, dtype=bool)
        else:
            mask = np.asarray(excluded_mask, dtype=bool)
            if mask.shape != depth.shape:
                raise ValueError("excluded obstacle mask does not match depth")
            excluded = mask[rows, columns]
        z = depth[rows, columns].astype(np.float64)
        points_camera = np.column_stack(
            [
                (columns - intrinsics[0, 2]) * z / intrinsics[0, 0],
                (rows - intrinsics[1, 2]) * z / intrinsics[1, 1],
                z,
            ]
        )
        down, planar_forward, planar_left = self._ground_axes()
        forward = points_camera @ planar_forward
        left = points_camera @ planar_left
        height_above_floor = self._active_ground_camera_height_m - points_camera @ down
        cosine = math.cos(bearing_rad)
        sine = math.sin(bearing_rad)
        along = cosine * forward + sine * left
        cross = -sine * forward + cosine * left
        in_corridor = (
            (along > 0.0) & (np.abs(cross) <= self.config.corridor_half_width_m)
        )
        sensor_values = along[in_corridor]
        sensor_cross = cross[in_corridor]
        sensor_rows = rows[in_corridor]
        sensor_columns = columns[in_corridor]
        sensor_excluded = excluded[in_corridor]
        sensor_pixels = int(sensor_values.size)
        if sensor_pixels < self.config.obstacle_min_sensor_pixels:
            return {
                "valid": False,
                "clear": False,
                "reason": "insufficient_corridor_depth",
                "sensor_pixels": sensor_pixels,
                "ground_filtered_pixels": 0,
                "height_candidate_pixels": 0,
                "target_mask_filtered_pixels": 0,
                "obstacle_pixels": 0,
                "depth_layer_pixels": 0,
                "largest_rejected_component_pixels": 0,
                "nearest_obstacle_m": None,
            }

        next_step_m = max(
            self.config.minimum_forward_command_m,
            min(self.config.maximum_forward_step_m, travel_distance_m),
        )
        path_limit = next_step_m + self.config.obstacle_path_margin_m
        if minimum_path_limit_m is not None:
            path_limit = max(path_limit, float(minimum_path_limit_m))
        structural = (
            (height_above_floor >= self.config.obstacle_min_height_m)
            & (height_above_floor <= self.config.obstacle_max_height_m)
        )
        structural_sensor = structural[in_corridor]
        candidate_mask = (
            structural_sensor
            & ~sensor_excluded
            & (sensor_values <= path_limit)
            & (sensor_values >= self.config.obstacle_min_depth_m)
        )
        candidates = sensor_values[candidate_mask]
        candidate_cross = sensor_cross[candidate_mask]
        candidate_rows = sensor_rows[candidate_mask]
        candidate_columns = sensor_columns[candidate_mask]
        order = np.argsort(candidates)
        candidates = candidates[order]
        candidate_cross = candidate_cross[order]
        candidate_rows = candidate_rows[order]
        candidate_columns = candidate_columns[order]
        required = self.config.obstacle_min_cluster_pixels
        nearest: float | None = None
        support = 0
        depth_layer_pixels = 0
        largest_rejected_component = 0
        obstacle_points = np.empty((0, 2), dtype=np.float64)
        start = 0
        while start <= candidates.size - required:
            stop = int(
                np.searchsorted(
                    candidates,
                    candidates[start] + self.config.obstacle_cluster_width_m,
                    side="right",
                )
            )
            if stop - start >= required:
                cluster = candidates[start:stop]
                depth_layer_pixels = max(depth_layer_pixels, int(cluster.size))
                component = self._largest_image_component(
                    candidate_rows[start:stop],
                    candidate_columns[start:stop],
                )
                largest_rejected_component = max(
                    largest_rejected_component, int(component.size)
                )
                if component.size >= required:
                    selected = cluster[component]
                    nearest = float(np.quantile(selected, 0.25))
                    support = int(component.size)
                    obstacle_points = np.column_stack(
                        [selected, candidate_cross[start:stop][component]]
                    )
                    if obstacle_points.shape[0] > 500:
                        keep = np.linspace(
                            0,
                            obstacle_points.shape[0] - 1,
                            500,
                            dtype=np.int64,
                        )
                        obstacle_points = obstacle_points[keep]
                    break
                # Avoid repeating an expensive connectivity check for almost
                # the same sliding depth window while retaining 50% overlap.
                next_depth = candidates[start] + 0.5 * self.config.obstacle_cluster_width_m
                start = max(
                    start + 1,
                    int(np.searchsorted(candidates, next_depth, side="right")),
                )
            else:
                start += 1
        return {
            "valid": True,
            "clear": nearest is None,
            "reason": "path_clear" if nearest is None else "path_blocked",
            "sensor_pixels": sensor_pixels,
            "ground_filtered_pixels": int(
                np.count_nonzero(
                    in_corridor
                    & (height_above_floor < self.config.obstacle_min_height_m)
                )
            ),
            "height_candidate_pixels": int(
                np.count_nonzero(in_corridor & structural)
            ),
            "target_mask_filtered_pixels": int(
                np.count_nonzero(
                    sensor_excluded
                    & structural_sensor
                    & (sensor_values <= path_limit)
                )
            ),
            "obstacle_pixels": support,
            "depth_layer_pixels": depth_layer_pixels,
            "largest_rejected_component_pixels": (
                0 if nearest is not None else largest_rejected_component
            ),
            "nearest_obstacle_m": nearest,
            "path_limit_m": float(path_limit),
            # Private diagnostic data is removed before the primitive result is
            # returned to generated code.
            "_obstacle_points_path_m": obstacle_points,
        }

    def _stable_path_status(
        self,
        depth: np.ndarray,
        intrinsics: np.ndarray,
        *,
        bearing_rad: float,
        travel_distance_m: float,
        target_mask: np.ndarray | None = None,
        minimum_path_limit_m: float | None = None,
    ) -> dict[str, Any]:
        """Median several stopped RGB-D frames before obstacle classification."""

        frames = [np.asarray(depth, dtype=np.float32)]
        valid_counts = [int(np.count_nonzero(np.isfinite(frames[0])))]
        for _ in range(1, self.config.obstacle_temporal_frames):
            _, next_depth, next_intrinsics = self._camera_input(require_new_frame=True)
            if next_depth.shape != frames[0].shape:
                raise RuntimeError("ZED depth shape changed during obstacle sampling")
            if not np.allclose(next_intrinsics, intrinsics, atol=1e-6, rtol=0.0):
                raise RuntimeError(
                    "ZED intrinsics changed during obstacle sampling"
                )
            frames.append(np.asarray(next_depth, dtype=np.float32))
            valid_counts.append(int(np.count_nonzero(np.isfinite(next_depth))))
        stack = np.stack(frames, axis=0)
        valid = (
            np.isfinite(stack)
            & (stack >= self.config.obstacle_min_depth_m)
            & (stack <= self.config.obstacle_max_depth_m)
        )
        stable_depth = np.ma.median(
            np.ma.array(stack, mask=~valid), axis=0
        ).filled(np.nan)
        temporal_support = np.count_nonzero(valid, axis=0)
        required_support = len(frames) // 2 + 1
        stable_depth[temporal_support < required_support] = np.nan
        status = self._straight_path_status(
            np.asarray(stable_depth, dtype=np.float32),
            intrinsics,
            bearing_rad=bearing_rad,
            travel_distance_m=travel_distance_m,
            excluded_mask=target_mask,
            minimum_path_limit_m=minimum_path_limit_m,
        )
        status["temporal_frames"] = len(frames)
        status["bearing_rad"] = float(bearing_rad)
        status["temporal_required_support"] = required_support
        status["temporal_supported_depth_pixels"] = int(
            np.count_nonzero(temporal_support >= required_support)
        )
        status["temporal_valid_depth_pixels"] = valid_counts
        status["_stable_depth_m"] = np.asarray(stable_depth, dtype=np.float32)
        return status

    @staticmethod
    def _largest_image_component(
        rows: np.ndarray, columns: np.ndarray
    ) -> np.ndarray:
        """Return indices of the largest 8-connected candidate-pixel group."""

        row_values = np.asarray(rows, dtype=np.int64).reshape(-1)
        column_values = np.asarray(columns, dtype=np.int64).reshape(-1)
        if row_values.shape != column_values.shape:
            raise ValueError("component row/column arrays must have equal shape")
        coordinate_to_index = {
            (int(row), int(column)): index
            for index, (row, column) in enumerate(
                zip(row_values, column_values, strict=True)
            )
        }
        unvisited = set(coordinate_to_index)
        largest: list[int] = []
        while unvisited:
            seed = unvisited.pop()
            stack = [seed]
            component = [coordinate_to_index[seed]]
            while stack:
                row, column = stack.pop()
                for row_delta in (-1, 0, 1):
                    for column_delta in (-1, 0, 1):
                        if row_delta == 0 and column_delta == 0:
                            continue
                        neighbor = (row + row_delta, column + column_delta)
                        if neighbor not in unvisited:
                            continue
                        unvisited.remove(neighbor)
                        stack.append(neighbor)
                        component.append(coordinate_to_index[neighbor])
            if len(component) > len(largest):
                largest = component
        return np.asarray(largest, dtype=np.int64)

    @staticmethod
    def _public_path_status(path: Mapping[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in path.items() if not key.startswith("_")}

    def _notify_docking_debug(
        self,
        *,
        phase: str,
        iteration: int,
        object_name: str,
        rgb: np.ndarray,
        depth: np.ndarray,
        intrinsics: np.ndarray,
        target: VisibleTarget,
        travel_distance_m: float,
        path_status: Mapping[str, Any] | None,
        avoidance_path_body: list[list[float]] | None = None,
        avoidance_waypoint_body: list[float] | None = None,
        avoidance_plan_reason: str | None = None,
    ) -> str | None:
        """Record and publish the latest state to an optional Viser observer."""

        payload = {
            "phase": str(phase),
            "iteration": int(iteration),
            "object_name": str(object_name),
            "rgb": rgb,
            "depth_m": depth,
            "intrinsics": intrinsics,
            "target": target,
            "travel_distance_m": float(travel_distance_m),
            "path_status": path_status,
            "avoidance_path_body_forward_left_m": avoidance_path_body,
            "avoidance_waypoint_body_forward_left_m": avoidance_waypoint_body,
            "avoidance_plan_reason": avoidance_plan_reason,
            "config": self.config,
            "ground_plane": dict(self.last_ground_plane_debug),
            "pose_xy_yaw": self._current_pose().tolist(),
        }
        self.last_docking_debug = payload
        callback = self._docking_debug_callback
        if callback is None:
            return None
        try:
            callback(payload)
        except Exception as exc:
            # Visualization must never change physical control behavior.
            return f"{type(exc).__name__}: {exc}"
        return None

    @staticmethod
    def _primitive_summary(result: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "primitive": result.get("primitive"),
            "success": bool(result.get("success", False)),
            "reason": str(result.get("reason", "unknown")),
            "metrics": dict(result.get("metrics", {})),
        }

    def _result(
        self,
        *,
        success: bool,
        reason: str,
        object_name: str,
        start_time: float,
        metrics: Mapping[str, Any],
        history: list[dict[str, Any]],
        stop_result: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        return {
            "primitive": "dock_to_visible_object",
            "success": bool(success),
            "reason": str(reason),
            "target": object_name,
            "elapsed_s": max(0.0, float(self._clock() - start_time)),
            "metrics": dict(metrics),
            "motion_history": history,
            "final_stop": None if stop_result is None else dict(stop_result),
        }

    def dock_to_visible_object(self, object_name: str) -> dict[str, Any]:
        """Dock to a visible semantic target chosen by the caller.

        The controller repeatedly segments the object and advances in bounded
        steps until its closest point reaches the docking distance.  A blocked
        direct corridor activates conservative static-obstacle A* planning;
        every short waypoint is followed by a stopped RGB-D replan.
        """

        if not isinstance(object_name, str) or not object_name.strip():
            raise ValueError("object_name must be a non-empty string")
        object_name = object_name.strip()
        self._occupancy_map = self._new_occupancy_map()
        self._virtual_pose_xy_yaw = np.zeros(3, dtype=np.float64)
        self._last_observed_pose_xy_yaw = None
        start_time = self._clock()
        history: list[dict[str, Any]] = []
        metrics: dict[str, Any] = {
            "docking_distance_m": self.config.docking_distance_m,
            "camera_to_robot_front_m": 0.0,
            "iterations": 0,
            "total_forward_command_m": 0.0,
            "total_lateral_command_m": 0.0,
            "total_planar_command_m": 0.0,
            "total_absolute_turn_rad": 0.0,
            "avoidance_active": False,
            "avoidance_replans": 0,
            "avoidance_guard_replans": 0,
            "target_temporarily_lost_iterations": 0,
        }
        success = False
        reason = "not_started"
        fatal_error: BaseException | None = None
        final_stop: Mapping[str, Any] | None = None
        target_world_xy: tuple[float, float] | None = None
        avoidance_side: int | None = None
        target_lost_iterations = 0

        entry_stop = self._motion.stop()
        history.append(self._primitive_summary(entry_stop))
        if not entry_stop.get("success", False):
            return self._result(
                success=False,
                reason=f"entry_stop_failed: {entry_stop.get('reason', 'unknown')}",
                object_name=object_name,
                start_time=start_time,
                metrics=metrics,
                history=history,
                stop_result=entry_stop,
            )

        try:
            for iteration in range(1, self.config.maximum_iterations + 1):
                metrics["iterations"] = iteration
                if self._clock() - start_time > self.config.timeout_s:
                    reason = "timeout"
                    break
                rgb, depth, intrinsics = self._camera_input()
                pose = self._current_pose()
                metrics["pose_xy_yaw"] = pose.tolist()
                target: VisibleTarget
                try:
                    target = self._detect_target(
                        object_name, rgb, depth, intrinsics
                    )
                except RuntimeError as exc:
                    target_loss_messages = (
                        "SAM3 found no instance",
                        "SAM3 mask has only",
                        "object mask has insufficient valid metric depth",
                    )
                    can_reacquire = target_world_xy is not None and any(
                        message in str(exc) for message in target_loss_messages
                    )
                    if not can_reacquire:
                        raise
                    target_lost_iterations += 1
                    metrics["target_temporarily_lost_iterations"] = int(
                        metrics["target_temporarily_lost_iterations"]
                    ) + 1
                    if target_lost_iterations > self.config.target_lost_max_iterations:
                        reason = "target_reacquisition_failed"
                        break
                    reacquired, reacquisition_reason = (
                        self._reacquire_target_with_fast_detector(
                            object_name,
                            rgb,
                            intrinsics,
                            history=history,
                            metrics=metrics,
                        )
                    )
                    metrics["last_reacquisition_reason"] = reacquisition_reason
                    if not reacquired:
                        reason = reacquisition_reason
                        break
                    # The base only rotated during search. Acquire a new frame
                    # and let SAM3 rebuild precise metric target geometry before
                    # any further translation or map planning.
                    continue
                target_lost_iterations = 0
                target_world_xy = self._target_world_xy(target, pose)
                metrics.update(target.metrics())
                metrics["target_world_xy_m"] = list(target_world_xy)
                metrics["ground_plane"] = dict(self.last_ground_plane_debug)

                if target.distance_m <= self.config.docking_distance_m:
                    self._notify_docking_debug(
                        phase="docked",
                        iteration=iteration,
                        object_name=object_name,
                        rgb=rgb,
                        depth=depth,
                        intrinsics=intrinsics,
                        target=target,
                        travel_distance_m=0.0,
                        path_status=None,
                    )
                    success = True
                    reason = "within_docking_distance"
                    break

                if (
                    not self.config.avoidance_enabled
                    and abs(target.bearing_rad)
                    > self.config.bearing_tolerance_rad
                ):
                    self._notify_docking_debug(
                        phase="turn_required",
                        iteration=iteration,
                        object_name=object_name,
                        rgb=rgb,
                        depth=depth,
                        intrinsics=intrinsics,
                        target=target,
                        travel_distance_m=(
                            target.distance_m - self.config.docking_distance_m
                        ),
                        path_status=None,
                    )
                    turn = float(
                        np.clip(
                            target.bearing_rad,
                            -self.config.maximum_turn_step_rad,
                            self.config.maximum_turn_step_rad,
                        )
                    )
                    total_turn = float(metrics["total_absolute_turn_rad"]) + abs(turn)
                    if total_turn > self.config.maximum_total_turn_rad:
                        reason = "turn_budget_exceeded"
                        break
                    motion = self._motion.turn_relative(
                        turn,
                        max_yaw_rad_s=self.config.turn_speed_rad_s,
                        timeout_s=self.config.turn_timeout_s,
                    )
                    history.append(self._primitive_summary(motion))
                    metrics["total_absolute_turn_rad"] = total_turn
                    if not motion.get("success", False):
                        reason = f"turn_failed: {motion.get('reason', 'unknown')}"
                        break
                    self._record_successful_motion(turn_rad=turn)
                    continue

                if target_world_xy is None:
                    reason = "target_anchor_unavailable"
                    break
                remaining = target.distance_m - self.config.docking_distance_m
                target_mask = target.mask
                path = self._stable_path_status(
                    depth,
                    intrinsics,
                    # Keep the old corridor result as a human-readable depth
                    # diagnostic; occupancy/A* below owns the motion decision.
                    bearing_rad=0.0,
                    travel_distance_m=self.config.planner_lookahead_m,
                    target_mask=target_mask,
                    minimum_path_limit_m=(
                        self.config.avoidance_detection_range_m
                        if self.config.avoidance_enabled
                        else None
                    ),
                )
                metrics["path_clearance"] = self._public_path_status(path)
                stable_depth = np.asarray(path.get("_stable_depth_m", depth))
                occupancy_update = self._integrate_occupancy(
                    stable_depth,
                    intrinsics,
                    pose_xy_yaw=pose,
                    excluded_mask=target_mask,
                )
                metrics["occupancy_update"] = occupancy_update
                self._notify_docking_debug(
                    phase="path_checked",
                    iteration=iteration,
                    object_name=object_name,
                    rgb=rgb,
                    depth=stable_depth,
                    intrinsics=intrinsics,
                    target=target,
                    travel_distance_m=remaining,
                    path_status=path,
                )
                planned_motion = False
                command_forward = 0.0
                command_left = 0.0
                precise_distance_tolerance_m: float | None = None
                for stale_key in (
                    "avoidance_command_waypoint_xy_m",
                    "avoidance_command_yaw_rad",
                    "avoidance_frontier_plan",
                    "planned_segment_clearance",
                    "precise_grid_step",
                    "command_distance_tolerance_m",
                ):
                    metrics.pop(stale_key, None)
                if not self.config.avoidance_enabled:
                    if not path["valid"] or not path["clear"]:
                        reason = str(path["reason"])
                        break
                    step = max(
                        self.config.minimum_forward_command_m,
                        min(self.config.maximum_forward_step_m, remaining),
                    )
                    command_forward = step
                else:
                    # The stopped RGB-D view above is the sole planning input.
                    # Always ask A* for a path to the semantic docking ring;
                    # the straight route naturally falls out as the cheapest
                    # path when it is free.  If the ring is still hidden, move
                    # only to an observed-free lateral frontier, then stop,
                    # rebuild the map from the new view, and plan again.
                    goal_plan = self._plan_avoidance(
                        pose_xy_yaw=pose,
                        target_world_xy=target_world_xy,
                    )
                    plan = goal_plan
                    using_frontier = False
                    metrics["avoidance_replans"] = int(
                        metrics["avoidance_replans"]
                    ) + 1
                    metrics["avoidance_goal_plan"] = goal_plan.metrics()
                    if not goal_plan.success:
                        frontier_plan = self._plan_avoidance_frontier(
                            pose_xy_yaw=pose,
                            target_world_xy=target_world_xy,
                            preferred_side=avoidance_side,
                        )
                        metrics["avoidance_frontier_plan"] = (
                            frontier_plan.metrics()
                        )
                        if not frontier_plan.success:
                            reason = "path_blocked"
                            break
                        plan = frontier_plan
                        using_frontier = True
                        metrics["frontier_replans"] = int(
                            metrics.get("frontier_replans", 0)
                        ) + 1
                        if avoidance_side is None:
                            avoidance_side = self._avoidance_path_side(
                                plan.path_xy,
                                start_xy=(float(pose[0]), float(pose[1])),
                                target_xy=target_world_xy,
                            )
                    metrics["avoidance_plan"] = plan.metrics()
                    metrics["avoidance_plan_kind"] = (
                        "frontier" if using_frontier else "docking_goal"
                    )
                    metrics["avoidance_side"] = avoidance_side
                    waypoint = self._occupancy_map.waypoint(
                        plan.path_xy, self.config.planner_lookahead_m
                    )
                    forward, left = self._waypoint_body_delta(waypoint, pose)
                    waypoint_distance = math.hypot(forward, left)
                    waypoint_bearing = math.atan2(left, forward)
                    if abs(left) > self.config.occupancy_resolution_m:
                        metrics["avoidance_active"] = True
                    metrics["avoidance_waypoint_xy_m"] = list(waypoint)
                    metrics["avoidance_waypoint_distance_m"] = waypoint_distance
                    metrics["avoidance_waypoint_bearing_rad"] = waypoint_bearing
                    avoidance_path_body = [
                        list(self._waypoint_body_delta(point, pose))
                        for point in plan.path_xy
                    ]
                    self._notify_docking_debug(
                        phase="avoidance_planned",
                        iteration=iteration,
                        object_name=object_name,
                        rgb=rgb,
                        depth=stable_depth,
                        intrinsics=intrinsics,
                        target=target,
                        travel_distance_m=remaining,
                        path_status=path,
                        avoidance_path_body=avoidance_path_body,
                        avoidance_waypoint_body=[forward, left],
                        avoidance_plan_reason=plan.reason,
                    )
                    if (
                        waypoint_distance < self.config.minimum_forward_command_m
                        and not using_frontier
                        and plan.reason == "already_on_docking_ring"
                    ):
                        # Grid quantization can place the current cell on the
                        # docking ring while precise SAM3 geometry is still a
                        # few centimetres outside it. Finish only that residual
                        # semantic displacement, then rebuild and replan.
                        scale = min(1.0, remaining / max(target.distance_m, 1e-9))
                        command_forward = target.closest_forward_m * scale
                        command_left = target.closest_left_m * scale
                        step = math.hypot(command_forward, command_left)
                        if step <= 1e-4:
                            # Sub-millimetre residuals are below ZED depth and
                            # grid precision; treating them as failure causes
                            # an otherwise completed docking run to stall.
                            success = True
                            reason = "within_docking_distance"
                            break
                        command_world_x, command_world_y = self._local_to_world(
                            command_forward, command_left, pose
                        )
                        command_waypoint = (
                            float(command_world_x),
                            float(command_world_y),
                        )
                        metrics["avoidance_command_waypoint_xy_m"] = list(
                            command_waypoint
                        )
                        segment_path = self._occupancy_map.segment_status(
                            (float(pose[0]), float(pose[1])),
                            command_waypoint,
                            allow_start_in_inflated=True,
                        )
                        metrics["planned_segment_clearance"] = dict(segment_path)
                        if not segment_path["clear"]:
                            reason = "path_blocked"
                            break
                        planned_motion = True
                    elif forward < -self.config.minimum_forward_command_m:
                        # Do not translate backwards without rear perception.
                        # Turn toward a behind-the-chassis route, stop, and map
                        # it from the now-forward-facing ZED view first.
                        turn = float(
                            np.clip(
                                waypoint_bearing,
                                -self.config.maximum_turn_step_rad,
                                self.config.maximum_turn_step_rad,
                            )
                        )
                        total_turn = float(metrics["total_absolute_turn_rad"]) + abs(
                            turn
                        )
                        if total_turn > self.config.maximum_total_turn_rad:
                            reason = "turn_budget_exceeded"
                            break
                        motion = self._motion.turn_relative(
                            turn,
                            max_yaw_rad_s=self.config.turn_speed_rad_s,
                            timeout_s=self.config.turn_timeout_s,
                        )
                        history.append(self._primitive_summary(motion))
                        metrics["total_absolute_turn_rad"] = total_turn
                        if not motion.get("success", False):
                            reason = (
                                f"turn_failed: {motion.get('reason', 'unknown')}"
                            )
                            break
                        self._record_successful_motion(turn_rad=turn)
                        continue
                    else:
                        # A* selects a 2-D line segment. Execution rotates the
                        # chassis/ZED to that segment and then drives straight,
                        # avoiding independent forward/lateral saturation that
                        # would otherwise change the planned direction.
                        command_forward = max(0.0, forward)
                        command_left = left
                        step = math.hypot(command_forward, command_left)
                        command_world_x, command_world_y = self._local_to_world(
                            command_forward, command_left, pose
                        )
                        command_waypoint = (
                            float(command_world_x),
                            float(command_world_y),
                        )
                        metrics["avoidance_command_waypoint_xy_m"] = list(
                            command_waypoint
                        )
                        segment_path = self._occupancy_map.segment_status(
                            (float(pose[0]), float(pose[1])),
                            command_waypoint,
                            allow_start_in_inflated=True,
                        )
                        metrics["planned_segment_clearance"] = dict(segment_path)
                        if not segment_path["clear"]:
                            reason = "path_blocked"
                            break
                        if (
                            step < self.config.minimum_forward_command_m
                        ):
                            precise_distance_tolerance_m = (
                                self._precise_grid_step_tolerance(step)
                            )
                            metrics["precise_grid_step"] = True
                            metrics["command_distance_tolerance_m"] = (
                                precise_distance_tolerance_m
                            )
                        planned_motion = True

                total_planar = float(metrics["total_planar_command_m"]) + step
                if total_planar > self.config.maximum_total_forward_m:
                    reason = "forward_budget_exceeded"
                    break
                if planned_motion:
                    path_bearing = math.atan2(command_left, command_forward)
                    metrics["avoidance_execution_mode"] = "turn_then_straight"
                    metrics["avoidance_command_yaw_rad"] = path_bearing
                    turn_failure = self._turn_to_path_heading(
                        path_bearing,
                        history=history,
                        metrics=metrics,
                    )
                    if turn_failure is not None:
                        reason = turn_failure
                        break
                    command_forward = step
                    command_left = 0.0
                    drive_options: dict[str, Any] = {
                        "max_speed_mps": self.config.forward_speed_mps,
                        "pose_callback": self._notify_actual_pose,
                    }
                    if precise_distance_tolerance_m is not None:
                        drive_options["distance_tolerance_m"] = (
                            precise_distance_tolerance_m
                        )
                    motion = self._motion.drive_straight(step, **drive_options)
                else:
                    motion = self._motion.drive_straight(
                        step,
                        max_speed_mps=self.config.forward_speed_mps,
                        pose_callback=self._notify_actual_pose,
                    )
                history.append(self._primitive_summary(motion))
                metrics["total_forward_command_m"] = float(
                    metrics["total_forward_command_m"]
                ) + abs(command_forward)
                metrics["total_lateral_command_m"] = float(
                    metrics["total_lateral_command_m"]
                ) + abs(command_left)
                metrics["total_planar_command_m"] = total_planar
                if not motion.get("success", False):
                    motion_reason = str(motion.get("reason", "unknown"))
                    if planned_motion and motion_reason == "obstacle_too_close":
                        metrics["avoidance_guard_replans"] = int(
                            metrics["avoidance_guard_replans"]
                        ) + 1
                        metrics["last_motion_guard_block"] = {
                            "reason": motion_reason,
                            "metrics": dict(motion.get("metrics", {})),
                        }
                        # drive_straight has already stopped. Reobserve and let
                        # the 2-D map choose another turn/straight segment.
                        continue
                    reason = f"drive_failed: {motion_reason}"
                    break
                self._record_successful_motion(
                    forward_m=command_forward,
                )
            else:
                reason = "iteration_budget_exceeded"
        except Exception as exc:
            reason = f"perception_failed: {type(exc).__name__}: {exc}"
        except BaseException as exc:
            fatal_error = exc
            reason = type(exc).__name__
        finally:
            final_stop = self._motion.stop()
            history.append(self._primitive_summary(final_stop))
            if not final_stop.get("success", False):
                success = False
                reason = (
                    f"{reason}; final_stop_failed: "
                    f"{final_stop.get('reason', 'unknown')}"
                )

        if fatal_error is not None:
            raise fatal_error
        return self._result(
            success=success,
            reason=reason,
            object_name=object_name,
            start_time=start_time,
            metrics=metrics,
            history=history,
            stop_result=final_stop,
        )
