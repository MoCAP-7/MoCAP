"""Nav2-backed implementation of ``dock_to_visible_object``.

SAM3 supplies one semantic metric target and verifies the final distance. Nav2
owns path planning, obstacle avoidance, rolling replanning, control, and
recoveries; this module deliberately contains no local navigation rules.

The only goal-selection rules here are the bearing the robot docks from and
how far from the object the goal sits. Without a readiness prior it approaches
straight in along the arrival bearing. With one
(``docs/readiness_prior_plan.md``), the bearing from the object toward where the
human stood in the passive video replaces it, with fixed offset retries and,
optionally, the arrival bearing as the final fallback. A caller may also name
the bearing outright. On every bearing the inner goal distance is tried first
and configured farther fallbacks after it, each optionally checked against the
Nav2 planner before the base drives; a fallback the base already stands at or
inside of, on a bearing it is aligned with, is taken without a goal. A goal
Nav2 gives up on without progress ends with a fresh observation: a camera on
the bearing inside that goal's arrival boundary docks where it stands. An
operator stop is honoured before the first goal and between goals as well as
during one. Every goal faces the object, optionally turned by a fixed offset
so the object ends up in front of one arm rather than between the shoulders.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
import math
import threading
import time
from typing import Any

from .nav2_client import Nav2Client
from .visible_object_navigation import VisibleObjectDockingController


_PERCEPTION_CONFIG_KEYS = {
    "docking_distance_m",
    "target_min_depth_m",
    "target_max_depth_m",
    "target_min_depth_pixels",
    "target_bbox_low_quantile",
    "target_bbox_high_quantile",
    "ground_camera_height_m",
    "ground_down_camera_xyz",
    "require_dynamic_ground_plane",
    "ground_plane_max_age_s",
    "ground_plane_max_height_step_m",
    "ground_plane_max_tilt_step_deg",
    "ground_plane_settle_s",
    "ground_plane_max_height_error_m",
    "ground_plane_max_tilt_error_deg",
}

_BLOCKED_RECOVERY_REASON = (
    "nav2_blocked_near_obstacle: the collision safety zone around the robot is "
    "occupied, so Nav2 cannot move from the current base pose. Do not retry "
    "dock_to_visible_object from the same pose. If the rear path is known to be "
    "clear, first call drive_straight with a short negative distance to back "
    "away, re-observe, and then retry dock_to_visible_object."
)

_STALLED_RECOVERY_REASON = (
    "nav2_stalled_no_progress: Nav2 made no meaningful base progress within "
    "the configured limit, so the goal was canceled and the robot was stopped. "
    "Do not retry dock_to_visible_object unchanged. Re-observe first. If the "
    "target is still visible and roughly ahead, and the forward path is clear, "
    "call drive_straight with a short positive distance of 0.10 m, "
    "re-observe, and then retry docking. Otherwise turn or choose another safe "
    "adjustment based on the new observation."
)


# A bearing on which no goal distance passed the Nav2 plan check, so the
# base never drove; recorded as the attempt's reason and, when it was the
# last attempt, as the primitive's.
_NO_PLANNABLE_GOAL_REASON = "no_plannable_goal_on_bearing"

# A fallback goal distance the base already stands at or inside of, on the
# attempt's bearing, so no goal is sent; the attempt's reason and, once the
# arrival check passes, the primitive's.
_DOCKED_AT_FALLBACK_REASON = "docked_at_fallback_distance"
# Nav2's own reason for a goal it cancelled without progress.
_NAV2_STALL_REASON = "stalled_no_progress"
# A stall ended the goal, and a fresh observation found the camera on the
# attempt's bearing inside that goal's arrival boundary: docked there.
_DOCKED_AFTER_STALL_REASON = "docked_after_stall"

# The operator stopped the base: the attempt's reason whether the stop cut a
# goal short or landed between goals, when it is recorded without one.
_OPERATOR_STOP_REASON = "operator_stop"

# Plan-check outcomes that say the goal itself is unreachable. Any other
# failure (planner unavailable, timeouts) says nothing about the goal, so the
# goal is sent and Nav2 decides, as it does without a plan check.
_UNREACHABLE_PLAN_REASONS = frozenset({"no_path", "goal_rejected"})


class _Nav2StartBlocked(RuntimeError):
    """Internal control-flow signal for an occupied Nav2 start footprint."""


# Bearing sources whose goal is the pre-prior arrival goal, computed with the
# literal expressions the controller has always used.
_ARRIVAL_BEARING_SOURCES = frozenset({"arrival", "arrival_fallback"})

_DEFAULT_MAX_BEARING_ERROR_DEG = 45.0
_DEFAULT_RETRY_BEARING_OFFSETS_DEG = (45.0, -45.0)


def _wrap_angle(angle_rad: float) -> float:
    """Wrap an angle into (-pi, pi]."""

    return math.atan2(math.sin(angle_rad), math.cos(angle_rad))


def _arm_for_side(left_m: float) -> str:
    """The arm on the side the target sits on, measured from the base centre."""

    return "left" if left_m > 0.0 else "right"


def _xy_or_none(point: Any) -> list[float] | None:
    """A 2-vector as ``[x, y]`` floats for the metrics, or None when absent."""

    if point is None:
        return None
    return [float(point[0]), float(point[1])]


def _finite_or_none(value: Any) -> float | None:
    """A float for the metrics, or None when absent or not finite (NaN, inf).

    The Web UI serializes metrics with ``allow_nan=False``, so a NaN here
    would take the whole result down with it.
    """

    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


@dataclass(frozen=True)
class _PriorOutcome:
    """What the docking flow needs from one readiness-prior query."""

    # Direction from the object toward the human's docked stance in odom;
    # None when the prior is absent, returned nothing, or failed.
    bearing_rad: float | None
    suggested_arm: str | None
    # Recorded verbatim as ``metrics["readiness_prior"]``.
    info: dict[str, Any]

    @property
    def used(self) -> bool:
        return self.bearing_rad is not None


@dataclass(frozen=True)
class Nav2VisibleObjectDockingConfig:
    """Only semantic-goal and Nav2-boundary parameters owned by the primitive."""

    docking_distance_m: float = 0.60
    nav2_goal_distance_m: float | None = None
    # Farther goal distances along the same bearing, tried in order when the
    # inner goal cannot be planned or navigated, so the base still gets as
    # close as the scene allows. Empty keeps the single inner goal.
    nav2_goal_distance_fallbacks_m: tuple[float, ...] = ()
    # Ask the Nav2 planner for a path before sending each goal and skip
    # goals it reports unreachable instead of driving toward them. A planner
    # that is unavailable or times out decides nothing: the goal is sent.
    plan_check_before_goal: bool = False
    nav2_planner_action_name: str = "compute_path_to_pose"
    # With an explicit approach bearing, already within the docking distance
    # counts as docked only when the arrival bearing is within this angle.
    explicit_bearing_alignment_tolerance_deg: float = 22.5
    robot_radius_m: float = 0.60
    # Convex polygon in metres from the swerve centre; when present the Nav2
    # renderer prefers it to the circle. Validated with the same hull the
    # renderer uses so a bad shape fails at agent start, not at Nav2 launch.
    robot_footprint_xy: tuple[tuple[float, float], ...] | None = None
    # Body outline for the structure collision monitor: the arms envelope
    # above only judges points measured at arm height, so everything else is
    # judged against this. Owned by the Nav2 renderer; the primitive itself
    # never uses it, but it is validated here so a bad shape fails at agent
    # start rather than at Nav2 launch.
    robot_structure_footprint_xy: tuple[tuple[float, float], ...] | None = None
    base_to_camera_forward_m: float = 0.2143
    base_to_camera_left_m: float = 0.0603
    nav2_frame_id: str = "odom"
    nav2_action_name: str = "navigate_to_pose"
    nav2_stop_service_name: str = "/yor_base_bridge/stop"
    nav2_halt_service_name: str = "/yor_base_bridge/halt"
    nav2_clear_stop_service_name: str = "/yor_base_bridge/clear_stop"
    nav2_server_timeout_s: float = 10.0
    timeout_s: float = 300.0
    final_distance_tolerance_m: float = 0.08
    # Lateral offset from the base centre past which the target's own side
    # decides which arm should reach for it. Inside the margin the object is
    # in front of the body rather than either shoulder, and the human's grasp
    # hand breaks the tie.
    arm_side_margin_m: float = 0.05
    # Facing the object dead ahead puts it between the shoulders, where a few
    # degrees of final heading decide which arm can reach it. A non-zero
    # offset turns the goal heading away from the chosen arm by this angle so
    # the object sits in front of that arm instead: the base turns right for
    # the left arm and left for the right arm. Zero keeps the heading that
    # faces the object.
    arm_side_heading_offset_deg: float = 0.0
    # The arm the offset favours when the readiness prior names no hand.
    arm_side_heading_default_arm: str = "left"
    progress_interval_s: float = 1.0
    no_progress_timeout_s: float = 10.0
    minimum_progress_translation_m: float = 0.03
    minimum_progress_yaw_rad: float = 0.05
    behavior_tree: str = ""

    @classmethod
    def from_mapping(
        cls, values: Mapping[str, Any] | None
    ) -> "Nav2VisibleObjectDockingConfig":
        source = dict(values or {})
        source.pop("backend", None)
        # The readiness-prior block is validated by ReadinessPriorConfig at
        # config load and consumed by the launch wiring, not by this config.
        source.pop("readiness_prior", None)
        known = set(cls.__dataclass_fields__)
        unknown = sorted(set(source) - known - _PERCEPTION_CONFIG_KEYS)
        if unknown:
            raise ValueError(f"unknown Nav2 docking settings: {unknown}")
        payload = {key: source[key] for key in known if key in source}
        fallbacks = payload.get("nav2_goal_distance_fallbacks_m")
        if isinstance(fallbacks, list):
            payload["nav2_goal_distance_fallbacks_m"] = tuple(fallbacks)
        config = cls(**payload)
        config._validate()
        return config

    def _validate(self) -> None:
        numeric = (
            self.docking_distance_m,
            self.robot_radius_m,
            self.base_to_camera_forward_m,
            self.base_to_camera_left_m,
            self.nav2_server_timeout_s,
            self.timeout_s,
            self.final_distance_tolerance_m,
            self.progress_interval_s,
            self.no_progress_timeout_s,
            self.minimum_progress_translation_m,
            self.minimum_progress_yaw_rad,
        )
        if not all(math.isfinite(float(value)) for value in numeric):
            raise ValueError("Nav2 docking settings must be finite")
        if not 0.1 <= self.docking_distance_m <= 1.5:
            raise ValueError("docking_distance_m must be in [0.1, 1.5]")
        if self.nav2_goal_distance_m is not None:
            if not math.isfinite(float(self.nav2_goal_distance_m)):
                raise ValueError("nav2_goal_distance_m must be finite")
            if not 0.1 <= self.nav2_goal_distance_m <= self.docking_distance_m:
                raise ValueError(
                    "nav2_goal_distance_m must be in [0.1, docking_distance_m]"
                )
        if not isinstance(self.nav2_goal_distance_fallbacks_m, tuple):
            raise ValueError("nav2_goal_distance_fallbacks_m must be a list")
        previous = self.effective_nav2_goal_distance_m
        for value in self.nav2_goal_distance_fallbacks_m:
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise ValueError(
                    "nav2_goal_distance_fallbacks_m entries must be finite numbers"
                )
            if not previous < float(value) <= 3.0:
                raise ValueError(
                    "nav2_goal_distance_fallbacks_m must ascend strictly from "
                    "above nav2_goal_distance_m to at most 3.0"
                )
            previous = float(value)
        if not isinstance(self.plan_check_before_goal, bool):
            raise ValueError("plan_check_before_goal must be a boolean")
        if not math.isfinite(
            float(self.explicit_bearing_alignment_tolerance_deg)
        ) or not 0.0 < self.explicit_bearing_alignment_tolerance_deg <= 180.0:
            raise ValueError(
                "explicit_bearing_alignment_tolerance_deg must be in (0, 180]"
            )
        if not 0.1 <= self.robot_radius_m <= 1.5:
            raise ValueError("robot_radius_m must be in [0.1, 1.5]")
        if self.robot_footprint_xy is not None:
            from ..nav2_params import polygon_footprint

            hull = polygon_footprint(self.robot_footprint_xy)
            if max(abs(v) for point in hull for v in point) > 1.5:
                raise ValueError("robot_footprint_xy vertices must lie within 1.5 m")
        if self.robot_structure_footprint_xy is not None:
            from ..nav2_params import polygon_footprint

            structure = polygon_footprint(self.robot_structure_footprint_xy)
            if max(abs(v) for point in structure for v in point) > 1.5:
                raise ValueError(
                    "robot_structure_footprint_xy vertices must lie within 1.5 m"
                )
        if not 0.0 <= self.base_to_camera_forward_m <= 1.0:
            raise ValueError("base_to_camera_forward_m must be in [0, 1]")
        if not -0.5 <= self.base_to_camera_left_m <= 0.5:
            raise ValueError("base_to_camera_left_m must be in [-0.5, 0.5]")
        if self.nav2_server_timeout_s <= 0.0 or not 1.0 <= self.timeout_s <= 600.0:
            raise ValueError("invalid Nav2 docking timeouts")
        if not 0.0 <= self.final_distance_tolerance_m <= 0.5:
            raise ValueError("final_distance_tolerance_m must be in [0, 0.5]")
        if not 0.0 <= self.arm_side_margin_m <= 0.5:
            raise ValueError("arm_side_margin_m must be in [0, 0.5]")
        if (
            isinstance(self.arm_side_heading_offset_deg, bool)
            or not isinstance(self.arm_side_heading_offset_deg, (int, float))
            or not math.isfinite(float(self.arm_side_heading_offset_deg))
            or not 0.0 <= float(self.arm_side_heading_offset_deg) <= 45.0
        ):
            raise ValueError("arm_side_heading_offset_deg must be in [0, 45]")
        if self.arm_side_heading_default_arm not in ("left", "right"):
            raise ValueError(
                "arm_side_heading_default_arm must be 'left' or 'right'"
            )
        if not 0.1 <= self.progress_interval_s <= 30.0:
            raise ValueError("progress_interval_s must be in [0.1, 30]")
        if not 2.0 <= self.no_progress_timeout_s <= self.timeout_s:
            raise ValueError(
                "no_progress_timeout_s must be in [2, timeout_s]"
            )
        if not 0.005 <= self.minimum_progress_translation_m <= 0.25:
            raise ValueError(
                "minimum_progress_translation_m must be in [0.005, 0.25]"
            )
        if not 0.01 <= self.minimum_progress_yaw_rad <= math.pi:
            raise ValueError("minimum_progress_yaw_rad must be in [0.01, pi]")
        if not all(
            (
                self.nav2_frame_id,
                self.nav2_action_name,
                self.nav2_planner_action_name,
                self.nav2_stop_service_name,
                self.nav2_halt_service_name,
                self.nav2_clear_stop_service_name,
            )
        ):
            raise ValueError("Nav2 frame, action, and service names must be non-empty")

    @property
    def effective_nav2_goal_distance_m(self) -> float:
        """Return the inner Nav2 target, defaulting to the success boundary."""

        if self.nav2_goal_distance_m is None:
            return self.docking_distance_m
        return float(self.nav2_goal_distance_m)

    @property
    def nav2_goal_distance_schedule_m(self) -> tuple[float, ...]:
        """The inner Nav2 target followed by its farther fallbacks, in order."""

        return (
            self.effective_nav2_goal_distance_m,
            *(float(value) for value in self.nav2_goal_distance_fallbacks_m),
        )


class Nav2VisibleObjectDockingController:
    """Convert one SAM3 observation into Nav2 goals and verify arrival.

    One observation yields one goal on the arrival bearing, or, with a
    readiness prior, a short sequence of goals on the prior bearing, its retry
    offsets, and finally the arrival bearing; a caller may instead name the
    single bearing to dock from. Each bearing is tried at the inner goal
    distance and then at the configured farther fallbacks.
    """

    def __init__(
        self,
        env: Any,
        *,
        docking_config: Mapping[str, Any] | None = None,
        segment_client_factory: Callable[[], Callable[..., Any]] | None = None,
        detector_factory: Callable[[Any], Any] | None = None,
        nav2_client_factory: Callable[[Nav2VisibleObjectDockingConfig], Any]
        | None = None,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
        clock: Callable[[], float] = time.monotonic,
        readiness_prior: Any | None = None,
    ) -> None:
        del detector_factory  # Nav2 does not use the legacy YOLOE yaw scan.
        source = dict(docking_config or {})
        self.config = Nav2VisibleObjectDockingConfig.from_mapping(source)
        perception_settings = {
            key: source[key] for key in _PERCEPTION_CONFIG_KEYS if key in source
        }
        self._perception = VisibleObjectDockingController(
            env,
            docking_config=perception_settings,
            segment_client_factory=segment_client_factory,
        )
        self._env = env
        self._motion = env.controller
        self._clock = clock
        self._nav2_client_factory = nav2_client_factory or self._default_client
        self._nav2_client: Any | None = None
        self._progress_callback = progress_callback
        # Duck-typed ``yor_agent.robot.readiness_prior.ReadinessPrior``: only
        # ``estimate(...)``, ``last_reason`` and ``config`` are read, so tests
        # and ablations can substitute any object with that shape.
        self._readiness_prior = readiness_prior

    def _default_client(self, config: Nav2VisibleObjectDockingConfig) -> Nav2Client:
        return Nav2Client(
            action_name=config.nav2_action_name,
            stop_service_name=config.nav2_stop_service_name,
            halt_service_name=config.nav2_halt_service_name,
            clear_stop_service_name=config.nav2_clear_stop_service_name,
            planner_action_name=config.nav2_planner_action_name,
        )

    def _client(self) -> Any:
        if self._nav2_client is None:
            self._nav2_client = self._nav2_client_factory(self.config)
        return self._nav2_client

    def _record_suggested_arm(
        self,
        metrics: dict[str, Any],
        target: Any,
        prior_arm: str | None,
    ) -> None:
        """Name the arm to reach with, from the side the target ended up on.

        Which arm can reach is decided by where the object sits relative to
        the shoulders, and the shoulders are offset to either side of the base
        centre. The human's grasp hand only says which hand a differently
        proportioned body used from a pose that is not exactly this one, so it
        breaks the tie when the object is in front of the body and decides
        nothing when it is clearly to one side. Following it regardless once
        cost three turns: the arm on the far side was reachable only after
        shifting the base sideways, a motion the forward-facing camera cannot
        clear, while the near arm reached from the docked pose.
        """

        left_m = float(target.closest_left_m) + self.config.base_to_camera_left_m
        if abs(left_m) >= self.config.arm_side_margin_m:
            arm, source = _arm_for_side(left_m), "target_side"
        elif prior_arm in ("left", "right"):
            arm, source = prior_arm, "human_hand"
        else:
            arm, source = _arm_for_side(left_m), "target_side"
        metrics["suggested_arm"] = arm
        metrics["suggested_arm_source"] = source
        metrics["target_left_of_base_m"] = left_m
        if prior_arm in ("left", "right"):
            metrics["human_hand_arm"] = prior_arm

    def _base_goal_from_camera_goal(
        self, camera_x: float, camera_y: float, yaw: float
    ) -> tuple[float, float, float]:
        """Return the swerve-center goal for a desired ZED-center pose."""

        cosine = math.cos(yaw)
        sine = math.sin(yaw)
        forward = self.config.base_to_camera_forward_m
        left = self.config.base_to_camera_left_m
        return (
            camera_x - (cosine * forward - sine * left),
            camera_y - (sine * forward + cosine * left),
            yaw,
        )

    def _prior_bearing_rules(self) -> tuple[float, tuple[float, ...], bool]:
        """Return (max bearing error deg, retry offsets deg, arrival fallback)
        from the prior."""

        prior_config = getattr(self._readiness_prior, "config", None)
        max_bearing_error_deg = float(
            getattr(
                prior_config,
                "max_bearing_error_deg",
                _DEFAULT_MAX_BEARING_ERROR_DEG,
            )
        )
        offsets = getattr(
            prior_config,
            "retry_bearing_offsets_deg",
            _DEFAULT_RETRY_BEARING_OFFSETS_DEG,
        )
        fallback_to_arrival = bool(
            getattr(prior_config, "fallback_to_arrival_bearing", True)
        )
        return (
            max_bearing_error_deg,
            tuple(float(offset) for offset in offsets),
            fallback_to_arrival,
        )

    def _query_readiness_prior(
        self,
        object_name: str,
        rgb: Any,
        depth: Any,
        intrinsics: Any,
        pose_xy_yaw: Any,
        *,
        target_xy: tuple[float, float],
        arrival_bearing_rad: float,
    ) -> _PriorOutcome:
        """Ask the passive-video prior where the human stood; never raise.

        The prior is best-effort: a missing prior, no estimate, or any
        exception leaves docking on the arrival bearing and records why.
        """

        prior = self._readiness_prior
        if prior is None:
            return _PriorOutcome(None, None, {"used": False, "reason": "not_configured"})
        try:
            down, planar_forward, planar_left = self._perception._ground_axes()
            # The camera height the perception controller resolved for this
            # observation (ZED floor plane when fresh, else the config value);
            # the vggt method needs it to put the human's head above the floor.
            camera_height_m = float(
                self._perception._active_ground_camera_height_m
            )
            estimate = prior.estimate(
                object_name,
                rgb,
                depth,
                intrinsics,
                pose_xy_yaw,
                down=down,
                planar_forward=planar_forward,
                planar_left=planar_left,
                target_xy=target_xy,
                camera_height_m=camera_height_m,
            )
            if estimate is None:
                reason = getattr(prior, "last_reason", None) or "no_estimate"
                return _PriorOutcome(
                    None,
                    None,
                    {
                        "used": False,
                        "arrival_bearing_deg": math.degrees(arrival_bearing_rad),
                        "reason": str(reason),
                    },
                )
            bearing_rad = float(estimate.bearing_rad)
            if not math.isfinite(bearing_rad):
                return _PriorOutcome(
                    None,
                    None,
                    {
                        "used": False,
                        "arrival_bearing_deg": math.degrees(arrival_bearing_rad),
                        "reason": "invalid_bearing",
                    },
                )
            suggested_arm = getattr(estimate, "suggested_arm", None)
            # pnp fills inliers/reprojection/focal and leaves the vggt-only
            # fields at None; vggt does the reverse. Both shapes are recorded.
            inliers = getattr(estimate, "inliers", None)
            reprojection_error_px = getattr(estimate, "reprojection_error_px", None)
            focal_px = getattr(estimate, "focal_px", None)
            human_height_m = getattr(estimate, "human_height_m", None)
            scale_iqr_ratio = getattr(estimate, "scale_iqr_ratio", None)
            frames = getattr(estimate, "frames", None)
            info = {
                "used": True,
                "method": getattr(estimate, "method", None),
                "bearing_deg": math.degrees(bearing_rad),
                "heading_deg": math.degrees(float(estimate.heading_rad)),
                "arrival_bearing_deg": math.degrees(arrival_bearing_rad),
                "bearing_error_deg": math.degrees(
                    _wrap_angle(arrival_bearing_rad - bearing_rad)
                ),
                "inliers": None if inliers is None else int(inliers),
                "reprojection_error_px": (
                    None
                    if reprojection_error_px is None
                    else float(reprojection_error_px)
                ),
                "focal_px": None if focal_px is None else float(focal_px),
                # NaN when the event carries no frame time (older prior builds).
                "frame_time_s": _finite_or_none(
                    getattr(estimate, "frame_time_s", None)
                ),
                "event_object": getattr(estimate, "event_object", None),
                "hand": getattr(estimate, "hand", None),
                "suggested_arm": suggested_arm,
                "human_xy": _xy_or_none(getattr(estimate, "human_xy", None)),
                "human_height_m": (
                    None if human_height_m is None else float(human_height_m)
                ),
                "scale_iqr_ratio": (
                    None if scale_iqr_ratio is None else float(scale_iqr_ratio)
                ),
                "stance_xy": _xy_or_none(getattr(estimate, "stance_xy", None)),
                "frames": [] if frames is None else [str(label) for label in frames],
                "reason": None,
            }
        except Exception as exc:  # the prior must never break docking
            return _PriorOutcome(
                None,
                None,
                {
                    "used": False,
                    "reason": f"exception:{type(exc).__name__}:{exc}",
                },
            )
        return _PriorOutcome(
            bearing_rad, str(suggested_arm) if suggested_arm else None, info
        )

    def _check_nav2_start_clearance(self, metrics: dict[str, Any]) -> None:
        """Raise ``_Nav2StartBlocked`` when the base cannot leave its pose."""

        start_clearance = getattr(
            self._motion, "nav2_start_clearance_status", None
        )
        if not callable(start_clearance):
            return
        try:
            clearance_status = start_clearance(
                radius_m=self.config.robot_radius_m,
            )
        except Exception as exc:  # Nav2 remains the fail-safe fallback.
            clearance_status = {
                "available": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
        metrics["nav2_start_clearance"] = dict(clearance_status)
        if bool(clearance_status.get("blocked", False)):
            metrics["recommended_recovery"] = {
                "action": "back_away_then_retry",
                "primitive": "drive_straight",
                "distance_sign": "negative",
                "requires_known_clear_rear_path": True,
            }
            raise _Nav2StartBlocked(_BLOCKED_RECOVERY_REASON)

    def dock_to_visible_object(
        self, object_name: str, *, approach_bearing_deg: float | None = None
    ) -> dict[str, Any]:
        """Dock along the prior, arrival, or an explicitly requested bearing.

        ``approach_bearing_deg`` is the direction from the object to the
        docking goal in the Nav2 frame, the quantity the metrics report as
        ``arrival_bearing_rad`` and ``reference_bearing_rad``. When given, the
        readiness prior is not consulted and that bearing is the only attempt.
        """

        if not isinstance(object_name, str) or not object_name.strip():
            raise ValueError("object_name must be a non-empty string")
        object_name = object_name.strip()
        explicit_bearing: float | None = None
        if approach_bearing_deg is not None:
            if isinstance(approach_bearing_deg, bool) or not math.isfinite(
                float(approach_bearing_deg)
            ):
                raise ValueError("approach_bearing_deg must be a finite angle")
            explicit_bearing = _wrap_angle(math.radians(float(approach_bearing_deg)))
        started = self._clock()
        navigation_id = f"dock-{time.time_ns()}"

        def publish_progress(
            state: str,
            *,
            terminal: bool = False,
            **payload: Any,
        ) -> None:
            if self._progress_callback is None:
                return
            event = {
                "primitive": "dock_to_visible_object",
                "target": object_name,
                "navigation_id": navigation_id,
                "state": state,
                "terminal": bool(terminal),
                "elapsed_s": max(0.0, self._clock() - started),
                "timestamp": time.time(),
                **payload,
            }
            try:
                self._progress_callback(event)
            except Exception:
                # Observability must never influence robot control.
                pass

        def publish_nav2_progress(event: dict[str, Any]) -> None:
            nav2_event = dict(event)
            nav2_terminal = bool(nav2_event.pop("terminal", False))
            state = str(nav2_event.pop("state", "navigating"))
            nav2_event.pop("elapsed_s", None)
            publish_progress(
                f"nav2_{state}" if nav2_terminal else state,
                nav2_terminal=nav2_terminal,
                **nav2_event,
            )

        publish_progress("detecting_target")
        history: list[dict[str, Any]] = []
        metrics: dict[str, Any] = {
            "backend": "nav2",
            "docking_distance_m": self.config.docking_distance_m,
            "nav2_goal_distance_m": (
                self.config.effective_nav2_goal_distance_m
            ),
            "robot_radius_m": self.config.robot_radius_m,
            "camera_to_robot_front_m": 0.0,
            "base_to_camera_forward_left_m": [
                self.config.base_to_camera_forward_m,
                self.config.base_to_camera_left_m,
            ],
        }
        entry_stop = self._motion.stop()
        history.append(self._summary(entry_stop))
        if not entry_stop.get("success", False):
            reason = (
                f"entry_stop_failed: {entry_stop.get('reason', 'unknown')}"
            )
            publish_progress(
                "failed",
                terminal=True,
                success=False,
                reason=reason,
            )
            return self._result(
                False,
                reason,
                object_name,
                started,
                metrics,
                history,
                entry_stop,
            )

        # The operator's Stop, whenever it lands during this call. Before the
        # Nav2 client exists it is remembered here alone and honoured at the
        # next goal; once the client exists it also cancels the active goal
        # and latches the bridge, as the client's own flag alone would.
        operator_stop = threading.Event()

        def on_operator_stop() -> None:
            operator_stop.set()
            nav2_client = self._nav2_client
            if nav2_client is not None:
                nav2_client.cancel_and_stop()

        register = getattr(self._env, "register_motion_stop_callback", None)
        if callable(register):
            register(on_operator_stop)

        success = False
        reason = "not_started"
        fatal_error: BaseException | None = None
        final_stop: Mapping[str, Any] | None = None
        try:
            rgb, depth, intrinsics = self._perception._camera_input()
            pose = self._perception._current_pose()
            target = self._perception._detect_target(
                object_name, rgb, depth, intrinsics
            )
            metrics.update(target.metrics())
            metrics["initial_pose_xy_yaw"] = pose.tolist()
            metrics["ground_plane"] = dict(
                self._perception.last_ground_plane_debug
            )
            goal_distance_m = self.config.effective_nav2_goal_distance_m
            target_x, target_y = self._perception._target_world_xy(target, pose)
            dx = target_x - float(pose[0])
            dy = target_y - float(pose[1])
            distance = math.hypot(dx, dy)
            if distance <= 1e-6:
                raise RuntimeError("semantic target produced a degenerate goal")
            # Direction from the object toward the robot: without a prior the
            # robot docks straight in along it.
            arrival_bearing = math.atan2(
                float(pose[1]) - target_y, float(pose[0]) - target_x
            )
            metrics["target_world_xy_m"] = [target_x, target_y]
            metrics["arrival_bearing_rad"] = arrival_bearing
            if explicit_bearing is None:
                prior = self._query_readiness_prior(
                    object_name,
                    rgb,
                    depth,
                    intrinsics,
                    pose,
                    target_xy=(target_x, target_y),
                    arrival_bearing_rad=arrival_bearing,
                )
            else:
                # A requested approach direction replaces the prior outright.
                metrics["approach_bearing_deg"] = float(approach_bearing_deg)
                prior = _PriorOutcome(
                    None, None, {"used": False, "reason": "explicit_bearing"}
                )
            metrics["readiness_prior"] = prior.info
            (
                max_bearing_error_deg,
                retry_offsets_deg,
                fallback_to_arrival,
            ) = self._prior_bearing_rules()
            # The bearing the first attempt docks from, and how far the
            # arrival bearing may sit from it for "already docked" to count.
            # The arrival bearing itself is aligned by definition.
            alignment_tolerance_rad: float | None
            if explicit_bearing is not None:
                reference_bearing = explicit_bearing
                alignment_tolerance_rad = math.radians(
                    self.config.explicit_bearing_alignment_tolerance_deg
                )
            elif prior.bearing_rad is not None:
                reference_bearing = prior.bearing_rad
                alignment_tolerance_rad = math.radians(max_bearing_error_deg)
            else:
                reference_bearing = arrival_bearing
                alignment_tolerance_rad = None
            metrics["reference_bearing_rad"] = reference_bearing
            within = target.distance_m <= self.config.docking_distance_m
            # Already close enough counts only when we are also on the side
            # the human worked from, or the side that was asked for;
            # otherwise re-dock to the reference bearing.
            aligned = alignment_tolerance_rad is None or abs(
                _wrap_angle(arrival_bearing - reference_bearing)
            ) <= alignment_tolerance_rad
            if within and aligned:
                success = True
                reason = "within_docking_distance"
                self._record_suggested_arm(metrics, target, prior.suggested_arm)
            else:
                if explicit_bearing is not None:
                    attempts: list[tuple[str, float]] = [
                        ("explicit", explicit_bearing)
                    ]
                elif prior.bearing_rad is None:
                    attempts = [("arrival", arrival_bearing)]
                else:
                    attempts = [("prior", prior.bearing_rad)]
                    attempts.extend(
                        (
                            "prior_offset",
                            _wrap_angle(prior.bearing_rad + math.radians(offset)),
                        )
                        for offset in retry_offsets_deg
                    )
                    if fallback_to_arrival:
                        attempts.append(("arrival_fallback", arrival_bearing))
                distances = self.config.nav2_goal_distance_schedule_m
                # The arm the heading offset places the object in front of:
                # the hand the human used when the prior chose the bearing,
                # else the configured default. Decided once, so every goal
                # of this call docks with the same heading rule.
                if prior.used and prior.suggested_arm in ("left", "right"):
                    heading_offset_arm = prior.suggested_arm
                else:
                    heading_offset_arm = self.config.arm_side_heading_default_arm
                # Signed change from the heading that faces the object: the
                # base turns right (negative) for the left arm and left
                # (positive) for the right arm, so the object sits on that
                # arm's side by the offset angle.
                offset_deg = float(self.config.arm_side_heading_offset_deg)
                if offset_deg == 0.0:
                    heading_offset_deg = 0.0
                elif heading_offset_arm == "left":
                    heading_offset_deg = -offset_deg
                else:
                    heading_offset_deg = offset_deg
                heading_offset_rad = math.radians(heading_offset_deg)
                metrics["heading_offset_arm"] = heading_offset_arm
                metrics["heading_offset_deg"] = heading_offset_deg

                def docking_yaw(facing_yaw: float) -> float:
                    # Applied only when there is an offset so a zero offset
                    # leaves the facing heading's own expression untouched.
                    if heading_offset_rad == 0.0:
                        return facing_yaw
                    return _wrap_angle(facing_yaw + heading_offset_rad)

                def camera_goal(
                    bearing_source: str, bearing: float, goal_distance: float
                ) -> tuple[float, float, float]:
                    if bearing_source in _ARRIVAL_BEARING_SOURCES:
                        # Kept literally as before the prior existed so the
                        # arrival goal is bit-identical to the old controller.
                        return (
                            target_x - goal_distance * dx / distance,
                            target_y - goal_distance * dy / distance,
                            docking_yaw(math.atan2(dy, dx)),
                        )
                    unit_x = math.cos(bearing)
                    unit_y = math.sin(bearing)
                    return (
                        target_x + goal_distance * unit_x,
                        target_y + goal_distance * unit_y,
                        # Face the object, then apply the arm-side offset.
                        docking_yaw(math.atan2(-unit_y, -unit_x)),
                    )

                metrics["goal_attempts"] = []
                # Duck-typed Nav2Client, acquired at the first goal.
                client: Any = None
                nav2_result: Mapping[str, Any] = {}
                operator_stopped = False
                # The fallback distance the base ended at, whether navigated
                # or already standing there; None when the inner goal was
                # navigated. It moves the arrival check's boundary out.
                fallback_distance_m: float | None = None
                # The observation a stall check docked from; None otherwise.
                stall_arrival: Any = None
                alignment_tolerance_rad = math.radians(
                    self.config.explicit_bearing_alignment_tolerance_deg
                )

                def record_attempt(
                    bearing_source: str,
                    bearing: float,
                    plan_checks: list[dict[str, Any]],
                    *,
                    success: bool,
                    reason: str,
                    goal_distance: float | None,
                    camera_goal_xy_yaw: list[float] | None,
                    nav2_goal_xy_yaw: list[float] | None,
                ) -> None:
                    metrics["goal_attempts"].append(
                        {
                            "bearing_source": bearing_source,
                            "bearing_rad": bearing,
                            "camera_goal_xy_yaw": camera_goal_xy_yaw,
                            "nav2_goal_xy_yaw": nav2_goal_xy_yaw,
                            "goal_distance_m": goal_distance,
                            # False when no goal was sent: every plan check
                            # failed, the operator stopped first, or the base
                            # already stood at the fallback distance.
                            "navigated": nav2_goal_xy_yaw is not None,
                            "plan_checks": list(plan_checks),
                            "success": success,
                            "reason": reason,
                        }
                    )

                def stop_pending() -> bool:
                    # A stop that lands while no goal is active, before the
                    # client exists, during a plan check or between two
                    # goals, is only remembered in a flag: this call's own,
                    # the client's (None before acquisition), or the motion
                    # controller's sticky one. Read them before every goal.
                    if operator_stop.is_set():
                        return True
                    for owner in (client, self._motion):
                        flag = getattr(owner, "stop_requested", None)
                        if callable(flag) and bool(flag()):
                            return True
                    return False

                for attempt_index, (bearing_source, bearing) in enumerate(attempts):
                    plan_checks: list[dict[str, Any]] = []
                    # Whether the bearing ended with a goal, in place, or on
                    # an operator stop, rather than running out of goals.
                    concluded = False
                    # The distance a stall check on this bearing measured with
                    # the camera on the bearing; None before any such check.
                    stalled_distance_m: float | None = None
                    for distance_index, goal_distance in enumerate(distances):
                        camera_goal_x, camera_goal_y, goal_yaw = camera_goal(
                            bearing_source, bearing, goal_distance
                        )
                        goal_x, goal_y, goal_yaw = self._base_goal_from_camera_goal(
                            camera_goal_x, camera_goal_y, goal_yaw
                        )
                        publish_progress(
                            "goal_ready",
                            goal_xy_yaw=[goal_x, goal_y, goal_yaw],
                            target_distance_m=target.distance_m,
                            bearing_source=bearing_source,
                            bearing_rad=bearing,
                            attempt=attempt_index,
                            goal_distance_m=goal_distance,
                        )

                        if client is None:
                            # First goal only: the start footprint does not
                            # change between goals sent from the same pose.
                            self._check_nav2_start_clearance(metrics)
                            client = self._client()
                            # The previous dock's final cancel_and_stop left
                            # the client's flag set; a stop that landed
                            # earlier in this call is still held by
                            # ``operator_stop``, so only that flag is reset.
                            reset = getattr(client, "reset_stop_request", None)
                            if callable(reset):
                                reset()
                        # The base already stands at or inside this fallback
                        # distance on the attempt's bearing, so reaching it
                        # would mean backing away: the pose is taken as
                        # docked there, without a plan check, once a stop
                        # has been ruled out. After a stall on this bearing
                        # the base stands where the stall check measured it,
                        # and that check found it on the bearing.
                        if stalled_distance_m is None:
                            standing_distance_m = float(target.distance_m)
                            standing_on_bearing = (
                                abs(_wrap_angle(arrival_bearing - bearing))
                                <= alignment_tolerance_rad
                            )
                        else:
                            standing_distance_m = stalled_distance_m
                            standing_on_bearing = True
                        in_place = (
                            distance_index > 0
                            and goal_distance >= standing_distance_m
                            and standing_on_bearing
                        )
                        unreachable = False
                        if self.config.plan_check_before_goal and not in_place:
                            plan = client.compute_path_to_pose(
                                goal_x,
                                goal_y,
                                goal_yaw,
                                frame_id=self.config.nav2_frame_id,
                                server_timeout_s=self.config.nav2_server_timeout_s,
                            )
                            plan_reason = str(plan.get("reason", "unknown"))
                            # True: a path exists. False: the goal itself is
                            # unreachable. None: the planner could not say.
                            plannable: bool | None
                            if plan.get("success", False):
                                plannable = True
                            elif plan_reason in _UNREACHABLE_PLAN_REASONS:
                                plannable = False
                            else:
                                plannable = None
                            unreachable = plannable is False
                            plan_check = {
                                "goal_distance_m": goal_distance,
                                "plannable": plannable,
                                "reason": plan_reason,
                                "path_length_m": _finite_or_none(
                                    plan.get("path_length_m")
                                ),
                                "nav2_goal_xy_yaw": [goal_x, goal_y, goal_yaw],
                            }
                            plan_checks.append(plan_check)
                            publish_progress(
                                "plan_checked",
                                bearing_source=bearing_source,
                                bearing_rad=bearing,
                                attempt=attempt_index,
                                **plan_check,
                            )
                        if stop_pending():
                            # The operator stopped the base while no goal
                            # was active: nothing may drive it again, at any
                            # distance or bearing, so no goal is sent, and a
                            # pose the base already holds is not reported as
                            # a dock either.
                            operator_stopped = True
                            nav2_result = {
                                "success": False,
                                "reason": _OPERATOR_STOP_REASON,
                            }
                            record_attempt(
                                bearing_source,
                                bearing,
                                plan_checks,
                                success=False,
                                reason=_OPERATOR_STOP_REASON,
                                goal_distance=None,
                                camera_goal_xy_yaw=None,
                                nav2_goal_xy_yaw=None,
                            )
                            publish_progress(
                                "stop_requested",
                                bearing_source=bearing_source,
                                bearing_rad=bearing,
                                attempt=attempt_index,
                                goal_distance_m=goal_distance,
                            )
                            concluded = True
                            break
                        if in_place:
                            fallback_distance_m = standing_distance_m
                            metrics["goal_distance_used_m"] = fallback_distance_m
                            metrics["bearing_source"] = None
                            metrics["desired_camera_goal_xy_yaw"] = None
                            metrics["nav2_goal_xy_yaw"] = None
                            nav2_result = {
                                "success": True,
                                "reason": _DOCKED_AT_FALLBACK_REASON,
                            }
                            record_attempt(
                                bearing_source,
                                bearing,
                                plan_checks,
                                success=True,
                                reason=_DOCKED_AT_FALLBACK_REASON,
                                goal_distance=fallback_distance_m,
                                camera_goal_xy_yaw=None,
                                nav2_goal_xy_yaw=None,
                            )
                            publish_progress(
                                "docked_in_place",
                                bearing_source=bearing_source,
                                bearing_rad=bearing,
                                attempt=attempt_index,
                                goal_distance_m=goal_distance,
                                target_distance_m=target.distance_m,
                            )
                            concluded = True
                            break
                        if unreachable:
                            # Skip a goal the planner cannot reach and try
                            # the next, farther one on this bearing.
                            continue
                        # These describe the goal actually navigated; a
                        # bearing that sends none leaves or clears them.
                        metrics["bearing_source"] = bearing_source
                        metrics["desired_camera_goal_xy_yaw"] = [
                            camera_goal_x,
                            camera_goal_y,
                            goal_yaw,
                        ]
                        metrics["nav2_goal_xy_yaw"] = [goal_x, goal_y, goal_yaw]
                        nav2_result = client.navigate_to_pose(
                            goal_x,
                            goal_y,
                            goal_yaw,
                            frame_id=self.config.nav2_frame_id,
                            server_timeout_s=self.config.nav2_server_timeout_s,
                            timeout_s=self.config.timeout_s,
                            behavior_tree=self.config.behavior_tree,
                            progress_callback=publish_nav2_progress,
                            progress_interval_s=self.config.progress_interval_s,
                            no_progress_timeout_s=self.config.no_progress_timeout_s,
                            minimum_progress_translation_m=(
                                self.config.minimum_progress_translation_m
                            ),
                            minimum_progress_yaw_rad=(
                                self.config.minimum_progress_yaw_rad
                            ),
                            self_filter_attached_objects=(
                                self._env.robot_self_filter_attached_objects()
                                if callable(
                                    getattr(
                                        self._env,
                                        "robot_self_filter_attached_objects",
                                        None,
                                    )
                                )
                                else {}
                            ),
                        )
                        concluded = True
                        fallback_distance_m = (
                            goal_distance if distance_index > 0 else None
                        )
                        metrics["goal_distance_used_m"] = goal_distance
                        attempt_success = bool(nav2_result.get("success", False))
                        attempt_reason = str(nav2_result.get("reason", "unknown"))
                        history.append(
                            {
                                "primitive": "nav2_navigate_to_pose",
                                "success": attempt_success,
                                "reason": attempt_reason,
                                "metrics": dict(nav2_result.get("feedback", {})),
                                "bearing_source": bearing_source,
                                "bearing_rad": bearing,
                                "goal_distance_m": goal_distance,
                            }
                        )
                        record_attempt(
                            bearing_source,
                            bearing,
                            plan_checks,
                            success=attempt_success,
                            reason=attempt_reason,
                            goal_distance=goal_distance,
                            camera_goal_xy_yaw=[
                                camera_goal_x,
                                camera_goal_y,
                                goal_yaw,
                            ],
                            nav2_goal_xy_yaw=[goal_x, goal_y, goal_yaw],
                        )
                        metrics["nav2"] = dict(nav2_result)
                        if attempt_success:
                            break
                        if attempt_reason == _OPERATOR_STOP_REASON:
                            # The operator stopped the base: nothing may
                            # drive it again, at any distance or bearing.
                            operator_stopped = True
                            break
                        if attempt_reason == _NAV2_STALL_REASON:
                            # A stall is Nav2's verdict on the way there, not
                            # on where the base ended: look again, and dock
                            # where it stands if the arrival check this goal
                            # would have run passes on the bearing.
                            stall_check, stalled_target = self._observe_after_stall(
                                object_name,
                                bearing_rad=bearing,
                                fallback_distance_m=fallback_distance_m,
                                inner_goal_distance_m=goal_distance_m,
                                alignment_tolerance_rad=alignment_tolerance_rad,
                            )
                            if stalled_target is not None:
                                stalled_distance_m = float(stalled_target.distance_m)
                            if stall_check["docked"] and stop_pending():
                                # A Stop that landed during the check: the
                                # pose the base holds is not a dock.
                                stall_check["docked"] = False
                                stall_check["refused_by_operator_stop"] = True
                            metrics.setdefault("stall_checks", []).append(
                                {
                                    "bearing_source": bearing_source,
                                    "bearing_rad": bearing,
                                    "goal_distance_m": goal_distance,
                                    **stall_check,
                                }
                            )
                            publish_progress(
                                "stall_checked",
                                bearing_source=bearing_source,
                                bearing_rad=bearing,
                                attempt=attempt_index,
                                goal_distance_m=goal_distance,
                                docked=stall_check["docked"],
                                target_distance_m=stall_check["target_distance_m"],
                            )
                            if stall_check["docked"]:
                                stall_arrival = stalled_target
                                break
                        # Any other failure falls through to the next,
                        # farther goal on the same bearing.
                    if not concluded:
                        # Every goal distance on this bearing failed the plan
                        # check; the bearing is exhausted without driving,
                        # so no navigated goal describes the base pose.
                        nav2_result = {
                            "success": False,
                            "reason": _NO_PLANNABLE_GOAL_REASON,
                        }
                        metrics["bearing_source"] = None
                        metrics["desired_camera_goal_xy_yaw"] = None
                        metrics["nav2_goal_xy_yaw"] = None
                        record_attempt(
                            bearing_source,
                            bearing,
                            plan_checks,
                            success=False,
                            reason=_NO_PLANNABLE_GOAL_REASON,
                            goal_distance=None,
                            camera_goal_xy_yaw=None,
                            nav2_goal_xy_yaw=None,
                        )
                        continue
                    if (
                        nav2_result.get("success", False)
                        or operator_stopped
                        or stall_arrival is not None
                    ):
                        break
                    if (
                        self._nav2_failure_reason(nav2_result)
                        == _BLOCKED_RECOVERY_REASON
                    ):
                        # The base cannot leave its pose, so another bearing
                        # from the same pose cannot help: return the back-away
                        # recovery now. Stalls and plain aborts fall through
                        # to the next bearing.
                        break
                if not nav2_result.get("success", False) and stall_arrival is None:
                    reason = self._nav2_failure_reason(nav2_result)
                    if reason == _BLOCKED_RECOVERY_REASON:
                        metrics["recommended_recovery"] = {
                            "action": "back_away_then_retry",
                            "primitive": "drive_straight",
                            "distance_sign": "negative",
                            "requires_known_clear_rear_path": True,
                        }
                    elif reason == _STALLED_RECOVERY_REASON:
                        metrics["recommended_recovery"] = {
                            "action": "small_forward_step_then_reobserve",
                            "primitive": "drive_straight",
                            "distance_sign": "positive",
                            "suggested_distance_m": 0.10,
                            "requires_visible_target_ahead": True,
                            "requires_clear_forward_path": True,
                        }
                else:
                    publish_progress("verifying_arrival")
                    if stall_arrival is None:
                        rgb, depth, intrinsics = self._perception._camera_input(
                            require_new_frame=True
                        )
                        final_target = self._perception._detect_target(
                            object_name, rgb, depth, intrinsics
                        )
                    else:
                        # The stall check's observation already passed this
                        # goal's arrival check on the bearing.
                        final_target = stall_arrival
                    metrics["final_target"] = final_target.metrics()
                    arrival_limit_m = self._arrival_limit_m(
                        fallback_distance_m, goal_distance_m
                    )
                    if final_target.distance_m <= arrival_limit_m:
                        success = True
                        if stall_arrival is not None:
                            reason = _DOCKED_AFTER_STALL_REASON
                        elif fallback_distance_m is not None:
                            reason = _DOCKED_AT_FALLBACK_REASON
                        else:
                            reason = "within_docking_distance"
                        self._record_suggested_arm(
                            metrics, final_target, prior.suggested_arm
                        )
                    else:
                        reason = "final_visual_verification_failed"
        except _Nav2StartBlocked as exc:
            reason = str(exc)
        except BaseException as exc:  # final zero must also cover interrupts
            fatal_error = exc
            reason = f"exception:{type(exc).__name__}:{exc}"
        finally:
            if self._nav2_client is not None:
                final_nav2_stop = (
                    self._nav2_client.halt
                    if success
                    else self._nav2_client.cancel_and_stop
                )
                final_nav2_stop_key = (
                    "nav2_final_halt"
                    if success
                    else "nav2_final_cancel_and_stop"
                )
                try:
                    metrics[final_nav2_stop_key] = final_nav2_stop()
                except Exception as exc:  # noqa: BLE001 - direct zero follows
                    metrics[final_nav2_stop_key] = {
                        "accepted": False,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
            unregister = getattr(
                self._env, "unregister_motion_stop_callback", None
            )
            if callable(unregister):
                unregister(on_operator_stop)
            try:
                final_stop = self._motion.stop()
                history.append(self._summary(final_stop))
                if not final_stop.get("success", False):
                    success = False
                    stop_reason = final_stop.get("reason", "unknown")
                    reason = f"{reason}; final_stop_failed: {stop_reason}"
            except BaseException as exc:
                success = False
                reason = (
                    f"{reason}; final_stop_exception:"
                    f"{type(exc).__name__}:{exc}"
                )
                if fatal_error is None:
                    fatal_error = exc

        result = self._result(
            success,
            reason,
            object_name,
            started,
            metrics,
            history,
            final_stop,
        )
        progress_state = "completed" if success else "failed"
        progress_reason = reason
        if isinstance(fatal_error, KeyboardInterrupt):
            progress_state = "operator_interrupted"
            progress_reason = "operator_interrupt"
        publish_progress(
            progress_state,
            terminal=True,
            success=bool(success),
            reason=progress_reason,
        )
        if fatal_error is not None and not isinstance(fatal_error, Exception):
            raise fatal_error
        return result

    def _result(
        self,
        success: bool,
        reason: str,
        object_name: str,
        started: float,
        metrics: Mapping[str, Any],
        history: list[dict[str, Any]],
        final_stop: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        result = {
            "primitive": "dock_to_visible_object",
            "success": bool(success),
            "reason": reason,
            "target": object_name,
            "elapsed_s": max(0.0, self._clock() - started),
            "metrics": dict(metrics),
            "motion_history": history,
            "final_stop": None if final_stop is None else dict(final_stop),
        }
        # Which arm the human used in the passive video, for the policy to
        # pass to prepare_for_manipulation; present only when the prior knows.
        suggested_arm = metrics.get("suggested_arm")
        if suggested_arm:
            result["suggested_arm"] = str(suggested_arm)
        return result

    @staticmethod
    def _summary(result: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "primitive": result.get("primitive"),
            "success": bool(result.get("success", False)),
            "reason": str(result.get("reason", "unknown")),
            "metrics": dict(result.get("metrics", {})),
        }

    def _arrival_limit_m(
        self, fallback_distance_m: float | None, inner_goal_distance_m: float
    ) -> float:
        """Largest measured object distance the arrival check accepts.

        A farther fallback goal moves the success boundary out with it,
        keeping the convergence margin the inner goal has to the docking
        distance. A base that stood still uses the distance it stood at.
        """

        if fallback_distance_m is None:
            return (
                self.config.docking_distance_m
                + self.config.final_distance_tolerance_m
            )
        return (
            fallback_distance_m
            + (self.config.docking_distance_m - inner_goal_distance_m)
            + self.config.final_distance_tolerance_m
        )

    def _observe_after_stall(
        self,
        object_name: str,
        *,
        bearing_rad: float,
        fallback_distance_m: float | None,
        inner_goal_distance_m: float,
        alignment_tolerance_rad: float,
    ) -> tuple[dict[str, Any], Any]:
        """Observe the object where a Nav2 stall left the base.

        Returns the stall-check record and the observed target when the
        camera stands on the attempt's bearing, else None (also when the
        object is not found). ``docked`` says whether that target also lies
        inside the arrival boundary of the goal that stalled.
        """

        limit_m = self._arrival_limit_m(fallback_distance_m, inner_goal_distance_m)
        check: dict[str, Any] = {
            "arrival_limit_m": limit_m,
            "target_distance_m": None,
            "bearing_error_deg": None,
            "on_bearing": None,
            "docked": False,
            "refused_by_operator_stop": False,
            "detection_error": None,
        }
        try:
            rgb, depth, intrinsics = self._perception._camera_input(
                require_new_frame=True
            )
            pose = self._perception._current_pose()
            target = self._perception._detect_target(
                object_name, rgb, depth, intrinsics
            )
        except RuntimeError as exc:
            # An object SAM3 or the depth no longer resolves leaves the stall
            # as it was: the next goal, or the stall failure, follows.
            check["detection_error"] = str(exc)
            return check, None
        target_x, target_y = self._perception._target_world_xy(target, pose)
        stood_bearing = math.atan2(
            float(pose[1]) - target_y, float(pose[0]) - target_x
        )
        bearing_error = abs(_wrap_angle(stood_bearing - bearing_rad))
        on_bearing = bearing_error <= alignment_tolerance_rad
        distance_m = float(target.distance_m)
        check.update(
            target_distance_m=distance_m,
            bearing_error_deg=math.degrees(bearing_error),
            on_bearing=on_bearing,
            docked=on_bearing and distance_m <= limit_m,
        )
        return check, (target if on_bearing else None)

    @staticmethod
    def _nav2_failure_reason(nav2_result: Mapping[str, Any]) -> str:
        nav2_reason = str(nav2_result.get("reason", "unknown"))
        feedback = nav2_result.get("feedback")
        if not isinstance(feedback, Mapping):
            feedback = {}
        try:
            recoveries = int(feedback.get("number_of_recoveries", 0))
            navigation_time_s = float(feedback.get("navigation_time_s", 0.0))
        except (TypeError, ValueError):
            recoveries = 0
            navigation_time_s = 0.0
        if nav2_reason == _NO_PLANNABLE_GOAL_REASON:
            return nav2_reason
        if nav2_reason == "stalled_no_progress":
            return _STALLED_RECOVERY_REASON
        if (
            nav2_reason == "blocked_no_progress"
            or (
                nav2_reason == "aborted"
                and recoveries >= 1
                and navigation_time_s >= 5.0
            )
        ):
            return _BLOCKED_RECOVERY_REASON
        return f"nav2_failed:{nav2_reason}"

    def settings(self) -> dict[str, Any]:
        return asdict(self.config)
