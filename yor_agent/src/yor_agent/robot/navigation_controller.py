"""Closed-loop, bounded navigation motion for the YOR mobile base.

Moved from ``YOR/Agent/agents_yor/controller.py``. Behavior is unchanged except
that the temporary ``turn_relative`` convergence instrumentation is commented out
rather than printed into every generated policy's stdout.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import threading
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .contracts import NavigationEnvironment
from .footprint_clearance import (
    CameraGeometry,
    ClearanceResult,
    FootprintConfig,
    StickyOccupancy,
    check_cells,
    depth_to_base_points,
    occupied_cells,
    arm_band_layers,
    sphere_layers,
    spheres_to_base,
    sweep_distance,
    sweep_visibility,
    points_in_convex_polygon,
    seen_through_cells,
)
from .self_filter import RobotDepthSelfFilter, RobotSelfFilterConfig


class PrimitiveReason(RuntimeError):
    """A failure whose message is written for the policy, not for a log.

    ``_exception_reason`` returns :attr:`reason` verbatim instead of the
    ``Type:token`` form, so the model reads a sentence it can act on rather
    than a name whose meaning it has to guess.
    """

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = str(reason)


# The camera looks forward only, so the space a sideways or backward step
# enters is unobserved rather than free and the gate fails closed. Naming the
# recovery matters: with the bare token the model spent three separate policy
# turns retrying drive_lateral in one run (2026-09-08 22:01), each refusal
# skipping the rest of that policy.
DEPTH_UNKNOWN_IN_SWEEP_REASON = (
    "depth_unknown_in_sweep: the space this step would move into returned no "
    "depth, so it is unobserved rather than known to be free, and the step was "
    "refused. The camera faces forward, so a sideways or backward step almost "
    "always enters space it cannot see. Do not retry the same step and do not "
    "just shorten it. Turn to face the direction you want to travel with "
    "turn_relative, re-observe, then use drive_straight. If this happened on a "
    "forward step instead, the floor ahead gave no returns (low texture, "
    "glass, a dark or shiny surface): re-observe, then try a shorter step or a "
    "slightly different heading."
)


def wrap_angle(angle: float) -> float:
    return float(math.atan2(math.sin(angle), math.cos(angle)))


# Remembered cells checked against each frame, nearest along the command
# first. The rest wait for a later cycle, so the check stays well inside the
# control period however much a call has remembered.
SEEN_THROUGH_MAX_CELLS_PER_CYCLE = 256


@dataclass(frozen=True)
class NavigationConfig:
    control_hz: float = 20.0
    pose_max_age_s: float = 0.30
    max_distance_m: float = 2.0
    max_reverse_distance_m: float = 0.50
    max_turn_rad: float = math.pi
    max_duration_s: float = 45.0
    default_linear_mps: float = 0.10
    default_reverse_mps: float = 0.06
    default_yaw_rad_s: float = 0.30
    min_linear_mps: float = 0.035
    min_yaw_rad_s: float = 0.055
    distance_tolerance_m: float = 0.025
    yaw_tolerance_rad: float = math.radians(2.5)
    settle_cycles: int = 3
    linear_kp: float = 0.9
    yaw_kp: float = 1.5
    heading_kp: float = 1.4
    cross_track_kp: float = 1.0
    min_front_clearance_m: float = 0.45
    depth_min_valid_pixels: int = 80
    depth_percentile: float = 10.0
    max_pose_jump_m: float = 0.25
    max_yaw_jump_rad: float = 0.65
    stop_timeout_s: float = 0.8
    auto_timeout_scale: float = 7.0
    auto_timeout_overhead_s: float = 5.0
    # Footprint-swept clearance (robot/footprint_clearance.py). It replaces the
    # central-cone depth gate whenever the camera geometry is configured; the
    # legacy gate remains for configurations without calibration.
    swept_clearance: bool = True
    footprint: Mapping[str, Any] | None = None
    command_latency_s: float = 0.10
    lease_s: float = 0.25
    brake_accel_mps2: float = 0.30
    clearance_margin_m: float = 0.05
    clearance_cell_m: float = 0.05
    clearance_cell_min_points: int = 3
    clearance_pixel_stride: int = 2
    clearance_max_depth_m: float = 6.0
    # Coarse "camera blind" guard only. The ZED service's confidence filters
    # leave about 0.20 of the lower band valid on an open textureless floor
    # (measured 2026-09-03), so unknown space ahead is judged per image
    # column by sweep_visibility instead of by this fraction.
    min_valid_depth_fraction: float = 0.05
    # Fraction of rays through the swept volume that may be invalid depth
    # before the frame is treated as unknown space ahead (fail closed).
    max_unknown_sweep_fraction: float = 0.30
    layer_z_tolerance_m: float = 0.03
    # The occupancy memory keeps cells a primitive call has seen so that an
    # obstacle leaving the view on the way in still stops the base. Kept
    # unconditionally, a single frame's stray depth points in mid-air refused
    # moves through space the camera was looking straight through. A cell is
    # now forgotten when, over at least the given fraction of valid pixels it
    # projects to, every depth lies beyond its far side by the margin. Cells
    # out of view, or seen with too little valid depth, are kept.
    memory_seen_through_clearing: bool = True
    memory_seen_through_margin_m: float = 0.15
    memory_seen_through_min_valid_fraction: float = 0.9
    memory_seen_through_lookahead_m: float = 1.0
    # Replace the configured arm layer with height-resolved bands built from
    # the collision spheres the depth self-filter already computes for the
    # live joint state. A single tall prism drawn around the gripper's
    # forward reach charges a table top against hardware that is nowhere near
    # its height, which stops the base ~0.6 m short of any desk.
    arm_footprint_from_spheres: bool = True
    arm_footprint_layer: str = "arms"
    arm_footprint_band_m: float = 0.05
    # Sideways pad, the same 0.05 m the rest of this model carries. It used to
    # be 0.16 to reach a 1.20 m arm span attributed to a 2026-09-03 tape
    # measurement that was never taken; the sphere envelope is the only
    # measurement there has ever been. Padding to that figure drew the arms
    # 0.638 m to each side against a true 0.478 m, so a 1.27 m-wide model
    # refused aisles the 0.95 m-wide robot fits through: on 2026-09-08 22:16 it
    # blocked three drive_straight calls and held the Nav2 arms monitor in
    # approach for 12.9 s beside a bookshelf, stalling the dock.
    arm_footprint_lateral_margin_m: float = 0.05
    # Forward pad, and much smaller on purpose. Forward the sphere model does
    # NOT under-state the hardware: FK reaches 0.482 m against the 0.47 m the
    # tape implies, so this only carries the repo's usual 0.05 m. Using the
    # sideways figure here pushed the modelled gripper to 0.615 m, past the
    # static outline's 0.52 m, and refused a prepare nudge by 7 mm that the
    # old model would have allowed (2026-09-04 15:29).
    arm_footprint_forward_margin_m: float = 0.05
    # Vertical pad, and deliberately tiny. Floor-plane drift is already paid
    # for on the obstacle side by layer_z_tolerance_m, which spreads every
    # depth point across neighbouring bands; padding here too would double
    # count. This covers only sphere_erosion_m (0.008), which shrinks the
    # spheres vertically as well as sideways. The gripper tops out 0.074 m
    # below an office desk top, so anything near the horizontal margin here
    # would smear the fingers into the table and undo the height resolution.
    arm_footprint_z_margin_m: float = 0.01
    # The vertical pad inside which a cell is at the arms' own height, when it
    # is smaller than arm_footprint_z_margin_m. Cells within this pad keep the
    # full horizontal pads and clearance_margin_m; cells only the larger pad
    # reaches (a support surface the arms pass above) are charged with the two
    # underside margins below instead. None keeps a single set of arm layers.
    arm_footprint_core_z_margin_m: float | None = None
    # Forward pad for those underside cells. The larger vertical pad already
    # keeps the arms from passing over such a surface; padding forward as
    # well refused steps that stop short of its edge. None: the forward pad.
    arm_footprint_underside_forward_margin_m: float | None = None
    # Fixed sweep margin for those underside cells, on top of the reaction and
    # braking travel every layer sweeps. None: clearance_margin_m.
    arm_footprint_underside_clearance_margin_m: float | None = None
    # Grown around always-occupied structure the URDF omits. The chassis and
    # lift column sit well inside the body polygon, but the mast and the ZED
    # head are unmeasured and the camera's own optical centre is already at
    # x = 0.2143 against a 0.22 body front. Measure them and tighten this.
    arm_footprint_structure_margin_m: float = 0.05
    # Vertical pad of the Nav2 arms collision monitor's band, when it differs
    # from arm_footprint_z_margin_m. This controller does not read it; the
    # Nav2 parameter renderer does, so both pads are set in one navigation
    # block. None: the Nav2 band uses arm_footprint_z_margin_m.
    nav2_arm_band_z_margin_m: float | None = None

    _NON_NUMERIC = frozenset(
        {
            "settle_cycles",
            "depth_min_valid_pixels",
            "swept_clearance",
            "memory_seen_through_clearing",
            "footprint",
            "clearance_cell_min_points",
            "clearance_pixel_stride",
            "arm_footprint_from_spheres",
            "arm_footprint_layer",
            "arm_footprint_core_z_margin_m",
            "arm_footprint_underside_forward_margin_m",
            "arm_footprint_underside_clearance_margin_m",
            "nav2_arm_band_z_margin_m",
        }
    )

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | None) -> "NavigationConfig":
        if not values:
            cfg = cls()
        else:
            known = set(cls.__dataclass_fields__)
            unknown = sorted(set(values) - known)
            if unknown:
                raise ValueError(f"unknown navigation settings: {unknown}")
            cfg = cls(**dict(values))
        cfg._validate()
        return cfg

    def _validate(self) -> None:
        numeric = [
            value
            for key, value in asdict(self).items()
            if key not in self._NON_NUMERIC
        ]
        if not all(math.isfinite(float(value)) for value in numeric):
            raise ValueError("navigation settings must be finite")
        if self.footprint is not None and not isinstance(self.footprint, Mapping):
            raise ValueError("navigation.footprint must be a mapping")
        if self.clearance_cell_min_points < 1 or self.clearance_pixel_stride < 1:
            raise ValueError("clearance cell threshold and pixel stride must be positive")
        if self.clearance_cell_m <= 0 or self.brake_accel_mps2 <= 0:
            raise ValueError("clearance cell size and brake acceleration must be positive")
        if self.clearance_max_depth_m <= 0.1:
            raise ValueError("clearance_max_depth_m must exceed 0.1 m")
        if not 0.0 <= self.min_valid_depth_fraction <= 1.0:
            raise ValueError("min_valid_depth_fraction must be in [0, 1]")
        if not 0.0 <= self.max_unknown_sweep_fraction <= 1.0:
            raise ValueError("max_unknown_sweep_fraction must be in [0, 1]")
        if self.layer_z_tolerance_m < 0.0:
            raise ValueError("layer_z_tolerance_m must be nonnegative")
        if not isinstance(self.memory_seen_through_clearing, bool):
            raise ValueError("memory_seen_through_clearing must be a boolean")
        if self.memory_seen_through_margin_m <= 0.0:
            raise ValueError("memory_seen_through_margin_m must be positive")
        if not 0.0 < self.memory_seen_through_min_valid_fraction <= 1.0:
            raise ValueError("memory_seen_through_min_valid_fraction must be in (0, 1]")
        if self.memory_seen_through_lookahead_m <= 0.0:
            raise ValueError("memory_seen_through_lookahead_m must be positive")
        if not isinstance(self.swept_clearance, bool):
            raise ValueError("swept_clearance must be a boolean")
        if not isinstance(self.arm_footprint_from_spheres, bool):
            raise ValueError("arm_footprint_from_spheres must be a boolean")
        if not isinstance(self.arm_footprint_layer, str) or not self.arm_footprint_layer:
            raise ValueError("arm_footprint_layer must be a non-empty layer name")
        if self.arm_footprint_band_m <= 0.0:
            raise ValueError("arm_footprint_band_m must be positive")
        if self.arm_footprint_lateral_margin_m < 0.0:
            raise ValueError("arm_footprint_lateral_margin_m must be nonnegative")
        if self.arm_footprint_forward_margin_m < 0.0:
            raise ValueError("arm_footprint_forward_margin_m must be nonnegative")
        if self.arm_footprint_structure_margin_m < 0.0:
            raise ValueError("arm_footprint_structure_margin_m must be nonnegative")
        if self.arm_footprint_z_margin_m < 0.0:
            raise ValueError("arm_footprint_z_margin_m must be nonnegative")
        band_margin = self.nav2_arm_band_z_margin_m
        # The same bound the Nav2 ZED bridge accepts for its arm band.
        if band_margin is not None and (
            isinstance(band_margin, bool)
            or not isinstance(band_margin, (int, float))
            or not (math.isfinite(float(band_margin)) and 0.0 <= float(band_margin) <= 0.20)
        ):
            raise ValueError("nav2_arm_band_z_margin_m must be in [0, 0.2] or null")
        # Each underside setting may only relax its full counterpart: a fringe
        # charged more strictly than the arms' own height would be pointless.
        for key, upper in (
            ("arm_footprint_core_z_margin_m", self.arm_footprint_z_margin_m),
            (
                "arm_footprint_underside_forward_margin_m",
                self.arm_footprint_forward_margin_m,
            ),
            (
                "arm_footprint_underside_clearance_margin_m",
                self.clearance_margin_m,
            ),
        ):
            value = getattr(self, key)
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{key} must be a number or null")
            if not (math.isfinite(float(value)) and 0.0 <= float(value) <= float(upper)):
                raise ValueError(f"{key} must be in [0, {float(upper):g}]")
        if self.arm_footprint_band_m < self.clearance_cell_m:
            # Bands finer than the obstacle grid cost cycles without resolving
            # anything, and below 0.01 m the generated layer names collide.
            raise ValueError("arm_footprint_band_m must be at least clearance_cell_m")
        if self.footprint is not None:
            FootprintConfig.from_mapping(self.footprint)
        if not 10.0 <= self.control_hz <= 40.0:
            raise ValueError("control_hz must be in [10, 40]")
        if (
            self.pose_max_age_s <= 0
            or self.max_distance_m <= 0
            or self.max_reverse_distance_m <= 0
        ):
            raise ValueError("pose age and distance limits must be positive")
        if self.default_linear_mps <= 0 or self.default_reverse_mps <= 0:
            raise ValueError("default linear speeds must be positive magnitudes")
        if self.auto_timeout_scale <= 0 or self.auto_timeout_overhead_s <= 0:
            raise ValueError("automatic timeout scale and overhead must be positive")
        if not 1 <= self.settle_cycles <= 10:
            raise ValueError("settle_cycles must be in [1, 10]")
        if self.depth_min_valid_pixels < 1:
            raise ValueError("depth_min_valid_pixels must be positive")
        if not 0.0 < self.depth_percentile <= 50.0:
            raise ValueError("depth_percentile must be in (0, 50]")


class NavigationController:
    """Synchronous primitive controller over the Pi's short velocity lease."""

    def __init__(
        self,
        env: NavigationEnvironment,
        *,
        config: NavigationConfig | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.env = env
        self.config = config or NavigationConfig.from_mapping(env.navigation_config)
        self._clock = clock
        self._sleep = sleep
        self._motion_lock = threading.Lock()
        self._command_lock = threading.Lock()
        self._stop_requested = threading.Event()
        self.last_front_clearance_debug: dict[str, Any] | None = None
        self.last_clearance: ClearanceResult | None = None
        self.last_visibility: dict[str, Any] | None = None
        # Why the last swept check failed closed; copied into the primitive
        # metrics so a trace explains a depth_* stop from the first cycle.
        self.last_clearance_debug: dict[str, Any] | None = None
        # Which footprint the last call used, and the front extent per height
        # band, so a trace can say where the arms were when the gate fired.
        self.last_footprint_debug: dict[str, Any] | None = None
        self._footprint: FootprintConfig | None = None
        self._static_footprint: FootprintConfig | None = None
        self._robot_self_filter: RobotDepthSelfFilter | None = None
        self._robot_self_filter_configured = False

    def request_stop(self) -> dict[str, Any]:
        """Request an operator stop without waiting for the motion lock.

        Motion loops check the flag before every non-zero command.  Serializing
        command submission separately guarantees that the zero sent here cannot
        be followed by another non-zero command from an already-running loop.
        The active primitive will then run its usual confirmed final stop.
        """

        self._stop_requested.set()
        return self._submit_velocity([0.0, 0.0, 0.0], allow_stopped=True)

    def stop_requested(self) -> bool:
        """Whether an operator stop has been requested on this controller."""

        return self._stop_requested.is_set()

    def stop(self) -> dict[str, Any]:
        """Command a normal stop and confirm the Pi reports a zero command."""

        with self._motion_lock:
            return self._stop_impl(start_time=self._clock())

    def turn_relative(
        self,
        angle_rad: float,
        *,
        max_yaw_rad_s: float | None = None,
        timeout_s: float | None = None,
    ) -> dict[str, Any]:
        """Turn in place by a relative angle using fresh ZED yaw feedback."""

        angle_rad = self._finite_float("angle_rad", angle_rad)
        if abs(angle_rad) < self.config.yaw_tolerance_rad:
            with self._motion_lock:
                start_time = self._clock()
                try:
                    self._preflight()
                except Exception as exc:
                    return self._result(
                        "turn_relative",
                        False,
                        self._exception_reason(exc),
                        start_time,
                    )
                stop_result = self._stop_impl(start_time=start_time)
                return self._result(
                    "turn_relative",
                    stop_result["success"],
                    (
                        "already_within_tolerance"
                        if stop_result["success"]
                        else f"final_stop_failed: {stop_result['reason']}"
                    ),
                    start_time,
                    status=stop_result.get("base_status"),
                )
        if abs(angle_rad) > self.config.max_turn_rad:
            raise ValueError(
                f"abs(angle_rad) must be <= {self.config.max_turn_rad:.3f}"
            )
        yaw_limit = self._bounded_speed(
            max_yaw_rad_s,
            default=self.config.default_yaw_rad_s,
            minimum=self.config.min_yaw_rad_s,
            status_key="max_yaw_rad_s",
        )
        timeout = self._timeout(
            timeout_s,
            nominal=(
                abs(angle_rad)
                / max(yaw_limit, 1e-6)
                * self.config.auto_timeout_scale
                + self.config.auto_timeout_overhead_s
            ),
        )

        with self._motion_lock:
            start_time = self._clock()
            start_pose: list[float] | None = None
            final_pose: list[float] | None = None
            reason = "timeout"
            success = False
            metrics: dict[str, Any] = {}
            fatal_error: BaseException | None = None
            try:
                status = self._preflight()
                yaw_limit = min(yaw_limit, float(status["limits"]["max_yaw_rad_s"]))
                frame = self._frame()
                start_pose = self._pose(frame)
                previous = start_pose
                target_yaw = wrap_angle(start_pose[2] + angle_rad)
                settled = 0
                # DEBUG instrumentation for the turn_relative convergence
                # investigation. Kept commented out: the agent loop feeds policy
                # stdout back to the model, so per-cycle prints would flood the
                # next prompt. Re-enable locally when the timeout/lag cause is
                # being investigated again.
                # _debug_iter = 0
                # _debug_loop_start = self._clock()
                # _debug_prev_t = _debug_loop_start
                while self._clock() - start_time <= timeout:
                    self._raise_if_stop_requested()
                    # _debug_frame_wait_start = self._clock()
                    frame = self._frame()
                    # _debug_now = self._clock()
                    pose = self._pose(frame)
                    self._check_pose_jump(previous, pose)
                    previous = pose
                    final_pose = pose
                    error = wrap_angle(target_yaw - pose[2])
                    metrics = {
                        "target_yaw_rad": target_yaw,
                        "yaw_error_rad": error,
                        "command_yaw_limit_rad_s": yaw_limit,
                        "minimum_yaw_command_rad_s": min(
                            self.config.min_yaw_rad_s, yaw_limit
                        ),
                    }
                    # _debug_iter += 1
                    # print(
                    #     f"[turn_relative debug] iter={_debug_iter} "
                    #     f"t={_debug_now - _debug_loop_start:.3f}s "
                    #     f"dt_since_prev={_debug_now - _debug_prev_t:.3f}s "
                    #     f"frame_wait={_debug_now - _debug_frame_wait_start:.3f}s "
                    #     f"yaw_error_deg={math.degrees(error):.2f} "
                    #     f"frame_ts_age_s={_debug_now - getattr(frame, 'timestamp_ns', _debug_now * 1e9) / 1e9:.3f}"
                    # )
                    # _debug_prev_t = _debug_now
                    if abs(error) <= self.config.yaw_tolerance_rad:
                        settled += 1
                        self._submit_velocity([0.0, 0.0, 0.0], allow_stopped=True)
                        if settled >= self.config.settle_cycles:
                            success = True
                            reason = "target_reached"
                            break
                    else:
                        settled = 0
                        omega = float(
                            np.clip(
                                self.config.yaw_kp * error,
                                -yaw_limit,
                                yaw_limit,
                            )
                        )
                        minimum_yaw = min(self.config.min_yaw_rad_s, yaw_limit)
                        if abs(omega) < minimum_yaw:
                            omega = math.copysign(minimum_yaw, error)
                        # print(f"[turn_relative debug] iter={_debug_iter} omega_cmd={omega:.4f} rad/s")
                        self._submit_velocity([0.0, 0.0, omega])
                    self._sleep(1.0 / self.config.control_hz)
            except Exception as exc:
                reason = self._exception_reason(exc)
            except BaseException as exc:
                fatal_error = exc
                reason = type(exc).__name__
            stop_result = self._stop_impl(start_time=self._clock())
            if not stop_result["success"]:
                success = False
                reason = f"{reason}; final_stop_failed: {stop_result['reason']}"
            if fatal_error is not None:
                raise fatal_error
            return self._result(
                "turn_relative",
                success,
                reason,
                start_time,
                start_pose=start_pose,
                final_pose=final_pose,
                metrics=metrics,
                status=stop_result.get("base_status"),
            )

    def drive_straight(
        self,
        distance_m: float,
        *,
        max_speed_mps: float | None = None,
        timeout_s: float | None = None,
        distance_tolerance_m: float | None = None,
        pose_callback: Callable[[list[float]], None] | None = None,
        obstacle_check: bool = True,
    ) -> dict[str, Any]:
        """Drive a signed relative distance with ZED planar-pose feedback.

        ``obstacle_check=False`` drives without the clearance gate; only a
        caller that deliberately reproduces a method without one (the Cap-X
        baseline) passes it.
        """

        distance_m = self._finite_float("distance_m", distance_m)
        distance_tolerance = self.config.distance_tolerance_m
        if distance_tolerance_m is not None:
            distance_tolerance = self._finite_float(
                "distance_tolerance_m", distance_tolerance_m
            )
            if not 0.002 <= distance_tolerance <= self.config.distance_tolerance_m:
                raise ValueError(
                    "distance_tolerance_m must be in "
                    f"[0.002, {self.config.distance_tolerance_m:.3f}]"
                )
        direction = 1.0 if distance_m > 0 else -1.0
        distance_limit = (
            self.config.max_distance_m
            if direction > 0
            else self.config.max_reverse_distance_m
        )
        if not distance_tolerance < abs(distance_m) <= distance_limit:
            raise ValueError(
                f"abs(distance_m) must be in "
                f"({distance_tolerance:.3f}, {distance_limit:.3f}] "
                f"for {'forward' if direction > 0 else 'reverse'} motion"
            )
        speed_limit = self._bounded_speed(
            max_speed_mps,
            default=(
                self.config.default_linear_mps
                if direction > 0
                else self.config.default_reverse_mps
            ),
            minimum=self.config.min_linear_mps,
            status_key="max_linear_mps",
        )
        timeout = self._timeout(
            timeout_s,
            nominal=(
                abs(distance_m)
                / max(speed_limit, 1e-6)
                * self.config.auto_timeout_scale
                + self.config.auto_timeout_overhead_s
            ),
        )

        with self._motion_lock:
            start_time = self._clock()
            self.last_front_clearance_debug = None
            self.last_clearance = None
            self.last_visibility = None
            self.last_clearance_debug = None
            start_pose: list[float] | None = None
            final_pose: list[float] | None = None
            reason = "timeout"
            success = False
            metrics: dict[str, Any] = {}
            fatal_error: BaseException | None = None
            try:
                status = self._preflight()
                speed_limit = min(
                    speed_limit, float(status["limits"]["max_linear_mps"])
                )
                yaw_limit = float(status["limits"]["max_yaw_rad_s"])
                frame = self._frame()
                self._prepare_robot_self_filter(frame)
                self._refresh_arm_footprint(frame)
                sticky = self._new_sticky()
                start_pose = self._pose(frame)
                if pose_callback is not None:
                    pose_callback(list(start_pose))
                previous = start_pose
                settled = 0
                clearance_metrics = self._clearance_metrics(None, None)
                while self._clock() - start_time <= timeout:
                    self._raise_if_stop_requested()
                    frame = self._frame()
                    frame_age = self._frame_age_s()
                    pose = self._pose(frame)
                    if pose_callback is not None:
                        pose_callback(list(pose))
                    self._check_pose_jump(previous, pose)
                    previous = pose
                    final_pose = pose
                    dx = pose[0] - start_pose[0]
                    dy = pose[1] - start_pose[1]
                    c, s = math.cos(start_pose[2]), math.sin(start_pose[2])
                    along = c * dx + s * dy
                    cross = -s * dx + c * dy
                    remaining = direction * (distance_m - along)
                    heading_error = wrap_angle(start_pose[2] - pose[2])
                    metrics = {
                        "target_distance_m": distance_m,
                        "distance_tolerance_m": distance_tolerance,
                        "progress_m": along,
                        "remaining_m": remaining,
                        "direction": "forward" if direction > 0 else "reverse",
                        "cross_track_error_m": cross,
                        "heading_error_rad": heading_error,
                        **clearance_metrics,
                        "rear_clearance_checked": False,
                    }
                    if remaining <= distance_tolerance:
                        settled += 1
                        self._submit_velocity([0.0, 0.0, 0.0], allow_stopped=True)
                        if settled >= self.config.settle_cycles:
                            success = True
                            reason = "target_reached"
                            break
                    else:
                        settled = 0
                        speed_magnitude = min(
                            speed_limit, self.config.linear_kp * remaining
                        )
                        speed_magnitude = max(
                            self.config.min_linear_mps, speed_magnitude
                        )
                        swept = (
                            self._swept_clearance(
                                frame,
                                [direction * speed_magnitude, 0.0],
                                sticky,
                                frame_age_s=frame_age,
                            )
                            if obstacle_check
                            else None
                        )
                        legacy_clearance = (
                            self._front_clearance(frame)
                            if obstacle_check and swept is None and direction > 0
                            else None
                        )
                        clearance_metrics = self._clearance_metrics(
                            swept, legacy_clearance
                        )
                        metrics.update(clearance_metrics)
                        if swept is not None and not swept.clear:
                            reason = "obstacle_too_close"
                            break
                        if (
                            legacy_clearance is not None
                            and legacy_clearance < self.config.min_front_clearance_m
                        ):
                            reason = "obstacle_too_close"
                            break
                        speed = direction * speed_magnitude
                        omega = (
                            self.config.heading_kp * heading_error
                            - direction * self.config.cross_track_kp * cross
                        )
                        omega = float(np.clip(omega, -yaw_limit, yaw_limit))
                        self._submit_velocity([speed, 0.0, omega])
                    self._sleep(1.0 / self.config.control_hz)
            except Exception as exc:
                reason = self._exception_reason(exc)
            except BaseException as exc:
                fatal_error = exc
                reason = type(exc).__name__
            stop_result = self._stop_impl(start_time=self._clock())
            if not stop_result["success"]:
                success = False
                reason = f"{reason}; final_stop_failed: {stop_result['reason']}"
            if fatal_error is not None:
                raise fatal_error
            if self.last_clearance_debug is not None:
                metrics = {**metrics, "clearance_debug": self.last_clearance_debug}
            return self._result(
                "drive_straight",
                success,
                reason,
                start_time,
                start_pose=start_pose,
                final_pose=final_pose,
                metrics=metrics,
                status=stop_result.get("base_status"),
            )

    def drive_lateral(
        self,
        distance_m: float,
        *,
        max_speed_mps: float | None = None,
        timeout_s: float | None = None,
    ) -> dict[str, Any]:
        """Drive sideways while holding the starting ZED heading.

        Positive distance is robot-left and negative distance is robot-right.
        Both directions use the ordinary forward distance/speed bounds because
        neither sign represents reverse motion.  YOR has no side-facing depth
        sensor, so this method deliberately makes no side-clearance claim.
        """

        distance_m = self._finite_float("distance_m", distance_m)
        distance_tolerance = self.config.distance_tolerance_m
        if not distance_tolerance < abs(distance_m) <= self.config.max_distance_m:
            raise ValueError(
                f"abs(distance_m) must be in "
                f"({distance_tolerance:.3f}, {self.config.max_distance_m:.3f}] "
                "for lateral motion"
            )
        speed_limit = self._bounded_speed(
            max_speed_mps,
            default=self.config.default_linear_mps,
            minimum=self.config.min_linear_mps,
            status_key="max_linear_mps",
        )
        timeout = self._timeout(
            timeout_s,
            nominal=(
                abs(distance_m)
                / max(speed_limit, 1e-6)
                * self.config.auto_timeout_scale
                + self.config.auto_timeout_overhead_s
            ),
        )
        result = self.move_planar_relative(
            0.0,
            distance_m,
            0.0,
            max_linear_mps=speed_limit,
            max_lateral_mps=speed_limit,
            max_yaw_rad_s=self.config.default_yaw_rad_s,
            position_tolerance_m=distance_tolerance,
            yaw_tolerance_rad=self.config.yaw_tolerance_rad,
            timeout_s=timeout,
            allow_reverse=True,
        )
        result["primitive"] = "drive_lateral"
        metrics = result.setdefault("metrics", {})
        start_pose = result.get("start_pose_xy_yaw")
        final_pose = result.get("final_pose_xy_yaw")
        progress = None
        cross_track = None
        if (
            isinstance(start_pose, list)
            and len(start_pose) == 3
            and isinstance(final_pose, list)
            and len(final_pose) == 3
        ):
            dx = float(final_pose[0]) - float(start_pose[0])
            dy = float(final_pose[1]) - float(start_pose[1])
            cosine = math.cos(float(start_pose[2]))
            sine = math.sin(float(start_pose[2]))
            progress = -sine * dx + cosine * dy
            cross_track = cosine * dx + sine * dy
        direction = 1.0 if distance_m > 0.0 else -1.0
        metrics.update(
            {
                "target_distance_m": distance_m,
                "distance_tolerance_m": distance_tolerance,
                "progress_m": progress,
                "remaining_m": (
                    None
                    if progress is None
                    else direction * (distance_m - progress)
                ),
                "direction": "left" if direction > 0.0 else "right",
                "cross_track_error_m": cross_track,
                "lateral_clearance_checked": False,
            }
        )
        return result

    def move_planar_relative(
        self,
        forward_m: float,
        left_m: float,
        yaw_rad: float,
        *,
        max_linear_mps: float = 0.06,
        max_lateral_mps: float = 0.04,
        max_yaw_rad_s: float = 0.20,
        position_tolerance_m: float = 0.025,
        yaw_tolerance_rad: float = math.radians(2.5),
        timeout_s: float = 20.0,
        allow_reverse: bool = False,
        target_pose_world: Sequence[float] | None = None,
        motion_guard: Callable[[Any, dict[str, float]], Any] | None = None,
        obstacle_check: bool = True,
    ) -> dict[str, Any]:
        """Track one nearby SE(2) goal with simultaneous holonomic velocity.

        This controller-only method is intentionally not registered as an LLM
        primitive.  A semantic controller must first select and safety-check
        the local goal; this method then closes the loop using fresh ZED pose
        feedback and the Pi's short velocity lease. ``obstacle_check=False``
        skips the clearance gate, for a caller that deliberately reproduces
        a method without one.
        """

        forward_m = self._finite_float("forward_m", forward_m)
        left_m = self._finite_float("left_m", left_m)
        yaw_rad = self._finite_float("yaw_rad", yaw_rad)
        max_linear_mps = self._finite_float("max_linear_mps", max_linear_mps)
        max_lateral_mps = self._finite_float("max_lateral_mps", max_lateral_mps)
        max_yaw_rad_s = self._finite_float("max_yaw_rad_s", max_yaw_rad_s)
        position_tolerance_m = self._finite_float(
            "position_tolerance_m", position_tolerance_m
        )
        yaw_tolerance_rad = self._finite_float(
            "yaw_tolerance_rad", yaw_tolerance_rad
        )
        timeout_s = self._finite_float("timeout_s", timeout_s)
        if target_pose_world is not None:
            absolute_target = np.asarray(target_pose_world, dtype=np.float64)
            if absolute_target.shape != (3,) or not np.all(
                np.isfinite(absolute_target)
            ):
                raise ValueError("target_pose_world must contain finite [x, y, yaw]")
        else:
            absolute_target = None
        distance = math.hypot(forward_m, left_m)
        if distance > self.config.max_distance_m:
            raise ValueError(
                f"planar goal distance must be <= {self.config.max_distance_m:.3f} m"
            )
        if forward_m < -position_tolerance_m and not allow_reverse:
            raise ValueError("reverse planar motion requires allow_reverse=True")
        if abs(yaw_rad) > self.config.max_turn_rad:
            raise ValueError(
                f"abs(yaw_rad) must be <= {self.config.max_turn_rad:.3f}"
            )
        if not 0.0 < max_lateral_mps <= max_linear_mps:
            raise ValueError("max_lateral_mps must be in (0, max_linear_mps]")
        if min(
            max_linear_mps,
            max_yaw_rad_s,
            position_tolerance_m,
            yaw_tolerance_rad,
            timeout_s,
        ) <= 0.0:
            raise ValueError("planar tracking limits and timeout must be positive")
        if timeout_s > self.config.max_duration_s:
            raise ValueError(
                f"timeout_s must be <= {self.config.max_duration_s:.1f}"
            )

        with self._motion_lock:
            start_time = self._clock()
            start_pose: list[float] | None = None
            final_pose: list[float] | None = None
            reason = "timeout"
            success = False
            metrics: dict[str, Any] = {}
            fatal_error: BaseException | None = None
            try:
                status = self._preflight()
                hardware_linear = float(status["limits"]["max_linear_mps"])
                hardware_yaw = float(status["limits"]["max_yaw_rad_s"])
                linear_limit = min(max_linear_mps, hardware_linear)
                # Diagonal normalization can round hypot(vx, vy) a few ULPs
                # above an exactly advertised RPC limit (for example
                # 0.18000000000000002 > 0.18). Keep negligible numerical
                # headroom so a valid holonomic command is never rejected as
                # overspeed by a strict server comparison.
                command_linear_limit = linear_limit * (1.0 - 1e-6)
                lateral_limit = min(max_lateral_mps, command_linear_limit)
                yaw_limit = min(max_yaw_rad_s, hardware_yaw)
                frame = self._frame()
                self._prepare_robot_self_filter(frame)
                self._refresh_arm_footprint(frame)
                sticky = self._new_sticky()
                self.last_clearance = None
                self.last_front_clearance_debug = None
                self.last_clearance_debug = None
                start_pose = self._pose(frame)
                if absolute_target is None:
                    c0, s0 = math.cos(start_pose[2]), math.sin(start_pose[2])
                    target_x = start_pose[0] + c0 * forward_m - s0 * left_m
                    target_y = start_pose[1] + s0 * forward_m + c0 * left_m
                    target_yaw = wrap_angle(start_pose[2] + yaw_rad)
                else:
                    target_x = float(absolute_target[0])
                    target_y = float(absolute_target[1])
                    target_yaw = wrap_angle(float(absolute_target[2]))
                    absolute_distance = math.hypot(
                        target_x - start_pose[0], target_y - start_pose[1]
                    )
                    absolute_yaw_delta = abs(
                        wrap_angle(target_yaw - start_pose[2])
                    )
                    if absolute_distance > self.config.max_distance_m:
                        raise ValueError(
                            "absolute planar goal exceeds the configured distance limit"
                        )
                    if absolute_yaw_delta > self.config.max_turn_rad:
                        raise ValueError(
                            "absolute planar goal exceeds the configured turn limit"
                        )
                    start_cosine = math.cos(start_pose[2])
                    start_sine = math.sin(start_pose[2])
                    initial_forward_error = (
                        start_cosine * (target_x - start_pose[0])
                        + start_sine * (target_y - start_pose[1])
                    )
                    if (
                        initial_forward_error < -position_tolerance_m
                        and not allow_reverse
                    ):
                        raise ValueError(
                            "absolute reverse planar motion requires allow_reverse=True"
                        )
                    if (
                        initial_forward_error < -position_tolerance_m
                        and absolute_distance > self.config.max_reverse_distance_m
                    ):
                        raise ValueError(
                            "absolute reverse planar goal exceeds the reverse distance limit"
                        )
                previous = start_pose
                settled = 0
                clearance_metrics = self._clearance_metrics(None, None)
                while self._clock() - start_time <= timeout_s:
                    self._raise_if_stop_requested()
                    frame = self._frame()
                    frame_age = self._frame_age_s()
                    pose = self._pose(frame)
                    self._check_pose_jump(previous, pose)
                    previous = pose
                    final_pose = pose
                    world_dx = target_x - pose[0]
                    world_dy = target_y - pose[1]
                    cosine, sine = math.cos(pose[2]), math.sin(pose[2])
                    forward_error = cosine * world_dx + sine * world_dy
                    left_error = -sine * world_dx + cosine * world_dy
                    position_error = math.hypot(forward_error, left_error)
                    yaw_error = wrap_angle(target_yaw - pose[2])
                    guard_state = {
                        "forward_error_m": float(forward_error),
                        "left_error_m": float(left_error),
                        "position_error_m": float(position_error),
                        "yaw_error_rad": float(yaw_error),
                    }
                    metrics = {
                        "target_relative_forward_m": forward_m,
                        "target_relative_left_m": left_m,
                        "target_relative_yaw_rad": yaw_rad,
                        "target_pose_world": [target_x, target_y, target_yaw],
                        "command_linear_limit_mps": command_linear_limit,
                        "minimum_linear_command_mps": min(
                            self.config.min_linear_mps,
                            command_linear_limit,
                        ),
                        "minimum_yaw_command_rad_s": min(
                            self.config.min_yaw_rad_s,
                            yaw_limit,
                        ),
                        **guard_state,
                        **clearance_metrics,
                        "holonomic": True,
                    }
                    if (
                        position_error <= position_tolerance_m
                        and abs(yaw_error) <= yaw_tolerance_rad
                    ):
                        settled += 1
                        self._submit_velocity(
                            [0.0, 0.0, 0.0], allow_stopped=True
                        )
                        if settled >= self.config.settle_cycles:
                            success = True
                            reason = "target_reached"
                            break
                    else:
                        settled = 0
                        if motion_guard is not None:
                            guard_result = motion_guard(frame, guard_state)
                            if isinstance(guard_result, Mapping):
                                metrics["motion_guard"] = dict(guard_result)
                            if guard_result is False or (
                                isinstance(guard_result, Mapping)
                                and not bool(guard_result.get("clear", False))
                            ):
                                detail = (
                                    guard_result.get("reason", "motion_guard_blocked")
                                    if isinstance(guard_result, Mapping)
                                    else "motion_guard_blocked"
                                )
                                reason = f"motion_guard_blocked:{detail}"
                                break
                        if position_error <= position_tolerance_m:
                            # Do not let the static-friction floor create
                            # translation while only yaw needs correction.
                            forward = 0.0
                            lateral = 0.0
                        else:
                            forward = float(
                                np.clip(
                                    self.config.linear_kp * forward_error,
                                    (
                                        -command_linear_limit
                                        if allow_reverse
                                        else 0.0
                                    ),
                                    command_linear_limit,
                                )
                            )
                            lateral = float(
                                np.clip(
                                    self.config.linear_kp * left_error,
                                    -lateral_limit,
                                    lateral_limit,
                                )
                            )
                            norm = math.hypot(forward, lateral)
                            if norm > command_linear_limit:
                                scale = command_linear_limit / norm
                                forward *= scale
                                lateral *= scale
                            elif 0.0 < norm < min(
                                self.config.min_linear_mps,
                                command_linear_limit,
                            ):
                                scale = min(
                                    self.config.min_linear_mps,
                                    command_linear_limit,
                                ) / norm
                                forward *= scale
                                lateral *= scale
                        swept = (
                            self._swept_clearance(
                                frame, [forward, lateral], sticky, frame_age_s=frame_age
                            )
                            if obstacle_check
                            else None
                        )
                        legacy_clearance = (
                            self._front_clearance(frame)
                            if obstacle_check and swept is None and forward > 0.0
                            else None
                        )
                        clearance_metrics = self._clearance_metrics(
                            swept, legacy_clearance
                        )
                        metrics.update(clearance_metrics)
                        if swept is not None and not swept.clear:
                            reason = "obstacle_too_close"
                            break
                        if (
                            legacy_clearance is not None
                            and legacy_clearance < self.config.min_front_clearance_m
                        ):
                            reason = "obstacle_too_close"
                            break
                        if abs(yaw_error) <= yaw_tolerance_rad:
                            # Preserve heading once it is within tolerance
                            # while the translation axes finish converging.
                            omega = 0.0
                        else:
                            omega = float(
                                np.clip(
                                    self.config.yaw_kp * yaw_error,
                                    -yaw_limit,
                                    yaw_limit,
                                )
                            )
                            minimum_yaw = min(
                                self.config.min_yaw_rad_s,
                                yaw_limit,
                            )
                            if 0.0 < abs(omega) < minimum_yaw:
                                omega = math.copysign(minimum_yaw, yaw_error)
                        self._submit_velocity([forward, lateral, omega])
                    self._sleep(1.0 / self.config.control_hz)
            except Exception as exc:
                reason = self._exception_reason(exc)
            except BaseException as exc:
                fatal_error = exc
                reason = type(exc).__name__
            stop_result = self._stop_impl(start_time=self._clock())
            if not stop_result["success"]:
                success = False
                reason = f"{reason}; final_stop_failed: {stop_result['reason']}"
            if fatal_error is not None:
                raise fatal_error
            if self.last_clearance_debug is not None:
                metrics = {**metrics, "clearance_debug": self.last_clearance_debug}
            return self._result(
                "move_planar_relative",
                success,
                reason,
                start_time,
                start_pose=start_pose,
                final_pose=final_pose,
                metrics=metrics,
                status=stop_result.get("base_status"),
            )

    def _preflight(self) -> dict[str, Any]:
        self._raise_if_stop_requested()
        status = self.env.base_status()
        required = {"lease_active", "last_velocity", "estop_latched", "limits"}
        missing = required - set(status)
        if missing:
            raise RuntimeError(f"base RPC status missing fields: {sorted(missing)}")
        limits = status["limits"]
        if not {"lease_s", "max_linear_mps", "max_yaw_rad_s"} <= set(limits):
            raise RuntimeError("base RPC does not advertise the required safety limits")
        values = [
            float(limits["lease_s"]),
            float(limits["max_linear_mps"]),
            float(limits["max_yaw_rad_s"]),
        ]
        if not all(math.isfinite(value) and value > 0 for value in values):
            raise RuntimeError("base RPC advertised invalid safety limits")
        if values[0] > 0.30:
            raise RuntimeError("Pi velocity lease exceeds the 0.30 s safety bound")
        if status["estop_latched"]:
            raise RuntimeError("emergency stop is latched")
        if status["lease_active"]:
            raise RuntimeError("another controller has an active base velocity lease")
        return status

    def _frame(self):
        return self.env.navigation_frame(max_age_s=self.config.pose_max_age_s)

    @staticmethod
    def _pose(frame: Any) -> list[float]:
        pose = getattr(frame, "planar_pose", None)
        if pose is None or not bool(getattr(pose, "valid", False)):
            raise RuntimeError("pose_invalid")
        values = [float(pose.x_m), float(pose.y_m), wrap_angle(float(pose.yaw_rad))]
        if not all(math.isfinite(value) for value in values):
            raise RuntimeError("pose_nonfinite")
        return values

    def _check_pose_jump(self, previous: list[float], current: list[float]) -> None:
        translation = math.hypot(current[0] - previous[0], current[1] - previous[1])
        rotation = abs(wrap_angle(current[2] - previous[2]))
        if translation > self.config.max_pose_jump_m:
            raise RuntimeError(f"localization_translation_jump_{translation:.3f}m")
        if rotation > self.config.max_yaw_jump_rad:
            raise RuntimeError(f"localization_yaw_jump_{rotation:.3f}rad")

    # ------------------------------------------------------------------
    # Footprint-swept clearance
    # ------------------------------------------------------------------

    def _footprint_config(self) -> FootprintConfig:
        if self._footprint is None:
            self._footprint = FootprintConfig.from_mapping(self.config.footprint)
        return self._footprint

    def _static_footprint_config(self) -> FootprintConfig:
        if self._static_footprint is None:
            self._static_footprint = FootprintConfig.from_mapping(self.config.footprint)
        return self._static_footprint

    def _refresh_arm_footprint(self, frame: Any) -> None:
        """Rebuild the arm layer from where the arms actually are right now.

        Called once per primitive call, straight after the depth self-filter
        has been updated for the current joint state, so the layer names stay
        fixed for the whole call and :class:`StickyOccupancy` keeps matching.
        Any failure leaves the configured static layers in place: the gate is
        then exactly as conservative as before this method existed.
        """

        static = self._static_footprint_config()
        self._footprint = static
        self.last_footprint_debug = {
            "source": "static",
            "layers": len(static.layers),
        }
        if not self.config.arm_footprint_from_spheres:
            return
        target = static.layer(self.config.arm_footprint_layer)
        if target is None or self._robot_self_filter is None:
            self.last_footprint_debug["reason"] = (
                "layer_not_configured" if target is None else "self_filter_absent"
            )
            return
        attached_getter = getattr(self.env, "robot_self_filter_attached_objects", None)
        if callable(attached_getter) and attached_getter():
            # A grasped payload is masked out of the depth image but is not in
            # the sphere envelope, so a live outline would model it nowhere.
            # The static layer covered anything held between the grippers.
            self.last_footprint_debug["reason"] = "attached_object"
            return
        covered = set(self._robot_self_filter.camera_sphere_arms())
        if covered != set(self._robot_self_filter.arm_from_camera):
            # With require_both_arms false the envelope can describe one arm of
            # a two-armed robot; half a collision model is worse than none.
            self.last_footprint_debug["reason"] = "partial_arm_status"
            return
        spheres_camera = self._robot_self_filter.camera_spheres()
        if not len(spheres_camera):
            self.last_footprint_debug["reason"] = "no_spheres"
            return
        depth = np.asarray(getattr(frame, "depth_m", None))
        if depth.ndim == 3 and depth.shape[-1] == 1:
            depth = depth[..., 0]
        if depth.ndim != 2:
            self.last_footprint_debug["reason"] = "depth_invalid"
            return
        ground = self._ground_clearance_config(frame, depth.shape)
        if ground is None:
            self.last_footprint_debug["reason"] = "camera_geometry_absent"
            return
        intrinsics, camera_height, down, *_ = ground
        geometry = CameraGeometry(
            intrinsics=intrinsics,
            camera_height_m=camera_height,
            down_camera_xyz=down,
        )
        spheres = spheres_to_base(spheres_camera, geometry, static)
        bands = arm_band_layers(
            spheres,
            z_min=target.z_min,
            z_max=target.z_max,
            band_m=self.config.arm_footprint_band_m,
            forward_margin_m=self.config.arm_footprint_forward_margin_m,
            lateral_margin_m=self.config.arm_footprint_lateral_margin_m,
            z_margin_m=self.config.arm_footprint_z_margin_m,
            always_occupied_xy=static.body_polygon_xy,
            structure_margin_m=self.config.arm_footprint_structure_margin_m,
            name=target.name,
            core_z_margin_m=self.config.arm_footprint_core_z_margin_m,
            underside_forward_margin_m=(
                self.config.arm_footprint_underside_forward_margin_m
            ),
            underside_sweep_margin_m=(
                self.config.arm_footprint_underside_clearance_margin_m
            ),
        )
        if not bands:
            self.last_footprint_debug["reason"] = "bands_unavailable"
            return
        kept = [layer for layer in static.layers if layer.name != target.name]
        self._footprint = static.with_layers([*kept, *bands])
        forward = np.asarray([1.0, 0.0], dtype=np.float64)
        fringe_prefix = f"{target.name}_fringe_"
        core_bands = [layer for layer in bands if not layer.name.startswith(fringe_prefix)]
        fringe_bands = [layer for layer in bands if layer.name.startswith(fringe_prefix)]
        self.last_footprint_debug = {
            "source": "arm_spheres",
            "layers": len(self._footprint.layers),
            "spheres": int(len(spheres)),
            "replaced_layer": target.name,
            "band_m": float(self.config.arm_footprint_band_m),
            "forward_margin_m": float(self.config.arm_footprint_forward_margin_m),
            "lateral_margin_m": float(self.config.arm_footprint_lateral_margin_m),
            "static_front_m": target.extent_along(forward),
            "front_by_band_m": {
                f"{layer.z_min:.2f}": round(layer.extent_along(forward), 4)
                for layer in core_bands
            },
        }
        if fringe_bands:
            self.last_footprint_debug.update(
                {
                    "core_z_margin_m": float(self.config.arm_footprint_core_z_margin_m),
                    "z_margin_m": float(self.config.arm_footprint_z_margin_m),
                    "underside_forward_margin_m": float(
                        self.config.arm_footprint_forward_margin_m
                        if self.config.arm_footprint_underside_forward_margin_m is None
                        else self.config.arm_footprint_underside_forward_margin_m
                    ),
                    "underside_clearance_margin_m": fringe_bands[0].sweep_margin_m,
                    "fringe_front_by_band_m": {
                        f"{layer.z_min:.2f}": round(layer.extent_along(forward), 4)
                        for layer in fringe_bands
                    },
                }
            )

    def _new_sticky(self) -> StickyOccupancy | None:
        if not self.config.swept_clearance:
            return None
        return StickyOccupancy(
            self._footprint_config(),
            cell_m=self.config.clearance_cell_m,
            cell_min_points=self.config.clearance_cell_min_points,
        )

    def _frame_age_s(self) -> float:
        """Arrival age of the frame used for a decision, never below zero.

        When the transport cannot report it, the configured maximum pose age
        is assumed so the horizon errs on the long side.
        """

        getter = getattr(self.env, "navigation_frame_age_s", None)
        age = None
        if callable(getter):
            try:
                age = getter()
            except Exception:  # noqa: BLE001 - fall back to the configured bound
                age = None
        if age is None or not math.isfinite(float(age)):
            return float(self.config.pose_max_age_s)
        return max(0.0, float(age))

    def _swept_clearance(
        self,
        frame: Any,
        velocity_xy: Sequence[float],
        sticky: StickyOccupancy | None,
        *,
        frame_age_s: float | None = None,
    ) -> ClearanceResult | None:
        """Footprint-swept check for one intended base velocity.

        Returns ``None`` when the check is disabled or the camera geometry is
        not configured; callers then use the legacy central-cone gate.  The
        depth image is only consulted when the command has a forward or
        lateral component the camera can observe; a pure reverse command is
        checked against remembered cells only, and a zero translation is
        always clear (yaw is not swept in this phase).
        """

        if not self.config.swept_clearance:
            return None
        vx, vy = float(velocity_xy[0]), float(velocity_xy[1])
        speed = math.hypot(vx, vy)
        age = self._frame_age_s() if frame_age_s is None else max(0.0, float(frame_age_s))
        footprint = self._footprint_config()
        horizon, brake, sweep = sweep_distance(
            speed,
            frame_age_s=age,
            command_latency_s=self.config.command_latency_s,
            lease_s=self.config.lease_s,
            brake_accel_mps2=self.config.brake_accel_mps2,
            margin_m=self.config.clearance_margin_m,
        )
        pose = self._pose(frame)
        uses_camera = vx > 1e-9 or abs(vy) > 1e-9
        # Read memory BEFORE this cycle's cells are added to it. Remembering a
        # cell quantizes it onto a 0.025 m odom grid and reads it back at the
        # cell center, so a cell added and re-read in the same cycle returns
        # displaced by up to half a cell. check_cells takes the minimum over
        # live and remembered cells, so that copy could only ever shorten the
        # free distance of a cell the same frame had just measured as clear.
        remembered: dict[str, np.ndarray] = {}
        if sticky is not None:
            remembered = sticky.cells_base(pose)
        memory_cells = int(sum(len(value) for value in remembered.values()))
        cells: dict[str, np.ndarray] = {
            layer.name: np.zeros((0, 2), dtype=np.float64) for layer in footprint.layers
        }
        valid_fraction = 1.0
        if uses_camera:
            depth = np.asarray(getattr(frame, "depth_m", None), dtype=np.float32)
            if depth.ndim == 3 and depth.shape[-1] == 1:
                depth = depth[..., 0]
            if depth.ndim != 2 or min(depth.shape) < 8:
                raise RuntimeError("depth_invalid_shape")
            ground = self._ground_clearance_config(frame, depth.shape)
            if ground is None:
                return None
            if self._robot_self_filter is not None:
                depth = self._robot_self_filter.filtered_depth(depth)
            intrinsics, camera_height, down, _, _, max_depth, _, _ = ground
            geometry = CameraGeometry(
                intrinsics=intrinsics,
                camera_height_m=camera_height,
                down_camera_xyz=down,
            )
            points = depth_to_base_points(
                depth,
                geometry,
                footprint,
                stride=self.config.clearance_pixel_stride,
                max_depth_m=min(float(max_depth), self.config.clearance_max_depth_m),
            )
            valid_fraction = points.valid_fraction
            self.last_clearance_debug = {
                "valid_depth_fraction": valid_fraction,
                "valid_pixels": points.valid_pixels,
                "sampled_pixels": points.sampled_pixels,
                "frame_age_s": age,
                "sweep_distance_m": sweep,
            }
            if valid_fraction < self.config.min_valid_depth_fraction:
                # The ZED returns no depth for a plain surface at close range;
                # a mostly empty image is treated as unknown, never as free.
                raise RuntimeError("depth_mostly_invalid")
            cells = occupied_cells(
                points.xyz,
                footprint,
                cell_m=self.config.clearance_cell_m,
                cell_min_points=self.config.clearance_cell_min_points,
                z_tolerance_m=self.config.layer_z_tolerance_m,
            )
            if (
                sticky is not None
                and memory_cells
                and speed > 1e-9
                and self.config.memory_seen_through_clearing
            ):
                # Before this cycle's cells are added: the forget masks index
                # the rows cells_base returned, and add_cells re-sorts them.
                remembered, memory_cells = self._forget_seen_through(
                    sticky,
                    remembered,
                    depth,
                    geometry,
                    footprint,
                    (vx / speed, vy / speed),
                    sweep,
                )
            if sticky is not None:
                sticky.add_cells(cells, pose)
            if speed > 1e-9:
                visibility = sweep_visibility(
                    points,
                    geometry,
                    footprint,
                    (vx, vy),
                    sweep,
                    image_height=int(depth.shape[0]),
                )
                self.last_visibility = visibility.summary()
                self.last_clearance_debug["visibility"] = self.last_visibility
                if (
                    visibility.lane_bins
                    and visibility.unknown_fraction > self.config.max_unknown_sweep_fraction
                ):
                    # No floor was seen behind the area about to be entered on
                    # most of the columns that look into it: unobserved space,
                    # not free space (a plain wall at close range, a black
                    # table top, glass).  Obstacles that were seen are handled
                    # by the cell check below, which runs first on the result.
                    unknown_error = PrimitiveReason(
                        DEPTH_UNKNOWN_IN_SWEEP_REASON
                    )
                else:
                    unknown_error = None
            else:
                unknown_error = None
        elif self._ground_clearance_config(frame, (8, 8)) is None:
            # No camera geometry at all: keep the legacy behaviour (no check).
            return None
        else:
            unknown_error = None
        # Captured before the remembered cells are appended, so a blocking
        # cell's row index says whether this frame saw it or the memory did.
        live_cell_counts = {name: int(len(value)) for name, value in cells.items()}
        if memory_cells:
            if set(remembered) != set(cells):
                # The memory is keyed by layer name and the footprint is fixed
                # for a whole call, so this cannot happen; dropping the
                # mismatched layers silently would forget obstacles instead.
                raise RuntimeError("clearance_memory_layer_mismatch")
            cells = {
                name: np.vstack([cells[name], remembered[name]]) for name in cells
            }
        result = check_cells(
            cells,
            footprint,
            velocity_xy,
            frame_age_s=age,
            command_latency_s=self.config.command_latency_s,
            lease_s=self.config.lease_s,
            brake_accel_mps2=self.config.brake_accel_mps2,
            margin_m=self.config.clearance_margin_m,
            valid_fraction=valid_fraction,
            memory_cells=memory_cells,
            live_cell_counts=live_cell_counts,
        )
        self.last_clearance = result
        if result.clear and unknown_error is not None:
            raise unknown_error
        return result

    def _forget_seen_through(
        self,
        sticky: StickyOccupancy,
        remembered: dict[str, np.ndarray],
        depth: np.ndarray,
        geometry: CameraGeometry,
        footprint: FootprintConfig,
        direction: tuple[float, float],
        sweep: float,
    ) -> tuple[dict[str, np.ndarray], int]:
        """Drop remembered cells this frame looks straight through.

        Only cells ahead along the command, where they could stop it, are
        checked, nearest first and a bounded number per cycle. Returns the
        remembered cells that are kept and how many there are.
        """

        unit = np.asarray(direction, dtype=np.float64)
        lookahead = max(float(sweep), float(self.config.memory_seen_through_lookahead_m))
        candidates: list[tuple[float, str, int]] = []
        for layer in footprint.layers:
            cells = remembered.get(layer.name)
            if cells is None or not len(cells):
                continue
            ahead = points_in_convex_polygon(
                cells, layer.swept(unit * lookahead)
            ) & ~points_in_convex_polygon(cells, footprint.body_polygon_xy)
            rows = np.flatnonzero(ahead)
            distances = cells[rows] @ unit
            candidates.extend(
                (float(distance), layer.name, int(row))
                for distance, row in zip(distances, rows)
            )
        candidates.sort(key=lambda item: item[0])
        checked = candidates[:SEEN_THROUGH_MAX_CELLS_PER_CYCLE]

        rows_by_layer: dict[str, list[int]] = {}
        for _, name, row in checked:
            rows_by_layer.setdefault(name, []).append(row)
        masks = {name: np.zeros(len(value), dtype=bool) for name, value in remembered.items()}
        for layer in footprint.layers:
            rows = rows_by_layer.get(layer.name)
            if not rows:
                continue
            index = np.asarray(rows, dtype=np.int64)
            seen = seen_through_cells(
                remembered[layer.name][index],
                layer,
                depth,
                geometry,
                footprint,
                cell_m=self.config.clearance_cell_m,
                margin_m=self.config.memory_seen_through_margin_m,
                min_valid_fraction=self.config.memory_seen_through_min_valid_fraction,
            )
            masks[layer.name][index[seen]] = True

        forgotten = sticky.forget(masks) if any(mask.any() for mask in masks.values()) else 0
        if isinstance(self.last_clearance_debug, dict):
            self.last_clearance_debug["memory_seen_through"] = {
                "checked_cells": len(checked),
                "not_checked_cells": len(candidates) - len(checked),
                "forgotten_cells": int(forgotten),
            }
        kept = {name: value[~masks[name]] for name, value in remembered.items()}
        return kept, int(sum(len(value) for value in kept.values()))

    def _clearance_metrics(
        self, swept: ClearanceResult | None, legacy_clearance: float | None
    ) -> dict[str, Any]:
        if swept is not None:
            summary = swept.summary()
            if self.last_visibility is not None:
                summary["visibility"] = dict(self.last_visibility)
            if self.last_footprint_debug is not None:
                # Which footprint decided this, and how far forward the robot
                # was modelled to reach in each height band. Without it a
                # trace cannot say where the arms were when the gate fired.
                summary["footprint"] = dict(self.last_footprint_debug)
            return {
                "front_clearance_m": swept.min_free_distance_m,
                "front_clearance_debug": None,
                # A flat one-liner beside the numbers: metrics are nested one
                # level deeper inside a guard block than the trace summariser
                # descends, and a plain string survives that where a dict of
                # per-layer evidence does not.
                "clearance_blocked_by": swept.describe_block(),
                "clearance": summary,
            }
        return {
            "front_clearance_m": legacy_clearance,
            "front_clearance_debug": self.last_front_clearance_debug,
            "clearance": {"mode": "legacy_central_depth"},
        }

    def _front_clearance(self, frame: Any) -> float | None:
        depth = np.asarray(getattr(frame, "depth_m", None), dtype=np.float32)
        if depth.ndim == 3 and depth.shape[-1] == 1:
            depth = depth[..., 0]
        if depth.ndim != 2 or min(depth.shape) < 8:
            raise RuntimeError("depth_invalid_shape")
        filter_debug = None
        if self._robot_self_filter is not None:
            depth = self._robot_self_filter.filtered_depth(depth)
            filter_debug = dict(self._robot_self_filter.last_debug)
        height, width = depth.shape
        row0, row1 = int(0.30 * height), int(0.78 * height)
        column0, column1 = int(0.35 * width), int(0.65 * width)
        roi = depth[row0:row1, column0:column1]
        raw_valid = np.isfinite(roi) & (roi > 0.05)
        if np.count_nonzero(raw_valid) < self.config.depth_min_valid_pixels:
            raise RuntimeError("depth_insufficient_valid_pixels")

        ground = self._ground_clearance_config(frame, depth.shape)
        if ground is None:
            values = roi[raw_valid]
            clearance = float(
                np.percentile(values, self.config.depth_percentile)
            )
            self.last_front_clearance_debug = {
                "mode": "raw_depth_percentile",
                "sensor_pixels": int(values.size),
                "ground_filtered_pixels": 0,
                "height_candidate_pixels": int(values.size),
                "robot_self_filter": filter_debug,
            }
            return clearance

        (
            intrinsics,
            camera_height,
            down,
            min_height,
            max_height,
            max_depth,
            ground_source,
            ground_age_s,
        ) = ground
        roi_valid = raw_valid & (roi <= max_depth)
        local_rows, local_columns = np.nonzero(roi_valid)
        if local_rows.size < self.config.depth_min_valid_pixels:
            raise RuntimeError("depth_insufficient_valid_pixels")
        rows = local_rows + row0
        columns = local_columns + column0
        z = depth[rows, columns].astype(np.float64)
        points_camera = np.column_stack(
            [
                (columns - intrinsics[0, 2]) * z / intrinsics[0, 0],
                (rows - intrinsics[1, 2]) * z / intrinsics[1, 1],
                z,
            ]
        )
        height_above_floor = camera_height - points_camera @ down
        optical_forward = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
        planar_forward = optical_forward - down * float(
            np.dot(optical_forward, down)
        )
        forward_norm = float(np.linalg.norm(planar_forward))
        if forward_norm <= 1e-6:
            raise RuntimeError("ground_clearance_forward_axis_invalid")
        planar_forward /= forward_norm
        forward = points_camera @ planar_forward
        structural = (
            (height_above_floor >= min_height)
            & (height_above_floor <= max_height)
            & (forward > 0.0)
        )
        candidates = forward[structural]
        self.last_front_clearance_debug = {
            "mode": "calibrated_floor_height",
            "sensor_pixels": int(local_rows.size),
            "ground_filtered_pixels": int(
                np.count_nonzero(height_above_floor < min_height)
            ),
            "height_candidate_pixels": int(candidates.size),
            "minimum_obstacle_height_m": float(min_height),
            "ground_camera_height_m": float(camera_height),
            "ground_plane_source": ground_source,
            "ground_plane_age_s": ground_age_s,
            "robot_self_filter": filter_debug,
        }
        if candidates.size == 0:
            return None
        return float(np.percentile(candidates, self.config.depth_percentile))

    def nav2_start_clearance_status(
        self,
        *,
        radius_m: float,
        point_stride: int = 3,
        max_points: int = 3,
    ) -> dict[str, Any]:
        """Mirror Nav2's camera-centered stop-zone test before sending a goal.

        The Nav2 collision monitor receives the same self-filtered ZED depth and
        treats points 0.10-1.45 m above the floor inside the configured circular
        footprint as a hard stop.  Checking that condition while the base is
        already stopped lets the policy receive an actionable failure instead
        of waiting through every Nav2 recovery while all velocity is suppressed.
        """

        radius = self._finite_float("radius_m", radius_m)
        if radius <= 0.0:
            raise ValueError("radius_m must be positive")
        stride = int(point_stride)
        threshold = int(max_points)
        if stride < 1:
            raise ValueError("point_stride must be positive")
        if threshold < 0:
            raise ValueError("max_points must be nonnegative")

        with self._motion_lock:
            frame = self._frame()
            self._prepare_robot_self_filter(frame)
            depth = np.asarray(getattr(frame, "depth_m", None), dtype=np.float32)
            if depth.ndim == 3 and depth.shape[-1] == 1:
                depth = depth[..., 0]
            if depth.ndim != 2 or min(depth.shape) < 8:
                raise RuntimeError("depth_invalid_shape")
            filter_debug = None
            if self._robot_self_filter is not None:
                depth = self._robot_self_filter.filtered_depth(depth)
                filter_debug = dict(self._robot_self_filter.last_debug)

            ground = self._ground_clearance_config(frame, depth.shape)
            if ground is None:
                raise RuntimeError("nav2_start_clearance_ground_config_unavailable")
            (
                intrinsics,
                camera_height,
                down,
                min_height,
                max_height,
                max_depth,
                ground_source,
                ground_age_s,
            ) = ground

            height, width = depth.shape
            rows, columns = np.mgrid[0:height:stride, 0:width:stride]
            z = depth[rows, columns]
            valid = np.isfinite(z) & (z > 0.10) & (z <= max_depth)
            if not np.any(valid):
                raise RuntimeError("depth_insufficient_valid_pixels")
            z = z[valid].astype(np.float64)
            sampled_rows = rows[valid].astype(np.float64)
            sampled_columns = columns[valid].astype(np.float64)
            points_camera = np.column_stack(
                [
                    (sampled_columns - intrinsics[0, 2]) * z / intrinsics[0, 0],
                    (sampled_rows - intrinsics[1, 2]) * z / intrinsics[1, 1],
                    z,
                ]
            )
            height_above_floor = camera_height - points_camera @ down
            optical_forward = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
            planar_forward = optical_forward - down * float(
                np.dot(optical_forward, down)
            )
            forward_norm = float(np.linalg.norm(planar_forward))
            if forward_norm <= 1e-6:
                raise RuntimeError("ground_clearance_forward_axis_invalid")
            planar_forward /= forward_norm
            planar_left = np.cross(planar_forward, down)
            planar_left /= np.linalg.norm(planar_left)
            forward = points_camera @ planar_forward
            left = points_camera @ planar_left
            planar_distance = np.hypot(forward, left)
            structural = (
                (height_above_floor >= min_height)
                & (height_above_floor <= max_height)
            )
            # The generated Nav2 stop zone is a regular 16-gon.  Its apothem
            # is an inner, false-positive-free approximation of that polygon.
            inner_radius = radius * math.cos(math.pi / 16.0)
            inside = structural & (planar_distance <= inner_radius)
            obstacle_points = int(np.count_nonzero(inside))
            structural_distances = planar_distance[structural]
            return {
                "available": True,
                "blocked": obstacle_points > threshold,
                "robot_radius_m": radius,
                "conservative_inner_radius_m": inner_radius,
                "obstacle_points": obstacle_points,
                "max_points": threshold,
                "point_stride": stride,
                "minimum_planar_obstacle_distance_m": (
                    None
                    if structural_distances.size == 0
                    else float(np.min(structural_distances))
                ),
                "ground_plane_source": ground_source,
                "ground_plane_age_s": ground_age_s,
                "robot_self_filter": filter_debug,
            }

    def _prepare_robot_self_filter(self, frame: Any) -> None:
        manipulation = getattr(self.env, "manipulation_config", None)
        if not isinstance(manipulation, Mapping):
            return
        if not self._robot_self_filter_configured:
            config = RobotSelfFilterConfig.from_mapping(
                manipulation.get("robot_self_filter")
            )
            self._robot_self_filter_configured = True
            if config is None:
                return
            calibrations = {
                name: manipulation.get(f"{name}_arm_from_camera")
                for name in ("left", "right")
                if manipulation.get(f"{name}_arm_from_camera") is not None
            }
            self._robot_self_filter = RobotDepthSelfFilter(
                config, arm_from_camera=calibrations
            )
        if self._robot_self_filter is None:
            return
        depth = np.asarray(getattr(frame, "depth_m", None))
        if depth.ndim == 3 and depth.shape[-1] == 1:
            depth = depth[..., 0]
        if depth.ndim != 2:
            raise RuntimeError("robot_self_filter_depth_invalid")
        attached_getter = getattr(
            self.env, "robot_self_filter_attached_objects", None
        )
        attached = attached_getter() if callable(attached_getter) else {}
        self._robot_self_filter.update(
            self.env.arm_status(),
            depth_shape=depth.shape,
            intrinsics=self._scaled_camera_intrinsics(depth.shape),
            attached_objects=attached,
            force=True,
        )

    def _scaled_camera_intrinsics(
        self, depth_shape: tuple[int, int]
    ) -> np.ndarray:
        manipulation = getattr(self.env, "manipulation_config", None)
        if not isinstance(manipulation, Mapping):
            raise RuntimeError("robot_self_filter_camera_calibration_missing")
        intrinsics = np.asarray(
            manipulation.get("camera_intrinsics"), dtype=np.float64
        )
        resolution = np.asarray(
            manipulation.get("camera_calibration_resolution"), dtype=np.float64
        ).reshape(-1)
        if (
            intrinsics.shape != (3, 3)
            or resolution.shape != (2,)
            or not np.all(np.isfinite(intrinsics))
            or not np.all(resolution > 0.0)
        ):
            raise RuntimeError("robot_self_filter_camera_calibration_invalid")
        image_height, image_width = depth_shape
        scale_x = image_width / float(resolution[0])
        scale_y = image_height / float(resolution[1])
        scaled = intrinsics.copy()
        scaled[0, 0] *= scale_x
        scaled[0, 2] = scale_x * (scaled[0, 2] + 0.5) - 0.5
        scaled[1, 1] *= scale_y
        scaled[1, 2] = scale_y * (scaled[1, 2] + 0.5) - 0.5
        return scaled

    def _ground_clearance_config(
        self, frame: Any, depth_shape: tuple[int, int]
    ) -> tuple[
        np.ndarray,
        float,
        np.ndarray,
        float,
        float,
        float,
        str,
        float | None,
    ] | None:
        manipulation = getattr(self.env, "manipulation_config", None)
        if not isinstance(manipulation, Mapping):
            return None
        docking = manipulation.get("visible_object_docking")
        if not isinstance(docking, Mapping):
            return None
        required = (
            "camera_intrinsics",
            "camera_calibration_resolution",
        )
        if any(key not in manipulation for key in required):
            return None
        intrinsics = np.asarray(
            manipulation["camera_intrinsics"], dtype=np.float64
        )
        resolution = np.asarray(
            manipulation["camera_calibration_resolution"], dtype=np.float64
        ).reshape(-1)
        if intrinsics.shape != (3, 3) or resolution.shape != (2,):
            raise RuntimeError("ground_clearance_camera_calibration_invalid")
        if not np.all(np.isfinite(intrinsics)) or not np.all(resolution > 0.0):
            raise RuntimeError("ground_clearance_camera_calibration_invalid")
        image_height, image_width = depth_shape
        scale_x = image_width / float(resolution[0])
        scale_y = image_height / float(resolution[1])
        intrinsics = intrinsics.copy()
        intrinsics[0, 0] *= scale_x
        intrinsics[0, 2] = scale_x * (intrinsics[0, 2] + 0.5) - 0.5
        intrinsics[1, 1] *= scale_y
        intrinsics[1, 2] = scale_y * (intrinsics[1, 2] + 0.5) - 0.5
        require_dynamic = bool(
            docking.get("require_dynamic_ground_plane", False)
        )
        maximum_age_s = float(docking.get("ground_plane_max_age_s", 2.0))
        dynamic_height = getattr(frame, "ground_camera_height_m", None)
        dynamic_down = getattr(frame, "ground_down_camera_xyz", None)
        dynamic_timestamp = getattr(frame, "ground_plane_timestamp_ns", None)
        frame_timestamp = getattr(frame, "timestamp_ns", None)
        dynamic_available = all(
            value is not None
            for value in (
                dynamic_height,
                dynamic_down,
                dynamic_timestamp,
                frame_timestamp,
            )
        )
        ground_age_s: float | None = None
        if dynamic_available:
            ground_age_s = max(
                0.0, (int(frame_timestamp) - int(dynamic_timestamp)) * 1e-9
            )
            dynamic_available = ground_age_s <= maximum_age_s
        if dynamic_available:
            camera_height = float(dynamic_height)
            down_values = dynamic_down
            ground_source = "zed_sdk_floor_plane"
        elif require_dynamic:
            raise RuntimeError("dynamic_ground_plane_unavailable")
        else:
            camera_height = float(docking.get("ground_camera_height_m"))
            down_values = docking.get("ground_down_camera_xyz")
            ground_source = "configured_fallback"
            ground_age_s = None
        down = np.asarray(down_values, dtype=np.float64).reshape(-1)
        if down.shape != (3,) or not np.all(np.isfinite(down)):
            raise RuntimeError("ground_clearance_down_vector_invalid")
        norm = float(np.linalg.norm(down))
        if norm <= 1e-6:
            raise RuntimeError("ground_clearance_down_vector_invalid")
        down /= norm
        # Nav2 owns the main obstacle map, so its docking settings do not need
        # to repeat the legacy direct-controller height filters.  Direct
        # primitives still use this hard forward-clearance check and therefore
        # need conservative defaults when those optional keys are absent.
        values = (
            camera_height,
            float(docking.get("obstacle_min_height_m", 0.10)),
            float(docking.get("obstacle_max_height_m", 1.45)),
            float(docking.get("obstacle_max_depth_m", 20.0)),
        )
        if not all(math.isfinite(value) for value in values):
            raise RuntimeError("ground_clearance_height_config_invalid")
        camera_height, min_height, max_height, max_depth = values
        if (
            camera_height <= 0.0
            or not 0.0 < min_height < max_height
            or max_depth <= 0.05
        ):
            raise RuntimeError("ground_clearance_height_config_invalid")
        return (
            intrinsics,
            camera_height,
            down,
            min_height,
            max_height,
            max_depth,
            ground_source,
            ground_age_s,
        )


    def _stop_impl(self, *, start_time: float) -> dict[str, Any]:
        deadline = self._clock() + self.config.stop_timeout_s
        confirmed = 0
        reason = "zero_not_confirmed"
        status: dict[str, Any] | None = None
        try:
            while self._clock() <= deadline:
                self._submit_velocity([0.0, 0.0, 0.0], allow_stopped=True)
                status = self.env.base_status()
                last = np.asarray(status.get("last_velocity", []), dtype=float)
                zero_confirmed = (
                    last.shape == (3,)
                    and np.all(np.isfinite(last))
                    and np.linalg.norm(last) <= 1e-6
                )
                if zero_confirmed:
                    confirmed += 1
                    if confirmed >= self.config.settle_cycles:
                        return self._result(
                            "stop",
                            True,
                            "zero_confirmed",
                            start_time,
                            metrics={"confirmed_cycles": confirmed},
                            status=status,
                        )
                else:
                    confirmed = 0
                self._sleep(1.0 / self.config.control_hz)
        except Exception as exc:
            reason = self._exception_reason(exc)
        return self._result(
            "stop",
            False,
            reason,
            start_time,
            metrics={"confirmed_cycles": confirmed},
            status=status,
        )

    def _raise_if_stop_requested(self) -> None:
        if self._stop_requested.is_set():
            raise RuntimeError("operator_stop_requested")

    def _submit_velocity(
        self, velocity: list[float], *, allow_stopped: bool = False
    ) -> dict[str, Any]:
        """Serialize velocity writes and reject motion after an operator stop."""

        with self._command_lock:
            if self._stop_requested.is_set() and not allow_stopped:
                raise RuntimeError("operator_stop_requested")
            return self.env.submit_base_velocity(velocity)

    def _bounded_speed(
        self,
        value: float | None,
        *,
        default: float,
        minimum: float,
        status_key: str,
    ) -> float:
        speed = default if value is None else self._finite_float(status_key, value)
        if speed < minimum:
            raise ValueError(f"{status_key} must be >= {minimum:.3f}")
        return speed

    def _timeout(self, value: float | None, *, nominal: float) -> float:
        timeout = (
            min(nominal, self.config.max_duration_s)
            if value is None
            else self._finite_float("timeout_s", value)
        )
        if timeout <= 0 or timeout > self.config.max_duration_s:
            raise ValueError(
                f"timeout_s must be in (0, {self.config.max_duration_s:.1f}]"
            )
        return timeout

    @staticmethod
    def _finite_float(name: str, value: float) -> float:
        if isinstance(value, bool):
            raise TypeError(f"{name} must be a real number")
        value = float(value)
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
        return value

    @staticmethod
    def _exception_reason(exc: Exception) -> str:
        reason = getattr(exc, "reason", None)
        if isinstance(reason, str) and reason.strip():
            # Already written for the policy; keep its wording and spacing.
            return reason.strip()
        text = str(exc).strip().replace(" ", "_")
        return f"{type(exc).__name__}:{text}" if text else type(exc).__name__

    def _result(
        self,
        primitive: str,
        success: bool,
        reason: str,
        start_time: float,
        *,
        start_pose: list[float] | None = None,
        final_pose: list[float] | None = None,
        metrics: dict[str, Any] | None = None,
        status: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {
            "success": bool(success),
            "status": "succeeded" if success else "failed",
            "primitive": primitive,
            "reason": reason,
            "elapsed_s": max(0.0, float(self._clock() - start_time)),
            "start_pose_xy_yaw": start_pose,
            "final_pose_xy_yaw": final_pose,
            "metrics": metrics or {},
            "base_status": status,
        }
