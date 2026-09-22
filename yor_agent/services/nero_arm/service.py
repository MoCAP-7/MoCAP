"""Restricted dual-Nero RPC service for the YOR Raspberry Pi.

Run this module on the Raspberry Pi with both the verified ``pyAgxArm`` stack
and the repository's MuJoCo/Mink IK dependencies available.  It imports only
``robot.arm.ik_solver`` from the robot package; the base remains owned by the
separate leased service on port 5557.
"""

from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import os
from pathlib import Path
import signal
import subprocess
import threading
import time
from typing import Any

import numpy as np


# Rest pose: the hand-taught pose of 2026-09-10 with the shoulder opened and
# the elbow bent to the most the arm executes (2026-09-13) so that every part
# reaching past the body outline stays above the lowest arm layer the
# direct-motion gate builds; with the previous pose a table top at working
# height fell into the forearm's layer and refused small forward steps. The
# arms are 0.94 m across and reach 0.44 m ahead of the swerve centre, inside
# the navigation footprint. The elbow motor itself stops at 2.147 rad
# whatever limit the driver is configured with (a 2.16 command settled at
# 2.147), so the target stays just under that. Must match NERO_HOME_JOINTS_RAD
# in yor_agent/robot/manipulation.py, which records the sphere-model numbers.
# The right arm mirrors the left by flipping joints 1, 3, 5 and 6.
LEFT_HOME = [-0.1500, 1.1000, -1.9500, 2.1400, -0.8258, -0.2723, -0.9225]
RIGHT_HOME = [0.1500, 1.1000, 1.9500, 2.1400, 0.8258, 0.2723, -0.9225]
# Nominal TCP for the currently planned AgileX Nero official gripper.
#
# AgileX's URDF/xacro gives:
#   link7 (the pyAgxArm flange frame) -> gripper_flange
#       xyz=[0.031, 0, -0.0235], rpy=[-pi/2, 0, -pi/2]
#   gripper_flange -> gripper_base: local Z +0.006 m
#   gripper_base -> open grasp-volume center: local Z +0.105 m. NVIDIA's
#       official piper_hand profile describes that useful volume as
#       Z=[0.0775, 0.1325] m.
# Thus the gripper mounting flange -> grasp-center TCP is +0.111 m, versus
# +0.144 m to the absolute fingertip. Composing this with the adapter gives the
# pose below. This is description-derived rather than physically calibrated;
# change it when measurement or a different gripper supersedes it.
# The TCP orientation follows the official gripper_base frame: local +Y is the
# parallel-jaw opening/closing axis and local +Z points forward toward the jaws.
NERO_OFFICIAL_GRIPPER_MOUNT_FLANGE_TO_TCP_M = 0.111
NERO_LINK7_TO_GRIPPER_MOUNT_X_M = 0.031
NERO_OFFICIAL_GRIPPER_TCP_OFFSET = [
    NERO_LINK7_TO_GRIPPER_MOUNT_X_M
    + NERO_OFFICIAL_GRIPPER_MOUNT_FLANGE_TO_TCP_M,
    0.0,
    -0.0235,
    -math.pi / 2,
    0.0,
    -math.pi / 2,
]
# The elbow's upper limit is 0.01 rad inside its mechanical stop, measured at
# 2.20 rad by bending the left arm by hand in calibration drag (2026-09-13).
# The motor firmware still saturates commands at the vendor's 2.147 rad, so
# the rest pose targets 2.14; the wider limit keeps the measured pose inside
# the range cuRobo plans within. The pyAgxArm driver clamps every joint
# command to the limits it was configured with, so _create_pair hands it this
# elbow range explicitly.
NERO_JOINT_POSITION_LIMITS_RAD = np.asarray(
    [
        [-2.70526, 2.70526],
        [-1.74, 1.74],
        [-2.75, 2.75],
        [-1.01, 2.19],
        [-2.75, 2.75],
        [-0.73, 0.95],
        [-math.pi / 2, math.pi / 2],
    ],
    dtype=float,
)
ELBOW_JOINT_NAME = "joint4"
TRAJECTORY_MAX_JOINT_STEP_RAD = 0.10
TRAJECTORY_START_TOLERANCE_RAD = 0.02
# One planner path per RPC. cuRobo emits 0.10 s samples, so 256 waypoints cover
# 25 s of motion; the longest path observed on 2026-09-07 had 101 samples.
TRAJECTORY_MAX_WAYPOINTS = 256
# Planner sample spacing (grasp_motion interpolation_dt). Waypoint k is sent at
# k * dt so the firmware's position-velocity smoothing receives one continuous
# target stream instead of settle-and-go MOVE_J segments.
TRAJECTORY_DEFAULT_DT_S = 0.10
# Planner samples are linearly interpolated to this command rate before they
# are sent, the way a joint-trajectory controller interpolates at its control
# period. The firmware only receives joint positions (four CAN frames per
# target, no time or velocity field), so a dense stream keeps the motion
# continuous whether or not the firmware blends between consecutive targets.
TRAJECTORY_DEFAULT_STREAM_HZ = 50.0
# Stop streaming when the measured joints lag the most recently sent target by
# more than this: the firmware then finishes that target and holds position,
# and the RPC returns ``joint_trajectory_lag_exceeded`` (no emergency stop; an
# electronic stop cuts torque and lets the arm drop). The planner checks
# collisions along samples at most 0.10 rad apart, so the lag bounds how far
# the firmware's own interpolation toward the current target can cut across
# the planned path.
TRAJECTORY_PATH_TOLERANCE_RAD = 0.15
# How long a lag-aborted stream waits for the arm to settle on the held target
# before returning, so the caller sees a stationary arm.
TRAJECTORY_LAG_HOLD_SETTLE_S = 5.0
# Arrival at a commanded target: three consecutive 50 ms reads within this.
TRAJECTORY_SETTLE_TOLERANCE_RAD = 0.035
# Torque-off leaves an arm hanging against the base: measured on 2026-09-08 the
# left arm came to rest with joints
# [0.094, 1.554, -2.064, 0.312, -1.103, -0.071, -0.081], its lowest link 0.228 m
# above the floor and thirteen collision spheres inside the base footprint,
# right against the yellow frame. Homing from there swings the whole arm and
# hits the frame, which trips the emergency stop before startup finishes.
#
# So the shoulder moves first, on its own: joint 1 alone swings the arm about
# 30 deg FORWARD while every other joint is powered off, and only then is the
# whole arm powered and homed. The powered-off joints are braked, so the arm
# turns about joint 1 as one rigid piece, and the operator judged on the robot
# that this carries the hand away from the frame without touching it on the
# way.
#
# What that does, from the measured pose (FK of nero-welded-base-and-lift.mjcf;
# that model's lift height and frame do not match the robot, so only the
# relative motion carries over): joint 1 turns about the robot's lateral axis,
# so the hand moves 0.30 m forward and 0.10 m up and not at all sideways. The
# robot gets no wider than its travel pose during start-up, and the hand ends
# up about as far forward as it is in the rest pose. Forward is DECREASING
# joint 1 on the left arm and INCREASING it on the right, mirrored like every
# odd joint. Homing then turns joint 1 back about 10 deg to the rest pose's
# -0.25 / +0.25; along that joint-space path no collision sphere comes lower
# than it is at the end of the swing (0.325 m in the sphere model).
#
# Until 2026-09-10 joint 2 swung the arm outward instead, which made the robot
# about 1.4 m across during start-up and, with the rest braked, lifted the
# whole arm sideways with a load the operator found alarming.
#
# Enabling joint 1 alone straight from power-off does not work: after the
# emergency-stop reset the arm sits at arm_status 6 (JOINT_BRAKE_NOT_RELEASED)
# with every joint off, a single-joint enable does not take it out of that
# state, and MOVE_J did nothing for the full timeout (2026-09-08). So the whole
# arm is enabled first, which clears that state without moving anything, and
# then every joint but joint 1 is disabled again before the swing.
#
# MOVE_J with only some joints powered is not documented; on 2026-09-10 it
# appeared to work with joint 2. If the shoulder has not started moving within
# SHOULDER_CLEARANCE_RESPONSE_S, the rest of the arm is powered again and the
# same swing runs with every joint held, and the result says so.
SHOULDER_CLEARANCE_JOINT = 1
SHOULDER_CLEARANCE_DIRECTION = {"left": -1.0, "right": 1.0}
SHOULDER_CLEARANCE_RAD = 0.52
SHOULDER_CLEARANCE_TIMEOUT_S = 10.0
SHOULDER_CLEARANCE_RESPONSE_S = 2.0
SHOULDER_CLEARANCE_RESPONSE_RAD = 0.02
YOR_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_NERO_MJCF = (
    YOR_ROOT / "robot" / "yor-description" / "nero-welded-base-and-lift.mjcf"
)
EXPECTED_SERIALS = {
    "can_left": "003900194148571320343133",
    "can_right": "004A00414148570D20343133",
}
CALIBRATION_DRAG_CONFIRMATION_PREFIX = "I_AM_PHYSICALLY_SUPPORTING"
GRIPPER_COMMISSIONING_CONFIRMATION = (
    "I_CONFIRM_SUPERVISED_GRIPPER_COMMISSIONING"
)


_MINK_WORKER_SOLVERS: dict[str, Any] = {}


def _initialize_mink_worker(
    solver_specs: dict[str, dict[str, Any]],
    ready_queue: Any | None = None,
) -> None:
    """Construct process-local Mink solvers once at worker startup."""

    # The foreground service owns terminal shutdown.  Without this, Ctrl-C is
    # delivered to every process in the foreground group and idle Pool workers
    # each print a KeyboardInterrupt traceback while the parent is already
    # performing the coordinated arm/pool shutdown.
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    from robot.arm.ik_solver import SingleArmIK

    global _MINK_WORKER_SOLVERS
    solvers: dict[str, Any] = {}
    for name, spec in solver_specs.items():
        solver = SingleArmIK(
            spec["mjcf_path"],
            solver_dt=float(spec["solver_dt"]),
            joint_names=list(spec["joint_names"]),
            ee_frame=str(spec["ee_frame"]),
            root_frame=str(spec["root_frame"]),
        )
        tcp_offset = spec.get("tcp_offset")
        if tcp_offset is not None:
            solver.set_end_effector_offset(list(tcp_offset))
        solvers[name] = solver
    _MINK_WORKER_SOLVERS = solvers
    if ready_queue is not None:
        ready_queue.put(os.getpid())


def _solve_mink_worker_task(task: tuple[Any, ...]) -> dict[str, Any]:
    """Solve one target in an isolated persistent worker process."""

    name, candidate_index, start_values, target_values, max_iterations = task
    try:
        solver = _MINK_WORKER_SOLVERS[str(name)]
        start_joints = np.asarray(start_values, dtype=float)
        target = np.asarray(target_values, dtype=float)
        solver.init(start_joints)
        joint_target, ik_converged, ik_error = solver.solve_pose_xyz_rpy(
            target,
            max_iter=int(max_iterations),
        )
        joint_target = np.asarray(joint_target, dtype=float)
        ik_error = np.asarray(ik_error, dtype=float)
        if joint_target.shape != (7,) or not np.all(np.isfinite(joint_target)):
            raise RuntimeError("Mink returned invalid joint values")
        if ik_error.shape != (6,) or not np.all(np.isfinite(ik_error)):
            raise RuntimeError("Mink returned an invalid task error")
        return {
            "candidate_index": int(candidate_index),
            "joint_target": joint_target.tolist(),
            "ik_converged": bool(ik_converged),
            "ik_error": ik_error.tolist(),
        }
    except Exception as exc:
        return {
            "candidate_index": int(candidate_index),
            "error": f"{type(exc).__name__}: {exc}",
        }


class ParallelMinkIK:
    """Persistent process pool for exact Raspberry Pi Mink IK queries."""

    def __init__(
        self,
        solver_specs: dict[str, dict[str, Any]],
        *,
        worker_count: int,
        batch_timeout_s: float,
    ) -> None:
        if not 2 <= int(worker_count) <= 16:
            raise ValueError("worker_count must be in [2, 16]")
        if not 1.0 <= float(batch_timeout_s) <= 60.0:
            raise ValueError("batch_timeout_s must be in [1, 60]")
        self.worker_count = int(worker_count)
        self.batch_timeout_s = float(batch_timeout_s)
        self._solver_specs = solver_specs
        # ``spawn`` avoids inheriting CAN drivers, RPC locks, and mutable
        # MuJoCo state from the service process. Startup is paid only once.
        self._context = mp.get_context("spawn")
        self._solve_lock = threading.Lock()
        # Do not create workers here. NeroArmPairRPC is constructed before the
        # hardware is enabled and homed; loading multiple MuJoCo models during
        # that safety-critical feedback loop can starve the CAN SDK threads.
        self._pool = None
        self._closed = False

    def _start_pool(self, *, wait_until_ready: bool):
        ready_queue = self._context.Queue() if wait_until_ready else None
        pool = self._context.Pool(
            processes=self.worker_count,
            initializer=_initialize_mink_worker,
            initargs=(self._solver_specs, ready_queue),
        )
        if not wait_until_ready:
            return pool
        ready_pids: set[int] = set()
        deadline = time.monotonic() + 60.0
        try:
            while len(ready_pids) < self.worker_count:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    raise TimeoutError(
                        "Mink IK workers did not initialize within 60 seconds"
                    )
                ready_pids.add(int(ready_queue.get(timeout=remaining)))
        except BaseException:
            pool.terminate()
            pool.join()
            raise
        finally:
            ready_queue.close()
        return pool

    def start(self) -> None:
        """Start workers only after both arms have completed homing."""

        with self._solve_lock:
            if self._closed:
                raise RuntimeError("parallel Mink IK pool is closed")
            if self._pool is None:
                self._pool = self._start_pool(wait_until_ready=True)

    def solve(
        self,
        name: str,
        start_joints: np.ndarray,
        targets: list[np.ndarray],
        *,
        max_iterations: int,
    ) -> list[dict[str, Any]]:
        if self._closed:
            raise RuntimeError("parallel Mink IK pool is closed")
        tasks = [
            (
                name,
                candidate_index,
                start_joints.tolist(),
                target.tolist(),
                int(max_iterations),
            )
            for candidate_index, target in enumerate(targets)
        ]
        # IK difficulty varies substantially by target.  Dispatch one target
        # at a time so an unlucky chunk of hard targets cannot leave the other
        # persistent workers idle while one process becomes the straggler.
        chunksize = 1
        with self._solve_lock:
            if self._pool is None:
                # Defensive fallback for direct library users. The production
                # service explicitly starts this pool after homing.
                self._pool = self._start_pool(wait_until_ready=True)
            result = self._pool.map_async(
                _solve_mink_worker_task,
                tasks,
                chunksize,
            )
            try:
                return result.get(timeout=self.batch_timeout_s)
            except mp.TimeoutError as exc:
                # Pool work cannot be cancelled safely. Replace the complete
                # pool so a timed-out DAQP call cannot poison later requests.
                self._pool.terminate()
                self._pool.join()
                self._pool = self._start_pool(wait_until_ready=False)
                raise TimeoutError(
                    "parallel Mink IK batch exceeded "
                    f"{self.batch_timeout_s:.1f}s for {len(tasks)} candidates"
                ) from exc

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._pool is not None:
            self._pool.close()
            self._pool.join()


def _arm_name(arm: int | str) -> str:
    if arm in (0, "0", "left", "can_left"):
        return "left"
    if arm in (1, "1", "right", "can_right"):
        return "right"
    raise ValueError("arm must be 0/'left' or 1/'right'")


def _message_payload(message: Any) -> Any:
    return None if message is None else getattr(message, "msg", message)


def _number(value: Any) -> int | float | str | bool | None:
    # Do not return int/float/str subclasses unchanged. pyAgxArm status enums
    # inherit from int, and pickling such an enum leaks the SDK's module path to
    # RPC clients that should only receive built-in wire types.
    if value is None or type(value) in (str, bool, int, float):
        return value
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return float(value)
        except (TypeError, ValueError):
            return str(value)


def _wrap(values: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(values), np.cos(values))


def _rpy_rotation_matrix(rpy: np.ndarray) -> np.ndarray:
    """Return the XYZ-RPY rotation matrix used by the Nero pose interface."""

    roll, pitch, yaw = np.asarray(rpy, dtype=float)
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.asarray(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=float,
    )


def _rotation_error_rad(current_rpy: np.ndarray, target_rpy: np.ndarray) -> float:
    """Return the geodesic SO(3) angle between two equivalent RPY poses."""

    current = _rpy_rotation_matrix(current_rpy)
    target = _rpy_rotation_matrix(target_rpy)
    cosine = float(
        np.clip((np.trace(current.T @ target) - 1.0) / 2.0, -1.0, 1.0)
    )
    return math.acos(cosine)


def _verify_interface_serial(channel: str, expected: str) -> None:
    interface_path = Path("/sys/class/net") / channel
    if not interface_path.exists():
        raise RuntimeError(
            f"required SocketCAN interface {channel!r} does not exist; "
            "restore the corresponding USB-CAN adapter/interface before "
            "enabling or recovering that arm"
        )
    result = subprocess.run(
        [
            "udevadm",
            "info",
            "--query=property",
            f"--path={interface_path}",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    properties = dict(
        line.split("=", 1)
        for line in result.stdout.splitlines()
        if "=" in line
    )
    actual = properties.get("ID_SERIAL_SHORT")
    if actual != expected:
        raise RuntimeError(
            f"{channel} USB serial mismatch: expected {expected}, got {actual}"
        )


class NeroArmPairRPC:
    """Small monitored RPC surface around two pyAgxArm V120 drivers."""

    def __init__(
        self,
        left_driver: Any,
        right_driver: Any,
        left_gripper: Any,
        right_gripper: Any,
        *,
        speed_percent: int = 20,
        open_width_m: float = 0.10,
        close_width_m: float = 0.0,
        gripper_force_n: float = 1.0,
        max_motion_s: float = 60.0,
        tcp_offsets: dict[str, list[float]] | None = None,
        ik_solvers: dict[str, Any] | None = None,
        parallel_ik: ParallelMinkIK | None = None,
        end_effector_frame: str = "tcp",
        ik_max_iterations: int = 100,
        allow_calibration_drag: bool = False,
        trajectory_dt_s: float = TRAJECTORY_DEFAULT_DT_S,
        trajectory_stream_hz: float = TRAJECTORY_DEFAULT_STREAM_HZ,
        trajectory_path_tolerance_rad: float = TRAJECTORY_PATH_TOLERANCE_RAD,
    ) -> None:
        if not 1 <= int(speed_percent) <= 30:
            raise ValueError("speed_percent must be in [1, 30]")
        if not 0.0 <= close_width_m < open_width_m <= 0.10:
            raise ValueError("invalid gripper open/close widths")
        if not 0.1 <= gripper_force_n <= 3.0:
            raise ValueError("gripper_force_n must be in [0.1, 3.0]")
        if not 1.0 <= max_motion_s <= 60.0:
            raise ValueError("max_motion_s must be in [1, 60]")
        if not 0.02 <= float(trajectory_dt_s) <= 1.0:
            raise ValueError("trajectory_dt_s must be in [0.02, 1.0] s")
        if not 5.0 <= float(trajectory_stream_hz) <= 200.0:
            raise ValueError("trajectory_stream_hz must be in [5, 200] Hz")
        if not (
            TRAJECTORY_MAX_JOINT_STEP_RAD
            <= float(trajectory_path_tolerance_rad)
            <= 0.5
        ):
            raise ValueError(
                "trajectory_path_tolerance_rad must be in "
                f"[{TRAJECTORY_MAX_JOINT_STEP_RAD:.2f}, 0.5] rad"
            )
        if end_effector_frame not in ("tcp", "flange"):
            raise ValueError("end_effector_frame must be 'tcp' or 'flange'")
        if not 1 <= int(ik_max_iterations) <= 1000:
            raise ValueError("ik_max_iterations must be in [1, 1000]")
        self._drivers = {"left": left_driver, "right": right_driver}
        self._grippers = {"left": left_gripper, "right": right_gripper}
        self._homes = {"left": LEFT_HOME, "right": RIGHT_HOME}
        self._locks = {"left": threading.Lock(), "right": threading.Lock()}
        self._calibration_drag_state_lock = threading.Lock()
        self._last_sequence = {"left": -1, "right": -1}
        self._speed_percent = int(speed_percent)
        self._open_width_m = float(open_width_m)
        self._close_width_m = float(close_width_m)
        self._gripper_force_n = float(gripper_force_n)
        self._max_motion_s = float(max_motion_s)
        self._trajectory_dt_s = float(trajectory_dt_s)
        self._trajectory_stream_hz = float(trajectory_stream_hz)
        self._trajectory_path_tolerance_rad = float(trajectory_path_tolerance_rad)
        self._tcp_offsets = tcp_offsets or {
            "left": list(NERO_OFFICIAL_GRIPPER_TCP_OFFSET),
            "right": list(NERO_OFFICIAL_GRIPPER_TCP_OFFSET),
        }
        self._ik_solvers = {} if ik_solvers is None else dict(ik_solvers)
        self._parallel_ik = parallel_ik
        self._end_effector_frame = end_effector_frame
        self._ik_max_iterations = int(ik_max_iterations)
        self._allow_calibration_drag = bool(allow_calibration_drag)
        self._calibration_drag_arm: str | None = None
        self._calibration_drag_started_monotonic: float | None = None
        self._calibration_drag_entry_joints: list[float] | None = None
        self._estop_latched = False
        if self._allow_calibration_drag:
            for name, driver in self._drivers.items():
                missing = [
                    method
                    for method in (
                        "set_leader_mode",
                        "set_follower_mode",
                        "get_leader_joint_angles",
                        "fk",
                    )
                    if not callable(getattr(driver, method, None))
                ]
                if missing:
                    raise RuntimeError(
                        f"{name} pyAgxArm driver lacks calibration drag APIs: "
                        f"{', '.join(missing)}"
                    )

    @staticmethod
    def _arm_status_value(driver: Any) -> Any:
        message = driver.get_arm_status()
        status = _message_payload(message)
        return None if status is None else getattr(status, "arm_status", None)

    @staticmethod
    def _arm_status_diagnostic(driver: Any) -> dict[str, Any] | None:
        message = driver.get_arm_status()
        status = _message_payload(message)
        if status is None:
            return None
        fields = (
            "ctrl_mode",
            "arm_status",
            "mode_feedback",
            "teach_status",
            "motion_status",
            "trajectory_num",
        )
        result = {
            field: _number(getattr(status, field, None)) for field in fields
        }
        error_status = getattr(status, "err_status", None)
        if error_status is not None:
            result["err_status"] = str(error_status)
        return result

    @staticmethod
    def _ik_joint_diagnostic(
        driver: Any, *, newer_than_timestamp: float | None = None
    ) -> list[float] | str | None:
        get_ik = getattr(driver, "get_ik_joint_angles", None)
        if not callable(get_ik):
            return None
        try:
            message = get_ik()
            if message is None:
                return None
            joints = np.asarray(message.msg, dtype=float)
            if joints.shape != (7,) or not np.all(np.isfinite(joints)):
                return f"invalid_ik_feedback:{getattr(message, 'msg', None)!r}"
            timestamp = getattr(message, "timestamp", None)
            if (
                newer_than_timestamp is not None
                and timestamp is not None
                and float(timestamp) <= newer_than_timestamp
            ):
                return f"stale_ik_feedback:{joints.tolist()}"
            return joints.tolist()
        except Exception as exc:
            return f"ik_feedback_error:{type(exc).__name__}:{exc}"

    @staticmethod
    def _status_value_is_emergency_stopped(value: Any) -> bool:
        try:
            return int(value) == 1
        except (TypeError, ValueError):
            return str(value).upper().startswith("EMERGENCY_STOP")

    @staticmethod
    def _status_value_is_no_solution(value: Any) -> bool:
        try:
            return int(value) == 2
        except (TypeError, ValueError):
            return str(value).upper().startswith("NO_SOLUTION")

    @classmethod
    def _has_completed_no_solution(cls, driver: Any) -> bool:
        """Return true only for retained NO_SOLUTION with target reached."""

        status = _message_payload(driver.get_arm_status())
        if status is None or not cls._status_value_is_no_solution(
            getattr(status, "arm_status", None)
        ):
            return False
        motion_status = getattr(status, "motion_status", None)
        try:
            return int(motion_status) == 0
        except (TypeError, ValueError):
            token = str(motion_status).upper()
            return token in ("0", "REACHED", "REACH_TARGET_POS_SUCCESS")

    @classmethod
    def _is_emergency_stopped(cls, driver: Any) -> bool:
        return cls._status_value_is_emergency_stopped(
            cls._arm_status_value(driver)
        )

    @classmethod
    def _wait_for_arm_status(
        cls, driver: Any, name: str, timeout_s: float = 3.0
    ) -> Any:
        """Wait for stable, non-empty 0x2A1 feedback after connecting."""

        deadline = time.monotonic() + float(timeout_s)
        stable = 0
        last_status: Any = None
        while time.monotonic() <= deadline:
            last_status = cls._arm_status_value(driver)
            if last_status is not None:
                stable += 1
                if stable >= 3:
                    return last_status
            else:
                stable = 0
            time.sleep(0.02)
        raise TimeoutError(
            f"{name} arm status feedback timeout: status={last_status}, "
            f"joints_enabled={driver.get_joints_enable_status_list()}"
        )

    @staticmethod
    def _enable_with_timeout(
        driver: Any, name: str, timeout_s: float = 5.0
    ) -> None:
        """Enable all joints with a bounded, low-rate retry loop.

        V120 ``enable()`` already waits for motor feedback for up to 1.5 s.
        Calling it at polling frequency both floods CAN and makes our outer
        timeout meaningless, so each SDK call gets an explicit bounded wait.
        """

        deadline = time.monotonic() + float(timeout_s)
        attempts = 0
        last_enabled: list[bool] = []
        while time.monotonic() < deadline:
            remaining = deadline - time.monotonic()
            sdk_timeout = max(0.0, min(1.5, remaining))
            try:
                acknowledged = bool(driver.enable(timeout=sdk_timeout))
            except TypeError as exc:
                # Test doubles and older SDKs do not expose the V112 timeout
                # keyword. Do not hide unrelated TypeErrors from enable().
                if "timeout" not in str(exc):
                    raise
                acknowledged = bool(driver.enable())
            attempts += 1
            last_enabled = [
                bool(value) for value in driver.get_joints_enable_status_list()
            ]
            if acknowledged and len(last_enabled) == 7 and all(last_enabled):
                return
            time.sleep(min(0.10, max(0.0, deadline - time.monotonic())))
        raise TimeoutError(
            f"{name} arm enable timeout: attempts={attempts}, "
            f"joints_enabled={last_enabled}, "
            f"status_detail={NeroArmPairRPC._arm_status_diagnostic(driver)}"
        )

    @classmethod
    def _enable_until_motion_ready(
        cls,
        driver: Any,
        name: str,
        timeout_s: float = 5.0,
        *,
        allow_no_solution_for_home: bool = False,
    ) -> None:
        """Enable and require stable power feedback before motion.

        ``NO_SOLUTION`` is the result of a previous Cartesian task, not an
        enable failure. During explicitly confirmed startup/home it is allowed
        through so a fresh feasible MOVE_J can replace that retained task.
        """

        cls._enable_with_timeout(driver, name, timeout_s=timeout_s)
        deadline = time.monotonic() + 1.0
        stable = 0
        last_status: Any = None
        last_enabled: list[bool] = []
        while time.monotonic() <= deadline:
            last_enabled = [
                bool(value) for value in driver.get_joints_enable_status_list()
            ]
            last_status = cls._arm_status_value(driver)
            if cls._status_value_is_emergency_stopped(last_status):
                raise RuntimeError(
                    f"{name} arm emergency stop became active during enable; "
                    "run explicit single-arm recovery"
                )
            try:
                normal = int(last_status) == 0
            except (TypeError, ValueError):
                normal = str(last_status).upper().startswith("NORMAL")
            acceptable = normal or (
                allow_no_solution_for_home
                and cls._status_value_is_no_solution(last_status)
            )
            if len(last_enabled) == 7 and all(last_enabled) and acceptable:
                stable += 1
                if stable >= 3:
                    return
            else:
                stable = 0
            time.sleep(0.02)
        raise TimeoutError(
            f"{name} arm did not become motion-ready: "
            f"status={last_status}, joints_enabled={last_enabled}, "
            f"no_solution_allowed_for_home={allow_no_solution_for_home}, "
            f"status_detail={cls._arm_status_diagnostic(driver)}"
        )

    @staticmethod
    def _control_mode_value(driver: Any) -> Any:
        status = _message_payload(driver.get_arm_status())
        return None if status is None else getattr(status, "ctrl_mode", None)

    @staticmethod
    def _control_mode_matches(value: Any, expected: int, token: str) -> bool:
        try:
            return int(value) == expected
        except (TypeError, ValueError):
            return token in str(value).upper()

    @staticmethod
    def _feedback_timestamp(message: Any) -> float | None:
        if message is None:
            return None
        value = getattr(message, "timestamp", None)
        try:
            timestamp = float(value)
        except (TypeError, ValueError):
            return None
        return timestamp if math.isfinite(timestamp) else None

    @classmethod
    def _wait_for_joint_feedback(
        cls,
        driver: Any,
        name: str,
        *,
        getter: Any,
        stream_name: str,
        timeout_s: float = 3.0,
        newer_than: float | None = None,
        minimum_fresh_samples: int = 1,
        require_timestamp: bool = False,
    ) -> np.ndarray:
        deadline = time.monotonic() + float(timeout_s)
        last_value: Any = None
        last_timestamp: float | None = None
        counted_timestamp: float | None = None
        fresh_samples = 0
        last_enabled: list[bool] = []
        while time.monotonic() <= deadline:
            message = getter()
            last_value = None if message is None else message.msg
            last_timestamp = cls._feedback_timestamp(message)
            last_enabled = [
                bool(value) for value in driver.get_joints_enable_status_list()
            ]
            if last_value is not None:
                joints = np.asarray(last_value, dtype=float)
                valid = joints.shape == (7,) and np.all(np.isfinite(joints))
                timestamp_ok = not require_timestamp or (
                    last_timestamp is not None
                    and (newer_than is None or last_timestamp > newer_than)
                )
                distinct = (
                    last_timestamp is None
                    or counted_timestamp is None
                    or last_timestamp > counted_timestamp
                )
                enabled = len(last_enabled) == 7 and all(last_enabled)
                if valid and timestamp_ok and distinct and enabled:
                    fresh_samples += 1
                    counted_timestamp = last_timestamp
                    if fresh_samples >= int(minimum_fresh_samples):
                        return joints
            time.sleep(0.02)
        raise TimeoutError(
            f"{name} {stream_name} joint feedback did not become fresh and "
            f"valid: value={last_value!r}, timestamp={last_timestamp}, "
            f"baseline={newer_than}, fresh_samples={fresh_samples}, "
            f"joints_enabled={last_enabled}"
        )

    @classmethod
    def _wait_for_leader_joints(
        cls,
        driver: Any,
        name: str,
        timeout_s: float = 3.0,
        *,
        newer_than: float | None = None,
        minimum_fresh_samples: int = 1,
        require_timestamp: bool = False,
    ) -> np.ndarray:
        return cls._wait_for_joint_feedback(
            driver,
            name,
            getter=driver.get_leader_joint_angles,
            stream_name="leader",
            timeout_s=timeout_s,
            newer_than=newer_than,
            minimum_fresh_samples=minimum_fresh_samples,
            require_timestamp=require_timestamp,
        )

    @classmethod
    def _wait_for_standard_joints(
        cls,
        driver: Any,
        name: str,
        timeout_s: float = 3.0,
        *,
        newer_than: float | None,
    ) -> np.ndarray:
        return cls._wait_for_joint_feedback(
            driver,
            name,
            getter=driver.get_joint_angles,
            stream_name="standard",
            timeout_s=timeout_s,
            newer_than=newer_than,
            minimum_fresh_samples=2,
            require_timestamp=True,
        )

    def _recover_emergency_stop(
        self, name: str, driver: Any, *, settle_s: float = 1.0
    ) -> dict[str, Any]:
        """Perform the Nero enable/reset sequence after explicit authorization.

        The official SDK documents that ``reset()`` only takes effect while an
        emergency-stopped arm is enabled, and that reset immediately removes
        motor power. Callers must therefore explicitly authorize this path and
        physically support/clear the arm before startup.
        """

        self._enable_with_timeout(driver, name)

        # The Nero documentation requires the stop and reset to be issued in
        # this order while the arm is enabled. Both are edge-triggered mode
        # commands, not cyclic control frames.
        driver.electronic_emergency_stop()
        stop_deadline = time.monotonic() + max(1.0, float(settle_s))
        last_status: Any = None
        while time.monotonic() <= stop_deadline:
            last_status = self._arm_status_value(driver)
            if self._status_value_is_emergency_stopped(last_status):
                break
            time.sleep(0.02)
        else:
            raise RuntimeError(
                f"{name} arm did not acknowledge electronic emergency stop: "
                f"status={last_status}, "
                f"joints_enabled={driver.get_joints_enable_status_list()}"
            )
        if settle_s:
            time.sleep(float(settle_s))

        reset_attempts = 0
        last_enabled: list[bool] = []
        partial_disable_seen = False
        # One frame should be sufficient. Permit two slow retries for a lost
        # CAN frame, but never reproduce the old 20 Hz reset burst.
        for _ in range(3):
            driver.reset()
            reset_attempts += 1
            attempt_deadline = time.monotonic() + 1.5
            stable_powered_off = 0
            while time.monotonic() <= attempt_deadline:
                last_enabled = [
                    bool(value)
                    for value in driver.get_joints_enable_status_list()
                ]
                last_status = self._arm_status_value(driver)
                if len(last_enabled) == 7 and not any(last_enabled):
                    stable_powered_off += 1
                    if stable_powered_off >= 3:
                        return {
                            "reset_attempts": reset_attempts,
                            "reset_acknowledgement": "all_joints_powered_off",
                            "status_after_reset": _number(last_status),
                        }
                else:
                    stable_powered_off = 0
                if len(last_enabled) == 7 and not all(last_enabled):
                    partial_disable_seen = True
                time.sleep(0.02)
            if partial_disable_seen:
                # A partial transition proves reset was accepted. Resending it
                # cannot help and risks confusing the controller; just wait.
                break

        if partial_disable_seen:
            transition_deadline = time.monotonic() + 3.0
            while time.monotonic() <= transition_deadline:
                last_enabled = [
                    bool(value)
                    for value in driver.get_joints_enable_status_list()
                ]
                last_status = self._arm_status_value(driver)
                if len(last_enabled) == 7 and not any(last_enabled):
                    return {
                        "reset_attempts": reset_attempts,
                        "reset_acknowledgement": "all_joints_powered_off",
                        "status_after_reset": _number(last_status),
                    }
                time.sleep(0.02)
        raise TimeoutError(
            f"{name} arm reset was not acknowledged after {reset_attempts} "
            f"low-rate attempts: status={last_status}, "
            f"joints_enabled={last_enabled}. No additional emergency-stop "
            "was sent; keep supporting the arm and inspect the physical stop "
            "chain/controller before retrying."
        )

    def _recover_estop_for_operator(
        self, name: str, *, settle_s: float = 1.0
    ) -> dict[str, Any]:
        """Recover one arm and verify that reset powered all joints off.

        This private operator path deliberately does not call :meth:`close` on
        success because normal service close latches another electronic stop.
        It never homes or sends a motion target.
        """

        name = _arm_name(name)
        if not 0.0 <= float(settle_s) <= 3.0:
            raise ValueError("settle_s must be in [0, 3]")
        driver = self._drivers[name]
        connected = False
        try:
            driver.set_joint_limits_enabled(True)
            driver.connect()
            connected = True
            deadline = time.monotonic() + 3.0
            while driver.get_joint_angles() is None:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"{name} arm joint feedback timeout")
                time.sleep(0.02)
            before = self._wait_for_arm_status(driver, name)
            enabled_before = [
                bool(value) for value in driver.get_joints_enable_status_list()
            ]
            if not self._is_emergency_stopped(driver):
                after = self._arm_status_value(driver)
                enabled_after = [
                    bool(value)
                    for value in driver.get_joints_enable_status_list()
                ]
                driver.disconnect()
                connected = False
                return {
                    "success": True,
                    "arm": name,
                    "status_before": _number(before),
                    "status_after": _number(after),
                    "joints_enabled": enabled_after,
                    "reason": "emergency_stop_already_clear_no_action",
                    "reset_attempts": 0,
                }

            recovery = self._recover_emergency_stop(
                name, driver, settle_s=settle_s
            )
            enabled_after = [
                bool(value) for value in driver.get_joints_enable_status_list()
            ]
            if len(enabled_after) != 7 or any(enabled_after):
                raise RuntimeError(
                    f"{name} reset returned without powering all joints off: "
                    f"joints_enabled={enabled_after}"
                )
            after = self._arm_status_value(driver)
            driver.disconnect()
            connected = False
            return {
                "success": True,
                "arm": name,
                "status_before": _number(before),
                "status_after": _number(after),
                "joints_enabled_before": enabled_before,
                "joints_enabled": enabled_after,
                "reason": "reset_accepted_and_all_joints_powered_off",
                **recovery,
            }
        except BaseException:
            if connected:
                try:
                    # Recovery already issued the documented stop before reset.
                    # Do not re-latch it during exception cleanup, especially
                    # after a delayed reset acknowledgement.
                    driver.disconnect()
                except Exception:
                    pass
            raise

    def _clear_base_frame_with_shoulder(
        self,
        driver: Any,
        name: str,
        *,
        timeout_s: float = SHOULDER_CLEARANCE_TIMEOUT_S,
        response_s: float | None = None,
    ) -> dict[str, Any]:
        """Swing a hanging arm forward off the base with joint 1 alone, before homing.

        The arm arrives powered and holding wherever gravity left it. Every
        joint but joint 1 is powered off again, joint 1 swings the arm forward
        and clear of the base with the rest braked, and then the whole arm is
        powered, holding where it is, ready for homing. If the firmware does
        not act on MOVE_J with the other joints off, the swing is redone with
        every joint held (see SHOULDER_CLEARANCE_RAD); the result's
        ``others`` and ``fallback_reason`` say which ran.

        The angle is relative to the measured pose and clamped to the official
        joint limit. Failure raises rather than latching the emergency stop:
        an electronic stop cuts torque and would drop the arm onto the very
        frame this is clearing.
        """

        message = driver.get_joint_angles()
        current = None if message is None else np.asarray(message.msg, dtype=float)
        if current is None or current.shape != (7,) or not np.all(np.isfinite(current)):
            raise RuntimeError(
                f"{name} arm has no valid joint feedback before the shoulder swing"
            )
        home = np.asarray(self._homes[name], dtype=float)
        if float(np.max(np.abs(_wrap(current - home)))) <= TRAJECTORY_START_TOLERANCE_RAD:
            return {"performed": False, "reason": "already_home"}
        index = SHOULDER_CLEARANCE_JOINT - 1
        lower, upper = (float(v) for v in NERO_JOINT_POSITION_LIMITS_RAD[index])
        start = float(current[index])
        target_value = min(
            upper,
            max(lower, start + SHOULDER_CLEARANCE_DIRECTION[name] * SHOULDER_CLEARANCE_RAD),
        )
        if abs(target_value - start) <= TRAJECTORY_SETTLE_TOLERANCE_RAD:
            return {"performed": False, "reason": "at_joint_limit", "joint_rad": start}

        response_s = (
            SHOULDER_CLEARANCE_RESPONSE_S if response_s is None else float(response_s)
        )
        target = current.copy()
        target[index] = target_value
        # The controller ignores settings while joints are off, so the motion
        # mode goes in before the rest of the arm is powered down.
        driver.set_motion_mode("j")
        time.sleep(0.10)

        fallback_reason = None
        others_off, enabled = self._power_off_all_but_shoulder(driver)
        if others_off:
            driver.move_j(target.tolist())
            outcome, last_value = self._wait_for_shoulder(
                driver,
                name,
                start=start,
                target_value=target_value,
                deadline=time.monotonic() + float(timeout_s),
                respond_by=time.monotonic() + response_s,
            )
            if outcome == "reached":
                # Re-target the powered-off joints to where they are now, so
                # powering them holds them there instead of pulling them back
                # towards the pose they had against the frame.
                reading = driver.get_joint_angles()
                values = None if reading is None else np.asarray(reading.msg, dtype=float)
                if values is not None and values.shape == (7,) and np.all(np.isfinite(values)):
                    driver.move_j(values.tolist())
            else:
                fallback_reason = (
                    f"joint {SHOULDER_CLEARANCE_JOINT} did not move within "
                    f"{response_s:.1f} s with the other joints off "
                    f"(arm_status {self._arm_status_value(driver)})"
                )
        else:
            fallback_reason = (
                f"the other joints did not report powered off (joints_enabled={enabled})"
            )
        self._enable_until_motion_ready(driver, name, allow_no_solution_for_home=True)
        driver.set_speed_percent(self._speed_percent)
        if fallback_reason is None:
            return {
                "performed": True,
                "joint": SHOULDER_CLEARANCE_JOINT,
                "others": "off",
                "from_rad": start,
                "to_rad": last_value,
            }

        print(
            f"[agents-yor-arm] {name} shoulder swing with the other joints off "
            f"failed ({fallback_reason}); redoing it with every joint held",
            flush=True,
        )
        driver.set_motion_mode("j")
        time.sleep(0.10)
        reading = driver.get_joint_angles()
        values = None if reading is None else np.asarray(reading.msg, dtype=float)
        held = (
            values.copy()
            if values is not None and values.shape == (7,) and np.all(np.isfinite(values))
            else current.copy()
        )
        held[index] = target_value
        driver.move_j(held.tolist())
        _, last_value = self._wait_for_shoulder(
            driver,
            name,
            start=start,
            target_value=target_value,
            deadline=time.monotonic() + float(timeout_s),
        )
        return {
            "performed": True,
            "joint": SHOULDER_CLEARANCE_JOINT,
            "others": "held",
            "fallback_reason": fallback_reason,
            "from_rad": start,
            "to_rad": last_value,
        }

    @staticmethod
    def _power_off_all_but_shoulder(driver: Any) -> tuple[bool, list[bool]]:
        """Disable every joint except joint 2 and report whether feedback agrees."""

        for joint in range(1, 8):
            if joint == SHOULDER_CLEARANCE_JOINT:
                continue
            try:
                driver.disable(joint, timeout=0.5)
            except TypeError as exc:
                # Older SDKs have no timeout keyword; do not hide other errors.
                if "timeout" not in str(exc):
                    raise
                driver.disable(joint)
        enabled = [bool(value) for value in driver.get_joints_enable_status_list()]
        expected = [joint == SHOULDER_CLEARANCE_JOINT for joint in range(1, 8)]
        return enabled == expected, enabled

    def _wait_for_shoulder(
        self,
        driver: Any,
        name: str,
        *,
        start: float,
        target_value: float,
        deadline: float,
        respond_by: float | None = None,
    ) -> tuple[str, float]:
        """Wait for joint 2 to settle on ``target_value``.

        Returns ``("reached", angle)``, or ``("no_response", angle)`` when
        ``respond_by`` passes with joint 2 still where it started. Raises on an
        emergency stop or when ``deadline`` passes.
        """

        index = SHOULDER_CLEARANCE_JOINT - 1
        stable = 0
        last_value = start
        responded = respond_by is None
        while time.monotonic() <= deadline:
            if self._is_emergency_stopped(driver):
                raise RuntimeError(
                    f"{name} arm emergency stop became active during the shoulder "
                    f"swing (joint {SHOULDER_CLEARANCE_JOINT} at {last_value:.3f} rad, "
                    f"target {target_value:.3f}); clear the obstruction around the "
                    "base and run the explicit single-arm recovery"
                )
            reading = driver.get_joint_angles()
            if reading is not None:
                values = np.asarray(reading.msg, dtype=float)
                if values.shape == (7,) and np.all(np.isfinite(values)):
                    last_value = float(values[index])
                    if abs(last_value - start) > SHOULDER_CLEARANCE_RESPONSE_RAD:
                        responded = True
                    error = abs(float(_wrap(np.asarray([last_value - target_value]))[0]))
                    if error <= TRAJECTORY_SETTLE_TOLERANCE_RAD:
                        stable += 1
                        if stable >= 3:
                            return "reached", last_value
                    else:
                        stable = 0
            if not responded and time.monotonic() >= respond_by:
                return "no_response", last_value
            time.sleep(0.05)
        raise TimeoutError(
            f"{name} arm shoulder swing did not reach {target_value:.3f} rad "
            f"(joint {SHOULDER_CLEARANCE_JOINT} at {last_value:.3f} rad); "
            "the arm may be resting against the base"
        )

    def initialize(
        self,
        *,
        home: bool = True,
        timeout_s: float = 15.0,
    ) -> None:
        """Connect, verify feedback, enable all joints, and optionally home.

        Emergency-stop recovery is deliberately separate from normal startup.
        Nero reset briefly removes motor power and may let a raised arm fall.
        """

        connected: list[Any] = []
        try:
            for name in ("left", "right"):
                driver = self._drivers[name]
                driver.set_joint_limits_enabled(True)
                driver.connect()
                connected.append(driver)
                deadline = time.monotonic() + 3.0
                while driver.get_joint_angles() is None:
                    if time.monotonic() >= deadline:
                        raise TimeoutError(f"{name} arm joint feedback timeout")
                    time.sleep(0.02)
                initial_status = self._wait_for_arm_status(driver, name)
                if self._status_value_is_emergency_stopped(initial_status):
                    raise RuntimeError(
                        f"{name} arm emergency stop is active; run the explicit "
                        "single-arm operator recovery before normal startup"
                    )
                self._enable_until_motion_ready(
                    driver,
                    name,
                    allow_no_solution_for_home=home,
                )
                driver.set_speed_percent(self._speed_percent)
                if home:
                    # Powering the arm does not move it, so this is still the
                    # first motion it makes: swing the hanging arm forward off
                    # the base frame with joint 1, the rest off, before homing
                    # sweeps the whole arm through it. See SHOULDER_CLEARANCE_RAD.
                    lift = self._clear_base_frame_with_shoulder(driver, name)
                    print(f"[agents-yor-arm] {name} shoulder swing: {lift}", flush=True)
            if home:
                for name in ("left", "right"):
                    self._move_joints_impl(
                        name,
                        self._homes[name],
                        timeout_s,
                        allow_initial_no_solution=True,
                    )
                # Leave both grippers open, so the arm starts from the same
                # state the grasp primitives assume. This is part of the same
                # physical bring-up the operator acknowledged with
                # --confirm-enable-and-home, and it runs after homing so a
                # held object is released clear of the rest pose. A gripper
                # that cannot be confirmed open aborts startup like homing.
                for name in ("left", "right"):
                    self.set_gripper(name, True, time.time_ns())
        except BaseException:
            for driver in connected:
                try:
                    enabled = [
                        bool(value)
                        for value in driver.get_joints_enable_status_list()
                    ]
                    # Only an arm powered by this startup needs damping. Do not
                    # manufacture a fresh stop on an already-stopped or still
                    # disabled controller merely because another arm failed.
                    if (
                        any(enabled)
                        and not self._is_emergency_stopped(driver)
                    ):
                        driver.electronic_emergency_stop()
                    driver.disconnect()
                except Exception:
                    pass
            raise

    def start_parallel_ik(self) -> None:
        """Load exact-IK workers after all startup arm motion has stopped."""

        if self._parallel_ik is not None:
            self._parallel_ik.start()

    def _accept_sequence(self, name: str, sequence: int) -> None:
        if isinstance(sequence, bool) or int(sequence) != sequence:
            raise ValueError("sequence must be an integer")
        sequence = int(sequence)
        if sequence <= self._last_sequence[name]:
            raise ValueError(
                f"{name} sequence must increase: last={self._last_sequence[name]}"
            )
        self._last_sequence[name] = sequence

    def _arm_snapshot(self, name: str) -> dict[str, Any]:
        driver = self._drivers[name]
        joints_msg = driver.get_joint_angles()
        flange_msg = driver.get_flange_pose()
        if self._calibration_drag_arm == name:
            joints_msg = driver.get_leader_joint_angles()
            flange_msg = (
                None if joints_msg is None else driver.fk(list(joints_msg.msg))
            )
        tcp_msg = driver.get_tcp_pose()
        status_msg = driver.get_arm_status()
        status = _message_payload(status_msg)
        gripper_msg = self._grippers[name].get_gripper_status()
        gripper = _message_payload(gripper_msg)
        return {
            "available": bool(driver.is_connected()),
            "connected": bool(driver.is_connected()),
            "communication_ok": bool(driver.is_ok() and not driver.has_comm_error()),
            "joint_pos": (
                None
                if joints_msg is None
                else [float(value) for value in joints_msg.msg]
            ),
            "flange_pose_xyz_rpy": (
                None
                if flange_msg is None
                else [
                    float(value)
                    for value in getattr(flange_msg, "msg", flange_msg)
                ]
            ),
            "tcp_pose_xyz_rpy": (
                None if tcp_msg is None else [float(value) for value in tcp_msg.msg]
            ),
            "joints_enabled": [
                bool(value) for value in driver.get_joints_enable_status_list()
            ],
            "arm_status": (
                None
                if status is None
                else {
                    "ctrl_mode": _number(getattr(status, "ctrl_mode", None)),
                    "arm_status": _number(getattr(status, "arm_status", None)),
                    "mode_feedback": _number(getattr(status, "mode_feedback", None)),
                    "teach_status": _number(getattr(status, "teach_status", None)),
                    "motion_status": _number(getattr(status, "motion_status", None)),
                    "err_code": _number(getattr(status, "err_code", None)),
                }
            ),
            "gripper": (
                {"available": False}
                if gripper is None
                else {
                    "available": True,
                    "width_m": float(gripper.value),
                    "force_n": float(gripper.force),
                    "mode": str(gripper.mode),
                    "enabled": bool(gripper.foc_status.driver_enable_status),
                }
            ),
            "last_sequence": int(self._last_sequence[name]),
        }

    def get_status(self) -> dict[str, Any]:
        drag_elapsed_s = None
        if self._calibration_drag_started_monotonic is not None:
            drag_elapsed_s = max(
                0.0,
                time.monotonic() - self._calibration_drag_started_monotonic,
            )
        return {
            "firmware_driver": "NeroFW.V120",
            "speed_percent": self._speed_percent,
            "max_motion_s": self._max_motion_s,
            "gripper_control": {
                "open_width_m": self._open_width_m,
                "close_width_m": self._close_width_m,
                "default_force_n": self._gripper_force_n,
                "commissioning_rpc": "commission_gripper_width",
            },
            "tcp_offsets_xyz_rpy": self._tcp_offsets,
            "cartesian_backend": "mink_move_j",
            "end_effector_frame": self._end_effector_frame,
            "ik_max_iterations": self._ik_max_iterations,
            "ik_parallel_workers": (
                1
                if self._parallel_ik is None
                else self._parallel_ik.worker_count
            ),
            "ik_batch_timeout_s": (
                None
                if self._parallel_ik is None
                else self._parallel_ik.batch_timeout_s
            ),
            "ik_planning_rpc": ["plan_tcp_pose", "plan_tcp_poses"],
            "trajectory_execution_rpc": "execute_joint_trajectory",
            "trajectory_dt_s": self._trajectory_dt_s,
            "trajectory_stream_hz": self._trajectory_stream_hz,
            "trajectory_path_tolerance_rad": self._trajectory_path_tolerance_rad,
            "trajectory_max_waypoints": TRAJECTORY_MAX_WAYPOINTS,
            "estop_latched": bool(self._estop_latched),
            "calibration_drag": {
                "allowed": self._allow_calibration_drag,
                "active_arm": self._calibration_drag_arm,
                "elapsed_s": drag_elapsed_s,
            },
            "left": self._arm_snapshot("left"),
            "right": self._arm_snapshot("right"),
        }

    @staticmethod
    def _calibration_drag_confirmation(name: str) -> str:
        return f"{CALIBRATION_DRAG_CONFIRMATION_PREFIX}_{name.upper()}"

    def enter_calibration_drag(
        self, arm: int | str, confirmation: str
    ) -> dict[str, Any]:
        """Put one explicitly authorized arm into V120 zero-force drag mode.

        This is an operator-only calibration path.  Normal deployments do not
        enable it, and generated robot code has no wrapper for this method.
        The operator must physically support the arm before the mode change.
        """

        name = _arm_name(arm)
        if not self._allow_calibration_drag:
            raise PermissionError(
                "calibration drag is disabled; restart with "
                "--allow-calibration-drag"
            )
        expected_confirmation = self._calibration_drag_confirmation(name)
        if confirmation != expected_confirmation:
            raise PermissionError(
                f"confirmation must be exactly {expected_confirmation!r}"
            )
        with self._locks[name]:
            with self._calibration_drag_state_lock:
                if self._calibration_drag_arm is not None:
                    raise RuntimeError(
                        "calibration drag is already active for "
                        f"{self._calibration_drag_arm}"
                    )
                self._ensure_motion_allowed(name)
                # Reserve the single drag slot before releasing the state lock.
                # This also blocks new motion commands during the mode transition.
                self._calibration_drag_arm = name
                self._calibration_drag_started_monotonic = time.monotonic()
                self._calibration_drag_entry_joints = None
            driver = self._drivers[name]
            try:
                joints_message = driver.get_joint_angles()
                if joints_message is None:
                    raise RuntimeError(f"{name} joint feedback is unavailable")
                entry_joints = np.asarray(joints_message.msg, dtype=float)
                if entry_joints.shape != (7,) or not np.all(
                    np.isfinite(entry_joints)
                ):
                    raise RuntimeError(f"invalid {name} joint feedback before drag")
                with self._calibration_drag_state_lock:
                    self._calibration_drag_entry_joints = entry_joints.tolist()
                leader_before = driver.get_leader_joint_angles()
                leader_before_timestamp = self._feedback_timestamp(leader_before)
                driver.set_leader_mode()
                self._wait_for_leader_joints(
                    driver,
                    name,
                    newer_than=leader_before_timestamp,
                    minimum_fresh_samples=2,
                    require_timestamp=True,
                )
            except BaseException:
                try:
                    driver.set_follower_mode()
                except Exception:
                    pass
                self.emergency_stop()
                raise
            return {
                "success": True,
                "reason": "leader_zero_force_drag_active",
                "arm": name,
                "control_mode": 6,
                "mode_verification": "fresh_leader_joint_feedback",
                "entry_joint_pos": entry_joints.tolist(),
                "warning": "physically support the arm throughout manual dragging",
            }

    def exit_calibration_drag(self, arm: int | str) -> dict[str, Any]:
        """Return the dragged arm to controlled mode and hold its current joints."""

        name = _arm_name(arm)
        if not self._allow_calibration_drag:
            raise PermissionError("calibration drag is disabled")
        with self._locks[name]:
            with self._calibration_drag_state_lock:
                active = self._calibration_drag_arm
            if active is None:
                return {
                    "success": True,
                    "reason": "calibration_drag_already_inactive",
                    "arm": name,
                }
            if active != name:
                raise RuntimeError(f"calibration drag is active for {active}, not {name}")
            driver = self._drivers[name]
            try:
                hold_joints = self._wait_for_leader_joints(driver, name)
            except BaseException:
                self.emergency_stop()
                raise
            elapsed_s = (
                None
                if self._calibration_drag_started_monotonic is None
                else time.monotonic() - self._calibration_drag_started_monotonic
            )
            try:
                standard_before = driver.get_joint_angles()
                standard_before_timestamp = self._feedback_timestamp(standard_before)
                driver.set_follower_mode()
                self._wait_for_standard_joints(
                    driver,
                    name,
                    newer_than=standard_before_timestamp,
                )
                # set_follower_mode() re-enables the ordinary CAN feedback
                # stream, but get_arm_status() may still contain the last
                # linkage-teaching status frame.  Fresh timestamped standard
                # joint frames are the authoritative transition signal, just
                # as fresh leader frames are on entry.
                control_mode = self._control_mode_value(driver)
                with self._calibration_drag_state_lock:
                    self._calibration_drag_arm = None
                    self._calibration_drag_started_monotonic = None
                    self._calibration_drag_entry_joints = None
                hold_result = self._move_joints_impl(
                    name, hold_joints.tolist(), min(5.0, self._max_motion_s)
                )
            except BaseException:
                with self._calibration_drag_state_lock:
                    self._calibration_drag_arm = None
                    self._calibration_drag_started_monotonic = None
                    self._calibration_drag_entry_joints = None
                self.emergency_stop()
                raise
            return {
                "success": True,
                "reason": "controlled_hold_restored",
                "arm": name,
                "control_mode": _number(control_mode),
                "mode_verification": "fresh_standard_joint_feedback",
                "hold_joint_pos": hold_joints.tolist(),
                "drag_elapsed_s": elapsed_s,
                "hold": hold_result,
            }

    def _ensure_motion_allowed(self, name: str) -> None:
        if self._estop_latched:
            raise RuntimeError("arm emergency stop is latched; restart service to re-arm")
        if self._calibration_drag_arm == name:
            raise RuntimeError(
                f"{name} arm is in calibration drag mode; exit drag before motion"
            )
        snapshot = self._arm_snapshot(name)
        if not snapshot["communication_ok"]:
            raise RuntimeError(f"{name} arm communication is unhealthy")
        if not all(snapshot["joints_enabled"]):
            raise RuntimeError(f"{name} arm joints are not all enabled")

    def _motion_error(
        self,
        name: str,
        *,
        allow_stale_no_solution: bool = False,
    ) -> str | None:
        snapshot = self._arm_snapshot(name)
        status = snapshot["arm_status"]
        if status is None:
            return "arm_status_unavailable"
        arm_status = status["arm_status"]
        if self._status_value_is_no_solution(arm_status) and (
            allow_stale_no_solution
            or self._has_completed_no_solution(self._drivers[name])
        ):
            # V120 can retain a previous Cartesian NO_SOLUTION even after a
            # later task has reached its target. The SDK documents
            # motion_status=0 as reached and 1 as reach-target failed.
            return None
        # pyAgxArm exposes enum-like values; NORMAL always serializes to zero.
        if arm_status not in (0, "0", "NORMAL"):
            return f"arm_status_{arm_status}"
        return None

    def _joint_error_rad(self, name: str, target: np.ndarray) -> float | None:
        """Return the largest wrapped joint error to ``target``, or None."""

        message = self._drivers[name].get_joint_angles()
        if message is None:
            return None
        current = np.asarray(message.msg, dtype=float)
        if current.shape != (7,) or not np.all(np.isfinite(current)):
            return None
        return float(np.max(np.abs(_wrap(current - target))))

    def _prepare_joint_motion(
        self, name: str, *, allow_initial_no_solution: bool
    ) -> bool:
        """Clear a retained firmware NO_SOLUTION before commanding joint motion.

        Returns whether a NO_SOLUTION that survives the clearing nudge must be
        tolerated while monitoring the motion that follows.
        """

        initial_no_solution = self._status_value_is_no_solution(
            self._arm_status_value(self._drivers[name])
        )
        completed_no_solution = self._has_completed_no_solution(
            self._drivers[name]
        )
        tolerate_initial_no_solution = (
            allow_initial_no_solution or completed_no_solution
        )
        if initial_no_solution and not tolerate_initial_no_solution:
            raise RuntimeError("arm_status_2")
        if initial_no_solution:
            # Explicit homing is authorized by --confirm-enable-and-home.
            # A zero-displacement hold/home does not replace V120's
            # retained failed MOVE_P task. Create a small, bounded MOVE_J
            # task first, verify it from this task's motion status and
            # joint feedback, then continue to the requested home target
            # below. V120 may leave arm_status at NO_SOLUTION even after
            # the MOVE_J has completed successfully.
            current_message = self._drivers[name].get_joint_angles()
            current_value = (
                None if current_message is None else current_message.msg
            )
            current_joints = np.asarray(current_value, dtype=float)
            if current_joints.shape != (7,) or not np.all(
                np.isfinite(current_joints)
            ):
                raise RuntimeError(
                    f"{name} cannot clear retained NO_SOLUTION because "
                    f"joint feedback is invalid: {current_value!r}"
                )
            nudge_target = current_joints.copy()
            # Move joint 7 about 1.15 degrees toward its range center. At
            # zero, use the positive direction. Nero J7 is limited to
            # approximately +/- pi/2, so this remains well inside limits.
            nudge_target[6] += 0.02 if current_joints[6] <= 0.0 else -0.02
            self._drivers[name].set_motion_mode("j")
            time.sleep(0.10)
            self._drivers[name].move_j(nudge_target.tolist())
            nudge_start = time.monotonic()
            successful_stable = 0
            last_nudge_joints = current_joints
            while time.monotonic() - nudge_start <= 3.0:
                feedback = self._drivers[name].get_joint_angles()
                if feedback is not None:
                    candidate = np.asarray(feedback.msg, dtype=float)
                    if candidate.shape == (7,) and np.all(
                        np.isfinite(candidate)
                    ):
                        last_nudge_joints = candidate
                nudge_error = float(
                    np.max(np.abs(_wrap(last_nudge_joints - nudge_target)))
                )
                if nudge_error <= 0.01:
                    successful_stable += 1
                    if successful_stable >= 3:
                        break
                else:
                    successful_stable = 0
                error = self._motion_error(
                    name,
                    allow_stale_no_solution=True,
                )
                if error is not None:
                    raise RuntimeError(
                        f"{name} NO_SOLUTION nudge failed: {error}; "
                        f"joint_error_max_rad={nudge_error}; "
                        "status_detail="
                        f"{self._arm_status_diagnostic(self._drivers[name])}"
                    )
                time.sleep(0.02)
            else:
                raise RuntimeError(
                    f"{name} did not reach the joint-7 NO_SOLUTION nudge: "
                    f"start={current_joints.tolist()}, "
                    f"target={nudge_target.tolist()}, "
                    f"feedback={last_nudge_joints.tolist()}, "
                    "status_detail="
                    f"{self._arm_status_diagnostic(self._drivers[name])}"
                )

        retained_no_solution = (
            initial_no_solution
            and self._status_value_is_no_solution(
                self._arm_status_value(self._drivers[name])
            )
        )
        return bool(tolerate_initial_no_solution and retained_no_solution)

    def _wait_for_joint_target(
        self,
        name: str,
        target: np.ndarray,
        timeout_s: float,
        *,
        allow_stale_no_solution: bool,
    ) -> dict[str, Any]:
        """Block until three consecutive reads settle on ``target``."""

        start = time.monotonic()
        target_stable = 0
        while time.monotonic() - start <= timeout_s:
            max_error = self._joint_error_rad(name, target)
            if max_error is not None:
                if max_error <= TRAJECTORY_SETTLE_TOLERANCE_RAD:
                    target_stable += 1
                    if target_stable >= 3:
                        return {
                            "elapsed_s": time.monotonic() - start,
                            "joint_error_max_rad": max_error,
                        }
                else:
                    target_stable = 0
            error = self._motion_error(
                name,
                allow_stale_no_solution=allow_stale_no_solution,
            )
            if error is not None:
                raise RuntimeError(
                    f"{error}; status_detail="
                    f"{self._arm_status_diagnostic(self._drivers[name])}"
                )
            time.sleep(0.05)
        raise TimeoutError(f"{name} arm joint motion timeout")

    def _move_joints_impl(
        self,
        name: str,
        joints: list[float],
        timeout_s: float,
        *,
        allow_initial_no_solution: bool = False,
    ) -> dict[str, Any]:
        target = np.asarray(joints, dtype=float)
        if target.shape != (7,) or not np.all(np.isfinite(target)):
            raise ValueError("joints must contain seven finite radians")
        timeout_s = min(float(timeout_s), self._max_motion_s)
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        try:
            allow_stale_no_solution = self._prepare_joint_motion(
                name, allow_initial_no_solution=allow_initial_no_solution
            )
            self._drivers[name].move_j(target.tolist())
            settled = self._wait_for_joint_target(
                name,
                target,
                timeout_s,
                allow_stale_no_solution=allow_stale_no_solution,
            )
            return {
                "success": True,
                "reason": "target_reached",
                "arm": name,
                "elapsed_s": settled["elapsed_s"],
                "joint_error_max_rad": settled["joint_error_max_rad"],
                "stale_no_solution_ignored": allow_stale_no_solution,
            }
        except BaseException:
            self.emergency_stop()
            raise

    def home_arm(self, arm: int | str, sequence: int, timeout_s: float = 20.0):
        name = _arm_name(arm)
        with self._locks[name]:
            self._accept_sequence(name, sequence)
            self._ensure_motion_allowed(name)
            return self._move_joints_impl(name, self._homes[name], timeout_s)

    @staticmethod
    def _validate_tcp_pose(pose_xyz_rpy: Any) -> np.ndarray:
        target = np.asarray(pose_xyz_rpy, dtype=float)
        if target.shape != (6,) or not np.all(np.isfinite(target)):
            raise ValueError("pose_xyz_rpy must contain six finite values")
        if np.linalg.norm(target[:3]) > 1.0:
            raise ValueError("target position exceeds the 1.0 m arm-base bound")
        if abs(target[3]) > math.pi or abs(target[5]) > math.pi:
            raise ValueError("roll/yaw must be within [-pi, pi]")
        if abs(target[4]) > math.pi / 2:
            raise ValueError("pitch must be within [-pi/2, pi/2]")
        return target

    def _plan_tcp_targets_locked(
        self, name: str, targets: list[np.ndarray]
    ) -> dict[str, Any]:
        """Solve targets from one measured joint seed without moving hardware."""

        solver = self._ik_solvers.get(name)
        common = {
            "arm": name,
            "no_motion": True,
            "cartesian_backend": "mink_move_j",
            "end_effector_frame": self._end_effector_frame,
            "estop_latched": bool(self._estop_latched),
        }
        if solver is None:
            return {
                **common,
                "success": False,
                "reason": "mink_ik_unavailable",
                "plans": [],
            }
        joint_message = self._drivers[name].get_joint_angles()
        start_joints = (
            None
            if joint_message is None
            else np.asarray(joint_message.msg, dtype=float)
        )
        if start_joints is None or start_joints.shape != (7,) or not np.all(
            np.isfinite(start_joints)
        ):
            return {
                **common,
                "success": False,
                "reason": "invalid_joint_feedback_for_mink_ik",
                "plans": [],
            }

        joint_limits = None
        get_joint_limits = getattr(solver, "joint_position_limits", None)
        if callable(get_joint_limits):
            candidate_limits = np.asarray(get_joint_limits(), dtype=float)
            if candidate_limits.shape == (7, 2) and np.all(
                np.isfinite(candidate_limits)
            ):
                joint_limits = candidate_limits

        solve_started = time.monotonic()
        parallel_results = None
        if self._parallel_ik is not None:
            parallel_results = self._parallel_ik.solve(
                name,
                start_joints,
                targets,
                max_iterations=self._ik_max_iterations,
            )
            if len(parallel_results) != len(targets):
                raise RuntimeError(
                    "parallel Mink IK returned an unexpected result count"
                )

        plans: list[dict[str, Any]] = []
        for candidate_index, target in enumerate(targets):
            plan: dict[str, Any] = {
                "candidate_index": candidate_index,
                "target_pose_xyz_rpy": target.tolist(),
            }
            try:
                if parallel_results is None:
                    # Every candidate starts from exactly the same measured
                    # state; an earlier solve must not bias the next candidate.
                    solver.init(start_joints)
                    joint_target, ik_converged, ik_error = (
                        solver.solve_pose_xyz_rpy(
                            target,
                            max_iter=self._ik_max_iterations,
                        )
                    )
                else:
                    result = parallel_results[candidate_index]
                    if int(result.get("candidate_index", -1)) != candidate_index:
                        raise RuntimeError(
                            "parallel Mink IK returned results out of order"
                        )
                    if "error" in result:
                        raise RuntimeError(str(result["error"]))
                    joint_target = result["joint_target"]
                    ik_converged = result["ik_converged"]
                    ik_error = result["ik_error"]
                joint_target = np.asarray(joint_target, dtype=float)
                ik_error = np.asarray(ik_error, dtype=float)
                if joint_target.shape != (7,) or not np.all(
                    np.isfinite(joint_target)
                ):
                    raise RuntimeError("Mink returned invalid joint values")
                if ik_error.shape != (6,) or not np.all(np.isfinite(ik_error)):
                    raise RuntimeError("Mink returned an invalid task error")
            except Exception as exc:
                plan.update(
                    {
                        "success": False,
                        "reason": "mink_ik_error",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                plans.append(plan)
                continue

            # Nero's seven revolute joints are bounded, not continuous.  The
            # move_j path cannot cross a +/-pi seam as a shortcut, so wrapping
            # this difference would under-report the real commanded motion and
            # could make a large flip appear cheaper than it is.
            joint_delta = joint_target - start_joints
            official_margins = np.minimum(
                joint_target - NERO_JOINT_POSITION_LIMITS_RAD[:, 0],
                NERO_JOINT_POSITION_LIMITS_RAD[:, 1] - joint_target,
            )
            margins = None
            if joint_limits is not None:
                margins = np.minimum(
                    joint_target - joint_limits[:, 0],
                    joint_limits[:, 1] - joint_target,
                )
            ik_details = {
                "ik_joint_target": joint_target.tolist(),
                "ik_converged": bool(ik_converged),
                "ik_task_error": ik_error.tolist(),
                "ik_position_error_m": float(np.linalg.norm(ik_error[:3])),
                "ik_rotation_error_rad": float(np.linalg.norm(ik_error[3:])),
                "joint_delta_rad": joint_delta.tolist(),
                "joint_travel_l2_rad": float(np.linalg.norm(joint_delta)),
                "joint_travel_max_rad": float(np.max(np.abs(joint_delta))),
                "joint_limit_margins_rad": (
                    None if margins is None else margins.tolist()
                ),
                "joint_limit_margin_min_rad": (
                    None if margins is None else float(np.min(margins))
                ),
                "official_joint_limit_margins_rad": official_margins.tolist(),
                "official_joint_limit_margin_min_rad": float(
                    np.min(official_margins)
                ),
            }
            violation_indices = np.flatnonzero(official_margins < 0.0)
            if violation_indices.size:
                violations = []
                for joint_index in violation_indices:
                    lower, upper = NERO_JOINT_POSITION_LIMITS_RAD[joint_index]
                    value = float(joint_target[joint_index])
                    violations.append(
                        {
                            "joint_index": int(joint_index),
                            "joint_number": int(joint_index) + 1,
                            "value_rad": value,
                            "minimum_rad": float(lower),
                            "maximum_rad": float(upper),
                            "violation_rad": float(
                                max(float(lower) - value, value - float(upper))
                            ),
                        }
                    )
                plan.update(
                    {
                        "success": False,
                        "reason": "ik_joint_target_exceeds_official_limits",
                        **ik_details,
                        "official_joint_limit_violations": violations,
                    }
                )
                plans.append(plan)
                continue
            plan.update(
                {
                    "success": True,
                    "reason": "planned",
                    **ik_details,
                }
            )
            plans.append(plan)

        solve_elapsed_s = time.monotonic() - solve_started
        valid_count = sum(bool(plan.get("success", False)) for plan in plans)
        return {
            **common,
            "success": valid_count > 0,
            "reason": "planned" if valid_count > 0 else "all_mink_ik_plans_failed",
            "start_joints": start_joints.tolist(),
            "candidate_count": len(targets),
            "valid_plan_count": valid_count,
            "ik_parallel_workers": (
                1
                if self._parallel_ik is None
                else self._parallel_ik.worker_count
            ),
            "ik_solve_elapsed_s": solve_elapsed_s,
            "plans": plans,
        }

    def plan_tcp_poses(
        self, arm: int | str, poses_xyz_rpy: Any
    ) -> dict[str, Any]:
        """Batch-plan TCP poses from current feedback without any arm motion."""

        name = _arm_name(arm)
        raw_targets = list(poses_xyz_rpy)
        if not raw_targets:
            raise ValueError("poses_xyz_rpy must contain at least one pose")
        if len(raw_targets) > 256:
            raise ValueError("poses_xyz_rpy is limited to 256 candidates per call")
        targets = [self._validate_tcp_pose(pose) for pose in raw_targets]
        with self._locks[name]:
            return self._plan_tcp_targets_locked(name, targets)

    def plan_tcp_pose(
        self, arm: int | str, pose_xyz_rpy: Any
    ) -> dict[str, Any]:
        """Plan one TCP pose without consuming a sequence or moving the arm."""

        result = self.plan_tcp_poses(arm, [pose_xyz_rpy])
        if not result.get("plans"):
            result["target_pose_xyz_rpy"] = self._validate_tcp_pose(
                pose_xyz_rpy
            ).tolist()
            return result
        plan = dict(result["plans"][0])
        plan.update(
            {
                key: result[key]
                for key in (
                    "arm",
                    "no_motion",
                    "cartesian_backend",
                    "end_effector_frame",
                    "estop_latched",
                    "start_joints",
                )
                if key in result
            }
        )
        return plan

    def execute_joint_trajectory(
        self,
        arm: int | str,
        waypoints: Any,
        sequence: int,
        timeout_s: float = 60.0,
        waypoint_dt_s: float | None = None,
    ) -> dict[str, Any]:
        """Stream a bounded, pre-planned joint path to the firmware at its rate.

        Planner waypoint ``k`` is reached at ``k * waypoint_dt_s`` after the
        stream starts. Between waypoints the path is linearly interpolated and
        sent with ``move_j`` at ``trajectory_stream_hz``, without waiting for
        arrival: in the Nero position-velocity mode each target overwrites the
        previous one and the firmware smooths the stream into one continuous
        motion (pyAgxArm ``move_j`` documentation). Only the final waypoint
        waits for settled arrival. Throughout, the measured joints must stay
        within the configured path tolerance of the most recently sent target;
        a larger lag means the firmware cannot follow the planned samples, so
        the stream stops, the firmware finishes and holds the last sent target
        (at most one tolerance from a collision-checked sample), and the call
        returns ``joint_trajectory_lag_exceeded`` with the per-waypoint lag so
        the stream rate or the firmware speed can be retuned. Arm faults and
        CAN communication errors still trigger the emergency stop.
        """

        name = _arm_name(arm)
        trajectory = np.asarray(waypoints, dtype=float)
        if (
            trajectory.ndim != 2
            or trajectory.shape[1] != 7
            or not np.all(np.isfinite(trajectory))
        ):
            raise ValueError("waypoints must have finite shape [N, 7]")
        if not 2 <= len(trajectory) <= TRAJECTORY_MAX_WAYPOINTS:
            raise ValueError(
                "joint trajectory must contain between 2 and "
                f"{TRAJECTORY_MAX_WAYPOINTS} waypoints"
            )
        if np.any(trajectory < NERO_JOINT_POSITION_LIMITS_RAD[:, 0]) or np.any(
            trajectory > NERO_JOINT_POSITION_LIMITS_RAD[:, 1]
        ):
            raise ValueError("joint trajectory exceeds official Nero joint limits")
        maximum_step = float(np.max(np.abs(np.diff(trajectory, axis=0))))
        if maximum_step > TRAJECTORY_MAX_JOINT_STEP_RAD:
            raise ValueError(
                f"adjacent trajectory waypoint step {maximum_step:.3f} rad exceeds "
                f"{TRAJECTORY_MAX_JOINT_STEP_RAD:.2f} rad"
            )
        timeout_s = min(float(timeout_s), self._max_motion_s)
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        dt_s = (
            self._trajectory_dt_s
            if waypoint_dt_s is None
            else float(waypoint_dt_s)
        )
        if not np.isfinite(dt_s) or not 0.02 <= dt_s <= 1.0:
            raise ValueError("waypoint_dt_s must be in [0.02, 1.0] s")
        tolerance_rad = self._trajectory_path_tolerance_rad
        substeps = max(1, int(round(dt_s * self._trajectory_stream_hz)))
        frame_dt_s = dt_s / substeps

        with self._locks[name]:
            self._accept_sequence(name, sequence)
            self._ensure_motion_allowed(name)
            start_error = self._joint_error_rad(name, trajectory[0])
            if start_error is None:
                return {
                    "success": False,
                    "reason": "invalid_joint_feedback_for_trajectory",
                    "arm": name,
                }
            if start_error > TRAJECTORY_START_TOLERANCE_RAD:
                return {
                    "success": False,
                    "reason": "trajectory_start_mismatch",
                    "arm": name,
                    "start_error_max_rad": start_error,
                    "allowed_start_error_rad": TRAJECTORY_START_TOLERANCE_RAD,
                }

            deadline = time.monotonic() + timeout_s
            sent: list[dict[str, Any]] = []
            max_lag_rad = 0.0

            def total_timeout() -> dict[str, Any]:
                return {
                    "success": False,
                    "reason": "joint_trajectory_total_timeout",
                    "arm": name,
                    "executed_waypoints": len(sent),
                    "waypoint_count": len(trajectory),
                    "maximum_step_rad": maximum_step,
                    "waypoint_dt_s": dt_s,
                    "stream_hz": self._trajectory_stream_hz,
                    "frames_sent": frames_sent,
                    "max_lag_rad": max_lag_rad,
                }

            frames_sent = 0
            lag_abort: tuple[int, float] | None = None
            try:
                allow_stale_no_solution = self._prepare_joint_motion(
                    name, allow_initial_no_solution=True
                )
                started = time.monotonic()
                last_target = trajectory[0]
                for waypoint_index in range(1, len(trajectory)):
                    previous = trajectory[waypoint_index - 1]
                    waypoint = trajectory[waypoint_index]
                    for substep in range(1, substeps + 1):
                        frame_index = (waypoint_index - 1) * substeps + substep
                        send_at = started + frame_index * frame_dt_s
                        while True:
                            now = time.monotonic()
                            if now >= deadline:
                                return total_timeout()
                            lag_rad = self._joint_error_rad(name, last_target)
                            if lag_rad is not None:
                                max_lag_rad = max(max_lag_rad, lag_rad)
                                if lag_rad > tolerance_rad:
                                    # Not a fault: the firmware is slower than
                                    # the stream. Stop feeding it and let it
                                    # hold the last target.
                                    lag_abort = (waypoint_index, lag_rad)
                                    break
                            if self._drivers[name].has_comm_error():
                                raise RuntimeError(
                                    f"{name} arm CAN communication error while "
                                    f"streaming waypoint {waypoint_index}; "
                                    "status_detail="
                                    f"{self._arm_status_diagnostic(self._drivers[name])}"
                                )
                            error = self._motion_error(
                                name,
                                allow_stale_no_solution=allow_stale_no_solution,
                            )
                            if error is not None:
                                raise RuntimeError(
                                    f"{error}; status_detail="
                                    f"{self._arm_status_diagnostic(self._drivers[name])}"
                                )
                            if now >= send_at:
                                break
                            time.sleep(min(frame_dt_s, send_at - now))
                        if lag_abort is not None:
                            break
                        if substep == substeps:
                            target = waypoint
                        else:
                            target = previous + (waypoint - previous) * (
                                substep / substeps
                            )
                        self._drivers[name].move_j(target.tolist())
                        frames_sent += 1
                        last_target = target
                    if lag_abort is not None:
                        break
                    sent.append(
                        {
                            "waypoint_index": waypoint_index,
                            "sent_at_s": round(time.monotonic() - started, 3),
                            "lag_rad": lag_rad,
                        }
                    )
                stream_s = time.monotonic() - started
                if lag_abort is not None:
                    # Wait (bounded) for the firmware to finish the held target
                    # so the caller sees a stationary arm; keep monitoring
                    # genuine faults meanwhile.
                    hold_deadline = min(
                        deadline, time.monotonic() + TRAJECTORY_LAG_HOLD_SETTLE_S
                    )
                    hold_error = self._joint_error_rad(name, last_target)
                    stable = 0
                    while time.monotonic() < hold_deadline:
                        hold_error = self._joint_error_rad(name, last_target)
                        if (
                            hold_error is not None
                            and hold_error <= TRAJECTORY_SETTLE_TOLERANCE_RAD
                        ):
                            stable += 1
                            if stable >= 3:
                                break
                        else:
                            stable = 0
                        error = self._motion_error(
                            name,
                            allow_stale_no_solution=allow_stale_no_solution,
                        )
                        if error is not None:
                            raise RuntimeError(
                                f"{error}; status_detail="
                                f"{self._arm_status_diagnostic(self._drivers[name])}"
                            )
                        time.sleep(0.05)
                    failed_index, failed_lag = lag_abort
                    return {
                        "success": False,
                        "reason": "joint_trajectory_lag_exceeded",
                        "arm": name,
                        "failed_waypoint_index": failed_index,
                        "lag_rad": failed_lag,
                        "path_tolerance_rad": tolerance_rad,
                        "max_lag_rad": max_lag_rad,
                        "held_target": last_target.tolist(),
                        "held_joint_error_max_rad": hold_error,
                        "executed": sent,
                        "waypoint_count": len(trajectory),
                        "waypoint_dt_s": dt_s,
                        "stream_hz": self._trajectory_stream_hz,
                        "frames_sent": frames_sent,
                        "stream_s": stream_s,
                        "maximum_step_rad": maximum_step,
                        "status_detail": self._arm_status_diagnostic(
                            self._drivers[name]
                        ),
                    }
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    return total_timeout()
                settled = self._wait_for_joint_target(
                    name,
                    trajectory[-1],
                    remaining,
                    allow_stale_no_solution=allow_stale_no_solution,
                )
            except BaseException:
                self.emergency_stop()
                raise
            return {
                "success": True,
                "reason": "joint_trajectory_completed",
                "arm": name,
                "waypoint_count": len(trajectory),
                "maximum_step_rad": maximum_step,
                "start_error_max_rad": start_error,
                "executed": sent,
                "waypoint_dt_s": dt_s,
                "stream_hz": self._trajectory_stream_hz,
                "frames_sent": frames_sent,
                "path_tolerance_rad": tolerance_rad,
                "max_lag_rad": max_lag_rad,
                "stream_s": stream_s,
                "settle_s": settled["elapsed_s"],
                "final_joint_error_max_rad": settled["joint_error_max_rad"],
                "execution_s": time.monotonic() - started,
                "stale_no_solution_ignored": allow_stale_no_solution,
            }

    def move_tcp_pose(
        self,
        arm: int | str,
        pose_xyz_rpy,
        sequence: int,
        timeout_s: float = 60.0,
    ) -> dict[str, Any]:
        name = _arm_name(arm)
        target = self._validate_tcp_pose(pose_xyz_rpy)
        timeout_s = min(float(timeout_s), self._max_motion_s)
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        with self._locks[name]:
            self._accept_sequence(name, sequence)
            self._ensure_motion_allowed(name)
            solver = self._ik_solvers.get(name)
            if solver is None:
                return {
                    "success": False,
                    "reason": "mink_ik_unavailable",
                    "arm": name,
                    "target_pose_xyz_rpy": target.tolist(),
                    "estop_latched": False,
                }
            joint_message = self._drivers[name].get_joint_angles()
            start_joints = (
                None
                if joint_message is None
                else np.asarray(joint_message.msg, dtype=float)
            )
            if start_joints is None or start_joints.shape != (7,) or not np.all(
                np.isfinite(start_joints)
            ):
                return {
                    "success": False,
                    "reason": "invalid_joint_feedback_for_mink_ik",
                    "arm": name,
                    "target_pose_xyz_rpy": target.tolist(),
                    "estop_latched": False,
                }
            try:
                # Seed every solve from measured hardware joints.  We retain
                # Mink's best result even when its strict convergence flag is
                # false, matching robot/arm's established IK behavior; the
                # residual is returned for diagnosis instead of manufacturing
                # another binary NO_SOLUTION gate.
                solver.init(start_joints)
                joint_target, ik_converged, ik_error = solver.solve_pose_xyz_rpy(
                    target,
                    max_iter=self._ik_max_iterations,
                )
            except Exception as exc:
                return {
                    "success": False,
                    "reason": "mink_ik_error",
                    "arm": name,
                    "target_pose_xyz_rpy": target.tolist(),
                    "error": f"{type(exc).__name__}: {exc}",
                    "estop_latched": False,
                }
            joint_target = np.asarray(joint_target, dtype=float)
            ik_error = np.asarray(ik_error, dtype=float)
            if joint_target.shape != (7,) or not np.all(np.isfinite(joint_target)):
                return {
                    "success": False,
                    "reason": "mink_ik_returned_invalid_joints",
                    "arm": name,
                    "target_pose_xyz_rpy": target.tolist(),
                    "estop_latched": False,
                }

            result = self._move_joints_impl(
                name,
                joint_target.tolist(),
                timeout_s,
                # Firmware NO_SOLUTION belongs to an earlier MOVE_P.  The new
                # Cartesian backend does not issue MOVE_P and must not inherit
                # that controller-side planning result.
                allow_initial_no_solution=True,
            )
            final_tcp_message = self._drivers[name].get_tcp_pose()
            final_tcp = (
                None
                if final_tcp_message is None
                else np.asarray(final_tcp_message.msg, dtype=float)
            )
            position_error_m = None
            rotation_error_rad = None
            if final_tcp is not None and final_tcp.shape == (6,) and np.all(
                np.isfinite(final_tcp)
            ):
                position_error_m = float(np.linalg.norm(final_tcp[:3] - target[:3]))
                rotation_error_rad = _rotation_error_rad(final_tcp[3:], target[3:])
            else:
                final_tcp = None
            result.update(
                {
                    "cartesian_backend": "mink_move_j",
                    "end_effector_frame": self._end_effector_frame,
                    "target_pose_xyz_rpy": target.tolist(),
                    "ik_joint_target": joint_target.tolist(),
                    "ik_converged": bool(ik_converged),
                    "ik_task_error": ik_error.tolist(),
                    "final_tcp_pose_xyz_rpy": (
                        None if final_tcp is None else final_tcp.tolist()
                    ),
                    "position_error_m": position_error_m,
                    "rotation_error_rad": rotation_error_rad,
                }
            )
            return result

    def set_gripper(
        self,
        arm: int | str,
        opened: bool,
        sequence: int,
        timeout_s: float = 3.0,
        force_n: float | None = None,
    ) -> dict[str, Any]:
        name = _arm_name(arm)
        if not isinstance(opened, bool):
            raise TypeError("opened must be bool")
        force = self._gripper_force_n if force_n is None else float(force_n)
        if not 0.1 <= force <= 3.0:
            raise ValueError("force_n must be in [0.1, 3.0]")
        target = self._open_width_m if opened else self._close_width_m
        if not 0.0 < float(timeout_s) <= 5.0:
            raise ValueError("timeout_s must be in (0, 5]")
        return self._command_gripper_width(
            name,
            target,
            sequence,
            timeout_s=float(timeout_s),
            force_n=force,
            allow_stable_contact=not opened,
            success_reason="open_confirmed" if opened else "close_stable",
        )

    def commission_gripper_width(
        self,
        arm: int | str,
        width_m: float,
        sequence: int,
        confirmation: str,
        timeout_s: float = 3.0,
        force_n: float | None = None,
    ) -> dict[str, Any]:
        """Move one gripper to an exact supervised commissioning width.

        This operator-only endpoint is intentionally absent from the Jetson
        hardware wrapper and generated-agent primitive registry.  It permits
        staged low-force validation before real ``open_gripper`` and
        ``close_gripper`` are enabled.
        """

        if confirmation != GRIPPER_COMMISSIONING_CONFIRMATION:
            raise PermissionError(
                "gripper commissioning requires the exact operator confirmation"
            )
        name = _arm_name(arm)
        target = float(width_m)
        if not 0.0 <= target <= self._open_width_m:
            raise ValueError(
                f"width_m must be in [0, {self._open_width_m:.3f}]"
            )
        force = self._gripper_force_n if force_n is None else float(force_n)
        if not 0.1 <= force <= 3.0:
            raise ValueError("force_n must be in [0.1, 3.0]")
        if not 0.0 < float(timeout_s) <= 5.0:
            raise ValueError("timeout_s must be in (0, 5]")
        return self._command_gripper_width(
            name,
            target,
            sequence,
            timeout_s=float(timeout_s),
            force_n=force,
            allow_stable_contact=False,
            success_reason="commissioning_width_confirmed",
        )

    def _command_gripper_width(
        self,
        name: str,
        target: float,
        sequence: int,
        *,
        timeout_s: float,
        force_n: float,
        allow_stable_contact: bool,
        success_reason: str,
    ) -> dict[str, Any]:
        with self._locks[name]:
            self._accept_sequence(name, sequence)
            self._ensure_motion_allowed(name)
            start = time.monotonic()
            initial_message = self._grippers[name].get_gripper_status()
            initial_payload = _message_payload(initial_message)
            initial_width = (
                None if initial_payload is None else float(initial_payload.value)
            )
            enabled_seen = bool(
                initial_payload is not None
                and initial_payload.foc_status.driver_enable_status
            )
            last_timestamp = self._feedback_timestamp(initial_message)
            deadline = start + min(timeout_s, 5.0)
            enable_deadline = min(deadline, start + 1.0)
            self._grippers[name].move_gripper_m(value=target, force=force_n)
            previous: float | None = None
            stable = 0
            while time.monotonic() <= deadline:
                message = self._grippers[name].get_gripper_status()
                payload = _message_payload(message)
                if payload is not None:
                    width = float(payload.value)
                    foc_status = payload.foc_status
                    enabled = bool(foc_status.driver_enable_status)
                    fault_fields = (
                        "voltage_too_low",
                        "motor_overheating",
                        "driver_overcurrent",
                        "driver_overheating",
                        "sensor_status",
                        "driver_error_status",
                    )
                    faults = [
                        field
                        for field in fault_fields
                        if bool(getattr(foc_status, field, False))
                    ]
                    if faults:
                        raise RuntimeError(
                            f"{name} gripper fault: {','.join(faults)}"
                        )
                    if not enabled:
                        # move_gripper_m also enables the gripper. The SDK
                        # initially returns cached disabled feedback while
                        # that command is taking effect (about 0.2 s on Nero).
                        # Only allow this delay before the first enable;
                        # losing enable during motion remains an error.
                        if enabled_seen or time.monotonic() >= enable_deadline:
                            raise RuntimeError(f"{name} gripper driver is disabled")
                        stable = 0
                        previous = None
                        time.sleep(0.05)
                        continue
                    timestamp = self._feedback_timestamp(message)
                    if (
                        timestamp is not None
                        and last_timestamp is not None
                        and timestamp <= last_timestamp
                    ):
                        # Do not confirm motion using a cached pre-command
                        # sample, or count the same sample multiple times.
                        time.sleep(0.05)
                        continue
                    last_timestamp = timestamp
                    enabled_seen = True
                    if abs(width - target) <= 0.005:
                        stable += 1
                    elif (
                        allow_stable_contact
                        and previous is not None
                        and abs(width - previous) <= 0.001
                        and (
                            width <= self._close_width_m + 0.003
                            or (
                                initial_width is not None
                                and initial_width - width >= 0.002
                            )
                        )
                    ):
                        # Closing on an object legitimately stops above zero width.
                        stable += 1
                    else:
                        stable = 0
                    previous = width
                    if stable >= 4:
                        return {
                            "success": True,
                            "reason": success_reason,
                            "arm": name,
                            "target_width_m": target,
                            "elapsed_s": time.monotonic() - start,
                            "width_m": width,
                            "force_n": float(payload.force),
                        }
                time.sleep(0.05)
            raise TimeoutError(
                f"{name} gripper motion timeout: target_width_m={target:.3f}"
            )

    def emergency_stop(self) -> dict[str, Any]:
        self._estop_latched = True
        self._calibration_drag_arm = None
        self._calibration_drag_started_monotonic = None
        self._calibration_drag_entry_joints = None
        errors = {}
        for name, driver in self._drivers.items():
            try:
                driver.electronic_emergency_stop()
            except Exception as exc:
                errors[name] = f"{type(exc).__name__}: {exc}"
        return {
            "stopped": not errors,
            "estop_latched": True,
            "errors": errors,
        }

    def close(self) -> None:
        self._estop_latched = True
        for name in ("left", "right"):
            for operation in (
                self._drivers[name].electronic_emergency_stop,
                self._grippers[name].disable_gripper,
            ):
                try:
                    operation()
                except Exception:
                    pass
            try:
                self._drivers[name].disconnect()
            except Exception:
                pass
        if self._parallel_ik is not None:
            self._parallel_ik.close()


def _create_pair(args: argparse.Namespace) -> NeroArmPairRPC:
    from pyAgxArm import AgxArmFactory, ArmModel, NeroFW, create_agx_arm_config

    drivers = {}
    grippers = {}
    tcp_offsets = _resolve_tcp_offsets(args)
    for name, channel in (("left", args.left_channel), ("right", args.right_channel)):
        cfg = create_agx_arm_config(
            robot=ArmModel.NERO,
            firmeware_version=NeroFW.V120,
            interface="socketcan",
            channel=channel,
            bitrate=1_000_000,
            # Only the elbow differs from the vendor preset; the driver would
            # otherwise clamp the rest pose's elbow to the preset.
            joint_limits={
                ELBOW_JOINT_NAME: [
                    float(value) for value in NERO_JOINT_POSITION_LIMITS_RAD[3]
                ]
            },
        )
        driver = AgxArmFactory.create_arm(cfg)
        driver.set_tcp_offset(tcp_offsets[name])
        gripper = driver.init_effector(driver.OPTIONS.EFFECTOR.AGX_GRIPPER)
        drivers[name] = driver
        grippers[name] = gripper

    ik_solvers: dict[str, Any] = {}
    parallel_ik: ParallelMinkIK | None = None
    if args.recover_estop_only is None:
        try:
            from robot.arm.ik_solver import SingleArmIK
        except ImportError as exc:
            raise RuntimeError(
                "Mink IK dependencies are unavailable. Start the Pi arm service "
                "from an environment containing mujoco, mink, and daqp, and add "
                "the YOR repository root to PYTHONPATH."
            ) from exc
        mjcf_path = str(Path(args.ik_mjcf_path).expanduser().resolve())
        for name in ("left", "right"):
            joint_names = [f"{name}_arm_joint{index}" for index in range(1, 8)]
            ee_frame = (
                f"{name}_arm_ee"
                if args.end_effector_frame == "tcp"
                else f"{name}_arm_flange"
            )
            solver = SingleArmIK(
                mjcf_path,
                solver_dt=args.ik_solver_dt,
                joint_names=joint_names,
                ee_frame=ee_frame,
                root_frame=f"{name}_arm_base_link",
            )
            if args.end_effector_frame == "tcp":
                solver.set_end_effector_offset(tcp_offsets[name])
            ik_solvers[name] = solver
        if args.ik_workers > 1:
            solver_specs = {
                name: {
                    "mjcf_path": mjcf_path,
                    "solver_dt": args.ik_solver_dt,
                    "joint_names": [
                        f"{name}_arm_joint{index}" for index in range(1, 8)
                    ],
                    "ee_frame": (
                        f"{name}_arm_ee"
                        if args.end_effector_frame == "tcp"
                        else f"{name}_arm_flange"
                    ),
                    "root_frame": f"{name}_arm_base_link",
                    "tcp_offset": (
                        tcp_offsets[name]
                        if args.end_effector_frame == "tcp"
                        else None
                    ),
                }
                for name in ("left", "right")
            }
            parallel_ik = ParallelMinkIK(
                solver_specs,
                worker_count=args.ik_workers,
                batch_timeout_s=args.ik_batch_timeout_s,
            )
    return NeroArmPairRPC(
        drivers["left"],
        drivers["right"],
        grippers["left"],
        grippers["right"],
        speed_percent=args.speed_percent,
        max_motion_s=args.max_motion_s,
        open_width_m=args.open_width_m,
        close_width_m=args.close_width_m,
        gripper_force_n=args.gripper_force_n,
        tcp_offsets=tcp_offsets,
        ik_solvers=ik_solvers,
        parallel_ik=parallel_ik,
        end_effector_frame=args.end_effector_frame,
        ik_max_iterations=args.ik_max_iterations,
        allow_calibration_drag=args.allow_calibration_drag,
        trajectory_dt_s=args.trajectory_dt_s,
        trajectory_stream_hz=args.trajectory_stream_hz,
        trajectory_path_tolerance_rad=args.trajectory_path_tolerance_rad,
    )


def _resolve_tcp_offsets(args: argparse.Namespace) -> dict[str, list[float]]:
    """Resolve the shared contact-TCP length and optional per-arm overrides."""

    shared_x_m = float(args.official_gripper_tcp_x_m)
    if not math.isfinite(shared_x_m) or not 0.0 <= shared_x_m <= 0.40:
        raise ValueError("official_gripper_tcp_x_m must be within [0, 0.40] m")
    shared = list(NERO_OFFICIAL_GRIPPER_TCP_OFFSET)
    shared[0] = shared_x_m
    offsets: dict[str, list[float]] = {}
    for name in ("left", "right"):
        override = getattr(args, f"{name}_tcp_offset")
        values = shared if override is None else [float(value) for value in override]
        if len(values) != 6 or not np.all(np.isfinite(values)):
            raise ValueError(f"{name}_tcp_offset must contain six finite values")
        offsets[name] = list(values)
    return offsets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="YOR dual Nero V120 arm RPC with Raspberry Pi Mink IK"
    )
    parser.add_argument("--port", type=int, default=5558)
    parser.add_argument("--left-channel", default="can_left")
    parser.add_argument("--right-channel", default="can_right")
    parser.add_argument(
        "--speed-percent",
        type=int,
        default=20,
        help=(
            "firmware position-velocity speed [1, 30]. 10 followed streamed "
            "cuRobo paths at about 0.3-0.4 rad/s and fell 0.15 rad behind their "
            "~0.47 rad/s peaks within 2 s (2026-09-07); 20 leaves headroom"
        ),
    )
    parser.add_argument(
        "--max-motion-s",
        type=float,
        default=60.0,
        help="hard monitoring deadline for one arm motion, in seconds",
    )
    parser.add_argument(
        "--trajectory-dt-s",
        type=float,
        default=TRAJECTORY_DEFAULT_DT_S,
        help=(
            "seconds between streamed execute_joint_trajectory waypoints when "
            "the caller does not pass waypoint_dt_s (planner sample spacing)"
        ),
    )
    parser.add_argument(
        "--trajectory-stream-hz",
        type=float,
        default=TRAJECTORY_DEFAULT_STREAM_HZ,
        help=(
            "rate at which linearly interpolated joint targets are streamed "
            "between planner waypoints during execute_joint_trajectory"
        ),
    )
    parser.add_argument(
        "--trajectory-path-tolerance-rad",
        type=float,
        default=TRAJECTORY_PATH_TOLERANCE_RAD,
        help=(
            "emergency-stop a streamed trajectory when the measured joints lag "
            "the most recent target by more than this many radians"
        ),
    )
    parser.add_argument("--open-width-m", type=float, default=0.10)
    parser.add_argument("--close-width-m", type=float, default=0.0)
    parser.add_argument("--gripper-force-n", type=float, default=1.0)
    parser.add_argument(
        "--official-gripper-tcp-x-m",
        type=float,
        default=float(NERO_OFFICIAL_GRIPPER_TCP_OFFSET[0]),
        help=(
            "shared pyAgxArm-flange +X distance to the official gripper's "
            "grasp-volume-center TCP; default is 0.142 m; "
            "use a full per-arm 6D offset for a differently mounted tool"
        ),
    )
    parser.add_argument(
        "--left-tcp-offset",
        type=float,
        nargs=6,
        metavar=("X", "Y", "Z", "ROLL", "PITCH", "YAW"),
        default=None,
        help=(
            "optional full left TCP override in the pyAgxArm/link7 flange "
            "frame; takes precedence over --official-gripper-tcp-x-m"
        ),
    )
    parser.add_argument(
        "--right-tcp-offset",
        type=float,
        nargs=6,
        metavar=("X", "Y", "Z", "ROLL", "PITCH", "YAW"),
        default=None,
        help=(
            "optional full right TCP override in the pyAgxArm/link7 flange "
            "frame; takes precedence over --official-gripper-tcp-x-m"
        ),
    )
    parser.add_argument(
        "--end-effector-frame",
        choices=("tcp", "flange"),
        default="tcp",
        help=(
            "pose controlled by Mink and exposed through move_tcp_pose; tcp uses "
            "the configured tool offset, flange forces a zero offset"
        ),
    )
    parser.add_argument(
        "--ik-mjcf-path",
        type=Path,
        default=DEFAULT_NERO_MJCF,
        help="MuJoCo model used by the Raspberry Pi Mink IK backend",
    )
    parser.add_argument(
        "--ik-solver-dt",
        type=float,
        default=0.01,
        help="Mink integration timestep",
    )
    parser.add_argument(
        "--ik-max-iterations",
        type=int,
        default=100,
        help="Mink iterations per requested final pose",
    )
    parser.add_argument(
        "--ik-workers",
        type=int,
        default=min(4, os.cpu_count() or 1),
        choices=range(1, 17),
        metavar="N",
        help=(
            "persistent CPU processes for batched Mink IK; 1 preserves the "
            "serial implementation"
        ),
    )
    parser.add_argument(
        "--ik-batch-timeout-s",
        type=float,
        default=8.0,
        help="hard deadline for one read-only parallel Mink IK batch",
    )
    parser.add_argument(
        "--confirm-enable-and-home",
        action="store_true",
        help="required acknowledgement that both arms may enable and move home",
    )
    parser.add_argument(
        "--allow-calibration-drag",
        action="store_true",
        help=(
            "expose operator-only leader zero-force drag RPC methods; the "
            "caller must still provide the per-arm physical-support confirmation"
        ),
    )
    parser.add_argument(
        "--confirm-reset-estop",
        action="store_true",
        help=(
            "authorize enable/reset of an emergency-stopped arm; reset removes "
            "motor power and a raised arm may fall immediately"
        ),
    )
    parser.add_argument(
        "--recover-estop-only",
        choices=("left", "right"),
        help=(
            "recover only the selected arm, verify reset powered all joints "
            "off, and exit without re-enabling or homing"
        ),
    )
    parser.add_argument(
        "--skip-usb-serial-check",
        action="store_true",
        help="unsafe unless adapter identity was verified by another mechanism",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    recovery_only = args.recover_estop_only is not None
    if recovery_only and not args.confirm_reset_estop:
        raise SystemExit(
            "refusing emergency-stop reset without --confirm-reset-estop"
        )
    if not recovery_only and not args.confirm_enable_and_home:
        raise SystemExit("refusing arm motion without --confirm-enable-and-home")
    if not args.skip_usb_serial_check:
        if recovery_only:
            recovery_name = str(args.recover_estop_only)
            recovery_channel = getattr(args, f"{recovery_name}_channel")
            _verify_interface_serial(
                recovery_channel,
                EXPECTED_SERIALS[f"can_{recovery_name}"],
            )
        else:
            _verify_interface_serial(
                args.left_channel, EXPECTED_SERIALS["can_left"]
            )
            _verify_interface_serial(
                args.right_channel, EXPECTED_SERIALS["can_right"]
            )
    api = _create_pair(args)
    if recovery_only:
        result = api._recover_estop_for_operator(args.recover_estop_only)
        print(f"[agents-yor-arm] recovery result={result}", flush=True)
        return 0
    try:
        from commlink import RPCServer
    except ImportError:
        raise SystemExit(
            "commlink is required for the Nero RPC service but is unavailable "
            "in the nero-official environment"
        )
    try:
        api.initialize(home=True, timeout_s=args.max_motion_s)
        # Worker startup is deliberately after all enable/home motion and waits
        # for model loading to finish before exposing the RPC port.
        api.start_parallel_ik()
    except BaseException:
        api.close()
        raise
    server = RPCServer(api, port=args.port, threaded=True)
    server.start()
    print(
        f"[agents-yor-arm] RPC port={args.port} firmware=V120 "
        f"speed={args.speed_percent}% max_motion={args.max_motion_s:.1f}s "
        f"cartesian=mink_move_j endpoint={args.end_effector_frame} "
        f"ik_workers={args.ik_workers} "
        f"ik_batch_timeout={args.ik_batch_timeout_s:.1f}s "
        f"gripper_open={args.open_width_m:.3f}m "
        f"gripper_force={args.gripper_force_n:.3f}N "
        f"tcp_offsets={api._tcp_offsets} "
        f"calibration_drag={'ENABLED' if args.allow_calibration_drag else 'disabled'}",
        flush=True,
    )
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("[agents-yor-arm] stopping", flush=True)
    finally:
        try:
            api.close()
        finally:
            server.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
