from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import signal
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock

import numpy as np

SERVICES_ROOT = Path(__file__).resolve().parents[2] / "services"
if str(SERVICES_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVICES_ROOT))

from nero_arm.service import (
    GRIPPER_COMMISSIONING_CONFIRMATION,
    NERO_JOINT_POSITION_LIMITS_RAD,
    NERO_OFFICIAL_GRIPPER_TCP_OFFSET,
    SHOULDER_CLEARANCE_DIRECTION,
    SHOULDER_CLEARANCE_JOINT,
    SHOULDER_CLEARANCE_RAD,
    NeroArmPairRPC,
    _initialize_mink_worker,
    _resolve_tcp_offsets,
    _rotation_error_rad,
    _verify_interface_serial,
)

# Where torque-off left the left arm, against the base frame (2026-09-08), and
# its mirror for the right arm (joints 1, 3, 5 and 6 flipped, as the rest pose).
LEFT_HANGING = [0.0936, 1.5535, -2.0642, 0.3118, -1.1034, -0.0711, -0.0812]
RIGHT_HANGING = [-0.0936, 1.5535, 2.0642, 0.3118, 1.1034, 0.0711, -0.0812]


@dataclass
class Message:
    msg: object
    timestamp: float | None = None


class SDKIntEnumLike(int):
    """Model an SDK enum whose pickle would require its defining module."""


@dataclass
class ArmStatus:
    ctrl_mode: int = 1
    arm_status: int = 0
    mode_feedback: int = 1
    teach_status: int = 0
    motion_status: int = 0
    err_code: int = 0


@dataclass
class FocStatus:
    driver_enable_status: bool = True
    voltage_too_low: bool = False
    motor_overheating: bool = False
    driver_overcurrent: bool = False
    driver_overheating: bool = False
    sensor_status: bool = False
    driver_error_status: bool = False


@dataclass
class GripperStatus:
    value: float
    force: float = 1.0
    mode: str = "width"
    foc_status: FocStatus = None

    def __post_init__(self):
        if self.foc_status is None:
            self.foc_status = FocStatus()


class FakeDriver:
    def __init__(self) -> None:
        self.connected = True
        self.enabled = True
        self.joints = [0.0] * 7
        self.leader_joints = [0.0] * 7
        self.tcp = [0.0] * 6
        self.estopped = False
        self.estop_calls = 0
        self.estop_raises = False
        self.disable_calls = 0
        self.arm_status = 0
        self.motion_status = 0
        self.reset_calls = 0
        self.speed_percent = None
        self.enable_calls = 0
        self.ready_after_enable_calls = 1
        self.ctrl_mode = 1
        self.leader_mode_calls = 0
        self.follower_mode_calls = 0
        self.move_j_calls = []
        self.leader_active = False
        self.standard_feedback_timestamp = 1.0
        self.leader_feedback_timestamp = 0.0
        self.status_none_reads_after_reset = 0
        self.status_none_reads_remaining = 0
        self.ignored_reset_calls_remaining = 0
        self.preserve_no_solution_until_joint_move = False
        self.retain_no_solution_after_successful_joint_move = False
        self.motion_mode_calls = []
        self.reset_keeps_status = False
        self.status_feedback_timestamp = 1.0
        self.status_none_reads_after_connect = 0
        self.joint_move_sets_motion_pending = False
        self.ik_message = None
        # Per-joint power on top of the whole-arm ``enabled`` flag.
        self.disabled_joints: set[int] = set()
        self.single_joint_disable_takes = True
        # The alternative the shoulder swing has to survive: a firmware
        # that reports JOINT_BRAKE_NOT_RELEASED and ignores motion while any
        # joint is off.
        self.ignores_motion_while_partly_off = False
        self.enable_status_at_move_j: list[list[bool]] = []

    def set_joint_limits_enabled(self, enabled):
        self.joint_limits_enabled = bool(enabled)

    def connect(self):
        self.connected = True
        self.status_none_reads_remaining = self.status_none_reads_after_connect

    def disconnect(self):
        self.connected = False

    def enable(self, joint_index=255, timeout=1.5):
        if joint_index != 255:
            self.disabled_joints.discard(int(joint_index))
            return True
        self.enable_calls += 1
        self.enabled = self.enable_calls >= self.ready_after_enable_calls
        if self.enabled:
            self.disabled_joints.clear()
        if (
            self.arm_status != 1
            and self.enabled
            and not (
                self.arm_status == 2
                and self.preserve_no_solution_until_joint_move
            )
        ):
            self.arm_status = 0
        return True

    def disable(self, joint_index=255, timeout=1.5):
        if joint_index != 255:
            if self.single_joint_disable_takes:
                self.disabled_joints.add(int(joint_index))
                if self.ignores_motion_while_partly_off:
                    self.arm_status = 6
            return True
        self.disable_calls += 1
        self.enabled = False
        self.arm_status = 6
        return True

    def reset(self):
        self.reset_calls += 1
        if self.ignored_reset_calls_remaining:
            self.ignored_reset_calls_remaining -= 1
            return
        self.enabled = False
        if not self.reset_keeps_status:
            self.arm_status = 6
        self.enable_calls = 0
        self.status_none_reads_remaining = self.status_none_reads_after_reset

    def set_speed_percent(self, percent):
        self.speed_percent = int(percent)

    def is_connected(self):
        return self.connected

    def is_ok(self):
        return True

    def has_comm_error(self):
        return False

    def get_joint_angles(self):
        if not self.leader_active:
            self.standard_feedback_timestamp += 0.01
        return Message(self.joints, self.standard_feedback_timestamp)

    def get_leader_joint_angles(self):
        if not self.leader_active:
            return None
        self.leader_feedback_timestamp += 0.01
        return Message(self.leader_joints, self.leader_feedback_timestamp)

    def fk(self, joints):
        return self.tcp

    def get_flange_pose(self):
        return Message(self.tcp)

    def get_tcp_pose(self):
        return Message(self.tcp)

    def get_arm_status(self):
        if self.status_none_reads_remaining:
            self.status_none_reads_remaining -= 1
            return None
        self.status_feedback_timestamp += 0.01
        return Message(
            ArmStatus(
                ctrl_mode=self.ctrl_mode,
                arm_status=self.arm_status,
                motion_status=self.motion_status,
            ),
            self.status_feedback_timestamp,
        )

    def get_ik_joint_angles(self):
        return self.ik_message

    def get_joints_enable_status_list(self):
        return [
            self.enabled and joint not in self.disabled_joints
            for joint in range(1, 8)
        ]

    def get_tcp2flange_pose(self, pose):
        return pose

    def move_j(self, joints):
        self.enable_status_at_move_j.append(self.get_joints_enable_status_list())
        if self.ignores_motion_while_partly_off and self.disabled_joints:
            self.move_j_calls.append(list(joints))
            return
        previous = self.joints.copy()
        self.joints = list(joints)
        self.move_j_calls.append(list(joints))
        if self.joint_move_sets_motion_pending:
            self.motion_status = 1
        changed = any(
            abs(float(after) - float(before)) > 1e-6
            for before, after in zip(previous, self.joints)
        )
        if (
            self.arm_status == 2
            and self.preserve_no_solution_until_joint_move
            and changed
            and not self.retain_no_solution_after_successful_joint_move
        ):
            self.arm_status = 0

    def set_motion_mode(self, mode):
        self.motion_mode_calls.append(mode)

    def set_leader_mode(self):
        self.leader_mode_calls += 1
        self.leader_joints = self.joints.copy()
        # The real V112/V120 SDK disables ordinary CAN status push before
        # linkage configuration, so ctrl_mode may remain cached at CAN_CTRL.
        self.leader_active = True

    def set_follower_mode(self):
        self.follower_mode_calls += 1
        self.leader_active = False

    def electronic_emergency_stop(self):
        self.estop_calls += 1
        self.estopped = True
        self.arm_status = 1
        if self.estop_raises:
            raise RuntimeError("simulated stop failure")


class FakeGripper:
    def __init__(self) -> None:
        self.width = 0.0
        self.force = 1.0
        self.commands = []
        self.follow_commands = True
        self.foc_status = FocStatus()

    def get_gripper_status(self):
        return Message(
            GripperStatus(self.width, self.force, foc_status=self.foc_status)
        )

    def move_gripper_m(self, value, force):
        self.commands.append((value, force))
        if self.follow_commands:
            self.width = value
        self.force = force

    def disable_gripper(self):
        return True


class SequencedGripper(FakeGripper):
    """Deliver controlled feedback samples only after a command is sent."""

    def __init__(self, initial: Message, samples: list[Message | None]) -> None:
        super().__init__()
        self.initial = initial
        self.samples = samples
        self.sample_index = 0

    def get_gripper_status(self):
        if not self.commands:
            return self.initial
        sample = self.samples[min(self.sample_index, len(self.samples) - 1)]
        self.sample_index += 1
        return sample


class GripperTestClock:
    def __init__(self) -> None:
        self.now = 0.0

    def monotonic(self):
        return self.now

    def sleep(self, duration):
        self.now += duration


class FakeIK:
    def __init__(self) -> None:
        self.init_calls = []
        self.solve_calls = []
        self.joint_target = [0.1, 0.2, -0.3, 0.4, -0.2, 0.1, 0.0]
        self.converged = True
        self.task_error = np.zeros(6)
        self.error = None

    def init(self, joints) -> None:
        self.init_calls.append(np.asarray(joints, dtype=float).tolist())

    def solve_pose_xyz_rpy(self, pose, *, max_iter):
        self.solve_calls.append((np.asarray(pose, dtype=float).tolist(), max_iter))
        if self.error is not None:
            raise self.error
        return (
            np.asarray(self.joint_target, dtype=float),
            self.converged,
            np.asarray(self.task_error, dtype=float),
        )

    def joint_position_limits(self):
        return np.asarray([[-2.0, 2.0]] * 7, dtype=float)


class FakeParallelIK:
    worker_count = 4

    def __init__(self) -> None:
        self.calls = []
        self.closed = False
        self.started = False

    def start(self):
        self.started = True

    def solve(self, name, start_joints, targets, *, max_iterations):
        self.calls.append(
            {
                "name": name,
                "start_joints": np.asarray(start_joints).copy(),
                "targets": [np.asarray(target).copy() for target in targets],
                "max_iterations": max_iterations,
            }
        )
        return [
            {
                "candidate_index": index,
                "joint_target": [0.1, 0.2, -0.3, 0.4, -0.2, 0.1, 0.0],
                "ik_converged": True,
                "ik_error": [0.0] * 6,
            }
            for index in range(len(targets))
        ]

    def close(self):
        self.closed = True


class NeroArmPairRPCTest(unittest.TestCase):
    def setUp(self) -> None:
        self.left = FakeDriver()
        self.right = FakeDriver()
        self.left_gripper = FakeGripper()
        self.right_gripper = FakeGripper()
        self.left_ik = FakeIK()
        self.right_ik = FakeIK()
        self.rpc = NeroArmPairRPC(
            self.left,
            self.right,
            self.left_gripper,
            self.right_gripper,
            ik_solvers={"left": self.left_ik, "right": self.right_ik},
        )

    def test_move_tcp_pose_is_monitored_and_replay_rejected(self) -> None:
        target = [0.3, 0.1, 0.4, 0.0, 0.2, 0.0]
        result = self.rpc.move_tcp_pose("left", target, 10, 1.0)

        self.assertTrue(result["success"])
        self.assertEqual(self.left.move_j_calls[-1], self.left_ik.joint_target)
        self.assertEqual(self.left_ik.solve_calls[-1], (target, 100))
        self.assertEqual(result["cartesian_backend"], "mink_move_j")
        with self.assertRaises(ValueError):
            self.rpc.move_tcp_pose("left", target, 10, 1.0)

    def test_missing_can_interface_reports_recovery_action(self) -> None:
        with self.assertRaisesRegex(
            RuntimeError,
            "restore the corresponding USB-CAN adapter/interface",
        ):
            _verify_interface_serial("definitely_missing_can", "unused")

    def test_mink_workers_ignore_terminal_interrupts(self) -> None:
        fake_ik_module = ModuleType("robot.arm.ik_solver")
        fake_ik_module.SingleArmIK = object
        with mock.patch.dict(
            sys.modules, {"robot.arm.ik_solver": fake_ik_module}
        ), mock.patch("nero_arm.service.signal.signal") as set_signal:
            _initialize_mink_worker({})

        set_signal.assert_called_once_with(signal.SIGINT, signal.SIG_IGN)

    def test_batch_ik_plan_is_read_only_and_reseeds_each_candidate(self) -> None:
        self.left.joints = [0.05] * 7
        targets = [
            [0.3, 0.1, 0.4, 0.0, 0.2, 0.0],
            [0.25, -0.1, 0.35, 0.1, 0.0, -0.2],
        ]

        result = self.rpc.plan_tcp_poses("left", targets)

        self.assertTrue(result["success"])
        self.assertTrue(result["no_motion"])
        self.assertEqual(result["candidate_count"], 2)
        self.assertEqual(result["valid_plan_count"], 2)
        self.assertEqual(self.left.move_j_calls, [])
        self.assertEqual(self.rpc._last_sequence["left"], -1)
        self.assertEqual(self.left_ik.init_calls, [[0.05] * 7, [0.05] * 7])
        self.assertEqual([plan["candidate_index"] for plan in result["plans"]], [0, 1])
        self.assertIn("joint_limit_margin_min_rad", result["plans"][0])
        self.assertIn("official_joint_limit_margin_min_rad", result["plans"][0])

    def test_batch_ik_uses_parallel_worker_pool_without_shared_solver_state(
        self,
    ) -> None:
        parallel = FakeParallelIK()
        rpc = NeroArmPairRPC(
            self.left,
            self.right,
            self.left_gripper,
            self.right_gripper,
            ik_solvers={"left": self.left_ik, "right": self.right_ik},
            parallel_ik=parallel,
        )
        targets = [
            [0.3, 0.1, 0.4, 0.0, 0.2, 0.0],
            [0.25, -0.1, 0.35, 0.1, 0.0, -0.2],
        ]

        result = rpc.plan_tcp_poses("left", targets)

        self.assertTrue(result["success"])
        self.assertEqual(result["ik_parallel_workers"], 4)
        self.assertEqual(len(parallel.calls), 1)
        self.assertEqual(parallel.calls[0]["max_iterations"], 100)
        self.assertEqual(self.left_ik.init_calls, [])
        self.assertEqual(self.left_ik.solve_calls, [])
        self.assertEqual(self.left.move_j_calls, [])

    def test_parallel_workers_start_only_after_explicit_post_home_hook(self) -> None:
        parallel = FakeParallelIK()
        rpc = NeroArmPairRPC(
            self.left,
            self.right,
            self.left_gripper,
            self.right_gripper,
            ik_solvers={"left": self.left_ik, "right": self.right_ik},
            parallel_ik=parallel,
        )

        self.assertFalse(parallel.started)
        rpc.start_parallel_ik()

        self.assertTrue(parallel.started)

    def test_batch_ik_rejects_official_limit_violation_and_keeps_searching(
        self,
    ) -> None:
        targets = [
            [0.3, 0.1, 0.4, 0.0, 0.2, 0.0],
            [0.25, -0.1, 0.35, 0.1, 0.0, -0.2],
        ]
        returned_joint_targets = iter(
            [
                [NERO_JOINT_POSITION_LIMITS_RAD[5, 1] + 0.01] * 7,
                [0.1, 0.2, -0.3, 0.4, -0.2, 0.1, 0.0],
            ]
        )

        def sequenced_solve(pose, *, max_iter):
            self.left_ik.solve_calls.append(
                (np.asarray(pose, dtype=float).tolist(), max_iter)
            )
            return np.asarray(next(returned_joint_targets)), True, np.zeros(6)

        self.left_ik.solve_pose_xyz_rpy = sequenced_solve

        result = self.rpc.plan_tcp_poses("left", targets)

        self.assertTrue(result["success"])
        self.assertEqual(result["valid_plan_count"], 1)
        rejected, accepted = result["plans"]
        self.assertFalse(rejected["success"])
        self.assertEqual(
            rejected["reason"], "ik_joint_target_exceeds_official_limits"
        )
        self.assertLess(rejected["official_joint_limit_margin_min_rad"], 0.0)
        self.assertTrue(rejected["official_joint_limit_violations"])
        self.assertTrue(accepted["success"])
        self.assertEqual(accepted["reason"], "planned")
        self.assertEqual(len(self.left_ik.solve_calls), 2)
        self.assertEqual(self.left.move_j_calls, [])

    def test_joint_trajectory_execution_checks_start_and_segments(self) -> None:
        trajectory = [[0.0] * 7, [0.05] * 7, [0.10] * 7]

        result = self.rpc.execute_joint_trajectory(
            "left", trajectory, sequence=9, timeout_s=2.0
        )

        self.assertTrue(result["success"])
        self.assertEqual(result["waypoint_count"], 3)
        # The planner samples are interpolated to the stream rate; every
        # planner waypoint is still sent exactly, in order, as the last frame
        # of its segment.
        substeps = round(0.10 * self.rpc._trajectory_stream_hz)
        self.assertEqual(len(self.left.move_j_calls), 2 * substeps)
        self.assertEqual(result["frames_sent"], 2 * substeps)
        self.assertEqual(
            self.left.move_j_calls[substeps - 1 :: substeps], trajectory[1:]
        )
        with self.assertRaises(ValueError):
            self.rpc.execute_joint_trajectory(
                "left", trajectory, sequence=9, timeout_s=2.0
            )

    def test_joint_trajectory_rejects_stale_start_without_motion(self) -> None:
        trajectory = [[0.2] * 7, [0.21] * 7]

        result = self.rpc.execute_joint_trajectory(
            "left", trajectory, sequence=9, timeout_s=2.0
        )

        self.assertFalse(result["success"])
        self.assertEqual(result["reason"], "trajectory_start_mismatch")
        self.assertEqual(self.left.move_j_calls, [])

    def test_joint_trajectory_rejects_large_step_and_joint_limit(self) -> None:
        with self.assertRaisesRegex(ValueError, "0.10 rad"):
            self.rpc.execute_joint_trajectory(
                "left", [[0.0] * 7, [0.11] * 7], sequence=9, timeout_s=2.0
            )
        invalid = [[0.0] * 7, [0.01] * 7]
        invalid[1][5] = 1.0
        with self.assertRaisesRegex(ValueError, "joint limits"):
            self.rpc.execute_joint_trajectory(
                "left", invalid, sequence=10, timeout_s=2.0
            )

    def test_batch_plan_reports_unwrapped_bounded_joint_motion(self) -> None:
        self.left.joints = [1.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.left_ik.joint_target = [-1.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        result = self.rpc.plan_tcp_pose(
            "left", [0.3, 0.1, 0.4, 0.0, 0.2, 0.0]
        )

        self.assertAlmostEqual(result["joint_delta_rad"][0], -3.8)
        self.assertAlmostEqual(result["joint_travel_l2_rad"], 3.8)
        self.assertAlmostEqual(result["joint_travel_max_rad"], 3.8)

    def test_nominal_official_gripper_tcp_uses_grasp_volume_center(self) -> None:
        self.assertAlmostEqual(NERO_OFFICIAL_GRIPPER_TCP_OFFSET[0], 0.142)

    def test_shared_tcp_x_parameter_and_per_arm_override(self) -> None:
        args = SimpleNamespace(
            official_gripper_tcp_x_m=0.165,
            left_tcp_offset=None,
            right_tcp_offset=[0.16, 0.001, -0.02, -1.5, 0.0, -1.5],
        )

        offsets = _resolve_tcp_offsets(args)

        self.assertAlmostEqual(offsets["left"][0], 0.165)
        self.assertEqual(offsets["left"][1:], NERO_OFFICIAL_GRIPPER_TCP_OFFSET[1:])
        self.assertEqual(offsets["right"], args.right_tcp_offset)

    def test_single_ik_plan_reports_best_effort_without_motion(self) -> None:
        self.left_ik.converged = False
        self.left_ik.task_error = np.asarray([0.002, 0.0, 0.0, 0.0, 0.03, 0.0])

        result = self.rpc.plan_tcp_pose(
            "left", [0.3, 0.1, 0.4, 0.0, 0.2, 0.0]
        )

        self.assertTrue(result["success"])
        self.assertFalse(result["ik_converged"])
        self.assertAlmostEqual(result["ik_position_error_m"], 0.002)
        self.assertAlmostEqual(result["ik_rotation_error_rad"], 0.03)
        self.assertEqual(self.left.move_j_calls, [])

    def test_cartesian_joint_timeout_latches_estop(self) -> None:
        self.left.joint_move_sets_motion_pending = True
        original_move_j = self.left.move_j
        self.left.move_j = lambda joints: self.left.move_j_calls.append(list(joints))
        target = [0.3, 0.1, 0.4, 0.0, 0.2, 0.0]

        with self.assertRaisesRegex(TimeoutError, "joint motion timeout"):
            self.rpc.move_tcp_pose("left", target, 11, 0.01)

        self.left.move_j = original_move_j
        self.assertTrue(self.left.estopped)
        self.assertTrue(self.right.estopped)

    def test_rotation_error_handles_equivalent_gimbal_lock_rpy(self) -> None:
        error = _rotation_error_rad(
            [0.5, math.pi / 2, 1.0],
            [0.0, math.pi / 2, 0.5],
        )

        self.assertAlmostEqual(error, 0.0, places=7)

    def test_status_reports_configured_motion_deadline(self) -> None:
        self.assertEqual(self.rpc.get_status()["max_motion_s"], 60.0)

    def test_move_tcp_pose_ignores_retained_firmware_no_solution(self) -> None:
        self.left.arm_status = 2
        self.left.motion_status = 0
        target = [0.3, 0.1, 0.4, 0.0, 0.2, 0.0]

        result = self.rpc.move_tcp_pose("left", target, 11, 1.0)

        self.assertTrue(result["success"])
        self.assertTrue(result["stale_no_solution_ignored"])

    def test_unavailable_mink_is_nonfatal_and_does_not_move(self) -> None:
        rpc = NeroArmPairRPC(
            self.left,
            self.right,
            self.left_gripper,
            self.right_gripper,
        )
        target = [0.3, 0.1, 0.4, 0.0, 0.2, 0.0]

        result = rpc.move_tcp_pose("left", target, 12, 2.0)

        self.assertFalse(result["success"])
        self.assertEqual(result["reason"], "mink_ik_unavailable")
        self.assertEqual(self.left.move_j_calls, [])
        self.assertFalse(result["estop_latched"])
        self.assertFalse(self.left.estopped)
        self.assertFalse(self.right.estopped)

    def test_mink_exception_is_nonfatal_and_does_not_move(self) -> None:
        self.left_ik.error = RuntimeError("solver failed")
        target = [0.3, 0.1, 0.4, 0.0, 0.2, 0.0]

        result = self.rpc.move_tcp_pose("left", target, 13, 2.0)

        self.assertFalse(result["success"])
        self.assertEqual(result["reason"], "mink_ik_error")
        self.assertEqual(self.left.move_j_calls, [])
        self.assertFalse(self.left.estopped)
        self.assertFalse(self.right.estopped)

    def test_nonconverged_mink_best_solution_is_still_commanded(self) -> None:
        self.left_ik.converged = False
        self.left_ik.task_error = np.asarray([0.02, 0.0, 0.0, 0.1, 0.0, 0.0])
        target = [0.3, 0.1, 0.4, 0.0, 0.2, 0.0]

        result = self.rpc.move_tcp_pose("left", target, 14, 2.0)

        self.assertTrue(result["success"])
        self.assertFalse(result["ik_converged"])
        self.assertEqual(self.left.move_j_calls[-1], self.left_ik.joint_target)
        self.assertFalse(self.left.estopped)
        self.assertFalse(self.right.estopped)

    def test_cartesian_nonplanning_fault_still_latches_both_arms(self) -> None:
        self.left.arm_status = 3
        target = [0.3, 0.1, 0.4, 0.0, 0.2, 0.0]

        with self.assertRaisesRegex(RuntimeError, "arm_status_3"):
            self.rpc.move_tcp_pose("left", target, 16, 2.0)

        self.assertTrue(self.left.estopped)
        self.assertTrue(self.right.estopped)

    def test_status_converts_sdk_integer_subclasses_to_builtin_ints(self) -> None:
        self.left.ctrl_mode = SDKIntEnumLike(1)
        self.left.arm_status = SDKIntEnumLike(0)

        status = self.rpc.get_status()["left"]["arm_status"]

        self.assertIs(type(status["ctrl_mode"]), int)
        self.assertIs(type(status["arm_status"]), int)

    def test_ik_diagnostic_labels_retained_feedback_as_stale(self) -> None:
        self.left.ik_message = Message([0.1] * 7, timestamp=5.0)

        result = self.rpc._ik_joint_diagnostic(
            self.left, newer_than_timestamp=5.0
        )

        self.assertTrue(str(result).startswith("stale_ik_feedback:"))

    def test_gripper_uses_native_width_feedback(self) -> None:
        result = self.rpc.set_gripper("right", True, 20, 1.0, 1.2)

        self.assertTrue(result["success"])
        self.assertAlmostEqual(result["width_m"], 0.10)
        self.assertAlmostEqual(result["target_width_m"], 0.10)
        self.assertAlmostEqual(result["force_n"], 1.2)

        result = self.rpc.set_gripper("right", False, 21, 1.0, 1.2)
        self.assertTrue(result["success"])
        self.assertAlmostEqual(result["width_m"], 0.0)

    def test_gripper_uses_configured_91_mm_opening_on_both_arms(self) -> None:
        rpc = NeroArmPairRPC(
            self.left, self.right, self.left_gripper, self.right_gripper,
            open_width_m=0.091,
        )
        clock = GripperTestClock()
        with mock.patch("nero_arm.service.time.monotonic", clock.monotonic), mock.patch(
            "nero_arm.service.time.sleep", clock.sleep
        ):
            for index, arm in enumerate(("left", "right")):
                with self.subTest(arm=arm):
                    opened = rpc.set_gripper(arm, True, 50 + 2 * index, 3.0)
                    closed = rpc.set_gripper(arm, False, 51 + 2 * index, 3.0)
                    self.assertTrue(opened["success"])
                    self.assertAlmostEqual(opened["target_width_m"], 0.091)
                    self.assertTrue(closed["success"])
                    self.assertAlmostEqual(closed["target_width_m"], 0.0)
                    self.assertEqual(
                        rpc._grippers[arm].commands, [(0.091, 1.0), (0.0, 1.0)]
                    )

    def test_gripper_commissioning_moves_to_exact_bounded_width(self) -> None:
        result = self.rpc.commission_gripper_width(
            "left",
            0.02,
            30,
            GRIPPER_COMMISSIONING_CONFIRMATION,
            timeout_s=1.0,
            force_n=1.0,
        )

        self.assertTrue(result["success"])
        self.assertEqual(result["reason"], "commissioning_width_confirmed")
        self.assertAlmostEqual(result["target_width_m"], 0.02)
        self.assertEqual(self.left_gripper.commands, [(0.02, 1.0)])

    def test_gripper_commissioning_requires_confirmation_before_motion(self) -> None:
        with self.assertRaisesRegex(PermissionError, "exact operator confirmation"):
            self.rpc.commission_gripper_width(
                "left", 0.02, 30, "wrong confirmation"
            )

        self.assertEqual(self.left_gripper.commands, [])

    def test_gripper_commissioning_rejects_width_above_configured_opening(self) -> None:
        with self.assertRaisesRegex(ValueError, r"\[0, 0\.100\]"):
            self.rpc.commission_gripper_width(
                "left",
                0.101,
                30,
                GRIPPER_COMMISSIONING_CONFIRMATION,
            )

        self.assertEqual(self.left_gripper.commands, [])

    def test_gripper_commissioning_rejects_disabled_feedback(self) -> None:
        self.left_gripper.foc_status.driver_enable_status = False
        clock = GripperTestClock()

        with mock.patch("nero_arm.service.time", clock), self.assertRaisesRegex(
            RuntimeError, "gripper driver is disabled"
        ):
            self.rpc.commission_gripper_width(
                "left",
                0.02,
                30,
                GRIPPER_COMMISSIONING_CONFIRMATION,
                timeout_s=3.0,
            )
        self.assertGreaterEqual(clock.now, 1.0)
        self.assertLess(clock.now, 1.1)

    def test_gripper_waits_for_initial_enable_on_both_rpc_paths(self) -> None:
        for commissioning in (False, True):
            with self.subTest(commissioning=commissioning):
                disabled = Message(
                    GripperStatus(0.0, foc_status=FocStatus(driver_enable_status=False)),
                    timestamp=1.0,
                )
                gripper = SequencedGripper(
                    disabled,
                    [disabled] * 4
                    + [Message(GripperStatus(0.10), timestamp=float(i))
                       for i in range(2, 6)],
                )
                self.rpc._grippers["left"] = gripper
                clock = GripperTestClock()
                with mock.patch("nero_arm.service.time", clock):
                    if commissioning:
                        result = self.rpc.commission_gripper_width(
                            "left", 0.10, 41, GRIPPER_COMMISSIONING_CONFIRMATION,
                            timeout_s=3.0, force_n=1.0,
                        )
                    else:
                        result = self.rpc.set_gripper("left", True, 40, 3.0, 1.0)
                self.assertTrue(result["success"])
                self.assertEqual(gripper.commands, [(0.10, 1.0)])
                self.assertGreaterEqual(clock.now, 0.35)

    def test_gripper_initial_enable_wait_respects_short_call_timeout(self) -> None:
        self.left_gripper.foc_status.driver_enable_status = False
        clock = GripperTestClock()
        with mock.patch("nero_arm.service.time", clock), self.assertRaisesRegex(
            TimeoutError, "target_width_m=0.100"
        ):
            self.rpc.set_gripper("left", True, 40, 0.12, 1.0)
        self.assertLessEqual(clock.now, 0.17)

    def test_gripper_fault_during_initial_enable_is_not_hidden(self) -> None:
        self.left_gripper.foc_status.driver_enable_status = False
        self.left_gripper.foc_status.driver_overcurrent = True
        clock = GripperTestClock()
        with mock.patch("nero_arm.service.time", clock), self.assertRaisesRegex(
            RuntimeError, "driver_overcurrent"
        ):
            self.rpc.set_gripper("left", True, 40, 3.0, 1.0)
        self.assertEqual(clock.now, 0.0)

    def test_gripper_losing_enable_after_command_is_still_an_error(self) -> None:
        for initially_enabled in (False, True):
            with self.subTest(initially_enabled=initially_enabled):
                initial = Message(
                    GripperStatus(0.0, foc_status=FocStatus(
                        driver_enable_status=initially_enabled,
                    )), timestamp=1.0,
                )
                samples = [Message(GripperStatus(0.02), timestamp=2.0)]
                samples.append(Message(
                    GripperStatus(0.02, foc_status=FocStatus(driver_enable_status=False)),
                    timestamp=3.0,
                ))
                self.rpc._grippers["left"] = SequencedGripper(initial, samples)
                clock = GripperTestClock()
                with mock.patch("nero_arm.service.time", clock), self.assertRaisesRegex(
                    RuntimeError, "gripper driver is disabled"
                ):
                    self.rpc.set_gripper("left", True, 40 + int(initially_enabled), 3.0, 1.0)
                self.assertLess(clock.now, 0.2)

    def test_gripper_cached_target_feedback_cannot_confirm_motion(self) -> None:
        for stamp in (1.0, 2.0):
            with self.subTest(stamp=stamp):
                initial = Message(GripperStatus(0.10), timestamp=1.0)
                self.rpc._grippers["left"] = SequencedGripper(
                    initial, [Message(GripperStatus(0.10), timestamp=stamp)],
                )
                clock = GripperTestClock()
                with mock.patch("nero_arm.service.time", clock), self.assertRaisesRegex(
                    TimeoutError, "target_width_m=0.100"
                ):
                    self.rpc.set_gripper("left", True, 40 + int(stamp), 0.3, 1.0)

    def test_gripper_waits_through_missing_initial_feedback(self) -> None:
        initial = Message(GripperStatus(0.0), timestamp=1.0)
        self.rpc._grippers["left"] = SequencedGripper(
            initial, [None, None]
            + [Message(GripperStatus(0.10), timestamp=float(i)) for i in range(2, 6)],
        )
        clock = GripperTestClock()
        with mock.patch("nero_arm.service.time", clock):
            result = self.rpc.set_gripper("left", True, 40, 3.0, 1.0)
        self.assertTrue(result["success"])

    def test_gripper_commissioning_rejects_fault_feedback(self) -> None:
        self.left_gripper.foc_status.driver_overcurrent = True

        with self.assertRaisesRegex(RuntimeError, "driver_overcurrent"):
            self.rpc.commission_gripper_width(
                "left",
                0.02,
                30,
                GRIPPER_COMMISSIONING_CONFIRMATION,
                timeout_s=1.0,
            )

    def test_gripper_commissioning_times_out_without_width_feedback(self) -> None:
        self.left_gripper.follow_commands = False

        with self.assertRaisesRegex(
            TimeoutError, "target_width_m=0.020"
        ):
            self.rpc.commission_gripper_width(
                "left",
                0.02,
                30,
                GRIPPER_COMMISSIONING_CONFIRMATION,
                timeout_s=0.01,
            )

    def test_emergency_stop_latches_both_arms(self) -> None:
        result = self.rpc.emergency_stop()

        self.assertTrue(result["estop_latched"])
        self.assertTrue(self.left.estopped)
        self.assertTrue(self.right.estopped)

    def test_motion_fault_latches_both_arms(self) -> None:
        self.left.arm_status = 3

        with self.assertRaises(RuntimeError):
            self.rpc.home_arm("left", 30, 1.0)

        self.assertTrue(self.left.estopped)
        self.assertTrue(self.right.estopped)

    def test_one_estop_failure_does_not_skip_other_arm(self) -> None:
        self.left.estop_raises = True

        result = self.rpc.emergency_stop()

        self.assertFalse(result["stopped"])
        self.assertIn("left", result["errors"])
        self.assertTrue(self.right.estopped)

    def test_initialize_refuses_implicit_emergency_stop_reset(self) -> None:
        self.left.arm_status = 1

        with self.assertRaisesRegex(RuntimeError, "single-arm operator recovery"):
            self.rpc.initialize(home=False)

        self.assertEqual(self.left.reset_calls, 0)
        self.assertEqual(self.left.disable_calls, 0)
        self.assertEqual(self.left.arm_status, 1)
        self.assertEqual(self.left.estop_calls, 0)

    def test_close_preserves_damped_estop_without_disabling_arms(self) -> None:
        self.rpc.close()

        self.assertTrue(self.left.estopped)
        self.assertTrue(self.right.estopped)
        self.assertEqual(self.left.disable_calls, 0)
        self.assertEqual(self.right.disable_calls, 0)

    def test_operator_recovery_resets_then_leaves_arm_disabled(self) -> None:
        self.left.arm_status = 1
        self.left.enabled = False
        self.left.status_none_reads_after_connect = 4
        self.left.status_none_reads_after_reset = 2

        result = self.rpc._recover_estop_for_operator("left", settle_s=0.0)

        self.assertTrue(result["success"])
        self.assertEqual(self.left.reset_calls, 1)
        self.assertEqual(self.left.estop_calls, 1)
        self.assertFalse(self.left.enabled)
        self.assertFalse(self.left.connected)
        self.assertEqual(self.left.arm_status, 6)
        self.assertEqual(
            result["reset_acknowledgement"], "all_joints_powered_off"
        )

    def test_operator_recovery_accepts_power_off_with_stale_estop_status(
        self,
    ) -> None:
        self.left.arm_status = 1
        self.left.enabled = True
        self.left.reset_keeps_status = True

        result = self.rpc._recover_estop_for_operator("left", settle_s=0.0)

        self.assertTrue(result["success"])
        self.assertEqual(self.left.reset_calls, 1)
        self.assertEqual(self.left.estop_calls, 1)
        self.assertFalse(self.left.enabled)
        self.assertEqual(result["status_after"], 1)

    def test_operator_recovery_limits_unacknowledged_reset_retries(self) -> None:
        self.left.arm_status = 1
        self.left.enabled = True
        self.left.ignored_reset_calls_remaining = 99

        with self.assertRaisesRegex(TimeoutError, "3 low-rate attempts"):
            self.rpc._recover_estop_for_operator("left", settle_s=0.0)

        self.assertEqual(self.left.reset_calls, 3)
        # One documented re-trigger before reset; exception cleanup must not
        # send another stop and start the next recovery in the same loop.
        self.assertEqual(self.left.estop_calls, 1)
        self.assertFalse(self.left.connected)

    def test_initialize_waits_for_stable_motion_ready_feedback(self) -> None:
        self.left.enabled = False
        self.left.arm_status = 6
        self.left.enable_calls = 0
        self.left.ready_after_enable_calls = 2

        self.rpc.initialize(home=False)

        self.assertEqual(self.left.enable_calls, 2)
        self.assertTrue(self.left.enabled)
        self.assertEqual(self.left.arm_status, 0)

    def test_operator_recovery_does_nothing_when_estop_is_already_clear(self) -> None:
        self.left.arm_status = 2
        self.left.enabled = False
        self.left.preserve_no_solution_until_joint_move = True
        self.left.joints = [0.1, 0.2, -0.3, 0.4, -0.2, 0.1, 0.0]

        result = self.rpc._recover_estop_for_operator("left", settle_s=0.0)

        self.assertTrue(result["success"])
        self.assertEqual(self.left.move_j_calls, [])
        self.assertEqual(
            result["reason"], "emergency_stop_already_clear_no_action"
        )
        self.assertEqual(self.left.arm_status, 2)
        self.assertFalse(self.left.enabled)

    def test_confirmed_home_recovers_disabled_no_solution_state(self) -> None:
        self.left.arm_status = 2
        self.left.motion_status = 1
        self.left.enabled = False
        self.left.enable_calls = 0
        self.left.preserve_no_solution_until_joint_move = True
        self.left.retain_no_solution_after_successful_joint_move = True
        self.left.joints = self.rpc._homes["left"].copy()

        self.rpc.initialize(home=True, timeout_s=1.0)

        self.assertTrue(self.left.enabled)
        self.assertEqual(self.left.enable_calls, 1)
        self.assertEqual(self.left.motion_mode_calls, ["j"])
        self.assertEqual(len(self.left.move_j_calls), 2)
        self.assertEqual(self.left.joints, self.rpc._homes["left"])

    def test_confirmed_home_leaves_both_grippers_open(self) -> None:
        self.left.joints = self.rpc._homes["left"].copy()
        self.right.joints = self.rpc._homes["right"].copy()

        self.rpc.initialize(home=True, timeout_s=1.0)

        for gripper in (self.left_gripper, self.right_gripper):
            self.assertTrue(gripper.commands)
            self.assertAlmostEqual(gripper.commands[-1][0], self.rpc._open_width_m)
            self.assertAlmostEqual(gripper.width, self.rpc._open_width_m)

    def test_home_disabled_leaves_the_grippers_untouched(self) -> None:
        self.rpc.initialize(home=False)

        self.assertEqual(self.left_gripper.commands, [])
        self.assertEqual(self.right_gripper.commands, [])

    def test_confirmed_home_replaces_sticky_no_solution_target(self) -> None:
        self.left.arm_status = 2
        self.left.enabled = True
        self.left.preserve_no_solution_until_joint_move = True
        self.left.retain_no_solution_after_successful_joint_move = True
        self.left.joints = self.rpc._homes["left"].copy()

        self.rpc.initialize(home=True, timeout_s=1.0)

        self.assertEqual(self.left.motion_mode_calls, ["j"])
        self.assertAlmostEqual(
            self.left.move_j_calls[0][6],
            self.rpc._homes["left"][6] + 0.02,
        )
        self.assertEqual(self.left.joints, self.rpc._homes["left"])
        self.assertEqual(self.left.arm_status, 2)

    def test_home_accepts_completed_retained_no_solution(self) -> None:
        self.left.arm_status = 2
        self.left.motion_status = 0
        self.left.joints = self.rpc._homes["left"].copy()
        self.left.retain_no_solution_after_successful_joint_move = True
        self.left.joint_move_sets_motion_pending = True

        result = self.rpc.home_arm("left", 200, 1.0)

        self.assertTrue(result["success"])
        self.assertEqual(len(self.left.move_j_calls), 2)
        self.assertFalse(self.left.estopped)

    def test_startup_swings_the_arm_forward_on_joint_1_before_homing(self) -> None:
        # Torque-off leaves the arm hanging against the base frame, so joint 1
        # swings it forward first with the rest off; homing the whole arm from
        # the hanging pose is what hits the frame and trips the stop.
        self.left.joints = list(LEFT_HANGING)
        self.right.joints = list(RIGHT_HANGING)
        for driver in (self.left, self.right):
            driver.enabled = False
            driver.ready_after_enable_calls = 1

        self.rpc.initialize(home=True, timeout_s=1.0)

        joint_1_only = [joint == 1 for joint in range(1, 8)]
        for name, driver, hanging in (
            ("left", self.left, LEFT_HANGING),
            ("right", self.right, RIGHT_HANGING),
        ):
            # The swing is the first motion, made with only joint 1 powered;
            # the rest of the arm is powered again before homing.
            self.assertEqual(SHOULDER_CLEARANCE_JOINT, 1)
            self.assertEqual(driver.enable_status_at_move_j[0], joint_1_only)
            self.assertEqual(driver.enable_status_at_move_j[-1], [True] * 7)
            swing = driver.move_j_calls[0]
            # FK moves the hand forward when joint 1 decreases on the left
            # arm and increases on the right, which mirror each other.
            forward = {"left": -1.0, "right": 1.0}[name]
            self.assertEqual(SHOULDER_CLEARANCE_DIRECTION[name], forward)
            self.assertAlmostEqual(
                swing[0], hanging[0] + forward * SHOULDER_CLEARANCE_RAD, places=6
            )
            for index, value in enumerate(swing[1:], start=1):
                self.assertAlmostEqual(value, hanging[index], places=6)
            # Only then is the whole arm powered and homed.
            self.assertEqual(driver.move_j_calls[-1], list(self.rpc._homes[name]))
            self.assertTrue(driver.enabled)
            self.assertFalse(driver.estopped)

    def test_startup_skips_the_shoulder_swing_when_already_home(self) -> None:
        for name, driver in (("left", self.left), ("right", self.right)):
            driver.joints = list(self.rpc._homes[name])

        self.rpc.initialize(home=True, timeout_s=1.0)

        for name, driver in (("left", self.left), ("right", self.right)):
            self.assertEqual(driver.move_j_calls, [list(self.rpc._homes[name])])

    def test_the_shoulder_swing_leaves_the_arm_powered_where_it_is(self) -> None:
        self.left.joints = list(LEFT_HANGING)

        lift = self.rpc._clear_base_frame_with_shoulder(self.left, "left", timeout_s=1.0)

        self.assertEqual(lift["others"], "off")
        self.assertNotIn("fallback_reason", lift)
        # Before the powered-off joints are powered, their target is moved to
        # where they are now, so powering them does not pull them back to the
        # pose they had against the frame.
        self.assertEqual(self.left.move_j_calls[-1], self.left.joints)
        self.assertEqual(self.left.get_joints_enable_status_list(), [True] * 7)
        self.assertEqual(self.left.arm_status, 0)

    def test_the_shoulder_swing_is_redone_held_when_the_firmware_ignores_it(
        self,
    ) -> None:
        self.left.joints = list(LEFT_HANGING)
        self.left.ignores_motion_while_partly_off = True

        lift = self.rpc._clear_base_frame_with_shoulder(
            self.left, "left", timeout_s=1.0, response_s=0.2
        )

        self.assertEqual(lift["others"], "held")
        self.assertIn("did not move", lift["fallback_reason"])
        self.assertIn("arm_status 6", lift["fallback_reason"])
        # The held swing goes to the same target, not a second 0.52 rad
        # further on, and only once the whole arm is powered again.
        self.assertEqual(self.left.enable_status_at_move_j[-1], [True] * 7)
        self.assertAlmostEqual(
            self.left.joints[0], LEFT_HANGING[0] - SHOULDER_CLEARANCE_RAD, places=6
        )
        for index, value in enumerate(self.left.joints[1:], start=1):
            self.assertAlmostEqual(value, LEFT_HANGING[index], places=6)
        self.assertFalse(self.left.estopped)

    def test_the_shoulder_swing_is_held_when_the_other_joints_stay_powered(
        self,
    ) -> None:
        self.right.joints = list(RIGHT_HANGING)
        self.right.single_joint_disable_takes = False

        lift = self.rpc._clear_base_frame_with_shoulder(self.right, "right", timeout_s=1.0)

        self.assertEqual(lift["others"], "held")
        self.assertIn("did not report powered off", lift["fallback_reason"])
        self.assertEqual(self.right.enable_status_at_move_j, [[True] * 7])
        self.assertAlmostEqual(
            self.right.joints[0], RIGHT_HANGING[0] + SHOULDER_CLEARANCE_RAD, places=6
        )

    def test_the_shoulder_swing_stops_at_the_official_joint_limit(self) -> None:
        lower, upper = (float(v) for v in NERO_JOINT_POSITION_LIMITS_RAD[0])
        # Each arm sits at the end of the range it would swing towards.
        self.left.joints = [lower, 1.5, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.right.joints = [upper, 1.5, 0.0, 0.0, 0.0, 0.0, 0.0]

        self.rpc.initialize(home=True, timeout_s=1.0)

        for driver in (self.left, self.right):
            # Nothing to give at the end of the range: skip, do not command a
            # target outside the limit.
            for call in driver.move_j_calls:
                self.assertGreaterEqual(call[0], lower - 1e-9)
                self.assertLessEqual(call[0], upper + 1e-9)
            self.assertEqual(driver.enable_status_at_move_j[0], [True] * 7)

    def test_a_stop_during_the_shoulder_swing_is_reported_not_latched(self) -> None:
        self.left.joints = list(LEFT_HANGING)
        self.left.arm_status = 1  # emergency stop, as a frame collision leaves it

        with self.assertRaisesRegex(RuntimeError, "shoulder swing"):
            self.rpc._clear_base_frame_with_shoulder(self.left, "left", timeout_s=0.3)

        # An electronic stop here would cut torque and drop the arm onto the
        # frame it is clearing, so the failure must not latch one.
        self.assertFalse(self.left.estopped)
        self.assertEqual(self.left.estop_calls, 0)

    def test_calibration_drag_is_disabled_by_default(self) -> None:
        with self.assertRaisesRegex(PermissionError, "--allow-calibration-drag"):
            self.rpc.enter_calibration_drag(
                "right", "I_AM_PHYSICALLY_SUPPORTING_RIGHT"
            )

    def test_calibration_drag_requires_exact_confirmation(self) -> None:
        rpc = NeroArmPairRPC(
            self.left,
            self.right,
            self.left_gripper,
            self.right_gripper,
            allow_calibration_drag=True,
        )

        with self.assertRaisesRegex(PermissionError, "confirmation"):
            rpc.enter_calibration_drag("right", "yes")

        self.assertEqual(self.right.leader_mode_calls, 0)

    def test_calibration_drag_enters_and_restores_current_joint_hold(self) -> None:
        rpc = NeroArmPairRPC(
            self.left,
            self.right,
            self.left_gripper,
            self.right_gripper,
            allow_calibration_drag=True,
        )
        self.right.joints = [0.1, 0.2, -0.3, 0.4, 0.5, -0.2, 0.1]

        entered = rpc.enter_calibration_drag(
            "right", "I_AM_PHYSICALLY_SUPPORTING_RIGHT"
        )

        self.assertTrue(entered["success"])
        # Nero disables the ordinary CAN status push before enabling linkage
        # teaching, so ctrl_mode may remain cached at CAN_CTRL.  Fresh leader
        # joint frames are the authoritative transition signal.
        self.assertEqual(self.right.ctrl_mode, 1)
        self.assertTrue(self.right.leader_active)
        self.assertEqual(
            entered["mode_verification"], "fresh_leader_joint_feedback"
        )
        self.assertEqual(rpc.get_status()["calibration_drag"]["active_arm"], "right")
        with self.assertRaisesRegex(RuntimeError, "calibration drag mode"):
            rpc.home_arm("right", 100, 1.0)

        dragged_joints = [0.2, 0.3, -0.4, 0.2, 0.3, -0.1, -0.2]
        dragged_flange = [0.41, -0.12, 0.58, 0.1, -0.2, 0.3]
        self.right.leader_joints = dragged_joints.copy()
        self.right.tcp = dragged_flange.copy()
        dragged_status = rpc.get_status()["right"]
        self.assertEqual(dragged_status["joint_pos"], dragged_joints)
        self.assertEqual(dragged_status["flange_pose_xyz_rpy"], dragged_flange)
        # A teaching-mode status frame can remain cached after ordinary joint
        # feedback resumes.  It must not turn a successful exit into an estop.
        self.right.ctrl_mode = 6
        exited = rpc.exit_calibration_drag("right")

        self.assertTrue(exited["success"])
        self.assertEqual(exited["reason"], "controlled_hold_restored")
        self.assertEqual(
            exited["mode_verification"], "fresh_standard_joint_feedback"
        )
        self.assertEqual(self.right.ctrl_mode, 6)
        self.assertFalse(self.right.leader_active)
        self.assertEqual(self.right.follower_mode_calls, 1)
        self.assertEqual(self.right.move_j_calls[-1], dragged_joints)
        self.assertIsNone(rpc.get_status()["calibration_drag"]["active_arm"])

    def test_calibration_drag_rejects_second_arm(self) -> None:
        rpc = NeroArmPairRPC(
            self.left,
            self.right,
            self.left_gripper,
            self.right_gripper,
            allow_calibration_drag=True,
        )
        rpc.enter_calibration_drag(
            "right", "I_AM_PHYSICALLY_SUPPORTING_RIGHT"
        )

        with self.assertRaisesRegex(RuntimeError, "already active for right"):
            rpc.enter_calibration_drag(
                "left", "I_AM_PHYSICALLY_SUPPORTING_LEFT"
            )


if __name__ == "__main__":
    unittest.main()
