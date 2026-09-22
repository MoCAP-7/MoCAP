"""Streaming execute_joint_trajectory on the Pi Nero arm service.

The service is exercised against a fake pyAgxArm driver whose joints move
toward the last ``move_j`` target at a bounded speed, and against a fake clock,
so the tests are deterministic and need no hardware.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types
import unittest

import numpy as np

SERVICE_PATH = (
    Path(__file__).resolve().parents[1] / "services" / "nero_arm" / "service.py"
)


def _load_service_module() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location("nero_arm_service", SERVICE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


service = _load_service_module()


class FakeClock:
    """Deterministic replacement for the ``time`` module inside the service."""

    def __init__(self) -> None:
        self.now = 1000.0

    def monotonic(self) -> float:
        return self.now

    def time(self) -> float:
        return self.now

    def time_ns(self) -> int:
        return int(self.now * 1e9)

    def sleep(self, seconds: float) -> None:
        self.now += max(0.0, float(seconds))


class _Msg:
    def __init__(self, payload) -> None:
        self.msg = payload


class _Status:
    def __init__(self) -> None:
        self.ctrl_mode = 1
        self.arm_status = 0
        self.mode_feedback = 1
        self.teach_status = 0
        self.motion_status = 0
        self.trajectory_num = 0
        self.err_status = None


class FakeDriver:
    """Joints move toward the latest ``move_j`` target at ``max_speed_rad_s``."""

    def __init__(self, clock: FakeClock, *, max_speed_rad_s: float) -> None:
        self._clock = clock
        self._speed = float(max_speed_rad_s)
        self.position = np.zeros(7)
        self.target = np.zeros(7)
        self._last_update = clock.now
        self.move_j_calls: list[tuple[float, list[float]]] = []
        self.estop_calls = 0
        self.comm_error = False
        self.status = _Status()

    def _advance(self) -> None:
        dt = self._clock.now - self._last_update
        self._last_update = self._clock.now
        if dt <= 0.0:
            return
        step = np.clip(self.target - self.position, -self._speed * dt, self._speed * dt)
        self.position = self.position + step

    # --- pyAgxArm surface used by the service -------------------------------
    def move_j(self, joints) -> None:
        self._advance()
        self.target = np.asarray(joints, dtype=float)
        self.move_j_calls.append((self._clock.now, list(map(float, joints))))

    def set_motion_mode(self, mode) -> None:  # noqa: ARG002
        return None

    def set_speed_percent(self, percent) -> None:  # noqa: ARG002
        return None

    def get_joint_angles(self):
        self._advance()
        return _Msg(self.position.tolist())

    def get_flange_pose(self):
        return _Msg([0.0] * 6)

    def get_tcp_pose(self):
        return _Msg([0.0] * 6)

    def get_arm_status(self):
        return _Msg(self.status)

    def get_joints_enable_status_list(self):
        return [True] * 7

    def is_connected(self) -> bool:
        return True

    def is_ok(self) -> bool:
        return True

    def has_comm_error(self) -> bool:
        return self.comm_error

    def electronic_emergency_stop(self) -> None:
        self.estop_calls += 1


class FakeGripper:
    def get_gripper_status(self):
        return None

    def disable_gripper(self) -> None:
        return None


def _ramp(count: int, step: float, joint: int = 0) -> np.ndarray:
    trajectory = np.zeros((count, 7))
    trajectory[:, joint] = np.arange(count) * step
    return trajectory


class StreamingTrajectoryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.clock = FakeClock()
        self._real_time = service.time
        service.time = self.clock

    def tearDown(self) -> None:
        service.time = self._real_time

    def _api(self, *, max_speed_rad_s: float, **kwargs):
        self.left = FakeDriver(self.clock, max_speed_rad_s=max_speed_rad_s)
        self.right = FakeDriver(self.clock, max_speed_rad_s=max_speed_rad_s)
        return service.NeroArmPairRPC(
            self.left,
            self.right,
            FakeGripper(),
            FakeGripper(),
            speed_percent=10,
            **kwargs,
        )

    def test_streams_waypoints_at_the_planned_cadence(self) -> None:
        api = self._api(max_speed_rad_s=5.0)
        trajectory = _ramp(6, 0.04)

        result = api.execute_joint_trajectory("left", trajectory.tolist(), 1, 30.0)

        self.assertTrue(result["success"], result)
        self.assertEqual(result["reason"], "joint_trajectory_completed")
        self.assertEqual(result["waypoint_count"], 6)
        self.assertEqual(len(result["executed"]), 5)
        substeps = round(0.10 * service.TRAJECTORY_DEFAULT_STREAM_HZ)
        calls = self.left.move_j_calls
        self.assertEqual(len(calls), 5 * substeps)
        self.assertEqual(result["frames_sent"], 5 * substeps)
        frame_dt = 0.10 / substeps
        start = calls[0][0] - frame_dt
        frame_times = [t - start for t, _ in calls]
        np.testing.assert_allclose(
            frame_times, [frame_dt * (i + 1) for i in range(len(calls))], atol=0.021
        )
        waypoint_times = [t - start for t, _ in calls[substeps - 1 :: substeps]]
        np.testing.assert_allclose(waypoint_times, [0.1, 0.2, 0.3, 0.4, 0.5], atol=0.021)
        for sent, expected in zip(calls[substeps - 1 :: substeps], trajectory[1:]):
            np.testing.assert_allclose(sent[1], expected)
        # Intermediate frames lie on the straight line between planner samples.
        np.testing.assert_allclose(calls[0][1][0], 0.04 / substeps, atol=1e-9)
        self.assertLessEqual(result["max_lag_rad"], service.TRAJECTORY_MAX_JOINT_STEP_RAD)
        self.assertLessEqual(
            result["final_joint_error_max_rad"], service.TRAJECTORY_SETTLE_TOLERANCE_RAD
        )
        self.assertEqual(self.left.estop_calls, 0)
        self.assertFalse(api._estop_latched)
        # The whole 6-sample path (0.5 s of plan) is not stretched to 0.15 s
        # per waypoint any more: the stream itself takes the planned 0.5 s.
        self.assertAlmostEqual(result["stream_s"], 0.5, delta=0.03)

    def test_caller_waypoint_dt_overrides_the_service_default(self) -> None:
        api = self._api(max_speed_rad_s=5.0)
        trajectory = _ramp(4, 0.04)

        result = api.execute_joint_trajectory(
            "left", trajectory.tolist(), 1, 30.0, waypoint_dt_s=0.25
        )

        self.assertTrue(result["success"], result)
        self.assertEqual(result["waypoint_dt_s"], 0.25)
        substeps = round(0.25 * service.TRAJECTORY_DEFAULT_STREAM_HZ)
        calls = self.left.move_j_calls
        self.assertEqual(len(calls), 3 * substeps)
        start = calls[0][0] - 0.25 / substeps
        waypoint_times = [t - start for t, _ in calls[substeps - 1 :: substeps]]
        np.testing.assert_allclose(waypoint_times, [0.25, 0.5, 0.75], atol=0.021)

    def test_lag_beyond_the_path_tolerance_holds_position_without_estop(self) -> None:
        # The arm can only do 0.05 rad/s while the plan asks for 0.5 rad/s, so
        # the lag behind the streamed target grows by ~0.045 rad per sample.
        api = self._api(max_speed_rad_s=0.05)
        trajectory = _ramp(30, 0.05)

        result = api.execute_joint_trajectory("left", trajectory.tolist(), 1, 30.0)

        self.assertFalse(result["success"])
        self.assertEqual(result["reason"], "joint_trajectory_lag_exceeded")
        self.assertGreater(result["lag_rad"], service.TRAJECTORY_PATH_TOLERANCE_RAD)
        self.assertLess(result["failed_waypoint_index"], 10)
        # A slow firmware is not a fault: no electronic stop on either arm.
        self.assertEqual(self.left.estop_calls, 0)
        self.assertEqual(self.right.estop_calls, 0)
        self.assertFalse(api._estop_latched)
        # The stream stopped early and the arm was left holding the last sent
        # target, which it reached before the call returned.
        substeps = round(0.10 * service.TRAJECTORY_DEFAULT_STREAM_HZ)
        self.assertLess(len(self.left.move_j_calls), 10 * substeps)
        self.assertEqual(result["held_target"], self.left.move_j_calls[-1][1])
        np.testing.assert_allclose(self.left.target, result["held_target"])
        self.assertLessEqual(
            result["held_joint_error_max_rad"], service.TRAJECTORY_SETTLE_TOLERANCE_RAD
        )
        self.assertEqual(len(result["executed"]), result["failed_waypoint_index"] - 1)

    def test_can_communication_error_still_emergency_stops(self) -> None:
        api = self._api(max_speed_rad_s=5.0)
        trajectory = _ramp(10, 0.04)
        original_move_j = self.left.move_j

        def flaky_move_j(joints) -> None:
            original_move_j(joints)
            if len(self.left.move_j_calls) >= 3:
                self.left.comm_error = True

        self.left.move_j = flaky_move_j

        with self.assertRaises(RuntimeError) as raised:
            api.execute_joint_trajectory("left", trajectory.tolist(), 1, 30.0)

        self.assertIn("CAN communication error", str(raised.exception))
        self.assertEqual(self.left.estop_calls, 1)
        self.assertEqual(self.right.estop_calls, 1)
        self.assertTrue(api._estop_latched)

    def test_total_timeout_returns_failure_without_emergency_stop(self) -> None:
        api = self._api(max_speed_rad_s=5.0)
        trajectory = _ramp(40, 0.04)  # 3.9 s of plan

        result = api.execute_joint_trajectory("left", trajectory.tolist(), 1, 1.0)

        self.assertFalse(result["success"])
        self.assertEqual(result["reason"], "joint_trajectory_total_timeout")
        self.assertLess(result["executed_waypoints"], 39)
        self.assertEqual(self.left.estop_calls, 0)
        self.assertFalse(api._estop_latched)

    def test_accepts_256_waypoints_and_rejects_257(self) -> None:
        api = self._api(max_speed_rad_s=5.0)
        accepted = _ramp(256, 0.004)
        result = api.execute_joint_trajectory("left", accepted.tolist(), 1, 60.0)
        self.assertTrue(result["success"], result)
        self.assertEqual(len(result["executed"]), 255)

        with self.assertRaises(ValueError):
            api.execute_joint_trajectory("left", _ramp(257, 0.004).tolist(), 2, 60.0)

    def test_pre_motion_checks_are_unchanged(self) -> None:
        api = self._api(max_speed_rad_s=5.0)

        mismatched = _ramp(4, 0.04)
        mismatched[:, 1] += 0.05  # the arm is at zero
        result = api.execute_joint_trajectory("left", mismatched.tolist(), 1, 10.0)
        self.assertFalse(result["success"])
        self.assertEqual(result["reason"], "trajectory_start_mismatch")
        self.assertEqual(self.left.move_j_calls, [])

        with self.assertRaises(ValueError):
            api.execute_joint_trajectory("left", _ramp(4, 0.11).tolist(), 2, 10.0)
        with self.assertRaises(ValueError):
            api.execute_joint_trajectory(
                "left", _ramp(4, 0.04).tolist(), 3, 10.0, waypoint_dt_s=0.01
            )
        self.assertEqual(self.left.move_j_calls, [])
        self.assertEqual(self.left.estop_calls, 0)

    def test_single_target_motion_still_waits_for_settled_arrival(self) -> None:
        api = self._api(max_speed_rad_s=1.0)
        target = [0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        result = api._move_joints_impl("left", target, 10.0)

        self.assertTrue(result["success"])
        self.assertEqual(result["reason"], "target_reached")
        self.assertLessEqual(
            result["joint_error_max_rad"], service.TRAJECTORY_SETTLE_TOLERANCE_RAD
        )
        # 0.3 rad at 1 rad/s plus three 50 ms confirmation reads.
        self.assertGreaterEqual(result["elapsed_s"], 0.3)
        self.assertEqual(len(self.left.move_j_calls), 1)

    def test_constructor_validates_streaming_settings(self) -> None:
        with self.assertRaises(ValueError):
            self._api(max_speed_rad_s=5.0, trajectory_dt_s=0.01)
        with self.assertRaises(ValueError):
            self._api(max_speed_rad_s=5.0, trajectory_path_tolerance_rad=0.05)
        api = self._api(
            max_speed_rad_s=5.0, trajectory_dt_s=0.2, trajectory_path_tolerance_rad=0.2
        )
        self.assertEqual(api._trajectory_dt_s, 0.2)
        self.assertEqual(api._trajectory_path_tolerance_rad, 0.2)


if __name__ == "__main__":
    unittest.main()
