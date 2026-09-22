"""Parity tests moved with the controller from ``YOR/Agent/tests/test_controller.py``."""

from __future__ import annotations

from dataclasses import dataclass
import math
import unittest

import numpy as np

from yor_agent.robot.navigation_controller import (
    NavigationConfig,
    NavigationController,
    wrap_angle,
)


@dataclass
class Pose:
    x_m: float
    y_m: float
    yaw_rad: float
    valid: bool = True


@dataclass
class Frame:
    planar_pose: Pose
    depth_m: np.ndarray


class ManualRobot:
    def __init__(self, *, clearance_m: float = 3.0) -> None:
        self.navigation_config = {
            "settle_cycles": 2,
            "stop_timeout_s": 0.4,
        }
        self.now = 0.0
        self.pose = Pose(0.0, 0.0, 0.0)
        self.velocity = np.zeros(3, dtype=float)
        self.clearance_m = clearance_m
        self.commands: list[list[float]] = []
        self.frame_error: Exception | None = None

    def clock(self) -> float:
        return self.now

    def sleep(self, duration: float) -> None:
        vx, vy, omega = self.velocity
        self.pose.yaw_rad = wrap_angle(self.pose.yaw_rad + omega * duration)
        cosine = math.cos(self.pose.yaw_rad)
        sine = math.sin(self.pose.yaw_rad)
        self.pose.x_m += (vx * cosine - vy * sine) * duration
        self.pose.y_m += (vx * sine + vy * cosine) * duration
        self.now += duration

    def navigation_frame(self, *, max_age_s: float):
        del max_age_s
        if self.frame_error is not None:
            raise self.frame_error
        return Frame(
            planar_pose=Pose(
                self.pose.x_m, self.pose.y_m, self.pose.yaw_rad, self.pose.valid
            ),
            depth_m=np.full((64, 96), self.clearance_m, dtype=np.float32),
        )

    def base_status(self):
        return {
            "lease_active": bool(np.any(self.velocity)),
            "lease_remaining_s": 0.25 if np.any(self.velocity) else 0.0,
            "last_velocity": self.velocity.tolist(),
            "estop_latched": False,
            "limits": {
                "lease_s": 0.25,
                "max_linear_mps": 0.12,
                "max_yaw_rad_s": 0.35,
            },
        }

    def submit_base_velocity(self, velocity):
        self.velocity = np.asarray(velocity, dtype=float)
        self.commands.append(self.velocity.tolist())
        return {"accepted": True, "velocity": self.velocity.tolist()}

    def get_observation(self):
        return {}


class StrictLimitRobot(ManualRobot):
    """Match the Pi RPC's strict comparison at its advertised limit."""

    def submit_base_velocity(self, velocity):
        linear = math.hypot(float(velocity[0]), float(velocity[1]))
        limit = float(self.base_status()["limits"]["max_linear_mps"])
        if linear > limit:
            raise ValueError(f"linear speed {linear!r} exceeds {limit!r}")
        return super().submit_base_velocity(velocity)


def make_controller(robot: ManualRobot) -> NavigationController:
    return NavigationController(
        robot,
        config=NavigationConfig.from_mapping(robot.navigation_config),
        clock=robot.clock,
        sleep=robot.sleep,
    )


class NavigationControllerTest(unittest.TestCase):
    def test_nav2_start_clearance_detects_camera_centered_obstacle(self) -> None:
        robot = ManualRobot(clearance_m=3.0)
        robot.manipulation_config = {
            "camera_calibration_resolution": [96, 64],
            "camera_intrinsics": [
                [100.0, 0.0, 48.0],
                [0.0, 100.0, 32.0],
                [0.0, 0.0, 1.0],
            ],
            "visible_object_docking": {
                "ground_camera_height_m": 1.0,
                "ground_down_camera_xyz": [0.0, 1.0, 0.0],
                "obstacle_min_height_m": 0.10,
                "obstacle_max_height_m": 1.45,
                "obstacle_max_depth_m": 20.0,
            },
        }
        depth = np.full((64, 96), 3.0, dtype=np.float32)
        depth[24:42, 39:57] = 0.40
        frame = Frame(planar_pose=robot.pose, depth_m=depth)
        robot.navigation_frame = lambda **_options: frame

        status = make_controller(robot).nav2_start_clearance_status(radius_m=0.60)

        self.assertTrue(status["available"])
        self.assertTrue(status["blocked"])
        self.assertGreater(status["obstacle_points"], status["max_points"])
        self.assertLess(status["minimum_planar_obstacle_distance_m"], 0.60)

    def test_automatic_motion_timeouts_include_physical_startup_margin(self) -> None:
        config = NavigationConfig.from_mapping({})

        self.assertEqual(config.auto_timeout_scale, 7.0)
        self.assertEqual(config.auto_timeout_overhead_s, 5.0)
        expected_drive_timeout = (
            0.30 / 0.18 * config.auto_timeout_scale
            + config.auto_timeout_overhead_s
        )
        expected_turn_timeout = (
            math.radians(15.0) / 0.35 * config.auto_timeout_scale
            + config.auto_timeout_overhead_s
        )
        self.assertGreater(expected_drive_timeout, 11.0)
        self.assertGreater(expected_turn_timeout, 6.5)

        turn_controller = make_controller(ManualRobot())
        turn_nominals = []
        turn_controller._timeout = lambda value, *, nominal: (
            turn_nominals.append((value, nominal)) or 0.05
        )
        turn_controller.turn_relative(
            math.radians(15.0), max_yaw_rad_s=0.35
        )

        drive_controller = make_controller(ManualRobot())
        drive_nominals = []
        drive_controller._timeout = lambda value, *, nominal: (
            drive_nominals.append((value, nominal)) or 0.05
        )
        drive_controller.drive_straight(0.30, max_speed_mps=0.18)

        lateral_controller = make_controller(ManualRobot())
        lateral_nominals = []
        lateral_controller._timeout = lambda value, *, nominal: (
            lateral_nominals.append((value, nominal)) or 0.05
        )
        lateral_controller.drive_lateral(0.30, max_speed_mps=0.18)

        self.assertIsNone(turn_nominals[0][0])
        self.assertAlmostEqual(turn_nominals[0][1], expected_turn_timeout)
        self.assertIsNone(drive_nominals[0][0])
        self.assertAlmostEqual(drive_nominals[0][1], expected_drive_timeout)
        self.assertIsNone(lateral_nominals[0][0])
        self.assertAlmostEqual(lateral_nominals[0][1], expected_drive_timeout)

    def test_move_planar_relative_tracks_lateral_goal_and_calls_guard(self) -> None:
        robot = ManualRobot()
        guard_calls = []

        def guard(_frame, state):
            guard_calls.append(dict(state))
            return {"clear": True, "reason": "segment_clear"}

        result = make_controller(robot).move_planar_relative(
            0.05,
            0.15,
            0.0,
            max_linear_mps=0.10,
            max_lateral_mps=0.10,
            timeout_s=8.0,
            motion_guard=guard,
        )

        self.assertTrue(result["success"])
        self.assertGreater(len(guard_calls), 0)
        self.assertGreater(robot.pose.y_m, 0.12)
        self.assertTrue(any(command[1] > 0.0 for command in robot.commands))
        self.assertEqual(robot.commands[-1], [0.0, 0.0, 0.0])

    def test_planar_diagonal_keeps_roundoff_below_hardware_limit(self) -> None:
        robot = StrictLimitRobot()

        result = make_controller(robot).move_planar_relative(
            0.20,
            0.20,
            0.0,
            max_linear_mps=0.12,
            max_lateral_mps=0.10,
            timeout_s=8.0,
            motion_guard=lambda _frame, _state: {
                "clear": True,
                "reason": "segment_clear",
            },
        )

        self.assertTrue(result["success"])
        nonzero = [command for command in robot.commands if any(command)]
        self.assertTrue(nonzero)
        self.assertTrue(
            all(math.hypot(command[0], command[1]) < 0.12 for command in nonzero)
        )

    def test_move_planar_relative_guard_blocks_before_nonzero_motion(self) -> None:
        robot = ManualRobot()

        result = make_controller(robot).move_planar_relative(
            0.0,
            0.15,
            0.0,
            max_linear_mps=0.10,
            max_lateral_mps=0.10,
            timeout_s=8.0,
            motion_guard=lambda _frame, _state: {
                "clear": False,
                "reason": "inflated_obstacle",
            },
        )

        self.assertFalse(result["success"])
        self.assertIn("inflated_obstacle", result["reason"])
        self.assertEqual(
            result["metrics"]["motion_guard"]["reason"],
            "inflated_obstacle",
        )
        self.assertTrue(
            all(command == [0.0, 0.0, 0.0] for command in robot.commands)
        )

    def test_operator_stop_blocks_future_nonzero_commands(self) -> None:
        robot = ManualRobot()
        controller = make_controller(robot)

        reply = controller.request_stop()
        result = controller.turn_relative(math.radians(20))

        self.assertTrue(reply["accepted"])
        self.assertFalse(result["success"])
        self.assertIn("operator_stop_requested", result["reason"])
        self.assertTrue(robot.commands)
        self.assertTrue(
            all(command == [0.0, 0.0, 0.0] for command in robot.commands)
        )

    def test_turn_relative_closes_loop_and_finishes_at_zero(self) -> None:
        robot = ManualRobot()
        result = make_controller(robot).turn_relative(0.50, timeout_s=8.0)

        self.assertTrue(result["success"])
        self.assertEqual(result["reason"], "target_reached")
        self.assertLess(
            abs(wrap_angle(robot.pose.yaw_rad - 0.50)), math.radians(2.5)
        )
        self.assertEqual(robot.commands[-1], [0.0, 0.0, 0.0])
        self.assertTrue(any(command[2] > 0 for command in robot.commands))

    def test_turn_relative_enforces_static_friction_yaw_floor(self) -> None:
        robot = ManualRobot()
        robot.navigation_config.update(
            {
                "min_yaw_rad_s": 0.25,
                "yaw_kp": 0.2,
            }
        )

        result = make_controller(robot).turn_relative(0.10, timeout_s=3.0)

        self.assertTrue(result["success"], result)
        nonzero_yaw = [
            abs(command[2]) for command in robot.commands if command[2] != 0.0
        ]
        self.assertTrue(nonzero_yaw)
        self.assertGreaterEqual(min(nonzero_yaw), 0.25)

    def test_planar_floors_only_axes_outside_their_tolerance(self) -> None:
        translating_robot = ManualRobot()
        translating_robot.navigation_config.update({"min_linear_mps": 0.05})
        translated = make_controller(translating_robot).move_planar_relative(
            0.04,
            0.0,
            0.0,
            max_linear_mps=0.10,
            max_lateral_mps=0.10,
            timeout_s=3.0,
        )

        self.assertTrue(translated["success"], translated)
        translation_commands = [
            command for command in translating_robot.commands if any(command)
        ]
        self.assertTrue(translation_commands)
        self.assertTrue(
            all(
                math.hypot(command[0], command[1]) >= 0.05 - 1e-9
                for command in translation_commands
            )
        )
        self.assertTrue(
            all(command[2] == 0.0 for command in translation_commands)
        )

        turning_robot = ManualRobot()
        turning_robot.navigation_config.update(
            {"min_linear_mps": 0.05, "min_yaw_rad_s": 0.25}
        )
        turned = make_controller(turning_robot).move_planar_relative(
            0.0,
            0.0,
            0.10,
            max_linear_mps=0.10,
            max_lateral_mps=0.10,
            max_yaw_rad_s=0.30,
            timeout_s=3.0,
        )

        self.assertTrue(turned["success"], turned)
        turn_commands = [
            command for command in turning_robot.commands if any(command)
        ]
        self.assertTrue(turn_commands)
        self.assertTrue(
            all(
                command[0] == 0.0 and command[1] == 0.0
                for command in turn_commands
            )
        )
        self.assertGreaterEqual(
            min(abs(command[2]) for command in turn_commands),
            0.25 - 1e-9,
        )

    def test_absolute_planar_target_can_reverse_correct_an_overshoot(self) -> None:
        robot = ManualRobot()
        robot.pose.x_m = 0.06
        robot.navigation_config.update({"min_linear_mps": 0.05})

        result = make_controller(robot).move_planar_relative(
            0.0,
            0.0,
            0.0,
            max_linear_mps=0.10,
            max_lateral_mps=0.10,
            position_tolerance_m=0.01,
            timeout_s=3.0,
            allow_reverse=True,
            target_pose_world=[0.04, 0.0, 0.0],
        )

        self.assertTrue(result["success"], result)
        self.assertTrue(any(command[0] < 0.0 for command in robot.commands))
        self.assertAlmostEqual(robot.pose.x_m, 0.04, delta=0.011)

    def test_near_target_motion_stays_at_the_static_friction_floor(self) -> None:
        robot = ManualRobot()
        robot.navigation_config.update({"min_linear_mps": 0.05})

        result = make_controller(robot).move_planar_relative(
            0.0,
            0.0,
            0.0,
            max_linear_mps=0.10,
            max_lateral_mps=0.10,
            position_tolerance_m=0.01,
            timeout_s=4.0,
            allow_reverse=True,
            target_pose_world=[0.04, 0.0, 0.0],
        )

        self.assertTrue(result["success"], result)
        moving_indices = [
            index
            for index, command in enumerate(robot.commands)
            if any(command)
        ]
        self.assertTrue(moving_indices)
        first_moving = moving_indices[0]
        last_moving = moving_indices[-1]
        self.assertTrue(
            all(
                any(robot.commands[index])
                for index in range(first_moving, last_moving + 1)
            )
        )
        self.assertGreaterEqual(
            min(abs(robot.commands[index][0]) for index in moving_indices),
            0.05 - 1e-9,
        )

    def test_drive_straight_closes_loop_and_checks_depth(self) -> None:
        robot = ManualRobot(clearance_m=2.5)
        result = make_controller(robot).drive_straight(0.15, timeout_s=8.0)

        self.assertTrue(result["success"])
        self.assertGreaterEqual(result["metrics"]["progress_m"], 0.125)
        self.assertLess(abs(result["metrics"]["cross_track_error_m"]), 0.02)
        self.assertEqual(robot.commands[-1], [0.0, 0.0, 0.0])
        self.assertTrue(any(command[0] > 0 for command in robot.commands))

    def test_drive_straight_supports_precise_short_planner_step(self) -> None:
        robot = ManualRobot(clearance_m=2.5)
        result = make_controller(robot).drive_straight(
            0.036,
            timeout_s=4.0,
            distance_tolerance_m=0.008,
        )

        self.assertTrue(result["success"], result)
        self.assertGreaterEqual(result["metrics"]["progress_m"], 0.028)
        self.assertEqual(result["metrics"]["distance_tolerance_m"], 0.008)
        self.assertEqual(robot.commands[-1], [0.0, 0.0, 0.0])

    def test_drive_straight_supports_bounded_reverse(self) -> None:
        # A low front clearance must not be mistaken for rear clearance.
        robot = ManualRobot(clearance_m=0.10)
        result = make_controller(robot).drive_straight(-0.15, timeout_s=8.0)

        self.assertTrue(result["success"])
        self.assertEqual(result["metrics"]["direction"], "reverse")
        self.assertFalse(result["metrics"]["rear_clearance_checked"])
        self.assertLessEqual(result["metrics"]["progress_m"], -0.125)
        self.assertEqual(robot.commands[-1], [0.0, 0.0, 0.0])
        self.assertTrue(any(command[0] < 0 for command in robot.commands))

    def test_drive_lateral_closes_loop_in_both_directions(self) -> None:
        robot = ManualRobot(clearance_m=0.10)
        controller = make_controller(robot)

        left = controller.drive_lateral(0.15, timeout_s=8.0)
        right = controller.drive_lateral(-0.15, timeout_s=8.0)

        self.assertTrue(left["success"], left)
        self.assertEqual(left["metrics"]["direction"], "left")
        self.assertFalse(left["metrics"]["lateral_clearance_checked"])
        self.assertTrue(right["success"], right)
        self.assertEqual(right["metrics"]["direction"], "right")
        self.assertTrue(any(command[1] > 0.0 for command in robot.commands))
        self.assertTrue(any(command[1] < 0.0 for command in robot.commands))
        self.assertAlmostEqual(robot.pose.y_m, 0.0, delta=0.05)
        self.assertEqual(robot.commands[-1], [0.0, 0.0, 0.0])

    def test_lateral_distance_uses_forward_distance_bound_for_both_signs(self) -> None:
        controller = make_controller(ManualRobot())

        with self.assertRaises(ValueError):
            controller.drive_lateral(2.01)
        with self.assertRaises(ValueError):
            controller.drive_lateral(-2.01)

    def test_reverse_distance_uses_shorter_safety_bound(self) -> None:
        robot = ManualRobot()

        with self.assertRaises(ValueError):
            make_controller(robot).drive_straight(-0.51)

    def test_drive_refuses_obstacle_without_nonzero_command(self) -> None:
        robot = ManualRobot(clearance_m=0.30)
        result = make_controller(robot).drive_straight(0.15, timeout_s=3.0)

        self.assertFalse(result["success"])
        self.assertEqual(result["reason"], "obstacle_too_close")
        self.assertTrue(
            all(command == [0.0, 0.0, 0.0] for command in robot.commands)
        )

    def test_stale_pose_failure_still_sends_final_zero(self) -> None:
        robot = ManualRobot()
        robot.frame_error = RuntimeError("stale_pose")
        result = make_controller(robot).turn_relative(-0.4, timeout_s=3.0)

        self.assertFalse(result["success"])
        self.assertIn("stale_pose", result["reason"])
        self.assertEqual(robot.commands[-1], [0.0, 0.0, 0.0])

    def test_stop_confirms_zero(self) -> None:
        robot = ManualRobot()
        robot.velocity[:] = [0.1, 0.0, 0.0]
        result = make_controller(robot).stop()

        self.assertTrue(result["success"])
        self.assertEqual(result["reason"], "zero_confirmed")
        self.assertEqual(robot.commands[-1], [0.0, 0.0, 0.0])

    def test_keyboard_interrupt_runs_final_stop_before_propagating(self) -> None:
        robot = ManualRobot()
        interrupted = False

        def interrupt_once(duration: float) -> None:
            nonlocal interrupted
            if not interrupted:
                interrupted = True
                raise KeyboardInterrupt
            robot.sleep(duration)

        controller = NavigationController(
            robot,
            config=NavigationConfig.from_mapping(robot.navigation_config),
            clock=robot.clock,
            sleep=interrupt_once,
        )
        with self.assertRaises(KeyboardInterrupt):
            controller.turn_relative(0.5, timeout_s=8.0)

        self.assertEqual(robot.commands[-1], [0.0, 0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
