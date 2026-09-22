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
    timestamp_ns: int = 2_000_000_000
    ground_camera_height_m: float | None = None
    ground_down_camera_xyz: tuple[float, float, float] | None = None
    ground_plane_timestamp_ns: int | None = None


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


def make_controller(robot: ManualRobot) -> NavigationController:
    return NavigationController(
        robot,
        config=NavigationConfig.from_mapping(robot.navigation_config),
        clock=robot.clock,
        sleep=robot.sleep,
    )


class NavigationControllerTest(unittest.TestCase):
    @staticmethod
    def _ground_aware_robot() -> ManualRobot:
        robot = ManualRobot()
        robot.manipulation_config = {
            "camera_calibration_resolution": [96, 64],
            "camera_intrinsics": [
                [100.0, 0.0, 47.5],
                [0.0, 100.0, 31.5],
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
        return robot

    @staticmethod
    def _floor_depth() -> np.ndarray:
        depth = np.full((64, 96), np.nan, dtype=np.float32)
        for row in range(37, 50):
            depth[row, :] = 100.0 / (row - 31.5)
        return depth

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

    def test_drive_straight_closes_loop_and_checks_depth(self) -> None:
        robot = ManualRobot(clearance_m=2.5)
        result = make_controller(robot).drive_straight(0.15, timeout_s=8.0)

        self.assertTrue(result["success"])
        self.assertGreaterEqual(result["metrics"]["progress_m"], 0.125)
        self.assertLess(abs(result["metrics"]["cross_track_error_m"]), 0.02)
        self.assertEqual(robot.commands[-1], [0.0, 0.0, 0.0])
        self.assertTrue(any(command[0] > 0 for command in robot.commands))

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

    def test_continuous_clearance_ignores_calibrated_floor(self) -> None:
        robot = self._ground_aware_robot()
        controller = make_controller(robot)
        frame = Frame(robot.pose, self._floor_depth())

        clearance = controller._front_clearance(frame)

        self.assertIsNone(clearance)
        self.assertEqual(
            controller.last_front_clearance_debug["mode"],
            "calibrated_floor_height",
        )
        self.assertGreater(
            controller.last_front_clearance_debug["ground_filtered_pixels"], 0
        )
        self.assertEqual(
            controller.last_front_clearance_debug["minimum_obstacle_height_m"],
            0.10,
        )

    def test_continuous_clearance_keeps_above_floor_obstacle(self) -> None:
        robot = self._ground_aware_robot()
        controller = make_controller(robot)
        depth = self._floor_depth()
        depth[25:35, 34:63] = 0.30
        frame = Frame(robot.pose, depth)

        clearance = controller._front_clearance(frame)

        self.assertAlmostEqual(clearance, 0.30, places=2)
        self.assertGreater(
            controller.last_front_clearance_debug["height_candidate_pixels"], 0
        )

    def test_continuous_clearance_uses_dynamic_lift_height(self) -> None:
        robot = self._ground_aware_robot()
        robot.manipulation_config["visible_object_docking"].update(
            {
                "require_dynamic_ground_plane": True,
                "ground_plane_max_age_s": 0.5,
            }
        )
        controller = make_controller(robot)
        depth = np.full((64, 96), np.nan, dtype=np.float32)
        for row in range(39, 50):
            depth[row, :] = 150.0 / (row - 31.5)
        frame = Frame(
            robot.pose,
            depth,
            ground_camera_height_m=1.5,
            ground_down_camera_xyz=(0.0, 1.0, 0.0),
            ground_plane_timestamp_ns=1_900_000_000,
        )

        clearance = controller._front_clearance(frame)

        self.assertIsNone(clearance)
        self.assertEqual(
            controller.last_front_clearance_debug["ground_plane_source"],
            "zed_sdk_floor_plane",
        )
        self.assertAlmostEqual(
            controller.last_front_clearance_debug["ground_camera_height_m"],
            1.5,
        )

    def test_nav2_settings_may_omit_legacy_obstacle_height_filters(self) -> None:
        robot = self._ground_aware_robot()
        docking = robot.manipulation_config["visible_object_docking"]
        for key in (
            "obstacle_min_height_m",
            "obstacle_max_height_m",
            "obstacle_max_depth_m",
        ):
            docking.pop(key)
        docking["require_dynamic_ground_plane"] = True
        controller = make_controller(robot)
        frame = Frame(
            robot.pose,
            self._floor_depth(),
            ground_camera_height_m=1.0,
            ground_down_camera_xyz=(0.0, 1.0, 0.0),
            ground_plane_timestamp_ns=1_900_000_000,
        )

        self.assertIsNone(controller._front_clearance(frame))
        self.assertEqual(
            controller.last_front_clearance_debug["minimum_obstacle_height_m"],
            0.10,
        )

    def test_continuous_clearance_requires_fresh_dynamic_ground(self) -> None:
        robot = self._ground_aware_robot()
        robot.manipulation_config["visible_object_docking"].update(
            {"require_dynamic_ground_plane": True}
        )
        controller = make_controller(robot)

        with self.assertRaisesRegex(
            RuntimeError, "dynamic_ground_plane_unavailable"
        ):
            controller._front_clearance(Frame(robot.pose, self._floor_depth()))

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
