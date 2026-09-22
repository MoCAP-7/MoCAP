from __future__ import annotations

import inspect
import math
import unittest

import numpy as np

from yor_agent.robot.visible_object_navigation import (
    VisibleObjectDockingConfig,
    VisibleObjectDockingController,
)
from yor_agent.robot.visualization.navigation import (
    _point_cloud_in_navigation_frame,
    _rounded_bbox_boundary,
    _world_xy_in_navigation_frame,
)


class FakeDockingEnv:
    def __init__(
        self,
        *,
        target_distance_m: float = 1.0,
        target_column: int = 50,
        obstacle_distance_m: float | None = None,
        target_visible: bool = True,
    ) -> None:
        self.navigation_config = {}
        self.manipulation_config = {
            "camera_calibration_resolution": [100, 80],
            "camera_intrinsics": [
                [100.0, 0.0, 50.0],
                [0.0, 100.0, 40.0],
                [0.0, 0.0, 1.0],
            ],
            "sam3_score_threshold": 0.05,
            "visible_object_docking": {
                "obstacle_min_cluster_pixels": 30,
                "obstacle_min_sensor_pixels": 40,
                "maximum_iterations": 20,
            },
        }
        self.target_distance_m = target_distance_m
        self.target_column = target_column
        self.obstacle_distance_m = obstacle_distance_m
        self.target_visible = target_visible
        self.current_mask = np.zeros((80, 100), dtype=bool)
        self.ground_plane: dict | None = None
        self.frame_index = 0
        self.next_observations = 0

    def observe(self):
        rgb = np.zeros((80, 100, 3), dtype=np.uint8)
        depth = np.full((80, 100), 3.0, dtype=np.float32)
        mask = np.zeros((80, 100), dtype=bool)
        column0 = max(0, self.target_column - 6)
        column1 = min(100, self.target_column + 6)
        mask[30:60, column0:column1] = True
        depth[mask] = self.target_distance_m
        if self.obstacle_distance_m is not None:
            # A separate obstacle inside the base corridor but outside the
            # target mask, so it cannot alter the target bbox estimate.
            depth[20:45, 30:36] = self.obstacle_distance_m
        self.current_mask = mask
        camera = {
            "images": {"rgb": rgb, "depth": depth[..., None]},
            "timestamp_ns": 2_000_000_000 + self.frame_index,
        }
        if self.ground_plane is not None:
            camera["ground_plane"] = dict(self.ground_plane)
        return {
            "robot0_robotview": {
                **camera,
            }
        }

    def observe_next(self):
        self.frame_index += 1
        self.next_observations += 1
        return self.observe()

    def segment(self, _rgb, *, text_prompt):
        self.last_prompt = text_prompt
        if not self.target_visible:
            return []
        return [{"mask": self.current_mask.copy(), "score": 0.9}]

    # NavigationController validates these methods during API construction.
    # Tests replace the motion object before executing a primitive.
    def navigation_frame(self, *, max_age_s):
        raise AssertionError(max_age_s)

    def base_status(self):
        return {}

    def submit_base_velocity(self, velocity):
        raise AssertionError(velocity)


class FakeMotion:
    def __init__(self, env: FakeDockingEnv) -> None:
        self.env = env
        self.stops = 0
        self.turns: list[float] = []
        self.turn_options: list[tuple[float | None, float | None]] = []
        self.drives: list[float] = []
        self.drive_tolerances: list[float | None] = []
        self.planar_moves: list[tuple[float, float]] = []
        self.planar_yaws: list[float] = []

    @staticmethod
    def _result(primitive: str):
        return {
            "primitive": primitive,
            "success": True,
            "reason": "target_reached",
            "metrics": {},
        }

    def stop(self):
        self.stops += 1
        result = self._result("stop")
        result["reason"] = "zero_confirmed"
        return result

    def turn_relative(
        self, angle_rad, *, max_yaw_rad_s=None, timeout_s=None
    ):
        self.turns.append(float(angle_rad))
        self.turn_options.append((max_yaw_rad_s, timeout_s))
        # Simulate that the bounded turn centers the target in the next image.
        self.env.target_column = 50
        return self._result("turn_relative")

    def drive_straight(
        self,
        distance_m,
        *,
        max_speed_mps,
        distance_tolerance_m=None,
        pose_callback=None,
    ):
        del pose_callback
        self.drives.append(float(distance_m))
        self.drive_tolerances.append(distance_tolerance_m)
        self.env.target_distance_m -= float(distance_m)
        result = self._result("drive_straight")
        result["metrics"] = {"max_speed_mps": float(max_speed_mps)}
        return result

    def move_planar_relative(
        self,
        forward_m,
        left_m,
        yaw_rad,
        *,
        max_linear_mps,
        max_lateral_mps,
        max_yaw_rad_s,
        timeout_s,
        allow_reverse,
        motion_guard,
    ):
        del max_yaw_rad_s, timeout_s, allow_reverse, motion_guard
        self.planar_moves.append((float(forward_m), float(left_m)))
        self.planar_yaws.append(float(yaw_rad))
        if abs(float(yaw_rad)) > 0.0:
            self.env.target_column = 50
        self.env.target_distance_m -= math.hypot(forward_m, left_m)
        result = self._result("move_planar_relative")
        result["metrics"] = {
            "max_linear_mps": float(max_linear_mps),
            "max_lateral_mps": float(max_lateral_mps),
        }
        return result


def make_api(
    env: FakeDockingEnv,
) -> tuple[VisibleObjectDockingController, FakeMotion]:
    motion = FakeMotion(env)
    env.controller = motion
    api = VisibleObjectDockingController(env)
    api._segment = env.segment
    return api, motion


class VisibleObjectNavigationTest(unittest.TestCase):
    def test_default_threshold_uses_camera_at_robot_front(self) -> None:
        config = VisibleObjectDockingConfig.from_mapping(None)

        self.assertEqual(config.docking_distance_m, 0.7)
        self.assertAlmostEqual(config.maximum_turn_step_rad, np.deg2rad(25.0))
        self.assertEqual(config.turn_speed_rad_s, 0.35)
        self.assertEqual(config.turn_timeout_s, 10.0)
        self.assertEqual(config.maximum_forward_step_m, 0.5)
        self.assertEqual(config.obstacle_path_margin_m, 0.1)
        self.assertAlmostEqual(config.bearing_tolerance_rad, np.deg2rad(10.0))
        self.assertEqual(config.forward_speed_mps, 0.18)
        self.assertEqual(config.corridor_half_width_m, 0.32)
        self.assertEqual(config.robot_radius_m, 0.30)
        self.assertEqual(config.obstacle_clearance_margin_m, 0.0)
        self.assertEqual(config.effective_footprint_radius_m, 0.30)
        self.assertEqual(config.planner_lookahead_m, 0.80)
        self.assertEqual(config.avoidance_lateral_speed_mps, 0.10)
        self.assertEqual(config.avoidance_motion_timeout_s, 18.0)
        self.assertEqual(config.target_max_depth_m, 20.0)
        self.assertEqual(config.target_min_depth_pixels, 20)
        self.assertEqual(config.obstacle_max_depth_m, 20.0)
        self.assertEqual(config.maximum_total_forward_m, 20.0)
        self.assertEqual(config.maximum_iterations, 100)
        self.assertEqual(config.timeout_s, 300.0)
        self.assertEqual(config.obstacle_temporal_frames, 3)
        self.assertAlmostEqual(config.ground_camera_height_m, 1.122339129447937)
        self.assertEqual(config.obstacle_min_height_m, 0.10)
        self.assertEqual(config.obstacle_max_height_m, 1.45)

    def test_required_dynamic_ground_plane_updates_camera_height(self) -> None:
        env = FakeDockingEnv()
        env.manipulation_config["visible_object_docking"].update(
            {
                "require_dynamic_ground_plane": True,
                "ground_plane_max_age_s": 0.5,
                "ground_down_camera_xyz": [0.0, 1.0, 0.0],
            }
        )
        # The first plane inside the gate's envelope replaces the configured
        # fallback at once.
        env.ground_plane = {
            "valid": True,
            "camera_height_m": 1.10,
            "down_camera_xyz": [0.0, 1.0, 0.0],
            "timestamp_ns": 1_900_000_000,
        }
        api, _ = make_api(env)

        api._camera_input()

        self.assertTrue(api.last_ground_plane_debug["dynamic"])
        self.assertAlmostEqual(
            api.last_ground_plane_debug["camera_height_m"], 1.10
        )
        self.assertAlmostEqual(api.last_ground_plane_debug["age_s"], 0.1)
        self.assertEqual(api.last_ground_plane_debug["gate"], "accepted")
        self.assertEqual(api.last_ground_plane_debug["held_for_s"], 0.0)
        self.assertNotIn("rejected_camera_height_m", api.last_ground_plane_debug)

    def test_ground_plane_far_from_the_previous_one_is_held(self) -> None:
        env = FakeDockingEnv()
        env.manipulation_config["visible_object_docking"].update(
            {
                "require_dynamic_ground_plane": True,
                "ground_down_camera_xyz": [0.0, 1.0, 0.0],
            }
        )
        env.ground_plane = {
            "valid": True,
            "camera_height_m": 1.10,
            "down_camera_xyz": [0.0, 1.0, 0.0],
            "timestamp_ns": 2_000_000_000,
        }
        api, _ = make_api(env)
        api._camera_input()
        env.frame_index = 500_000_000
        env.ground_plane.update(
            {"camera_height_m": 1.70, "timestamp_ns": 2_500_000_000}
        )

        api._camera_input()

        debug = api.last_ground_plane_debug
        self.assertTrue(debug["dynamic"])
        self.assertEqual(debug["source"], "zed_sdk_floor_plane")
        self.assertEqual(debug["gate"], "held")
        self.assertAlmostEqual(debug["camera_height_m"], 1.10)
        self.assertAlmostEqual(debug["rejected_camera_height_m"], 1.70)
        self.assertAlmostEqual(debug["held_for_s"], 0.5)
        self.assertIn("envelope", debug["gate_reason"])
        self.assertAlmostEqual(api._active_ground_camera_height_m, 1.10)

    def test_ground_plane_offset_held_steadily_is_adopted_after_settling(
        self,
    ) -> None:
        env = FakeDockingEnv()
        env.manipulation_config["visible_object_docking"].update(
            {
                "require_dynamic_ground_plane": True,
                "ground_down_camera_xyz": [0.0, 1.0, 0.0],
                "ground_plane_settle_s": 2.0,
                "ground_plane_max_height_error_m": 0.8,
            }
        )
        env.ground_plane = {
            "valid": True,
            "camera_height_m": 1.10,
            "down_camera_xyz": [0.0, 1.0, 0.0],
            "timestamp_ns": 2_000_000_000,
        }
        api, _ = make_api(env)
        api._camera_input()
        statuses = []
        # The same 0.6 m offset, observation after observation, over more
        # than the settle time: a lift move, not a burst.
        for offset_ns in (500_000_000, 1_000_000_000, 1_500_000_000, 2_500_000_000):
            env.frame_index = offset_ns
            env.ground_plane.update(
                {"camera_height_m": 1.70, "timestamp_ns": 2_000_000_000 + offset_ns}
            )
            api._camera_input()
            statuses.append(api.last_ground_plane_debug["gate"])

        self.assertEqual(statuses, ["held", "settling", "settling", "accepted"])
        self.assertAlmostEqual(
            api.last_ground_plane_debug["camera_height_m"], 1.70
        )
        self.assertAlmostEqual(api._active_ground_camera_height_m, 1.70)
        self.assertNotIn("rejected_camera_height_m", api.last_ground_plane_debug)

    def test_ground_plane_gate_settings_are_validated(self) -> None:
        for settings in (
            {"ground_plane_max_height_step_m": 0.0},
            {"ground_plane_max_tilt_step_deg": 60.0},
            {"ground_plane_settle_s": 0.0},
            {"ground_plane_max_height_error_m": 0.01},
            {"ground_plane_max_tilt_error_deg": 1.0},
        ):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                VisibleObjectDockingConfig.from_mapping(settings)

    def test_required_dynamic_ground_plane_fails_when_absent(self) -> None:
        env = FakeDockingEnv()
        env.manipulation_config["visible_object_docking"].update(
            {"require_dynamic_ground_plane": True}
        )
        api, _ = make_api(env)

        with self.assertRaisesRegex(
            RuntimeError, "dynamic_ground_plane_unavailable"
        ):
            api._camera_input()

    def test_controller_keeps_semantic_docking_documentation(self) -> None:
        api, _ = make_api(FakeDockingEnv())

        documentation = inspect.getdoc(api.dock_to_visible_object)
        self.assertIn("visible semantic target", documentation)
        self.assertIn("chosen by the caller", documentation)

    def test_already_docked_stops_without_nonzero_motion(self) -> None:
        api, motion = make_api(FakeDockingEnv(target_distance_m=0.68))

        result = api.dock_to_visible_object("table")

        self.assertTrue(result["success"])
        self.assertEqual(result["reason"], "within_docking_distance")
        self.assertEqual(result["metrics"]["camera_to_robot_front_m"], 0.0)
        self.assertLessEqual(result["metrics"]["target_distance_m"], 0.7)
        self.assertEqual(motion.turns, [])
        self.assertEqual(motion.drives, [])
        self.assertEqual(motion.stops, 2)

    def test_resegments_and_advances_in_bounded_steps(self) -> None:
        env = FakeDockingEnv(target_distance_m=1.80)
        api, motion = make_api(env)

        result = api.dock_to_visible_object("table")

        self.assertTrue(result["success"])
        self.assertGreaterEqual(len(motion.drives), 2)
        distances = motion.drives
        self.assertTrue(all(step <= 0.80 for step in distances))
        self.assertTrue(all(step > 0.0 for step in distances))
        self.assertTrue(any(step > 0.50 for step in distances))
        self.assertEqual(motion.planar_moves, [])
        self.assertLessEqual(result["metrics"]["target_distance_m"], 0.7)
        self.assertEqual(env.last_prompt, "table")
        self.assertEqual(motion.stops, 2)

    def test_target_left_produces_positive_bounded_turn(self) -> None:
        env = FakeDockingEnv(target_distance_m=1.20, target_column=20)
        api, motion = make_api(env)

        result = api.dock_to_visible_object("table")

        self.assertTrue(result["success"])
        self.assertGreater(motion.turns[0], 0.0)
        self.assertLessEqual(
            motion.turns[0], api.config.maximum_turn_step_rad
        )
        self.assertEqual(motion.planar_moves, [])

    def test_target_right_produces_negative_bounded_turn(self) -> None:
        env = FakeDockingEnv(target_distance_m=1.20, target_column=80)
        api, motion = make_api(env)

        result = api.dock_to_visible_object("table")

        self.assertTrue(result["success"])
        self.assertLess(motion.turns[0], 0.0)
        self.assertGreaterEqual(
            motion.turns[0], -api.config.maximum_turn_step_rad
        )
        self.assertEqual(motion.planar_moves, [])

    def test_sub_five_cm_goal_grid_step_uses_precise_straight_motion(self) -> None:
        env = FakeDockingEnv(target_distance_m=1.20, target_column=30)
        api, motion = make_api(env)

        result = api.dock_to_visible_object("table")

        self.assertTrue(result["success"])
        self.assertEqual(motion.turns, [])
        self.assertGreater(len(motion.drives), 0)
        self.assertEqual(motion.planar_moves, [])
        self.assertAlmostEqual(motion.drives[-1], 0.05)
        self.assertEqual(motion.drive_tolerances[-1], 0.01)
        self.assertTrue(result["metrics"]["precise_grid_step"])
        drive_updates = [
            update
            for update in result["motion_history"]
            if update["primitive"] == "drive_straight"
        ]
        self.assertGreater(len(drive_updates), 0)
        self.assertEqual(result["metrics"]["total_lateral_command_m"], 0.0)

    def test_coherent_obstacle_blocks_path_without_drive(self) -> None:
        env = FakeDockingEnv(
            target_distance_m=1.20,
            obstacle_distance_m=0.30,
        )
        api, motion = make_api(env)

        result = api.dock_to_visible_object("table")

        self.assertFalse(result["success"])
        self.assertEqual(result["reason"], "path_blocked")
        self.assertEqual(motion.drives, [])
        self.assertGreater(
            result["metrics"]["path_clearance"]["nearest_obstacle_m"], 0.27
        )
        self.assertLess(
            result["metrics"]["path_clearance"]["nearest_obstacle_m"], 0.31
        )
        self.assertNotIn(
            "_obstacle_points_path_m", result["metrics"]["path_clearance"]
        )
        self.assertEqual(motion.stops, 2)

    def test_same_depth_but_disconnected_artifacts_do_not_block_path(self) -> None:
        api, _ = make_api(FakeDockingEnv())
        depth = np.full((80, 100), 3.0, dtype=np.float32)
        for row0 in (18, 30, 42):
            depth[row0 : row0 + 4, 47:52] = 0.30
        intrinsics = np.asarray(
            [[100.0, 0.0, 50.0], [0.0, 100.0, 40.0], [0.0, 0.0, 1.0]]
        )

        status = api._straight_path_status(
            depth,
            intrinsics,
            bearing_rad=0.0,
            travel_distance_m=0.40,
        )

        self.assertTrue(status["valid"])
        self.assertTrue(status["clear"])
        self.assertEqual(status["obstacle_pixels"], 0)
        self.assertEqual(status["depth_layer_pixels"], 60)
        self.assertEqual(status["largest_rejected_component_pixels"], 20)

    def test_path_check_only_covers_next_bounded_step(self) -> None:
        api, _ = make_api(FakeDockingEnv())
        depth = np.full((80, 100), 3.0, dtype=np.float32)
        depth[20:45, 30:36] = 1.4
        intrinsics = np.asarray(
            [[100.0, 0.0, 50.0], [0.0, 100.0, 40.0], [0.0, 0.0, 1.0]]
        )

        status = api._straight_path_status(
            depth,
            intrinsics,
            bearing_rad=0.0,
            travel_distance_m=3.0,
        )

        self.assertTrue(status["clear"])
        self.assertAlmostEqual(status["path_limit_m"], 0.60)

    def test_semantic_target_mask_is_not_treated_as_obstacle(self) -> None:
        api, _ = make_api(FakeDockingEnv())
        depth = np.full((80, 100), 3.0, dtype=np.float32)
        target_mask = np.zeros((80, 100), dtype=bool)
        target_mask[20:45, 30:36] = True
        depth[target_mask] = 0.30
        intrinsics = np.asarray(
            [[100.0, 0.0, 50.0], [0.0, 100.0, 40.0], [0.0, 0.0, 1.0]]
        )

        status = api._straight_path_status(
            depth,
            intrinsics,
            bearing_rad=0.0,
            travel_distance_m=0.40,
            excluded_mask=target_mask,
        )

        self.assertTrue(status["clear"])
        self.assertGreater(status["target_mask_filtered_pixels"], 0)

    def test_calibrated_height_filter_removes_floor_points(self) -> None:
        env = FakeDockingEnv()
        env.manipulation_config["visible_object_docking"].update(
            {
                "ground_camera_height_m": 1.0,
                "ground_down_camera_xyz": [0.0, 1.0, 0.0],
                "obstacle_top_row_fraction": 0.0,
                "obstacle_bottom_row_fraction": 1.0,
            }
        )
        api, _ = make_api(env)
        depth = np.full((80, 100), np.nan, dtype=np.float32)
        for row in range(61, 80):
            depth[row, :] = 100.0 / (row - 40.0)
        intrinsics = np.asarray(
            [[100.0, 0.0, 50.0], [0.0, 100.0, 40.0], [0.0, 0.0, 1.0]]
        )

        status = api._straight_path_status(
            depth,
            intrinsics,
            bearing_rad=0.0,
            travel_distance_m=4.0,
        )

        self.assertTrue(status["valid"])
        self.assertTrue(status["clear"])
        self.assertGreater(status["ground_filtered_pixels"], 0)
        self.assertEqual(status["height_candidate_pixels"], 0)

    def test_temporal_median_removes_one_frame_depth_noise(self) -> None:
        env = FakeDockingEnv(target_distance_m=2.0)
        api, _ = make_api(env)
        depth = np.full((80, 100), 3.0, dtype=np.float32)
        depth[20:45, 30:36] = 0.30
        intrinsics = np.asarray(
            [[100.0, 0.0, 50.0], [0.0, 100.0, 40.0], [0.0, 0.0, 1.0]]
        )

        status = api._stable_path_status(
            depth,
            intrinsics,
            bearing_rad=0.0,
            travel_distance_m=0.40,
        )

        self.assertTrue(status["clear"])
        self.assertEqual(status["temporal_frames"], 3)
        self.assertEqual(env.next_observations, 2)

    def test_debug_callback_receives_live_closed_loop_updates(self) -> None:
        env = FakeDockingEnv(target_distance_m=0.92)
        api, _ = make_api(env)
        updates = []
        api._docking_debug_callback = updates.append

        result = api.dock_to_visible_object("table")

        self.assertTrue(result["success"])
        phases = [update["phase"] for update in updates]
        self.assertIn("path_checked", phases)
        self.assertEqual(phases[-1], "docked")
        self.assertIs(api.last_docking_debug, updates[-1])

    def test_target_loss_returns_failure_and_final_stop(self) -> None:
        env = FakeDockingEnv(target_visible=False)
        api, motion = make_api(env)

        result = api.dock_to_visible_object("table")

        self.assertFalse(result["success"])
        self.assertIn("SAM3 found no instance", result["reason"])
        self.assertEqual(motion.drives, [])
        self.assertEqual(motion.stops, 2)

    def test_out_of_range_target_reports_depth_counts_and_distribution(self) -> None:
        env = FakeDockingEnv(target_distance_m=20.5)
        api, motion = make_api(env)

        result = api.dock_to_visible_object("right orange sofa")

        self.assertFalse(result["success"])
        self.assertIn("in_range=0", result["reason"])
        self.assertIn("observed_p05_p50_p95_m", result["reason"])
        diagnostic = api.last_target_detection_debug
        self.assertIsNotNone(diagnostic)
        self.assertEqual(diagnostic["in_range_depth_pixels"], 0)
        self.assertEqual(diagnostic["configured_depth_range_m"], [0.1, 20.0])
        self.assertEqual(motion.drives, [])
        self.assertEqual(motion.stops, 2)

    def test_keyboard_interrupt_runs_final_stop_before_propagating(self) -> None:
        env = FakeDockingEnv()
        api, motion = make_api(env)

        def interrupt(_rgb, *, text_prompt):
            del text_prompt
            raise KeyboardInterrupt

        api._segment = interrupt
        with self.assertRaises(KeyboardInterrupt):
            api.dock_to_visible_object("table")

        self.assertEqual(motion.drives, [])
        self.assertEqual(motion.stops, 2)

    def test_viser_point_cloud_uses_forward_left_up_axes(self) -> None:
        depth = np.full((2, 2), 2.0, dtype=np.float32)
        rgb = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
        intrinsics = np.asarray(
            [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]]
        )
        mask = np.zeros((2, 2), dtype=bool)
        mask[1, 1] = True

        points, colors = _point_cloud_in_navigation_frame(
            depth, rgb, intrinsics, mask=mask
        )

        np.testing.assert_allclose(points, [[2.0, -1.0, -1.0]])
        np.testing.assert_array_equal(colors, [rgb[1, 1]])

    def test_viser_calibrated_point_cloud_places_floor_at_zero_height(self) -> None:
        depth = np.full((81, 101), np.nan, dtype=np.float32)
        rgb = np.zeros((81, 101, 3), dtype=np.uint8)
        depth[60, 50] = 5.0
        intrinsics = np.asarray(
            [[100.0, 0.0, 50.0], [0.0, 100.0, 40.0], [0.0, 0.0, 1.0]]
        )

        points, _ = _point_cloud_in_navigation_frame(
            depth,
            rgb,
            intrinsics,
            ground_camera_height_m=1.0,
            ground_down_camera_xyz=np.asarray([0.0, 1.0, 0.0]),
        )

        np.testing.assert_allclose(points[0], [5.0, 0.0, 0.0], atol=1e-6)

    def test_actual_world_trajectory_uses_current_zed_navigation_frame(self) -> None:
        local = _world_xy_in_navigation_frame(
            np.asarray([[1.0, 2.0], [1.0, 3.0], [0.5, 3.0]]),
            np.asarray([1.0, 2.0, math.pi / 2.0]),
        )

        np.testing.assert_allclose(
            local,
            np.asarray([[0.0, 0.0], [1.0, 0.0], [1.0, 0.5]]),
            atol=1e-9,
        )

    def test_viser_docking_envelope_inflates_planar_bbox(self) -> None:
        boundary = _rounded_bbox_boundary(1.0, 2.0, -0.5, 0.5, 0.5)

        self.assertEqual(boundary.shape, (48, 2))
        np.testing.assert_allclose(boundary[0], [2.5, 0.5])
        self.assertAlmostEqual(float(np.max(boundary[:, 0])), 2.5)
        self.assertAlmostEqual(float(np.min(boundary[:, 0])), 0.5)


if __name__ == "__main__":
    unittest.main()
