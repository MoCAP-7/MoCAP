from __future__ import annotations

import math
import unittest

import numpy as np

from yor_agent.exceptions import PrimitiveFailed
from yor_agent.primitives.registry import PrimitiveRegistry
from yor_agent.primitives.visible_object_navigation import (
    register_visible_object_navigation_primitives,
)
from yor_agent.robot.local_occupancy_planner import OccupancyPlan
from yor_agent.robot.visible_object_navigation import (
    VisibleObjectDockingConfig,
    VisibleObjectDockingController,
)


class FakeDockingEnv:
    """Duck-typed stand-in: only ``observe()``, ``manipulation_config``, and
    ``controller`` are used by :class:`VisibleObjectDockingController`."""

    def __init__(
        self,
        *,
        target_distance_m: float = 0.8,
        target_column: int = 50,
        obstacle_distance_m: float | None = None,
        target_visible: bool = True,
    ) -> None:
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
        self.controller = FakeMotion(self)
        self.last_prompt: str | None = None

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
        return {"robot0_robotview": {"images": {"rgb": rgb, "depth": depth[..., None]}}}

    def segment(self, _rgb, *, text_prompt):
        self.last_prompt = text_prompt
        if not self.target_visible:
            return []
        return [{"mask": self.current_mask.copy(), "score": 0.9}]


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

    def turn_relative(self, angle_rad, *, max_yaw_rad_s=None, timeout_s=None):
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


class _ScriptedDetector:
    def __init__(self, boxes):
        self.boxes = list(boxes)
        self.calls = 0

    def __call__(self, _rgb, *, text_prompt):
        self.calls += 1
        box = self.boxes.pop(0) if self.boxes else None
        if box is None:
            return []
        return [{"box_xyxy": box, "score": 0.9, "label": text_prompt}]


def make_controller(
    env: FakeDockingEnv,
    *,
    settings=None,
    detector=None,
) -> VisibleObjectDockingController:
    docking_config = None
    if settings is not None:
        docking_config = dict(
            env.manipulation_config.get("visible_object_docking", {})
        )
        docking_config.update(settings)
    return VisibleObjectDockingController(
        env,
        docking_config=docking_config,
        segment_client_factory=lambda: env.segment,
        detector_factory=(None if detector is None else lambda _config: detector),
    )


def make_registry(env: FakeDockingEnv):
    registry = PrimitiveRegistry()
    register_visible_object_navigation_primitives(
        registry, env, segment_client_factory=lambda: env.segment
    )
    return registry


class VisibleObjectDockingConfigTest(unittest.TestCase):
    def test_default_threshold_uses_camera_at_robot_front(self) -> None:
        config = VisibleObjectDockingConfig.from_mapping(None)

        self.assertEqual(config.docking_distance_m, 0.7)
        self.assertAlmostEqual(config.maximum_turn_step_rad, np.deg2rad(25.0))
        self.assertEqual(config.turn_speed_rad_s, 0.35)
        self.assertEqual(config.forward_speed_mps, 0.18)
        self.assertEqual(config.corridor_half_width_m, 0.32)
        self.assertTrue(config.avoidance_enabled)
        self.assertEqual(config.occupancy_resolution_m, 0.05)
        self.assertEqual(config.robot_radius_m, 0.30)
        self.assertEqual(config.obstacle_clearance_margin_m, 0.0)
        self.assertEqual(config.effective_footprint_radius_m, 0.30)
        self.assertEqual(config.planner_lookahead_m, 0.8)
        self.assertEqual(config.avoidance_lateral_speed_mps, 0.10)
        self.assertEqual(config.reacquisition_detector_model, "yoloe-26s-seg.pt")
        self.assertEqual(config.reacquisition_detector_image_size, 640)
        self.assertEqual(
            config.reacquisition_search_yaw_offsets_rad[:2],
            (np.deg2rad(15.0), np.deg2rad(30.0)),
        )
        self.assertEqual(config.maximum_iterations, 100)
        self.assertAlmostEqual(config.ground_camera_height_m, 1.122339129447937)

    def test_human_friendly_motion_aliases_convert_to_internal_units(self) -> None:
        config = VisibleObjectDockingConfig.from_mapping(
            {
                "docking_distance_m": 0.7,
                "forward_step_m": 0.4,
                "forward_speed_mps": 0.12,
                "turn_step_deg": 20.0,
                "turn_speed_deg_s": 15.0,
                "bearing_tolerance_deg": 8.0,
                "maximum_total_turn_deg": 80.0,
            }
        )

        self.assertEqual(config.docking_distance_m, 0.7)
        self.assertEqual(config.maximum_forward_step_m, 0.4)
        self.assertEqual(config.forward_speed_mps, 0.12)
        self.assertAlmostEqual(config.maximum_turn_step_rad, np.deg2rad(20.0))
        self.assertAlmostEqual(config.turn_speed_rad_s, np.deg2rad(15.0))
        self.assertAlmostEqual(config.bearing_tolerance_rad, np.deg2rad(8.0))
        self.assertAlmostEqual(config.maximum_total_turn_rad, np.deg2rad(80.0))

    def test_alias_and_legacy_name_cannot_both_be_configured(self) -> None:
        with self.assertRaisesRegex(ValueError, "configure only one"):
            VisibleObjectDockingConfig.from_mapping(
                {"turn_step_deg": 20.0, "maximum_turn_step_rad": 0.4}
            )

    def test_legacy_inflation_value_becomes_the_robot_radius(self) -> None:
        config = VisibleObjectDockingConfig.from_mapping(
            {"obstacle_inflation_radius_m": 0.31}
        )

        self.assertEqual(config.robot_radius_m, 0.31)
        self.assertEqual(config.obstacle_clearance_margin_m, 0.0)
        self.assertEqual(config.effective_footprint_radius_m, 0.31)

    def test_legacy_and_explicit_robot_radius_cannot_be_mixed(self) -> None:
        with self.assertRaisesRegex(ValueError, "legacy"):
            VisibleObjectDockingConfig.from_mapping(
                {
                    "obstacle_inflation_radius_m": 0.30,
                    "robot_radius_m": 0.30,
                }
            )


class VisibleObjectDockingPrimitiveTest(unittest.TestCase):
    def test_registers_a_single_primitive_with_docstring(self) -> None:
        registry = make_registry(FakeDockingEnv())

        self.assertEqual(registry.names(), ["dock_to_visible_object"])
        self.assertIn("caller chooses", registry.documentation())

    def test_documentation_asks_for_a_short_object_name(self) -> None:
        # A prompt that described the object's colour and where it sat made
        # SAM3 return no instance, while the bare category noun docked.
        documentation = make_registry(FakeDockingEnv()).documentation()

        self.assertIn("short category noun", documentation)
        self.assertIn("retry", documentation)
        self.assertIn("with a shorter, more generic name", documentation)

    def test_argument_validation_happens_before_any_motion(self) -> None:
        env = FakeDockingEnv()
        registry = make_registry(env)

        with self.assertRaises(ValueError):
            registry.functions()["dock_to_visible_object"]("   ")

        self.assertEqual(env.controller.drives, [])
        self.assertEqual(env.controller.turns, [])

    def test_already_docked_stops_without_nonzero_motion(self) -> None:
        env = FakeDockingEnv(target_distance_m=0.68)
        registry = make_registry(env)

        result = registry.functions()["dock_to_visible_object"]("table")

        self.assertTrue(result["success"])
        self.assertEqual(result["reason"], "within_docking_distance")
        self.assertEqual(env.controller.turns, [])
        self.assertEqual(env.controller.drives, [])
        self.assertEqual(env.controller.stops, 2)

    def test_resegments_and_advances_in_bounded_steps(self) -> None:
        env = FakeDockingEnv(target_distance_m=1.80)
        registry = make_registry(env)

        result = registry.functions()["dock_to_visible_object"]("table")

        self.assertTrue(result["success"])
        self.assertGreaterEqual(len(env.controller.drives), 2)
        distances = env.controller.drives
        self.assertTrue(all(step <= 0.80 for step in distances))
        self.assertTrue(all(step > 0.0 for step in distances))
        self.assertTrue(any(step > 0.50 for step in distances))
        self.assertEqual(env.controller.planar_moves, [])
        self.assertGreater(result["metrics"]["avoidance_replans"], 0)
        self.assertEqual(env.last_prompt, "table")

    def test_target_left_produces_positive_bounded_turn(self) -> None:
        env = FakeDockingEnv(target_distance_m=1.20, target_column=20)
        controller = make_controller(env)

        result = controller.dock_to_visible_object("table")

        self.assertTrue(result["success"])
        self.assertGreater(env.controller.turns[0], 0.0)
        self.assertLessEqual(
            env.controller.turns[0], controller.config.maximum_turn_step_rad
        )
        self.assertEqual(env.controller.planar_moves, [])

    def test_target_right_produces_negative_bounded_turn(self) -> None:
        env = FakeDockingEnv(target_distance_m=1.20, target_column=80)
        controller = make_controller(env)

        result = controller.dock_to_visible_object("table")

        self.assertTrue(result["success"])
        self.assertLess(env.controller.turns[0], 0.0)
        self.assertEqual(env.controller.planar_moves, [])

    def test_coherent_obstacle_blocks_path_and_raises_primitive_failed(self) -> None:
        env = FakeDockingEnv(target_distance_m=1.20, obstacle_distance_m=0.30)
        registry = make_registry(env)

        with self.assertRaises(PrimitiveFailed) as caught:
            registry.functions()["dock_to_visible_object"]("table")

        self.assertEqual(caught.exception.reason, "path_blocked")
        self.assertEqual(env.controller.drives, [])
        self.assertEqual(env.controller.stops, 2)

    def test_static_obstacle_activates_astar_detour_without_api_change(self) -> None:
        env = FakeDockingEnv(target_distance_m=2.0)
        controller = make_controller(env)
        controller._detector = _ScriptedDetector([(45.0, 25.0, 55.0, 60.0)])
        controller._target_world_xy = lambda _target, _pose: (2.0, 0.0)
        original_segment = env.segment
        segment_calls = 0

        def intermittently_hidden(rgb, *, text_prompt):
            nonlocal segment_calls
            segment_calls += 1
            if segment_calls == 2:
                return []
            return original_segment(rgb, text_prompt=text_prompt)

        controller._segment = intermittently_hidden
        original_status = controller._stable_path_status
        status_calls = 0

        def scripted_status(*args, **kwargs):
            nonlocal status_calls
            status_calls += 1
            status = original_status(*args, **kwargs)
            if status_calls == 1:
                status.update(
                    {
                        "valid": True,
                        "clear": False,
                        "reason": "path_blocked",
                        "nearest_obstacle_m": 0.8,
                    }
                )
            else:
                status.update(
                    {
                        "valid": True,
                        "clear": True,
                        "reason": "path_clear",
                        "nearest_obstacle_m": None,
                    }
                )
            return status

        controller._stable_path_status = scripted_status
        original_plan = controller._plan_avoidance
        map_seeded = False

        def seeded_plan(*, pose_xy_yaw, target_world_xy):
            nonlocal map_seeded
            if not map_seeded:
                occupancy = controller._occupancy_map
                for x_index in range(23):
                    for y_index in range(-10, 11):
                        occupancy.integrate_rays(
                            (0.0, 0.0),
                            [(x_index * 0.1, y_index * 0.1)],
                            [False],
                        )
                for y_index in range(-3, 4):
                    occupancy.integrate_rays(
                        (0.0, 0.0), [(0.8, y_index * 0.1)], [True]
                    )
                map_seeded = True
            return original_plan(
                pose_xy_yaw=pose_xy_yaw,
                target_world_xy=target_world_xy,
            )

        controller._plan_avoidance = seeded_plan

        result = controller.dock_to_visible_object("table")

        self.assertTrue(result["success"])
        self.assertTrue(result["metrics"]["avoidance_active"])
        self.assertGreater(result["metrics"]["avoidance_replans"], 0)
        self.assertGreater(
            result["metrics"]["target_temporarily_lost_iterations"], 0
        )
        self.assertGreater(len(env.controller.turns), 0)
        self.assertGreater(len(env.controller.drives), 0)
        self.assertEqual(env.controller.planar_moves, [])
        drive_updates = [
            update
            for update in result["motion_history"]
            if update["primitive"] == "drive_straight"
        ]
        self.assertGreater(len(drive_updates), 0)
        self.assertEqual(
            result["metrics"]["avoidance_execution_mode"],
            "turn_then_straight",
        )
        self.assertEqual(result["metrics"]["total_lateral_command_m"], 0.0)

    def test_sam_target_loss_runs_stopped_yoloe_yaw_search(self) -> None:
        env = FakeDockingEnv(target_distance_m=1.20)
        detector = _ScriptedDetector(
            [None, None, (45.0, 25.0, 55.0, 60.0)]
        )
        controller = make_controller(
            env,
            settings={
                "reacquisition_search_yaw_offsets_deg": [10.0, -10.0],
            },
            detector=detector,
        )
        original_segment = env.segment
        segment_calls = 0

        def lose_once(rgb, *, text_prompt):
            nonlocal segment_calls
            segment_calls += 1
            if segment_calls == 2:
                return []
            return original_segment(rgb, text_prompt=text_prompt)

        controller._segment = lose_once

        result = controller.dock_to_visible_object("blue trash bin")

        self.assertTrue(result["success"], result)
        self.assertEqual(detector.calls, 3)
        self.assertEqual(
            result["metrics"]["last_reacquisition_reason"],
            "target_found_centered",
        )
        self.assertEqual(result["metrics"]["reacquisition_search_turns"], 2)
        self.assertEqual(
            result["metrics"]["target_temporarily_lost_iterations"], 1
        )
        self.assertTrue(any(turn > 0.0 for turn in env.controller.turns))
        self.assertTrue(any(turn < 0.0 for turn in env.controller.turns))

    def test_target_is_recentered_after_a_turn_straight_detour(self) -> None:
        env = FakeDockingEnv(target_distance_m=1.30)
        controller = make_controller(env)
        original_drive = env.controller.drive_straight
        drive_calls = 0

        def drive_then_shift_target(*args, **kwargs):
            nonlocal drive_calls
            result = original_drive(*args, **kwargs)
            drive_calls += 1
            if drive_calls == 1:
                env.target_column = 15
            return result

        env.controller.drive_straight = drive_then_shift_target

        result = controller.dock_to_visible_object("blue trash bin")

        self.assertTrue(result["success"], result)
        self.assertGreater(drive_calls, 0)
        self.assertTrue(any(turn > 0.0 for turn in env.controller.turns))

    def test_tiny_sam_mask_after_motion_uses_yoloe_reacquisition(self) -> None:
        env = FakeDockingEnv(target_distance_m=1.20)
        detector = _ScriptedDetector([(45.0, 25.0, 55.0, 60.0)])
        controller = make_controller(env, detector=detector)
        original_segment = env.segment
        segment_calls = 0

        def tiny_once(rgb, *, text_prompt):
            nonlocal segment_calls
            segment_calls += 1
            if segment_calls == 2:
                mask = np.zeros_like(env.current_mask)
                mask[40:42, 49:51] = True
                return [{"mask": mask, "score": 0.9}]
            return original_segment(rgb, text_prompt=text_prompt)

        controller._segment = tiny_once

        result = controller.dock_to_visible_object("blue trash bin")

        self.assertTrue(result["success"], result)
        self.assertEqual(detector.calls, 1)
        self.assertEqual(
            result["metrics"]["last_reacquisition_reason"],
            "target_found_centered",
        )

    def test_occluded_complete_route_uses_known_free_lateral_frontier(self) -> None:
        env = FakeDockingEnv(target_distance_m=1.0)
        controller = make_controller(env)
        controller._target_world_xy = lambda _target, _pose: (2.0, 0.0)

        def blocked_status(*args, **kwargs):
            depth = np.asarray(args[0], dtype=np.float32)
            return {
                "valid": True,
                "clear": False,
                "reason": "path_blocked",
                "nearest_obstacle_m": 0.8,
                "_stable_depth_m": depth,
            }

        controller._stable_path_status = blocked_status
        controller._plan_avoidance = lambda **_kwargs: OccupancyPlan(
            False, "no_safe_path"
        )
        def frontier_plan(**_kwargs):
            controller._occupancy_map.waypoint = (
                lambda path_xy, _lookahead: path_xy[-1]
            )
            controller._occupancy_map.segment_status = lambda *_args, **_kwargs: {
                "clear": True,
                "reason": "segment_clear",
            }
            return OccupancyPlan(
                True,
                "lateral_frontier_found",
                ((0.0, 0.0), (0.20, 0.32)),
            )

        controller._plan_avoidance_frontier = frontier_plan

        result = controller.dock_to_visible_object("table")

        self.assertTrue(result["success"])
        self.assertEqual(len(env.controller.drives), 1)
        self.assertTrue(any(turn > 0.0 for turn in env.controller.turns))
        self.assertEqual(env.controller.planar_moves, [])
        self.assertFalse(result["metrics"]["avoidance_goal_plan"]["success"])
        self.assertTrue(result["metrics"]["avoidance_frontier_plan"]["success"])
        self.assertEqual(
            result["metrics"]["avoidance_plan"]["reason"],
            "lateral_frontier_found",
        )

    def test_short_frontier_corner_never_becomes_direct_target_motion(self) -> None:
        env = FakeDockingEnv(target_distance_m=1.0)
        controller = make_controller(env)
        controller._target_world_xy = lambda _target, _pose: (4.0, 0.0)
        controller._stable_path_status = lambda depth, *_args, **_kwargs: {
            "valid": True,
            "clear": True,
            "reason": "path_clear",
            "_stable_depth_m": np.asarray(depth, dtype=np.float32),
        }
        controller._plan_avoidance = lambda **_kwargs: OccupancyPlan(
            False, "no_safe_path"
        )
        def short_frontier(**_kwargs):
            controller._occupancy_map.waypoint = (
                lambda _path_xy, _lookahead: (0.03, 0.0)
            )
            return OccupancyPlan(
                True,
                "lateral_frontier_found",
                ((0.0, 0.0), (0.03, 0.0), (0.03, 0.20)),
            )

        controller._plan_avoidance_frontier = short_frontier
        original_drive = env.controller.drive_straight

        def finish_after_short_corner(*args, **kwargs):
            result = original_drive(*args, **kwargs)
            env.target_distance_m = 0.65
            return result

        env.controller.drive_straight = finish_after_short_corner

        result = controller.dock_to_visible_object("blue trash bin")

        self.assertTrue(result["success"], result)
        self.assertEqual(env.controller.drives, [0.03])
        self.assertAlmostEqual(env.controller.drive_tolerances[0], 0.0075)
        self.assertEqual(result["metrics"]["avoidance_plan_kind"], "frontier")
        self.assertTrue(result["metrics"]["precise_grid_step"])
        self.assertEqual(
            result["metrics"]["avoidance_command_waypoint_xy_m"],
            [0.03, 0.0],
        )

    def test_short_goal_path_corner_never_becomes_direct_target_motion(self) -> None:
        env = FakeDockingEnv(target_distance_m=1.0)
        controller = make_controller(env)
        controller._target_world_xy = lambda _target, _pose: (4.0, 0.0)
        controller._stable_path_status = lambda depth, *_args, **_kwargs: {
            "valid": True,
            "clear": True,
            "reason": "path_clear",
            "_stable_depth_m": np.asarray(depth, dtype=np.float32),
        }
        def short_goal(**_kwargs):
            controller._occupancy_map.waypoint = (
                lambda _path_xy, _lookahead: (0.03, 0.0)
            )
            return OccupancyPlan(
                True,
                "path_found",
                ((0.0, 0.0), (0.03, 0.0), (0.50, 0.0)),
            )

        controller._plan_avoidance = short_goal
        original_drive = env.controller.drive_straight

        def finish_after_short_corner(*args, **kwargs):
            result = original_drive(*args, **kwargs)
            env.target_distance_m = 0.65
            return result

        env.controller.drive_straight = finish_after_short_corner

        result = controller.dock_to_visible_object("blue trash bin")

        self.assertTrue(result["success"], result)
        self.assertEqual(env.controller.drives, [0.03])
        self.assertAlmostEqual(env.controller.drive_tolerances[0], 0.0075)
        self.assertEqual(
            result["metrics"]["avoidance_plan_kind"], "docking_goal"
        )
        self.assertTrue(result["metrics"]["precise_grid_step"])

    def test_front_clearance_interruption_stops_and_replans(self) -> None:
        env = FakeDockingEnv(target_distance_m=1.0)
        controller = make_controller(env)
        controller._target_world_xy = lambda _target, _pose: (2.0, 0.0)

        def blocked_status(*args, **kwargs):
            depth = np.asarray(args[0], dtype=np.float32)
            return {
                "valid": True,
                "clear": False,
                "reason": "path_blocked",
                "nearest_obstacle_m": 0.8,
                "_stable_depth_m": depth,
            }

        def complete_plan(*, pose_xy_yaw, target_world_xy):
            del target_world_xy
            start = (float(pose_xy_yaw[0]), float(pose_xy_yaw[1]))
            endpoint = (start[0] + 0.20, start[1] + 0.32)
            controller._occupancy_map.waypoint = (
                lambda path_xy, _lookahead: path_xy[-1]
            )
            controller._occupancy_map.segment_status = lambda *_args, **_kwargs: {
                "clear": True,
                "reason": "segment_clear",
            }
            return OccupancyPlan(
                True, "path_found", (start, endpoint)
            )

        controller._stable_path_status = blocked_status
        controller._plan_avoidance = complete_plan
        original_drive = env.controller.drive_straight
        drive_attempts = 0

        def clearance_blocks_once(*args, **kwargs):
            nonlocal drive_attempts
            drive_attempts += 1
            if drive_attempts == 1:
                return {
                    "primitive": "drive_straight",
                    "success": False,
                    "reason": "obstacle_too_close",
                    "metrics": {"front_clearance_m": 0.44},
                }
            return original_drive(*args, **kwargs)

        env.controller.drive_straight = clearance_blocks_once

        result = controller.dock_to_visible_object("table")

        self.assertTrue(result["success"])
        self.assertEqual(drive_attempts, 2)
        self.assertEqual(result["metrics"]["avoidance_guard_replans"], 1)
        self.assertEqual(
            result["metrics"]["last_motion_guard_block"]["reason"],
            "obstacle_too_close",
        )

    def test_target_loss_raises_primitive_failed_after_a_final_stop(self) -> None:
        env = FakeDockingEnv(target_visible=False)
        registry = make_registry(env)

        with self.assertRaises(PrimitiveFailed) as caught:
            registry.functions()["dock_to_visible_object"]("table")

        self.assertIn("SAM3 found no instance", caught.exception.reason)
        self.assertEqual(env.controller.drives, [])
        self.assertEqual(env.controller.stops, 2)

    def test_out_of_range_target_reports_depth_diagnostics(self) -> None:
        env = FakeDockingEnv(target_distance_m=20.5)
        controller = make_controller(env)

        result = controller.dock_to_visible_object("right orange sofa")

        self.assertFalse(result["success"])
        self.assertIn("in_range=0", result["reason"])
        diagnostic = controller.last_target_detection_debug
        self.assertIsNotNone(diagnostic)
        self.assertEqual(diagnostic["in_range_depth_pixels"], 0)
        self.assertEqual(env.controller.drives, [])

    def test_keyboard_interrupt_runs_final_stop_before_propagating(self) -> None:
        env = FakeDockingEnv()
        controller = make_controller(env)

        def interrupt(_rgb, *, text_prompt):
            del text_prompt
            raise KeyboardInterrupt

        controller._segment = interrupt
        with self.assertRaises(KeyboardInterrupt):
            controller.dock_to_visible_object("table")

        self.assertEqual(env.controller.drives, [])
        self.assertEqual(env.controller.stops, 2)

    def test_last_docking_debug_reflects_the_final_phase(self) -> None:
        env = FakeDockingEnv(target_distance_m=0.72)
        controller = make_controller(env)

        result = controller.dock_to_visible_object("table")

        self.assertTrue(result["success"])
        self.assertEqual(controller.last_docking_debug["phase"], "docked")


if __name__ == "__main__":
    unittest.main()
