from __future__ import annotations

import math
import unittest

from fakes import FakeHardware, make_environment

from yor_agent.exceptions import PrimitiveFailed
from yor_agent.executor import PolicyExecutor
from yor_agent.primitives.navigation import register_navigation_primitives
from yor_agent.primitives.registry import PrimitiveRegistry


def build(hardware: FakeHardware | None = None, **navigation):
    environment = make_environment(hardware, **navigation)
    registry = PrimitiveRegistry()
    register_navigation_primitives(registry, environment)
    return environment, registry, PolicyExecutor(registry)


class NavigationPrimitiveTest(unittest.TestCase):
    def test_registers_the_v1_vocabulary_only(self) -> None:
        environment, registry, _ = build()
        self.addCleanup(environment.safe_shutdown)

        self.assertEqual(
            sorted(registry.names()),
            [
                "drive_lateral",
                "drive_straight",
                "observe",
                "stop",
                "turn_relative",
            ],
        )

    def test_documentation_states_units_and_signs(self) -> None:
        environment, registry, _ = build()
        self.addCleanup(environment.safe_shutdown)

        docs = registry.documentation()

        self.assertIn("def turn_relative(angle_deg: float, *,", docs)
        self.assertIn("Positive angles turn left", docs)
        self.assertIn("def drive_straight(distance_m: float, *,", docs)
        self.assertIn("no rear-facing clearance sensor", docs)
        self.assertIn("def drive_lateral(distance_m: float, *,", docs)
        self.assertIn("Positive distance moves robot-left", docs)
        self.assertIn("no side-facing", docs)
        self.assertIn("clearance sensor", docs)

    def test_turn_relative_converts_degrees_and_reports_degree_metrics(self) -> None:
        environment, _, _ = build()
        self.addCleanup(environment.safe_shutdown)

        outcome = _functions(environment)["turn_relative"](30.0, timeout_s=8.0)

        self.assertTrue(outcome["success"])
        self.assertAlmostEqual(outcome["metrics"]["requested_angle_deg"], 30.0)
        self.assertAlmostEqual(
            outcome["metrics"]["target_yaw_deg"], 30.0, delta=1e-6
        )
        self.assertLess(
            abs(environment._latest_frame.planar_pose.yaw_rad - math.radians(30.0)),
            math.radians(2.5),
        )

    def test_argument_validation_happens_before_any_motion(self) -> None:
        hardware = FakeHardware()
        environment, _, _ = build(hardware)
        self.addCleanup(environment.safe_shutdown)
        functions = _functions(environment)

        with self.assertRaises(ValueError):
            functions["turn_relative"](181.0)
        with self.assertRaises(ValueError):
            functions["turn_relative"](10.0, max_yaw_deg_s=-1.0)
        with self.assertRaises(TypeError):
            functions["drive_straight"](True)
        with self.assertRaises(ValueError):
            functions["drive_straight"](float("nan"))
        with self.assertRaises(ValueError):
            functions["drive_straight"](5.0)  # beyond max_distance_m
        with self.assertRaises(TypeError):
            functions["drive_lateral"](True)
        with self.assertRaises(ValueError):
            functions["drive_lateral"](float("nan"))
        with self.assertRaises(ValueError):
            functions["drive_lateral"](-5.0)

        self.assertEqual(hardware.commands, [])

    def test_drive_lateral_uses_left_positive_and_right_negative(self) -> None:
        environment, _, _ = build()
        self.addCleanup(environment.safe_shutdown)
        lateral = _functions(environment)["drive_lateral"]

        left = lateral(0.15, timeout_s=8.0)
        left_y = environment._latest_frame.planar_pose.y_m
        right = lateral(-0.15, timeout_s=8.0)

        self.assertTrue(left["success"], left)
        self.assertEqual(left["metrics"]["direction"], "left")
        self.assertFalse(left["metrics"]["lateral_clearance_checked"])
        self.assertGreater(left_y, 0.12)
        self.assertTrue(right["success"], right)
        self.assertEqual(right["metrics"]["direction"], "right")
        self.assertLess(environment._latest_frame.planar_pose.y_m, 0.03)

    def test_failed_drive_stops_and_interrupts_the_policy(self) -> None:
        hardware = FakeHardware(clearance_m=0.30)
        environment, _, executor = build(hardware)
        self.addCleanup(environment.safe_shutdown)

        record = executor.execute(
            "drive_straight(0.5, timeout_s=3.0)\n"
            "turn_relative(90.0)\n"
            "drive_straight(0.5)\n"
        )

        self.assertIsNotNone(record["interrupted_by"])
        self.assertEqual(record["interrupted_by"]["primitive"], "drive_straight")
        self.assertEqual(record["interrupted_by"]["reason"], "obstacle_too_close")
        self.assertEqual(len(record["primitive_calls"]), 1)
        self.assertTrue(
            record["interrupted_by"]["result"]["stop_after_failure"]["success"]
        )
        self.assertEqual(hardware.commands[-1], [0.0, 0.0, 0.0])
        self.assertTrue(all(command == [0.0, 0.0, 0.0] for command in hardware.commands))

    def test_drive_without_obstacle_check_ignores_the_clearance_gate(self) -> None:
        hardware = FakeHardware(clearance_m=0.30)
        environment, _, _ = build(hardware)
        self.addCleanup(environment.safe_shutdown)
        controller = environment.controller

        gated = controller.drive_straight(0.3, timeout_s=3.0)
        self.assertFalse(gated["success"])
        self.assertEqual(gated["reason"], "obstacle_too_close")

        ungated = controller.drive_straight(0.3, timeout_s=8.0, obstacle_check=False)
        self.assertTrue(ungated["success"], ungated)
        self.assertEqual(ungated["reason"], "target_reached")

        planar = controller.move_planar_relative(
            0.2, 0.0, 0.0, timeout_s=8.0, obstacle_check=False
        )
        self.assertTrue(planar["success"], planar)
        self.assertEqual(planar["reason"], "target_reached")

    def test_failed_primitive_raises_primitive_failed_directly(self) -> None:
        hardware = FakeHardware()
        hardware.frame_error = RuntimeError("stale_pose")
        environment, _, _ = build(hardware)
        self.addCleanup(environment.safe_shutdown)

        with self.assertRaises(PrimitiveFailed) as caught:
            _functions(environment)["turn_relative"](45.0, timeout_s=3.0)

        self.assertIn("stale_pose", caught.exception.reason)

    def test_observe_returns_the_unified_observation(self) -> None:
        environment, _, _ = build()
        self.addCleanup(environment.safe_shutdown)

        observation = _functions(environment)["observe"]()

        self.assertEqual(observation["robot0_robotview"]["images"]["rgb"].shape[2], 3)
        self.assertEqual(len(observation["base"]["pose_xy_yaw"]), 3)
        self.assertTrue(observation["lift"]["available"])


def _functions(environment):
    registry = PrimitiveRegistry()
    register_navigation_primitives(registry, environment)
    return registry.functions()


if __name__ == "__main__":
    unittest.main()
