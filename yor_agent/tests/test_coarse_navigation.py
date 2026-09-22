"""The CaP-X-style coarse navigation primitives move the base in fixed steps."""

from __future__ import annotations

import contextlib
import io
import math
import unittest

from fakes import FakeHardware, make_environment

from yor_agent.exceptions import PrimitiveFailed
from yor_agent.primitives.coarse_navigation import (
    COARSE_NAVIGATION_PRIMITIVES,
    GO_FORWARD_DISTANCE_M,
    register_coarse_navigation_primitives,
)
from yor_agent.primitives.registry import PrimitiveRegistry


def build(hardware: FakeHardware | None = None, **navigation):
    environment = make_environment(hardware, **navigation)
    registry = PrimitiveRegistry()
    register_coarse_navigation_primitives(registry, environment)
    return environment, registry


class CoarseNavigationPrimitiveTest(unittest.TestCase):
    def setUp(self) -> None:
        self.environment, self.registry = build()
        self.addCleanup(self.environment.safe_shutdown)
        self.functions = self.registry.functions()
        self.pose = self.environment.hardware.pose

    def test_registers_the_coarse_vocabulary_only(self) -> None:
        self.assertEqual(sorted(self.registry.names()), sorted(COARSE_NAVIGATION_PRIMITIVES))
        docs = self.registry.documentation()
        self.assertIn("def go_forward(", docs)
        self.assertIn("exactly 1 meter", docs)
        self.assertIn("def goto_planar_position(forward_m: float, left_m: float, *,", docs)
        self.assertIn("no side-facing", docs)

    def test_go_forward_drives_one_meter(self) -> None:
        result = self.functions["go_forward"]()

        self.assertTrue(result["success"])
        self.assertAlmostEqual(self.pose.x_m, GO_FORWARD_DISTANCE_M, delta=0.05)
        self.assertAlmostEqual(self.pose.y_m, 0.0, delta=0.05)

    def test_turns_are_45_degrees_to_the_left_and_to_the_right(self) -> None:
        left = self.functions["turn_left_45_degrees"]()
        self.assertAlmostEqual(math.degrees(self.pose.yaw_rad), 45.0, delta=3.0)
        right = self.functions["turn_right_45_degrees"]()
        self.assertAlmostEqual(math.degrees(self.pose.yaw_rad), 0.0, delta=3.0)

        self.assertTrue(left["success"] and right["success"])
        self.assertEqual(left["metrics"]["requested_angle_deg"], 45.0)
        self.assertEqual(right["metrics"]["requested_angle_deg"], -45.0)

    def test_goto_planar_position_reaches_the_relative_position_and_keeps_heading(self) -> None:
        result = self.functions["goto_planar_position"](0.6, 0.3)

        self.assertTrue(result["success"])
        self.assertAlmostEqual(self.pose.x_m, 0.6, delta=0.04)
        self.assertAlmostEqual(self.pose.y_m, 0.3, delta=0.04)
        self.assertAlmostEqual(self.pose.yaw_rad, 0.0, delta=math.radians(3.0))
        self.assertEqual(
            (result["metrics"]["requested_forward_m"], result["metrics"]["requested_left_m"]),
            (0.6, 0.3),
        )

    def test_goto_planar_position_validates_before_any_motion(self) -> None:
        for args in ((-0.2, 0.0), (1.5, 1.5), (float("nan"), 0.0)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                self.functions["goto_planar_position"](*args)
        with self.assertRaises(ValueError):
            self.functions["goto_planar_position"](0.5, 0.0, max_speed_mps=0.0)

        self.assertEqual((self.pose.x_m, self.pose.y_m), (0.0, 0.0))

    def test_a_blocked_go_forward_stops_and_interrupts_the_policy(self) -> None:
        environment, registry = build(FakeHardware(clearance_m=0.2))
        self.addCleanup(environment.safe_shutdown)

        with self.assertRaises(PrimitiveFailed):
            registry.functions()["go_forward"]()

    def test_say_something_prints_the_message_without_moving(self) -> None:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            result = self.functions["say_something"]("  heading to the table ")

        self.assertEqual(result, {"success": True, "text": "heading to the table"})
        self.assertIn("[say] heading to the table", out.getvalue())
        self.assertEqual((self.pose.x_m, self.pose.y_m), (0.0, 0.0))
        with self.assertRaises(ValueError):
            self.functions["say_something"]("   ")


if __name__ == "__main__":
    unittest.main()
