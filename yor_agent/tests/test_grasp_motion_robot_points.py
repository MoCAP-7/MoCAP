"""The cuRobo service must drop the arm's own depth returns from the scene."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np

GRASP_MOTION_DIRECTORY = Path(__file__).resolve().parents[1] / "services" / "grasp_motion"
if str(GRASP_MOTION_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(GRASP_MOTION_DIRECTORY))

import robot_points  # noqa: E402  (curobo-free helper module)


class RobotPointRemovalTest(unittest.TestCase):
    # A finger sphere from the deployed shell model: 5.7 mm radius.
    SHELL = np.array([[0.30, -0.20, 0.60, 0.0057]])

    def test_a_return_on_the_shell_surface_is_removed(self) -> None:
        # Depth noise puts the arm's own return 3 mm outside the modelled
        # surface. The old 12 mm erosion left a 2 mm sphere and kept it.
        on_surface = self.SHELL[0, :3] + np.array([0.0057 + 0.003, 0.0, 0.0])
        far_away = self.SHELL[0, :3] + np.array([0.05, 0.0, 0.0])
        points = np.vstack([on_surface, far_away])

        kept, removed = robot_points.remove_points_near_spheres(points, self.SHELL)

        self.assertEqual(removed, 1)
        np.testing.assert_allclose(kept, [far_away], atol=1e-6)

    def test_the_old_erosion_rule_would_have_kept_it(self) -> None:
        # Documents the failure the margin replaces: strictly inside a sphere
        # eroded to the 2 mm floor is a test almost no real return passes.
        on_surface = self.SHELL[0, :3] + np.array([0.0057 + 0.003, 0.0, 0.0])
        eroded_radius = max(0.002, 0.0057 - 0.012)

        inside = np.sum((on_surface - self.SHELL[0, :3]) ** 2) <= eroded_radius**2

        self.assertFalse(inside)

    def test_margin_is_the_only_knob(self) -> None:
        point = self.SHELL[0, :3] + np.array([0.0057 + 0.010, 0.0, 0.0])

        _, removed_default = robot_points.remove_points_near_spheres(point, self.SHELL)
        _, removed_tight = robot_points.remove_points_near_spheres(
            point, self.SHELL, margin_m=0.005
        )

        self.assertEqual(removed_default, 1)
        self.assertEqual(removed_tight, 0)

    def test_empty_inputs_are_safe(self) -> None:
        kept, removed = robot_points.remove_points_near_spheres(
            np.zeros((0, 3)), self.SHELL
        )
        self.assertEqual((len(kept), removed), (0, 0))
        kept, removed = robot_points.remove_points_near_spheres(
            np.ones((3, 3)), np.zeros((0, 4))
        )
        self.assertEqual((len(kept), removed), (3, 0))


if __name__ == "__main__":
    unittest.main()
