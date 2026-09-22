from __future__ import annotations

import unittest

from yor_agent.robot.local_occupancy_planner import StaticOccupancyMap


def observed_rectangle(
    occupancy: StaticOccupancyMap,
    *,
    x_max_m: float = 2.2,
    y_half_width_m: float = 1.0,
) -> None:
    resolution = occupancy.resolution_m
    for x_index in range(int(round(x_max_m / resolution)) + 1):
        for y_index in range(
            -int(round(y_half_width_m / resolution)),
            int(round(y_half_width_m / resolution)) + 1,
        ):
            occupancy.integrate_rays(
                (0.0, 0.0),
                [(x_index * resolution, y_index * resolution)],
                [False],
            )


class StaticOccupancyMapTest(unittest.TestCase):
    def test_rasterization_never_rounds_robot_radius_down(self) -> None:
        occupancy = StaticOccupancyMap(
            resolution_m=0.08, inflation_radius_m=0.30
        )
        occupancy.integrate_rays((0.0, 0.0), [(0.0, 0.0)], [True])

        inflated = occupancy._inflated_cells()

        self.assertIn((4, 0), inflated)
        self.assertIn((3, 3), inflated)
        self.assertNotIn((5, 0), inflated)

    def test_astar_detours_around_inflated_static_wall(self) -> None:
        occupancy = StaticOccupancyMap(
            resolution_m=0.1, inflation_radius_m=0.2
        )
        observed_rectangle(occupancy)
        for y_index in range(-3, 4):
            occupancy.integrate_rays(
                (0.0, 0.0), [(0.8, y_index * 0.1)], [True]
            )

        plan = occupancy.plan_to_docking_ring(
            start_xy=(0.0, 0.0),
            target_xy=(2.0, 0.0),
            docking_distance_m=0.3,
            maximum_expansions=10_000,
        )

        self.assertTrue(plan.success)
        self.assertEqual(plan.reason, "path_found")
        self.assertTrue(any(abs(y) >= 0.6 for _, y in plan.path_xy))
        self.assertAlmostEqual(
            ((plan.path_xy[-1][0] - 2.0) ** 2 + plan.path_xy[-1][1] ** 2)
            ** 0.5,
            0.3,
            delta=0.15,
        )

    def test_unknown_space_is_not_plannable(self) -> None:
        occupancy = StaticOccupancyMap(
            resolution_m=0.1, inflation_radius_m=0.2
        )
        occupancy.integrate_rays((0.0, 0.0), [(0.5, 0.0)], [False])

        plan = occupancy.plan_to_docking_ring(
            start_xy=(0.0, 0.0),
            target_xy=(2.0, 0.0),
            docking_distance_m=0.3,
            maximum_expansions=10_000,
        )

        self.assertFalse(plan.success)
        self.assertEqual(plan.reason, "no_observed_docking_goal")

    def test_inflation_can_close_a_too_narrow_gap(self) -> None:
        occupancy = StaticOccupancyMap(
            resolution_m=0.1, inflation_radius_m=0.3
        )
        observed_rectangle(occupancy, y_half_width_m=0.5)
        for y_index in (-5, -4, -3, 3, 4, 5):
            occupancy.integrate_rays(
                (0.0, 0.0), [(0.8, y_index * 0.1)], [True]
            )

        plan = occupancy.plan_to_docking_ring(
            start_xy=(0.0, 0.0),
            target_xy=(2.0, 0.0),
            docking_distance_m=0.3,
            maximum_expansions=10_000,
        )

        self.assertFalse(plan.success)
        self.assertEqual(plan.reason, "no_safe_path")

    def test_lateral_frontier_repositions_without_entering_occluded_space(self) -> None:
        occupancy = StaticOccupancyMap(
            resolution_m=0.1, inflation_radius_m=0.2
        )
        # The near-side floor is observed, while an occlusion leaves it
        # disconnected from a separately observed docking region.
        for x_index in range(7):
            for y_index in range(-10, 11):
                occupancy.free_cells.add((x_index, y_index))
        for x_index in range(14, 23):
            for y_index in range(-10, 11):
                occupancy.free_cells.add((x_index, y_index))

        complete = occupancy.plan_to_docking_ring(
            start_xy=(0.0, 0.0),
            target_xy=(2.0, 0.0),
            docking_distance_m=0.3,
            maximum_expansions=10_000,
        )
        frontier = occupancy.plan_to_lateral_frontier(
            start_xy=(0.0, 0.0),
            target_xy=(2.0, 0.0),
            minimum_lateral_m=0.4,
            minimum_target_progress_m=0.1,
            maximum_expansions=10_000,
        )

        self.assertFalse(complete.success)
        self.assertEqual(complete.reason, "no_safe_path")
        self.assertTrue(frontier.success)
        self.assertEqual(frontier.reason, "lateral_frontier_found")
        self.assertGreaterEqual(abs(frontier.path_xy[-1][1]), 0.4)
        self.assertLess(frontier.path_xy[-1][0], 0.7)
        for start, end in zip(frontier.path_xy, frontier.path_xy[1:]):
            self.assertTrue(occupancy.segment_status(start, end)["clear"])

    def test_lateral_frontier_respects_a_preferred_detour_side(self) -> None:
        occupancy = StaticOccupancyMap(
            resolution_m=0.1, inflation_radius_m=0.2
        )
        observed_rectangle(occupancy, x_max_m=0.8, y_half_width_m=1.0)

        plan = occupancy.plan_to_lateral_frontier(
            start_xy=(0.0, 0.0),
            target_xy=(2.0, 0.0),
            minimum_lateral_m=0.4,
            minimum_target_progress_m=0.1,
            maximum_expansions=10_000,
            preferred_side=1,
        )

        self.assertTrue(plan.success)
        self.assertGreaterEqual(plan.path_xy[-1][1], 0.4)

    def test_waypoint_never_cuts_through_inflated_obstacle(self) -> None:
        occupancy = StaticOccupancyMap(
            resolution_m=0.1, inflation_radius_m=0.2
        )
        observed_rectangle(occupancy)
        for y_index in range(-3, 4):
            occupancy.integrate_rays(
                (0.0, 0.0), [(0.8, y_index * 0.1)], [True]
            )
        plan = occupancy.plan_to_docking_ring(
            start_xy=(0.0, 0.0),
            target_xy=(2.0, 0.0),
            docking_distance_m=0.3,
            maximum_expansions=10_000,
        )

        waypoint = occupancy.waypoint(plan.path_xy, 0.5)

        self.assertLessEqual(
            (waypoint[0] ** 2 + waypoint[1] ** 2) ** 0.5, 0.5 + 1e-9
        )
        self.assertLess(waypoint[1], 0.0)

    def test_segment_status_rejects_obstacle_and_unknown_space(self) -> None:
        occupancy = StaticOccupancyMap(
            resolution_m=0.1, inflation_radius_m=0.2
        )
        observed_rectangle(occupancy)
        occupancy.integrate_rays((0.0, 0.0), [(0.8, 0.0)], [True])

        blocked = occupancy.segment_status((0.0, 0.0), (1.2, 0.0))
        clear = occupancy.segment_status((0.0, 0.0), (0.5, 0.5))
        unknown = occupancy.segment_status((0.0, 0.0), (0.0, 1.5))

        self.assertFalse(blocked["clear"])
        self.assertEqual(blocked["reason"], "inflated_obstacle")
        self.assertTrue(clear["clear"])
        self.assertFalse(unknown["clear"])
        self.assertEqual(unknown["reason"], "unknown_space")

    def test_segment_can_leave_only_its_current_inflated_cell(self) -> None:
        occupancy = StaticOccupancyMap(
            resolution_m=0.1, inflation_radius_m=0.2
        )
        observed_rectangle(occupancy, x_max_m=1.0, y_half_width_m=0.5)
        occupancy.integrate_rays((0.0, 0.0), [(0.5, 0.0)], [True])

        strict = occupancy.segment_status((0.3, 0.0), (0.0, 0.0))
        egress = occupancy.segment_status(
            (0.3, 0.0),
            (0.0, 0.0),
            allow_start_in_inflated=True,
        )
        deeper = occupancy.segment_status(
            (0.3, 0.0),
            (0.6, 0.0),
            allow_start_in_inflated=True,
        )

        self.assertFalse(strict["clear"])
        self.assertTrue(egress["clear"])
        self.assertEqual(egress["reason"], "segment_clear_from_inflated_start")
        self.assertTrue(egress["allowed_inflated_start"])
        self.assertFalse(deeper["clear"])
        self.assertGreater(deeper["blocked_cells"], 0)


if __name__ == "__main__":
    unittest.main()
