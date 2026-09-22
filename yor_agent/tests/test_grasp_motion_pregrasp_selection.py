"""Unit tests for the pre-grasp selection used by the grasp-motion planner service."""

from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np


GRASP_MOTION_DIRECTORY = Path(__file__).resolve().parents[1] / "services/grasp_motion"
if str(GRASP_MOTION_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(GRASP_MOTION_DIRECTORY))

from pregrasp_selection import select_pregrasp  # noqa: E402  (curobo-free helper module)


def planned_result() -> SimpleNamespace:
    return SimpleNamespace(success=np.array([True]))


def failed_result() -> SimpleNamespace:
    return SimpleNamespace(success=np.array([False, False]))


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class SelectPregraspTest(unittest.TestCase):
    def test_a_reached_goalset_is_used_without_single_goal_plans(self) -> None:
        singles: list[int] = []
        result = planned_result()
        selection = select_pregrasp(
            3, lambda: result, singles.append, single_goal_budget_s=60.0
        )
        self.assertIs(selection.result, result)
        self.assertEqual((selection.mode, selection.candidate_index), ("goalset", None))
        self.assertEqual(singles, [])
        self.assertEqual([attempt["outcome"] for attempt in selection.attempts], ["planned"])

    def test_single_goal_plans_follow_request_order_after_a_failed_goalset(self) -> None:
        order: list[int] = []
        results = {0: None, 1: failed_result(), 2: planned_result()}

        def plan_single_goal(index: int) -> object:
            order.append(index)
            return results[index]

        selection = select_pregrasp(
            4, failed_result, plan_single_goal, single_goal_budget_s=60.0
        )
        self.assertEqual(order, [0, 1, 2])
        self.assertTrue(selection.success)
        self.assertEqual((selection.mode, selection.candidate_index), ("single_goal", 2))
        self.assertIs(selection.result, results[2])
        self.assertEqual(
            [attempt["outcome"] for attempt in selection.attempts],
            ["no_collision_free_trajectory", "no_ik_solution", "no_collision_free_trajectory", "planned"],
        )
        self.assertEqual(
            [attempt["candidate_index"] for attempt in selection.attempts], [None, 0, 1, 2]
        )

    def test_the_budget_stops_further_single_goal_plans(self) -> None:
        clock = FakeClock()
        order: list[int] = []

        def plan_goalset() -> object:
            clock.now += 50.0
            return failed_result()

        def plan_single_goal(index: int) -> object:
            order.append(index)
            clock.now += 20.0
            return failed_result()

        selection = select_pregrasp(
            5, plan_goalset, plan_single_goal, single_goal_budget_s=75.0, clock=clock
        )
        # Candidate 1 starts at 70 s, inside the budget, and runs to 90 s.
        self.assertEqual(order, [0, 1])
        self.assertFalse(selection.success)
        self.assertEqual(selection.attempts[1]["elapsed_s"], 20.0)
        self.assertEqual(
            selection.attempts[-1],
            {
                "mode": "single_goal",
                "skipped_candidate_indices": [2, 3, 4],
                "outcome": "skipped_time_budget",
            },
        )

    def test_a_lone_candidate_is_planned_once(self) -> None:
        singles: list[int] = []
        selection = select_pregrasp(1, lambda: None, singles.append, single_goal_budget_s=60.0)
        self.assertEqual(singles, [])
        self.assertFalse(selection.success)
        self.assertIsNone(selection.result)
        self.assertEqual(
            [attempt["outcome"] for attempt in selection.attempts], ["no_ik_solution"]
        )

    def test_a_zero_budget_plans_only_the_goalset(self) -> None:
        singles: list[int] = []
        goalset = failed_result()
        selection = select_pregrasp(3, lambda: goalset, singles.append, single_goal_budget_s=0.0)
        self.assertEqual(singles, [])
        self.assertIs(selection.result, goalset)
        self.assertEqual(selection.attempts[-1]["skipped_candidate_indices"], [0, 1, 2])

    def test_invalid_arguments_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "candidate_count"):
            select_pregrasp(0, failed_result, lambda index: None, single_goal_budget_s=1.0)
        with self.assertRaisesRegex(ValueError, "single_goal_budget_s"):
            select_pregrasp(2, failed_result, lambda index: None, single_goal_budget_s=-1.0)


if __name__ == "__main__":
    unittest.main()
