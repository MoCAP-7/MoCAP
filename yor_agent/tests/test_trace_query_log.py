"""The prepare search's per-query log has to reach trace.json whole.

The IK-budget curve is replayed from it; a list collapsed to its length says
how many queries ran but not when the first base certified.
"""

from __future__ import annotations

import unittest

from yor_agent.executor import _summarize


def rows(count: int) -> list[dict]:
    return [
        {
            "order": index,
            "stage": "nominal",
            "batch_index": index // 32,
            "base_index": index % 7,
            "movement_cost_tier": index % 5,
            "movement_cost": 0.5,
            "forward_m": 0.05,
            "left_m": 0.0,
            "yaw_rad": 0.0,
            "grasp_index": index,
            "robustness_variant": "nominal",
            "ik_converged": index == 90,
        }
        for index in range(count)
    ]


class QueryLogTraceTest(unittest.TestCase):
    def test_a_long_query_log_is_kept_row_for_row(self) -> None:
        summarized = _summarize({"primitive": "prepare_for_manipulation", "pi_ik_query_log": rows(512)})

        log = summarized["pi_ik_query_log"]
        self.assertIsInstance(log, list)
        self.assertEqual(len(log), 512)
        self.assertEqual(log[90], rows(512)[90])

    def test_the_log_inside_failure_diagnostics_is_kept_too(self) -> None:
        summarized = _summarize({"reason": "budget", "diagnostics": {"pi_ik_query_log": rows(328)}})

        self.assertEqual(len(summarized["diagnostics"]["pi_ik_query_log"]), 328)

    def test_other_long_lists_still_collapse(self) -> None:
        summarized = _summarize({"pi_batches": list(range(40)), "pi_ik_query_log": rows(40)})

        self.assertEqual(summarized["pi_batches"], "<list len=40>")
        self.assertEqual(len(summarized["pi_ik_query_log"]), 40)

    def test_the_exemption_does_not_reach_lists_nested_inside_a_row(self) -> None:
        row = dict(rows(1)[0], seeds=list(range(50)))
        summarized = _summarize({"pi_ik_query_log": [row]})

        self.assertEqual(summarized["pi_ik_query_log"][0]["seeds"], "<list len=50>")


if __name__ == "__main__":
    unittest.main()
