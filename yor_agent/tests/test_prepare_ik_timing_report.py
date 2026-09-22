"""The IK-budget curve has to count every search that certified a base.

A prepare whose search certified a base pose but whose motion was then refused
spent its Pi queries exactly like a successful one. Dropping it would bias the
curve towards scenes where the robot happened not to need to move.
"""

from __future__ import annotations

import unittest

from tools.prepare_ik_timing_report import first_certified, summarise


def query(order: int, *, base: int, tier: int, converged: bool, batch: int = 0) -> dict:
    return {
        "order": order,
        "stage": "nominal",
        "batch_index": batch,
        "base_index": base,
        "movement_cost_tier": tier,
        "movement_cost": float(tier),
        "forward_m": 0.0,
        "left_m": 0.0,
        "yaw_rad": 0.0,
        "grasp_index": order,
        "robustness_variant": "nominal",
        "ik_converged": converged,
    }


LOG = [
    query(0, base=0, tier=0, converged=False),
    query(1, base=0, tier=0, converged=False),
    query(2, base=5, tier=2, converged=False, batch=1),
    query(3, base=9, tier=5, converged=True, batch=1),
]
BATCHES = [
    {"stage": "nominal", "candidate_count": 2, "rpc_elapsed_s": 1.0},
    {"stage": "nominal", "candidate_count": 2, "rpc_elapsed_s": 1.5},
]


class FirstCertifiedTest(unittest.TestCase):
    def test_counts_queries_up_to_the_first_base_with_enough_converged_grasps(self) -> None:
        self.assertEqual(first_certified(LOG, 1), (4, 5))

    def test_robustness_rows_never_certify(self) -> None:
        log = [dict(LOG[3], robustness_variant="forward_plus")]
        self.assertEqual(first_certified(log, 1), (None, None))


class SummariseTest(unittest.TestCase):
    def test_a_refused_motion_keeps_its_search_in_the_curve(self) -> None:
        result = {
            "primitive": "prepare_for_manipulation",
            "success": False,
            "reason": "RuntimeError:single-stage SE(2) motion failed: obstacle_too_close",
            "stage_timings_s": {"virtual_candidate_evaluation": 9.7, "motion": 0.3},
            "motion_executed": True,
            "pi_ik_query_log": LOG,
            "pi_batches": BATCHES,
            "minimum_feasible_grasps": 1,
        }

        row = summarise("run", result)

        self.assertFalse(row["ok"])
        self.assertEqual(row["q_first"], 4)
        self.assertEqual(row["tier"], 5)
        self.assertAlmostEqual(row["s_first"], 2.5)

    def test_a_budget_failure_still_reads_its_diagnostics(self) -> None:
        result = {
            "primitive": "prepare_for_manipulation",
            "success": False,
            "reason": "_CandidatePlanningError:Pi IK compute budget ended",
            "diagnostics": {
                "pi_ik_query_log": LOG[:3],
                "pi_batches": BATCHES,
                "pi_ik_budget_exhausted": True,
            },
        }

        row = summarise("run", result)

        self.assertIsNone(row["q_first"])
        self.assertEqual(row["exit"], "budget")

    def test_a_trace_whose_log_was_collapsed_to_its_length_leaves_the_column_blank(self) -> None:
        result = {
            "primitive": "prepare_for_manipulation",
            "success": True,
            "reason": "grasp_execution_ready",
            "pi_ik_query_log": "<list len=116>",
        }

        row = summarise("run", result)

        self.assertIsNone(row["q_first"])


if __name__ == "__main__":
    unittest.main()
