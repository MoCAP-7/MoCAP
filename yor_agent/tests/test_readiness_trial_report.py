"""The trial report reads each bearing and the cost back from trace.json alone."""

from __future__ import annotations

import contextlib
import csv
import io
import json
import math
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from tools.readiness_trial_report import (
    anchor_bearing_deg,
    bearing_groups,
    bearing_outcome,
    condition,
    configured_prior_enabled,
    goal_distance_used_m,
    main,
    nav2_goal_count,
    queries_until_base,
    solving_cost_queries,
    summarise,
    summary_rows,
    wrap_degrees,
)

ANCHOR_RAD = 0.5
ANCHOR_DEG = math.degrees(ANCHOR_RAD)
LLM_CONFIG = {
    "primitive_config": {
        "primitives": {
            "dock_to_visible_object": {"settings": {"readiness_prior": {"enabled": True}}}
        }
    },
    "model": {"provider": "deepseek", "name": "deepseek-v4-flash-vision-exp"},
    "agent": {"max_turns": 100, "time_limit_s": 120.0},
}


def query(order: int, *, base: int, converged: bool) -> dict:
    return {
        "order": order,
        "stage": "nominal",
        "batch_index": 0,
        "base_index": base,
        "movement_cost_tier": 1,
        "movement_cost": 1.0,
        "robustness_variant": "nominal",
        "ik_converged": converged,
    }


# Six queries; the first base with a converged nominal grasp certifies at the fourth.
LOG = [
    query(0, base=0, converged=False),
    query(1, base=0, converged=False),
    query(2, base=3, converged=False),
    query(3, base=7, converged=True),
    query(4, base=7, converged=True),
    query(5, base=9, converged=True),
]


def goal(distance_m: float | None, reason: str = "succeeded", **extra) -> dict:
    """One ``goal_attempts`` entry: a navigated goal carries its pose, an
    attempt that never drove records ``None`` for the pose and distance."""

    pose = None if distance_m is None else [1.0, 2.0, 0.5]
    return {
        "bearing_source": "explicit",
        "bearing_rad": ANCHOR_RAD,
        "camera_goal_xy_yaw": pose,
        "nav2_goal_xy_yaw": pose,
        "goal_distance_m": distance_m,
        "plan_checks": [],
        "success": reason == "succeeded",
        "reason": reason,
        **extra,
    }


def run_config(prior_enabled: bool | None) -> dict:
    """The part of the run config the report reads the condition from."""

    settings: dict = {"backend": "nav2"}
    if prior_enabled is not None:
        settings["readiness_prior"] = {"enabled": prior_enabled, "method": "vggt"}
    return {
        "primitive_config": {
            "primitives": {"dock_to_visible_object": {"settings": settings}}
        }
    }


def dock_call(result: dict, approach: float | None = None) -> dict:
    return {
        "name": "dock_to_visible_object",
        "args": ["can"],
        "kwargs": {} if approach is None else {"approach_bearing_deg": approach},
        "result": {"primitive": "dock_to_visible_object", **result},
    }


def prepare_call(result: dict) -> dict:
    return {
        "name": "prepare_for_manipulation",
        "args": ["can"],
        "kwargs": {"arm": "left"},
        "result": {"primitive": "prepare_for_manipulation", **result},
    }


def retreat_call(result: dict, distance_m: float = -0.3) -> dict:
    """The trial's retreat between two bearings, as the trace records it."""

    return {
        "name": "drive_straight",
        "args": [distance_m],
        "kwargs": {},
        "result": {"primitive": "drive_straight", **result},
    }


RETREATED = {"success": True, "reason": "reached", "progress_m": -0.3}
RETREAT_REFUSED = {"success": False, "reason": "obstacle_too_close", "progress_m": -0.02}


UNPLANNABLE = {
    "success": False,
    "reason": "no_plannable_goal_on_bearing",
    "metrics": {
        "readiness_prior": {"used": True, "reason": "prior_bearing"},
        "reference_bearing_rad": ANCHOR_RAD,
        "arrival_bearing_rad": 1.0,
        "goal_attempts": [goal(None, "no_plannable_goal_on_bearing")],
    },
}
DOCKED_FAR = {
    "success": True,
    "reason": "docked_at_fallback_distance",
    "suggested_arm": "left",
    "metrics": {
        "readiness_prior": {"used": False, "reason": "explicit_bearing"},
        "reference_bearing_rad": ANCHOR_RAD + math.radians(45.0),
        "goal_attempts": [goal(0.75, "aborted"), goal(0.9)],
        "goal_distance_used_m": 0.9,
    },
}
SOLVED = {
    "success": True,
    "reason": "grasp_execution_ready",
    "ik_query_count": 6,
    "minimum_feasible_grasps": 1,
    "pi_ik_query_log": LOG,
}
SEARCH_FAILED = {
    "success": False,
    "reason": "RuntimeError:no local base pose has a collision-safe, strictly Pi-converged nominal grasp",
    "diagnostics": {"pi_ik_query_log": [query(i, base=i, converged=False) for i in range(3)]},
}
MOTION_EXHAUSTED = {
    "success": False,
    "reason": "RuntimeError:no certified base reachable: 2 motions refused (obstacle_too_closex2)",
    "pi_ik_query_log": LOG,
    "motion_exhausted": True,
}
TOO_FAR = {
    "success": False,
    "reason": "target_too_far_for_local_manipulation",
    "target_distance_m": 1.31,
    "maximum_distance_m": 1.2,
}
DOCK_MISS = {
    "success": False,
    "reason": "exception:RuntimeError:SAM3 found no instance for 'can'",
    "metrics": {"backend": "nav2"},
}


def trace(
    run_id: str, calls: list[dict], finish: str | None = None, config: dict | None = None
) -> dict:
    return {
        "run_id": run_id,
        "config": {} if config is None else config,
        "primitive_calls": [{**call, "index": index, "turn": index + 1} for index, call in enumerate(calls)],
        "finish": None if finish is None else {"reason": finish},
    }


SOLVED_TRACE = trace(
    "web-solved",
    [
        dock_call(UNPLANNABLE),
        dock_call(DOCKED_FAR, approach=ANCHOR_DEG + 45.0),
        prepare_call(MOTION_EXHAUSTED),
        retreat_call(RETREATED),
        dock_call(DOCKED_FAR, approach=ANCHOR_DEG - 45.0),
        prepare_call(SOLVED),
    ],
    finish="trial solved at bearing 2 (offset -45 deg): ...",
    config=run_config(True),
)
UNSOLVED_TRACE = trace(
    "web-unsolved",
    [
        dock_call({**DOCKED_FAR, "metrics": {**DOCKED_FAR["metrics"], "readiness_prior": {"used": False, "reason": "not_configured"}, "reference_bearing_rad": ANCHOR_RAD}}),
        prepare_call(SEARCH_FAILED),
        retreat_call(RETREAT_REFUSED),
        dock_call(DOCK_MISS, approach=ANCHOR_DEG + 45.0),
    ],
    finish="trial unsolved after 2 of 2 bearings; ...",
    config=run_config(False),
)


class SummariseTest(unittest.TestCase):
    def test_a_solved_trial_reports_its_condition_bearings_and_cost(self) -> None:
        row, bearings = summarise("web-solved", SOLVED_TRACE)

        self.assertEqual(row["run"], "web-solved")
        self.assertEqual(row["condition"], "prior")
        self.assertAlmostEqual(row["anchor_deg"], round(ANCHOR_DEG, 1))
        self.assertEqual(row["dock_calls"], 3)
        # Bearing 0 sent no goal; the two fallback docks sent two each.
        self.assertEqual(row["nav2_goals"], 4)
        self.assertTrue(row["solved"])
        # The exhausted search's six queries plus the four the solving search
        # spent until its first base certified.
        self.assertEqual(row["cost_queries"], 6 + 4)
        self.assertEqual(row["total_queries"], 12)
        self.assertEqual(
            row["bearings"],
            "+0: dock_failed dock=no_plannable_goal_on_bearing prepare=- q=0; "
            "+45: prepare_failed dock=docked_at_fallback_distance@0.90 prepare=RuntimeError:no certified base reachable: 2 motions refused (obstacle_too_closex2) q=6; "
            "-45: solved dock=docked_at_fallback_distance@0.90 prepare=grasp_execution_ready q=6",
        )
        self.assertEqual([b["offset_deg"] for b in bearings], [0.0, 45.0, -45.0])
        self.assertEqual([b["outcome"] for b in bearings], ["dock_failed", "prepare_failed", "solved"])
        self.assertEqual([b["target_distance_m"] for b in bearings], [None, None, None])
        self.assertEqual([b["docks"] for b in bearings], [1, 1, 1])
        self.assertEqual([b["prepares"] for b in bearings], [0, 1, 1])
        self.assertEqual([b["nav2_goals"] for b in bearings], [0, 2, 2])
        self.assertEqual([b["goal_distance_m"] for b in bearings], [None, 0.9, 0.9])
        self.assertEqual(bearings[1]["prepare_reason"], MOTION_EXHAUSTED["reason"])
        self.assertAlmostEqual(bearings[2]["approach_deg"], ANCHOR_DEG - 45.0)
        # The retreat between the two docked bearings belongs to the one it follows.
        self.assertEqual(row["retreats"], 1)
        self.assertEqual([b["retreats"] for b in bearings], [0, 1, 0])
        self.assertEqual([b["retreat"] for b in bearings], ["", "reached", ""])

    def test_times_come_from_the_run_stamps_and_the_calls_elapsed(self) -> None:
        timed = trace(
            "web-timed",
            [
                {**dock_call(DOCKED_FAR), "elapsed_s": 30.5},
                {**prepare_call({**MOTION_EXHAUSTED, "pi_ik_compute_elapsed_s": 6.0}), "elapsed_s": 20.0},
                {**retreat_call(RETREATED), "elapsed_s": 4.0},
                {**dock_call(DOCKED_FAR, approach=ANCHOR_DEG + 45.0), "elapsed_s": 25.5},
                {**prepare_call({**SOLVED, "pi_ik_compute_elapsed_s": 4.5}), "elapsed_s": 18.0},
            ],
            finish="trial solved at bearing 1 (offset +45 deg): ...",
            config=run_config(True),
        )
        timed["created_at"] = "2026-09-14T00:07:23-0400"
        timed["finish"]["at"] = "2026-09-14T00:10:03-0400"

        row, _ = summarise("web-timed", timed)

        self.assertAlmostEqual(row["time_s"], 160.0)
        self.assertAlmostEqual(row["dock_time_s"], 56.0)
        self.assertAlmostEqual(row["prepare_time_s"], 38.0)
        self.assertAlmostEqual(row["ik_time_s"], 10.5)
        # Without a finish stamp there is no time to the solution; the call
        # times still add up.
        del timed["finish"]["at"]
        row, _ = summarise("web-timed", timed)
        self.assertIsNone(row["time_s"])
        self.assertAlmostEqual(row["dock_time_s"], 56.0)

    def test_an_llm_run_reports_its_end_times_turns_and_model_usage(self) -> None:
        started = datetime.strptime("2026-09-14T00:07:23-0400", "%Y-%m-%dT%H:%M:%S%z").timestamp()
        run = trace(
            "web-llm",
            [
                {**dock_call(DOCKED_FAR), "started_at": started + 5.0, "elapsed_s": 30.0},
                {**prepare_call(SEARCH_FAILED), "started_at": started + 36.0, "elapsed_s": 4.0},
                {**prepare_call(SOLVED), "started_at": started + 42.0, "elapsed_s": 18.0},
            ],
            finish="the robot is ready to grasp the can",
            config=LLM_CONFIG,
        )
        run["created_at"] = "2026-09-14T00:07:23-0400"
        run["finish"]["at"] = "2026-09-14T00:08:33-0400"
        run["policies"] = [{"turn": 1}, {"turn": 2}, {"turn": 3}]
        run["model_usage"] = {"n_calls": 4, "prompt_tokens": 12000, "output_tokens": 800}

        row, _ = summarise("web-llm", run)

        self.assertEqual(row["policy"], "llm")
        self.assertEqual(row["model"], "deepseek/deepseek-v4-flash-vision-exp")
        self.assertEqual(row["end"], "finished")
        self.assertEqual(row["time_limit_s"], 120.0)
        self.assertTrue(row["solved"])
        self.assertAlmostEqual(row["time_s"], 70.0)
        # The first successful prepare ended 60 s in; the policy finished later.
        self.assertAlmostEqual(row["solved_time_s"], 60.0)
        self.assertTrue(row["solved_in_time"])
        self.assertEqual(row["turns"], 3)
        self.assertEqual(
            (row["model_calls"], row["prompt_tokens"], row["output_tokens"]), (4, 12000, 800)
        )
        self.assertEqual(row["prepare_calls"], 2)
        self.assertEqual(row["explicit_bearing_docks"], 0)
        # The scripted protocol records no model usage.
        scripted = summarise(
            "web-solved", {**SOLVED_TRACE, "config": {**run_config(True), "model": {"provider": "scripted", "name": "readiness-trial"}}}
        )[0]
        self.assertEqual(scripted["policy"], "scripted")
        self.assertIsNone(scripted["model_calls"])
        self.assertEqual(scripted["explicit_bearing_docks"], 2)
        self.assertIsNone(summarise("web-solved", SOLVED_TRACE)[0]["policy"])

    def test_times_count_from_the_agent_start_when_the_trace_records_it(self) -> None:
        started = datetime.strptime("2026-09-14T00:07:23-0400", "%Y-%m-%dT%H:%M:%S%z").timestamp()
        run = trace(
            "web-slow-build",
            [dock_call(DOCKED_FAR), {**prepare_call(SOLVED), "started_at": started + 40.0, "elapsed_s": 20.0}],
            finish="ready to grasp",
            config=LLM_CONFIG,
        )
        # The runtime took 93 s to build before the agent loop started.
        run["created_at"] = "2026-09-14T00:05:50-0400"
        run["started_at"] = "2026-09-14T00:07:23-0400"
        run["finish"]["at"] = "2026-09-14T00:08:33-0400"

        row, _ = summarise("web-slow-build", run)

        self.assertAlmostEqual(row["time_s"], 70.0)
        self.assertAlmostEqual(row["solved_time_s"], 60.0)
        self.assertTrue(row["solved_in_time"])
        # A time-limit stop is timed from the agent start as well.
        stopped = {**run, "finish": None, "stop": {"reason": "time limit of 120 s reached", "at": "2026-09-14T00:09:23-0400"}}
        self.assertAlmostEqual(summarise("web-slow-build", stopped)[0]["time_s"], 120.0)
        # A trace from before the agent start was stamped counts from its creation.
        del run["started_at"]
        row, _ = summarise("web-slow-build", run)
        self.assertAlmostEqual(row["time_s"], 163.0)
        self.assertAlmostEqual(row["solved_time_s"], 153.0)

    def test_a_time_limit_stop_ends_and_times_the_run_and_an_operator_stop_does_not(self) -> None:
        run = trace(
            "web-timeout",
            [dock_call(DOCKED_FAR, approach=ANCHOR_DEG + 30.0), prepare_call(SEARCH_FAILED)],
            config=LLM_CONFIG,
        )
        run["created_at"] = "2026-09-14T00:07:23-0400"
        run["stop"] = {"reason": "time limit of 120 s reached", "at": "2026-09-14T00:09:25-0400"}

        row, _ = summarise("web-timeout", run)

        self.assertEqual(row["end"], "time_limit")
        self.assertAlmostEqual(row["time_s"], 122.0)
        self.assertFalse(row["solved"])
        self.assertIsNone(row["solved_time_s"])
        self.assertEqual(row["explicit_bearing_docks"], 1)
        self.assertEqual(row["turns"], 0)

        # A prepare already running when the limit stopped the run can still
        # succeed after it: solved, but not in time.
        started = datetime.strptime("2026-09-14T00:07:23-0400", "%Y-%m-%dT%H:%M:%S%z").timestamp()
        late = trace(
            "web-late",
            [dock_call(DOCKED_FAR), {**prepare_call(SOLVED), "started_at": started + 110.0, "elapsed_s": 18.0}],
            config=LLM_CONFIG,
        )
        late["created_at"] = "2026-09-14T00:07:23-0400"
        late["stop"] = {"reason": "time limit of 120 s reached", "at": "2026-09-14T00:09:31-0400"}
        late_row, _ = summarise("web-late", late)
        self.assertEqual(late_row["end"], "time_limit")
        self.assertTrue(late_row["solved"])
        self.assertAlmostEqual(late_row["solved_time_s"], 128.0)
        self.assertFalse(late_row["solved_in_time"])
        # Without a limit a solve is always in time.
        unlimited = {**late, "config": {**LLM_CONFIG, "agent": {"max_turns": 100}}}
        self.assertTrue(summarise("web-late", unlimited)[0]["solved_in_time"])

        run["stop"] = {"reason": "operator requested stop", "at": "2026-09-14T00:08:00-0400"}
        row, _ = summarise("web-timeout", run)
        self.assertEqual(row["end"], "stopped")
        self.assertIsNone(row["time_s"])

        run["error"] = {"type": "RuntimeError", "message": "boom"}
        self.assertEqual(summarise("web-timeout", run)[0]["end"], "error")
        budget = trace("web-budget", [], finish="turn budget exhausted after 100 turns", config=LLM_CONFIG)
        self.assertEqual(summarise("web-budget", budget)[0]["end"], "turn_budget")
        self.assertEqual(summarise("web-none", trace("web-none", []))[0]["end"], "unfinished")

    def test_the_summary_gives_success_rate_times_and_queries_per_policy_and_condition(self) -> None:
        rows = [
            {"policy": "llm", "condition": "prior", "solved": True, "time_s": 60.0, "solved_time_s": 50.0, "total_queries": 100},
            {"policy": "llm", "condition": "prior", "solved": False, "time_s": 122.0, "solved_time_s": None, "total_queries": 300},
            {"policy": "llm", "condition": "prior", "solved": True, "time_s": 80.0, "solved_time_s": 70.0, "total_queries": 200},
            {"policy": "llm", "condition": "no_prior", "solved": False, "time_s": None, "solved_time_s": None, "total_queries": 0},
            # Solved after its time limit: not a success of the summary.
            {"policy": "llm", "condition": "no_prior", "solved": True, "solved_in_time": False, "time_s": 128.0, "solved_time_s": 128.0, "total_queries": 40},
        ]

        no_prior, prior = summary_rows(rows)

        self.assertEqual(
            prior,
            {
                "policy": "llm",
                "condition": "prior",
                "trials": 3,
                "solved": 2,
                "success_rate": 2 / 3,
                "median_time_s": 80.0,
                "median_solved_time_s": 60.0,
                "mean_total_queries": 200.0,
            },
        )
        self.assertEqual((no_prior["condition"], no_prior["trials"], no_prior["solved"]), ("no_prior", 2, 0))
        self.assertEqual(no_prior["success_rate"], 0.0)
        self.assertEqual(no_prior["median_time_s"], 128.0)
        self.assertIsNone(no_prior["median_solved_time_s"])
        self.assertEqual(no_prior["mean_total_queries"], 20.0)

    def test_the_start_label_comes_from_the_run_config(self) -> None:
        labelled = trace(
            "web-labelled",
            [dock_call(DOCKED_FAR), prepare_call(SOLVED)],
            finish="trial solved at bearing 0 (offset +0 deg): ...",
            config={**run_config(True), "trial": {"start_label": "S45"}},
        )

        row, _ = summarise("web-labelled", labelled)

        self.assertEqual(row["start_label"], "S45")
        self.assertIsNone(summarise("web-solved", SOLVED_TRACE)[0]["start_label"])

    def test_an_unsolved_trial_has_no_cost_but_its_total(self) -> None:
        row, bearings = summarise("web-unsolved", UNSOLVED_TRACE)

        self.assertEqual(row["condition"], "no_prior")
        self.assertFalse(row["solved"])
        self.assertIsNone(row["cost_queries"])
        self.assertEqual(row["total_queries"], 3)
        self.assertEqual(row["dock_calls"], 2)
        self.assertEqual(row["nav2_goals"], 2)
        self.assertEqual(row["retreats"], 1)
        self.assertEqual([b["offset_deg"] for b in bearings], [0.0, 45.0])
        self.assertEqual(bearings[1]["dock_reason"], "exception:RuntimeError:SAM3 found no instance for 'can'")
        self.assertEqual(bearings[1]["nav2_goals"], 0)
        self.assertEqual([b["outcome"] for b in bearings], ["prepare_failed", "dock_names_exhausted"])
        # A refused retreat shows its reason; the bearing's outcome is unaffected.
        self.assertEqual([b["retreat"] for b in bearings], ["obstacle_too_close", ""])

    def test_retreats_are_counted_from_drive_straight_calls_only(self) -> None:
        row, bearings = summarise(
            "x",
            trace(
                "x",
                [
                    dock_call(DOCKED_FAR),
                    prepare_call(SEARCH_FAILED),
                    retreat_call(RETREAT_REFUSED),
                    dock_call({**UNPLANNABLE, "reason": "nav2_failed:nav2_blocked_near_obstacle"}, approach=ANCHOR_DEG + 45.0),
                    retreat_call(RETREATED),
                    dock_call(DOCKED_FAR, approach=ANCHOR_DEG - 45.0),
                    prepare_call(SOLVED),
                ],
            ),
        )

        self.assertEqual(row["retreats"], 2)
        self.assertEqual(row["dock_calls"], 3)
        self.assertEqual([b["outcome"] for b in bearings], ["prepare_failed", "dock_failed", "solved"])
        self.assertEqual([b["retreats"] for b in bearings], [1, 1, 0])
        self.assertEqual([b["retreat"] for b in bearings], ["obstacle_too_close", "reached", ""])
        without = summarise("y", trace("y", [dock_call(DOCKED_FAR), prepare_call(SOLVED)]))
        self.assertEqual(without[0]["retreats"], 0)
        self.assertEqual(without[1][0]["retreat"], "")
        # A retreat before any dock opens the anchor group like a prepare does.
        groups = bearing_groups(trace("z", [retreat_call(RETREATED), dock_call(DOCKED_FAR)]))
        self.assertEqual([len(g["retreats"]) for g in groups], [1])
        self.assertEqual([len(g["docks"]) for g in groups], [1])

    def test_a_too_far_bearing_is_named_with_its_target_distance_and_no_queries(self) -> None:
        row, bearings = summarise(
            "x",
            trace(
                "x",
                [
                    dock_call(DOCKED_FAR),
                    prepare_call(TOO_FAR),
                    dock_call(DOCKED_FAR, approach=ANCHOR_DEG + 45.0),
                    prepare_call(SOLVED),
                ],
            ),
        )

        self.assertEqual([b["outcome"] for b in bearings], ["prepare_too_far", "solved"])
        self.assertEqual(bearings[0]["target_distance_m"], 1.31)
        self.assertEqual(bearings[0]["ik_queries"], 0)
        self.assertIn(
            "+0: prepare_too_far dock=docked_at_fallback_distance@0.90 "
            "prepare=target_too_far_for_local_manipulation@1.31 q=0; ",
            row["bearings"],
        )
        self.assertEqual(row["cost_queries"], 4)
        self.assertEqual(row["total_queries"], 6)

    def test_bearing_outcomes_follow_the_last_call(self) -> None:
        certification = {"success": False, "reason": "RuntimeError:actual_pose_grasp_certification_failed", "pi_ik_query_log": LOG}
        prepare_miss = {"success": False, "reason": "RuntimeError:SAM3 produced no usable mask after 3 attempts: SAM3 found no instance for 'can'"}
        cases = [
            ({"docks": [], "prepares": []}, None),
            ({"docks": [DOCK_MISS], "prepares": []}, "dock_names_exhausted"),
            ({"docks": [UNPLANNABLE], "prepares": []}, "dock_failed"),
            ({"docks": [DOCKED_FAR], "prepares": []}, "unfinished"),
            ({"docks": [DOCKED_FAR], "prepares": [SEARCH_FAILED]}, "prepare_failed"),
            ({"docks": [DOCKED_FAR], "prepares": [certification]}, "certification_failed"),
            ({"docks": [DOCKED_FAR], "prepares": [prepare_miss]}, "prepare_names_exhausted"),
            ({"docks": [DOCKED_FAR], "prepares": [TOO_FAR]}, "prepare_too_far"),
            ({"docks": [DOCKED_FAR], "prepares": [TOO_FAR, SOLVED]}, "solved"),
        ]
        for group, expected in cases:
            with self.subTest(expected=expected):
                self.assertEqual(bearing_outcome(group), expected)

    def test_the_anchor_comes_from_the_first_dock_that_reported_one(self) -> None:
        self.assertAlmostEqual(anchor_bearing_deg(SOLVED_TRACE), ANCHOR_DEG)
        no_bearing = trace("x", [dock_call({"success": False, "reason": "x", "metrics": {"backend": "nav2"}})])
        self.assertIsNone(anchor_bearing_deg(no_bearing))
        by_attempt = trace("x", [dock_call({"success": False, "reason": "x", "metrics": {"goal_attempts": [goal(0.6)], "arrival_bearing_rad": 1.0}})])
        self.assertAlmostEqual(anchor_bearing_deg(by_attempt), ANCHOR_DEG)
        by_arrival = trace("x", [dock_call({"success": False, "reason": "x", "metrics": {"arrival_bearing_rad": 1.0}})])
        self.assertAlmostEqual(anchor_bearing_deg(by_arrival), math.degrees(1.0))

    def test_bearing_groups_split_on_the_requested_bearing_and_attach_prepares(self) -> None:
        groups = bearing_groups(
            trace(
                "x",
                [
                    dock_call({"success": False, "reason": "exception:RuntimeError:SAM3 found no instance for 'can'", "metrics": {}}),
                    dock_call({**UNPLANNABLE, "metrics": {**UNPLANNABLE["metrics"]}}),
                    dock_call(DOCKED_FAR, approach=ANCHOR_DEG + 45.0),
                    dock_call(DOCKED_FAR, approach=ANCHOR_DEG + 45.0),
                    prepare_call(SEARCH_FAILED),
                    prepare_call(SOLVED),
                    retreat_call(RETREATED),
                ],
            )
        )

        self.assertEqual([len(g["docks"]) for g in groups], [2, 2])
        self.assertEqual([len(g["prepares"]) for g in groups], [0, 2])
        self.assertEqual([len(g["retreats"]) for g in groups], [0, 1])
        self.assertEqual([g["offset_deg"] for g in groups], [0.0, 45.0])

    def test_offsets_wrap_and_an_unknown_anchor_leaves_them_blank(self) -> None:
        self.assertEqual(wrap_degrees(ANCHOR_DEG + 180.0 - ANCHOR_DEG), 180.0)
        self.assertEqual(wrap_degrees(-200.0), 160.0)
        groups = bearing_groups(
            trace("x", [dock_call({"success": False, "reason": "x", "metrics": {}}, approach=30.0)])
        )
        self.assertIsNone(groups[0]["offset_deg"])
        row, _ = summarise("x", trace("x", [dock_call({"success": False, "reason": "x", "metrics": {}}, approach=30.0)]))
        self.assertIn("?: dock_failed dock=x", row["bearings"])

    def test_the_condition_is_the_config_confirmed_by_the_first_prior_record(self) -> None:
        self.assertEqual(condition(SOLVED_TRACE), "prior")
        self.assertEqual(condition(UNSOLVED_TRACE), "no_prior")

        def with_record(record: dict | None, enabled: bool | None) -> dict:
            metrics = {"reference_bearing_rad": ANCHOR_RAD, "goal_attempts": [goal(0.6)]}
            if record is not None:
                metrics["readiness_prior"] = record
            # The first name misses (no record); the second dock carries it.
            return trace(
                "x",
                [
                    dock_call(DOCK_MISS),
                    dock_call({"success": True, "reason": "within_docking_distance", "metrics": metrics}),
                    dock_call(DOCKED_FAR, approach=ANCHOR_DEG + 45.0),
                ],
                config=run_config(enabled),
            )

        # The record decides: used, not configured, or enabled but failed.
        self.assertEqual(condition(with_record({"used": True, "reason": None}, True)), "prior")
        self.assertEqual(condition(with_record({"used": False, "reason": "not_configured"}, False)), "no_prior")
        self.assertEqual(
            condition(with_record({"used": False, "reason": "no_estimate"}, True)),
            "prior_failed:no_estimate",
        )
        self.assertEqual(
            condition(with_record({"used": False, "reason": "exception:RuntimeError:vggt down"}, None)),
            "prior_failed:exception:RuntimeError:vggt down",
        )
        # Without a record the config alone names the condition.
        self.assertEqual(condition(with_record(None, True)), "prior")
        self.assertEqual(condition(with_record(None, False)), "no_prior")
        self.assertIsNone(condition(with_record(None, None)))
        self.assertIsNone(condition(trace("x", [dock_call({"success": False, "reason": "x", "metrics": {}})])))
        self.assertIsNone(condition(trace("x", [])))
        # An explicit-bearing dock's record says nothing about the condition.
        self.assertEqual(
            condition(trace("x", [dock_call(DOCKED_FAR, approach=30.0)], config=run_config(True))),
            "prior",
        )
        self.assertTrue(configured_prior_enabled(SOLVED_TRACE))
        self.assertFalse(configured_prior_enabled(UNSOLVED_TRACE))
        self.assertIsNone(configured_prior_enabled(trace("x", [])))
        self.assertIsNone(configured_prior_enabled({"config": run_config(None)}))

    def test_a_solving_call_without_a_query_log_costs_its_whole_count(self) -> None:
        solved = {"success": True, "reason": "grasp_execution_ready", "ik_query_count": 40}
        row, _ = summarise("x", trace("x", [dock_call(DOCKED_FAR), prepare_call(solved)]))

        self.assertEqual(row["cost_queries"], 40)
        self.assertEqual(row["total_queries"], 40)

    def test_queries_until_a_base_certified_count_to_its_minimum_converged_grasp(self) -> None:
        # Base 7 certifies at the fourth query, base 9 at the sixth; base 3
        # never converges, base 0 never appears converged.
        self.assertEqual(queries_until_base(LOG, 7, 1), 4)
        self.assertEqual(queries_until_base(LOG, 7, 2), 5)
        self.assertEqual(queries_until_base(LOG, 9, 1), 6)
        self.assertIsNone(queries_until_base(LOG, 3, 1))
        self.assertIsNone(queries_until_base(LOG, 0, 1))
        self.assertIsNone(queries_until_base(LOG, 7, 3))
        self.assertIsNone(queries_until_base([], 7, 1))
        self.assertIsNone(queries_until_base(["<row>", {"order": 0}], 7, 1))

    def test_the_solving_cost_runs_until_the_executed_base_certified(self) -> None:
        # The search selected base 7 (first certified) but its motion was
        # refused and base 9 was executed: the cost is what base 9 took.
        executed_later = {**SOLVED, "executed_candidate_index": 9, "motion_attempts": [
            {"attempt": 0, "candidate_index": 7, "success": False, "reason": "obstacle_too_close"},
            {"attempt": 1, "candidate_index": 9, "success": True, "reason": "reached"},
        ]}
        self.assertEqual(solving_cost_queries(executed_later), 6)
        self.assertEqual(solving_cost_queries({**SOLVED, "executed_candidate_index": 7}), 4)
        # Without the key, the first certified base as before.
        self.assertEqual(solving_cost_queries(SOLVED), 4)
        self.assertEqual(solving_cost_queries({**SOLVED, "executed_candidate_index": None}), 4)
        # A base the log does not show certified: the whole call's count.
        self.assertEqual(solving_cost_queries({**SOLVED, "executed_candidate_index": 3}), 6)
        self.assertEqual(solving_cost_queries({**SOLVED, "minimum_feasible_grasps": 2}), 5)
        row, _ = summarise(
            "x",
            trace("x", [dock_call(DOCKED_FAR), prepare_call(SEARCH_FAILED), dock_call(DOCKED_FAR, approach=ANCHOR_DEG + 45.0), prepare_call(executed_later)]),
        )
        self.assertEqual(row["cost_queries"], 3 + 6)
        self.assertEqual(row["total_queries"], 9)

    def test_nav2_goals_count_sent_goals_and_the_goal_distance_falls_back_to_the_last_attempt(self) -> None:
        stopped_before_goal = {
            "success": False,
            "reason": "nav2_failed:operator_stop",
            "metrics": {"goal_attempts": [goal(0.6, "aborted"), goal(None, "operator_stop")]},
        }
        self.assertEqual(nav2_goal_count(stopped_before_goal), 1)
        self.assertEqual(goal_distance_used_m(stopped_before_goal), 0.6)
        with_results = {"metrics": {"goal_attempts": [goal(0.6, "aborted", nav2={"success": False}), goal(0.75, "operator_stop", nav2=None)]}}
        self.assertEqual(nav2_goal_count(with_results), 1)
        without_goal = {"success": True, "reason": "docked_at_fallback_distance", "metrics": {"goal_attempts": [goal(None, "succeeded", goal_distance_m=0.83)], "goal_distance_used_m": 0.83}}
        self.assertEqual(nav2_goal_count(without_goal), 0)
        self.assertEqual(goal_distance_used_m(without_goal), 0.83)
        self.assertIsNone(goal_distance_used_m({"metrics": {"goal_attempts": [goal(None, "no_plannable_goal_on_bearing")]}}))
        self.assertIsNone(goal_distance_used_m({}))

    def test_collapsed_lists_are_counted_from_their_length(self) -> None:
        dock = {"success": True, "reason": "within_docking_distance", "metrics": {"goal_attempts": "<list len=3>"}}
        prepare = {"success": False, "reason": "RuntimeError:x", "pi_ik_query_log": "<list len=200>"}
        row, _ = summarise("x", trace("x", [dock_call(dock), prepare_call(prepare)]))

        self.assertEqual(row["nav2_goals"], 3)
        self.assertEqual(row["total_queries"], 200)


class MainTest(unittest.TestCase):
    def test_main_prints_markdown_tables_and_writes_the_csv(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        for name, data in (("run-a", SOLVED_TRACE), ("run-b", UNSOLVED_TRACE)):
            (root / name).mkdir()
            (root / name / "trace.json").write_text(json.dumps(data))
        (root / "broken").mkdir()
        (root / "broken" / "trace.json").write_text("{not json")
        out = io.StringIO()
        err = io.StringIO()

        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            status = main([str(root), "--csv", str(root / "trials.csv")])

        self.assertEqual(status, 0)
        text = out.getvalue()
        self.assertIn("| run | condition | start_label | anchor_deg | bearings | dock_calls | nav2_goals | solved | cost_queries | total_queries | retreats |", text)
        self.assertIn("| web-solved | prior | - | 28.60 |", text)
        self.assertIn("| yes | 10 | 12 | 1 |", text)
        self.assertIn("| web-unsolved | no_prior |", text)
        self.assertIn("| no | - | 3 | 1 |", text)
        self.assertIn("| run | offset_deg | approach_deg | outcome | docks | dock_reason | goal_distance_m | nav2_goals | prepares | prepare_reason | target_distance_m | ik_queries | retreat |", text)
        self.assertIn("| web-solved | -45.00 |", text)
        self.assertIn("| solved | 1 | docked_at_fallback_distance | 0.90 | 2 | 1 | grasp_execution_ready | - | 6 | - |", text)
        self.assertIn("| prepare_failed | 1 | docked_at_fallback_distance | 0.90 | 2 | 1 | RuntimeError:no certified base reachable: 2 motions refused (obstacle_too_closex2) | - | 6 | reached |", text)
        self.assertIn("| 3 | obstacle_too_close |", text)
        self.assertIn("2 trials, 1 solved", text)
        self.assertIn(
            "| policy | condition | trials | solved | success_rate | median_time_s | median_solved_time_s | mean_total_queries |",
            text,
        )
        self.assertIn("skipping", err.getvalue())
        with (root / "trials.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual([row["run"] for row in rows], ["web-solved", "web-unsolved"])
        self.assertEqual(rows[0]["cost_queries"], "10")
        self.assertEqual(rows[1]["cost_queries"], "")
        self.assertEqual([row["retreats"] for row in rows], ["1", "1"])

    def test_main_reports_when_no_trace_is_found(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        err = io.StringIO()

        with contextlib.redirect_stderr(err):
            status = main([tmp.name])

        self.assertEqual(status, 1)
        self.assertIn("no traces found", err.getvalue())


if __name__ == "__main__":
    unittest.main()
