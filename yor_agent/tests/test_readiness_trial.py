"""The readiness trial walks the bearings around the object and counts its cost."""

from __future__ import annotations

import asyncio
import contextlib
import io
import json
import math
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from fakes import FakeHardware, make_environment

from yor_agent.agents.default import DefaultAgent
from yor_agent.exceptions import ModelError, PrimitiveFailed, Stopped
from yor_agent.executor import PolicyExecutor
from yor_agent.experiments.readiness_trial import (
    DEFAULT_LLM_TIME_LIMIT_S,
    DEFAULT_OBJECT_NAMES,
    LLM_TRIAL_HIDDEN_PRIMITIVES,
    LLM_TRIAL_INSTRUCTION,
    LLM_TRIAL_MODEL_NAME,
    LLM_TRIAL_MODEL_PROVIDER,
    TRIAL_ARM_SIDE_HEADING_OFFSET_DEG,
    TRIAL_CONTINUE_SEARCH_AFTER_REFUSED_MOTIONS,
    TRIAL_MAXIMUM_INITIAL_TARGET_DISTANCE_M,
    TRIAL_MOTION_ALTERNATIVE_LIMIT,
    TRIAL_NAV2_GOAL_DISTANCE_FALLBACKS_M,
    TRIAL_PI_IK_COMPUTE_BUDGET_S,
    TRIAL_RETREAT_DISTANCE_M,
    TRIAL_TARGET_HEIGHT_MARGIN_M,
    WEB_UI_MAX_TURNS,
    check_fallback_distances_within_prepare_limit,
    default_output_dir,
    fallback_target_distance_limit_m,
    main as trial_main,
    parse_bearing_offsets_deg,
    trial_config,
    trial_turn_budget,
    unmatched_object_names,
)
from yor_agent.launch import apply_readiness_prior_overrides, build_model, load_config
from yor_agent.models.llm import LLM, extract_code
from yor_agent.models.scripted import (
    DEFAULT_BEARING_OFFSETS_DEG,
    DEFAULT_RETREAT_DISTANCE_M,
    ReadinessTrialModel,
    anchor_bearing_deg,
    attempt_sent_nav2_goal,
    dock_navigated,
    nav2_goal_count,
    normalize_bearing_offsets_deg,
    normalize_object_names,
    normalize_retreat_distance_m,
    prepare_ik_query_count,
    primitive_outcome,
    wrap_degrees,
)
from yor_agent.primitive_config import primitive_exposed, primitive_settings
from yor_agent.primitives.registry import PrimitiveRegistry
from yor_agent.robot.readiness_prior import EVENTS_SCHEMA
from yor_agent.trace import Trace

try:
    from yor_agent.web.server import WebRunController
except ModuleNotFoundError as exc:
    if exc.name not in {"fastapi", "uvicorn"}:
        raise
    WebRunController = None


CONFIGS = Path(__file__).resolve().parents[1] / "configs"
TASKS = CONFIGS / "passive_video_tasks.yaml"
DOCK_MISS = "exception:RuntimeError:SAM3 found no instance for 'can'"
PREPARE_MISS = (
    "RuntimeError:SAM3 produced no usable mask after 3 attempts: "
    "SAM3 found no instance for 'can'"
)
SEARCH_FAILED = (
    "RuntimeError:no local base pose has a collision-safe, strictly "
    "Pi-converged nominal grasp: bases=343 queries=512/512"
)
MOTION_EXHAUSTED = (
    "RuntimeError:no certified base reachable: 3 motions refused (obstacle_too_closex3)"
)
MOTION_REFUSED = "RuntimeError:single-stage SE(2) motion failed: obstacle_too_close"
CERTIFICATION_FAILED = "actual_pose_grasp_certification_failed"
TARGET_TOO_FAR = "target_too_far_for_local_manipulation"
# Docking's refusal when the start-clearance check found the base next to an
# obstacle before Nav2 was started: the base has not moved.
BLOCKED_START = "nav2_blocked_near_obstacle"
# Docking's reason when Nav2 stopped making progress on a goal it was sent:
# the base stands wherever the stall left it.
STALLED = (
    "nav2_stalled_no_progress: Nav2 made no meaningful base progress within "
    "the configured limit, so the goal was canceled and the robot was stopped."
)
# Docking's reason when Nav2 reached its goal but the fresh check at the
# arrival pose found the target beyond the docking distance.
FINAL_CHECK_FAILED = "final_visual_verification_failed"
RETREAT_REFUSED = "obstacle_too_close"
ANCHOR_RAD = 0.5
ANCHOR_DEG = math.degrees(ANCHOR_RAD)


def retreat_program(bearing: int, distance_m: float = DEFAULT_RETREAT_DISTANCE_M) -> str:
    """The retreat program the model issues after bearing ``bearing``."""

    return (
        f"result = drive_straight(-{distance_m!r})\n"
        f"print({{'bearing': {bearing!r}, 'retreat_m': {distance_m!r}, "
        "'reason': result.get('reason')})"
    )


def approach(offset_deg: float, anchor_deg: float = ANCHOR_DEG) -> float:
    """The bearing the model asks for at ``offset_deg`` from the anchor."""

    return round(wrap_degrees(anchor_deg + offset_deg), 3)


def goal(distance_m: float | None, reason: str = "succeeded", **extra) -> dict:
    """One ``goal_attempts`` entry as docking records it.

    A navigated goal carries the pose it was sent to and ``navigated`` true;
    an attempt that never drove (every distance failed the plan check, or the
    operator's stop came first) records ``None`` for the pose and the
    distance and ``navigated`` false.
    """

    pose = None if distance_m is None else [1.0, 2.0, 0.5]
    return {
        "bearing_source": "explicit",
        "bearing_rad": ANCHOR_RAD,
        "camera_goal_xy_yaw": pose,
        "nav2_goal_xy_yaw": pose,
        "goal_distance_m": distance_m,
        "navigated": pose is not None,
        "plan_checks": [],
        "success": reason == "succeeded",
        "reason": reason,
        **extra,
    }


def too_far(distance_m: float = 1.31) -> dict:
    """prepare_for_manipulation's refusal before any IK query."""

    return {
        "success": False,
        "reason": TARGET_TOO_FAR,
        "target_distance_m": distance_m,
        "maximum_distance_m": TRIAL_MAXIMUM_INITIAL_TARGET_DISTANCE_M,
    }


def dock_metrics(*goals: dict, bearing_rad: float | None = ANCHOR_RAD, **extra) -> dict:
    metrics = {"goal_attempts": list(goals), **extra}
    if bearing_rad is not None:
        metrics["reference_bearing_rad"] = bearing_rad
    return metrics


def docked(*, arm: str | None = "left", goals: int = 1, **extra) -> dict:
    result = {
        "success": True,
        "reason": "within_docking_distance",
        "metrics": dock_metrics(*([goal(0.6)] * goals), **extra),
    }
    if arm:
        result["suggested_arm"] = arm
    return result


def dock_failed(reason: str, *goals: dict, **extra) -> dict:
    return {"success": False, "reason": reason, "metrics": dock_metrics(*goals, **extra)}


def prepared(queries: int = 5) -> dict:
    return {"success": True, "reason": "grasp_execution_ready", "ik_query_count": queries}


def prepare_failed(reason: str, *, queries: int = 3, diagnostics: bool = False) -> dict:
    log = [{"order": index} for index in range(queries)]
    result = {"success": False, "reason": reason}
    if diagnostics:
        result["diagnostics"] = {"pi_ik_query_log": log, "ik_query_count": queries}
    else:
        result["pi_ik_query_log"] = log
    return result


def record(primitive: str, *, result=None, failure=None, error=None, error_type="RuntimeError") -> dict:
    """An execution record shaped like PolicyExecutor.execute's."""

    calls: list[dict] = []
    execution = {
        "stdout": "",
        "stderr": "",
        "error": None,
        "interrupted_by": None,
        "primitive_calls": calls,
        "finish_reason": None,
        "elapsed_s": 0.0,
    }
    if error is not None:
        execution["error"] = {"type": error_type, "message": error, "traceback": error}
    elif failure is not None:
        failed = {**(result or {}), "success": False, "reason": failure}
        calls.append(
            {"name": primitive, "result": failed, "error": {"type": "PrimitiveFailed", "message": failure}}
        )
        execution["interrupted_by"] = {"primitive": primitive, "reason": failure, "result": failed}
    else:
        calls.append({"name": primitive, "result": {"success": True, **(result or {})}, "error": None})
    return execution


def dock_record(result: dict) -> dict:
    if result.get("success"):
        return record("dock_to_visible_object", result=result)
    return record("dock_to_visible_object", failure=result["reason"], result=result)


def prepare_record(result: dict) -> dict:
    if result.get("success"):
        return record("prepare_for_manipulation", result=result)
    return record("prepare_for_manipulation", failure=result["reason"], result=result)


def retreated(distance_m: float = -DEFAULT_RETREAT_DISTANCE_M) -> dict:
    return {"success": True, "reason": "reached", "progress_m": distance_m}


def retreat_record(result: dict | None = None) -> dict:
    result = retreated() if result is None else result
    if result.get("success"):
        return record("drive_straight", result=result)
    return record("drive_straight", failure=result["reason"], result=result)


class ReadinessTrialModelTest(unittest.TestCase):
    def after(self, model: ReadinessTrialModel, code: str, execution: dict) -> str:
        model.format_feedback(code, execution, {})
        return model.query([])

    def after_retreat(
        self, model: ReadinessTrialModel, code: str, execution: dict, *, bearing: int
    ) -> str:
        """Feed ``execution``, which ends bearing ``bearing`` with the base at
        the object; expect the retreat, feed its success, return what follows."""

        code = self.after(model, code, execution)
        self.assertEqual(code, retreat_program(bearing))
        return self.after(model, code, retreat_record())

    def test_the_first_turn_docks_with_the_first_name_without_a_bearing(self) -> None:
        model = ReadinessTrialModel({"object_names": ["can", "beverage can"]})

        code = model.query([])

        self.assertIn("dock_to_visible_object('can')\n", code)
        self.assertNotIn("approach_bearing_deg", code.split("\n")[0])
        self.assertIn("'bearing': 0", code)
        self.assertEqual(model.provider, "scripted")
        self.assertEqual(model.name, "readiness-trial")
        self.assertEqual(model.bearing_offsets_deg, DEFAULT_BEARING_OFFSETS_DEG)
        self.assertEqual(model.certification_retries, 1)

    def test_a_sam3_miss_tries_the_next_name_then_prepares_with_the_first_name(self) -> None:
        model = ReadinessTrialModel({"object_names": ["can", "beverage can"]})
        code = model.query([])

        code = self.after(model, code, record("dock_to_visible_object", failure=DOCK_MISS))
        self.assertIn("dock_to_visible_object('beverage can')", code)

        code = self.after(model, code, dock_record(docked(arm="left")))
        # The fallback name only helped detection at docking range.
        self.assertIn("prepare_for_manipulation('can', arm='left')", code)
        self.assertIn("'bearing': 0", code)

    def test_docking_names_exhausted_without_an_anchor_ends_the_trial(self) -> None:
        model = ReadinessTrialModel({"object_names": ["can", "soda can"], "name_rounds": 2})
        code = model.query([])
        tried = [code]
        for _ in range(3):
            code = self.after(model, code, record("dock_to_visible_object", failure=DOCK_MISS))
            tried.append(code)

        self.assertEqual(
            [("'soda can'" in item) for item in tried], [False, True, False, True]
        )
        code = self.after(model, code, record("dock_to_visible_object", failure=DOCK_MISS))
        self.assertIn("finish(", code)
        self.assertIn("trial unsolved after 1 of 8 bearings: docking reported no reference bearing", code)
        self.assertIn("+0: dock_names_exhausted", code)
        self.assertEqual(model.trial_summary()["dock_calls"], 4)

    def test_a_docking_failure_moves_to_the_next_bearing_at_the_anchor_plus_offset(self) -> None:
        model = ReadinessTrialModel({"object_names": ["can", "beverage can"]})
        code = model.query([])

        code = self.after(
            model,
            code,
            dock_record(dock_failed("no_plannable_goal_on_bearing", goal(None, "no_plannable_goal_on_bearing"))),
        )

        self.assertIn(
            f"dock_to_visible_object('can', approach_bearing_deg={approach(45.0)!r})", code
        )
        self.assertIn("'bearing': 1", code)
        summary = model.trial_summary()
        self.assertAlmostEqual(summary["anchor_bearing_deg"], ANCHOR_DEG)
        self.assertEqual(summary["bearings"][0]["outcome"], "dock_failed")
        self.assertEqual(summary["bearings"][0]["dock_reason"], "no_plannable_goal_on_bearing")
        self.assertEqual(summary["bearings"][0]["nav2_goals"], 0)
        self.assertEqual(summary["bearings"][1]["approach_bearing_deg"], approach(45.0))

    def test_the_names_restart_at_every_bearing_and_exhausting_them_moves_on(self) -> None:
        model = ReadinessTrialModel({"object_names": ["can", "soda can"]})
        code = model.query([])
        # Nav2 aborted under way, so the base is backed off before bearing 1.
        code = self.after_retreat(
            model, code, dock_record(dock_failed("nav2_failed:aborted", goal(0.6, "aborted"))), bearing=0
        )
        self.assertIn("dock_to_visible_object('can', approach_bearing_deg", code)

        code = self.after(model, code, record("dock_to_visible_object", failure=DOCK_MISS))
        self.assertIn("dock_to_visible_object('soda can', approach_bearing_deg", code)
        code = self.after(model, code, record("dock_to_visible_object", failure=DOCK_MISS))

        self.assertIn(
            f"dock_to_visible_object('can', approach_bearing_deg={approach(-45.0)!r})", code
        )
        bearings = model.trial_summary()["bearings"]
        self.assertEqual([b["outcome"] for b in bearings], ["dock_failed", "dock_names_exhausted", None])
        self.assertEqual(bearings[0]["nav2_goals"], 1)

    def test_later_bearings_try_the_accepted_name_first_and_every_other_name_once(self) -> None:
        model = ReadinessTrialModel(
            {"object_names": ["can", "beverage can", "soda can"], "name_rounds": 2}
        )
        code = model.query([])
        # Bearing 0: the first name misses, the second is accepted and docks.
        code = self.after(model, code, record("dock_to_visible_object", failure=DOCK_MISS))
        code = self.after(model, code, dock_record(docked(arm="left")))
        code = self.after_retreat(model, code, prepare_record(prepare_failed(MOTION_EXHAUSTED)), bearing=0)

        # Bearing 1: the accepted name first, then each other name once; no
        # second round even though the anchor bearing had two.
        names = []
        for _ in range(3):
            self.assertIn("approach_bearing_deg", code)
            names.append(code.split("dock_to_visible_object(")[1].split(",")[0])
            code = self.after(model, code, record("dock_to_visible_object", failure=DOCK_MISS))
        self.assertEqual(names, ["'beverage can'", "'can'", "'soda can'"])
        # Nothing docked at bearing 1, so bearing 2 is docked without a retreat.
        self.assertIn(f"dock_to_visible_object('beverage can', approach_bearing_deg={approach(-45.0)!r})", code)
        bearings = model.trial_summary()["bearings"]
        self.assertEqual([b["outcome"] for b in bearings], ["prepare_failed", "dock_names_exhausted", None])
        self.assertEqual(bearings[1]["dock_calls"], 3)
        self.assertEqual([b["retreat_calls"] for b in bearings], [1, 0, 0])

    def test_a_name_accepted_by_a_failed_dock_is_tried_first_at_the_next_bearing(self) -> None:
        model = ReadinessTrialModel({"object_names": ["can", "soda can"]})
        code = model.query([])
        code = self.after(model, code, record("dock_to_visible_object", failure=DOCK_MISS))
        # SAM3 accepted 'soda can'; Nav2 then failed, which is not a miss.
        code = self.after_retreat(
            model, code, dock_record(dock_failed("nav2_failed:aborted", goal(0.6, "aborted"))), bearing=0
        )

        self.assertIn(f"dock_to_visible_object('soda can', approach_bearing_deg={approach(45.0)!r})", code)

    def test_a_prepare_sam3_miss_tries_the_other_names_then_the_next_bearing(self) -> None:
        model = ReadinessTrialModel({"object_names": ["can", "beverage can", "soda can"]})
        code = model.query([])
        code = self.after(model, code, record("dock_to_visible_object", failure=DOCK_MISS))
        code = self.after(model, code, dock_record(docked(arm="right")))
        # 'beverage can' docked, but prepare starts from the first name and
        # goes through the list in order.
        self.assertIn("prepare_for_manipulation('can', arm='right')", code)
        self.assertEqual(model.trial_summary()["bearings"][0]["docked_name"], "beverage can")

        code = self.after(model, code, record("prepare_for_manipulation", failure=PREPARE_MISS))
        self.assertIn("prepare_for_manipulation('beverage can', arm='right')", code)
        code = self.after(model, code, record("prepare_for_manipulation", failure=PREPARE_MISS))
        self.assertIn("prepare_for_manipulation('soda can', arm='right')", code)
        code = self.after_retreat(
            model, code, record("prepare_for_manipulation", failure=PREPARE_MISS), bearing=0
        )

        # The next bearing docks with the name SAM3 accepted at this one first.
        self.assertIn(f"dock_to_visible_object('beverage can', approach_bearing_deg={approach(45.0)!r})", code)
        self.assertEqual(model.trial_summary()["bearings"][0]["outcome"], "prepare_names_exhausted")

    def test_a_successful_prepare_finishes_solved_with_the_counts(self) -> None:
        model = ReadinessTrialModel({"object_names": ["can"]})
        code = model.query([])
        code = self.after(model, code, dock_record(docked(arm="left", goals=2)))

        code = self.after(model, code, prepare_record(prepared(queries=96)))

        self.assertIn("finish(", code)
        self.assertIn(
            "trial solved at bearing 0 (offset +0 deg): prepare_for_manipulation "
            "grasp_execution_ready with the left arm for 'can'; bearings [+0: solved]; "
            "1 dock calls, 2 Nav2 goals, 96 Pi IK queries, 0 retreats",
            code,
        )
        summary = model.trial_summary()
        self.assertTrue(summary["solved"])
        self.assertEqual(summary["solved_bearing"], 0)
        self.assertEqual((summary["dock_calls"], summary["nav2_goals"], summary["ik_queries"]), (1, 2, 96))
        self.assertEqual(summary["retreats"], 0)
        self.assertIn("print({'trial': {", code)

    def test_a_prepare_search_failure_moves_to_the_next_bearing_and_counts_its_queries(self) -> None:
        model = ReadinessTrialModel({"object_names": ["can"]})
        code = model.query([])
        code = self.after(model, code, dock_record(docked()))

        code = self.after_retreat(
            model,
            code,
            prepare_record(prepare_failed(SEARCH_FAILED, queries=512, diagnostics=True)),
            bearing=0,
        )

        self.assertIn(f"dock_to_visible_object('can', approach_bearing_deg={approach(45.0)!r})", code)
        bearing = model.trial_summary()["bearings"][0]
        self.assertEqual(bearing["outcome"], "prepare_failed")
        self.assertEqual(bearing["prepare_reason"], SEARCH_FAILED)
        self.assertEqual(bearing["ik_queries"], 512)

    def test_motion_exhausted_and_other_prepare_failures_move_to_the_next_bearing(self) -> None:
        for reason in (MOTION_EXHAUSTED, MOTION_REFUSED):
            with self.subTest(reason=reason):
                model = ReadinessTrialModel({"object_names": ["can"]})
                code = model.query([])
                code = self.after(model, code, dock_record(docked()))
                code = self.after_retreat(
                    model, code, prepare_record(prepare_failed(reason, queries=7)), bearing=0
                )

                self.assertIn("dock_to_visible_object('can', approach_bearing_deg", code)
                self.assertEqual(model.trial_summary()["bearings"][0]["outcome"], "prepare_failed")
                self.assertEqual(model.trial_summary()["ik_queries"], 7)

    def test_a_target_too_far_for_prepare_is_its_own_outcome_and_moves_on(self) -> None:
        model = ReadinessTrialModel({"object_names": ["can"], "bearing_offsets_deg": [0, 45]})
        code = model.query([])
        code = self.after(model, code, dock_record(docked()))

        code = self.after_retreat(model, code, prepare_record(too_far(1.31)), bearing=0)

        self.assertIn("dock_to_visible_object('can', approach_bearing_deg", code)
        bearing = model.trial_summary()["bearings"][0]
        self.assertEqual(bearing["outcome"], "prepare_too_far")
        self.assertEqual(bearing["prepare_reason"], TARGET_TOO_FAR)
        self.assertEqual(bearing["prepare_target_distance_m"], 1.31)
        self.assertEqual(bearing["ik_queries"], 0)

        code = self.after(model, code, dock_record(docked()))
        code = self.after(model, code, prepare_record(too_far(1.25)))
        self.assertIn("finish(", code)
        self.assertIn(
            "bearings [+0: prepare_too_far (target 1.31 m); +45: prepare_too_far (target 1.25 m)]",
            code,
        )
        # No retreat after the last bearing: there is no dock left to make room for.
        self.assertIn("2 dock calls, 2 Nav2 goals, 0 Pi IK queries, 1 retreats", code)

    def test_a_failed_certification_is_retried_at_the_same_bearing_then_the_next_bearing(self) -> None:
        model = ReadinessTrialModel({"object_names": ["can"]})
        code = model.query([])
        code = self.after(model, code, dock_record(docked(arm="left")))

        code = self.after(model, code, prepare_record(prepare_failed(CERTIFICATION_FAILED, queries=4)))
        self.assertIn("prepare_for_manipulation('can', arm='left')", code)
        self.assertIn("'bearing': 0", code)

        code = self.after_retreat(
            model, code, prepare_record(prepare_failed(CERTIFICATION_FAILED, queries=4)), bearing=0
        )
        self.assertIn(f"dock_to_visible_object('can', approach_bearing_deg={approach(45.0)!r})", code)
        bearing = model.trial_summary()["bearings"][0]
        self.assertEqual(bearing["outcome"], "certification_failed")
        self.assertEqual(bearing["prepare_calls"], 2)
        self.assertEqual(bearing["ik_queries"], 8)

    def test_certification_retries_zero_moves_on_at_once(self) -> None:
        model = ReadinessTrialModel({"object_names": ["can"], "certification_retries": 0})
        code = model.query([])
        code = self.after(model, code, dock_record(docked()))

        code = self.after_retreat(
            model, code, prepare_record(prepare_failed(CERTIFICATION_FAILED)), bearing=0
        )

        self.assertIn("dock_to_visible_object('can', approach_bearing_deg", code)

    def test_a_docked_but_unsolved_bearing_is_followed_by_a_retreat_then_the_next_dock(self) -> None:
        model = ReadinessTrialModel({"object_names": ["can"]})
        code = model.query([])
        code = self.after(model, code, dock_record(docked(arm="left")))

        code = self.after(model, code, prepare_record(prepare_failed(SEARCH_FAILED, queries=9)))

        self.assertEqual(
            code,
            "result = drive_straight(-0.3)\n"
            "print({'bearing': 0, 'retreat_m': 0.3, 'reason': result.get('reason')})",
        )
        self.assertEqual(extract_code(code), code)
        code = self.after(model, code, retreat_record(retreated()))
        self.assertIn(f"dock_to_visible_object('can', approach_bearing_deg={approach(45.0)!r})", code)
        self.assertIn("'bearing': 1", code)
        summary = model.trial_summary()
        self.assertEqual(summary["retreats"], 1)
        bearing = summary["bearings"][0]
        self.assertEqual(bearing["outcome"], "prepare_failed")
        self.assertEqual(
            (bearing["retreat_calls"], bearing["retreat_reason"], bearing["retreat_success"]),
            (1, "reached", True),
        )
        self.assertEqual(summary["bearings"][1]["retreat_calls"], 0)
        self.assertEqual(summary["bearings"][1]["retreat_success"], None)

    def test_no_retreat_before_the_first_dock_or_after_a_bearing_where_nothing_drove(self) -> None:
        model = ReadinessTrialModel({"object_names": ["can", "soda can"]})
        code = model.query([])
        self.assertIn("dock_to_visible_object('can')", code)
        # Bearing 0: no goal distance passed the plan check, so no goal was
        # sent and the base never moved; bearing 1: SAM3 misses only.
        unplannable = dock_failed("no_plannable_goal_on_bearing", goal(None, "no_plannable_goal_on_bearing"))
        self.assertFalse(any(a["navigated"] for a in unplannable["metrics"]["goal_attempts"]))
        code = self.after(model, code, dock_record(unplannable))
        self.assertIn(f"dock_to_visible_object('can', approach_bearing_deg={approach(45.0)!r})", code)
        code = self.after(model, code, record("dock_to_visible_object", failure=DOCK_MISS))
        code = self.after(model, code, record("dock_to_visible_object", failure=DOCK_MISS))

        self.assertIn(f"dock_to_visible_object('can', approach_bearing_deg={approach(-45.0)!r})", code)
        summary = model.trial_summary()
        self.assertEqual(summary["retreats"], 0)
        self.assertEqual([b["outcome"] for b in summary["bearings"]], ["dock_failed", "dock_names_exhausted", None])
        self.assertEqual([b["retreat_calls"] for b in summary["bearings"]], [0, 0, 0])
        self.assertEqual([b["retreat_success"] for b in summary["bearings"]], [None, None, None])

    def test_a_dock_that_stalled_after_sending_goals_is_followed_by_a_retreat(self) -> None:
        model = ReadinessTrialModel({"object_names": ["can"]})
        code = model.query([])
        # Every goal Nav2 was sent stalled: the base stands where the last
        # one left it, next to the object, and a dock started there would be
        # refused by the start-clearance check.
        code = self.after(
            model,
            code,
            dock_record(
                dock_failed(
                    STALLED,
                    goal(0.6, "stalled_no_progress"),
                    goal(0.75, "stalled_no_progress"),
                    goal(0.9, "stalled_no_progress"),
                )
            ),
        )

        self.assertEqual(code, retreat_program(0))
        code = self.after(model, code, retreat_record())
        self.assertIn(f"dock_to_visible_object('can', approach_bearing_deg={approach(45.0)!r})", code)
        self.assertIn("'bearing': 1", code)
        summary = model.trial_summary()
        self.assertEqual(summary["retreats"], 1)
        bearing = summary["bearings"][0]
        self.assertEqual((bearing["outcome"], bearing["dock_reason"], bearing["nav2_goals"]), ("dock_failed", STALLED, 3))
        self.assertEqual(
            (bearing["retreat_calls"], bearing["retreat_reason"], bearing["retreat_success"]),
            (1, "reached", True),
        )
        self.assertEqual(summary["bearings"][1]["retreat_calls"], 0)

    def test_a_failed_final_visual_check_after_navigating_is_followed_by_a_retreat(self) -> None:
        model = ReadinessTrialModel({"object_names": ["can"], "bearing_offsets_deg": [0, 45]})
        code = model.query([])
        # Nav2 reached its goal (the attempt itself succeeded); only the
        # check at the arrival pose failed, so the base is at the object.
        code = self.after(model, code, dock_record(dock_failed(FINAL_CHECK_FAILED, goal(0.6))))

        self.assertEqual(code, retreat_program(0))
        code = self.after(model, code, retreat_record())
        self.assertIn(f"dock_to_visible_object('can', approach_bearing_deg={approach(45.0)!r})", code)
        code = self.after(model, code, dock_record(dock_failed(FINAL_CHECK_FAILED, goal(0.6))))
        self.assertIn("finish(", code)
        # No retreat after the last bearing: there is no dock left to make room for.
        self.assertIn(
            "bearings [+0: dock_failed (final_visual_verification_failed); "
            "+45: dock_failed (final_visual_verification_failed)]; "
            "2 dock calls, 2 Nav2 goals, 0 Pi IK queries, 1 retreats",
            code,
        )
        self.assertEqual([b["retreat_calls"] for b in model.trial_summary()["bearings"]], [1, 0])

    def test_a_dock_refused_next_to_an_obstacle_is_followed_by_a_retreat(self) -> None:
        model = ReadinessTrialModel({"object_names": ["can"]})
        code = model.query([])
        code = self.after(model, code, dock_record(docked()))
        # The retreat after bearing 0 is refused, so the base is still at the
        # object and bearing 1's dock is refused before Nav2 starts.
        code = self.after(model, code, prepare_record(prepare_failed(MOTION_EXHAUSTED, queries=3)))
        self.assertEqual(code, retreat_program(0))
        code = self.after(model, code, retreat_record({"success": False, "reason": RETREAT_REFUSED}))
        self.assertIn(f"dock_to_visible_object('can', approach_bearing_deg={approach(45.0)!r})", code)

        code = self.after(
            model,
            code,
            dock_record(dock_failed(f"nav2_failed:{BLOCKED_START}", goal(None, BLOCKED_START))),
        )

        self.assertEqual(code, retreat_program(1))
        code = self.after(model, code, retreat_record(retreated()))
        self.assertIn(f"dock_to_visible_object('can', approach_bearing_deg={approach(-45.0)!r})", code)
        summary = model.trial_summary()
        self.assertEqual(summary["retreats"], 2)
        bearings = summary["bearings"]
        self.assertEqual([b["outcome"] for b in bearings], ["prepare_failed", "dock_failed", None])
        self.assertEqual([b["retreat_success"] for b in bearings], [False, True, None])
        self.assertEqual(bearings[0]["retreat_reason"], RETREAT_REFUSED)
        self.assertEqual(bearings[1]["nav2_goals"], 0)

    def test_a_dock_refused_next_to_an_obstacle_at_the_first_bearing_retreats_too(self) -> None:
        model = ReadinessTrialModel({"object_names": ["can"]})
        code = model.query([])

        code = self.after(
            model,
            code,
            dock_record(dock_failed(f"nav2_failed:{BLOCKED_START}", goal(None, BLOCKED_START))),
        )

        self.assertEqual(code, retreat_program(0))
        code = self.after(model, code, retreat_record())
        self.assertIn(f"dock_to_visible_object('can', approach_bearing_deg={approach(45.0)!r})", code)

    def test_a_refused_retreat_is_recorded_and_the_trial_goes_on(self) -> None:
        model = ReadinessTrialModel({"object_names": ["can"], "bearing_offsets_deg": [0, 45]})
        code = model.query([])
        code = self.after(model, code, dock_record(docked()))
        code = self.after(model, code, prepare_record(prepare_failed(SEARCH_FAILED, queries=5)))
        self.assertEqual(code, retreat_program(0))

        code = self.after(model, code, retreat_record({"success": False, "reason": RETREAT_REFUSED}))

        self.assertNotIn("finish(", code)
        self.assertIn(f"dock_to_visible_object('can', approach_bearing_deg={approach(45.0)!r})", code)
        code = self.after(model, code, dock_record(dock_failed("nav2_failed:aborted", goal(0.6, "aborted"))))
        self.assertIn("finish(", code)
        self.assertIn(
            "trial unsolved after 2 of 2 bearings; bearings [+0: prepare_failed (RuntimeError:no "
            "local base pose has a collision-); +45: dock_failed (nav2_failed:aborted)]; "
            "2 dock calls, 2 Nav2 goals, 5 Pi IK queries, 1 retreats",
            code,
        )
        bearing = model.trial_summary()["bearings"][0]
        self.assertEqual((bearing["retreat_calls"], bearing["retreat_reason"], bearing["retreat_success"]), (1, RETREAT_REFUSED, False))

    def test_a_retreat_error_the_primitive_does_not_report_aborts_the_trial(self) -> None:
        model = ReadinessTrialModel({"object_names": ["can"]})
        code = model.query([])
        code = self.after(model, code, dock_record(docked()))
        code = self.after(model, code, prepare_record(prepare_failed(SEARCH_FAILED, queries=5)))

        code = self.after(
            model, code, record("drive_straight", error="name 'drive_straight' is not defined", error_type="NameError")
        )

        self.assertIn("finish(", code)
        self.assertIn(
            "trial aborted at bearing 0 (offset +0 deg): drive_straight raised NameError: "
            "name 'drive_straight' is not defined; bearings [+0: prepare_failed",
            code,
        )
        # The bearing had ended before the retreat; its own outcome stays.
        bearing = model.trial_summary()["bearings"][0]
        self.assertEqual(bearing["outcome"], "prepare_failed")
        self.assertEqual(bearing["retreat_calls"], 1)
        self.assertFalse(bearing["retreat_success"])

    def test_a_retreat_distance_of_zero_disables_the_retreat(self) -> None:
        model = ReadinessTrialModel({"object_names": ["can"], "retreat_distance_m": 0})
        self.assertEqual(model.retreat_distance_m, 0.0)
        code = model.query([])
        code = self.after(model, code, dock_record(docked()))

        code = self.after(model, code, prepare_record(prepare_failed(SEARCH_FAILED, queries=5)))

        self.assertIn(f"dock_to_visible_object('can', approach_bearing_deg={approach(45.0)!r})", code)
        code = self.after(model, code, dock_record(dock_failed(f"nav2_failed:{BLOCKED_START}", goal(None, BLOCKED_START))))
        self.assertIn(f"dock_to_visible_object('can', approach_bearing_deg={approach(-45.0)!r})", code)
        summary = model.trial_summary()
        self.assertEqual(summary["retreats"], 0)
        self.assertEqual([b["retreat_calls"] for b in summary["bearings"]], [0, 0, 0])

    def test_the_retreat_distance_is_written_as_configured(self) -> None:
        model = ReadinessTrialModel({"object_names": ["can"], "retreat_distance_m": 0.5})
        code = model.query([])
        code = self.after(model, code, dock_record(docked()))

        code = self.after(model, code, prepare_record(prepare_failed(SEARCH_FAILED)))

        self.assertEqual(code, retreat_program(0, 0.5))
        self.assertTrue(code.startswith("result = drive_straight(-0.5)\n"))

    def test_all_bearings_exhausted_finishes_unsolved_with_the_counts(self) -> None:
        model = ReadinessTrialModel({"object_names": ["can"], "bearing_offsets_deg": [0, 45]})
        code = model.query([])
        # Both goals of bearing 0 drove before Nav2 aborted, so bearing 1 is
        # docked after a retreat; nothing follows the last bearing.
        code = self.after_retreat(
            model,
            code,
            dock_record(dock_failed("nav2_failed:aborted", goal(0.6, "aborted"), goal(0.75, "aborted"))),
            bearing=0,
        )
        code = self.after(model, code, dock_record(docked(goals=1)))
        code = self.after(model, code, prepare_record(prepare_failed(MOTION_EXHAUSTED, queries=40)))

        self.assertIn("finish(", code)
        self.assertIn(
            "trial unsolved after 2 of 2 bearings; bearings [+0: dock_failed "
            "(nav2_failed:aborted); +45: prepare_failed (RuntimeError:no certified base "
            "reachable: 3 moti)]; 2 dock calls, 3 Nav2 goals, 40 Pi IK queries, 1 retreats",
            code,
        )
        summary = model.trial_summary()
        self.assertFalse(summary["solved"])
        self.assertIsNone(summary["solved_bearing"])
        self.assertEqual(summary["nav2_goals"], 3)

    def test_the_anchor_is_read_from_the_reference_bearing_then_the_fallbacks(self) -> None:
        by_reference = {"metrics": {"reference_bearing_rad": 0.5, "goal_attempts": [goal(0.6)], "arrival_bearing_rad": 1.0}}
        by_attempt = {"metrics": {"goal_attempts": [{**goal(0.6), "bearing_rad": 0.7}], "arrival_bearing_rad": 1.0}}
        by_arrival = {"metrics": {"arrival_bearing_rad": 1.0}}
        self.assertAlmostEqual(anchor_bearing_deg(by_reference), math.degrees(0.5))
        self.assertAlmostEqual(anchor_bearing_deg(by_attempt), math.degrees(0.7))
        self.assertAlmostEqual(anchor_bearing_deg(by_arrival), math.degrees(1.0))
        self.assertIsNone(anchor_bearing_deg({"metrics": {"backend": "nav2"}}))
        self.assertIsNone(anchor_bearing_deg({}))

        for result, navigated in ((by_attempt, True), (by_arrival, False)):
            with self.subTest(result=result):
                model = ReadinessTrialModel({"object_names": ["can"]})
                code = model.query([])
                code = self.after(model, code, dock_record({"success": False, "reason": "nav2_failed:aborted", **result}))
                if navigated:
                    # The goal drove, so the retreat comes before the next bearing.
                    self.assertEqual(code, retreat_program(0))
                    code = self.after(model, code, retreat_record())
                expected = approach(45.0, anchor_bearing_deg(result))
                self.assertIn(f"approach_bearing_deg={expected!r}", code)

    def test_an_anchor_from_a_later_name_after_a_miss_is_still_used(self) -> None:
        model = ReadinessTrialModel({"object_names": ["can", "soda can"]})
        code = model.query([])
        code = self.after(model, code, record("dock_to_visible_object", failure=DOCK_MISS))
        code = self.after_retreat(
            model, code, dock_record(dock_failed("nav2_failed:aborted", goal(0.6, "aborted"))), bearing=0
        )

        self.assertIn(f"approach_bearing_deg={approach(45.0)!r}", code)

    def test_bearings_wrap_past_180_degrees(self) -> None:
        model = ReadinessTrialModel({"object_names": ["can"], "bearing_offsets_deg": [0, 180, -135]})
        code = model.query([])
        code = self.after(model, code, dock_record(dock_failed("nav2_failed:aborted", bearing_rad=math.radians(170.0))))
        self.assertIn(f"approach_bearing_deg={approach(180.0, 170.0)!r}", code)
        self.assertEqual(approach(180.0, 170.0), -10.0)

        code = self.after(model, code, dock_record(dock_failed("nav2_failed:aborted", bearing_rad=math.radians(170.0))))
        self.assertIn(f"approach_bearing_deg={approach(-135.0, 170.0)!r}", code)
        self.assertEqual(approach(-135.0, 170.0), 35.0)

    def test_only_bearing_zero_is_tried_without_the_bearing_search(self) -> None:
        model = ReadinessTrialModel({"object_names": ["can"], "bearing_offsets_deg": [0]})
        code = model.query([])

        code = self.after(model, code, dock_record(dock_failed("nav2_failed:aborted", goal(0.6, "aborted"))))

        self.assertIn("finish(", code)
        self.assertIn("trial unsolved after 1 of 1 bearings; bearings [+0: dock_failed (nav2_failed:aborted)]", code)

    def test_an_unexpected_error_aborts_the_trial_with_the_error(self) -> None:
        model = ReadinessTrialModel({"object_names": ["can"]})
        code = model.query([])

        code = self.after(
            model,
            code,
            record(
                "dock_to_visible_object",
                error="dock_to_visible_object() got an unexpected keyword argument 'approach_bearing_deg'",
                error_type="TypeError",
            ),
        )

        self.assertIn("finish(", code)
        self.assertIn(
            "trial aborted at bearing 0 (offset +0 deg): dock_to_visible_object raised "
            "TypeError: dock_to_visible_object() got an unexpected keyword argument",
            code,
        )
        self.assertEqual(model.trial_summary()["bearings"][0]["outcome"], "aborted")

        model = ReadinessTrialModel({"object_names": ["can"]})
        code = model.query([])
        code = self.after(model, code, dock_record(docked()))
        code = self.after(model, code, record("prepare_for_manipulation", error="bad arm", error_type="ValueError"))
        self.assertIn("prepare_for_manipulation raised ValueError: bad arm", code)

    def test_a_dock_without_a_suggested_arm_uses_the_default_arm(self) -> None:
        model = ReadinessTrialModel({"object_names": ["can"], "default_arm": "left"})
        code = model.query([])

        code = self.after(model, code, dock_record(docked(arm=None)))

        self.assertIn("arm='left'", code)

    def test_every_program_is_one_extractable_python_policy(self) -> None:
        model = ReadinessTrialModel({"object_names": ["can", "beverage can"]})
        programs = [model.query([])]
        programs.append(self.after(model, programs[-1], record("dock_to_visible_object", failure=DOCK_MISS)))
        programs.append(self.after(model, programs[-1], dock_record(dock_failed("nav2_failed:aborted"))))
        programs.append(self.after(model, programs[-1], dock_record(docked(arm="left"))))
        programs.append(self.after(model, programs[-1], prepare_record(prepare_failed(SEARCH_FAILED))))
        self.assertIn("drive_straight(", programs[-1])
        programs.append(self.after(model, programs[-1], retreat_record()))
        programs.append(self.after(model, programs[-1], dock_record(docked(arm="left"))))
        programs.append(self.after(model, programs[-1], prepare_record(prepared())))

        for program in programs:
            self.assertEqual(extract_code(program), program)

    def test_primitive_outcome_flags_an_executor_error_as_unexpected(self) -> None:
        outcome = primitive_outcome(record("dock_to_visible_object", error="RPC timeout"), "dock_to_visible_object")

        self.assertFalse(outcome.succeeded)
        self.assertFalse(outcome.sam3_miss)
        self.assertTrue(outcome.unexpected_error)
        self.assertIn("RPC timeout", outcome.reason)
        failed = primitive_outcome(dock_record(dock_failed("nav2_failed:aborted")), "dock_to_visible_object")
        self.assertFalse(failed.unexpected_error)

    def test_nav2_goals_and_ik_queries_are_read_from_the_results(self) -> None:
        self.assertEqual(nav2_goal_count(dock_failed("x", goal(0.6, "aborted"), goal(0.75, "aborted"))), 2)
        self.assertEqual(nav2_goal_count(dock_failed("x", goal(None, "no_plannable_goal_on_bearing"))), 0)
        self.assertEqual(nav2_goal_count({"metrics": {"goal_attempts": "<list len=40>"}}), 40)
        self.assertEqual(nav2_goal_count({"success": True}), 0)
        self.assertEqual(prepare_ik_query_count(prepared(96)), 96)
        self.assertEqual(prepare_ik_query_count(prepare_failed(SEARCH_FAILED, queries=9, diagnostics=True)), 9)
        self.assertEqual(prepare_ik_query_count(prepare_failed(MOTION_REFUSED, queries=6)), 6)
        self.assertEqual(prepare_ik_query_count({"success": False, "pi_ik_query_log": "<list len=512>"}), 512)
        self.assertEqual(prepare_ik_query_count(too_far()), 0)

    def test_a_goal_attempt_counts_as_sent_by_its_nav2_result_else_by_its_pose_and_reason(self) -> None:
        # With a per-attempt Nav2 result, that result alone decides.
        self.assertTrue(attempt_sent_nav2_goal(goal(0.6, "operator_stop", nav2={"success": False})))
        self.assertFalse(attempt_sent_nav2_goal(goal(0.6, "operator_stop", nav2=None)))
        self.assertTrue(attempt_sent_nav2_goal({"nav2": {"success": True}}))
        # Without it: a navigated goal counts, an attempt that never drove
        # (no pose: the plan check failed everywhere, or the operator's stop
        # came first) does not, and an operator stop is taken as pre-empting.
        self.assertTrue(attempt_sent_nav2_goal(goal(0.6, "aborted")))
        self.assertTrue(attempt_sent_nav2_goal(goal(0.9)))
        self.assertFalse(attempt_sent_nav2_goal(goal(None, "no_plannable_goal_on_bearing")))
        self.assertFalse(attempt_sent_nav2_goal(goal(None, "operator_stop")))
        self.assertFalse(attempt_sent_nav2_goal(goal(0.6, "operator_stop")))
        self.assertFalse(attempt_sent_nav2_goal({"reason": "aborted"}))
        self.assertTrue(attempt_sent_nav2_goal("<dict>"))
        # A fallback dock that succeeded without sending a goal records no pose.
        self.assertFalse(attempt_sent_nav2_goal(goal(None, "succeeded")))
        self.assertEqual(
            nav2_goal_count(
                dock_failed(
                    "nav2_failed:operator_stop",
                    goal(0.6, "aborted"),
                    goal(None, "operator_stop"),
                )
            ),
            1,
        )

    def test_a_dock_navigated_when_any_goal_attempt_drove(self) -> None:
        # The entry's own flag decides, whatever the goal then ended as.
        self.assertTrue(dock_navigated(dock_failed(STALLED, goal(0.6, "stalled_no_progress"))))
        self.assertTrue(dock_navigated(dock_failed(FINAL_CHECK_FAILED, goal(0.6))))
        self.assertTrue(dock_navigated(dock_failed("nav2_failed:aborted", goal(None, "no_plannable_goal_on_bearing"), goal(0.6, "aborted"))))
        self.assertTrue(dock_navigated(docked(goals=2)))
        self.assertFalse(dock_navigated(dock_failed("no_plannable_goal_on_bearing", goal(None, "no_plannable_goal_on_bearing"), goal(None, "no_plannable_goal_on_bearing"))))
        self.assertFalse(dock_navigated(dock_failed("nav2_failed:operator_stop", goal(None, "operator_stop"))))
        # A pose alone is not the flag: only what docking wrote counts.
        self.assertFalse(dock_navigated(dock_failed("nav2_failed:aborted", goal(0.6, "aborted", navigated=False))))
        self.assertFalse(dock_navigated({"success": False, "reason": DOCK_MISS}))
        self.assertFalse(dock_navigated({"metrics": {"backend": "nav2"}}))
        self.assertFalse(dock_navigated({"metrics": {"goal_attempts": []}}))
        # A trace-collapsed list counts as navigated, as nav2_goal_count counts it as sent.
        self.assertTrue(dock_navigated({"metrics": {"goal_attempts": "<list len=2>"}}))
        self.assertFalse(dock_navigated({"metrics": {"goal_attempts": "<list len=0>"}}))
        self.assertTrue(dock_navigated({"metrics": {"goal_attempts": ["<dict>"]}}))
        outcome = primitive_outcome(dock_record(dock_failed(STALLED, goal(0.6, "stalled_no_progress"))), "dock_to_visible_object")
        self.assertTrue(outcome.navigated)
        self.assertFalse(outcome.blocked_start)
        self.assertFalse(primitive_outcome(record("dock_to_visible_object", failure=DOCK_MISS), "dock_to_visible_object").navigated)
        # The refusal before any goal was sent is the blocked-start case alone.
        refused = primitive_outcome(
            dock_record(dock_failed(f"nav2_failed:{BLOCKED_START}", goal(None, BLOCKED_START))), "dock_to_visible_object"
        )
        self.assertFalse(refused.navigated)
        self.assertTrue(refused.blocked_start)

    def test_a_stop_request_releases_the_turn(self) -> None:
        stop_event = threading.Event()
        stop_event.set()
        model = ReadinessTrialModel(
            {"object_names": ["can"]}, stop_event=stop_event, stop_reason=lambda: "operator pressed stop"
        )

        with self.assertRaises(Stopped) as raised:
            model.query([])

        self.assertEqual(raised.exception.reason, "operator pressed stop")

    def test_the_turn_budget_finishes_unsolved_on_the_last_turn(self) -> None:
        # Budget 5: dock, prepare, retreat, dock; the fifth program must be
        # the finish even though the protocol would prepare again.
        model = ReadinessTrialModel({"object_names": ["can"], "max_turns": 5})
        code = model.query([])
        self.assertIn("dock_to_visible_object('can')", code)
        code = self.after(model, code, dock_record(docked(arm="left")))
        self.assertIn("prepare_for_manipulation('can', arm='left')", code)
        code = self.after_retreat(
            model, code, prepare_record(prepare_failed(MOTION_EXHAUSTED, queries=30)), bearing=0
        )
        self.assertIn("dock_to_visible_object('can', approach_bearing_deg", code)

        code = self.after(model, code, dock_record(docked(arm="right")))

        self.assertIn("finish(", code)
        self.assertNotIn("prepare_for_manipulation(", code)
        self.assertIn(
            "trial unsolved: turn budget reached (5 turns) at bearing 1 (offset +45 deg), "
            "2 of 8 bearings started; bearings [+0: prepare_failed (RuntimeError:no certified "
            "base reachable: 3 moti); +45: turn_budget]; 2 dock calls, 2 Nav2 goals, 30 Pi IK queries, "
            "1 retreats",
            code,
        )
        self.assertEqual(model.n_calls, 5)
        summary = model.trial_summary()
        self.assertFalse(summary["solved"])
        self.assertEqual(summary["bearings"][1]["outcome"], "turn_budget")
        self.assertEqual(summary["bearings"][1]["dock_calls"], 1)
        # Once finished, the finish program is repeated.
        self.assertIn("finish(", self.after(model, code, {}))

    def test_a_budget_hit_when_a_bearing_would_start_marks_that_bearing(self) -> None:
        model = ReadinessTrialModel({"object_names": ["can"], "max_turns": 4})
        code = model.query([])
        code = self.after(model, code, dock_record(docked()))
        code = self.after_retreat(
            model, code, prepare_record(prepare_failed(SEARCH_FAILED, queries=5)), bearing=0
        )

        self.assertIn("finish(", code)
        self.assertIn("turn budget reached (4 turns) at bearing 1", code)
        self.assertEqual([b["outcome"] for b in model.trial_summary()["bearings"]], ["prepare_failed", "turn_budget"])
        self.assertEqual(model.trial_summary()["bearings"][1]["dock_calls"], 0)

    def test_a_budget_hit_when_the_retreat_would_be_the_last_turn_keeps_the_bearing_outcome(self) -> None:
        model = ReadinessTrialModel({"object_names": ["can"], "max_turns": 3})
        code = model.query([])
        code = self.after(model, code, dock_record(docked()))

        code = self.after(model, code, prepare_record(prepare_failed(SEARCH_FAILED, queries=5)))

        self.assertIn("finish(", code)
        self.assertNotIn("drive_straight(", code)
        self.assertIn("turn budget reached (3 turns) at bearing 0 (offset +0 deg), 1 of 8 bearings started", code)
        summary = model.trial_summary()
        self.assertEqual([b["outcome"] for b in summary["bearings"]], ["prepare_failed"])
        self.assertEqual(summary["retreats"], 0)

    def test_a_natural_finish_on_the_last_turn_keeps_its_own_reason(self) -> None:
        model = ReadinessTrialModel({"object_names": ["can"], "max_turns": 3})
        code = model.query([])
        code = self.after(model, code, dock_record(docked()))

        code = self.after(model, code, prepare_record(prepared(queries=12)))

        self.assertIn("trial solved at bearing 0", code)
        self.assertNotIn("turn budget", code)

    def test_a_budget_of_one_turn_finishes_at_once_and_none_leaves_the_budget_to_the_loop(self) -> None:
        model = ReadinessTrialModel({"object_names": ["can"], "max_turns": 1})
        code = model.query([])
        self.assertIn("finish(", code)
        self.assertIn("turn budget reached (1 turns) at bearing 0", code)
        self.assertEqual(model.trial_summary()["dock_calls"], 0)

        model = ReadinessTrialModel({"object_names": ["can"]})
        self.assertIsNone(model.max_turns)
        code = model.query([])
        for _ in range(5):
            code = self.after(
                model,
                code,
                dock_record(dock_failed("no_plannable_goal_on_bearing", goal(None, "no_plannable_goal_on_bearing"))),
            )
        self.assertIn("dock_to_visible_object(", code)

    def test_invalid_configuration_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ReadinessTrialModel({"object_names": []})
        with self.assertRaises(ValueError):
            ReadinessTrialModel({"object_names": ["can"], "max_turns": 0})
        with self.assertRaises(ValueError):
            ReadinessTrialModel({"object_names": ["can"], "max_turns": True})
        with self.assertRaises(ValueError):
            ReadinessTrialModel({"object_names": ["can"], "max_turns": "3"})
        with self.assertRaises(ValueError):
            ReadinessTrialModel({"object_names": "can"})
        with self.assertRaises(ValueError):
            ReadinessTrialModel({"object_names": ["can"], "name_rounds": 0})
        with self.assertRaises(ValueError):
            ReadinessTrialModel({"object_names": ["can"], "bearing_offsets_deg": [45, 0]})
        with self.assertRaises(ValueError):
            ReadinessTrialModel({"object_names": ["can"], "bearing_offsets_deg": []})
        with self.assertRaises(ValueError):
            ReadinessTrialModel({"object_names": ["can"], "bearing_offsets_deg": [0, float("nan")]})
        with self.assertRaises(ValueError):
            ReadinessTrialModel({"object_names": ["can"], "bearing_offsets_deg": "0,45"})
        with self.assertRaises(ValueError):
            ReadinessTrialModel({"object_names": ["can"], "certification_retries": -1})
        with self.assertRaises(ValueError):
            ReadinessTrialModel({"object_names": ["can"], "certification_retries": True})
        for retreat in (True, "0.3", -0.1, 1.5, float("nan"), float("inf"), None):
            with self.subTest(retreat=retreat), self.assertRaises(ValueError):
                ReadinessTrialModel({"object_names": ["can"], "retreat_distance_m": retreat})
        self.assertEqual(normalize_object_names([" can ", "can", "soda can"]), ("can", "soda can"))
        self.assertEqual(normalize_bearing_offsets_deg([0, 45, -45]), (0.0, 45.0, -45.0))
        self.assertEqual(normalize_retreat_distance_m(0), 0.0)
        self.assertEqual(normalize_retreat_distance_m(1), 1.0)
        self.assertEqual(ReadinessTrialModel({"object_names": ["can"]}).retreat_distance_m, 0.3)
        self.assertEqual(DEFAULT_RETREAT_DISTANCE_M, 0.3)

    def test_wrap_degrees_keeps_angles_in_the_half_open_range(self) -> None:
        self.assertEqual([wrap_degrees(v) for v in (0, 180, -180, 200, -200, 540)], [0.0, 180.0, 180.0, -160.0, 160.0, 180.0])

    def test_build_model_selects_the_trial_model_and_a_plain_llm_refuses(self) -> None:
        stop_event = threading.Event()

        model = build_model(
            {"provider": "scripted", "object_names": ["can"], "bearing_offsets_deg": [0, 90]},
            stop_event=stop_event,
        )

        self.assertIsInstance(model, ReadinessTrialModel)
        self.assertIs(model._stop_event, stop_event)
        self.assertEqual(model.bearing_offsets_deg, (0.0, 90.0))
        with self.assertRaises(ModelError):
            LLM({"provider": "scripted"}).query([])


class ScriptedPrimitives:
    """Fake dock, prepare and retreat primitives whose outcomes are scripted per call.

    Without a ``retreat`` script every ``drive_straight`` call succeeds.
    """

    def __init__(
        self, dock: list[dict], prepare: list[dict], retreat: list[dict] | None = None
    ) -> None:
        self.dock_script = list(dock)
        self.prepare_script = list(prepare)
        self.retreat_script = None if retreat is None else list(retreat)
        self.calls: list[tuple] = []

    def register(self, registry: PrimitiveRegistry) -> None:
        primitives = self

        def dock_to_visible_object(object_name: str, *, approach_bearing_deg: float | None = None) -> dict:
            """Fake docking."""

            primitives.calls.append(("dock", object_name, approach_bearing_deg))
            return primitives._outcome("dock_to_visible_object", primitives.dock_script)

        def prepare_for_manipulation(object_name: str, *, arm: str) -> dict:
            """Fake readiness search."""

            primitives.calls.append(("prepare", object_name, arm))
            return primitives._outcome("prepare_for_manipulation", primitives.prepare_script)

        def drive_straight(
            distance_m: float, *, max_speed_mps: float | None = None, timeout_s: float | None = None
        ) -> dict:
            """Fake straight drive; the trial reverses with it."""

            primitives.calls.append(("retreat", distance_m))
            if primitives.retreat_script is None:
                return retreated(distance_m)
            return primitives._outcome("drive_straight", primitives.retreat_script)

        registry.register("dock_to_visible_object", dock_to_visible_object)
        registry.register("prepare_for_manipulation", prepare_for_manipulation)
        registry.register("drive_straight", drive_straight)

    @staticmethod
    def _outcome(primitive: str, script: list[dict]) -> dict:
        if not script:
            raise AssertionError(f"{primitive} was called more often than scripted")
        result = script.pop(0)
        if result.get("success", False):
            return result
        raise PrimitiveFailed(primitive, result)


class ReadinessTrialLoopTest(unittest.TestCase):
    """The trial model drives the unchanged DefaultAgent loop and executor."""

    def run_trial(self, primitives: ScriptedPrimitives, model_config: dict, *, max_turns: int = 20) -> dict:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        environment = make_environment(FakeHardware())
        self.addCleanup(environment.safe_shutdown)
        registry = PrimitiveRegistry()
        primitives.register(registry)
        trace = Trace(output_dir=Path(tmp.name))
        executor = PolicyExecutor(registry, on_primitive_call=trace.record_primitive_call)
        model = ReadinessTrialModel(model_config)
        agent = DefaultAgent(model, environment, executor, trace, max_turns=max_turns)
        result = agent.run("Readiness trial")
        result["trace"] = json.loads(trace.path.read_text())
        result["summary"] = model.trial_summary()
        return result

    def test_a_sam3_miss_falls_back_and_the_trial_is_solved_at_bearing_zero(self) -> None:
        primitives = ScriptedPrimitives(
            dock=[{"success": False, "reason": DOCK_MISS}, docked(arm="left")],
            prepare=[prepared(96)],
        )

        result = self.run_trial(primitives, {"object_names": ["can", "beverage can"]})

        self.assertEqual(result["status"], "finished")
        self.assertEqual(result["turns"], 4)
        self.assertIn("trial solved at bearing 0", result["reason"])
        self.assertIn("2 dock calls, 1 Nav2 goals, 96 Pi IK queries, 0 retreats", result["reason"])
        self.assertEqual(
            primitives.calls,
            [("dock", "can", None), ("dock", "beverage can", None), ("prepare", "can", "left")],
        )
        self.assertEqual(result["trace"]["finish"]["reason"], result["reason"])
        self.assertIn("'trial': {'solved': True", result["trace"]["policies"][-1]["execution"]["stdout"])

    def test_an_unplannable_bearing_zero_is_followed_by_the_anchor_plus_45(self) -> None:
        primitives = ScriptedPrimitives(
            dock=[
                dock_failed("no_plannable_goal_on_bearing", goal(None, "no_plannable_goal_on_bearing")),
                {**docked(arm="right"), "reason": "docked_at_fallback_distance", "metrics": dock_metrics(goal(0.75, "aborted"), goal(0.9), goal_distance_used_m=0.9)},
            ],
            prepare=[prepared(120)],
        )

        result = self.run_trial(primitives, {"object_names": ["can"]})

        self.assertEqual(result["status"], "finished")
        # Bearing 0 never drove, so bearing 1 is docked without a retreat.
        self.assertEqual(result["turns"], 4)
        self.assertIn("trial solved at bearing 1 (offset +45 deg)", result["reason"])
        self.assertIn("bearings [+0: dock_failed (no_plannable_goal_on_bearing); +45: solved]", result["reason"])
        self.assertIn("2 dock calls, 2 Nav2 goals, 120 Pi IK queries, 0 retreats", result["reason"])
        self.assertEqual(primitives.calls[0], ("dock", "can", None))
        self.assertEqual(primitives.calls[1], ("dock", "can", approach(45.0)))
        self.assertEqual(primitives.calls[2], ("prepare", "can", "right"))
        # The requested bearing is what the trace records for the report.
        self.assertEqual(result["trace"]["primitive_calls"][1]["kwargs"], {"approach_bearing_deg": approach(45.0)})

    def test_a_prepare_search_failure_moves_to_the_next_bearing_where_it_is_solved(self) -> None:
        primitives = ScriptedPrimitives(
            dock=[docked(arm="left"), docked(arm="left")],
            prepare=[prepare_failed(SEARCH_FAILED, queries=512, diagnostics=True), prepared(64)],
        )

        result = self.run_trial(primitives, {"object_names": ["can"]})

        self.assertEqual(result["status"], "finished")
        self.assertIn("trial solved at bearing 1 (offset +45 deg)", result["reason"])
        self.assertIn("2 dock calls, 2 Nav2 goals, 576 Pi IK queries, 1 retreats", result["reason"])
        self.assertEqual(
            [call[0] for call in primitives.calls], ["dock", "prepare", "retreat", "dock", "prepare"]
        )
        self.assertEqual(primitives.calls[2], ("retreat", -0.3))
        self.assertEqual(primitives.calls[3], ("dock", "can", approach(45.0)))
        self.assertEqual(result["turns"], 6)
        # The retreat is in the trace as a plain drive_straight call.
        retreat = result["trace"]["primitive_calls"][2]
        self.assertEqual((retreat["name"], retreat["args"], retreat["kwargs"]), ("drive_straight", [-0.3], {}))
        self.assertEqual(result["summary"]["bearings"][0]["retreat_reason"], "reached")

    def test_a_refused_retreat_is_traced_and_the_next_bearing_is_still_docked(self) -> None:
        primitives = ScriptedPrimitives(
            dock=[docked(arm="left"), docked(arm="left")],
            prepare=[prepare_failed(SEARCH_FAILED, queries=12), prepared(8)],
            retreat=[{"success": False, "reason": RETREAT_REFUSED, "progress_m": -0.02}],
        )

        result = self.run_trial(primitives, {"object_names": ["can"]})

        self.assertEqual(result["status"], "finished")
        self.assertIn("trial solved at bearing 1 (offset +45 deg)", result["reason"])
        self.assertIn("2 dock calls, 2 Nav2 goals, 20 Pi IK queries, 1 retreats", result["reason"])
        self.assertEqual(
            [call[0] for call in primitives.calls], ["dock", "prepare", "retreat", "dock", "prepare"]
        )
        retreat = result["trace"]["primitive_calls"][2]
        self.assertEqual(retreat["error"], {"type": "PrimitiveFailed", "message": RETREAT_REFUSED})
        bearing = result["summary"]["bearings"][0]
        self.assertEqual((bearing["retreat_reason"], bearing["retreat_success"]), (RETREAT_REFUSED, False))

    def test_a_stalled_dock_is_followed_by_a_retreat_before_the_next_bearing_is_docked(self) -> None:
        primitives = ScriptedPrimitives(
            dock=[
                dock_failed(STALLED, goal(0.6, "stalled_no_progress"), goal(0.75, "stalled_no_progress")),
                docked(arm="left"),
            ],
            prepare=[prepared(40)],
        )

        result = self.run_trial(primitives, {"object_names": ["can"]})

        self.assertEqual(result["status"], "finished")
        self.assertIn("trial solved at bearing 1 (offset +45 deg)", result["reason"])
        self.assertIn(f"bearings [+0: dock_failed ({STALLED[:48]}); +45: solved]", result["reason"])
        self.assertIn("2 dock calls, 3 Nav2 goals, 40 Pi IK queries, 1 retreats", result["reason"])
        self.assertEqual(
            primitives.calls,
            [("dock", "can", None), ("retreat", -0.3), ("dock", "can", approach(45.0)), ("prepare", "can", "left")],
        )
        self.assertEqual(result["turns"], 5)
        bearings = result["summary"]["bearings"]
        self.assertEqual([b["retreat_calls"] for b in bearings], [1, 0])
        self.assertEqual((bearings[0]["retreat_reason"], bearings[0]["retreat_success"]), ("reached", True))

    def test_motion_exhausted_then_certification_retry_then_every_bearing_used(self) -> None:
        primitives = ScriptedPrimitives(
            dock=[docked(arm="left"), docked(arm="right"), {"success": False, "reason": DOCK_MISS}],
            prepare=[
                prepare_failed(MOTION_EXHAUSTED, queries=30),
                prepare_failed(CERTIFICATION_FAILED, queries=20),
                prepare_failed(CERTIFICATION_FAILED, queries=25),
            ],
        )

        result = self.run_trial(
            primitives, {"object_names": ["can"], "bearing_offsets_deg": [0, 45, -45]}
        )

        self.assertEqual(result["status"], "finished")
        self.assertIn("trial unsolved after 3 of 3 bearings", result["reason"])
        self.assertIn(
            "bearings [+0: prepare_failed (RuntimeError:no certified base reachable: 3 moti); "
            "+45: certification_failed (actual_pose_grasp_certification_failed); -45: dock_names_exhausted]",
            result["reason"],
        )
        self.assertIn("3 dock calls, 2 Nav2 goals, 75 Pi IK queries, 2 retreats", result["reason"])
        self.assertEqual(
            primitives.calls,
            [
                ("dock", "can", None),
                ("prepare", "can", "left"),
                ("retreat", -0.3),
                ("dock", "can", approach(45.0)),
                ("prepare", "can", "right"),
                ("prepare", "can", "right"),
                ("retreat", -0.3),
                ("dock", "can", approach(-45.0)),
            ],
        )
        self.assertEqual(result["turns"], 9)

    def test_a_run_that_would_overrun_the_budget_ends_with_a_finish_reason(self) -> None:
        # Every dock succeeds and every prepare fails: with 8 bearings the
        # protocol wants 24 turns (a retreat before every bearing after the
        # first); the budget of 8 ends it with finish() on turn 8 instead of
        # the loop's max_turns cut-off.
        primitives = ScriptedPrimitives(
            dock=[docked(arm="left")] * 3,
            prepare=[prepare_failed(SEARCH_FAILED, queries=10)] * 3,
        )

        result = self.run_trial(primitives, {"object_names": ["can"], "max_turns": 8}, max_turns=8)

        self.assertEqual(result["status"], "finished")
        self.assertEqual(result["turns"], 8)
        self.assertIn("trial unsolved: turn budget reached (8 turns) at bearing 2 (offset -45 deg)", result["reason"])
        self.assertIn("bearings [+0: prepare_failed", result["reason"])
        self.assertIn("-45: turn_budget]", result["reason"])
        self.assertIn("3 dock calls, 3 Nav2 goals, 20 Pi IK queries, 2 retreats", result["reason"])
        self.assertEqual(result["trace"]["finish"]["reason"], result["reason"])
        self.assertIn("'trial': {'solved': False", result["trace"]["policies"][-1]["execution"]["stdout"])
        self.assertEqual(
            [call[0] for call in primitives.calls],
            ["dock", "prepare", "retreat", "dock", "prepare", "retreat", "dock"],
        )

    def test_without_a_model_budget_the_loop_cuts_the_run_off_as_before(self) -> None:
        primitives = ScriptedPrimitives(
            dock=[docked(arm="left")] * 3,
            prepare=[prepare_failed(SEARCH_FAILED, queries=10)] * 3,
        )

        result = self.run_trial(primitives, {"object_names": ["can"]}, max_turns=5)

        # The loop's own cut-off: no trial finish reason, no printed summary.
        self.assertEqual(result["status"], "max_turns")
        self.assertEqual(result["trace"]["finish"]["reason"], "turn budget exhausted after 5 turns")
        self.assertNotIn("trial unsolved", result["reason"])
        self.assertNotIn("'trial':", result["trace"]["policies"][-1]["execution"]["stdout"])
        self.assertEqual(
            [call[0] for call in primitives.calls], ["dock", "prepare", "retreat", "dock", "prepare"]
        )

    def test_a_stop_during_the_run_never_starts_another_bearing(self) -> None:
        stop_event = threading.Event()
        primitives = ScriptedPrimitives(
            dock=[dock_failed("nav2_failed:operator_stop", goal(0.6, "operator_stop"))], prepare=[]
        )
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        environment = make_environment(FakeHardware())
        self.addCleanup(environment.safe_shutdown)
        registry = PrimitiveRegistry()
        primitives.register(registry)
        trace = Trace(output_dir=Path(tmp.name))

        def stopping_dock(event: dict) -> None:
            trace.record_primitive_call(event)
            stop_event.set()

        executor = PolicyExecutor(registry, on_primitive_call=stopping_dock)
        model = ReadinessTrialModel({"object_names": ["can"]}, stop_event=stop_event)
        agent = DefaultAgent(model, environment, executor, trace, max_turns=10, stop_event=stop_event)

        result = agent.run("Readiness trial")

        # The dock navigated, so the model would retreat next; the stop ends
        # the run before that program is asked for.
        self.assertEqual(result["status"], "stopped")
        self.assertEqual(primitives.calls, [("dock", "can", None)])


class TrialConfigTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_config(TASKS, task_id="find_can_on_table")

    def test_the_trial_fixes_the_model_and_changes_only_the_listed_settings(self) -> None:
        output = Path(tempfile.gettempdir()) / "readiness-trial-test"

        result = trial_config(
            self.config,
            object_names=DEFAULT_OBJECT_NAMES,
            name_rounds=2,
            pi_ik_compute_budget_s=TRIAL_PI_IK_COMPUTE_BUDGET_S,
            output_dir=output,
        )

        turns = trial_turn_budget(
            name_count=len(DEFAULT_OBJECT_NAMES),
            name_rounds=2,
            bearing_count=len(DEFAULT_BEARING_OFFSETS_DEG),
            certification_retries=1,
            retreat_distance_m=TRIAL_RETREAT_DISTANCE_M,
        )
        self.assertEqual(
            result["model"],
            {
                "provider": "scripted",
                "name": "readiness-trial",
                "object_names": list(DEFAULT_OBJECT_NAMES),
                "name_rounds": 2,
                "bearing_offsets_deg": list(DEFAULT_BEARING_OFFSETS_DEG),
                "certification_retries": 1,
                "retreat_distance_m": TRIAL_RETREAT_DISTANCE_M,
                "max_turns": turns,
            },
        )
        self.assertEqual(TRIAL_RETREAT_DISTANCE_M, 0.30)
        self.assertEqual(TRIAL_ARM_SIDE_HEADING_OFFSET_DEG, 10.0)
        self.assertFalse(result["navigation_planner"]["enabled"])
        self.assertNotIn("memory_path", result["navigation_planner"])

        before = primitive_settings(self.config["primitive_config"], "prepare_for_manipulation")
        after = primitive_settings(result["primitive_config"], "prepare_for_manipulation")
        changed = {key for key in set(before) | set(after) if before.get(key) != after.get(key)}
        # The shipped config already has the trial's IK budget, motion
        # alternatives and resumed search; only the initial distance widens.
        self.assertEqual(changed, {"maximum_initial_target_distance_m"})
        self.assertEqual(after["pi_ik_compute_budget_s"], TRIAL_PI_IK_COMPUTE_BUDGET_S)
        self.assertEqual(after["maximum_initial_target_distance_m"], TRIAL_MAXIMUM_INITIAL_TARGET_DISTANCE_M)
        self.assertEqual(after["motion_alternative_limit"], TRIAL_MOTION_ALTERNATIVE_LIMIT)
        self.assertIs(after["continue_search_after_refused_motions"], True)
        self.assertIs(TRIAL_CONTINUE_SEARCH_AFTER_REFUSED_MOTIONS, True)
        # The primitives validate their own settings only when the run starts,
        # so the trial's values have to pass that validation here, not at Start.
        from yor_agent.robot.manipulation_readiness import ManipulationReadinessConfig
        from yor_agent.robot.nav2_visible_object_navigation import Nav2VisibleObjectDockingConfig

        ManipulationReadinessConfig.from_mapping(after)

        before = primitive_settings(self.config["primitive_config"], "dock_to_visible_object")
        after = primitive_settings(result["primitive_config"], "dock_to_visible_object")
        changed = {key for key in set(before) | set(after) if before.get(key) != after.get(key)}
        self.assertEqual(
            changed,
            {
                "nav2_goal_distance_fallbacks_m",
                "plan_check_before_goal",
                "explicit_bearing_alignment_tolerance_deg",
                "arm_side_heading_offset_deg",
                "readiness_prior",
            },
        )
        self.assertEqual(after["nav2_goal_distance_fallbacks_m"], list(TRIAL_NAV2_GOAL_DISTANCE_FALLBACKS_M))
        self.assertTrue(after["plan_check_before_goal"])
        self.assertEqual(after["explicit_bearing_alignment_tolerance_deg"], 22.5)
        self.assertEqual(after["arm_side_heading_offset_deg"], 10.0)
        prior_before, prior_after = before["readiness_prior"], after["readiness_prior"]
        changed = {key for key in set(prior_before) | set(prior_after) if prior_before.get(key) != prior_after.get(key)}
        self.assertEqual(
            changed, {"retry_bearing_offsets_deg", "fallback_to_arrival_bearing", "event_object"}
        )
        self.assertEqual(prior_after["retry_bearing_offsets_deg"], [])
        self.assertFalse(prior_after["fallback_to_arrival_bearing"])
        self.assertEqual(prior_after["event_object"], "can")
        docking = Nav2VisibleObjectDockingConfig.from_mapping(after)
        self.assertEqual(docking.arm_side_heading_offset_deg, 10.0)

        self.assertEqual(result["trace"]["output_dir"], str(output))
        # The default protocol overruns the Web UI's cap once the retreats and
        # the sixth name are counted, so the model's budget is the cap itself.
        self.assertEqual(result["agent"]["max_turns"], turns)
        self.assertEqual(turns, WEB_UI_MAX_TURNS)
        self.assertEqual(len(DEFAULT_OBJECT_NAMES), 6)
        # Bottle is the first fallback after the can prompt.
        self.assertEqual(DEFAULT_OBJECT_NAMES[:2], ("can", "bottle"))
        self.assertEqual((6 * 2 + 6 + 1) + 7 * (6 + 6 + 1 + 1) + 1, 118)

    def test_the_start_label_is_recorded_in_the_run_config(self) -> None:
        options = dict(
            object_names=DEFAULT_OBJECT_NAMES,
            name_rounds=2,
            pi_ik_compute_budget_s=TRIAL_PI_IK_COMPUTE_BUDGET_S,
            output_dir=Path(tempfile.gettempdir()) / "readiness-trial-test",
        )

        labelled = trial_config(self.config, start_label=" S45 ", **options)
        unlabelled = trial_config(self.config, **options)

        self.assertEqual(labelled["trial"]["start_label"], "S45")
        self.assertNotIn("trial", unlabelled)
        for bad in ("", "   ", 45, "x" * 65):
            with self.subTest(label=bad), self.assertRaises(ValueError):
                trial_config(self.config, start_label=bad, **options)

    def test_the_retreat_stays_within_the_reverse_step_the_controller_allows(self) -> None:
        # drive_straight rejects a reverse step outside the navigation
        # settings' (distance_tolerance_m, max_reverse_distance_m]; the trial
        # would only meet that as an aborted run, so it is refused at launch.
        from yor_agent.experiments.readiness_trial import (
            check_retreat_within_reverse_limits,
        )

        config = {"robot": {"navigation": {"max_reverse_distance_m": 0.15}}}
        check_retreat_within_reverse_limits(config, 0.15)
        check_retreat_within_reverse_limits(config, 0.0)
        with self.assertRaises(ValueError):
            check_retreat_within_reverse_limits(config, 0.30)
        with self.assertRaises(ValueError):
            check_retreat_within_reverse_limits(config, 0.02)
        # The loaded task config carries the same limit, so the shipped
        # default passes and a larger retreat is refused by trial_config.
        trial_config(
            self.config,
            object_names=DEFAULT_OBJECT_NAMES,
            name_rounds=2,
            pi_ik_compute_budget_s=TRIAL_PI_IK_COMPUTE_BUDGET_S,
            output_dir="/tmp/trial",
            retreat_distance_m=TRIAL_RETREAT_DISTANCE_M,
        )
        with self.assertRaises(ValueError):
            trial_config(
                self.config,
                object_names=DEFAULT_OBJECT_NAMES,
                name_rounds=2,
                pi_ik_compute_budget_s=TRIAL_PI_IK_COMPUTE_BUDGET_S,
                output_dir="/tmp/trial",
                retreat_distance_m=0.35,
            )

    def test_the_trial_settings_can_be_changed_from_the_launcher(self) -> None:
        result = trial_config(
            self.config,
            object_names=DEFAULT_OBJECT_NAMES,
            name_rounds=2,
            pi_ik_compute_budget_s=TRIAL_PI_IK_COMPUTE_BUDGET_S,
            output_dir=Path(tempfile.gettempdir()) / "readiness-trial-test",
            retreat_distance_m=0,
            arm_side_heading_offset_deg=0,
        )

        self.assertEqual(result["model"]["retreat_distance_m"], 0.0)
        self.assertEqual(
            primitive_settings(result["primitive_config"], "dock_to_visible_object")["arm_side_heading_offset_deg"],
            0.0,
        )
        # The bottle name keeps the default protocol above the cap even
        # without retreats; with the five can names alone it is back under it.
        self.assertEqual(result["agent"]["max_turns"], WEB_UI_MAX_TURNS)
        self.assertEqual(result["model"]["max_turns"], WEB_UI_MAX_TURNS)
        five = trial_config(
            self.config,
            object_names=tuple(name for name in DEFAULT_OBJECT_NAMES if name != "bottle"),
            name_rounds=2,
            pi_ik_compute_budget_s=TRIAL_PI_IK_COMPUTE_BUDGET_S,
            output_dir=Path(tempfile.gettempdir()) / "readiness-trial-test",
            retreat_distance_m=0,
            arm_side_heading_offset_deg=0,
        )
        self.assertEqual(five["agent"]["max_turns"], 94)
        self.assertEqual(five["model"]["max_turns"], 94)
        for retreat, heading in ((-0.1, 10.0), (True, 10.0), ("0.3", 10.0), (0.3, "10"), (0.3, float("nan")), (0.3, True)):
            with self.subTest(retreat=retreat, heading=heading), self.assertRaises(ValueError):
                trial_config(
                    self.config,
                    object_names=("can",),
                    name_rounds=1,
                    pi_ik_compute_budget_s=TRIAL_PI_IK_COMPUTE_BUDGET_S,
                    output_dir=Path(tempfile.gettempdir()) / "readiness-trial-test",
                    retreat_distance_m=retreat,
                    arm_side_heading_offset_deg=heading,
                )
        # A heading offset the docking settings refuse fails at launch too.
        with self.assertRaisesRegex(ValueError, "arm_side_heading_offset_deg"):
            trial_config(
                self.config,
                object_names=("can",),
                name_rounds=1,
                pi_ik_compute_budget_s=TRIAL_PI_IK_COMPUTE_BUDGET_S,
                output_dir=Path(tempfile.gettempdir()) / "readiness-trial-test",
                arm_side_heading_offset_deg=90.0,
            )

    def test_the_turn_budget_covers_every_fallback_and_is_capped_for_the_web_ui(self) -> None:
        # First bearing: names x rounds docks; later bearings: every name once;
        # every bearing: every name once as a prepare fallback plus the
        # certification retries; one turn to finish.
        self.assertEqual(
            trial_turn_budget(name_count=1, name_rounds=1, bearing_count=2, certification_retries=1),
            (1 + 1 + 1) + (1 + 1 + 1) + 1,
        )
        self.assertEqual(
            trial_turn_budget(name_count=3, name_rounds=2, bearing_count=3, certification_retries=1),
            (3 * 2 + 3 + 1) + 2 * (3 + 3 + 1) + 1,
        )
        self.assertEqual(
            trial_turn_budget(name_count=5, name_rounds=2, bearing_count=8, certification_retries=1),
            (5 * 2 + 5 + 1) + 7 * (5 + 5 + 1) + 1,
        )
        self.assertEqual(
            trial_turn_budget(name_count=5, name_rounds=2, bearing_count=10, certification_retries=1),
            WEB_UI_MAX_TURNS,
        )
        self.assertGreater((5 * 2 + 5 + 1) + 9 * (5 + 5 + 1) + 1, WEB_UI_MAX_TURNS)
        # With a retreat distance, one retreat turn before every bearing after
        # the first; a single bearing has none.
        self.assertEqual(
            trial_turn_budget(name_count=1, name_rounds=1, bearing_count=2, certification_retries=1, retreat_distance_m=0.3),
            (1 + 1 + 1) + (1 + 1 + 1 + 1) + 1,
        )
        self.assertEqual(
            trial_turn_budget(name_count=1, name_rounds=1, bearing_count=1, certification_retries=1, retreat_distance_m=0.3),
            (1 + 1 + 1) + 1,
        )
        self.assertEqual(
            trial_turn_budget(name_count=3, name_rounds=2, bearing_count=3, certification_retries=1, retreat_distance_m=0.3),
            (3 * 2 + 3 + 1) + 2 * (3 + 3 + 1 + 1) + 1,
        )
        self.assertEqual(
            trial_turn_budget(name_count=3, name_rounds=2, bearing_count=3, certification_retries=1, retreat_distance_m=0.0),
            (3 * 2 + 3 + 1) + 2 * (3 + 3 + 1) + 1,
        )
        self.assertEqual(
            trial_turn_budget(name_count=5, name_rounds=2, bearing_count=8, certification_retries=1, retreat_distance_m=0.3),
            WEB_UI_MAX_TURNS,
        )
        result = trial_config(
            self.config,
            object_names=("can",),
            name_rounds=1,
            pi_ik_compute_budget_s=TRIAL_PI_IK_COMPUTE_BUDGET_S,
            output_dir=Path(tempfile.gettempdir()) / "readiness-trial-test",
            bearing_offsets_deg=(0.0, 45.0),
            certification_retries=1,
        )
        self.assertEqual(result["agent"]["max_turns"], 8)
        self.assertEqual(result["model"]["max_turns"], 8)
        self.assertEqual(result["model"]["bearing_offsets_deg"], [0.0, 45.0])
        result = trial_config(
            self.config,
            object_names=("can",),
            name_rounds=1,
            pi_ik_compute_budget_s=TRIAL_PI_IK_COMPUTE_BUDGET_S,
            output_dir=Path(tempfile.gettempdir()) / "readiness-trial-test",
            bearing_offsets_deg=(0.0, 45.0),
            certification_retries=1,
            retreat_distance_m=0.0,
        )
        self.assertEqual(result["agent"]["max_turns"], 7)

    def test_no_bearing_search_keeps_only_bearing_zero(self) -> None:
        result = trial_config(
            self.config,
            object_names=("can",),
            name_rounds=1,
            pi_ik_compute_budget_s=TRIAL_PI_IK_COMPUTE_BUDGET_S,
            output_dir=Path(tempfile.gettempdir()) / "readiness-trial-test",
            bearing_offsets_deg=(0.0,),
            certification_retries=0,
        )

        self.assertEqual(result["model"]["bearing_offsets_deg"], [0.0])
        self.assertEqual(result["model"]["certification_retries"], 0)
        self.assertEqual(result["agent"]["max_turns"], 3)

    def test_bearing_offsets_are_parsed_from_the_cli_and_validated(self) -> None:
        self.assertEqual(parse_bearing_offsets_deg("0,45,-45, 90"), (0.0, 45.0, -45.0, 90.0))
        self.assertEqual(parse_bearing_offsets_deg("0"), (0.0,))
        with self.assertRaisesRegex(ValueError, "start with 0"):
            parse_bearing_offsets_deg("45,0")
        with self.assertRaisesRegex(ValueError, "comma-separated"):
            parse_bearing_offsets_deg("0,north")
        with self.assertRaises(ValueError):
            trial_config(
                self.config,
                object_names=("can",),
                name_rounds=1,
                pi_ik_compute_budget_s=TRIAL_PI_IK_COMPUTE_BUDGET_S,
                output_dir=Path(tempfile.gettempdir()) / "readiness-trial-test",
                certification_retries=-1,
            )

    def test_the_docking_fallbacks_stay_within_prepares_initial_distance_gate(self) -> None:
        from yor_agent.robot.manipulation_readiness import ManipulationReadinessConfig
        from yor_agent.robot.nav2_visible_object_navigation import Nav2VisibleObjectDockingConfig

        self.assertEqual(TRIAL_NAV2_GOAL_DISTANCE_FALLBACKS_M, (0.75, 0.90))
        result = trial_config(
            self.config,
            object_names=("can",),
            name_rounds=1,
            pi_ik_compute_budget_s=TRIAL_PI_IK_COMPUTE_BUDGET_S,
            output_dir=Path(tempfile.gettempdir()) / "readiness-trial-test",
        )
        docking = Nav2VisibleObjectDockingConfig.from_mapping(
            primitive_settings(result["primitive_config"], "dock_to_visible_object")
        )
        prepare = ManipulationReadinessConfig.from_mapping(
            primitive_settings(result["primitive_config"], "prepare_for_manipulation")
        )
        # The last fallback's acceptance limit, plus the height margin, is
        # the farthest camera distance a docked target can have.
        limit = fallback_target_distance_limit_m(docking)
        self.assertAlmostEqual(
            limit,
            0.90
            + (docking.docking_distance_m - docking.effective_nav2_goal_distance_m)
            + docking.final_distance_tolerance_m,
        )
        self.assertLessEqual(
            limit + TRIAL_TARGET_HEIGHT_MARGIN_M, prepare.maximum_initial_target_distance_m
        )
        check_fallback_distances_within_prepare_limit(docking, prepare)

        # One more fallback step would let prepare refuse the target unsearched.
        settings = primitive_settings(result["primitive_config"], "dock_to_visible_object")
        farther = Nav2VisibleObjectDockingConfig.from_mapping(
            {**settings, "nav2_goal_distance_fallbacks_m": [0.75, 0.90, 1.05]}
        )
        with self.assertRaisesRegex(ValueError, r"1\.05 m .*= 1\.33 m > 1\.20 m maximum_initial_target_distance_m"):
            check_fallback_distances_within_prepare_limit(farther, prepare)
        self.assertIsNone(
            fallback_target_distance_limit_m(
                Nav2VisibleObjectDockingConfig.from_mapping(
                    {**settings, "nav2_goal_distance_fallbacks_m": []}
                )
            )
        )

    def test_a_docking_tolerance_that_breaks_the_gate_fails_at_launch(self) -> None:
        config = json.loads(json.dumps(self.config))
        settings = config["primitive_config"]["primitives"]["dock_to_visible_object"]["settings"]
        settings["final_distance_tolerance_m"] = 0.20

        with self.assertRaisesRegex(ValueError, "maximum_initial_target_distance_m"):
            trial_config(
                config,
                object_names=("can",),
                name_rounds=1,
                pi_ik_compute_budget_s=TRIAL_PI_IK_COMPUTE_BUDGET_S,
                output_dir=Path(tempfile.gettempdir()) / "readiness-trial-test",
            )

    def test_a_budget_the_primitive_rejects_fails_at_launch(self) -> None:
        with self.assertRaisesRegex(ValueError, "pi_ik_compute_budget_s"):
            trial_config(
                self.config,
                object_names=DEFAULT_OBJECT_NAMES,
                name_rounds=2,
                pi_ik_compute_budget_s=60.0,
                output_dir=Path(tempfile.gettempdir()) / "readiness-trial-test",
            )

    def test_a_legacy_docking_backend_is_rejected_at_launch(self) -> None:
        config = json.loads(json.dumps(self.config))
        config["primitive_config"]["primitives"]["dock_to_visible_object"]["settings"]["backend"] = "legacy"

        with self.assertRaisesRegex(ValueError, "backend 'nav2'"):
            trial_config(
                config,
                object_names=DEFAULT_OBJECT_NAMES,
                name_rounds=2,
                pi_ik_compute_budget_s=TRIAL_PI_IK_COMPUTE_BUDGET_S,
                output_dir=Path(tempfile.gettempdir()) / "readiness-trial-test",
            )

    def test_default_output_dir_sits_beside_the_other_task_outputs(self) -> None:
        self.assertEqual(
            default_output_dir(TASKS, "find_can_on_table"),
            CONFIGS.resolve().parent / "outputs" / "readiness_trials" / "find_can_on_table",
        )

    def events_config(self, objects: list[str]) -> dict:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = Path(tmp.name) / "manipulation_events.json"
        path.write_text(
            json.dumps(
                {
                    "schema_version": EVENTS_SCHEMA,
                    "events": [{"object": name, "frames": {}} for name in objects],
                }
            ),
            encoding="utf-8",
        )
        return apply_readiness_prior_overrides(
            self.config, enabled=None, events_path=str(path), source=None
        )

    def trial(self, config: dict, names=DEFAULT_OBJECT_NAMES) -> dict:
        return trial_config(
            config,
            object_names=names,
            name_rounds=2,
            pi_ik_compute_budget_s=TRIAL_PI_IK_COMPUTE_BUDGET_S,
            output_dir=Path(tempfile.gettempdir()) / "readiness-trial-test",
        )

    def test_default_names_all_resolve_to_the_can_event_once_pinned(self) -> None:
        # "bottle" shares no word with the event; the trial pins the event to
        # the first name, so it docks with the can's event all the same.
        self.assertEqual(
            unmatched_object_names(self.trial(self.events_config(["can"])), DEFAULT_OBJECT_NAMES),
            [],
        )

    def test_a_pinned_first_name_without_an_event_reports_every_name(self) -> None:
        self.assertEqual(
            unmatched_object_names(self.trial(self.events_config(["mug"])), DEFAULT_OBJECT_NAMES),
            list(DEFAULT_OBJECT_NAMES),
        )

    def test_a_name_without_the_event_word_is_reported(self) -> None:
        self.assertEqual(
            unmatched_object_names(self.events_config(["can"]), ("can", "bottle")), ["bottle"]
        )

    def test_a_name_that_resolves_to_another_event_is_reported(self) -> None:
        config = self.events_config(["drink bottle", "can"])

        self.assertEqual(unmatched_object_names(config, ("can", "drink can")), ["drink can"])

    def test_without_an_events_file_nothing_is_checked(self) -> None:
        self.assertEqual(unmatched_object_names(self.config, ("bottle",)), [])

    def test_a_with_prior_config_keeps_the_prior_on_its_own_bearing(self) -> None:
        result = trial_config(
            self.events_config(["can"]),
            object_names=DEFAULT_OBJECT_NAMES,
            name_rounds=2,
            pi_ik_compute_budget_s=TRIAL_PI_IK_COMPUTE_BUDGET_S,
            output_dir=Path(tempfile.gettempdir()) / "readiness-trial-test",
        )

        prior = primitive_settings(result["primitive_config"], "dock_to_visible_object")["readiness_prior"]
        self.assertTrue(prior["enabled"])
        self.assertEqual(prior["retry_bearing_offsets_deg"], [])
        self.assertFalse(prior["fallback_to_arrival_bearing"])
        self.assertEqual(prior["event_object"], "can")


@unittest.skipUnless(
    WebRunController is not None,
    "FastAPI/Uvicorn web dependencies are not installed in this test environment",
)
class ScriptedWebStartTest(unittest.TestCase):
    def test_web_start_keeps_the_trial_model_and_its_turn_budget(self) -> None:
        config = trial_config(
            load_config(TASKS, task_id="find_can_on_table"),
            object_names=("can", "soda can"),
            name_rounds=1,
            pi_ik_compute_budget_s=TRIAL_PI_IK_COMPUTE_BUDGET_S,
            output_dir=Path(tempfile.gettempdir()) / "readiness-trial-web-test",
        )
        controller = WebRunController(TASKS, config)
        captured: dict = {}

        async def scenario() -> dict:
            async def fake_run(run_config) -> None:
                captured.update(run_config)

            controller._run = fake_run
            # The form pre-fills its turns field from the config and submits it back.
            result = await controller.start(
                {
                    "instruction": config["task"]["instruction"],
                    "provider": "scripted",
                    "model": "readiness-trial",
                    "temperature": None,
                    "max_turns": config["agent"]["max_turns"],
                }
            )
            await controller._worker
            return result

        result = asyncio.run(scenario())

        self.assertEqual(result["status"], "started")
        self.assertEqual(captured["model"]["provider"], "scripted")
        self.assertEqual(captured["model"]["object_names"], ["can", "soda can"])
        self.assertEqual(captured["model"]["bearing_offsets_deg"], list(DEFAULT_BEARING_OFFSETS_DEG))
        self.assertEqual(captured["model"]["retreat_distance_m"], TRIAL_RETREAT_DISTANCE_M)
        self.assertEqual(captured["agent"]["max_turns"], config["agent"]["max_turns"])
        self.assertEqual(captured["model"]["max_turns"], config["agent"]["max_turns"])
        self.assertEqual(controller.provider, "scripted")


class LlmTrialConfigTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_config(TASKS, task_id="find_can_on_table")

    def options(self, **extra) -> dict:
        return {
            "object_names": DEFAULT_OBJECT_NAMES,
            "name_rounds": 2,
            "pi_ik_compute_budget_s": TRIAL_PI_IK_COMPUTE_BUDGET_S,
            "output_dir": Path(tempfile.gettempdir()) / "readiness-trial-test",
            "start_label": "S0",
            **extra,
        }

    def test_an_llm_trial_keeps_the_model_and_the_trial_settings(self) -> None:
        llm = trial_config(
            self.config,
            **self.options(
                policy="llm",
                time_limit_s=DEFAULT_LLM_TIME_LIMIT_S,
                instruction=LLM_TRIAL_INSTRUCTION,
            ),
        )
        scripted = trial_config(self.config, **self.options())

        # GPT-5.6 Sol writes the programs; the configured model's other
        # settings and turn budget carry over.
        self.assertEqual(
            llm["model"],
            {**self.config["model"], "provider": "openai", "name": "gpt-5.6-sol"},
        )
        self.assertEqual((LLM_TRIAL_MODEL_PROVIDER, LLM_TRIAL_MODEL_NAME), ("openai", "gpt-5.6-sol"))
        self.assertEqual(scripted["model"]["provider"], "scripted")
        base_agent = self.config.get("agent") or {}
        if "max_turns" in base_agent:
            self.assertEqual(
                llm["agent"]["max_turns"], min(base_agent["max_turns"], WEB_UI_MAX_TURNS)
            )
        else:
            self.assertNotIn("max_turns", llm["agent"])
        self.assertEqual(llm["agent"]["time_limit_s"], 120.0)
        self.assertEqual(DEFAULT_LLM_TIME_LIMIT_S, 120.0)
        self.assertNotIn("time_limit_s", scripted["agent"])
        self.assertEqual(llm["task"]["instruction"], LLM_TRIAL_INSTRUCTION)
        self.assertIn("prepare_for_manipulation", LLM_TRIAL_INSTRUCTION)
        self.assertFalse(llm["navigation_planner"]["enabled"])
        # Docking and prepare run exactly as in the scripted trial.
        for name in ("dock_to_visible_object", "prepare_for_manipulation"):
            self.assertEqual(
                primitive_settings(llm["primitive_config"], name),
                primitive_settings(scripted["primitive_config"], name),
            )
        self.assertEqual(llm["trial"], {"start_label": "S0"})
        self.assertEqual(llm["trace"], scripted["trace"])
        # Only the grasp, arm and gripper primitives are hidden.
        for name in llm["primitive_config"]["primitives"]:
            with self.subTest(primitive=name):
                self.assertEqual(
                    primitive_exposed(llm["primitive_config"], name),
                    name not in LLM_TRIAL_HIDDEN_PRIMITIVES
                    and primitive_exposed(self.config["primitive_config"], name),
                )
        for name in LLM_TRIAL_HIDDEN_PRIMITIVES:
            self.assertIn(name, self.config["primitive_config"]["primitives"])

    def test_grasping_and_the_planner_can_be_kept_as_configured(self) -> None:
        kept = trial_config(
            self.config, **self.options(policy="llm", allow_grasp=True, navigation_planner=True)
        )

        self.assertEqual(kept["navigation_planner"], self.config["navigation_planner"])
        for name in LLM_TRIAL_HIDDEN_PRIMITIVES:
            self.assertEqual(
                primitive_exposed(kept["primitive_config"], name),
                primitive_exposed(self.config["primitive_config"], name),
            )
        self.assertEqual(kept["task"], self.config["task"])

    def test_the_trial_options_are_validated_at_launch(self) -> None:
        for bad in (
            {"policy": "gpt"},
            {"time_limit_s": 0},
            {"time_limit_s": -5.0},
            {"time_limit_s": float("nan")},
            {"time_limit_s": True},
            {"instruction": "  "},
        ):
            with self.subTest(**{key: repr(value) for key, value in bad.items()}):
                with self.assertRaises(ValueError):
                    trial_config(self.config, **self.options(**bad))
        # Whatever model the config names, an LLM trial starts from GPT-5.6 Sol.
        manual = {**self.config, "model": {"provider": "manual", "name": "operator"}}
        self.assertEqual(
            trial_config(manual, **self.options(policy="llm"))["model"],
            {"provider": "openai", "name": "gpt-5.6-sol"},
        )


@unittest.skipUnless(
    WebRunController is not None,
    "FastAPI/Uvicorn web dependencies are not installed in this test environment",
)
class LlmTrialLauncherTest(unittest.TestCase):
    def launch(self, *extra: str) -> dict:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        captured: dict = {}
        argv = [
            "--config", str(TASKS),
            "--task-id", "find_can_on_table",
            "--no-bearing-search",
            "--start-label", "S0",
            "--output-dir", tmp.name,
            *extra,
        ]
        with mock.patch(
            "yor_agent.web.server.run_web_ui", side_effect=lambda **kwargs: captured.update(kwargs)
        ), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(trial_main(argv), 0)
        return captured["config"]

    def test_the_launcher_runs_the_llm_with_a_two_minute_limit_by_default(self) -> None:
        base = load_config(TASKS, task_id="find_can_on_table")

        config = self.launch()

        self.assertEqual(
            config["model"], {**base["model"], "provider": "openai", "name": "gpt-5.6-sol"}
        )
        self.assertEqual(config["agent"]["time_limit_s"], 120.0)
        self.assertEqual(config["task"]["instruction"], LLM_TRIAL_INSTRUCTION)
        self.assertFalse(primitive_exposed(config["primitive_config"], "goto_grasp_pose"))
        self.assertEqual(config["trial"], {"start_label": "S0"})

        self.assertEqual(self.launch("--time-limit-s", "90")["agent"]["time_limit_s"], 90.0)
        self.assertNotIn("time_limit_s", self.launch("--time-limit-s", "0")["agent"])
        self.assertEqual(
            self.launch("--instruction", "Get ready to grasp the can")["task"]["instruction"],
            "Get ready to grasp the can",
        )

    def test_the_scripted_policy_stays_the_fixed_protocol_without_a_limit(self) -> None:
        config = self.launch("--policy", "scripted")

        self.assertEqual(config["model"]["provider"], "scripted")
        self.assertNotIn("time_limit_s", config["agent"])
        self.assertEqual(
            config["task"], load_config(TASKS, task_id="find_can_on_table")["task"]
        )
        self.assertEqual(
            self.launch("--policy", "scripted", "--time-limit-s", "300")["agent"]["time_limit_s"],
            300.0,
        )
        for bad in (
            ("--policy", "scripted", "--allow-grasp"),
            ("--policy", "scripted", "--navigation-planner"),
            ("--time-limit-s", "-1"),
        ):
            with self.subTest(argv=bad), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    self.launch(*bad)

    def test_web_start_keeps_the_llm_the_limit_and_the_hidden_primitives(self) -> None:
        config = self.launch()
        controller = WebRunController(TASKS, config)
        captured: dict = {}

        async def scenario() -> dict:
            async def fake_run(run_config) -> None:
                captured.update(run_config)

            controller._run = fake_run
            result = await controller.start(
                {
                    "instruction": config["task"]["instruction"],
                    "provider": config["model"]["provider"],
                    "model": config["model"]["name"],
                    "temperature": None,
                    "max_turns": config["agent"].get("max_turns"),
                    "readiness_prior": False,
                }
            )
            await controller._worker
            return result

        result = asyncio.run(scenario())

        self.assertEqual(result["status"], "started")
        self.assertEqual(captured["model"]["provider"], config["model"]["provider"])
        self.assertEqual(captured["agent"]["time_limit_s"], 120.0)
        self.assertEqual(captured["task"]["instruction"], LLM_TRIAL_INSTRUCTION)
        for name in LLM_TRIAL_HIDDEN_PRIMITIVES:
            self.assertFalse(primitive_exposed(captured["primitive_config"], name))


if __name__ == "__main__":
    unittest.main()
