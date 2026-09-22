#!/usr/bin/env python3
"""Summarise readiness-trial traces: one row per trial with its cost.

Every trial run by ``yor_agent.experiments.readiness_trial`` leaves a
``trace.json``. For each one this prints its condition, the anchor bearing,
the bearings the trial went through with how each ended, and the cost the
conditions are compared on: Pi IK queries and Nav2 goals sent.

Columns of the trial table:

- ``condition``: ``prior`` or ``no_prior`` from the run config's
  ``readiness_prior.enabled``, confirmed by the first dock call that carries
  a ``readiness_prior`` record: used, ``prior``; ``not_configured``,
  ``no_prior``; enabled but not used, ``prior_failed:<reason>``. Blank when
  neither is in the trace.
- ``start_label``: the start position the launcher was given with
  ``--start-label`` (``trial.start_label`` of the run config); blank without.
- ``dock_calls``: ``dock_to_visible_object`` calls, SAM3-name retries included.
- ``nav2_goals``: Nav2 goals sent. A goal attempt counts when it records its
  Nav2 result (``nav2``); an attempt without that key counts when it carries
  its goal pose and its reason is neither ``no_plannable_goal_on_bearing``
  nor ``operator_stop``.
- ``solved``: a ``prepare_for_manipulation`` call succeeded.
- ``cost_queries``: Pi IK queries of every failed prepare call, plus the
  queries the solving call's search spent until the base it executed
  certified: the ``executed_candidate_index`` base reaching
  ``minimum_feasible_grasps`` converged nominal rows of ``pi_ik_query_log``;
  the first certified base (``first_certified`` of
  ``prepare_ik_timing_report``) when the result names no executed base; the
  call's whole count when the log is missing or does not show that base.
  Blank when the trial is unsolved.
- ``total_queries``: Pi IK queries of every prepare call.
- ``retreats``: ``drive_straight`` calls, which the trial makes only to back
  the base off the object before the next bearing.
- ``time_s``: seconds from the run's start to its finish (``finish.at``), or
  to the stop its time limit made (``stop.at``): the time to the solution
  when solved, the time to giving up otherwise. Blank for a run without
  either (stopped by the operator or aborted). The run starts when the agent
  loop does (``started_at``), which is also when the time limit starts; a
  trace from before that stamp existed counts from its creation
  (``created_at``), which includes building the runtime.
- ``dock_time_s`` / ``prepare_time_s``: seconds spent inside every
  ``dock_to_visible_object`` / ``prepare_for_manipulation`` call (the
  executor's ``elapsed_s``); ``ik_time_s`` is the Pi IK compute time the
  prepare calls report (``pi_ik_compute_elapsed_s``), a part of
  ``prepare_time_s``. The rest of ``time_s`` is the retreats, the
  observations and the policy's own turns.
- ``policy``: ``scripted`` for the fixed protocol, ``llm`` for any other
  model provider; ``model`` is ``provider/name`` of the run config.
- ``end``: how the run ended: ``finished`` (the policy called ``finish()``),
  ``turn_budget``, ``time_limit`` (``agent.time_limit_s`` stopped it),
  ``stopped`` (the operator), ``error``, or ``unfinished`` without any.
- ``time_limit_s``: the run config's ``agent.time_limit_s``; blank without.
- ``solved_time_s``: seconds from the run's start to the end of the first
  successful ``prepare_for_manipulation`` call; blank when unsolved. An LLM
  run may keep going after it until it calls ``finish()``.
- ``solved_in_time``: solved within the run's time limit (``solved_time_s``
  at most ``time_limit_s``); the same as ``solved`` without a limit. A
  prepare that was already running when the limit stopped the run can still
  succeed after it, which ``solved`` counts and this does not.
- ``turns``: policies the model wrote and the executor ran; ``model_calls``,
  ``prompt_tokens`` and ``output_tokens`` from the model usage the trace
  records (blank for the scripted model, which records none).
- ``prepare_calls``: ``prepare_for_manipulation`` calls.
- ``explicit_bearing_docks``: dock calls made with an
  ``approach_bearing_deg``. Such a dock skips the readiness prior, so in an
  LLM run with the prior it marks a dock the condition did not decide.

The summary table below the trial table gives, per policy and condition, the
trials, the ones solved in time (``solved_in_time``) and their success rate,
the median ``time_s``, the median ``solved_time_s`` of the trials solved in
time, and the mean ``total_queries``.

The per-bearing table below it lists, for each bearing, how it ended (named
as the trial model names its outcomes: ``solved``, ``prepare_too_far``,
``certification_failed``, ``prepare_names_exhausted``, ``prepare_failed``,
``dock_names_exhausted``, ``dock_failed``; ``unfinished`` for a dock that no
prepare followed), the last dock's reason and goal distance, the last
prepare's reason, the target distance it measured, the queries spent and the
reason of the retreat made after it (``-`` without one). Bearings are
recognised from the ``approach_bearing_deg`` each dock call was made with;
the offset is measured from the anchor the trial read back; a retreat
belongs to the bearing it follows.

Examples::

    python tools/readiness_trial_report.py yor_agent/outputs/readiness_trials/find_can_on_table
    python tools/readiness_trial_report.py run_a/trace.json run_b/trace.json --csv trials.csv
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

try:
    from tools.prepare_ik_timing_report import first_certified, trace_files
except ImportError:  # run as a script: the sibling module is next to this file
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from prepare_ik_timing_report import first_certified, trace_files

DOCK = "dock_to_visible_object"
PREPARE = "prepare_for_manipulation"
# The primitive the trial backs the base off the object with; it makes no
# other call of it.
RETREAT = "drive_straight"
NO_PLANNABLE_GOAL_REASON = "no_plannable_goal_on_bearing"
OPERATOR_STOP_REASON = "operator_stop"
# The trial model's markers (yor_agent.models.scripted), repeated here so the
# report runs without the package on the path.
SAM3_MISS_MARKERS = ("SAM3 found no instance", "SAM3 produced no usable mask")
CERTIFICATION_FAILED_MARKER = "actual_pose_grasp_certification_failed"
TARGET_TOO_FAR_MARKER = "target_too_far_for_local_manipulation"
# How the agent loop words a stop by its time limit and a finish by its turn
# budget (yor_agent.agents.default), repeated for the same reason.
TIME_LIMIT_STOP_PREFIX = "time limit"
TURN_BUDGET_FINISH_PREFIX = "turn budget exhausted"
TRIAL_COLUMNS = (
    "run",
    "condition",
    "start_label",
    "anchor_deg",
    "bearings",
    "dock_calls",
    "nav2_goals",
    "solved",
    "cost_queries",
    "total_queries",
    "retreats",
    "time_s",
    "dock_time_s",
    "prepare_time_s",
    "ik_time_s",
    "policy",
    "model",
    "end",
    "time_limit_s",
    "solved_time_s",
    "solved_in_time",
    "turns",
    "model_calls",
    "prompt_tokens",
    "output_tokens",
    "prepare_calls",
    "explicit_bearing_docks",
)
SUMMARY_COLUMNS = (
    "policy",
    "condition",
    "trials",
    "solved",
    "success_rate",
    "median_time_s",
    "median_solved_time_s",
    "mean_total_queries",
)
BEARING_COLUMNS = (
    "run",
    "offset_deg",
    "approach_deg",
    "outcome",
    "docks",
    "dock_reason",
    "goal_distance_m",
    "nav2_goals",
    "prepares",
    "prepare_reason",
    "target_distance_m",
    "ik_queries",
    "retreat",
)
_COLLAPSED_LIST = re.compile(r"^<(?:list|tuple) len=(\d+)>$")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="trace.json files, or run directories searched recursively for them",
    )
    parser.add_argument("--csv", type=Path, help="also write the trial rows to this CSV")
    return parser.parse_args(argv)


# ----------------------------------------------------------------------
# Reading one trace
# ----------------------------------------------------------------------


def wrap_degrees(value: float) -> float:
    wrapped = (float(value) + 180.0) % 360.0 - 180.0
    return 180.0 if wrapped == -180.0 else wrapped


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(float(value)) else None


def _list_length(value: Any) -> int | None:
    if isinstance(value, (list, tuple)):
        return len(value)
    if isinstance(value, str):
        match = _COLLAPSED_LIST.match(value.strip())
        if match:
            return int(match.group(1))
    return None


def primitive_calls(trace: dict[str, Any], primitive: str) -> list[dict[str, Any]]:
    """The trace's calls of ``primitive``, in order, each with a result dict."""

    calls = []
    for call in trace.get("primitive_calls") or []:
        if not isinstance(call, dict):
            continue
        result = call.get("result")
        name = call.get("name") or (result.get("primitive") if isinstance(result, dict) else None)
        if name == primitive:
            calls.append(call)
    return calls


def _result(call: dict[str, Any]) -> dict[str, Any]:
    result = call.get("result")
    return result if isinstance(result, dict) else {}


def _metrics(result: dict[str, Any]) -> dict[str, Any]:
    metrics = result.get("metrics")
    return metrics if isinstance(metrics, dict) else {}


def configured_prior_enabled(trace: dict[str, Any]) -> bool | None:
    """``readiness_prior.enabled`` of the run config the trace was started with."""

    node: Any = trace.get("config")
    for key in ("primitive_config", "primitives", DOCK, "settings", "readiness_prior"):
        if not isinstance(node, dict):
            return None
        node = node.get(key)
    if not isinstance(node, dict) or "enabled" not in node:
        return None
    return bool(node.get("enabled"))


def start_label(trace: dict[str, Any]) -> str | None:
    """``trial.start_label`` of the run config, the start position's name."""

    config = trace.get("config")
    trial = config.get("trial") if isinstance(config, dict) else None
    label = trial.get("start_label") if isinstance(trial, dict) else None
    return label if isinstance(label, str) and label else None


def prior_record(trace: dict[str, Any]) -> dict[str, Any] | None:
    """The first dock call's ``readiness_prior`` record that says whether it was used.

    A dock that failed before it asked the prior (a SAM3 miss) records none,
    and one made with an explicit bearing records ``explicit_bearing``, which
    says nothing about the condition; both are skipped.
    """

    for call in primitive_calls(trace, DOCK):
        prior = _metrics(_result(call)).get("readiness_prior")
        if not isinstance(prior, dict) or "used" not in prior:
            continue
        if not prior.get("used") and prior.get("reason") == "explicit_bearing":
            continue
        return prior
    return None


def condition(trace: dict[str, Any]) -> str | None:
    """``prior``, ``no_prior`` or ``prior_failed:<reason>``.

    The run config's ``readiness_prior.enabled`` names the condition the run
    was started in; the first dock's prior record confirms what happened:
    used, ``prior``; ``not_configured``, ``no_prior``; enabled but not used
    (no estimate, an invalid bearing, an exception), ``prior_failed`` with
    that reason. ``None`` when the trace carries neither.
    """

    enabled = configured_prior_enabled(trace)
    record = prior_record(trace)
    if record is None:
        if enabled is None:
            return None
        return "prior" if enabled else "no_prior"
    if record.get("used"):
        return "prior"
    reason = str(record.get("reason") or "unknown")
    if reason == "not_configured":
        return "no_prior"
    return f"prior_failed:{reason}"


def anchor_bearing_deg(trace: dict[str, Any]) -> float | None:
    """The bearing the first dock that detected the object docked from."""

    for call in primitive_calls(trace, DOCK):
        metrics = _metrics(_result(call))
        candidates: list[Any] = [metrics.get("reference_bearing_rad")]
        attempts = metrics.get("goal_attempts")
        if isinstance(attempts, list) and attempts and isinstance(attempts[0], dict):
            candidates.append(attempts[0].get("bearing_rad"))
        candidates.append(metrics.get("arrival_bearing_rad"))
        for candidate in candidates:
            bearing = _finite(candidate)
            if bearing is not None:
                return math.degrees(bearing)
    return None


def attempt_sent_nav2_goal(attempt: Any) -> bool:
    """Whether one ``goal_attempts`` entry sent a Nav2 goal.

    Same rule as ``yor_agent.models.scripted.attempt_sent_nav2_goal``: an
    entry recording its Nav2 result under ``nav2`` sent a goal iff that result
    is present; an entry without that key sent one iff it carries its goal
    pose and its reason is neither the plan-check exhaustion nor an operator
    stop that pre-empted the goal. A collapsed entry counts as sent.
    """

    if not isinstance(attempt, dict):
        return True
    if "nav2" in attempt:
        return isinstance(attempt["nav2"], dict)
    if attempt.get("nav2_goal_xy_yaw") is None:
        return False
    return attempt.get("reason") not in (NO_PLANNABLE_GOAL_REASON, OPERATOR_STOP_REASON)


def nav2_goal_count(result: dict[str, Any]) -> int:
    """Nav2 goals one dock sent (see ``attempt_sent_nav2_goal``)."""

    attempts = _metrics(result).get("goal_attempts")
    if not isinstance(attempts, list):
        return _list_length(attempts) or 0
    return sum(1 for attempt in attempts if attempt_sent_nav2_goal(attempt))


def goal_distance_used_m(result: dict[str, Any]) -> float | None:
    """The goal distance a dock ended on, else the last attempt's distance."""

    metrics = _metrics(result)
    used = _finite(metrics.get("goal_distance_used_m"))
    if used is not None:
        return used
    attempts = metrics.get("goal_attempts")
    if isinstance(attempts, list):
        for attempt in reversed(attempts):
            if isinstance(attempt, dict):
                distance = _finite(attempt.get("goal_distance_m"))
                if distance is not None:
                    return distance
    return None


def prepare_ik_query_count(result: dict[str, Any]) -> int:
    """Pi IK queries one prepare call spent, on success or failure."""

    if result.get("success", False):
        count = result.get("ik_query_count")
        if isinstance(count, int) and not isinstance(count, bool):
            return count
    for source in (result.get("diagnostics"), result):
        if not isinstance(source, dict):
            continue
        length = _list_length(source.get("pi_ik_query_log"))
        if length is not None:
            return length
        count = source.get("ik_query_count")
        if isinstance(count, int) and not isinstance(count, bool):
            return count
    return 0


def queries_until_base(
    log: list[dict[str, Any]], base_index: int, minimum_feasible_grasps: int
) -> int | None:
    """Queries spent when ``base_index`` had its ``minimum_feasible_grasps``-th
    converged nominal grasp: that row's ``order`` plus one. ``None`` when the
    log never certifies that base."""

    converged = 0
    for row in log:
        if not isinstance(row, dict):
            continue
        if row.get("robustness_variant") != "nominal" or not row.get("ik_converged"):
            continue
        if _int(row.get("base_index")) != base_index:
            continue
        converged += 1
        if converged >= minimum_feasible_grasps:
            order = _int(row.get("order"))
            return None if order is None else order + 1
    return None


def _int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def solving_cost_queries(result: dict[str, Any]) -> int:
    """Queries the solving prepare call spent until the base it executed certified.

    With ``executed_candidate_index`` (prepare may execute a later-ranked
    candidate when the selected one's motion is refused) the count runs until
    that base certified; without it, until the first base certified. The whole
    call's count when there is no query log or it does not show that base.
    """

    log = result.get("pi_ik_query_log")
    if isinstance(log, list):
        minimum = int(result.get("minimum_feasible_grasps", 1))
        executed = _int(result.get("executed_candidate_index"))
        if executed is not None:
            queries = queries_until_base(log, executed, minimum)
        else:
            queries, _ = first_certified(log, minimum)
        if queries is not None:
            return queries
    return prepare_ik_query_count(result)


def bearing_groups(trace: dict[str, Any]) -> list[dict[str, Any]]:
    """Consecutive calls made at one requested bearing, in trial order.

    A dock call opens a new group whenever its ``approach_bearing_deg`` differs
    from the current group's (the first dock, made without one, is the anchor
    group); prepare calls and the retreat (``drive_straight``) join the group
    of the dock before them.
    """

    anchor = anchor_bearing_deg(trace)
    groups: list[dict[str, Any]] = []

    def new_group(approach: float | None, offset: float | None) -> dict[str, Any]:
        return {
            "approach_deg": approach,
            "offset_deg": offset,
            "docks": [],
            "prepares": [],
            "retreats": [],
        }

    for call in trace.get("primitive_calls") or []:
        if not isinstance(call, dict):
            continue
        result = _result(call)
        name = call.get("name") or result.get("primitive")
        if name == DOCK:
            kwargs = call.get("kwargs") if isinstance(call.get("kwargs"), dict) else {}
            requested = _finite(kwargs.get("approach_bearing_deg"))
            if not groups or groups[-1]["approach_deg"] != requested:
                offset: float | None
                if requested is None:
                    offset = 0.0
                elif anchor is None:
                    offset = None
                else:
                    offset = wrap_degrees(requested - anchor)
                groups.append(new_group(requested, offset))
            groups[-1]["docks"].append(result)
        elif name in (PREPARE, RETREAT):
            if not groups:
                groups.append(new_group(None, 0.0))
            groups[-1]["prepares" if name == PREPARE else "retreats"].append(result)
    return groups


def bearing_outcome(group: dict[str, Any]) -> str | None:
    """How a bearing ended, named as the trial model names its outcomes.

    Read from the bearing's last call: a prepare decides (``solved``,
    ``prepare_too_far``, ``certification_failed``, ``prepare_names_exhausted``
    or ``prepare_failed``), else the last dock (``dock_names_exhausted``,
    ``dock_failed``, or ``unfinished`` for a successful dock that no prepare
    followed because the run ended). ``None`` without any call.
    """

    if group["prepares"]:
        last = group["prepares"][-1]
        if last.get("success", False):
            return "solved"
        reason = str(last.get("reason", ""))
        if TARGET_TOO_FAR_MARKER in reason:
            return "prepare_too_far"
        if CERTIFICATION_FAILED_MARKER in reason:
            return "certification_failed"
        if any(marker in reason for marker in SAM3_MISS_MARKERS):
            return "prepare_names_exhausted"
        return "prepare_failed"
    if group["docks"]:
        last = group["docks"][-1]
        if last.get("success", False):
            return "unfinished"
        reason = str(last.get("reason", ""))
        if any(marker in reason for marker in SAM3_MISS_MARKERS):
            return "dock_names_exhausted"
        return "dock_failed"
    return None


def bearing_rows(run: str, groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for group in groups:
        last_dock = group["docks"][-1] if group["docks"] else {}
        last_prepare = group["prepares"][-1] if group["prepares"] else {}
        last_retreat = group["retreats"][-1] if group["retreats"] else {}
        rows.append(
            {
                "run": run,
                "offset_deg": group["offset_deg"],
                "approach_deg": group["approach_deg"],
                "outcome": bearing_outcome(group),
                "docks": len(group["docks"]),
                "dock_reason": str(last_dock.get("reason", "")) if last_dock else "",
                "goal_distance_m": goal_distance_used_m(last_dock) if last_dock else None,
                "nav2_goals": sum(nav2_goal_count(dock) for dock in group["docks"]),
                "prepares": len(group["prepares"]),
                "prepare_reason": str(last_prepare.get("reason", "")) if last_prepare else "",
                "target_distance_m": _finite(last_prepare.get("target_distance_m")),
                "ik_queries": sum(prepare_ik_query_count(p) for p in group["prepares"]),
                "retreats": len(group["retreats"]),
                "retreat": str(last_retreat.get("reason", "")) if last_retreat else "",
            }
        )
    return rows


def _bearings_text(rows: list[dict[str, Any]]) -> str:
    items = []
    for row in rows:
        offset = "?" if row["offset_deg"] is None else f"{row['offset_deg']:+.0f}"
        dock = row["dock_reason"] or "-"
        if row["goal_distance_m"] is not None:
            dock += f"@{row['goal_distance_m']:.2f}"
        prepare = row["prepare_reason"] or "-"
        if row["target_distance_m"] is not None:
            prepare += f"@{row['target_distance_m']:.2f}"
        items.append(
            f"{offset}: {row['outcome'] or '-'} dock={dock} prepare={prepare} q={row['ik_queries']}"
        )
    return "; ".join(items)


def _parse_time_s(text: Any) -> float | None:
    """Epoch seconds of a trace timestamp (ISO 8601 with a UTC offset)."""

    if not isinstance(text, str) or not text.strip():
        return None
    value = text.strip()
    for parser in (
        datetime.fromisoformat,
        lambda v: datetime.strptime(v, "%Y-%m-%dT%H:%M:%S%z"),
        lambda v: datetime.strptime(v, "%Y-%m-%dT%H:%M:%S.%f%z"),
    ):
        try:
            return parser(value).timestamp()
        except ValueError:
            continue
    return None


def _record(trace: dict[str, Any], key: str) -> dict[str, Any]:
    value = trace.get(key)
    return value if isinstance(value, dict) else {}


def run_started_s(trace: dict[str, Any]) -> float | None:
    """Epoch seconds the agent loop started, else the trace's creation."""

    started = _parse_time_s(trace.get("started_at"))
    return started if started is not None else _parse_time_s(trace.get("created_at"))


def _time_limit_stop(trace: dict[str, Any]) -> bool:
    return str(_record(trace, "stop").get("reason", "")).startswith(TIME_LIMIT_STOP_PREFIX)


def run_time_s(trace: dict[str, Any]) -> float | None:
    """Seconds from the run's start to its finish or its time-limit stop.

    ``None`` without the start or either end; an operator stop is no end the
    trial measures.
    """

    started = run_started_s(trace)
    ended = _parse_time_s(_record(trace, "finish").get("at"))
    if ended is None and _time_limit_stop(trace):
        ended = _parse_time_s(_record(trace, "stop").get("at"))
    if started is None or ended is None:
        return None
    return max(0.0, ended - started)


def policy(trace: dict[str, Any]) -> str | None:
    """``scripted`` for the fixed protocol, ``llm`` for any other provider."""

    model = _record(trace, "config").get("model")
    provider = model.get("provider") if isinstance(model, dict) else None
    if not isinstance(provider, str) or not provider:
        return None
    return "scripted" if provider == "scripted" else "llm"


def model_name(trace: dict[str, Any]) -> str | None:
    model = _record(trace, "config").get("model")
    if not isinstance(model, dict) or not model.get("provider"):
        return None
    return f"{model.get('provider')}/{model.get('name') or '-'}"


def run_end(trace: dict[str, Any]) -> str:
    """How the run ended (see the ``end`` column)."""

    if trace.get("error"):
        return "error"
    finish = _record(trace, "finish")
    if finish:
        reason = str(finish.get("reason", ""))
        return "turn_budget" if reason.startswith(TURN_BUDGET_FINISH_PREFIX) else "finished"
    if _record(trace, "stop"):
        return "time_limit" if _time_limit_stop(trace) else "stopped"
    return "unfinished"


def configured_time_limit_s(trace: dict[str, Any]) -> float | None:
    agent = _record(trace, "config").get("agent")
    return _finite(agent.get("time_limit_s")) if isinstance(agent, dict) else None


def solved_time_s(trace: dict[str, Any]) -> float | None:
    """Seconds from the run's start to the end of its first successful prepare."""

    started = run_started_s(trace)
    if started is None:
        return None
    for call in primitive_calls(trace, PREPARE):
        if not _result(call).get("success", False):
            continue
        call_started = _finite(call.get("started_at"))
        elapsed = _finite(call.get("elapsed_s"))
        if call_started is None or elapsed is None:
            return None
        return max(0.0, call_started + elapsed - started)
    return None


def model_usage(trace: dict[str, Any], key: str) -> int | None:
    value = _record(trace, "model_usage").get(key)
    return _int(value)


def explicit_bearing_docks(trace: dict[str, Any]) -> int:
    count = 0
    for call in primitive_calls(trace, DOCK):
        kwargs = call.get("kwargs")
        if isinstance(kwargs, dict):
            count += _finite(kwargs.get("approach_bearing_deg")) is not None
    return count


def calls_elapsed_s(calls: list[dict[str, Any]]) -> float:
    """Seconds the executor spent inside these primitive calls."""

    total = 0.0
    for call in calls:
        elapsed = _finite(call.get("elapsed_s"))
        if elapsed is None:
            elapsed = _finite(_result(call).get("elapsed_s"))
        total += max(0.0, elapsed or 0.0)
    return total


def ik_compute_s(prepares: list[dict[str, Any]]) -> float:
    """Pi IK compute seconds the prepare results report."""

    return sum(
        max(0.0, _finite(p.get("pi_ik_compute_elapsed_s")) or 0.0) for p in prepares
    )


def summarise(run: str, trace: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """The trial row and its per-bearing rows."""

    dock_calls = primitive_calls(trace, DOCK)
    prepare_calls = primitive_calls(trace, PREPARE)
    docks = [_result(call) for call in dock_calls]
    prepares = [_result(call) for call in prepare_calls]
    retreats = primitive_calls(trace, RETREAT)
    solving = next((p for p in prepares if p.get("success", False)), None)
    failed_queries = sum(
        prepare_ik_query_count(p) for p in prepares if not p.get("success", False)
    )
    anchor = anchor_bearing_deg(trace)
    bearings = bearing_rows(run, bearing_groups(trace))
    time_limit = configured_time_limit_s(trace)
    time_to_solve = solved_time_s(trace)
    row = {
        "run": run,
        "condition": condition(trace),
        "start_label": start_label(trace),
        "anchor_deg": None if anchor is None else round(anchor, 1),
        "bearings": _bearings_text(bearings),
        "dock_calls": len(docks),
        "nav2_goals": sum(nav2_goal_count(dock) for dock in docks),
        "solved": solving is not None,
        "cost_queries": (
            None if solving is None else failed_queries + solving_cost_queries(solving)
        ),
        "total_queries": sum(prepare_ik_query_count(p) for p in prepares),
        "retreats": len(retreats),
        "time_s": run_time_s(trace),
        "dock_time_s": calls_elapsed_s(dock_calls),
        "prepare_time_s": calls_elapsed_s(prepare_calls),
        "ik_time_s": ik_compute_s(prepares),
        "policy": policy(trace),
        "model": model_name(trace),
        "end": run_end(trace),
        "time_limit_s": time_limit,
        "solved_time_s": time_to_solve,
        "solved_in_time": solving is not None
        and (time_limit is None or (time_to_solve is not None and time_to_solve <= time_limit)),
        "turns": len([p for p in trace.get("policies") or [] if isinstance(p, dict)]),
        "model_calls": model_usage(trace, "n_calls"),
        "prompt_tokens": model_usage(trace, "prompt_tokens"),
        "output_tokens": model_usage(trace, "output_tokens"),
        "prepare_calls": len(prepare_calls),
        "explicit_bearing_docks": explicit_bearing_docks(trace),
    }
    return row, bearings


def _median(values: list[float]) -> float | None:
    values = sorted(values)
    if not values:
        return None
    middle = len(values) // 2
    return values[middle] if len(values) % 2 else (values[middle - 1] + values[middle]) / 2.0


def summary_rows(trial_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per policy and condition: trials, success rate, median times, mean queries.

    A trial counts as solved only when it was solved within its time limit.
    """

    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in trial_rows:
        key = (row.get("policy") or "-", row.get("condition") or "-")
        groups.setdefault(key, []).append(row)
    rows = []
    for (policy_name, condition_name), members in sorted(groups.items()):
        in_time = [row for row in members if row.get("solved_in_time", row["solved"])]
        solved = len(in_time)
        times = [row["time_s"] for row in members if row.get("time_s") is not None]
        solved_times = [
            row["solved_time_s"] for row in in_time if row.get("solved_time_s") is not None
        ]
        rows.append(
            {
                "policy": policy_name,
                "condition": condition_name,
                "trials": len(members),
                "solved": solved,
                "success_rate": solved / len(members),
                "median_time_s": _median(times),
                "median_solved_time_s": _median(solved_times),
                "mean_total_queries": sum(row["total_queries"] for row in members) / len(members),
            }
        )
    return rows


# ----------------------------------------------------------------------
# Output
# ----------------------------------------------------------------------


def _cell(value: Any) -> str:
    if value is None or value == "":
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value).replace("|", "\\|")


def markdown_table(columns: tuple[str, ...], rows: list[dict[str, Any]]) -> str:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_cell(row.get(column)) for column in columns) + " |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    trial_rows: list[dict[str, Any]] = []
    bearing_table: list[dict[str, Any]] = []
    for path in trace_files(args.paths):
        try:
            trace = json.loads(path.read_text())
        except (OSError, ValueError) as exc:
            print(f"warning: skipping {path}: {exc}", file=sys.stderr)
            continue
        if not isinstance(trace, dict):
            print(f"warning: skipping {path}: not a trace object", file=sys.stderr)
            continue
        run = str(trace.get("run_id") or path.parent.name)
        row, bearings = summarise(run, trace)
        trial_rows.append(row)
        bearing_table.extend(bearings)
    if not trial_rows:
        print("no traces found", file=sys.stderr)
        return 1

    print(markdown_table(TRIAL_COLUMNS, trial_rows))
    print()
    print(markdown_table(SUMMARY_COLUMNS, summary_rows(trial_rows)))
    print()
    print(markdown_table(BEARING_COLUMNS, bearing_table))
    solved = [row for row in trial_rows if row["solved"]]
    print(f"\n{len(trial_rows)} trials, {len(solved)} solved")
    if args.csv:
        with args.csv.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(TRIAL_COLUMNS))
            writer.writeheader()
            writer.writerows(trial_rows)
        print(f"wrote {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
