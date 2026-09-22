#!/usr/bin/env python3
"""Summarise prepare_for_manipulation timing from trace.json files.

For every ``prepare_for_manipulation`` call found under the given paths this
prints one row with the stage split of the primitive, the local substage
split hidden inside ``virtual_candidate_evaluation``, the Pi IK waves, and,
when the trace carries ``pi_ik_query_log``, how many Pi queries were spent
before the first base pose was certified.  That last number is what the
budget-versus-success curve is built from: a run replayed at budget ``b``
succeeds iff ``queries_to_first_certified <= b``.

Failed calls are included; their evaluation figures come from the failure
diagnostics, so an unreachable target shows up as a full budget with no
certified query.  Traces recorded before the query log existed still get the
timing columns, with the certification columns left blank.

Examples::

    python tools/prepare_ik_timing_report.py yor_agent/outputs/passive_video_tasks
    python tools/prepare_ik_timing_report.py run_a/trace.json run_b/trace.json --csv out.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path
from typing import Any

STAGES = (
    "grasp_candidate_generation",
    "virtual_candidate_evaluation",
    "motion",
    "actual_pose_certification",
)
SUBSTAGES = (
    "virtual_candidates",
    "collision_scene_points",
    "collision_check_rpc",
    "record_expansion",
    "shortlist_ranking",
    "post_ik_ranking",
)
COLUMNS = (
    ("run", 34),
    ("ok", 5),
    *((stage.replace("_", " ")[:14], 8) for stage in STAGES),
    *((substage[:12], 8) for substage in SUBSTAGES),
    ("pi_ik_s", 8),
    ("nom_n", 6),
    ("nom_s", 7),
    ("rob_n", 6),
    ("rob_s", 7),
    ("pairs", 7),
    ("safe", 5),
    ("q_first", 8),
    ("tier", 5),
    ("s_first", 8),
    ("exit", 6),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="trace.json files, or directories searched recursively for them",
    )
    parser.add_argument("--csv", type=Path, help="also write every row to this CSV")
    parser.add_argument(
        "--successes-only",
        action="store_true",
        help="skip failed prepare_for_manipulation calls",
    )
    return parser.parse_args()


def trace_files(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            files.extend(sorted(path.rglob("trace.json")))
        elif path.is_file():
            files.append(path)
        else:
            print(f"warning: {path} does not exist", file=sys.stderr)
    return files


def prepare_results(trace: dict[str, Any]) -> list[dict[str, Any]]:
    results = []
    for call in trace.get("primitive_calls", []):
        result = call.get("result") if isinstance(call, dict) else None
        if isinstance(result, dict) and result.get("primitive") == "prepare_for_manipulation":
            results.append(result)
    return results


def first_certified(
    log: list[dict[str, Any]], minimum_feasible_grasps: int
) -> tuple[int | None, int | None]:
    """Return (queries spent, base tier) when the first base became certified."""

    converged_per_base: dict[int, int] = {}
    for row in log:
        if row.get("robustness_variant") != "nominal" or not row.get("ik_converged"):
            continue
        base = int(row["base_index"])
        converged_per_base[base] = converged_per_base.get(base, 0) + 1
        if converged_per_base[base] >= minimum_feasible_grasps:
            return int(row["order"]) + 1, int(row["movement_cost_tier"])
    return None, None


def summarise(run: str, result: dict[str, Any]) -> dict[str, Any]:
    # Successful calls carry the evaluation fields at the top level; failed
    # calls carry whatever the search reached inside ``diagnostics``. A search
    # that certified a base whose motion was then refused has no diagnostics
    # but records its query log and batches at the top level before moving.
    evaluation = result if result.get("success") else dict(result.get("diagnostics") or {})
    if not result.get("success") and "pi_ik_query_log" not in evaluation:
        for key in ("pi_ik_query_log", "pi_batches", "minimum_feasible_grasps"):
            if key in result:
                evaluation[key] = result[key]
    stages = result.get("stage_timings_s") or {}
    substages = (
        result.get("evaluation_substage_timings_s")
        or evaluation.get("substage_timings_s")
        or {}
    )
    batches = evaluation.get("pi_batches") or []
    nominal = [b for b in batches if str(b.get("stage", "")).startswith("nominal")]
    robustness = [b for b in batches if b.get("stage") == "robustness"]
    log = evaluation.get("pi_ik_query_log")
    row: dict[str, Any] = {
        "run": run,
        "ok": bool(result.get("success")),
        "reason": str(result.get("reason", ""))[:80],
        **{stage: stages.get(stage) for stage in STAGES},
        **{substage: substages.get(substage) for substage in SUBSTAGES},
        "pi_ik_s": evaluation.get("pi_ik_compute_elapsed_s"),
        "nom_n": sum(int(b.get("candidate_count", 0)) for b in nominal) or None,
        "nom_s": sum(float(b.get("rpc_elapsed_s", 0.0)) for b in nominal) or None,
        "rob_n": sum(int(b.get("candidate_count", 0)) for b in robustness) or None,
        "rob_s": sum(float(b.get("rpc_elapsed_s", 0.0)) for b in robustness) or None,
        "pairs": evaluation.get("pi_eligible_pair_count"),
        "safe": evaluation.get("collision_safe_grasp_count"),
        "q_first": None,
        "tier": None,
        "s_first": None,
        "exit": (
            "early" if evaluation.get("pi_ik_early_exit")
            else "budget" if evaluation.get("pi_ik_budget_exhausted")
            else ""
        ),
        "reused_certificate": result.get("reused_virtual_certificate"),
        "motion_executed": result.get("motion_executed"),
    }
    if isinstance(log, list):
        queries, tier = first_certified(
            log, int(evaluation.get("minimum_feasible_grasps", 1))
        )
        row["q_first"] = queries
        row["tier"] = tier
        if queries is not None:
            # Batch granularity: the wall clock at the end of the batch that
            # held the certifying query.
            batch_index = int(log[queries - 1]["batch_index"])
            row["s_first"] = sum(
                float(b.get("rpc_elapsed_s", 0.0)) for b in batches[: batch_index + 1]
            )
    return row


def fmt(value: Any, width: int) -> str:
    if value is None or value == "":
        text = "-"
    elif isinstance(value, bool):
        text = "yes" if value else "no"
    elif isinstance(value, float):
        text = f"{value:.2f}"
    else:
        text = str(value)
    return text[:width].rjust(width)


def main() -> int:
    args = parse_args()
    rows: list[dict[str, Any]] = []
    for path in trace_files(args.paths):
        try:
            trace = json.loads(path.read_text())
        except (OSError, ValueError) as exc:
            print(f"warning: skipping {path}: {exc}", file=sys.stderr)
            continue
        run = path.parent.name
        for index, result in enumerate(prepare_results(trace)):
            if args.successes_only and not result.get("success"):
                continue
            rows.append(summarise(f"{run}#{index}", result))
    if not rows:
        print("no prepare_for_manipulation calls found", file=sys.stderr)
        return 1

    keys = ["run", "ok", *STAGES, *SUBSTAGES, "pi_ik_s", "nom_n", "nom_s", "rob_n",
            "rob_s", "pairs", "safe", "q_first", "tier", "s_first", "exit"]
    print(" ".join(name.rjust(width) for name, width in COLUMNS))
    for row in rows:
        print(" ".join(fmt(row[key], width) for key, (_, width) in zip(keys, COLUMNS)))

    successes = [row for row in rows if row["ok"]]
    print(f"\n{len(rows)} calls, {len(successes)} succeeded; medians over successes:")
    for key in keys[2:]:
        values = [
            row[key] for row in successes
            if isinstance(row[key], (int, float)) and not isinstance(row[key], bool)
        ]
        if values:
            print(
                f"  {key:>28}: median {statistics.median(values):8.3f}"
                f"  min {min(values):8.3f}  max {max(values):8.3f}  (n={len(values)})"
            )
    if args.csv:
        with args.csv.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
