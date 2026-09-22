"""Comparison-experiment episodes of Web UI runs.

The episode layout, the operator review and the report are the repository's
``baselines/nav/experiment`` protocol, so YOR's conditions and the navigation
baselines are recorded the same way.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
#: YOR's conditions write to ``outputs/yor/<condition>/<experiment>/``.
OUTPUT_ROOT = "outputs/yor"


def experiment_protocol() -> ModuleType:
    """The shared protocol module, importable from any launcher of YOR."""

    if str(REPOSITORY_ROOT) not in sys.path:
        sys.path.insert(0, str(REPOSITORY_ROOT))
    from baselines.nav.experiment import protocol

    return protocol


def default_start_label() -> str:
    return str(experiment_protocol().DEFAULT_START_LABEL)


def experiment_block(*, name: str, start_label: str, condition: str) -> dict[str, str]:
    """The ``experiment`` config block that turns Web UI runs into episodes."""

    protocol = experiment_protocol()
    return {
        "name": protocol.checked_name(name, "--experiment"),
        "start_label": protocol.checked_name(start_label, "--start-label"),
        "condition": protocol.checked_name(condition, "condition"),
        "output_root": f"{OUTPUT_ROOT}/{condition}",
    }


def planner_condition(config: Mapping[str, Any]) -> str:
    """The condition of a run's episodes.

    A config that names its own condition (``comparison_condition``, as a
    baseline config does) keeps it; otherwise ``ours`` with the passive-video
    startup planner and ``no_prior`` without it.
    """

    named = config.get("comparison_condition")
    if named:
        return str(named)
    planner = config.get("navigation_planner") or {}
    return "ours" if bool(planner.get("enabled", False)) else "no_prior"
