"""Import the pinned CoW checkout without modifying any of its files."""

from __future__ import annotations

import importlib
import inspect
import json
import subprocess
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any


BASELINE_DIR = Path(__file__).resolve().parent
LOCK_PATH = BASELINE_DIR / "upstream.lock"

# CoW's localizer names in hparams/*.json, keyed by this baseline's localizer option.
LOCALIZER_HPARAM_KEYS = {
    "clip_grad": "grad-b32-openai",
    "owl": "owl-b32-openai",
}


def load_lock() -> dict[str, str]:
    return json.loads(LOCK_PATH.read_text(encoding="utf-8"))


def verify_checkout(repo: Path) -> str:
    """Refuse a checkout that is not the reviewed commit or has local edits."""

    if not (repo / "src" / "models" / "agent_fbe.py").is_file():
        raise FileNotFoundError(f"CoW checkout not found: {repo}")
    expected = load_lock()["commit"]
    actual = _git(repo, "rev-parse", "HEAD")
    if actual != expected:
        raise RuntimeError(f"refusing unreviewed CoW commit {actual}; expected {expected}")
    modified = _git(repo, "status", "--porcelain", "--untracked-files=no")
    if modified:
        raise RuntimeError(f"CoW checkout has local modifications:\n{modified}")
    return actual


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


@dataclass(frozen=True)
class CowModules:
    repo: Path
    commit: str
    exploration: types.ModuleType
    agent_mode: Any
    agent_class: type
    hparams: dict[str, float]


def import_cow(repo: str | Path, *, localizer: str, verify: bool = True) -> CowModules:
    """Import CoW's agent for ``localizer`` from ``repo``.

    Two import-time dependencies of the simulator code are satisfied from
    outside: the X11 client library that ``src/shared/utils.py`` imports for
    starting simulator displays, and the OWL-ViT box helper that newer
    ``transformers`` releases moved to another module.
    """

    if localizer not in LOCALIZER_HPARAM_KEYS:
        raise ValueError(f"localizer must be one of {sorted(LOCALIZER_HPARAM_KEYS)}")
    root = Path(repo).expanduser().resolve()
    commit = verify_checkout(root) if verify else "unverified"
    _provide_display_library_if_missing()
    if localizer == "owl":
        _provide_owlvit_box_helper()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    exploration = importlib.import_module("src.models.exploration.frontier_based_exploration")
    _require_inside(exploration, root)
    agent_module_name = (
        "src.models.agent_fbe_grad" if localizer == "clip_grad" else "src.models.agent_fbe_owl"
    )
    agent_module = importlib.import_module(agent_module_name)
    _require_inside(agent_module, root)
    agent_class = agent_module.AgentFbeGrad if localizer == "clip_grad" else agent_module.AgentFbeOwl
    agent_mode = importlib.import_module("src.models.agent_mode").AgentMode
    _require_stop_test(exploration)
    hparams = json.loads((root / "hparams" / "robo.json").read_text(encoding="utf-8"))
    return CowModules(
        repo=root,
        commit=commit,
        exploration=exploration,
        agent_mode=agent_mode,
        agent_class=agent_class,
        hparams=hparams,
    )


def localizer_threshold(modules: CowModules, localizer: str) -> float:
    """CoW's own RoboTHOR/PASTURE threshold for the B/32 localizer."""

    return float(modules.hparams[LOCALIZER_HPARAM_KEYS[localizer]])


def set_stop_radius(exploration: types.ModuleType, *, voxel_size_m: float, stop_radius_m: float) -> None:
    """Change the distance at which CoW issues Stop next to a localized target.

    CoW stops when the agent voxel is closer than ``1 / VOXEL_SIZE_M`` voxels
    to the target voxel. The module-level ``VOXEL_SIZE_M`` is read only by that
    test (the map itself uses the instance voxel size), so rebinding it moves
    the stop distance without changing the map resolution.
    """

    if not 0.0 < stop_radius_m <= 5.0:
        raise ValueError("stop_radius_m must be in (0, 5]")
    exploration.VOXEL_SIZE_M = voxel_size_m / stop_radius_m


def _require_stop_test(exploration: types.ModuleType) -> None:
    source = inspect.getsource(exploration.FrontierBasedExploration)
    if source.count("VOXEL_SIZE_M") != 1 or "< 1/VOXEL_SIZE_M" not in source:
        raise RuntimeError("CoW stop test no longer matches the reviewed commit")


def _require_inside(module: types.ModuleType, root: Path) -> None:
    location = Path(getattr(module, "__file__", "")).resolve()
    if root not in location.parents:
        raise ImportError(f"{module.__name__} resolved outside the CoW checkout: {location}")


def _provide_display_library_if_missing() -> None:
    try:
        importlib.import_module("Xlib.display")
    except ImportError:
        xlib = types.ModuleType("Xlib")
        display = types.ModuleType("Xlib.display")
        xlib.display = display
        sys.modules.setdefault("Xlib", xlib)
        sys.modules.setdefault("Xlib.display", display)


def _provide_owlvit_box_helper() -> None:
    feature_extraction = importlib.import_module(
        "transformers.models.owlvit.feature_extraction_owlvit"
    )
    if not hasattr(feature_extraction, "center_to_corners_format"):
        from transformers.image_transforms import center_to_corners_format

        feature_extraction.center_to_corners_format = center_to_corners_format
