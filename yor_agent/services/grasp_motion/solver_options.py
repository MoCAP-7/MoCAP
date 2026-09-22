"""Per-request cuRobo solver knobs that the planner API does not expose.

cuRoboV2's ``MotionPlanner`` hard-codes how many time-optimal *finetune*
passes each trajectory optimization runs: ``plan_cspace`` runs 3 extra passes
on every attempt, ``_plan_pose_single`` 1 (3 once graph seeding starts) and
the goalset path uses the solver default of 1. Every pass is a full fixed
100-iteration L-BFGS solve, so ``goto_pose("home")`` costs four solves and a
grasp segment two. The Pi executes waypoints at a fixed 0.1 s pace and ignores
the trajectory's time parametrization, so those passes buy nothing here.

``cap_finetune_attempts`` rewrites the keyword arguments of one
``TrajOptSolver.solve_pose`` / ``solve_cspace`` call: the pass count is capped
(never raised) at the requested value. Collision, joint-limit and tool-pose
checks are unchanged; only the number of refinement solves drops. This module
is free of ``curobo`` imports so it is unit-tested on a development machine.
"""

from __future__ import annotations

from typing import Any

FINETUNE_ATTEMPTS_MAX = 3
# cuRobo TrajOptSolver.solve_pose / solve_cspace default when the caller
# does not pass finetune_attempts (motion_planner goalset path).
CUROBO_DEFAULT_FINETUNE_ATTEMPTS = 1


def parse_finetune_cap(value: Any) -> int | None:
    """Validate an optional per-request ``finetune_attempts`` cap."""

    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("finetune_attempts must be an integer")
    cap = int(value)
    if not 0 <= cap <= FINETUNE_ATTEMPTS_MAX:
        raise ValueError(f"finetune_attempts must be in [0, {FINETUNE_ATTEMPTS_MAX}]")
    return cap


def cap_finetune_attempts(kwargs: dict[str, Any], cap: int | None) -> dict[str, Any]:
    """Return solver kwargs with ``finetune_attempts`` capped at ``cap``."""

    if cap is None:
        return kwargs
    current = kwargs.get("finetune_attempts", CUROBO_DEFAULT_FINETUNE_ATTEMPTS)
    capped = dict(kwargs)
    capped["finetune_attempts"] = min(int(current), int(cap))
    return capped


def install_finetune_cap(trajopt_solver: Any, cap_getter: Any) -> list[str]:
    """Wrap ``solve_pose`` / ``solve_cspace`` so each call honours ``cap_getter()``.

    Instance-level wrappers only touch Python keyword arguments; the CUDA
    graphs captured inside the solver see identical tensor shapes. Returns the
    method names that were wrapped (empty when the solver refuses attributes).
    """

    wrapped: list[str] = []
    for name in ("solve_pose", "solve_cspace"):
        original = getattr(trajopt_solver, name, None)
        if original is None:
            continue

        def make(original_method: Any) -> Any:
            def solve(*args: Any, **kwargs: Any) -> Any:
                return original_method(*args, **cap_finetune_attempts(kwargs, cap_getter()))

            solve.__name__ = getattr(original_method, "__name__", "solve")
            solve.__wrapped__ = original_method  # type: ignore[attr-defined]
            return solve

        try:
            setattr(trajopt_solver, name, make(original))
        except (AttributeError, TypeError):
            continue
        wrapped.append(name)
    return wrapped
