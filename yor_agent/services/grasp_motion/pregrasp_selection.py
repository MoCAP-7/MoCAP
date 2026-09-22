"""Select the pre-grasp a grasp plan approaches, without importing cuRobo.

cuRoboV2 plans a multi-member goalset with IK and trajectory optimisation only;
the graph seeding that routes a trajectory around obstacles runs only for a
single goal. ``select_pregrasp`` therefore plans every pre-grasp as one goalset
first and, when that reaches no member, plans the candidates one at a time in
request order until one is reached or the time budget is spent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any, Callable

PLANNED = "planned"
#: cuRobo returns no result when IK finds no solution.
NO_IK_SOLUTION = "no_ik_solution"
#: A result whose success flags are all false: IK was solved but trajectory
#: optimisation found no collision-free path.
NO_COLLISION_FREE_TRAJECTORY = "no_collision_free_trajectory"
SKIPPED_TIME_BUDGET = "skipped_time_budget"


def planned(result: Any) -> bool:
    success = getattr(result, "success", None)
    return result is not None and success is not None and bool(success.any())


def outcome(result: Any) -> str:
    if planned(result):
        return PLANNED
    return NO_IK_SOLUTION if result is None else NO_COLLISION_FREE_TRAJECTORY


@dataclass
class PregraspSelection:
    """The approach plan and how it was found."""

    #: The successful plan, or the last planning result when none succeeded.
    result: Any | None
    #: ``goalset`` when the joint goalset reached a member, ``single_goal`` when
    #: a one-at-a-time plan did, ``None`` when no pre-grasp was reached.
    mode: str | None
    #: The candidate a single-goal plan reached; a goalset plan reports its own.
    candidate_index: int | None
    attempts: list[dict[str, Any]] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return self.mode is not None


def select_pregrasp(
    candidate_count: int,
    plan_goalset: Callable[[], Any],
    plan_single_goal: Callable[[int], Any],
    *,
    single_goal_budget_s: float,
    clock: Callable[[], float] = time.monotonic,
) -> PregraspSelection:
    """Plan all pre-grasps as one goalset, then one at a time within a budget.

    ``plan_goalset`` plans every candidate together and ``plan_single_goal``
    plans candidate ``index`` alone; both return a cuRobo planning result or
    ``None``. A lone candidate is planned once, since its goalset already is a
    single goal. A one-at-a-time plan starts only while less than
    ``single_goal_budget_s`` has passed since the goalset plan started; a plan
    already running when the budget ends is completed.
    """

    if candidate_count < 1:
        raise ValueError("candidate_count must be positive")
    if not single_goal_budget_s >= 0.0:
        raise ValueError("single_goal_budget_s must be non-negative")
    started = clock()
    attempts: list[dict[str, Any]] = []

    def record(mode: str, index: int | None, result: Any, attempt_started: float) -> None:
        attempts.append(
            {
                "mode": mode,
                "candidate_index": index,
                "outcome": outcome(result),
                "elapsed_s": round(clock() - attempt_started, 3),
            }
        )

    result = plan_goalset()
    record("goalset", None, result, started)
    if planned(result):
        return PregraspSelection(result, "goalset", None, attempts)
    last = result
    if candidate_count > 1:
        for index in range(candidate_count):
            if clock() - started >= single_goal_budget_s:
                attempts.append(
                    {
                        "mode": "single_goal",
                        "skipped_candidate_indices": list(range(index, candidate_count)),
                        "outcome": SKIPPED_TIME_BUDGET,
                    }
                )
                break
            attempt_started = clock()
            single = plan_single_goal(index)
            record("single_goal", index, single, attempt_started)
            if planned(single):
                return PregraspSelection(single, "single_goal", index, attempts)
            if single is not None:
                last = single
    return PregraspSelection(last, None, None, attempts)
