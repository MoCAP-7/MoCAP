"""Deterministic policy model for the readiness-prior trial.

``ReadinessTrialModel`` keeps the outer agent loop, the executor, the trace,
the Web UI and the stop path exactly as an LLM run uses them. The only change
is where each turn's program comes from: every turn is one primitive call,
chosen from the previous turn's execution record, so every trial runs the same
protocol and conditions differ only in the configuration a run was started
with (for example the readiness prior switch).

The protocol is a round-robin over approach bearings around the object. Each
bearing is an offset from an anchor: the bearing docking chose on its own for
the first attempt (the prior's bearing when the prior is on, otherwise the
arrival direction), read back from that dock's metrics.

1. ``dock_to_visible_object(name)`` at the anchor bearing; at every later
   bearing ``dock_to_visible_object(name, approach_bearing_deg=...)``. When
   SAM3 found no instance for the name, the next name is tried at the same
   bearing; docking does not move the base before its detection succeeds. At
   the anchor bearing the name list is gone through ``name_rounds`` times; at
   every later bearing the name SAM3 last accepted is tried first and each
   other name once. Once the names are exhausted, or docking fails for any
   other reason, the next bearing is tried.
2. After a successful dock, ``prepare_for_manipulation(name, arm=...)`` with
   the first name, whichever name docked, and the dock's ``suggested_arm``. A
   SAM3 miss there is retried with the other names in order; a failed
   actual-pose certification is
   retried at the same bearing up to ``certification_retries`` times; a
   target beyond the primitive's initial distance limit (refused before any
   IK query) ends the bearing as ``prepare_too_far``; any other failure moves
   on to the next bearing.
3. Before the next bearing is docked, when the base is presumed to stand at
   the object, ``drive_straight(-retreat_distance_m)`` backs the base off
   first. That is the case when the bearing's dock succeeded, when it sent
   Nav2 any goal at all (a stall, an abort under way or a failed final
   visual check still leaves the base where Nav2 drove it), or when docking
   refused to start Nav2 because the base already stood next to an obstacle,
   so it is still where the previous dock left it. A dock that starts at the
   object is refused by that same clearance check, so without the retreat
   no later bearing would be driven. A dock that never drove (SAM3 misses,
   no plannable goal on the bearing) is followed by the next dock directly.
   A refused retreat is recorded and the next dock is tried anyway;
   ``retreat_distance_m`` 0 disables the retreat.
4. The first successful preparation ends the trial as solved; exhausting the
   bearings ends it as unsolved. An exception the primitives do not report as
   a failure (a ``TypeError`` or ``ValueError`` from a bad call, for example)
   ends the trial at once with that error as the reason.

The model knows the run's turn budget (``max_turns``). When the next program
would take the last turn it finishes unsolved with a turn-budget reason
instead of issuing another primitive call, so every run ends with a finish
reason and the printed trial summary rather than the loop's budget cut-off.

The finish reason and the printed trial summary carry the per-bearing
outcomes, the number of dock calls and Nav2 goals sent, the Pi IK queries
spent and the retreats made, which is what the experiment compares between
conditions.

The readiness prior matches its passive-video event by the words of the object
name, so every fallback name has to resolve to the same event;
``yor_agent.experiments.readiness_trial`` checks that at launch.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
import math
import re
import threading
from typing import Any

from ..exceptions import Stopped
from .llm import DEFAULT_MODEL_NAMES, LLM

DOCK = "dock_to_visible_object"
PREPARE = "prepare_for_manipulation"
# The primitive the retreat between two bearings uses, with a negative
# distance; it is also the name of the model's phase while one is under way.
RETREAT = "drive_straight"
# Reasons the two primitives report when SAM3 returned no usable instance for
# the prompt. Both are raised before the primitive commands any base motion.
SAM3_MISS_MARKERS = ("SAM3 found no instance", "SAM3 produced no usable mask")
# Docking's reason when Nav2 could not get away from an obstacle: the
# start-clearance check refused to start next to one, or Nav2 gave up blocked
# after a recovery. Either way the base still stands close to where the
# previous dock left it. The marker decides the retreat only for the refusal
# before any goal was sent; a dock that sent one is presumed at the object
# by that alone.
BLOCKED_START_MARKER = "nav2_blocked_near_obstacle"
# prepare_for_manipulation's reason when the virtual pose certified but the
# fresh check at the arrival pose did not: the same bearing is worth another
# attempt before moving on.
CERTIFICATION_FAILED_MARKER = "actual_pose_grasp_certification_failed"
# prepare_for_manipulation's reason when the target's median camera distance
# is beyond its initial distance limit: refused before any Pi IK query.
TARGET_TOO_FAR_MARKER = "target_too_far_for_local_manipulation"
# Docking's reason for a bearing it never drove towards because every goal
# distance failed the Nav2 plan check; such an attempt sends no Nav2 goal.
NO_PLANNABLE_GOAL_REASON = "no_plannable_goal_on_bearing"
# Nav2's reason when the operator stopped the base; an attempt the stop
# pre-empted records this without a goal pose and sent nothing.
OPERATOR_STOP_REASON = "operator_stop"
#: Offsets from the anchor bearing, in the order they are tried. The first
#: entry has to be 0: that bearing is the one docking chooses on its own.
DEFAULT_BEARING_OFFSETS_DEG = (0.0, 45.0, -45.0, 90.0, -90.0, 135.0, -135.0, 180.0)
DEFAULT_CERTIFICATION_RETRIES = 1
#: How far the base reverses before the next bearing is docked once it stood
#: at the object; 0 disables the retreat.
DEFAULT_RETREAT_DISTANCE_M = 0.30
MAX_RETREAT_DISTANCE_M = 1.0
DEFAULT_STOP_REASON = "operator requested stop"
MAX_FINISH_REASON_CHARS = 1200
# Per-bearing reasons are shortened to this in the finish reason; the printed
# trial summary keeps them whole.
BEARING_REASON_CHARS = 48
_COLLAPSED_LIST = re.compile(r"^<(?:list|tuple) len=(\d+)>$")


def normalize_object_names(names: Iterable[Any] | None) -> tuple[str, ...]:
    """Non-empty, stripped, de-duplicated names in the order given."""

    if names is None or isinstance(names, str):
        raise ValueError("object_names must be a list of SAM3 names")
    result: list[str] = []
    for name in names:
        text = str(name).strip()
        if not text:
            raise ValueError("object_names must not contain an empty name")
        if text not in result:
            result.append(text)
    if not result:
        raise ValueError("object_names must contain at least one name")
    return tuple(result)


def normalize_bearing_offsets_deg(offsets: Iterable[Any] | None) -> tuple[float, ...]:
    """Finite offsets in degrees, the first of which must be 0."""

    if offsets is None or isinstance(offsets, (str, bytes)):
        raise ValueError("bearing_offsets_deg must be a list of angles in degrees")
    result: list[float] = []
    for value in offsets:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise ValueError("bearing_offsets_deg entries must be finite angles in degrees")
        result.append(float(value))
    if not result or result[0] != 0.0:
        raise ValueError(
            "bearing_offsets_deg must start with 0, the bearing docking chooses on its own"
        )
    return tuple(result)


def normalize_retreat_distance_m(value: Any) -> float:
    """A finite retreat distance in metres from 0 (no retreat) to the maximum."""

    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 0.0 <= float(value) <= MAX_RETREAT_DISTANCE_M
    ):
        raise ValueError(
            f"retreat_distance_m must be a finite distance in [0, {MAX_RETREAT_DISTANCE_M}]"
        )
    return float(value)


def wrap_degrees(value: float) -> float:
    """Wrap an angle in degrees to (-180, 180]."""

    wrapped = (float(value) + 180.0) % 360.0 - 180.0
    return 180.0 if wrapped == -180.0 else wrapped


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(float(value)) else None


def _list_length(value: Any) -> int | None:
    """Length of a list, also when the trace collapsed it to ``<list len=N>``."""

    if isinstance(value, (list, tuple)):
        return len(value)
    if isinstance(value, str):
        match = _COLLAPSED_LIST.match(value.strip())
        if match:
            return int(match.group(1))
    return None


def anchor_bearing_deg(result: Mapping[str, Any]) -> float | None:
    """The bearing a dock result docked from, in degrees, or ``None``.

    Read from ``metrics["reference_bearing_rad"]`` (the bearing the first
    attempt was built on), else the first goal attempt's bearing, else the
    arrival bearing. Docking records these before it drives, so a failed dock
    reports its bearing too; a dock that never detected the object has none.
    """

    metrics = result.get("metrics") if isinstance(result, Mapping) else None
    if not isinstance(metrics, Mapping):
        return None
    candidates: list[Any] = [metrics.get("reference_bearing_rad")]
    attempts = metrics.get("goal_attempts")
    if isinstance(attempts, list) and attempts and isinstance(attempts[0], Mapping):
        candidates.append(attempts[0].get("bearing_rad"))
    candidates.append(metrics.get("arrival_bearing_rad"))
    for candidate in candidates:
        bearing = _finite_number(candidate)
        if bearing is not None:
            return math.degrees(bearing)
    return None


def attempt_sent_nav2_goal(attempt: Any) -> bool:
    """Whether one ``goal_attempts`` entry of a dock result sent a Nav2 goal.

    Docking appends one entry per bearing it gave up on without driving and
    one per goal it navigated. An entry that records its Nav2 result under
    ``nav2`` sent a goal iff that result is present. An entry without that key
    sent one iff it carries the goal pose it was built for and its reason is
    neither the plan-check exhaustion nor an operator stop, which pre-empts
    the goal; in such traces a goal the operator stopped after it was sent
    is not counted, which only affects stopped runs. An entry the trace
    collapsed to something other than a mapping counts as sent.
    """

    if not isinstance(attempt, Mapping):
        return True
    if "nav2" in attempt:
        return isinstance(attempt["nav2"], Mapping)
    if attempt.get("nav2_goal_xy_yaw") is None:
        return False
    return attempt.get("reason") not in (NO_PLANNABLE_GOAL_REASON, OPERATOR_STOP_REASON)


def nav2_goal_count(result: Mapping[str, Any]) -> int:
    """How many Nav2 goals a dock result sent (see ``attempt_sent_nav2_goal``)."""

    metrics = result.get("metrics") if isinstance(result, Mapping) else None
    if not isinstance(metrics, Mapping):
        return 0
    attempts = metrics.get("goal_attempts")
    if not isinstance(attempts, list):
        return _list_length(attempts) or 0
    return sum(1 for attempt in attempts if attempt_sent_nav2_goal(attempt))


def dock_navigated(result: Mapping[str, Any]) -> bool:
    """Whether a dock result drove the base towards the object at all.

    Docking writes ``navigated`` on every ``goal_attempts`` entry: true for a
    goal it sent Nav2, false for a bearing it gave up on without driving, a
    goal the operator's stop pre-empted, or a fallback distance the base
    already stood at. Wherever a sent goal then ended (docked, stalled,
    aborted, a failed final check), the base stands where Nav2 drove it. An
    entry the trace collapsed to something other than a mapping counts as
    navigated, as ``nav2_goal_count`` counts it as sent.
    """

    metrics = result.get("metrics") if isinstance(result, Mapping) else None
    if not isinstance(metrics, Mapping):
        return False
    attempts = metrics.get("goal_attempts")
    if not isinstance(attempts, list):
        return bool(_list_length(attempts))
    return any(
        not isinstance(attempt, Mapping) or attempt.get("navigated") is True
        for attempt in attempts
    )


def prepare_ik_query_count(result: Mapping[str, Any]) -> int:
    """Pi IK queries one prepare_for_manipulation call spent.

    A successful call reports ``ik_query_count``; a failed one carries the
    per-query log either in its search-failure diagnostics or, when the
    search certified a pose and a later stage failed, at the top level.
    """

    if not isinstance(result, Mapping):
        return 0
    if result.get("success", False):
        count = result.get("ik_query_count")
        if isinstance(count, int) and not isinstance(count, bool):
            return count
    diagnostics = result.get("diagnostics")
    for source in (diagnostics, result):
        if not isinstance(source, Mapping):
            continue
        length = _list_length(source.get("pi_ik_query_log"))
        if length is not None:
            return length
        count = source.get("ik_query_count")
        if isinstance(count, int) and not isinstance(count, bool):
            return count
    return 0


@dataclass(frozen=True)
class PrimitiveOutcome:
    """How one primitive call ended, read from an execution record."""

    succeeded: bool
    reason: str
    result: Mapping[str, Any]
    # The program raised something other than a primitive failure: a bad call
    # or a fault the primitives do not report as a result.
    unexpected_error: bool = False

    @property
    def sam3_miss(self) -> bool:
        return not self.succeeded and any(
            marker in self.reason for marker in SAM3_MISS_MARKERS
        )

    @property
    def certification_failed(self) -> bool:
        return not self.succeeded and CERTIFICATION_FAILED_MARKER in self.reason

    @property
    def target_too_far(self) -> bool:
        return not self.succeeded and TARGET_TOO_FAR_MARKER in self.reason

    @property
    def blocked_start(self) -> bool:
        return not self.succeeded and BLOCKED_START_MARKER in self.reason

    @property
    def navigated(self) -> bool:
        return dock_navigated(self.result)


def primitive_outcome(
    execution: Mapping[str, Any] | None, primitive: str
) -> PrimitiveOutcome:
    """Read the outcome of the last ``primitive`` call in ``execution``."""

    if not isinstance(execution, Mapping):
        return PrimitiveOutcome(False, "no execution record", {})
    interrupted = execution.get("interrupted_by")
    if isinstance(interrupted, Mapping):
        raw = interrupted.get("result")
        result = raw if isinstance(raw, Mapping) else {}
        reason = str(interrupted.get("reason") or result.get("reason") or "unknown")
        if interrupted.get("primitive") != primitive:
            reason = f"{interrupted.get('primitive')} failed: {reason}"
        return PrimitiveOutcome(False, reason, result)
    error = execution.get("error")
    if isinstance(error, Mapping):
        return PrimitiveOutcome(
            False,
            f"{error.get('type', 'Error')}: {error.get('message', '')}",
            {},
            unexpected_error=True,
        )
    calls = [
        call
        for call in execution.get("primitive_calls") or []
        if isinstance(call, Mapping) and call.get("name") == primitive
    ]
    if not calls:
        return PrimitiveOutcome(False, f"{primitive} was not called", {})
    raw = calls[-1].get("result")
    result = raw if isinstance(raw, Mapping) else {}
    if not result.get("success", False):
        return PrimitiveOutcome(False, str(result.get("reason") or "unknown"), result)
    return PrimitiveOutcome(True, str(result.get("reason") or "succeeded"), result)


class ReadinessTrialModel(LLM):
    """Policy model that runs the bearing round-robin dock-then-prepare trial."""

    def __init__(
        self,
        config: Mapping[str, Any] | None = None,
        *,
        stop_event: threading.Event | None = None,
        stop_reason: Callable[[], str] | None = None,
    ) -> None:
        config = dict(config or {})
        names = config.pop("object_names", None)
        rounds = config.pop("name_rounds", 1)
        default_arm = config.pop("default_arm", "right")
        offsets = config.pop("bearing_offsets_deg", DEFAULT_BEARING_OFFSETS_DEG)
        retries = config.pop("certification_retries", DEFAULT_CERTIFICATION_RETRIES)
        retreat = config.pop("retreat_distance_m", DEFAULT_RETREAT_DISTANCE_M)
        max_turns = config.pop("max_turns", None)
        config["provider"] = "scripted"
        config.setdefault("name", DEFAULT_MODEL_NAMES["scripted"])
        super().__init__(config)
        self.object_names = normalize_object_names(names)
        if isinstance(rounds, bool) or not isinstance(rounds, int) or rounds < 1:
            raise ValueError("name_rounds must be a positive integer")
        self.name_rounds = rounds
        if default_arm not in {"left", "right"}:
            raise ValueError("default_arm must be 'left' or 'right'")
        # Used only when a successful dock reports no suggested arm.
        self.default_arm = default_arm
        self.bearing_offsets_deg = normalize_bearing_offsets_deg(offsets)
        if isinstance(retries, bool) or not isinstance(retries, int) or retries < 0:
            raise ValueError("certification_retries must be a non-negative integer")
        self.certification_retries = retries
        self.retreat_distance_m = normalize_retreat_distance_m(retreat)
        if max_turns is not None and (
            isinstance(max_turns, bool) or not isinstance(max_turns, int) or max_turns < 1
        ):
            raise ValueError("max_turns must be a positive integer or None")
        # The run's turn budget (the agent's ``max_turns``); ``None`` leaves
        # the budget to the loop, which then ends a long run without a finish.
        self.max_turns = max_turns
        self._stop_event = stop_event if stop_event is not None else threading.Event()
        self.stop_reason = stop_reason
        self._dock_names: tuple[str, ...] = ()
        self._prepare_names: tuple[str, ...] = ()
        # The name SAM3 last accepted at docking, whether or not the dock then
        # succeeded; later bearings try it first.
        self._accepted_name: str | None = None
        self._phase = DOCK
        self._dock_attempt = 0
        self._prepare_attempt = 0
        self._certification_retries_used = 0
        self._arm: str | None = None
        self._bearing_index = 0
        self._anchor_bearing_deg: float | None = None
        self._bearings: list[dict[str, Any]] = []
        self._solved_bearing: int | None = None
        # Whether the base is presumed to stand at the object: set by a
        # successful dock, by any dock that sent Nav2 a goal (a stall, an
        # abort or a failed final check still leaves the base where Nav2
        # drove it) and by a dock refused for standing next to an obstacle;
        # cleared once a retreat was made.
        self._base_at_object = False
        self._dock_calls = 0
        self._nav2_goals = 0
        self._ik_queries = 0
        self._retreats = 0
        self._finish_reason: str | None = None
        self._execution: Mapping[str, Any] | None = None

    # ------------------------------------------------------------------
    # Loop entry points
    # ------------------------------------------------------------------

    def format_feedback(
        self,
        code: str,
        execution: Mapping[str, Any],
        observation: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        # The loop hands every execution record through here, which is where
        # the next program is decided from.
        self._execution = execution
        return super().format_feedback(code, execution, observation)

    def _generate(self, messages: list[Mapping[str, Any]]) -> str:
        del messages  # The sequence is fixed; the transcript is only traced.
        if self._stop_event.is_set():
            reason = DEFAULT_STOP_REASON
            if callable(self.stop_reason):
                reason = str(self.stop_reason() or "").strip() or reason
            raise Stopped(reason)
        program = self._next_program()
        self.n_calls += 1
        return program

    def trial_summary(self) -> dict[str, Any]:
        """Per-bearing outcomes and the cost counters, as the finish prints them."""

        return {
            "solved": self._solved_bearing is not None,
            "solved_bearing": self._solved_bearing,
            "anchor_bearing_deg": self._anchor_bearing_deg,
            "bearing_offsets_deg": list(self.bearing_offsets_deg),
            "bearings": [dict(bearing) for bearing in self._bearings],
            "dock_calls": self._dock_calls,
            "nav2_goals": self._nav2_goals,
            "ik_queries": self._ik_queries,
            "retreats": self._retreats,
        }

    # ------------------------------------------------------------------
    # Protocol
    # ------------------------------------------------------------------

    def _next_program(self) -> str:
        execution, self._execution = self._execution, None
        if self._finish_reason is not None:
            return self._finish_program()
        if not self._bearings:
            self._start_bearing()
            program = self._dock_program()
        elif self._phase == DOCK:
            program = self._after_dock(primitive_outcome(execution, DOCK))
        elif self._phase == RETREAT:
            program = self._after_retreat(primitive_outcome(execution, RETREAT))
        else:
            program = self._after_prepare(primitive_outcome(execution, PREPARE))
        if self._finish_reason is None and self._next_turn_is_last():
            # The record above is accounted for; the call it asks for would
            # leave no turn for finish(), so finish now instead.
            return self._finish_on_turn_budget()
        return program

    def _next_turn_is_last(self) -> bool:
        # ``n_calls`` counts the programs already issued; the one being built
        # takes turn ``n_calls + 1``.
        return self.max_turns is not None and self.n_calls + 1 >= self.max_turns

    def _start_bearing(self) -> None:
        offset = self.bearing_offsets_deg[self._bearing_index]
        approach: float | None = None
        if self._bearing_index > 0:
            assert self._anchor_bearing_deg is not None
            approach = round(wrap_degrees(self._anchor_bearing_deg + offset), 3)
        if self._bearing_index == 0 or self._accepted_name is None:
            self._dock_names = self.object_names * (
                self.name_rounds if self._bearing_index == 0 else 1
            )
        else:
            # One round, the name SAM3 last accepted first.
            self._dock_names = (
                self._accepted_name,
                *(name for name in self.object_names if name != self._accepted_name),
            )
        self._bearings.append(
            {
                "bearing": self._bearing_index,
                "offset_deg": offset,
                "approach_bearing_deg": approach,
                "outcome": None,
                "dock_calls": 0,
                "nav2_goals": 0,
                "dock_reason": None,
                "docked_name": None,
                "prepare_calls": 0,
                "prepare_reason": None,
                "prepare_target_distance_m": None,
                "ik_queries": 0,
                "retreat_calls": 0,
                "retreat_reason": None,
                "retreat_success": None,
            }
        )
        self._phase = DOCK
        self._dock_attempt = 0
        self._prepare_attempt = 0
        self._certification_retries_used = 0
        self._arm = None

    def _after_dock(self, outcome: PrimitiveOutcome) -> str:
        bearing = self._bearings[-1]
        goals = nav2_goal_count(outcome.result)
        bearing["dock_calls"] += 1
        bearing["nav2_goals"] += goals
        bearing["dock_reason"] = outcome.reason
        self._dock_calls += 1
        self._nav2_goals += goals
        if self._anchor_bearing_deg is None:
            self._anchor_bearing_deg = anchor_bearing_deg(outcome.result)
        if outcome.unexpected_error:
            return self._abort(DOCK, outcome.reason)
        name = self._dock_names[self._dock_attempt]
        if not outcome.sam3_miss:
            self._accepted_name = name
        if outcome.succeeded or outcome.navigated or outcome.blocked_start:
            # The base ended near the object: Nav2 drove it there, whatever
            # the goal then ended as, or it never left where the previous
            # dock put it. A dock started there is refused by the
            # start-clearance check.
            self._base_at_object = True
        if outcome.succeeded:
            arm = outcome.result.get("suggested_arm")
            self._arm = arm if arm in {"left", "right"} else self.default_arm
            bearing["docked_name"] = name
            # Prepare always starts from the first name, whichever name docked.
            # A later name is a fallback for detection at docking range, where
            # the object is small in the image; at prepare's range the first
            # name finds it, while a fallback name can match another object
            # or nothing once the base has moved closer.
            self._prepare_names = tuple(self.object_names)
            self._prepare_attempt = 0
            self._certification_retries_used = 0
            self._phase = PREPARE
            return self._prepare_program()
        if outcome.sam3_miss and self._dock_attempt + 1 < len(self._dock_names):
            self._dock_attempt += 1
            return self._dock_program()
        if outcome.sam3_miss:
            return self._end_bearing("dock_names_exhausted")
        return self._end_bearing("dock_failed")

    def _after_prepare(self, outcome: PrimitiveOutcome) -> str:
        bearing = self._bearings[-1]
        queries = prepare_ik_query_count(outcome.result)
        bearing["prepare_calls"] += 1
        bearing["ik_queries"] += queries
        bearing["prepare_reason"] = outcome.reason
        self._ik_queries += queries
        if outcome.unexpected_error:
            return self._abort(PREPARE, outcome.reason)
        name = self._prepare_names[self._prepare_attempt]
        if outcome.succeeded:
            bearing["outcome"] = "solved"
            self._solved_bearing = self._bearing_index
            return self._finish(
                f"trial solved at bearing {self._bearing_index} "
                f"(offset {bearing['offset_deg']:+g} deg): "
                f"prepare_for_manipulation {outcome.reason} with the {self._arm} "
                f"arm for {name!r}; {self._bearings_text()}; {self._counts_text()}"
            )
        if outcome.sam3_miss and self._prepare_attempt + 1 < len(self._prepare_names):
            self._prepare_attempt += 1
            return self._prepare_program()
        if outcome.sam3_miss:
            return self._end_bearing("prepare_names_exhausted")
        if outcome.certification_failed:
            if self._certification_retries_used < self.certification_retries:
                self._certification_retries_used += 1
                return self._prepare_program()
            return self._end_bearing("certification_failed")
        if outcome.target_too_far:
            # Refused before the search ran: the dock left the target beyond
            # the primitive's initial distance limit, visible as its own outcome.
            bearing["prepare_target_distance_m"] = _finite_number(
                outcome.result.get("target_distance_m")
            )
            return self._end_bearing("prepare_too_far")
        return self._end_bearing("prepare_failed")

    def _after_retreat(self, outcome: PrimitiveOutcome) -> str:
        bearing = self._bearings[-1]
        bearing["retreat_calls"] += 1
        bearing["retreat_reason"] = outcome.reason
        bearing["retreat_success"] = outcome.succeeded
        self._retreats += 1
        # Whether or not the base moved, the next dock is tried from here; a
        # dock refused for standing next to an obstacle sets the flag again.
        self._base_at_object = False
        if outcome.unexpected_error:
            return self._abort(RETREAT, outcome.reason)
        return self._next_bearing()

    def _end_bearing(self, outcome: str) -> str:
        self._bearings[-1]["outcome"] = outcome
        attempted = len(self._bearings)
        total = len(self.bearing_offsets_deg)
        if attempted >= total:
            return self._finish(
                f"trial unsolved after {attempted} of {total} bearings; "
                f"{self._bearings_text()}; {self._counts_text()}"
            )
        if self._anchor_bearing_deg is None:
            # Only a dock that detected the object reports the bearing the
            # offsets are measured from; without one no bearing can be asked for.
            return self._finish(
                f"trial unsolved after {attempted} of {total} bearings: docking "
                "reported no reference bearing, so no further bearing could be "
                f"requested; {self._bearings_text()}; {self._counts_text()}"
            )
        if self.retreat_distance_m > 0.0 and self._base_at_object:
            # A dock started at the object is refused by the clearance
            # check, so back off before the next bearing is asked for.
            self._phase = RETREAT
            return self._retreat_program()
        return self._next_bearing()

    def _next_bearing(self) -> str:
        self._bearing_index += 1
        self._start_bearing()
        return self._dock_program()

    def _abort(self, primitive: str, reason: str) -> str:
        bearing = self._bearings[-1]
        if bearing["outcome"] is None:
            # A retreat aborts after its bearing has ended; that outcome stays.
            bearing["outcome"] = "aborted"
        return self._finish(
            f"trial aborted at bearing {self._bearing_index} "
            f"(offset {bearing['offset_deg']:+g} deg): {primitive} raised {reason}; "
            f"{self._bearings_text()}; {self._counts_text()}"
        )

    def _finish_on_turn_budget(self) -> str:
        # The current bearing may have been started for the call that cannot
        # be made any more, be part-way through its dock/prepare sequence, or
        # have ended and be waiting for the retreat before the next one.
        bearing = self._bearings[-1]
        if bearing["outcome"] is None:
            bearing["outcome"] = "turn_budget"
        attempted = len(self._bearings)
        total = len(self.bearing_offsets_deg)
        return self._finish(
            f"trial unsolved: turn budget reached ({self.max_turns} turns) at "
            f"bearing {self._bearing_index} (offset {bearing['offset_deg']:+g} deg), "
            f"{attempted} of {total} bearings started; "
            f"{self._bearings_text()}; {self._counts_text()}"
        )

    # ------------------------------------------------------------------
    # Programs and reason text
    # ------------------------------------------------------------------

    def _dock_program(self) -> str:
        name = self._dock_names[self._dock_attempt]
        bearing = self._bearings[-1]
        approach = bearing["approach_bearing_deg"]
        call = (
            f"dock_to_visible_object({name!r})"
            if approach is None
            else f"dock_to_visible_object({name!r}, approach_bearing_deg={approach!r})"
        )
        return (
            f"result = {call}\n"
            f"print({{'bearing': {bearing['bearing']!r}, "
            f"'approach_bearing_deg': {approach!r}, "
            "'reason': result.get('reason'), "
            "'suggested_arm': result.get('suggested_arm')})"
        )

    def _prepare_program(self) -> str:
        name = self._prepare_names[self._prepare_attempt]
        bearing = self._bearings[-1]
        return (
            f"result = prepare_for_manipulation({name!r}, arm={self._arm!r})\n"
            f"print({{'bearing': {bearing['bearing']!r}, "
            "'reason': result.get('reason'), "
            "'selected_base_pose': result.get('selected_base_pose'), "
            "'ik_query_count': result.get('ik_query_count')})"
        )

    def _retreat_program(self) -> str:
        bearing = self._bearings[-1]
        distance = self.retreat_distance_m
        return (
            f"result = drive_straight(-{distance!r})\n"
            f"print({{'bearing': {bearing['bearing']!r}, "
            f"'retreat_m': {distance!r}, "
            "'reason': result.get('reason')})"
        )

    def _finish(self, reason: str) -> str:
        self._finish_reason = reason[:MAX_FINISH_REASON_CHARS]
        return self._finish_program()

    def _finish_program(self) -> str:
        return (
            f"print({{'trial': {self.trial_summary()!r}}})\n"
            f"finish(reason={self._finish_reason!r})"
        )

    def _bearings_text(self) -> str:
        items = []
        for bearing in self._bearings:
            text = f"{bearing['offset_deg']:+g}: {bearing['outcome']}"
            distance = bearing["prepare_target_distance_m"]
            detail = {
                "dock_failed": bearing["dock_reason"],
                "prepare_failed": bearing["prepare_reason"],
                "certification_failed": bearing["prepare_reason"],
                "prepare_too_far": None if distance is None else f"target {distance:.2f} m",
            }.get(bearing["outcome"])
            if detail:
                text += f" ({str(detail)[:BEARING_REASON_CHARS]})"
            items.append(text)
        return f"bearings [{'; '.join(items)}]"

    def _counts_text(self) -> str:
        return (
            f"{self._dock_calls} dock calls, {self._nav2_goals} Nav2 goals, "
            f"{self._ik_queries} Pi IK queries, {self._retreats} retreats"
        )
