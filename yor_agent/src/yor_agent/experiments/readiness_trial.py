"""Run the readiness-prior trial behind the usual Web UI.

The operator places the robot anywhere in the start area with the object in
view, opens the Web UI and presses Start. The run's policy is one of

- ``--policy llm`` (the default): GPT-5.6 Sol (``openai`` / ``gpt-5.6-sol``,
  with the config's other model settings; the Web UI form can pick another
  model for a run) writes and runs its own programs as in a normal run, on a
  task asking it to reach a base pose
  certified for grasping the object and finish there without grasping. The
  run stops after ``--time-limit-s`` (120 s by default) the way the Web UI's
  Stop does, and the primitives that move the arm or the gripper are hidden
  unless ``--allow-grasp`` is given, so the run measures readiness only;
- ``--policy scripted``: one fixed protocol
  (``yor_agent.models.scripted.ReadinessTrialModel``): dock to the object
  along a bearing, falling back through several SAM3 names, then
  ``prepare_for_manipulation`` with the arm the dock suggests; when that does
  not certify a base pose, back the base off the object and dock again from
  the next bearing around it, and finish once a pose certified or every
  bearing was tried. It reverses ``retreat_distance_m`` before the next
  bearing whenever the base stood at the object: its dock succeeded, sent
  Nav2 any goal (a stall or a failed final check still leaves the base where
  Nav2 drove it) or was refused for standing next to an obstacle (a dock
  started there is refused by the start-clearance check).

The Web UI's "Readiness prior" checkbox selects the condition of each run, so
trials with and without the prior share every other setting.

Compared with ``python -m yor_agent.launch``, this launcher

- disables the one-shot LLM navigation planner (``--navigation-planner``
  keeps it as configured for an LLM trial),
- lets docking fall back to farther, plan-checked goal distances along the
  requested bearing, so a wrong side still gets as close as it can and the
  local search runs (and is measured) instead of the dock failing outright,
- turns the docking heading toward the arm the dock suggests
  (``arm_side_heading_offset_deg``), so the object ends in front of that arm
  rather than between the two, where a few degrees of final heading decide
  how far the local search has to go,
- keeps a with-prior dock on the prior's bearing alone, so the bearings the
  trial asks for are the only ones it drives,
- pins the prior's passive-video event to the first SAM3 name
  (``readiness_prior.event_object``), so a fallback name that shares no word
  with it still docks with the same event,
- raises prepare_for_manipulation's Pi IK compute budget and initial target
  distance for these trials only, lets it try the next certified base poses
  when a motion is refused and, once every certified pose has been refused,
  continue the IK search into the farther tiers its early exit skipped, and
- records ``--start-label`` in every run's config (``trial.start_label``) and
  writes traces under ``outputs/readiness_trials/<task id>``.

Example::

    cd /home/yor/codefield/YOR/yor_agent
    source scripts/source_nav2_env.sh
    /home/yor/venvs/capx-jetson/bin/python -m yor_agent.experiments.readiness_trial \\
      --config configs/passive_video_tasks.yaml --task-id find_can_on_table \\
      --readiness-events ../nav_planner/outputs/<recording>/manipulation_events_<version>.json
"""

from __future__ import annotations

import argparse
import copy
import math
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from ..launch import (
    READINESS_PRIOR_SOURCES,
    apply_readiness_prior_overrides,
    load_config,
)
from ..models.llm import DEFAULT_MODEL_NAMES
from ..models.scripted import (
    DEFAULT_BEARING_OFFSETS_DEG,
    DEFAULT_CERTIFICATION_RETRIES,
    normalize_bearing_offsets_deg,
    normalize_object_names,
    normalize_retreat_distance_m,
)
from ..primitive_config import primitive_settings

#: SAM3 names tried in order. The prior's event is pinned to the first name
#: (``trial_config``), so the later ones need not share its words. ``bottle``
#: comes right after ``can``: it is what SAM3 still returns for a can too
#: small in the image for the ``can`` prompts, which return nothing at about
#: 2 m at the ZED's depth resolution, so the other can names rarely help
#: where ``can`` failed.
DEFAULT_OBJECT_NAMES = ("can", "bottle", "beverage can", "soda can", "drink can", "aluminum can")
DEFAULT_NAME_ROUNDS = 2
#: The largest budget prepare_for_manipulation's settings accept. At the Pi's
#: batch rate it still covers the whole nominal shortlist, so a scene the
#: default budget would cut off is still measured.
TRIAL_PI_IK_COMPUTE_BUDGET_S = 30.0
#: Farther Nav2 goal distances along the requested bearing, tried in order
#: when the inner goal cannot be planned or navigated. A dock at fallback
#: ``d`` accepts a final planar distance up to ``d + (docking_distance_m -
#: nav2_goal_distance_m) + final_distance_tolerance_m``; prepare_for_manipulation
#: measures the target's median 3D distance from the camera, which sits above
#: the object, and refuses anything beyond ``maximum_initial_target_distance_m``
#: with zero IK queries. So the last fallback's acceptance limit plus
#: ``TRIAL_TARGET_HEIGHT_MARGIN_M`` must stay within that maximum, which
#: :func:`trial_config` checks at launch.
TRIAL_NAV2_GOAL_DISTANCE_FALLBACKS_M = (0.75, 0.90)
#: How much the camera's 3D distance to the object may exceed the planar
#: distance docking verifies, from the camera's height above the object.
TRIAL_TARGET_HEIGHT_MARGIN_M = 0.15
TRIAL_EXPLICIT_BEARING_ALIGNMENT_TOLERANCE_DEG = 22.5
#: The largest initial target distance prepare_for_manipulation's settings
#: accept.
TRIAL_MAXIMUM_INITIAL_TARGET_DISTANCE_M = 1.2
#: Further certified base poses a refused motion may hand its turn to.
TRIAL_MOTION_ALTERNATIVE_LIMIT = 8
#: Whether prepare_for_manipulation resumes its IK search into the tiers the
#: best-tier early exit skipped once every certified base pose's motion has
#: been refused, within the same Pi IK budget and motion attempts.
TRIAL_CONTINUE_SEARCH_AFTER_REFUSED_MOTIONS = True
#: How far the base reverses before the next bearing once it stood at the
#: object: enough to leave the start-clearance check's refusal zone, and no
#: more than the drive controller lets a single reverse step go (reverse
#: motion has no camera behind it), which ``trial_config`` checks against the
#: navigation settings of the loaded config.
TRIAL_RETREAT_DISTANCE_M = 0.30
#: How far the docking heading turns toward the arm the dock suggests, so the
#: object ends in front of that arm instead of between the two.
TRIAL_ARM_SIDE_HEADING_OFFSET_DEG = 10.0
#: Longest start label accepted; it only has to name a marked start position.
MAX_START_LABEL_CHARS = 64
#: The Web UI start form submits the config's ``agent.max_turns`` back and
#: the server rejects values above this, so the trial's turn budget is capped
#: here (see :func:`trial_turn_budget`).
WEB_UI_MAX_TURNS = 100
#: ``llm``: the configured LLM writes and runs its own programs, as in a normal
#: run. ``scripted``: the fixed protocol of ``ReadinessTrialModel``.
TRIAL_POLICIES = ("llm", "scripted")
DEFAULT_TRIAL_POLICY = "llm"
#: Wall-clock limit of an LLM trial run unless ``--time-limit-s`` says otherwise.
DEFAULT_LLM_TIME_LIMIT_S = 120.0
#: The LLM trial's task. It names the primitive whose success the trial
#: counts, so a run that docks and finishes is not mistaken for readiness.
LLM_TRIAL_INSTRUCTION = (
    "Get ready to grasp the can on the table: bring the robot to a base pose "
    "from which one arm can grasp the can, as certified by a successful "
    "prepare_for_manipulation call, then call finish(). Do not grasp the can."
)
#: The model an LLM trial starts from; the Web UI form can pick another one
#: for a run.
LLM_TRIAL_MODEL_PROVIDER = "openai"
LLM_TRIAL_MODEL_NAME = "gpt-5.6-sol"
#: Primitives that sample or execute a grasp or move the arm or gripper. An
#: LLM trial hides them unless grasping is allowed, so a run ends at readiness
#: and its time and IK queries count only what reaching readiness took.
LLM_TRIAL_HIDDEN_PRIMITIVES = (
    "sample_grasp_pose",
    "goto_pose",
    "goto_grasp_pose",
    "open_gripper",
    "close_gripper",
    "lift_grasped_object",
)


def default_output_dir(config_path: str | Path, task_id: str | None) -> Path:
    """``outputs/readiness_trials/<task id>`` beside the configs directory."""

    configs = Path(config_path).expanduser().resolve().parent
    return configs.parent / "outputs" / "readiness_trials" / (task_id or "trial")


def check_retreat_within_reverse_limits(config: Mapping[str, Any], retreat_m: float) -> None:
    """Refuse a retreat the drive controller would reject at run time.

    ``drive_straight`` raises for a reverse step outside
    ``(distance_tolerance_m, max_reverse_distance_m]`` of the navigation
    settings, which the trial would only see as an aborted run at its first
    retreat. Checked at launch against the loaded config's ``robot.navigation``
    block, falling back to the controller's defaults when a key is absent.
    """

    if retreat_m <= 0.0:
        return
    from ..robot.navigation_controller import NavigationConfig

    navigation = (config.get("robot") or {}).get("navigation") or {}
    tolerance = float(
        navigation.get("distance_tolerance_m", getattr(NavigationConfig, "distance_tolerance_m", 0.025))
    )
    limit = float(navigation.get("max_reverse_distance_m", NavigationConfig.max_reverse_distance_m))
    if not tolerance < retreat_m <= limit:
        raise ValueError(
            f"retreat_distance_m {retreat_m:.3f} m must be in ({tolerance:.3f}, "
            f"{limit:.3f}] m, the reverse step the navigation settings allow "
            "(robot.navigation.max_reverse_distance_m)"
        )


def trial_turn_budget(
    *,
    name_count: int,
    name_rounds: int,
    bearing_count: int,
    certification_retries: int,
    retreat_distance_m: float = 0.0,
) -> int:
    """Turns the protocol can take at most, capped at what the Web UI accepts.

    At the first bearing every SAM3 name for every round at docking; at each
    later bearing every name once (the accepted name first). At every bearing
    every name once more as prepare fallbacks plus the certification retries.
    With a retreat distance, one retreat before every bearing after the
    first. One more turn for ``finish()``. The model gets the same budget
    and, when a trial would still exceed it, finishes unsolved on the last
    turn with a turn-budget reason.
    """

    prepare_turns = name_count + certification_retries
    first_bearing = name_count * name_rounds + prepare_turns
    later_bearing = name_count + prepare_turns
    if retreat_distance_m > 0.0:
        later_bearing += 1
    turns = first_bearing + max(0, bearing_count - 1) * later_bearing + 1
    return min(WEB_UI_MAX_TURNS, turns)


def trial_config(
    config: Mapping[str, Any],
    *,
    object_names: Iterable[str],
    name_rounds: int,
    pi_ik_compute_budget_s: float,
    output_dir: str | Path,
    bearing_offsets_deg: Iterable[float] = DEFAULT_BEARING_OFFSETS_DEG,
    certification_retries: int = DEFAULT_CERTIFICATION_RETRIES,
    retreat_distance_m: float = TRIAL_RETREAT_DISTANCE_M,
    arm_side_heading_offset_deg: float = TRIAL_ARM_SIDE_HEADING_OFFSET_DEG,
    start_label: str | None = None,
    policy: str = "scripted",
    time_limit_s: float | None = None,
    instruction: str | None = None,
    allow_grasp: bool = False,
    navigation_planner: bool = False,
) -> dict[str, Any]:
    """Return a copy of ``config`` set up for the trial.

    ``start_label`` names the start position the runs are started from; it is
    recorded as ``trial.start_label`` in the config every run's trace keeps.

    ``policy`` ``scripted`` replaces the model with the fixed protocol and its
    turn budget. ``llm`` runs :data:`LLM_TRIAL_MODEL_NAME` with the configured
    model's other settings and the configured turn budget (capped at what the
    Web UI accepts), and hides :data:`LLM_TRIAL_HIDDEN_PRIMITIVES`
    unless ``allow_grasp``; ``navigation_planner`` keeps the configured planner
    for it. The SAM3 names, name rounds, bearing offsets, certification
    retries and retreat are the scripted model's; an LLM trial uses only the
    first name, which the prior's event is pinned to. ``time_limit_s`` (either
    policy) becomes ``agent.time_limit_s``; ``instruction`` replaces the task
    instruction the Web UI form starts from.
    """

    if policy not in TRIAL_POLICIES:
        raise ValueError(f"policy must be one of {TRIAL_POLICIES}, got {policy!r}")
    scripted = policy == "scripted"
    if time_limit_s is not None:
        if isinstance(time_limit_s, bool) or not isinstance(time_limit_s, (int, float)):
            raise ValueError("time_limit_s must be a number of seconds")
        if not math.isfinite(float(time_limit_s)) or float(time_limit_s) <= 0.0:
            raise ValueError("time_limit_s must be positive and finite")
    if instruction is not None and (not isinstance(instruction, str) or not instruction.strip()):
        raise ValueError("instruction must be a non-empty string")

    result = copy.deepcopy(dict(config))
    names = normalize_object_names(object_names)
    if isinstance(name_rounds, bool) or not isinstance(name_rounds, int) or name_rounds < 1:
        raise ValueError("name_rounds must be a positive integer")
    budget = float(pi_ik_compute_budget_s)
    if not budget > 0.0:
        raise ValueError("the Pi IK compute budget must be positive")
    offsets = normalize_bearing_offsets_deg(bearing_offsets_deg)
    if (
        isinstance(certification_retries, bool)
        or not isinstance(certification_retries, int)
        or certification_retries < 0
    ):
        raise ValueError("certification_retries must be a non-negative integer")
    retreat = normalize_retreat_distance_m(retreat_distance_m)
    if scripted:
        check_retreat_within_reverse_limits(result, retreat)
    if (
        isinstance(arm_side_heading_offset_deg, bool)
        or not isinstance(arm_side_heading_offset_deg, (int, float))
        or not math.isfinite(float(arm_side_heading_offset_deg))
    ):
        raise ValueError("arm_side_heading_offset_deg must be a finite angle in degrees")
    heading_offset = float(arm_side_heading_offset_deg)
    label: str | None = None
    if start_label is not None:
        if not isinstance(start_label, str) or not start_label.strip():
            raise ValueError("start_label must be a non-empty string")
        label = start_label.strip()
        if len(label) > MAX_START_LABEL_CHARS:
            raise ValueError(
                f"start_label must be at most {MAX_START_LABEL_CHARS} characters"
            )

    if scripted:
        result["model"] = {
            "provider": "scripted",
            "name": DEFAULT_MODEL_NAMES["scripted"],
            "object_names": list(names),
            "name_rounds": name_rounds,
            "bearing_offsets_deg": list(offsets),
            "certification_retries": certification_retries,
            "retreat_distance_m": retreat,
        }
    else:
        # The configured model's other settings (temperature, token budget)
        # carry over to the trial's model.
        result["model"] = {
            **dict(result.get("model") or {}),
            "provider": LLM_TRIAL_MODEL_PROVIDER,
            "name": LLM_TRIAL_MODEL_NAME,
        }
    if scripted or not navigation_planner:
        planner = dict(result.get("navigation_planner") or {})
        planner["enabled"] = False
        planner.pop("memory_path", None)
        result["navigation_planner"] = planner

    primitive_config = result.get("primitive_config") or {}
    if primitive_settings(primitive_config, "dock_to_visible_object") is None:
        raise ValueError("dock_to_visible_object has no settings in the primitive config")
    if primitive_settings(primitive_config, "prepare_for_manipulation") is None:
        raise ValueError("prepare_for_manipulation has no settings in the primitive config")
    primitives = result["primitive_config"]["primitives"]
    if not scripted and not allow_grasp:
        for name in LLM_TRIAL_HIDDEN_PRIMITIVES:
            primitives[name] = {**dict(primitives.get(name) or {}), "exposed": False}

    docking = primitives["dock_to_visible_object"]
    docking_settings = {
        **dict(docking["settings"]),
        "nav2_goal_distance_fallbacks_m": list(TRIAL_NAV2_GOAL_DISTANCE_FALLBACKS_M),
        "plan_check_before_goal": True,
        "explicit_bearing_alignment_tolerance_deg": (
            TRIAL_EXPLICIT_BEARING_ALIGNMENT_TOLERANCE_DEG
        ),
        "arm_side_heading_offset_deg": heading_offset,
    }
    prior = docking_settings.get("readiness_prior")
    if isinstance(prior, Mapping):
        # The trial itself walks the bearings around the object, so a
        # with-prior dock must not add offsets or the arrival bearing of its
        # own. Its event is the first name's whichever fallback name SAM3
        # accepted, so the condition does not change with the name.
        docking_settings["readiness_prior"] = {
            **dict(prior),
            "retry_bearing_offsets_deg": [],
            "fallback_to_arrival_bearing": False,
            "event_object": names[0],
        }
    docking["settings"] = docking_settings
    # The primitives validate their settings only when a run starts; check the
    # trial's values at launch instead of at the operator's Start. Explicit
    # bearings, goal-distance fallbacks and the arm-side heading offset exist
    # on the Nav2 backend only.
    backend = str(docking_settings.get("backend", "legacy")).strip().lower()
    if backend != "nav2":
        raise ValueError(
            "the readiness trial needs dock_to_visible_object backend 'nav2', "
            f"got {backend!r}"
        )
    from ..robot.nav2_visible_object_navigation import Nav2VisibleObjectDockingConfig

    docking_config = Nav2VisibleObjectDockingConfig.from_mapping(docking_settings)
    if isinstance(prior, Mapping):
        from ..robot.readiness_prior import ReadinessPriorConfig

        ReadinessPriorConfig.from_mapping(docking_settings["readiness_prior"])

    prepare = primitives["prepare_for_manipulation"]
    prepare["settings"] = {
        **dict(prepare["settings"]),
        "pi_ik_compute_budget_s": budget,
        "maximum_initial_target_distance_m": TRIAL_MAXIMUM_INITIAL_TARGET_DISTANCE_M,
        "motion_alternative_limit": TRIAL_MOTION_ALTERNATIVE_LIMIT,
        "continue_search_after_refused_motions": (
            TRIAL_CONTINUE_SEARCH_AFTER_REFUSED_MOTIONS
        ),
    }
    from ..robot.manipulation_readiness import ManipulationReadinessConfig

    prepare_config = ManipulationReadinessConfig.from_mapping(prepare["settings"])
    check_fallback_distances_within_prepare_limit(docking_config, prepare_config)

    agent = dict(result.get("agent") or {})
    if scripted:
        turns = trial_turn_budget(
            name_count=len(names),
            name_rounds=name_rounds,
            bearing_count=len(offsets),
            certification_retries=certification_retries,
            retreat_distance_m=retreat,
        )
        agent["max_turns"] = turns
        # The model counts its own programs against the same budget, so a trial
        # that would overrun it still ends with finish() on the last turn.
        result["model"]["max_turns"] = turns
    elif "max_turns" in agent:
        agent["max_turns"] = min(int(agent["max_turns"]), WEB_UI_MAX_TURNS)
    if time_limit_s is not None:
        agent["time_limit_s"] = float(time_limit_s)
    else:
        # The trial's own limit replaces one the task config carries, so a
        # trial without a limit has none.
        agent.pop("time_limit_s", None)
    result["agent"] = agent
    if instruction is not None:
        result["task"] = {**dict(result.get("task") or {}), "instruction": instruction.strip()}
    result["trace"] = {**dict(result.get("trace") or {}), "output_dir": str(output_dir)}
    if label is not None:
        # The Web UI starts every run from a copy of this config and the trace
        # records it, so each run carries the start position it was run from.
        result["trial"] = {**dict(result.get("trial") or {}), "start_label": label}
    return result


def fallback_target_distance_limit_m(docking_config: Any) -> float | None:
    """The farthest target a dock at the last goal-distance fallback accepts.

    ``None`` without fallbacks. The camera's 3D distance the prepare gate
    measures may exceed this planar distance by the height margin.
    """

    schedule = tuple(docking_config.nav2_goal_distance_schedule_m)
    if len(schedule) < 2:
        return None
    return (
        schedule[-1]
        + (docking_config.docking_distance_m - schedule[0])
        + docking_config.final_distance_tolerance_m
    )


def check_fallback_distances_within_prepare_limit(
    docking_config: Any, prepare_config: Any
) -> None:
    """Refuse fallbacks that could leave prepare's initial distance gate failing.

    A bearing docked at the last fallback would otherwise end as
    ``prepare_too_far`` with zero IK queries, which measures nothing.
    """

    limit = fallback_target_distance_limit_m(docking_config)
    if limit is None:
        return
    reach = limit + TRIAL_TARGET_HEIGHT_MARGIN_M
    maximum = float(prepare_config.maximum_initial_target_distance_m)
    if reach > maximum + 1e-9:
        schedule = docking_config.nav2_goal_distance_schedule_m
        raise ValueError(
            "the last docking fallback can leave the target beyond "
            "prepare_for_manipulation's initial distance limit: "
            f"{schedule[-1]:.2f} m + ({docking_config.docking_distance_m:.2f} m "
            f"docking_distance_m - {schedule[0]:.2f} m nav2_goal_distance_m) + "
            f"{docking_config.final_distance_tolerance_m:.2f} m "
            f"final_distance_tolerance_m + {TRIAL_TARGET_HEIGHT_MARGIN_M:.2f} m "
            f"height margin = {reach:.2f} m > {maximum:.2f} m "
            "maximum_initial_target_distance_m"
        )


def parse_bearing_offsets_deg(text: str) -> tuple[float, ...]:
    """``"0,45,-45"`` -> ``(0.0, 45.0, -45.0)``, validated like the model does."""

    items = [item.strip() for item in str(text).split(",")]
    try:
        values = [float(item) for item in items if item]
    except ValueError as exc:
        raise ValueError(
            f"--bearing-offsets-deg must be comma-separated angles in degrees, got {text!r}"
        ) from exc
    return normalize_bearing_offsets_deg(values)


def _event_index(events: Sequence[Mapping[str, Any]], name: str, tokens: Any) -> int | None:
    wanted = tokens(name)
    for index, event in enumerate(events):
        if wanted & tokens(event.get("object", "")):
            return index
    return None


def unmatched_object_names(
    config: Mapping[str, Any], object_names: Iterable[str]
) -> list[str]:
    """Names that would not resolve to the readiness event the first name does.

    ``ReadinessPrior.match_event`` takes the first event sharing any word with
    the name, so a fallback name that matches nothing, or matches a different
    event, would silently change the condition of a with-prior trial. With
    ``readiness_prior.event_object`` set (as ``trial_config`` does) every
    name resolves to that object's event, so either all names resolve or,
    when no event carries its words, none does. Returns an empty list when no
    events file is configured.
    """

    settings = primitive_settings(config.get("primitive_config") or {}, "dock_to_visible_object") or {}
    block = settings.get("readiness_prior") or {}
    events_path = block.get("events_path") if isinstance(block, Mapping) else None
    if not events_path:
        return []
    from ..robot.readiness_prior import ReadinessPrior, _tokens

    events = ReadinessPrior._load_events(Path(str(events_path)).expanduser())["events"]
    names = list(normalize_object_names(object_names))
    pinned = block.get("event_object") if isinstance(block, Mapping) else None
    if isinstance(pinned, str) and pinned.strip():
        return [] if _event_index(events, pinned, _tokens) is not None else names
    first = _event_index(events, names[0], _tokens)
    return [
        name
        for name in names
        if first is None or _event_index(events, name, _tokens) != first
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="yor-readiness-trial",
        description=(
            "Readiness-prior trial behind the Web UI: the configured LLM policy "
            "or the scripted bearing round-robin dock-then-prepare protocol."
        ),
    )
    parser.add_argument("--config", required=True, help="path to a task YAML file")
    parser.add_argument("--task-id", help="select one task from config.task_suite.tasks")
    parser.add_argument(
        "--policy",
        choices=TRIAL_POLICIES,
        default=DEFAULT_TRIAL_POLICY,
        help=(
            f"llm: {LLM_TRIAL_MODEL_NAME} writes and runs its own programs; scripted: "
            "the fixed dock-then-prepare protocol (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--time-limit-s",
        type=float,
        metavar="S",
        help=(
            "stop each run after this many seconds, as the Web UI's Stop does; 0 "
            f"disables (default: {DEFAULT_LLM_TIME_LIMIT_S:g} for llm, none for scripted)"
        ),
    )
    parser.add_argument(
        "--instruction",
        help=(
            "task instruction the Web UI form starts from (default for llm: a "
            "grasp-readiness task on the can; scripted: the config's)"
        ),
    )
    parser.add_argument(
        "--allow-grasp",
        action="store_true",
        help=(
            "llm: keep the grasp, arm and gripper primitives exposed as configured "
            "(hidden by default so a run ends at readiness)"
        ),
    )
    parser.add_argument(
        "--navigation-planner",
        action="store_true",
        help="llm: keep the one-shot navigation planner as configured (disabled by default)",
    )
    parser.add_argument(
        "--readiness-events",
        metavar="PATH",
        help="manipulation events JSON; required for the Web UI 'Readiness prior' checkbox",
    )
    parser.add_argument(
        "--readiness-prior-source",
        choices=READINESS_PRIOR_SOURCES,
        help="ready_frame (the method) or nav_frame (the walking-frame ablation)",
    )
    parser.add_argument(
        "--object-name",
        dest="object_names",
        action="append",
        help=(
            "SAM3 name to try, in order; repeat for fallbacks "
            f"(default: {', '.join(DEFAULT_OBJECT_NAMES)})"
        ),
    )
    parser.add_argument(
        "--name-rounds",
        type=int,
        default=DEFAULT_NAME_ROUNDS,
        help="how many times the docking fallback list is tried at each bearing",
    )
    parser.add_argument(
        "--bearing-offsets-deg",
        default=",".join(f"{value:g}" for value in DEFAULT_BEARING_OFFSETS_DEG),
        metavar="DEG,DEG,...",
        help=(
            "approach bearings to try, as offsets from the bearing the first dock "
            "chose, in order; the first must be 0 (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--no-bearing-search",
        action="store_true",
        help=(
            "try bearing 0 only (the bearing docking chooses on its own); the "
            "docking fallbacks, plan check and motion alternatives stay on"
        ),
    )
    parser.add_argument(
        "--certification-retries",
        type=int,
        default=DEFAULT_CERTIFICATION_RETRIES,
        help=(
            "prepare_for_manipulation retries at the same bearing after a failed "
            "actual-pose certification"
        ),
    )
    parser.add_argument(
        "--pi-ik-budget-s",
        type=float,
        default=TRIAL_PI_IK_COMPUTE_BUDGET_S,
        help="prepare_for_manipulation Pi IK compute budget for these trials",
    )
    parser.add_argument(
        "--retreat-m",
        type=float,
        default=TRIAL_RETREAT_DISTANCE_M,
        metavar="M",
        help=(
            "how far the base reverses before the next bearing once it stood at "
            "the object (a dock started there is refused); 0 disables the retreat "
            "(default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--arm-heading-offset-deg",
        type=float,
        default=TRIAL_ARM_SIDE_HEADING_OFFSET_DEG,
        metavar="DEG",
        help=(
            "how far docking turns its final heading toward the arm it suggests, "
            "so the object ends in front of that arm rather than between the two; "
            "0 keeps the object dead ahead (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--start-label",
        metavar="LABEL",
        help=(
            "name of the start position these runs are started from (for example "
            "S45); recorded as trial.start_label in every run's trace and shown by "
            "tools/readiness_trial_report.py. Restart the launcher when the start "
            "position changes"
        ),
    )
    parser.add_argument(
        "--output-dir",
        help="trace root; default outputs/readiness_trials/<task id> next to the configs",
    )
    parser.add_argument("--web-ui-host", default="0.0.0.0", help="Web UI bind host")
    parser.add_argument("--web-ui-port", type=int, default=8200, help="Web UI bind port")
    args = parser.parse_args(argv)
    scripted = args.policy == "scripted"
    if args.time_limit_s is not None and not (
        math.isfinite(args.time_limit_s) and args.time_limit_s >= 0.0
    ):
        parser.error("--time-limit-s must be a non-negative number of seconds")
    if args.time_limit_s is None:
        time_limit_s = None if scripted else DEFAULT_LLM_TIME_LIMIT_S
    else:
        time_limit_s = args.time_limit_s or None
    instruction = args.instruction
    if instruction is None and not scripted:
        instruction = LLM_TRIAL_INSTRUCTION
    if scripted and (args.allow_grasp or args.navigation_planner):
        parser.error("--allow-grasp and --navigation-planner apply to --policy llm only")

    config = load_config(args.config, task_id=args.task_id)
    try:
        config = apply_readiness_prior_overrides(
            config,
            enabled=None,
            events_path=args.readiness_events,
            source=args.readiness_prior_source,
        )
        names = normalize_object_names(args.object_names or DEFAULT_OBJECT_NAMES)
        offsets = (
            (0.0,)
            if args.no_bearing_search
            else parse_bearing_offsets_deg(args.bearing_offsets_deg)
        )
        output_dir = (
            Path(args.output_dir).expanduser()
            if args.output_dir
            else default_output_dir(args.config, args.task_id)
        )
        config = trial_config(
            config,
            object_names=names,
            name_rounds=args.name_rounds,
            pi_ik_compute_budget_s=args.pi_ik_budget_s,
            output_dir=output_dir,
            bearing_offsets_deg=offsets,
            certification_retries=args.certification_retries,
            retreat_distance_m=args.retreat_m,
            arm_side_heading_offset_deg=args.arm_heading_offset_deg,
            start_label=args.start_label,
            policy=args.policy,
            time_limit_s=time_limit_s,
            instruction=instruction,
            allow_grasp=args.allow_grasp,
            navigation_planner=args.navigation_planner,
        )
        unmatched = unmatched_object_names(config, names)
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))
    if unmatched:
        parser.error(
            f"no readiness event shares a word with the first name {names[0]!r}, "
            f"which every name docks with: {unmatched}; pass --object-name with "
            "the event's object word first"
        )

    settings = primitive_settings(config["primitive_config"], "dock_to_visible_object") or {}
    events = (settings.get("readiness_prior") or {}).get("events_path")
    agent = config.get("agent") or {}
    if scripted:
        print(f"readiness trial: scripted policy, SAM3 names {list(names)} x {args.name_rounds} round(s)")
        print(
            "readiness trial: bearing offsets "
            f"[{', '.join(f'{value:+g}' for value in offsets)}] deg from the first dock's bearing"
        )
        print(
            f"readiness trial: certification retries {args.certification_retries}, "
            f"turn budget {agent['max_turns']}"
        )
        print(
            "readiness trial: retreat "
            + (
                f"{args.retreat_m:.2f} m before the next bearing after standing at the object"
                if args.retreat_m > 0.0
                else "disabled"
            )
        )
    else:
        model = config.get("model") or {}
        print(
            f"readiness trial: LLM policy {model.get('provider')} / {model.get('name')}, "
            f"turn budget {agent.get('max_turns', 'default')}"
        )
        print(f"readiness trial: instruction {config['task']['instruction']!r}")
        print(
            "readiness trial: grasp primitives "
            + ("exposed as configured" if args.allow_grasp else f"hidden {list(LLM_TRIAL_HIDDEN_PRIMITIVES)}")
        )
        print(
            "readiness trial: navigation planner "
            + ("as configured" if args.navigation_planner else "disabled")
        )
        print(f"readiness trial: prior event pinned to {names[0]!r}")
    print(
        "readiness trial: time limit "
        + (f"{agent['time_limit_s']:g} s per run" if agent.get("time_limit_s") else "none")
    )
    print(f"readiness trial: Pi IK budget {args.pi_ik_budget_s:.1f} s")
    print(
        f"readiness trial: docking heading offset {args.arm_heading_offset_deg:+g} deg "
        "toward the suggested arm"
    )
    print(
        "readiness trial: start label "
        + (
            repr(config["trial"]["start_label"])
            if isinstance(config.get("trial"), Mapping) and config["trial"].get("start_label")
            else "not set (pass --start-label to record the start position)"
        )
    )
    print(f"readiness trial: traces under {output_dir}")
    print(
        "readiness trial: readiness events "
        + (str(events) if events else "not configured (only the no-prior condition can run)")
    )

    from ..web.server import run_web_ui

    run_web_ui(
        config_path=Path(args.config).expanduser(),
        config=config,
        host=args.web_ui_host,
        port=args.web_ui_port,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
