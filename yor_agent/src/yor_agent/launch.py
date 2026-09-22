"""Composition root: build the robot, registry, model, executor, agent, and run.

This file loads one YAML file, selects an optional task-suite entry, and wires
the pieces together. It does not implement a reasoning loop of its own. The
registry remains the capability set, and its generated documentation is what
the policy model sees.
"""

from __future__ import annotations

import argparse
import copy
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
import threading
from typing import Any

import yaml

from .agents.default import DefaultAgent
from .environment import YorEnvironment
from .executor import PolicyExecutor
from .experiment_episodes import default_start_label, experiment_block, planner_condition
from .models.llm import DEFAULT_MODEL_NAMES, LLM
from .models.manual import ManualModel
from .models.navigation_planner import StartupNavigationPlanner
from .models.scripted import ReadinessTrialModel
from .primitive_config import (
    apply_primitive_overrides,
    load_primitive_config,
    normalize_primitive_config,
    primitive_defaults,
    primitive_exposed,
    primitive_settings,
)
from .primitives.manipulation import register_manipulation_primitives
from .primitives.manipulation_readiness import (
    register_manipulation_readiness_primitive,
)
from .primitives.coarse_navigation import (
    COARSE_NAVIGATION_PRIMITIVES,
    register_coarse_navigation_primitives,
)
from .primitives.navigation import register_navigation_primitives
from .primitives.registry import PrimitiveRegistry
from .primitives.visible_object_navigation import (
    register_visible_object_navigation_primitives,
)
from .trace import Trace

#: Which passive-video frame the readiness prior matches against the ZED
#: image: ``ready_frame`` is the method, ``nav_frame`` the walking-frame
#: ablation (docs/readiness_prior_plan.md).
READINESS_PRIOR_SOURCES = ("ready_frame", "nav_frame")


def load_config(
    path: str | Path, *, task_id: str | None = None
) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    config = _load_config_mapping(config_path, seen=set())
    suite = config.pop("task_suite", None)
    if suite is not None:
        if not isinstance(suite, Mapping):
            raise ValueError(f"{path} task_suite must be a mapping")
        tasks = suite.get("tasks") or {}
        if not isinstance(tasks, Mapping) or not tasks:
            raise ValueError(f"{path} task_suite.tasks must be a non-empty mapping")
        available = sorted(str(name) for name in tasks)
        if not task_id:
            raise ValueError(
                f"{path} is a task suite; pass --task-id. Available tasks: "
                + ", ".join(available)
            )
        if task_id not in tasks:
            raise ValueError(
                f"unknown task id {task_id!r}; available tasks: "
                + ", ".join(available)
            )
        selected = tasks[task_id]
        if isinstance(selected, str):
            selected = {"instruction": selected}
        if not isinstance(selected, Mapping):
            raise ValueError(f"task {task_id!r} must be a string or mapping")
        config["task"] = {"id": task_id, **dict(selected)}
        output_root = suite.get("output_root", "../outputs/task_suite")
        output_root = Path(str(output_root)).expanduser()
        if not output_root.is_absolute():
            output_root = config_path.parent / output_root
        config["trace"] = {
            **dict(config.get("trace") or {}),
            "output_dir": str((output_root / task_id).resolve()),
        }
    elif task_id:
        raise ValueError(f"{path} does not define task_suite; omit --task-id")

    task = config.get("task") or {}
    if not str(task.get("instruction", "")).strip():
        raise ValueError(f"{path} must define task.instruction")
    primitive_source = config.get("primitive_config")
    if isinstance(primitive_source, (str, Path)):
        primitive_path = Path(primitive_source).expanduser()
        if not primitive_path.is_absolute():
            primitive_path = config_path.parent / primitive_path
        config["primitive_config"] = load_primitive_config(primitive_path)
    else:
        config["primitive_config"] = normalize_primitive_config(primitive_source)
    # A task configuration may change a few primitives (for example which are
    # exposed) on top of the complete primitive configuration it loads.
    primitive_overrides = config.pop("primitive_overrides", None)
    if primitive_overrides is not None:
        config["primitive_config"] = apply_primitive_overrides(
            config["primitive_config"], primitive_overrides
        )
    planner_config = dict(config.get("navigation_planner") or {})
    memory_source = planner_config.get("memory_path")
    if memory_source:
        memory_path = Path(str(memory_source)).expanduser()
        if not memory_path.is_absolute():
            memory_path = config_path.parent / memory_path
        planner_config["memory_path"] = str(memory_path.resolve())
        config["navigation_planner"] = planner_config
    # A task-YAML-relative readiness events file resolves against the task
    # file, exactly like navigation_planner.memory_path above.
    docking_settings = primitive_settings(
        config["primitive_config"], "dock_to_visible_object"
    )
    prior_block = (docking_settings or {}).get("readiness_prior")
    if isinstance(prior_block, Mapping) and prior_block.get("events_path"):
        events_path = Path(str(prior_block["events_path"])).expanduser()
        if not events_path.is_absolute():
            events_path = config_path.parent / events_path
        config["primitive_config"]["primitives"]["dock_to_visible_object"][
            "settings"
        ]["readiness_prior"] = {
            **prior_block,
            "events_path": str(events_path.resolve()),
        }
    return config


def _load_config_mapping(path: Path, *, seen: set[Path]) -> dict[str, Any]:
    if path in seen:
        raise ValueError(f"cyclic config extends chain at {path}")
    seen.add(path)
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must contain a YAML mapping")
    child = dict(value)
    parent_source = child.pop("extends", None)
    if not parent_source:
        return child
    parent_path = Path(str(parent_source)).expanduser()
    if not parent_path.is_absolute():
        parent_path = path.parent / parent_path
    parent = _load_config_mapping(parent_path.resolve(), seen=seen)
    return _deep_merge(parent, child)


def _deep_merge(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in overlay.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def apply_navigation_planner_overrides(
    config: Mapping[str, Any],
    *,
    enabled: bool | None,
    memory_path: str | None,
    model: str | None,
) -> dict[str, Any]:
    """Apply the CLI planner switch without changing the policy-only path."""

    result = dict(config)
    planner_config = dict(result.get("navigation_planner") or {})
    if enabled is not None:
        planner_config["enabled"] = enabled
    if memory_path:
        if enabled is False:
            raise ValueError(
                "--navigation-memory cannot be used with --no-navigation-planner"
            )
        planner_config["memory_path"] = str(
            Path(memory_path).expanduser().resolve()
        )
        if enabled is None:
            planner_config["enabled"] = True
    if model:
        if enabled is False:
            raise ValueError(
                "--navigation-planner-model cannot be used with "
                "--no-navigation-planner"
            )
        planner_config["model"] = model

    if bool(planner_config.get("enabled", False)):
        if not planner_config.get("memory_path"):
            raise ValueError(
                "--navigation-memory is required when navigation planner is enabled"
            )
    else:
        # Make the disabled trace unambiguous and ensure no stale configured
        # memory can be mistaken for an input used by the policy-only run.
        planner_config.pop("memory_path", None)
    result["navigation_planner"] = planner_config
    return result


def apply_readiness_prior_overrides(
    config: Mapping[str, Any],
    *,
    enabled: bool | None,
    events_path: str | None,
    source: str | None,
) -> dict[str, Any]:
    """Apply the CLI readiness-prior switch to the docking primitive's settings.

    The prior lives in ``primitive_config.primitives.dock_to_visible_object
    .settings.readiness_prior`` and is an ablation independent of the
    navigation planner, so this mirrors
    :func:`apply_navigation_planner_overrides` without touching its block: an
    events file implies ``enabled``, an explicitly disabled prior drops the
    events file so the trace is unambiguous, and enabling without an events
    file is an error. With no flag given the loaded config is returned as is
    (deep-copied), so a YAML that configures ``events_path`` with ``enabled:
    false`` keeps the path for the Web UI checkbox.
    """

    result = copy.deepcopy(dict(config))
    if enabled is None and events_path is None and source is None:
        return result

    primitive_config = result.get("primitive_config")
    primitives = (
        primitive_config.get("primitives")
        if isinstance(primitive_config, Mapping)
        else None
    )
    docking = (
        primitives.get("dock_to_visible_object")
        if isinstance(primitives, Mapping)
        else None
    )
    settings = docking.get("settings") if isinstance(docking, Mapping) else None
    if not isinstance(settings, Mapping):
        raise ValueError("dock_to_visible_object has no settings to override")

    prior_config = dict(settings.get("readiness_prior") or {})
    if enabled is not None:
        prior_config["enabled"] = enabled
    if events_path:
        if enabled is False:
            raise ValueError(
                "--readiness-events cannot be used with --no-readiness-prior"
            )
        prior_config["events_path"] = str(
            Path(events_path).expanduser().resolve()
        )
        if enabled is None:
            prior_config["enabled"] = True
    if source:
        if enabled is False:
            raise ValueError(
                "--readiness-prior-source cannot be used with --no-readiness-prior"
            )
        if source not in READINESS_PRIOR_SOURCES:
            raise ValueError(
                "--readiness-prior-source must be one of "
                + ", ".join(READINESS_PRIOR_SOURCES)
                + f", got {source!r}"
            )
        prior_config["source"] = source

    if bool(prior_config.get("enabled", False)):
        if not prior_config.get("events_path"):
            raise ValueError(
                "--readiness-events is required when the readiness prior is enabled"
            )
    else:
        # Make the disabled trace unambiguous: no stale events file can be
        # mistaken for an input the docking controller used.
        prior_config.pop("events_path", None)

    settings = {**settings, "readiness_prior": prior_config}
    docking = {**docking, "settings": settings}
    primitives = {**primitives, "dock_to_visible_object": docking}
    result["primitive_config"] = {**primitive_config, "primitives": primitives}
    return result


def build_environment(config: Mapping[str, Any]) -> YorEnvironment:
    robot_config = dict(config.get("robot") or {})
    primitive_config = normalize_primitive_config(config.get("primitive_config"))
    docking_config = primitive_settings(
        primitive_config, "dock_to_visible_object"
    )
    if docking_config is not None:
        manipulation = dict(robot_config.get("manipulation") or {})
        manipulation["visible_object_docking"] = docking_config
        robot_config["manipulation"] = manipulation
    readiness_config = primitive_settings(
        primitive_config, "prepare_for_manipulation"
    )
    if readiness_config is not None:
        manipulation = dict(robot_config.get("manipulation") or {})
        manipulation["manipulation_readiness"] = readiness_config
        robot_config["manipulation"] = manipulation
    return YorEnvironment(**robot_config)


@dataclass
class Runtime:
    """The objects shared by CLI and Web UI around the one agent loop."""

    environment: YorEnvironment
    registry: PrimitiveRegistry
    trace: Trace
    executor: PolicyExecutor
    model: LLM
    navigation_planner: StartupNavigationPlanner | None
    agent: DefaultAgent


def manual_provider_error(config: Mapping[str, Any], *, web_ui: bool) -> str | None:
    """Explain why a manual-provider run cannot start, or return ``None``.

    Operator programs only arrive through the Web UI, so a headless CLI run
    with provider ``manual`` would initialize the robot and then wait forever.
    """

    provider = str((config.get("model") or {}).get("provider", "vertex"))
    if provider.strip().lower() == "manual" and not web_ui:
        return (
            "the manual provider requires --web-ui: operator programs are "
            "submitted from the Web UI"
        )
    return None


def build_model(
    model_config: Mapping[str, Any] | None,
    *,
    stop_event: threading.Event | None = None,
) -> LLM:
    """Select the policy model for ``config["model"]``.

    Provider ``manual`` returns an operator-driven :class:`ManualModel` that
    shares the run's stop event, so a Web UI stop also releases a turn that
    is waiting for a program.  Provider ``scripted`` returns the fixed
    readiness trial (:class:`ReadinessTrialModel`).  Every other provider is
    an API-backed :class:`LLM`.
    """

    config = dict(model_config or {})
    provider = str(config.get("provider", "vertex")).strip().lower()
    if provider == "manual":
        return ManualModel(config, stop_event=stop_event)
    if provider == "scripted":
        return ReadinessTrialModel(config, stop_event=stop_event)
    return LLM(config)


def _build_readiness_prior(docking_settings: Mapping[str, Any] | None) -> Any | None:
    """Construct the passive-video readiness prior for ``dock_to_visible_object``.

    Returns ``None`` unless ``docking_settings["readiness_prior"]["enabled"]``
    is true. The prior module is imported only on that path, so a launch
    without the prior never needs it (nor cv2, which the module itself loads
    lazily). Loading the events JSON happens here, at construction, so a
    missing or malformed file fails the launch rather than the first dock.
    """

    values = (docking_settings or {}).get("readiness_prior")
    if not isinstance(values, Mapping) or not bool(values.get("enabled", False)):
        return None
    from .robot.readiness_prior import ReadinessPrior, ReadinessPriorConfig

    return ReadinessPrior(ReadinessPriorConfig.from_mapping(values))


def build_runtime(
    config: Mapping[str, Any],
    *,
    on_event: Any | None = None,
    stop_event: threading.Event | None = None,
) -> Runtime:
    """Construct the shared runtime without starting a second execution loop."""

    # One stop event is the single source of truth for the agent loop and for
    # a ManualModel waiting on the operator; a caller that passes none still
    # gets a shared event so ``agent.request_stop()`` releases a waiting turn.
    stop_event = stop_event if stop_event is not None else threading.Event()
    trace = Trace(config.get("trace") or {})
    environment: YorEnvironment | None = None
    try:
        environment = build_environment(config)
        primitive_config = normalize_primitive_config(config.get("primitive_config"))
        registry = PrimitiveRegistry()

        def emit_primitive_progress(event: dict[str, Any]) -> None:
            """Stream long-running primitive state without changing its API."""

            if on_event is None:
                return
            try:
                on_event(
                    {
                        "type": "primitive_progress",
                        "turn": trace.turn,
                        **event,
                    }
                )
            except Exception:
                # A display subscriber must never affect a primitive result.
                pass

        register_navigation_primitives(
            registry,
            environment,
            primitive_defaults={
                name: primitive_defaults(primitive_config, name)
                for name in (
                    "observe",
                    "stop",
                    "turn_relative",
                    "drive_straight",
                    "drive_lateral",
                )
            },
        )
        # The coarse navigation baseline's vocabulary is registered only where
        # a primitive configuration exposes it, so no other run lists it.
        if any(
            name in primitive_config.get("primitives", {})
            and primitive_exposed(primitive_config, name)
            for name in COARSE_NAVIGATION_PRIMITIVES
        ):
            register_coarse_navigation_primitives(
                registry,
                environment,
                primitive_defaults={
                    name: primitive_defaults(primitive_config, name)
                    for name in COARSE_NAVIGATION_PRIMITIVES
                },
            )
        # No task-level YAML switch is needed: the model sees whatever is
        # registered, gated only on whether the required hardware/config is
        # present (arm RPC reachable, camera calibration configured).
        if environment.has_manipulation:
            manipulation_controller = register_manipulation_primitives(
                registry,
                environment,
                primitive_defaults={
                    name: primitive_defaults(primitive_config, name)
                    for name in (
                        "get_object_pose",
                        "sample_grasp_pose",
                        "goto_pose",
                        "goto_grasp_pose",
                        "open_gripper",
                        "close_gripper",
                        "lift_grasped_object",
                    )
                },
                primitive_settings={
                    name: primitive_settings(primitive_config, name) or {}
                    for name in (
                        "get_object_pose",
                        "sample_grasp_pose",
                        "open_gripper",
                        "close_gripper",
                    )
                },
                attached_lift_enabled=primitive_exposed(
                    primitive_config, "lift_grasped_object"
                ),
            )
            if primitive_settings(
                primitive_config, "prepare_for_manipulation"
            ) is not None:
                register_manipulation_readiness_primitive(
                    registry,
                    environment,
                    settings=primitive_settings(
                        primitive_config, "prepare_for_manipulation"
                    ),
                    manipulation_controller=manipulation_controller,
                )
        if environment.has_visible_object_docking:
            docking_settings = primitive_settings(
                primitive_config, "dock_to_visible_object"
            )
            register_visible_object_navigation_primitives(
                registry,
                environment,
                docking_config=docking_settings,
                progress_callback=emit_primitive_progress,
                readiness_prior=_build_readiness_prior(docking_settings),
            )

        # Exposure is a policy capability boundary, independent of whether the
        # implementation and settings remain available to the robot stack.
        for name in tuple(registry.names()):
            if not primitive_exposed(primitive_config, name):
                registry.unregister(name)

        def record_primitive_call(event: dict[str, Any]) -> None:
            trace.record_primitive_call(event)
            if on_event is not None:
                try:
                    on_event(
                        {
                            "type": "primitive_call",
                            "turn": trace.turn,
                            "timestamp": event.get("started_at"),
                            **event,
                        }
                    )
                except Exception:
                    # A display subscriber must never affect a primitive result.
                    pass

        executor = PolicyExecutor(registry, on_primitive_call=record_primitive_call)
        model = build_model(config.get("model"), stop_event=stop_event)
        planner_config = dict(config.get("navigation_planner") or {})
        navigation_planner = (
            StartupNavigationPlanner(planner_config)
            if bool(planner_config.get("enabled", False))
            else None
        )
        agent = DefaultAgent(
            model,
            environment,
            executor,
            trace,
            navigation_planner=navigation_planner,
            on_event=on_event,
            stop_event=stop_event,
            **dict(config.get("agent") or {}),
        )
        if isinstance(model, (ManualModel, ReadinessTrialModel)):
            # Report the operator's stop reason (recorded by DefaultAgent when
            # request_stop() sets the shared event) from a manual or scripted turn.
            model.stop_reason = lambda: getattr(
                agent, "_stop_reason", "operator requested stop"
            )
        return Runtime(
            environment,
            registry,
            trace,
            executor,
            model,
            navigation_planner,
            agent,
        )
    except BaseException as exc:
        trace.record_error(exc)
        if environment is not None:
            environment.safe_shutdown()
        raise


def launch(
    config: Mapping[str, Any],
    *,
    on_event: Any | None = None,
    stop_event: threading.Event | None = None,
) -> dict[str, Any]:
    """Construct every component from ``config`` and run one task."""

    runtime = build_runtime(config, on_event=on_event, stop_event=stop_event)
    return runtime.agent.run(config["task"]["instruction"], config)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="yor-agent", description=__doc__)
    parser.add_argument("--config", required=True, help="path to a task YAML file")
    parser.add_argument(
        "--task-id", help="select one task from config.task_suite.tasks"
    )
    parser.add_argument("--instruction", help="override task.instruction")
    parser.add_argument("--output-dir", help="override trace.output_dir")
    parser.add_argument(
        "--navigation-memory",
        help="timestamped offline navigation-memory JSON for this run",
    )
    parser.add_argument(
        "--navigation-planner",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "enable or disable one-shot startup planning; disable with "
            "--no-navigation-planner"
        ),
    )
    parser.add_argument(
        "--navigation-planner-model",
        help="override the Gemini or GPT model used by one-shot startup planning",
    )
    parser.add_argument(
        "--readiness-prior",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "enable or disable the passive-video readiness prior that picks "
            "the docking bearing in dock_to_visible_object; disable with "
            "--no-readiness-prior"
        ),
    )
    parser.add_argument(
        "--readiness-events",
        metavar="PATH",
        help=(
            "manipulation events JSON from `nav-planner-readiness VIDEO` for "
            "this run; implies --readiness-prior"
        ),
    )
    parser.add_argument(
        "--readiness-prior-source",
        choices=READINESS_PRIOR_SOURCES,
        help=(
            "which event frame the prior matches against the ZED image: "
            "ready_frame (the method) or nav_frame (the t_ready - 2 s ablation)"
        ),
    )
    parser.add_argument(
        "--max-turns", type=int, help="override agent.max_turns for this run"
    )
    parser.add_argument(
        "--provider",
        choices=("vertex", "qwen", "deepseek", "openai", "manual"),
        help=(
            "override model.provider for this run; 'manual' replaces the LLM "
            "with operator-typed programs from the Web UI"
        ),
    )
    parser.add_argument("--model", help="override model.name for this run")
    parser.add_argument(
        "--reasoning-effort",
        choices=("none", "low", "medium", "high", "xhigh", "max"),
        help="override model.reasoning_effort for OpenAI reasoning models",
    )
    parser.add_argument(
        "--qwen-base-url",
        help="override Qwen's regional OpenAI-compatible endpoint",
    )
    parser.add_argument(
        "--deepseek-base-url",
        help="override DeepSeek's official OpenAI-compatible endpoint",
    )
    parser.add_argument(
        "--web-ui",
        action="store_true",
        help="launch the interactive browser UI instead of starting immediately",
    )
    parser.add_argument(
        "--web-ui-host", default="0.0.0.0", help="Web UI bind host"
    )
    parser.add_argument(
        "--web-ui-port", type=int, default=8200, help="Web UI bind port"
    )
    parser.add_argument(
        "--experiment",
        help=(
            "comparison experiment name: every Web UI run becomes an episode under "
            "outputs/yor/<condition>/<experiment>/<task>/ that the operator reviews"
        ),
    )
    parser.add_argument(
        "--start-label",
        help="start position of the experiment episodes (default kitchen)",
    )
    args = parser.parse_args(argv)

    config = load_config(args.config, task_id=args.task_id)
    if args.instruction:
        config["task"] = {**config.get("task", {}), "instruction": args.instruction}
    if args.output_dir:
        config["trace"] = {**(config.get("trace") or {}), "output_dir": args.output_dir}
    try:
        config = apply_navigation_planner_overrides(
            config,
            enabled=args.navigation_planner,
            memory_path=args.navigation_memory,
            model=args.navigation_planner_model,
        )
        config = apply_readiness_prior_overrides(
            config,
            enabled=args.readiness_prior,
            events_path=args.readiness_events,
            source=args.readiness_prior_source,
        )
    except ValueError as exc:
        parser.error(str(exc))
    if args.max_turns is not None:
        config["agent"] = {**(config.get("agent") or {}), "max_turns": args.max_turns}
    if (
        args.provider
        or args.model
        or args.reasoning_effort
        or args.qwen_base_url
        or args.deepseek_base_url
    ):
        model_config = dict(config.get("model") or {})
        if args.provider:
            previous_provider = str(model_config.get("provider", "vertex")).lower()
            model_config["provider"] = args.provider
            if not args.model and args.provider != previous_provider:
                model_config["name"] = DEFAULT_MODEL_NAMES[args.provider]
        if args.model:
            model_config["name"] = args.model
        if args.reasoning_effort:
            model_config["reasoning_effort"] = args.reasoning_effort
        if args.qwen_base_url:
            model_config["base_url"] = args.qwen_base_url
        if args.deepseek_base_url:
            model_config["base_url"] = args.deepseek_base_url
        config["model"] = model_config

    if args.experiment:
        if not args.web_ui:
            parser.error(
                "--experiment needs --web-ui: the operator reviews every episode in the Web UI"
            )
        try:
            config["experiment"] = experiment_block(
                name=args.experiment,
                start_label=args.start_label or default_start_label(),
                condition=planner_condition(config),
            )
        except ValueError as exc:
            parser.error(str(exc))
    elif args.start_label:
        parser.error("--start-label needs --experiment")

    manual_error = manual_provider_error(config, web_ui=args.web_ui)
    if manual_error:
        parser.error(manual_error)

    if args.web_ui:
        from .web.server import run_web_ui

        run_web_ui(
            config_path=Path(args.config).expanduser(),
            config=config,
            host=args.web_ui_host,
            port=args.web_ui_port,
        )
        return 0

    result = launch(config)
    # This reports why the loop stopped. It is not a physical-success claim.
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
