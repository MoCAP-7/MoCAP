"""One-robot, one-episode launcher for the Cap-X baselines on YOR.

``r1pro.yaml`` is the CaP-X baseline: Cap-X's R1Pro mobile-manipulation API
and prompt on YOR, with no YOR clearance gate, advice or planner. The earlier
YOR-authored variants remain: ``config.yaml`` gives Cap-X YOR's production
navigation primitives and the startup navigation planner, and
``coarse_navigation.yaml`` the coarse CaP-X vocabulary without a planner;
both manipulate with Cap-X's top SAM3 instance, top Contact-GraspNet grasp
and direct Cartesian motion. Every variant records each episode like the
navigation comparison methods, in
``<output root>/<experiment>/<task>/<timestamp>_<start label>/`` with the
operator's review in ``result.json``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import subprocess
import sys
import threading
import time
import traceback
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from .bootstrap import REPO_ROOT, configure_import_paths

CAPX_REPO, _ = configure_import_paths()

from baselines.nav.experiment.protocol import (  # noqa: E402
    add_experiment_arguments,
    episode_directory,
    experiment_directory,
    record_review,
    review_episode,
    yor_version,
)
from yor_agent.agents.default import TIME_LIMIT_STOP_REASON_PREFIX  # noqa: E402
from yor_agent.launch import load_config as load_yor_config  # noqa: E402


DEFAULT_CONFIG = Path(__file__).with_name("config.yaml")
NAVIGATION_MODES = ("yor", "coarse", "r1pro")
OPERATOR_STOP_REASON = "operator requested stop"
NAVIGATION_IMPLEMENTATIONS = {
    "yor": (
        "Exact current yor_agent drive_straight, turn_relative, "
        "drive_lateral, and dock_to_visible_object registrations."
    ),
    "coarse": (
        "Current yor_agent coarse CaP-X registrations: go_forward (1 m "
        "drive_straight), turn_left_45_degrees and turn_right_45_degrees "
        "(45-degree turn_relative), goto_planar_position (relative planar "
        "move keeping the heading), and say_something."
    ),
    "r1pro": (
        "Cap-X's R1Pro control API (capx/integrations/r1pro/control.py): "
        "navigate_to_pose turns toward the goal, drives straight and turns to "
        "the goal heading with the base controller's obstacle check off, in "
        "the ZED odometry frame with Cap-X's five-waypoint fallback; "
        "find_object_base_rotate through 0.5 rad in-place turns, "
        "get_robot_position and get_navigation_pose as in Cap-X; no YOR "
        "clearance gate, Nav2, docking or advice text."
    ),
}
MANIPULATION_IMPLEMENTATIONS = {
    "r1pro": (
        "Cap-X R1Pro semantics: top SAM3 instance, Open3D-style statistical "
        "outlier removal and PCA oriented bounding box, top Contact-GraspNet "
        "grasp at the calibrated Nero TCP, solve_ik/move_to_joint_positions "
        "through the Pi Mink IK and joint streaming RPC, move_hand/lift_arm "
        "through the Pi Cartesian RPC, grasp_object with Cap-X's reach check "
        "and correction; failures print and return False/None."
    ),
    "default": (
        "Cap-X top-SAM3/top-Contact-GraspNet/direct-Cartesian semantics "
        "adapted only for YOR calibration and guarded hardware RPC."
    ),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--task-id", required=True)
    parser.add_argument(
        "--navigation-memory",
        help=(
            "JSON memory built from the passive human video; required when the "
            "config enables the startup navigation planner."
        ),
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="explicit episode directory instead of the experiment layout",
    )
    parser.add_argument("--model")
    parser.add_argument("--server-url")
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--reasoning-effort")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate configuration without connecting to or moving the robot.",
    )
    add_experiment_arguments(parser)
    return parser


def main(argv: list[str] | None = None) -> int:
    cli = build_parser().parse_args(argv)
    config_path = cli.config.expanduser().resolve()
    config = _load_mapping(config_path)
    if not (CAPX_REPO / "capx").is_dir():
        raise FileNotFoundError(f"CAPX_REPO is not a Cap-X checkout: {CAPX_REPO}")
    allowed_tasks = [str(value) for value in config.get("allowed_tasks", [])]
    if cli.task_id not in allowed_tasks:
        raise ValueError(
            f"task {cli.task_id!r} is not enabled; choose one of {allowed_tasks}"
        )
    navigation = str(config.get("navigation", "yor"))
    if navigation not in NAVIGATION_MODES:
        raise ValueError(
            f"navigation must be one of {list(NAVIGATION_MODES)}: {navigation!r}"
        )
    condition = str(config.get("condition") or config_path.stem)

    yor_config_path = _resolve_from(config_path, config["yor_agent_config"])
    resolved_yor_config = load_yor_config(yor_config_path, task_id=cli.task_id)
    task_instruction = str(resolved_yor_config["task"]["instruction"]).strip()
    if not task_instruction:
        raise ValueError(f"task {cli.task_id!r} has an empty instruction")
    time_limit_s = _time_limit_s(resolved_yor_config)
    capx_manipulation = dict(config.get("capx_manipulation", {}))
    simulate_gripper = capx_manipulation.get("simulate_gripper", False)
    if type(simulate_gripper) is not bool:
        raise TypeError("capx_manipulation.simulate_gripper must be boolean")
    task_hints = str(config.get("task_hints") or "").strip()

    planner_config = dict(config.get("navigation_planner") or {})
    # A planner block without ``enabled`` turns the planner on; no block, off.
    planner_enabled = bool(planner_config.pop("enabled", bool(planner_config)))
    image_max_width = int(planner_config.pop("image_max_width", 1024))
    navigation_memory: Path | None = None
    navigation_planner = None
    if planner_enabled:
        if cli.navigation_memory is None:
            raise ValueError(
                f"{config_path.name} enables the startup navigation planner; "
                "pass --navigation-memory"
            )
        navigation_memory = Path(cli.navigation_memory).expanduser().resolve()
        if not navigation_memory.is_file():
            raise FileNotFoundError(
                f"navigation memory does not exist: {navigation_memory}"
            )
        planner_config["memory_path"] = str(navigation_memory)
        # Constructor validation is cheap and guarantees validate-only checks the
        # exact current planner contract without contacting its model endpoint.
        from yor_agent.models.navigation_planner import (
            SUPPORTED_MEMORY_SCHEMAS,
            StartupNavigationPlanner,
        )

        navigation_planner = StartupNavigationPlanner(planner_config)
        _validate_navigation_memory(navigation_memory, SUPPORTED_MEMORY_SCHEMAS)
    elif cli.navigation_memory is not None:
        raise ValueError(
            f"{config_path.name} runs without the startup navigation planner; "
            "drop --navigation-memory"
        )

    policy = dict(config.get("policy", {}))
    provider = str(policy.get("provider", "openai")).strip().lower()
    model = str(cli.model or policy["model"])
    server_url = str(cli.server_url or policy["server_url"])
    temperature = float(
        policy.get("temperature", 1.0)
        if cli.temperature is None
        else cli.temperature
    )
    max_tokens = int(
        policy.get("max_tokens", 20480)
        if cli.max_tokens is None
        else cli.max_tokens
    )
    reasoning_effort = str(
        cli.reasoning_effort or policy.get("reasoning_effort", "medium")
    )
    experiment_dir = experiment_directory(
        _resolve_from(config_path, config["output_root"]), cli.experiment
    )

    if cli.validate_only:
        print(
            json.dumps(
                {
                    "valid": True,
                    "task_id": cli.task_id,
                    "task_instruction": task_instruction,
                    "condition": condition,
                    "navigation": navigation,
                    "navigation_planner": planner_enabled,
                    "navigation_memory": _path_or_none(navigation_memory),
                    "time_limit_s": time_limit_s,
                    "policy_model": model,
                    "reasoning_effort": reasoning_effort,
                    "yor_agent_config": str(yor_config_path),
                    "experiment_dir": str(experiment_dir),
                    "capx_repo": str(CAPX_REPO),
                },
                indent=2,
            )
        )
        return 0

    output_dir = episode_directory(
        cli.output_dir, experiment_dir, cli.task_id, cli.start_label
    )

    # Keep validate-only usable on a deployment host before Cap-X's optional
    # runtime dependencies have been installed.
    from capx.envs import trial as capx_trial
    from capx.envs.launch import LaunchArgs
    from capx.llm.client import VLM_MODELS

    from .code_env import CapXYorCodeEnv, CapXYorCodeExecConfig
    from .environment import CapXYorLowLevelEnv
    from .llm import OpenAIResponsesQuery
    from .recorder import EpisodeRecorder

    # Cap-X gates image attachment with this list. Runtime registration permits
    # the same custom visual policy model used by yor_agent without editing the
    # upstream checkout.
    for visual_model in (
        model,
        str(policy.get("visual_differencing_model", model)),
    ):
        if visual_model not in VLM_MODELS:
            VLM_MODELS.append(visual_model)

    low_level = CapXYorLowLevelEnv(
        yor_config_path=yor_config_path,
        task_id=cli.task_id,
        capx_manipulation_config=capx_manipulation,
        episode_directory=output_dir,
    )
    try:
        env_config = CapXYorCodeExecConfig(
            low_level=low_level,
            apis=[],
            task_instruction=task_instruction,
            navigation=navigation,
            navigation_planner=planner_config if planner_enabled else None,
            planner_image_max_width=image_max_width,
            events_directory=str(output_dir),
            time_limit_s=time_limit_s,
            task_hints=task_hints,
            enable_render=True,
            viser_debug=False,
        )
        env = CapXYorCodeEnv(env_config, planner=navigation_planner)
    except BaseException:
        low_level.close()
        raise
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except BaseException:
        env.close()
        raise

    args = LaunchArgs(
        config_path=str(config_path),
        server_url=server_url,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        reasoning_effort=reasoning_effort,
        api_key=_secret_from_env(policy.get("api_key_env")),
        use_visual_feedback=bool(policy.get("use_visual_feedback", True)),
        use_img_differencing=bool(policy.get("use_img_differencing", False)),
        use_video_differencing=bool(policy.get("use_video_differencing", False)),
        use_wrist_camera=False,
        use_legacy_multi_turn_decision_prompt=False,
        visual_differencing_model=str(
            policy.get("visual_differencing_model", model)
        ),
        visual_differencing_model_server_url=str(
            policy.get("visual_differencing_model_server_url", server_url)
        ),
        visual_differencing_model_api_key=_secret_from_env(
            policy.get("visual_differencing_api_key_env")
        ),
        total_trials=1,
        num_workers=1,
        record_video=bool(policy.get("record_video", False)),
        output_dir=str(output_dir),
        debug=bool(policy.get("debug", False)),
        use_oracle_code=False,
        use_parallel_ensemble=False,
        use_multimodel=False,
        web_ui=False,
    )
    run_config = {
        "output_dir": str(output_dir),
        "record_video": bool(args.record_video),
        "use_visual_feedback": bool(args.use_visual_feedback),
        "use_img_differencing": bool(args.use_img_differencing),
        "use_video_differencing": bool(args.use_video_differencing),
        "use_wrist_camera": False,
        "use_oracle_code": False,
        "use_parallel_ensemble": False,
        "use_multimodel": False,
        "save_multiturn_prompts": bool(
            policy.get("save_multiturn_prompts", True)
        ),
    }
    multiturn_limit = int(capx_trial.MULTITURN_LIMIT)
    metadata = _baseline_metadata(
        task_id=cli.task_id,
        instruction=task_instruction,
        condition=condition,
        navigation=navigation,
        planner_enabled=planner_enabled,
        yor_config_path=yor_config_path,
        navigation_memory=navigation_memory,
        model=model,
        server_url=server_url,
        output_dir=output_dir,
        gripper_simulated=simulate_gripper,
        provider=provider,
        time_limit_s=time_limit_s,
        multiturn_limit=multiturn_limit,
    )
    metadata_path = output_dir / "baseline_metadata.json"
    _write_json(metadata_path, metadata)

    version = yor_version()
    _log(
        f"experiment {experiment_dir.name}, task {cli.task_id}, start "
        f"{cli.start_label}, {condition}; episode directory {output_dir}"
    )
    if version["yor_dirty"]:
        _log(
            f"warning: the YOR checkout at {version['yor_commit']} has "
            "uncommitted changes to tracked files"
        )

    summary = None
    error: BaseException | None = None
    timer: threading.Timer | None = None
    recorder = EpisodeRecorder(output_dir, getattr(low_level, "yor_environment", None))
    recording: dict[str, Any] | None = None
    original_query_model = capx_trial._query_model
    previous_sigint = signal.getsignal(signal.SIGINT)
    # The episode clock starts before Cap-X resets the robot, like the
    # clocks of the YOR conditions.
    started_at = _now()
    started = time.monotonic()
    try:
        recorder.start()
        if provider == "openai":
            capx_trial._query_model = OpenAIResponsesQuery(
                api_key=_secret_from_env(policy.get("api_key_env")),
                timeout_s=float(policy.get("timeout_s", 180.0)),
            )
        signal.signal(signal.SIGINT, _operator_stop_handler(env))
        if time_limit_s is not None:
            timer = threading.Timer(
                time_limit_s,
                env.request_stop,
                args=(f"{TIME_LIMIT_STOP_REASON_PREFIX} of {time_limit_s:g} s reached",),
            )
            timer.daemon = True
            timer.start()
        summary = capx_trial._run_single_trial(
            env,
            0,
            args,
            run_config,
            str(config.get("multi_turn_prompt") or "").strip() or None,
        )
    except (Exception, KeyboardInterrupt) as exc:
        # The episode still ends with the robot stopped and the operator's
        # review, like an episode that ran to its end.
        traceback.print_exc()
        error = exc
    finally:
        if timer is not None:
            timer.cancel()
        elapsed_s = round(time.monotonic() - started, 3)
        ended_at = _now()
        signal.signal(signal.SIGINT, previous_sigint)
        capx_trial._query_model = original_query_model
        # Finish the recording while the camera stream is still open.
        recording = recorder.stop()
        try:
            env.close()
        except Exception as exc:  # noqa: BLE001 - the episode is still recorded
            _log(f"closing the robot environment failed: {exc!r}")

    stop_reason = env.stop_reason
    termination = _termination(stop_reason, summary, error, multiturn_limit)
    metadata["navigation_advice"] = env.navigation_advice
    if summary is not None:
        metadata["capx_execution"] = {
            "program_executed_without_exception": bool(summary.success),
            "sandbox_rc": int(summary.sandbox_rc),
            "num_regenerations": int(summary.num_regenerations),
            "num_code_blocks": int(summary.num_code_blocks),
        }
        _write_json(output_dir / "trial_summary.json", asdict(summary))
    if error is not None:
        metadata["run_error"] = repr(error)
    metadata["termination"] = termination
    metadata["physical_task_success"] = None
    metadata["physical_task_success_note"] = (
        "Assigned by the operator's review in result.json; Cap-X sandbox "
        "success is not physical task success."
    )
    _write_json(metadata_path, metadata)

    result = {
        "task_id": cli.task_id,
        "instruction": task_instruction,
        "experiment": experiment_dir.name,
        "start_label": cli.start_label,
        "condition": condition,
        "navigation": navigation,
        "termination": termination,
        "reason": stop_reason
        or (None if error is None else f"{type(error).__name__}: {error}"),
        "elapsed_s": elapsed_s,
        "started_at": started_at,
        "ended_at": ended_at,
        "time_limit_s": time_limit_s,
        "multiturn_limit": multiturn_limit,
        "num_code_blocks": _summary_int(summary, "num_code_blocks"),
        "num_regenerations": _summary_int(summary, "num_regenerations"),
        "num_finishes": _summary_int(summary, "num_finishes"),
        "sandbox_rc": _summary_int(summary, "sandbox_rc"),
        "model": {
            "provider": provider,
            "name": model,
            "reasoning_effort": reasoning_effort,
        },
        "navigation_planner": {
            "enabled": planner_enabled,
            "model": planner_config.get("model") if planner_enabled else None,
            "memory_path": _path_or_none(navigation_memory),
        },
        "gripper_simulated": simulate_gripper,
        "recording": recording,
        "error": None
        if error is None
        else {"type": type(error).__name__, "message": str(error)},
        "task_success": None,
        "task_success_source": None,
        "adopted": None,
        "exclusion_reason": None,
        "note": None,
        **version,
    }
    (output_dir / "result.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))
    review = review_episode(
        cli.adopt, no_motion=False, interactive=sys.stdin.isatty(), log=_log
    )
    record_review(
        output_dir,
        experiment_dir=experiment_dir,
        review=review,
        result=result,
        log=_log,
    )
    return 1 if error is not None and not isinstance(error, KeyboardInterrupt) else 0


def _termination(
    stop_reason: str | None,
    summary: Any,
    error: BaseException | None,
    multiturn_limit: int,
) -> str:
    """How an episode ended, in the terms of the YOR conditions where they match."""

    if stop_reason is not None:
        if stop_reason.startswith(TIME_LIMIT_STOP_REASON_PREFIX):
            return "time_limit"
        return "operator_stop"
    if isinstance(error, KeyboardInterrupt):
        return "operator_stop"
    if error is not None or summary is None:
        return "error"
    if summary.num_finishes:
        return "finished"
    # Cap-X runs at most multiturn_limit + 1 programs in one trial.
    if summary.num_code_blocks > multiturn_limit:
        return "multiturn_limit"
    return "program_ended"


def _operator_stop_handler(env: Any):
    """Ctrl+C stops the episode; a second Ctrl+C aborts the trial."""

    def handler(signum: int, frame: Any) -> None:
        del signum, frame
        if env.stop_reason is not None:
            raise KeyboardInterrupt
        # Unbuffered: the interrupted main thread may be inside a print.
        os.write(
            2,
            b"[capx] Ctrl+C: stopping the base; the episode ends before its next "
            b"program (Ctrl+C again to abort)\n",
        )
        # The base stop takes the controller's command lock, which the
        # interrupted main thread may be holding.
        threading.Thread(
            target=env.request_stop, args=(OPERATOR_STOP_REASON,), daemon=True
        ).start()

    return handler


def _time_limit_s(resolved_yor_config: dict[str, Any]) -> float | None:
    """The YOR config's agent wall-clock limit, shared with the YOR conditions."""

    value = (resolved_yor_config.get("agent") or {}).get("time_limit_s")
    if value is None:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(
            f"agent.time_limit_s must be a positive number of seconds: {value!r}"
        )
    return float(value)


def _summary_int(summary: Any, name: str) -> int | None:
    return None if summary is None else int(getattr(summary, name))


def _path_or_none(path: Path | None) -> str | None:
    return None if path is None else str(path)


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _log(message: str) -> None:
    print(f"[capx] {message}", file=sys.stderr, flush=True)


def _load_mapping(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"configuration must be a mapping: {path}")
    return payload


def _resolve_from(config_path: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = config_path.parent / path
    return path.resolve()


def _validate_navigation_memory(path: Path, supported_schemas: set[str]) -> None:
    memory = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(memory, dict):
        raise TypeError(f"navigation memory must be a JSON object: {path}")
    if memory.get("schema_version") not in supported_schemas:
        raise ValueError(f"unsupported navigation memory schema: {path}")
    if not str(memory.get("memory_text", "")).strip():
        raise ValueError(f"navigation memory has no memory_text: {path}")


def _secret_from_env(variable: Any) -> str | None:
    name = str(variable or "").strip()
    return os.environ.get(name) if name else None


def _baseline_metadata(
    *,
    task_id: str,
    instruction: str,
    condition: str,
    navigation: str,
    planner_enabled: bool,
    yor_config_path: Path,
    navigation_memory: Path | None,
    model: str,
    server_url: str,
    output_dir: Path,
    gripper_simulated: bool,
    provider: str,
    time_limit_s: float | None,
    multiturn_limit: int,
) -> dict[str, Any]:
    return {
        "schema_version": "yor-capx-baseline-v2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "task_id": task_id,
        "task_instruction": instruction,
        "condition": condition,
        "navigation": navigation,
        "navigation_planner_enabled": planner_enabled,
        "policy_model": model,
        "policy_provider": provider,
        "policy_server_url": server_url,
        "gripper_simulated": gripper_simulated,
        "time_limit_s": time_limit_s,
        "multiturn_limit": multiturn_limit,
        "output_dir": str(output_dir),
        "yor_agent_config": str(yor_config_path),
        "navigation_memory": _path_or_none(navigation_memory),
        "capx_repo": str(CAPX_REPO),
        "yor_repo": str(REPO_ROOT),
        "capx_source": _git_source_state(CAPX_REPO),
        "yor_source": _git_source_state(REPO_ROOT),
        "navigation_implementation": NAVIGATION_IMPLEMENTATIONS[navigation],
        "manipulation_implementation": MANIPULATION_IMPLEMENTATIONS.get(
            navigation, MANIPULATION_IMPLEMENTATIONS["default"]
        ),
        "excluded_yor_manipulation_features": [
            "GraspGen-X",
            "target-association filtering",
            "collision filtering",
            "IK candidate precheck",
            "cuRobo motion planning",
            "prepare_for_manipulation",
        ],
    }


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )


def _git_source_state(repository: Path) -> dict[str, Any]:
    """Return revision provenance without failing a physical trial on git."""

    try:
        revision = subprocess.run(
            ["git", "-C", str(repository), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5.0,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(repository), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5.0,
        ).stdout
        return {"revision": revision, "worktree_dirty": bool(status.strip())}
    except (OSError, subprocess.SubprocessError) as exc:
        return {"revision": None, "worktree_dirty": None, "error": repr(exc)}


if __name__ == "__main__":
    raise SystemExit(main())
