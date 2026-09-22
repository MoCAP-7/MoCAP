"""Command-line entry point for one supervised real-robot CoW episode."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

from ..experiment.protocol import (
    add_experiment_arguments,
    episode_directory,
    experiment_directory,
    record_review,
    review_episode,
    yor_version,
)
from .camera import LevelCamera, LevelCameraModel
from .config import load_config
from .console import stderr_log
from .episode import CowEpisode, build_agent
from .robot import build_yor_environment, camera_geometry_from_yor
from .tasks import DEFAULT_TASK_CONFIG, instruction_to_goal, load_task
from .upstream import import_cow, localizer_threshold, set_stop_radius


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-id", required=True, help="task in the shared YOR task suite")
    parser.add_argument("--task-config", default=str(DEFAULT_TASK_CONFIG), help="YOR task suite YAML")
    parser.add_argument(
        "--instruction",
        help="override the suite instruction for a debugging run (the YOR environment still uses --task-id)",
    )
    parser.add_argument("--config", default=str(Path(__file__).with_name("config.yaml")))
    parser.add_argument("--output-dir", help="explicit episode directory instead of the experiment layout")
    parser.add_argument(
        "--no-motion",
        action="store_true",
        help="observe, localize, map and decide once without moving the base",
    )
    add_experiment_arguments(parser)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    log = stderr_log

    config = load_config(args.config)
    task = load_task(args.task_id, args.task_config)
    if args.instruction:
        instruction = " ".join(args.instruction.split())
        task = replace(task, instruction=instruction, goal=instruction_to_goal(instruction))
    experiment_dir = experiment_directory(config.experiment.output_root, args.experiment)
    output_dir = episode_directory(args.output_dir, experiment_dir, str(task.task_id), args.start_label)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "task.json").write_text(json.dumps(task.as_dict(), indent=2), encoding="utf-8")
    version = yor_version()
    log(
        f"experiment {experiment_dir.name}, task {task.task_id} (goal {task.goal!r}), start {args.start_label}; "
        f"episode directory {output_dir}"
    )
    if version["yor_dirty"]:
        log(f"warning: the YOR checkout at {version['yor_commit']} has uncommitted changes to tracked files")

    log(f"loading CoW from {config.upstream.repository_path}")
    modules = import_cow(
        config.upstream.repository_path,
        localizer=config.agent.localizer,
        verify=config.upstream.verify_commit,
    )
    set_stop_radius(
        modules.exploration,
        voxel_size_m=config.agent.voxel_size_m,
        stop_radius_m=config.agent.stop_radius_m,
    )
    threshold = (
        config.agent.threshold
        if config.agent.threshold is not None
        else localizer_threshold(modules, config.agent.localizer)
    )
    log(
        f"CoW {modules.commit[:12]} loaded: {config.agent.localizer} threshold {threshold}, "
        f"stop radius {config.agent.stop_radius_m} m, at most {config.agent.max_steps} steps "
        f"and {config.experiment.maximum_duration_s:.0f} s"
    )
    environment = None
    try:
        log("connecting to YOR: ZED stream, Pi base and arm services")
        environment, resolved = build_yor_environment(args.task_config, str(task.task_id))
        camera_height, down = camera_geometry_from_yor(resolved)
        if config.camera.camera_height_m is not None:
            camera_height = config.camera.camera_height_m
        if config.camera.down_camera_xyz is not None:
            down = config.camera.down_camera_xyz
        camera = LevelCamera(
            LevelCameraModel(
                source_width=config.camera.source_resolution[0],
                source_height=config.camera.source_resolution[1],
                intrinsics=config.camera.intrinsics,
                down_camera_xyz=down,
                output_size=config.agent.image_size,
                fov_deg=config.agent.fov_deg,
            )
        )
        log(f"loading the {config.agent.localizer} model on {config.agent.device}")
        agent = build_agent(
            config,
            agent_class=modules.agent_class,
            repository=modules.repo,
            goal=task.goal,
            camera_height_m=camera_height,
            threshold=threshold,
        )
        episode = CowEpisode(
            config,
            agent=agent,
            environment=environment,
            camera=camera,
            task=task,
            output_dir=output_dir,
            voxel_types=modules.exploration.VoxelType,
            no_motion=args.no_motion,
            log=log,
        )
        result = {
            "experiment": experiment_dir.name,
            "start_label": args.start_label,
            **episode.run(),
            "adopted": None,
            "exclusion_reason": None,
            "note": None,
            **version,
            "cow_commit": modules.commit,
            "localizer": config.agent.localizer,
            "threshold": threshold,
            "camera_height_m": camera_height,
            "down_camera_xyz": list(down),
            "level_view_coverage": camera.coverage,
        }
        (output_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2))
    finally:
        if environment is not None:
            environment.safe_shutdown()
    review = review_episode(args.adopt, no_motion=args.no_motion, interactive=sys.stdin.isatty(), log=log)
    record_review(output_dir, experiment_dir=experiment_dir, review=review, result=result, log=log)
    return 1 if result["termination"] == "error" else 0


if __name__ == "__main__":
    raise SystemExit(main())
