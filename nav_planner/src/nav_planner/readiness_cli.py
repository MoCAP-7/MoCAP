"""Offline command that labels manipulation events for the readiness prior."""

from __future__ import annotations

import argparse
import json
import sys

from .cli import resolve_output_dir
from .readiness import (
    ManipulationEventsBuilder,
    ReadinessConfig,
    parse_manual_event,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nav-planner-readiness",
        description=(
            "Find the moments in a passive egocentric video where the wearer "
            "stood ready to manipulate an object, and save the nav/ready/grasp "
            "frames plus an events JSON for the robot's readiness prior."
        ),
    )
    parser.add_argument("video")
    parser.add_argument(
        "--output-dir",
        help="override the default nav_planner/outputs/<video-parent-directory>",
    )
    parser.add_argument(
        "--intrinsics",
        help=(
            "pinhole intrinsics JSON for the video; by default "
            "<video>.intrinsics.json next to the video is used when present"
        ),
    )
    parser.add_argument(
        "--label",
        choices=("vlm", "manual"),
        default="vlm",
        help=(
            "vlm: a Gemini/GPT model labels every low-motion window; "
            "manual: take the events from --event (default: vlm)"
        ),
    )
    parser.add_argument("--model", default="gemini-3.7-flash")
    parser.add_argument(
        "--thinking-level",
        choices=("none", "low", "medium", "high", "xhigh", "max"),
        default="high",
        help=(
            "reasoning effort; Gemini accepts low/medium/high and GPT also "
            "accepts none/xhigh/max"
        ),
    )
    parser.add_argument("--request-timeout", type=float, default=900.0)
    parser.add_argument(
        "--event",
        action="append",
        default=[],
        metavar="OBJECT:HAND:T_READY:T_GRASP",
        help=(
            "one manual event, for example can:right:16.5:19.0 (hand is left, "
            "right, or both; times in seconds); repeat for several events"
        ),
    )
    parser.add_argument(
        "--motion-threshold",
        type=float,
        default=None,
        help=(
            "absolute per-second YAVG below which the wearer counts as still "
            "(default: --motion-threshold-factor x the median second of the video)"
        ),
    )
    parser.add_argument(
        "--motion-threshold-factor",
        type=float,
        default=1.0,
        help=(
            "multiple of the median per-second YAVG used as the still threshold "
            "(default 1.0); ignored when --motion-threshold is given"
        ),
    )
    parser.add_argument(
        "--frame-width",
        type=int,
        default=1280,
        help="width of the saved nav/ready/grasp frames",
    )
    parser.add_argument(
        "--sample-fps",
        type=float,
        default=2.0,
        help="frames per second sent to the VLM per low-motion window",
    )
    parser.add_argument(
        "--sample-width",
        type=int,
        default=768,
        help="maximum width of the frames sent to the VLM",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.label == "manual" and not args.event:
        parser.error(
            "--label manual requires at least one --event OBJECT:HAND:T_READY:T_GRASP"
        )
    if args.label == "vlm" and args.event:
        parser.error("--event is only valid with --label manual")
    try:
        manual_events = [parse_manual_event(text) for text in args.event]
        config = ReadinessConfig(
            motion_threshold=args.motion_threshold,
            motion_threshold_factor=args.motion_threshold_factor,
            sample_fps=args.sample_fps,
            sample_width=args.sample_width,
            frame_width=args.frame_width,
            model=args.model,
            thinking_level=args.thinking_level,
            request_timeout_s=args.request_timeout,
        )
    except ValueError as exc:
        parser.error(str(exc))
    builder = ManipulationEventsBuilder(
        config,
        on_progress=lambda event: print(json.dumps(event), file=sys.stderr),
    )
    output_dir = resolve_output_dir(args.video, args.output_dir)
    result = builder.build(
        args.video,
        output_dir,
        label=args.label,
        manual_events=manual_events,
        intrinsics_path=args.intrinsics,
    )
    print(
        json.dumps(
            {
                "artifact": result["artifact_path"],
                "events": len(result["events"]),
                "windows": len(result["motion"]["windows"]),
                "motion_threshold": result["motion"]["threshold"],
                "motion_windows": result["motion"]["windows"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
