"""Offline task-agnostic navigation-memory command."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .memory import VideoMemoryBuilder, VideoMemoryConfig

DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parents[2] / "outputs"


def resolve_output_dir(
    video_path: str | Path,
    explicit_output_dir: str | Path | None = None,
) -> Path:
    """Group memory versions by the source recording directory name."""

    if explicit_output_dir is not None:
        return Path(explicit_output_dir).expanduser().resolve()
    video = Path(video_path).expanduser().resolve()
    recording_name = video.parent.name
    if not recording_name:
        raise ValueError(f"cannot derive recording directory from {video}")
    return DEFAULT_OUTPUT_ROOT / recording_name


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nav-planner",
        description=(
            "Use Gemini native video or GPT timestamped frames to save "
            "task-agnostic navigation memory."
        ),
    )
    parser.add_argument("video")
    parser.add_argument(
        "--output-dir",
        help=(
            "override the default nav_planner/outputs/<video-parent-directory>"
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
    parser.add_argument("--upload-timeout", type=float, default=1800.0)
    parser.add_argument("--request-timeout", type=float, default=900.0)
    parser.add_argument("--max-reference-frames", type=int, default=32)
    parser.add_argument(
        "--video-frame-fps",
        type=float,
        default=2.0,
        help="sampling rate used only by image-only GPT video understanding",
    )
    parser.add_argument(
        "--max-video-input-frames",
        type=int,
        default=300,
        help="maximum timestamped frames sent to GPT (maximum 1500)",
    )
    parser.add_argument(
        "--video-frame-width",
        type=int,
        default=2048,
        help="maximum width of frames sent to GPT",
    )
    parser.add_argument(
        "--image-detail",
        choices=("low", "high", "original", "auto"),
        default="high",
        help="OpenAI image detail level used for sampled video frames",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    builder = VideoMemoryBuilder(
        VideoMemoryConfig(
            model=args.model,
            thinking_level=args.thinking_level,
            upload_timeout_s=args.upload_timeout,
            request_timeout_s=args.request_timeout,
            max_reference_frames=args.max_reference_frames,
            video_frame_fps=args.video_frame_fps,
            max_video_input_frames=args.max_video_input_frames,
            video_frame_width=args.video_frame_width,
            image_detail=args.image_detail,
        ),
        on_progress=lambda event: print(json.dumps(event), file=sys.stderr),
    )
    output_dir = resolve_output_dir(args.video, args.output_dir)
    result = builder.build(args.video, output_dir)
    artifact = Path(result["artifact_path"])
    print(
        json.dumps(
            {
                "artifact": str(artifact),
                "schema_version": result.get("schema_version"),
                "model": result.get("model"),
            },
            indent=2,
        )
    )
    return 0
