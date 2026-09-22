#!/usr/bin/env python3
"""Supervised real-robot test with Viser for manipulation readiness."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


YOR_AGENT_ROOT = Path(__file__).resolve().parents[1]
YOR_ROOT = YOR_AGENT_ROOT.parent
for source_root in reversed((YOR_AGENT_ROOT / "src", YOR_ROOT / "navdp")):
    source = str(source_root)
    if source_root.is_dir():
        if source in sys.path:
            sys.path.remove(source)
        sys.path.insert(0, source)

import numpy as np

from yor_agent.launch import build_environment, load_config
from yor_agent.primitive_config import primitive_settings
from yor_agent.robot.manipulation_readiness import ManipulationReadinessController


DEFAULT_CONFIG = YOR_AGENT_ROOT / "configs" / "mobile_manipulation.yaml"


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"cannot serialize {type(value).__name__}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--object-name", default="pen")
    parser.add_argument("--arm", choices=("left", "right"), default="left")
    parser.add_argument("--viser-host", default="0.0.0.0")
    parser.add_argument("--viser-port", type=int, default=8080)
    parser.add_argument("--viser-url-host", default="127.0.0.1")
    parser.add_argument(
        "--viser-wait",
        action="store_true",
        help="keep Viser alive until Enter or Ctrl-C after the run",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="enable physical base motion after exact typed confirmation",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    object_name = args.object_name.strip()
    if not object_name:
        raise SystemExit("--object-name must be non-empty")
    if not args.execute:
        raise SystemExit(
            "This primitive includes physical coarse and holonomic motion. "
            "Re-run with --execute when the robot is supervised."
        )
    config = load_config(args.config.expanduser().resolve())
    environment = build_environment(config)
    visualizer = None
    try:
        from yor_agent.robot.visualization.manipulation_readiness import (
            ManipulationReadinessViser,
        )

        visualizer = ManipulationReadinessViser(
            host=args.viser_host, port=args.viser_port
        )
        settings = primitive_settings(
            config["primitive_config"], "prepare_for_manipulation"
        )
        controller = ManipulationReadinessController(
            environment,
            config=settings,
            debug_callback=visualizer.publish,
        )
        print(
            f"VISER BROWSER URL: http://{args.viser_url_host}:{visualizer.port}",
            flush=True,
        )
        print(
            "The test may turn, then strafe, then move forward. "
            "Keep the emergency stop in hand.",
            flush=True,
        )
        confirmation = input(
            f"Type MOVE {object_name} {args.arm} exactly to continue: "
        ).strip()
        expected = f"MOVE {object_name} {args.arm}"
        if confirmation != expected:
            print("Confirmation did not match; no primitive motion was started.")
            return 2
        result = controller.prepare_for_manipulation(object_name, args.arm)
        print(json.dumps(result, indent=2, default=_json_default))
        return 0 if result.get("success", False) else 1
    finally:
        environment.safe_shutdown()
        if visualizer is not None and args.viser_wait:
            try:
                input("Viser remains available; press Enter to close: ")
            except (EOFError, KeyboardInterrupt):
                pass
        if visualizer is not None:
            visualizer.close()


if __name__ == "__main__":
    raise SystemExit(main())
