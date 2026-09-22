"""CLI for preserving an existing text-only memory as a frame-backed bundle."""

from __future__ import annotations

import argparse
import json

from .enrich import enrich_existing_memory


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="nav-planner-enrich-memory")
    parser.add_argument("memory")
    parser.add_argument("--output-dir")
    parser.add_argument("--max-reference-frames", type=int, default=32)
    args = parser.parse_args(argv)
    memory = enrich_existing_memory(
        args.memory,
        output_dir=args.output_dir,
        max_reference_frames=args.max_reference_frames,
    )
    print(
        json.dumps(
            {
                "artifact": memory["artifact_path"],
                "reference_frames": len(memory["reference_frames"]),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
