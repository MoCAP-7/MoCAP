from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from nav_planner.cli import DEFAULT_OUTPUT_ROOT, build_parser, resolve_output_dir


class OutputDirectoryTest(unittest.TestCase):
    def test_default_uses_the_recording_directory_name(self) -> None:
        video = Path(
            "~/AriaGen2_recordings/"
            "NavigationAroundKitchen2_20260829_232327/"
            "NavigationAroundKitchen2_20260829_232327_undistorted.mp4"
        )

        output = resolve_output_dir(video)

        self.assertEqual(
            output,
            DEFAULT_OUTPUT_ROOT / "NavigationAroundKitchen2_20260829_232327",
        )

    def test_explicit_output_directory_still_overrides_the_default(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            override = Path(directory) / "custom"

            output = resolve_output_dir("/recording/tour.mp4", override)

            self.assertEqual(output, override.resolve())

    def test_gpt_video_sampling_options_are_exposed(self) -> None:
        args = build_parser().parse_args(
            [
                "tour.mp4",
                "--model",
                "gpt-5.6-sol",
                "--video-frame-fps",
                "2.5",
                "--max-video-input-frames",
                "240",
                "--image-detail",
                "original",
                "--thinking-level",
                "max",
            ]
        )

        self.assertEqual(args.model, "gpt-5.6-sol")
        self.assertEqual(args.video_frame_fps, 2.5)
        self.assertEqual(args.max_video_input_frames, 240)
        self.assertEqual(args.image_detail, "original")
        self.assertEqual(args.thinking_level, "max")


if __name__ == "__main__":
    unittest.main()
