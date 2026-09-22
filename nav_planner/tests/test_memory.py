from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from nav_planner.memory import (
    GeminiConfig,
    GeminiVideoMemoryBuilder,
    VideoMemoryBuilder,
    VideoMemoryConfig,
)


class _Files:
    def __init__(self) -> None:
        self.uploaded = None
        self.get_calls = 0

    def upload(self, *, file, config=None):
        self.uploaded = file
        self.upload_config = config
        return SimpleNamespace(
            name="files/video", uri="gemini://video", mime_type="video/mp4",
            state=SimpleNamespace(name="PROCESSING"),
        )

    def get(self, *, name):
        self.get_calls += 1
        return SimpleNamespace(
            name=name, uri="gemini://video", mime_type="video/mp4",
            state=SimpleNamespace(name="ACTIVE"),
        )


class _Interactions:
    def __init__(self) -> None:
        self.request = None

    def create(self, **kwargs):
        self.request = kwargs
        return SimpleNamespace(output_text="Generic scene memory")


class _Responses:
    def __init__(self) -> None:
        self.request = None

    def create(self, **kwargs):
        self.request = kwargs
        return SimpleNamespace(output_text="Generic GPT scene memory")


class GeminiMemoryTest(unittest.TestCase):
    def test_whole_video_is_sent_once_without_task_conditioning(self) -> None:
        files = _Files()
        interactions = _Interactions()
        client = SimpleNamespace(files=files, interactions=interactions)
        config = GeminiConfig(upload_poll_interval_s=0.0)
        builder = GeminiVideoMemoryBuilder(config, client=client)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "tour.mp4"
            video.write_bytes(b"fake video")
            memory = builder.build(video, root / "outputs")
            output = Path(memory["artifact_path"])

            self.assertEqual(memory["memory_text"], "Generic scene memory")
            self.assertTrue(output.is_file())
            self.assertRegex(output.name, r"^memory_\d{8}T\d{6}\.\d{6}Z\.json$")
            self.assertEqual(memory["generation_prompt"], request_prompt := interactions.request["input"][1]["text"])
            self.assertEqual(memory["reference_frames"], [])
            request = interactions.request
            self.assertEqual(request["model"], "gemini-3.7-flash")
            self.assertEqual(request["generation_config"], {"thinking_level": "high"})
            self.assertEqual(request["input"][0]["type"], "video")
            self.assertEqual(request["input"][0]["uri"], "gemini://video")
            self.assertEqual(
                files.upload_config["http_options"]["timeout"], 1800000
            )
            prompt = request_prompt.lower()
            normalized_prompt = " ".join(prompt.split())
            self.assertIn("reusable navigation memory", normalized_prompt)
            self.assertIn("traversable connections", normalized_prompt)
            self.assertIn("what should become visible next", normalized_prompt)
            self.assertIn("different height and viewpoint", normalized_prompt)
            self.assertIn(
                "small, portable, or visually subtle objects", normalized_prompt
            )
            self.assertIn("support surface", normalized_prompt)
            self.assertIn("best supporting timestamp", normalized_prompt)
            self.assertIn("partial occlusion", normalized_prompt)
            for held_out_target in ("trash bin", "marker", "cardbox", "water bottle"):
                self.assertNotIn(held_out_target, normalized_prompt)

    def test_each_generation_creates_a_new_timestamped_artifact(self) -> None:
        files = _Files()
        interactions = _Interactions()
        client = SimpleNamespace(files=files, interactions=interactions)
        builder = GeminiVideoMemoryBuilder(
            GeminiConfig(upload_poll_interval_s=0.0), client=client
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "tour.mp4"
            video.write_bytes(b"fake video")
            output_dir = root / "outputs"
            first = builder.build(video, output_dir)
            second = builder.build(video, output_dir)
            self.assertNotEqual(first["artifact_path"], second["artifact_path"])
            self.assertTrue(Path(first["artifact_path"]).is_file())
            self.assertTrue(Path(second["artifact_path"]).is_file())


class OpenAIMemoryTest(unittest.TestCase):
    def test_timestamped_frames_are_sent_to_responses_api(self) -> None:
        responses = _Responses()
        client = SimpleNamespace(responses=responses)

        def fake_extract(video, output_dir, **kwargs):
            output_dir.mkdir(parents=True)
            first = output_dir / "frame_000000.jpg"
            second = output_dir / "frame_000001.jpg"
            first.write_bytes(b"first-frame")
            second.write_bytes(b"second-frame")
            return [
                {
                    "timestamp": "0:00.000",
                    "time_s": 0.0,
                    "path": first,
                    "mime_type": "image/jpeg",
                },
                {
                    "timestamp": "0:00.500",
                    "time_s": 0.5,
                    "path": second,
                    "mime_type": "image/jpeg",
                },
            ]

        builder = VideoMemoryBuilder(
            VideoMemoryConfig(model="gpt-5.6-sol"), client=client
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "tour.mp4"
            video.write_bytes(b"fake video")
            with mock.patch(
                "nav_planner.memory.extract_video_sample_frames",
                side_effect=fake_extract,
            ):
                memory = builder.build(video, root / "outputs")

        self.assertEqual(memory["provider"], "openai")
        self.assertEqual(memory["schema_version"], "yor-video-memory-v1")
        self.assertEqual(memory["memory_text"], "Generic GPT scene memory")
        self.assertEqual(memory["video_input"]["frame_count"], 2)
        request = responses.request
        self.assertEqual(request["model"], "gpt-5.6-sol")
        self.assertEqual(request["reasoning"], {"effort": "high"})
        content = request["input"][0]["content"]
        self.assertEqual(content[2]["text"], "VIDEO FRAME at 0:00.000:")
        self.assertEqual(content[3]["type"], "input_image")
        self.assertEqual(content[3]["detail"], "high")
        self.assertTrue(
            content[3]["image_url"].endswith("Zmlyc3QtZnJhbWU=")
        )
        self.assertEqual(content[4]["text"], "VIDEO FRAME at 0:00.500:")


if __name__ == "__main__":
    unittest.main()
