from __future__ import annotations

import base64
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from yor_agent.models.llm import LLM, observation_rgb_data_url
from yor_agent.models.navigation_planner import (
    GeminiStartupNavigationPlanner,
    StartupNavigationPlanner,
)


class _Interactions:
    def __init__(self) -> None:
        self.request = None

    def create(self, **kwargs):
        self.request = kwargs
        return SimpleNamespace(output_text="Go toward the recognizable central area.")


class _Responses:
    def __init__(self) -> None:
        self.request = None

    def create(self, **kwargs):
        self.request = kwargs
        return SimpleNamespace(output_text="Turn left toward the orange wall.")


class StartupPlannerTest(unittest.TestCase):
    def test_planner_receives_exact_same_png_as_policy_vlm(self) -> None:
        interactions = _Interactions()
        client = SimpleNamespace(interactions=interactions)
        with tempfile.TemporaryDirectory() as directory:
            memory_path = Path(directory) / "memory.json"
            reference_path = Path(directory) / "frame_000003.jpg"
            reference_path.write_bytes(b"passive-frame")
            memory_path.write_text(
                json.dumps(
                    {
                        "schema_version": "yor-gemini-video-memory-v1",
                        "memory_text": "A generic memory with an orange central area.",
                        "reference_frames": [
                            {
                                "timestamp": "00:03",
                                "path": reference_path.name,
                                "mime_type": "image/jpeg",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            planner = GeminiStartupNavigationPlanner(
                {"memory_path": str(memory_path)}, client=client
            )
            rgb = np.arange(80 * 120 * 3, dtype=np.uint8).reshape(80, 120, 3)
            observation = {"robot0_robotview": {"images": {"rgb": rgb}}}

            advice = planner.plan(
                task="An arbitrary held-out task",
                observation=observation,
                image_max_width=64,
            )

            self.assertIn("recognizable central area", advice)
            image_input = interactions.request["input"][2]
            self.assertEqual(image_input["type"], "image")
            self.assertEqual(image_input["mime_type"], "image/png")
            expected_url = observation_rgb_data_url(observation, max_width=64)
            expected_payload = expected_url.partition(",")[2]
            self.assertEqual(image_input["data"], expected_payload)

            policy_model = LLM({"provider": "vertex", "image_max_width": 64})
            policy_messages = policy_model.initial_messages(
                task="An arbitrary held-out task",
                observation=observation,
                primitive_docs="def observe(): ...",
            )
            policy_url = policy_messages[-1]["content"][1]["image_url"]["url"]
            self.assertEqual(
                base64.b64decode(image_input["data"]),
                base64.b64decode(policy_url.partition(",")[2]),
            )
            prompt = interactions.request["input"][0]["text"]
            self.assertIn("A generic memory", prompt)
            self.assertIn("An arbitrary held-out task", prompt)
            self.assertIn("at most 120 English words", prompt)
            self.assertIn("wrong direction to avoid", prompt)
            self.assertIn("about 0.43 m to each side", prompt)
            self.assertIn("move the full footprint away", prompt)
            self.assertIn("do not merely point the camera away", prompt)
            self.assertIn("rear path is known clear", prompt)
            self.assertIn("out-of-view as proof", prompt)
            self.assertIn("prefer dock_to_visible_object", prompt)
            self.assertEqual(
                interactions.request["input"][3]["text"],
                "PASSIVE-VIDEO REFERENCE FRAME at 00:03:",
            )
            self.assertEqual(
                base64.b64decode(interactions.request["input"][4]["data"]),
                b"passive-frame",
            )

    def test_openai_planner_receives_current_and_memory_images(self) -> None:
        responses = _Responses()
        client = SimpleNamespace(responses=responses)
        with tempfile.TemporaryDirectory() as directory:
            memory_path = Path(directory) / "memory.json"
            reference_path = Path(directory) / "frame_000003.jpg"
            reference_path.write_bytes(b"passive-frame")
            memory_path.write_text(
                json.dumps(
                    {
                        "schema_version": "yor-video-memory-v1",
                        "memory_text": "Reception has a bright orange wall.",
                        "reference_frames": [
                            {
                                "timestamp": "00:03",
                                "path": reference_path.name,
                                "mime_type": "image/jpeg",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            planner = StartupNavigationPlanner(
                {
                    "memory_path": str(memory_path),
                    "model": "gpt-5.6-sol",
                    "thinking_level": "high",
                },
                client=client,
            )
            rgb = np.arange(80 * 120 * 3, dtype=np.uint8).reshape(80, 120, 3)
            observation = {"robot0_robotview": {"images": {"rgb": rgb}}}

            advice = planner.plan(
                task="Find an arbitrary object",
                observation=observation,
                image_max_width=64,
            )

            self.assertIn("Turn left", advice)
            self.assertEqual(planner.provider, "openai")
            request = responses.request
            self.assertEqual(request["model"], "gpt-5.6-sol")
            self.assertEqual(request["reasoning"], {"effort": "high"})
            content = request["input"][0]["content"]
            expected_url = observation_rgb_data_url(observation, max_width=64)
            self.assertEqual(content[2]["image_url"], expected_url)
            self.assertEqual(content[2]["detail"], "high")
            self.assertEqual(
                content[3]["text"],
                "PASSIVE-VIDEO REFERENCE FRAME at 00:03:",
            )
            self.assertTrue(
                content[4]["image_url"].endswith("cGFzc2l2ZS1mcmFtZQ==")
            )


if __name__ == "__main__":
    unittest.main()
