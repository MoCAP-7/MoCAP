from __future__ import annotations

import json
import tempfile
import threading
import time
import unittest
from pathlib import Path

from fakes import FakeHardware, RecordingEnvironment, ScriptedModel, make_environment

from yor_agent.agents.default import DefaultAgent
from yor_agent.exceptions import ModelError
from yor_agent.executor import PolicyExecutor
from yor_agent.primitives.navigation import register_navigation_primitives
from yor_agent.primitives.registry import PrimitiveRegistry
from yor_agent.trace import Trace


class AgentLoopTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.output_dir = Path(self._tmp.name)

    def build(
        self,
        responses: list[str],
        *,
        hardware=None,
        max_turns: int = 5,
        primitives: dict | None = None,
        **agent_options,
    ):
        hardware = hardware or FakeHardware()
        environment = make_environment(hardware)
        self.addCleanup(environment.safe_shutdown)
        registry = PrimitiveRegistry()
        register_navigation_primitives(registry, environment)
        for name, function in (primitives or {}).items():
            registry.register(name, function)
        trace = Trace(output_dir=self.output_dir)
        executor = PolicyExecutor(
            registry, on_primitive_call=trace.record_primitive_call
        )
        model = ScriptedModel(responses)
        agent = DefaultAgent(
            model, environment, executor, trace, max_turns=max_turns, **agent_options
        )
        return agent, model, trace, hardware, environment

    def read_trace(self) -> dict:
        return json.loads((self.output_dir / "trace.json").read_text())

    def test_fenced_policy_is_executed_and_fed_back(self) -> None:
        agent, model, _, hardware, _ = self.build(
            [
                "Here is my plan.\n```python\n"
                "obs = observe()\n"
                "print('pose', obs['base']['pose_xy_yaw'])\n"
                "turn_relative(20.0, timeout_s=8.0)\n"
                "```",
                '```python\nfinish(reason="navigation attempt complete")\n```',
            ]
        )

        result = agent.run("Go to the orange sofa.")

        self.assertEqual(result["status"], "finished")
        self.assertEqual(result["reason"], "navigation attempt complete")
        # The second prompt must contain the executed code, its stdout, and a
        # fresh observation.
        second_prompt = model.prompts[1]
        feedback = second_prompt[-1]["content"][0]["text"]
        self.assertIn("turn_relative(20.0, timeout_s=8.0)", feedback)
        self.assertIn("pose", feedback)
        self.assertIn("Current observation:", feedback)
        self.assertIn("pose_xy_yaw", feedback)
        # An image part accompanies every observation message.
        self.assertEqual(second_prompt[-1]["content"][1]["type"], "image_url")
        self.assertTrue(any(command[2] > 0 for command in hardware.commands))

    def test_trace_records_turns_policies_and_primitive_calls(self) -> None:
        agent, _, _, _, _ = self.build(
            [
                "```python\nobserve()\nstop()\n```",
                '```python\nfinish(reason="done looking")\n```',
            ]
        )

        agent.run("Look around.")
        trace = self.read_trace()

        self.assertEqual(trace["task"], "Look around.")
        self.assertEqual(trace["finish"]["reason"], "done looking")
        self.assertIsNone(trace["error"])
        self.assertEqual(len(trace["policies"]), 2)
        self.assertEqual(
            [call["name"] for call in trace["primitive_calls"]], ["observe", "stop"]
        )
        self.assertEqual(trace["primitive_calls"][0]["turn"], 1)
        # Images are stored as artifacts, never inline base64.
        image_parts = [
            part
            for message in trace["messages"]
            if isinstance(message["content"], list)
            for part in message["content"]
            if isinstance(part, dict) and part.get("type") == "image_ref"
        ]
        self.assertTrue(image_parts)
        for part in image_parts:
            self.assertTrue((self.output_dir / part["path"]).exists())
        self.assertNotIn("base64", json.dumps(trace))

    def test_format_error_is_fed_back_without_touching_the_robot(self) -> None:
        agent, model, _, hardware, _ = self.build(
            [
                "I would drive forward now, but let me think about it first.",
                '```python\nfinish(reason="gave up")\n```',
            ]
        )

        result = agent.run("Go somewhere.")

        self.assertEqual(result["status"], "finished")
        self.assertEqual(result["turns"], 2)
        retry_prompt = model.prompts[1][-1]["content"]
        self.assertIn("could not be read as one Python program", retry_prompt)
        # Only the shutdown stop was ever commanded; no motion was attempted.
        self.assertTrue(all(command == [0.0, 0.0, 0.0] for command in hardware.commands))
        # Turn 1 produced no policy at all; only turn 2's finish was executed.
        policies = self.read_trace()["policies"]
        self.assertEqual([policy["turn"] for policy in policies], [2])

    def test_primitive_failure_reaches_the_next_turn_with_a_fresh_observation(
        self,
    ) -> None:
        agent, model, _, hardware, _ = self.build(
            [
                "```python\ndrive_straight(0.5, timeout_s=3.0)\nturn_relative(90.0)\n```",
                '```python\nfinish(reason="blocked, stopping")\n```',
            ],
            hardware=FakeHardware(clearance_m=0.30),
        )

        agent.run("Drive into the wall.")

        feedback = model.prompts[1][-1]["content"][0]["text"]
        self.assertIn("A primitive failed", feedback)
        self.assertIn("obstacle_too_close", feedback)
        self.assertIn("Current observation:", feedback)
        self.assertTrue(all(command == [0.0, 0.0, 0.0] for command in hardware.commands))
        trace = self.read_trace()
        self.assertEqual(
            trace["policies"][0]["execution"]["interrupted_by"]["primitive"],
            "drive_straight",
        )

    def test_runtime_error_is_feedback_and_never_a_success_claim(self) -> None:
        agent, model, _, _, _ = self.build(
            [
                "```python\ndrive_straight(0.1, timeout_s=8.0)\nboom()\n```",
                '```python\nfinish(reason="recovered")\n```',
            ]
        )

        result = agent.run("Break something.")
        trace = self.read_trace()

        self.assertEqual(result["status"], "finished")
        self.assertIn("raised an exception", model.prompts[1][-1]["content"][0]["text"])
        self.assertEqual(trace["policies"][0]["execution"]["error"]["type"], "NameError")
        self.assertNotIn("success", trace["finish"])

    def test_finish_runs_safe_shutdown_and_persists_the_trace(self) -> None:
        agent, _, _, hardware, environment = self.build(
            ['```python\nfinish(reason="nothing to do")\n```']
        )

        agent.run("Stand still.")

        self.assertTrue(hardware.closed)
        self.assertTrue(environment._closed)
        self.assertEqual(hardware.commands[-1], [0.0, 0.0, 0.0])
        self.assertEqual(self.read_trace()["finish"]["reason"], "nothing to do")

    def test_model_error_still_shuts_down_and_records_the_failure(self) -> None:
        agent, model, _, hardware, _ = self.build(["```python\nobserve()\n```"])
        model.error = ModelError("vertex/gemini-2.5-pro query failed: boom")

        with self.assertRaises(ModelError):
            agent.run("Go somewhere.")

        self.assertTrue(hardware.closed)
        trace = self.read_trace()
        self.assertEqual(trace["error"]["type"], "ModelError")
        self.assertIsNone(trace["finish"])

    def test_keyboard_interrupt_still_shuts_down(self) -> None:
        agent, model, _, hardware, _ = self.build(["```python\nobserve()\n```"])
        model.error = KeyboardInterrupt()

        with self.assertRaises(KeyboardInterrupt):
            agent.run("Go somewhere.")

        self.assertTrue(hardware.closed)
        self.assertEqual(self.read_trace()["error"]["type"], "KeyboardInterrupt")

    def test_interrupt_during_a_policy_keeps_the_partial_record(self) -> None:
        hardware = FakeHardware()
        agent, _, _, _, _ = self.build(
            ["```python\nobserve()\nturn_relative(45.0, timeout_s=8.0)\n```"],
            hardware=hardware,
        )
        interrupted = False
        real_sleep = hardware.sleep

        def interrupt_once(duration: float) -> None:
            nonlocal interrupted
            if not interrupted:
                interrupted = True
                raise KeyboardInterrupt
            real_sleep(duration)

        agent.environment.controller._sleep = interrupt_once

        with self.assertRaises(KeyboardInterrupt):
            agent.run("Turn a bit.")

        trace = self.read_trace()
        self.assertEqual(trace["error"]["type"], "KeyboardInterrupt")
        self.assertEqual(len(trace["policies"]), 1)
        self.assertIn("turn_relative", trace["policies"][0]["code"])
        self.assertEqual(
            [call["name"] for call in trace["primitive_calls"]],
            ["observe", "turn_relative"],
        )
        self.assertEqual(hardware.commands[-1], [0.0, 0.0, 0.0])
        self.assertTrue(hardware.closed)

    def test_turn_budget_stops_the_loop_without_claiming_success(self) -> None:
        agent, _, _, _, _ = self.build(
            ["```python\nobserve()\n```"] * 6, max_turns=2
        )

        result = agent.run("Loop forever.")

        self.assertEqual(result["status"], "max_turns")
        self.assertEqual(result["turns"], 2)
        self.assertEqual(len(self.read_trace()["policies"]), 2)

    def test_reset_failure_shuts_down_before_any_model_call(self) -> None:
        environment = RecordingEnvironment(
            reset_error=RuntimeError("YOR emergency stop is latched")
        )
        trace = Trace(output_dir=self.output_dir)
        model = ScriptedModel([])
        agent = DefaultAgent(
            model, environment, PolicyExecutor(PrimitiveRegistry()), trace
        )

        with self.assertRaises(RuntimeError):
            agent.run("Go somewhere.")

        self.assertEqual(environment.shutdowns, 1)
        self.assertEqual(model.prompts, [])
        self.assertEqual(self.read_trace()["error"]["type"], "RuntimeError")

    def test_events_include_exact_model_image_code_and_execution(self) -> None:
        events: list[dict] = []
        hardware = FakeHardware()
        environment = make_environment(hardware)
        self.addCleanup(environment.safe_shutdown)
        registry = PrimitiveRegistry()
        register_navigation_primitives(registry, environment)
        trace = Trace(output_dir=self.output_dir)
        model = ScriptedModel(['```python\nprint("hello")\nfinish(reason="done")\n```'])
        agent = DefaultAgent(
            model,
            environment,
            PolicyExecutor(registry, on_primitive_call=trace.record_primitive_call),
            trace,
            on_event=events.append,
        )

        result = agent.run("Look around.")

        self.assertEqual(result["status"], "finished")
        model_input = next(event for event in events if event["type"] == "model_input")
        exact_prompt_url = model.prompts[0][-1]["content"][1]["image_url"]["url"]
        self.assertEqual(model_input["image_url"], exact_prompt_url)
        response = next(event for event in events if event["type"] == "model_response")
        self.assertIn('print("hello")', response["code"])
        execution = next(
            event for event in events if event["type"] == "policy_execution_finished"
        )
        self.assertEqual(execution["execution"]["finish_reason"], "done")
        self.assertIn("hello", execution["execution"]["stdout"])

    def test_one_shot_navigation_advice_enriches_only_initial_task(self) -> None:
        class Planner:
            model = "gemini-3.7-flash"
            memory_path = "/tmp/generic-memory.json"

            def __init__(self) -> None:
                self.calls = []

            def plan(self, **kwargs):
                self.calls.append(kwargs)
                return "Use the central landmark to orient, then look for the target."

        planner = Planner()
        hardware = FakeHardware()
        environment = make_environment(hardware)
        self.addCleanup(environment.safe_shutdown)
        registry = PrimitiveRegistry()
        register_navigation_primitives(registry, environment)
        trace = Trace(output_dir=self.output_dir)
        model = ScriptedModel(['```python\nfinish(reason="done")\n```'])
        agent = DefaultAgent(
            model,
            environment,
            PolicyExecutor(registry),
            trace,
            navigation_planner=planner,
        )

        agent.run("Find an arbitrary object.")

        self.assertEqual(len(planner.calls), 1)
        initial_prompt = model.prompts[0][-1]["content"][0]["text"]
        self.assertIn("Operator task: Find an arbitrary object.", initial_prompt)
        self.assertIn("Use the central landmark", initial_prompt)
        persisted = self.read_trace()["navigation_plan"]
        self.assertEqual(persisted["model"], "gemini-3.7-flash")
        self.assertIn("central landmark", persisted["advice"])
        raw = json.loads((self.output_dir / "trace.json").read_text())
        self.assertEqual(next(iter(raw)), "navigation_plan")

    def test_pre_requested_operator_stop_is_status_neutral(self) -> None:
        stop_event = threading.Event()
        stop_event.set()
        environment = RecordingEnvironment()
        trace = Trace(output_dir=self.output_dir)
        model = ScriptedModel([])
        agent = DefaultAgent(
            model,
            environment,
            PolicyExecutor(PrimitiveRegistry()),
            trace,
            stop_event=stop_event,
        )

        result = agent.run("Do not move.")

        self.assertEqual(result["status"], "stopped")
        self.assertEqual(model.prompts, [])
        persisted = self.read_trace()
        self.assertEqual(persisted["stop"]["reason"], "operator requested stop")
        self.assertIsNone(persisted["finish"])

    def test_the_time_limit_stops_a_running_policy_as_the_operator_stop_does(self) -> None:
        stop_event = threading.Event()

        def wait_for_stop() -> dict:
            """Block until the run is asked to stop."""

            return {"success": True, "stopped": stop_event.wait(5.0)}

        agent, _, _, _, _ = self.build(
            ["```python\nprint(wait_for_stop())\n```"],
            primitives={"wait_for_stop": wait_for_stop},
            time_limit_s=0.05,
            stop_event=stop_event,
        )

        result = agent.run("Wait for the time limit.")

        self.assertEqual(result["status"], "stopped")
        self.assertEqual(result["reason"], "time limit of 0.05 s reached")
        persisted = self.read_trace()
        self.assertEqual(persisted["stop"]["reason"], "time limit of 0.05 s reached")
        self.assertIsNone(persisted["finish"])
        # The program the limit interrupted is still recorded.
        self.assertEqual(len(persisted["policies"]), 1)
        self.assertIn("'stopped': True", persisted["policies"][0]["execution"]["stdout"])

    def test_a_run_that_ends_first_disarms_its_time_limit(self) -> None:
        stop_event = threading.Event()
        agent, _, _, _, _ = self.build(
            ['```python\nfinish(reason="done")\n```'],
            time_limit_s=0.05,
            stop_event=stop_event,
        )

        result = agent.run("Finish at once.")
        time.sleep(0.15)

        self.assertEqual(result["status"], "finished")
        self.assertFalse(stop_event.is_set())
        self.assertIsNone(self.read_trace()["stop"])

    def test_the_trace_records_when_the_agent_loop_started(self) -> None:
        agent, _, _, _, _ = self.build(['```python\nfinish(reason="done")\n```'])
        self.assertIsNone(self.read_trace()["started_at"])

        agent.run("Finish at once.")

        persisted = self.read_trace()
        self.assertIsNotNone(persisted["started_at"])
        self.assertLessEqual(persisted["created_at"], persisted["started_at"])
        self.assertLessEqual(persisted["started_at"], persisted["finish"]["at"])

    def test_the_time_limit_must_be_positive_seconds(self) -> None:
        agent, _, _, _, _ = self.build([])
        self.assertIsNone(agent.time_limit_s)
        for bad in (0, -1.0, float("inf"), float("nan"), True, "120"):
            with self.subTest(limit=bad), self.assertRaises(ValueError):
                self.build([], time_limit_s=bad)


if __name__ == "__main__":
    unittest.main()
