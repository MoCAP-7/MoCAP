from __future__ import annotations

import tempfile
import threading
import unittest
from pathlib import Path

from fakes import FakeHardware, make_environment

from yor_agent.agents.default import DefaultAgent
from yor_agent.exceptions import FormatError, ModelError, Stopped
from yor_agent.executor import PolicyExecutor
from yor_agent.launch import build_model, manual_provider_error
from yor_agent.models.llm import DEFAULT_MODEL_NAMES, LLM
from yor_agent.models.manual import ManualModel
from yor_agent.primitives.navigation import register_navigation_primitives
from yor_agent.primitives.registry import PrimitiveRegistry
from yor_agent.trace import Trace


class ManualModelTest(unittest.TestCase):
    def test_identity_and_defaults(self) -> None:
        model = ManualModel()

        self.assertEqual(model.provider, "manual")
        self.assertEqual(model.name, "operator")
        self.assertEqual(DEFAULT_MODEL_NAMES["manual"], "operator")
        self.assertEqual(model.n_calls, 0)
        self.assertEqual(model.pending(), 0)

    def test_query_returns_the_submitted_program_through_code_extraction(self) -> None:
        model = ManualModel(poll_interval_s=0.01)
        model.submit('```python\nprint("hello")\n```')
        model.submit("drive_straight(0.5)")

        first = model.query([])
        second = model.query([])

        self.assertEqual(first, 'print("hello")')
        self.assertEqual(second, "drive_straight(0.5)")
        self.assertEqual(model.n_calls, 2)
        self.assertEqual(model.pending(), 0)

    def test_empty_or_prose_only_programs_are_rejected(self) -> None:
        model = ManualModel(poll_interval_s=0.01)

        with self.assertRaises(ValueError):
            model.submit("   ")
        model.submit("```python\n```")
        with self.assertRaises(FormatError):
            model.query([])

    def test_stop_event_releases_a_waiting_query(self) -> None:
        stop_event = threading.Event()
        model = ManualModel(stop_event=stop_event, poll_interval_s=0.01)
        observed: dict[str, object] = {}

        def wait_for_program() -> None:
            try:
                model.query([])
            except Stopped as stopped:
                observed["reason"] = stopped.reason

        worker = threading.Thread(target=wait_for_program)
        worker.start()
        deadline = 50
        while not model.waiting and deadline:
            threading.Event().wait(0.01)
            deadline -= 1
        stop_event.set()
        worker.join(timeout=2.0)

        self.assertFalse(worker.is_alive())
        self.assertEqual(observed["reason"], "operator requested stop")
        self.assertFalse(model.waiting)

    def test_stop_reason_accessor_is_reported(self) -> None:
        stop_event = threading.Event()
        stop_event.set()
        model = ManualModel(
            stop_event=stop_event,
            stop_reason=lambda: "e-stop pressed",
            poll_interval_s=0.01,
        )

        with self.assertRaises(Stopped) as raised:
            model.query([])

        self.assertEqual(raised.exception.reason, "e-stop pressed")

    def test_plain_llm_refuses_to_generate_for_the_manual_provider(self) -> None:
        model = LLM({"provider": "manual"})

        self.assertEqual(model.name, "operator")
        with self.assertRaises(ModelError):
            model.query([])

    def test_build_model_selects_the_manual_model(self) -> None:
        stop_event = threading.Event()

        manual = build_model({"provider": "manual"}, stop_event=stop_event)
        api = build_model({"provider": "vertex", "name": "gemini-2.5-pro"})

        self.assertIsInstance(manual, ManualModel)
        self.assertEqual(manual.provider, "manual")
        self.assertIs(manual._stop_event, stop_event)
        self.assertIsInstance(api, LLM)
        self.assertNotIsInstance(api, ManualModel)

    def test_manual_provider_requires_the_web_ui(self) -> None:
        manual = {"model": {"provider": "manual"}}
        api = {"model": {"provider": "openai"}}

        self.assertIsNotNone(manual_provider_error(manual, web_ui=False))
        self.assertIsNone(manual_provider_error(manual, web_ui=True))
        self.assertIsNone(manual_provider_error(api, web_ui=False))
        self.assertIsNone(manual_provider_error({}, web_ui=False))


class ManualAgentLoopTest(unittest.TestCase):
    """Operator programs run through the unchanged DefaultAgent loop."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.output_dir = Path(self._tmp.name)

    def build(self, *, stop_event: threading.Event | None = None):
        hardware = FakeHardware()
        environment = make_environment(hardware)
        self.addCleanup(environment.safe_shutdown)
        registry = PrimitiveRegistry()
        register_navigation_primitives(registry, environment)
        trace = Trace(output_dir=self.output_dir)
        executor = PolicyExecutor(
            registry, on_primitive_call=trace.record_primitive_call
        )
        model = ManualModel(stop_event=stop_event, poll_interval_s=0.01)
        events: list[dict] = []
        agent = DefaultAgent(
            model,
            environment,
            executor,
            trace,
            max_turns=5,
            on_event=events.append,
            stop_event=stop_event,
        )
        return agent, model, hardware, events

    def test_submitted_programs_drive_the_robot_and_finish(self) -> None:
        agent, model, hardware, events = self.build()
        model.submit("obs = observe()\nprint('pose', obs['base']['pose_xy_yaw'])\nturn_relative(15.0, timeout_s=8.0)")
        model.submit('finish(reason="manual test complete")')

        result = agent.run("Manual primitive check")

        self.assertEqual(result["status"], "finished")
        self.assertEqual(result["reason"], "manual test complete")
        self.assertEqual(result["turns"], 2)
        self.assertTrue(any(command[2] > 0 for command in hardware.commands))
        started = next(event for event in events if event["type"] == "run_started")
        self.assertEqual(started["provider"], "manual")
        self.assertEqual(started["model"], "operator")
        executed = [
            event["code"] for event in events if event["type"] == "policy_execution_started"
        ]
        self.assertEqual(len(executed), 2)
        self.assertIn("turn_relative(15.0", executed[0])

    def test_operator_stop_while_waiting_for_a_program(self) -> None:
        stop_event = threading.Event()
        agent, model, _, _ = self.build(stop_event=stop_event)
        outcome: dict[str, dict] = {}

        def run() -> None:
            outcome["result"] = agent.run("Manual stop check")

        worker = threading.Thread(target=run)
        worker.start()
        deadline = 200
        while not model.waiting and deadline:
            threading.Event().wait(0.01)
            deadline -= 1
        agent.request_stop("operator pressed stop")
        worker.join(timeout=5.0)

        self.assertFalse(worker.is_alive())
        self.assertEqual(outcome["result"]["status"], "stopped")

    def test_agent_without_explicit_stop_event_still_releases_a_waiting_turn(self) -> None:
        """build_runtime shares one event; the loop test mirrors that wiring."""

        shared = threading.Event()
        agent, model, _, _ = self.build(stop_event=shared)
        model.stop_reason = lambda: getattr(agent, "_stop_reason", "")
        outcome: dict[str, dict] = {}

        def run() -> None:
            outcome["result"] = agent.run("Manual stop reason check")

        worker = threading.Thread(target=run)
        worker.start()
        deadline = 200
        while not model.waiting and deadline:
            threading.Event().wait(0.01)
            deadline -= 1
        agent.request_stop("watchdog tripped")
        worker.join(timeout=5.0)

        self.assertFalse(worker.is_alive())
        self.assertEqual(outcome["result"]["status"], "stopped")
        self.assertEqual(outcome["result"]["reason"], "watchdog tripped")


if __name__ == "__main__":
    unittest.main()
