from __future__ import annotations

import unittest

import numpy as np

from yor_agent.exceptions import Finished, PrimitiveFailed
from yor_agent.executor import PolicyExecutor
from yor_agent.primitives.registry import PrimitiveRegistry


class Recorder:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.fail_on: set[str] = set()

    def observe(self) -> dict:
        """Return a fake observation."""

        self.calls.append("observe")
        return {"rgb": np.zeros((4, 4, 3), dtype=np.uint8), "pose": [0.0, 0.0, 0.0]}

    def drive(self, distance_m: float) -> dict:
        """Drive forward."""

        self.calls.append(f"drive({distance_m})")
        if "drive" in self.fail_on:
            raise PrimitiveFailed("drive", {"success": False, "reason": "obstacle"})
        return {"success": True, "reason": "target_reached"}


def build(recorder: Recorder, events: list | None = None) -> PolicyExecutor:
    registry = PrimitiveRegistry()
    registry.register("observe", recorder.observe)
    registry.register("drive", recorder.drive)
    on_call = None if events is None else events.append
    return PolicyExecutor(registry, on_primitive_call=on_call)


class PolicyExecutorTest(unittest.TestCase):
    def test_runs_multi_call_policy_and_records_calls(self) -> None:
        recorder = Recorder()
        events: list = []
        executor = build(recorder, events)

        record = executor.execute(
            "for step in range(2):\n"
            "    observe()\n"
            "    drive(0.1 * (step + 1))\n"
            "print('done')\n"
        )

        self.assertEqual(
            recorder.calls, ["observe", "drive(0.1)", "observe", "drive(0.2)"]
        )
        self.assertEqual(record["stdout"], "done\n")
        self.assertIsNone(record["error"])
        self.assertIsNone(record["interrupted_by"])
        self.assertEqual(len(record["primitive_calls"]), 4)
        self.assertEqual([event["name"] for event in events][:2], ["observe", "drive"])

    def test_large_arrays_are_summarized_not_embedded(self) -> None:
        recorder = Recorder()
        executor = build(recorder)

        record = executor.execute("observe()")

        result = record["primitive_calls"][0]["result"]
        self.assertEqual(result["rgb"], "<array shape=(4, 4, 3) dtype=uint8>")
        self.assertEqual(result["pose"], [0.0, 0.0, 0.0])

    def test_small_arrays_stay_readable_in_the_trace(self) -> None:
        registry = PrimitiveRegistry()
        registry.register("pose", lambda: np.array([1.5, -2.0, 0.25]))
        executor = PolicyExecutor(registry)

        record = executor.execute("pose()")

        self.assertEqual(record["primitive_calls"][0]["result"], [1.5, -2.0, 0.25])

    def test_nested_planner_diagnostics_remain_structured_in_trace(self) -> None:
        registry = PrimitiveRegistry()

        def reject() -> None:
            raise PrimitiveFailed(
                "prepare_for_manipulation",
                {
                    "success": False,
                    "reason": "grasp_goal_ik_failed",
                    "diagnostics": {
                        "attempts": [
                            {
                                "ik_diagnostics": {
                                    "success_by_returned_seed": [False, False],
                                    "position_error_m_by_returned_seed": [
                                        0.031,
                                        0.044,
                                    ],
                                }
                            }
                        ]
                    },
                },
            )

        registry.register("reject", reject)
        record = PolicyExecutor(registry).execute("reject()")

        diagnostics = record["primitive_calls"][0]["result"]["diagnostics"]
        ik_diagnostics = diagnostics["attempts"][0]["ik_diagnostics"]
        self.assertEqual(
            ik_diagnostics["position_error_m_by_returned_seed"],
            [0.031, 0.044],
        )

    def test_finish_propagates_and_carries_its_execution_record(self) -> None:
        executor = build(Recorder())

        with self.assertRaises(Finished) as caught:
            executor.execute("print('bye')\nfinish(reason='operator asked')")

        self.assertEqual(caught.exception.reason, "operator asked")
        self.assertIsNotNone(caught.exception.execution)
        self.assertEqual(caught.exception.execution["stdout"], "bye\n")
        self.assertEqual(caught.exception.execution["finish_reason"], "operator asked")

    def test_syntax_error_becomes_feedback(self) -> None:
        executor = build(Recorder())

        record = executor.execute("drive(0.1")

        self.assertEqual(record["error"]["type"], "SyntaxError")
        self.assertIsNone(record["interrupted_by"])
        self.assertEqual(record["primitive_calls"], [])

    def test_runtime_error_becomes_feedback_and_skips_the_rest(self) -> None:
        recorder = Recorder()
        executor = build(recorder)

        record = executor.execute("drive(0.1)\nraise ValueError('bad plan')\ndrive(0.2)")

        self.assertEqual(record["error"]["type"], "ValueError")
        self.assertIn("bad plan", record["error"]["message"])
        self.assertIn("<policy>", record["error"]["traceback"])
        self.assertEqual(recorder.calls, ["drive(0.1)"])

    def test_primitive_failure_interrupts_later_motion(self) -> None:
        recorder = Recorder()
        recorder.fail_on.add("drive")
        executor = build(recorder)

        record = executor.execute("drive(0.5)\ndrive(0.5)\nobserve()")

        self.assertEqual(record["interrupted_by"]["primitive"], "drive")
        self.assertEqual(record["interrupted_by"]["reason"], "obstacle")
        self.assertIsNone(record["error"])
        self.assertEqual(recorder.calls, ["drive(0.5)"])

    def test_namespace_is_capability_limited(self) -> None:
        executor = build(Recorder())

        for snippet, expected in [
            ("import os", "ImportError"),
            ("open('/etc/passwd')", "NameError"),
            ("env", "NameError"),
            ("submit_base_velocity([1, 0, 0])", "NameError"),
            ("eval('1+1')", "NameError"),
            ("__import__('os')", "NameError"),
            ("globals()", "NameError"),
        ]:
            with self.subTest(snippet=snippet):
                record = executor.execute(snippet)
                self.assertEqual(record["error"]["type"], expected)

    def test_numpy_is_bound_without_reopening_imports(self) -> None:
        """Primitive signatures show ``np.ndarray``, so ``np`` must resolve."""

        executor = build(Recorder())

        record = executor.execute("print(np.asarray([1.0, 2.0]).sum())")

        self.assertIsNone(record["error"], record)
        self.assertEqual(record["stdout"].strip(), "3.0")
        # Binding the module must not hand the policy an import hook back.
        blocked = executor.execute("import numpy")
        self.assertEqual(blocked["error"]["type"], "ImportError")

    def test_namespace_is_fresh_every_turn(self) -> None:
        executor = build(Recorder())

        first = executor.execute("carried = 41")
        second = executor.execute("print(carried)")

        self.assertIsNone(first["error"])
        self.assertEqual(second["error"]["type"], "NameError")

    def test_documentation_includes_primitives_and_finish(self) -> None:
        docs = build(Recorder()).documentation()

        self.assertIn("def drive(distance_m: float)", docs)
        self.assertIn("def finish(reason: str)", docs)
        self.assertIn("does not verify", docs)


if __name__ == "__main__":
    unittest.main()
