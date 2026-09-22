from __future__ import annotations

import contextlib
import copy
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from fakes import FakeHardware, make_environment

from yor_agent import launch as launch_module
from yor_agent.primitives.registry import PrimitiveRegistry
from yor_agent.trace import Trace, redact_config

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "navigation.yaml"
TASK_SUITE_PATH = (
    Path(__file__).resolve().parents[1] / "configs" / "passive_video_tasks.yaml"
)
_NAVIGATION_ONLY = [
    "drive_lateral",
    "drive_straight",
    "observe",
    "stop",
    "turn_relative",
]
_PRIOR_KEYS = ("primitive_config", "primitives", "dock_to_visible_object", "settings")


def _docking_config(readiness_prior: dict | None = None, **overrides) -> dict:
    """A minimal task config whose docking primitive has Nav2 settings."""

    settings: dict = {"backend": "nav2"}
    if readiness_prior is not None:
        settings["readiness_prior"] = readiness_prior
    return {
        "task": {"instruction": "Dock to the can."},
        "model": {"provider": "vertex"},
        "navigation_planner": {"enabled": False},
        "primitive_config": {
            "version": 1,
            "primitives": {
                "dock_to_visible_object": {"defaults": {}, "settings": settings}
            },
        },
        **overrides,
    }


def _prior_block(config: dict) -> dict:
    node = config
    for key in _PRIOR_KEYS:
        node = node[key]
    return node["readiness_prior"]


def _write_events_json(path: Path) -> None:
    """One manipulation event in the locked ``yor-manipulation-events-v1`` shape."""

    frames = {
        name: {
            "path": f"{path.stem}_frames/event00_{name}.jpg",
            "time_s": time_s,
            "width": 1280,
            "height": 960,
            "scale": 0.5,
        }
        for name, time_s in (("nav", 14.5), ("ready", 16.5), ("grasp", 19.0))
    }
    path.write_text(
        json.dumps(
            {
                "schema_version": "yor-manipulation-events-v1",
                "created_at": "2026-09-11T00:00:00Z",
                "artifact_path": str(path),
                "source_video": {
                    "path": "video.mp4",
                    "size_bytes": 1,
                    "mtime_ns": 1,
                    "width": 2560,
                    "height": 1920,
                    "fps": 10.0,
                    "duration_s": 43.8,
                },
                "intrinsics": None,
                "motion": {
                    "threshold": 1.0,
                    "median_yavg": 1.4,
                    "per_second_yavg": [1.4],
                    "windows": [],
                },
                "labeling": {
                    "mode": "manual",
                    "model": None,
                    "prompt_sha256": None,
                    "raw_response": None,
                },
                "events": [
                    {
                        "object": "can",
                        "hand": "right",
                        "t_ready_s": 16.5,
                        "t_grasp_s": 19.0,
                        "confidence": None,
                        "notes": "",
                        "frames": frames,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def _closure_value(function, name: str):
    """Read one free variable (for example the controller) out of a closure."""

    index = function.__code__.co_freevars.index(name)
    return function.__closure__[index].cell_contents


class LoadConfigTest(unittest.TestCase):
    def test_shipped_navigation_config_loads(self) -> None:
        config = launch_module.load_config(CONFIG_PATH)

        self.assertEqual(config["model"]["provider"], "vertex")
        self.assertEqual(config["robot"]["navigation"]["min_front_clearance_m"], 0.45)
        self.assertIn("orange sofa", config["task"]["instruction"])

    def test_missing_instruction_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.yaml"
            path.write_text("task:\n  id: nothing\n")

            with self.assertRaises(ValueError):
                launch_module.load_config(path)

    def test_task_suite_requires_and_selects_one_task(self) -> None:
        with self.assertRaises(ValueError) as caught:
            launch_module.load_config(TASK_SUITE_PATH)
        self.assertIn("--task-id", str(caught.exception))

        config = launch_module.load_config(
            TASK_SUITE_PATH, task_id="find_blue_trash_bin"
        )

        self.assertEqual(config["task"]["id"], "find_blue_trash_bin")
        self.assertEqual(config["task"]["instruction"], "Find the blue trash bin")
        self.assertEqual(
            Path(config["trace"]["output_dir"]).name, "find_blue_trash_bin"
        )
        self.assertNotIn("memory_path", config["navigation_planner"])

    def test_navigation_planner_can_be_explicitly_disabled(self) -> None:
        config = launch_module.load_config(
            TASK_SUITE_PATH, task_id="find_blue_trash_bin"
        )

        config = launch_module.apply_navigation_planner_overrides(
            config,
            enabled=False,
            memory_path=None,
            model=None,
        )

        self.assertFalse(config["navigation_planner"]["enabled"])
        self.assertNotIn("memory_path", config["navigation_planner"])

    def test_navigation_planner_switch_accepts_an_explicit_memory(self) -> None:
        config = launch_module.load_config(
            TASK_SUITE_PATH, task_id="find_blue_trash_bin"
        )

        config = launch_module.apply_navigation_planner_overrides(
            config,
            enabled=True,
            memory_path="./memory.json",
            model="gemini-3.6-flash",
        )

        self.assertTrue(config["navigation_planner"]["enabled"])
        self.assertEqual(
            config["navigation_planner"]["memory_path"],
            str(Path("./memory.json").resolve()),
        )
        self.assertEqual(
            config["navigation_planner"]["model"], "gemini-3.6-flash"
        )

    def test_disabled_planner_rejects_a_memory_argument(self) -> None:
        config = launch_module.load_config(
            TASK_SUITE_PATH, task_id="find_blue_trash_bin"
        )

        with self.assertRaisesRegex(ValueError, "cannot be used"):
            launch_module.apply_navigation_planner_overrides(
                config,
                enabled=False,
                memory_path="./memory.json",
                model=None,
            )

    def test_disabled_planner_rejects_a_model_argument(self) -> None:
        config = launch_module.load_config(
            TASK_SUITE_PATH, task_id="find_blue_trash_bin"
        )

        with self.assertRaisesRegex(ValueError, "cannot be used"):
            launch_module.apply_navigation_planner_overrides(
                config,
                enabled=False,
                memory_path=None,
                model="gemini-3.6-flash",
            )

    def test_relative_readiness_events_path_resolves_against_the_task_yaml(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "task.yaml"
            config_path.write_text(
                yaml.safe_dump(
                    _docking_config(
                        {"enabled": True, "events_path": "./events/can.json"}
                    )
                )
            )

            config = launch_module.load_config(config_path)

            self.assertEqual(
                _prior_block(config)["events_path"],
                str((Path(directory) / "events" / "can.json").resolve()),
            )

    def test_absolute_and_absent_readiness_events_paths_are_left_alone(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "task.yaml"
            config_path.write_text(
                yaml.safe_dump(
                    _docking_config({"enabled": False, "events_path": None})
                )
            )
            config = launch_module.load_config(config_path)
            self.assertIsNone(_prior_block(config)["events_path"])

            absolute = str(Path(directory).resolve() / "abs.json")
            config_path.write_text(
                yaml.safe_dump(
                    _docking_config({"enabled": True, "events_path": absolute})
                )
            )
            config = launch_module.load_config(config_path)
            self.assertEqual(_prior_block(config)["events_path"], absolute)


class ReadinessPriorOverrideTest(unittest.TestCase):
    def test_readiness_events_alone_enables_the_prior_and_resolves_the_path(
        self,
    ) -> None:
        config = _docking_config()
        before = copy.deepcopy(config)

        result = launch_module.apply_readiness_prior_overrides(
            config, enabled=None, events_path="./events.json", source=None
        )

        block = _prior_block(result)
        self.assertTrue(block["enabled"])
        self.assertEqual(block["events_path"], str(Path("./events.json").resolve()))
        self.assertNotIn("source", block)
        # The input config is never mutated.
        self.assertEqual(config, before)

    def test_no_readiness_prior_disables_and_drops_the_events_path(self) -> None:
        config = _docking_config(
            {"enabled": True, "events_path": "/data/events.json", "source": "nav_frame"}
        )

        result = launch_module.apply_readiness_prior_overrides(
            config, enabled=False, events_path=None, source=None
        )

        block = _prior_block(result)
        self.assertFalse(block["enabled"])
        self.assertNotIn("events_path", block)
        # Only the switch and the events file change; the rest of the block
        # (here the frame source) is still the configured value.
        self.assertEqual(block["source"], "nav_frame")

    def test_readiness_prior_without_an_events_path_is_rejected(self) -> None:
        for block in (None, {"enabled": False, "events_path": None}):
            with self.subTest(block=block), self.assertRaisesRegex(
                ValueError, "required"
            ):
                launch_module.apply_readiness_prior_overrides(
                    _docking_config(block), enabled=True, events_path=None, source=None
                )

    def test_readiness_prior_enables_with_an_events_path_from_the_config(
        self,
    ) -> None:
        config = _docking_config({"enabled": False, "events_path": "/data/events.json"})

        result = launch_module.apply_readiness_prior_overrides(
            config, enabled=True, events_path=None, source="nav_frame"
        )

        block = _prior_block(result)
        self.assertTrue(block["enabled"])
        self.assertEqual(block["events_path"], "/data/events.json")
        self.assertEqual(block["source"], "nav_frame")

    def test_disabled_readiness_prior_rejects_an_events_argument(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot be used"):
            launch_module.apply_readiness_prior_overrides(
                _docking_config(), enabled=False, events_path="./e.json", source=None
            )

    def test_disabled_readiness_prior_rejects_a_source_argument(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot be used"):
            launch_module.apply_readiness_prior_overrides(
                _docking_config(), enabled=False, events_path=None, source="nav_frame"
            )

    def test_readiness_prior_source_must_be_a_known_frame(self) -> None:
        with self.assertRaisesRegex(ValueError, "ready_frame, nav_frame"):
            launch_module.apply_readiness_prior_overrides(
                _docking_config(),
                enabled=None,
                events_path="./e.json",
                source="grasp_frame",
            )

    def test_source_alone_on_a_disabled_config_drops_the_events_path(self) -> None:
        # Mirrors --navigation-planner-model alone: the setting is recorded but
        # the prior stays off, so the events file must not linger in the trace.
        config = _docking_config({"enabled": False, "events_path": "/data/e.json"})

        result = launch_module.apply_readiness_prior_overrides(
            config, enabled=None, events_path=None, source="nav_frame"
        )

        block = _prior_block(result)
        self.assertFalse(block["enabled"])
        self.assertEqual(block["source"], "nav_frame")
        self.assertNotIn("events_path", block)

    def test_no_flags_return_the_config_unchanged(self) -> None:
        # A YAML that keeps an events file next to enabled: false is what lets
        # the Web UI checkbox turn the prior on; a flagless CLI run must not
        # strip it.
        config = _docking_config({"enabled": False, "events_path": "/data/e.json"})

        result = launch_module.apply_readiness_prior_overrides(
            config, enabled=None, events_path=None, source=None
        )

        self.assertEqual(result, config)
        self.assertIsNot(result["primitive_config"], config["primitive_config"])

    def test_override_needs_docking_settings(self) -> None:
        config = _docking_config()
        config["primitive_config"]["primitives"]["dock_to_visible_object"] = {
            "defaults": {}
        }

        with self.assertRaisesRegex(ValueError, "no settings to override"):
            launch_module.apply_readiness_prior_overrides(
                config, enabled=None, events_path="./e.json", source=None
            )
        # Without any flag there is nothing to override, so nothing to reject.
        launch_module.apply_readiness_prior_overrides(
            config, enabled=None, events_path=None, source=None
        )

    def test_planner_and_readiness_prior_overrides_are_independent(self) -> None:
        config = _docking_config(
            {"enabled": True, "events_path": "/data/events.json"},
            navigation_planner={
                "enabled": True,
                "memory_path": "/data/memory.json",
                "model": "gemini-3.6-flash",
            },
        )

        planner_off = launch_module.apply_navigation_planner_overrides(
            config, enabled=False, memory_path=None, model=None
        )
        self.assertFalse(planner_off["navigation_planner"]["enabled"])
        self.assertEqual(_prior_block(planner_off), _prior_block(config))

        prior_off = launch_module.apply_readiness_prior_overrides(
            config, enabled=False, events_path=None, source=None
        )
        self.assertFalse(_prior_block(prior_off)["enabled"])
        self.assertEqual(prior_off["navigation_planner"], config["navigation_planner"])

    def test_main_applies_the_readiness_prior_flags(self) -> None:
        captured: dict = {}

        def fake_launch(config, **_):
            captured["config"] = config
            return {"status": "finished", "reason": "test"}

        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "task.yaml"
            config_path.write_text(yaml.safe_dump(_docking_config()))
            events = Path(directory) / "events.json"
            argv = [
                "--config",
                str(config_path),
                "--readiness-events",
                str(events),
                "--readiness-prior-source",
                "nav_frame",
            ]
            with _patched(launch_module, "launch", fake_launch):
                with contextlib.redirect_stdout(io.StringIO()):
                    code = launch_module.main(argv)

            self.assertEqual(code, 0)
            block = _prior_block(captured["config"])
            self.assertTrue(block["enabled"])
            self.assertEqual(block["events_path"], str(events.resolve()))
            self.assertEqual(block["source"], "nav_frame")
            # The planner switch is untouched by the prior flags.
            self.assertFalse(captured["config"]["navigation_planner"]["enabled"])

            with _patched(launch_module, "launch", fake_launch):
                with contextlib.redirect_stdout(io.StringIO()):
                    launch_module.main(["--config", str(config_path), "--no-readiness-prior"])
            block = _prior_block(captured["config"])
            self.assertFalse(block["enabled"])
            self.assertNotIn("events_path", block)

    def test_main_reports_conflicting_readiness_prior_flags_as_usage_errors(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "task.yaml"
            config_path.write_text(yaml.safe_dump(_docking_config()))
            for extra in (
                ["--no-readiness-prior", "--readiness-events", "e.json"],
                ["--no-readiness-prior", "--readiness-prior-source", "nav_frame"],
                ["--readiness-prior"],
                ["--readiness-prior-source", "grasp_frame"],
            ):
                with self.subTest(extra=extra):
                    stderr = io.StringIO()
                    with contextlib.redirect_stderr(stderr):
                        with self.assertRaises(SystemExit) as caught:
                            launch_module.main(["--config", str(config_path), *extra])
                    self.assertEqual(caught.exception.code, 2)
                    self.assertIn("readiness", stderr.getvalue())


class LaunchWiringTest(unittest.TestCase):
    def test_launch_wires_one_shared_loop_over_fake_hardware(self) -> None:
        from fakes import ScriptedModel

        hardware = FakeHardware()
        environment = make_environment(hardware)
        self.addCleanup(environment.safe_shutdown)
        with tempfile.TemporaryDirectory() as directory:
            config = {
                "task": {"instruction": "Look around."},
                "model": {"provider": "vertex"},
                "agent": {"max_turns": 3},
                "trace": {"output_dir": directory},
            }
            responses = ['```python\nobserve()\nfinish(reason="looked")\n```']
            with _patched(launch_module, "build_environment", lambda _: environment):
                with _patched(launch_module, "LLM", lambda _: ScriptedModel(responses)):
                    result = launch_module.launch(config)

            self.assertEqual(result["status"], "finished")
            self.assertEqual(result["reason"], "looked")
            self.assertTrue((Path(directory) / "trace.json").exists())

    def test_the_agent_time_limit_reaches_the_loop(self) -> None:
        environment = make_environment()
        self.addCleanup(environment.safe_shutdown)
        with tempfile.TemporaryDirectory() as directory:
            config = {
                "task": {"instruction": "Look around."},
                "model": {"provider": "vertex"},
                "agent": {"max_turns": 3, "time_limit_s": 30},
                "trace": {"output_dir": directory},
            }

            with _patched(launch_module, "build_environment", lambda _: environment):
                runtime = launch_module.build_runtime(config)

            self.assertEqual(runtime.agent.time_limit_s, 30.0)
            self.assertEqual(runtime.agent.max_turns, 3)

    def test_navigation_primitives_are_registered_without_a_yaml_switch(self) -> None:
        environment = make_environment()
        self.addCleanup(environment.safe_shutdown)
        registry = PrimitiveRegistry()

        launch_module.register_navigation_primitives(registry, environment)

        self.assertEqual(len(registry), 5)

    def test_any_configured_primitive_can_be_hidden_from_policy_scope(self) -> None:
        environment = make_environment()
        self.addCleanup(environment.safe_shutdown)
        config = {
            "task": {"instruction": "Look around."},
            "model": {"provider": "vertex"},
            "primitive_config": {
                "version": 1,
                "primitives": {
                    "observe": {"defaults": {}, "exposed": False},
                },
            },
        }

        with _patched(launch_module, "build_environment", lambda _: environment):
            runtime = launch_module.build_runtime(config)

        self.assertNotIn("observe", runtime.registry.names())
        self.assertIn("stop", runtime.registry.names())
        self.assertNotIn("def observe", runtime.executor.documentation())
        execution = runtime.executor.execute("observe()")
        self.assertEqual(execution["error"]["type"], "NameError")

    def test_manipulation_primitives_register_only_when_arms_are_reachable(self) -> None:
        config = {
            "task": {"instruction": "Look around."},
            "model": {"provider": "vertex"},
        }

        without_arms = make_environment(FakeHardware(arms=None))
        self.addCleanup(without_arms.safe_shutdown)
        with _patched(launch_module, "build_environment", lambda _: without_arms):
            runtime = launch_module.build_runtime(config)
        self.assertEqual(sorted(runtime.registry.names()), _NAVIGATION_ONLY)

        with_arms = make_environment(
            FakeHardware(
                arms={
                    "left": {},
                    "right": {},
                    "estop_latched": False,
                    "end_effector_frame": "tcp",
                }
            )
        )
        self.addCleanup(with_arms.safe_shutdown)
        with _patched(launch_module, "build_environment", lambda _: with_arms):
            runtime = launch_module.build_runtime(config)
        self.assertTrue(
            set(_NAVIGATION_ONLY).issubset(set(runtime.registry.names()))
        )
        self.assertIn("goto_pose", runtime.registry.names())
        self.assertIn("goto_grasp_pose", runtime.registry.names())
        self.assertIn("sample_grasp_pose", runtime.registry.names())

    def test_visible_object_docking_registers_only_with_camera_calibration(self) -> None:
        from fakes import FAKE_MANIPULATION_CONFIG

        config = {
            "task": {"instruction": "Look around."},
            "model": {"provider": "vertex"},
        }

        without_calibration = make_environment(FakeHardware(arms=None))
        self.addCleanup(without_calibration.safe_shutdown)
        with _patched(launch_module, "build_environment", lambda _: without_calibration):
            runtime = launch_module.build_runtime(config)
        self.assertNotIn("dock_to_visible_object", runtime.registry.names())

        with_calibration = make_environment(
            FakeHardware(arms=None), manipulation=FAKE_MANIPULATION_CONFIG
        )
        self.addCleanup(with_calibration.safe_shutdown)
        with _patched(launch_module, "build_environment", lambda _: with_calibration):
            runtime = launch_module.build_runtime(config)
        self.assertIn("dock_to_visible_object", runtime.registry.names())
        # Camera calibration alone must not also register arm primitives.
        self.assertNotIn("goto_pose", runtime.registry.names())
        self.assertNotIn("goto_grasp_pose", runtime.registry.names())

    def test_readiness_prior_is_not_built_or_imported_while_disabled(self) -> None:
        # ``None`` in sys.modules makes any import of the prior module raise,
        # so this proves the disabled path never touches it (nor cv2).
        with patch.dict(sys.modules, {"yor_agent.robot.readiness_prior": None}):
            self.assertIsNone(launch_module._build_readiness_prior(None))
            self.assertIsNone(launch_module._build_readiness_prior({"backend": "nav2"}))
            self.assertIsNone(
                launch_module._build_readiness_prior(
                    {"readiness_prior": {"enabled": False, "events_path": "/e.json"}}
                )
            )
            with self.assertRaises(ImportError):
                launch_module._build_readiness_prior(
                    {"readiness_prior": {"enabled": True, "events_path": "/e.json"}}
                )

    def test_build_runtime_hands_the_readiness_prior_to_the_docking_controller(
        self,
    ) -> None:
        from fakes import FAKE_MANIPULATION_CONFIG

        environment = make_environment(
            FakeHardware(arms=None), manipulation=FAKE_MANIPULATION_CONFIG
        )
        self.addCleanup(environment.safe_shutdown)
        with tempfile.TemporaryDirectory() as directory:
            events = Path(directory) / "manipulation_events_test.json"
            _write_events_json(events)
            config = _docking_config(
                {"enabled": True, "events_path": str(events), "source": "nav_frame"}
            )
            config["trace"] = {"output_dir": directory}

            with _patched(launch_module, "build_environment", lambda _: environment):
                runtime = launch_module.build_runtime(config)

            dock = runtime.registry.functions()["dock_to_visible_object"]
            controller = _closure_value(dock, "controller")
            prior = controller._readiness_prior
            self.assertIsNotNone(prior)
            self.assertTrue(prior.config.enabled)
            self.assertEqual(prior.config.events_path, str(events))
            self.assertEqual(prior.config.source, "nav_frame")
            self.assertEqual(len(prior.events), 1)

            disabled = _docking_config({"enabled": False, "events_path": str(events)})
            disabled["trace"] = {"output_dir": directory}
            with _patched(launch_module, "build_environment", lambda _: environment):
                runtime = launch_module.build_runtime(disabled)
            dock = runtime.registry.functions()["dock_to_visible_object"]
            self.assertIsNone(_closure_value(dock, "controller")._readiness_prior)

    def test_build_runtime_fails_at_launch_when_the_events_file_is_missing(
        self,
    ) -> None:
        from fakes import FAKE_MANIPULATION_CONFIG

        environment = make_environment(
            FakeHardware(arms=None), manipulation=FAKE_MANIPULATION_CONFIG
        )
        self.addCleanup(environment.safe_shutdown)
        with tempfile.TemporaryDirectory() as directory:
            config = _docking_config(
                {"enabled": True, "events_path": str(Path(directory) / "missing.json")}
            )
            config["trace"] = {"output_dir": directory}

            with _patched(launch_module, "build_environment", lambda _: environment):
                with self.assertRaises(FileNotFoundError):
                    launch_module.build_runtime(config)


class TraceRedactionTest(unittest.TestCase):
    def test_credentials_never_reach_the_trace(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            trace = Trace(output_dir=directory)
            trace.start("task", {"model": {"api_key": "sk-secret", "name": "gemini"}})

            self.assertEqual(trace.data["config"]["model"]["api_key"], "<redacted>")
            self.assertEqual(trace.data["config"]["model"]["name"], "gemini")

    def test_redaction_is_recursive(self) -> None:
        redacted = redact_config({"a": {"b": {"token": "t", "port": 1}}})

        self.assertEqual(redacted, {"a": {"b": {"token": "<redacted>", "port": 1}}})


class _patched:
    """Minimal attribute patcher so the tests need no external dependencies."""

    def __init__(self, target, name, value) -> None:
        self.target, self.name, self.value = target, name, value

    def __enter__(self):
        self.original = getattr(self.target, self.name)
        setattr(self.target, self.name, self.value)
        return self.value

    def __exit__(self, *exc_info) -> None:
        setattr(self.target, self.name, self.original)


class ExperimentOptionTest(unittest.TestCase):
    COMPARISON_PATH = Path(__file__).resolve().parents[1] / "configs" / "nav_comparison.yaml"

    def test_comparison_config_uses_gpt_and_the_baseline_time_limit(self) -> None:
        config = launch_module.load_config(self.COMPARISON_PATH, task_id="find_blue_trash_bin")
        self.assertEqual(
            (config["model"]["provider"], config["model"]["name"]), ("openai", "gpt-5.6-sol")
        )
        self.assertEqual(config["navigation_planner"]["model"], "gpt-5.6-sol")
        self.assertEqual(config["agent"]["time_limit_s"], 900)

    def test_passive_video_tasks_stop_after_ten_minutes(self) -> None:
        config = launch_module.load_config(
            self.COMPARISON_PATH.with_name("passive_video_tasks.yaml"), task_id="grasp_can"
        )
        self.assertEqual(config["agent"]["time_limit_s"], 600)
        # The limit is merged into the inherited agent block.
        self.assertEqual(config["agent"]["max_turns"], 100)

    def _web_ui_config(self, *arguments: str) -> dict:
        try:
            import yor_agent.web.server  # noqa: F401
        except ModuleNotFoundError as exc:
            self.skipTest(f"web dependencies unavailable: {exc}")
        captured: dict = {}

        def fake_run_web_ui(*, config_path, config, host, port):
            captured["config"] = config

        with patch("yor_agent.web.server.run_web_ui", fake_run_web_ui):
            launch_module.main(
                [
                    "--config",
                    str(self.COMPARISON_PATH),
                    "--task-id",
                    "find_blue_trash_bin",
                    *arguments,
                    "--web-ui",
                ]
            )
        return captured["config"]

    def test_experiment_condition_follows_the_startup_planner(self) -> None:
        ours = self._web_ui_config(
            "--navigation-memory",
            "/tmp/memory.json",
            "--experiment",
            "nav_main",
            "--start-label",
            "hallway",
        )
        self.assertEqual(
            ours["experiment"],
            {
                "name": "nav_main",
                "start_label": "hallway",
                "condition": "ours",
                "output_root": "outputs/yor/ours",
            },
        )
        no_prior = self._web_ui_config("--no-navigation-planner", "--experiment", "nav_main")
        self.assertEqual(no_prior["experiment"]["condition"], "no_prior")
        self.assertEqual(no_prior["experiment"]["start_label"], "kitchen")

    def test_experiment_needs_the_web_ui_and_a_valid_name(self) -> None:
        for arguments in (
            ["--no-navigation-planner", "--experiment", "nav_main"],
            ["--no-navigation-planner", "--start-label", "kitchen"],
            ["--no-navigation-planner", "--experiment", "bad name", "--web-ui"],
        ):
            with self.subTest(arguments=arguments), contextlib.redirect_stderr(
                io.StringIO()
            ), self.assertRaises(SystemExit):
                launch_module.main(
                    [
                        "--config",
                        str(self.COMPARISON_PATH),
                        "--task-id",
                        "find_blue_trash_bin",
                        *arguments,
                    ]
                )


class CoarseNavigationBaselineTest(unittest.TestCase):
    CONFIGS = Path(__file__).resolve().parents[1] / "configs"
    BASELINE_PATH = CONFIGS / "capx_coarse_navigation.yaml"
    COMPARISON_PATH = CONFIGS / "nav_comparison.yaml"

    def test_the_baseline_swaps_yor_navigation_for_the_coarse_vocabulary(self) -> None:
        from yor_agent.experiment_episodes import planner_condition
        from yor_agent.primitive_config import primitive_exposed, primitive_settings
        from yor_agent.primitives.coarse_navigation import COARSE_NAVIGATION_PRIMITIVES

        config = launch_module.load_config(self.BASELINE_PATH, task_id="find_can_and_give_back")
        comparison = launch_module.load_config(
            self.COMPARISON_PATH, task_id="find_can_and_give_back"
        )
        primitives = config["primitive_config"]

        for name in (
            "dock_to_visible_object",
            "prepare_for_manipulation",
            "turn_relative",
            "drive_straight",
            "drive_lateral",
        ):
            self.assertFalse(primitive_exposed(primitives, name), name)
        for name in (
            *COARSE_NAVIGATION_PRIMITIVES,
            "observe",
            "get_object_pose",
            "sample_grasp_pose",
            "goto_pose",
            "goto_grasp_pose",
            "open_gripper",
            "close_gripper",
        ):
            self.assertTrue(primitive_exposed(primitives, name), name)
        for name in ("get_object_pose", "sample_grasp_pose", "open_gripper", "close_gripper"):
            self.assertEqual(
                primitive_settings(primitives, name),
                primitive_settings(comparison["primitive_config"], name),
            )
        self.assertNotIn("primitive_overrides", config)
        self.assertFalse(config["navigation_planner"]["enabled"])
        self.assertEqual(config["model"]["system_prompt"], "capx_coarse_navigation")
        self.assertEqual(
            (config["model"]["provider"], config["model"]["name"], config["agent"]["time_limit_s"]),
            ("openai", "gpt-5.6-sol", 900),
        )
        self.assertEqual(config["task"]["instruction"], comparison["task"]["instruction"])
        self.assertTrue(
            config["trace"]["output_dir"].endswith(
                "outputs/capx_coarse_navigation/find_can_and_give_back"
            )
        )
        self.assertEqual(planner_condition(config), "capx_coarse_navigation")
        self.assertEqual(planner_condition(comparison), "ours")

    def test_only_the_baseline_runtime_lists_the_coarse_vocabulary(self) -> None:
        from yor_agent.primitives.coarse_navigation import COARSE_NAVIGATION_PRIMITIVES

        for path, baseline in ((self.BASELINE_PATH, True), (self.COMPARISON_PATH, False)):
            with self.subTest(config=path.name), tempfile.TemporaryDirectory() as directory:
                config = launch_module.load_config(path, task_id="find_can_and_give_back")
                config["trace"] = {"output_dir": directory}
                config["navigation_planner"] = {"enabled": False}
                environment = make_environment()
                self.addCleanup(environment.safe_shutdown)

                with _patched(
                    launch_module, "build_environment", lambda _config, env=environment: env
                ):
                    runtime = launch_module.build_runtime(config)

                names = set(runtime.registry.names())
                self.assertIn("observe", names)
                self.assertEqual(set(COARSE_NAVIGATION_PRIMITIVES) <= names, baseline)
                self.assertEqual("drive_straight" in names, not baseline)
                self.assertEqual(
                    runtime.agent.model.system_prompt_name,
                    "capx_coarse_navigation" if baseline else "default",
                )


if __name__ == "__main__":
    unittest.main()
