from __future__ import annotations

import asyncio
import base64
import copy
from pathlib import Path
import unittest

import numpy as np

from yor_agent.launch import load_config

try:
    from yor_agent.web.server import (
        READINESS_PRIOR_EVENTS_REQUIRED,
        SERVER_SHUTDOWN_STOP_REASON,
        WebRunController,
        _preview_data_url,
        apply_readiness_prior_toggle,
        create_app,
    )
except ModuleNotFoundError as exc:
    if exc.name not in {"fastapi", "uvicorn"}:
        raise
    WEB_DEPENDENCIES_AVAILABLE = False
    WebRunController = _preview_data_url = create_app = None
    READINESS_PRIOR_EVENTS_REQUIRED = apply_readiness_prior_toggle = None
    SERVER_SHUTDOWN_STOP_REASON = None
else:
    WEB_DEPENDENCIES_AVAILABLE = True


CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "navigation.yaml"

READINESS_PRIOR_BLOCK = {
    "enabled": True,
    "events_path": "/tmp/manipulation_events.json",
    "source": "ready_frame",
    "min_inliers": 30,
}


def _docking_settings(config: dict) -> dict:
    return config["primitive_config"]["primitives"]["dock_to_visible_object"]["settings"]


def _with_readiness_prior(config: dict, block: dict | None) -> dict:
    """Deep-copy ``config`` with ``dock_to_visible_object.settings.readiness_prior`` set."""

    result = copy.deepcopy(config)
    _docking_settings(result)["readiness_prior"] = copy.deepcopy(block)
    return result


@unittest.skipUnless(
    WEB_DEPENDENCIES_AVAILABLE,
    "FastAPI/Uvicorn web dependencies are not installed in this test environment",
)
class WebServerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_config(CONFIG_PATH)

    def test_app_exposes_config_and_packaged_static_assets(self) -> None:
        app = create_app(config_path=CONFIG_PATH, config=self.config)
        routes = {
            route.path: route for route in app.routes if hasattr(route, "endpoint")
        }

        public = asyncio.run(routes["/api/config"].endpoint())
        index = asyncio.run(routes["/"].endpoint())

        self.assertEqual(
            public["task"]["instruction"],
            "Go to the orange sofa until it is clearly in front of you, "
            "instead of still having some distance.",
        )
        static_dir = Path(index.path).parent
        self.assertTrue((static_dir / "index.html").is_file())
        self.assertTrue((static_dir / "styles.css").is_file())
        self.assertTrue((static_dir / "app.js").is_file())
        html = (static_dir / "index.html").read_text(encoding="utf-8")
        javascript = (static_dir / "app.js").read_text(encoding="utf-8")
        self.assertIn(
            'id="start-button" class="button primary" type="button" disabled',
            html,
        )
        self.assertIn('value="qwen::qwen3.7-plus"', html)
        self.assertIn('value="qwen::qwen3.6-plus"', html)
        self.assertIn('value="qwen::qwen3.6-flash"', html)
        self.assertIn(
            'value="deepseek::deepseek-v4-flash-vision-exp"', html
        )
        self.assertIn('value="openai::gpt-5.6-sol"', html)
        self.assertIn('value="manual::operator"', html)
        self.assertIn('id="manual-button"', html)
        self.assertIn('id="program-panel"', html)
        self.assertIn('id="run-program"', html)
        self.assertIn("!configReady || !socketReady", javascript)
        self.assertIn("WebSocket could not connect", javascript)
        self.assertIn("/api/policy", javascript)
        self.assertIn("policy_submitted", javascript)
        self.assertIn('id="primitive-chips"', html)
        self.assertIn("case 'primitives'", javascript)
        stylesheet = (static_dir / "styles.css").read_text(encoding="utf-8")
        self.assertIn('id="readiness-prior-input" type="checkbox"', html)
        self.assertIn('class="small-field checkbox-field"', html)
        self.assertIn(".checkbox-field", stylesheet)
        self.assertIn("readiness_prior: ui.readinessPrior.checked", javascript)
        # Start and Debug both send the checkbox; Debug still skips the planner.
        self.assertEqual(javascript.count("readiness_prior: ui.readinessPrior.checked"), 2)
        self.assertIn("navigation_planner: false", javascript)
        self.assertIn("config.readiness_prior?.enabled", javascript)
        self.assertIn("config.readiness_prior?.events_path", javascript)

    def test_public_config_exposes_the_readiness_prior_block(self) -> None:
        # The shipped navigation config names no events JSON, so the checkbox
        # is not usable whether or not the block itself is present.
        public = WebRunController(CONFIG_PATH, self.config).public_config()

        self.assertEqual(
            public["readiness_prior"],
            {
                "configured": False,
                "enabled": False,
                "events_path": None,
                "source": "ready_frame",
            },
        )

        configured = WebRunController(
            CONFIG_PATH, _with_readiness_prior(self.config, READINESS_PRIOR_BLOCK)
        ).public_config()
        self.assertEqual(
            configured["readiness_prior"],
            {
                "configured": True,
                "enabled": True,
                "events_path": "/tmp/manipulation_events.json",
                "source": "ready_frame",
            },
        )

        no_settings = copy.deepcopy(self.config)
        del no_settings["primitive_config"]["primitives"]["dock_to_visible_object"]["settings"]
        absent = WebRunController(CONFIG_PATH, no_settings).public_config()
        self.assertEqual(absent["readiness_prior"]["configured"], False)
        self.assertEqual(absent["readiness_prior"]["enabled"], False)
        self.assertIsNone(absent["readiness_prior"]["events_path"])

    def test_readiness_prior_toggle_is_a_pure_deep_copy(self) -> None:
        config = _with_readiness_prior(self.config, READINESS_PRIOR_BLOCK)
        before = copy.deepcopy(config)

        off = apply_readiness_prior_toggle(config, False)
        on = apply_readiness_prior_toggle(config, True)

        self.assertEqual(config, before)
        self.assertEqual(
            _docking_settings(off)["readiness_prior"],
            {"enabled": False, "source": "ready_frame", "min_inliers": 30},
        )
        self.assertEqual(
            _docking_settings(on)["readiness_prior"], READINESS_PRIOR_BLOCK
        )
        # Every other docking setting survives the toggle.
        self.assertEqual(
            {k: v for k, v in _docking_settings(off).items() if k != "readiness_prior"},
            {k: v for k, v in _docking_settings(config).items() if k != "readiness_prior"},
        )

    def test_readiness_prior_toggle_tolerates_a_missing_block(self) -> None:
        absent = copy.deepcopy(self.config)
        _docking_settings(absent).pop("readiness_prior", None)
        for config in (absent, _with_readiness_prior(self.config, None)):
            self.assertEqual(apply_readiness_prior_toggle(config, False), config)
            with self.assertRaises(ValueError) as rejected:
                apply_readiness_prior_toggle(config, True)
            self.assertEqual(str(rejected.exception), READINESS_PRIOR_EVENTS_REQUIRED)

        no_settings = copy.deepcopy(self.config)
        del no_settings["primitive_config"]["primitives"]["dock_to_visible_object"]["settings"]
        # Switching off never creates settings: that would wire the docking
        # primitive into the environment.
        self.assertEqual(apply_readiness_prior_toggle(no_settings, False), no_settings)
        with self.assertRaises(ValueError):
            apply_readiness_prior_toggle(no_settings, True)

    def test_readiness_prior_toggle_off_drops_a_null_events_path(self) -> None:
        # The shipped block (enabled false, events_path null) loses the null
        # key on "off" so the trace never shows a prior input, and "on" is
        # rejected because null is not an events JSON.
        config = _with_readiness_prior(
            self.config, {"enabled": False, "events_path": None, "source": "ready_frame"}
        )

        off = apply_readiness_prior_toggle(config, False)

        self.assertEqual(
            _docking_settings(off)["readiness_prior"],
            {"enabled": False, "source": "ready_frame"},
        )
        with self.assertRaises(ValueError):
            apply_readiness_prior_toggle(config, True)

    def _start_with_readiness_prior(self, config: dict, values: dict) -> dict:
        controller = WebRunController(CONFIG_PATH, config)
        captured: dict = {}

        async def scenario() -> dict:
            async def fake_run(run_config) -> None:
                captured.update(run_config)

            controller._run = fake_run
            result = await controller.start(
                {
                    "instruction": config["task"]["instruction"],
                    "provider": "manual",
                    "model": "operator",
                    "temperature": None,
                    "max_turns": None,
                    **values,
                }
            )
            await controller._worker
            return result

        result = asyncio.run(scenario())
        self.assertEqual(result["status"], "started")
        return captured

    def test_start_with_readiness_prior_off_disables_and_drops_events_path(self) -> None:
        config = _with_readiness_prior(self.config, READINESS_PRIOR_BLOCK)

        captured = self._start_with_readiness_prior(config, {"readiness_prior": False})

        block = _docking_settings(captured)["readiness_prior"]
        self.assertFalse(block["enabled"])
        self.assertNotIn("events_path", block)
        self.assertEqual(block["source"], "ready_frame")
        self.assertEqual(block["min_inliers"], 30)
        # The loaded config is untouched for the next run.
        self.assertEqual(_docking_settings(config)["readiness_prior"], READINESS_PRIOR_BLOCK)

    def test_start_with_readiness_prior_on_enables_a_configured_prior(self) -> None:
        config = _with_readiness_prior(
            self.config, {**READINESS_PRIOR_BLOCK, "enabled": False}
        )

        captured = self._start_with_readiness_prior(config, {"readiness_prior": True})

        self.assertEqual(
            _docking_settings(captured)["readiness_prior"], READINESS_PRIOR_BLOCK
        )

    def test_start_with_readiness_prior_on_requires_an_events_path(self) -> None:
        from fastapi import HTTPException

        config = _with_readiness_prior(self.config, {"enabled": False, "events_path": None})
        controller = WebRunController(CONFIG_PATH, config)

        with self.assertRaises(HTTPException) as rejected:
            asyncio.run(
                controller.start(
                    {
                        "instruction": config["task"]["instruction"],
                        "provider": "manual",
                        "readiness_prior": True,
                    }
                )
            )

        self.assertEqual(rejected.exception.status_code, 422)
        self.assertEqual(
            rejected.exception.detail,
            "readiness prior needs dock_to_visible_object.readiness_prior.events_path "
            "(config or --readiness-events)",
        )
        self.assertIsNone(controller._worker)

    def test_start_without_readiness_prior_key_leaves_the_block_untouched(self) -> None:
        config = _with_readiness_prior(self.config, READINESS_PRIOR_BLOCK)

        captured = self._start_with_readiness_prior(config, {})
        self.assertEqual(
            _docking_settings(captured)["readiness_prior"], READINESS_PRIOR_BLOCK
        )

        captured = self._start_with_readiness_prior(config, {"readiness_prior": None})
        self.assertEqual(
            _docking_settings(captured)["readiness_prior"], READINESS_PRIOR_BLOCK
        )

    def test_start_rejects_a_non_boolean_readiness_prior(self) -> None:
        from fastapi import HTTPException

        controller = WebRunController(CONFIG_PATH, self.config)

        with self.assertRaises(HTTPException) as rejected:
            asyncio.run(
                controller.start(
                    {
                        "instruction": self.config["task"]["instruction"],
                        "provider": "manual",
                        "readiness_prior": "yes",
                    }
                )
            )

        self.assertEqual(rejected.exception.status_code, 422)

    def test_debug_start_passes_readiness_prior_independently_of_the_planner(self) -> None:
        config = _with_readiness_prior(
            self.config, {**READINESS_PRIOR_BLOCK, "enabled": False}
        )
        config["navigation_planner"] = {
            "enabled": True,
            "memory_path": "/tmp/memory.json",
            "model": "gemini-3.7-flash",
        }

        captured = self._start_with_readiness_prior(
            config, {"navigation_planner": False, "readiness_prior": True}
        )

        self.assertFalse(captured["navigation_planner"]["enabled"])
        self.assertNotIn("memory_path", captured["navigation_planner"])
        self.assertEqual(
            _docking_settings(captured)["readiness_prior"], READINESS_PRIOR_BLOCK
        )

        captured = self._start_with_readiness_prior(
            config, {"navigation_planner": False, "readiness_prior": False}
        )
        self.assertFalse(captured["navigation_planner"]["enabled"])
        self.assertFalse(_docking_settings(captured)["readiness_prior"]["enabled"])
        self.assertNotIn("events_path", _docking_settings(captured)["readiness_prior"])

    def test_primitive_catalog_describes_signatures_and_required_params(self) -> None:
        from types import SimpleNamespace

        from yor_agent.primitives.registry import PrimitiveRegistry
        from yor_agent.web.server import primitive_catalog

        def drive_straight(distance_m: float, *, max_speed_mps: float | None = None) -> dict:
            """Drive a signed relative distance.

            More detail here.
            """

            return {}

        def sample_grasp_pose(object_name: str, *, arm: str) -> dict:
            """Sample one grasp."""

            return {}

        registry = PrimitiveRegistry()
        registry.register("drive_straight", drive_straight)
        registry.register("sample_grasp_pose", sample_grasp_pose)

        items = {item["name"]: item for item in primitive_catalog(registry)}

        self.assertEqual(
            list(items), ["drive_straight", "sample_grasp_pose", "finish"]
        )
        drive = items["drive_straight"]
        self.assertEqual(drive["summary"], "Drive a signed relative distance.")
        self.assertIn("distance_m: float", drive["signature"])
        self.assertEqual(
            [(p["name"], p["required"], p["kind"]) for p in drive["params"]],
            [
                ("distance_m", True, "positional_or_keyword"),
                ("max_speed_mps", False, "keyword_only"),
            ],
        )
        grasp = items["sample_grasp_pose"]
        self.assertEqual(
            [(p["name"], p["required"], p["annotation"]) for p in grasp["params"]],
            [("object_name", True, "str"), ("arm", True, "str")],
        )
        self.assertTrue(items["finish"]["params"][0]["required"])

        controller = WebRunController(CONFIG_PATH, self.config)
        published: list[dict] = []

        async def scenario() -> None:
            controller._loop = asyncio.get_running_loop()

            async def record(event, *, remember=True):
                published.append(dict(event))

            controller.publish = record
            controller.publish_primitives(SimpleNamespace(registry=registry))
            await asyncio.sleep(0)

        asyncio.run(scenario())

        self.assertEqual(published[-1]["type"], "primitives")
        self.assertEqual(published[-1]["items"][0]["name"], "drive_straight")

    def test_camera_previews_are_not_kept_in_reconnect_history(self) -> None:
        controller = WebRunController(CONFIG_PATH, self.config)

        asyncio.run(
            controller.publish(
                {"type": "camera_preview", "image_url": "data:image/jpeg;base64,x"},
                remember=False,
            )
        )

        self.assertEqual(controller.history, [])

    def test_qwen_cli_override_is_accepted_by_web_start(self) -> None:
        controller = WebRunController(CONFIG_PATH, self.config)
        captured = {}

        async def scenario() -> dict:
            async def fake_run(config) -> None:
                captured.update(config)

            controller._run = fake_run
            result = await controller.start(
                {
                    "instruction": self.config["task"]["instruction"],
                    "provider": "qwen",
                    "model": "qwen3.6-plus",
                    "temperature": None,
                    "max_turns": None,
                }
            )
            await controller._worker
            return result

        result = asyncio.run(scenario())

        self.assertEqual(result["status"], "started")
        self.assertEqual(captured["model"]["provider"], "qwen")
        self.assertEqual(captured["model"]["name"], "qwen3.6-plus")
        self.assertEqual(
            captured["agent"]["max_turns"], self.config["agent"]["max_turns"]
        )

    def test_deepseek_override_is_accepted_by_web_start(self) -> None:
        controller = WebRunController(CONFIG_PATH, self.config)
        captured = {}

        async def scenario() -> dict:
            async def fake_run(config) -> None:
                captured.update(config)

            controller._run = fake_run
            result = await controller.start(
                {
                    "instruction": self.config["task"]["instruction"],
                    "provider": "deepseek",
                    "model": "deepseek-v4-flash-vision-exp",
                    "temperature": None,
                    "max_turns": None,
                }
            )
            await controller._worker
            return result

        result = asyncio.run(scenario())

        self.assertEqual(result["status"], "started")
        self.assertEqual(captured["model"]["provider"], "deepseek")
        self.assertEqual(
            captured["model"]["name"], "deepseek-v4-flash-vision-exp"
        )

    def test_openai_override_is_accepted_by_web_start(self) -> None:
        controller = WebRunController(CONFIG_PATH, self.config)
        captured = {}

        async def scenario() -> dict:
            async def fake_run(config) -> None:
                captured.update(config)

            controller._run = fake_run
            result = await controller.start(
                {
                    "instruction": self.config["task"]["instruction"],
                    "provider": "openai",
                    "model": "gpt-5.6-sol",
                    "temperature": None,
                    "max_turns": None,
                }
            )
            await controller._worker
            return result

        result = asyncio.run(scenario())

        self.assertEqual(result["status"], "started")
        self.assertEqual(captured["model"]["provider"], "openai")
        self.assertEqual(captured["model"]["name"], "gpt-5.6-sol")

    def test_manual_provider_is_accepted_by_web_start(self) -> None:
        controller = WebRunController(CONFIG_PATH, self.config)
        captured = {}

        async def scenario() -> dict:
            async def fake_run(config) -> None:
                captured.update(config)

            controller._run = fake_run
            result = await controller.start(
                {
                    "instruction": self.config["task"]["instruction"],
                    "provider": "manual",
                    "model": "",
                    "temperature": None,
                    "max_turns": None,
                }
            )
            await controller._worker
            return result

        result = asyncio.run(scenario())

        self.assertEqual(result["status"], "started")
        self.assertEqual(captured["model"]["provider"], "manual")
        self.assertEqual(captured["model"]["name"], "operator")
        self.assertEqual(controller.provider, "manual")
        self.assertEqual(controller.public_config()["provider"], "manual")

    def test_debug_start_disables_the_navigation_planner(self) -> None:
        config = dict(self.config)
        config["navigation_planner"] = {
            "enabled": True,
            "memory_path": "/tmp/memory.json",
            "model": "gemini-3.7-flash",
        }
        controller = WebRunController(CONFIG_PATH, config)
        captured = {}

        async def scenario() -> dict:
            async def fake_run(config) -> None:
                captured.update(config)

            controller._run = fake_run
            result = await controller.start(
                {
                    "instruction": config["task"]["instruction"],
                    "provider": "manual",
                    "model": "operator",
                    "temperature": None,
                    "max_turns": None,
                    "navigation_planner": False,
                }
            )
            await controller._worker
            return result

        result = asyncio.run(scenario())

        self.assertEqual(result["status"], "started")
        self.assertFalse(captured["navigation_planner"]["enabled"])
        self.assertNotIn("memory_path", captured["navigation_planner"])
        self.assertEqual(captured["navigation_planner"]["model"], "gemini-3.7-flash")

    def test_policy_submission_requires_an_active_manual_run(self) -> None:
        from fastapi import HTTPException

        controller = WebRunController(CONFIG_PATH, self.config)

        with self.assertRaises(HTTPException) as no_run:
            asyncio.run(controller.submit_policy({"code": "observe()"}))
        with self.assertRaises(HTTPException) as empty:
            asyncio.run(controller.submit_policy({"code": "   "}))

        self.assertEqual(no_run.exception.status_code, 409)
        self.assertEqual(empty.exception.status_code, 422)

    def test_policy_submission_queues_into_the_manual_model(self) -> None:
        from types import SimpleNamespace

        from yor_agent.models.manual import ManualModel

        controller = WebRunController(CONFIG_PATH, self.config)
        published: list[dict] = []

        async def scenario() -> dict:
            async def record(event, *, remember=True):
                published.append(dict(event))

            controller.publish = record
            model = ManualModel()
            controller.runtime = SimpleNamespace(
                model=model, trace=SimpleNamespace(turn=1)
            )
            controller.state = "running"
            controller._worker = asyncio.get_running_loop().create_future()
            try:
                return await controller.submit_policy({"code": "drive_straight(0.3)"}), model
            finally:
                controller._worker.cancel()

        result, model = asyncio.run(scenario())

        self.assertEqual(result["status"], "queued")
        self.assertEqual(result["pending"], 1)
        self.assertEqual(model.pending(), 1)
        self.assertEqual(published[-1]["type"], "policy_submitted")
        self.assertEqual(published[-1]["code"], "drive_straight(0.3)")

    def test_policy_submission_is_one_program_per_turn(self) -> None:
        from fastapi import HTTPException
        from types import SimpleNamespace

        from yor_agent.models.manual import ManualModel

        controller = WebRunController(CONFIG_PATH, self.config)

        async def scenario() -> int:
            async def record(event, *, remember=True):
                pass

            controller.publish = record
            model = ManualModel()
            model.submit("observe()")
            controller.runtime = SimpleNamespace(
                model=model, trace=SimpleNamespace(turn=1)
            )
            controller.state = "running"
            controller._worker = asyncio.get_running_loop().create_future()
            try:
                await controller.submit_policy({"code": "drive_straight(0.3)"})
            except HTTPException as exc:
                return exc.status_code
            finally:
                controller._worker.cancel()
            return 200

        self.assertEqual(asyncio.run(scenario()), 409)

    def test_run_end_events_leave_the_running_state_immediately(self) -> None:
        controller = WebRunController(CONFIG_PATH, self.config)

        async def scenario() -> list[str]:
            controller._loop = asyncio.get_running_loop()
            states = []
            controller.emit_from_runtime({"type": "run_started"})
            states.append(controller.state)
            controller.emit_from_runtime({"type": "run_finished", "status": "finished"})
            states.append(controller.state)
            controller.emit_from_runtime({"type": "run_error", "message": "x"})
            states.append(controller.state)
            await asyncio.sleep(0)
            return states

        self.assertEqual(asyncio.run(scenario()), ["running", "finished", "error"])

    def _block_run_until_agent_stop(self, controller) -> tuple[dict, object]:
        """Replace the robot run with one that only ends on the agent's stop.

        Like a primitive driving the base, the fake run ignores the loop's stop
        event; only ``runtime.agent.request_stop`` (the Stop button's motion
        cancel) ends it.
        """

        import threading
        from types import SimpleNamespace

        agent_stopped = threading.Event()
        record: dict = {"stop_reasons": [], "finished": []}

        def request_stop(reason="operator requested stop"):
            record["stop_reasons"].append(reason)
            agent_stopped.set()

        runtime = SimpleNamespace(agent=SimpleNamespace(request_stop=request_stop))

        def run_sync(config):
            controller.runtime = runtime
            stopped = agent_stopped.wait(timeout=5.0)
            # Stands in for DefaultAgent.run's safe shutdown and trace save.
            record["finished"].append(stopped)
            return {"status": "stopped" if stopped else "ran_without_stop"}

        async def no_camera():
            return None

        controller._run_sync = run_sync
        controller._camera_stream = no_camera
        return record, runtime

    def test_server_shutdown_stops_the_active_run_and_waits_for_it(self) -> None:
        controller = WebRunController(CONFIG_PATH, self.config)
        record, runtime = self._block_run_until_agent_stop(controller)

        async def scenario() -> bool:
            async def quiet(event, *, remember=True):
                pass

            controller.publish = quiet
            controller.runtime = runtime
            controller._worker = asyncio.create_task(controller._run({}))
            await controller.shutdown(timeout_s=5.0)
            return controller._worker.done()

        self.assertTrue(asyncio.run(scenario()))
        self.assertEqual(record["stop_reasons"], [SERVER_SHUTDOWN_STOP_REASON])
        self.assertEqual(record["finished"], [True])
        self.assertEqual(controller.state, "stopped")

    def test_server_shutdown_without_a_run_does_nothing(self) -> None:
        controller = WebRunController(CONFIG_PATH, self.config)
        record, _ = self._block_run_until_agent_stop(controller)

        asyncio.run(controller.shutdown(timeout_s=0.1))

        self.assertEqual(record["stop_reasons"], [])
        self.assertFalse(controller._stop_event.is_set())

    def test_event_loop_teardown_stops_the_run_before_waiting_for_it(self) -> None:
        # A second Ctrl+C skips the server's shutdown; asyncio.run then cancels
        # the run task on its way out and waits for the run's thread.
        controller = WebRunController(CONFIG_PATH, self.config)
        record, runtime = self._block_run_until_agent_stop(controller)

        async def scenario() -> None:
            async def quiet(event, *, remember=True):
                pass

            controller.publish = quiet
            controller.runtime = runtime
            controller._worker = asyncio.create_task(controller._run({}))
            await asyncio.sleep(0.05)

        asyncio.run(scenario())

        self.assertEqual(record["stop_reasons"], [SERVER_SHUTDOWN_STOP_REASON])
        self.assertEqual(record["finished"], [True])
        self.assertEqual(controller.state, "stopped")

    def test_app_shutdown_stops_a_run_started_from_the_ui(self) -> None:
        import time

        try:
            from fastapi.testclient import TestClient
        except (ImportError, RuntimeError) as exc:  # needs httpx
            self.skipTest(f"FastAPI TestClient unavailable: {exc}")

        app = create_app(config_path=CONFIG_PATH, config=self.config)
        controller = app.state.controller
        record, _ = self._block_run_until_agent_stop(controller)

        with TestClient(app) as client:
            response = client.post(
                "/api/start",
                json={
                    "instruction": "stay put",
                    "provider": "manual",
                    "navigation_planner": False,
                },
            )
            self.assertEqual(response.status_code, 200)
            deadline = time.monotonic() + 2.0
            while controller.runtime is None and time.monotonic() < deadline:
                time.sleep(0.01)

        self.assertEqual(record["stop_reasons"], [SERVER_SHUTDOWN_STOP_REASON])
        self.assertEqual(record["finished"], [True])
        self.assertEqual(controller.state, "stopped")

    def test_preview_encoder_returns_a_jpeg_data_url(self) -> None:
        url = _preview_data_url(np.zeros((12, 20, 3), dtype=np.uint8))

        header, payload = url.split(",", 1)
        self.assertEqual(header, "data:image/jpeg;base64")
        self.assertTrue(base64.b64decode(payload).startswith(b"\xff\xd8"))


@unittest.skipUnless(
    WEB_DEPENDENCIES_AVAILABLE,
    "FastAPI/Uvicorn web dependencies are not installed in this test environment",
)
class ExperimentEpisodeTest(unittest.TestCase):
    def setUp(self) -> None:
        import tempfile

        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.config = load_config(CONFIG_PATH)
        self.config["experiment"] = {
            "name": "nav_main",
            "start_label": "kitchen",
            "condition": "ours",
            "output_root": str(self.root / "ours"),
        }

    def _fake_run(self, controller, result: dict) -> None:
        def run_sync(config):
            controller._timing = {"started_at": "t0", "ended_at": "t1", "elapsed_s": 12.5}
            return dict(result)

        async def no_camera():
            return None

        controller._run_sync = run_sync
        controller._camera_stream = no_camera

    def test_episode_is_recorded_and_reviewed_before_the_next_run(self) -> None:
        import json
        import time

        try:
            from fastapi.testclient import TestClient
        except (ImportError, RuntimeError) as exc:  # needs httpx
            self.skipTest(f"FastAPI TestClient unavailable: {exc}")

        app = create_app(config_path=CONFIG_PATH, config=self.config)
        controller = app.state.controller
        self._fake_run(
            controller, {"status": "stopped", "reason": "time limit of 900 s reached", "turns": 7}
        )
        start = {"instruction": "Navigate to the blue trash bin", "provider": "openai", "model": "gpt-5.6-sol"}
        with TestClient(app) as client:
            self.assertEqual(client.post("/api/start", json=start).status_code, 200)
            deadline = time.monotonic() + 3.0
            while controller.review_pending is None and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertIsNotNone(controller.review_pending)
            episode_dir = Path(controller.review_pending["output_dir"])
            result = json.loads((episode_dir / "result.json").read_text())
            self.assertEqual(result["termination"], "time_limit")
            self.assertEqual(result["elapsed_s"], 12.5)
            self.assertEqual(
                (result["experiment"], result["condition"], result["start_label"]),
                ("nav_main", "ours", "kitchen"),
            )
            self.assertIsNone(result["adopted"])
            self.assertEqual(episode_dir.parents[1], self.root / "ours" / "nav_main")
            self.assertTrue(episode_dir.name.endswith("_kitchen"))
            self.assertEqual(client.get("/api/config").json()["review"]["termination"], "time_limit")
            self.assertEqual(client.post("/api/start", json=start).status_code, 409)
            self.assertEqual(client.post("/api/review", json={"task_success": True}).status_code, 422)
            response = client.post(
                "/api/review",
                json={
                    "task_success": False,
                    "adopted": False,
                    "exclusion_reason": "a person stepped in front",
                    "note": "retry",
                },
            )
            self.assertEqual(response.status_code, 200)
            moved = Path(response.json()["episode_dir"])
            self.assertEqual(moved.parents[1].name, "_excluded")
            reviewed = json.loads((moved / "result.json").read_text())
            self.assertIs(reviewed["task_success"], False)
            self.assertEqual(reviewed["task_success_source"], "operator")
            self.assertEqual(reviewed["exclusion_reason"], "a person stepped in front")
            self.assertIsNone(client.get("/api/config").json()["review"])
            self.assertEqual(
                client.post("/api/review", json={"task_success": True, "adopted": True}).status_code,
                409,
            )

    def test_manual_debug_runs_are_not_experiment_episodes(self) -> None:
        controller = WebRunController(CONFIG_PATH, self.config)

        async def scenario() -> None:
            async def quiet(event, *, remember=True):
                pass

            controller.publish = quiet
            self._fake_run(controller, {"status": "finished", "reason": "done", "turns": 1})
            await controller.start({"instruction": "stay put", "provider": "manual", "navigation_planner": False})
            await controller._worker

        asyncio.run(scenario())
        self.assertIsNone(controller.review_pending)
        self.assertFalse((self.root / "ours").exists())

    def test_run_sync_decorates_the_runtime_and_times_the_agent_loop(self) -> None:
        from types import SimpleNamespace
        from unittest import mock

        decorated = []
        agent = SimpleNamespace(run=lambda task, config: {"status": "finished", "reason": task, "turns": 1})
        runtime = SimpleNamespace(agent=agent)
        controller = WebRunController(CONFIG_PATH, self.config, decorate_runtime=decorated.append)
        controller.publish_primitives = lambda _runtime: None
        with mock.patch("yor_agent.web.server.build_runtime", return_value=runtime):
            result = controller._run_sync({"task": {"instruction": "go"}})
        self.assertEqual(result["reason"], "go")
        self.assertEqual(decorated, [runtime])
        self.assertIs(controller.runtime, runtime)
        self.assertGreaterEqual(controller._timing["elapsed_s"], 0.0)

    def test_termination_names_time_limit_and_operator_stops(self) -> None:
        from yor_agent.web.server import episode_termination

        self.assertEqual(episode_termination({"status": "stopped", "reason": "time limit of 900 s reached"}), "time_limit")
        self.assertEqual(episode_termination({"status": "stopped", "reason": "operator requested stop"}), "operator_stop")
        self.assertEqual(episode_termination({"status": "finished", "reason": "arrived"}), "finished")
        self.assertEqual(episode_termination({"status": "error", "reason": "boom"}), "error")


if __name__ == "__main__":
    unittest.main()
