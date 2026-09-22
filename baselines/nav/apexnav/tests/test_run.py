from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import yaml

from baselines.nav.apexnav import run
from baselines.nav.apexnav.config import ViewerConfig
from baselines.nav.apexnav.run import (
    VIEWER_TOPIC_WHITELIST,
    _ignore_interrupts,
    _start_viewer,
    _stop_children,
    _stop_planner,
    _viewer_address,
    _viewer_parameters,
)


class PlannerProcessTests(unittest.TestCase):
    def test_stop_planner_reaps_its_process_group(self):
        process = subprocess.Popen(["sleep", "60"], start_new_session=True)
        try:
            _stop_planner(process)
            self.assertIsNotNone(process.poll())
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=2.0)

    def test_stop_children_stops_the_relay_when_planner_stop_is_interrupted(self):
        relay = subprocess.Popen(["sleep", "60"], start_new_session=True)
        try:
            with mock.patch.object(run, "_stop_planner", side_effect=KeyboardInterrupt):
                with self.assertRaises(KeyboardInterrupt):
                    _stop_children(object(), relay)
            self.assertIsNotNone(relay.poll())
        finally:
            if relay.poll() is None:
                relay.kill()
                relay.wait(timeout=2.0)

    def test_stop_children_stops_recorders_last_even_after_a_failure(self):
        planner, relay, base_status, bag = object(), object(), object(), object()
        names = {id(relay): "relay", id(base_status): "base_status", id(bag): "bag"}
        stopped = []

        def stop_group(process):
            stopped.append(names[id(process)])
            if process is base_status:
                raise RuntimeError("recorder already gone")

        with mock.patch.object(
            run, "_stop_planner", side_effect=lambda process: stopped.append("planner")
        ), mock.patch.object(run, "_stop_process_group", side_effect=stop_group):
            _stop_children(planner, relay, base_status, bag)

        self.assertEqual(stopped, ["planner", "relay", "base_status", "bag"])

    def test_ignore_interrupts_ignores_ctrl_c_hangup_and_sigterm(self):
        signals = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
        saved = {value: signal.getsignal(value) for value in signals}
        try:
            _ignore_interrupts()
            for value in signals:
                self.assertIs(signal.getsignal(value), signal.SIG_IGN)
        finally:
            for value, handler in saved.items():
                signal.signal(value, handler)


class ViewerTests(unittest.TestCase):
    def test_stop_children_stops_the_viewer_after_the_recorders(self):
        planner, relay, base_status, bag, viewer = (object() for _ in range(5))
        names = {
            id(relay): "relay",
            id(base_status): "base_status",
            id(bag): "bag",
            id(viewer): "viewer",
        }
        stopped = []

        def stop_group(process):
            stopped.append(names[id(process)])
            if process is bag:
                raise RuntimeError("recorder already gone")

        with mock.patch.object(
            run, "_stop_planner", side_effect=lambda process: stopped.append("planner")
        ), mock.patch.object(run, "_stop_process_group", side_effect=stop_group):
            _stop_children(planner, relay, base_status, bag, viewer)

        self.assertEqual(stopped, ["planner", "relay", "base_status", "bag", "viewer"])

    def test_viewer_parameters_allow_reading_whitelisted_topics_only(self):
        settings = _viewer_parameters(ViewerConfig(), "127.0.0.1")
        self.assertEqual(list(settings), ["/**"])
        parameters = settings["/**"]["ros__parameters"]
        self.assertEqual(parameters["capabilities"], ["none"])
        for key in (
            "service_whitelist",
            "param_whitelist",
            "client_topic_whitelist",
            "asset_uri_allowlist",
        ):
            self.assertEqual(parameters[key], ["(?!)"])
        self.assertEqual(parameters["topic_whitelist"], list(VIEWER_TOPIC_WHITELIST))
        self.assertEqual((parameters["address"], parameters["port"]), ("127.0.0.1", 8765))
        self.assertFalse(parameters["sysinfo"])
        self.assertEqual(yaml.safe_load(yaml.safe_dump(settings)), settings)
        self.assertEqual(_yaml_anchors(yaml.safe_dump(settings)), [])

    def test_viewer_address_prefers_configuration_then_tailscale_then_loopback(self):
        with mock.patch.object(run.subprocess, "run") as tailscale:
            self.assertEqual(_viewer_address("10.0.0.5"), "10.0.0.5")
            tailscale.assert_not_called()
            tailscale.return_value = subprocess.CompletedProcess(
                [], 0, stdout="100.64.0.7\n", stderr=""
            )
            self.assertEqual(_viewer_address(""), "100.64.0.7")
            tailscale.return_value = subprocess.CompletedProcess([], 0, stdout="", stderr="")
            self.assertEqual(_viewer_address(""), "127.0.0.1")
            tailscale.side_effect = FileNotFoundError("tailscale")
            self.assertEqual(_viewer_address(""), "127.0.0.1")

    def test_missing_viewer_executable_skips_the_viewer(self):
        with tempfile.TemporaryDirectory() as directory:
            log_dir = Path(directory)
            with mock.patch.object(run, "start_logged_process") as start:
                result = _start_viewer(ViewerConfig(), log_dir, log_dir / "missing")
            self.assertEqual(result, (None, None))
            start.assert_not_called()
            self.assertEqual(list(log_dir.iterdir()), [])

    def test_start_viewer_runs_the_bridge_with_its_parameter_file(self):
        viewer = ViewerConfig(address="127.0.0.1")
        process = object()
        with tempfile.TemporaryDirectory() as directory:
            log_dir = Path(directory)
            with mock.patch.object(
                run, "start_logged_process", return_value=process
            ) as start:
                result = _start_viewer(viewer, log_dir, Path(sys.executable))
            self.assertEqual(result, (process, "ws://127.0.0.1:8765"))
            command, log_path = start.call_args.args
            parameters_path = log_dir.resolve() / "foxglove_bridge.params.yaml"
            self.assertEqual(
                command,
                [sys.executable, "--ros-args", "--params-file", str(parameters_path)],
            )
            self.assertEqual(log_path, log_dir / "foxglove_bridge.log")
            text = parameters_path.read_text(encoding="utf-8")
            self.assertEqual(yaml.safe_load(text), _viewer_parameters(viewer, "127.0.0.1"))
            self.assertEqual(_yaml_anchors(text), [])


def _yaml_anchors(text: str) -> list[str]:
    """Anchors and aliases in a YAML document; rcl's parameter file parser rejects both."""

    return [event.anchor for event in yaml.parse(text) if getattr(event, "anchor", None)]


if __name__ == "__main__":
    unittest.main()
