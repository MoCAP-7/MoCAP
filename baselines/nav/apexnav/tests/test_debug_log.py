import json
from pathlib import Path
import socket
import subprocess
import tempfile
import threading
import time
import unittest

from baselines.nav.apexnav.base_status_logger import poll_status
from baselines.nav.apexnav.debug_log import (
    BAG_TOPICS,
    JsonlWriter,
    bag_record_command,
    environment_snapshot,
    start_console_tee,
)


def _unused_port():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


class JsonlWriterTests(unittest.TestCase):
    def test_records_from_several_threads_are_all_written(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "records.jsonl"
            writer = JsonlWriter(path)

            def produce(worker):
                for index in range(50):
                    writer.write({"worker": worker, "index": index})

            threads = [threading.Thread(target=produce, args=(worker,)) for worker in range(4)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            writer.close()

            records = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(len(records), 200)
            self.assertEqual(writer.dropped, 0)
            self.assertIn("t_wall", records[0])
            self.assertIn("t_monotonic", records[0])


class RecorderCommandTests(unittest.TestCase):
    def test_bag_command_records_every_debug_topic(self):
        command = bag_record_command(Path("/tmp/run/debug/bag"))
        self.assertEqual(command[:5], ["ros2", "bag", "record", "-o", "/tmp/run/debug/bag"])
        self.assertEqual(tuple(command[5:]), BAG_TOPICS)
        self.assertIn("/apexnav/cmd_vel_raw", BAG_TOPICS)
        self.assertIn("/mpc_car/track_err", BAG_TOPICS)

    def test_console_tee_copies_child_output_to_the_log(self):
        with tempfile.TemporaryDirectory() as directory:
            log_path = Path(directory) / "console.log"
            process = subprocess.Popen(
                ["printf", "first\\nsecond\\n"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            thread = start_console_tee(process, log_path)
            process.wait(timeout=5.0)
            thread.join(timeout=5.0)
            process.stdout.close()
            self.assertEqual(log_path.read_text(), "first\nsecond\n")

    def test_environment_snapshot_reports_unreachable_endpoints(self):
        port = _unused_port()
        snapshot = environment_snapshot(
            Path.cwd(),
            Path("config.yaml"),
            {"closed": f"http://127.0.0.1:{port}/x", "no_port": "http://127.0.0.1/"},
        )
        self.assertEqual(
            snapshot["endpoints_reachable"], {"closed": False, "no_port": False}
        )
        self.assertIn("head", snapshot["git"])
        self.assertIsInstance(snapshot["service_processes"], list)


class _StatusClient:
    def __init__(self, stop, fail_first=False, stop_after=3):
        self.stop = stop
        self.fail_first = fail_first
        self.stop_after = stop_after
        self.calls = 0

    def get_status(self):
        self.calls += 1
        if self.calls >= self.stop_after:
            self.stop.set()
        if self.fail_first and self.calls == 1:
            raise RuntimeError("Again: Resource temporarily unavailable")
        return {"lease_active": False, "cmd_vel": [[0.1, 0.0, 0.0], 1.0]}


class PollStatusTests(unittest.TestCase):
    def test_writes_status_and_reconnects_after_an_error(self):
        stop = threading.Event()
        connections = []

        def connect():
            client = _StatusClient(stop, fail_first=not connections, stop_after=2)
            connections.append(client)
            return client

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "base_status.jsonl"
            started = time.monotonic()
            written = poll_status(connect, output, 0.01, stop, error_backoff_s=0.01)
            self.assertLess(time.monotonic() - started, 5.0)
            records = [json.loads(line) for line in output.read_text().splitlines()]

        self.assertEqual(written, len(records))
        self.assertEqual(len(connections), 2)
        self.assertIn("error", records[0])
        self.assertEqual(records[-1]["status"]["cmd_vel"][0], [0.1, 0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
