import os
import subprocess
import sys
import tempfile
import time
import unittest

from geometry_msgs.msg import Twist
import rclpy
from baselines.nav.apexnav.config import ApexNavConfig
from baselines.nav.apexnav.control_relay import ApexNavControlRelay


class _RPC:
    def __init__(self):
        self.commands = []

    def get_status(self):
        return {
            "estop_latched": False,
            "lease_active": False,
            "limits": {"lease_s": 0.25},
        }

    def submit_velocity(self, velocity, sequence):
        self.commands.append((list(velocity), int(sequence)))
        return {"accepted": True}


class _SensorMonitor:
    max_gap_s = 0.05

    def wait_until_fresh(self, max_age_s):
        return None

    def is_fresh(self, max_age_s):
        return True

    def close(self):
        return None


class ControlRelayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def test_relay_renews_continuous_command_and_watchdog_stops(self):
        rpc = _RPC()
        node = ApexNavControlRelay(
            ApexNavConfig(), rpc=rpc, sensor_monitor=_SensorMonitor()
        )
        try:
            command = Twist()
            command.linear.x = 0.01
            command.angular.z = -0.02
            node._on_twist(command)
            node._renew_base_lease()
            node._renew_base_lease()
            self.assertEqual(rpc.commands[-1][0], [0.05, 0.0, -0.16])
            self.assertEqual(node.base_submit_count, 2)

            node._command_received_at = time.monotonic() - 1.0
            node._renew_base_lease()
            self.assertEqual(rpc.commands[-1][0], [0.0, 0.0, 0.0])
            self.assertEqual(node.watchdog_zero_count, 1)
            self.assertEqual(node.stale_command_zero_count, 1)
            self.assertEqual(node.stale_sensor_zero_count, 0)
            self.assertEqual(node.zero_submit_count, 1)
        finally:
            node.stop()
            node.destroy_node()

    def test_command_log_records_raw_twists_and_submitted_commands(self):
        rpc = _RPC()
        log = _CommandLog()
        node = ApexNavControlRelay(
            ApexNavConfig(), rpc=rpc, sensor_monitor=_SensorMonitor(), command_log=log
        )
        try:
            command = Twist()
            command.linear.x = 0.01
            command.angular.z = -0.02
            node._on_twist(command)
            node._renew_base_lease()
            node._command_received_at = time.monotonic() - 1.0
            node._renew_base_lease()
        finally:
            node.stop()
            node.destroy_node()

        twist = log.records[0]
        self.assertEqual(twist["event"], "twist")
        self.assertEqual(twist["raw_linear_x"], 0.01)
        self.assertEqual(twist["command"], [0.05, 0.0, -0.16])
        submits = [record for record in log.records if record["event"] == "submit"]
        self.assertEqual(
            [record["reason"] for record in submits], ["active", "stale_command", "stop"]
        )
        self.assertEqual(submits[0]["command"], [0.05, 0.0, -0.16])
        self.assertTrue(all(record["accepted"] for record in submits))


class _CommandLog:
    def __init__(self):
        self.records = []

    def write(self, record):
        self.records.append(dict(record))


class CleanupWatchdogTests(unittest.TestCase):
    def test_watchdog_exits_a_blocked_process_and_records_its_stacks(self):
        with tempfile.TemporaryDirectory() as directory:
            traceback_path = os.path.join(directory, "cleanup_traceback.txt")
            code = (
                "import pathlib, time\n"
                "from baselines.nav.apexnav.control_relay import _arm_exit_watchdog\n"
                f"_arm_exit_watchdog(0.2, 3, pathlib.Path({traceback_path!r}))\n"
                "time.sleep(30)\n"
            )
            started = time.monotonic()
            completed = subprocess.run(
                [sys.executable, "-c", code],
                env=os.environ.copy(),
                cwd=os.getcwd(),
                timeout=20,
                check=False,
            )
            self.assertEqual(completed.returncode, 3)
            self.assertLess(time.monotonic() - started, 15.0)
            with open(traceback_path, encoding="utf-8") as handle:
                self.assertIn("most recent call first", handle.read())


if __name__ == "__main__":
    unittest.main()
