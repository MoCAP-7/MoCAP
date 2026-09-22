"""Isolated ROS2-to-Pi velocity relay for ApexNav on YOR."""

from __future__ import annotations

import argparse
import faulthandler
import json
import math
import os
from pathlib import Path
import signal
import threading
import time
from typing import Any
import warnings

from geometry_msgs.msg import Twist
import numpy as np
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions

from .config import ApexNavConfig, load_config
from .debug_log import JsonlWriter

# Upper bound on the relay's own shutdown. The Pi's velocity lease expires on
# its own, so exiting before cleanup finishes cannot leave the base moving.
CLEANUP_TIMEOUT_S = 4.0


class ZEDFreshnessMonitor:
    """Track fresh valid ZED poses without unpacking RGB or depth arrays."""

    _topic = "navdp/rgbd"

    def __init__(self, host: str, port: int) -> None:
        from commlink import Subscriber

        # Use the subscriber's single global socket. Supplying an explicit topic
        # would also retain its always-created global socket and duplicate every
        # large RGB-D message on loopback.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            self._subscriber = Subscriber(
                host=host,
                port=int(port),
                topics=[],
                buffer=False,
                queue_size=1,
            )
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._last_timestamp_ns = -1
        self._last_valid_received_at = 0.0
        self.max_gap_s = 0.0
        self.error: str | None = None
        self._thread = threading.Thread(
            target=self._receiver_loop,
            name="apexnav-zed-freshness-monitor",
            daemon=True,
        )
        self._thread.start()

    def _receiver_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                topic, message = self._subscriber.get()
                if topic != self._topic:
                    continue
                timestamp_ns = int(message["timestamp_ns"])
                if timestamp_ns <= self._last_timestamp_ns:
                    time.sleep(0.005)
                    continue
                self._last_timestamp_ns = timestamp_ns
                if not bool(message.get("pose_valid", False)):
                    continue
                now = time.monotonic()
                with self._lock:
                    if self._last_valid_received_at > 0.0:
                        self.max_gap_s = max(
                            self.max_gap_s, now - self._last_valid_received_at
                        )
                    self._last_valid_received_at = now
            except BaseException as exc:  # noqa: BLE001 - surfaced to relay
                if not self._stop_event.is_set():
                    self.error = f"{type(exc).__name__}: {exc}"
                return

    def wait_until_fresh(self, max_age_s: float, timeout_s: float = 5.0) -> None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self.error is not None:
                raise RuntimeError(f"ZED freshness monitor failed: {self.error}")
            if self.is_fresh(max_age_s):
                return
            time.sleep(0.01)
        raise RuntimeError("ZED freshness monitor received no valid pose within 5 s")

    def is_fresh(self, max_age_s: float) -> bool:
        with self._lock:
            received_at = self._last_valid_received_at
        return received_at > 0.0 and time.monotonic() - received_at <= max_age_s

    def close(self) -> None:
        self._stop_event.set()
        self._subscriber.stop()
        self._thread.join(timeout=1.0)


class ApexNavControlRelay(Node):
    """Keep safety-critical base lease callbacks out of the perception process."""

    def __init__(
        self,
        config: ApexNavConfig,
        *,
        rpc: Any | None = None,
        sensor_monitor: Any | None = None,
        command_log: Any | None = None,
    ) -> None:
        super().__init__("yor_apexnav_control_relay")
        self.config = config
        # Receives every incoming Twist and every submitted base command.
        self._command_log = command_log
        if rpc is None:
            from commlink import RPCClient

            rpc = RPCClient(
                host=config.robot.base_rpc_host,
                port=config.robot.base_rpc_port,
            )
            import zmq

            timeout_ms = int(round(config.robot.base_rpc_timeout_s * 1000.0))
            rpc.socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
            rpc.socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
            rpc.socket.setsockopt(zmq.LINGER, 0)
        self._rpc = rpc
        self._preflight_base()
        self._sensor_monitor = sensor_monitor or ZEDFreshnessMonitor(
            config.robot.zed_host, config.robot.zed_port
        )
        self._sensor_monitor.wait_until_fresh(config.robot.sensor_timeout_s)

        self._lock = threading.Lock()
        self._command = np.zeros(3, dtype=np.float64)
        self._command_received_at = 0.0
        self._command_active = False
        self._zero_pending = False
        self._sequence = time.time_ns()
        self.finished = threading.Event()
        self.fatal_error: str | None = None

        self.command_count = 0
        self.nonzero_command_count = 0
        self.base_submit_count = 0
        self.zero_submit_count = 0
        self.watchdog_zero_count = 0
        self.stale_command_zero_count = 0
        self.stale_sensor_zero_count = 0
        self.max_command_gap_s = 0.0
        self._last_command_at = 0.0

        self.create_subscription(Twist, "/apexnav/cmd_vel_raw", self._on_twist, 20)
        self.create_timer(1.0 / config.robot.control_hz, self._renew_base_lease)

    def _preflight_base(self) -> None:
        status = self._rpc.get_status()
        if not isinstance(status, dict):
            raise RuntimeError("base RPC returned an invalid status")
        if status.get("estop_latched", False):
            raise RuntimeError("Pi base emergency stop is latched")
        if status.get("lease_active", False):
            raise RuntimeError("another process owns the Pi base velocity lease")
        limits = status.get("limits")
        if not isinstance(limits, dict):
            raise RuntimeError("base RPC did not publish velocity limits")
        if float(limits.get("lease_s", 1.0)) > 0.30:
            raise RuntimeError("Pi base lease exceeds the 0.30 s safety bound")

    def _log(self, record: dict[str, Any]) -> None:
        if self._command_log is not None:
            self._command_log.write(record)

    @staticmethod
    def _speed_floor(value: float, floor: float) -> float:
        if abs(value) <= 1e-9:
            return 0.0
        return math.copysign(max(abs(value), floor), value)

    def _on_twist(self, message: Twist) -> None:
        command = np.asarray(
            [float(message.linear.x), 0.0, float(message.angular.z)],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(command)):
            self.get_logger().error("discarded non-finite ApexNav Twist")
            return
        command[0] = float(
            np.clip(
                command[0],
                -self.config.robot.maximum_linear_mps,
                self.config.robot.maximum_linear_mps,
            )
        )
        command[2] = float(
            np.clip(
                command[2],
                -self.config.robot.maximum_yaw_rad_s,
                self.config.robot.maximum_yaw_rad_s,
            )
        )
        command[0] = self._speed_floor(
            command[0], self.config.robot.minimum_linear_mps
        )
        command[2] = self._speed_floor(
            command[2], self.config.robot.minimum_yaw_rad_s
        )
        self._log(
            {
                "event": "twist",
                "raw_linear_x": float(message.linear.x),
                "raw_angular_z": float(message.angular.z),
                "command": command.tolist(),
            }
        )
        now = time.monotonic()
        with self._lock:
            if self._last_command_at > 0.0:
                self.max_command_gap_s = max(
                    self.max_command_gap_s, now - self._last_command_at
                )
            self._last_command_at = now
            self.command_count += 1
            if np.any(np.abs(command) > 1e-9):
                self.nonzero_command_count += 1
                self._command = command
                self._command_received_at = now
                self._command_active = True
            else:
                self._command.fill(0.0)
                self._command_active = False
                self._zero_pending = True

    def _renew_base_lease(self) -> None:
        with self._lock:
            now = time.monotonic()
            stale_command = (
                now - self._command_received_at
                > self.config.robot.command_timeout_s
            )
            stale_sensor = not self._sensor_monitor.is_fresh(
                self.config.robot.sensor_timeout_s
            )
            reason = "active"
            if self._command_active and (stale_command or stale_sensor):
                self._command_active = False
                self._command.fill(0.0)
                self._zero_pending = True
                self.watchdog_zero_count += 1
                if stale_command:
                    self.stale_command_zero_count += 1
                if stale_sensor:
                    self.stale_sensor_zero_count += 1
                reason = "stale_command" if stale_command else "stale_sensor"
            if self._command_active:
                command = self._command.copy()
            elif self._zero_pending:
                command = np.zeros(3, dtype=np.float64)
                self._zero_pending = False
                if reason == "active":
                    reason = "zero_command"
            else:
                return
        try:
            self._send_base(command, reason)
        except Exception as exc:  # noqa: BLE001 - relay must fail closed
            self.fatal_error = f"{type(exc).__name__}: {exc}"
            self.get_logger().error(f"base command failure: {self.fatal_error}")
            self.finished.set()

    def _send_base(self, command: np.ndarray, reason: str = "direct") -> None:
        self._sequence = max(time.time_ns(), self._sequence + 1)
        started = time.monotonic()
        try:
            reply = self._rpc.submit_velocity(command.tolist(), self._sequence)
        except Exception as exc:
            self._log(
                {
                    "event": "submit",
                    "reason": reason,
                    "command": command.tolist(),
                    "sequence": self._sequence,
                    "rpc_s": round(time.monotonic() - started, 6),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            raise
        accepted = isinstance(reply, dict) and bool(reply.get("accepted", False))
        self._log(
            {
                "event": "submit",
                "reason": reason,
                "command": command.tolist(),
                "sequence": self._sequence,
                "rpc_s": round(time.monotonic() - started, 6),
                "accepted": accepted,
            }
        )
        if not accepted:
            raise RuntimeError(f"base RPC rejected ApexNav velocity: {reply!r}")
        self.base_submit_count += 1
        if not np.any(np.abs(command) > 1e-9):
            self.zero_submit_count += 1

    def stop(self) -> None:
        with self._lock:
            self._command.fill(0.0)
            self._command_active = False
            self._zero_pending = False
        try:
            self._send_base(np.zeros(3, dtype=np.float64), "stop")
        except Exception as exc:  # noqa: BLE001 - retain original failure
            self.get_logger().error(f"failed to stop base: {exc}")

    def stats(self, reason: str) -> dict[str, Any]:
        return {
            "reason": reason,
            "fatal_error": self.fatal_error,
            "command_count": self.command_count,
            "nonzero_command_count": self.nonzero_command_count,
            "base_submit_count": self.base_submit_count,
            "zero_submit_count": self.zero_submit_count,
            "watchdog_zero_count": self.watchdog_zero_count,
            "stale_command_zero_count": self.stale_command_zero_count,
            "stale_sensor_zero_count": self.stale_sensor_zero_count,
            "max_command_gap_s": round(self.max_command_gap_s, 6),
            "max_sensor_gap_s": round(self._sensor_monitor.max_gap_s, 6),
        }

    def destroy_node(self) -> bool:
        self._sensor_monitor.close()
        return super().destroy_node()


def _arm_exit_watchdog(
    timeout_s: float, exit_code: int, traceback_path: Path | None = None
) -> threading.Timer:
    """Exit with ``exit_code`` unless cancelled within ``timeout_s``.

    Before exiting, every thread's stack is written to ``traceback_path`` so the
    cleanup step that blocked can be identified afterwards.
    """

    def expire() -> None:
        if traceback_path is not None:
            try:
                with open(traceback_path, "w", encoding="utf-8") as handle:
                    faulthandler.dump_traceback(file=handle, all_threads=True)
            except OSError:
                pass
        os._exit(exit_code)

    timer = threading.Timer(timeout_s, expire)
    timer.daemon = True
    timer.start()
    return timer


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--debug-dir", help="record every incoming Twist and base command here"
    )
    args = parser.parse_args(argv)
    config = load_config(Path(args.config).expanduser().resolve())
    output_dir = Path(args.output_dir).expanduser().resolve()
    result_path = output_dir / "control_result.json"
    ready_path = output_dir / "control_ready"
    command_log = (
        JsonlWriter(Path(args.debug_dir).expanduser() / "control_commands.jsonl")
        if args.debug_dir
        else None
    )

    node = None
    executor = None
    reason = "exception"
    stop_requested = threading.Event()

    def request_stop(signum: int, frame: Any) -> None:
        del signum, frame
        stop_requested.set()

    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    # A signal only sets a flag. Raising KeyboardInterrupt inside a base RPC
    # would leave its request socket waiting for a reply, and the zero command
    # sent during cleanup would then fail.
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    try:
        node = ApexNavControlRelay(config, command_log=command_log)
        executor = SingleThreadedExecutor()
        executor.add_node(node)
        ready_path.write_text("ready\n", encoding="utf-8")
        reason = "running"
        while (
            rclpy.ok()
            and not node.finished.is_set()
            and not stop_requested.is_set()
        ):
            executor.spin_once(timeout_sec=0.05)
        if node.finished.is_set():
            reason = "control_failure" if node.fatal_error else "finished"
        elif stop_requested.is_set():
            reason = "operator_interrupt"
    except KeyboardInterrupt:
        reason = "operator_interrupt"
    except Exception as exc:  # noqa: BLE001 - persist startup failures
        reason = "control_failure"
        result_path.write_text(
            json.dumps(
                {
                    "reason": reason,
                    "fatal_error": f"{type(exc).__name__}: {exc}",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    finally:
        watchdog = _arm_exit_watchdog(
            CLEANUP_TIMEOUT_S,
            1 if reason == "control_failure" else 0,
            output_dir / "control_cleanup_traceback.txt",
        )
        if node is not None:
            node.stop()
            # Persist the counters before any cleanup step that could block.
            result_path.write_text(
                json.dumps(node.stats(reason), indent=2), encoding="utf-8"
            )
            if command_log is not None:
                command_log.close(timeout_s=1.0)
            if executor is not None:
                executor.shutdown(timeout_sec=1.0)
                executor.remove_node(node)
            node.destroy_node()
        elif command_log is not None:
            command_log.close(timeout_s=1.0)
        if rclpy.ok():
            rclpy.shutdown()
        watchdog.cancel()
    return 1 if reason == "control_failure" else 0


if __name__ == "__main__":
    raise SystemExit(main())
