"""Convert the final Nav2-safe Twist into YOR's existing leased base RPC."""

from __future__ import annotations

import math
import threading
import time

from geometry_msgs.msg import Twist
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2
from std_srvs.srv import Trigger

# A command whose magnitude is at or below this is a stop request, not a slow
# motion request, and is never rounded up to the drivetrain floor.
_MOTION_EPSILON = 1e-3


class BaseBridge(Node):
    def __init__(self) -> None:
        super().__init__("yor_base_bridge")
        self.declare_parameter("rpc_host", "192.168.1.10")
        self.declare_parameter("rpc_port", 5557)
        self.declare_parameter("cmd_vel_topic", "/yor/cmd_vel_safe")
        self.declare_parameter("command_timeout_s", 0.15)
        self.declare_parameter("sensor_timeout_s", 0.50)
        self.declare_parameter("halt_duration_s", 0.50)
        self.declare_parameter("publish_hz", 20.0)
        self.declare_parameter("max_linear_mps", 0.18)
        self.declare_parameter("max_yaw_rad_s", 0.35)
        # Static-friction floor of the YOR drivetrain. A command under it is
        # accepted by the Pi and produces no motion, so a planner asking for a
        # slow correction gets nothing. Rounding up here, at the last point
        # before the RPC, lets the planner sample freely while the wheels still
        # see a command they can execute; below the epsilon a command is still
        # exactly zero, so "stop" remains "stop".
        self.declare_parameter("min_linear_mps", 0.05)
        self.declare_parameter("min_yaw_rad_s", 0.20)
        from commlink import RPCClient

        self._rpc = RPCClient(
            host=str(self.get_parameter("rpc_host").value),
            port=int(self.get_parameter("rpc_port").value),
        )
        status = self._rpc.get_status()
        if not isinstance(status, dict):
            raise RuntimeError("YOR base RPC returned an invalid status")
        if status.get("estop_latched", False):
            raise RuntimeError("Pi base emergency stop is latched")
        if status.get("lease_active", False):
            # A killed/restarted bridge can leave at most one 250 ms lease
            # behind.  Let that expire, while still rejecting a controller
            # that is actively renewing ownership.
            remaining_s = float(status.get("lease_remaining_s", 0.25))
            time.sleep(min(0.35, max(0.05, remaining_s + 0.05)))
            status = self._rpc.get_status()
            if not isinstance(status, dict):
                raise RuntimeError("YOR base RPC returned an invalid status")
            if status.get("estop_latched", False):
                raise RuntimeError("Pi base emergency stop is latched")
            if status.get("lease_active", False):
                raise RuntimeError(
                    "Pi base lease is already active; stop the other base "
                    "controller before starting the ROS bridge"
                )
        self._timeout_s = float(self.get_parameter("command_timeout_s").value)
        self._sensor_timeout_s = float(
            self.get_parameter("sensor_timeout_s").value
        )
        self._halt_duration_s = float(
            self.get_parameter("halt_duration_s").value
        )
        publish_hz = float(self.get_parameter("publish_hz").value)
        self._max_linear = float(self.get_parameter("max_linear_mps").value)
        self._max_yaw = float(self.get_parameter("max_yaw_rad_s").value)
        self._min_linear = float(self.get_parameter("min_linear_mps").value)
        self._min_yaw = float(self.get_parameter("min_yaw_rad_s").value)
        if self._min_linear > self._max_linear or self._min_yaw > self._max_yaw:
            raise ValueError("base bridge velocity floors must not exceed the limits")
        if min(
            self._timeout_s,
            self._sensor_timeout_s,
            self._halt_duration_s,
            publish_hz,
            self._max_linear,
            self._max_yaw,
        ) <= 0.0:
            raise ValueError("base bridge timing and velocity limits must be positive")
        self._lock = threading.Lock()
        self._velocity = [0.0, 0.0, 0.0]
        self._received_at = 0.0
        self._sensor_received_at = 0.0
        self._hold_until = 0.0
        self._stop_latched = False
        self._command_active = False
        self._zero_pending = False
        self._sequence = time.time_ns()
        topic = str(self.get_parameter("cmd_vel_topic").value)
        self.create_subscription(Twist, topic, self._on_twist, 10)
        self.create_subscription(
            PointCloud2,
            "/yor/zed/points",
            self._on_sensor,
            qos_profile_sensor_data,
        )
        self.create_service(Trigger, "~/stop", self._on_stop)
        self.create_service(Trigger, "~/halt", self._on_halt)
        self.create_service(Trigger, "~/clear_stop", self._on_clear_stop)
        self.create_timer(1.0 / publish_hz, self._renew_lease)
        self.get_logger().info(f"bridging {topic} to the leased Pi base RPC")

    def _on_twist(self, message: Twist) -> None:
        command = [
            float(message.linear.x),
            float(message.linear.y),
            float(message.angular.z),
        ]
        if not all(math.isfinite(value) for value in command):
            self.get_logger().error("discarded non-finite Twist")
            return
        linear = math.hypot(command[0], command[1])
        if linear > self._max_linear:
            scale = self._max_linear / linear
            command[0] *= scale
            command[1] *= scale
        command[2] = max(-self._max_yaw, min(self._max_yaw, command[2]))
        # Raise a below-deadband request to the floor rather than dropping it.
        # The loop closes at publish_hz, so one cycle of a rounded-up command
        # moves the base by a fraction of a degree; a request that is exactly
        # zero stays zero.
        if _MOTION_EPSILON < abs(command[2]) < self._min_yaw:
            command[2] = math.copysign(self._min_yaw, command[2])
        linear = math.hypot(command[0], command[1])
        if _MOTION_EPSILON < linear < self._min_linear:
            scale = self._min_linear / linear
            command[0] *= scale
            command[1] *= scale
        with self._lock:
            if not self._stop_latched and time.monotonic() >= self._hold_until:
                is_nonzero = any(abs(value) > 1e-9 for value in command)
                if is_nonzero:
                    self._velocity = command
                    self._received_at = time.monotonic()
                    self._command_active = True
                else:
                    if self._command_active or any(
                        abs(value) > 1e-9 for value in self._velocity
                    ):
                        self._zero_pending = True
                    self._velocity = [0.0, 0.0, 0.0]
                    self._received_at = 0.0
                    self._command_active = False

    def _on_sensor(self, _message: PointCloud2) -> None:
        with self._lock:
            self._sensor_received_at = time.monotonic()

    def _renew_lease(self) -> None:
        with self._lock:
            now = time.monotonic()
            stale = now - self._received_at > self._timeout_s
            sensor_stale = (
                now - self._sensor_received_at > self._sensor_timeout_s
            )
            held = now < self._hold_until
            if self._command_active and (
                stale or sensor_stale or held or self._stop_latched
            ):
                self._command_active = False
                self._velocity = [0.0, 0.0, 0.0]
                self._received_at = 0.0
                self._zero_pending = True
            if self._command_active:
                command = list(self._velocity)
            elif self._zero_pending:
                command = [0.0, 0.0, 0.0]
                self._zero_pending = False
            else:
                # A zero command releases the Pi lease.  Remaining silent
                # while idle lets the agent's direct motion primitives own the
                # same RPC without sequence races or zero-command overrides.
                return
        try:
            self._send(command)
        except Exception as exc:  # noqa: BLE001 - Pi watchdog independently zeros
            self.get_logger().error(f"base RPC command failed: {exc}")

    def _send(self, velocity: list[float]) -> None:
        self._sequence = max(time.time_ns(), self._sequence + 1)
        reply = self._rpc.submit_velocity(velocity, self._sequence)
        if not isinstance(reply, dict) or not reply.get("accepted", False):
            raise RuntimeError(f"base RPC rejected velocity: {reply!r}")

    def _on_stop(self, _request: Trigger.Request, response: Trigger.Response):
        with self._lock:
            self._stop_latched = True
            self._velocity = [0.0, 0.0, 0.0]
            self._received_at = 0.0
            self._command_active = False
            self._zero_pending = False
        try:
            self._send([0.0, 0.0, 0.0])
            response.success = True
            response.message = "ROS base bridge stopped and latched"
        except Exception as exc:  # noqa: BLE001
            response.success = False
            response.message = str(exc)
        return response

    def _on_halt(
        self, _request: Trigger.Request, response: Trigger.Response
    ) -> Trigger.Response:
        with self._lock:
            self._velocity = [0.0, 0.0, 0.0]
            self._received_at = 0.0
            self._hold_until = time.monotonic() + self._halt_duration_s
            self._command_active = False
            self._zero_pending = False
        try:
            self._send([0.0, 0.0, 0.0])
            response.success = True
            response.message = "cached Twist discarded; temporary zero hold active"
        except Exception as exc:  # noqa: BLE001
            response.success = False
            response.message = str(exc)
        return response

    def _on_clear_stop(
        self, _request: Trigger.Request, response: Trigger.Response
    ) -> Trigger.Response:
        with self._lock:
            self._velocity = [0.0, 0.0, 0.0]
            self._received_at = 0.0
            self._hold_until = 0.0
            self._stop_latched = False
            self._command_active = False
            self._zero_pending = False
        response.success = True
        response.message = "ROS base bridge stop latch cleared; awaiting fresh Twist"
        return response

    def destroy_node(self) -> bool:
        try:
            self._send([0.0, 0.0, 0.0])
        except Exception:  # noqa: BLE001
            pass
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = BaseBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
