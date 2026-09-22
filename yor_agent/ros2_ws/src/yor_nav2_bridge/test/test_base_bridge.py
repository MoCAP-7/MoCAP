import threading
import time

from geometry_msgs.msg import Twist

from yor_nav2_bridge.base_bridge import BaseBridge


class _Logger:
    def error(self, _message: str) -> None:
        pass


def _bridge_without_ros() -> tuple[BaseBridge, list[list[float]]]:
    bridge = BaseBridge.__new__(BaseBridge)
    bridge._lock = threading.Lock()
    bridge._timeout_s = 0.15
    bridge._sensor_timeout_s = 0.50
    bridge._hold_until = 0.0
    bridge._stop_latched = False
    bridge._velocity = [0.0, 0.0, 0.0]
    bridge._received_at = 0.0
    bridge._sensor_received_at = time.monotonic()
    bridge._command_active = False
    bridge._zero_pending = False
    bridge._max_linear = 0.18
    bridge._max_yaw = 0.35
    sent: list[list[float]] = []
    bridge._send = lambda velocity: sent.append(list(velocity))
    bridge.get_logger = lambda: _Logger()
    return bridge, sent


def _twist(x_mps: float = 0.0, yaw_rad_s: float = 0.0) -> Twist:
    message = Twist()
    message.linear.x = x_mps
    message.angular.z = yaw_rad_s
    return message


def test_idle_bridge_does_not_write_zero_commands_to_shared_rpc() -> None:
    bridge, sent = _bridge_without_ros()

    bridge._renew_lease()
    bridge._renew_lease()

    assert sent == []


def test_zero_twist_releases_once_then_bridge_remains_silent() -> None:
    bridge, sent = _bridge_without_ros()
    bridge._on_twist(_twist(x_mps=0.10))
    bridge._renew_lease()
    bridge._on_twist(_twist())

    bridge._renew_lease()
    bridge._on_twist(_twist())
    bridge._renew_lease()

    assert sent == [[0.10, 0.0, 0.0], [0.0, 0.0, 0.0]]


def test_stale_active_command_releases_once_then_bridge_remains_silent() -> None:
    bridge, sent = _bridge_without_ros()
    bridge._on_twist(_twist(yaw_rad_s=0.20))
    bridge._received_at = time.monotonic() - 1.0

    bridge._renew_lease()
    bridge._renew_lease()

    assert sent == [[0.0, 0.0, 0.0]]
