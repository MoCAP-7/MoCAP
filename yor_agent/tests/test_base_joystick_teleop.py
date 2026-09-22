"""The gamepad tool shares YOR's leased base RPC with the agent."""

from __future__ import annotations

import importlib.util
import math
import time
import unittest
from pathlib import Path

from tools.base_joystick_teleop import (
    CONTROLLER_MAPS,
    DEFAULT_DEAD_ZONE,
    LIMIT_MARGIN,
    GamepadSample,
    LeasedBaseTeleop,
    apply_dead_zone,
    check_rate_renews_lease,
    read_sample,
)

LEASED_BASE_PATH = (
    Path(__file__).resolve().parents[2] / "navdp" / "navdp_deploy" / "control" / "leased_base.py"
)


def leased_base_rpc_class():
    """The base service's own lease logic, loaded without the navdp package."""

    spec = importlib.util.spec_from_file_location("leased_base_for_gamepad_test", LEASED_BASE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.LeasedBaseRPC


class FakeBase:
    def __init__(self) -> None:
        self.velocities: list[list[float]] = []

    def set_base_velocity(self, velocity) -> None:
        self.velocities.append([float(value) for value in velocity])

    def get_cmd_vel(self):
        return (self.velocities[-1] if self.velocities else [0.0, 0.0, 0.0]), 0.0

    def get_base_encoders(self) -> dict:
        return {}


class AgentLikeClient:
    """Numbers its commands as the Jetson's hardware client does: wall-clock nanoseconds."""

    def __init__(self, service) -> None:
        self._service = service
        self._last = time.time_ns()

    def submit(self, velocity):
        self._last = max(time.time_ns(), self._last + 1)
        return self._service.submit_velocity(velocity, self._last)


class ServiceWithoutTeleopLimits:
    """A base service from before teleoperation limits existed."""

    def __init__(self, service) -> None:
        self._service = service

    def get_status(self) -> dict:
        status = self._service.get_status()
        status.pop("teleop_limits", None)
        return status

    def submit_velocity(self, velocity, sequence):
        return self._service.submit_velocity(velocity, sequence)


class LeasedBaseTeleopTest(unittest.TestCase):
    def setUp(self) -> None:
        self.now = 100.0
        self.base = FakeBase()
        self.service = leased_base_rpc_class()(
            self.base,
            lease_s=0.25,
            max_linear_mps=0.18,
            max_yaw_rad_s=0.35,
            clock=lambda: self.now,
            start_watchdog=False,
        )
        self.log: list[str] = []
        self.teleop = LeasedBaseTeleop(self.service, log=self.log.append)

    def test_nothing_is_sent_until_enabled_and_a_stick_leaves_the_dead_zone(self) -> None:
        self.assertEqual(self.teleop.step(GamepadSample(forward=1.0)), "idle")
        self.assertEqual(
            self.teleop.step(GamepadSample(start=True, forward=DEFAULT_DEAD_ZONE / 2)), "idle"
        )

        self.assertTrue(self.teleop.enabled)
        self.assertEqual(self.base.velocities, [])
        self.assertFalse(self.service.get_status()["lease_active"])

    def test_driving_renews_the_lease_and_letting_go_ends_it(self) -> None:
        self.teleop.step(GamepadSample(start=True))

        self.assertEqual(self.teleop.step(GamepadSample(forward=1.0)), "driving")
        status = self.service.get_status()
        self.assertTrue(status["lease_active"])
        self.assertEqual(status["last_sequence"], 0)
        self.assertAlmostEqual(self.base.velocities[-1][0], 0.18 * LIMIT_MARGIN)
        self.now += 0.05
        self.assertEqual(self.teleop.step(GamepadSample(forward=1.0)), "driving")
        self.assertEqual(self.service.get_status()["last_sequence"], 1)

        self.assertEqual(self.teleop.step(GamepadSample()), "released")
        self.assertEqual(self.base.velocities[-1], [0.0, 0.0, 0.0])
        self.assertFalse(self.service.get_status()["lease_active"])
        self.assertEqual(self.teleop.step(GamepadSample()), "idle")

    def test_commands_stay_within_the_advertised_limits_and_l1_halves_them(self) -> None:
        self.teleop.step(GamepadSample(start=True))

        self.teleop.step(GamepadSample(forward=1.0, left=1.0, turn_left=-1.0))

        vx, vy, yaw = self.base.velocities[-1]
        self.assertAlmostEqual(math.hypot(vx, vy), 0.18 * LIMIT_MARGIN)
        self.assertAlmostEqual(vx, vy)
        self.assertAlmostEqual(yaw, -0.35 * LIMIT_MARGIN)
        self.teleop.step(GamepadSample(slow_toggle=True, forward=1.0))
        self.assertAlmostEqual(self.base.velocities[-1][0], 0.18 * LIMIT_MARGIN * 0.5)
        # Holding L1 does not toggle it back.
        self.teleop.step(GamepadSample(slow_toggle=True, forward=1.0))
        self.assertTrue(self.teleop.slow)

    def test_it_does_not_start_while_another_controller_holds_the_lease(self) -> None:
        agent = AgentLikeClient(self.service)
        agent.submit([0.1, 0.0, 0.0])
        self.teleop.step(GamepadSample(start=True))

        self.assertEqual(self.teleop.step(GamepadSample(forward=-1.0)), "blocked")

        self.assertEqual(self.base.velocities, [[0.1, 0.0, 0.0]])
        self.assertIn("another controller holds the base lease", self.log[-1])
        # Once the other controller lets go, the gamepad drives.
        agent.submit([0.0, 0.0, 0.0])
        self.assertEqual(self.teleop.step(GamepadSample(forward=-1.0)), "driving")

    def test_a_latched_emergency_stop_blocks_driving(self) -> None:
        self.service.emergency_stop()
        self.teleop.step(GamepadSample(start=True))

        self.assertEqual(self.teleop.step(GamepadSample(forward=1.0)), "blocked")

        self.assertIn("emergency stop is latched", self.log[-1])

    def test_it_stops_when_another_controller_commands_while_it_drives(self) -> None:
        agent = AgentLikeClient(self.service)
        self.teleop.step(GamepadSample(start=True))
        self.assertEqual(self.teleop.step(GamepadSample(forward=1.0)), "driving")

        agent.submit([0.0, 0.1, 0.0])

        self.assertEqual(self.teleop.step(GamepadSample(forward=1.0)), "refused")
        self.assertFalse(self.teleop.driving)
        self.assertEqual(self.base.velocities[-1], [0.0, 0.1, 0.0])
        self.assertIn("refused", self.log[-1])
        # The other controller keeps the base; the gamepad waits for it.
        self.assertEqual(self.teleop.step(GamepadSample(forward=1.0)), "blocked")
        self.assertTrue(agent.submit([0.0, 0.1, 0.0])["accepted"])

    def test_the_agent_is_never_refused_after_the_gamepad_drove(self) -> None:
        agent = AgentLikeClient(self.service)
        agent.submit([0.1, 0.0, 0.0])
        agent.submit([0.0, 0.0, 0.0])
        self.teleop.step(GamepadSample(start=True))

        for _ in range(200):
            self.assertEqual(self.teleop.step(GamepadSample(turn_left=1.0)), "driving")
        self.teleop.step(GamepadSample())

        # The gamepad continued the service's numbering one by one, so the
        # agent's next wall-clock sequence is still ahead of it.
        self.assertTrue(agent.submit([0.05, 0.0, 0.0])["accepted"])

    def test_back_and_close_let_go_of_the_base(self) -> None:
        self.teleop.step(GamepadSample(start=True))
        self.teleop.step(GamepadSample(forward=1.0))

        self.assertEqual(self.teleop.step(GamepadSample(back=True, forward=1.0)), "released")
        self.assertFalse(self.teleop.enabled)
        self.assertFalse(self.service.get_status()["lease_active"])

        self.teleop.step(GamepadSample(start=True))
        self.teleop.step(GamepadSample(forward=1.0))
        self.teleop.close()
        self.assertEqual(self.base.velocities[-1], [0.0, 0.0, 0.0])
        self.assertFalse(self.service.get_status()["lease_active"])

    def test_a_stalled_loop_leaves_the_stop_to_the_lease(self) -> None:
        self.teleop.step(GamepadSample(start=True))
        self.teleop.step(GamepadSample(forward=1.0))

        self.now += 0.30

        self.assertTrue(self.service.expire_if_needed())
        self.assertEqual(self.base.velocities[-1], [0.0, 0.0, 0.0])


class TeleopLimitsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.now = 100.0
        self.base = FakeBase()
        self.service = leased_base_rpc_class()(
            self.base,
            lease_s=0.25,
            max_linear_mps=0.18,
            max_yaw_rad_s=0.35,
            teleop_max_linear_mps=0.50,
            teleop_max_yaw_rad_s=1.57,
            clock=lambda: self.now,
            start_watchdog=False,
        )

    def test_the_gamepad_drives_at_the_teleoperation_limits(self) -> None:
        teleop = LeasedBaseTeleop(self.service, log=lambda _message: None)
        teleop.step(GamepadSample(start=True))

        self.assertEqual(teleop.step(GamepadSample(forward=1.0, turn_left=1.0)), "driving")

        self.assertTrue(teleop.uses_teleop_limits)
        vx, _vy, yaw = self.base.velocities[-1]
        self.assertAlmostEqual(vx, 0.50 * LIMIT_MARGIN)
        self.assertAlmostEqual(yaw, 1.57 * LIMIT_MARGIN)
        self.assertEqual(teleop.step(GamepadSample()), "released")
        self.assertFalse(self.service.get_status()["lease_active"])

    def test_autonomous_controllers_keep_the_shared_limits(self) -> None:
        agent = AgentLikeClient(self.service)

        with self.assertRaises(ValueError):
            agent.submit([0.50 * LIMIT_MARGIN, 0.0, 0.0])
        with self.assertRaises(ValueError):
            agent.submit([0.0, 0.0, 1.0])

        self.assertTrue(agent.submit([0.18, 0.0, 0.35])["accepted"])
        limits = self.service.get_status()["limits"]
        self.assertEqual((limits["max_linear_mps"], limits["max_yaw_rad_s"]), (0.18, 0.35))

    def test_the_gamepad_gets_its_acceleration_and_the_agent_gets_its_own_back(self) -> None:
        applied: list[list[float]] = []
        service = leased_base_rpc_class()(
            self.base,
            lease_s=0.25,
            max_linear_mps=0.18,
            max_yaw_rad_s=0.35,
            teleop_max_linear_mps=0.50,
            teleop_max_yaw_rad_s=1.57,
            max_accel=[0.30, 0.30, 0.80],
            teleop_max_accel=[1.9, 1.9, 6.5],
            set_max_accel=lambda accel: applied.append(accel.tolist()),
            clock=lambda: self.now,
            start_watchdog=False,
        )
        teleop = LeasedBaseTeleop(service, log=lambda _message: None)
        teleop.step(GamepadSample(start=True))

        teleop.step(GamepadSample(forward=1.0))
        self.assertEqual(applied[-1], [1.9, 1.9, 6.5])
        teleop.step(GamepadSample())
        AgentLikeClient(service).submit([0.1, 0.0, 0.0])

        self.assertEqual(applied, [[0.30, 0.30, 0.80], [1.9, 1.9, 6.5], [0.30, 0.30, 0.80]])

    def test_an_older_base_service_caps_the_gamepad_at_its_shared_limits(self) -> None:
        teleop = LeasedBaseTeleop(
            ServiceWithoutTeleopLimits(self.service), log=lambda _message: None
        )
        teleop.step(GamepadSample(start=True))

        self.assertEqual(teleop.step(GamepadSample(forward=1.0)), "driving")

        self.assertFalse(teleop.uses_teleop_limits)
        self.assertAlmostEqual(self.base.velocities[-1][0], 0.18 * LIMIT_MARGIN)


class HelpersTest(unittest.TestCase):
    def test_the_dead_zone_zeroes_small_deflections_and_rescales_the_rest(self) -> None:
        self.assertEqual(apply_dead_zone(0.05, 0.08), 0.0)
        self.assertAlmostEqual(apply_dead_zone(1.0, 0.08), 1.0)
        self.assertAlmostEqual(apply_dead_zone(-0.54, 0.08), -0.5)
        self.assertAlmostEqual(apply_dead_zone(1.7, 0.08), 1.0)

    def test_the_loop_must_renew_the_lease_twice_per_lease(self) -> None:
        check_rate_renews_lease(20.0, 0.25)
        check_rate_renews_lease(8.0, 0.25)
        for bad in (5.0, 0.0, float("nan"), float("inf")):
            with self.subTest(rate=bad), self.assertRaises(ValueError):
                check_rate_renews_lease(bad, 0.25)

    def test_invalid_advertised_limits_are_refused(self) -> None:
        class BadRpc:
            def get_status(self) -> dict:
                return {"limits": {"lease_s": 0.25, "max_linear_mps": 0.0, "max_yaw_rad_s": 0.35}}

        with self.assertRaises(ValueError):
            LeasedBaseTeleop(BadRpc())

    def test_sticks_are_read_so_forward_left_and_a_left_turn_are_positive(self) -> None:
        class Pad:
            axes = {4: -1.0, 3: -0.5, 0: -0.25}
            buttons = {7: 1, 6: 0, 4: 1}

            def get_axis(self, index: int) -> float:
                return self.axes.get(index, 0.0)

            def get_button(self, index: int) -> int:
                return self.buttons.get(index, 0)

        sample = read_sample(Pad(), CONTROLLER_MAPS["xbox"])

        self.assertEqual(
            sample,
            GamepadSample(
                start=True, back=False, slow_toggle=True, forward=1.0, left=0.5, turn_left=0.25
            ),
        )


if __name__ == "__main__":
    unittest.main()
