#!/usr/bin/env python3
"""Drive YOR's base with a gamepad through the leased base RPC, beside the services.

``robot/teleop/joystick.py`` is a client of ``robot/yor.py``, which opens the
base motors, both arm CAN buses and RPC port 5557 itself, so it only runs
with the base and nero services stopped (and restarting nero homes the arms
again). This tool is a client of the base service
(``yor_agent/services/base/service.py``) instead: the arm service keeps its
arms, and the agent can start its next run as soon as the base is back.

How it shares the base with the agent:

- It drives only while enabled (Start enables, Back disables) and a stick is
  outside its dead zone. Every command renews the service's short lease;
  letting go of the sticks sends one zero command, which ends the lease at
  once. If the loop stalls, the lease runs out and the service stops the
  base.
- It does not start driving while another controller holds the lease or the
  emergency stop is latched. The agent likewise refuses to start a motion
  while a lease is active.
- The service accepts only increasing sequence numbers, across all of its
  clients. This tool numbers its commands from the service's last sequence
  rather than from a clock: the agent numbers its commands by its own
  wall-clock nanoseconds, which a sequence taken from a clock running ahead
  could overtake, and the service would then refuse the agent's commands.
  When another controller commands the base while this tool is driving, the
  service refuses this tool's next command and the tool stops driving.
- Speeds are fractions of the teleoperation limits the service advertises
  (``teleop_limits`` in ``get_status``), which its ``submit_teleop_velocity``
  checks; the limits autonomous controllers are held to stay as they are.
  Against a base service without teleoperation limits the tool uses
  ``submit_velocity`` and those shared limits. L1 toggles half speed. The
  lift is not available: the leased RPC does not expose it.

``start_services.sh pi`` starts it in the ``gamepad`` tmux session next to
the base and nero services (``YOR_START_GAMEPAD=0`` skips it). To run it by
hand on the Raspberry Pi, where the gamepad's receiver is plugged in, while
the base service is running::

    cd /home/cone-e2/YOR
    /home/cone-e2/miniconda3/envs/yor-nero/bin/python yor_agent/tools/base_joystick_teleop.py

Run one instance at a time: two would read the same gamepad and both command
the base.
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from typing import Any, Callable

#: Button and axis indices, as in ``robot/teleop/joystick.py``.
CONTROLLER_MAPS = {
    "xbox": {
        "start": 7,
        "back": 6,
        "l1": 4,
        "left_horizontal_axis": 0,
        "right_horizontal_axis": 3,
        "right_vertical_axis": 4,
    },
    "ps4": {
        "start": 3,
        "back": 0,
        "l1": 4,
        "left_horizontal_axis": 0,
        "right_horizontal_axis": 3,
        "right_vertical_axis": 4,
    },
}
DEFAULT_RPC_PORT = 5557
DEFAULT_RATE_HZ = 20.0
DEFAULT_DEAD_ZONE = 0.08
DEFAULT_SLOW_SCALE = 0.5
#: Commands stay this fraction of the advertised limits, so rounding never
#: takes one over a limit the service would refuse.
LIMIT_MARGIN = 0.98


@dataclass(frozen=True)
class GamepadSample:
    """One gamepad reading: buttons held and stick deflections in [-1, 1].

    ``forward``, ``left`` and ``turn_left`` are positive for moving forward,
    moving left and turning left.
    """

    start: bool = False
    back: bool = False
    slow_toggle: bool = False
    forward: float = 0.0
    left: float = 0.0
    turn_left: float = 0.0


def apply_dead_zone(value: float, dead_zone: float) -> float:
    """Zero inside the dead zone and rescale the rest back onto [-1, 1]."""

    value = max(-1.0, min(1.0, float(value)))
    if abs(value) <= dead_zone:
        return 0.0
    return math.copysign((abs(value) - dead_zone) / (1.0 - dead_zone), value)


def check_rate_renews_lease(rate_hz: float, lease_s: float) -> None:
    """Refuse a loop that cannot renew the lease at least twice per lease."""

    if not (math.isfinite(rate_hz) and rate_hz > 0.0):
        raise ValueError("the loop rate must be a positive number of hertz")
    period = 1.0 / rate_hz
    if period > lease_s / 2.0:
        raise ValueError(
            f"{rate_hz:g} Hz sends a command every {period:.3f} s, too slow to "
            f"renew the base service's {lease_s:.3f} s lease twice per lease"
        )


def _first_line(exc: BaseException) -> str:
    # commlink's RPCException carries the server's traceback in str().
    message = getattr(exc, "message", None) or str(exc)
    return str(message).strip().splitlines()[0] if str(message).strip() else type(exc).__name__


class LeasedBaseTeleop:
    """Gamepad-to-velocity logic over a leased base RPC client (no pygame here)."""

    def __init__(
        self,
        rpc: Any,
        *,
        dead_zone: float = DEFAULT_DEAD_ZONE,
        slow_scale: float = DEFAULT_SLOW_SCALE,
        log: Callable[[str], None] = print,
    ) -> None:
        if not 0.0 <= dead_zone < 1.0:
            raise ValueError("dead_zone must be in [0, 1)")
        if not 0.0 < slow_scale <= 1.0:
            raise ValueError("slow_scale must be in (0, 1]")
        self._rpc = rpc
        self._log = log
        self.dead_zone = float(dead_zone)
        self.slow_scale = float(slow_scale)
        status = rpc.get_status()
        limits = dict(status["limits"])
        teleop_limits = status.get("teleop_limits")
        #: Whether commands go through the service's teleoperation limits
        #: (``submit_teleop_velocity``) rather than the shared ones.
        self.uses_teleop_limits = isinstance(teleop_limits, dict)
        if self.uses_teleop_limits:
            limits.update(teleop_limits)
        self.lease_s = float(limits["lease_s"])
        self.max_linear_mps = float(limits["max_linear_mps"])
        self.max_yaw_rad_s = float(limits["max_yaw_rad_s"])
        if not all(
            math.isfinite(value) and value > 0.0
            for value in (self.lease_s, self.max_linear_mps, self.max_yaw_rad_s)
        ):
            raise ValueError(f"the base service advertised invalid limits: {limits}")
        self.enabled = False
        self.slow = False
        self.driving = False
        self._sequence: int | None = None
        self._previous_slow_toggle = False
        self._last_note: str | None = None

    def command(self, sample: GamepadSample) -> list[float]:
        """The velocity ``[vx, vy, yaw_rate]`` the sample asks for, within the limits."""

        forward = apply_dead_zone(sample.forward, self.dead_zone)
        left = apply_dead_zone(sample.left, self.dead_zone)
        turn = apply_dead_zone(sample.turn_left, self.dead_zone)
        norm = math.hypot(forward, left)
        if norm > 1.0:
            forward, left = forward / norm, left / norm
        scale = LIMIT_MARGIN * (self.slow_scale if self.slow else 1.0)
        return [
            forward * self.max_linear_mps * scale,
            left * self.max_linear_mps * scale,
            turn * self.max_yaw_rad_s * scale,
        ]

    def step(self, sample: GamepadSample) -> str:
        """Act on one reading.

        Returns what happened: ``idle``, ``driving``, ``released`` (a zero
        command ended this tool's lease), ``blocked`` (another controller
        holds the lease or the emergency stop is latched) or ``refused`` (the
        service refused a command, so this tool stopped driving).
        """

        if sample.back and self.enabled:
            self.enabled = False
            self._log("control disabled (Start enables)")
        elif sample.start and not self.enabled:
            self.enabled = True
            self._log("control enabled (Back disables)")
        if sample.slow_toggle and not self._previous_slow_toggle:
            self.slow = not self.slow
            self._log("half speed" if self.slow else "full speed")
        self._previous_slow_toggle = sample.slow_toggle

        velocity = self.command(sample) if self.enabled else [0.0, 0.0, 0.0]
        if not any(velocity):
            if self.driving:
                self.release()
                return "released"
            self._last_note = None
            return "idle"
        if not self.driving:
            reason = self._start_refusal()
            if reason is not None:
                self._note(f"not driving: {reason}")
                return "blocked"
        try:
            self._submit(velocity)
        except Exception as exc:  # noqa: BLE001 - any refusal ends this drive
            self.driving = False
            self._sequence = None
            self._note(
                "stopped driving: the base service refused the command "
                f"({_first_line(exc)})"
            )
            return "refused"
        self.driving = True
        self._last_note = None
        return "driving"

    def release(self) -> None:
        """Send one zero command, which ends this tool's lease at once."""

        if not self.driving:
            return
        self.driving = False
        try:
            self._submit([0.0, 0.0, 0.0])
        except Exception as exc:  # noqa: BLE001 - the lease runs out on its own
            self._note(
                f"zero command refused ({_first_line(exc)}); the lease runs out on its own"
            )
        self._sequence = None

    def close(self) -> None:
        self.release()

    def _start_refusal(self) -> str | None:
        status = self._rpc.get_status()
        if status.get("estop_latched", False):
            return "the base emergency stop is latched"
        if status.get("lease_active", False):
            return "another controller holds the base lease"
        self._sequence = int(status["last_sequence"])
        return None

    def _submit(self, velocity: list[float]) -> None:
        if self._sequence is None:
            raise RuntimeError("no sequence to continue from")
        sequence = self._sequence + 1
        submit = (
            self._rpc.submit_teleop_velocity
            if self.uses_teleop_limits
            else self._rpc.submit_velocity
        )
        reply = submit(velocity, sequence)
        if not isinstance(reply, dict) or not reply.get("accepted", False):
            raise RuntimeError(f"unexpected reply {reply!r}")
        self._sequence = sequence

    def _note(self, message: str) -> None:
        if message != self._last_note:
            self._log(message)
            self._last_note = message


def read_sample(joystick: Any, mapping: dict[str, int]) -> GamepadSample:
    """Read one sample; stick axes are inverted so forward, left and a left turn are positive."""

    return GamepadSample(
        start=bool(joystick.get_button(mapping["start"])),
        back=bool(joystick.get_button(mapping["back"])),
        slow_toggle=bool(joystick.get_button(mapping["l1"])),
        forward=-float(joystick.get_axis(mapping["right_vertical_axis"])),
        left=-float(joystick.get_axis(mapping["right_horizontal_axis"])),
        turn_left=-float(joystick.get_axis(mapping["left_horizontal_axis"])),
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Drive YOR's base with a gamepad through the leased base RPC, "
            "beside the running base and arm services."
        )
    )
    parser.add_argument("--host", default="localhost", help="base RPC host (default: %(default)s)")
    parser.add_argument(
        "--port", type=int, default=DEFAULT_RPC_PORT, help="base RPC port (default: %(default)s)"
    )
    parser.add_argument(
        "--controller",
        choices=sorted(CONTROLLER_MAPS),
        default="xbox",
        help="button layout (default: %(default)s)",
    )
    parser.add_argument(
        "--rate-hz",
        type=float,
        default=DEFAULT_RATE_HZ,
        help="command rate; must renew the service's lease twice per lease (default: %(default)s)",
    )
    parser.add_argument(
        "--dead-zone", type=float, default=DEFAULT_DEAD_ZONE, help="stick dead zone (default: %(default)s)"
    )
    parser.add_argument(
        "--slow-scale",
        type=float,
        default=DEFAULT_SLOW_SCALE,
        help="fraction of the limits at half speed (L1) (default: %(default)s)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    import pygame
    from commlink import RPCClient

    pygame.init()
    pygame.joystick.init()
    if pygame.joystick.get_count() < 1:
        raise SystemExit("no gamepad detected; plug its receiver into this computer")
    joystick = pygame.joystick.Joystick(0)
    joystick.init()

    print(
        f"connecting to the base RPC at {args.host}:{args.port} "
        "(start the base service if this does not return)",
        flush=True,
    )
    rpc = RPCClient(host=args.host, port=args.port)
    try:
        teleop = LeasedBaseTeleop(rpc, dead_zone=args.dead_zone, slow_scale=args.slow_scale)
        check_rate_renews_lease(args.rate_hz, teleop.lease_s)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    source = (
        "the base service's teleoperation limits"
        if teleop.uses_teleop_limits
        else "the base service has no teleoperation limits, so its shared limits"
    )
    print(
        f"gamepad {joystick.get_name()!r}: up to {teleop.max_linear_mps:.2f} m/s and "
        f"{teleop.max_yaw_rad_s:.2f} rad/s ({source}), "
        f"{args.rate_hz:g} Hz over a {teleop.lease_s:.2f} s lease\n"
        "Start enables, Back disables; right stick moves, left stick turns, "
        "L1 toggles half speed; Ctrl+C quits",
        flush=True,
    )

    mapping = CONTROLLER_MAPS[args.controller]
    period = 1.0 / args.rate_hz
    deadline = time.monotonic()
    try:
        while True:
            pygame.event.pump()
            teleop.step(read_sample(joystick, mapping))
            deadline += period
            delay = deadline - time.monotonic()
            if delay > 0.0:
                time.sleep(delay)
            else:
                deadline = time.monotonic()
    except KeyboardInterrupt:
        print("stopping", flush=True)
    finally:
        teleop.close()
        pygame.quit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
