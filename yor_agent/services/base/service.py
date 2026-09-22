#!/usr/bin/env python3
"""Run the leased, base-only NavDP RPC service on YOR's Raspberry Pi."""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
NAVDP_ROOT = REPO_ROOT / "navdp"
for path in (REPO_ROOT, NAVDP_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

DEFAULT_RPC_PORT = 5557
#: Base acceleration limits (m/s^2, m/s^2, rad/s^2) for autonomous controllers.
DEFAULT_MAX_ACCEL = (0.30, 0.30, 0.80)


def _accelerations(text: str) -> list[float]:
    """``"0.3,0.3,0.8"`` -> ``[0.3, 0.3, 0.8]``: x, y and yaw accelerations."""

    try:
        values = [float(item) for item in str(text).split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"expected three comma-separated accelerations, got {text!r}"
        ) from exc
    if len(values) != 3 or not all(math.isfinite(v) and v > 0.0 for v in values):
        raise argparse.ArgumentTypeError(
            f"expected three positive accelerations (x, y, yaw), got {text!r}"
        )
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Leased base-only RPC service for supervised NavDP tests"
    )
    parser.add_argument("--port", type=int, default=DEFAULT_RPC_PORT)
    parser.add_argument("--lease-ms", type=float, default=250.0)
    parser.add_argument("--max-linear-mps", type=float, default=0.18)
    parser.add_argument("--max-yaw-rad-s", type=float, default=0.35)
    parser.add_argument(
        "--teleop-max-linear-mps",
        type=float,
        default=None,
        help=(
            "linear limit of submit_teleop_velocity, for an operator driving by "
            "hand (default: --max-linear-mps)"
        ),
    )
    parser.add_argument(
        "--teleop-max-yaw-rad-s",
        type=float,
        default=None,
        help="yaw limit of submit_teleop_velocity (default: --max-yaw-rad-s)",
    )
    parser.add_argument(
        "--max-accel",
        type=_accelerations,
        default=list(DEFAULT_MAX_ACCEL),
        metavar="AX,AY,AYAW",
        help=(
            "base S-curve acceleration limits while submit_velocity drives "
            "(default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--teleop-max-accel",
        type=_accelerations,
        default=None,
        metavar="AX,AY,AYAW",
        help=(
            "base S-curve acceleration limits while submit_teleop_velocity drives "
            "(default: --max-accel)"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not Path("/sys/class/net/can0").exists():
        raise SystemExit("can0 not found; run this service on the Raspberry Pi")

    try:
        from commlink import RPCServer
    except ImportError:
        raise SystemExit(
            "commlink is unavailable in this interpreter; on the YOR "
            "Raspberry Pi use /home/cone-e2/miniconda3/envs/yor-nero/bin/python"
        )

    from navdp_deploy.control import LeasedBaseRPC
    from robot.yor import YOR

    teleop_linear = (
        args.max_linear_mps
        if args.teleop_max_linear_mps is None
        else args.teleop_max_linear_mps
    )
    teleop_yaw = (
        args.max_yaw_rad_s if args.teleop_max_yaw_rad_s is None else args.teleop_max_yaw_rad_s
    )
    teleop_accel = args.max_accel if args.teleop_max_accel is None else args.teleop_max_accel
    yor = YOR(no_arms=True)
    yor.base._a_max = np.array(args.max_accel, dtype=float)
    yor.init()

    def set_max_accel(accel: np.ndarray) -> None:
        # The base reads these whenever it starts a new S-curve velocity segment.
        yor.base._a_max = np.array(accel, dtype=float)

    api = LeasedBaseRPC(
        yor,
        lease_s=args.lease_ms / 1000.0,
        max_linear_mps=args.max_linear_mps,
        max_yaw_rad_s=args.max_yaw_rad_s,
        teleop_max_linear_mps=teleop_linear,
        teleop_max_yaw_rad_s=teleop_yaw,
        max_accel=args.max_accel,
        teleop_max_accel=teleop_accel,
        set_max_accel=set_max_accel,
    )
    server = RPCServer(api, port=args.port, threaded=True)
    server.start()
    print(
        "[navdp-pi] leased base RPC started "
        f"port={args.port} lease={args.lease_ms:.0f}ms "
        f"linear<={args.max_linear_mps:.3f}m/s "
        f"yaw<={args.max_yaw_rad_s:.3f}rad/s "
        f"accel<={','.join(f'{value:g}' for value in args.max_accel)} "
        f"teleop linear<={teleop_linear:.3f}m/s "
        f"teleop yaw<={teleop_yaw:.3f}rad/s "
        f"teleop accel<={','.join(f'{value:g}' for value in teleop_accel)}",
        flush=True,
    )
    print("[navdp-pi] no arms initialized; Ctrl+C stops and zeros the base", flush=True)
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("[navdp-pi] stopping", flush=True)
    finally:
        try:
            api.close()
            time.sleep(0.3)
        finally:
            server.stop()
            yor.base_controller.stop()
            yor.base.stop_control()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
