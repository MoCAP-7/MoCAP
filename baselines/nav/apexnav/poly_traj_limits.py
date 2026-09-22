"""Check ApexNav PolyTraj trajectories against platform limits and time-scale them.

The planner's YOR patch retimes each optimized trajectory along its own path to
the limits, and slows it uniformly in time if the retiming fails
(trajectory_manager/platform_time_scale.hpp). This module holds the uniform
scaling without ROS so it can be unit-tested, and ``check_bag`` checks every
trajectory published in a run's debug bag.

Piece i covers tau in [0, duration[i]] and coef[8*i + j] multiplies
tau**(7 - j). Scaling q(t) = p(t / k) multiplies each duration by k and divides
the coefficient of tau**n by k**n, so speed and heading rate scale by 1/k and
acceleration and heading acceleration by 1/k**2 while the path is unchanged.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
from pathlib import Path
import sys
from typing import Sequence

import numpy as np

ORDER = 7


@dataclass(frozen=True)
class Limits:
    """A non-positive limit disables its term."""

    max_vel: float
    max_acc: float
    max_omega: float
    max_domega: float


@dataclass(frozen=True)
class Peaks:
    speed: float
    acceleration: float
    yaw_rate: float
    yaw_acceleration: float
    start_speed: float
    duration: float


def _pieces(durations: Sequence[float], coef: Sequence[float]) -> np.ndarray:
    count = len(durations)
    values = np.asarray(coef, dtype=np.float64)
    if values.size != (ORDER + 1) * count:
        raise ValueError(f"expected {(ORDER + 1) * count} coefficients, got {values.size}")
    return values.reshape(count, ORDER + 1)


def _derivative(row: np.ndarray, tau: np.ndarray, order: int) -> np.ndarray:
    """Evaluate the order-th derivative of a highest-power-first row at tau."""

    total = np.zeros_like(tau)
    for column, power in enumerate(range(ORDER, -1, -1)):
        if power < order:
            continue
        total = total + math.perm(power, order) * row[column] * tau ** (power - order)
    return total


def measure(
    durations: Sequence[float],
    coef_x: Sequence[float],
    coef_y: Sequence[float],
    sample_dt: float = 0.002,
    min_speed: float = 0.05,
) -> Peaks:
    """Sampled peaks; heading terms use the optimizer's penalty expressions.

    Heading is undefined at rest, so heading rate and heading acceleration are
    only evaluated where the planar speed exceeds ``min_speed``.
    """

    cx = _pieces(durations, coef_x)
    cy = _pieces(durations, coef_y)
    speed = acceleration = yaw_rate = yaw_acceleration = 0.0
    for index, duration in enumerate(durations):
        steps = max(1, math.ceil(float(duration) / sample_dt))
        tau = np.linspace(0.0, float(duration), steps + 1)
        vx, vy = _derivative(cx[index], tau, 1), _derivative(cy[index], tau, 1)
        ax, ay = _derivative(cx[index], tau, 2), _derivative(cy[index], tau, 2)
        jx, jy = _derivative(cx[index], tau, 3), _derivative(cy[index], tau, 3)
        planar_speed = np.hypot(vx, vy)
        speed = max(speed, float(planar_speed.max()))
        acceleration = max(acceleration, float(np.hypot(ax, ay).max()))
        moving = planar_speed > min_speed
        if moving.any():
            v2 = planar_speed[moving] ** 2
            cross_va = (vx * ay - vy * ax)[moving]
            cross_vj = (vx * jy - vy * jx)[moving]
            dot_va = (vx * ax + vy * ay)[moving]
            yaw_rate = max(yaw_rate, float(np.abs(cross_va / v2).max()))
            yaw_acceleration = max(
                yaw_acceleration,
                float(np.abs(cross_vj / v2 - 2.0 * cross_va * dot_va / v2**2).max()),
            )
    origin = np.zeros(1)
    start_speed = float(
        np.hypot(_derivative(cx[0], origin, 1), _derivative(cy[0], origin, 1))[0]
    )
    return Peaks(
        speed, acceleration, yaw_rate, yaw_acceleration, start_speed, float(np.sum(durations))
    )


def measure_published(
    durations: Sequence[float],
    coef_x: Sequence[float],
    coef_y: Sequence[float],
    sample_dt: float = 0.01,
    window: float = 0.2,
    min_speed: float = 0.05,
) -> Peaks:
    """Peaks of a published trajectory, as the planner reports its retimed ones.

    A retimed trajectory is cut into pieces of constant rate: inside a piece the
    acceleration still follows the optimizer's own timing, and the velocity
    changes by small steps between pieces. Speed and heading rate are sampled;
    acceleration and heading acceleration are differences over ``window``.
    Heading terms are left out where the planar speed is at most ``min_speed``.
    """

    cx = _pieces(durations, coef_x)
    cy = _pieces(durations, coef_y)
    times, speeds, turns = [], [], []
    start = 0.0
    for index, duration in enumerate(durations):
        steps = max(1, math.ceil(float(duration) / sample_dt))
        tau = np.linspace(0.0, float(duration), steps + 1)
        vx, vy = _derivative(cx[index], tau, 1), _derivative(cy[index], tau, 1)
        ax, ay = _derivative(cx[index], tau, 2), _derivative(cy[index], tau, 2)
        speed = np.hypot(vx, vy)
        turn = np.full_like(speed, np.nan)
        moving = speed > min_speed
        turn[moving] = (vx * ay - vy * ax)[moving] / speed[moving] ** 2
        times.append(start + tau)
        speeds.append(speed)
        turns.append(turn)
        start += float(duration)
    times = np.concatenate(times)
    speeds = np.concatenate(speeds)
    turns = np.concatenate(turns)

    later = np.searchsorted(times, times + window)
    earlier = np.nonzero(later < times.size)[0]
    later = later[earlier]
    acceleration = yaw_acceleration = 0.0
    if earlier.size:
        dt = times[later] - times[earlier]
        tangential = (speeds[later] - speeds[earlier]) / dt
        normal = np.nan_to_num(np.abs(turns[earlier])) * speeds[earlier]
        acceleration = float(np.hypot(tangential, normal).max())
        turn_change = np.abs(turns[later] - turns[earlier]) / dt
        turn_change = turn_change[np.isfinite(turn_change)]
        if turn_change.size:
            yaw_acceleration = float(turn_change.max())
    finite_turns = np.abs(turns[np.isfinite(turns)])
    return Peaks(
        float(speeds.max()),
        acceleration,
        float(finite_turns.max()) if finite_turns.size else 0.0,
        yaw_acceleration,
        float(speeds[0]),
        float(np.sum(durations)),
    )


def time_scale(peaks: Peaks, limits: Limits) -> tuple[float, str]:
    """Smallest k >= 1 that brings every peak within its limit, and the binding term."""

    candidates = {"none": 1.0}
    if limits.max_vel > 0.0:
        candidates["vel"] = peaks.speed / limits.max_vel
    if limits.max_omega > 0.0:
        candidates["yaw_rate"] = peaks.yaw_rate / limits.max_omega
    if limits.max_acc > 0.0:
        candidates["acc"] = math.sqrt(peaks.acceleration / limits.max_acc)
    if limits.max_domega > 0.0:
        candidates["yaw_acc"] = math.sqrt(peaks.yaw_acceleration / limits.max_domega)
    binding = max(candidates, key=lambda name: candidates[name])
    return max(1.0, candidates[binding]), binding


def scale(
    durations: Sequence[float], coef: Sequence[float], k: float
) -> tuple[list[float], list[float]]:
    rows = _pieces(durations, coef)
    divisor = k ** np.arange(ORDER, -1, -1, dtype=np.float64)
    return [float(value) * k for value in durations], (rows / divisor).reshape(-1).tolist()


def feasible_scale(
    durations: Sequence[float],
    coef_x: Sequence[float],
    coef_y: Sequence[float],
    limits: Limits,
    sample_dt: float = 0.002,
    min_speed: float = 0.05,
    max_refinements: int = 3,
) -> tuple[float, str]:
    """k from one sampling pass, enlarged by denser passes on the scaled result.

    A narrow heading-rate spike near low speed can fall between samples, so the
    scaled trajectory is re-measured four times more densely and k grows by any
    remaining ratio, as in the planner's C++ helper.
    """

    k, binding = time_scale(measure(durations, coef_x, coef_y, sample_dt, min_speed), limits)
    for _ in range(max_refinements):
        scaled_durations, scaled_x = scale(durations, coef_x, k)
        _, scaled_y = scale(durations, coef_y, k)
        residual, residual_binding = time_scale(
            measure(scaled_durations, scaled_x, scaled_y, sample_dt * k / 4.0, min_speed / k),
            limits,
        )
        if residual <= 1.0 + 1e-9:
            break
        k, binding = k * residual, residual_binding
    return k, binding


def check_bag(bag: str, limits: Limits, tolerance: float, window: float, min_speed: float) -> int:
    """Print each published trajectory's peaks; 1 if any exceeds a limit."""

    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=bag, storage_id="sqlite3"),
        rosbag2_py.ConverterOptions("cdr", "cdr"),
    )
    types = {item.name: item.type for item in reader.get_all_topics_and_types()}
    count = violations = 0
    print(
        "traj_id pieces duration_s speed_peak acc_peak yaw_rate_peak yaw_acc_peak "
        "start_speed exceeded"
    )
    while reader.has_next():
        topic, data, _ = reader.read_next()
        if not topic.endswith("/planning/trajectory"):
            continue
        message = deserialize_message(data, get_message(types[topic]))
        peaks = measure_published(
            message.duration, message.coef_x, message.coef_y, window=window, min_speed=min_speed
        )
        exceeded = [
            name
            for name, value, limit in (
                ("vel", peaks.speed, limits.max_vel),
                ("acc", peaks.acceleration, limits.max_acc),
                ("yaw_rate", peaks.yaw_rate, limits.max_omega),
                ("yaw_acc", peaks.yaw_acceleration, limits.max_domega),
            )
            if limit > 0.0 and value > limit * (1.0 + tolerance)
        ]
        count += 1
        violations += int(bool(exceeded))
        print(
            f"{message.traj_id} {len(message.duration)} {peaks.duration:.2f} "
            f"{peaks.speed:.3f} {peaks.acceleration:.3f} {peaks.yaw_rate:.3f} "
            f"{peaks.yaw_acceleration:.3f} {peaks.start_speed:.3f} "
            f"{','.join(exceeded) if exceeded else '-'}"
        )
    print(f"trajectories={count} violations={violations}")
    return 1 if violations else 0


def main(argv: Sequence[str] | None = None) -> int:
    from .config import load_config

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("bag", help="a run's debug/bag directory")
    parser.add_argument("--config", default=str(Path(__file__).with_name("config.yaml")))
    parser.add_argument(
        "--max-acc", type=float, default=0.30, help="optimizer.max_acc in the YOR launch file"
    )
    parser.add_argument(
        "--max-domega",
        type=float,
        default=0.70,
        help="optimizer.max_domega in the YOR launch file",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.05,
        help="allowed relative excess; differences over the window include the small velocity "
        "steps between retimed pieces",
    )
    parser.add_argument("--window", type=float, default=0.2)
    parser.add_argument("--min-speed", type=float, default=0.05)
    args = parser.parse_args(argv)
    planner = load_config(args.config).planner
    limits = Limits(
        planner.reference_max_linear_mps,
        args.max_acc,
        planner.reference_max_yaw_rad_s,
        args.max_domega,
    )
    return check_bag(args.bag, limits, args.tolerance, args.window, args.min_speed)


if __name__ == "__main__":
    sys.exit(main())
