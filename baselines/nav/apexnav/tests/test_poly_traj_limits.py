import json
import math
import os
import unittest

import numpy as np

from baselines.nav.apexnav import poly_traj_limits as limits_module

LIMITS = limits_module.Limits(max_vel=0.18, max_acc=0.30, max_omega=0.35, max_domega=0.70)


def _position(durations, coef, t):
    rows = np.asarray(coef, dtype=np.float64).reshape(len(durations), 8)
    for index, duration in enumerate(durations):
        if t <= duration or index == len(durations) - 1:
            return float(np.polyval(rows[index], min(t, duration)))
        t -= duration
    raise AssertionError("time outside the trajectory")


def _random_trajectory(seed, pieces=4):
    rng = np.random.default_rng(seed)
    durations = rng.uniform(0.3, 1.2, pieces).tolist()
    return (
        durations,
        rng.normal(0.0, 0.3, 8 * pieces).tolist(),
        rng.normal(0.0, 0.3, 8 * pieces).tolist(),
    )


class TimeScalingTests(unittest.TestCase):
    def test_scaled_trajectory_traces_the_same_path(self):
        durations, coef_x, coef_y = _random_trajectory(1)
        k = 3.7
        scaled_durations, scaled_x = limits_module.scale(durations, coef_x, k)
        _, scaled_y = limits_module.scale(durations, coef_y, k)
        self.assertAlmostEqual(sum(scaled_durations), k * sum(durations), places=12)
        for t in np.linspace(0.0, sum(durations), 57):
            self.assertAlmostEqual(
                _position(scaled_durations, scaled_x, k * t), _position(durations, coef_x, t), places=9
            )
            self.assertAlmostEqual(
                _position(scaled_durations, scaled_y, k * t), _position(durations, coef_y, t), places=9
            )

    def test_peaks_scale_with_the_expected_powers_of_k(self):
        durations, coef_x, coef_y = _random_trajectory(2)
        k = 2.5
        before = limits_module.measure(durations, coef_x, coef_y, sample_dt=0.002, min_speed=0.0)
        scaled_durations, scaled_x = limits_module.scale(durations, coef_x, k)
        _, scaled_y = limits_module.scale(durations, coef_y, k)
        after = limits_module.measure(
            scaled_durations, scaled_x, scaled_y, sample_dt=0.002 * k, min_speed=0.0
        )
        self.assertAlmostEqual(after.speed, before.speed / k, places=9)
        self.assertAlmostEqual(after.yaw_rate, before.yaw_rate / k, places=6)
        self.assertAlmostEqual(after.acceleration, before.acceleration / k**2, places=9)
        self.assertAlmostEqual(after.yaw_acceleration, before.yaw_acceleration / k**2, places=6)

    def test_straight_line_binds_on_speed(self):
        # x(t) = 0.72 t: constant speed, no acceleration or turning.
        coef_x = [0, 0, 0, 0, 0, 0, 0.72, 0.0]
        k, binding = limits_module.time_scale(
            limits_module.measure([2.0], coef_x, [0.0] * 8), LIMITS
        )
        self.assertEqual(binding, "vel")
        self.assertAlmostEqual(k, 4.0, places=9)

    def test_tight_arc_binds_on_yaw_rate(self):
        # A constant-speed arc as a degree-7 Taylor polynomial of r (sin wt, 1 - cos wt).
        coef_x = [0.0] * 8
        coef_y = [0.0] * 8
        for power in range(1, 8):
            term = 0.1 / math.factorial(power)
            if power % 2 == 1:
                coef_x[7 - power] = term * (1 if power % 4 == 1 else -1)
            else:
                coef_y[7 - power] = term * (1 if power % 4 == 2 else -1)
        peaks = limits_module.measure([0.5], coef_x, coef_y, sample_dt=0.001)
        k, binding = limits_module.time_scale(peaks, LIMITS)
        self.assertEqual(binding, "yaw_rate")
        self.assertAlmostEqual(k, peaks.yaw_rate / LIMITS.max_omega, places=9)
        self.assertGreater(k, 2.5)

    def test_feasible_trajectory_is_left_alone(self):
        coef_x = [0, 0, 0, 0, 0, 0, 0.1, 0.0]
        result = limits_module.time_scale(limits_module.measure([1.0], coef_x, [0.0] * 8), LIMITS)
        self.assertEqual(result, (1.0, "none"))

    def test_refined_scale_keeps_limits_after_a_float32_round_trip(self):
        # A heading-rate spike this narrow is under-sampled by the first pass.
        durations, coef_x, coef_y = _random_trajectory(3, pieces=6)
        k, _ = limits_module.feasible_scale(durations, coef_x, coef_y, LIMITS)
        scaled_durations, scaled_x = limits_module.scale(durations, coef_x, k)
        _, scaled_y = limits_module.scale(durations, coef_y, k)

        def float32(values):
            return np.asarray(values, dtype=np.float32).astype(np.float64).tolist()

        after = limits_module.measure(
            float32(scaled_durations),
            float32(scaled_x),
            float32(scaled_y),
            sample_dt=0.001,
            min_speed=0.05 / k,
        )
        self.assertLessEqual(after.speed, LIMITS.max_vel * 1.001)
        self.assertLessEqual(after.acceleration, LIMITS.max_acc * 1.001)
        self.assertLessEqual(after.yaw_rate, LIMITS.max_omega * 1.001)
        self.assertLessEqual(after.yaw_acceleration, LIMITS.max_domega * 1.001)


class PublishedTrajectoryTests(unittest.TestCase):
    def test_constant_acceleration_cut_into_pieces_is_measured_over_the_window(self):
        # x(t) = 0.15 t^2 over 1 s, as ten pieces in local time.
        acceleration = 0.30
        durations = [0.1] * 10
        coef_x = []
        for index in range(10):
            start = 0.1 * index
            coef_x += [0, 0, 0, 0, 0, acceleration / 2, acceleration * start, acceleration * start**2 / 2]
        peaks = limits_module.measure_published(durations, coef_x, [0.0] * 80)
        self.assertAlmostEqual(peaks.speed, acceleration, places=9)
        self.assertAlmostEqual(peaks.acceleration, acceleration, places=6)
        self.assertEqual((peaks.yaw_rate, peaks.yaw_acceleration), (0.0, 0.0))
        self.assertAlmostEqual(peaks.duration, 1.0, places=9)

    def test_speed_steps_between_pieces_count_as_acceleration(self):
        # Constant speeds 0.100 then 0.102 m/s along x, continuous in position.
        durations = [1.0, 1.0]
        coef_x = [0, 0, 0, 0, 0, 0, 0.100, 0.0] + [0, 0, 0, 0, 0, 0, 0.102, 0.100]
        peaks = limits_module.measure_published(durations, coef_x, [0.0] * 16, window=0.2)
        self.assertAlmostEqual(peaks.speed, 0.102, places=9)
        self.assertAlmostEqual(peaks.acceleration, 0.002 / 0.2, delta=1e-3)
        self.assertEqual(peaks.start_speed, 0.1)


RECORDED = os.environ.get("APEXNAV_RECORDED_TRAJ_JSON")


@unittest.skipUnless(RECORDED, "set APEXNAV_RECORDED_TRAJ_JSON to decoded PolyTraj messages")
class RecordedTrajectoryTests(unittest.TestCase):
    def test_recorded_trajectories_need_the_measured_scale(self):
        with open(RECORDED, encoding="utf-8") as stream:
            recorded = json.load(stream)
        scales = [
            limits_module.time_scale(
                limits_module.measure(item["duration"], item["coef_x"], item["coef_y"]), LIMITS
            )[0]
            for item in recorded
        ]
        self.assertEqual(len(scales), 13)
        self.assertAlmostEqual(min(scales), 3.26, delta=0.01)
        self.assertAlmostEqual(max(scales), 5.63, delta=0.01)


if __name__ == "__main__":
    unittest.main()
