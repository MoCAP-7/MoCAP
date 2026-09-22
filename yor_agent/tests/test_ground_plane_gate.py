"""The floor-plane gate against replayed ZED plane sequences.

The frame times are the caller's, so every sequence below is a deterministic
replay of what the SDK reported: the 2026-09-13 docking stall's bursts, a
lift move, a fallback that was off by a few centimetres, static jitter.
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from yor_agent.robot.ground_plane_gate import (
    ACCEPTED,
    HELD,
    SETTLING,
    GroundPlaneGate,
)

DOWN = np.asarray([0.0, 1.0, 0.0])


def tilted(degrees: float) -> np.ndarray:
    """``DOWN`` rotated about the optical x axis by ``degrees``."""

    angle = math.radians(degrees)
    return np.asarray([0.0, math.cos(angle), math.sin(angle)])


def make_gate(fallback_height_m: float = 1.12, **tolerances) -> GroundPlaneGate:
    return GroundPlaneGate(fallback_height_m, DOWN, **tolerances)


def confirmed_gate(height_m: float = 1.04, time_s: float = 0.0, **tolerances) -> GroundPlaneGate:
    """A gate whose plane in use came from the SDK, not the fallback."""

    gate = make_gate(1.12, **tolerances)
    decision = gate.offer(height_m, DOWN, time_s)
    assert decision.status == ACCEPTED
    return gate


class GroundPlaneGateTest(unittest.TestCase):
    def test_the_first_plane_inside_the_envelope_replaces_the_fallback(self) -> None:
        # The configured fallback is a few centimetres above the live camera,
        # more than one step; the first observation must not pay for that.
        gate = make_gate(1.12)

        decision = gate.offer(1.04, DOWN, 0.0)

        self.assertEqual(decision.status, ACCEPTED)
        self.assertAlmostEqual(decision.camera_height_m, 1.04)
        self.assertAlmostEqual(gate.camera_height_m, 1.04)
        self.assertEqual(decision.held_for_s, 0.0)
        self.assertFalse(decision.settled)
        self.assertIn("fallback", decision.reason)

    def test_the_first_plane_outside_the_envelope_is_held_at_the_fallback(self) -> None:
        gate = make_gate(1.12)

        decision = gate.offer(1.67, DOWN, 0.0)

        self.assertEqual(decision.status, HELD)
        self.assertAlmostEqual(gate.camera_height_m, 1.12)
        # The fallback is still a placeholder: the next plausible plane
        # replaces it at once.
        self.assertEqual(gate.offer(1.04, DOWN, 0.5).status, ACCEPTED)
        self.assertAlmostEqual(gate.camera_height_m, 1.04)

    def test_candidate_within_step_is_accepted_at_once(self) -> None:
        gate = confirmed_gate(1.04)

        decision = gate.offer(1.07, DOWN, 0.5)

        self.assertEqual(decision.status, ACCEPTED)
        self.assertAlmostEqual(decision.camera_height_m, 1.07)
        self.assertAlmostEqual(gate.camera_height_m, 1.07)
        self.assertEqual(decision.held_for_s, 0.0)
        self.assertFalse(decision.settled)

    def test_burst_is_held_and_the_plane_in_use_is_unchanged(self) -> None:
        gate = confirmed_gate(1.04)
        burst = [0.31, 1.67, 1.04, 0.28, 1.04, 1.72]

        for index in range(24):
            offered = burst[index % len(burst)]
            decision = gate.offer(offered, DOWN, 0.5 + 0.05 * index)
            self.assertEqual(
                decision.status, ACCEPTED if offered == 1.04 else HELD, offered
            )
            self.assertAlmostEqual(decision.camera_height_m, 1.04)
            np.testing.assert_allclose(decision.down_camera_xyz, DOWN)

        self.assertAlmostEqual(gate.camera_height_m, 1.04)

    def test_in_envelope_burst_is_dropped_by_the_next_good_frame(self) -> None:
        gate = confirmed_gate(1.04, settle_s=2.0)

        for index in range(40):
            offered = 1.20 if index % 4 == 0 else 1.04
            decision = gate.offer(offered, DOWN, 0.5 + 0.05 * index)
            self.assertEqual(decision.status, HELD if offered == 1.20 else ACCEPTED)

        self.assertAlmostEqual(gate.camera_height_m, 1.04)

    def test_steady_change_is_adopted_after_the_settle_time_not_before(
        self,
    ) -> None:
        gate = confirmed_gate(1.12, time_s=9.0, settle_s=2.0)

        first = gate.offer(1.04, DOWN, 10.0)
        self.assertEqual(first.status, HELD)
        self.assertAlmostEqual(first.camera_height_m, 1.12)
        for time_s in (10.5, 11.0, 11.5):
            decision = gate.offer(1.04, DOWN, time_s)
            self.assertEqual(decision.status, SETTLING)
            self.assertAlmostEqual(decision.camera_height_m, 1.12)
            self.assertAlmostEqual(gate.camera_height_m, 1.12)

        adopted = gate.offer(1.04, DOWN, 12.0)

        self.assertEqual(adopted.status, ACCEPTED)
        self.assertTrue(adopted.settled)
        self.assertAlmostEqual(adopted.camera_height_m, 1.04)
        self.assertAlmostEqual(gate.camera_height_m, 1.04)
        self.assertEqual(adopted.held_for_s, 0.0)

    def test_jitter_wider_than_the_step_still_settles_on_its_mean(self) -> None:
        # Static jitter spans about twice the step; a pending candidate is
        # judged against the running mean of its offers with twice the step,
        # so the extremes on either side still count as agreement.
        gate = confirmed_gate(1.20, time_s=0.0, settle_s=2.0)
        jitter = [0.95, 1.05, 0.97, 1.03, 1.05, 0.95, 1.00, 1.04, 0.96]

        statuses = []
        heights = []
        settled_height_m = None
        for index in range(50):
            decision = gate.offer(jitter[index % len(jitter)], DOWN, 0.5 + 0.1 * index)
            statuses.append(decision.status)
            heights.append(decision.camera_height_m)
            if decision.settled and settled_height_m is None:
                settled_height_m = decision.camera_height_m

        self.assertIn(ACCEPTED, statuses)
        self.assertEqual(statuses[0], HELD)
        first_accept = statuses.index(ACCEPTED)
        self.assertEqual(set(statuses[1:first_accept]), {SETTLING})
        self.assertLessEqual(0.5 + 0.1 * first_accept, 2.6)
        # Settled on the mean; afterwards the plane in use stays inside the
        # jitter band whichever frames are followed or held.
        self.assertIsNotNone(settled_height_m)
        self.assertAlmostEqual(float(settled_height_m or 0.0), 1.00, delta=0.03)
        for height_m in heights[first_accept:]:
            self.assertAlmostEqual(height_m, 1.00, delta=0.06)

    def test_a_lift_move_is_followed(self) -> None:
        # The camera lowers slowly for several seconds; every frame is within
        # one step of the plane in use, so the plane follows frame by frame.
        # The envelope is around the fallback, so it has to cover the travel.
        gate = confirmed_gate(1.04, time_s=0.0, settle_s=2.0, max_height_error_m=0.8)

        height = 1.04
        for index in range(1, 121):
            height = 1.04 - 0.004 * index  # 0.08 m/s at 20 Hz
            gate.offer(height, DOWN, 0.05 * index)

        self.assertAlmostEqual(gate.camera_height_m, height, delta=0.01)

    def test_a_lift_move_beyond_the_envelope_freezes_the_plane(self) -> None:
        gate = confirmed_gate(1.04, time_s=0.0, max_height_error_m=0.25)

        for index in range(1, 121):
            gate.offer(1.04 - 0.004 * index, DOWN, 0.05 * index)

        # Frozen where the envelope around the fallback ends, not lost.
        self.assertAlmostEqual(gate.camera_height_m, 1.12 - 0.25, delta=0.01)

    def test_pending_candidate_is_dropped_by_a_disagreeing_offer(self) -> None:
        gate = confirmed_gate(1.12, time_s=0.0, settle_s=2.0)

        gate.offer(1.04, DOWN, 0.5)
        gate.offer(1.04, DOWN, 1.0)
        gate.offer(1.30, DOWN, 1.5)
        decision = gate.offer(1.04, DOWN, 2.5)

        # Persistence restarted at 2.5 s: the 1.04 m candidate was not seen
        # through the whole window, so it is a fresh jump again.
        self.assertEqual(decision.status, HELD)
        self.assertAlmostEqual(gate.camera_height_m, 1.12)

    def test_envelope_violation_never_settles(self) -> None:
        gate = make_gate(1.12, max_height_error_m=0.25)

        for index in range(40):
            decision = gate.offer(1.50, DOWN, 0.25 * index)
            self.assertEqual(decision.status, HELD)
            self.assertIn("envelope", decision.reason)

        self.assertAlmostEqual(gate.camera_height_m, 1.12)

    def test_tilt_is_measured_as_an_angle(self) -> None:
        gate = confirmed_gate(1.12, time_s=0.0, max_tilt_step_deg=3.0)

        held = gate.offer(1.12, tilted(5.0), 0.5)
        accepted = gate.offer(1.12, tilted(2.0), 1.0)

        self.assertEqual(held.status, HELD)
        np.testing.assert_allclose(held.down_camera_xyz, DOWN)
        self.assertEqual(accepted.status, ACCEPTED)
        np.testing.assert_allclose(accepted.down_camera_xyz, tilted(2.0))

    def test_tilt_outside_the_envelope_never_settles(self) -> None:
        gate = make_gate(1.12, max_tilt_error_deg=15.0)

        for index in range(20):
            decision = gate.offer(1.12, tilted(20.0), 0.5 * index)
            self.assertEqual(decision.status, HELD)

        np.testing.assert_allclose(gate.down_camera_xyz, DOWN)

    def test_held_for_grows_across_held_frames_and_resets_on_accept(self) -> None:
        gate = make_gate(1.12)

        held_for = [
            gate.offer(1.67, DOWN, time_s).held_for_s for time_s in (5.0, 5.5, 6.0)
        ]
        accepted = gate.offer(1.13, DOWN, 6.5)
        held_again = gate.offer(1.67, DOWN, 7.5)

        self.assertEqual(held_for, [0.0, 0.5, 1.0])
        self.assertEqual(accepted.status, ACCEPTED)
        self.assertEqual(accepted.held_for_s, 0.0)
        self.assertAlmostEqual(held_again.held_for_s, 1.0)

    def test_zero_norm_down_is_held_without_crashing(self) -> None:
        gate = make_gate(1.12)

        decision = gate.offer(1.12, np.zeros(3), 0.0)

        self.assertEqual(decision.status, HELD)
        self.assertAlmostEqual(decision.camera_height_m, 1.12)
        np.testing.assert_allclose(decision.down_camera_xyz, DOWN)

    def test_decision_reports_the_offered_height(self) -> None:
        gate = make_gate(1.12)

        decision = gate.offer(1.67, DOWN, 0.0)

        self.assertAlmostEqual(decision.offered_height_m, 1.67)
        self.assertAlmostEqual(decision.camera_height_m, 1.12)

    def test_tolerances_are_validated(self) -> None:
        with self.assertRaises(ValueError):
            make_gate(1.12, max_height_step_m=0.0)
        with self.assertRaises(ValueError):
            make_gate(1.12, max_tilt_step_deg=60.0)
        with self.assertRaises(ValueError):
            make_gate(1.12, settle_s=0.0)
        with self.assertRaises(ValueError):
            make_gate(1.12, max_height_error_m=0.01)
        with self.assertRaises(ValueError):
            make_gate(1.12, max_tilt_error_deg=1.0)
        with self.assertRaises(ValueError):
            GroundPlaneGate(1.12, np.zeros(3))


if __name__ == "__main__":
    unittest.main()
