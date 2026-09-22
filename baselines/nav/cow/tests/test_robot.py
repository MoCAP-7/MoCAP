import unittest

from baselines.nav.cow.robot import safety_stop_reason


def failed(reason):
    return {"success": False, "status": "failed", "primitive": "drive_straight", "reason": reason}


class SafetyStopReasonTest(unittest.TestCase):
    def test_clearance_refusals_are_safety_stops(self):
        self.assertEqual(safety_stop_reason(failed("obstacle_too_close")), "obstacle_too_close")
        self.assertEqual(
            safety_stop_reason(failed("depth_unknown_in_sweep: the space this step would move into returned no depth")),
            "depth_unknown_in_sweep",
        )
        self.assertEqual(safety_stop_reason(failed("RuntimeError:depth_mostly_invalid")), "depth_mostly_invalid")
        self.assertEqual(
            safety_stop_reason(failed("obstacle_too_close; final_stop_failed: timeout")), "obstacle_too_close"
        )

    def test_other_failures_and_successes_are_not(self):
        self.assertIsNone(safety_stop_reason(failed("timeout")))
        self.assertIsNone(safety_stop_reason(failed("RuntimeError:localization_yaw_jump_0.400rad")))
        self.assertIsNone(safety_stop_reason(failed("RuntimeError:operator_stop_requested")))
        self.assertIsNone(safety_stop_reason({"success": True, "reason": "target_reached"}))


if __name__ == "__main__":
    unittest.main()
