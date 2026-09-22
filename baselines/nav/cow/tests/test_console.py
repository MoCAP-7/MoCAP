import unittest

from baselines.nav.cow.console import INCOMPLETE_ACTION_WARNING, EpisodeNarrator


def decision(step, **fields):
    return {
        "type": "decision",
        "step": step,
        "elapsed_s": 1.0 + step,
        "mode": "EXPLORE",
        "action": "MoveAhead",
        "attention_max": 0.0,
        "roi_exists": False,
        "exploration_targets": 12,
        "decide_s": 0.3,
        "predicted_failed_previous_action": False,
        "map_reset_this_step": False,
        **fields,
    }


def primitive(step, *, success, primitive_name="drive_straight", reason="target_reached", start=None, final=None):
    result = {
        "success": success,
        "status": "succeeded" if success else "failed",
        "primitive": primitive_name,
        "reason": reason,
        "elapsed_s": 0.4,
    }
    if start is not None:
        result.update(start_pose_xy_yaw=start, final_pose_xy_yaw=final)
    return {"type": "primitive", "step": step, "action": "MoveAhead", "result": result}


def refused(step):
    return primitive(
        step,
        success=False,
        reason="obstacle_too_close: something is in the way",
        start=[1.0, 2.0, 0.5],
        final=[1.0, 2.0, 0.5 - 1e-5],
    )


class NarratorTest(unittest.TestCase):
    def setUp(self):
        self.lines = []
        self.narrator = EpisodeNarrator(self.lines.append)

    def test_decision_and_action_lines(self):
        self.narrator.event(decision(3, mode="EXPLOIT", attention_max=1.0, roi_exists=True))
        done = primitive(3, success=True, start=[0.0, 0.0, 0.0], final=[0.229, 0.0, 0.03])
        done["result"]["elapsed_s"] = 2.9
        self.narrator.event(done)
        self.assertIn("step   3", self.lines[0])
        self.assertIn("EXPLOIT -> MoveAhead", self.lines[0])
        self.assertIn("target pixel 1.00, target in map yes", self.lines[0])
        self.assertIn("drive_straight done: camera moved 0.229 m, turned +1.7 deg in 2.9 s", self.lines[1])

    def test_mode_changes_and_map_resets_are_announced(self):
        self.narrator.event(decision(0, mode="SPIN", action="RotateLeft"))
        self.narrator.event(decision(1, mode="EXPLOIT", map_reset_this_step=True))
        self.assertIn("CoW mode SPIN -> EXPLOIT", self.lines)
        self.assertTrue(any("reset its map" in line for line in self.lines))

    def test_repeated_incomplete_actions_warn_with_the_reason(self):
        for step in range(INCOMPLETE_ACTION_WARNING):
            self.narrator.event(refused(step))
        self.assertIn(
            "drive_straight failed (obstacle_too_close): camera moved 0.000 m, turned +0.0 deg in 0.4 s", self.lines[0]
        )
        warnings = [line for line in self.lines if line.startswith("warning:")]
        self.assertEqual(len(warnings), 1)
        self.assertIn(f"{INCOMPLETE_ACTION_WARNING} actions in a row did not complete", warnings[0])

    def test_a_completed_action_resets_the_warning_count(self):
        for step in range(INCOMPLETE_ACTION_WARNING - 1):
            self.narrator.event(refused(step))
        self.narrator.event(primitive(9, success=True, primitive_name="turn_relative"))
        for step in range(INCOMPLETE_ACTION_WARNING - 1):
            self.narrator.event(refused(step))
        self.assertFalse(any(line.startswith("warning:") for line in self.lines))

    def test_unknown_events_and_missing_fields_do_not_raise(self):
        self.narrator.event({"type": "something_new"})
        self.narrator.event({"type": "episode_start", "goal": "blue trash bin", "pose": [None, None, None]})
        self.assertIn("camera pose unavailable", self.lines[-1])


if __name__ == "__main__":
    unittest.main()
