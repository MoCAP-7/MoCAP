import unittest

from baselines.nav.cow.tasks import direct_task, instruction_to_goal, load_task


class InstructionToGoalTest(unittest.TestCase):
    def test_keeps_the_whole_description(self):
        self.assertEqual(instruction_to_goal("Navigate to the can on the table"), "can on the table")
        self.assertEqual(
            instruction_to_goal("Navigate to the small cardbox on the white storage cabinet"),
            "small cardbox on the white storage cabinet",
        )
        self.assertEqual(instruction_to_goal("Find the blue trash bin."), "blue trash bin")

    def test_rejects_non_navigation_instructions(self):
        for instruction in (
            "Grasp the small cardboard box",
            "",
            "Navigate to",
            "Find the can, grasp it, and return to the starting location while holding it",
        ):
            with self.subTest(instruction=instruction), self.assertRaises(ValueError):
                instruction_to_goal(instruction)

    def test_descriptions_with_commas_are_still_goals(self):
        self.assertEqual(instruction_to_goal("Find the small, red apple"), "small, red apple")

    def test_direct_task_normalizes_whitespace(self):
        task = direct_task("  Navigate   to the red marker on the table ")
        self.assertIsNone(task.task_id)
        self.assertEqual(task.goal, "red marker on the table")

    def test_shared_task_suite(self):
        task = load_task("find_blue_trash_bin")
        self.assertEqual(task.goal, "blue trash bin")
        with self.assertRaisesRegex(ValueError, "unknown task id"):
            load_task("no_such_task")
        for manipulation_task in ("find_can_and_give_back", "grasp_cardboard_box", "throw_box_to_trash_bin"):
            with self.subTest(task=manipulation_task), self.assertRaises(ValueError):
                load_task(manipulation_task)


if __name__ == "__main__":
    unittest.main()
