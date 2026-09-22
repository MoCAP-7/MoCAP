import unittest

from baselines.nav.apexnav import run


class ExperimentInterfaceTests(unittest.TestCase):
    def parse(self, *arguments):
        return run.build_parser().parse_args(list(arguments))

    def test_runner_takes_the_shared_experiment_options(self):
        args = self.parse("--task-id", "find_blue_trash_bin", "--experiment", "apexnav_main")
        self.assertEqual((args.experiment, args.start_label, args.adopt), ("apexnav_main", "kitchen", "ask"))

    def test_task_id_uses_the_suite_instruction_as_the_detector_target(self):
        self.assertEqual(
            run.resolve_task(self.parse("--task-id", "find_blue_trash_bin")),
            ("find_blue_trash_bin", "Navigate to the blue trash bin", "blue trash bin"),
        )

    def test_debugging_targets_and_instructions_still_work(self):
        self.assertEqual(run.resolve_task(self.parse("--target", "blue trash bin")), (None, None, "blue trash bin"))
        self.assertEqual(
            run.resolve_task(self.parse("--instruction", "Find the blue trash bin")),
            (None, "Find the blue trash bin", "blue trash bin"),
        )

    def test_tasks_that_ask_for_manipulation_are_rejected(self):
        with self.assertRaises(ValueError):
            run.resolve_task(self.parse("--task-id", "find_can_and_give_back"))

    def test_only_one_way_to_name_the_target(self):
        with self.assertRaises(SystemExit):
            self.parse("--task-id", "find_blue_trash_bin", "--target", "blue trash bin")


if __name__ == "__main__":
    unittest.main()
