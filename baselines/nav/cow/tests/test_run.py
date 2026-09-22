import unittest

from baselines.nav.cow.run import build_parser


class ParserTest(unittest.TestCase):
    def test_cow_runner_takes_the_shared_experiment_options(self):
        args = build_parser().parse_args(["--task-id", "find_blue_trash_bin", "--experiment", "cow_main"])
        self.assertEqual((args.experiment, args.start_label, args.adopt), ("cow_main", "kitchen", "ask"))


if __name__ == "__main__":
    unittest.main()
