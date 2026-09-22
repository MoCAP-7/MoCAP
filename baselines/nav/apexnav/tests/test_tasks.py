import unittest

from baselines.nav.apexnav.tasks import instruction_to_target


class TaskTest(unittest.TestCase):
    def test_open_vocabulary_description_is_preserved(self):
        self.assertEqual(
            instruction_to_target("Find the blue recycling bin."),
            "blue recycling bin",
        )

    def test_non_navigation_instruction_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "must begin"):
            instruction_to_target("The chair is over there")


if __name__ == "__main__":
    unittest.main()
