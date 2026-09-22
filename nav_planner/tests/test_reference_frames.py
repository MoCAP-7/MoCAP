from __future__ import annotations

import unittest

from nav_planner.reference_frames import cited_timestamps


class ReferenceFrameTest(unittest.TestCase):
    def test_extracts_unique_timestamps_from_ranges_in_chronological_order(self) -> None:
        text = "Seen at 00:43-00:46, revisited at 00:43 and 01:06."

        self.assertEqual(
            cited_timestamps(text),
            [("00:43", 43.0), ("00:46", 46.0), ("01:06", 66.0)],
        )

    def test_limit_spreads_frames_across_the_full_timeline(self) -> None:
        text = " ".join(f"00:{second:02d}" for second in range(10))

        timestamps = cited_timestamps(text, limit=3)

        self.assertEqual(timestamps[0], ("00:00", 0.0))
        self.assertEqual(timestamps[-1], ("00:09", 9.0))


if __name__ == "__main__":
    unittest.main()
