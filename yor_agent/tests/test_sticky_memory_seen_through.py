"""Remembered clearance cells that the current frame looks straight through.

The occupancy memory keeps the cells a primitive call has seen so that an
obstacle leaving the view on the way in still stops the base. It used to keep
every cell for the whole call, so a single frame's stray depth points in
mid-air refused a move through space the camera was looking straight through.
These tests pin both sides: an artefact is forgotten, and anything the camera
cannot actually show to be empty is not.
"""

from __future__ import annotations

import unittest

import numpy as np

from test_footprint_clearance import Box, SceneRobot, make_controller, make_geometry, render_depth
from yor_agent.robot.footprint_clearance import (
    FootprintConfig,
    StickyOccupancy,
    seen_through_cells,
)
from yor_agent.robot.navigation_controller import NavigationConfig

# A chassis and one arm band high enough that a cell just ahead of it stays
# inside this test camera's field of view.
FOOTPRINT = {
    "layers": [
        {
            "name": "chassis",
            "z_min": 0.10,
            "z_max": 0.27,
            "polygon_xy": [[0.22, 0.27], [-0.22, 0.27], [-0.22, -0.27], [0.22, -0.27]],
        },
        {
            "name": "arms_high",
            "z_min": 0.85,
            "z_max": 0.95,
            "polygon_xy": [[0.30, 0.43], [-0.22, 0.43], [-0.22, -0.43], [0.30, -0.43]],
        },
    ]
}
# Inside the 0.18 m/s sweep ahead of the arm band, and in view.
CELL_XY = (0.47, 0.0)
# A real object filling exactly that cell.
CELL_BOX = Box(0.445, 0.495, -0.025, 0.025, 0.85, 0.95)


def footprint() -> FootprintConfig:
    return FootprintConfig.from_mapping(FOOTPRINT)


class SeenThroughCellsTest(unittest.TestCase):
    """The per-cell judgement, against rendered depth."""

    def setUp(self) -> None:
        self.footprint = footprint()
        self.geometry, _ = make_geometry()
        self.layer = self.footprint.layer("arms_high")

    def depth(self, boxes=()) -> np.ndarray:
        return render_depth(self.footprint, base_pose=(0.0, 0.0, 0.0), boxes=tuple(boxes))

    def seen(self, depth, cells=(CELL_XY,)) -> np.ndarray:
        return seen_through_cells(
            np.array(cells, dtype=np.float64),
            self.layer,
            depth,
            self.geometry,
            self.footprint,
            cell_m=0.05,
            margin_m=0.15,
            min_valid_fraction=0.9,
        )

    def test_open_space_is_seen_through(self) -> None:
        self.assertTrue(self.seen(self.depth())[0])

    def test_an_object_in_the_cell_is_not(self) -> None:
        self.assertFalse(self.seen(self.depth(boxes=(CELL_BOX,)))[0])

    def test_no_depth_is_no_evidence_of_empty_space(self) -> None:
        self.assertFalse(self.seen(np.full_like(self.depth(), np.nan))[0])

    def test_an_object_that_returns_no_depth_is_kept(self) -> None:
        """A dark or shiny object: the background around it still measures."""

        empty = self.depth()
        dark = self.depth(boxes=(CELL_BOX,))
        dark[np.abs(dark - empty) > 1e-3] = np.nan

        self.assertFalse(self.seen(dark)[0])

    def test_a_dark_object_beside_the_remembered_centre_is_kept(self) -> None:
        """A thin object returning no depth, just past the edge of the cell.

        The memory reports a centre up to a quarter cell off where the
        obstacle was seen, and the pose drifts; the object must still hold the
        cell rather than let the background around it read as empty space.
        """

        empty = self.depth()
        pole = Box(0.455, 0.470, 0.030, 0.045, 0.0, 1.2)
        dark = self.depth(boxes=(pole,))
        dark[np.abs(dark - empty) > 1e-3] = np.nan

        self.assertFalse(self.seen(dark)[0])

    def test_something_nearer_across_the_cell_keeps_it(self) -> None:
        pole = Box(0.40, 0.42, -0.10, 0.02, 0.0, 1.2)

        self.assertFalse(self.seen(self.depth(boxes=(pole,)))[0])

    def test_a_cell_below_the_view_is_kept(self) -> None:
        self.assertFalse(self.seen(self.depth(), cells=((0.25, 0.0),))[0])

    def test_a_cell_behind_the_camera_is_kept(self) -> None:
        self.assertFalse(self.seen(self.depth(), cells=((-0.50, 0.0),))[0])

    def test_a_cell_cut_by_the_image_edge_is_kept(self) -> None:
        self.assertFalse(self.seen(self.depth(), cells=((0.47, 0.60),))[0])

    def test_cells_are_judged_independently(self) -> None:
        judged = self.seen(self.depth(), cells=(CELL_XY, (0.25, 0.0), (-0.50, 0.0)))

        self.assertEqual(judged.tolist(), [True, False, False])


class StickyForgetTest(unittest.TestCase):
    def setUp(self) -> None:
        self.sticky = StickyOccupancy(footprint(), cell_m=0.05, cell_min_points=3)
        self.pose = (0.0, 0.0, 0.0)
        self.sticky.add_cells({"arms_high": np.array([[0.6, 0.0], [1.0, 0.0], [1.4, 0.0]])}, self.pose)

    def test_only_the_marked_rows_are_dropped(self) -> None:
        before = self.sticky.cells_base(self.pose)["arms_high"]
        mask = np.zeros(len(before), dtype=bool)
        mask[1] = True

        dropped = self.sticky.forget({"arms_high": mask, "chassis": np.zeros(0, dtype=bool)})

        self.assertEqual(dropped, 1)
        np.testing.assert_allclose(self.sticky.cells_base(self.pose)["arms_high"], before[[0, 2]])

    def test_a_mask_of_the_wrong_length_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            self.sticky.forget({"arms_high": np.zeros(2, dtype=bool)})


class ControllerForgetsSeenThroughCellsTest(unittest.TestCase):
    """The clearance check the drive loop runs, with one remembered cell."""

    def build(self, *, boxes=(), clearing=True, depth_override=None):
        robot = SceneRobot(footprint(), boxes=boxes)
        robot.navigation_config["footprint"] = FOOTPRINT
        robot.navigation_config["memory_seen_through_clearing"] = clearing
        robot.depth_override = depth_override
        controller = make_controller(robot)
        frame = robot.navigation_frame(max_age_s=1.0)
        sticky = controller._new_sticky()
        sticky.add_cells({"arms_high": np.array([CELL_XY])}, controller._pose(frame))
        return controller, frame, sticky

    def test_a_cell_the_frame_sees_straight_through_no_longer_blocks(self) -> None:
        controller, frame, sticky = self.build()

        result = controller._swept_clearance(frame, (0.18, 0.0), sticky)

        self.assertTrue(result.clear, result.describe_block())
        self.assertEqual(result.memory_cells, 0)
        self.assertEqual(controller.last_clearance_debug["memory_seen_through"]["forgotten_cells"], 1)
        self.assertEqual(sticky.count, 0)

    def test_with_the_clearing_switched_off_the_same_cell_still_blocks(self) -> None:
        controller, frame, sticky = self.build(clearing=False)

        result = controller._swept_clearance(frame, (0.18, 0.0), sticky)

        self.assertFalse(result.clear)
        self.assertEqual(result.blocking_layers()[0].blocking.remembered_cells, 1)
        self.assertNotIn("memory_seen_through", controller.last_clearance_debug)

    def test_a_real_object_in_view_still_blocks(self) -> None:
        controller, frame, sticky = self.build(boxes=(CELL_BOX,))

        result = controller._swept_clearance(frame, (0.18, 0.0), sticky)

        self.assertFalse(result.clear)
        self.assertEqual(controller.last_clearance_debug["memory_seen_through"]["forgotten_cells"], 0)

    def test_a_remembered_object_the_camera_cannot_measure_still_blocks(self) -> None:
        """The case the memory exists for: the object's pixels return no depth."""

        empty = render_depth(footprint(), base_pose=(0.0, 0.0, 0.0))
        dark = render_depth(footprint(), base_pose=(0.0, 0.0, 0.0), boxes=(CELL_BOX,))
        dark[np.abs(dark - empty) > 1e-3] = np.nan
        controller, frame, sticky = self.build(depth_override=dark)

        result = controller._swept_clearance(frame, (0.18, 0.0), sticky)

        self.assertFalse(result.clear)
        blocking = result.blocking_layers()[0].blocking
        self.assertEqual((blocking.live_cells, blocking.remembered_cells), (0, 1))
        self.assertEqual(sticky.count, 1)

    def test_a_reverse_command_consults_no_depth_and_forgets_nothing(self) -> None:
        controller, frame, sticky = self.build()

        controller._swept_clearance(frame, (-0.06, 0.0), sticky)

        self.assertEqual(sticky.count, 1)


class SeenThroughConfigTest(unittest.TestCase):
    def test_the_clearing_is_on_by_default(self) -> None:
        self.assertTrue(NavigationConfig.from_mapping(None).memory_seen_through_clearing)

    def test_out_of_range_settings_are_refused(self) -> None:
        for bad in (
            {"memory_seen_through_clearing": "yes"},
            {"memory_seen_through_margin_m": 0.0},
            {"memory_seen_through_min_valid_fraction": 0.0},
            {"memory_seen_through_min_valid_fraction": 1.5},
            {"memory_seen_through_lookahead_m": -1.0},
        ):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                NavigationConfig.from_mapping(bad)


if __name__ == "__main__":
    unittest.main()
