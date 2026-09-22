"""What a refused base motion leaves behind for whoever reads the trace.

A refusal used to record only that some height band held a blocking cell.
With ten bands in the footprint and the per-layer list arriving as a repr cut
off at ``MAX_REPR_CHARS``, neither the band nor the cells in it could be named
once the run was over, so a real obstacle and the robot blocking itself looked
identical. These tests pin both halves of the fix: the evidence the gate now
computes, and the trace summarising that has to carry it out intact.
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from yor_agent.executor import MAX_REPR_CHARS, _summarize
from yor_agent.robot.footprint_clearance import (
    ClearanceResult,
    FootprintConfig,
    check_cells,
    distance_outside_polygon,
)


def band(
    name: str, z_min: float, z_max: float, *, front: float, half_width: float
) -> dict:
    return {
        "name": name,
        "z_min": z_min,
        "z_max": z_max,
        "polygon_xy": [
            [front, half_width],
            [-0.22, half_width],
            [-0.22, -half_width],
            [front, -half_width],
        ],
    }


class BlockingEvidenceTest(unittest.TestCase):
    """The per-band evidence the clearance gate records when it refuses."""

    # No frame age, latency or lease, so the swept distance is the braking
    # distance plus the margin, and every cell below is placed against a
    # number this test states outright.
    SWEEP_M = math.pi * 0.10 * 0.10 / (4.0 * 0.6) + 0.05

    def setUp(self) -> None:
        self.body = np.array(
            [[0.22, 0.27], [-0.22, 0.27], [-0.22, -0.27], [0.22, -0.27]]
        )
        self.footprint = FootprintConfig.from_mapping(
            {
                "body_polygon_xy": self.body.tolist(),
                "layers": [
                    band("chassis", 0.00, 0.27, front=0.22, half_width=0.27),
                    band("arms_0p77", 0.77, 0.82, front=0.30, half_width=0.43),
                    band("arms_0p82", 0.82, 0.87, front=0.30, half_width=0.43),
                ],
            }
        )

    def check(self, cells, live_counts, *, memory_cells=0):
        return check_cells(
            cells,
            self.footprint,
            (0.10, 0.0),
            frame_age_s=0.0,
            command_latency_s=0.0,
            lease_s=0.0,
            brake_accel_mps2=0.6,
            margin_m=0.05,
            live_cell_counts=live_counts,
            memory_cells=memory_cells,
        )

    def test_sweep_is_the_distance_the_cells_below_are_placed_against(self) -> None:
        result = self.check({}, {})

        self.assertAlmostEqual(result.sweep_distance_m, self.SWEEP_M, places=6)
        self.assertGreater(self.SWEEP_M, 0.06)
        self.assertLess(self.SWEEP_M, 0.08)

    def test_the_blocking_band_is_named_with_its_height(self) -> None:
        cells = {
            "chassis": np.zeros((0, 2)),
            # Far enough ahead to stay out of the sweep, so it sets min_free
            # for its own band without blocking.
            "arms_0p77": np.array([[0.50, 0.0]]),
            "arms_0p82": np.array([[0.36, 0.02], [0.38, -0.08], [0.33, 0.05]]),
        }
        result = self.check(cells, {"chassis": 0, "arms_0p77": 1, "arms_0p82": 2})
        summary = result.summary()

        self.assertFalse(result.clear)
        self.assertEqual(summary["blocking_layer_names"], ["arms_0p82"])
        self.assertEqual(summary["binding_layer_name"], "arms_0p82")
        self.assertIn("arms_0p82", summary["blocked_by"])
        self.assertIn("0.82", summary["blocked_by"])
        self.assertIn("0.87", summary["blocked_by"])

    def test_a_cell_beyond_the_sweep_does_not_join_the_evidence(self) -> None:
        cells = {
            "chassis": np.zeros((0, 2)),
            "arms_0p77": np.array([[0.50, 0.0]]),
            "arms_0p82": np.array([[0.36, 0.02], [0.38, -0.08], [0.33, 0.05]]),
        }
        result = self.check(cells, {"chassis": 0, "arms_0p77": 1, "arms_0p82": 2})
        evidence = result.blocking_layers()[0].blocking

        # 0.06 and 0.03 m ahead of the outline block; 0.08 m ahead does not.
        self.assertEqual(evidence.count, 2)
        self.assertAlmostEqual(evidence.nearest_free_distance_m, 0.03, places=6)
        self.assertAlmostEqual(evidence.nearest_cell_xy[0], 0.33, places=6)
        self.assertAlmostEqual(evidence.nearest_cell_xy[1], 0.05, places=6)

    def test_evidence_says_whether_the_frame_or_the_memory_saw_the_cell(self) -> None:
        cells = {
            "chassis": np.zeros((0, 2)),
            "arms_0p77": np.zeros((0, 2)),
            # Two live cells, then the remembered one appended behind them, in
            # the order the controller stacks them.
            "arms_0p82": np.array([[0.36, 0.02], [0.38, -0.08], [0.33, 0.05]]),
        }
        result = self.check(
            cells,
            {"chassis": 0, "arms_0p77": 0, "arms_0p82": 2},
            memory_cells=1,
        )
        evidence = result.blocking_layers()[0].blocking

        self.assertEqual(evidence.live_cells, 1)
        self.assertEqual(evidence.remembered_cells, 1)
        self.assertIn("1 live/1 remembered", result.describe_block())

    def test_the_split_reads_the_live_half_from_the_front_of_the_stack(self) -> None:
        """Pins the direction of the convention, which an asymmetric count cannot.

        Cells arrive live-first and remembered-second, and the live count says
        where the boundary is. Reading it from the other end is invisible while
        every occupied cell blocks, because both readings then count the same
        number of rows. It only shows up when the band also holds a cell that
        does NOT block, which moves the blocking rows off centre.
        """

        near, far = [0.33, 0.05], [0.50, 0.0]

        live_first = self.check(
            {
                "chassis": np.zeros((0, 2)),
                "arms_0p77": np.zeros((0, 2)),
                # The blocker is the live row; the remembered row is too far
                # ahead to block.
                "arms_0p82": np.array([near, far]),
            },
            {"chassis": 0, "arms_0p77": 0, "arms_0p82": 1},
            memory_cells=1,
        ).blocking_layers()[0].blocking

        self.assertEqual(live_first.count, 1)
        self.assertEqual(live_first.live_cells, 1)
        self.assertEqual(live_first.remembered_cells, 0)

        remembered_blocks = self.check(
            {
                "chassis": np.zeros((0, 2)),
                "arms_0p77": np.zeros((0, 2)),
                # Now the other way round: the frame saw only the far cell and
                # the memory is the one holding the blocker.
                "arms_0p82": np.array([far, near]),
            },
            {"chassis": 0, "arms_0p77": 0, "arms_0p82": 1},
            memory_cells=1,
        ).blocking_layers()[0].blocking

        self.assertEqual(remembered_blocks.count, 1)
        self.assertEqual(remembered_blocks.live_cells, 0)
        self.assertEqual(remembered_blocks.remembered_cells, 1)

    def test_a_band_the_frame_did_not_see_reports_only_remembered_cells(self) -> None:
        """The case that matters on a robot: the camera cannot see it any more.

        A reverse command consults no depth at all, so every cell in the band
        came from the occupancy memory and none of it should read as evidence
        this frame measured.
        """

        cells = {
            "chassis": np.zeros((0, 2)),
            "arms_0p77": np.zeros((0, 2)),
            "arms_0p82": np.array([[0.36, 0.02], [0.33, 0.05]]),
        }
        result = self.check(
            cells,
            {"chassis": 0, "arms_0p77": 0, "arms_0p82": 0},
            memory_cells=2,
        )
        evidence = result.blocking_layers()[0].blocking

        self.assertEqual(evidence.live_cells, 0)
        self.assertEqual(evidence.remembered_cells, 2)
        self.assertIn("0 live/2 remembered", result.describe_block())

    def test_all_cells_count_as_live_when_no_split_is_given(self) -> None:
        cells = {
            "chassis": np.zeros((0, 2)),
            "arms_0p77": np.zeros((0, 2)),
            "arms_0p82": np.array([[0.36, 0.02], [0.33, 0.05]]),
        }
        evidence = self.check(cells, {}).blocking_layers()[0].blocking

        self.assertEqual(evidence.live_cells, 2)
        self.assertEqual(evidence.remembered_cells, 0)

    def test_a_cell_just_outside_the_body_reads_as_the_robot_itself(self) -> None:
        """The signature of a self-filter leftover, against a real obstacle.

        Both refuse the move. Only the distance outside the body outline says
        which one the robot should have driven through.
        """

        leftover = {
            "chassis": np.zeros((0, 2)),
            "arms_0p77": np.zeros((0, 2)),
            # Inside the band outline already and 0.01 m past the body edge.
            "arms_0p82": np.array([[0.23, 0.0]]),
        }
        obstacle = {
            "chassis": np.zeros((0, 2)),
            "arms_0p77": np.zeros((0, 2)),
            "arms_0p82": np.array([[0.33, 0.05]]),
        }

        near = self.check(leftover, {}).blocking_layers()[0]
        far = self.check(obstacle, {}).blocking_layers()[0]

        self.assertAlmostEqual(near.blocking.nearest_outside_body_m, 0.01, places=6)
        self.assertAlmostEqual(far.blocking.nearest_outside_body_m, 0.11, places=6)
        # The overlapping case is the one that reports a negative free
        # distance, as the robot's own outline already covers the cell.
        self.assertLess(near.blocking.nearest_free_distance_m, 0.0)
        self.assertGreater(far.blocking.nearest_free_distance_m, 0.0)

    def test_samples_stay_short_enough_to_survive_trace_summarising(self) -> None:
        many = np.stack(
            [
                np.full(40, 0.31) + np.arange(40) * 1e-4,
                np.linspace(-0.20, 0.20, 40),
            ],
            axis=1,
        )
        cells = {
            "chassis": np.zeros((0, 2)),
            "arms_0p77": np.zeros((0, 2)),
            "arms_0p82": many,
        }
        evidence = self.check(cells, {}).blocking_layers()[0].blocking

        self.assertEqual(evidence.count, 40)
        self.assertLessEqual(len(evidence.sample_cells_xy), 8)
        # Nearest first, so a reader sees the cluster that actually stopped it.
        self.assertAlmostEqual(
            evidence.sample_cells_xy[0][0], float(np.min(many[:, 0])), places=6
        )

    def test_a_cell_the_outline_already_overlaps_sorts_to_the_front(self) -> None:
        """An overlapping cell has a negative free distance, not a missing one.

        It is the cell that matters most — the robot is standing in it — so
        the evidence has to name it, not a neighbour further ahead.
        """

        cells = {
            "chassis": np.zeros((0, 2)),
            "arms_0p77": np.zeros((0, 2)),
            "arms_0p82": np.array([[0.36, 0.02], [0.25, 0.10]]),
        }
        evidence = self.check(cells, {}).blocking_layers()[0].blocking

        self.assertEqual(evidence.count, 2)
        self.assertAlmostEqual(evidence.nearest_free_distance_m, -0.05, places=6)
        self.assertAlmostEqual(evidence.nearest_cell_xy[0], 0.25, places=6)

    def test_a_blocking_cell_always_has_a_measurable_free_distance(self) -> None:
        """Why the non-finite guard in the ranking is defensive, not load-bearing.

        polygon_support_along has no front to offer on a lateral line that
        misses the outline, and a cell there would come back at +inf. It
        cannot happen for a blocking cell: the swept hull is the outline
        translated ALONG the direction, so it spans exactly the same lateral
        range as the outline, and anything inside it has a finite front.
        This pins that invariant on a diagonal sweep, where it is least
        obvious, so a future change to the sweep shape has to face it.
        """

        for velocity in ((0.10, 0.0), (0.0707, 0.0707), (0.0, 0.10)):
            with self.subTest(velocity=velocity):
                result = check_cells(
                    {
                        "chassis": np.zeros((0, 2)),
                        "arms_0p77": np.zeros((0, 2)),
                        # A spray of cells all around the band, well past the
                        # outline's lateral span on both sides.
                        "arms_0p82": np.stack(
                            [
                                np.linspace(-0.60, 0.60, 25),
                                np.linspace(0.60, -0.60, 25),
                            ],
                            axis=1,
                        ),
                    },
                    self.footprint,
                    velocity,
                    frame_age_s=0.0,
                    command_latency_s=0.0,
                    lease_s=0.0,
                    brake_accel_mps2=0.6,
                    margin_m=0.30,
                )
                for layer in result.blocking_layers():
                    self.assertTrue(
                        math.isfinite(layer.blocking.nearest_free_distance_m),
                        layer.describe(),
                    )

    def test_a_clear_result_says_what_the_nearest_thing_was(self) -> None:
        cells = {
            "chassis": np.zeros((0, 2)),
            "arms_0p77": np.array([[0.90, 0.0]]),
            "arms_0p82": np.zeros((0, 2)),
        }
        result = self.check(cells, {})

        self.assertTrue(result.clear)
        self.assertEqual(result.blocking_layers(), [])
        self.assertIn("clear:", result.describe_block())
        self.assertIn("arms_0p77", result.describe_block())

    def test_distance_outside_the_body_does_not_depend_on_winding(self) -> None:
        points = np.array([[0.30, 0.31], [0.0, 0.0], [0.23, 0.0]])

        counter_clockwise = distance_outside_polygon(points, self.body)
        clockwise = distance_outside_polygon(points, self.body[::-1])

        np.testing.assert_allclose(counter_clockwise, clockwise)
        self.assertEqual(counter_clockwise[1], 0.0)
        self.assertAlmostEqual(counter_clockwise[2], 0.01, places=6)




def deep_footprint(band_count: int = 9) -> FootprintConfig:
    """A chassis plus enough arm bands to outrun ``MAX_REPR_CHARS``."""

    layers = [band("chassis", 0.00, 0.27, front=0.22, half_width=0.27)]
    for index in range(band_count):
        z = 0.52 + 0.05 * index
        layers.append(
            band(f"arms_{int(round(z * 100)):02d}", z, z + 0.05, front=0.30, half_width=0.43)
        )
    return FootprintConfig.from_mapping({"layers": [dict(l) for l in layers]})


def refusal() -> tuple[FootprintConfig, ClearanceResult]:
    """A real refusal from the real gate, with the top band the one that blocks.

    Built rather than hand-written so the assertions below are about what
    ``ClearanceResult.summary()`` actually emits — key order included — and
    not about a literal that could drift away from it.
    """

    footprint = deep_footprint()
    names = [layer.name for layer in footprint.layers]
    cells = {name: np.zeros((0, 2)) for name in names}
    # Something far ahead in the lower bands: a free distance, never a block.
    for name in names[1:-1]:
        cells[name] = np.array([[0.80, 0.0], [0.82, 0.10]])
    # The refusing band is last, where a truncated repr never reached. Two
    # cells this frame saw and one the occupancy memory carried.
    cells[names[-1]] = np.array([[0.36, 0.02], [0.34, -0.02], [0.33, 0.05]])
    live_counts = {name: len(value) for name, value in cells.items()}
    live_counts[names[-1]] = 2
    result = check_cells(
        cells,
        footprint,
        (0.10, 0.0),
        frame_age_s=0.0,
        command_latency_s=0.0,
        lease_s=0.0,
        brake_accel_mps2=0.6,
        margin_m=0.05,
        live_cell_counts=live_counts,
        memory_cells=1,
    )
    assert not result.clear
    return footprint, result


def guard_block_payload() -> dict:
    """A refused move nested the way the primitives actually record one.

    ``dock_to_visible_object`` stores the drive's gate metrics under
    ``metrics.last_motion_guard_block.metrics``, one level below where trace
    summarising stops descending. ``motion_history`` carries the same kind of
    payload for every drive of the run and is included so the depth exception
    can be shown not to lift those too.
    """

    _, result = refusal()
    drive_metrics = {
        "front_clearance_m": result.min_free_distance_m,
        "clearance_blocked_by": result.describe_block(),
        "clearance": result.summary(),
    }
    return {
        "primitive": "dock_to_visible_object",
        "metrics": {
            "last_motion_guard_block": {
                "reason": "obstacle_too_close",
                "metrics": drive_metrics,
            },
        },
        "motion_history": [
            {"primitive": "drive_straight", "metrics": dict(drive_metrics)}
            for _ in range(3)
        ],
    }


class GuardBlockTraceTest(unittest.TestCase):
    """The refusal evidence has to reach the trace as data, not as a repr."""

    def setUp(self) -> None:
        self.payload = guard_block_payload()
        self.summarized = _summarize(self.payload)
        self.guard = self.summarized["metrics"]["last_motion_guard_block"]["metrics"]
        _, self.result = refusal()
        self.band_name = self.result.blocking_layers()[0].name

    def test_the_per_layer_list_survives_as_structured_data(self) -> None:
        self.assertIsInstance(self.guard["clearance"], dict)
        self.assertEqual(
            len(self.guard["clearance"]["layers"]), len(self.result.layers)
        )

    def test_the_band_that_refused_is_recoverable_from_the_layer_list(self) -> None:
        blocking = [
            layer["name"]
            for layer in self.guard["clearance"]["layers"]
            if layer["blocking_cells"]
        ]

        self.assertEqual(blocking, [self.band_name])

    def test_the_blocking_cells_keep_their_position_and_provenance(self) -> None:
        layers = self.guard["clearance"]["layers"]
        evidence = next(
            layer["blocking"] for layer in layers if layer["blocking_cells"]
        )

        self.assertEqual(evidence["live_cells"], 2)
        self.assertEqual(evidence["remembered_cells"], 1)
        self.assertEqual(len(evidence["nearest_cell_xy"]), 2)

    def test_the_band_is_named_before_the_list_that_would_be_cut(self) -> None:
        """The property that makes the evidence survive any future truncation.

        The per-band list alone outruns ``MAX_REPR_CHARS`` and the refusing
        band is last in it, so a repr of the list loses the answer. The
        summary's own keys carry the name, and they are emitted first.
        """

        summary = self.result.summary()

        self.assertGreater(len(repr(summary["layers"])), MAX_REPR_CHARS)
        self.assertNotIn(self.band_name, repr(summary["layers"])[:MAX_REPR_CHARS])
        # Emitted before "layers", so a repr of the whole summary keeps it.
        keys = list(summary)
        self.assertLess(keys.index("blocked_by"), keys.index("layers"))
        self.assertLess(keys.index("blocking_layer_names"), keys.index("layers"))
        self.assertIn(self.band_name, repr(summary)[:MAX_REPR_CHARS])

    def test_the_flat_one_liner_fits_inside_the_repr_budget(self) -> None:
        flat = self.guard["clearance_blocked_by"]

        self.assertLess(len(flat), MAX_REPR_CHARS)
        self.assertIn(self.band_name, flat)

    def test_the_exception_does_not_lift_every_drive_in_the_history(self) -> None:
        """The depth budget is keyed on the guard block, not on "clearance".

        Every entry of a motion history carries gate metrics too. Lifting
        those out of the repr as well doubled a docking trace and bought
        nothing the refusal does not already record.
        """

        history = self.summarized["motion_history"]

        self.assertEqual(len(history), 3)
        for entry in history:
            self.assertIsInstance(entry["metrics"]["clearance"], str)

    def test_a_collapsed_history_entry_still_names_the_band(self) -> None:
        """What the flat one-liner buys where the structure is given up.

        A history entry's gate metrics stay a repr, but the one-liner beside
        them is a short string and rides out at full length, so even there the
        band that refused is recoverable.
        """

        for entry in self.summarized["motion_history"]:
            flat = entry["metrics"]["clearance_blocked_by"]
            self.assertIn(self.band_name, flat)

    def test_ordinary_primitive_data_keeps_its_shallow_bound(self) -> None:
        """The depth exception is for failure evidence, not for everything."""

        deep = {"a": {"b": {"c": {"d": {"e": {"f": "too deep"}}}}}}

        summarized = _summarize(deep)

        self.assertIsInstance(summarized["a"]["b"]["c"]["d"], str)


class DirectDriveTraceTest(unittest.TestCase):
    """A drive's own result, not a guard block, as the executor traces it."""

    def setUp(self) -> None:
        _, self.result = refusal()
        self.summarized = _summarize(
            {"primitive": "drive_straight", "metrics": {"clearance": self.result.summary()}}
        )

    def test_layer_items_stay_structured_in_a_drive_result(self) -> None:
        """The list sits at depth three here, one short of the usual bound."""

        layers = self.summarized["metrics"]["clearance"]["layers"]

        self.assertEqual(len(layers), len(self.result.layers))
        self.assertTrue(all(isinstance(layer, dict) for layer in layers))

    def test_the_blocking_evidence_is_readable_without_parsing(self) -> None:
        layers = self.summarized["metrics"]["clearance"]["layers"]
        blocking = [layer for layer in layers if layer["blocking_cells"]]

        self.assertEqual(len(blocking), 1)
        self.assertEqual(blocking[0]["blocking"]["live_cells"], 2)
        self.assertEqual(blocking[0]["blocking"]["remembered_cells"], 1)


class ManyBlockingBandsTest(unittest.TestCase):
    """Which band leads, and what the one-liner keeps when several refuse.

    A docking pose puts a surface in front of most arm bands at once, so the
    several-band case is the normal one on the robot, not an edge case.
    """

    def setUp(self) -> None:
        self.footprint = deep_footprint()
        names = [layer.name for layer in self.footprint.layers]
        self.arm_names = names[1:]
        cells = {name: np.zeros((0, 2)) for name in names}
        # Five bands block, each a little nearer than the one below it, so the
        # expected order is the reverse of the band order.
        for index, name in enumerate(self.arm_names[:5]):
            cells[name] = np.array([[0.36 - 0.005 * index, 0.0]])
        self.result = check_cells(
            cells,
            self.footprint,
            (0.10, 0.0),
            frame_age_s=0.0,
            command_latency_s=0.0,
            lease_s=0.0,
            brake_accel_mps2=0.6,
            margin_m=0.05,
        )

    def test_the_nearest_obstacle_leads_whatever_the_band_order(self) -> None:
        blocking = self.result.blocking_layers()

        self.assertEqual(len(blocking), 5)
        distances = [layer.blocking.nearest_free_distance_m for layer in blocking]
        self.assertEqual(distances, sorted(distances))
        # Band 5 holds the nearest cell; the footprint lists it last of the five.
        self.assertEqual(blocking[0].name, self.arm_names[4])
        self.assertEqual(
            self.result.summary()["blocking_layer_names"][0], self.arm_names[4]
        )

    def test_the_one_liner_keeps_three_bands_and_counts_the_rest(self) -> None:
        text = self.result.describe_block()

        self.assertEqual(text.count("m outside body"), 3)
        self.assertIn("+2 more bands", text)
        self.assertLess(len(text), MAX_REPR_CHARS)

    def test_every_band_keeps_its_height_in_the_summary(self) -> None:
        layers = self.result.summary()["layers"]

        self.assertEqual(len(layers), len(self.footprint.layers))
        for layer in layers:
            self.assertLess(layer["z_min"], layer["z_max"])


if __name__ == "__main__":
    unittest.main()
