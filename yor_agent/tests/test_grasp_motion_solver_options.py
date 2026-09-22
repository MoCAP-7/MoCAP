"""Per-request finetune cap for the cuRobo trajectory optimizer."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest


GRASP_MOTION_DIRECTORY = Path(__file__).resolve().parents[1] / "services/grasp_motion"
if str(GRASP_MOTION_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(GRASP_MOTION_DIRECTORY))

import solver_options  # noqa: E402  (curobo-free helper module)


class _FakeTrajOpt:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def solve_pose(self, goal, state, **kwargs):
        self.calls.append(("solve_pose", dict(kwargs)))
        return "pose"

    def solve_cspace(self, goal, state, **kwargs):
        self.calls.append(("solve_cspace", dict(kwargs)))
        return "cspace"


class SolverOptionsTest(unittest.TestCase):
    def test_cap_never_raises_and_defaults_to_curobo_default(self) -> None:
        self.assertEqual(
            solver_options.cap_finetune_attempts({"finetune_attempts": 3}, 1),
            {"finetune_attempts": 1},
        )
        self.assertEqual(
            solver_options.cap_finetune_attempts({"finetune_attempts": 0}, 2),
            {"finetune_attempts": 0},
        )
        # Goalset path passes no finetune_attempts: cuRobo's default is 1.
        self.assertEqual(
            solver_options.cap_finetune_attempts({"seed_config": "s"}, 0),
            {"seed_config": "s", "finetune_attempts": 0},
        )
        untouched = {"finetune_attempts": 3}
        self.assertIs(solver_options.cap_finetune_attempts(untouched, None), untouched)

    def test_parse_bounds(self) -> None:
        self.assertIsNone(solver_options.parse_finetune_cap(None))
        self.assertEqual(solver_options.parse_finetune_cap(0), 0)
        self.assertEqual(solver_options.parse_finetune_cap("2"), 2)
        for bad in (-1, 4, True):
            with self.assertRaises(ValueError):
                solver_options.parse_finetune_cap(bad)

    def test_install_wraps_both_solvers_and_reads_the_cap_each_call(self) -> None:
        solver = _FakeTrajOpt()
        cap = {"value": None}
        wrapped = solver_options.install_finetune_cap(solver, lambda: cap["value"])
        self.assertEqual(wrapped, ["solve_pose", "solve_cspace"])

        solver.solve_cspace("g", "s", seed_traj=None, finetune_attempts=3, finetune_dt_scale=0.75)
        cap["value"] = 1
        solver.solve_cspace("g", "s", seed_traj=None, finetune_attempts=3, finetune_dt_scale=0.75)
        solver.solve_pose("g", "s", seed_config="c", use_implicit_goal=True)
        cap["value"] = 0
        self.assertEqual(solver.solve_pose("g", "s", finetune_attempts=1), "pose")

        self.assertEqual(
            [call[1]["finetune_attempts"] for call in solver.calls], [3, 1, 1, 0]
        )
        self.assertEqual(solver.calls[2][1]["seed_config"], "c")


if __name__ == "__main__":
    unittest.main()
