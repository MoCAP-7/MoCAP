"""The cuRobo mesh-query patch must edit exactly the two kernel sites and be reversible."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest


GRASP_MOTION_DIRECTORY = Path(__file__).resolve().parents[1] / "services/grasp_motion"
if str(GRASP_MOTION_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(GRASP_MOTION_DIRECTORY))

import patch_curobo_mesh_query as patch  # noqa: E402


SAMPLE = """
def compute_local_sdf(obs_set, env_idx, local_idx, local_pt):
    bounding_box_size = wp.vec3(obs_set.dims[flat_idx,0],
    obs_set.dims[flat_idx,1],
    obs_set.dims[flat_idx,2])
    max_distance = wp.length(bounding_box_size) * 0.5

    result = wp.mesh_query_point(mesh_id, local_pt, max_distance)


def compute_local_sdf_with_grad(obs_set, env_idx, local_idx, local_pt, query_distance):
    bounding_box_size = wp.vec3(obs_set.dims[flat_idx,0],
    obs_set.dims[flat_idx,1],
    obs_set.dims[flat_idx,2])
    max_distance = wp.length(bounding_box_size) * 0.5
    max_distance = wp.max(max_distance, query_distance)

    def to_warp(self):
        s.max_dist = max_dist
"""


class CuroboMeshQueryPatchTest(unittest.TestCase):
    def test_apply_is_idempotent_and_reversible(self) -> None:
        patched, status = patch.apply_patch(SAMPLE)
        self.assertEqual(status, "applied")
        self.assertEqual(patched.count(patch.PATCHED), 2)
        self.assertEqual(patched.count(patch.ORIGINAL), 0)
        # The with-grad site keeps the query_distance floor after the clamp.
        self.assertIn(
            patch.PATCHED + "    max_distance = wp.max(max_distance, query_distance)\n",
            patched,
        )
        again, status = patch.apply_patch(patched)
        self.assertEqual(status, "unchanged")
        self.assertEqual(again, patched)
        reverted, status = patch.apply_patch(patched, revert=True)
        self.assertEqual(status, "reverted")
        self.assertEqual(reverted, SAMPLE)

    def test_refuses_an_unexpected_curobo_source(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "expected 2 occurrences"):
            patch.apply_patch(SAMPLE.replace(patch.ORIGINAL, "    max_distance = 1.0\n", 1))
        with self.assertRaisesRegex(RuntimeError, "max_dist is not populated"):
            patch.apply_patch(SAMPLE.replace("s.max_dist = max_dist", ""))


if __name__ == "__main__":
    unittest.main()
