import importlib.util
import os
import unittest
from pathlib import Path

from baselines.nav.cow import upstream


COW_REPO = Path(os.environ.get("COW_REPO", "/home/yor/codefield/cow")).expanduser()
HAS_TORCH = importlib.util.find_spec("torch") is not None


class LockTest(unittest.TestCase):
    def test_lock_pins_a_full_commit(self):
        lock = upstream.load_lock()
        self.assertEqual(len(lock["commit"]), 40)
        self.assertIn("real-stanford/cow", lock["repository"])


@unittest.skipUnless(COW_REPO.is_dir() and HAS_TORCH, "set COW_REPO to the pinned CoW checkout")
class ImportCowTest(unittest.TestCase):
    def test_imports_unmodified_clip_grad_agent(self):
        modules = upstream.import_cow(COW_REPO, localizer="clip_grad")
        self.assertEqual(modules.commit, upstream.load_lock()["commit"])
        self.assertEqual(modules.agent_class.__name__, "AgentFbeGrad")
        self.assertEqual(upstream.localizer_threshold(modules, "clip_grad"), 0.625)
        self.assertIn(COW_REPO.resolve(), Path(modules.exploration.__file__).resolve().parents)

    def test_stop_radius_only_rebinds_the_stop_test_constant(self):
        modules = upstream.import_cow(COW_REPO, localizer="clip_grad")
        exploration = modules.exploration
        original = exploration.VOXEL_SIZE_M
        try:
            upstream.set_stop_radius(exploration, voxel_size_m=0.125, stop_radius_m=0.65)
            self.assertAlmostEqual(1.0 / exploration.VOXEL_SIZE_M * 0.125, 0.65)
            with self.assertRaises(ValueError):
                upstream.set_stop_radius(exploration, voxel_size_m=0.125, stop_radius_m=0.0)
        finally:
            exploration.VOXEL_SIZE_M = original


if __name__ == "__main__":
    unittest.main()
