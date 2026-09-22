"""Unit tests for the cuRoboV2 Warp mesh cache eviction used by the planner service."""

from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest


GRASP_MOTION_DIRECTORY = Path(__file__).resolve().parents[1] / "services/grasp_motion"
if str(GRASP_MOTION_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(GRASP_MOTION_DIRECTORY))

import scene_cache  # noqa: E402  (curobo-free helper module)


class FakeMeshData:
    """Mimic cuRoboV2 MeshData: name-keyed Warp cache plus per-env active names."""

    def __init__(self, max_n: int = 1, num_envs: int = 1) -> None:
        self.max_n = max_n
        self.num_envs = num_envs
        self.wp_cache: dict[str, SimpleNamespace] = {}
        self.names: list[list[str | None]] = [
            [None] * max_n for _ in range(num_envs)
        ]
        self._next_id = 100
        self.clear_calls: list[tuple[object, bool]] = []

    def clear(self, env_idx=None, clear_warp_cache: bool = False) -> None:
        self.clear_calls.append((env_idx, clear_warp_cache))
        if env_idx is None:
            self.names = [[None] * self.max_n for _ in range(self.num_envs)]
        else:
            self.names[env_idx] = [None] * self.max_n
        if clear_warp_cache:
            self.wp_cache.clear()

    def add(self, name: str, env_idx: int = 0) -> None:
        # cuRobo's real behavior: a cached name is reused with its OLD geometry.
        if name not in self.wp_cache:
            self.wp_cache[name] = SimpleNamespace(name=name, mesh_id=self._next_id)
            self._next_id += 1
        self.names[env_idx][0] = name

    def load_scene(self, name: str, env_idx: int = 0) -> None:
        """Mimic load_from_scene_cfg: clear active slots (not the cache), then add."""

        self.clear(env_idx)
        self.add(name, env_idx)


class SceneCacheTest(unittest.TestCase):
    def test_reusing_one_name_keeps_the_first_geometry_without_eviction(self) -> None:
        meshes = FakeMeshData()
        meshes.load_scene("observed_scene")
        first_id = meshes.wp_cache["observed_scene"].mesh_id
        meshes.load_scene("observed_scene")
        # This is the stale-world failure mode the service must never hit.
        self.assertEqual(meshes.wp_cache["observed_scene"].mesh_id, first_id)

    def test_evict_then_fresh_name_yields_new_mesh_and_verifies(self) -> None:
        meshes = FakeMeshData()
        meshes.load_scene(scene_cache.observed_scene_name(0))
        first_id = scene_cache.verify_active_mesh(
            meshes, scene_cache.observed_scene_name(0)
        )

        evicted = scene_cache.evict_cached_meshes(meshes)
        self.assertEqual(evicted, [scene_cache.observed_scene_name(0)])
        self.assertEqual(meshes.clear_calls[-1], (None, True))
        self.assertEqual(meshes.wp_cache, {})

        second_name = scene_cache.observed_scene_name(1)
        meshes.load_scene(second_name)
        second_id = scene_cache.verify_active_mesh(meshes, second_name)
        self.assertNotEqual(first_id, second_id)
        self.assertEqual(list(meshes.wp_cache), [second_name])

    def test_evict_falls_back_when_clear_lacks_warp_cache_flag(self) -> None:
        class LegacyMeshData(FakeMeshData):
            def clear(self, env_idx=None):  # type: ignore[override]
                self.clear_calls.append((env_idx, False))
                self.names = [[None] * self.max_n for _ in range(self.num_envs)]

        meshes = LegacyMeshData()
        meshes.load_scene("observed_scene_000000")
        scene_cache.evict_cached_meshes(meshes)
        self.assertEqual(meshes.wp_cache, {})

    def test_verify_rejects_stale_or_mismatched_active_mesh(self) -> None:
        meshes = FakeMeshData(max_n=2)
        meshes.load_scene("observed_scene_000000")
        with self.assertRaises(scene_cache.SceneCacheError):
            scene_cache.verify_active_mesh(meshes, "observed_scene_000001")
        # A leftover cache entry beside the active mesh is also a stale world.
        meshes.wp_cache["leftover"] = SimpleNamespace(name="leftover", mesh_id=7)
        with self.assertRaises(scene_cache.SceneCacheError):
            scene_cache.verify_active_mesh(meshes, "observed_scene_000000")

    def test_mesh_cache_of_walks_curobo_attribute_path(self) -> None:
        meshes = FakeMeshData()
        # cuRoboV2 MotionPlanner: scene_collision_checker is a SceneCollision
        # whose ``scene_model`` is the last SceneCfg (no ``.data``) or None.
        planner = SimpleNamespace(
            scene_collision_checker=SimpleNamespace(
                data=SimpleNamespace(meshes=meshes),
                scene_model=SimpleNamespace(mesh=[], cuboid=[]),
            )
        )
        self.assertIs(scene_cache.mesh_cache_of(planner), meshes)
        wrapped = SimpleNamespace(
            scene_collision_checker=SimpleNamespace(
                scene_model=SimpleNamespace(data=SimpleNamespace(meshes=meshes))
            )
        )
        self.assertIs(scene_cache.mesh_cache_of(wrapped), meshes)
        with self.assertRaises(scene_cache.SceneCacheError):
            scene_cache.mesh_cache_of(SimpleNamespace(scene_collision_checker=None))
        no_cache = SimpleNamespace(
            scene_collision_checker=SimpleNamespace(
                scene_model=SimpleNamespace(
                    data=SimpleNamespace(meshes=SimpleNamespace(wp_cache=None))
                )
            )
        )
        with self.assertRaises(scene_cache.SceneCacheError):
            scene_cache.mesh_cache_of(no_cache)

    def test_verify_active_meshes_accepts_two_fresh_meshes_in_order(self) -> None:
        meshes = FakeMeshData(max_n=2)
        meshes.clear(None)
        meshes.add("observed_scene_000004_fine", 0)
        # FakeMeshData.add writes slot 0; emulate cuRobo appending a second slot.
        meshes.wp_cache["observed_scene_000004_coarse"] = SimpleNamespace(
            name="observed_scene_000004_coarse", mesh_id=555
        )
        meshes.names[0][1] = "observed_scene_000004_coarse"
        ids = scene_cache.verify_active_meshes(
            meshes, ["observed_scene_000004_fine", "observed_scene_000004_coarse"]
        )
        self.assertEqual(set(ids), {"observed_scene_000004_fine", "observed_scene_000004_coarse"})
        with self.assertRaises(scene_cache.SceneCacheError):
            scene_cache.verify_active_meshes(
                meshes, ["observed_scene_000004_coarse", "observed_scene_000004_fine"]
            )
        with self.assertRaises(scene_cache.SceneCacheError):
            scene_cache.verify_active_meshes(meshes, ["observed_scene_000004_fine"])
        with self.assertRaises(scene_cache.SceneCacheError):
            scene_cache.verify_active_meshes(meshes, [])

    def test_observed_scene_names_are_unique_and_prefixed(self) -> None:
        names = {scene_cache.observed_scene_name(index) for index in range(1000)}
        self.assertEqual(len(names), 1000)
        self.assertTrue(
            all(name.startswith(scene_cache.OBSERVED_SCENE_PREFIX) for name in names)
        )
        with self.assertRaises(ValueError):
            scene_cache.observed_scene_name(-1)


if __name__ == "__main__":
    unittest.main()
