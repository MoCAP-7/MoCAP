"""Greedy voxel-surface meshing must reproduce cuRobo's per-voxel surface exactly."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


GRASP_MOTION_DIRECTORY = Path(__file__).resolve().parents[1] / "services/grasp_motion"
if str(GRASP_MOTION_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(GRASP_MOTION_DIRECTORY))

import voxel_surface  # noqa: E402  (curobo-free helper module)


def _unit_faces(vertices: np.ndarray, triangles: np.ndarray, pitch: float, origin: np.ndarray):
    """Rasterize triangles (all axis-aligned rectangles) back into unit voxel faces.

    Returns a set of (axis, slab, row, column, orientation) keys so two meshes
    with different rectangle decompositions can be compared face by face.
    Orientation is the sign of the triangle normal along ``axis``.
    """

    faces = set()
    grid = np.round((vertices - origin) / pitch).astype(np.int64)
    for a, b, c in triangles:
        pa, pb, pc = grid[a], grid[b], grid[c]
        normal = np.cross(pb - pa, pc - pa)
        axis = int(np.flatnonzero(normal)[0])
        orientation = int(np.sign(normal[axis]))
        slab = int(pa[axis])
        in_plane = [i for i in range(3) if i != axis]
        lo = np.minimum(np.minimum(pa, pb), pc)
        hi = np.maximum(np.maximum(pa, pb), pc)
        # Each triangle covers half of its rectangle; collect the rectangle
        # once per pair by adding all unit faces (set semantics dedupe).
        for row in range(lo[in_plane[0]], hi[in_plane[0]]):
            for column in range(lo[in_plane[1]], hi[in_plane[1]]):
                faces.add((axis, slab, row, column, orientation))
    return faces


class VoxelSurfaceTest(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(3)
        xs = np.arange(-0.3, 0.3, 0.008)
        ys = np.arange(-0.6, -0.2, 0.008)
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        table = np.column_stack([X.ravel(), Y.ravel(), np.full(X.size, -0.03)])
        zs = np.arange(-0.03, 0.25, 0.008)
        Xw, Zw = np.meshgrid(xs, zs, indexing="ij")
        wall = np.column_stack([Xw.ravel(), np.full(Xw.size, -0.6), Zw.ravel()])
        box = rng.uniform([0.0, -0.45, -0.03], [0.06, -0.39, 0.09], size=(400, 3))
        self.points = np.concatenate([table, wall, box]).astype(np.float32)
        self.points += rng.normal(0.0, 0.002, self.points.shape).astype(np.float32)
        self.pitch = 0.012

    def test_greedy_surface_equals_per_voxel_surface_with_fewer_triangles(self) -> None:
        occupied, origin = voxel_surface.voxelize(self.points, self.pitch)
        reference_vertices, reference_triangles = voxel_surface.per_voxel_surface(
            self.points, self.pitch
        )
        vertices, triangles, stats = voxel_surface.greedy_voxel_surface(
            self.points, self.pitch
        )

        self.assertEqual(stats["per_voxel_triangles"], len(reference_triangles))
        self.assertLess(len(triangles), len(reference_triangles) / 4)
        self.assertEqual(stats["triangles"], len(triangles))
        self.assertEqual(stats["occupied_voxels"], int(occupied.sum()))
        # Same vertex lattice.
        lattice = (vertices - origin) / self.pitch
        np.testing.assert_allclose(lattice, np.round(lattice), atol=1e-9)
        # Same set of exposed unit faces with the same orientation.
        self.assertEqual(
            _unit_faces(vertices, triangles, self.pitch, origin),
            _unit_faces(reference_vertices, reference_triangles, self.pitch, origin),
        )

    def test_reference_matches_curobo_face_templates_for_one_voxel(self) -> None:
        vertices, triangles = voxel_surface.per_voxel_surface(
            np.asarray([[0.0, 0.0, 0.0]]), 0.01
        )
        self.assertEqual(len(triangles), 12)
        self.assertEqual(len(vertices), 24)
        # cuRobo's from_pointcloud templates wind every face the same way
        # (normals point into the voxel). The greedy mesh must keep exactly that
        # convention, so assert consistency rather than a particular direction.
        centre = vertices.mean(axis=0)
        signs = {
            int(np.sign(np.dot(np.cross(vertices[b] - vertices[a], vertices[c] - vertices[a]), vertices[a] - centre)))
            for a, b, c in triangles
        }
        self.assertEqual(signs, {-1})
        greedy_vertices, greedy_triangles, _ = voxel_surface.greedy_voxel_surface(
            np.asarray([[0.0, 0.0, 0.0]]), 0.01
        )
        np.testing.assert_allclose(greedy_vertices, vertices)
        np.testing.assert_array_equal(greedy_triangles, triangles)

    def test_greedy_rectangles_cover_each_cell_once(self) -> None:
        mask = np.zeros((6, 7), dtype=bool)
        mask[1:4, 1:6] = True
        mask[4, 2:4] = True
        mask[0, 6] = True
        rectangles = voxel_surface.greedy_rectangles(mask)
        cover = np.zeros_like(mask, dtype=int)
        for row, column, height, width in rectangles:
            cover[row : row + height, column : column + width] += 1
        np.testing.assert_array_equal(cover, mask.astype(int))
        self.assertEqual(len(rectangles), 3)

    def test_tiled_surfaces_partition_the_per_voxel_surface_exactly(self) -> None:
        occupied, origin = voxel_surface.voxelize(self.points, self.pitch)
        reference_vertices, reference_triangles = voxel_surface.per_voxel_surface(
            self.points, self.pitch
        )
        tiles, stats = voxel_surface.tiled_voxel_surfaces(self.points, self.pitch, 0.25)

        self.assertGreater(stats["tiles"], 3)
        self.assertEqual(stats["triangles"], len(reference_triangles))
        self.assertEqual(sum(len(t) for _, t in tiles), len(reference_triangles))
        union = set()
        for vertices, triangles in tiles:
            faces = _unit_faces(vertices, triangles, self.pitch, origin)
            self.assertTrue(union.isdisjoint(faces))
            union |= faces
            # Every tile fits in a tile_m cube plus one voxel of face extent.
            extent = vertices.max(axis=0) - vertices.min(axis=0)
            self.assertTrue(np.all(extent <= stats["tile_m"] + self.pitch + 1e-9))
        self.assertEqual(
            union, _unit_faces(reference_vertices, reference_triangles, self.pitch, origin)
        )

    def test_service_builds_the_world_from_the_greedy_surface(self) -> None:
        source = (GRASP_MOTION_DIRECTORY / "service.py").read_text(encoding="utf-8")
        self.assertIn("greedy_voxel_surface(points, pitch_m)", source)
        self.assertIn("tiled_voxel_surfaces(points, pitch_m, tile_m)", source)
        self.assertIn("crop_points(filtered, options[\"crop_radius_m\"])", source)
        self.assertIn('collision_cache={"mesh": self.mesh_slots}', source)
        self.assertIn("SCENE_MESH_SLOTS = 1", source)
        self.assertIn('"--scene-mesh-slots"', source)
        self.assertIn('SCENE_MESH_MODES = ("tiled", "greedy", "per_voxel")', source)
        self.assertIn('request.get("scene_mesh", "per_voxel")', source)
        # Finetune cap is installed before warmup and reset after every request.
        self.assertIn("install_finetune_cap(", source)
        self.assertIn("planner.set_finetune_cap(None)", source)
        self.assertIn("SCENE_CROP_RADIUS_MINIMUM_M = 1.5", source)
        # Every world-loading action passes tool centres for the fine region.
        self.assertEqual(source.count("tool_centres=["), 3)

    def test_crop_points_keeps_boundary_and_counts_dropped(self) -> None:
        points = np.asarray([[1.0, 0.0, 0.0], [1.0 + 1e-6, 0.0, 0.0], [0.2, 0.1, 0.0]])
        kept, dropped = voxel_surface.crop_points(points, 1.0)
        self.assertEqual(dropped, 1)
        np.testing.assert_array_equal(kept, points[[0, 2]])
        with self.assertRaises(ValueError):
            voxel_surface.crop_points(points, 0.0)


if __name__ == "__main__":
    unittest.main()
