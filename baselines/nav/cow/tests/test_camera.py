import math
import unittest

import numpy as np

from baselines.nav.cow.camera import LevelCamera, LevelCameraModel, level_rotation


INTRINSICS = ((267.2, 0.0, 337.2), (0.0, 267.2, 182.3), (0.0, 0.0, 1.0))
WIDTH, HEIGHT = 672, 376


def tilted_down(pitch_deg):
    pitch = math.radians(pitch_deg)
    return (0.0, math.cos(pitch), math.sin(pitch))


def floor_depth(pitch_deg, camera_height_m):
    """Z-depth image of an infinite floor seen by a camera pitched down by pitch_deg."""

    k = np.asarray(INTRINSICS)
    u, v = np.meshgrid(np.arange(WIDTH, dtype=np.float64), np.arange(HEIGHT, dtype=np.float64))
    rays = np.stack([(u - k[0, 2]) / k[0, 0], (v - k[1, 2]) / k[1, 1], np.ones_like(u)], axis=-1)
    down = np.asarray(tilted_down(pitch_deg))
    along_down = rays @ down
    depth = np.where(along_down > 1e-6, camera_height_m / np.maximum(along_down, 1e-6), 0.0)
    return depth.astype(np.float32)


def cow_unproject(depth, fov_deg):
    """CoW's square single-FOV unprojection, as optical-frame x right, y down."""

    size = depth.shape[0]
    focal = (size / 2.0) / math.tan(math.radians(fov_deg) / 2.0)
    centres = np.arange(size) + 0.5 - size / 2.0
    x, y = np.meshgrid(centres / focal, centres / focal)
    return x * depth, y * depth


class LevelRotationTest(unittest.TestCase):
    def test_level_camera_is_identity(self):
        np.testing.assert_allclose(level_rotation((0.0, 1.0, 0.0)), np.eye(3), atol=1e-12)

    def test_tilted_camera_keeps_heading_and_levels_forward(self):
        rotation = level_rotation(tilted_down(20.0))
        forward_in_camera = rotation[2]
        self.assertAlmostEqual(float(np.dot(forward_in_camera, tilted_down(20.0))), 0.0, places=12)
        self.assertLess(forward_in_camera[1], 0.0)  # the horizon is above the image centre
        np.testing.assert_allclose(rotation[0], [1.0, 0.0, 0.0], atol=1e-12)

    def test_rejects_degenerate_down_vector(self):
        with self.assertRaises(ValueError):
            level_rotation((0.0, 0.0, 0.0))
        with self.assertRaises(ValueError):
            level_rotation((0.0, 0.0, 1.0))


class LevelCameraTest(unittest.TestCase):
    def make(self, pitch_deg=20.0, fov_deg=90.0, size=672):
        return LevelCamera(
            LevelCameraModel(
                source_width=WIDTH,
                source_height=HEIGHT,
                intrinsics=INTRINSICS,
                down_camera_xyz=tilted_down(pitch_deg),
                output_size=size,
                fov_deg=fov_deg,
            )
        )

    def test_floor_lands_at_camera_height_under_cow_unprojection(self):
        camera_height = 1.05
        camera = self.make()
        level = camera.render_depth(floor_depth(20.0, camera_height))
        valid = level > 0.0
        self.assertGreater(valid.mean(), 0.2)
        _, y_down = cow_unproject(level, 90.0)
        error = np.abs(y_down[valid] - camera_height)
        near = level[valid] <= 8.0
        self.assertLess(float(error[near].max()), 0.03)
        # Beyond that the error comes only from the angle between a virtual
        # pixel and its nearest source pixel, so it stays proportional to range.
        self.assertTrue(bool(np.all(error <= 0.004 * level[valid] + 0.005)))

    def test_level_forward_ray_samples_the_source_horizon_pixel(self):
        camera = self.make(size=64)
        rotation = level_rotation(tilted_down(20.0))
        k = np.asarray(INTRINSICS)
        # The virtual pixel just below and right of the image centre looks almost straight ahead.
        ray_level = np.array([0.5, 0.5, 32.0]) / 32.0
        ray_camera = rotation.T @ ray_level
        u = int(round(k[0, 0] * ray_camera[0] / ray_camera[2] + k[0, 2]))
        v = int(round(k[1, 1] * ray_camera[1] / ray_camera[2] + k[1, 2]))
        rgb = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        rgb[v - 3 : v + 4, u - 3 : u + 4] = (255, 40, 10)
        rendered = camera.render_rgb(rgb)
        np.testing.assert_array_equal(rendered[32, 32], [255, 40, 10])

    def test_invalid_depth_and_outside_pixels_are_zero(self):
        camera = self.make()
        depth = np.full((HEIGHT, WIDTH), 2.0, dtype=np.float32)
        depth[:, : WIDTH // 2] = np.nan
        depth[:10] = np.inf
        level = camera.render_depth(depth)
        self.assertTrue(np.isfinite(level).all())
        self.assertEqual(float(level[0, 0]), 0.0)  # above the tilted camera's view
        self.assertEqual(float(level[400, 10]), 0.0)  # left half was NaN
        self.assertGreater(float(level[400, 600]), 0.0)
        self.assertLess(camera.coverage, 1.0)

    def test_frame_shape_is_checked(self):
        camera = self.make()
        with self.assertRaises(ValueError):
            camera.render(np.zeros((10, 10, 3), dtype=np.uint8), np.zeros((10, 10), dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
