import math
import unittest

import numpy as np

from baselines.nav.apexnav.geometry import (
    base_pose_from_camera,
    camera_rotation_world,
    masked_depth_to_world,
    relative_camera_pose,
    rotation_matrix_to_quaternion,
)


class GeometryTest(unittest.TestCase):
    def test_episode_pose_starts_at_camera_origin(self):
        origin = np.asarray([4.0, -2.0, math.pi / 2.0])
        np.testing.assert_allclose(relative_camera_pose(origin, origin), np.zeros(3))

    def test_base_pose_subtracts_calibrated_camera_lever_arm(self):
        pose = base_pose_from_camera(np.zeros(3), 0.2143, 0.0603)
        np.testing.assert_allclose(pose, [-0.2143, -0.0603, 0.0])

    def test_flat_zed_projection_uses_optical_forward_as_world_forward(self):
        rotation = camera_rotation_world(0.0, [0.0, 1.0, 0.0])
        points = masked_depth_to_world(
            np.asarray([[2.0]], dtype=np.float32),
            np.asarray([[True]]),
            np.eye(3),
            np.asarray([0.0, 0.0, 1.0]),
            rotation,
            minimum_depth_m=0.1,
            maximum_depth_m=5.0,
        )
        np.testing.assert_allclose(points, [[2.0, 0.0, 1.0]], atol=1e-6)

    def test_rotation_quaternion_is_normalized(self):
        rotation = camera_rotation_world(0.7, [0.0, 1.0, 0.0])
        quaternion = rotation_matrix_to_quaternion(rotation)
        self.assertAlmostEqual(float(np.linalg.norm(quaternion)), 1.0)


if __name__ == "__main__":
    unittest.main()
