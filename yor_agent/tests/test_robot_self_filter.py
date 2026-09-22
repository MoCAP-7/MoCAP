from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from fakes import FakeHardware, make_environment
from yor_agent.robot.self_filter import (
    RobotDepthSelfFilter,
    RobotSelfFilterConfig,
)
from yor_agent.robot.manipulation import ManipulationController


_URDF = """<?xml version="1.0"?>
<robot name="test">
  <link name="base"/>
  <link name="grasp_tcp"/>
  <joint name="tcp_fixed" type="fixed">
    <parent link="base"/><child link="grasp_tcp"/>
    <origin xyz="0 0 1" rpy="0 0 0"/>
  </joint>
</robot>
"""

_SPHERES = """robot_cfg:
  kinematics:
    lock_joints: {}
    collision_spheres:
      base:
        - center: [0.0, 0.0, 1.0]
          radius: 0.1
"""


_STATUS = {
    "left": {
        "joint_pos": [0.0] * 7,
        "gripper": {"available": True, "width_m": 0.08},
    }
}
_INTRINSICS = np.asarray([[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]])


class RobotSelfFilterTest(unittest.TestCase):
    def _filter(
        self, directory: Path, *, mask_padding_m: float = 0.0
    ) -> RobotDepthSelfFilter:
        urdf = directory / "robot.urdf"
        spheres = directory / "robot.yml"
        urdf.write_text(_URDF, encoding="utf-8")
        spheres.write_text(_SPHERES, encoding="utf-8")
        config = RobotSelfFilterConfig(
            urdf_path=urdf,
            spheres_path=spheres,
            sphere_erosion_m=0.0,
            depth_tolerance_m=0.01,
            require_both_arms=False,
            mask_padding_m=mask_padding_m,
        )
        config.validate()
        return RobotDepthSelfFilter(
            config, arm_from_camera={"left": np.eye(4)}
        )

    def test_masks_depth_the_robot_occludes_however_far_off_it_reads(self) -> None:
        # The sphere spans depth 0.9-1.1 on the central ray. At the left image
        # border the ZED reports a gripper about 0.10 m too far; anything the
        # camera reports behind the robot's nearest surface is the robot.
        with tempfile.TemporaryDirectory() as temporary:
            self_filter = self._filter(Path(temporary))
            self_filter.update(_STATUS, depth_shape=(100, 100), intrinsics=_INTRINSICS)
            depth = np.full((100, 100), 2.0, dtype=np.float32)
            depth[50, 50] = 1.25

            mask = self_filter.mask(depth)

            self.assertTrue(mask[50, 50])
            # Off the silhouette the background stays: the sphere subtends
            # about 11 px, so column 80 is clear of it.
            self.assertFalse(mask[50, 80])

    def test_mask_padding_grows_the_mask_but_not_the_footprint_spheres(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            plain = self._filter(Path(temporary))
            padded = self._filter(Path(temporary), mask_padding_m=0.03)
            for self_filter in (plain, padded):
                self_filter.update(_STATUS, depth_shape=(100, 100), intrinsics=_INTRINSICS)
            depth = np.full((100, 100), 2.0, dtype=np.float32)

            # Column 62 looks 0.119 m past the sphere centre: outside the
            # 0.10 m sphere, inside it grown by 0.03 m.
            self.assertFalse(plain.mask(depth)[50, 62])
            self.assertTrue(padded.mask(depth)[50, 62])
            np.testing.assert_allclose(
                padded.camera_spheres(), plain.camera_spheres()
            )

    def test_mask_padding_is_bounded(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "mask padding"):
                self._filter(Path(temporary), mask_padding_m=0.06)

    def test_masks_predicted_robot_depth_but_preserves_nearer_obstacle(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            self_filter = self._filter(Path(temporary))
            status = {
                "left": {
                    "joint_pos": [0.0] * 7,
                    "gripper": {"available": True, "width_m": 0.08},
                }
            }
            intrinsics = np.asarray(
                [[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]]
            )
            self.assertTrue(
                self_filter.update(
                    status, depth_shape=(100, 100), intrinsics=intrinsics
                )
            )
            self.assertFalse(
                self_filter.update(
                    status, depth_shape=(100, 100), intrinsics=intrinsics
                )
            )
            depth = np.full((100, 100), 2.0, dtype=np.float32)
            depth[50, 50] = 0.90
            depth[50, 51] = 0.50

            mask = self_filter.mask(depth)

            self.assertTrue(mask[50, 50])
            self.assertFalse(mask[50, 51])

    def test_environment_attachment_registry_is_detached(self) -> None:
        environment = make_environment(FakeHardware())
        environment.set_robot_self_filter_attached_object(
            "right", [[-0.01, -0.02, 0.0], [0.03, 0.02, 0.08]]
        )
        snapshot = environment.robot_self_filter_attached_objects()
        snapshot["right"]["bounds_local"][0][0] = 99.0

        fresh = environment.robot_self_filter_attached_objects()
        self.assertEqual(fresh["right"]["bounds_local"][0][0], -0.01)
        environment.clear_robot_self_filter_attached_object("right")
        self.assertEqual(environment.robot_self_filter_attached_objects(), {})

    def test_simulated_gripper_close_and_open_update_filter_attachment(self) -> None:
        environment = make_environment(FakeHardware())
        controller = ManipulationController(environment)
        controller._pending_self_filter_grasps["right"] = {
            "attached_bounds_local": np.asarray(
                [[-0.01, -0.02, 0.0], [0.03, 0.02, 0.08]]
            ),
            "gripper_closed": False,
        }

        self.assertTrue(controller.close_gripper("right", simulated=True)["success"])
        self.assertIn(
            "right", environment.robot_self_filter_attached_objects()
        )
        self.assertTrue(controller.open_gripper("right", simulated=True)["success"])
        self.assertEqual(environment.robot_self_filter_attached_objects(), {})


if __name__ == "__main__":
    unittest.main()
