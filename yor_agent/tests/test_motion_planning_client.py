from __future__ import annotations

import math
import unittest

import numpy as np

from yor_agent.robot.motion_planning_client import GraspMotionPlanningClient


NERO_JOINT_LIMIT_BOUNDARIES = np.asarray(
    [2.70526, -1.74, 2.75, 2.14, -2.75, -0.73, math.pi / 2],
    dtype=np.float64,
)


class RecordingMotionPlanningClient(GraspMotionPlanningClient):
    def _request(self, payload):
        return payload


class MotionPlanningClientPrecisionTests(unittest.TestCase):
    def setUp(self):
        self.client = RecordingMotionPlanningClient({})
        self.current_joints = np.zeros(7, dtype=np.float64)
        self.tcp_pose = np.eye(4, dtype=np.float64)
        self.obstacle_points = np.zeros((10, 3), dtype=np.float32)

    def test_plan_grasp_preserves_joint_seed_limit_precision(self):
        request = self.client.plan_grasp(
            self.current_joints,
            self.tcp_pose,
            self.obstacle_points,
            clearance_m=0.008,
            approach_m=0.10,
            approach_samples=8,
            scene_voxel_m=0.012,
            goal_joint_seed=NERO_JOINT_LIMIT_BOUNDARIES,
        )

        self.assertEqual(request["current_joints"].dtype, np.dtype(np.float64))
        self.assertEqual(request["tcp_poses"].dtype, np.dtype(np.float32))
        self.assertEqual(request["goal_joint_seed"].dtype, np.dtype(np.float64))
        np.testing.assert_array_equal(
            request["goal_joint_seed"], NERO_JOINT_LIMIT_BOUNDARIES
        )

    def test_plan_joint_target_preserves_joint_limit_precision(self):
        request = self.client.plan_joint_target(
            self.current_joints,
            NERO_JOINT_LIMIT_BOUNDARIES,
            self.obstacle_points,
            scene_voxel_m=0.012,
        )

        self.assertEqual(request["current_joints"].dtype, np.dtype(np.float64))
        self.assertEqual(request["target_joints"].dtype, np.dtype(np.float64))
        np.testing.assert_array_equal(
            request["target_joints"], NERO_JOINT_LIMIT_BOUNDARIES
        )

    def test_attached_object_payloads_are_forwarded(self):
        bounds = np.asarray([[-0.02, -0.03, -0.04], [0.02, 0.03, 0.04]])
        lift = self.client.plan_attached_lift(
            self.current_joints,
            self.obstacle_points,
            bounds,
            lift_m=0.15,
            scene_voxel_m=0.012,
        )
        self.assertEqual(lift["action"], "plan_attached_lift")
        self.assertEqual(lift["attached_bounds_local"].dtype, np.dtype(np.float32))
        np.testing.assert_allclose(lift["attached_bounds_local"], bounds)

        home = self.client.plan_joint_target(
            self.current_joints,
            NERO_JOINT_LIMIT_BOUNDARIES,
            self.obstacle_points,
            scene_voxel_m=0.012,
            attached_bounds_local=bounds,
        )
        self.assertEqual(home["action"], "plan_joint_target")
        np.testing.assert_allclose(home["attached_bounds_local"], bounds)

    def test_planner_attempts_are_forwarded_only_when_set(self):
        common = dict(
            clearance_m=0.008,
            approach_m=0.10,
            approach_samples=8,
            scene_voxel_m=0.012,
        )
        default = self.client.plan_grasp(
            self.current_joints, self.tcp_pose, self.obstacle_points, **common
        )
        self.assertNotIn("max_attempts", default)
        self.assertNotIn("enable_graph_attempt", default)

        bounded = self.client.plan_grasp(
            self.current_joints,
            self.tcp_pose,
            self.obstacle_points,
            planner_attempts={
                "max_attempts": 3,
                "enable_graph_attempt": 1,
                "finetune_attempts": 0,
            },
            **common,
        )
        self.assertEqual(bounded["max_attempts"], 3)
        self.assertEqual(bounded["enable_graph_attempt"], 1)
        self.assertEqual(bounded["finetune_attempts"], 0)

        partial = self.client.plan_joint_target(
            self.current_joints,
            NERO_JOINT_LIMIT_BOUNDARIES,
            self.obstacle_points,
            scene_voxel_m=0.012,
            planner_attempts={"max_attempts": 2, "unknown_key": 9},
        )
        self.assertEqual(partial["max_attempts"], 2)
        self.assertNotIn("enable_graph_attempt", partial)
        self.assertNotIn("unknown_key", partial)

        lift = self.client.plan_attached_lift(
            self.current_joints,
            self.obstacle_points,
            np.asarray([[-0.02, -0.03, -0.04], [0.02, 0.03, 0.04]]),
            lift_m=0.15,
            scene_voxel_m=0.012,
            planner_attempts={"enable_graph_attempt": 0},
        )
        self.assertEqual(lift["enable_graph_attempt"], 0)
        self.assertNotIn("max_attempts", lift)

        scene = self.client.plan_joint_target(
            self.current_joints,
            NERO_JOINT_LIMIT_BOUNDARIES,
            self.obstacle_points,
            scene_voxel_m=0.012,
            planner_attempts={
                "scene_mesh": "greedy",
                "scene_coarse_voxel_m": 0.03,
                "scene_fine_radius_m": 0.3,
                "scene_crop_radius_m": 1.6,
            },
        )
        self.assertEqual(scene["scene_mesh"], "greedy")
        self.assertEqual(scene["scene_coarse_voxel_m"], 0.03)
        self.assertEqual(scene["scene_fine_radius_m"], 0.3)
        self.assertEqual(scene["scene_crop_radius_m"], 1.6)


if __name__ == "__main__":
    unittest.main()
