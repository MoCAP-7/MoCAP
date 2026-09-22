from __future__ import annotations

from dataclasses import dataclass, field
import inspect
import sys
import types
import unittest

import numpy as np


from yor_agent.robot.geometry import (
    matrix_to_quaternion_wxyz,
    quaternion_wxyz_to_matrix,
    rpy_to_quaternion_wxyz,
    slerp_quaternion_wxyz,
)
from yor_agent.robot.hardware import YorHardwareBridge
from yor_agent.robot.visualization.grasp import (
    _add_parallel_jaw_goal_icon,
    _candidate_status,
    _point_cloud_in_arm_frame,
    _selected_candidate,
)
from yor_agent.robot.manipulation import ManipulationController


@dataclass
class Pose:
    x_m: float = 1.0
    y_m: float = 2.0
    yaw_rad: float = 0.25
    valid: bool = True


@dataclass
class Frame:
    rgb: np.ndarray
    depth_m: np.ndarray
    timestamp_ns: int = 1234
    planar_pose: Pose = field(default_factory=Pose)
    ground_camera_height_m: float | None = None
    ground_down_camera_xyz: tuple[float, float, float] | None = None
    ground_plane_timestamp_ns: int | None = None


class FakeHardware:
    def __init__(self) -> None:
        self.frame = Frame(
            rgb=np.zeros((12, 16, 3), dtype=np.uint8),
            depth_m=np.ones((12, 16), dtype=np.float32),
            ground_camera_height_m=1.55,
            ground_down_camera_xyz=(0.0, 1.0, 0.0),
            ground_plane_timestamp_ns=1200,
        )
        self.closed = False

    def latest_frame(self, *, max_age_s):
        del max_age_s
        return self.frame

    def next_frame(self, timeout_s=2.0):
        del timeout_s
        return self.frame

    def get_base_status(self):
        return {
            "lease_active": False,
            "lease_remaining_s": 0.0,
            "last_velocity": [0.0, 0.0, 0.0],
            "estop_latched": False,
            "telemetry": {"lift_height_m": 0.42},
            "limits": {
                "lease_s": 0.25,
                "max_linear_mps": 0.18,
                "max_yaw_rad_s": 0.35,
            },
        }

    def submit_base_velocity(self, velocity):
        return {"accepted": True, "velocity": velocity}

    def close(self):
        self.closed = True


class FakeRPC:
    def __init__(self) -> None:
        self.calls = []
        self.closed = False

    def submit_velocity(self, velocity, sequence):
        self.calls.append((velocity, sequence))
        return {"accepted": True}

    def get_status(self):
        return {"lease_active": False}

    def close(self):
        self.closed = True


class FakeArmRPC(FakeRPC):
    def __init__(self) -> None:
        super().__init__()
        self.estopped = False

    def emergency_stop(self):
        self.estopped = True
        return {"stopped": True, "estop_latched": True}


class FakeZed:
    def __init__(self) -> None:
        self.closed = False

    def latest_frame(self, *, max_age_s):
        return max_age_s

    def next_frame(self, timeout_s=2.0):
        return timeout_s

    def close(self):
        self.closed = True


class RobotContractTest(unittest.TestCase):
    def test_hardware_bridge_adds_monotonic_sequence_and_final_zero(self) -> None:
        rpc = FakeRPC()
        zed = FakeZed()
        bridge = YorHardwareBridge(rpc_client=rpc, zed_source=zed)

        bridge.submit_base_velocity([0.1, 0.0, 0.0])
        bridge.submit_base_velocity([0.0, 0.0, 0.2])
        bridge.close()

        sequences = [call[1] for call in rpc.calls]
        self.assertEqual(sequences, sorted(sequences))
        self.assertEqual(len(sequences), len(set(sequences)))
        self.assertEqual(rpc.calls[-1][0], [0.0, 0.0, 0.0])
        self.assertTrue(rpc.closed)
        self.assertTrue(zed.closed)

    def test_hardware_bridge_close_does_not_latch_arm_stop(self) -> None:
        rpc = FakeRPC()
        arm_rpc = FakeArmRPC()
        bridge = YorHardwareBridge(
            rpc_client=rpc,
            zed_source=FakeZed(),
            arm_rpc_client=arm_rpc,
            require_arms=True,
        )

        bridge.close()
        bridge.close()

        self.assertFalse(arm_rpc.estopped)
        self.assertTrue(arm_rpc.closed)

    def test_quaternion_matrix_round_trip(self) -> None:
        quaternion = np.asarray([0.8, -0.1, 0.3, 0.5], dtype=float)
        quaternion /= np.linalg.norm(quaternion)
        recovered = matrix_to_quaternion_wxyz(
            quaternion_wxyz_to_matrix(quaternion)
        )
        self.assertAlmostEqual(abs(float(np.dot(quaternion, recovered))), 1.0, places=7)

    def test_quaternion_slerp_endpoints_and_midpoint(self) -> None:
        start = rpy_to_quaternion_wxyz([0.0, 0.0, 0.0])
        end = rpy_to_quaternion_wxyz([0.0, 0.0, np.pi / 2])

        np.testing.assert_allclose(slerp_quaternion_wxyz(start, end, 0.0), start)
        np.testing.assert_allclose(slerp_quaternion_wxyz(start, end, 1.0), end)
        midpoint = slerp_quaternion_wxyz(start, end, 0.5)
        expected = rpy_to_quaternion_wxyz([0.0, 0.0, np.pi / 4])
        self.assertAlmostEqual(abs(float(np.dot(midpoint, expected))), 1.0, places=7)



class FakeManipulationEnv:
    def __init__(self) -> None:
        self.manipulation_config = {
            "sam3_score_threshold": 0.05,
            # Legacy CGN contract tests opt in explicitly now that the product
            # default is GraspGen-X.
            "grasp_backend": "contact_graspnet",
            "grasp_candidate_max_object_distance_m": 0.10,
            "grasp_candidate_distance_reference": "origin",
            # Most contract tests isolate candidate selection from frame
            # conversion; dedicated tests below cover the real CGN depth.
            "contact_graspnet_origin_to_tcp_m": 0.0,
            "contact_graspnet_to_nero_tcp_rpy_rad": [0.0, 0.0, 0.0],
            "grasp_ik_parallel_jaw_symmetry_enabled": False,
            "grasp_collision_check_enabled": True,
        }
        self.arm_calls = []
        self.gripper_calls = []
        self.plan_calls = []
        self.arm_poses = {"left": [0.0] * 6, "right": [0.0] * 6}

    def observe(self):
        return {
            "robot0_robotview": {
                "images": {
                    "rgb": np.zeros((40, 60, 3), dtype=np.uint8),
                    "depth": np.ones((40, 60, 1), dtype=np.float32) * 0.5,
                }
            }
        }

    def manipulation_calibration(self, arm):
        del arm
        return {
            "camera_intrinsics": np.asarray(
                [[100.0, 0.0, 30.0], [0.0, 100.0, 20.0], [0.0, 0.0, 1.0]]
            ),
            "camera_calibration_resolution": np.asarray([60, 40]),
            "arm_from_camera": np.eye(4),
        }

    def move_arm_pose(self, arm, pose, timeout_s):
        self.arm_calls.append((arm, pose, timeout_s))
        self.arm_poses[arm] = list(pose)
        return {"success": True, "reason": "target_reached"}

    def plan_arm_poses(self, arm, poses):
        self.plan_calls.append((arm, poses))
        return {
            "success": True,
            "plans": [
                {
                    "candidate_index": index,
                    "success": True,
                    "ik_converged": True,
                    "ik_position_error_m": 0.0001,
                    "ik_rotation_error_rad": 0.0001,
                }
                for index in range(len(poses))
            ],
        }

    def arm_status(self):
        return {
            "estop_latched": False,
            "left": {"tcp_pose_xyz_rpy": self.arm_poses["left"]},
            "right": {"tcp_pose_xyz_rpy": self.arm_poses["right"]},
        }

    def set_gripper(self, arm, opened, timeout_s, force_n):
        self.gripper_calls.append((arm, opened, timeout_s, force_n))
        return {"success": True, "reason": "confirmed"}


class _AlwaysSafeGraspPlanner:
    def check_grasps(self, tcp_poses, obstacle_points, **kwargs):
        del obstacle_points, kwargs
        return {
            "safe": [True] * len(tcp_poses),
            "minimum_clearance_m": [0.10] * len(tcp_poses),
        }


def _api(env: FakeManipulationEnv) -> ManipulationController:
    return ManipulationController(
        env,
        motion_planner_factory=lambda config: _AlwaysSafeGraspPlanner(),
    )


class ManipulationApiTest(unittest.TestCase):
    def test_grasp_viser_selects_only_final_candidate(self) -> None:
        candidates = [
            {"selected": False, "candidate_index": 4},
            {"selected": True, "candidate_index": 7},
            {"selected": False, "candidate_index": 9},
        ]

        rank, selected = _selected_candidate({"candidates": candidates})

        self.assertEqual(rank, 2)
        self.assertEqual(selected["candidate_index"], 7)
        with self.assertRaisesRegex(RuntimeError, "exactly one"):
            _selected_candidate({"candidates": candidates[:1]})

    def test_grasp_viser_goal_icon_uses_one_frame_and_parallel_jaw_glyph(
        self,
    ) -> None:
        class RecordingScene:
            def __init__(self) -> None:
                self.frames = []
                self.spheres = []
                self.boxes = []

            def add_frame(self, **kwargs):
                self.frames.append(kwargs)

            def add_icosphere(self, **kwargs):
                self.spheres.append(kwargs)

            def add_box(self, **kwargs):
                self.boxes.append(kwargs)

        scene = RecordingScene()
        transform = np.eye(4)
        transform[:3, 3] = [0.1, -0.2, 0.3]

        _add_parallel_jaw_goal_icon(
            scene,
            name="/left/grasp_goal",
            transform=transform,
            status_color=np.asarray([30, 220, 80], dtype=np.uint8),
        )

        self.assertEqual(len(scene.frames), 1)
        self.assertEqual(scene.frames[0]["name"], "/left/grasp_goal")
        np.testing.assert_allclose(
            scene.frames[0]["position"],
            [0.1, -0.2, 0.3],
        )
        self.assertEqual(len(scene.spheres), 1)
        self.assertEqual(len(scene.boxes), 5)
        box_names = {box["name"] for box in scene.boxes}
        self.assertIn(
            "/left/grasp_goal/parallel_jaw_icon/finger_positive_y",
            box_names,
        )
        self.assertIn(
            "/left/grasp_goal/parallel_jaw_icon/finger_negative_y",
            box_names,
        )

    def test_grasp_viser_candidate_status_reports_ik_quality(self) -> None:
        self.assertEqual(_candidate_status({"ik_plan": None}), "not_evaluated")
        self.assertEqual(
            _candidate_status({"ik_plan": {"success": False}}),
            "failed",
        )
        self.assertEqual(
            _candidate_status(
                {"ik_plan": {"success": True, "ik_converged": True}}
            ),
            "converged",
        )
        self.assertEqual(
            _candidate_status(
                {
                    "ik_plan": {
                        "success": True,
                        "ik_converged": False,
                        "ik_quality_acceptable": True,
                    }
                }
            ),
            "acceptable",
        )

    def test_parallel_jaw_symmetry_is_disabled_by_default(self) -> None:
        env = FakeManipulationEnv()
        env.manipulation_config.pop("grasp_ik_parallel_jaw_symmetry_enabled")
        api = _api(env)
        mask = np.zeros((40, 60), dtype=bool)
        mask[10:30, 20:40] = True
        api._segment = lambda rgb, text_prompt: [
            {"mask": mask, "score": 0.9, "box": [20, 10, 40, 30]}
        ]
        grasp = np.eye(4)
        grasp[:3, 3] = [0.0, 0.0, 0.5]
        api._plan_grasp = lambda *args, **kwargs: (
            np.asarray([grasp]),
            np.asarray([0.8]),
            np.asarray([[0.0, 0.0, 0.5]]),
        )

        api.sample_grasp_pose("cup", arm="left")

        self.assertEqual(len(env.plan_calls[0][1]), 1)
        candidate = api.last_grasp_debug["candidates"][0]
        self.assertFalse(candidate["symmetry_flipped"])
        self.assertEqual(len(candidate["ik_variants"]), 1)

    def test_grasp_viser_point_cloud_uses_arm_frame_and_mask(self) -> None:
        depth = np.asarray([[1.0, 2.0], [0.0, 1.0]], dtype=np.float64)
        rgb = np.asarray(
            [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]],
            dtype=np.uint8,
        )
        intrinsics = np.asarray(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        )
        arm_from_camera = np.eye(4)
        arm_from_camera[:3, 3] = [0.1, 0.2, 0.3]
        mask = np.asarray([[False, False], [False, True]])

        points, colors = _point_cloud_in_arm_frame(
            depth,
            rgb,
            intrinsics,
            arm_from_camera,
            mask=mask,
        )

        np.testing.assert_allclose(points, [[1.1, 1.2, 1.3]])
        np.testing.assert_array_equal(colors, [[10, 11, 12]])

    def test_get_object_pose_does_not_initialize_graspnet(self) -> None:
        env = FakeManipulationEnv()
        api = _api(env)
        mask = np.zeros((40, 60), dtype=bool)
        mask[10:30, 20:40] = True
        api._segment = lambda rgb, text_prompt: [
            {"mask": mask, "score": 0.9, "box": [20, 10, 40, 30]}
        ]

        position, quaternion, extent = api.get_object_pose(
            "marker", arm="left", return_bbox_extent=True
        )

        self.assertIsNone(api._plan_grasp)
        self.assertEqual(position.shape, (3,))
        self.assertEqual(quaternion.shape, (4,))
        self.assertEqual(extent.shape, (3,))

    def test_get_object_pose_optionally_returns_zed_metric_distance(self) -> None:
        env = FakeManipulationEnv()
        api = _api(env)
        mask = np.zeros((40, 60), dtype=bool)
        mask[10:30, 20:40] = True
        api._segment = lambda rgb, text_prompt: [
            {"mask": mask, "score": 0.9, "box": [20, 10, 40, 30]}
        ]

        position, quaternion, extent, distance_m = api.get_object_pose(
            "marker", arm="left", return_zed_distance=True
        )

        self.assertEqual(position.shape, (3,))
        self.assertEqual(quaternion.shape, (4,))
        self.assertIsNone(extent)
        self.assertAlmostEqual(
            distance_m,
            float(np.linalg.norm([-0.0025, -0.0025, 0.5])),
            places=7,
        )

    def test_sample_grasp_and_goto_use_same_arm_frame(self) -> None:
        env = FakeManipulationEnv()
        api = _api(env)
        mask = np.zeros((40, 60), dtype=bool)
        mask[10:30, 20:40] = True
        api._segment = lambda rgb, text_prompt: [
            {"mask": mask, "score": 0.9, "box": [20, 10, 40, 30]}
        ]
        grasp = np.eye(4)
        grasp[:3, 3] = [0.0, 0.0, 0.5]
        api._plan_grasp = lambda *args, **kwargs: (
            np.asarray([grasp]),
            np.asarray([0.8]),
            np.zeros((1, 3)),
        )

        position, quaternion = api.sample_grasp_pose("cup", arm="right")
        np.testing.assert_allclose(position, [0.0, 0.0, 0.5])
        result = api.goto_pose(position, quaternion, arm="right")

        self.assertTrue(result["success"])
        self.assertFalse(result["segmented"])
        self.assertEqual(len(env.arm_calls), 1)
        self.assertEqual(env.arm_calls[-1][0], "right")
        np.testing.assert_allclose(env.arm_calls[-1][1][:3], position)

    def test_sample_grasp_rejects_off_object_high_score(self) -> None:
        env = FakeManipulationEnv()
        api = _api(env)
        mask = np.zeros((40, 60), dtype=bool)
        mask[10:30, 20:40] = True
        api._segment = lambda rgb, text_prompt: [
            {"mask": mask, "score": 0.9, "box": [20, 10, 40, 30]}
        ]
        valid_grasp = np.eye(4)
        valid_grasp[:3, 3] = [0.0, 0.0, 0.5]
        off_object_grasp = np.eye(4)
        off_object_grasp[:3, 3] = [0.0, 0.0, 0.8]
        api._plan_grasp = lambda *args, **kwargs: (
            np.asarray([valid_grasp, off_object_grasp]),
            np.asarray([0.7, 0.99]),
            np.zeros((2, 3)),
        )

        position, _ = api.sample_grasp_pose("cup", arm="left")

        np.testing.assert_allclose(position, [0.0, 0.0, 0.5])

    def test_sample_grasp_prefers_ik_converged_candidate(self) -> None:
        env = FakeManipulationEnv()
        api = _api(env)
        mask = np.zeros((40, 60), dtype=bool)
        mask[10:30, 20:40] = True
        api._segment = lambda rgb, text_prompt: [
            {"mask": mask, "score": 0.9, "box": [20, 10, 40, 30]}
        ]
        high_score = np.eye(4)
        high_score[:3, 3] = [0.0, 0.0, 0.50]
        reachable = np.eye(4)
        reachable[:3, 3] = [0.01, 0.0, 0.50]
        api._plan_grasp = lambda *args, **kwargs: (
            np.asarray([high_score, reachable]),
            np.asarray([0.95, 0.80]),
            np.zeros((2, 3)),
        )

        def plan_arm_poses(arm, poses):
            env.plan_calls.append((arm, poses))
            return {
                "success": True,
                "plans": [
                    {
                        "candidate_index": 0,
                        "success": True,
                        "ik_converged": False,
                        "ik_position_error_m": 0.02,
                        "ik_rotation_error_rad": 0.1,
                    },
                    {
                        "candidate_index": 1,
                        "success": True,
                        "ik_converged": True,
                        "ik_position_error_m": 0.0001,
                        "ik_rotation_error_rad": 0.0001,
                    },
                ],
            }

        env.plan_arm_poses = plan_arm_poses

        position, _ = api.sample_grasp_pose("cup", arm="left")

        np.testing.assert_allclose(position, reachable[:3, 3])
        self.assertEqual(len(env.plan_calls), 1)
        cached = api._sampled_grasp_goalsets["left"]["tcp_poses"]
        self.assertEqual(len(cached), 1)
        np.testing.assert_allclose(cached[0][:3, 3], reachable[:3, 3])

    def test_sample_grasp_cannot_disable_ik_precheck(self) -> None:
        env = FakeManipulationEnv()
        env.manipulation_config["grasp_ik_precheck_enabled"] = False
        api = _api(env)
        mask = np.zeros((40, 60), dtype=bool)
        mask[10:30, 20:40] = True
        api._segment = lambda rgb, text_prompt: [
            {"mask": mask, "score": 0.9, "box": [20, 10, 40, 30]}
        ]
        lower = np.eye(4)
        lower[:3, 3] = [0.01, 0.0, 0.50]
        higher = np.eye(4)
        higher[:3, 3] = [0.0, 0.0, 0.50]
        api._plan_grasp = lambda *args, **kwargs: (
            np.asarray([lower, higher]),
            np.asarray([0.8, 0.95]),
            np.zeros((2, 3)),
        )

        with self.assertRaisesRegex(RuntimeError, "requires grasp_ik_precheck_enabled"):
            api.sample_grasp_pose("cup", arm="left")

    def test_sample_grasp_cannot_disable_collision_check(self) -> None:
        env = FakeManipulationEnv()
        env.manipulation_config["grasp_collision_check_enabled"] = False
        api = _api(env)

        with self.assertRaisesRegex(
            RuntimeError, "requires grasp_collision_check_enabled"
        ):
            api.sample_grasp_pose("cup", arm="left")

    def test_sample_grasp_maps_cgn_origin_to_tcp_jaw_center(self) -> None:
        env = FakeManipulationEnv()
        env.manipulation_config["contact_graspnet_origin_to_tcp_m"] = 0.1034
        api = _api(env)
        mask = np.zeros((40, 60), dtype=bool)
        mask[10:30, 20:40] = True
        api._segment = lambda rgb, text_prompt: [
            {"mask": mask, "score": 0.9, "box": [20, 10, 40, 30]}
        ]
        grasp = np.eye(4)
        grasp[:3, 3] = [0.0, 0.0, 0.5]
        api._plan_grasp = lambda *args, **kwargs: (
            np.asarray([grasp]),
            np.asarray([0.8]),
            np.asarray([[0.0, 0.0, 0.5]]),
        )

        position, _ = api.sample_grasp_pose("cup", arm="left")

        np.testing.assert_allclose(position, [0.0, 0.0, 0.6034])
        candidate = api.last_grasp_debug["candidates"][0]
        np.testing.assert_allclose(
            candidate["arm_from_cgn_origin"][:3, 3], [0.0, 0.0, 0.5]
        )
        np.testing.assert_allclose(
            candidate["arm_from_grasp"][:3, 3], [0.0, 0.0, 0.6034]
        )

    def test_sample_grasp_maps_graspgenx_origin_to_nero_tcp(self) -> None:
        env = FakeManipulationEnv()
        env.manipulation_config.update(
            {
                "grasp_backend": "graspgenx",
                "graspgenx_origin_to_tcp_m": 0.13,
                "graspgenx_to_nero_tcp_rpy_rad": [0.0, 0.0, 0.0],
            }
        )
        api = _api(env)
        mask = np.zeros((40, 60), dtype=bool)
        mask[10:30, 20:40] = True
        api._segment = lambda rgb, text_prompt: [
            {"mask": mask, "score": 0.9, "box": [20, 10, 40, 30]}
        ]
        grasp = np.eye(4)
        grasp[:3, 3] = [0.0, 0.0, 0.5]

        class FakeGraspGenX:
            name = "graspgenx"

            def __call__(self, *args, **kwargs):
                del args, kwargs
                return types.SimpleNamespace(
                    poses=np.asarray([grasp]),
                    scores=np.asarray([0.8]),
                    anchor_points=np.asarray([[0.0, 0.0, 0.5]]),
                    metadata={"planner": "diffusion"},
                )

        api._plan_grasp = FakeGraspGenX()

        position, _ = api.sample_grasp_pose("cup", arm="left")

        np.testing.assert_allclose(position, [0.0, 0.0, 0.63])
        self.assertEqual(api.last_grasp_debug["grasp_backend"], "graspgenx")
        self.assertEqual(
            api.last_grasp_debug["backend_metadata"],
            {"planner": "diffusion"},
        )
        self.assertAlmostEqual(
            api.last_grasp_debug["grasp_model_origin_to_tcp_m"], 0.13
        )
        candidate = api.last_grasp_debug["candidates"][0]
        np.testing.assert_allclose(candidate["arm_from_model_origin"], grasp)
        np.testing.assert_allclose(
            candidate["arm_from_canonical_endpoint"][:3, 3],
            [0.0, 0.0, 0.63],
        )
        np.testing.assert_allclose(
            candidate["arm_from_symmetric_endpoint"][:3, :3],
            np.diag([-1.0, -1.0, 1.0]),
            atol=1e-12,
        )

    def test_graspgenx_default_axes_use_official_piper_canonical_mapping(
        self,
    ) -> None:
        env = FakeManipulationEnv()
        api = _api(env)

        model_from_tcp, endpoint, _ = api._grasp_model_to_controlled_endpoint(
            "left", backend_name="graspgenx"
        )

        self.assertEqual(endpoint, "tcp")
        # The official +90-degree mapping makes Nero TCP +Y coincide with the
        # unsigned GraspGen-X closing axis in its -X direction.
        np.testing.assert_allclose(
            model_from_tcp[:3, 1], [-1.0, 0.0, 0.0], atol=1e-12
        )
        # Both frames retain +Z as gripper-forward/approach.
        np.testing.assert_allclose(
            model_from_tcp[:3, 2], [0.0, 0.0, 1.0], atol=1e-12
        )

    def test_cgn_closing_and_approach_axes_align_with_nero_tcp(self) -> None:
        env = FakeManipulationEnv()
        env.manipulation_config["contact_graspnet_to_nero_tcp_rpy_rad"] = [
            0.0,
            0.0,
            -np.pi / 2.0,
        ]
        api = _api(env)

        cgn_from_tcp, endpoint, _ = api._contact_graspnet_to_controlled_endpoint(
            "left"
        )

        self.assertEqual(endpoint, "tcp")
        # Nero TCP +Y is the official gripper closing axis and must equal CGN +X.
        np.testing.assert_allclose(cgn_from_tcp[:3, 1], [1.0, 0.0, 0.0], atol=1e-12)
        # Both conventions use local +Z as gripper-forward/approach.
        np.testing.assert_allclose(cgn_from_tcp[:3, 2], [0.0, 0.0, 1.0], atol=1e-12)

    def test_sample_grasp_converts_same_tcp_goal_to_flange_endpoint(self) -> None:
        env = FakeManipulationEnv()
        env.manipulation_config["contact_graspnet_origin_to_tcp_m"] = 0.1034
        env.arm_status = lambda: {
            "end_effector_frame": "flange",
            "tcp_offsets_xyz_rpy": {
                "left": [0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
                "right": [0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
            },
            "left": {"tcp_pose_xyz_rpy": [0.0] * 6},
            "right": {"tcp_pose_xyz_rpy": [0.0] * 6},
        }
        api = _api(env)
        mask = np.zeros((40, 60), dtype=bool)
        mask[10:30, 20:40] = True
        api._segment = lambda rgb, text_prompt: [
            {"mask": mask, "score": 0.9, "box": [20, 10, 40, 30]}
        ]
        grasp = np.eye(4)
        grasp[:3, 3] = [0.0, 0.0, 0.5]
        api._plan_grasp = lambda *args, **kwargs: (
            np.asarray([grasp]),
            np.asarray([0.8]),
            np.asarray([[0.0, 0.0, 0.5]]),
        )

        position, _ = api.sample_grasp_pose("cup", arm="left")

        np.testing.assert_allclose(position, [-0.1, 0.0, 0.6034])
        self.assertEqual(api.last_grasp_debug["controlled_endpoint"], "flange")

    def test_sample_grasp_excludes_nonconverged_low_travel_symmetry(self) -> None:
        env = FakeManipulationEnv()
        env.manipulation_config["grasp_ik_parallel_jaw_symmetry_enabled"] = True
        api = _api(env)
        mask = np.zeros((40, 60), dtype=bool)
        mask[10:30, 20:40] = True
        api._segment = lambda rgb, text_prompt: [
            {"mask": mask, "score": 0.9, "box": [20, 10, 40, 30]}
        ]
        grasp = np.eye(4)
        grasp[:3, 3] = [0.0, 0.0, 0.5]
        api._plan_grasp = lambda *args, **kwargs: (
            np.asarray([grasp]),
            np.asarray([0.8]),
            np.asarray([[0.0, 0.0, 0.5]]),
        )

        def plan_arm_poses(arm, poses):
            env.plan_calls.append((arm, poses))
            return {
                "success": True,
                "plans": [
                    {
                        "candidate_index": 0,
                        "success": True,
                        "ik_converged": True,
                        "ik_position_error_m": 0.0001,
                        "ik_rotation_error_rad": 0.0001,
                        "joint_travel_l2_rad": 5.0,
                    },
                    {
                        "candidate_index": 1,
                        "success": True,
                        # Lower travel is irrelevant when this branch did not
                        # strictly converge.
                        "ik_converged": False,
                        "ik_position_error_m": 0.002,
                        "ik_rotation_error_rad": 0.02,
                        "joint_travel_l2_rad": 1.0,
                    },
                ],
            }

        env.plan_arm_poses = plan_arm_poses

        _, quaternion = api.sample_grasp_pose("cup", arm="left")

        self.assertEqual(len(env.plan_calls[0][1]), 2)
        self.assertFalse(
            api.last_grasp_debug["candidates"][0]["symmetry_flipped"]
        )
        self.assertEqual(
            len(api.last_grasp_debug["candidates"][0]["ik_variants"]), 2
        )
        candidate = api.last_grasp_debug["candidates"][0]
        # The raw backend prediction remains unmodified and the only strict
        # branch is the canonical endpoint.
        np.testing.assert_allclose(candidate["arm_from_model_origin"], grasp)
        np.testing.assert_allclose(
            candidate["arm_from_selected_model_origin"][:3, :3],
            np.eye(3),
            atol=1e-12,
        )
        np.testing.assert_allclose(
            candidate["arm_from_canonical_endpoint"][:3, :3],
            np.eye(3),
            atol=1e-12,
        )
        np.testing.assert_allclose(
            candidate["arm_from_symmetric_endpoint"][:3, :3],
            np.diag([-1.0, -1.0, 1.0]),
            atol=1e-12,
        )
        np.testing.assert_allclose(np.abs(quaternion), [1.0, 0.0, 0.0, 0.0])

    def test_sample_grasp_rejects_all_finite_nonconverged_candidates(self) -> None:
        env = FakeManipulationEnv()
        api = _api(env)
        mask = np.zeros((40, 60), dtype=bool)
        mask[10:30, 20:40] = True
        api._segment = lambda rgb, text_prompt: [
            {"mask": mask, "score": 0.9, "box": [20, 10, 40, 30]}
        ]
        high_score = np.eye(4)
        high_score[:3, 3] = [0.0, 0.0, 0.50]
        lower_residual = np.eye(4)
        lower_residual[:3, 3] = [0.01, 0.0, 0.50]
        api._plan_grasp = lambda *args, **kwargs: (
            np.asarray([high_score, lower_residual]),
            np.asarray([0.95, 0.80]),
            np.zeros((2, 3)),
        )

        def plan_arm_poses(arm, poses):
            env.plan_calls.append((arm, poses))
            return {
                "success": True,
                "plans": [
                    {
                        "candidate_index": 0,
                        "success": True,
                        "ik_converged": False,
                        "ik_position_error_m": 0.03,
                        "ik_rotation_error_rad": 0.01,
                    },
                    {
                        "candidate_index": 1,
                        "success": True,
                        "ik_converged": False,
                        "ik_position_error_m": 0.022,
                        "ik_rotation_error_rad": 0.08,
                    },
                ],
            }

        env.plan_arm_poses = plan_arm_poses

        with self.assertRaisesRegex(
            RuntimeError,
            "did not converge for any collision-safe grasp variant",
        ):
            api.sample_grasp_pose("cup", arm="left")

        self.assertNotIn("left", api._sampled_grasp_goalsets)

    def test_bimanual_gripper_selection(self) -> None:
        env = FakeManipulationEnv()
        api = _api(env)

        self.assertTrue(api.open_gripper(arm=0)["success"])
        self.assertTrue(api.close_gripper(arm=1, force_n=1.5)["success"])
        self.assertEqual(env.gripper_calls[0][0:2], ("left", True))
        self.assertEqual(env.gripper_calls[1][0:2], ("right", False))


if __name__ == "__main__":
    unittest.main()
