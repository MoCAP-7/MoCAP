from __future__ import annotations

import inspect
import unittest
from unittest import mock

import numpy as np
from fakes import FAKE_MANIPULATION_CONFIG, FakeHardware, make_environment

from yor_agent.exceptions import PrimitiveFailed
from yor_agent.primitives.manipulation import register_manipulation_primitives
from yor_agent.primitives.registry import PrimitiveRegistry
from yor_agent.robot import manipulation as manipulation_module
from yor_agent.robot.geometry import quaternion_wxyz_to_matrix
from yor_agent.robot.manipulation import (
    NERO_HOME_JOINTS_RAD,
    NERO_MIRROR_SIGNS,
    ManipulationController,
)

ARMS = {
    "left": {"joint_positions": [0.0] * 6, "joint_pos": [0.0] * 7},
    "right": {"joint_positions": [0.0] * 6, "joint_pos": [0.0] * 7},
    "estop_latched": False,
    "end_effector_frame": "tcp",
}

# A small square mask (8x8 = 64 px, above the >=30 px floor) roughly centered
# in the 48x64 fake camera frame, all at a uniform 1.0 m depth.
_MASK = np.zeros((48, 64), dtype=bool)
_MASK[20:28, 28:36] = True

_GRASP_POSE = np.eye(4)
# Keep the controlled TCP below the Pi's 1.0 m arm-base protocol bound after
# the default 0.105 m GraspGen-X origin-to-TCP offset is applied.
_GRASP_POSE[:3, 3] = [0.0, 0.0, 0.8]
_GRASP_POSES = _GRASP_POSE[None]
_GRASP_SCORES = np.asarray([0.9])
# A masked pixel at (row=24, col=32) back-projects to exactly (0, 0, 1.0)
# given the fake intrinsics below, so the object-distance check passes.
_GRASP_CONTACT_POINTS = np.asarray([[0.0, 0.0, 1.0]])


def _fake_segment(rgb, text_prompt):
    del rgb, text_prompt
    return [{"score": 0.9, "mask": _MASK}]


def _fake_grasp_backend(depth, intrinsics, segmentation, instance_id):
    del depth, intrinsics, segmentation, instance_id
    return _GRASP_POSES, _GRASP_SCORES, _GRASP_CONTACT_POINTS


def _fake_goalset_grasp_backend(depth, intrinsics, segmentation, instance_id):
    del depth, intrinsics, segmentation, instance_id
    second_pose = _GRASP_POSE.copy()
    second_pose[0, 3] = 0.01
    return (
        np.stack([_GRASP_POSE, second_pose]),
        np.asarray([0.9, 0.8]),
        np.asarray([[0.0, 0.0, 1.0], [0.01, 0.0, 1.0]]),
    )


class FakeMotionPlanner:
    def __init__(self):
        self.plan_grasp_requests = []
        self.attached_lift_requests = []
        self.joint_target_requests = []

    def check_grasps(self, tcp_poses, obstacle_points, **kwargs):
        del obstacle_points, kwargs
        return {
            "success": True,
            "safe": [True] * len(tcp_poses),
            "minimum_clearance_m": [0.2] * len(tcp_poses),
        }

    def plan_grasp(self, current_joints, tcp_poses, obstacle_points, **kwargs):
        tcp_poses = np.asarray(tcp_poses, dtype=float)
        if tcp_poses.ndim == 2:
            tcp_poses = tcp_poses[None]
        self.plan_grasp_requests.append(
            {**dict(kwargs), "tcp_poses": tcp_poses.copy()}
        )
        del obstacle_points
        current = np.asarray(current_joints, dtype=float)
        pregrasp = current + 0.005
        grasp = current + 0.01
        return {
            "success": True,
            "reason": "collision_free_path_planned",
            "waypoints": [
                current.tolist(),
                grasp.tolist(),
            ],
            "segments": [
                {
                    "phase": "approach",
                    "waypoints": [current.tolist(), pregrasp.tolist()],
                },
                {"phase": "grasp", "waypoints": [grasp.tolist()]},
            ],
            "planning_time_s": 0.01,
            "planning_scene_points": 100,
            "removed_robot_points": 0,
            "trajectory_dt_s": 0.1,
            "terminal_check": {"safe": [True]},
            "selected_goalset_index": 0,
            "selected_tcp_pose": tcp_poses[0].tolist(),
        }

    def plan_attached_lift(
        self,
        current_joints,
        obstacle_points,
        attached_bounds_local,
        **kwargs,
    ):
        current = np.asarray(current_joints, dtype=float)
        target = current + 0.01
        self.attached_lift_requests.append(
            {
                "current_joints": current.copy(),
                "obstacle_points": np.asarray(obstacle_points).copy(),
                "attached_bounds_local": np.asarray(
                    attached_bounds_local
                ).copy(),
                "kwargs": dict(kwargs),
            }
        )
        return {
            "success": True,
            "reason": "collision_free_attached_lift_planned",
            "waypoints": [current.tolist(), target.tolist()],
            "planning_time_s": 0.01,
            "planning_scene_points": 100,
            "removed_robot_points": 0,
            "trajectory_dt_s": 0.1,
            "attached_object_sphere_count": 32,
        }

    def plan_joint_target(
        self, current_joints, target_joints, obstacle_points, **kwargs
    ):
        current = np.asarray(current_joints, dtype=float)
        target = np.asarray(target_joints, dtype=float)
        self.joint_target_requests.append(
            {
                "current_joints": current.copy(),
                "target_joints": target.copy(),
                "obstacle_points": np.asarray(obstacle_points).copy(),
                "kwargs": dict(kwargs),
            }
        )
        interval_count = max(
            1,
            int(np.ceil(np.max(np.abs(target - current)) / 0.05)),
        )
        waypoints = np.linspace(current, target, interval_count + 1)
        return {
            "success": True,
            "reason": "collision_free_joint_target_path_planned",
            "waypoints": waypoints.tolist(),
            "planning_time_s": 0.01,
            "planning_scene_points": 100,
            "removed_robot_points": 0,
            "trajectory_dt_s": 0.1,
            "maximum_step_rad": 0.05,
            "goal_error_rad": 0.0,
        }


class UnsafeMotionPlanner(FakeMotionPlanner):
    def check_grasps(self, tcp_poses, obstacle_points, **kwargs):
        del obstacle_points, kwargs
        return {
            "success": True,
            "safe": [False] * len(tcp_poses),
            "minimum_clearance_m": [-0.001] * len(tcp_poses),
        }


class LongMotionPlanner(FakeMotionPlanner):
    def plan_grasp(self, current_joints, tcp_poses, obstacle_points, **kwargs):
        del tcp_poses, obstacle_points, kwargs
        start = np.asarray(current_joints, dtype=float)
        waypoints = np.tile(start, (81, 1))
        waypoints[:, 0] += np.linspace(0.0, 0.8, len(waypoints))
        return {
            "success": True,
            "reason": "collision_free_path_planned",
            "waypoints": waypoints.tolist(),
            "planning_time_s": 0.01,
            "planning_scene_points": 100,
            "removed_robot_points": 0,
            "trajectory_dt_s": 0.1,
            "maximum_step_rad": 0.01,
            "terminal_check": {"safe": [True]},
        }


def build(
    hardware: FakeHardware | None = None,
    motion_planner: FakeMotionPlanner | None = None,
    *,
    primitive_settings=None,
    segment_client_factory=None,
    attached_lift_enabled=False,
):
    hardware = hardware or FakeHardware(clearance_m=1.0, arms=ARMS)
    environment = make_environment(hardware, manipulation=FAKE_MANIPULATION_CONFIG)
    registry = PrimitiveRegistry()
    register_manipulation_primitives(
        registry,
        environment,
        primitive_settings=primitive_settings,
        segment_client_factory=segment_client_factory or (lambda: _fake_segment),
        grasp_backend_factory=lambda config: _fake_grasp_backend,
        motion_planner_factory=lambda config: motion_planner or FakeMotionPlanner(),
        attached_lift_enabled=attached_lift_enabled,
    )
    return environment, registry


class ManipulationPrimitiveTest(unittest.TestCase):
    def test_home_right_arm_mirrors_the_left(self) -> None:
        # The right arm is the left arm with joints 1, 3, 5 and 6 negated;
        # FK places every link within 0.04 m of the left arm's y-mirror. The
        # old pose only exercised joint 3, so a joint-3-only mirror looked
        # right until a taught pose with nonzero odd joints sent the right
        # gripper 0.58 m off. Keep the two constants tied to the rule.
        np.testing.assert_allclose(
            NERO_HOME_JOINTS_RAD["right"],
            NERO_HOME_JOINTS_RAD["left"] * NERO_MIRROR_SIGNS,
        )
        # Every joint stays at least 0.03 rad inside the URDF limits cuRobo
        # plans within. The elbow limit is the one the asset generator writes
        # (0.01 rad inside the measured mechanical stop), and the rest pose
        # bends the elbow to exactly that margin.
        lower = np.asarray([-2.705, -1.74, -2.75, -1.01, -2.75, -0.73, -1.5708])
        upper = np.asarray([2.705, 1.74, 2.75, 2.19, 2.75, 0.95, 1.5708])
        for name, joints in NERO_HOME_JOINTS_RAD.items():
            self.assertTrue(np.all(joints - lower >= 0.03 - 1e-9), name)
            self.assertTrue(np.all(upper - joints >= 0.03 - 1e-9), name)

    def test_registers_the_v1_vocabulary_only(self) -> None:
        environment, registry = build()
        self.addCleanup(environment.safe_shutdown)

        self.assertEqual(
            sorted(registry.names()),
            [
                "close_gripper",
                "get_object_pose",
                "goto_grasp_pose",
                "goto_pose",
                "lift_grasped_object",
                "open_gripper",
                "sample_grasp_pose",
            ],
        )

    def test_documentation_states_units_without_arm_selection_advice(self) -> None:
        environment, registry = build()
        self.addCleanup(environment.safe_shutdown)

        docs = registry.documentation()

        self.assertIn("def get_object_pose(object_name: str,", docs)
        self.assertIn("return_zed_distance=True", docs)
        self.assertIn("def sample_grasp_pose(object_name: str,", docs)
        self.assertIn("arm: Literal['left', 'right']", docs)
        self.assertIn("def goto_pose(position", docs)
        self.assertIn("def goto_grasp_pose(object_name: str,", docs)
        self.assertIn("def lift_grasped_object(*, arm:", docs)
        self.assertIn("relative offset, or home", docs)
        self.assertIn("Do not use this primitive to calculate or execute a grasp pose", docs)
        self.assertIn("+X horizontal right", docs)
        self.assertIn('``goto_pose("home")`` immediately after', docs)
        self.assertIn("Open one native Nero CAN gripper", docs)
        self.assertIn("Close one native Nero CAN gripper", docs)
        self.assertNotIn("Pass the same", docs)
        self.assertNotIn("must match", docs)
        self.assertNotIn("arm=0", docs)
        self.assertNotIn("arm=1", docs)

    def test_object_names_are_asked_for_as_short_nouns(self) -> None:
        # Prompts that described the object's colour and position found no
        # instance where the bare category noun did.
        environment, registry = build()
        self.addCleanup(environment.safe_shutdown)

        docs = registry.documentation()

        self.assertEqual(docs.count("short category noun"), 3)
        self.assertIn("Use the name that worked for", docs)
        self.assertNotIn("Natural-language SAM3", docs)
        self.assertNotIn('``0``/``"left"``', docs)

    def test_manipulation_primitives_require_explicit_string_arm(self) -> None:
        environment, registry = build()
        self.addCleanup(environment.safe_shutdown)
        functions = registry.functions()

        for name in (
            "get_object_pose",
            "sample_grasp_pose",
            "goto_pose",
            "goto_grasp_pose",
            "open_gripper",
            "close_gripper",
            "lift_grasped_object",
        ):
            arm = inspect.signature(functions[name]).parameters["arm"]
            self.assertEqual(arm.kind, inspect.Parameter.KEYWORD_ONLY)
            self.assertIs(arm.default, inspect.Parameter.empty)

        with self.assertRaises(TypeError):
            functions["get_object_pose"]("pen")
        with self.assertRaises(TypeError):
            functions["sample_grasp_pose"]("pen")
        with self.assertRaises(TypeError):
            functions["goto_pose"]("home")
        with self.assertRaises(TypeError):
            functions["open_gripper"]()
        with self.assertRaisesRegex(ValueError, "string 'left' or 'right'"):
            functions["sample_grasp_pose"]("pen", arm=0)
        with self.assertRaisesRegex(ValueError, "string 'left' or 'right'"):
            functions["goto_grasp_pose"](
                "pen",
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0, 0.0],
                arm="1",
            )

    def test_attached_lift_requires_close_and_carries_geometry_into_home(
        self,
    ) -> None:
        hardware = FakeHardware(clearance_m=1.0, arms=ARMS)
        planner = FakeMotionPlanner()
        environment, registry = build(
            hardware, planner, attached_lift_enabled=True
        )
        self.addCleanup(environment.safe_shutdown)
        functions = registry.functions()

        functions["goto_grasp_pose"](
            "pen", [0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 0.0], arm="left"
        )
        with self.assertRaises(PrimitiveFailed) as caught:
            functions["lift_grasped_object"](arm="left")
        self.assertIn("close_gripper_required", caught.exception.reason)

        functions["close_gripper"](arm="left")
        lift = functions["lift_grasped_object"](arm="left", lift_m=0.12)
        self.assertTrue(lift["success"])
        self.assertTrue(lift["attached_object_collision_enabled"])
        self.assertEqual(planner.attached_lift_requests[0]["kwargs"]["lift_m"], 0.12)
        bounds = planner.attached_lift_requests[0]["attached_bounds_local"]
        self.assertEqual(bounds.shape, (2, 3))
        self.assertTrue(np.all(bounds[1] > bounds[0]))

        with self.assertRaises(PrimitiveFailed) as caught:
            functions["goto_pose"](
                [0.1, 0.0, 0.2], [1.0, 0.0, 0.0, 0.0], arm="left"
            )
        self.assertIn("forbidden_while_carrying", caught.exception.reason)

        home = functions["goto_pose"]("home", arm="left")
        self.assertTrue(home["success"])
        self.assertTrue(home["attached_object_collision_enabled"])
        np.testing.assert_allclose(
            planner.joint_target_requests[-1]["kwargs"][
                "attached_bounds_local"
            ],
            bounds,
        )

        functions["open_gripper"](arm="left")
        cartesian = functions["goto_pose"](
            [0.1, 0.0, 0.2], [1.0, 0.0, 0.0, 0.0], arm="left"
        )
        self.assertTrue(cartesian["success"])

    def test_argument_validation_happens_before_any_motion(self) -> None:
        hardware = FakeHardware(clearance_m=1.0, arms=ARMS)
        environment, registry = build(hardware)
        self.addCleanup(environment.safe_shutdown)
        functions = registry.functions()

        with self.assertRaises(ValueError):
            functions["goto_pose"]([0.1, 0.0, 0.2], [1.0, 0.0, 0.0, 0.0], arm=5)
        with self.assertRaises(ValueError):
            functions["goto_pose"](
                [0.1, 0.0, 0.2],
                [1.0, 0.0, 0.0, 0.0],
                arm="left",
                z_approach=999.0,
            )
        with self.assertRaises(ValueError):
            functions["open_gripper"](arm="not-an-arm")
        with self.assertRaises(ValueError):
            functions["close_gripper"](arm=2)
        with self.assertRaises(ValueError):
            functions["goto_pose"]("not-home", arm="left")
        with self.assertRaises(ValueError):
            functions["goto_pose"](
                "home", [1.0, 0.0, 0.0, 0.0], arm="left"
            )

        self.assertEqual(hardware.arm_pose_commands, [])
        self.assertEqual(hardware.gripper_commands, [])

    def test_goto_pose_moves_and_reports_success(self) -> None:
        hardware = FakeHardware(clearance_m=1.0, arms=ARMS)
        environment, registry = build(hardware)
        self.addCleanup(environment.safe_shutdown)

        result = registry.functions()["goto_pose"](
            [0.1, 0.0, 0.2], [1.0, 0.0, 0.0, 0.0], arm="left"
        )

        self.assertTrue(result["success"])
        self.assertEqual(result["arm"], "left")
        self.assertEqual(len(hardware.arm_pose_commands), 1)

    def test_goto_pose_approach_sends_two_commands(self) -> None:
        hardware = FakeHardware(clearance_m=1.0, arms=ARMS)
        environment, registry = build(hardware)
        self.addCleanup(environment.safe_shutdown)

        result = registry.functions()["goto_pose"](
            [0.1, 0.0, 0.2],
            [1.0, 0.0, 0.0, 0.0],
            arm="left",
            z_approach=0.05,
        )

        self.assertTrue(result["success"])
        self.assertEqual(len(hardware.arm_pose_commands), 2)
        self.assertEqual(result["submotions"][0]["phase"], "approach")
        self.assertEqual(result["submotions"][1]["phase"], "target")

    def test_goto_pose_current_applies_pitch_and_arm_calibration(self) -> None:
        arms = {
            "left": {
                "joint_positions": [0.0] * 6,
                "joint_pos": [0.0] * 7,
                "tcp_pose_xyz_rpy": [0.20, 0.30, 0.40, 0.1, 0.2, 0.3],
            },
            "right": {
                "joint_positions": [0.0] * 6,
                "joint_pos": [0.0] * 7,
            },
            "estop_latched": False,
            "end_effector_frame": "tcp",
        }
        hardware = FakeHardware(clearance_m=1.0, arms=arms)
        root_half = np.sqrt(0.5)
        hardware.ground_down_camera_xyz = np.asarray(
            [0.0, root_half, root_half], dtype=np.float64
        )
        environment, registry = build(hardware)
        self.addCleanup(environment.safe_shutdown)
        arm_from_camera = np.eye(4, dtype=np.float64)
        arm_from_camera[:3, :3] = np.asarray(
            [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        )
        environment.manipulation_config[
            "left_arm_from_camera"
        ] = arm_from_camera.tolist()

        result = registry.functions()["goto_pose"](
            "current", arm="left", camera_offset_xyz=[0.0, -0.10, 0.0]
        )

        self.assertTrue(result["success"])
        self.assertTrue(result["relative_to_measured_tcp"])
        np.testing.assert_allclose(
            result["delta_camera_xyz_m"],
            [0.0, -0.10 * root_half, -0.10 * root_half],
        )
        np.testing.assert_allclose(
            result["delta_arm_xyz_m"],
            [-0.10 * root_half, 0.0, -0.10 * root_half],
        )
        np.testing.assert_allclose(
            hardware.arm_pose_commands[0]["pose_xyz_rpy"],
            [
                0.20 - 0.10 * root_half,
                0.30,
                0.40 - 0.10 * root_half,
                0.1,
                0.2,
                0.3,
            ],
        )

    def test_goto_pose_current_rejects_invalid_relative_requests(self) -> None:
        hardware = FakeHardware(clearance_m=1.0, arms=ARMS)
        environment, registry = build(hardware)
        self.addCleanup(environment.safe_shutdown)
        goto_pose = registry.functions()["goto_pose"]

        with self.assertRaises(ValueError):
            goto_pose("current", arm="left")
        with self.assertRaises(ValueError):
            goto_pose(
                "current",
                [1.0, 0.0, 0.0, 0.0],
                arm="left",
                camera_offset_xyz=[0.0, -0.1, 0.0],
            )
        with self.assertRaises(ValueError):
            goto_pose(
                [0.1, 0.0, 0.2],
                [1.0, 0.0, 0.0, 0.0],
                arm="left",
                camera_offset_xyz=[0.0, -0.1, 0.0],
            )

        self.assertEqual(hardware.arm_pose_commands, [])

    def test_failed_goto_pose_raises_primitive_failed(self) -> None:
        hardware = FakeHardware(clearance_m=1.0, arms=ARMS)
        hardware.move_arm_pose_result = {"success": False, "reason": "ik_failed"}
        environment, registry = build(hardware)
        self.addCleanup(environment.safe_shutdown)

        with self.assertRaises(PrimitiveFailed) as caught:
            registry.functions()["goto_pose"](
                [0.1, 0.0, 0.2], [1.0, 0.0, 0.0, 0.0], arm="left"
            )

        self.assertIn("ik_failed", caught.exception.reason)

    def test_goto_grasp_pose_uses_guarded_joint_trajectory(self) -> None:
        hardware = FakeHardware(clearance_m=1.0, arms=ARMS)
        planner = FakeMotionPlanner()
        environment, registry = build(hardware, planner)
        self.addCleanup(environment.safe_shutdown)

        result = registry.functions()["goto_grasp_pose"](
            "pen", [0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 0.0], arm="left"
        )

        self.assertTrue(result["success"])
        self.assertEqual(result["primitive"], "goto_grasp_pose")
        self.assertEqual(planner.plan_grasp_requests[0]["clearance_m"], 0.008)
        self.assertEqual(planner.plan_grasp_requests[0]["approach_samples"], 8)
        np.testing.assert_allclose(
            planner.plan_grasp_requests[0]["goal_joint_seed"], [0.0] * 7
        )
        self.assertTrue(result["pi_goal_joint_seed_forwarded"])
        self.assertEqual(len(hardware.arm_trajectory_commands), 1)
        self.assertEqual(hardware.arm_pose_commands, [])
        self.assertTrue(result["home_retreat_available"])
        self.assertEqual(result["home_retreat_waypoint_count"], 2)

    def test_planner_attempt_bounds_and_timings_flow_through(self) -> None:
        class TimedMotionPlanner(FakeMotionPlanner):
            def plan_grasp(self, current_joints, tcp_poses, obstacle_points, **kwargs):
                plan = super().plan_grasp(
                    current_joints, tcp_poses, obstacle_points, **kwargs
                )
                return {
                    **plan,
                    "timings": {"pregrasp_plan_s": 0.5, "total_s": 0.7},
                    "scene": {
                        "mesh_name": "observed_scene_000003",
                        "evicted_mesh_names": ["observed_scene_000002"],
                    },
                    "planner_attempts": dict(kwargs.get("planner_attempts") or {}),
                    "pregrasp_solver": {"solve_time_s": 0.4},
                }

        hardware = FakeHardware(clearance_m=1.0, arms=ARMS)
        planner = TimedMotionPlanner()
        environment = make_environment(
            hardware,
            manipulation={
                **FAKE_MANIPULATION_CONFIG,
                "grasp_motion_max_attempts": 3,
                "grasp_motion_enable_graph_attempt": 1,
            },
        )
        registry = PrimitiveRegistry()
        register_manipulation_primitives(
            registry,
            environment,
            segment_client_factory=lambda: _fake_segment,
            grasp_backend_factory=lambda config: _fake_grasp_backend,
            motion_planner_factory=lambda config: planner,
            attached_lift_enabled=False,
        )
        self.addCleanup(environment.safe_shutdown)
        functions = registry.functions()

        result = functions["goto_grasp_pose"](
            "pen", [0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 0.0], arm="left"
        )

        self.assertTrue(result["success"])
        self.assertEqual(
            planner.plan_grasp_requests[0]["planner_attempts"],
            {"max_attempts": 3, "enable_graph_attempt": 1},
        )
        # The service's per-phase diagnostics must reach trace.json unchanged.
        self.assertEqual(result["planner_timings"]["total_s"], 0.7)
        self.assertEqual(result["planner_scene"]["mesh_name"], "observed_scene_000003")
        self.assertEqual(
            result["planner_solver"]["pregrasp_solver"]["solve_time_s"], 0.4
        )
        self.assertEqual(
            result["planner_attempts"], {"max_attempts": 3, "enable_graph_attempt": 1}
        )

        home = functions["goto_pose"]("home", arm="left")

        self.assertTrue(home["success"])
        self.assertEqual(
            planner.joint_target_requests[0]["kwargs"]["planner_attempts"],
            {"max_attempts": 3, "enable_graph_attempt": 1},
        )
        self.assertIn("planner_timings", home)

    def test_planner_attempt_bounds_default_to_unset(self) -> None:
        hardware = FakeHardware(clearance_m=1.0, arms=ARMS)
        planner = FakeMotionPlanner()
        environment, registry = build(hardware, planner)
        self.addCleanup(environment.safe_shutdown)

        registry.functions()["goto_grasp_pose"](
            "pen", [0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 0.0], arm="left"
        )

        self.assertEqual(planner.plan_grasp_requests[0]["planner_attempts"], {})

    def test_scene_options_flow_through_to_the_planner(self) -> None:
        hardware = FakeHardware(clearance_m=1.0, arms=ARMS)
        planner = FakeMotionPlanner()
        environment = make_environment(
            hardware,
            manipulation={
                **FAKE_MANIPULATION_CONFIG,
                "grasp_motion_scene_mesh": "greedy",
                "grasp_motion_scene_coarse_voxel_m": 0.03,
                "grasp_motion_scene_fine_radius_m": 0.3,
            },
        )
        registry = PrimitiveRegistry()
        register_manipulation_primitives(
            registry,
            environment,
            segment_client_factory=lambda: _fake_segment,
            grasp_backend_factory=lambda config: _fake_grasp_backend,
            motion_planner_factory=lambda config: planner,
            attached_lift_enabled=False,
        )
        self.addCleanup(environment.safe_shutdown)

        registry.functions()["goto_grasp_pose"](
            "pen", [0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 0.0], arm="left"
        )

        self.assertEqual(
            planner.plan_grasp_requests[0]["planner_attempts"],
            {
                "scene_mesh": "greedy",
                "scene_coarse_voxel_m": 0.03,
                "scene_fine_radius_m": 0.3,
            },
        )

    def test_invalid_planner_attempt_bounds_are_rejected(self) -> None:
        for key, value in (
            ("grasp_motion_max_attempts", 0),
            ("grasp_motion_max_attempts", 11),
            ("grasp_motion_max_attempts", True),
            ("grasp_motion_enable_graph_attempt", -1),
            ("grasp_motion_enable_graph_attempt", 2.5),
            ("grasp_motion_finetune_attempts", 4),
            ("grasp_motion_finetune_attempts", -1),
            ("grasp_motion_scene_mesh", "octree"),
            ("grasp_motion_scene_coarse_voxel_m", 0.06),
            ("grasp_motion_scene_fine_radius_m", 0.05),
            ("grasp_motion_scene_crop_radius_m", 1.0),
        ):
            with self.subTest(key=key, value=value):
                hardware = FakeHardware(clearance_m=1.0, arms=ARMS)
                environment = make_environment(
                    hardware,
                    manipulation={**FAKE_MANIPULATION_CONFIG, key: value},
                )
                registry = PrimitiveRegistry()
                register_manipulation_primitives(
                    registry,
                    environment,
                    segment_client_factory=lambda: _fake_segment,
                    grasp_backend_factory=lambda config: _fake_grasp_backend,
                    motion_planner_factory=lambda config: FakeMotionPlanner(),
                    attached_lift_enabled=False,
                )
                self.addCleanup(environment.safe_shutdown)
                with self.assertRaises((ValueError, PrimitiveFailed)):
                    registry.functions()["goto_grasp_pose"](
                        "pen", [0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 0.0], arm="left"
                    )

    def test_sampled_alternatives_reach_curobo_when_attached_lift_disabled(
        self,
    ) -> None:
        hardware = FakeHardware(clearance_m=1.0, arms=ARMS)
        planner = FakeMotionPlanner()
        environment = make_environment(
            hardware, manipulation=FAKE_MANIPULATION_CONFIG
        )
        registry = PrimitiveRegistry()
        register_manipulation_primitives(
            registry,
            environment,
            segment_client_factory=lambda: _fake_segment,
            grasp_backend_factory=lambda config: _fake_goalset_grasp_backend,
            motion_planner_factory=lambda config: planner,
            attached_lift_enabled=False,
        )
        self.addCleanup(environment.safe_shutdown)
        functions = registry.functions()

        position, quaternion = functions["sample_grasp_pose"]("pen", arm="left")
        result = functions["goto_grasp_pose"](
            "pen", position, quaternion, arm="left"
        )

        self.assertTrue(result["success"])
        goalset = planner.plan_grasp_requests[0]["tcp_poses"]
        self.assertGreater(len(goalset), 1)
        np.testing.assert_allclose(goalset[0][:3, 3], position)
        self.assertEqual(result["goalset_candidate_count"], len(goalset))

    def test_sample_grasp_filters_overbound_pose_before_pi_batch_ik(self) -> None:
        hardware = FakeHardware(clearance_m=1.0, arms=ARMS)
        environment = make_environment(
            hardware, manipulation=FAKE_MANIPULATION_CONFIG
        )
        registry = PrimitiveRegistry()

        near_pose = np.eye(4)
        near_pose[:3, 3] = [0.0, 0.0, 0.50]
        overbound_pose = np.eye(4)
        # The default GraspGen-X origin-to-TCP shift adds another 0.105 m.
        overbound_pose[:3, 3] = [0.0, 0.0, 0.95]
        overbound_pregrasp_pose = np.eye(4)
        overbound_pregrasp_pose[:3, :3] = np.diag([1.0, -1.0, -1.0])
        # TCP target is 0.95 m, but its far-side 10 cm pre-grasp is 1.05 m.
        overbound_pregrasp_pose[:3, 3] = [0.0, 0.0, 1.055]

        def grasp_backend(depth, intrinsics, segmentation, instance_id):
            del depth, intrinsics, segmentation, instance_id
            return (
                np.stack(
                    [overbound_pose, overbound_pregrasp_pose, near_pose]
                ),
                np.asarray([0.95, 0.90, 0.80]),
                np.asarray(
                    [
                        [0.0, 0.0, 1.0],
                        [0.0, 0.0, 1.0],
                        [0.0, 0.0, 1.0],
                    ]
                ),
            )

        original_plan_arm_poses = hardware.plan_arm_poses

        def reject_overbound_batch(arm, poses_xyz_rpy):
            assert all(
                np.linalg.norm(np.asarray(pose[:3], dtype=float)) <= 1.0
                for pose in poses_xyz_rpy
            )
            return original_plan_arm_poses(arm, poses_xyz_rpy)

        hardware.plan_arm_poses = reject_overbound_batch
        register_manipulation_primitives(
            registry,
            environment,
            segment_client_factory=lambda: _fake_segment,
            grasp_backend_factory=lambda config: grasp_backend,
            motion_planner_factory=lambda config: FakeMotionPlanner(),
        )
        self.addCleanup(environment.safe_shutdown)

        position, quaternion = registry.functions()["sample_grasp_pose"](
            "pen", arm="left"
        )

        self.assertEqual(quaternion.shape, (4,))
        self.assertLessEqual(float(np.linalg.norm(position)), 1.0)
        self.assertEqual(len(hardware.arm_plan_requests), 1)
        self.assertEqual(len(hardware.arm_plan_requests[0]["poses_xyz_rpy"]), 1)

    def test_sample_grasp_prefers_near_side_approach_over_model_score(
        self,
    ) -> None:
        hardware = FakeHardware(clearance_m=1.0, arms=ARMS)
        environment = make_environment(
            hardware, manipulation=FAKE_MANIPULATION_CONFIG
        )
        planner = FakeMotionPlanner()

        near_side = np.eye(4)
        near_side[:3, 3] = [0.0, 0.0, 0.794]
        far_side = np.eye(4)
        far_side[:3, :3] = np.diag([1.0, -1.0, -1.0])
        far_side[:3, 3] = [0.0, 0.0, 1.004]

        def grasp_backend(depth, intrinsics, segmentation, instance_id):
            del depth, intrinsics, segmentation, instance_id
            return (
                # The far-side grasp deliberately has the higher model score.
                np.stack([far_side, near_side]),
                np.asarray([0.99, 0.50]),
                np.asarray([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]),
            )

        controller = ManipulationController(
            environment,
            segment_client_factory=lambda: _fake_segment,
            grasp_backend_factory=lambda config: grasp_backend,
            motion_planner_factory=lambda config: planner,
        )
        self.addCleanup(environment.safe_shutdown)

        position, quaternion = controller.sample_grasp_pose("pen", arm=0)
        result = controller.goto_grasp_pose(
            "pen", position, quaternion, arm=0
        )

        self.assertTrue(result["success"])
        self.assertGreater(quaternion_wxyz_to_matrix(quaternion)[2, 2], 0.99)
        selected = next(
            candidate
            for candidate in controller.last_grasp_debug["candidates"]
            if candidate["selected"]
        )
        self.assertEqual(selected["candidate_index"], 1)
        self.assertTrue(selected["near_side_approach"])
        self.assertLess(
            selected["pregrasp_position_norm_m"],
            selected["position_norm_m"],
        )
        first_goal = planner.plan_grasp_requests[0]["tcp_poses"][0]
        self.assertGreater(first_goal[2, 2], 0.99)

    def test_goto_grasp_pose_lets_curobo_solve_when_pi_ik_did_not_converge(
        self,
    ) -> None:
        hardware = FakeHardware(clearance_m=1.0, arms=ARMS)
        hardware.plan_arm_poses_result = {
            "plans": [
                {
                    "success": True,
                    "candidate_index": 0,
                    "ik_converged": False,
                    "ik_position_error_m": 0.10,
                    "ik_rotation_error_rad": 0.55,
                    "ik_joint_target": [0.1] * 7,
                }
            ]
        }
        planner = FakeMotionPlanner()
        environment, registry = build(hardware, planner)
        self.addCleanup(environment.safe_shutdown)

        result = registry.functions()["goto_grasp_pose"](
            "pen", [0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 0.0], arm="left"
        )

        self.assertTrue(result["success"])
        self.assertFalse(result["pi_goal_ik_converged"])
        self.assertFalse(result["pi_goal_joint_seed_forwarded"])
        self.assertIsNone(planner.plan_grasp_requests[0]["goal_joint_seed"])
        self.assertEqual(len(hardware.arm_trajectory_commands), 1)

    def test_goto_grasp_pose_rejects_malformed_nonconverged_pi_seed(self) -> None:
        hardware = FakeHardware(clearance_m=1.0, arms=ARMS)
        hardware.plan_arm_poses_result = {
            "plans": [
                {
                    "success": True,
                    "candidate_index": 0,
                    "ik_converged": False,
                    "ik_position_error_m": 0.10,
                    "ik_rotation_error_rad": 0.55,
                    "ik_joint_target": [0.1] * 6,
                }
            ]
        }
        planner = FakeMotionPlanner()
        environment, registry = build(hardware, planner)
        self.addCleanup(environment.safe_shutdown)

        with self.assertRaises(PrimitiveFailed) as caught:
            registry.functions()["goto_grasp_pose"](
                "pen", [0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 0.0], arm="left"
            )

        self.assertIn("pi_grasp_goal_joint_seed_unavailable", caught.exception.reason)
        self.assertEqual(planner.plan_grasp_requests, [])

    def test_goto_grasp_pose_chunks_long_planner_path_without_dropping_points(
        self,
    ) -> None:
        hardware = FakeHardware(clearance_m=1.0, arms=ARMS)
        environment, registry = build(hardware, LongMotionPlanner())
        self.addCleanup(environment.safe_shutdown)

        # The Pi accepts a whole planner path in one RPC nowadays; force the
        # old 64-waypoint limit to keep exercising the chunk splitter.
        with mock.patch.object(manipulation_module, "PI_MAX_TRAJECTORY_WAYPOINTS", 64):
            result = registry.functions()["goto_grasp_pose"](
                "pen", [0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 0.0], arm="left"
            )

        self.assertTrue(result["success"])
        self.assertEqual(result["trajectory_waypoint_count"], 81)
        self.assertEqual(result["trajectory_chunk_count"], 2)
        self.assertEqual(
            [len(command["waypoints"]) for command in hardware.arm_trajectory_commands],
            [64, 18],
        )
        first = hardware.arm_trajectory_commands[0]["waypoints"]
        second = hardware.arm_trajectory_commands[1]["waypoints"]
        np.testing.assert_allclose(first[-1], second[0])
        np.testing.assert_allclose(second[-1][0], 0.8)

    def test_goto_grasp_pose_streams_a_planner_path_as_one_rpc(self) -> None:
        hardware = FakeHardware(clearance_m=1.0, arms=ARMS)
        environment, registry = build(hardware, LongMotionPlanner())
        self.addCleanup(environment.safe_shutdown)

        result = registry.functions()["goto_grasp_pose"](
            "pen", [0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 0.0], arm="left"
        )

        self.assertTrue(result["success"])
        self.assertEqual(result["trajectory_waypoint_count"], 81)
        self.assertEqual(result["trajectory_chunk_count"], 1)
        self.assertEqual(len(hardware.arm_trajectory_commands), 1)
        self.assertEqual(len(hardware.arm_trajectory_commands[0]["waypoints"]), 81)

    def test_execution_reports_timing_and_totals_without_decimation(self) -> None:
        hardware = FakeHardware(clearance_m=1.0, arms=ARMS)
        environment, registry = build(hardware, LongMotionPlanner())
        self.addCleanup(environment.safe_shutdown)

        result = registry.functions()["goto_grasp_pose"](
            "pen", [0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 0.0], arm="left"
        )

        self.assertTrue(result["success"])
        self.assertIsNone(result["execution_step_cap_rad"])
        self.assertEqual(result["trajectory_waypoint_count"], 81)
        self.assertEqual(result["executed_waypoint_count"], 81)
        self.assertEqual(result["last_chunk_waypoint_count"], 81)
        self.assertGreaterEqual(result["execution_elapsed_s"], 0.0)
        self.assertEqual(len(result["trajectory_chunks"]), 1)
        for chunk in result["trajectory_chunks"]:
            self.assertGreaterEqual(chunk["rpc_elapsed_s"], 0.0)
            self.assertIsNone(chunk["pi_executed_count"])

    def test_decimation_keeps_planner_samples_and_bounds_steps(self) -> None:
        rng = np.random.default_rng(7)
        # A smooth 7-joint path sampled finely, as cuRobo emits it.
        steps = np.linspace(0.0, 1.0, 120)[:, None] * np.asarray(
            [0.8, -0.5, 0.3, 0.0, 0.2, -0.1, 0.05]
        )[None, :]
        trajectory = steps + 0.002 * rng.standard_normal(steps.shape)
        decimate = ManipulationController._decimate_for_execution

        kept = decimate(trajectory, 0.09)

        self.assertLess(len(kept), len(trajectory))
        np.testing.assert_array_equal(kept[0], trajectory[0])
        np.testing.assert_array_equal(kept[-1], trajectory[-1])
        self.assertLessEqual(float(np.max(np.abs(np.diff(kept, axis=0)))), 0.09)
        # Every kept row is an original planner sample, in order.
        rows = {tuple(row) for row in trajectory}
        self.assertTrue(all(tuple(row) in rows for row in kept))
        indices = [int(np.flatnonzero((trajectory == row).all(axis=1))[0]) for row in kept]
        self.assertEqual(indices, sorted(indices))
        # A path whose steps already exceed the cap is returned unchanged.
        coarse = np.linspace(0.0, 1.0, 11)[:, None] * np.ones((1, 7))
        np.testing.assert_array_equal(decimate(coarse, 0.09), coarse)
        two = trajectory[:2]
        np.testing.assert_array_equal(decimate(two, 0.09), two)

    def test_execution_step_cap_decimates_approach_but_not_contact_segment(
        self,
    ) -> None:
        class SegmentedMotionPlanner(FakeMotionPlanner):
            def plan_grasp(self, current_joints, tcp_poses, obstacle_points, **kwargs):
                del tcp_poses, obstacle_points, kwargs
                start = np.asarray(current_joints, dtype=float)
                approach = np.tile(start, (61, 1))
                approach[:, 0] += np.linspace(0.0, 0.6, 61)
                contact = np.tile(approach[-1], (20, 1))
                contact[:, 1] += np.linspace(0.005, 0.1, 20)
                waypoints = np.concatenate([approach, contact], axis=0)
                return {
                    "success": True,
                    "reason": "collision_free_path_planned",
                    "waypoints": waypoints.tolist(),
                    "segments": [
                        {"phase": "approach", "waypoints": approach.tolist()},
                        {"phase": "grasp", "waypoints": contact.tolist()},
                    ],
                    "planning_time_s": 0.01,
                    "planning_scene_points": 100,
                    "removed_robot_points": 0,
                    "trajectory_dt_s": 0.1,
                    "maximum_step_rad": 0.01,
                    "terminal_check": {"safe": [True]},
                }

        hardware = FakeHardware(clearance_m=1.0, arms=ARMS)
        environment = make_environment(
            hardware,
            manipulation={
                **FAKE_MANIPULATION_CONFIG,
                "grasp_motion_execution_step_rad": 0.09,
            },
        )
        registry = PrimitiveRegistry()
        register_manipulation_primitives(
            registry,
            environment,
            segment_client_factory=lambda: _fake_segment,
            grasp_backend_factory=lambda config: _fake_grasp_backend,
            motion_planner_factory=lambda config: SegmentedMotionPlanner(),
            attached_lift_enabled=False,
        )
        self.addCleanup(environment.safe_shutdown)

        result = registry.functions()["goto_grasp_pose"](
            "pen", [0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 0.0], arm="left"
        )

        self.assertTrue(result["success"], result)
        self.assertEqual(result["execution_step_cap_rad"], 0.09)
        self.assertEqual(result["trajectory_waypoint_count"], 81)
        self.assertEqual(result["protected_tail_count"], 21)
        sent = np.concatenate(
            [
                np.asarray(command["waypoints"])
                if index == 0
                else np.asarray(command["waypoints"])[1:]
                for index, command in enumerate(hardware.arm_trajectory_commands)
            ],
            axis=0,
        )
        self.assertEqual(len(sent), result["executed_waypoint_count"])
        # Approach: 61 samples 0.01 rad apart -> at most ceil(0.6 / 0.09) + 1.
        self.assertLessEqual(len(sent), 8 + 20)
        self.assertLessEqual(float(np.max(np.abs(np.diff(sent, axis=0)))), 0.09)
        # The pre-grasp sample and the whole 20-point contact segment are intact.
        planned = np.asarray(SegmentedMotionPlanner().plan_grasp(
            [0.0] * 7, None, None)["waypoints"])
        np.testing.assert_allclose(sent[-21:], planned[-21:])
        np.testing.assert_allclose(sent[0], planned[0])
        # The cached retreat (reversed contact segment) is never decimated.
        self.assertEqual(result["home_retreat_waypoint_count"], 21)

    def test_invalid_execution_step_cap_is_rejected(self) -> None:
        for value in (0.0, 0.11, "fast"):
            with self.subTest(value=value):
                hardware = FakeHardware(clearance_m=1.0, arms=ARMS)
                environment = make_environment(
                    hardware,
                    manipulation={
                        **FAKE_MANIPULATION_CONFIG,
                        "grasp_motion_execution_step_rad": value,
                    },
                )
                registry = PrimitiveRegistry()
                register_manipulation_primitives(
                    registry,
                    environment,
                    segment_client_factory=lambda: _fake_segment,
                    grasp_backend_factory=lambda config: _fake_grasp_backend,
                    motion_planner_factory=lambda config: FakeMotionPlanner(),
                    attached_lift_enabled=False,
                )
                self.addCleanup(environment.safe_shutdown)
                with self.assertRaises((ValueError, PrimitiveFailed)):
                    registry.functions()["goto_grasp_pose"](
                        "pen", [0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 0.0], arm="left"
                    )
                self.assertEqual(hardware.arm_trajectory_commands, [])

    def test_goto_pose_home_uses_guarded_collision_planned_trajectory(self) -> None:
        hardware = FakeHardware(clearance_m=1.0, arms=ARMS)
        planner = FakeMotionPlanner()
        environment, registry = build(hardware, planner)
        self.addCleanup(environment.safe_shutdown)

        result = registry.functions()["goto_pose"]("home", arm="right")

        self.assertTrue(result["success"])
        self.assertEqual(result["primitive"], "goto_pose")
        self.assertEqual(result["target"], "home")
        self.assertEqual(
            result["target_joints"],
            NERO_HOME_JOINTS_RAD["right"].tolist(),
        )
        self.assertEqual(len(hardware.arm_trajectory_commands), 1)
        self.assertEqual(hardware.arm_pose_commands, [])
        self.assertGreater(len(planner.joint_target_requests[0]["obstacle_points"]), 0)
        np.testing.assert_allclose(
            planner.joint_target_requests[0]["target_joints"],
            NERO_HOME_JOINTS_RAD["right"].tolist(),
        )

    def test_home_after_grasp_reverses_checked_grasp_segment_before_planning(
        self,
    ) -> None:
        hardware = FakeHardware(clearance_m=1.0, arms=ARMS)
        planner = FakeMotionPlanner()
        environment, registry = build(hardware, planner)
        self.addCleanup(environment.safe_shutdown)
        functions = registry.functions()

        functions["goto_grasp_pose"](
            "pen", [0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 0.0], arm="left"
        )
        result = functions["goto_pose"]("home", arm="left")

        self.assertTrue(result["success"])
        self.assertTrue(result["grasp_retreat"]["executed"])
        self.assertEqual(len(hardware.arm_trajectory_commands), 3)
        retreat = np.asarray(hardware.arm_trajectory_commands[1]["waypoints"])
        np.testing.assert_allclose(retreat[0], [0.01] * 7)
        np.testing.assert_allclose(retreat[-1], [0.005] * 7)
        self.assertEqual(len(planner.joint_target_requests), 1)

    def test_home_while_carrying_attaches_the_held_object_without_attached_lift(
        self,
    ) -> None:
        # With the attached-lift feature off, close_gripper left the held
        # object in the obstacle cloud, so cuRobo saw the gripper start inside
        # it and every goto_pose('home') after a grasp failed with "Start or
        # End state in collision" (2026-09-05). The object must leave the
        # cloud and join the robot model regardless of that feature.
        hardware = FakeHardware(clearance_m=1.0, arms=ARMS)
        planner = FakeMotionPlanner()
        environment, registry = build(hardware, planner)
        self.addCleanup(environment.safe_shutdown)
        functions = registry.functions()

        functions["goto_grasp_pose"](
            "pen", [0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 0.0], arm="left"
        )
        functions["close_gripper"](arm="left")
        self.assertIn("left", environment.robot_self_filter_attached_objects())
        # Free-form Cartesian motion is refused while carrying, which also
        # keeps the cached grasp retreat from being thrown away.
        with self.assertRaises(PrimitiveFailed) as refused:
            functions["goto_pose"]([0.1, 0.0, 0.2], [1.0, 0.0, 0.0, 0.0], arm="left")
        self.assertIn(
            "freeform_cartesian_motion_forbidden_while_carrying_object",
            str(refused.exception),
        )

        result = functions["goto_pose"]("home", arm="left")

        self.assertTrue(result["success"])
        self.assertTrue(result["attached_object_collision_enabled"])
        self.assertEqual(result["object_name"], "pen")
        # The cached grasp retreat survived the refused Cartesian call, but a
        # blind replay is deliberately skipped while carrying: the whole path
        # is planned fresh with the object attached.
        self.assertTrue(result["grasp_retreat"]["available"])
        self.assertFalse(result["grasp_retreat"]["executed"])
        bounds = planner.joint_target_requests[-1]["kwargs"]["attached_bounds_local"]
        self.assertIsNotNone(bounds)
        self.assertEqual(np.asarray(bounds).shape, (2, 3))

        functions["open_gripper"](arm="left")
        self.assertNotIn("left", environment.robot_self_filter_attached_objects())
        released = functions["goto_pose"]("home", arm="left")
        self.assertTrue(released["success"])
        self.assertNotIn("attached_object_collision_enabled", released)

    def test_attached_object_bounds_ignore_far_mask_points(self) -> None:
        # One masked pixel sees the background 1 m behind the object (still
        # inside the 2.5 m metric-depth window), as when the SAM3 mask leaks
        # onto the desk or a silhouette flying pixel survives. It must not
        # stretch the attached box past what the grasp-motion planner accepts
        # (0.40 m), which is what stopped home after a grasp on 2026-09-07.
        hardware = FakeHardware(clearance_m=1.0, arms=ARMS)
        original_latest_frame = hardware.latest_frame

        def far_pixel_frame(*, max_age_s: float):
            frame = original_latest_frame(max_age_s=max_age_s)
            frame.depth_m[21, 29] = 2.0
            return frame

        hardware.latest_frame = far_pixel_frame
        planner = FakeMotionPlanner()
        environment, registry = build(hardware, planner)
        self.addCleanup(environment.safe_shutdown)
        functions = registry.functions()

        result = functions["goto_grasp_pose"](
            "pen", [0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 0.0], arm="left"
        )

        self.assertTrue(result["success"])
        self.assertEqual(
            result["attached_point_total"], result["attached_point_count"] + 1
        )
        self.assertFalse(result["attached_radius_fallback"])
        self.assertEqual(result["attached_clamped_axes"], [])
        self.assertEqual(len(result["attached_extent_m"]), 3)
        for extent in result["attached_extent_m"]:
            self.assertGreaterEqual(extent, 0.01)
            self.assertLessEqual(extent, 0.40)

        functions["close_gripper"](arm="left")
        home = functions["goto_pose"]("home", arm="left")
        self.assertTrue(home["success"])
        bounds = np.asarray(
            planner.joint_target_requests[-1]["kwargs"]["attached_bounds_local"]
        )
        np.testing.assert_array_less(bounds[1] - bounds[0], 0.40 + 1e-9)

    def test_home_never_executes_a_stale_grasp_retreat(self) -> None:
        hardware = FakeHardware(clearance_m=1.0, arms=ARMS)
        planner = FakeMotionPlanner()
        environment, registry = build(hardware, planner)
        self.addCleanup(environment.safe_shutdown)
        functions = registry.functions()

        functions["goto_grasp_pose"](
            "pen", [0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 0.0], arm="left"
        )
        hardware.arms["left"]["joint_pos"] = [0.5] * 7
        result = functions["goto_pose"]("home", arm="left")

        self.assertTrue(result["success"])
        self.assertFalse(result["grasp_retreat"]["executed"])
        self.assertEqual(
            result["grasp_retreat"]["skipped_reason"],
            "cached_grasp_retreat_start_mismatch",
        )
        self.assertEqual(len(hardware.arm_trajectory_commands), 2)

    def test_motion_away_from_open_grasp_invalidates_pending_attachment(
        self,
    ) -> None:
        hardware = FakeHardware(clearance_m=1.0, arms=ARMS)
        planner = FakeMotionPlanner()
        environment, registry = build(
            hardware, planner, attached_lift_enabled=True
        )
        self.addCleanup(environment.safe_shutdown)
        functions = registry.functions()

        functions["goto_grasp_pose"](
            "pen", [0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 0.0], arm="left"
        )
        functions["goto_pose"](
            [0.1, 0.0, 0.2], [1.0, 0.0, 0.0, 0.0], arm="left"
        )
        functions["close_gripper"](arm="left")

        with self.assertRaises(PrimitiveFailed) as caught:
            functions["lift_grasped_object"](arm="left")
        self.assertIn("no_pending_grasp", caught.exception.reason)

    def test_disabled_attached_lift_allows_home_directly_after_close(self) -> None:
        # With the attached-lift feature off there is no lift step, so home
        # must be allowed straight after close (no attached_lift_required gate)
        # while still carrying the held object into the plan.
        hardware = FakeHardware(clearance_m=1.0, arms=ARMS)
        planner = FakeMotionPlanner()
        environment, registry = build(hardware, planner)
        self.addCleanup(environment.safe_shutdown)
        functions = registry.functions()

        functions["goto_grasp_pose"](
            "pen", [0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 0.0], arm="left"
        )
        functions["close_gripper"](arm="left")
        home = functions["goto_pose"]("home", arm="left")

        self.assertTrue(home["success"])
        self.assertNotEqual(home.get("reason"), "attached_lift_required_before_home")
        self.assertEqual(len(planner.plan_grasp_requests[0]["tcp_poses"]), 1)
        self.assertIsNotNone(
            planner.joint_target_requests[0]["kwargs"][
                "attached_bounds_local"
            ]
        )
        self.assertTrue(home["attached_object_collision_enabled"])

    def test_gripper_primitives_report_success_and_failure(self) -> None:
        hardware = FakeHardware(clearance_m=1.0, arms=ARMS)
        environment, registry = build(hardware)
        self.addCleanup(environment.safe_shutdown)
        functions = registry.functions()

        opened = functions["open_gripper"](arm="right")
        self.assertTrue(opened["success"])
        closed = functions["close_gripper"](arm="left")
        self.assertTrue(closed["success"])
        self.assertEqual(
            [command["arm"] for command in hardware.gripper_commands],
            ["right", "left"],
        )

        hardware.set_gripper_result = {"success": False, "reason": "gripper_stall"}
        with self.assertRaises(PrimitiveFailed) as caught:
            functions["close_gripper"](arm="left")
        self.assertIn("gripper_stall", caught.exception.reason)

    def test_get_object_pose_returns_a_finite_pose(self) -> None:
        environment, registry = build()
        self.addCleanup(environment.safe_shutdown)

        position, quaternion, extent = registry.functions()["get_object_pose"](
            "pen", arm="left"
        )

        self.assertEqual(position.shape, (3,))
        self.assertEqual(quaternion.shape, (4,))
        self.assertTrue(np.all(np.isfinite(position)))
        self.assertTrue(np.all(np.isfinite(quaternion)))
        self.assertIsNone(extent)

    def test_get_object_pose_retries_fresh_frames_for_metric_depth(self) -> None:
        attempts = 0

        def flaky_segment(rgb, text_prompt):
            nonlocal attempts
            del rgb, text_prompt
            attempts += 1
            return [] if attempts < 3 else [{"score": 0.9, "mask": _MASK}]

        environment, registry = build(
            primitive_settings={"get_object_pose": {"depth_retry_count": 2}},
            segment_client_factory=lambda: flaky_segment,
        )
        self.addCleanup(environment.safe_shutdown)

        position, _, _ = registry.functions()["get_object_pose"](
            "pen", arm="left"
        )

        self.assertTrue(np.all(np.isfinite(position)))
        self.assertEqual(attempts, 3)

    def test_sample_grasp_pose_returns_a_finite_pose(self) -> None:
        environment, registry = build()
        self.addCleanup(environment.safe_shutdown)

        position, quaternion = registry.functions()["sample_grasp_pose"](
            "pen", arm="left"
        )

        self.assertEqual(position.shape, (3,))
        self.assertEqual(quaternion.shape, (4,))
        self.assertTrue(np.all(np.isfinite(position)))
        self.assertTrue(np.all(np.isfinite(quaternion)))

    def test_sample_grasp_pose_retries_fresh_frames_for_metric_depth(self) -> None:
        attempts = 0

        def flaky_segment(rgb, text_prompt):
            nonlocal attempts
            del rgb, text_prompt
            attempts += 1
            return [] if attempts < 3 else [{"score": 0.9, "mask": _MASK}]

        environment, registry = build(
            primitive_settings={"sample_grasp_pose": {"depth_retry_count": 2}},
            segment_client_factory=lambda: flaky_segment,
        )
        self.addCleanup(environment.safe_shutdown)

        position, _ = registry.functions()["sample_grasp_pose"](
            "pen", arm="left"
        )

        self.assertTrue(np.all(np.isfinite(position)))
        self.assertEqual(attempts, 3)

    def test_sample_grasp_pose_hard_rejects_non_target_collision(self) -> None:
        hardware = FakeHardware(clearance_m=1.0, arms=ARMS)
        manipulation = {
            **FAKE_MANIPULATION_CONFIG,
            "grasp_collision_check_enabled": True,
        }
        environment = make_environment(hardware, manipulation=manipulation)
        self.addCleanup(environment.safe_shutdown)
        registry = PrimitiveRegistry()
        register_manipulation_primitives(
            registry,
            environment,
            segment_client_factory=lambda: _fake_segment,
            grasp_backend_factory=lambda config: _fake_grasp_backend,
            motion_planner_factory=lambda config: UnsafeMotionPlanner(),
        )

        with self.assertRaisesRegex(RuntimeError, "all top grasp candidates collide"):
            registry.functions()["sample_grasp_pose"]("pen", arm="left")

        self.assertEqual(hardware.arm_plan_requests, [])


if __name__ == "__main__":
    unittest.main()
