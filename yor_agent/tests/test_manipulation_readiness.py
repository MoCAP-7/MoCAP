from __future__ import annotations

import unittest
from unittest import mock

import numpy as np

from fakes import FAKE_MANIPULATION_CONFIG, FakeHardware, make_environment

from yor_agent.robot import manipulation_readiness
from yor_agent.robot.manipulation_readiness import (
    ManipulationReadinessConfig,
    ManipulationReadinessController,
)
from yor_agent.robot.perception.fast_detector import FastDetection
from yor_agent.robot.visualization.manipulation_readiness import (
    ManipulationReadinessViser,
)


ARMS = {
    "left": {"joint_pos": [0.0] * 7},
    "right": {"joint_pos": [0.0] * 7},
    "estop_latched": False,
    "end_effector_frame": "tcp",
}


class _MotionPlanner:
    def __init__(self, *, success: bool = True, reason: str | None = None) -> None:
        self.success = success
        self.reason = reason or (
            "collision_free_path_planned" if success else "grasp_goal_ik_failed"
        )
        self.requests: list[dict] = []

    def check_grasps(self, tcp_poses, obstacle_points, **kwargs):
        goals = np.asarray(tcp_poses, dtype=np.float64)
        return {
            "safe": [True] * len(goals),
            "minimum_clearance_m": [0.10] * len(goals),
            "scene_point_count": int(len(obstacle_points)),
            **kwargs,
        }

    def plan_grasp(
        self,
        current_joints,
        tcp_poses,
        obstacle_points,
        **kwargs,
    ):
        goals = np.asarray(tcp_poses, dtype=np.float64)
        self.requests.append(
            {
                "current_joints": np.asarray(current_joints).copy(),
                "tcp_poses": goals.copy(),
                "obstacle_points": np.asarray(obstacle_points).copy(),
                **kwargs,
            }
        )
        if not self.success:
            return {"success": False, "reason": self.reason}
        return {
            "success": True,
            "reason": self.reason,
            "selected_goalset_index": 0,
            "selected_tcp_pose": goals[0].tolist(),
            "waypoints": [[0.0] * 7, [0.0] * 7],
            "planning_time_s": 0.01,
            "terminal_check": {"safe": [True] * len(goals)},
        }


class _ScriptedDetector:
    def __init__(
        self,
        boxes: list[tuple[float, float, float, float] | None],
    ) -> None:
        self.boxes = list(boxes)
        self.calls = 0

    def __call__(self, image: np.ndarray, *, text_prompt: str):
        del image
        index = min(self.calls, len(self.boxes) - 1)
        self.calls += 1
        box = self.boxes[index]
        return [] if box is None else [FastDetection(box, 0.9, text_prompt)]


class _GraspBackend:
    name = "graspgenx"

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, depth, intrinsics, segmentation, instance_id):
        self.calls += 1
        del intrinsics, segmentation, instance_id
        transforms = []
        anchors = []
        for index in range(16):
            transform = np.eye(4, dtype=np.float64)
            transform[:3, 3] = [0.0, 0.0, float(depth[24, 32])]
            transforms.append(transform)
            anchors.append([0.0, 0.0, float(depth[24, 32])])
        return (
            np.asarray(transforms),
            np.linspace(0.95, 0.70, len(transforms)),
            np.asarray(anchors),
        )


def _segment(image: np.ndarray, *, text_prompt: str):
    del text_prompt
    mask = np.zeros(image.shape[:2], dtype=bool)
    mask[17:31, 25:39] = True
    return [{"mask": mask, "score": 0.9}]


class _SequenceSegmenter:
    def __init__(self, masks: list[np.ndarray]) -> None:
        self.masks = [np.asarray(mask, dtype=bool) for mask in masks]
        self.calls = 0

    def __call__(self, image: np.ndarray, *, text_prompt: str):
        del image, text_prompt
        index = min(self.calls, len(self.masks) - 1)
        self.calls += 1
        return [{"mask": self.masks[index], "score": 0.9}]


def _controller(
    hardware: FakeHardware,
    detector: _ScriptedDetector,
    *,
    settings: dict | None = None,
    manipulation_settings: dict | None = None,
    segment_client=None,
    motion_planner: _MotionPlanner | None = None,
) -> tuple[ManipulationReadinessController, object]:
    # Retained as a positional fixture argument across these tests to make it
    # explicit that prepare no longer consumes the old YOLO detector.
    del detector
    environment = make_environment(
        hardware,
        manipulation={
            **FAKE_MANIPULATION_CONFIG,
            "grasp_ik_precheck_enabled": True,
            "grasp_collision_check_enabled": True,
            "graspgenx_origin_to_tcp_m": 0.0,
            "graspgenx_to_nero_tcp_rpy_rad": [0.0, 0.0, 0.0],
            **(manipulation_settings or {}),
        },
        min_front_clearance_m=0.20,
    )
    controller = ManipulationReadinessController(
        environment,
        config={
            # Keep unit tests small while production uses the complete 729-pose
            # grid. These values form 4 x 5 x 3 = 60 local base candidates.
            "candidate_forward_offsets_m": [0.0, 0.05, 0.10, 0.15],
            "candidate_lateral_offsets_m": [-0.10, -0.05, 0.0, 0.05, 0.10],
            "candidate_yaw_offsets_deg": [-5.0, 0.0, 5.0],
            "candidate_base_limit": 60,
            "grasp_candidate_limit": 2,
            # Individual tests opt into the production seven-pose robustness
            # bundle when that behavior is under test.
            "robustness_enabled": False,
            **(settings or {}),
        },
        segment_client_factory=lambda: segment_client or _segment,
        grasp_backend_factory=lambda config: _GraspBackend(),
        motion_planner_factory=lambda config: motion_planner or _MotionPlanner(),
        clock=hardware.clock,
    )
    return controller, environment


def _forward_poses_ready_scene(
    hardware: FakeHardware, *, settings: dict | None = None
) -> tuple[ManipulationReadinessController, object]:
    """A scene where several forward-moved base poses certify, none at rest.

    The fixture's TCP z drops with the forward offset, so the virtual batch
    converges every forward-moved pose and refuses the current one; the
    actual-pose certification after the motion converges everything. The
    grid is small enough for one Pi batch and the tier early exit is off, so
    every forward pose is strict-ready and ranked behind the selected one.
    """

    detector = _ScriptedDetector([(27.0, 15.0, 37.0, 33.0)])
    controller, environment = _controller(
        hardware,
        detector,
        settings={
            "candidate_forward_offsets_m": [0.0, 0.05, 0.10],
            "candidate_lateral_offsets_m": [-0.05, 0.0, 0.05],
            "candidate_yaw_offsets_deg": [0.0],
            "pi_ik_best_tier_early_exit": False,
            **(settings or {}),
        },
    )
    pi_call = 0

    def forward_poses_then_everything(arm, poses_xyz_rpy):
        nonlocal pi_call
        del arm
        pi_call += 1
        return {
            "plans": [
                {
                    "candidate_index": index,
                    "success": True,
                    "ik_converged": pi_call > 1 or float(pose[2]) < 0.88,
                    "ik_position_error_m": 0.001,
                    "ik_rotation_error_rad": 0.001,
                    "ik_joint_target": [0.0] * 7,
                    "joint_travel_l2_rad": 0.1,
                    "joint_travel_max_rad": 0.05,
                }
                for index, pose in enumerate(poses_xyz_rpy)
            ]
        }

    environment.plan_arm_poses = forward_poses_then_everything
    return controller, environment


def _refusal(reason: str = "obstacle_too_close") -> dict:
    """What move_planar_relative returns when the clearance gate refuses."""

    return {
        "success": False,
        "status": "failed",
        "reason": reason,
        "metrics": {"last_command": [0.0, 0.0, 0.0]},
    }


def _refused_near_tier_scene(
    hardware: FakeHardware,
    *,
    settings: dict | None = None,
    lateral_offsets_m: tuple[float, ...] = (0.0,),
    pi_call_s: tuple[float, ...] = (),
) -> tuple[ManipulationReadinessController, object, dict]:
    """A near tier whose ready base the motion refuses, with farther tiers left.

    Fixture TCP z: current 0.90, forward 0.05 -> 0.85, 0.10 -> 0.80. Every
    forward-moved pose converges and the current one does not, so the tier
    early exit settles the 0.05 m tier after its two-query batch and never
    queries the 0.10 m poses. Once ``state["moved"]`` is set, as a test's
    successful motion does, the actual-pose certification converges
    everything. The n-th Pi call advances the fake clock by ``pi_call_s[n]``.
    """

    detector = _ScriptedDetector([(27.0, 15.0, 37.0, 33.0)])
    controller, environment = _controller(
        hardware,
        detector,
        settings={
            "candidate_forward_offsets_m": [0.0, 0.05, 0.10],
            "candidate_lateral_offsets_m": list(lateral_offsets_m),
            "candidate_yaw_offsets_deg": [0.0],
            "ik_batch_size": 2,
            **(settings or {}),
        },
    )
    state: dict = {"moved": False, "requested_z": [], "calls": 0}

    def forward_poses_then_everything(arm, poses_xyz_rpy):
        del arm
        if state["calls"] < len(pi_call_s):
            hardware.now += pi_call_s[state["calls"]]
        state["calls"] += 1
        state["requested_z"].extend(
            round(float(pose[2]), 3) for pose in poses_xyz_rpy
        )
        return {
            "plans": [
                {
                    "candidate_index": index,
                    "success": True,
                    "ik_converged": state["moved"] or float(pose[2]) < 0.88,
                    "ik_position_error_m": 0.001,
                    "ik_rotation_error_rad": 0.001,
                    "ik_joint_target": [0.0] * 7,
                    "joint_travel_l2_rad": 0.1,
                    "joint_travel_max_rad": 0.05,
                }
                for index, pose in enumerate(poses_xyz_rpy)
            ]
        }

    environment.plan_arm_poses = forward_poses_then_everything
    return controller, environment, state


class ManipulationReadinessConfigTest(unittest.TestCase):
    def test_defaults_cover_only_sam_grasp_and_local_search(self) -> None:
        config = ManipulationReadinessConfig()

        self.assertEqual(config.sam3_depth_retry_count, 2)
        self.assertEqual(config.maximum_initial_target_distance_m, 0.95)
        self.assertEqual(
            config.candidate_forward_offsets_m,
            (-0.10, -0.05, 0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30),
        )
        self.assertEqual(
            config.candidate_lateral_offsets_m,
            (-0.20, -0.15, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15, 0.20),
        )
        self.assertEqual(config.candidate_base_limit, 729)
        self.assertEqual(config.grasp_candidate_limit, 64)
        self.assertEqual(config.pi_ik_shortlist_base_limit, 128)
        self.assertEqual(config.pi_ik_grasps_per_base, 4)
        self.assertEqual(config.pi_ik_candidate_limit, 1024)
        self.assertEqual(config.pi_ik_nominal_candidate_limit, 512)
        self.assertEqual(config.ik_batch_size, 32)
        self.assertEqual(config.pi_ik_compute_budget_s, 15.0)
        self.assertTrue(config.robustness_enabled)
        self.assertEqual(config.candidate_translation_bucket_m, 0.020)
        self.assertAlmostEqual(config.candidate_yaw_bucket_rad, np.deg2rad(2.0))
        self.assertEqual(
            config.candidate_yaw_offsets_rad,
            tuple(np.deg2rad(np.arange(-20.0, 20.1, 5.0))),
        )
        self.assertEqual(config.max_linear_mps, 0.10)
        self.assertEqual(config.max_lateral_mps, 0.07)
        self.assertEqual(config.max_yaw_rad_s, 0.25)
        self.assertEqual(config.base_to_camera_forward_m, 0.2143)
        self.assertEqual(config.base_to_camera_left_m, 0.0603)

    def test_reverse_candidates_are_bounded_to_ten_centimeters(self) -> None:
        config = ManipulationReadinessConfig.from_mapping(
            {"candidate_forward_offsets_m": [-0.10, 0.0]}
        )
        self.assertEqual(config.candidate_forward_offsets_m, (-0.10, 0.0))

        with self.assertRaisesRegex(ValueError, "reverse"):
            ManipulationReadinessConfig.from_mapping(
                {"candidate_forward_offsets_m": [-0.11, 0.0]}
            )

    def test_degree_aliases_are_converted(self) -> None:
        config = ManipulationReadinessConfig.from_mapping(
            {"candidate_yaw_offsets_deg": [-5.0, 0.0, 5.0]}
        )
        self.assertAlmostEqual(config.candidate_yaw_offsets_rad[-1], np.deg2rad(5))

    def test_camera_lever_arm_is_bounded(self) -> None:
        with self.assertRaisesRegex(ValueError, "base-to-camera"):
            ManipulationReadinessConfig.from_mapping(
                {"base_to_camera_forward_m": 0.51}
            )

    def test_pi_search_dimensions_are_bounded(self) -> None:
        with self.assertRaisesRegex(ValueError, "candidate_base_limit"):
            ManipulationReadinessConfig.from_mapping(
                {"candidate_base_limit": 1025}
            )
        with self.assertRaisesRegex(ValueError, "grasp_candidate_limit"):
            ManipulationReadinessConfig.from_mapping(
                {"grasp_candidate_limit": 65}
            )
        with self.assertRaisesRegex(ValueError, "pi_ik_candidate_limit"):
            ManipulationReadinessConfig.from_mapping(
                {"pi_ik_candidate_limit": 4097}
            )
        with self.assertRaisesRegex(ValueError, "positive"):
            ManipulationReadinessConfig.from_mapping(
                {"pi_ik_compute_budget_s": 0.0}
            )
        with self.assertRaisesRegex(ValueError, "at most 30"):
            ManipulationReadinessConfig.from_mapping(
                {"pi_ik_compute_budget_s": 30.1}
            )
        with self.assertRaisesRegex(
            ValueError, "pi_ik_nominal_candidate_limit cannot exceed"
        ):
            ManipulationReadinessConfig.from_mapping(
                {
                    "pi_ik_candidate_limit": 6,
                    "pi_ik_nominal_candidate_limit": 7,
                }
            )

    def test_motion_alternatives_default_off_and_are_a_bounded_integer(
        self,
    ) -> None:
        self.assertEqual(ManipulationReadinessConfig().motion_alternative_limit, 0)
        config = ManipulationReadinessConfig.from_mapping(
            {"motion_alternative_limit": 24}
        )
        self.assertEqual(config.motion_alternative_limit, 24)
        for rejected in (-1, 1.5, True, 25):
            with self.assertRaisesRegex(ValueError, "motion_alternative_limit"):
                ManipulationReadinessConfig.from_mapping(
                    {"motion_alternative_limit": rejected}
                )

    def test_continuing_the_search_after_refusals_defaults_off_and_is_boolean(
        self,
    ) -> None:
        self.assertIs(
            ManipulationReadinessConfig().continue_search_after_refused_motions,
            False,
        )
        config = ManipulationReadinessConfig.from_mapping(
            {"continue_search_after_refused_motions": True}
        )
        self.assertIs(config.continue_search_after_refused_motions, True)
        for rejected in (1, 0, "true", None):
            with self.subTest(rejected=rejected):
                with self.assertRaisesRegex(
                    TypeError, "continue_search_after_refused_motions"
                ):
                    ManipulationReadinessConfig.from_mapping(
                        {"continue_search_after_refused_motions": rejected}
                    )


class ManipulationReadinessControllerTest(unittest.TestCase):
    def test_sparse_sam_depth_retries_and_uses_first_acceptable_observation(
        self,
    ) -> None:
        hardware = FakeHardware(clearance_m=0.9, arms=ARMS)
        detector = _ScriptedDetector([(27.0, 15.0, 37.0, 33.0)])
        sparse = np.zeros((48, 64), dtype=bool)
        sparse[21:27, 29:35] = True  # 36 valid points, below the configured 40.
        dense = np.zeros((48, 64), dtype=bool)
        dense[17:31, 25:39] = True
        segmenter = _SequenceSegmenter([sparse, dense])
        controller, environment = _controller(
            hardware,
            detector,
            settings={"sam3_depth_retry_count": 2},
            manipulation_settings={"minimum_metric_depth_points": 40},
            segment_client=segmenter,
        )
        self.addCleanup(environment.safe_shutdown)

        result = controller.prepare_for_manipulation("marker", "left")

        self.assertTrue(result["success"], result)
        # Two virtual-evaluation observations (sparse, then dense). The
        # current pose is selected, so no fresh certification observation
        # follows; the certificate comes from the virtual evaluation.
        self.assertEqual(segmenter.calls, 2)
        self.assertEqual(result["certification"]["source"], "virtual_evaluation")
        self.assertEqual(result["sam3_depth_attempt_count"], 2)
        self.assertEqual(
            [
                item["valid_depth_points"]
                for item in controller.last_debug["sam3_depth_attempts"]
            ],
            [36, 196],
        )

    def test_sparse_sam_depth_failure_reports_all_attempts_to_debug(self) -> None:
        hardware = FakeHardware(clearance_m=0.9, arms=ARMS)
        detector = _ScriptedDetector([(27.0, 15.0, 37.0, 33.0)])
        sparse = np.zeros((48, 64), dtype=bool)
        sparse[21:27, 29:35] = True
        segmenter = _SequenceSegmenter([sparse])
        controller, environment = _controller(
            hardware,
            detector,
            settings={"sam3_depth_retry_count": 2},
            manipulation_settings={"minimum_metric_depth_points": 40},
            segment_client=segmenter,
        )
        self.addCleanup(environment.safe_shutdown)

        result = controller.prepare_for_manipulation("marker", "left")

        self.assertFalse(result["success"])
        self.assertEqual(segmenter.calls, 3)
        self.assertIn("best 36/40", result["reason"])
        self.assertIn("attempt counts=[36, 36, 36]", result["reason"])
        self.assertEqual(controller.last_debug["phase"], "failed")
        self.assertEqual(
            controller.last_debug["failed_after_phase"],
            "sam3_depth_validation",
        )
        self.assertEqual(controller.last_debug["valid_depth_points"], 36)

    def test_pi_ik_failure_rejects_candidates_without_calling_curobo(self) -> None:
        hardware = FakeHardware(clearance_m=0.9, arms=ARMS)
        detector = _ScriptedDetector([(27.0, 15.0, 37.0, 33.0)])
        controller, environment = _controller(
            hardware,
            detector,
            settings={"candidate_yaw_offsets_deg": [0.0]},
        )
        self.addCleanup(environment.safe_shutdown)

        def reject_virtual_batch(arm, poses_xyz_rpy):
            del arm
            return {
                "plans": [
                    {"candidate_index": index, "success": False}
                    for index in range(len(poses_xyz_rpy))
                ]
            }

        environment.plan_arm_poses = reject_virtual_batch

        result = controller.prepare_for_manipulation("marker", "left")

        self.assertFalse(result["success"])
        self.assertIn("strictly Pi-converged", result["reason"])
        diagnostics = result["diagnostics"]
        self.assertEqual(
            diagnostics["stage"],
            "parallel_pi_virtual_certification",
        )
        # A failed search still reports its local split and per-query log,
        # since the unreachable case is exactly what a budget replay compares.
        self.assertIn("post_ik_ranking", diagnostics["substage_timings_s"])
        self.assertEqual(
            len(diagnostics["pi_ik_query_log"]), diagnostics["ik_query_count"]
        )
        self.assertGreater(len(diagnostics["pi_ik_query_log"]), 0)
        self.assertFalse(
            any(row["ik_converged"] for row in diagnostics["pi_ik_query_log"])
        )
        planner = controller.manipulation._motion_planner_client()
        self.assertEqual(planner.requests, [])

    def test_missing_visible_target_fails_without_search_motion(self) -> None:
        hardware = FakeHardware(clearance_m=0.9, arms=ARMS)
        detector = _ScriptedDetector([None])
        controller, environment = _controller(
            hardware,
            detector,
            segment_client=lambda image, *, text_prompt: [],
        )
        self.addCleanup(environment.safe_shutdown)

        result = controller.prepare_for_manipulation("pen", "left")

        self.assertFalse(result["success"])
        self.assertIn("SAM3", result["reason"])
        self.assertTrue(result["final_stop"]["success"])
        self.assertEqual(hardware.velocity.tolist(), [0.0, 0.0, 0.0])
        self.assertFalse(
            any(np.linalg.norm(command) > 1e-8 for command in hardware.commands)
        )

    def test_current_pose_can_be_selected_without_nonzero_planar_motion(self) -> None:
        hardware = FakeHardware(clearance_m=0.9, arms=ARMS)
        detector = _ScriptedDetector([(27.0, 15.0, 37.0, 33.0)])
        controller, environment = _controller(
            hardware, detector, settings={"candidate_yaw_offsets_deg": [0.0]}
        )
        self.addCleanup(environment.safe_shutdown)

        result = controller.prepare_for_manipulation("pen", "left")

        self.assertTrue(result["success"], result)
        self.assertEqual(result["reason"], "grasp_execution_ready")
        self.assertFalse(result["requires_fresh_grasp_sampling"])
        self.assertTrue(result["certified_goalset_available"])
        self.assertNotIn("grasp_position", result)
        self.assertNotIn("grasp_quaternion_wxyz", result)
        largest_batch = max(
            len(item["poses_xyz_rpy"]) for item in hardware.arm_plan_requests
        )
        self.assertLessEqual(largest_batch, 64)
        self.assertIsNotNone(controller.last_debug)
        self.assertEqual(controller.last_debug["phase"], "grasp_execution_ready")

    def test_successful_preparation_reports_a_stage_timing_split(self) -> None:
        hardware = FakeHardware(clearance_m=0.9, arms=ARMS)
        detector = _ScriptedDetector([(27.0, 15.0, 37.0, 33.0)])
        controller, environment = _controller(
            hardware, detector, settings={"candidate_yaw_offsets_deg": [0.0]}
        )
        self.addCleanup(environment.safe_shutdown)

        result = controller.prepare_for_manipulation("pen", "left")

        self.assertTrue(result["success"], result)
        timings = result["stage_timings_s"]
        self.assertEqual(
            set(timings),
            {
                "grasp_candidate_generation",
                "navigation_frame_fetch",
                "virtual_candidate_evaluation",
                "motion",
                "actual_pose_certification",
            },
        )
        for stage, value in timings.items():
            self.assertGreaterEqual(value, 0.0, stage)
        # The stages are disjoint sub-intervals of the primitive's own span.
        self.assertLessEqual(sum(timings.values()), result["elapsed_s"] + 1e-9)
        # This scene keeps the current base pose, so the actual-pose
        # certification re-observes from an already strictly certified pose.
        self.assertFalse(result["motion_executed"])
        self.assertEqual(timings["motion"], 0.0)
        self.assertEqual(controller.last_debug["stage_timings_s"], timings)

    def test_evaluation_reports_the_local_substage_split_around_pi_ik(self) -> None:
        """The evaluation bucket hides local expansion work around the Pi waves."""

        hardware = FakeHardware(clearance_m=0.9, arms=ARMS)
        detector = _ScriptedDetector([(27.0, 15.0, 37.0, 33.0)])
        controller, environment = _controller(
            hardware, detector, settings={"candidate_yaw_offsets_deg": [0.0]}
        )
        self.addCleanup(environment.safe_shutdown)

        result = controller.prepare_for_manipulation("pen", "left")

        self.assertTrue(result["success"], result)
        substages = result["evaluation_substage_timings_s"]
        self.assertEqual(
            set(substages),
            {
                "ground_geometry",
                "virtual_candidates",
                "collision_scene_points",
                "collision_check_rpc",
                "record_expansion",
                "shortlist_ranking",
                "post_ik_ranking",
            },
        )
        for name, value in substages.items():
            self.assertGreaterEqual(value, 0.0, name)
        # Every substage lies outside the Pi IK window (before pi_started or
        # after the last wave), so together with it they must fit inside the
        # evaluation stage.
        evaluation_stage = result["stage_timings_s"]["virtual_candidate_evaluation"]
        self.assertLessEqual(
            sum(substages.values()) + result["pi_ik_compute_elapsed_s"],
            evaluation_stage + 1e-9,
        )

    def test_pose_conversion_runs_for_the_shortlist_not_the_whole_lattice(self) -> None:
        """matrix_to_quaternion_wxyz costs an SVD, so it must stay off the lattice."""

        hardware = FakeHardware(clearance_m=0.9, arms=ARMS)
        detector = _ScriptedDetector([(27.0, 15.0, 37.0, 33.0)])
        controller, environment = _controller(
            hardware,
            detector,
            settings={
                "candidate_yaw_offsets_deg": [0.0],
                # Force a shortlist that is strictly smaller than the lattice.
                "pi_ik_shortlist_base_limit": 4,
                "pi_ik_grasps_per_base": 1,
            },
        )
        self.addCleanup(environment.safe_shutdown)

        calls = 0
        original = manipulation_readiness.matrix_to_quaternion_wxyz

        def counted(rotation):
            nonlocal calls
            calls += 1
            return original(rotation)

        with mock.patch.object(
            manipulation_readiness, "matrix_to_quaternion_wxyz", counted
        ):
            result = controller.prepare_for_manipulation("pen", "left")

        self.assertTrue(result["success"], result)
        self.assertGreater(result["pi_eligible_pair_count"], 0)
        # One conversion per shortlisted pair (plus any robustness variant),
        # never one per eligible base/grasp pair.
        self.assertLessEqual(calls, result["pi_ik_requested_count"])
        self.assertLess(calls, result["pi_eligible_pair_count"])

    def test_failed_preparation_retains_completed_stage_timings(self) -> None:
        hardware = FakeHardware(clearance_m=0.9, arms=ARMS)
        detector = _ScriptedDetector([(27.0, 15.0, 37.0, 33.0)])
        controller, environment = _controller(
            hardware, detector, settings={"candidate_yaw_offsets_deg": [0.0]}
        )
        self.addCleanup(environment.safe_shutdown)

        def reject_virtual_batch(arm, poses_xyz_rpy):
            del arm
            return {
                "plans": [
                    {"candidate_index": index, "success": False}
                    for index in range(len(poses_xyz_rpy))
                ]
            }

        environment.plan_arm_poses = reject_virtual_batch

        result = controller.prepare_for_manipulation("marker", "left")

        self.assertFalse(result["success"])
        timings = result["stage_timings_s"]
        # Candidate generation completed before the evaluation raised, so its
        # cost stays measurable even though the later stages never ran.
        self.assertIn("grasp_candidate_generation", timings)
        self.assertIn("navigation_frame_fetch", timings)
        self.assertNotIn("virtual_candidate_evaluation", timings)
        self.assertNotIn("actual_pose_certification", timings)

    def test_finite_nonconverged_pi_candidate_is_not_ready(self) -> None:
        hardware = FakeHardware(clearance_m=0.9, arms=ARMS)
        detector = _ScriptedDetector([(27.0, 15.0, 37.0, 33.0)])
        controller, environment = _controller(
            hardware, detector, settings={"candidate_yaw_offsets_deg": [0.0]}
        )
        self.addCleanup(environment.safe_shutdown)

        def finite_but_outside_quality_band(arm, poses_xyz_rpy):
            del arm
            plans = []
            for index, pose in enumerate(poses_xyz_rpy):
                better = float(pose[0]) > 0.04
                plans.append(
                    {
                        "candidate_index": index,
                        "success": True,
                        "ik_converged": False,
                        # The "better" result is finite but remains just
                        # outside the shared cuRobo/Pi 0.020 m / 0.10 rad gate.
                        "ik_position_error_m": 0.021 if better else 0.030,
                        "ik_rotation_error_rad": 0.110 if better else 0.250,
                        "ik_joint_target": [0.0] * 7,
                        "joint_travel_l2_rad": 0.1,
                    }
                )
            return {"plans": plans}

        environment.plan_arm_poses = finite_but_outside_quality_band
        result = controller.prepare_for_manipulation("marker", "left")

        self.assertFalse(result["success"])
        self.assertIn("strictly Pi-converged", result["reason"])
        planner = controller.manipulation._motion_planner_client()
        self.assertEqual(planner.requests, [])

    def test_candidate_ranking_prioritizes_the_nearest_motion_tier(self) -> None:
        hardware = FakeHardware(clearance_m=0.9, arms=ARMS)
        detector = _ScriptedDetector([(27.0, 15.0, 37.0, 33.0)])
        controller, environment = _controller(hardware, detector)
        self.addCleanup(environment.safe_shutdown)
        generated = controller.manipulation.generate_grasp_candidates(
            "marker", "left"
        )
        frame = environment.navigation_frame(max_age_s=1.0)

        evaluation = controller._evaluate_candidates(generated, frame)

        ready = [
            candidate
            for candidate in evaluation["candidates"]
            if candidate["strict_ready"]
        ]
        self.assertGreater(len(ready), 1)
        selected = evaluation["selected"]
        self.assertEqual(
            selected["movement_cost_tier"],
            min(candidate["movement_cost_tier"] for candidate in ready),
        )
        current = next(
            candidate
            for candidate in evaluation["candidates"]
            if candidate["forward_m"] == 0.0
            and candidate["left_m"] == 0.0
            and candidate["yaw_rad"] == 0.0
        )
        self.assertEqual(current["proxy_score"], 0.0)
        self.assertEqual(current["movement_cost"], 0.0)
        self.assertEqual(current["movement_cost_tier"], 0)

    def test_nearer_tier_wins_even_with_worse_joint_travel(self) -> None:
        hardware = FakeHardware(clearance_m=0.9, arms=ARMS)
        detector = _ScriptedDetector([(27.0, 15.0, 37.0, 33.0)])
        controller, environment = _controller(
            hardware,
            detector,
            settings={
                "candidate_forward_offsets_m": [0.0, 0.05],
                "candidate_lateral_offsets_m": [0.0],
                "candidate_yaw_offsets_deg": [0.0],
                "candidate_base_limit": 2,
                "pi_ik_shortlist_base_limit": 2,
                "pi_ik_grasps_per_base": 2,
                "pi_ik_candidate_limit": 4,
                "pi_ik_nominal_candidate_limit": 4,
                "ik_batch_size": 4,
            },
        )
        self.addCleanup(environment.safe_shutdown)

        def favor_far_joint_travel(arm, poses_xyz_rpy):
            del arm
            return {
                "plans": [
                    {
                        "candidate_index": index,
                        "success": True,
                        "ik_converged": True,
                        "ik_position_error_m": 0.001,
                        "ik_rotation_error_rad": 0.001,
                        # Forward base motion shortens this fixture's TCP Z.
                        "joint_travel_l2_rad": (
                            10.0 if float(pose[2]) > 0.88 else 0.0
                        ),
                        "joint_travel_max_rad": (
                            1.0 if float(pose[2]) > 0.88 else 0.0
                        ),
                    }
                    for index, pose in enumerate(poses_xyz_rpy)
                ]
            }

        environment.plan_arm_poses = favor_far_joint_travel
        generated = controller.manipulation.generate_grasp_candidates(
            "marker", "left"
        )
        frame = environment.navigation_frame(max_age_s=1.0)

        selected = controller._evaluate_candidates(generated, frame)["selected"]

        self.assertEqual(selected["forward_m"], 0.0)
        self.assertEqual(selected["movement_cost_tier"], 0)

    def test_grasp_count_breaks_ties_inside_one_motion_tier(self) -> None:
        hardware = FakeHardware(clearance_m=0.9, arms=ARMS)
        detector = _ScriptedDetector([(27.0, 15.0, 37.0, 33.0)])
        controller, environment = _controller(
            hardware,
            detector,
            settings={
                "candidate_forward_offsets_m": [0.0, 0.01],
                "candidate_lateral_offsets_m": [0.0],
                "candidate_yaw_offsets_deg": [0.0],
                "candidate_base_limit": 2,
                "pi_ik_shortlist_base_limit": 2,
                "pi_ik_grasps_per_base": 2,
                "pi_ik_candidate_limit": 4,
                "pi_ik_nominal_candidate_limit": 4,
                "ik_batch_size": 4,
            },
        )
        self.addCleanup(environment.safe_shutdown)

        def give_moved_pose_more_grasps(arm, poses_xyz_rpy):
            del arm
            return {
                "plans": [
                    {
                        "candidate_index": index,
                        # Both moved-pose grasps converge; only the first
                        # current-pose grasp converges.
                        "success": float(pose[2]) < 0.895 or index == 1,
                        "ik_converged": float(pose[2]) < 0.895 or index == 1,
                        "ik_position_error_m": 0.001,
                        "ik_rotation_error_rad": 0.001,
                        "joint_travel_l2_rad": 0.1,
                        "joint_travel_max_rad": 0.05,
                    }
                    for index, pose in enumerate(poses_xyz_rpy)
                ]
            }

        environment.plan_arm_poses = give_moved_pose_more_grasps
        generated = controller.manipulation.generate_grasp_candidates(
            "marker", "left"
        )
        frame = environment.navigation_frame(max_age_s=1.0)

        selected = controller._evaluate_candidates(generated, frame)["selected"]

        self.assertEqual(selected["movement_cost_tier"], 0)
        self.assertEqual(selected["forward_m"], 0.01)
        self.assertEqual(selected["converged_grasp_count"], 2)

    def test_pi_shortlist_bounds_exact_work_and_preserves_base_diversity(self) -> None:
        hardware = FakeHardware(clearance_m=0.9, arms=ARMS)
        detector = _ScriptedDetector([(27.0, 15.0, 37.0, 33.0)])
        controller, environment = _controller(
            hardware,
            detector,
            settings={
                "pi_ik_shortlist_base_limit": 5,
                "pi_ik_grasps_per_base": 2,
                "pi_ik_candidate_limit": 10,
                "ik_batch_size": 10,
            },
        )
        self.addCleanup(environment.safe_shutdown)
        generated = controller.manipulation.generate_grasp_candidates(
            "marker", "left"
        )
        frame = environment.navigation_frame(max_age_s=1.0)

        evaluation = controller._evaluate_candidates(generated, frame)

        self.assertEqual(evaluation["ik_query_count"], 10)
        self.assertEqual(evaluation["pi_shortlisted_base_count"], 5)
        self.assertGreater(evaluation["pi_eligible_pair_count"], 10)
        queried_bases = {
            candidate["candidate_index"]
            for candidate in evaluation["candidates"]
            if candidate["plans"]
        }
        self.assertEqual(len(queried_bases), 5)
        self.assertEqual(evaluation["pi_batches"][0]["candidate_count"], 10)
        self.assertEqual(evaluation["nominal_grasp_evaluated_count"], 10)
        self.assertEqual(evaluation["robustness_query_count"], 0)
        self.assertTrue(
            all(
                candidate["nominal_grasp_evaluated_count"] == 2
                for candidate in evaluation["candidates"]
                if candidate["plans"]
            )
        )

    def test_evaluation_logs_every_pi_query_in_schedule_order(self) -> None:
        """A budget curve needs each query's outcome, not just per-batch counts."""

        hardware = FakeHardware(clearance_m=0.9, arms=ARMS)
        detector = _ScriptedDetector([(27.0, 15.0, 37.0, 33.0)])
        controller, environment = _controller(
            hardware,
            detector,
            settings={
                "pi_ik_shortlist_base_limit": 5,
                "pi_ik_grasps_per_base": 2,
                "pi_ik_candidate_limit": 10,
                "ik_batch_size": 4,
            },
        )
        self.addCleanup(environment.safe_shutdown)
        generated = controller.manipulation.generate_grasp_candidates(
            "marker", "left"
        )
        frame = environment.navigation_frame(max_age_s=1.0)

        evaluation = controller._evaluate_candidates(generated, frame)

        log = evaluation["pi_ik_query_log"]
        self.assertEqual(len(log), evaluation["ik_query_count"])
        self.assertEqual([row["order"] for row in log], list(range(len(log))))
        self.assertEqual(
            [row["batch_index"] for row in log],
            [
                index
                for index, batch in enumerate(evaluation["pi_batches"])
                for _ in range(batch["candidate_count"])
            ],
        )
        self.assertEqual(
            {row["stage"] for row in log},
            {batch["stage"] for batch in evaluation["pi_batches"]},
        )
        # Each row points at exactly one recorded plan and repeats its verdict.
        by_base = {
            candidate["candidate_index"]: candidate
            for candidate in evaluation["candidates"]
        }
        for row in log:
            base = by_base[row["base_index"]]
            plans = [
                plan
                for plan in base["plans"]
                if plan["grasp_index"] == row["grasp_index"]
                and plan["robustness_variant"] == row["robustness_variant"]
            ]
            self.assertEqual(len(plans), 1, row)
            self.assertEqual(plans[0]["ik_converged"], row["ik_converged"])
            self.assertEqual(row["movement_cost_tier"], base["movement_cost_tier"])
            self.assertEqual(row["forward_m"], base["forward_m"])
        self.assertEqual(
            evaluation["minimum_feasible_grasps"],
            controller.config.minimum_feasible_grasps,
        )

    def test_pi_search_finishes_nominal_grasp_diversity_before_robustness(
        self,
    ) -> None:
        hardware = FakeHardware(clearance_m=0.9, arms=ARMS)
        detector = _ScriptedDetector([(27.0, 15.0, 37.0, 33.0)])
        controller, environment = _controller(
            hardware,
            detector,
            settings={
                "candidate_forward_offsets_m": [0.0, 0.05],
                "candidate_lateral_offsets_m": [0.0],
                "candidate_yaw_offsets_deg": [0.0],
                "candidate_base_limit": 2,
                "grasp_candidate_limit": 2,
                "pi_ik_shortlist_base_limit": 2,
                "pi_ik_grasps_per_base": 2,
                "pi_ik_candidate_limit": 10,
                "pi_ik_nominal_candidate_limit": 4,
                "ik_batch_size": 10,
                "robustness_enabled": True,
            },
        )
        self.addCleanup(environment.safe_shutdown)
        generated = controller.manipulation.generate_grasp_candidates(
            "marker", "left"
        )
        frame = environment.navigation_frame(max_age_s=1.0)

        evaluation = controller._evaluate_candidates(generated, frame)

        self.assertEqual(evaluation["nominal_grasp_evaluated_count"], 4)
        self.assertEqual(evaluation["robustness_query_count"], 6)
        self.assertEqual(
            [batch["stage"] for batch in evaluation["pi_batches"]],
            ["nominal", "robustness"],
        )
        evaluated = [
            candidate["nominal_grasp_evaluated_count"]
            for candidate in evaluation["candidates"]
        ]
        self.assertEqual(evaluated, [2, 2])

    def test_pi_waves_keep_completed_results_and_stop_before_budget_overshoot(
        self,
    ) -> None:
        hardware = FakeHardware(clearance_m=0.9, arms=ARMS)
        detector = _ScriptedDetector([(27.0, 15.0, 37.0, 33.0)])
        controller, environment = _controller(
            hardware,
            detector,
            settings={
                "candidate_yaw_offsets_deg": [0.0],
                "pi_ik_shortlist_base_limit": 5,
                "pi_ik_grasps_per_base": 2,
                "pi_ik_candidate_limit": 10,
                "ik_batch_size": 4,
                "pi_ik_compute_budget_s": 2.5,
                # This test exercises the budget guard, not the tier early exit.
                "pi_ik_best_tier_early_exit": False,
            },
        )
        self.addCleanup(environment.safe_shutdown)

        def one_second_strict_batch(arm, poses_xyz_rpy):
            del arm
            hardware.now += 1.0
            return {
                "plans": [
                    {
                        "candidate_index": index,
                        "success": True,
                        "ik_converged": True,
                        "ik_position_error_m": 0.0005,
                        "ik_rotation_error_rad": 0.0005,
                        "joint_travel_l2_rad": 0.1,
                        "joint_travel_max_rad": 0.05,
                    }
                    for index in range(len(poses_xyz_rpy))
                ]
            }

        environment.plan_arm_poses = one_second_strict_batch
        generated = controller.manipulation.generate_grasp_candidates(
            "marker", "left"
        )
        frame = environment.navigation_frame(max_age_s=1.0)

        evaluation = controller._evaluate_candidates(generated, frame)

        self.assertEqual(evaluation["pi_ik_requested_count"], 10)
        self.assertEqual(evaluation["ik_query_count"], 8)
        self.assertEqual(len(evaluation["pi_batches"]), 2)
        self.assertTrue(evaluation["pi_ik_budget_exhausted"])
        self.assertEqual(evaluation["pi_ik_compute_elapsed_s"], 2.0)
        self.assertEqual(len(evaluation["pi_ik_query_log"]), 8)
        self.assertGreater(evaluation["strict_pi_candidate_count"], 0)

    def test_the_search_ranks_the_runner_up_base_poses(self) -> None:
        hardware = FakeHardware(clearance_m=0.9, arms=ARMS)
        detector = _ScriptedDetector([(27.0, 15.0, 37.0, 33.0)])
        controller, environment = _controller(hardware, detector)
        generated = controller.manipulation.generate_grasp_candidates(
            "marker", "left"
        )
        frame = environment.navigation_frame(max_age_s=1.0)

        evaluation = controller._evaluate_candidates(generated, frame)

        alternatives = evaluation["ranked_alternatives"]
        self.assertTrue(alternatives)
        self.assertLessEqual(
            len(alternatives), evaluation["strict_pi_candidate_count"]
        )
        self.assertLessEqual(
            len(alternatives), manipulation_readiness.RANKED_ALTERNATIVES_LOGGED
        )
        self.assertEqual(
            [item["rank"] for item in alternatives], list(range(len(alternatives)))
        )
        # Distinct poses, not the same one repeated: several of this fixture's
        # candidates sit at forward 0 and left 0, so identity has to be checked
        # on the candidate index rather than on the offsets alone.
        indices = [item["candidate_index"] for item in alternatives]
        self.assertEqual(len(set(indices)), len(indices))
        # Rank 0 is the pose the primitive actually drives to.
        self.assertEqual(
            alternatives[0]["candidate_index"],
            int(evaluation["selected"]["candidate_index"]),
        )
        self.assertAlmostEqual(
            alternatives[0]["forward_m"], evaluation["selected"]["forward_m"]
        )
        self.assertAlmostEqual(
            alternatives[0]["left_m"], evaluation["selected"]["left_m"]
        )
        self.assertAlmostEqual(
            float(np.radians(alternatives[0]["yaw_deg"])),
            float(evaluation["selected"]["yaw_rad"]),
            places=9,
        )
        # Ranking is by movement cost tier first, so the list never steps back
        # to a cheaper tier further down.
        tiers = [item["movement_cost_tier"] for item in alternatives]
        self.assertEqual(tiers, sorted(tiers))

    def test_a_prepare_that_raises_still_carries_the_runner_up_poses(self) -> None:
        """The case the list exists for, and the one that nearly lost it.

        Only ``ranked[0]`` is executed. When the clearance gate refuses the
        motion to it, prepare raises and every search aggregate assembled on
        the success path below is skipped, so a list recorded there would
        never reach trace.json. The refusal and the stop failure used here
        land in the same handler; this pins that the list survives it.
        """

        hardware = FakeHardware(clearance_m=0.9, arms=ARMS)
        detector = _ScriptedDetector([(27.0, 15.0, 37.0, 33.0)])
        controller, environment = _controller(hardware, detector)
        # The post-motion stop, which runs whether or not the selected pose
        # needed driving to, so this does not depend on which pose the search
        # happens to rank first. The first stop guards the primitive's entry
        # and has to keep succeeding or nothing is searched at all.
        real_stop = environment.controller.stop
        calls = {"count": 0}

        def stop_after_the_search(*args, **kwargs):
            calls["count"] += 1
            if calls["count"] == 1:
                return real_stop(*args, **kwargs)
            return {"success": False, "reason": "stop_rpc_unavailable"}

        environment.controller.stop = stop_after_the_search

        result = controller.prepare_for_manipulation("marker", "left")

        self.assertFalse(result["success"])
        alternatives = result["ranked_alternatives"]
        self.assertTrue(alternatives)
        self.assertEqual(alternatives[0]["rank"], 0)
        # The pose the primitive drove to, or would have, is rank 0.
        self.assertAlmostEqual(
            alternatives[0]["forward_m"], result["selected_base_pose"]["forward_m"]
        )
        self.assertAlmostEqual(
            alternatives[0]["left_m"], result["selected_base_pose"]["left_m"]
        )
        self.assertIsNotNone(result["current_pose_evaluation"])
        self.assertTrue(result["tier_evaluation"])
        # The IK-budget curve is replayed from the per-query order, so a
        # refused motion must not cost the search it followed.
        log = result["pi_ik_query_log"]
        self.assertTrue(log)
        self.assertEqual([row["order"] for row in log], list(range(len(log))))
        self.assertTrue(any(row["ik_converged"] for row in log))
        self.assertTrue(result["pi_batches"])
        self.assertEqual(
            sum(int(batch["candidate_count"]) for batch in result["pi_batches"]),
            len(log),
        )
        self.assertGreaterEqual(result["minimum_feasible_grasps"], 1)

    def test_a_successful_prepare_also_carries_the_runner_up_poses(self) -> None:
        hardware = FakeHardware(clearance_m=0.9, arms=ARMS)
        detector = _ScriptedDetector([(27.0, 15.0, 37.0, 33.0)])
        controller, _ = _controller(hardware, detector)

        result = controller.prepare_for_manipulation("marker", "left")

        self.assertTrue(result["success"])
        self.assertTrue(result["ranked_alternatives"])

    def test_a_prepare_accounts_for_the_pose_it_stood_at(self) -> None:
        """The docked pose, stage by stage, beside the pose it chose instead."""

        hardware = FakeHardware(clearance_m=0.9, arms=ARMS)
        detector = _ScriptedDetector([(27.0, 15.0, 37.0, 33.0)])
        controller, _ = _controller(hardware, detector)

        result = controller.prepare_for_manipulation("marker", "left")

        current = result["current_pose_evaluation"]
        self.assertEqual(
            (current["forward_m"], current["left_m"], current["yaw_deg"]), (0.0, 0.0, 0.0)
        )
        self.assertEqual(current["movement_cost_tier"], 0)
        self.assertLessEqual(current["collision_safe_grasps"], current["grasp_pool"])
        self.assertEqual(
            current["eligible_grasps"] + current["reach_bound_rejected_grasps"],
            current["collision_safe_grasps"],
        )
        self.assertLessEqual(current["ik_converged_grasps"], current["ik_queried_grasps"])
        self.assertEqual(current["strict_ready"], current["rejected_at"] is None)
        self.assertIsNone(result["selected_pose_evaluation"]["rejected_at"])
        tiers = result["tier_evaluation"]
        self.assertEqual(tiers[0]["tier"], 0)
        self.assertEqual([row["tier"] for row in tiers], sorted(row["tier"] for row in tiers))

    def test_the_account_names_the_first_stage_that_rejected_a_pose(self) -> None:
        hardware = FakeHardware(clearance_m=0.9, arms=ARMS)
        detector = _ScriptedDetector([(27.0, 15.0, 37.0, 33.0)])
        controller, _ = _controller(hardware, detector)

        def base(**overrides):
            candidate = {
                "candidate_index": 0,
                "forward_m": 0.0,
                "left_m": 0.0,
                "yaw_rad": 0.0,
                "movement_cost_tier": 0,
                "eligible_grasps": 3,
                "pi_bound_rejected_grasps": 1,
                "shortlisted_grasps": 3,
                "nearest_tcp_radius_m": 0.5,
                "plans": [],
                "strict_ready": False,
            }
            candidate.update(overrides)
            return candidate

        def plan(converged, position_error=0.02):
            return {
                "robustness_variant": "nominal",
                "ik_converged": converged,
                "ik_finite": True,
                "grasp_index": 4,
                "tcp_radius_m": 0.5,
                "ik_plan": {
                    "ik_position_error_m": position_error,
                    "ik_rotation_error_rad": 0.1,
                },
            }

        context = {
            "grasp_pool_count": 10,
            "collision_safe_count": 4,
            "shortlisted_base_indices": {0},
        }
        cases = [
            (base(eligible_grasps=0, pi_bound_rejected_grasps=4, shortlisted_grasps=0), "reach_bound"),
            (base(candidate_index=7), "shortlist"),
            (base(), "not_queried"),
            (base(plans=[plan(False), plan(False, 0.05)]), "ik_not_converged"),
            (base(plans=[plan(True, 0.001)], strict_ready=True), None),
        ]
        for candidate, expected in cases:
            with self.subTest(expected=expected):
                account = controller._base_evaluation(candidate, **context)
                self.assertEqual(account["rejected_at"], expected)

        account = controller._base_evaluation(
            base(plans=[plan(False, 0.05), plan(False, 0.02)]), **context
        )
        self.assertAlmostEqual(account["best_ik"]["ik_position_error_m"], 0.02)
        self.assertFalse(account["best_ik"]["ik_converged"])

    def test_later_pi_wave_failure_keeps_an_earlier_strict_result(self) -> None:
        hardware = FakeHardware(clearance_m=0.9, arms=ARMS)
        detector = _ScriptedDetector([(27.0, 15.0, 37.0, 33.0)])
        controller, environment = _controller(
            hardware,
            detector,
            settings={
                "candidate_yaw_offsets_deg": [0.0],
                "pi_ik_shortlist_base_limit": 5,
                "pi_ik_grasps_per_base": 2,
                "pi_ik_candidate_limit": 10,
                "ik_batch_size": 4,
                # A later-wave failure needs a later wave to exist.
                "pi_ik_best_tier_early_exit": False,
            },
        )
        self.addCleanup(environment.safe_shutdown)
        planning_call = 0

        def strict_then_timeout(arm, poses_xyz_rpy):
            nonlocal planning_call
            del arm
            planning_call += 1
            if planning_call > 1:
                raise TimeoutError("scripted later-wave timeout")
            return {
                "plans": [
                    {
                        "candidate_index": index,
                        "success": True,
                        "ik_converged": True,
                        "ik_position_error_m": 0.0005,
                        "ik_rotation_error_rad": 0.0005,
                        "joint_travel_l2_rad": 0.1,
                        "joint_travel_max_rad": 0.05,
                    }
                    for index in range(len(poses_xyz_rpy))
                ]
            }

        environment.plan_arm_poses = strict_then_timeout
        generated = controller.manipulation.generate_grasp_candidates(
            "marker", "left"
        )
        frame = environment.navigation_frame(max_age_s=1.0)

        evaluation = controller._evaluate_candidates(generated, frame)

        self.assertEqual(evaluation["ik_query_count"], 4)
        self.assertEqual(evaluation["pi_ik_requested_count"], 10)
        self.assertGreater(evaluation["strict_pi_candidate_count"], 0)
        self.assertIn(
            "scripted later-wave timeout",
            evaluation["pi_ik_terminal_batch_error"]["error"],
        )

    def test_stage_three_tracks_one_absolute_se2_target(self) -> None:
        hardware = FakeHardware(clearance_m=0.9, arms=ARMS)
        detector = _ScriptedDetector([(27.0, 15.0, 37.0, 33.0)])
        controller, environment = _controller(hardware, detector)
        self.addCleanup(environment.safe_shutdown)
        target = controller._single_stage_motion_target(
            forward_m=0.15,
            left_m=0.15,
            yaw_rad=np.deg2rad(15.0),
        )
        motion = controller._execute_single_stage_motion(target)

        self.assertTrue(motion["success"], motion)
        self.assertEqual(motion["primitive"], "move_planar_single_stage")
        self.assertEqual(
            motion["metrics"]["execution_mode"], "single_absolute_se2"
        )
        self.assertNotIn("phases", motion)
        self.assertTrue(motion["metrics"]["reverse_correction_enabled"])
        np.testing.assert_allclose(
            motion["metrics"]["target_pose_world"], target["target_pose_world"]
        )
        self.assertAlmostEqual(hardware.pose.x_m, 0.15, delta=0.04)
        self.assertAlmostEqual(hardware.pose.y_m, 0.15, delta=0.04)
        self.assertAlmostEqual(hardware.pose.yaw_rad, np.deg2rad(15.0), delta=0.06)

    def test_single_stage_target_allows_reverse_correction(self) -> None:
        hardware = FakeHardware(clearance_m=0.9, arms=ARMS)
        detector = _ScriptedDetector([(27.0, 15.0, 37.0, 33.0)])
        controller, environment = _controller(hardware, detector)
        self.addCleanup(environment.safe_shutdown)

        target = controller._single_stage_motion_target(
            forward_m=-0.10,
            left_m=0.0,
            yaw_rad=0.0,
        )
        motion = controller._execute_single_stage_motion(target)

        self.assertTrue(motion["success"], motion)
        self.assertEqual(
            motion["metrics"]["execution_mode"], "single_absolute_se2"
        )
        self.assertFalse(motion["metrics"]["rear_clearance_checked"])
        self.assertTrue(any(command[0] < 0.0 for command in hardware.commands))
        self.assertAlmostEqual(hardware.pose.x_m, -0.10, delta=0.04)

    def test_motion_failure_retains_selected_and_absolute_target_for_trace(
        self,
    ) -> None:
        hardware = FakeHardware(clearance_m=0.9, arms=ARMS)
        detector = _ScriptedDetector([(27.0, 15.0, 37.0, 33.0)])
        controller, environment = _controller(
            hardware,
            detector,
            settings={
                "candidate_forward_offsets_m": [0.05],
                "candidate_lateral_offsets_m": [0.0],
                "candidate_yaw_offsets_deg": [5.0],
                "candidate_base_limit": 1,
                "pi_ik_shortlist_base_limit": 1,
                "pi_ik_grasps_per_base": 1,
                "pi_ik_candidate_limit": 1,
                "pi_ik_nominal_candidate_limit": 1,
                "ik_batch_size": 1,
            },
        )
        self.addCleanup(environment.safe_shutdown)

        def fail_motion(*args, **kwargs):
            del args, kwargs
            return {
                "success": False,
                "status": "failed",
                "reason": "operator_stop_requested",
                "metrics": {"last_command": [0.1, 0.0, 0.2]},
            }

        environment.controller.move_planar_relative = fail_motion

        result = controller.prepare_for_manipulation("marker", "left")

        self.assertFalse(result["success"])
        self.assertIn("single-stage SE(2) motion failed", result["reason"])
        self.assertEqual(result["selected_base_pose"]["forward_m"], 0.05)
        self.assertAlmostEqual(
            result["selected_base_pose"]["yaw_rad"], np.deg2rad(5.0)
        )
        self.assertEqual(
            result["motion_target"]["target_relative_forward_m"], 0.05
        )
        self.assertEqual(
            result["motion"]["metrics"]["target_pose_world"],
            result["motion_target"]["target_pose_world"],
        )

    def test_the_search_exposes_every_ready_pose_in_ranked_order(self) -> None:
        hardware = FakeHardware(clearance_m=0.9, arms=ARMS)
        controller, environment = _forward_poses_ready_scene(hardware)
        self.addCleanup(environment.safe_shutdown)
        generated = controller.manipulation.generate_grasp_candidates(
            "marker", "left"
        )
        frame = environment.navigation_frame(max_age_s=1.0)

        evaluation = controller._evaluate_candidates(generated, frame)

        ranked = evaluation["ranked_candidates"]
        self.assertEqual(len(ranked), evaluation["strict_pi_candidate_count"])
        self.assertGreaterEqual(len(ranked), 3)
        self.assertIs(ranked[0], evaluation["selected"])
        self.assertTrue(all(candidate["strict_ready"] for candidate in ranked))
        alternatives = evaluation["ranked_alternatives"]
        self.assertEqual(
            [candidate["candidate_index"] for candidate in ranked[: len(alternatives)]],
            [item["candidate_index"] for item in alternatives],
        )

    def test_single_stage_target_can_start_from_a_given_pose(self) -> None:
        hardware = FakeHardware(clearance_m=0.9, arms=ARMS)
        detector = _ScriptedDetector([(27.0, 15.0, 37.0, 33.0)])
        controller, environment = _controller(hardware, detector)
        self.addCleanup(environment.safe_shutdown)
        # The live pose is elsewhere; the given start pose must win.
        hardware.pose.x_m = 1.0

        target = controller._single_stage_motion_target(
            forward_m=0.10,
            left_m=0.0,
            yaw_rad=0.0,
            start_pose_world=[0.0, 0.0, np.pi / 2.0],
        )

        self.assertEqual(target["start_pose_world"], [0.0, 0.0, np.pi / 2.0])
        np.testing.assert_allclose(
            target["target_pose_world"], [0.0, 0.10, np.pi / 2.0], atol=1e-12
        )

    def test_the_single_attempt_is_recorded_when_no_alternative_is_allowed(
        self,
    ) -> None:
        hardware = FakeHardware(clearance_m=0.9, arms=ARMS)
        controller, environment = _forward_poses_ready_scene(hardware)
        self.addCleanup(environment.safe_shutdown)
        refused: list[list[float]] = []

        def refuse_every_goal(*args, **kwargs):
            del args
            refused.append(list(kwargs["target_pose_world"]))
            return _refusal()

        environment.controller.move_planar_relative = refuse_every_goal

        result = controller.prepare_for_manipulation("marker", "left")

        self.assertFalse(result["success"])
        self.assertEqual(
            result["reason"],
            "RuntimeError:single-stage SE(2) motion failed: obstacle_too_close",
        )
        self.assertEqual(len(refused), 1)
        attempts = result["motion_attempts"]
        self.assertEqual(len(attempts), 1)
        self.assertEqual(attempts[0]["attempt"], 0)
        self.assertFalse(attempts[0]["success"])
        self.assertEqual(attempts[0]["reason"], "obstacle_too_close")
        self.assertTrue(attempts[0]["needed_motion"])
        self.assertNotIn("motion_exhausted", result)
        self.assertIsNone(result["executed_candidate_index"])
        # The one refusal is recorded once, in ``motion`` alone.
        self.assertNotIn("first_refused_motion", result)

    def test_a_pose_kept_without_motion_is_one_trivially_successful_attempt(
        self,
    ) -> None:
        hardware = FakeHardware(clearance_m=0.9, arms=ARMS)
        detector = _ScriptedDetector([(27.0, 15.0, 37.0, 33.0)])
        controller, environment = _controller(
            hardware, detector, settings={"candidate_yaw_offsets_deg": [0.0]}
        )
        self.addCleanup(environment.safe_shutdown)

        result = controller.prepare_for_manipulation("pen", "left")

        self.assertTrue(result["success"], result)
        attempts = result["motion_attempts"]
        self.assertEqual(len(attempts), 1)
        self.assertFalse(attempts[0]["needed_motion"])
        self.assertTrue(attempts[0]["success"])
        self.assertEqual(attempts[0]["reason"], "selected_current_pose")
        self.assertEqual(
            result["executed_candidate_index"], attempts[0]["candidate_index"]
        )
        self.assertNotIn("executed_pose_evaluation", result)
        self.assertNotIn("first_refused_motion", result)
        self.assertEqual(result["certification"]["source"], "virtual_evaluation")

    def test_a_refused_motion_hands_its_turn_to_the_next_certified_pose(
        self,
    ) -> None:
        hardware = FakeHardware(clearance_m=0.9, arms=ARMS)
        controller, environment = _forward_poses_ready_scene(
            hardware, settings={"motion_alternative_limit": 2}
        )
        self.addCleanup(environment.safe_shutdown)
        real_move = environment.controller.move_planar_relative
        goals: list[list[float]] = []

        def refuse_the_first_goal(*args, **kwargs):
            goals.append(list(kwargs["target_pose_world"]))
            if len(goals) == 1:
                return _refusal()
            return real_move(*args, **kwargs)

        environment.controller.move_planar_relative = refuse_the_first_goal

        result = controller.prepare_for_manipulation("marker", "left")

        self.assertTrue(result["success"], result)
        self.assertEqual(result["reason"], "grasp_execution_ready")
        attempts = result["motion_attempts"]
        self.assertEqual([item["attempt"] for item in attempts], [0, 1])
        self.assertEqual([item["success"] for item in attempts], [False, True])
        self.assertEqual(attempts[0]["reason"], "obstacle_too_close")
        self.assertEqual(attempts[1]["reason"], "target_reached")
        self.assertTrue(all(item["needed_motion"] for item in attempts))
        ranked = result["ranked_alternatives"]
        self.assertEqual(
            [item["candidate_index"] for item in attempts],
            [ranked[0]["candidate_index"], ranked[1]["candidate_index"]],
        )
        self.assertEqual(
            result["executed_candidate_index"], ranked[1]["candidate_index"]
        )
        # Every record that describes the motion now describes the executed
        # pose, and the executed motion tracked that pose's absolute goal.
        self.assertEqual(result["selected_base_pose"]["forward_m"], ranked[1]["forward_m"])
        self.assertEqual(result["selected_base_pose"]["left_m"], ranked[1]["left_m"])
        self.assertEqual(
            result["selected_base_pose"]["movement_cost_tier"],
            ranked[1]["movement_cost_tier"],
        )
        self.assertEqual(
            result["motion_target"]["target_relative_left_m"], ranked[1]["left_m"]
        )
        np.testing.assert_allclose(
            result["motion"]["metrics"]["target_pose_world"],
            result["motion_target"]["target_pose_world"],
        )
        np.testing.assert_allclose(goals[1], result["motion_target"]["target_pose_world"])
        self.assertTrue(result["motion_executed"])
        self.assertNotIn("motion_exhausted", result)
        self.assertAlmostEqual(hardware.pose.x_m, ranked[1]["forward_m"], delta=0.04)
        self.assertAlmostEqual(hardware.pose.y_m, ranked[1]["left_m"], delta=0.04)
        # The search's own choice keeps its account; the executed pose gains
        # one, and the first refusal keeps its full diagnostics.
        self.assertEqual(
            result["selected_pose_evaluation"]["left_m"], ranked[0]["left_m"]
        )
        executed = result["executed_pose_evaluation"]
        self.assertEqual(
            (executed["forward_m"], executed["left_m"]),
            (ranked[1]["forward_m"], ranked[1]["left_m"]),
        )
        self.assertTrue(executed["strict_ready"])
        self.assertEqual(result["first_refused_motion"]["reason"], "obstacle_too_close")
        self.assertEqual(result["certification"]["source"], "fresh_sample")
        self.assertEqual(controller.last_debug["phase"], "grasp_execution_ready")
        self.assertEqual(
            controller.last_debug["executed_candidate_index"],
            ranked[1]["candidate_index"],
        )

    def test_prepare_fails_only_after_every_allowed_alternative_is_refused(
        self,
    ) -> None:
        hardware = FakeHardware(clearance_m=0.9, arms=ARMS)
        controller, environment = _forward_poses_ready_scene(
            hardware, settings={"motion_alternative_limit": 2}
        )
        self.addCleanup(environment.safe_shutdown)
        refused: list[list[float]] = []

        def refuse_every_goal(*args, **kwargs):
            del args
            refused.append(list(kwargs["target_pose_world"]))
            return _refusal()

        environment.controller.move_planar_relative = refuse_every_goal

        result = controller.prepare_for_manipulation("marker", "left")

        self.assertFalse(result["success"])
        self.assertTrue(
            result["reason"].startswith(
                "RuntimeError:no certified base reachable"
            ),
            result["reason"],
        )
        self.assertIn("3 motions refused (obstacle_too_closex3)", result["reason"])
        self.assertTrue(result["motion_exhausted"])
        self.assertIsNone(result["executed_candidate_index"])
        attempts = result["motion_attempts"]
        self.assertEqual(len(attempts), 3)
        self.assertEqual(len(refused), 3)
        self.assertFalse(any(item["success"] for item in attempts))
        self.assertEqual({item["reason"] for item in attempts}, {"obstacle_too_close"})
        ranked = result["ranked_alternatives"]
        self.assertEqual(
            [item["candidate_index"] for item in attempts],
            [item["candidate_index"] for item in ranked[:3]],
        )
        # Three distinct absolute goals were commanded, tier order kept.
        self.assertEqual(len({tuple(goal) for goal in refused}), 3)
        tiers = [item["movement_cost_tier"] for item in attempts]
        self.assertEqual(tiers, sorted(tiers))
        self.assertEqual(result["motion"]["reason"], "obstacle_too_close")
        self.assertTrue(result["motion_executed"])
        self.assertEqual(hardware.velocity.tolist(), [0.0, 0.0, 0.0])
        self.assertTrue(result["final_stop"]["success"])
        # ``motion`` is the last refusal; the first keeps its own record.
        first_refused = result["first_refused_motion"]
        self.assertIsNot(first_refused, result["motion"])
        self.assertEqual(first_refused["reason"], "obstacle_too_close")
        np.testing.assert_allclose(
            first_refused["metrics"]["target_pose_world"], refused[0]
        )
        np.testing.assert_allclose(
            result["motion"]["metrics"]["target_pose_world"], refused[2]
        )
        # The search records survive the exhausted motion as they survive
        # a single refusal.
        self.assertTrue(result["pi_ik_query_log"])
        self.assertTrue(result["ranked_alternatives"])
        self.assertEqual(controller.last_debug["phase"], "failed")
        self.assertEqual(len(controller.last_debug["motion_attempts"]), 3)

    def test_later_attempts_are_aimed_from_the_observation_pose(self) -> None:
        """A refused motion can leave the base part-way; the candidates are
        relative to where the search observed from, not to where it stopped."""

        hardware = FakeHardware(clearance_m=0.9, arms=ARMS)
        controller, environment = _forward_poses_ready_scene(
            hardware, settings={"motion_alternative_limit": 1}
        )
        self.addCleanup(environment.safe_shutdown)
        goals: list[list[float]] = []

        def move_part_way_then_refuse(*args, **kwargs):
            del args
            goals.append(list(kwargs["target_pose_world"]))
            if len(goals) == 1:
                hardware.pose.x_m += 0.03
                hardware.pose.y_m -= 0.02
                return _refusal()
            return {
                "success": True,
                "status": "succeeded",
                "reason": "target_reached",
                "metrics": {},
            }

        environment.controller.move_planar_relative = move_part_way_then_refuse

        result = controller.prepare_for_manipulation("marker", "left")

        self.assertTrue(result["success"], result)
        attempts = result["motion_attempts"]
        self.assertEqual(len(attempts), 2)
        self.assertTrue(attempts[1]["needed_motion"])
        second = attempts[1]
        # Observation pose (the origin) plus the candidate's own offset ...
        np.testing.assert_allclose(
            goals[1],
            [second["forward_m"], second["left_m"], np.deg2rad(second["yaw_deg"])],
            atol=1e-12,
        )
        # ... and not the displaced live pose plus that offset.
        self.assertGreater(abs(goals[1][0] - (0.03 + second["forward_m"])), 0.01)
        self.assertEqual(result["motion_target"]["start_pose_world"], [0.0, 0.0, 0.0])

    def test_a_refusal_whose_stop_failed_ends_the_attempts(self) -> None:
        hardware = FakeHardware(clearance_m=0.9, arms=ARMS)
        controller, environment = _forward_poses_ready_scene(
            hardware, settings={"motion_alternative_limit": 2}
        )
        self.addCleanup(environment.safe_shutdown)
        calls = {"count": 0}

        def refuse_and_fail_to_stop(*args, **kwargs):
            del args, kwargs
            calls["count"] += 1
            return _refusal(
                "obstacle_too_close; final_stop_failed: stop_rpc_unavailable"
            )

        environment.controller.move_planar_relative = refuse_and_fail_to_stop

        result = controller.prepare_for_manipulation("marker", "left")

        self.assertFalse(result["success"])
        self.assertEqual(calls["count"], 1)
        self.assertEqual(len(result["motion_attempts"]), 1)
        self.assertTrue(result["motion_exhausted"])
        self.assertNotIn("first_refused_motion", result)
        self.assertTrue(
            result["reason"].startswith(
                "RuntimeError:no certified base reachable: 1 motions refused"
            ),
            result["reason"],
        )

    def test_an_operator_stop_during_an_attempt_ends_the_attempts(self) -> None:
        hardware = FakeHardware(clearance_m=0.9, arms=ARMS)
        controller, environment = _forward_poses_ready_scene(
            hardware, settings={"motion_alternative_limit": 2}
        )
        self.addCleanup(environment.safe_shutdown)
        real_move = environment.controller.move_planar_relative
        calls = {"count": 0}

        def stop_during_the_first_motion(*args, **kwargs):
            calls["count"] += 1
            # The operator's stop lands while this motion tracks its goal.
            environment.controller.request_stop()
            return real_move(*args, **kwargs)

        environment.controller.move_planar_relative = stop_during_the_first_motion

        result = controller.prepare_for_manipulation("marker", "left")

        self.assertFalse(result["success"])
        # The remaining candidates are not commanded ...
        self.assertEqual(calls["count"], 1)
        attempts = result["motion_attempts"]
        self.assertEqual(len(attempts), 1)
        self.assertEqual(attempts[0]["reason"], "RuntimeError:operator_stop_requested")
        # ... and the result reads as a single refusal, not an exhausted search.
        self.assertEqual(
            result["reason"],
            "RuntimeError:single-stage SE(2) motion failed: "
            "RuntimeError:operator_stop_requested",
        )
        self.assertNotIn("motion_exhausted", result)
        self.assertNotIn("first_refused_motion", result)
        self.assertIsNone(result["executed_candidate_index"])
        self.assertEqual(result["motion"]["reason"], "RuntimeError:operator_stop_requested")
        self.assertTrue(result["final_stop"]["success"])
        self.assertEqual(hardware.velocity.tolist(), [0.0, 0.0, 0.0])

    def test_attempts_made_before_a_later_attempt_raises_stay_in_the_result(
        self,
    ) -> None:
        hardware = FakeHardware(clearance_m=0.9, arms=ARMS)
        controller, environment = _forward_poses_ready_scene(
            hardware, settings={"motion_alternative_limit": 2}
        )
        self.addCleanup(environment.safe_shutdown)
        calls = {"count": 0}

        def refuse_then_lose_the_frame(*args, **kwargs):
            del args, kwargs
            calls["count"] += 1
            # The live pose the next attempt reads after a refusal is gone.
            hardware.frame_error = RuntimeError("zed_frame_stale")
            return _refusal()

        environment.controller.move_planar_relative = refuse_then_lose_the_frame

        result = controller.prepare_for_manipulation("marker", "left")

        self.assertFalse(result["success"])
        self.assertEqual(result["reason"], "RuntimeError:zed_frame_stale")
        self.assertEqual(calls["count"], 1)
        # The attempt that completed is still on record, with its outcome.
        attempts = result["motion_attempts"]
        self.assertEqual(len(attempts), 1)
        self.assertEqual(attempts[0]["attempt"], 0)
        self.assertEqual(attempts[0]["reason"], "obstacle_too_close")
        self.assertEqual(result["motion"]["reason"], "obstacle_too_close")
        self.assertTrue(result["motion_executed"])
        self.assertIsNone(result["executed_candidate_index"])
        self.assertNotIn("motion_exhausted", result)
        self.assertNotIn("first_refused_motion", result)
        self.assertEqual(controller.last_debug["phase"], "failed")
        self.assertEqual(len(controller.last_debug["motion_attempts"]), 1)

    def test_by_default_a_refused_near_tier_fails_with_farther_tiers_unqueried(
        self,
    ) -> None:
        hardware = FakeHardware(clearance_m=0.9, arms=ARMS)
        controller, environment, pi = _refused_near_tier_scene(
            hardware, settings={"motion_alternative_limit": 2}
        )
        self.addCleanup(environment.safe_shutdown)
        refused: list[list[float]] = []

        def refuse_every_goal(*args, **kwargs):
            del args
            refused.append(list(kwargs["target_pose_world"]))
            return _refusal()

        environment.controller.move_planar_relative = refuse_every_goal

        result = controller.prepare_for_manipulation("marker", "left")

        self.assertFalse(result["success"])
        self.assertEqual(
            result["reason"],
            "RuntimeError:no certified base reachable: "
            "1 motions refused (obstacle_too_closex1)",
        )
        self.assertTrue(result["motion_exhausted"])
        self.assertEqual(len(refused), 1)
        self.assertEqual(len(result["motion_attempts"]), 1)
        self.assertEqual(result["motion_attempts"][0]["forward_m"], 0.05)
        # The 0.10 m tier would certify, but the early exit never queried it.
        self.assertEqual(pi["requested_z"], [0.9, 0.9, 0.85, 0.85])
        self.assertEqual(
            {row["stage"] for row in result["pi_ik_query_log"]}, {"nominal"}
        )
        self.assertNotIn("search_resumes", result)
        self.assertNotIn("search_passes", result)
        self.assertNotIn("pi_ik_compute_elapsed_s", result)
        self.assertNotIn("resumed_candidate_evaluation", result["stage_timings_s"])

    def test_refusals_of_every_ready_pose_resume_the_search_in_farther_tiers(
        self,
    ) -> None:
        hardware = FakeHardware(clearance_m=0.9, arms=ARMS)
        controller, environment, pi = _refused_near_tier_scene(
            hardware,
            settings={
                "motion_alternative_limit": 2,
                "continue_search_after_refused_motions": True,
            },
            pi_call_s=(0.25, 0.25, 0.25),
        )
        self.addCleanup(environment.safe_shutdown)
        real_move = environment.controller.move_planar_relative
        goals: list[list[float]] = []
        budget_s = controller.config.pi_ik_compute_budget_s

        def refuse_the_first_goal_slowly(*args, **kwargs):
            goals.append(list(kwargs["target_pose_world"]))
            if len(goals) == 1:
                # A refusal slower than the whole Pi IK budget: only the Pi
                # time spent may count against the resumed search.
                hardware.now += 2.0 * budget_s
                return _refusal()
            motion = real_move(*args, **kwargs)
            pi["moved"] = bool(motion.get("success", False))
            return motion

        environment.controller.move_planar_relative = refuse_the_first_goal_slowly

        result = controller.prepare_for_manipulation("marker", "left")

        self.assertTrue(result["success"], result)
        attempts = result["motion_attempts"]
        self.assertEqual([item["success"] for item in attempts], [False, True])
        self.assertEqual(
            [(item["forward_m"], item["movement_cost_tier"]) for item in attempts],
            [(0.05, 2), (0.10, 5)],
        )
        self.assertEqual(
            result["executed_candidate_index"], attempts[1]["candidate_index"]
        )
        self.assertEqual(result["selected_base_pose"]["forward_m"], 0.10)
        self.assertEqual(result["executed_pose_evaluation"]["forward_m"], 0.10)
        # The resumed rows follow the first pass's, in one continuous order.
        log = result["pi_ik_query_log"]
        self.assertEqual(result["ik_query_count"], len(log))
        self.assertEqual([row["order"] for row in log], list(range(len(log))))
        self.assertEqual(
            [row["stage"] for row in log],
            ["nominal"] * 4 + ["nominal_resumed"] * 2,
        )
        self.assertEqual({row["forward_m"] for row in log[4:]}, {0.10})
        self.assertEqual(pi["requested_z"][:6], [0.9, 0.9, 0.85, 0.85, 0.8, 0.8])
        self.assertEqual(
            sum(int(batch["candidate_count"]) for batch in result["pi_batches"]),
            len(log),
        )
        self.assertEqual(result["pi_batches"][-1]["stage"], "nominal_resumed")
        self.assertEqual(result["nominal_grasp_evaluated_count"], 6)
        self.assertEqual(result["strict_pi_candidate_count"], 2)
        self.assertEqual(
            [item["candidate_index"] for item in result["ranked_alternatives"]],
            [item["candidate_index"] for item in attempts],
        )
        tiers = {row["tier"]: row for row in result["tier_evaluation"]}
        self.assertEqual(tiers[5]["grasps_queried"], 2)
        self.assertEqual(tiers[5]["bases_ready"], 1)
        resumes = result["search_resumes"]
        self.assertEqual(len(resumes), 1)
        self.assertEqual(
            {key: resumes[0][key] for key in resumes[0] if key != "pi_ik_compute_elapsed_s"},
            {
                "after_attempts": 1,
                "queries": 2,
                "new_ready_bases": 1,
                "tiers_queried": [5],
                "budget_exhausted": False,
                "early_exit": True,
                "terminal_batch_error": None,
            },
        )
        # The first pass scheduled the 0.10 m pairs before its early exit; the
        # resume that queries them does not request them a second time.
        self.assertEqual(result["pi_ik_requested_count"], len(log))
        self.assertEqual(
            [(item["pass"], item["early_exit"]) for item in result["search_passes"]],
            [("initial", True), ("resumed", True)],
        )
        # The budget counts Pi time alone: the slow refusal in between did not
        # use it up, and the call's Pi figure covers both passes.
        self.assertAlmostEqual(resumes[0]["pi_ik_compute_elapsed_s"], 0.25)
        self.assertAlmostEqual(result["pi_ik_compute_elapsed_s"], 0.75)
        self.assertFalse(result["pi_ik_budget_exhausted"])
        timings = result["stage_timings_s"]
        self.assertAlmostEqual(timings["resumed_candidate_evaluation"], 0.25)
        self.assertGreaterEqual(timings["motion"], 2.0 * budget_s)
        # The resumed search is not also counted inside ``motion``.
        self.assertLessEqual(sum(timings.values()), result["elapsed_s"] + 1e-9)
        self.assertEqual(controller.last_debug["ik_query_count"], len(log))
        self.assertEqual(controller.last_debug["search_resumes"], resumes)

    def test_a_budget_the_first_pass_used_up_leaves_nothing_to_resume(
        self,
    ) -> None:
        cases = {
            # The last batch overran the budget: nothing may be resumed.
            "overrun": ((0.1, 5.0), 0),
            # Some budget is left, but less than one full Pi wave needs: the
            # resume stops before its first query.
            "remainder_below_one_wave": ((1.0, 1.0), 1),
        }
        for name, (pi_call_s, expected_resumes) in cases.items():
            with self.subTest(name):
                hardware = FakeHardware(clearance_m=0.9, arms=ARMS)
                controller, environment, pi = _refused_near_tier_scene(
                    hardware,
                    settings={
                        "motion_alternative_limit": 2,
                        "continue_search_after_refused_motions": True,
                        "pi_ik_compute_budget_s": 2.5,
                    },
                    pi_call_s=pi_call_s,
                )
                self.addCleanup(environment.safe_shutdown)
                environment.controller.move_planar_relative = (
                    lambda *args, **kwargs: _refusal()
                )

                result = controller.prepare_for_manipulation("marker", "left")

                self.assertFalse(result["success"])
                self.assertEqual(
                    result["reason"],
                    "RuntimeError:no certified base reachable: "
                    "1 motions refused (obstacle_too_closex1)",
                )
                self.assertTrue(result["motion_exhausted"])
                self.assertEqual(pi["requested_z"], [0.9, 0.9, 0.85, 0.85])
                self.assertEqual(
                    {row["stage"] for row in result["pi_ik_query_log"]},
                    {"nominal"},
                )
                resumes = result["search_resumes"]
                self.assertEqual(len(resumes), expected_resumes)
                for resume in resumes:
                    self.assertEqual(resume["queries"], 0)
                    self.assertEqual(resume["new_ready_bases"], 0)
                    self.assertTrue(resume["budget_exhausted"])

    def test_an_operator_stop_is_not_resumed_into_farther_tiers(self) -> None:
        hardware = FakeHardware(clearance_m=0.9, arms=ARMS)
        controller, environment, pi = _refused_near_tier_scene(
            hardware,
            settings={
                "motion_alternative_limit": 2,
                "continue_search_after_refused_motions": True,
            },
        )
        self.addCleanup(environment.safe_shutdown)
        real_move = environment.controller.move_planar_relative
        calls = {"count": 0}

        def stop_during_the_first_motion(*args, **kwargs):
            calls["count"] += 1
            environment.controller.request_stop()
            return real_move(*args, **kwargs)

        environment.controller.move_planar_relative = stop_during_the_first_motion

        result = controller.prepare_for_manipulation("marker", "left")

        self.assertFalse(result["success"])
        self.assertEqual(
            result["reason"],
            "RuntimeError:single-stage SE(2) motion failed: "
            "RuntimeError:operator_stop_requested",
        )
        self.assertEqual(calls["count"], 1)
        self.assertEqual(len(result["motion_attempts"]), 1)
        self.assertNotIn("motion_exhausted", result)
        self.assertEqual(pi["requested_z"], [0.9, 0.9, 0.85, 0.85])
        self.assertEqual(result["search_resumes"], [])
        self.assertNotIn("resumed_candidate_evaluation", result["stage_timings_s"])

    def test_a_refusal_whose_stop_failed_is_not_resumed(self) -> None:
        hardware = FakeHardware(clearance_m=0.9, arms=ARMS)
        controller, environment, pi = _refused_near_tier_scene(
            hardware,
            settings={
                "motion_alternative_limit": 2,
                "continue_search_after_refused_motions": True,
            },
        )
        self.addCleanup(environment.safe_shutdown)
        environment.controller.move_planar_relative = lambda *args, **kwargs: _refusal(
            "obstacle_too_close; final_stop_failed: stop_rpc_unavailable"
        )

        result = controller.prepare_for_manipulation("marker", "left")

        self.assertFalse(result["success"])
        self.assertTrue(
            result["reason"].startswith(
                "RuntimeError:no certified base reachable: 1 motions refused"
            ),
            result["reason"],
        )
        self.assertEqual(pi["requested_z"], [0.9, 0.9, 0.85, 0.85])
        self.assertEqual(result["search_resumes"], [])

    def test_a_resume_a_pi_batch_error_cut_short_records_the_error(self) -> None:
        hardware = FakeHardware(clearance_m=0.9, arms=ARMS)
        controller, environment, pi = _refused_near_tier_scene(
            hardware,
            settings={
                "motion_alternative_limit": 2,
                "continue_search_after_refused_motions": True,
            },
        )
        self.addCleanup(environment.safe_shutdown)
        scene_plan = environment.plan_arm_poses

        def drop_the_resumed_batch(arm, poses_xyz_rpy):
            # The first pass takes two calls; the third is the resume's.
            if pi["calls"] == 2:
                pi["calls"] += 1
                raise ConnectionError("pi rpc dropped")
            return scene_plan(arm, poses_xyz_rpy)

        environment.plan_arm_poses = drop_the_resumed_batch
        environment.controller.move_planar_relative = (
            lambda *args, **kwargs: _refusal()
        )

        result = controller.prepare_for_manipulation("marker", "left")

        self.assertFalse(result["success"])
        self.assertEqual(
            result["reason"],
            "RuntimeError:no certified base reachable: "
            "1 motions refused (obstacle_too_closex1)",
        )
        expected_error = {
            "stage": "nominal_resumed",
            "failed_batch_start": 0,
            "failed_batch_size": 2,
            "error": "ConnectionError: pi rpc dropped",
        }
        resumes = result["search_resumes"]
        self.assertEqual(len(resumes), 1)
        self.assertEqual(resumes[0]["queries"], 0)
        # Not a resume that simply found nothing: the error says why.
        self.assertEqual(resumes[0]["terminal_batch_error"], expected_error)
        self.assertEqual(result["pi_ik_terminal_batch_error"], expected_error)
        self.assertEqual(
            [item["terminal_batch_error"] for item in result["search_passes"]],
            [None, expected_error],
        )

    def test_a_resume_that_raises_keeps_the_motion_timing_and_the_failed_resume(
        self,
    ) -> None:
        hardware = FakeHardware(clearance_m=0.9, arms=ARMS)
        controller, environment, pi = _refused_near_tier_scene(
            hardware,
            settings={
                "motion_alternative_limit": 2,
                "continue_search_after_refused_motions": True,
            },
        )
        self.addCleanup(environment.safe_shutdown)
        scene_plan = environment.plan_arm_poses

        def malformed_resumed_reply(arm, poses_xyz_rpy):
            if pi["calls"] == 2:
                pi["calls"] += 1
                hardware.now += 0.5
                return {"no_plans": True}
            return scene_plan(arm, poses_xyz_rpy)

        def refuse_slowly(*args, **kwargs):
            del args, kwargs
            hardware.now += 1.0
            return _refusal()

        environment.plan_arm_poses = malformed_resumed_reply
        environment.controller.move_planar_relative = refuse_slowly

        result = controller.prepare_for_manipulation("marker", "left")

        self.assertFalse(result["success"])
        self.assertEqual(
            result["reason"],
            "RuntimeError:Pi IK returned no plans: {'no_plans': True}",
        )
        self.assertEqual(len(result["motion_attempts"]), 1)
        self.assertEqual(
            result["search_resumes"],
            [
                {
                    "after_attempts": 1,
                    "error": "RuntimeError: Pi IK returned no plans: "
                    "{'no_plans': True}",
                }
            ],
        )
        timings = result["stage_timings_s"]
        self.assertAlmostEqual(timings["resumed_candidate_evaluation"], 0.5)
        # The refusal's time, without the resume that raised after it.
        self.assertGreaterEqual(timings["motion"], 1.0)
        self.assertLess(timings["motion"], 1.5)

    def test_resumed_attempts_stay_within_the_attempt_budget(self) -> None:
        hardware = FakeHardware(clearance_m=0.9, arms=ARMS)
        # Lateral poses at the current forward offset do not converge, so
        # tier 2 holds one ready base, tier 3 two and tier 5 three.
        controller, environment, pi = _refused_near_tier_scene(
            hardware,
            settings={
                "motion_alternative_limit": 1,
                "continue_search_after_refused_motions": True,
            },
            lateral_offsets_m=(-0.05, 0.0, 0.05),
        )
        self.addCleanup(environment.safe_shutdown)
        refused: list[list[float]] = []

        def refuse_every_goal(*args, **kwargs):
            del args
            refused.append(list(kwargs["target_pose_world"]))
            return _refusal()

        environment.controller.move_planar_relative = refuse_every_goal

        result = controller.prepare_for_manipulation("marker", "left")

        self.assertFalse(result["success"])
        self.assertEqual(
            result["reason"],
            "RuntimeError:no certified base reachable: "
            "2 motions refused (obstacle_too_closex2)",
        )
        self.assertEqual(len(refused), 2)
        attempts = result["motion_attempts"]
        self.assertEqual(
            [(item["forward_m"], item["movement_cost_tier"]) for item in attempts],
            [(0.05, 2), (0.05, 3)],
        )
        resumes = result["search_resumes"]
        self.assertEqual(len(resumes), 1)
        self.assertEqual(resumes[0]["tiers_queried"], [3])
        self.assertEqual(resumes[0]["new_ready_bases"], 2)
        # A ready pose was still untried when the budget ran out.
        self.assertEqual(len(result["ranked_alternatives"]), 3)
        self.assertNotIn(0.8, pi["requested_z"])
        # A failed call still reports the Pi time and the passes it spent.
        self.assertEqual(len(result["search_passes"]), 2)
        self.assertEqual(
            result["pi_ik_compute_elapsed_s"],
            controller.last_debug["pi_ik_compute_elapsed_s"],
        )

    def test_a_resumed_search_runs_robustness_for_the_poses_it_certifies(
        self,
    ) -> None:
        hardware = FakeHardware(clearance_m=0.9, arms=ARMS)
        controller, environment, _ = _refused_near_tier_scene(
            hardware, settings={"robustness_enabled": True}
        )
        self.addCleanup(environment.safe_shutdown)
        generated = controller.manipulation.generate_grasp_candidates(
            "marker", "left"
        )
        frame = environment.navigation_frame(max_age_s=1.0)

        evaluation = controller._evaluate_candidates(generated, frame)

        self.assertEqual(len(evaluation["ranked_candidates"]), 1)
        self.assertEqual(len(evaluation["search_passes"]), 1)
        continue_search = evaluation["continue_search"]
        self.assertIsNotNone(continue_search)
        first_rows = len(evaluation["pi_ik_query_log"])

        resumed = continue_search(
            {int(evaluation["selected"]["candidate_index"])}
        )

        log = resumed["pi_ik_query_log"]
        self.assertEqual(resumed["ik_query_count"], len(log))
        self.assertEqual([row["order"] for row in log], list(range(len(log))))
        self.assertEqual(
            [row["stage"] for row in log[:first_rows]],
            ["nominal"] * 4 + ["robustness"] * 12,
        )
        self.assertEqual(
            [row["stage"] for row in log[first_rows:]],
            ["nominal_resumed"] * 2 + ["robustness"] * 12,
        )
        self.assertEqual(
            {row["forward_m"] for row in log[first_rows:]}, {0.10}
        )
        self.assertEqual(resumed["robustness_query_count"], 24)
        self.assertEqual(resumed["pi_ik_requested_count"], len(log))
        self.assertEqual(
            [candidate["forward_m"] for candidate in resumed["ranked_candidates"]],
            [0.05, 0.10],
        )
        self.assertEqual(
            resumed["ranked_candidates"][1]["selection_quality"],
            "strict_pi_robustness_ranked",
        )
        self.assertEqual(
            [item["pass"] for item in resumed["search_passes"]],
            ["initial", "resumed"],
        )
        # Every shortlisted pair has been queried, so nothing is left.
        self.assertIsNone(resumed["continue_search"])

    def test_robust_pi_certificate_uses_a_fixed_seven_pose_bundle(self) -> None:
        hardware = FakeHardware(clearance_m=0.9, arms=ARMS)
        detector = _ScriptedDetector([(27.0, 15.0, 37.0, 33.0)])
        controller, environment = _controller(
            hardware,
            detector,
            settings={
                "candidate_forward_offsets_m": [0.05],
                "candidate_lateral_offsets_m": [0.0],
                "candidate_yaw_offsets_deg": [0.0],
                "candidate_base_limit": 1,
                "grasp_candidate_limit": 1,
                "pi_ik_shortlist_base_limit": 1,
                "pi_ik_grasps_per_base": 1,
                "pi_ik_candidate_limit": 7,
                "ik_batch_size": 7,
                "robustness_enabled": True,
            },
        )
        self.addCleanup(environment.safe_shutdown)
        generated = controller.manipulation.generate_grasp_candidates(
            "marker", "left"
        )
        frame = environment.navigation_frame(max_age_s=1.0)

        evaluation = controller._evaluate_candidates(generated, frame)

        self.assertEqual(evaluation["pi_ik_requested_count"], 7)
        self.assertEqual(evaluation["robustness_variant_count"], 7)
        self.assertTrue(evaluation["selected"]["strict_ready"])
        self.assertEqual(
            evaluation["selected"]["selection_quality"],
            "strict_pi_robustness_ranked",
        )
        selected = evaluation["selected"]["selected_grasp"]
        self.assertEqual(selected["robustness_converged_variants"], 7)
        requested = np.concatenate(
            [
                np.asarray(request["poses_xyz_rpy"], dtype=np.float64)
                for request in hardware.arm_plan_requests[-2:]
            ],
            axis=0,
        )
        self.assertEqual(requested.shape, (7, 6))
        self.assertGreater(np.max(np.ptp(requested[:, :3], axis=0)), 0.0)

    def test_robust_pi_keeps_nominal_success_when_perturbations_fail(self) -> None:
        hardware = FakeHardware(clearance_m=0.9, arms=ARMS)
        detector = _ScriptedDetector([(27.0, 15.0, 37.0, 33.0)])
        controller, environment = _controller(
            hardware,
            detector,
            settings={
                "candidate_forward_offsets_m": [0.05],
                "candidate_lateral_offsets_m": [0.0],
                "candidate_yaw_offsets_deg": [0.0],
                "candidate_base_limit": 1,
                "grasp_candidate_limit": 1,
                "pi_ik_shortlist_base_limit": 1,
                "pi_ik_grasps_per_base": 1,
                "pi_ik_candidate_limit": 7,
                "ik_batch_size": 7,
                "robustness_enabled": True,
                "robustness_preferred_converged_variants": 5,
            },
        )
        self.addCleanup(environment.safe_shutdown)

        def only_four_variants_converge(arm, poses_xyz_rpy):
            del arm
            return {
                "plans": [
                    {
                        "candidate_index": index,
                        "success": True,
                        "ik_converged": (
                            len(poses_xyz_rpy) == 1 or index < 3
                        ),
                        "ik_position_error_m": 0.001,
                        "ik_rotation_error_rad": 0.001,
                        "ik_joint_target": [0.0] * 7,
                        "joint_travel_l2_rad": 0.1,
                        "joint_travel_max_rad": 0.05,
                    }
                    for index in range(len(poses_xyz_rpy))
                ]
            }

        environment.plan_arm_poses = only_four_variants_converge
        generated = controller.manipulation.generate_grasp_candidates(
            "marker", "left"
        )
        frame = environment.navigation_frame(max_age_s=1.0)

        evaluation = controller._evaluate_candidates(generated, frame)

        self.assertEqual(evaluation["bases_with_nominal_converged_grasp"], 1)
        self.assertTrue(evaluation["selected"]["strict_ready"])
        self.assertEqual(
            evaluation["selected"]["selected_grasp"][
                "robustness_converged_variants"
            ],
            4,
        )

    def test_virtual_lattice_keeps_the_negative_forward_candidate(self) -> None:
        hardware = FakeHardware(clearance_m=0.9, arms=ARMS)
        detector = _ScriptedDetector([(27.0, 15.0, 37.0, 33.0)])
        controller, environment = _controller(
            hardware,
            detector,
            settings={
                "candidate_forward_offsets_m": [-0.10, 0.0],
                "candidate_lateral_offsets_m": [0.0],
                "candidate_yaw_offsets_deg": [0.0],
            },
        )
        self.addCleanup(environment.safe_shutdown)
        generated = controller.manipulation.generate_grasp_candidates(
            "marker", "left"
        )
        frame = environment.navigation_frame(max_age_s=1.0)

        evaluation = controller._evaluate_candidates(generated, frame)

        self.assertEqual(
            [candidate["forward_m"] for candidate in evaluation["candidates"]],
            [0.0, -0.10],
        )

    def test_virtual_base_transform_moves_goals_and_scene_consistently(self) -> None:
        hardware = FakeHardware(clearance_m=0.9, arms=ARMS)
        detector = _ScriptedDetector([(27.0, 15.0, 37.0, 33.0)])
        controller, environment = _controller(
            hardware,
            detector,
            settings={
                "candidate_forward_offsets_m": [0.10],
                "candidate_lateral_offsets_m": [-0.10],
                "candidate_yaw_offsets_deg": [10.0],
            },
        )
        self.addCleanup(environment.safe_shutdown)
        generated = controller.manipulation.generate_grasp_candidates(
            "marker", "left"
        )
        frame = environment.navigation_frame(max_age_s=1.0)
        _, _, navigation_from_camera = controller._ground_geometry(frame)

        candidates, grasp_indices = controller._virtual_candidates(
            generated, navigation_from_camera
        )

        candidate = candidates[0]
        grasp_index = grasp_indices[0]
        yaw = np.deg2rad(10.0)
        cosine, sine = np.cos(yaw), np.sin(yaw)
        rotation_navigation = np.asarray(
            [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]]
        )
        camera_offset = np.asarray([0.2143, 0.0603, 0.0])
        expected_camera_translation = navigation_from_camera.T @ (
            np.asarray([0.10, -0.10, 0.0])
            + rotation_navigation @ camera_offset
            - camera_offset
        )
        np.testing.assert_allclose(
            candidate["camera_motion"][:3, 3],
            expected_camera_translation,
            atol=1e-10,
        )
        current_arm_target = (
            np.asarray(generated["arm_from_camera"])
            @ np.asarray(generated["grasps_camera_from_model"])[grasp_index]
            @ np.asarray(generated["model_from_controlled_endpoint"])
        )
        candidate_from_current = np.asarray(
            candidate["candidate_arm_from_current_arm"]
        )
        np.testing.assert_allclose(
            candidate["arm_targets"][0],
            candidate_from_current @ current_arm_target,
            atol=1e-10,
        )
        current_point = np.asarray([0.21, -0.04, 0.55, 1.0])
        np.testing.assert_allclose(
            (candidate_from_current @ current_point)[:3],
            current_point[:3] @ candidate_from_current[:3, :3].T
            + candidate_from_current[:3, 3],
            atol=1e-10,
        )

    def test_holonomic_lateral_target(self) -> None:
        hardware = FakeHardware(clearance_m=0.9, arms=ARMS)
        detector = _ScriptedDetector(
            [
                (45.0, 15.0, 61.0, 33.0),
                (27.0, 15.0, 37.0, 33.0),
            ]
        )
        controller, environment = _controller(
            hardware, detector, settings={"candidate_yaw_offsets_deg": [0.0]}
        )
        self.addCleanup(environment.safe_shutdown)

        def lateral_only_plans(arm, poses_xyz_rpy):
            del arm
            hardware.arm_plan_requests.append(
                {"arm": "left", "poses_xyz_rpy": [list(pose) for pose in poses_xyz_rpy]}
            )
            plans = []
            for index, pose in enumerate(poses_xyz_rpy):
                success = hardware.pose.y_m > 0.02 or float(pose[0]) > 0.04
                plans.append(
                    {
                        "candidate_index": index,
                        "success": success,
                        "ik_converged": success,
                        "ik_position_error_m": 0.001 if success else 1.0,
                        "ik_rotation_error_rad": 0.001 if success else 1.0,
                        "ik_joint_target": [0.0] * 7,
                        "joint_travel_l2_rad": 0.1,
                        "joint_travel_max_rad": 0.05,
                    }
                )
            return {"plans": plans}

        environment.plan_arm_poses = lateral_only_plans
        result = controller.prepare_for_manipulation("pen", "left")

        self.assertTrue(result["success"], result)
        self.assertTrue(any(abs(command[1]) > 1e-5 for command in hardware.commands))
        self.assertTrue(result["motion"]["metrics"]["holonomic"])
        self.assertGreater(result["selected_base_pose"]["left_m"], 0.0)

    def test_arrival_resamples_and_retains_a_certified_goalset(self) -> None:
        hardware = FakeHardware(clearance_m=0.9, arms=ARMS)
        detector = _ScriptedDetector([(27.0, 15.0, 37.0, 33.0)])
        controller, environment = _controller(
            hardware,
            detector,
            settings={
                "candidate_yaw_offsets_deg": [0.0],
                "ik_batch_size": 256,
                "pi_ik_candidate_limit": 256,
            },
        )
        self.addCleanup(environment.safe_shutdown)

        planning_call = 0

        def requires_forward_then_accepts_current(arm, poses_xyz_rpy):
            nonlocal planning_call
            del arm
            planning_call += 1
            hardware.arm_plan_requests.append(
                {
                    "arm": "left",
                    "poses_xyz_rpy": [list(pose) for pose in poses_xyz_rpy],
                }
            )
            plans = []
            for index, pose in enumerate(poses_xyz_rpy):
                # Only forward-moved virtual poses (fixture TCP z < 0.88 m)
                # converge on the virtual grid, so the base must move and the
                # actual arrival pose is certified by a second Pi request.
                success = planning_call > 1 or float(pose[2]) < 0.88
                plans.append(
                    {
                        "candidate_index": index,
                        "success": success,
                        "ik_converged": success,
                        "ik_position_error_m": 0.001 if success else 1.0,
                        "ik_rotation_error_rad": 0.001 if success else 1.0,
                        "ik_joint_target": [0.0] * 7,
                        "joint_travel_l2_rad": 0.1,
                        "joint_travel_max_rad": 0.05,
                    }
                )
            return {"plans": plans}

        environment.plan_arm_poses = requires_forward_then_accepts_current

        result = controller.prepare_for_manipulation("pen", "left")

        self.assertTrue(result["success"], result)
        self.assertFalse(result["requires_fresh_grasp_sampling"])
        self.assertTrue(result["certified_goalset_available"])
        self.assertNotIn("grasp_position", result)
        self.assertNotIn("grasp_quaternion_wxyz", result)
        self.assertIn("certification", result)
        self.assertEqual(result["certification"]["source"], "fresh_sample")
        self.assertEqual(result["selected_base_pose"]["forward_m"], 0.05)
        # One virtual-grid batch plus one fresh actual-pose strict Pi request.
        # prepare does not run the old extra cuRobo-seed validation request.
        self.assertEqual(planning_call, 2)
        backend = controller.manipulation._grasp_client()
        backend_calls = backend.calls

        position, quaternion = controller.manipulation.sample_grasp_pose(
            "pen", "left"
        )

        self.assertEqual(backend.calls, backend_calls)
        self.assertEqual(position.shape, (3,))
        self.assertEqual(quaternion.shape, (4,))
        self.assertIn("left", controller.manipulation._sampled_grasp_goalsets)
        certified_goals = controller.manipulation._sampled_grasp_goalsets["left"][
            "tcp_poses"
        ].copy()

        execution = controller.manipulation.goto_grasp_pose(
            "pen", position, quaternion, "left"
        )

        self.assertTrue(execution["success"], execution)
        planner = controller.manipulation._motion_planner_client()
        np.testing.assert_allclose(
            planner.requests[-1]["tcp_poses"], certified_goals
        )

    def test_target_too_far_returns_explicit_docking_recovery(self) -> None:
        hardware = FakeHardware(clearance_m=1.10, arms=ARMS)
        detector = _ScriptedDetector([(27.0, 15.0, 37.0, 33.0)])
        controller, environment = _controller(hardware, detector)
        self.addCleanup(environment.safe_shutdown)

        result = controller.prepare_for_manipulation("box", "left")

        self.assertFalse(result["success"])
        self.assertEqual(result["reason"], "target_too_far_for_local_manipulation")
        self.assertEqual(result["recovery"]["action"], "dock_to_visible_object")
        self.assertIn("Move closer", result["recovery"]["message"])
        self.assertGreater(result["target_distance_m"], result["maximum_distance_m"])

    def test_prepare_never_calls_curobo_even_when_planner_would_reject(self) -> None:
        hardware = FakeHardware(clearance_m=0.9, arms=ARMS)
        detector = _ScriptedDetector([(27.0, 15.0, 37.0, 33.0)])
        planner = _MotionPlanner(success=False)
        controller, environment = _controller(
            hardware, detector, motion_planner=planner
        )
        self.addCleanup(environment.safe_shutdown)

        result = controller.prepare_for_manipulation("box", "left")

        self.assertTrue(result["success"], result)
        self.assertFalse(result["curobo_called"])
        self.assertEqual(planner.requests, [])

    def test_actual_pose_nonconvergence_is_a_structured_failure(self) -> None:
        hardware = FakeHardware(clearance_m=0.9, arms=ARMS)
        detector = _ScriptedDetector([(27.0, 15.0, 37.0, 33.0)])
        controller, environment = _controller(hardware, detector)
        self.addCleanup(environment.safe_shutdown)
        pi_call = 0

        def converge_virtual_only(arm, poses_xyz_rpy):
            nonlocal pi_call
            del arm
            pi_call += 1
            return {
                "plans": [
                    {
                        "candidate_index": index,
                        "success": True,
                        # Forward-moved virtual poses converge; the arrival
                        # certification (second request) converges nothing.
                        "ik_converged": pi_call == 1 and float(pose[2]) < 0.88,
                        "ik_position_error_m": 0.001,
                        "ik_rotation_error_rad": 0.001,
                        "ik_joint_target": [0.0] * 7,
                        "joint_travel_l2_rad": 0.1,
                        "joint_travel_max_rad": 0.05,
                    }
                    for index, pose in enumerate(poses_xyz_rpy)
                ]
            }

        environment.plan_arm_poses = converge_virtual_only
        result = controller.prepare_for_manipulation("box", "left")

        self.assertFalse(result["success"])
        self.assertEqual(result["reason"], "actual_pose_grasp_certification_failed")
        self.assertIn("Pi Mink IK did not converge", result["certification"]["reason"])
        self.assertEqual(result["certification"]["source"], "fresh_sample")
        self.assertEqual(
            controller.last_debug["failed_after_phase"],
            "actual_pose_strict_pi_certification",
        )

    def test_no_motion_reuses_the_virtual_certificate_without_a_second_pi_call(
        self,
    ) -> None:
        hardware = FakeHardware(clearance_m=0.9, arms=ARMS)
        detector = _ScriptedDetector([(27.0, 15.0, 37.0, 33.0)])
        controller, environment = _controller(
            hardware, detector, settings={"candidate_yaw_offsets_deg": [0.0]}
        )
        self.addCleanup(environment.safe_shutdown)

        result = controller.prepare_for_manipulation("pen", "left")

        self.assertTrue(result["success"], result)
        self.assertEqual(result["motion"]["reason"], "selected_current_pose")
        self.assertEqual(result["certification"]["source"], "virtual_evaluation")
        self.assertEqual(
            result["certification"]["reason"], "strict_pi_grasp_goalset_ready"
        )
        self.assertFalse(result["certification"]["curobo_called"])
        self.assertTrue(result["certified_goalset_available"])
        # The current pose is settled by the first Pi batch and no fresh
        # actual-pose request is issued because the base never moved.
        self.assertEqual(len(hardware.arm_plan_requests), 1)
        self.assertTrue(result["pi_ik_early_exit"])
        self.assertEqual(result["ik_query_count"], 32)
        backend = controller.manipulation._grasp_client()
        backend_calls = backend.calls
        segmenter_calls_before = len(hardware.arm_plan_requests)

        position, quaternion = controller.manipulation.sample_grasp_pose(
            "pen", "left"
        )

        # Consumption revalidates the SAM3 target only: no grasp model call and
        # no Pi IK batch.
        self.assertEqual(backend.calls, backend_calls)
        self.assertEqual(len(hardware.arm_plan_requests), segmenter_calls_before)
        selected = np.asarray(result["certification"]["selected_tcp_pose"])
        np.testing.assert_allclose(position, selected[:3, 3])
        goalset = controller.manipulation._sampled_grasp_goalsets["left"]
        self.assertEqual(goalset["object_name"], "pen")
        np.testing.assert_allclose(goalset["tcp_poses"][0], selected)
        self.assertLessEqual(len(goalset["tcp_poses"]), 16)

        execution = controller.manipulation.goto_grasp_pose(
            "pen", position, quaternion, "left"
        )

        self.assertTrue(execution["success"], execution)
        planner = controller.manipulation._motion_planner_client()
        np.testing.assert_allclose(
            planner.requests[-1]["tcp_poses"], goalset["tcp_poses"]
        )

    def test_virtual_certificate_reuse_can_be_disabled(self) -> None:
        hardware = FakeHardware(clearance_m=0.9, arms=ARMS)
        detector = _ScriptedDetector([(27.0, 15.0, 37.0, 33.0)])
        controller, environment = _controller(
            hardware,
            detector,
            settings={
                "candidate_yaw_offsets_deg": [0.0],
                "reuse_virtual_certificate_without_motion": False,
            },
        )
        self.addCleanup(environment.safe_shutdown)

        result = controller.prepare_for_manipulation("pen", "left")

        self.assertTrue(result["success"], result)
        self.assertEqual(result["motion"]["reason"], "selected_current_pose")
        self.assertEqual(result["certification"]["source"], "fresh_sample")
        self.assertEqual(len(hardware.arm_plan_requests), 2)

    def test_virtual_certificate_is_never_used_after_base_motion(self) -> None:
        hardware = FakeHardware(clearance_m=0.9, arms=ARMS)
        detector = _ScriptedDetector([(27.0, 15.0, 37.0, 33.0)])
        controller, environment = _controller(
            hardware, detector, settings={"candidate_yaw_offsets_deg": [0.0]}
        )
        self.addCleanup(environment.safe_shutdown)

        pi_call = 0

        def forward_poses_only(arm, poses_xyz_rpy):
            nonlocal pi_call
            del arm
            pi_call += 1
            return {
                "plans": [
                    {
                        "candidate_index": index,
                        "success": True,
                        # Virtual grid: only forward-moved poses converge, so
                        # the base moves; the actual-pose request converges.
                        "ik_converged": pi_call > 1 or float(pose[2]) < 0.88,
                        "ik_position_error_m": 0.001,
                        "ik_rotation_error_rad": 0.001,
                        "ik_joint_target": [0.0] * 7,
                        "joint_travel_l2_rad": 0.1,
                        "joint_travel_max_rad": 0.05,
                    }
                    for index, pose in enumerate(poses_xyz_rpy)
                ]
            }

        environment.plan_arm_poses = forward_poses_only
        result = controller.prepare_for_manipulation("pen", "left")

        self.assertTrue(result["success"], result)
        self.assertNotEqual(result["motion"]["reason"], "selected_current_pose")
        self.assertEqual(result["selected_base_pose"]["forward_m"], 0.05)
        self.assertEqual(result["certification"]["source"], "fresh_sample")
        self.assertEqual(pi_call, 2)
        self.assertNotIn("virtual_certificate_attempt", controller.last_debug)

    def test_best_tier_early_exit_stops_the_nominal_wave(self) -> None:
        hardware = FakeHardware(clearance_m=0.9, arms=ARMS)
        detector = _ScriptedDetector([(27.0, 15.0, 37.0, 33.0)])
        settings = {
            "candidate_yaw_offsets_deg": [0.0],
            "pi_ik_shortlist_base_limit": 5,
            "pi_ik_grasps_per_base": 2,
            "pi_ik_candidate_limit": 10,
            "ik_batch_size": 4,
        }
        controller, environment = _controller(hardware, detector, settings=settings)
        self.addCleanup(environment.safe_shutdown)
        generated = controller.manipulation.generate_grasp_candidates(
            "marker", "left"
        )
        frame = environment.navigation_frame(max_age_s=1.0)

        evaluation = controller._evaluate_candidates(generated, frame)

        # The current pose's two grasps head the shortlist and both converge,
        # so the wave stops after its first batch of four.
        self.assertTrue(evaluation["pi_ik_early_exit"])
        self.assertEqual(evaluation["pi_ik_early_exit_stage"], "nominal")
        self.assertEqual(evaluation["ik_query_count"], 4)
        self.assertEqual(evaluation["pi_ik_scheduled_nominal_count"], 10)
        self.assertFalse(evaluation["pi_ik_budget_exhausted"])
        self.assertEqual(evaluation["selected"]["movement_cost_tier"], 0)
        self.assertEqual(evaluation["selected"]["converged_grasp_count"], 2)
        first_batch = hardware.arm_plan_requests[0]["poses_xyz_rpy"]
        self.assertEqual([round(pose[2], 3) for pose in first_batch[:2]], [0.9, 0.9])

        environment.safe_shutdown()
        hardware = FakeHardware(clearance_m=0.9, arms=ARMS)
        controller, environment = _controller(
            hardware,
            detector,
            settings={**settings, "pi_ik_best_tier_early_exit": False},
        )
        self.addCleanup(environment.safe_shutdown)
        generated = controller.manipulation.generate_grasp_candidates(
            "marker", "left"
        )
        frame = environment.navigation_frame(max_age_s=1.0)

        evaluation = controller._evaluate_candidates(generated, frame)

        self.assertFalse(evaluation["pi_ik_early_exit"])
        self.assertEqual(evaluation["ik_query_count"], 10)
        self.assertEqual(evaluation["selected"]["movement_cost_tier"], 0)

    def test_early_exit_waits_until_the_nearest_ready_tier_is_settled(
        self,
    ) -> None:
        hardware = FakeHardware(clearance_m=0.9, arms=ARMS)
        detector = _ScriptedDetector([(27.0, 15.0, 37.0, 33.0)])
        controller, environment = _controller(
            hardware,
            detector,
            settings={
                # Fixture TCP z: current 0.90, forward 0.05 -> 0.85, 0.10 -> 0.80.
                "candidate_forward_offsets_m": [0.0, 0.05, 0.10],
                "candidate_lateral_offsets_m": [0.0],
                "candidate_yaw_offsets_deg": [0.0],
                "candidate_base_limit": 3,
                "pi_ik_shortlist_base_limit": 3,
                "pi_ik_grasps_per_base": 2,
                "pi_ik_candidate_limit": 6,
                "pi_ik_nominal_candidate_limit": 6,
                "ik_batch_size": 1,
            },
        )
        self.addCleanup(environment.safe_shutdown)

        requested_z: list[float] = []

        def only_the_five_centimeter_pose_converges(arm, poses_xyz_rpy):
            del arm
            requested_z.extend(round(float(pose[2]), 3) for pose in poses_xyz_rpy)
            return {
                "plans": [
                    {
                        "candidate_index": index,
                        "success": True,
                        "ik_converged": abs(float(pose[2]) - 0.85) < 0.01,
                        "ik_position_error_m": 0.001,
                        "ik_rotation_error_rad": 0.001,
                        "joint_travel_l2_rad": 0.1,
                        "joint_travel_max_rad": 0.05,
                    }
                    for index, pose in enumerate(poses_xyz_rpy)
                ]
            }

        environment.plan_arm_poses = only_the_five_centimeter_pose_converges
        generated = controller.manipulation.generate_grasp_candidates(
            "marker", "left"
        )
        frame = environment.navigation_frame(max_age_s=1.0)

        evaluation = controller._evaluate_candidates(generated, frame)

        # Query order is one whole tier at a time: current x2 (tier 0, both
        # fail), then the 0.05 m pose x2 (tier 2). That tier is settled and
        # ready after the fourth query, so the tier 5 pose at 0.10 m is never
        # queried at all. Interleaving the far tier instead would spend a
        # query on 0.80 before finishing tier 2.
        self.assertEqual(requested_z, [0.9, 0.9, 0.85, 0.85])
        self.assertNotIn(0.8, requested_z)
        self.assertTrue(evaluation["pi_ik_early_exit"])
        self.assertEqual(evaluation["ik_query_count"], 4)
        self.assertEqual(evaluation["selected"]["forward_m"], 0.05)
        self.assertEqual(evaluation["selected"]["converged_grasp_count"], 2)


class ManipulationReadinessViserTest(unittest.TestCase):
    def test_candidate_transform_uses_forward_left_and_yaw(self) -> None:
        transform = ManipulationReadinessViser._base_transform(
            {"forward_m": 0.1, "left_m": -0.2, "yaw_rad": np.pi / 2.0}
        )
        np.testing.assert_allclose(transform[:3, 3], [0.1, -0.2, 0.0])
        np.testing.assert_allclose(
            transform[:3, :3]
            @ np.asarray([1.0, 0.0, 0.0]),
            [0.0, 1.0, 0.0],
            atol=1e-8,
        )


if __name__ == "__main__":
    unittest.main()
