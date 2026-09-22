from __future__ import annotations

import ast
import copy
import inspect
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from fakes import FAKE_MANIPULATION_CONFIG, FakeHardware, make_environment

from yor_agent import launch as launch_module
from yor_agent.launch import load_config
from yor_agent.nav2_params import (
    INFLATION_OUTSIDE_FOOTPRINT_M,
    polygon_footprint,
    render_nav2_parameters,
)
from yor_agent.primitive_config import (
    PRIMITIVE_NAMES,
    apply_primitive_overrides,
    load_primitive_config,
    normalize_primitive_config,
    primitive_exposed,
    primitive_settings,
)
from yor_agent.primitives.manipulation import register_manipulation_primitives
from yor_agent.primitives.navigation import register_navigation_primitives
from yor_agent.primitives.registry import PrimitiveRegistry
from yor_agent.robot.visible_object_navigation import VisibleObjectDockingController


CONFIG_DIRECTORY = Path(__file__).resolve().parents[1] / "configs"
NAV2_PARAMETERS = (
    Path(__file__).resolve().parents[1]
    / "ros2_ws/src/yor_nav2_bridge/config/nav2_params.yaml"
)
GRASP_MOTION_DIRECTORY = Path(__file__).resolve().parents[1] / "services/grasp_motion"
GRASP_MOTION_CONFIG = (
    Path(__file__).resolve().parents[1]
    / ".runtime/grasp_motion/nero_with_official_gripper.yml"
)


class PrimitiveConfigLoadingTest(unittest.TestCase):
    def test_grasp_motion_keeps_all_installed_gripper_links_collision_active(
        self,
    ) -> None:
        payload = yaml.safe_load(GRASP_MOTION_CONFIG.read_text(encoding="utf-8"))
        kinematics = payload["robot_cfg"]["kinematics"]
        gripper_links = {
            "gripper_flange",
            "gripper_base",
            "gripper_link1",
            "gripper_link2",
        }
        arm_links = {"base_link", *(f"link{index}" for index in range(1, 8))}

        self.assertEqual(kinematics["grasp_contact_link_names"], [])
        self.assertTrue(gripper_links.issubset(kinematics["collision_link_names"]))
        self.assertTrue(gripper_links.issubset(kinematics["collision_spheres"]))
        self.assertEqual(kinematics["collision_sphere_buffer"], 0.0)
        self.assertTrue(
            all(
                sphere["radius"] <= 0.015
                for link_name in gripper_links
                for sphere in kinematics["collision_spheres"][link_name]
            )
        )
        arm_spheres = [
            sphere
            for link_name in arm_links
            for sphere in kinematics["collision_spheres"][link_name]
        ]
        self.assertEqual(len(arm_spheres), 124)
        self.assertTrue(
            all(sphere["radius"] <= 0.040 for sphere in arm_spheres)
        )

        service_source = (GRASP_MOTION_DIRECTORY / "service.py").read_text(
            encoding="utf-8"
        )
        tree = ast.parse(service_source)
        grasp_planner_calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "plan_grasp"
            and any(
                keyword.arg == "disable_collision_links"
                for keyword in node.keywords
            )
        ]
        # The single execution planner calls plan_pose directly on the complete
        # pre-grasp goalset and never disables a gripper collision link. The
        # removed multi-environment batch planner was the only plan_grasp call.
        self.assertEqual(grasp_planner_calls, [])

    def test_grasp_motion_uses_only_the_single_environment_planner(self) -> None:
        service_source = (GRASP_MOTION_DIRECTORY / "service.py").read_text(
            encoding="utf-8"
        )

        self.assertNotIn("BatchMotionPlanner", service_source)
        self.assertNotIn("plan_grasp_batch", service_source)

    def test_grasp_motion_plans_the_complete_pregrasp_goalset(self) -> None:
        service_source = (GRASP_MOTION_DIRECTORY / "service.py").read_text(
            encoding="utf-8"
        )
        tree = ast.parse(service_source)
        plan_grasp = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "plan_grasp"
        )
        # One helper builds a goalset of however many pre-grasps it is given.
        goalset_builds = [
            node
            for node in ast.walk(plan_grasp)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "from_poses"
            and any(
                keyword.arg == "num_goalset"
                and isinstance(keyword.value, ast.Call)
                and isinstance(keyword.value.func, ast.Name)
                and keyword.value.func.id == "len"
                for keyword in node.keywords
            )
        ]
        self.assertEqual(len(goalset_builds), 1)

        def pregrasp_plans(accepts_argument):
            return [
                node
                for node in ast.walk(plan_grasp)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "plan_pose"
                and node.args
                and isinstance(node.args[0], ast.Call)
                and isinstance(node.args[0].func, ast.Name)
                and node.args[0].func.id == "pregrasp_goal"
                and node.args[0].args
                and accepts_argument(node.args[0].args[0])
            ]

        # Every pre-grasp is planned together once, then one at a time only
        # when that goalset reaches none of them.
        all_pregrasp_plans = pregrasp_plans(
            lambda argument: isinstance(argument, ast.Name)
            and argument.id == "pregrasp_targets"
        )
        self.assertEqual(len(all_pregrasp_plans), 1)
        single_pregrasp_plans = pregrasp_plans(
            lambda argument: isinstance(argument, ast.Subscript)
            and isinstance(argument.value, ast.Name)
            and argument.value.id == "pregrasp_targets"
        )
        self.assertEqual(len(single_pregrasp_plans), 1)

    def test_shipped_file_hides_the_coarse_navigation_vocabulary(self) -> None:
        from yor_agent.primitives.coarse_navigation import COARSE_NAVIGATION_PRIMITIVES

        config = load_primitive_config(CONFIG_DIRECTORY / "primitive_config.yaml")

        for name in COARSE_NAVIGATION_PRIMITIVES:
            self.assertIn(name, PRIMITIVE_NAMES)
            self.assertFalse(primitive_exposed(config, name), name)

    def test_overrides_change_only_the_keys_they_name(self) -> None:
        config = load_primitive_config(CONFIG_DIRECTORY / "primitive_config.yaml")
        original = copy.deepcopy(config)

        overridden = apply_primitive_overrides(
            config,
            {
                "drive_straight": {"exposed": False, "defaults": {"timeout_s": 30}},
                "go_forward": {"exposed": True},
                "get_object_pose": {"settings": {"depth_retry_count": 1}},
            },
        )

        self.assertFalse(primitive_exposed(overridden, "drive_straight"))
        self.assertEqual(
            overridden["primitives"]["drive_straight"]["defaults"],
            {**config["primitives"]["drive_straight"]["defaults"], "timeout_s": 30.0},
        )
        self.assertTrue(primitive_exposed(overridden, "go_forward"))
        self.assertEqual(primitive_settings(overridden, "get_object_pose")["depth_retry_count"], 1)
        self.assertEqual(
            overridden["primitives"]["prepare_for_manipulation"],
            config["primitives"]["prepare_for_manipulation"],
        )
        self.assertEqual(config, original)
        for bad in (
            {"no_such_primitive": {"exposed": False}},
            {"go_forward": {"settings": {}}},
            {"go_forward": {"defaults": {"timeout_s": 60}}},
            {"drive_straight": {"exposed": "no"}},
            {"drive_straight": []},
        ):
            with self.subTest(overrides=bad), self.assertRaises((TypeError, ValueError)):
                apply_primitive_overrides(config, bad)

    def test_shipped_file_lists_every_configured_robot_primitive(self) -> None:
        config = load_primitive_config(CONFIG_DIRECTORY / "primitive_config.yaml")

        self.assertEqual(tuple(config["primitives"]), PRIMITIVE_NAMES)
        docking = primitive_settings(config, "dock_to_visible_object")
        self.assertEqual(docking["backend"], "nav2")
        self.assertEqual(docking["docking_distance_m"], 0.65)
        self.assertEqual(docking["nav2_goal_distance_m"], 0.60)
        self.assertEqual(docking["robot_radius_m"], 0.30)
        self.assertEqual(docking["base_to_camera_forward_m"], 0.2143)
        self.assertEqual(docking["base_to_camera_left_m"], 0.0603)
        self.assertEqual(docking["nav2_frame_id"], "odom")
        self.assertEqual(docking["nav2_action_name"], "navigate_to_pose")
        self.assertIn("navigate_no_reverse.xml", docking["behavior_tree"])
        self.assertTrue(docking["require_dynamic_ground_plane"])
        self.assertTrue(primitive_exposed(config, "prepare_for_manipulation"))
        self.assertTrue(primitive_exposed(config, "open_gripper"))
        self.assertFalse(primitive_exposed(config, "lift_grasped_object"))
        self.assertFalse(primitive_settings(config, "open_gripper")["simulated"])
        self.assertFalse(primitive_settings(config, "close_gripper")["simulated"])
        self.assertEqual(
            primitive_settings(config, "get_object_pose")["depth_retry_count"], 2
        )
        self.assertFalse(
            config["primitives"]["get_object_pose"]["defaults"][
                "return_zed_distance"
            ]
        )
        for name in (
            "get_object_pose",
            "sample_grasp_pose",
            "goto_pose",
            "goto_grasp_pose",
            "open_gripper",
            "close_gripper",
            "lift_grasped_object",
            "prepare_for_manipulation",
        ):
            self.assertNotIn("arm", config["primitives"][name]["defaults"])
        self.assertEqual(
            primitive_settings(config, "sample_grasp_pose")["depth_retry_count"], 2
        )
        readiness = primitive_settings(config, "prepare_for_manipulation")
        self.assertEqual(
            readiness["candidate_forward_offsets_m"],
            [-0.10, -0.05, 0.0, 0.05, 0.10, 0.15, 0.20],
        )
        self.assertEqual(readiness["grasp_candidate_limit"], 64)
        self.assertEqual(readiness["candidate_base_limit"], 729)
        self.assertEqual(readiness["pi_ik_shortlist_base_limit"], 128)
        self.assertEqual(readiness["pi_ik_grasps_per_base"], 4)
        self.assertEqual(readiness["pi_ik_candidate_limit"], 1024)
        self.assertEqual(readiness["pi_ik_nominal_candidate_limit"], 512)
        self.assertEqual(readiness["ik_batch_size"], 32)
        self.assertEqual(readiness["pi_ik_compute_budget_s"], 30.0)
        self.assertEqual(readiness["motion_alternative_limit"], 8)
        self.assertIs(readiness["continue_search_after_refused_motions"], True)
        self.assertTrue(readiness["robustness_enabled"])
        self.assertEqual(readiness["robustness_preferred_converged_variants"], 3)
        self.assertEqual(readiness["candidate_translation_bucket_m"], 0.020)
        self.assertEqual(readiness["candidate_yaw_bucket_deg"], 2.0)
        self.assertEqual(readiness["position_tolerance_m"], 0.030)
        self.assertEqual(readiness["yaw_tolerance_deg"], 3.0)
        self.assertFalse(any(key.startswith("refinement") for key in readiness))
        self.assertEqual(readiness["base_to_camera_forward_m"], 0.2143)
        self.assertNotIn("curobo_top_k", readiness)
        self.assertNotIn("curobo_goalset_limit", readiness)
        self.assertFalse(any(key.startswith("fast_detector") for key in readiness))
        self.assertFalse(any(key.startswith("coarse_") for key in readiness))

    @staticmethod
    def _rendered_footprints(nav2):
        footprint_strings = (
            nav2["local_costmap"]["local_costmap"]["ros__parameters"]["footprint"],
            nav2["global_costmap"]["global_costmap"]["ros__parameters"]["footprint"],
        )
        footprints = [json.loads(value) for value in footprint_strings]
        stop_points = nav2["collision_monitor_arms"]["ros__parameters"][
            "ArmsStop"
        ]["points"]
        footprints.append(
            [[x_m, y_m] for x_m, y_m in zip(stop_points[::2], stop_points[1::2])]
        )
        return footprints

    def test_nav2_footprints_match_the_shipped_footprint_polygon(self) -> None:
        # The shipped footprint is the rest pose's envelope as a polygon, so
        # Nav2 keeps the elbows and the chassis rear out of walls, which the
        # ZED-centred circle could not without pushing its front past the
        # grippers. Both costmaps and the collision monitor must agree on it.
        config = load_primitive_config(CONFIG_DIRECTORY / "primitive_config.yaml")
        docking = primitive_settings(config, "dock_to_visible_object")
        template = yaml.safe_load(NAV2_PARAMETERS.read_text())
        nav2 = render_nav2_parameters(config, template)
        center = (
            docking["base_to_camera_forward_m"],
            docking["base_to_camera_left_m"],
        )
        expected = polygon_footprint(docking["robot_footprint_xy"])

        footprints = self._rendered_footprints(nav2)

        self.assertTrue(all(footprint == footprints[0] for footprint in footprints))
        self.assertEqual(footprints[0], expected)
        xs = [x_m for x_m, _ in expected]
        ys = [y_m for _, y_m in expected]
        # Front covers the arms' forward reach (0.457 m), sides their half
        # width (0.433 m), rear the chassis (0.22 m), each with the repo's
        # 0.05 m margin. These come from the deployed collision-sphere
        # envelope of the travel pose, which is the only measurement of the
        # arms there has ever been, and they must not be padded beyond it:
        # a 1.20 m span that was never measured once made this 1.27 m wide
        # and refused aisles the robot fits through (2026-09-08).
        self.assertGreaterEqual(max(xs), 0.457 + 0.05 - 1e-9)
        self.assertLess(max(xs), 0.457 + 0.10)
        self.assertGreaterEqual(max(ys), 0.433 + 0.05 - 1e-9)
        self.assertLess(max(ys), 0.433 + 0.10)
        self.assertLessEqual(min(ys), -(0.433 + 0.05) + 1e-9)
        self.assertLessEqual(min(xs), -(0.22 + 0.05) + 1e-9)
        circumscribed = max(math.hypot(x_m, y_m) for x_m, y_m in expected)
        self.assertEqual(
            nav2["local_costmap"]["local_costmap"]["ros__parameters"][
                "inflation_layer"
            ]["inflation_radius"],
            round(circumscribed + INFLATION_OUTSIDE_FOOTPRINT_M, 6),
        )
        self.assertEqual(
            nav2["yor_zed_bridge"]["ros__parameters"][
                "base_to_camera_forward_m"
            ],
            center[0],
        )

    def test_nav2_falls_back_to_the_circle_without_a_polygon(self) -> None:
        config = load_primitive_config(CONFIG_DIRECTORY / "primitive_config.yaml")
        config = copy.deepcopy(config)
        settings = config["primitives"]["dock_to_visible_object"]["settings"]
        settings.pop("robot_footprint_xy")
        template = yaml.safe_load(NAV2_PARAMETERS.read_text())
        center = (settings["base_to_camera_forward_m"], settings["base_to_camera_left_m"])
        radius = settings["robot_radius_m"]

        nav2 = render_nav2_parameters(config, template)
        footprints = self._rendered_footprints(nav2)

        self.assertTrue(all(footprint == footprints[0] for footprint in footprints))
        for x_m, y_m in footprints[0]:
            self.assertAlmostEqual(
                math.hypot(x_m - center[0], y_m - center[1]), radius, delta=1e-4
            )
        self.assertEqual(
            nav2["local_costmap"]["local_costmap"]["ros__parameters"][
                "inflation_layer"
            ]["inflation_radius"],
            round(radius + INFLATION_OUTSIDE_FOOTPRINT_M, 6),
        )

    def test_polygon_footprint_is_a_convex_hull_in_any_vertex_order(self) -> None:
        scrambled = [[0.49, -0.51], [-0.27, 0.51], [0.0, 0.0], [0.49, 0.51], [-0.27, -0.51]]

        hull = polygon_footprint(scrambled)

        self.assertEqual(len(hull), 4)
        self.assertNotIn([0.0, 0.0], hull)
        with self.assertRaises(ValueError):
            polygon_footprint([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        with self.assertRaises(ValueError):
            polygon_footprint([[0.0, 0.0], [1.0, float("nan")], [0.0, 1.0]])

    def test_changed_primitive_radius_changes_every_nav2_footprint(self) -> None:
        config = load_primitive_config(CONFIG_DIRECTORY / "primitive_config.yaml")
        config = copy.deepcopy(config)
        settings = config["primitives"]["dock_to_visible_object"]["settings"]
        settings.pop("robot_footprint_xy")
        settings["robot_radius_m"] = 0.42
        template = yaml.safe_load(NAV2_PARAMETERS.read_text())

        nav2 = render_nav2_parameters(config, template)

        local = json.loads(
            nav2["local_costmap"]["local_costmap"]["ros__parameters"]["footprint"]
        )
        center = (
            settings["base_to_camera_forward_m"],
            settings["base_to_camera_left_m"],
        )
        for x_m, y_m in local:
            self.assertAlmostEqual(
                math.hypot(x_m - center[0], y_m - center[1]), 0.42, delta=1e-5
            )
        self.assertEqual(
            nav2["global_costmap"]["global_costmap"]["ros__parameters"][
                "footprint"
            ],
            nav2["local_costmap"]["local_costmap"]["ros__parameters"][
                "footprint"
            ],
        )
        self.assertEqual(
            nav2["local_costmap"]["local_costmap"]["ros__parameters"][
                "inflation_layer"
            ]["inflation_radius"],
            round(0.42 + INFLATION_OUTSIDE_FOOTPRINT_M, 6),
        )

    def test_structure_monitor_only_truncates_the_envelope_in_front(self) -> None:
        # Splitting by height must not cost any protection except the forward
        # strip the arms actually occupy. The structure outline every point is
        # judged against is therefore the arms envelope with its front cut
        # back to the body (body_polygon_xy grown by the gate's structure
        # margin) and its width and rear left exactly as they are, so the
        # elbows, which reach sideways at desk height without reaching
        # forward, keep their full lateral coverage.
        primitive_config = load_primitive_config(
            CONFIG_DIRECTORY / "primitive_config.yaml"
        )
        robot_config = yaml.safe_load(
            (CONFIG_DIRECTORY / "mobile_manipulation.yaml").read_text()
        )
        navigation = robot_config["robot"]["navigation"]
        body = navigation["footprint"]["body_polygon_xy"]
        margin = float(navigation["arm_footprint_structure_margin_m"])
        template = yaml.safe_load(NAV2_PARAMETERS.read_text())

        nav2 = render_nav2_parameters(primitive_config, template, robot_config)
        structure = nav2["collision_monitor_structure"]["ros__parameters"][
            "StructureStop"
        ]["points"]
        arms = nav2["collision_monitor_arms"]["ros__parameters"]["ArmsStop"][
            "points"
        ]

        body_front = max(x for x, _ in body) + margin
        self.assertAlmostEqual(max(structure[::2]), body_front, places=6)
        self.assertLess(max(structure[::2]), max(arms[::2]))
        # No lateral or rear regression against today's single polygon.
        self.assertAlmostEqual(max(structure[1::2]), max(arms[1::2]), places=6)
        self.assertAlmostEqual(min(structure[1::2]), min(arms[1::2]), places=6)
        self.assertAlmostEqual(min(structure[::2]), min(arms[::2]), places=6)
        self.assertEqual(
            nav2["yor_zed_bridge"]["ros__parameters"]["structure_footprint_xy"],
            structure,
        )

    def test_collision_monitors_are_chained_with_split_sources(self) -> None:
        primitive_config = load_primitive_config(
            CONFIG_DIRECTORY / "primitive_config.yaml"
        )
        robot_config = yaml.safe_load(
            (CONFIG_DIRECTORY / "mobile_manipulation.yaml").read_text()
        )
        template = yaml.safe_load(NAV2_PARAMETERS.read_text())

        nav2 = render_nav2_parameters(primitive_config, template, robot_config)
        structure = nav2["collision_monitor_structure"]["ros__parameters"]
        arms = nav2["collision_monitor_arms"]["ros__parameters"]
        zed = nav2["yor_zed_bridge"]["ros__parameters"]

        # cmd_vel -> structure -> arms -> the Pi bridge, so a restriction from
        # either monitor survives to the base.
        self.assertEqual(structure["cmd_vel_in_topic"], "cmd_vel")
        self.assertEqual(
            arms["cmd_vel_in_topic"], structure["cmd_vel_out_topic"]
        )
        self.assertEqual(arms["cmd_vel_out_topic"], "/yor/cmd_vel_safe")
        # The structure monitor sees every point; the arms monitor sees only
        # what the ZED bridge measured at arm height.
        self.assertEqual(structure["zed_points"]["topic"], "/yor/zed/points")
        self.assertEqual(
            arms["zed_points_arms"]["topic"], zed["arm_band_cloud_topic"]
        )
        self.assertNotEqual(
            arms["zed_points_arms"]["topic"], structure["zed_points"]["topic"]
        )
        # An approach polygon can only take a footprint topic in Humble.
        self.assertEqual(
            structure["StructureApproach"]["footprint_topic"],
            zed["structure_footprint_topic"],
        )
        self.assertEqual(
            arms["ArmsApproach"]["footprint_topic"],
            "/local_costmap/published_footprint",
        )
        self.assertNotIn("points", structure["StructureApproach"])
        self.assertNotIn("points", arms["ArmsApproach"])
        # The band the bridge applies is the gate's, not a second copy.
        navigation = robot_config["robot"]["navigation"]
        self.assertEqual(zed["arm_band_m"], navigation["arm_footprint_band_m"])
        # The arms monitor bands with its own vertical pad, at least as wide as
        # the direct-motion gate's, so it sees a surface just below the arms.
        self.assertEqual(
            zed["arm_band_z_margin_m"], navigation["nav2_arm_band_z_margin_m"]
        )
        self.assertGreaterEqual(
            navigation["nav2_arm_band_z_margin_m"], navigation["arm_footprint_z_margin_m"]
        )
        # The bridge's floor-plane gate is centred on the docking fallback plane.
        docking = primitive_config["primitives"]["dock_to_visible_object"]["settings"]
        self.assertEqual(zed["fallback_camera_height_m"], docking["ground_camera_height_m"])
        self.assertEqual(zed["fallback_ground_down_xyz"], docking["ground_down_camera_xyz"])
        self.assertEqual(
            zed["ground_plane_max_height_error_m"],
            docking["ground_plane_max_height_error_m"],
        )
        self.assertEqual(
            zed["arm_band_point_tolerance_m"], navigation["layer_z_tolerance_m"]
        )
        self.assertEqual(zed["arm_band_z_min_m"], 0.27)
        self.assertEqual(zed["arm_band_z_max_m"], 1.45)
        self.assertEqual(len(zed["structure_footprint_xy"]), 8)

    def test_nav2_renderer_rejects_a_structure_outline_outside_the_arms(
        self,
    ) -> None:
        config = copy.deepcopy(
            load_primitive_config(CONFIG_DIRECTORY / "primitive_config.yaml")
        )
        settings = config["primitives"]["dock_to_visible_object"]["settings"]
        settings["robot_structure_footprint_xy"] = [
            [0.90, 0.32],
            [-0.27, 0.32],
            [-0.27, -0.32],
            [0.90, -0.32],
        ]
        template = yaml.safe_load(NAV2_PARAMETERS.read_text())

        with self.assertRaisesRegex(ValueError, "must lie inside"):
            render_nav2_parameters(config, template)

    def test_nav2_renderer_fails_closed_without_both_monitors(self) -> None:
        config = load_primitive_config(CONFIG_DIRECTORY / "primitive_config.yaml")
        template = yaml.safe_load(NAV2_PARAMETERS.read_text())
        template.pop("collision_monitor_arms")

        with self.assertRaisesRegex(ValueError, "collision_monitor"):
            render_nav2_parameters(config, template)

    def test_nav2_renderer_injects_robot_self_filter_configuration(self) -> None:
        primitive_config = load_primitive_config(
            CONFIG_DIRECTORY / "primitive_config.yaml"
        )
        robot_config = yaml.safe_load(
            (CONFIG_DIRECTORY / "mobile_manipulation.yaml").read_text()
        )
        template = yaml.safe_load(NAV2_PARAMETERS.read_text())

        nav2 = render_nav2_parameters(
            primitive_config, template, robot_config
        )
        zed = nav2["yor_zed_bridge"]["ros__parameters"]

        self.assertTrue(zed["self_filter_enabled"])
        self.assertEqual(zed["self_filter_arm_status_poll_hz"], 2.0)
        self.assertEqual(zed["arm_rpc_host"], "192.168.1.10")
        self.assertEqual(len(zed["left_arm_from_camera"]), 16)
        self.assertIn(
            "nero_with_official_gripper.yml",
            zed["self_filter_spheres_path"],
        )

    def test_shipped_readiness_prior_block_is_off_and_round_trips(self) -> None:
        from yor_agent.robot.readiness_prior import ReadinessPriorConfig

        config = load_primitive_config(CONFIG_DIRECTORY / "primitive_config.yaml")
        docking = primitive_settings(config, "dock_to_visible_object")

        self.assertEqual(
            docking["readiness_prior"],
            {
                "enabled": False,
                "events_path": None,
                "source": "ready_frame",
                "method": "vggt",
                "vggt_service_url": "http://127.0.0.1:8117",
                "vggt_timeout_s": 60.0,
                "vggt_frames": ["nav", "ready"],
                "stance_offset_m": 0.30,
                "near_object_m": 0.30,
                "min_height_m": 1.2,
                "max_height_m": 2.1,
                "max_scale_iqr_ratio": 2.0,
                "min_inliers": 30,
                "max_reprojection_error_px": 8.0,
                "max_bearing_error_deg": 45.0,
                "retry_bearing_offsets_deg": [45.0, -45.0],
            },
        )
        prior = ReadinessPriorConfig.from_mapping(docking["readiness_prior"])
        self.assertFalse(prior.enabled)
        self.assertIsNone(prior.events_path)
        self.assertEqual(prior.source, "ready_frame")
        self.assertEqual(prior.method, "vggt")
        self.assertEqual(prior.vggt_service_url, "http://127.0.0.1:8117")
        self.assertEqual(prior.vggt_frames, ("nav", "ready"))
        self.assertEqual(prior.retry_bearing_offsets_deg, (45.0, -45.0))
        # The block is the launch wiring's, not the docking backend's: the
        # shipped Nav2 config still loads with it present.
        self.assertEqual(docking["backend"], "nav2")

    def test_unknown_readiness_prior_key_is_rejected_at_load(self) -> None:
        payload = yaml.safe_load(
            (CONFIG_DIRECTORY / "primitive_config.yaml").read_text(encoding="utf-8")
        )
        block = payload["primitives"]["dock_to_visible_object"]["settings"][
            "readiness_prior"
        ]
        block["invented_setting"] = 1
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "primitive_config.yaml"
            path.write_text(yaml.safe_dump(payload))

            with self.assertRaisesRegex(ValueError, "unknown readiness_prior"):
                load_primitive_config(path)

    def test_readiness_prior_block_is_validated_for_either_backend(self) -> None:
        for backend in ("nav2", "legacy"):
            with self.subTest(backend=backend), self.assertRaisesRegex(
                ValueError, "unknown readiness_prior"
            ):
                normalize_primitive_config(
                    {
                        "version": 1,
                        "primitives": {
                            "dock_to_visible_object": {
                                "settings": {
                                    "backend": backend,
                                    "readiness_prior": {
                                        "enabled": False,
                                        "invented_setting": 1,
                                    },
                                }
                            }
                        },
                    }
                )
        with self.assertRaisesRegex(ValueError, "events_path"):
            normalize_primitive_config(
                {
                    "version": 1,
                    "primitives": {
                        "dock_to_visible_object": {
                            "settings": {
                                "backend": "nav2",
                                "readiness_prior": {"enabled": True},
                            }
                        }
                    },
                }
            )

    def test_readiness_prior_block_must_be_a_mapping(self) -> None:
        with self.assertRaisesRegex(TypeError, "readiness_prior must be a mapping"):
            normalize_primitive_config(
                {
                    "version": 1,
                    "primitives": {
                        "dock_to_visible_object": {
                            "settings": {"backend": "nav2", "readiness_prior": "yes"}
                        }
                    },
                }
            )

    def test_task_config_resolves_primitive_file_relative_to_itself(self) -> None:
        config = load_config(CONFIG_DIRECTORY / "mobile_manipulation.yaml")

        self.assertEqual(
            tuple(config["primitive_config"]["primitives"]), PRIMITIVE_NAMES
        )
        self.assertEqual(config["robot"]["navigation"]["min_front_clearance_m"], 0.50)
        manipulation = config["robot"]["manipulation"]
        self.assertEqual(manipulation["grasp_ik_acceptable_position_error_m"], 0.020)
        self.assertEqual(manipulation["grasp_ik_acceptable_rotation_error_rad"], 0.10)

    def test_complete_file_rejects_a_missing_primitive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "primitive_config.yaml"
            path.write_text("version: 1\nprimitives:\n  observe:\n    defaults: {}\n")

            with self.assertRaisesRegex(ValueError, "is missing"):
                load_primitive_config(path)

    def test_unknown_default_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown defaults"):
            normalize_primitive_config(
                {
                    "version": 1,
                    "primitives": {
                        "stop": {"defaults": {"invented_parameter": 1}}
                    },
                }
            )

    def test_manipulation_arm_default_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown defaults"):
            normalize_primitive_config(
                {
                    "version": 1,
                    "primitives": {
                        "get_object_pose": {"defaults": {"arm": 0}}
                    },
                }
            )

    def test_exposed_must_be_boolean(self) -> None:
        with self.assertRaisesRegex(TypeError, "exposed must be boolean"):
            normalize_primitive_config(
                {
                    "version": 1,
                    "primitives": {
                        "prepare_for_manipulation": {"exposed": "false"}
                    },
                }
            )

    def test_simulated_gripper_setting_must_be_boolean(self) -> None:
        with self.assertRaisesRegex(TypeError, "simulated must be boolean"):
            normalize_primitive_config(
                {
                    "version": 1,
                    "primitives": {
                        "open_gripper": {
                            "settings": {"simulated": "true"}
                        }
                    },
                }
            )

    def test_depth_retry_count_must_be_a_bounded_integer(self) -> None:
        for invalid in (True, -1, 6, 1.5):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(
                TypeError, "depth_retry_count"
            ):
                normalize_primitive_config(
                    {
                        "version": 1,
                        "primitives": {
                            "sample_grasp_pose": {
                                "settings": {"depth_retry_count": invalid}
                            }
                        },
                    }
                )


class PrimitiveDefaultInjectionTest(unittest.TestCase):
    def test_environment_and_navigation_controller_receive_docking_settings(self) -> None:
        hardware = FakeHardware(arms=None)
        config = {
            "robot": {
                "hardware": hardware,
                "manipulation": FAKE_MANIPULATION_CONFIG,
            },
            "primitive_config": {
                "version": 1,
                "primitives": {
                    "dock_to_visible_object": {
                        "defaults": {},
                        "settings": {
                            "docking_distance_m": 0.81,
                            "require_dynamic_ground_plane": True,
                        },
                    }
                },
            },
        }

        environment = launch_module.build_environment(config)
        self.addCleanup(environment.safe_shutdown)

        docking = environment.manipulation_config["visible_object_docking"]
        self.assertEqual(docking["docking_distance_m"], 0.81)
        self.assertIs(docking["require_dynamic_ground_plane"], True)

    def test_launch_wires_shipped_config_into_registry_documentation(self) -> None:
        config = load_config(CONFIG_DIRECTORY / "mobile_manipulation.yaml")
        environment = make_environment(
            FakeHardware(
                arms={
                    "left": {},
                    "right": {},
                    "estop_latched": False,
                    "end_effector_frame": "tcp",
                }
            ),
            manipulation=FAKE_MANIPULATION_CONFIG,
        )
        self.addCleanup(environment.safe_shutdown)

        with patch.object(
            launch_module, "build_environment", return_value=environment
        ):
            runtime = launch_module.build_runtime(config)

        documentation = runtime.registry.documentation()
        self.assertIn("dock_to_visible_object", runtime.registry.names())
        self.assertIn("Configured success boundary: 0.65 m", documentation)
        self.assertIn("Immediately after close-range manipulation", documentation)
        self.assertIn("prepare_for_manipulation", runtime.registry.names())
        self.assertIn("def prepare_for_manipulation", documentation)
        self.assertIn("batched Pi IK", documentation)
        prepare = runtime.registry.functions()["prepare_for_manipulation"]
        prepare_arm = inspect.signature(prepare).parameters["arm"]
        self.assertEqual(prepare_arm.kind, inspect.Parameter.KEYWORD_ONLY)
        self.assertIs(prepare_arm.default, inspect.Parameter.empty)
        with self.assertRaises(TypeError):
            prepare("pen")
        with self.assertRaisesRegex(ValueError, "string 'left' or 'right'"):
            prepare("pen", arm=1)
        self.assertIn("open_gripper", runtime.registry.names())
        self.assertIn("close_gripper", runtime.registry.names())
        self.assertNotIn("lift_grasped_object", runtime.registry.names())
        self.assertNotIn("def lift_grasped_object", documentation)
        opened = runtime.registry.functions()["open_gripper"](arm="left")
        closed = runtime.registry.functions()["close_gripper"](arm="left")
        self.assertTrue(opened["success"])
        self.assertTrue(closed["success"])
        self.assertFalse(opened.get("simulated", False))
        self.assertFalse(closed.get("simulated", False))
        self.assertEqual(
            environment._hardware.gripper_commands,
            [
                {"arm": "left", "opened": True, "timeout_s": 3.0, "force_n": None},
                {"arm": "left", "opened": False, "timeout_s": 3.0, "force_n": None},
            ],
        )

    def test_navigation_defaults_appear_in_model_facing_signatures(self) -> None:
        environment = make_environment()
        self.addCleanup(environment.safe_shutdown)
        registry = PrimitiveRegistry()
        register_navigation_primitives(
            registry,
            environment,
            primitive_defaults={
                "turn_relative": {"max_yaw_deg_s": 18.0, "timeout_s": 12.0},
                "drive_straight": {"max_speed_mps": 0.09, "timeout_s": 20.0},
                "drive_lateral": {"max_speed_mps": 0.08, "timeout_s": 18.0},
            },
        )

        turn = inspect.signature(registry.functions()["turn_relative"])
        drive = inspect.signature(registry.functions()["drive_straight"])
        lateral = inspect.signature(registry.functions()["drive_lateral"])
        self.assertEqual(turn.parameters["max_yaw_deg_s"].default, 18.0)
        self.assertEqual(turn.parameters["timeout_s"].default, 12.0)
        self.assertEqual(drive.parameters["max_speed_mps"].default, 0.09)
        self.assertEqual(drive.parameters["timeout_s"].default, 20.0)
        self.assertEqual(lateral.parameters["max_speed_mps"].default, 0.08)
        self.assertEqual(lateral.parameters["timeout_s"].default, 18.0)
        documentation = registry.documentation()
        self.assertIn("max_yaw_deg_s: float | None = 18.0", documentation)
        self.assertIn("max_speed_mps: float | None = 0.09", documentation)
        self.assertIn("max_speed_mps: float | None = 0.08", documentation)

    def test_non_arm_manipulation_defaults_appear_in_callable_signatures(self) -> None:
        hardware = FakeHardware(
            arms={"left": {}, "right": {}, "estop_latched": False}
        )
        environment = make_environment(
            hardware, manipulation=FAKE_MANIPULATION_CONFIG
        )
        self.addCleanup(environment.safe_shutdown)
        registry = PrimitiveRegistry()
        register_manipulation_primitives(
            registry,
            environment,
            primitive_defaults={
                "get_object_pose": {
                    "return_bbox_extent": True,
                    "return_zed_distance": True,
                },
                "goto_pose": {"z_approach": 0.04, "timeout_s": 40.0},
                "goto_grasp_pose": {
                    "approach_m": 0.08,
                    "timeout_s": 45.0,
                },
                "open_gripper": {"timeout_s": 2.0, "force_n": 0.8},
                "lift_grasped_object": {
                    "lift_m": 0.11,
                    "timeout_s": 44.0,
                },
            },
        )

        functions = registry.functions()
        get_pose = inspect.signature(functions["get_object_pose"])
        goto = inspect.signature(functions["goto_pose"])
        goto_grasp = inspect.signature(functions["goto_grasp_pose"])
        gripper = inspect.signature(functions["open_gripper"])
        lift = inspect.signature(functions["lift_grasped_object"])
        for signature in (get_pose, goto, goto_grasp, gripper, lift):
            self.assertEqual(
                signature.parameters["arm"].kind,
                inspect.Parameter.KEYWORD_ONLY,
            )
            self.assertIs(
                signature.parameters["arm"].default,
                inspect.Parameter.empty,
            )
        self.assertIs(get_pose.parameters["return_bbox_extent"].default, True)
        self.assertIs(get_pose.parameters["return_zed_distance"].default, True)
        self.assertEqual(goto.parameters["z_approach"].default, 0.04)
        self.assertEqual(goto.parameters["timeout_s"].default, 40.0)
        self.assertEqual(goto_grasp.parameters["approach_m"].default, 0.08)
        self.assertEqual(goto_grasp.parameters["timeout_s"].default, 45.0)
        self.assertEqual(gripper.parameters["force_n"].default, 0.8)
        self.assertEqual(lift.parameters["lift_m"].default, 0.11)
        self.assertEqual(lift.parameters["timeout_s"].default, 44.0)

    def test_docking_settings_override_legacy_environment_block(self) -> None:
        manipulation = {
            **FAKE_MANIPULATION_CONFIG,
            "visible_object_docking": {"docking_distance_m": 0.5},
        }
        environment = make_environment(manipulation=manipulation)
        self.addCleanup(environment.safe_shutdown)

        controller = VisibleObjectDockingController(
            environment,
            docking_config={"docking_distance_m": 0.83},
        )

        self.assertEqual(controller.config.docking_distance_m, 0.83)


if __name__ == "__main__":
    unittest.main()
