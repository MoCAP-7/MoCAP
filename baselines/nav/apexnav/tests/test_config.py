import unittest

from baselines.nav.apexnav.config import (
    ApexNavConfig,
    PlannerConfig,
    ViewerConfig,
    load_config,
)


class ConfigTest(unittest.TestCase):
    def test_shipped_config_is_isolated_and_uses_yor_limits(self):
        config = load_config("baselines/nav/apexnav/config.yaml")
        self.assertEqual(config.planner.ros_domain_id, 47)
        self.assertEqual(config.robot.base_rpc_timeout_s, 2.0)
        self.assertEqual(config.robot.maximum_linear_mps, 0.18)
        self.assertEqual(config.robot.maximum_yaw_rad_s, 0.35)
        self.assertEqual(config.planner.obstacle_inflation_m, 0.50)
        self.assertEqual(config.planner.optimizer_safe_distance_m, 0.20)
        self.assertEqual(config.planner.footprint_length_m, 0.16)
        self.assertEqual(config.planner.footprint_width_m, 0.16)
        self.assertGreaterEqual(config.planner.minimum_map_free_cells, 100)
        self.assertEqual(config.planner.initial_scan_timeout_s, 50.0)
        self.assertTrue(config.planner.time_scale_to_limits)
        self.assertTrue(config.planner.adaptive_replan_time)
        self.assertLess(
            config.planner.reference_max_linear_mps, config.robot.maximum_linear_mps
        )
        self.assertLess(
            config.planner.reference_max_yaw_rad_s, config.robot.maximum_yaw_rad_s
        )

    def test_reference_limits_above_the_robot_limits_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "reference speed"):
            ApexNavConfig(
                planner=PlannerConfig(reference_max_linear_mps=0.5)
            ).validate()
        with self.assertRaisesRegex(ValueError, "reference yaw rate"):
            ApexNavConfig(
                planner=PlannerConfig(reference_max_yaw_rad_s=0.0)
            ).validate()

    def test_nonpositive_footprint_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "footprint"):
            ApexNavConfig(
                planner=PlannerConfig(footprint_length_m=0.0)
            ).validate()

    def test_shipped_viewer_binds_tailscale_on_the_default_port(self):
        viewer = load_config("baselines/nav/apexnav/config.yaml").viewer
        self.assertTrue(viewer.enabled)
        self.assertEqual(viewer.address, "")
        self.assertEqual(viewer.port, 8765)
        self.assertTrue(viewer.publish_tf)

    def test_invalid_viewer_address_or_port_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "viewer address"):
            ApexNavConfig(viewer=ViewerConfig(address="tailscale")).validate()
        with self.assertRaisesRegex(ValueError, "viewer port"):
            ApexNavConfig(viewer=ViewerConfig(port=80)).validate()


if __name__ == "__main__":
    unittest.main()
