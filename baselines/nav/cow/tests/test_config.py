import unittest
from pathlib import Path

import yaml

from baselines.nav.cow.config import AgentConfig, CowConfig, RobotConfig, build_config, load_config
from baselines.nav.cow.robot import REPOSITORY_ROOT, camera_geometry_from_yor


class ConfigTest(unittest.TestCase):
    def test_bundled_config_is_cow_with_the_comparison_stop_distance(self):
        config = load_config(Path(__file__).resolve().parents[1] / "config.yaml")
        self.assertEqual(config.agent.localizer, "clip_grad")
        self.assertIsNone(config.agent.threshold)
        self.assertEqual(config.agent.fov_deg, 90.0)
        self.assertEqual(config.agent.voxel_size_m, 0.125)
        self.assertEqual(config.agent.stop_radius_m, 0.65)
        self.assertEqual(config.agent.max_steps, 250)
        self.assertEqual(config.robot.pose_source, "zed")
        self.assertEqual(config.experiment.maximum_duration_s, 900.0)
        self.assertIs(config.experiment.end_on_safety_stop, True)

    def test_unknown_keys_and_invalid_values_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown AgentConfig keys"):
            build_config(CowConfig, {"agent": {"stop_distance": 1.0}})
        with self.assertRaisesRegex(ValueError, "multiple of agent.rotation_deg"):
            AgentConfig(fov_deg=103.0)
        with self.assertRaises(ValueError):
            RobotConfig(pose_source="nav2")

    def test_camera_geometry_comes_from_the_yor_primitive_configuration(self):
        raw = yaml.safe_load((REPOSITORY_ROOT / "yor_agent/configs/primitive_config.yaml").read_text())
        settings = raw["primitives"]["dock_to_visible_object"]["settings"]
        height, down = camera_geometry_from_yor({"primitive_config": raw})
        self.assertEqual(height, float(settings["ground_camera_height_m"]))
        self.assertEqual(list(down), [float(value) for value in settings["ground_down_camera_xyz"]])
        # Optical frame (x right, y down, z forward) of a camera tilted towards the floor.
        norm = sum(value * value for value in down) ** 0.5
        self.assertGreater(down[1], 0.8 * norm)
        self.assertGreater(down[2], 0.0)
        with self.assertRaises(ValueError):
            camera_geometry_from_yor({"primitive_config": {}})


if __name__ == "__main__":
    unittest.main()
