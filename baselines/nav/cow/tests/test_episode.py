import importlib.util
import inspect
import json
import math
import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from baselines.nav.cow.camera import LevelCamera, LevelCameraModel
from baselines.nav.cow.config import CowConfig
from baselines.nav.cow.episode import CowEpisode, ZedPoseInjector, build_agent, planar_pose_matrix
from baselines.nav.cow.tasks import direct_task


INTRINSICS = ((267.2, 0.0, 337.2), (0.0, 267.2, 182.3), (0.0, 0.0, 1.0))
PITCH = math.radians(20.0)
DOWN = (0.0, math.cos(PITCH), math.sin(PITCH))
CAMERA_HEIGHT = 1.05
COW_REPO = Path(os.environ.get("COW_REPO", "/home/yor/codefield/cow")).expanduser()
HAS_COW_DEPENDENCIES = all(
    importlib.util.find_spec(name) is not None
    for name in ("networkx", "sklearn", "scipy", "threadpoolctl", "torchvision")
)
NEEDS_COW = unittest.skipUnless(
    COW_REPO.is_dir() and HAS_COW_DEPENDENCIES, "set COW_REPO and install CoW's dependencies"
)


def camera_rays():
    u, v = np.meshgrid(np.arange(672, dtype=np.float64), np.arange(376, dtype=np.float64))
    return np.stack([(u - 337.2) / 267.2, (v - 182.3) / 267.2, np.ones_like(u)], axis=-1)


def floor_depth(offset_m=0.0):
    along_down = camera_rays() @ np.asarray(DOWN)
    depth = np.where(along_down > 0.02, CAMERA_HEIGHT / np.maximum(along_down, 1e-6), 0.0)
    depth = np.where(depth > 0.0, np.minimum(depth + offset_m, 20.0), 0.0)
    return depth.astype(np.float32)


def scene_depth(pose, boxes, max_range_m=12.0):
    """Z-depth of a floor plus axis-aligned boxes (world x forward, y left, z up) from a camera pose."""

    x, y, yaw = pose
    right = np.array([math.sin(yaw), -math.cos(yaw), 0.0])
    forward = np.array([math.cos(PITCH) * math.cos(yaw), math.cos(PITCH) * math.sin(yaw), -math.sin(PITCH)])
    down = np.cross(forward, right)
    directions = camera_rays() @ np.stack([right, down, forward], axis=1).T
    origin = np.array([x, y, CAMERA_HEIGHT])
    hit = np.full(directions.shape[:2], np.inf)
    with np.errstate(divide="ignore", invalid="ignore"):
        floor = -CAMERA_HEIGHT / directions[..., 2]
        hit = np.where((directions[..., 2] < 0) & (floor > 0), floor, hit)
        for box in boxes:
            low = np.array(box[0::2]) - origin
            high = np.array(box[1::2]) - origin
            first = low / directions
            second = high / directions
            entry = np.nanmax(np.minimum(first, second), axis=-1)
            leave = np.nanmin(np.maximum(first, second), axis=-1)
            inside = (leave >= entry) & (entry > 0)
            hit = np.where(inside & (entry < hit), entry, hit)
    return np.where(hit <= max_range_m, hit, 0.0).astype(np.float32)


class FakeController:
    """Moves the camera by fixed amounts that differ from CoW's nominal deltas."""

    def __init__(self, environment, *, turn_scale=0.9, drive_scale=0.8, fail_on=None, drive_refusal=None):
        self.environment = environment
        self.turn_scale = turn_scale
        self.drive_scale = drive_scale
        self.fail_on = fail_on
        self.drive_refusal = drive_refusal
        self.calls = []

    def turn_relative(self, angle_rad):
        self.calls.append(("turn_relative", angle_rad))
        self.environment.pose[2] += angle_rad * self.turn_scale
        self.environment.frame += 1
        return {"success": True, "status": "succeeded", "primitive": "turn_relative", "reason": "target_reached"}

    def drive_straight(self, distance_m):
        self.calls.append(("drive_straight", distance_m))
        if self.fail_on == "drive_straight":
            raise RuntimeError("base RPC connection lost")
        if self.drive_refusal is not None:
            return {"success": False, "status": "failed", "primitive": "drive_straight", "reason": self.drive_refusal}
        x, y, yaw = self.environment.pose
        travelled = distance_m * self.drive_scale
        self.environment.pose = [x + travelled * math.cos(yaw), y + travelled * math.sin(yaw), yaw]
        self.environment.frame += 1
        return {"success": True, "status": "succeeded", "primitive": "drive_straight", "reason": "target_reached"}

    def stop(self):
        self.calls.append(("stop",))
        return {"success": True, "status": "succeeded", "primitive": "stop", "reason": "stopped"}


class FakeEnvironment:
    """A camera over an endless floor, or over a scene of boxes.

    ``depth_offset_per_frame`` makes consecutive floor-only frames differ so
    CoW's depth-difference test treats motions as successful; it also puts the
    apparent floor below its true height, so tests of the real CoW map use 0
    or a box scene. The first ``invalid_pose_observations`` observations carry
    the non-finite pose YOR reports while ZED tracking is not OK.
    """

    def __init__(
        self,
        *,
        depth_offset_per_frame=0.2,
        invalid_pose_observations=0,
        boxes=None,
        start_pose=(0.3, -0.2, 0.4),
        **controller_options,
    ):
        self.pose = list(start_pose)
        self.frame = 0
        self.observations = 0
        self.depth_offset_per_frame = depth_offset_per_frame
        self.invalid_pose_observations = invalid_pose_observations
        self.boxes = boxes
        self.estop_latched = False
        self.controller = FakeController(self, **controller_options)

    def reset(self):
        return self._observation()

    def observe_next(self):
        return self._observation()

    def _observation(self):
        self.observations += 1
        valid = self.observations > self.invalid_pose_observations
        if self.boxes is None:
            depth = floor_depth(self.depth_offset_per_frame * self.frame)
        else:
            depth = scene_depth(self.pose, self.boxes)
        return {
            "robot0_robotview": {
                "images": {"rgb": np.zeros((376, 672, 3), dtype=np.uint8), "depth": depth[:, :, None]}
            },
            "base": {
                "pose_xy_yaw": np.asarray(self.pose if valid else [math.nan] * 3),
                "estop_latched": self.estop_latched,
            },
        }


def cow_movement(action):
    """CoW's nominal action delta, as in FrontierBasedExploration._action_to_movement_matrix."""

    matrix = torch.eye(4)
    if action in ("RotateLeft", "RotateRight"):
        angle = math.radians(30.0 if action == "RotateLeft" else -30.0)
        c, s = math.cos(angle), math.sin(angle)
        matrix[:3, :3] = torch.tensor([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    elif action == "MoveAhead":
        matrix[2, 3] = 0.25
    return matrix


class FakeExploration:
    """The pose bookkeeping of CoW's update_map and reset, without the map."""

    def __init__(self):
        self.camera_to_agent = torch.eye(4)
        self.last_observation = None
        self.fail_stop = True
        self.roi_targets = []
        self.exploration_targets = []
        self.voxels = SimpleNamespace(number_of_nodes=lambda: 0)

    def reset(self):
        self.camera_to_agent = torch.eye(4)
        self.last_observation = None

    def poll_roi_exists(self):
        return False

    def _action_to_movement_matrix(self, action):
        return cow_movement(action)

    def update_map(self, depth, last_action):
        new = torch.as_tensor(depth).squeeze()
        failed = False
        if self.last_observation is not None and self.fail_stop:
            difference = torch.abs(self.last_observation - new)
            failed = difference.mean().item() < 0.09 and difference.std().item() < 0.09
        self.last_observation = new
        movement = torch.eye(4) if failed else cow_movement(last_action)
        self.camera_to_agent = self.camera_to_agent @ movement


class ScriptedAgent:
    def __init__(self, actions, *, reset_at_step=None):
        self.fbe = FakeExploration()
        self.actions = list(actions)
        self.reset_at_step = reset_at_step
        self.last_action = None
        self.agent_mode = SimpleNamespace(name="EXPLORE")
        self.step = 0
        self.camera_after_act = []

    def reset(self):
        self.fbe.reset()
        self.last_action = None

    def localize_object(self, observations):
        return torch.zeros(224, 224)

    def act(self, observations):
        self.localize_object(observations)
        self.fbe.update_map(observations["depth"], self.last_action)
        if self.step == self.reset_at_step:
            # CoW's explore/exploit "confused" branch: reset, then integrate the frame again.
            self.fbe.reset()
            self.fbe.update_map(observations["depth"], self.last_action)
        self.camera_after_act.append(self.fbe.camera_to_agent.clone())
        self.step += 1
        action = self.actions.pop(0)
        self.last_action = action
        return action


def make_camera():
    return LevelCamera(
        LevelCameraModel(source_width=672, source_height=376, intrinsics=INTRINSICS, down_camera_xyz=DOWN)
    )


def make_config(**agent):
    config = CowConfig()
    return replace(
        config,
        agent=replace(config.agent, device="cpu", **agent),
        experiment=replace(config.experiment, save_images=False),
    )


def run_episode(agent, environment, *, config=None, no_motion=False, clock=None, log=None):
    directory = Path(tempfile.mkdtemp())
    episode = CowEpisode(
        config or make_config(),
        agent=agent,
        environment=environment,
        camera=make_camera(),
        task=direct_task("Navigate to the blue trash bin"),
        output_dir=directory,
        no_motion=no_motion,
        log=log or (lambda message: None),
        **({} if clock is None else {"clock": clock}),
    )
    result = episode.run()
    events = [json.loads(line) for line in (directory / "trace.jsonl").read_text().splitlines()]
    return result, events


def record_poses(agent, environment):
    poses = []
    original_act = agent.act

    def act_recording_pose(observations):
        poses.append(list(environment.pose))
        return original_act(observations)

    agent.act = act_recording_pose
    return poses


class EpisodeTest(unittest.TestCase):
    def test_actions_run_yor_primitives_until_cow_stops(self):
        environment = FakeEnvironment()
        agent = ScriptedAgent(["RotateLeft", "MoveAhead", "RotateRight", "Stop"])
        result, events = run_episode(agent, environment)
        self.assertEqual(result["termination"], "cow_stop")
        self.assertEqual(result["steps"], 4)
        self.assertIsNone(result["task_success"])
        self.assertIsNone(result["error"])
        calls = environment.controller.calls
        self.assertEqual(calls[0], ("turn_relative", math.radians(30.0)))
        self.assertEqual(calls[1], ("drive_straight", 0.25))
        self.assertEqual(calls[2], ("turn_relative", -math.radians(30.0)))
        self.assertEqual(calls[-1], ("stop",))
        self.assertEqual([event["type"] for event in events].count("decision"), 4)
        self.assertEqual(events[-1]["type"], "episode_end")

    def test_console_follows_decisions_and_actions(self):
        lines = []
        run_episode(ScriptedAgent(["RotateLeft", "MoveAhead", "Stop"]), FakeEnvironment(), log=lines.append)
        self.assertIn("waiting for the Pi base status", lines[0])
        self.assertTrue(any(line.startswith("step   0") and "RotateLeft" in line for line in lines))
        self.assertTrue(any("turn_relative done" in line for line in lines))
        self.assertTrue(any("drive_straight done" in line for line in lines))
        self.assertIn("episode end: cow_stop after 3 steps", lines[-1])

    def test_yor_safety_stop_ends_the_episode(self):
        environment = FakeEnvironment(drive_refusal="obstacle_too_close")
        agent = ScriptedAgent(["RotateLeft", "MoveAhead", "RotateLeft", "Stop"])
        result, events = run_episode(agent, environment)
        self.assertEqual(result["termination"], "safety_stop")
        self.assertEqual(result["safety_stop"], {"step": 1, "action": "MoveAhead", "reason": "obstacle_too_close"})
        self.assertEqual(result["steps"], 2)
        self.assertEqual([event["type"] for event in events].count("decision"), 2)
        self.assertEqual(environment.controller.calls[-1], ("stop",))

    def test_safety_stop_can_be_left_to_cow(self):
        config = make_config()
        config = replace(config, experiment=replace(config.experiment, end_on_safety_stop=False))
        environment = FakeEnvironment(drive_refusal="obstacle_too_close")
        result, _ = run_episode(ScriptedAgent(["MoveAhead", "MoveAhead", "Stop"]), environment, config=config)
        self.assertEqual(result["termination"], "cow_stop")
        self.assertIsNone(result["safety_stop"])

    def test_no_motion_decides_once_without_primitives(self):
        environment = FakeEnvironment()
        result, _ = run_episode(ScriptedAgent(["MoveAhead"]), environment, no_motion=True)
        self.assertEqual(result["termination"], "no_motion_first_decision")
        self.assertEqual(environment.controller.calls, [("stop",)])

    def test_time_budget_and_estop_end_the_episode(self):
        ticks = iter(range(0, 10_000, 400))
        result, _ = run_episode(
            ScriptedAgent(["RotateLeft"] * 20), FakeEnvironment(), clock=lambda: float(next(ticks))
        )
        self.assertEqual(result["termination"], "time_budget_exhausted")
        environment = FakeEnvironment()
        environment.estop_latched = True
        result, _ = run_episode(ScriptedAgent(["RotateLeft"]), environment)
        self.assertEqual(result["termination"], "estop_latched")

    def test_unexpected_exception_still_stops_and_records_the_episode(self):
        environment = FakeEnvironment(fail_on="drive_straight")
        result, events = run_episode(ScriptedAgent(["RotateLeft", "MoveAhead", "Stop"]), environment)
        self.assertEqual(result["termination"], "error")
        self.assertIn("RuntimeError: base RPC connection lost", result["error"])
        self.assertEqual(environment.controller.calls[-1], ("stop",))
        self.assertIn("error", [event["type"] for event in events])
        self.assertEqual(events[-1]["type"], "episode_end")

    def test_map_pose_is_the_measured_pose_not_the_nominal_delta(self):
        environment = FakeEnvironment()
        start = list(environment.pose)
        agent = ScriptedAgent(["RotateLeft", "MoveAhead", "MoveAhead", "RotateRight", "Stop"])
        poses = record_poses(agent, environment)
        run_episode(agent, environment)
        world_to_map = torch.linalg.inv(planar_pose_matrix(start))
        for pose, camera_to_agent in zip(poses, agent.camera_after_act):
            expected = world_to_map @ planar_pose_matrix(pose)
            torch.testing.assert_close(camera_to_agent, expected, atol=1e-5, rtol=0)

    def test_mid_episode_map_reset_reanchors_the_pose(self):
        environment = FakeEnvironment()
        agent = ScriptedAgent(["RotateLeft", "MoveAhead", "RotateLeft", "MoveAhead", "Stop"], reset_at_step=2)
        poses = record_poses(agent, environment)
        result, _ = run_episode(agent, environment)
        self.assertEqual(result["map_resets"], 1)
        # CoW's reset restarts the map at identity and re-applies the previous action.
        anchor = cow_movement("MoveAhead") @ torch.linalg.inv(planar_pose_matrix(poses[2]))
        torch.testing.assert_close(agent.camera_after_act[2], cow_movement("MoveAhead"), atol=1e-5, rtol=0)
        for index in (3, 4):
            expected = anchor @ planar_pose_matrix(poses[index])
            torch.testing.assert_close(agent.camera_after_act[index], expected, atol=1e-5, rtol=0)

    def test_invalid_tracking_pose_is_waited_out(self):
        environment = FakeEnvironment(invalid_pose_observations=3)
        agent = ScriptedAgent(["RotateLeft", "Stop"])
        result, events = run_episode(agent, environment)
        self.assertEqual(result["termination"], "cow_stop")
        self.assertEqual([event["type"] for event in events].count("pose_invalid_frame"), 3)
        for camera_to_agent in agent.camera_after_act:
            self.assertTrue(bool(torch.isfinite(camera_to_agent).all()))

    def test_persistent_invalid_pose_ends_before_cow_acts(self):
        environment = FakeEnvironment(invalid_pose_observations=10_000)
        agent = ScriptedAgent(["MoveAhead"])
        result, _ = run_episode(agent, environment)
        self.assertEqual(result["termination"], "pose_invalid")
        self.assertEqual(agent.camera_after_act, [])
        self.assertEqual(environment.controller.calls, [("stop",)])

    def test_dead_reckoning_does_not_require_a_tracking_pose(self):
        environment = FakeEnvironment(invalid_pose_observations=10_000)
        config = make_config()
        config = replace(config, robot=replace(config.robot, pose_source="dead_reckoning"))
        result, _ = run_episode(ScriptedAgent(["RotateLeft", "Stop"]), environment, config=config)
        self.assertEqual(result["termination"], "cow_stop")


class InjectorTest(unittest.TestCase):
    def test_predicted_failure_leaves_pose_unmoved(self):
        exploration = FakeExploration()
        injector = ZedPoseInjector(exploration)
        injector.start([0.0, 0.0, 0.0])
        depth = floor_depth()
        exploration.update_map(depth, None)
        self.assertTrue(injector.before_act([0.0, 0.0, 0.0], depth, "MoveAhead"))
        exploration.update_map(depth, "MoveAhead")
        torch.testing.assert_close(exploration.camera_to_agent, torch.eye(4), atol=1e-6, rtol=0)

    def test_without_fail_stop_cow_always_applies_the_delta(self):
        exploration = FakeExploration()
        exploration.fail_stop = False
        injector = ZedPoseInjector(exploration)
        injector.start([0.0, 0.0, 0.0])
        depth = floor_depth()
        exploration.update_map(depth, None)
        self.assertFalse(injector.before_act([0.0, 0.0, 0.0], depth, "MoveAhead"))
        exploration.update_map(depth, "MoveAhead")
        torch.testing.assert_close(exploration.camera_to_agent, torch.eye(4), atol=1e-6, rtol=0)

    def test_non_finite_pose_is_rejected(self):
        with self.assertRaises(ValueError):
            planar_pose_matrix([math.nan, 0.0, 0.0])


@NEEDS_COW
class UpstreamAgentTest(unittest.TestCase):
    def setUp(self):
        from baselines.nav.cow.upstream import import_cow

        self.modules = import_cow(COW_REPO, localizer="clip_grad")

    def no_detection_agent(self):
        agent_fbe = __import__("src.models.agent_fbe", fromlist=["AgentFbe"]).AgentFbe

        class NoDetectionAgent(agent_fbe):
            def __init__(self):
                super().__init__(90.0, torch.device("cpu"), CAMERA_HEIGHT, 0.15, fail_stop=True)
                self.transform = None
                self.clip_module = lambda image, goal: torch.zeros(224, 224)

        return NoDetectionAgent()

    def test_exact_motions_reproduce_cows_own_dead_reckoning(self):
        """With the robot moving exactly CoW's steps, the injected pose equals CoW's nominal chain."""

        real = self.modules.exploration.FrontierBasedExploration(
            90.0, torch.device("cpu"), 1.9, CAMERA_HEIGHT, 0.15, 30.0, 0.25, 0.125, True, False, False, True
        )
        environment = FakeEnvironment(turn_scale=1.0, drive_scale=1.0)
        actions = ["RotateLeft", "MoveAhead", "RotateRight", "RotateRight", "MoveAhead", "Stop"]
        agent = ScriptedAgent(list(actions))
        run_episode(agent, environment)
        nominal = torch.eye(4)
        previous = None
        for index, camera_to_agent in enumerate(agent.camera_after_act):
            nominal = nominal @ real._action_to_movement_matrix(previous).float()
            torch.testing.assert_close(camera_to_agent, nominal, atol=1e-5, rtol=0)
            previous = actions[index]

    def test_build_agent_binds_cows_constructor_and_resize(self):
        import torchvision.transforms as T

        signature = inspect.signature(self.modules.agent_class.__init__)
        captured = {}

        class Capture:
            def __init__(self, *args, **kwargs):
                captured.update(signature.bind(self, *args, **kwargs).arguments)

        config = make_config()
        agent = build_agent(
            config,
            agent_class=Capture,
            repository=COW_REPO,
            goal="blue trash bin",
            camera_height_m=1.045,
            threshold=0.625,
        )
        self.assertTrue(str(captured["clip_model_name"]).endswith("ViT-B-32.pt"))
        self.assertEqual(captured["classes"], ["blue trash bin"])
        self.assertEqual(captured["classes_clip"], ["blue trash bin"])
        self.assertEqual(captured["fov"], 90.0)
        self.assertEqual((captured["height"], captured["width"]), (672, 672))
        self.assertEqual(captured["agent_height"], 1.045)
        self.assertEqual(captured["floor_tolerance"], config.agent.floor_tolerance_m)
        self.assertEqual(captured["threshold"], 0.625)
        self.assertIs(captured["center_only"], True)
        self.assertIs(captured["fail_stop"], True)
        resizes = [step for step in agent.transform.transforms if isinstance(step, T.Resize)]
        self.assertEqual(len(resizes), 1)
        self.assertIs(resizes[0].antialias, False)

    def test_unmodified_cow_agent_spins_first_and_maps_the_floor_free(self):
        config = make_config(max_steps=10)
        environment = FakeEnvironment(depth_offset_per_frame=0.0, turn_scale=1.0, drive_scale=1.0)
        agent = self.no_detection_agent()
        result, _ = run_episode(agent, environment, config=config)
        self.assertEqual(result["termination"], "max_steps")
        turns = [call for call in environment.controller.calls if call[0] == "turn_relative"]
        self.assertGreaterEqual(len(turns), 9)
        voxel_type = self.modules.exploration.VoxelType
        near_occupied = [
            key
            for key, data in agent.fbe.voxels.nodes(data=True)
            if data["voxel_type"] == voxel_type.OCCUPIED and math.hypot(key[0], key[2]) * 0.125 < 3.0
        ]
        self.assertEqual(near_occupied, [])

    def test_obstacles_land_where_they_are_with_measured_turns(self):
        """A box on the left maps to the left, and nothing is mapped where no structure exists."""

        walls = [
            (-2.6, -2.5, -3.0, 3.0, 0.0, 2.5),
            (4.0, 4.1, -3.0, 3.0, 0.0, 2.5),
            (-2.6, 4.1, 3.0, 3.1, 0.0, 2.5),
            (-2.6, 4.1, -3.1, -3.0, 0.0, 2.5),
        ]
        box = (1.8, 2.2, 0.6, 1.0, 0.0, 1.0)
        environment = FakeEnvironment(
            boxes=walls + [box], start_pose=(0.0, 0.0, 0.0), turn_scale=0.9, drive_scale=1.0
        )
        agent = self.no_detection_agent()
        result, events = run_episode(agent, environment, config=make_config(max_steps=10))
        self.assertEqual(result["termination"], "max_steps")
        accepted = [
            event
            for event in events
            if event["type"] == "decision" and event["step"] > 0 and event["predicted_failed_previous_action"] is False
        ]
        self.assertTrue(accepted, "at least one measured turn must be integrated as a real motion")

        voxel_type = self.modules.exploration.VoxelType
        occupied = [
            (key[2] * 0.125, key[0] * 0.125)
            for key, data in agent.fbe.voxels.nodes(data=True)
            if data["voxel_type"] == voxel_type.OCCUPIED
        ]

        def near(point, footprint, margin):
            forward, left = point
            return footprint[0] - margin <= forward <= footprint[1] + margin and (
                footprint[2] - margin <= left <= footprint[3] + margin
            )

        on_box = [point for point in occupied if near(point, box, 0.3)]
        mirrored_box = (box[0], box[1], -box[3], -box[2])
        on_mirror = [point for point in occupied if near(point, mirrored_box, 0.3)]
        stray = [point for point in occupied if not any(near(point, item, 0.4) for item in walls + [box])]
        self.assertTrue(on_box)
        self.assertEqual(on_mirror, [])
        self.assertEqual(stray, [])


if __name__ == "__main__":
    unittest.main()
