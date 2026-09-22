from __future__ import annotations

import ast
import importlib
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest


BASELINE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[4]
COARSE_CONFIG = BASELINE_DIR / "coarse_navigation.yaml"
R1PRO_CONFIG = BASELINE_DIR / "r1pro.yaml"


def _import_runtime_module(name: str):
    """Import the minimal Cap-X path when gymnasium is absent in the test shell."""

    if "gymnasium" not in sys.modules:
        try:
            import gymnasium  # noqa: F401
        except ImportError:
            gym = types.ModuleType("gymnasium")

            class Env:
                def __init__(self):
                    pass

                def reset(self, **kwargs):
                    del kwargs

            class Space:
                def __init__(self, *args, **kwargs):
                    del args, kwargs

            gym.Env = Env
            gym.spaces = types.SimpleNamespace(Text=Space, Dict=Space)
            sys.modules["gymnasium"] = gym
    if "omegaconf" not in sys.modules:
        try:
            import omegaconf  # noqa: F401
        except ImportError:
            omega = types.ModuleType("omegaconf")

            class DictConfig(dict):
                pass

            class ListConfig(list):
                pass

            class OmegaConf:
                @staticmethod
                def to_container(value, **kwargs):
                    del kwargs
                    return value

            omega.DictConfig = DictConfig
            omega.ListConfig = ListConfig
            omega.OmegaConf = OmegaConf
            sys.modules["omegaconf"] = omega
    if "cloudpickle" not in sys.modules:
        try:
            import cloudpickle  # noqa: F401
        except ImportError:
            cloudpickle = types.ModuleType("cloudpickle")
            cloudpickle.dump = lambda *args, **kwargs: None
            cloudpickle.load = lambda *args, **kwargs: None
            sys.modules["cloudpickle"] = cloudpickle
    return importlib.import_module(name)


def _resolved_yor_config(config_name: str, task_id: str):
    from yor_agent.launch import load_config

    return load_config(REPO_ROOT / "yor_agent/configs" / config_name, task_id=task_id)


class _FakeMotionController:
    def __init__(self):
        self.config = types.SimpleNamespace(min_yaw_rad_s=0.1)


class _FakeYorEnvironment:
    def __init__(self):
        self.controller = _FakeMotionController()
        self.manipulation_config = {}


def test_low_level_reset_resets_yor_without_the_abstract_capx_reset():
    module = _import_runtime_module("baselines.mobilemanip.capx.environment")
    resolved = _resolved_yor_config("nav_comparison.yaml", "grasp_can")
    observation = {
        "robot0_robotview": {
            "images": {
                "rgb": np.full((5, 5, 3), 7, dtype=np.uint8),
                "depth": np.ones((5, 5, 1), dtype=np.float32),
            }
        }
    }

    class ResettableYorEnvironment(_FakeYorEnvironment):
        def __init__(self):
            super().__init__()
            self.resets = 0

        def reset(self):
            self.resets += 1
            return observation

    yor_environment = ResettableYorEnvironment()
    env = module.CapXYorLowLevelEnv(
        yor_environment=yor_environment,
        resolved_yor_config=resolved,
        task_id="grasp_can",
    )
    returned, info = env.reset(seed=3)
    assert returned == observation
    assert info == {"task_id": "grasp_can"}
    assert yor_environment.resets == 1
    assert np.all(env.render() == 7)


def test_navigation_api_exposes_only_the_four_production_primitives():
    module = _import_runtime_module("baselines.mobilemanip.capx.navigation")
    resolved = _resolved_yor_config("passive_video_tasks.yaml", "grasp_cardboard_box")
    wrapper = types.SimpleNamespace(
        primitive_config=resolved["primitive_config"],
        yor_environment=_FakeYorEnvironment(),
    )
    api = module.CapXYorNavigationApi(
        wrapper,
        docking_factories={
            "segment_client_factory": lambda: lambda *args, **kwargs: [],
            "nav2_client_factory": lambda config: object(),
        },
    )
    assert tuple(api.functions()) == module.NAVIGATION_PRIMITIVES
    assert set(api.functions()) == {
        "drive_straight",
        "turn_relative",
        "drive_lateral",
        "dock_to_visible_object",
    }


def test_coarse_navigation_api_exposes_only_the_capx_vocabulary():
    module = _import_runtime_module("baselines.mobilemanip.capx.navigation")
    resolved = _resolved_yor_config("nav_comparison.yaml", "find_can_and_give_back")
    wrapper = types.SimpleNamespace(
        primitive_config=resolved["primitive_config"],
        yor_environment=_FakeYorEnvironment(),
    )
    api = module.CapXYorCoarseNavigationApi(wrapper)
    assert tuple(api.functions()) == (
        "go_forward",
        "turn_left_45_degrees",
        "turn_right_45_degrees",
        "goto_planar_position",
        "say_something",
    )


def test_low_level_rgb_adapter_reads_the_current_yor_observation_schema():
    module = _import_runtime_module("baselines.mobilemanip.capx.environment")
    rgb = np.full((3, 4, 3), 7, dtype=np.uint8)
    observation = {
        "robot0_robotview": {"images": {"rgb": rgb, "depth": None}}
    }
    assert module._observation_rgb(observation) is rgb


class _FakeManipulationEnv:
    def __init__(self):
        self.primitive_config = {
            "primitives": {
                "open_gripper": {"settings": {"simulated": False}},
                "close_gripper": {"settings": {"simulated": False}},
            }
        }
        self.manipulation_config = {
            "contact_graspnet_origin_to_tcp_m": 0.1,
            "contact_graspnet_to_nero_tcp_rpy_rad": [0.0, 0.0, 0.0],
        }
        self.moves = []
        self.gripper_calls = []

    def get_observation(self):
        return {
            "robot0_robotview": {
                "images": {
                    "rgb": np.zeros((5, 5, 3), dtype=np.uint8),
                    "depth": np.ones((5, 5, 1), dtype=np.float32),
                }
            }
        }

    def manipulation_calibration(self, arm):
        del arm
        return {
            "camera_intrinsics": np.asarray(
                [[4.0, 0.0, 2.0], [0.0, 4.0, 2.0], [0.0, 0.0, 1.0]]
            ),
            "camera_calibration_resolution": np.asarray([5, 5]),
            "arm_from_camera": np.eye(4),
        }

    def arm_status(self):
        return {"end_effector_frame": "tcp"}

    def move_arm_pose(self, arm, pose, timeout_s):
        self.moves.append((arm, pose, timeout_s))
        return {"success": True, "reason": "done"}

    def set_gripper(self, arm, opened, timeout_s, force_n):
        self.gripper_calls.append((arm, opened, timeout_s, force_n))
        return {"success": True, "reason": "done"}


def test_manipulation_selects_top_capx_candidates_and_moves_directly():
    module = _import_runtime_module("baselines.mobilemanip.capx.manipulation")
    masks = [
        {"mask": np.ones((5, 5), dtype=bool), "score": 0.1},
        {"mask": np.ones((5, 5), dtype=bool), "score": 0.9},
    ]
    grasps = np.stack((np.eye(4), np.eye(4)))
    grasps[0, 0, 3] = 0.1
    grasps[1, 0, 3] = 0.2
    scores = np.asarray([0.2, 0.8])
    env = _FakeManipulationEnv()
    api = module.CapXYorManipulationApi(
        env,
        segment_client_factory=lambda: lambda *args, **kwargs: masks,
        grasp_client_factory=lambda: (
            lambda *args, **kwargs: (grasps, scores, np.zeros((2, 3)))
        ),
    )
    position, quaternion = api.sample_grasp_pose("box", "left")
    assert np.isclose(position[0], 0.2)
    assert quaternion.shape == (4,)
    api.goto_pose(position, quaternion, "left", z_approach=0.05)
    assert len(env.moves) == 2
    assert env.moves[0][0] == "left"
    assert not hasattr(env, "plan_arm_poses")


def test_manipulation_source_does_not_call_yor_enhanced_manipulation_stack():
    tree = ast.parse((BASELINE_DIR / "manipulation.py").read_text(encoding="utf-8"))
    names = {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
    }
    assert names.isdisjoint(
        {
            "plan_arm_poses",
            "execute_arm_trajectory",
            "prepare_for_manipulation",
            "goto_grasp_pose",
        }
    )


class _FakePlanner:
    def __init__(self):
        self.observation = None

    def plan(self, *, task, observation, image_max_width):
        assert task == "Find the box"
        assert image_max_width == 640
        self.observation = observation
        return "Location: likely beside the table.\n1. Face the cabinet.\nTarget cue: box."


class _FakeLowLevel:
    def __init__(self, primitive_config):
        self.primitive_config = primitive_config
        self.yor_environment = _FakeYorEnvironment()
        self.manipulation_config = {}
        self.capx_manipulation_config = {"simulate_gripper": False}
        self._sim_step_count = 0
        self.closed = False
        self.latest = None
        self.stop_requests = 0

    def _observation(self, value):
        image = np.full((5, 5, 3), value, dtype=np.uint8)
        return {
            "robot0_robotview": {
                "images": {
                    "rgb": image,
                    "depth": np.ones((5, 5, 1), dtype=np.float32),
                }
            },
        }

    def reset(self, **kwargs):
        del kwargs
        self.latest = self._observation(1)
        return self.latest, {}

    def get_observation(self):
        self.latest = self._observation(2)
        return self.latest

    def render(self, mode="rgb_array"):
        del mode
        return self.latest["robot0_robotview"]["images"]["rgb"].copy()

    def compute_reward(self):
        return 0.0

    def task_completed(self):
        return False

    def request_stop(self):
        self.stop_requests += 1
        return {"accepted": True}

    def close(self):
        self.closed = True


def test_code_env_injects_one_shot_advice_into_capx_prompt():
    module = _import_runtime_module("baselines.mobilemanip.capx.code_env")
    resolved = _resolved_yor_config("passive_video_tasks.yaml", "grasp_cardboard_box")
    low_level = _FakeLowLevel(resolved["primitive_config"])
    planner = _FakePlanner()
    config = module.CapXYorCodeExecConfig(
        low_level=low_level,
        apis=[],
        task_instruction="Find the box",
        navigation_planner={"memory_path": "unused-in-injected-test"},
        planner_image_max_width=640,
    )
    env = module.CapXYorCodeEnv(config, planner=planner)
    observation, info = env.reset()
    prompt = observation["full_prompt"][-1]["content"][0]["text"]
    assert "Operator task: Find the box" in prompt
    assert "Fallible one-time navigation advice" in prompt
    assert "dock_to_visible_object(" in prompt
    assert info["navigation_advice"] == env.navigation_advice
    planner_image = planner.observation["robot0_robotview"]["images"]["rgb"]
    assert np.all(planner_image == 2)
    assert np.array_equal(low_level.render(), planner_image)
    env.close()
    assert low_level.closed is True


def _coarse_code_env(module):
    resolved = _resolved_yor_config("nav_comparison.yaml", "find_can_and_give_back")
    low_level = _FakeLowLevel(resolved["primitive_config"])
    config = module.CapXYorCodeExecConfig(
        low_level=low_level,
        apis=[],
        task_instruction="Find the can, grasp it, and return",
        navigation="coarse",
        navigation_planner=None,
    )
    return module.CapXYorCodeEnv(config), low_level


def test_coarse_code_env_documents_the_coarse_vocabulary_without_a_planner():
    module = _import_runtime_module("baselines.mobilemanip.capx.code_env")
    env, _ = _coarse_code_env(module)
    observation, info = env.reset()
    prompt = observation["full_prompt"][-1]["content"][0]["text"]
    assert "Find the can, grasp it, and return" in prompt
    assert "go_forward drives exactly 1 meter forward" in prompt
    for call in ("go_forward(", "goto_planar_position(", "sample_grasp_pose(", "goto_pose("):
        assert call in prompt
    for hidden in ("dock_to_visible_object", "drive_straight(", "turn_relative(", "drive_lateral("):
        assert hidden not in prompt
    assert env.navigation_advice is None
    assert "navigation_advice" not in info
    # The stoppable wrappers document the same signatures and docstrings.
    assert env._get_complete_prompt() == prompt
    env.close()


def test_stop_stops_the_base_and_ends_the_episode_before_the_next_program():
    module = _import_runtime_module("baselines.mobilemanip.capx.code_env")
    env, low_level = _coarse_code_env(module)
    env.reset()
    env._exec_globals["stop_now"] = lambda: env.request_stop(
        "time limit of 900 s reached"
    )
    _, _, _, truncated, info = env.step(
        "say_something('before')\nstop_now()\nsay_something('after')"
    )
    assert truncated is True
    assert "[say] before" in info["stdout"]
    assert "[say] after" not in info["stdout"]
    assert "EpisodeStopped" in info["stderr"]
    assert info["stderr"].endswith(
        "executing action in terminated episode: time limit of 900 s reached\n"
    )
    assert low_level.stop_requests == 1
    assert env.request_stop("operator requested stop") is False
    assert env.stop_reason == "time limit of 900 s reached"

    _, _, _, truncated, info = env.step("say_something('next program')")
    assert truncated is True
    assert info["stdout"] == ""
    assert "terminated episode" in info["stderr"]
    with pytest.raises(module.EpisodeStopped):
        env._exec_globals["go_forward"]()
    env.close()


def test_event_log_records_results_failures_and_frames(tmp_path):
    events = _import_runtime_module("baselines.mobilemanip.capx.events")
    from yor_agent.exceptions import PrimitiveFailed

    snapshots = []

    def snapshot():
        snapshots.append(len(snapshots))
        return events.RobotSnapshot(
            pose_xy_yaw=[1.0, 2.0, 0.5 + len(snapshots)],
            frame_timestamp_ns=100 + len(snapshots),
            rgb=np.full((4, 6, 3), 7, dtype=np.uint8),
            base={"lease_active": False, "last_velocity": np.zeros(3)},
        )

    def failed_turn():
        raise PrimitiveFailed(
            "turn_right_45_degrees",
            {"success": False, "reason": "timeout", "metrics": {"yaw_error_rad": 0.7}},
        )

    def interrupted():
        raise KeyboardInterrupt

    episode = tmp_path / "episode"
    log = events.EpisodeEventLog(episode, snapshot=snapshot)
    assert not episode.exists()
    log.code_block_started("go_forward()")
    value = log.call(
        "go_forward",
        (),
        {"max_speed_mps": 0.2},
        lambda: {"success": True, "metrics": {"cells": np.arange(100), "error": float("nan")}},
    )
    assert value["success"] is True
    with pytest.raises(PrimitiveFailed):
        log.call("turn_right_45_degrees", (), {"timeout_s": 30.0}, failed_turn)
    log.call("say_something", ("hello",), {}, lambda: {"success": True, "text": "hello"})
    with pytest.raises(KeyboardInterrupt):
        log.call("goto_planar_position", (0.0, 0.25), {}, interrupted)
    log.code_block_finished({"sandbox_rc": 1, "stdout": "[say] hello\n", "stderr": "x" * 5000})

    records = [json.loads(line) for line in (episode / "events.jsonl").read_text().splitlines()]
    assert [record["event"] for record in records] == [
        "code_block_started",
        "api_call",
        "api_call",
        "api_call",
        "api_call",
        "code_block_finished",
    ]
    started, forward, turn, say, planar, finished = records
    assert started["block"] == forward["block"] == finished["block"] == 1
    assert forward["outcome"] == "ok"
    assert forward["kwargs"] == {"max_speed_mps": 0.2}
    assert forward["result"]["metrics"] == {
        "cells": {"array_shape": [100], "dtype": str(np.arange(100).dtype)},
        "error": None,
    }
    assert forward["before"]["pose_xy_yaw"] == [1.0, 2.0, 1.5]
    assert forward["after"]["pose_xy_yaw"] == [1.0, 2.0, 2.5]
    assert forward["after"]["base"] == {"lease_active": False, "last_velocity": [0.0, 0.0, 0.0]}
    assert forward["frame"] == "event_frames/0001_go_forward.jpg"
    assert (episode / forward["frame"]).is_file()
    assert turn["outcome"] == "primitive_failed"
    assert turn["reason"] == "timeout"
    assert turn["result"]["metrics"] == {"yaw_error_rad": 0.7}
    assert say["outcome"] == "ok"
    assert say["args"] == ["hello"]
    assert say["before"] is None and say["after"] is None and "frame" not in say
    assert planar["outcome"] == "interrupted"
    assert planar["error"]["type"] == "KeyboardInterrupt"
    assert finished["sandbox_rc"] == 1
    assert finished["stderr"] == "..." + "x" * 4000
    assert all(record["elapsed_s"] >= 0 for record in (forward, turn, say, planar))


class _FakeFrameEnvironment(_FakeYorEnvironment):
    def navigation_frame(self, *, max_age_s):
        assert max_age_s > 0
        return types.SimpleNamespace(
            planar_pose=types.SimpleNamespace(x_m=0.5, y_m=-0.25, yaw_rad=0.1),
            timestamp_ns=42,
            rgb=np.full((4, 4, 3), 9, dtype=np.uint8),
        )

    def base_status(self):
        return {
            "lease_active": True,
            "lease_remaining_s": 0.2,
            "last_velocity": [0.0, 0.0, 0.3],
            "estop_latched": False,
            "limits": {},
        }


def test_code_env_logs_each_api_call_and_code_block(tmp_path):
    module = _import_runtime_module("baselines.mobilemanip.capx.code_env")
    resolved = _resolved_yor_config("nav_comparison.yaml", "find_can_and_give_back")
    low_level = _FakeLowLevel(resolved["primitive_config"])
    low_level.yor_environment = _FakeFrameEnvironment()
    config = module.CapXYorCodeExecConfig(
        low_level=low_level,
        apis=[],
        task_instruction="Find the can, grasp it, and return",
        navigation="coarse",
        navigation_planner=None,
        events_directory=str(tmp_path),
    )
    env = module.CapXYorCodeEnv(config)
    env.reset()
    env.step("say_something('looking for the can')")
    env.request_stop("operator requested stop")
    with pytest.raises(module.EpisodeStopped):
        env._exec_globals["go_forward"]()
    env.close()

    records = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert [(r["event"], r.get("name"), r.get("outcome")) for r in records] == [
        ("code_block_started", None, None),
        ("api_call", "say_something", "ok"),
        ("code_block_finished", None, None),
        ("api_call", "go_forward", "refused_after_stop"),
    ]
    assert records[1]["result"] == {"success": True, "text": "looking for the can"}
    assert "[say] looking for the can" in records[2]["stdout"]
    refused = records[3]
    assert refused["before"]["pose_xy_yaw"] == [0.5, -0.25, 0.1]
    assert refused["after"]["base"]["last_velocity"] == [0.0, 0.0, 0.3]
    assert refused["frame"] == "event_frames/0002_go_forward.jpg"
    assert (tmp_path / refused["frame"]).is_file()


def test_yor_snapshot_tolerates_a_missing_frame_and_status():
    events = _import_runtime_module("baselines.mobilemanip.capx.events")
    assert events.yor_snapshot(None) is None
    assert events.yor_snapshot(_FakeYorEnvironment()) is None

    class Broken:
        def navigation_frame(self, *, max_age_s):
            raise RuntimeError("no synchronized ZED RGB-D/pose frame")

        def base_status(self):
            raise TimeoutError("base RPC")

    assert events.yor_snapshot(Broken())() == events.RobotSnapshot(None, None, None, None)


class _RecordingEnvironment:
    def __init__(self):
        self.stamp = 0
        self.commands = []

    def navigation_frame(self, *, max_age_s):
        assert max_age_s > 0
        self.stamp += 1
        return types.SimpleNamespace(
            planar_pose=types.SimpleNamespace(
                x_m=0.1 * self.stamp, y_m=0.0, yaw_rad=0.2, valid=True
            ),
            timestamp_ns=self.stamp,
            rgb=np.full((47, 65, 3), 40 * self.stamp, dtype=np.uint8),
        )

    def submit_base_velocity(self, velocity):
        self.commands.append(list(velocity))
        return {"accepted": True}


def test_recorder_writes_video_trajectory_and_commands(tmp_path):
    import cv2

    recorder_module = _import_runtime_module("baselines.mobilemanip.capx.recorder")
    environment = _RecordingEnvironment()
    recorder = recorder_module.EpisodeRecorder(tmp_path, environment, hz=5.0)
    recorder.attach()
    recorder.capture()
    assert environment.submit_base_velocity([0.0, 0.0, 0.3]) == {"accepted": True}
    recorder.capture()
    summary = recorder.stop()

    assert environment.commands == [[0.0, 0.0, 0.3]]
    assert "submit_base_velocity" not in vars(environment)
    assert summary["enabled"] is True
    assert summary["errors"] == []
    assert (
        summary["frames"],
        summary["poses"],
        summary["commands"],
        summary["frames_missing"],
    ) == (2, 2, 1, 0)
    assert summary["video"] == "video.mp4"
    assert summary["video_codec"] in {"libx264", "mp4v"}
    capture = cv2.VideoCapture(str(tmp_path / "video.mp4"))
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    assert len(frames) == 2
    assert frames[0].shape[:2] == (46, 64)
    records = [
        json.loads(line) for line in (tmp_path / "trajectory.jsonl").read_text().splitlines()
    ]
    assert [record["kind"] for record in records] == ["pose", "command", "pose"]
    assert records[0]["pose_xy_yaw"] == [0.1, 0.0, 0.2]
    assert records[0]["pose_valid"] is True
    assert records[1]["velocity"] == [0.0, 0.0, 0.3]
    assert records[1]["reply"] == {"accepted": True}
    assert records[2]["frame_timestamp_ns"] == 2


def test_video_encoder_survives_the_operator_ctrl_c(tmp_path, monkeypatch):
    recorder_module = _import_runtime_module("baselines.mobilemanip.capx.recorder")
    launched = {}

    class FakePopen:
        def __init__(self, command, **kwargs):
            launched["command"] = command
            launched["kwargs"] = kwargs

    monkeypatch.setattr(recorder_module.subprocess, "Popen", FakePopen)
    recorder_module._FfmpegSink("ffmpeg", tmp_path / "video.mp4", 64, 46, 5.0)
    assert launched["kwargs"]["start_new_session"] is True
    command = launched["command"]
    assert command[command.index("-movflags") + 1] == "+frag_keyframe+empty_moov+default_base_moof"
    assert command[command.index("-g") + 1] == "5"


def test_recorder_thread_stops_and_survives_a_missing_camera_frame(tmp_path):
    import time

    recorder_module = _import_runtime_module("baselines.mobilemanip.capx.recorder")

    class NoFrame(_RecordingEnvironment):
        def navigation_frame(self, *, max_age_s):
            raise RuntimeError("no synchronized ZED RGB-D/pose frame")

    recorder = recorder_module.EpisodeRecorder(tmp_path, NoFrame(), hz=50.0)
    recorder.start()
    deadline = time.monotonic() + 2.0
    while recorder._counts["poses"] < 3 and time.monotonic() < deadline:
        time.sleep(0.01)
    summary = recorder.stop()
    assert summary["poses"] >= 3
    assert summary["frames"] == 0
    assert summary["frames_missing"] == summary["poses"]
    assert summary["video"] is None
    records = [
        json.loads(line) for line in (tmp_path / "trajectory.jsonl").read_text().splitlines()
    ]
    assert all(record["pose_xy_yaw"] is None for record in records)

    disabled = recorder_module.EpisodeRecorder(tmp_path / "none", _FakeYorEnvironment())
    disabled.start()
    assert disabled.stop() == {
        "enabled": False,
        "reason": "the environment has no camera stream",
    }
    assert not (tmp_path / "none").exists()


def test_capx_trial_loop_ends_at_the_stop_without_another_model_query(
    tmp_path, monkeypatch, capsys
):
    trial = _import_runtime_module("capx.envs.trial")
    from capx.envs.launch import LaunchArgs

    module = _import_runtime_module("baselines.mobilemanip.capx.code_env")
    run = importlib.import_module("baselines.mobilemanip.capx.run")
    env, _ = _coarse_code_env(module)
    queries = []

    def query_model(args, prompt):
        del args, prompt
        queries.append(len(queries))
        if len(queries) == 1:
            return {"content": "```python\nsay_something('first')\n```", "reasoning": ""}
        # The limit fires while the model is deciding on the next program.
        env.request_stop("time limit of 900 s reached")
        return {
            "content": "REGENERATE\n```python\nsay_something('second')\n```",
            "reasoning": "",
        }

    monkeypatch.setattr(trial, "_query_model", query_model)
    args = LaunchArgs(
        config_path="unused",
        model="fake-policy",
        use_visual_feedback=False,
        use_img_differencing=False,
        use_video_differencing=False,
        use_wrist_camera=False,
        use_legacy_multi_turn_decision_prompt=False,
        total_trials=1,
        num_workers=1,
        record_video=False,
        output_dir=str(tmp_path),
        use_oracle_code=False,
        use_parallel_ensemble=False,
        use_multimodel=False,
        web_ui=False,
    )
    run_config = {
        "output_dir": str(tmp_path),
        "record_video": False,
        "use_visual_feedback": False,
        "use_img_differencing": False,
        "use_video_differencing": False,
        "use_wrist_camera": False,
        "use_oracle_code": False,
        "use_parallel_ensemble": False,
        "use_multimodel": False,
        "save_multiturn_prompts": True,
    }
    import yaml

    multi_turn_prompt = yaml.safe_load(COARSE_CONFIG.read_text(encoding="utf-8"))[
        "multi_turn_prompt"
    ]
    summary = trial._run_single_trial(env, 0, args, run_config, multi_turn_prompt)
    output = capsys.readouterr().out
    assert queries == [0, 1]
    assert "[say] first" in output
    assert "[say] second" not in output
    assert summary.truncated is True
    assert (summary.num_code_blocks, summary.num_regenerations, summary.num_finishes) == (2, 1, 0)
    assert (
        run._termination(env.stop_reason, summary, None, trial.MULTITURN_LIMIT)
        == "time_limit"
    )
    env.close()


def test_main_records_an_episode_for_the_operator_review(tmp_path, monkeypatch):
    import yaml

    trial = _import_runtime_module("capx.envs.trial")
    environment = _import_runtime_module("baselines.mobilemanip.capx.environment")
    run = importlib.import_module("baselines.mobilemanip.capx.run")
    config = yaml.safe_load(COARSE_CONFIG.read_text(encoding="utf-8"))
    config["yor_agent_config"] = str(REPO_ROOT / "yor_agent/configs/nav_comparison.yaml")
    config["policy"]["provider"] = "scripted"
    config["policy"]["use_visual_feedback"] = False
    config["output_root"] = str(tmp_path / "outputs")
    config_path = tmp_path / "capx.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    resolved = _resolved_yor_config("nav_comparison.yaml", "find_can_and_give_back")
    low_levels = []

    def low_level_env(**kwargs):
        low_levels.append(_FakeLowLevel(resolved["primitive_config"]))
        assert kwargs["task_id"] == "find_can_and_give_back"
        return low_levels[-1]

    replies = iter(["```python\nsay_something('looking for the can')\n```", "FINISH"])
    monkeypatch.setattr(environment, "CapXYorLowLevelEnv", low_level_env)
    monkeypatch.setattr(
        trial, "_query_model", lambda args, prompt: {"content": next(replies), "reasoning": ""}
    )

    assert run.main(
        [
            "--config",
            str(config_path),
            "--task-id",
            "find_can_and_give_back",
            "--experiment",
            "mm_main",
            "--start-label",
            "kitchen",
        ]
    ) == 0
    (episode,) = (tmp_path / "outputs/mm_main/find_can_and_give_back").iterdir()
    assert episode.name.endswith("_kitchen")
    result = json.loads((episode / "result.json").read_text(encoding="utf-8"))
    assert result["termination"] == "finished"
    assert (result["condition"], result["navigation"], result["time_limit_s"]) == (
        "capx_coarse",
        "coarse",
        900.0,
    )
    assert (result["num_code_blocks"], result["num_finishes"]) == (1, 1)
    assert result["navigation_planner"] == {"enabled": False, "model": None, "memory_path": None}
    assert (result["task_success"], result["adopted"]) == (None, None)
    assert result["recording"] == {
        "enabled": False,
        "reason": "the environment has no camera stream",
    }
    metadata = json.loads((episode / "baseline_metadata.json").read_text(encoding="utf-8"))
    assert metadata["navigation"] == "coarse"
    assert low_levels[0].closed is True


def test_termination_names_match_the_yor_conditions():
    run = importlib.import_module("baselines.mobilemanip.capx.run")
    ran = types.SimpleNamespace(num_finishes=0, num_code_blocks=3)
    assert run._termination("time limit of 900 s reached", ran, None, 10) == "time_limit"
    assert run._termination("operator requested stop", ran, None, 10) == "operator_stop"
    assert run._termination(None, None, KeyboardInterrupt(), 10) == "operator_stop"
    assert run._termination(None, None, RuntimeError("lost"), 10) == "error"
    finished = types.SimpleNamespace(num_finishes=1, num_code_blocks=3)
    assert run._termination(None, finished, None, 10) == "finished"
    exhausted = types.SimpleNamespace(num_finishes=0, num_code_blocks=11)
    assert run._termination(None, exhausted, None, 10) == "multiturn_limit"
    assert run._termination(None, ran, None, 10) == "program_ended"


def test_validate_only_uses_current_yor_task_and_valid_memory(tmp_path, capsys):
    memory = tmp_path / "memory.json"
    memory.write_text(
        json.dumps(
            {
                "schema_version": "yor-video-memory-v1",
                "memory_text": "The cabinet is beyond the table.",
            }
        ),
        encoding="utf-8",
    )
    run = importlib.import_module("baselines.mobilemanip.capx.run")
    result = run.main(
        [
            "--task-id",
            "grasp_cardboard_box",
            "--navigation-memory",
            str(memory),
            "--validate-only",
        ]
    )
    assert result == 0
    output = json.loads(capsys.readouterr().out)
    assert output["valid"] is True
    assert (output["navigation"], output["navigation_planner"]) == ("yor", True)
    assert output["task_instruction"] == (
        "Grasp the small cardboard box on the white storage cabinet and keep it securely held"
    )


def test_validate_only_runs_the_full_capx_baseline_without_memory(capsys):
    run = importlib.import_module("baselines.mobilemanip.capx.run")
    result = run.main(
        [
            "--config",
            str(COARSE_CONFIG),
            "--task-id",
            "find_can_and_give_back",
            "--experiment",
            "mm_main",
            "--validate-only",
        ]
    )
    assert result == 0
    output = json.loads(capsys.readouterr().out)
    assert output["condition"] == "capx_coarse"
    assert (output["navigation"], output["navigation_planner"]) == ("coarse", False)
    assert output["navigation_memory"] is None
    assert output["time_limit_s"] == 900.0
    assert output["experiment_dir"] == str(REPO_ROOT / "outputs/capx/mm_main")


def test_validate_only_runs_the_r1pro_capx_baseline(capsys):
    run = importlib.import_module("baselines.mobilemanip.capx.run")
    result = run.main(
        [
            "--config",
            str(R1PRO_CONFIG),
            "--task-id",
            "grasp_cardboard_box",
            "--experiment",
            "capx_r1pro",
            "--validate-only",
        ]
    )
    assert result == 0
    output = json.loads(capsys.readouterr().out)
    assert output["condition"] == "capx"
    assert (output["navigation"], output["navigation_planner"]) == ("r1pro", False)
    assert output["time_limit_s"] == 900.0
    assert output["experiment_dir"] == str(REPO_ROOT / "outputs/capx/capx_r1pro")


def test_full_capx_baseline_refuses_a_navigation_memory(tmp_path):
    run = importlib.import_module("baselines.mobilemanip.capx.run")
    with pytest.raises(ValueError, match="without the startup navigation planner"):
        run.main(
            [
                "--config",
                str(COARSE_CONFIG),
                "--task-id",
                "find_can_and_give_back",
                "--navigation-memory",
                str(tmp_path / "memory.json"),
                "--validate-only",
            ]
        )


def test_gpt56_adapter_uses_responses_and_persists_policy_reasoning():
    module = importlib.import_module("baselines.mobilemanip.capx.llm")

    class Responses:
        def __init__(self):
            self.requests = []

        def create(self, **request):
            self.requests.append(request)
            index = len(self.requests)
            return types.SimpleNamespace(
                id=f"response-{index}",
                output_text="```python\nprint('ok')\n```",
            )

    responses = Responses()
    client = types.SimpleNamespace(responses=responses)
    query = module.OpenAIResponsesQuery(client=client)
    args = types.SimpleNamespace(
        model="gpt-5.6-sol",
        reasoning_effort="high",
        temperature=1.0,
        max_tokens=20480,
    )
    prompt = [
        {"role": "system", "content": "Generate Cap-X policy code."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Move to the box."},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,AAAA"},
                },
            ],
        },
    ]
    assert query(args, prompt)["content"].startswith("```python")
    query(args, prompt)
    first, second = responses.requests
    assert first["model"] == "gpt-5.6-sol"
    assert first["reasoning"] == {"effort": "high"}
    assert first["instructions"] == "Generate Cap-X policy code."
    assert first["input"][0]["content"][1]["type"] == "input_image"
    assert second["previous_response_id"] == "response-1"
    assert len(second["input"]) == 1


def test_default_config_uses_gpt56_for_planner_and_policy():
    import yaml

    config = yaml.safe_load(
        (BASELINE_DIR / "config.yaml").read_text(encoding="utf-8")
    )
    assert config["navigation_planner"]["model"] == "gpt-5.6-sol"
    assert config["policy"]["provider"] == "openai"
    assert config["policy"]["model"] == "gpt-5.6-sol"


def test_full_capx_config_shares_the_comparison_settings():
    import yaml

    coarse = yaml.safe_load(COARSE_CONFIG.read_text(encoding="utf-8"))
    default = yaml.safe_load((BASELINE_DIR / "config.yaml").read_text(encoding="utf-8"))
    assert (coarse["condition"], coarse["navigation"]) == ("capx_coarse", "coarse")
    assert "navigation_planner" not in coarse
    assert coarse["yor_agent_config"].endswith("/nav_comparison.yaml")
    assert coarse["policy"]["model"] == "gpt-5.6-sol"
    assert coarse["multi_turn_prompt"] == default["multi_turn_prompt"]
    assert coarse["allowed_tasks"] == default["allowed_tasks"]


def test_r1pro_config_uses_capx_prompts_and_the_comparison_settings():
    import yaml

    r1pro = yaml.safe_load(R1PRO_CONFIG.read_text(encoding="utf-8"))
    coarse = yaml.safe_load(COARSE_CONFIG.read_text(encoding="utf-8"))
    assert (r1pro["condition"], r1pro["navigation"]) == ("capx", "r1pro")
    assert "navigation_planner" not in r1pro
    assert r1pro["yor_agent_config"].endswith("/nav_comparison.yaml")
    assert r1pro["policy"] == coarse["policy"]
    assert r1pro["allowed_tasks"] == coarse["allowed_tasks"]
    assert r1pro["task_hints"] == ""
    # Cap-X's R1Pro multi-turn prompt, not the YOR-authored one.
    prompt = r1pro["multi_turn_prompt"]
    assert prompt.startswith("The following code blocks were just executed:")
    assert "Please respond with EXACTLY ONE of the following:" in prompt
    assert "REGENERATE" in prompt and "FINISH" in prompt
    assert "conservative code block" not in prompt


# ----------------------------------------------------------------------
# Cap-X's R1Pro API on YOR
# ----------------------------------------------------------------------


class _R1ProController:
    """Base controller fake: turns and gate-free drives move the env's pose."""

    def __init__(self, env):
        self.env = env
        self.config = types.SimpleNamespace(max_distance_m=2.0, max_turn_rad=np.pi)
        self.turns = []
        self.drives = []
        self.drive_failures = 0

    def turn_relative(self, angle_rad, **kwargs):
        self.turns.append(angle_rad)
        self.env.pose[2] = float(np.arctan2(np.sin(self.env.pose[2] + angle_rad), np.cos(self.env.pose[2] + angle_rad)))
        return {"success": True, "reason": "target_reached"}

    def drive_straight(self, distance_m, **kwargs):
        assert kwargs.get("obstacle_check") is False
        assert 0.0 < distance_m <= self.config.max_distance_m
        self.drives.append(distance_m)
        if len(self.drives) <= self.drive_failures:
            return {"success": False, "reason": "timeout"}
        self.env.pose[0] += distance_m * np.cos(self.env.pose[2])
        self.env.pose[1] += distance_m * np.sin(self.env.pose[2])
        return {"success": True, "reason": "target_reached"}


class _R1ProYorEnvironment:
    def __init__(self, env):
        self.controller = _R1ProController(env)
        self.manipulation_config = {}


class _FakeR1ProEnv:
    """The low-level facade as CapXYorR1ProApi uses it, flat floor, yaw 0."""

    def __init__(self, primitive_config, *, camera_height_m=1.0, pose=(0.0, 0.0, 0.0)):
        self.primitive_config = primitive_config
        self.pose = list(pose)
        self.yor_environment = _R1ProYorEnvironment(self)
        self.controller = self.yor_environment.controller
        self.manipulation_config = {
            "contact_graspnet_origin_to_tcp_m": 0.0,
            "contact_graspnet_to_nero_tcp_rpy_rad": [0.0, 0.0, 0.0],
        }
        self.capx_manipulation_config = {"simulate_gripper": False}
        self.episode_directory = None
        self.camera_height_m = camera_height_m
        self.depth = np.full((5, 5), 2.0, dtype=np.float32)
        self.joints = {"left": np.zeros(7), "right": np.zeros(7)}
        # TCP straight below the arm origin, which sits at the camera.
        self.tcp = {"left": [0.0, 0.0, -0.5, 0.0, 0.0, 0.0], "right": [0.0, 0.0, -0.5, 0.0, 0.0, 0.0]}
        self.gripper_width = {"left": 0.0, "right": 0.0}
        self.moves = []
        self.trajectories = []
        self.planned = []
        self.gripper_calls = []
        self.plan_success = True

    def get_observation(self):
        return {
            "robot0_robotview": {
                "images": {
                    "rgb": np.zeros((5, 5, 3), dtype=np.uint8),
                    "depth": self.depth[:, :, None],
                },
                "ground_plane": {
                    "valid": True,
                    "camera_height_m": self.camera_height_m,
                    "down_camera_xyz": [0.0, 1.0, 0.0],
                },
            },
            "base": {"pose_xy_yaw": np.asarray(self.pose, dtype=np.float64)},
        }

    def manipulation_calibration(self, arm):
        del arm
        return {
            "camera_intrinsics": np.asarray(
                [[4.0, 0.0, 2.0], [0.0, 4.0, 2.0], [0.0, 0.0, 1.0]]
            ),
            "camera_calibration_resolution": np.asarray([5, 5]),
            # Arm base at the camera: x forward, y left, z up.
            "arm_from_camera": np.asarray(
                [[0.0, 0.0, 1.0, 0.0], [-1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
            ),
        }

    def arm_status(self):
        return {
            "end_effector_frame": "tcp",
            "gripper_control": {"open_width_m": 0.08, "close_width_m": 0.0},
            **{
                name: {
                    "joint_pos": self.joints[name].tolist(),
                    "tcp_pose_xyz_rpy": list(self.tcp[name]),
                    "gripper": {"width_m": self.gripper_width[name], "force_n": 0.0},
                }
                for name in ("left", "right")
            },
        }

    def move_arm_pose(self, arm, pose, timeout_s):
        self.moves.append((arm, pose, timeout_s))
        self.tcp[arm] = list(pose)
        return {"success": True, "reason": "done"}

    def plan_arm_poses(self, arm, poses_xyz_rpy):
        self.planned.append((arm, poses_xyz_rpy))
        if not self.plan_success:
            return {"success": False, "reason": "all_mink_ik_plans_failed", "plans": [
                {"success": False, "reason": "mink_ik_unreachable"}
            ]}
        return {
            "success": True,
            "reason": "planned",
            "plans": [{"success": True, "reason": "planned", "ik_joint_target": [0.3] * 7}],
        }

    def execute_arm_trajectory(self, arm, waypoints, timeout_s):
        self.trajectories.append((arm, np.asarray(waypoints), timeout_s))
        self.joints[arm] = np.asarray(waypoints[-1], dtype=np.float64)
        # The reach lands the TCP on the planned pose.
        self.tcp[arm] = list(self.planned[-1][1][0])
        return {"success": True, "reason": "done"}

    def set_gripper(self, arm, opened, timeout_s, force_n):
        self.gripper_calls.append((arm, opened))
        self.gripper_width[arm] = 0.08 if opened else 0.0
        return {"success": True, "reason": "done"}


def _r1pro_api(module, env, **kwargs):
    kwargs.setdefault("segment_client_factory", lambda: lambda *a, **k: [])
    kwargs.setdefault("point_segment_client_factory", lambda: lambda *a, **k: [])
    kwargs.setdefault(
        "grasp_client_factory",
        lambda: (lambda *a, **k: (np.zeros((0, 4, 4)), np.zeros(0), np.zeros((0, 3)))),
    )
    return module.CapXYorR1ProApi(env, **kwargs)


def _r1pro_primitive_config():
    return _resolved_yor_config("nav_comparison.yaml", "grasp_cardboard_box")["primitive_config"]


def test_r1pro_api_exposes_capx_functions_without_yor_primitives():
    module = _import_runtime_module("baselines.mobilemanip.capx.r1pro_api")
    env = _FakeR1ProEnv(_r1pro_primitive_config())
    api = _r1pro_api(module, env)
    names = tuple(api.functions())
    assert names == module.R1PRO_FUNCTIONS
    assert {"navigate_to_pose", "grasp_object", "sample_grasp_pose", "find_object_base_rotate"} <= set(names)
    assert set(names).isdisjoint(
        {"go_forward", "goto_planar_position", "say_something", "goto_pose", "dock_to_visible_object", "drive_straight"}
    )
    assert set(names).isdisjoint(module.OMITTED_R1PRO_FUNCTIONS)


def test_r1pro_source_has_no_clearance_gate_or_advice():
    # The docking primitive's settings are read for the floor-plane fallback;
    # the primitive itself, Nav2 and the gates are never used, and the base
    # controller is driven with its obstacle check off.
    source = (BASELINE_DIR / "r1pro_api.py").read_text(encoding="utf-8")
    assert "register_visible_object_navigation_primitives" not in source
    assert "register_navigation_primitives" not in source
    assert "obstacle_check=False" in source
    for forbidden in (
        "Nav2Client",
        "navigate_to_pose(x",
        "goto_planar_position",
        "PrimitiveFailed",
        "DEPTH_UNKNOWN_IN_SWEEP",
        "swept_clearance",
        "min_front_clearance",
        "_front_clearance",
    ):
        assert forbidden not in source, forbidden


def test_world_from_camera_maps_floor_axes_and_object_points():
    module = _import_runtime_module("baselines.mobilemanip.capx.r1pro_api")
    yaw = 0.7
    # Camera pitched down: the world down direction leans into the optical axis.
    down = np.asarray([0.0, np.cos(0.3), np.sin(0.3)])
    transform = module.world_from_camera((1.0, -2.0, yaw), 1.2, down)
    rotation = transform[:3, :3]
    assert np.allclose(rotation @ rotation.T, np.eye(3))
    assert np.allclose(rotation @ -down, [0.0, 0.0, 1.0])
    _, forward, left = module.floor_axes(down)
    assert np.allclose(rotation @ forward, [np.cos(yaw), np.sin(yaw), 0.0])
    assert np.allclose(rotation @ left, [-np.sin(yaw), np.cos(yaw), 0.0])
    assert np.allclose(transform[:3, 3], [1.0, -2.0, 1.2])

    env = _FakeR1ProEnv(_r1pro_primitive_config(), pose=(1.0, 0.5, 0.0))
    masks = [{"mask": np.ones((5, 5), dtype=bool), "score": 0.9}]
    api = _r1pro_api(module, env, segment_client_factory=lambda: lambda *a, **k: masks)
    position, quaternion, extent, points, obb = api.get_object_pose("box", return_bbox_extent=True)
    # 2 m ahead of a camera 1 m above the floor at (1, 0.5): x 3, y 0.5, z 1,
    # spanning 1 m to each side (world y) and the full height (world z).
    assert np.allclose(position, [3.0, 0.5, 1.0])
    assert np.isclose(np.linalg.norm(quaternion), 1.0)
    assert points.shape == (25, 3) and np.allclose(points[:, 0], 3.0)
    assert np.allclose(points.min(axis=0), [3.0, -0.5, 0.0])
    assert np.allclose(points.max(axis=0), [3.0, 1.5, 2.0])
    # A flat square: one zero extent; the two others are its side or, since
    # PCA's axes of a square are ambiguous, its diagonal.
    assert np.isclose(np.min(extent), 0.0, atol=1e-9)
    assert 2.0 - 1e-9 <= np.max(extent) <= 2.0 * np.sqrt(2.0) + 1e-9
    assert obb.extent.shape == (3,) and obb.R.shape == (3, 3)
    assert np.allclose(obb.center, position)


def test_r1pro_perception_follows_capx_thresholds():
    module = _import_runtime_module("baselines.mobilemanip.capx.r1pro_api")
    env = _FakeR1ProEnv(_r1pro_primitive_config())
    api = _r1pro_api(module, env, segment_client_factory=lambda: lambda *a, **k: [])
    with pytest.raises(ValueError, match="No sam3 detections"):
        api.get_object_pose("box")
    with pytest.raises(ValueError, match="No sam3 detections"):
        api.sample_grasp_pose("box")
    with pytest.raises(ValueError, match="No sam3 detections"):
        api.get_sam3_mask("box")
    weak = [{"mask": np.ones((5, 5), dtype=bool), "score": 0.05}]
    api = _r1pro_api(module, env, segment_client_factory=lambda: lambda *a, **k: weak)
    assert api.get_object_pose("box") == (None, None, None, None, None)
    assert api.sample_grasp_pose("box") == (None, None)
    assert api.get_sam3_mask("box") == 25
    assert api.detect_object_sam3("box") is False
    assert api.find_object_base_rotate("box") is False
    assert env.controller.turns == [0.5] * 11
    rgb, depth = api.get_env_observation()
    assert rgb.shape == (5, 5, 3) and depth.shape == (5, 5)


def test_r1pro_sample_grasp_pose_returns_capx_pose_lists_in_the_world():
    module = _import_runtime_module("baselines.mobilemanip.capx.r1pro_api")
    env = _FakeR1ProEnv(_r1pro_primitive_config())
    masks = [{"mask": np.ones((5, 5), dtype=bool), "score": 0.9}]
    grasps = np.stack((np.eye(4), np.eye(4)))
    grasps[0, 2, 3] = 1.5
    grasps[1, 2, 3] = 2.0
    scores = np.asarray([0.2, 0.8])
    api = _r1pro_api(
        module,
        env,
        segment_client_factory=lambda: lambda *a, **k: masks,
        grasp_client_factory=lambda: (lambda *a, **k: (grasps, scores, np.zeros((2, 3)))),
    )
    pregrasps, grasp_poses = api.sample_grasp_pose("box")
    assert len(pregrasps) == len(grasp_poses) == 2
    simple_pregrasp, pregrasp_topdown = pregrasps
    simple_grasp, grasp_topdown = grasp_poses
    # The top Contact-GraspNet grasp is 2 m along the optical axis: world x.
    assert np.allclose(grasp_topdown[0], [2.0, 0.0, 1.0])
    assert np.allclose(grasp_topdown[1], [1.0, 0.0, 0.0, 0.0])
    # Its approach is the camera z axis, so the pregrasp backs off 0.1 m.
    assert np.allclose(pregrasp_topdown[0], [1.9, 0.0, 1.0])
    # The simple grasp is the object centre, approached from 0.25 m above.
    assert np.allclose(simple_grasp[0], [2.0, 0.0, 1.0])
    assert np.allclose(simple_pregrasp[0], [2.0, 0.0, 1.25])
    assert np.isclose(np.linalg.norm(simple_grasp[1]), 1.0)

    no_grasps = _r1pro_api(module, env, segment_client_factory=lambda: lambda *a, **k: masks)
    pregrasps, grasp_poses = no_grasps.sample_grasp_pose("box")
    assert len(pregrasps) == len(grasp_poses) == 1


def test_r1pro_navigate_to_pose_turns_drives_and_interpolates_like_capx():
    module = _import_runtime_module("baselines.mobilemanip.capx.r1pro_api")
    env = _FakeR1ProEnv(_r1pro_primitive_config(), pose=(0.0, 0.0, 0.0))
    api = _r1pro_api(module, env)
    controller = env.controller
    position, quaternion, yaw = api.get_robot_position()
    assert np.allclose(position, [0.0, 0.0, 0.0]) and np.allclose(quaternion, [0, 0, 0, 1])

    # 4 m straight ahead: no turn, two gate-free segments within the limit.
    assert api.navigate_to_pose((4.0, 0.0, 0.0)) is True
    assert controller.turns == []
    assert controller.drives == [2.0, 2.0]
    assert np.allclose(env.pose, [4.0, 0.0, 0.0])

    # Behind and to the left, with a final heading: turn, drive, turn.
    controller.turns.clear()
    controller.drives.clear()
    assert api.navigate_to_pose((4.0, 1.0, 0.0)) is True
    assert np.allclose(controller.turns, [np.pi / 2, -np.pi / 2])
    assert np.allclose(controller.drives, [1.0])
    assert np.allclose(env.pose, [4.0, 1.0, 0.0], atol=1e-9)

    # A goal within the tolerance moves nothing.
    controller.turns.clear()
    controller.drives.clear()
    assert api.navigate_to_pose((4.02, 1.0, 0.0)) is True
    assert controller.turns == [] and controller.drives == []

    # A failed drive fails the waypoint; Cap-X then tries the interpolated
    # waypoints toward the robot, the last of which is the robot's own pose
    # and so "succeeds" without moving, as in Cap-X.
    controller.drive_failures = 99
    controller.drives.clear()
    assert api.navigate_to_pose((8.0, 1.0, 0.0)) is True
    assert np.allclose(controller.drives, [2.0, 2.0, 2.0, 1.0])
    assert np.allclose(env.pose, [4.0, 1.0, 0.0], atol=1e-9)

    env.pose = [float("nan")] * 3
    assert api.get_robot_position() == (None, None, None)
    assert api.navigate_to_pose((1.0, 0.0, 0.0)) is False


def test_r1pro_arm_calls_run_capx_grasp_sequence_through_the_arm_rpc():
    module = _import_runtime_module("baselines.mobilemanip.capx.r1pro_api")
    env = _FakeR1ProEnv(_r1pro_primitive_config())
    api = _r1pro_api(module, env)

    # TCP 0.5 m below the arm origin (at the camera, 1 m up): world z 0.5.
    position, quaternion = api.get_current_eef_pose(arm=0)
    assert np.allclose(position, [0.0, 0.0, 0.5])
    relative_position, _ = api.get_robot_relative_eef_pose(arm=0)
    assert np.allclose(relative_position, [0.0, 0.0, 0.5])
    assert np.allclose(api.get_current_joint_positions(arm=1), np.zeros(7))

    joints = api.solve_ik(np.asarray([0.5, 0.0, 0.8]), np.asarray([1.0, 0.0, 0.0, 0.0]), arm=0)
    assert np.allclose(joints, 0.3)
    (arm, poses) = env.planned[-1]
    assert arm == "left" and np.allclose(poses[0][:3], [0.5, 0.0, -0.2])
    assert api.move_to_joint_positions(joints, arm=0) is True
    (arm, waypoints, _) = env.trajectories[-1]
    assert arm == "left" and len(waypoints) == 7
    assert np.allclose(waypoints[0], 0.0) and np.allclose(waypoints[-1], 0.3)
    assert np.max(np.abs(np.diff(waypoints, axis=0))) <= 0.05 + 1e-9

    env.plan_success = False
    assert api.solve_ik(np.zeros(3), np.asarray([1.0, 0.0, 0.0, 0.0]), arm=0) is None
    env.plan_success = True

    assert api.move_hand((np.asarray([0.5, 0.0, 0.8]), np.asarray([0.0, 0.0, 0.0, 1.0])), arm=1) is True
    assert env.moves[-1][0] == "right" and np.allclose(env.moves[-1][1][:3], [0.5, 0.0, -0.2])
    assert api.lift_arm(arm=1) is True
    assert np.isclose(env.moves[-1][1][2], -0.2 + 0.10)

    # Before any gripper command, and after opening, the hand is empty
    # whatever width it reads; only a closed gripper stopped wide holds.
    assert api.check_object_in_hand(arm=0) is False
    assert api.open_gripper(arm=0) is True
    assert env.gripper_width["left"] == 0.08
    assert api.check_object_in_hand(arm=0) is False
    assert api.close_gripper(arm=0) is True
    assert api.check_object_in_hand(arm=0) is False
    env.gripper_width["left"] = 0.03
    assert api.check_object_in_hand(arm=0) is True
    assert api.check_object_in_hand(arm=1) is False

    env.gripper_calls.clear()
    env.moves.clear()
    grasp = (np.asarray([0.5, 0.0, 0.8]), np.asarray([1.0, 0.0, 0.0, 0.0]))
    assert api.grasp_object(grasp, grasp, "box", arm=0) is True
    assert [call[1] for call in env.gripper_calls] == [True, False]
    assert len(env.trajectories) == 2
    assert env.moves[-1][0] == "left"  # the lift
    assert api.grasp_object(None, grasp, "box", arm=0) is False


def test_r1pro_navigation_pose_stands_outside_the_table_edge_facing_the_object():
    module = _import_runtime_module("baselines.mobilemanip.capx.r1pro_api")
    table = np.asarray([[x, y, 0.7] for x in np.linspace(1.0, 2.0, 6) for y in np.linspace(-1.0, 1.0, 9)])
    obj = np.asarray([[1.2, 0.1, 0.8], [1.25, 0.1, 0.8], [1.2, 0.15, 0.8]])
    x, y, yaw = module.get_navigation_pose(table, obj)
    assert np.isclose(x, 0.7) and np.isclose(y, 0.1, atol=1e-6)
    assert np.isclose(yaw, 0.0, atol=1e-6)
    hull = module.convex_hull_xy(table[:, :2])
    assert len(hull) == 4


def test_r1pro_code_env_uses_capx_prompt_without_advice(tmp_path):
    module = _import_runtime_module("baselines.mobilemanip.capx.code_env")
    r1pro = _import_runtime_module("baselines.mobilemanip.capx.r1pro_api")
    resolved = _resolved_yor_config("nav_comparison.yaml", "grasp_cardboard_box")
    low_level = _FakeR1ProEnv(resolved["primitive_config"])
    low_level.reset = lambda **kwargs: (low_level.get_observation(), {})
    low_level.render = lambda mode="rgb_array": np.zeros((5, 5, 3), dtype=np.uint8)
    low_level.compute_reward = lambda: 0.0
    low_level.task_completed = lambda: False
    low_level.request_stop = lambda: {"accepted": True}
    low_level.close = lambda: None
    low_level._sim_step_count = 0
    factories = {
        "init_sam3": lambda: (lambda *a, **k: []),
        "init_sam3_point_prompt": lambda: (lambda *a, **k: []),
        "init_contact_graspnet": lambda: (lambda *a, **k: (np.zeros((0, 4, 4)), np.zeros(0), np.zeros((0, 3)))),
    }
    originals = {name: getattr(r1pro, name) for name in factories}
    for name, value in factories.items():
        setattr(r1pro, name, value)
    try:
        config = module.CapXYorCodeExecConfig(
            low_level=low_level,
            apis=[],
            task_instruction="Grasp the small cardboard box on the white storage cabinet",
            navigation="r1pro",
            navigation_planner=None,
            events_directory=str(tmp_path),
            time_limit_s=900.0,
            task_hints="The box is straight ahead.",
        )
        env = module.CapXYorCodeEnv(config)
        observation, info = env.reset()
        prompt = observation["full_prompt"][-1]["content"][0]["text"]
        assert prompt.startswith("You are controlling a YOR robot with API described below.\nGoal: Grasp the small cardboard box on the white storage cabinet The box is straight ahead. There is a time limit of 900s to finish the task.")
        assert "do not write it in code fences" in prompt
        assert "APIs:" in prompt
        for call in ("navigate_to_pose(", "find_object_base_rotate(", "sample_grasp_pose(", "grasp_object(", "get_navigation_pose("):
            assert call in prompt
        for hidden in ("goto_pose(", "go_forward(", "goto_planar_position(", "say_something(", "drive_straight(", "dock_to_visible_object"):
            assert hidden not in prompt
        for advice in ("Operator task:", "navigation advice", "swept footprint", "fenced ```python"):
            assert advice not in prompt
        assert env.navigation_advice is None
        assert env._get_complete_prompt() == prompt
        _, _, _, _, step_info = env.step("print(get_robot_position()[2])")
        assert step_info["stdout"].strip() == "0.0"
        env.close()
    finally:
        for name, value in originals.items():
            setattr(r1pro, name, value)

    with pytest.raises(ValueError, match="no place for startup navigation advice"):
        module.CapXYorCodeEnv(
            module.CapXYorCodeExecConfig(
                low_level=low_level,
                apis=[],
                task_instruction="Find the box",
                navigation="r1pro",
                navigation_planner={"memory_path": "x"},
            ),
            planner=_FakePlanner(),
        )
