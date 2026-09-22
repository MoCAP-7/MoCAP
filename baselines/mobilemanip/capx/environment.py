"""Gym facade that lets Cap-X drive the production YOR environment."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .bootstrap import configure_import_paths

configure_import_paths()

from capx.envs.base import BaseEnv  # noqa: E402
from gymnasium import Env  # noqa: E402
from yor_agent.launch import build_environment, load_config  # noqa: E402


class CapXYorLowLevelEnv(BaseEnv):
    """A deliberately thin adapter around the current ``YorEnvironment``.

    It never reimplements motion primitives.  The API adapters receive the
    wrapped production environment and register the same primitives used by
    ``yor_agent``.  ``render()`` returns the most recently observed frame so
    the startup navigation planner and Cap-X policy see the same initial image.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 10}

    def __init__(
        self,
        *,
        yor_config_path: str | Path | None = None,
        task_id: str | None = None,
        yor_environment: Any | None = None,
        resolved_yor_config: Mapping[str, Any] | None = None,
        capx_manipulation_config: Mapping[str, Any] | None = None,
        episode_directory: str | Path | None = None,
    ) -> None:
        super().__init__()
        #: Where an API saves files the policy asks for; ``None``: the cwd.
        self.episode_directory = (
            None if episode_directory is None else Path(episode_directory)
        )
        if yor_environment is None:
            if yor_config_path is None or not task_id:
                raise ValueError(
                    "yor_config_path and task_id are required when no environment is injected"
                )
            resolved = load_config(Path(yor_config_path), task_id=task_id)
            yor_environment = build_environment(resolved)
        else:
            resolved = copy.deepcopy(dict(resolved_yor_config or {}))

        self.yor_environment = yor_environment
        self.resolved_yor_config = resolved
        self.task_id = task_id or str(resolved.get("task", {}).get("id", ""))
        self.primitive_config = copy.deepcopy(dict(resolved.get("primitive_config", {})))
        robot_config = dict(resolved.get("robot", {}))
        self.manipulation_config = copy.deepcopy(
            dict(
                getattr(
                    yor_environment,
                    "manipulation_config",
                    robot_config.get("manipulation", {}),
                )
            )
        )
        self.navigation_config = copy.deepcopy(
            dict(robot_config.get("navigation", {}))
        )
        self.capx_manipulation_config = copy.deepcopy(
            dict(capx_manipulation_config or {})
        )
        self._latest_observation: dict[str, Any] | None = None
        self._record_video = False
        self._recorded_frames: list[np.ndarray] = []
        self._closed = False
        self._sim_step_count = 0

    @property
    def controller(self) -> Any:
        return self.yor_environment.controller

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        del options
        # Seed through gymnasium directly: Cap-X's BaseEnv.reset is an abstract
        # stub that only raises NotImplementedError.
        Env.reset(self, seed=seed)
        observation = self.yor_environment.reset()
        self._cache_observation(observation)
        return observation, {"task_id": self.task_id}

    def step(self, action: Any):
        del action
        raise RuntimeError(
            "Direct Gym actions are disabled; Cap-X must use the injected YOR APIs."
        )

    def get_observation(self) -> dict[str, Any]:
        observation = self.yor_environment.observe()
        self._cache_observation(observation)
        return observation

    def compute_reward(self, obs: Any = None) -> float:
        del obs
        return 0.0

    def task_completed(self) -> bool:
        # Physical task success is scored by the experiment evaluator, not by
        # successful execution of a generated Python program.
        return False

    def render(self, mode: str = "rgb_array") -> np.ndarray:
        del mode
        if self._latest_observation is None:
            self.get_observation()
        assert self._latest_observation is not None
        image = _observation_rgb(self._latest_observation)
        if image is None:
            raise RuntimeError("The YOR observation did not contain an RGB image")
        return np.asarray(image).copy()

    def start_video_recording(self) -> None:
        self._record_video = True
        self._recorded_frames = []

    def stop_video_recording(self) -> list[np.ndarray]:
        self._record_video = False
        return list(self._recorded_frames)

    def enable_video_capture(
        self, enabled: bool = True, *, clear: bool = True
    ) -> None:
        if clear:
            self._recorded_frames = []
        self._record_video = bool(enabled)

    def get_video_frames(self, *, clear: bool = False) -> list[np.ndarray]:
        frames = [frame.copy() for frame in self._recorded_frames]
        if clear:
            self._recorded_frames = []
        return frames

    def get_video_frame_count(self) -> int:
        return len(self._recorded_frames)

    def get_video_frames_range(self, start: int, end: int) -> list[np.ndarray]:
        return [frame.copy() for frame in self._recorded_frames[start:end]]

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.yor_environment.safe_shutdown()

    def _cache_observation(self, observation: Mapping[str, Any]) -> None:
        self._latest_observation = dict(observation)
        image = _observation_rgb(observation)
        if self._record_video and image is not None:
            self._recorded_frames.append(np.asarray(image).copy())

    def __getattr__(self, name: str) -> Any:
        # Only public production-environment methods are delegated.  This keeps
        # the adapter small while retaining the exact hardware implementation.
        if name.startswith("_"):
            raise AttributeError(name)
        wrapped = self.__dict__.get("yor_environment")
        if wrapped is None:
            raise AttributeError(name)
        return getattr(wrapped, name)


def _observation_rgb(observation: Mapping[str, Any]) -> Any | None:
    image = observation.get("image")
    if image is not None:
        return image
    camera = observation.get("robot0_robotview")
    if not isinstance(camera, Mapping):
        return None
    images = camera.get("images")
    if not isinstance(images, Mapping):
        return None
    return images.get("rgb")
