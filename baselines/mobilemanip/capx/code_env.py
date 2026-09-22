"""Cap-X code environment for the YOR baselines.

The vocabulary is chosen per run: Cap-X's R1Pro mobile-manipulation API with
Cap-X's own prompt (``r1pro``), or one of the earlier YOR-authored variants
(YOR's production navigation primitives with the one-shot startup navigation
advice, or the coarse CaP-X vocabulary without it). A stop requested by the
time limit or the operator stops the base, makes every later API call refuse,
and ends the Cap-X trial before it asks the model for another program. When
the run names an episode directory, every API call and code block is appended
to its ``events.jsonl``.
"""

from __future__ import annotations

import copy
import functools
import sys
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .bootstrap import configure_import_paths

configure_import_paths()

from capx.envs.tasks.base import CodeExecEnvConfig, CodeExecutionEnvBase  # noqa: E402
from capx.integrations.base_api import ApiBase, register_api  # noqa: E402
from yor_agent.models.navigation_planner import (  # noqa: E402
    StartupNavigationPlanner,
    policy_task_with_navigation_advice,
)

from .events import EpisodeEventLog, yor_snapshot
from .manipulation import CapXYorManipulationApi
from .navigation import CapXYorCoarseNavigationApi, CapXYorNavigationApi
from .r1pro_api import CapXYorR1ProApi


NAVIGATION_API_NAME = "CapXYorNavigationApi"
COARSE_NAVIGATION_API_NAME = "CapXYorCoarseNavigationApi"
MANIPULATION_API_NAME = "CapXYorManipulationApi"
R1PRO_API_NAME = "CapXYorR1ProApi"
#: Navigation mode of a run -> the Cap-X APIs it exposes. ``r1pro`` is one
#: API for the base and the arms, like Cap-X's R1ProControlApi.
NAVIGATION_APIS = {
    "yor": [NAVIGATION_API_NAME, MANIPULATION_API_NAME],
    "coarse": [COARSE_NAVIGATION_API_NAME, MANIPULATION_API_NAME],
    "r1pro": [R1PRO_API_NAME],
}
#: Cap-X's trial loop ends on a program whose stderr names a terminated
#: episode, before it queries the model again.
TERMINATED_EPISODE = "executing action in terminated episode"

# Cap-X's registry expects an env -> API callable; classes implement exactly
# that constructor contract. Registration is local process state only.
register_api(NAVIGATION_API_NAME, CapXYorNavigationApi)
register_api(COARSE_NAVIGATION_API_NAME, CapXYorCoarseNavigationApi)
register_api(MANIPULATION_API_NAME, CapXYorManipulationApi)
register_api(R1PRO_API_NAME, CapXYorR1ProApi)


class EpisodeStopped(BaseException):
    """An API call after the episode was stopped.

    It derives from ``BaseException`` so that an ``except Exception`` in a
    generated program cannot keep a stopped program running.
    """


@dataclass
class CapXYorCodeExecConfig(CodeExecEnvConfig):
    """Extra immutable run inputs for the YOR Cap-X environment."""

    task_instruction: str = ""
    navigation: str = "yor"
    #: Startup navigation planner settings; ``None`` runs without the planner.
    navigation_planner: dict[str, Any] | None = field(default_factory=dict)
    planner_image_max_width: int = 1024
    #: Episode directory for ``events.jsonl``; ``None`` records no events.
    events_directory: str | None = None
    #: The episode's wall-clock limit, told to the policy in Cap-X's words
    #: (``r1pro`` only); ``None`` mentions no limit.
    time_limit_s: float | None = None
    #: Sentences appended to the Goal line (``r1pro`` only).
    task_hints: str = ""


class CapXYorCodeEnv(CodeExecutionEnvBase):
    """Cap-X execution on YOR, optionally advised once by the startup planner."""

    def __init__(
        self,
        cfg: CapXYorCodeExecConfig,
        *,
        planner: StartupNavigationPlanner | None = None,
    ) -> None:
        task = str(cfg.task_instruction).strip()
        if not task:
            raise ValueError("task_instruction must not be empty")
        if int(cfg.planner_image_max_width) <= 0:
            raise ValueError("planner_image_max_width must be positive")
        if cfg.navigation not in NAVIGATION_APIS:
            raise ValueError(
                f"navigation must be one of {sorted(NAVIGATION_APIS)}: {cfg.navigation!r}"
            )
        cfg.apis = list(NAVIGATION_APIS[cfg.navigation])
        if cfg.navigation == "r1pro":
            if planner is not None or cfg.navigation_planner is not None:
                raise ValueError(
                    "Cap-X's R1Pro prompt has no place for startup navigation "
                    "advice; run r1pro without navigation_planner"
                )
            cfg.prompt = r1pro_prompt(
                task, time_limit_s=cfg.time_limit_s, hints=cfg.task_hints
            )
        else:
            cfg.prompt = _policy_prompt(task, cfg.navigation)
        self.task_instruction = task
        self.navigation = cfg.navigation
        self.navigation_advice: str | None = None
        self._planner_image_max_width = int(cfg.planner_image_max_width)
        if planner is None and cfg.navigation_planner is not None:
            planner = StartupNavigationPlanner(cfg.navigation_planner)
        self._navigation_planner = planner
        self._stop_lock = threading.Lock()
        self._stop_reason: str | None = None
        super().__init__(cfg)
        self.events = (
            None
            if cfg.events_directory is None
            else EpisodeEventLog(
                Path(cfg.events_directory),
                snapshot=yor_snapshot(
                    getattr(self.low_level_env, "yor_environment", None)
                ),
            )
        )
        # functools.wraps keeps each signature and docstring, so the prompt
        # documents the same calls.
        self._apis = {
            name: _StoppableApi(api, self.raise_if_stopped, self.events)
            for name, api in self._apis.items()
        }
        self._init_exec_globals()

    @property
    def stop_reason(self) -> str | None:
        """Why the episode was stopped, or ``None`` while it runs."""

        return self._stop_reason

    def request_stop(self, reason: str) -> bool:
        """Stop the episode like YOR's operator stop; thread-safe and idempotent.

        The base's active motion is cancelled and every later base command is
        refused by the controller; every later API call raises
        ``EpisodeStopped``, and the trial ends before its next program. An arm
        motion already under way finishes. Returns whether this call stopped
        the episode.
        """

        with self._stop_lock:
            if self._stop_reason is not None:
                return False
            self._stop_reason = str(reason).strip() or "operator requested stop"
        request = getattr(self.low_level_env, "request_stop", None)
        if callable(request):
            try:
                request()
            except Exception as exc:  # noqa: BLE001 - the API calls still refuse
                print(
                    f"[capx] the base stop request failed: {exc!r}",
                    file=sys.stderr,
                    flush=True,
                )
        return True

    def raise_if_stopped(self) -> None:
        reason = self._stop_reason
        if reason is not None:
            raise EpisodeStopped(f"{TERMINATED_EPISODE}: {reason}")

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ):
        observation, info = super().reset(seed=seed, options=options)
        if self._navigation_planner is None:
            return observation, info
        advice = self._navigation_planner.plan(
            task=self.task_instruction,
            observation=observation,
            image_max_width=self._planner_image_max_width,
        )
        if not isinstance(advice, str) or not advice.strip():
            raise RuntimeError("startup navigation planner returned empty advice")
        self.navigation_advice = advice.strip()
        advised_task = policy_task_with_navigation_advice(
            self.task_instruction, self.navigation_advice
        )
        self._task_prompt = _policy_prompt(advised_task, self.navigation)
        self._full_prompt = [
            {"role": "system", "content": self._system_prompt},
            {
                "role": "user",
                "content": [{"type": "text", "text": self._get_complete_prompt()}],
            },
        ]
        observation["full_prompt"] = copy.deepcopy(self._full_prompt)
        self._exec_globals["INPUTS"] = observation
        info["task_prompt"] = self._task_prompt
        info["navigation_advice"] = self.navigation_advice
        return observation, info

    def step(self, action: str):
        if self._stop_reason is not None:
            # A stopped episode runs no further program.
            info = {
                "sandbox_rc": 1,
                "stdout": "",
                "stderr": self._terminated_message(),
                "task_prompt": self._task_prompt,
                "task_completed": False,
            }
            return {"full_prompt": copy.deepcopy(self._full_prompt)}, 0.0, False, True, info
        if self.events is not None:
            self.events.code_block_started(action)
        observation, reward, terminated, truncated, info = super().step(action)
        if self._stop_reason is not None:
            info["stderr"] = f"{info.get('stderr') or ''}{self._terminated_message()}"
            truncated = True
        if self.events is not None:
            self.events.code_block_finished(info)
        return observation, reward, terminated, truncated, info

    def close(self) -> None:
        self.low_level_env.close()

    def _terminated_message(self) -> str:
        return f"{TERMINATED_EPISODE}: {self._stop_reason}\n"


class _StoppableApi(ApiBase):
    """One Cap-X API whose calls refuse to start once the episode is stopped."""

    def __init__(
        self,
        api: ApiBase,
        raise_if_stopped: Callable[[], None],
        events: EpisodeEventLog | None = None,
    ) -> None:
        super().__init__(api._env)
        self.api = api
        self._functions = {
            name: _stoppable(name, function, raise_if_stopped, events)
            for name, function in api.functions().items()
        }

    def functions(self) -> dict[str, Callable[..., Any]]:
        return dict(self._functions)


def _stoppable(
    name: str,
    function: Callable[..., Any],
    raise_if_stopped: Callable[[], None],
    events: EpisodeEventLog | None = None,
) -> Callable[..., Any]:
    @functools.wraps(function)
    def call(*args: Any, **kwargs: Any) -> Any:
        if events is None:
            raise_if_stopped()
            return function(*args, **kwargs)

        def run() -> Any:
            raise_if_stopped()
            return function(*args, **kwargs)

        return events.call(name, args, kwargs, run)

    return call


#: Cap-X's R1Pro prompt (capx/envs/tasks/r1pro/r1pro_behavior.py) with the
#: robot's name and the goal filled in; the time limit sentence is the one of
#: Cap-X's R1Pro pick-up-trash prompt.
R1PRO_PROMPT = """\
You are controlling a YOR robot with API described below.
Goal: {goal}
You may write python code comments for reasoning but ONLY write the executable Python code and do not write it in code fences.
If you want to use numpy, scipy for spatial transformations, opencv, pytorch, or any other libraries, you need to import them explicitly.
Note that API may fail. Make sure the code is fault tolerant.
You should consider retrying, try and except, and retrying other combinations of APIs or write your own code to recreate the same capability.
The functions (APIs) below are already imported to the environment. If you want to use numpy, you need to import it explicitly.
"""
R1PRO_TIME_LIMIT_SENTENCE = (
    "There is a time limit of {limit:g}s to finish the task. So avoid getting "
    "stuck and try to finish the task in the given time limit."
)


def r1pro_prompt(
    task: str, *, time_limit_s: float | None = None, hints: str = ""
) -> str:
    goal = [task.strip()]
    if hints and hints.strip():
        goal.append(hints.strip())
    if time_limit_s is not None:
        goal.append(R1PRO_TIME_LIMIT_SENTENCE.format(limit=float(time_limit_s)))
    return R1PRO_PROMPT.format(goal=" ".join(goal))


COARSE_NAVIGATION_PROMPT = """\
The base moves only through four navigation calls: go_forward drives exactly 1 meter forward, turn_left_45_degrees and turn_right_45_degrees turn in place by 45 degrees, and goto_planar_position moves to a position relative to the robot while keeping its heading (forward_m positive forward and at least 0, left_m positive robot-left). Compose exploration and the approach from them with Python loops and conditionals decided from perception results and the visual feedback, and use goto_planar_position when a continuous offset is needed. say_something tells the supervising operator what you intend to do.

YOR is a wide wheeled base: with the arms in their travel pose it spans about 0.87 m side to side (about 0.43 m to each side of the base center) and the grippers reach about 0.46 m ahead of the base center. go_forward and the forward part of goto_planar_position stop when live depth, or anything seen earlier in the same call, enters the swept footprint ahead. The camera looks down from the mast, so an object lower than about 0.7 m that is already within the gripper reach, or lower than about 0.45 m within half a metre of the camera, is invisible unless it was seen earlier in the same call. The 45-degree turns have no obstacle check, the sideways part of goto_planar_position has no side check, and the base cannot drive backward. When a wall or furniture appears close in the current view, treat it as a collision concern for the whole 0.87 m span: move the full swept footprint away instead of merely pointing the camera away. Before manipulating, bring the base close enough that the target is comfortably within arm reach, and use the arm on the side of the image where the target is.

"""


def _policy_prompt(task_with_optional_advice: str, navigation: str = "yor") -> str:
    navigation_text = COARSE_NAVIGATION_PROMPT if navigation == "coarse" else ""
    return f"""\
You control the physical dual-arm YOR mobile-manipulation robot.

{task_with_optional_advice}

{navigation_text}Generate executable Python that solves the task using the documented injected APIs. Use explicit arm arguments (0/'left' or 1/'right') for all manipulation calls. Use current visual evidence and returned primitive feedback; do not invent metric coordinates. Navigation and arm/gripper calls are effectful and failures interrupt the current code block. For a grasp, open the selected gripper, call sample_grasp_pose, move with goto_pose using a modest z_approach when appropriate, and close the gripper. Re-observe through normal Cap-X visual feedback after meaningful motion instead of assuming success.

Return only Python in a fenced ```python code block. Do not call private environment methods, undocumented robot controls, shell commands, or network clients."""
