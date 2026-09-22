"""One CoW episode on YOR: unmodified CoW decisions, YOR sensing and actuation."""

from __future__ import annotations

import importlib
import json
import math
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import torch

from .camera import LevelCamera
from .config import CowConfig
from .console import EpisodeNarrator, stderr_log
from .robot import CowActuator, safety_stop_reason
from .tasks import CowTask
from .visualization import attention_overlay, map_image


# Frames to wait for a valid ZED tracking pose before the episode gives up.
POSE_RETRY_FRAMES = 15


def build_agent(
    config: CowConfig,
    *,
    agent_class: type,
    repository: Path,
    goal: str,
    camera_height_m: float,
    threshold: float,
) -> Any:
    """Construct CoW's agent with the episode goal as its only class."""

    settings = config.agent
    templates = json.loads((repository / settings.prompt_templates).read_text(encoding="utf-8"))
    model_name = settings.clip_checkpoint if settings.localizer == "clip_grad" else "ViT-B/32"
    agent = agent_class(
        model_name,
        [goal],
        [goal],
        templates,
        settings.fov_deg,
        settings.image_size,
        settings.image_size,
        camera_height_m,
        settings.floor_tolerance_m,
        threshold,
        torch.device(settings.device),
        max_ceiling_height=settings.max_ceiling_height_m,
        rotation_degrees=settings.rotation_deg,
        forward_distance=settings.forward_m,
        voxel_size_m=settings.voxel_size_m,
        in_cspace=settings.in_cspace,
        fail_stop=settings.fail_stop,
        center_only=settings.center_only,
    )
    if settings.localizer == "clip_grad":
        agent.transform = clip_grad_transform()
    return agent


def clip_grad_transform() -> Any:
    """CoW's CLIP-Grad input transform with the resize CoW was evaluated with.

    CoW pins a torchvision release whose tensor ``Resize`` does not antialias.
    Newer releases antialias tensors by default, which changes the image CLIP
    sees, so the original behaviour is requested explicitly.
    """

    import torchvision.transforms as T

    constants = importlib.import_module("src.simulation.constants")
    return T.Compose(
        [
            T.ToTensor(),
            T.Resize(224, interpolation=T.InterpolationMode.BICUBIC, antialias=False),
            T.Normalize(constants.CLIP_MEAN, constants.CLIP_STD),
        ]
    )


def planar_pose_matrix(pose_xy_yaw: Any) -> torch.Tensor:
    """Planar camera pose (x forward, y left, counter-clockwise yaw) in CoW's frame.

    CoW's agent frame has x to the left, y up and z forward; a left turn is a
    positive rotation about y.
    """

    x, y, yaw = (float(value) for value in list(pose_xy_yaw)[:3])
    if not all(math.isfinite(value) for value in (x, y, yaw)):
        raise ValueError(f"camera pose must be finite: {[x, y, yaw]}")
    matrix = torch.eye(4, dtype=torch.float32)
    c, s = math.cos(yaw), math.sin(yaw)
    matrix[:3, :3] = torch.tensor([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    matrix[0, 3] = y
    matrix[2, 3] = x
    return matrix


class ZedPoseInjector:
    """Give CoW's map the measured camera pose instead of nominal action deltas.

    CoW's ``update_map`` multiplies ``camera_to_agent`` by the delta of the
    previous action unless its depth-difference test marks that action as
    failed. The injector writes the measured pose pre-multiplied by the inverse
    of exactly that delta, so the pose CoW uses is the measured one. When CoW
    resets its map mid-episode it restarts from identity and applies the
    previous action again; the injector re-anchors to that frame.
    """

    def __init__(self, exploration: Any) -> None:
        self.exploration = exploration
        self.world_to_map: torch.Tensor | None = None
        self.resets = 0
        self._reset_pending = False
        original_reset = exploration.reset

        def reset_and_mark() -> None:
            original_reset()
            self._reset_pending = True

        exploration.reset = reset_and_mark

    def start(self, pose_xy_yaw: Any) -> None:
        self.world_to_map = torch.linalg.inv(planar_pose_matrix(pose_xy_yaw))
        self._reset_pending = False

    def before_act(self, pose_xy_yaw: Any, depth: np.ndarray, last_action: str | None) -> bool:
        if self.world_to_map is None:
            raise RuntimeError("ZedPoseInjector.start must be called first")
        failed = self._predicts_failed_action(depth)
        movement = (
            torch.eye(4)
            if failed or last_action is None
            else self.exploration._action_to_movement_matrix(last_action).float()
        )
        self.exploration.camera_to_agent = (
            self.world_to_map @ planar_pose_matrix(pose_xy_yaw) @ torch.linalg.inv(movement)
        )
        return failed

    def after_act(self, pose_xy_yaw: Any, last_action_before_act: str | None) -> bool:
        if not self._reset_pending:
            return False
        self._reset_pending = False
        self.resets += 1
        movement = self.exploration._action_to_movement_matrix(last_action_before_act).float()
        self.world_to_map = movement @ torch.linalg.inv(planar_pose_matrix(pose_xy_yaw))
        return True

    def _predicts_failed_action(self, depth: np.ndarray) -> bool:
        # Same test as CoW's update_map applies before integrating a motion.
        previous = self.exploration.last_observation
        if previous is None or not self.exploration.fail_stop:
            return False
        difference = torch.abs(previous - torch.as_tensor(depth).squeeze())
        return bool(difference.mean().item() < 0.09 and difference.std().item() < 0.09)


class CowEpisode:
    def __init__(
        self,
        config: CowConfig,
        *,
        agent: Any,
        environment: Any,
        camera: LevelCamera,
        task: CowTask,
        output_dir: str | Path,
        voxel_types: Any = None,
        no_motion: bool = False,
        clock: Callable[[], float] = time.monotonic,
        log: Callable[[str], None] = stderr_log,
    ) -> None:
        self.config = config
        self.narrator = EpisodeNarrator(log)
        self.agent = agent
        self.environment = environment
        self.camera = camera
        self.task = task
        self.output_dir = Path(output_dir)
        self.voxel_types = voxel_types
        self.no_motion = no_motion
        self.clock = clock
        self.actuator = CowActuator(
            environment.controller,
            rotation_deg=config.agent.rotation_deg,
            forward_m=config.agent.forward_m,
        )
        self.injector = (
            ZedPoseInjector(agent.fbe) if config.robot.pose_source == "zed" else None
        )
        self.last_attention: torch.Tensor | None = None
        original_localize = agent.localize_object

        def localize_and_record(observations: Mapping[str, Any]) -> torch.Tensor:
            attention = original_localize(observations)
            self.last_attention = attention
            return attention

        agent.localize_object = localize_and_record
        self._trace_path = self.output_dir / "trace.jsonl"
        self._artifacts = self.output_dir / "artifacts"

    def run(self) -> dict[str, Any]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.config.experiment.save_images:
            self._artifacts.mkdir(exist_ok=True)
        (self.output_dir / "config.json").write_text(
            json.dumps(self.config.as_dict(), indent=2), encoding="utf-8"
        )
        started = self.clock()
        steps = 0
        termination = "max_steps"
        action: str | None = None
        error: str | None = None
        safety_stop: dict[str, Any] | None = None
        try:
            self.narrator.say("waiting for the Pi base status and the first ZED frame")
            observation = self._with_valid_pose(self.environment.reset())
            self.agent.reset()
            if observation is None:
                termination = "pose_invalid"
            else:
                if self.injector is not None:
                    self.injector.start(_pose(observation))
                self._write_event("episode_start", goal=self.task.goal, pose=_pose(observation))
                for step in range(self.config.agent.max_steps):
                    if self.clock() - started >= self.config.experiment.maximum_duration_s:
                        termination = "time_budget_exhausted"
                        break
                    if bool(observation["base"].get("estop_latched", False)):
                        termination = "estop_latched"
                        break
                    steps = step + 1
                    action = self._decide(step, observation, started)
                    if action == "Stop":
                        termination = "cow_stop"
                        break
                    if self.no_motion:
                        termination = "no_motion_first_decision"
                        break
                    primitive = self.actuator.execute(action)
                    self._write_event(
                        "primitive", step=step, action=action, result=_primitive_summary(primitive)
                    )
                    reason = safety_stop_reason(primitive)
                    if reason is not None and self.config.experiment.end_on_safety_stop:
                        # CoW has no collision handling of its own; YOR's refusal stands in for a collision.
                        termination = "safety_stop"
                        safety_stop = {"step": step, "action": action, "reason": reason}
                        self.narrator.say(f"YOR safety stop on {action} ({reason}): the episode ends here")
                        break
                    observation = self._with_valid_pose(self.environment.observe_next())
                    if observation is None:
                        termination = "pose_invalid"
                        break
        except KeyboardInterrupt:
            termination = "operator_interrupt"
            self.narrator.say("operator interrupt: stopping the base")
        except Exception as exc:  # noqa: BLE001 - the base stop and the episode record must still happen
            termination = "error"
            error = f"{type(exc).__name__}: {exc}"
            self._write_event("error", error=error, traceback=traceback.format_exc())
        finally:
            stop = self._confirm_stop()
        try:
            self._save_map(steps, final=True)
        except Exception as exc:  # noqa: BLE001 - a map image must not hide the episode result
            self._write_event("map_image_error", error=f"{type(exc).__name__}: {exc}")
        result = {
            "task_id": self.task.task_id,
            "instruction": self.task.instruction,
            "goal": self.task.goal,
            "termination": termination,
            "error": error,
            "last_action": action,
            "steps": steps,
            "elapsed_s": self.clock() - started,
            "map_resets": None if self.injector is None else self.injector.resets,
            "stop": stop,
            "safety_stop": safety_stop,
            "task_success": None,
            "task_success_source": None,
        }
        self._write_event("episode_end", **result)
        return result

    def _with_valid_pose(self, observation: Mapping[str, Any]) -> Mapping[str, Any] | None:
        """Return an observation whose pose CoW's map can use, or None.

        A frame without valid ZED tracking carries a non-finite pose. With the
        ZED pose source such a pose would corrupt CoW's map, so the episode
        waits a few frames for tracking and otherwise ends.
        """

        if self.injector is None:
            return observation
        for attempt in range(POSE_RETRY_FRAMES + 1):
            pose = _pose(observation)
            if all(math.isfinite(value) for value in pose):
                return observation
            self._write_event("pose_invalid_frame", attempt=attempt, pose=pose)
            if attempt < POSE_RETRY_FRAMES:
                observation = self.environment.observe_next()
        return None

    def _decide(self, step: int, observation: Mapping[str, Any], started: float) -> str:
        images = observation["robot0_robotview"]["images"]
        rgb, depth = self.camera.render(images["rgb"], images["depth"])
        pose = _pose(observation)
        previous_action = self.agent.last_action
        failed = None
        if self.injector is not None:
            failed = self.injector.before_act(pose, depth, previous_action)
        decide_started = self.clock()
        action = self.agent.act({"rgb": rgb, "depth": depth, "object_goal": self.task.goal})
        decide_s = self.clock() - decide_started
        reanchored = False
        if self.injector is not None:
            reanchored = self.injector.after_act(pose, previous_action)
        attention = self.last_attention
        hits = [] if attention is None else torch.nonzero(attention > 0).tolist()
        exploration = self.agent.fbe
        self._write_event(
            "decision",
            step=step,
            elapsed_s=self.clock() - started,
            pose=pose,
            action=action,
            mode=getattr(self.agent.agent_mode, "name", str(self.agent.agent_mode)),
            attention_pixels=hits[:8],
            attention_max=None if attention is None else float(attention.max()),
            roi_exists=bool(exploration.poll_roi_exists()),
            roi_targets=len(exploration.roi_targets),
            exploration_targets=len(exploration.exploration_targets),
            map_voxels=exploration.voxels.number_of_nodes(),
            predicted_failed_previous_action=failed,
            map_reset_this_step=reanchored,
            decide_s=decide_s,
            level_valid_depth_fraction=float((depth > 0).mean()),
        )
        if self.config.experiment.save_images:
            attention_overlay(rgb, attention).save(self._artifacts / f"step_{step:04d}_level_rgb.jpg", quality=90)
            if step % self.config.experiment.map_image_every_steps == 0:
                self._save_map(step)
        return action

    def _save_map(self, step: int, *, final: bool = False) -> None:
        if not self.config.experiment.save_images or self.voxel_types is None:
            return
        image = map_image(self.agent.fbe, self.voxel_types)
        if image is not None:
            name = "final_map.png" if final else f"step_{step:04d}_map.png"
            image.save(self._artifacts / name)

    def _confirm_stop(self) -> dict[str, Any]:
        try:
            return _primitive_summary(dict(self.environment.controller.stop()))
        except Exception as exc:  # noqa: BLE001 - the episode result must still be written
            return {"success": False, "reason": f"{type(exc).__name__}: {exc}"}

    def _write_event(self, event_type: str, **payload: Any) -> None:
        record = _json_safe({"type": event_type, "wall_time": time.time(), **payload})
        with self._trace_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        try:
            self.narrator.event(record)
        except Exception:  # noqa: BLE001 - console output must never end the episode
            pass


def _pose(observation: Mapping[str, Any]) -> list[float]:
    return [float(value) for value in list(observation["base"]["pose_xy_yaw"])[:3]]


def _primitive_summary(result: Mapping[str, Any]) -> dict[str, Any]:
    keys = ("success", "status", "primitive", "reason", "elapsed_s", "start_pose_xy_yaw", "final_pose_xy_yaw")
    return {key: result.get(key) for key in keys if key in result}


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, torch.Tensor):
        return _json_safe(value.detach().cpu().tolist())
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value
