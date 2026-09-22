"""Configuration and validation for the real-robot CoW baseline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Mapping, TypeVar, get_type_hints

import yaml


@dataclass(frozen=True)
class UpstreamConfig:
    repository_path: str = "/home/yor/codefield/cow"
    verify_commit: bool = True


@dataclass(frozen=True)
class AgentConfig:
    localizer: str = "clip_grad"
    clip_checkpoint: str = "/home/yor/models/yoloe/clip/ViT-B-32.pt"
    prompt_templates: str = "prompt_templates/imagenet_template.json"
    threshold: float | None = None
    center_only: bool = True
    device: str = "cuda"
    image_size: int = 672
    fov_deg: float = 90.0
    voxel_size_m: float = 0.125
    rotation_deg: float = 30.0
    forward_m: float = 0.25
    max_ceiling_height_m: float = 1.9
    floor_tolerance_m: float = 0.15
    in_cspace: bool = True
    fail_stop: bool = True
    stop_radius_m: float = 0.65
    max_steps: int = 250

    def __post_init__(self) -> None:
        if self.localizer not in {"clip_grad", "owl"}:
            raise ValueError("agent.localizer must be clip_grad or owl")
        if self.threshold is not None and not 0.0 < self.threshold:
            raise ValueError("agent.threshold must be positive or null")
        if self.image_size <= 0 or not 0.0 < self.fov_deg < 180.0:
            raise ValueError("agent.image_size must be positive and agent.fov_deg in (0, 180)")
        # CoW's spin phase asserts that the unseen arc is a whole number of turns.
        if self.rotation_deg <= 0 or (360 - int(self.fov_deg)) % self.rotation_deg != 0:
            raise ValueError("(360 - fov_deg) must be a multiple of agent.rotation_deg")
        for name in ("voxel_size_m", "forward_m", "max_ceiling_height_m", "floor_tolerance_m"):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"agent.{name} must be positive")
        if not 0.0 < self.stop_radius_m <= 5.0:
            raise ValueError("agent.stop_radius_m must be in (0, 5]")
        if self.max_steps <= 0:
            raise ValueError("agent.max_steps must be positive")


@dataclass(frozen=True)
class CameraConfig:
    source_resolution: tuple[int, int] = (672, 376)
    intrinsics: tuple[tuple[float, float, float], ...] = (
        (267.214, 0.0, 337.181),
        (0.0, 267.150, 182.288),
        (0.0, 0.0, 1.0),
    )
    camera_height_m: float | None = None
    down_camera_xyz: tuple[float, float, float] | None = None

    def __post_init__(self) -> None:
        if len(self.source_resolution) != 2 or min(self.source_resolution) <= 0:
            raise ValueError("camera.source_resolution must be [width, height]")
        if len(self.intrinsics) != 3 or any(len(row) != 3 for row in self.intrinsics):
            raise ValueError("camera.intrinsics must be a 3x3 matrix")
        if self.camera_height_m is not None and not 0.2 <= self.camera_height_m <= 2.5:
            raise ValueError("camera.camera_height_m must be in [0.2, 2.5] or null")
        if self.down_camera_xyz is not None and len(self.down_camera_xyz) != 3:
            raise ValueError("camera.down_camera_xyz must have three values or be null")


@dataclass(frozen=True)
class RobotConfig:
    pose_source: str = "zed"

    def __post_init__(self) -> None:
        if self.pose_source not in {"zed", "dead_reckoning"}:
            raise ValueError("robot.pose_source must be zed or dead_reckoning")


@dataclass(frozen=True)
class ExperimentConfig:
    output_root: str = "outputs/cow"
    maximum_duration_s: float = 900.0
    save_images: bool = True
    map_image_every_steps: int = 10
    end_on_safety_stop: bool = True

    def __post_init__(self) -> None:
        if self.maximum_duration_s <= 0.0:
            raise ValueError("experiment.maximum_duration_s must be positive")
        if self.map_image_every_steps <= 0:
            raise ValueError("experiment.map_image_every_steps must be positive")


@dataclass(frozen=True)
class CowConfig:
    upstream: UpstreamConfig = field(default_factory=UpstreamConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    robot: RobotConfig = field(default_factory=RobotConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


T = TypeVar("T")


def load_config(path: str | Path) -> CowConfig:
    loaded = yaml.safe_load(Path(path).expanduser().read_text(encoding="utf-8")) or {}
    return build_config(CowConfig, loaded)


def build_config(cls: type[T], data: Any) -> T:
    if not isinstance(data, Mapping):
        raise ValueError(f"{cls.__name__} must be a mapping")
    names = {item.name for item in fields(cls)}
    unknown = sorted(set(data) - names)
    if unknown:
        raise ValueError(f"unknown {cls.__name__} keys: {', '.join(unknown)}")
    hints = get_type_hints(cls)
    values: dict[str, Any] = {}
    for item in fields(cls):
        if item.name not in data:
            continue
        hint = hints[item.name]
        value = data[item.name]
        values[item.name] = build_config(hint, value) if is_dataclass(hint) else _freeze(value)
    return cls(**values)


def _freeze(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value
