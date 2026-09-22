"""Validated configuration for the isolated ApexNav-on-YOR baseline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
import ipaddress
from pathlib import Path
from typing import Any, Mapping, TypeVar

import yaml


@dataclass(frozen=True)
class CameraConfig:
    calibration_resolution: tuple[int, int] = (672, 376)
    intrinsics: tuple[tuple[float, float, float], ...] = (
        (267.214, 0.0, 337.181),
        (0.0, 267.150, 182.288),
        (0.0, 0.0, 1.0),
    )
    minimum_depth_m: float = 0.10
    maximum_depth_m: float = 5.0
    fallback_camera_height_m: float = 1.122339129447937
    fallback_ground_down_xyz: tuple[float, float, float] = (
        0.013630234132681617,
        0.9477663115351749,
        0.31867418382495005,
    )
    require_dynamic_ground_plane: bool = True
    ground_plane_max_age_s: float = 2.0
    odometry_hz: float = 30.0
    mapping_hz: float = 8.0
    perception_hz: float = 1.0
    object_cloud_stride: int = 2
    maximum_object_points: int = 12000


@dataclass(frozen=True)
class RobotConfig:
    zed_host: str = "127.0.0.1"
    zed_port: int = 6000
    base_rpc_host: str = "192.168.1.10"
    base_rpc_port: int = 5557
    base_rpc_timeout_s: float = 2.0
    base_to_camera_forward_m: float = 0.2143
    base_to_camera_left_m: float = 0.0603
    command_timeout_s: float = 0.15
    sensor_timeout_s: float = 0.50
    control_hz: float = 20.0
    maximum_linear_mps: float = 0.18
    maximum_yaw_rad_s: float = 0.35
    minimum_linear_mps: float = 0.05
    minimum_yaw_rad_s: float = 0.16


@dataclass(frozen=True)
class VLMConfig:
    grounding_dino_url: str = "http://127.0.0.1:12181/gdino"
    blip2_itm_url: str = "http://127.0.0.1:12182/blip2itm"
    mobile_sam_url: str = "http://127.0.0.1:12183/mobile_sam"
    yolov7_url: str = "http://127.0.0.1:12184/yolov7"
    request_timeout_s: float = 45.0
    jpeg_quality: int = 90
    yolo_confidence_threshold: float = 0.30
    yolo_iou_threshold: float = 0.50
    grounding_dino_box_threshold: float = 0.40
    grounding_dino_text_threshold: float = 0.25
    mask_erosion_pixels: int = 1
    maximum_consecutive_errors: int = 3


@dataclass(frozen=True)
class PlannerConfig:
    ros_domain_id: int = 47
    map_size_m: float = 80.0
    map_resolution_m: float = 0.05
    # The robot's largest reach from the base center, arms included, plus a margin.
    obstacle_inflation_m: float = 0.50
    optimizer_safe_distance_m: float = 0.20
    footprint_length_m: float = 0.16
    footprint_width_m: float = 0.16
    wheel_base_m: float = 0.50
    # Retime every optimized trajectory along its own path to the reference
    # limits below. They stay under the robot limits so the tracker keeps
    # authority to close the lag of a base that under-delivers.
    time_scale_to_limits: bool = True
    reference_max_linear_mps: float = 0.12
    reference_max_yaw_rad_s: float = 0.24
    # Predict replans made while moving as far ahead as recent plans took.
    adaptive_replan_time: bool = True
    minimum_obstacle_height_m: float = 0.08
    maximum_obstacle_height_m: float = 1.60
    virtual_ground_height_m: float = -0.34
    maximum_episode_s: float = 900.0
    trigger_delay_s: float = 3.0
    trigger_retry_s: float = 1.0
    initial_scan_timeout_s: float = 50.0
    minimum_map_free_cells: int = 500


@dataclass(frozen=True)
class ExperimentConfig:
    output_dir: str = "./outputs/apexnav"
    priors_path: str = "baselines/nav/apexnav/target_priors.yaml"
    save_detection_images: bool = True
    # Keep the per-run debug recording under <output>/debug.
    record_debug: bool = True


@dataclass(frozen=True)
class ViewerConfig:
    # Live map viewer: foxglove_bridge serving whitelisted topics to Lichtblick.
    enabled: bool = True
    # Empty binds this host's Tailscale IPv4, or 127.0.0.1 without Tailscale.
    address: str = ""
    port: int = 8765
    # Publish world to base_footprint on /tf while a viewer subscribes, for the
    # layout that follows the robot.
    publish_tf: bool = True
    # Messages queued for each viewer before the oldest are dropped.
    message_backlog_size: int = 128
    # Upper bound on the bridge's ROS subscription depth. Marker topics publish
    # in bursts that a shallow queue would drop.
    max_qos_depth: int = 100
    num_threads: int = 2


@dataclass(frozen=True)
class ApexNavConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    robot: RobotConfig = field(default_factory=RobotConfig)
    vlm: VLMConfig = field(default_factory=VLMConfig)
    planner: PlannerConfig = field(default_factory=PlannerConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    viewer: ViewerConfig = field(default_factory=ViewerConfig)

    def validate(self) -> "ApexNavConfig":
        width, height = self.camera.calibration_resolution
        if width <= 0 or height <= 0:
            raise ValueError("camera calibration resolution must be positive")
        if len(self.camera.intrinsics) != 3 or any(
            len(row) != 3 for row in self.camera.intrinsics
        ):
            raise ValueError("camera intrinsics must be a 3x3 matrix")
        if not (
            0.0
            < self.camera.minimum_depth_m
            < self.camera.maximum_depth_m
            <= 20.0
        ):
            raise ValueError("invalid camera depth range")
        if not 1.0 <= self.camera.odometry_hz <= 60.0:
            raise ValueError("odometry_hz must be in [1, 60]")
        if not 0.1 <= self.camera.mapping_hz <= 30.0:
            raise ValueError("mapping_hz must be in [0.1, 30]")
        if not 0.05 <= self.camera.perception_hz <= 5.0:
            raise ValueError("perception_hz must be in [0.05, 5]")
        if not 1 <= self.camera.object_cloud_stride <= 16:
            raise ValueError("object_cloud_stride must be in [1, 16]")
        if self.camera.maximum_object_points < 100:
            raise ValueError("maximum_object_points must be at least 100")
        if not 0 <= self.planner.ros_domain_id <= 232:
            raise ValueError("ROS_DOMAIN_ID must be in [0, 232]")
        if not 0.2 <= self.planner.trigger_retry_s <= 5.0:
            raise ValueError("trigger_retry_s must be in [0.2, 5.0]")
        if not 10.0 <= self.planner.initial_scan_timeout_s <= 60.0:
            raise ValueError("initial_scan_timeout_s must be in [10, 60]")
        if self.planner.minimum_map_free_cells < 100:
            raise ValueError("minimum_map_free_cells must be at least 100")
        if self.planner.obstacle_inflation_m <= 0.0:
            raise ValueError("planner obstacle inflation must be positive")
        if self.planner.optimizer_safe_distance_m <= 0.0:
            raise ValueError("planner optimizer safety distance must be positive")
        if min(self.planner.footprint_length_m, self.planner.footprint_width_m) <= 0.0:
            raise ValueError("planner footprint dimensions must be positive")
        if not (
            0.0 < self.robot.minimum_linear_mps <= self.robot.maximum_linear_mps
        ):
            raise ValueError("invalid robot linear velocity limits")
        if not (
            0.0
            < self.planner.reference_max_linear_mps
            <= self.robot.maximum_linear_mps
        ):
            raise ValueError("planner reference speed must be positive and within the robot limit")
        if not (
            0.0 < self.planner.reference_max_yaw_rad_s <= self.robot.maximum_yaw_rad_s
        ):
            raise ValueError("planner reference yaw rate must be positive and within the robot limit")
        if not (
            0.0 < self.robot.minimum_yaw_rad_s <= self.robot.maximum_yaw_rad_s
        ):
            raise ValueError("invalid robot yaw velocity limits")
        if self.robot.control_hz < 10.0:
            raise ValueError("control_hz must be at least 10 Hz")
        if not 0.2 <= self.robot.base_rpc_timeout_s <= 5.0:
            raise ValueError("base RPC timeout must be in [0.2, 5.0] s")
        if not 0.0 < self.robot.command_timeout_s < 0.25:
            raise ValueError("command timeout must be shorter than the Pi lease")
        if not 1 <= self.vlm.maximum_consecutive_errors <= 10:
            raise ValueError("maximum_consecutive_errors must be in [1, 10]")
        if self.viewer.address:
            try:
                ipaddress.ip_address(self.viewer.address)
            except ValueError as exc:
                raise ValueError("viewer address must be empty or an IP address") from exc
        if not 1024 <= self.viewer.port <= 65535:
            raise ValueError("viewer port must be in [1024, 65535]")
        if not 1 <= self.viewer.message_backlog_size <= 4096:
            raise ValueError("viewer message_backlog_size must be in [1, 4096]")
        if not 1 <= self.viewer.max_qos_depth <= 1000:
            raise ValueError("viewer max_qos_depth must be in [1, 1000]")
        if not 1 <= self.viewer.num_threads <= 8:
            raise ValueError("viewer num_threads must be in [1, 8]")
        return self

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


T = TypeVar("T")


def _construct(cls: type[T], value: Mapping[str, Any] | None) -> T:
    source = dict(value or {})
    known = {item.name for item in fields(cls)}
    unknown = sorted(set(source) - known)
    if unknown:
        raise ValueError(f"unknown {cls.__name__} settings: {unknown}")
    return cls(**source)


def load_config(path: str | Path | None = None) -> ApexNavConfig:
    source: Mapping[str, Any] = {}
    if path is not None:
        loaded = yaml.safe_load(Path(path).expanduser().read_text(encoding="utf-8"))
        if not isinstance(loaded, Mapping):
            raise ValueError("ApexNav config must contain a YAML mapping")
        source = loaded
    sections = {item.name for item in fields(ApexNavConfig)}
    unknown = sorted(set(source) - sections)
    if unknown:
        raise ValueError(f"unknown ApexNav config sections: {unknown}")
    return ApexNavConfig(
        camera=_construct(CameraConfig, source.get("camera")),
        robot=_construct(RobotConfig, source.get("robot")),
        vlm=_construct(VLMConfig, source.get("vlm")),
        planner=_construct(PlannerConfig, source.get("planner")),
        experiment=_construct(ExperimentConfig, source.get("experiment")),
        viewer=_construct(ViewerConfig, source.get("viewer")),
    ).validate()
