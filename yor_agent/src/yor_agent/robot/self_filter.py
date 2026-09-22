"""Depth-aware robot self filtering from cached articulated geometry.

The expensive collision-sphere fitting remains an offline operation.  At
runtime this module performs FK only when an arm configuration changes, renders
the resulting spheres (and optional TCP-attached boxes) into one camera-depth
envelope, and applies that cached envelope with a vectorized per-frame mask.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Mapping
import xml.etree.ElementTree as ET

import numpy as np
import yaml


def _numbers(text: str | None, *, default: tuple[float, ...]) -> np.ndarray:
    if text is None:
        return np.asarray(default, dtype=np.float64)
    values = np.asarray([float(value) for value in text.split()], dtype=np.float64)
    if values.shape != (len(default),) or not np.all(np.isfinite(values)):
        raise ValueError(f"invalid numeric vector: {text!r}")
    return values


def _rpy_matrix(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.asarray(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )


def _transform(xyz: np.ndarray, rpy: np.ndarray) -> np.ndarray:
    output = np.eye(4, dtype=np.float64)
    output[:3, :3] = _rpy_matrix(rpy)
    output[:3, 3] = xyz
    return output


def _axis_rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    norm = float(np.linalg.norm(axis))
    if norm <= 1e-12:
        raise ValueError("URDF joint axis must be nonzero")
    x, y, z = axis / norm
    cosine, sine = math.cos(angle), math.sin(angle)
    one_minus = 1.0 - cosine
    output = np.eye(4, dtype=np.float64)
    output[:3, :3] = np.asarray(
        [
            [
                cosine + x * x * one_minus,
                x * y * one_minus - z * sine,
                x * z * one_minus + y * sine,
            ],
            [
                y * x * one_minus + z * sine,
                cosine + y * y * one_minus,
                y * z * one_minus - x * sine,
            ],
            [
                z * x * one_minus - y * sine,
                z * y * one_minus + x * sine,
                cosine + z * z * one_minus,
            ],
        ],
        dtype=np.float64,
    )
    return output


def _axis_translation(axis: np.ndarray, distance: float) -> np.ndarray:
    norm = float(np.linalg.norm(axis))
    if norm <= 1e-12:
        raise ValueError("URDF joint axis must be nonzero")
    output = np.eye(4, dtype=np.float64)
    output[:3, 3] = axis / norm * float(distance)
    return output


@dataclass(frozen=True)
class RobotSelfFilterConfig:
    urdf_path: Path
    spheres_path: Path
    sphere_erosion_m: float = 0.008
    depth_tolerance_m: float = 0.010
    joint_change_tolerance_rad: float = 0.002
    require_both_arms: bool = True
    # How far past the eroded spheres the depth mask reaches. The footprint
    # keeps using the eroded spheres; only the image mask grows, to cover the
    # few pixels by which calibration and stereo edges miss the silhouette.
    mask_padding_m: float = 0.0

    @classmethod
    def from_mapping(
        cls, values: Mapping[str, Any] | None
    ) -> "RobotSelfFilterConfig | None":
        if not values or not bool(values.get("enabled", False)):
            return None
        unknown = set(values) - {
            "enabled",
            "urdf_path",
            "spheres_path",
            "sphere_erosion_m",
            "depth_tolerance_m",
            "joint_change_tolerance_rad",
            "require_both_arms",
            "arm_status_poll_hz",
            "mask_padding_m",
        }
        if unknown:
            raise ValueError(f"unknown robot_self_filter settings: {sorted(unknown)}")
        config = cls(
            urdf_path=Path(str(values.get("urdf_path", ""))).expanduser(),
            spheres_path=Path(str(values.get("spheres_path", ""))).expanduser(),
            sphere_erosion_m=float(values.get("sphere_erosion_m", 0.008)),
            depth_tolerance_m=float(values.get("depth_tolerance_m", 0.010)),
            joint_change_tolerance_rad=float(
                values.get("joint_change_tolerance_rad", 0.002)
            ),
            require_both_arms=bool(values.get("require_both_arms", True)),
            mask_padding_m=float(values.get("mask_padding_m", 0.0)),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if not self.urdf_path.is_file():
            raise FileNotFoundError(f"robot self-filter URDF missing: {self.urdf_path}")
        if not self.spheres_path.is_file():
            raise FileNotFoundError(
                f"robot self-filter sphere config missing: {self.spheres_path}"
            )
        values = (
            self.sphere_erosion_m,
            self.depth_tolerance_m,
            self.joint_change_tolerance_rad,
            self.mask_padding_m,
        )
        if not all(math.isfinite(value) and value >= 0.0 for value in values):
            raise ValueError(
                "robot self-filter tolerances must be finite and nonnegative"
            )
        if self.sphere_erosion_m > 0.03 or self.depth_tolerance_m > 0.03:
            raise ValueError("robot self-filter spatial tolerances must be <= 0.03 m")
        if self.mask_padding_m > 0.05:
            raise ValueError("robot self-filter mask padding must be <= 0.05 m")
        if self.joint_change_tolerance_rad > 0.05:
            raise ValueError("robot self-filter joint tolerance must be <= 0.05 rad")


@dataclass(frozen=True)
class _Joint:
    name: str
    kind: str
    parent: str
    child: str
    origin: np.ndarray
    axis: np.ndarray
    mimic: tuple[str, float, float] | None


class RobotSphereModel:
    """Small NumPy URDF FK model over offline-fitted collision spheres."""

    def __init__(self, config: RobotSelfFilterConfig) -> None:
        root = ET.parse(config.urdf_path).getroot()
        joints: list[_Joint] = []
        children: set[str] = set()
        links = {str(link.attrib["name"]) for link in root.findall("link")}
        for element in root.findall("joint"):
            origin = element.find("origin")
            mimic_element = element.find("mimic")
            mimic = None
            if mimic_element is not None:
                mimic = (
                    str(mimic_element.attrib["joint"]),
                    float(mimic_element.attrib.get("multiplier", 1.0)),
                    float(mimic_element.attrib.get("offset", 0.0)),
                )
            joint = _Joint(
                name=str(element.attrib["name"]),
                kind=str(element.attrib["type"]),
                parent=str(element.find("parent").attrib["link"]),
                child=str(element.find("child").attrib["link"]),
                origin=_transform(
                    _numbers(
                        None if origin is None else origin.attrib.get("xyz"),
                        default=(0.0, 0.0, 0.0),
                    ),
                    _numbers(
                        None if origin is None else origin.attrib.get("rpy"),
                        default=(0.0, 0.0, 0.0),
                    ),
                ),
                axis=_numbers(
                    None
                    if element.find("axis") is None
                    else element.find("axis").attrib.get("xyz"),
                    default=(1.0, 0.0, 0.0),
                ),
                mimic=mimic,
            )
            joints.append(joint)
            children.add(joint.child)
        roots = sorted(links - children)
        if len(roots) != 1:
            raise ValueError(f"self-filter URDF must have one root link, got {roots}")
        self._root_link = roots[0]
        self._pending_joints = joints

        payload = yaml.safe_load(config.spheres_path.read_text(encoding="utf-8"))
        try:
            sphere_mapping = payload["robot_cfg"]["kinematics"]["collision_spheres"]
            lock_joints = payload["robot_cfg"]["kinematics"].get("lock_joints", {})
        except (KeyError, TypeError) as exc:
            raise ValueError("invalid robot collision-sphere YAML") from exc
        self._lock_joints = {
            str(name): float(value) for name, value in dict(lock_joints).items()
        }
        self._spheres: dict[str, np.ndarray] = {}
        for link, entries in dict(sphere_mapping).items():
            if link == "attached_object":
                continue
            values = np.asarray(
                [
                    [*map(float, entry["center"]), float(entry["radius"])]
                    for entry in entries
                ],
                dtype=np.float64,
            )
            if values.ndim != 2 or values.shape[1] != 4 or not np.all(
                np.isfinite(values)
            ):
                raise ValueError(f"invalid collision spheres for link {link!r}")
            self._spheres[str(link)] = values
        missing = sorted(set(self._spheres) - links)
        if missing:
            raise ValueError(f"sphere links absent from URDF: {missing}")

    @staticmethod
    def _joint_values(joints: np.ndarray, gripper_width_m: float) -> dict[str, float]:
        if joints.shape != (7,) or not np.all(np.isfinite(joints)):
            raise ValueError("robot self-filter requires seven finite arm joints")
        values = {
            f"joint{index + 1}": float(value)
            for index, value in enumerate(joints)
        }
        values["gripper"] = float(np.clip(gripper_width_m, 0.0, 0.10))
        return values

    def link_transforms(
        self, joints: np.ndarray, *, gripper_width_m: float
    ) -> dict[str, np.ndarray]:
        values = {**self._lock_joints, **self._joint_values(joints, gripper_width_m)}
        transforms = {self._root_link: np.eye(4, dtype=np.float64)}
        pending = list(self._pending_joints)
        while pending:
            progress = False
            for joint in list(pending):
                parent = transforms.get(joint.parent)
                if parent is None:
                    continue
                value = float(values.get(joint.name, 0.0))
                if joint.mimic is not None:
                    source, multiplier, offset = joint.mimic
                    value = float(values.get(source, 0.0)) * multiplier + offset
                motion = np.eye(4, dtype=np.float64)
                if joint.kind in {"revolute", "continuous"}:
                    motion = _axis_rotation(joint.axis, value)
                elif joint.kind == "prismatic":
                    motion = _axis_translation(joint.axis, value)
                elif joint.kind != "fixed":
                    raise ValueError(f"unsupported URDF joint type: {joint.kind}")
                transforms[joint.child] = parent @ joint.origin @ motion
                pending.remove(joint)
                progress = True
            if not progress:
                unresolved = [joint.name for joint in pending]
                raise ValueError(f"unresolved URDF joint tree: {unresolved}")
        return transforms

    def camera_spheres(
        self,
        joints: np.ndarray,
        arm_from_camera: np.ndarray,
        *,
        gripper_width_m: float,
        erosion_m: float,
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        arm_from_camera = np.asarray(arm_from_camera, dtype=np.float64)
        if arm_from_camera.shape != (4, 4) or not np.all(np.isfinite(arm_from_camera)):
            raise ValueError("arm_from_camera must be a finite 4x4 transform")
        camera_from_arm = np.linalg.inv(arm_from_camera)
        links = self.link_transforms(joints, gripper_width_m=gripper_width_m)
        output: list[np.ndarray] = []
        for link, spheres in self._spheres.items():
            arm_from_link = links[link]
            centers_arm = (
                spheres[:, :3] @ arm_from_link[:3, :3].T
                + arm_from_link[:3, 3]
            )
            centers_camera = (
                centers_arm @ camera_from_arm[:3, :3].T
                + camera_from_arm[:3, 3]
            )
            radii = np.maximum(0.002, spheres[:, 3] - float(erosion_m))
            output.append(np.column_stack((centers_camera, radii)))
        if not output:
            return np.empty((0, 4), dtype=np.float64), links
        return np.concatenate(output, axis=0), links


class RobotDepthSelfFilter:
    """Cache a depth envelope for two stationary arms and attached objects."""

    def __init__(
        self,
        config: RobotSelfFilterConfig,
        *,
        arm_from_camera: Mapping[str, Any],
    ) -> None:
        self.config = config
        self.model = RobotSphereModel(config)
        self.arm_from_camera = {
            name: np.asarray(transform, dtype=np.float64)
            for name, transform in arm_from_camera.items()
        }
        if config.require_both_arms and set(self.arm_from_camera) != {"left", "right"}:
            raise ValueError(
                "robot self-filter requires left and right arm calibration"
            )
        self._cache_key: tuple[Any, ...] | None = None
        self._near: np.ndarray | None = None
        self._camera_spheres: np.ndarray = np.empty((0, 4), dtype=np.float64)
        self._camera_sphere_arms: tuple[str, ...] = ()
        self.last_debug: dict[str, Any] = {"ready": False}

    def camera_spheres(self) -> np.ndarray:
        """Collision spheres of the arms in camera optical coordinates.

        ``(N, 4)`` of ``[x, y, z, radius]`` for the joint state of the last
        :meth:`update`, empty before the first one.  The depth self-filter
        already builds this envelope every time the arms move; exposing it
        lets the navigation footprint use where the arms actually are instead
        of a static polygon.  Eroded by ``sphere_erosion_m`` (the depth mask
        grows them again by ``mask_padding_m``), so a footprint built from it
        must add its own margin.

        Covers the arm links only.  Attached objects are masked out of the
        depth image but are NOT in here, so a caller that builds a collision
        outline from this must refuse to do so while something is attached.
        Pair it with :meth:`camera_sphere_arms`: with ``require_both_arms``
        false the envelope may describe one arm of a two-armed robot.
        """

        return self._camera_spheres

    def camera_sphere_arms(self) -> tuple[str, ...]:
        """Which arms the last :meth:`camera_spheres` envelope covers."""

        return self._camera_sphere_arms

    @staticmethod
    def _snapshot(
        status: Mapping[str, Any], require_both: bool
    ) -> dict[str, tuple[np.ndarray, float]]:
        if bool(status.get("estop_latched", False)):
            raise RuntimeError("arm emergency stop is latched")
        output: dict[str, tuple[np.ndarray, float]] = {}
        for name in ("left", "right"):
            arm = status.get(name)
            if not isinstance(arm, Mapping):
                if require_both:
                    raise RuntimeError(f"arm status has no {name} snapshot")
                continue
            joints = np.asarray(arm.get("joint_pos"), dtype=np.float64)
            if joints.shape != (7,) or not np.all(np.isfinite(joints)):
                raise RuntimeError(f"{name} arm has no valid seven-joint feedback")
            gripper = arm.get("gripper")
            width = 0.10
            if isinstance(gripper, Mapping) and bool(gripper.get("available", False)):
                candidate = float(gripper.get("width_m", width))
                if math.isfinite(candidate):
                    width = candidate
            output[name] = (joints, float(np.clip(width, 0.0, 0.10)))
        return output

    @staticmethod
    def _attached_key(attached: Mapping[str, Any] | None) -> tuple[Any, ...]:
        if not attached:
            return ()
        values: list[Any] = []
        for name in sorted(attached):
            entry = attached[name]
            if not isinstance(entry, Mapping):
                raise ValueError(
                    f"attached-object entry for {name!r} must be a mapping"
                )
            bounds = np.asarray(entry.get("bounds_local"), dtype=np.float64)
            if bounds.shape != (2, 3) or not np.all(np.isfinite(bounds)):
                raise ValueError(f"attached-object bounds for {name!r} are invalid")
            if np.any(bounds[1] <= bounds[0]):
                raise ValueError(
                    f"attached-object bounds for {name!r} must be positive"
                )
            values.extend((name, *np.round(bounds.reshape(-1), 6)))
        return tuple(values)

    def update(
        self,
        status: Mapping[str, Any],
        *,
        depth_shape: tuple[int, int],
        intrinsics: np.ndarray,
        attached_objects: Mapping[str, Any] | None = None,
        force: bool = False,
    ) -> bool:
        height, width = (int(depth_shape[0]), int(depth_shape[1]))
        if height < 8 or width < 8:
            raise ValueError("robot self-filter depth resolution is invalid")
        intrinsics = np.asarray(intrinsics, dtype=np.float64)
        if intrinsics.shape != (3, 3) or not np.all(np.isfinite(intrinsics)):
            raise ValueError("robot self-filter intrinsics must be a finite 3x3 matrix")
        arms = self._snapshot(status, self.config.require_both_arms)
        quantization = max(self.config.joint_change_tolerance_rad, 1e-9)
        key_values: list[Any] = [height, width, *np.round(intrinsics.reshape(-1), 8)]
        for name in sorted(arms):
            joints, width_m = arms[name]
            key_values.extend(
                (
                    name,
                    *np.round(joints / quantization).astype(np.int64),
                    round(width_m, 4),
                )
            )
        key_values.extend(self._attached_key(attached_objects))
        key = tuple(key_values)
        if not force and key == self._cache_key:
            return False

        spheres: list[np.ndarray] = []
        link_transforms: dict[str, dict[str, np.ndarray]] = {}
        for name, (joints, width_m) in arms.items():
            calibration = self.arm_from_camera.get(name)
            if calibration is None:
                raise RuntimeError(f"robot self-filter has no {name} calibration")
            camera_spheres, links = self.model.camera_spheres(
                joints,
                calibration,
                gripper_width_m=width_m,
                erosion_m=self.config.sphere_erosion_m,
            )
            spheres.append(camera_spheres)
            link_transforms[name] = links

        near = np.full((height, width), np.inf, dtype=np.float32)
        fx, fy = float(intrinsics[0, 0]), float(intrinsics[1, 1])
        cx, cy = float(intrinsics[0, 2]), float(intrinsics[1, 2])
        rendered_spheres = 0
        all_spheres = (
            np.concatenate(spheres, axis=0)
            if spheres
            else np.empty((0, 4), dtype=np.float64)
        )
        self._camera_spheres = all_spheres
        self._camera_sphere_arms = tuple(sorted(arms))
        padding = float(self.config.mask_padding_m)
        for center_x, center_y, center_z, sphere_radius in all_spheres:
            radius = float(sphere_radius) + padding
            if center_z <= radius + 0.01:
                continue
            projected_x = fx * center_x / center_z + cx
            projected_y = fy * center_y / center_z + cy
            pixel_radius_x = fx * radius / max(0.01, center_z - radius)
            pixel_radius_y = fy * radius / max(0.01, center_z - radius)
            column0 = max(0, int(math.floor(projected_x - pixel_radius_x - 1.0)))
            column1 = min(width, int(math.ceil(projected_x + pixel_radius_x + 1.0)))
            row0 = max(0, int(math.floor(projected_y - pixel_radius_y - 1.0)))
            row1 = min(height, int(math.ceil(projected_y + pixel_radius_y + 1.0)))
            if column0 >= column1 or row0 >= row1:
                continue
            rows, columns = np.mgrid[row0:row1, column0:column1]
            dx = (columns.astype(np.float64) - cx) / fx
            dy = (rows.astype(np.float64) - cy) / fy
            a = dx * dx + dy * dy + 1.0
            b = -2.0 * (dx * center_x + dy * center_y + center_z)
            c = (
                center_x * center_x
                + center_y * center_y
                + center_z * center_z
                - radius * radius
            )
            discriminant = b * b - 4.0 * a * c
            valid = discriminant >= 0.0
            if not np.any(valid):
                continue
            root = np.sqrt(np.maximum(discriminant, 0.0))
            patch_near = (-b - root) / (2.0 * a)
            patch_far = (-b + root) / (2.0 * a)
            valid &= patch_far > 0.05
            target_near = near[row0:row1, column0:column1]
            # A depth camera can only observe the first opaque robot surface
            # on a ray, so only the nearest one matters (see mask()).
            closer = valid & (patch_near < target_near)
            np.copyto(target_near, patch_near, where=closer)
            rendered_spheres += 1

        attached_count = 0
        for name, entry in (attached_objects or {}).items():
            if name not in link_transforms:
                raise ValueError(f"attached object references unavailable arm {name!r}")
            bounds = np.asarray(entry["bounds_local"], dtype=np.float64)
            arm_from_tcp = link_transforms[name].get("grasp_tcp")
            if arm_from_tcp is None:
                raise RuntimeError("self-filter URDF has no grasp_tcp link")
            camera_from_arm = np.linalg.inv(self.arm_from_camera[name])
            camera_from_box = camera_from_arm @ arm_from_tcp
            box_from_camera = np.linalg.inv(camera_from_box)
            rows, columns = np.mgrid[0:height, 0:width]
            directions_camera = np.stack(
                (
                    (columns.astype(np.float64) - cx) / fx,
                    (rows.astype(np.float64) - cy) / fy,
                    np.ones((height, width), dtype=np.float64),
                ),
                axis=-1,
            )
            directions_box = directions_camera @ box_from_camera[:3, :3].T
            origin_box = box_from_camera[:3, 3]
            with np.errstate(divide="ignore", invalid="ignore"):
                t1 = (bounds[0] - origin_box) / directions_box
                t2 = (bounds[1] - origin_box) / directions_box
            parallel = np.abs(directions_box) <= 1e-12
            outside_parallel = parallel & (
                (origin_box < bounds[0]) | (origin_box > bounds[1])
            )
            lower = np.where(parallel, -np.inf, np.minimum(t1, t2))
            upper = np.where(parallel, np.inf, np.maximum(t1, t2))
            box_near = np.max(lower, axis=-1)
            box_far = np.min(upper, axis=-1)
            valid = (~np.any(outside_parallel, axis=-1)) & (
                box_far >= np.maximum(box_near, 0.05)
            )
            closer = valid & (box_near < near)
            np.copyto(near, box_near, where=closer)
            attached_count += 1

        envelope = np.isfinite(near)
        self._near = near
        self._cache_key = key
        self.last_debug = {
            "ready": True,
            "arm_count": len(arms),
            "sphere_count": int(sum(len(values) for values in spheres)),
            "rendered_sphere_count": rendered_spheres,
            "attached_object_count": attached_count,
            "envelope_pixels": int(np.count_nonzero(envelope)),
            "depth_shape": [height, width],
        }
        return True

    def mask(self, depth_m: np.ndarray) -> np.ndarray:
        """Pixels whose ray meets the robot and whose depth is not in front of it.

        The robot occludes everything behind its nearest surface on a ray, so
        any depth the camera reports at or beyond that surface is the robot
        itself, however far off the reading is. That matters at the left image
        border, where a gripper 0.5 m away is visible to the left camera only:
        the ZED fills that band with depth about 0.10 m too far, which a
        near-to-far interval around the spheres never matched (2026-09-10, the
        left gripper stopped drive_straight from a standstill). Only a reading
        clearly in front of the robot is kept, as a real obstacle.
        """

        depth = np.asarray(depth_m, dtype=np.float32)
        if self._near is None:
            raise RuntimeError("robot self-filter has no current geometry snapshot")
        if depth.shape != self._near.shape:
            raise ValueError("robot self-filter depth resolution changed")
        tolerance = float(self.config.depth_tolerance_m)
        mask = (
            np.isfinite(depth)
            & np.isfinite(self._near)
            & (depth >= self._near - tolerance)
        )
        self.last_debug = {
            **self.last_debug,
            "removed_depth_pixels": int(np.count_nonzero(mask)),
        }
        return mask

    def filtered_depth(self, depth_m: np.ndarray) -> np.ndarray:
        depth = np.asarray(depth_m, dtype=np.float32).copy()
        depth[self.mask(depth)] = np.nan
        return depth
