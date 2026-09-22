"""Calibrated ZED-to-ApexNav geometry with no Habitat coordinate assumptions."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def wrap_angle(value: float) -> float:
    return math.atan2(math.sin(float(value)), math.cos(float(value)))


def scaled_intrinsics(
    intrinsics: np.ndarray,
    calibration_resolution: tuple[int, int],
    image_shape: tuple[int, int],
) -> np.ndarray:
    matrix = np.asarray(intrinsics, dtype=np.float64).copy()
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError("intrinsics must be a finite 3x3 matrix")
    calibration_width, calibration_height = calibration_resolution
    height, width = image_shape
    sx = width / float(calibration_width)
    sy = height / float(calibration_height)
    matrix[0, 0] *= sx
    matrix[0, 2] = (matrix[0, 2] + 0.5) * sx - 0.5
    matrix[1, 1] *= sy
    matrix[1, 2] = (matrix[1, 2] + 0.5) * sy - 0.5
    return matrix


def ground_axes(down_camera_xyz: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    down = np.asarray(down_camera_xyz, dtype=np.float64).reshape(-1)
    if down.shape != (3,) or not np.all(np.isfinite(down)):
        raise ValueError("ground down axis must contain three finite values")
    norm = float(np.linalg.norm(down))
    if norm <= 1e-6:
        raise ValueError("ground down axis is zero")
    down /= norm
    optical_forward = np.asarray([0.0, 0.0, 1.0])
    forward = optical_forward - down * float(optical_forward @ down)
    forward /= float(np.linalg.norm(forward))
    right = np.cross(down, forward)
    right /= float(np.linalg.norm(right))
    return forward, right, down


def camera_rotation_world(yaw: float, down_camera_xyz: Any) -> np.ndarray:
    """Return R_world_camera for the ZED optical frame (x right, y down, z forward)."""

    forward_camera, right_camera, down_camera = ground_axes(down_camera_xyz)
    cosine, sine = math.cos(float(yaw)), math.sin(float(yaw))
    forward_world = np.asarray([cosine, sine, 0.0])
    right_world = np.asarray([sine, -cosine, 0.0])
    up_world = np.asarray([0.0, 0.0, 1.0])
    rotation = (
        np.outer(forward_world, forward_camera)
        + np.outer(right_world, right_camera)
        + np.outer(up_world, -down_camera)
    )
    if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-5):
        raise ValueError("computed camera rotation is not orthonormal")
    return rotation


def rotation_matrix_to_quaternion(matrix: np.ndarray) -> tuple[float, float, float, float]:
    """Return a normalized ROS quaternion in xyzw order."""

    m = np.asarray(matrix, dtype=np.float64)
    trace = float(np.trace(m))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        q = np.asarray(
            [(m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s,
             (m[1, 0] - m[0, 1]) / s, 0.25 * s]
        )
    else:
        index = int(np.argmax(np.diag(m)))
        if index == 0:
            s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
            q = np.asarray([0.25 * s, (m[0, 1] + m[1, 0]) / s,
                            (m[0, 2] + m[2, 0]) / s, (m[2, 1] - m[1, 2]) / s])
        elif index == 1:
            s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
            q = np.asarray([(m[0, 1] + m[1, 0]) / s, 0.25 * s,
                            (m[1, 2] + m[2, 1]) / s, (m[0, 2] - m[2, 0]) / s])
        else:
            s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
            q = np.asarray([(m[0, 2] + m[2, 0]) / s,
                            (m[1, 2] + m[2, 1]) / s, 0.25 * s,
                            (m[1, 0] - m[0, 1]) / s])
    q /= float(np.linalg.norm(q))
    return tuple(float(value) for value in q)


def relative_camera_pose(
    raw_pose: np.ndarray, origin_pose: np.ndarray
) -> np.ndarray:
    """Express a ZED camera planar pose in an episode-local frame."""

    raw = np.asarray(raw_pose, dtype=np.float64).reshape(3)
    origin = np.asarray(origin_pose, dtype=np.float64).reshape(3)
    delta = raw[:2] - origin[:2]
    cosine, sine = math.cos(origin[2]), math.sin(origin[2])
    local_xy = np.asarray(
        [cosine * delta[0] + sine * delta[1], -sine * delta[0] + cosine * delta[1]]
    )
    return np.asarray([local_xy[0], local_xy[1], wrap_angle(raw[2] - origin[2])])


def base_pose_from_camera(
    camera_pose: np.ndarray, forward_offset_m: float, left_offset_m: float
) -> np.ndarray:
    pose = np.asarray(camera_pose, dtype=np.float64).reshape(3)
    cosine, sine = math.cos(pose[2]), math.sin(pose[2])
    dx = cosine * forward_offset_m - sine * left_offset_m
    dy = sine * forward_offset_m + cosine * left_offset_m
    return np.asarray([pose[0] - dx, pose[1] - dy, pose[2]])


def masked_depth_to_world(
    depth_m: np.ndarray,
    mask: np.ndarray,
    intrinsics: np.ndarray,
    camera_xyz: np.ndarray,
    rotation_world_camera: np.ndarray,
    *,
    minimum_depth_m: float,
    maximum_depth_m: float,
    stride: int = 1,
    maximum_points: int = 12000,
) -> np.ndarray:
    """Project a segmented metric-depth mask to an Nx3 world point cloud."""

    depth = np.asarray(depth_m, dtype=np.float32)
    selected = np.asarray(mask, dtype=bool)
    if depth.ndim != 2 or selected.shape != depth.shape:
        raise ValueError("depth and mask must be matching 2-D arrays")
    rows, columns = np.mgrid[0 : depth.shape[0] : stride, 0 : depth.shape[1] : stride]
    z = depth[rows, columns]
    valid = (
        selected[rows, columns]
        & np.isfinite(z)
        & (z >= float(minimum_depth_m))
        & (z <= float(maximum_depth_m))
    )
    rows = rows[valid].astype(np.float64)
    columns = columns[valid].astype(np.float64)
    z = z[valid].astype(np.float64)
    if z.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    points_camera = np.column_stack(
        [
            (columns - intrinsics[0, 2]) * z / intrinsics[0, 0],
            (rows - intrinsics[1, 2]) * z / intrinsics[1, 1],
            z,
        ]
    )
    points_world = points_camera @ np.asarray(rotation_world_camera).T
    points_world += np.asarray(camera_xyz, dtype=np.float64).reshape(1, 3)
    if len(points_world) > maximum_points:
        # Deterministic spatially uniform subsampling, unlike upstream random sampling.
        indices = np.linspace(0, len(points_world) - 1, maximum_points, dtype=np.int64)
        points_world = points_world[indices]
    return np.ascontiguousarray(points_world, dtype=np.float32)
