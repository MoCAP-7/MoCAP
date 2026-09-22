"""Dependency-light rigid-transform helpers for YOR manipulation.

Ported unchanged from ``YOR/Agent/agents_yor/geometry.py`` so ``yor_agent``
owns the whole high-level-agent-to-low-level-robot path.
"""

from __future__ import annotations

import math

import numpy as np


def normalize_quaternion_wxyz(quaternion) -> np.ndarray:
    quat = np.asarray(quaternion, dtype=np.float64).reshape(-1)
    if quat.size != 4 or not np.all(np.isfinite(quat)):
        raise ValueError("quaternion_wxyz must contain four finite values")
    norm = float(np.linalg.norm(quat))
    if norm <= 1e-10:
        raise ValueError("quaternion_wxyz has zero norm")
    return quat / norm


def quaternion_wxyz_to_matrix(quaternion) -> np.ndarray:
    w, x, y, z = normalize_quaternion_wxyz(quaternion)
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def matrix_to_quaternion_wxyz(rotation) -> np.ndarray:
    matrix = np.asarray(rotation, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError("rotation must be a finite 3x3 matrix")
    # Project small numerical errors back onto SO(3).
    u, _, vh = np.linalg.svd(matrix)
    matrix = u @ vh
    if np.linalg.det(matrix) < 0:
        u[:, -1] *= -1
        matrix = u @ vh
    trace = float(np.trace(matrix))
    if trace > 0:
        scale = math.sqrt(trace + 1.0) * 2
        quat = np.asarray(
            [
                0.25 * scale,
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
            ]
        )
    else:
        index = int(np.argmax(np.diag(matrix)))
        if index == 0:
            scale = math.sqrt(1 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2
            quat = np.asarray(
                [
                    (matrix[2, 1] - matrix[1, 2]) / scale,
                    0.25 * scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                ]
            )
        elif index == 1:
            scale = math.sqrt(1 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2
            quat = np.asarray(
                [
                    (matrix[0, 2] - matrix[2, 0]) / scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    0.25 * scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                ]
            )
        else:
            scale = math.sqrt(1 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2
            quat = np.asarray(
                [
                    (matrix[1, 0] - matrix[0, 1]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    0.25 * scale,
                ]
            )
    quat = normalize_quaternion_wxyz(quat)
    return -quat if quat[0] < 0 else quat


def quaternion_wxyz_to_rpy(quaternion) -> np.ndarray:
    matrix = quaternion_wxyz_to_matrix(quaternion)
    pitch = math.asin(float(np.clip(-matrix[2, 0], -1.0, 1.0)))
    if abs(math.cos(pitch)) > 1e-7:
        roll = math.atan2(matrix[2, 1], matrix[2, 2])
        yaw = math.atan2(matrix[1, 0], matrix[0, 0])
    else:
        roll = math.atan2(-matrix[1, 2], matrix[1, 1])
        yaw = 0.0
    return np.asarray([roll, pitch, yaw], dtype=np.float64)


def rpy_to_quaternion_wxyz(rpy) -> np.ndarray:
    angles = np.asarray(rpy, dtype=np.float64).reshape(-1)
    if angles.size != 3 or not np.all(np.isfinite(angles)):
        raise ValueError("rpy must contain three finite values")
    roll, pitch, yaw = angles
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rotation = np.asarray(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )
    return matrix_to_quaternion_wxyz(rotation)


def slerp_quaternion_wxyz(start, end, fraction: float) -> np.ndarray:
    """Interpolate two WXYZ quaternions along their shortest SO(3) arc."""

    amount = float(fraction)
    if not np.isfinite(amount) or not 0.0 <= amount <= 1.0:
        raise ValueError("fraction must be finite and within [0, 1]")
    first = normalize_quaternion_wxyz(start)
    second = normalize_quaternion_wxyz(end)
    dot = float(np.dot(first, second))
    if dot < 0.0:
        second = -second
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        return normalize_quaternion_wxyz(first + amount * (second - first))
    angle = math.acos(dot)
    scale = math.sin(angle)
    return normalize_quaternion_wxyz(
        math.sin((1.0 - amount) * angle) / scale * first
        + math.sin(amount * angle) / scale * second
    )


def pose_matrix(position, quaternion_wxyz) -> np.ndarray:
    position = np.asarray(position, dtype=np.float64).reshape(-1)
    if position.size != 3 or not np.all(np.isfinite(position)):
        raise ValueError("position must contain three finite values")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = quaternion_wxyz_to_matrix(quaternion_wxyz)
    transform[:3, 3] = position
    return transform


def validated_transform(value, *, name: str) -> np.ndarray:
    transform = np.asarray(value, dtype=np.float64)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError(f"{name} must be a finite 4x4 transform")
    if not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-6):
        raise ValueError(f"{name} has an invalid homogeneous last row")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=2e-3):
        raise ValueError(f"{name} rotation is not orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=2e-3):
        raise ValueError(f"{name} rotation determinant must be +1")
    return transform.copy()
