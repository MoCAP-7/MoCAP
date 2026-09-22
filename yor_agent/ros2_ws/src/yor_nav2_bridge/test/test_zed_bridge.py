import math
from types import SimpleNamespace

import numpy as np

from yor_nav2_bridge.zed_bridge import ZedBridge


def _bridge_without_ros() -> ZedBridge:
    bridge = ZedBridge.__new__(ZedBridge)
    bridge._base_to_camera = np.asarray([0.2143, 0.0603])
    bridge._twist_alpha = 1.0
    bridge._last_pose_sample = None
    bridge._filtered_twist = np.zeros(3, dtype=np.float64)
    return bridge


def test_camera_pose_is_shifted_to_swerve_center() -> None:
    bridge = _bridge_without_ros()
    pose = SimpleNamespace(x_m=0.0, y_m=0.0, yaw_rad=0.0, timestamp_ns=1)

    base_x, base_y, yaw, twist = bridge._base_pose_and_twist(pose)

    assert math.isclose(base_x, -0.2143)
    assert math.isclose(base_y, -0.0603)
    assert yaw == 0.0
    np.testing.assert_allclose(twist, 0.0)


def test_camera_arc_during_pure_yaw_does_not_become_base_translation() -> None:
    bridge = _bridge_without_ros()
    base_x = -0.2143
    base_y = -0.0603
    bridge._base_pose_and_twist(
        SimpleNamespace(x_m=0.0, y_m=0.0, yaw_rad=0.0, timestamp_ns=1_000_000_000)
    )
    yaw = 0.1
    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    camera_x = base_x + cosine * 0.2143 - sine * 0.0603
    camera_y = base_y + sine * 0.2143 + cosine * 0.0603

    result_x, result_y, _, twist = bridge._base_pose_and_twist(
        SimpleNamespace(
            x_m=camera_x,
            y_m=camera_y,
            yaw_rad=yaw,
            timestamp_ns=1_100_000_000,
        )
    )

    assert math.isclose(result_x, base_x, abs_tol=1e-12)
    assert math.isclose(result_y, base_y, abs_tol=1e-12)
    assert math.isclose(twist[0], 0.0, abs_tol=1e-12)
    assert math.isclose(twist[1], 0.0, abs_tol=1e-12)
    assert math.isclose(twist[2], 1.0, abs_tol=1e-12)
