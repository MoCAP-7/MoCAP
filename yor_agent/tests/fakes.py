"""Tiny test doubles with the same surface as the real robot and model.

These live under ``tests/`` on purpose: production code must never be able to
import a simulated robot, and the executor/agent must not be able to tell
whether an injected callable moves hardware or mutates test state.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from yor_agent.models.llm import LLM
from yor_agent.robot.navigation_controller import wrap_angle


@dataclass
class FakePose:
    x_m: float = 0.0
    y_m: float = 0.0
    yaw_rad: float = 0.0
    valid: bool = True


@dataclass
class FakeFrame:
    planar_pose: FakePose
    rgb: np.ndarray
    depth_m: np.ndarray
    timestamp_ns: int
    ground_camera_height_m: float | None = 1.0
    ground_down_camera_xyz: np.ndarray | None = field(
        default_factory=lambda: np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
    )
    ground_plane_timestamp_ns: int | None = 0


class FakeHardware:
    """Stand-in for :class:`YorHardwareBridge` that integrates commands in place."""

    def __init__(self, *, clearance_m: float = 3.0, arms: dict | None = None) -> None:
        self.now = 0.0
        self.pose = FakePose()
        self.velocity = np.zeros(3, dtype=float)
        self.clearance_m = clearance_m
        self.commands: list[list[float]] = []
        self.arms = arms
        self.closed = False
        self.frame_error: Exception | None = None
        self.ground_down_camera_xyz = np.asarray(
            [0.0, 1.0, 0.0], dtype=np.float64
        )
        # -- manipulation fakes ------------------------------------------
        self.arm_pose_commands: list[dict[str, Any]] = []
        self.arm_plan_requests: list[dict[str, Any]] = []
        self.gripper_commands: list[dict[str, Any]] = []
        self.arm_trajectory_commands: list[dict[str, Any]] = []
        self.move_arm_pose_result: dict[str, Any] | None = None
        self.plan_arm_poses_result: dict[str, Any] | None = None
        self.set_gripper_result: dict[str, Any] | None = None
        self.execute_arm_trajectory_result: dict[str, Any] | None = None

    # -- clock/sleep injected into NavigationController ------------------
    def clock(self) -> float:
        return self.now

    def sleep(self, duration: float) -> None:
        vx, vy, omega = self.velocity
        yaw = self.pose.yaw_rad
        self.pose.x_m += (vx * math.cos(yaw) - vy * math.sin(yaw)) * duration
        self.pose.y_m += (vx * math.sin(yaw) + vy * math.cos(yaw)) * duration
        self.pose.yaw_rad = wrap_angle(yaw + omega * duration)
        self.now += duration

    # -- NavigationHardware ---------------------------------------------
    def latest_frame(self, *, max_age_s: float):
        del max_age_s
        if self.frame_error is not None:
            raise self.frame_error
        return FakeFrame(
            planar_pose=FakePose(
                self.pose.x_m, self.pose.y_m, self.pose.yaw_rad, self.pose.valid
            ),
            rgb=np.zeros((48, 64, 3), dtype=np.uint8),
            depth_m=np.full((48, 64), self.clearance_m, dtype=np.float32),
            timestamp_ns=int(self.now * 1e9),
            ground_down_camera_xyz=self.ground_down_camera_xyz.copy(),
            ground_plane_timestamp_ns=int(self.now * 1e9),
        )

    def next_frame(self, timeout_s: float | None = 2.0):
        del timeout_s
        return self.latest_frame(max_age_s=1.0)

    def get_base_status(self) -> dict[str, Any]:
        return {
            "lease_active": bool(np.any(self.velocity)),
            "lease_remaining_s": 0.25 if np.any(self.velocity) else 0.0,
            "last_velocity": self.velocity.tolist(),
            "estop_latched": False,
            "limits": {
                "lease_s": 0.25,
                "max_linear_mps": 0.12,
                "max_yaw_rad_s": 0.35,
            },
            "telemetry": {"lift_height_m": 0.42},
        }

    def submit_base_velocity(self, velocity: list[float]) -> dict[str, Any]:
        self.velocity = np.asarray(velocity, dtype=float)
        self.commands.append(self.velocity.tolist())
        return {"accepted": True, "velocity": self.velocity.tolist()}

    def get_arm_status(self) -> dict[str, Any] | None:
        return self.arms

    # -- manipulation transport ------------------------------------------
    def move_arm_pose(
        self, arm: int | str, pose_xyz_rpy: list[float], timeout_s: float = 60.0
    ) -> dict[str, Any]:
        self.arm_pose_commands.append(
            {"arm": arm, "pose_xyz_rpy": list(pose_xyz_rpy), "timeout_s": timeout_s}
        )
        if self.move_arm_pose_result is not None:
            return dict(self.move_arm_pose_result)
        return {"success": True, "reason": "ok", "pose_xyz_rpy": list(pose_xyz_rpy)}

    def plan_arm_poses(
        self, arm: int | str, poses_xyz_rpy: list[list[float]]
    ) -> dict[str, Any]:
        self.arm_plan_requests.append(
            {"arm": arm, "poses_xyz_rpy": [list(pose) for pose in poses_xyz_rpy]}
        )
        if self.plan_arm_poses_result is not None:
            return dict(self.plan_arm_poses_result)
        plans = [
            {
                "success": True,
                "candidate_index": index,
                "ik_converged": True,
                "ik_position_error_m": 0.0005,
                "ik_rotation_error_rad": 0.001,
                "ik_joint_target": [0.0] * 7,
                "joint_travel_l2_rad": 0.1,
                "joint_travel_max_rad": 0.05,
            }
            for index in range(len(poses_xyz_rpy))
        ]
        return {"plans": plans}

    def set_gripper(
        self,
        arm: int | str,
        opened: bool,
        timeout_s: float = 3.0,
        force_n: float | None = None,
    ) -> dict[str, Any]:
        self.gripper_commands.append(
            {"arm": arm, "opened": opened, "timeout_s": timeout_s, "force_n": force_n}
        )
        if self.set_gripper_result is not None:
            return dict(self.set_gripper_result)
        return {"success": True, "reason": "ok", "opened": opened}

    def execute_arm_trajectory(
        self,
        arm: int | str,
        waypoints: list[list[float]],
        timeout_s: float = 60.0,
    ) -> dict[str, Any]:
        self.arm_trajectory_commands.append(
            {"arm": arm, "waypoints": waypoints, "timeout_s": timeout_s}
        )
        if self.execute_arm_trajectory_result is not None:
            return dict(self.execute_arm_trajectory_result)
        return {
            "success": True,
            "reason": "joint_trajectory_completed",
            "waypoint_count": len(waypoints),
        }

    def close(self) -> None:
        self.closed = True


#: A valid manipulation calibration block, reused by manipulation tests so
#: they don't each repeat a fabricated eye-to-hand calibration.
FAKE_MANIPULATION_CONFIG: dict[str, Any] = {
    "camera_calibration_resolution": [64, 48],
    "camera_intrinsics": [
        [50.0, 0.0, 32.0],
        [0.0, 50.0, 24.0],
        [0.0, 0.0, 1.0],
    ],
    "left_arm_from_camera": np.eye(4).tolist(),
    "right_arm_from_camera": np.eye(4).tolist(),
    "grasp_backend": "graspgenx",
    "grasp_ik_precheck_enabled": True,
    "grasp_collision_check_enabled": True,
}


def make_environment(
    hardware: FakeHardware | None = None,
    *,
    manipulation: dict[str, Any] | None = None,
    **navigation: Any,
):
    """Build a real ``YorEnvironment`` around fake hardware and a fake clock."""

    from yor_agent.environment import YorEnvironment
    from yor_agent.robot.navigation_controller import (
        NavigationConfig,
        NavigationController,
    )

    hardware = hardware or FakeHardware()
    settings = {"settle_cycles": 2, "stop_timeout_s": 0.4, **navigation}
    environment = YorEnvironment(
        hardware=hardware, navigation=settings, manipulation=manipulation
    )
    environment.controller = NavigationController(
        environment,
        config=NavigationConfig.from_mapping(settings),
        clock=hardware.clock,
        sleep=hardware.sleep,
    )
    environment.hardware = hardware
    return environment


class ScriptedModel(LLM):
    """Real prompt/extraction behavior with a scripted provider response."""

    def __init__(self, responses: list[str], **config: Any) -> None:
        super().__init__({"provider": "vertex", **config})
        self.responses = list(responses)
        self.prompts: list[list[dict]] = []
        self.error: BaseException | None = None

    def _generate(self, messages: list[dict]) -> str:  # type: ignore[override]
        self.prompts.append([dict(message) for message in messages])
        if self.error is not None:
            raise self.error
        self.n_calls += 1
        if not self.responses:
            return '```python\nfinish(reason="script exhausted")\n```'
        return self.responses.pop(0)


@dataclass
class RecordingEnvironment:
    """Minimal environment double for loop tests that need no robot at all."""

    observation: dict[str, Any] = field(default_factory=lambda: {"base": {}})
    reset_error: BaseException | None = None
    observe_error: BaseException | None = None
    shutdowns: int = 0
    observations: int = 0

    def reset(self) -> dict[str, Any]:
        if self.reset_error is not None:
            raise self.reset_error
        return self.observe()

    def observe(self) -> dict[str, Any]:
        if self.observe_error is not None:
            raise self.observe_error
        self.observations += 1
        return self.observation

    def safe_shutdown(self) -> None:
        self.shutdowns += 1
