"""Thin ZMQ client for the isolated GPU grasp-motion planner service."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np


LOGGER = logging.getLogger(__name__)


class GraspMotionPlanningClient:
    def __init__(self, config: dict[str, Any]) -> None:
        self.host = str(config.get("grasp_motion_host", "127.0.0.1"))
        self.port = int(config.get("grasp_motion_port", 5559))
        self.timeout_ms = int(config.get("grasp_motion_timeout_ms", 300_000))
        if not 1 <= self.port <= 65535:
            raise ValueError("grasp_motion_port must be in [1, 65535]")
        if not 1_000 <= self.timeout_ms <= 600_000:
            raise ValueError("grasp_motion_timeout_ms must be in [1000, 600000]")
        self._context = None
        self._socket = None

    @property
    def address(self) -> str:
        return f"tcp://{self.host}:{self.port}"

    def close(self) -> None:
        if self._socket is not None:
            self._socket.close(linger=0)
            self._socket = None

    def _connect(self) -> None:
        if self._socket is not None:
            return
        try:
            import zmq
        except ImportError as exc:
            raise RuntimeError(
                "collision-aware grasp planning requires the manipulation extra"
            ) from exc
        self._context = zmq.Context.instance()
        self._socket = self._context.socket(zmq.REQ)
        self._socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self._socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.connect(self.address)
        LOGGER.info("Connected to grasp-motion planner at %s", self.address)

    def _request(self, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            import msgpack_numpy
            import zmq
        except ImportError as exc:
            raise RuntimeError(
                "collision-aware grasp planning requires pyzmq and msgpack-numpy"
            ) from exc
        self._connect()
        assert self._socket is not None
        try:
            self._socket.send(msgpack_numpy.packb(payload, use_bin_type=True))
            raw = self._socket.recv()
        except zmq.error.Again as exc:
            self.close()
            raise TimeoutError(
                f"grasp-motion planner at {self.address} did not respond within "
                f"{self.timeout_ms} ms"
            ) from exc
        response = msgpack_numpy.unpackb(raw, raw=False)
        if not isinstance(response, dict):
            raise RuntimeError("grasp-motion planner returned a non-dictionary")
        if response.get("error"):
            raise RuntimeError(f"grasp-motion planner error: {response['error']}")
        return response

    def health(self) -> dict[str, Any]:
        return self._request({"action": "health"})

    def check_grasps(
        self,
        tcp_poses: np.ndarray,
        obstacle_points: np.ndarray,
        *,
        clearance_m: float,
        approach_m: float,
        approach_samples: int,
    ) -> dict[str, Any]:
        return self._request(
            {
                "action": "check_grasps",
                "tcp_poses": np.asarray(tcp_poses, dtype=np.float32),
                "obstacle_points": np.asarray(obstacle_points, dtype=np.float32),
                "clearance_m": float(clearance_m),
                "approach_m": float(approach_m),
                "approach_samples": int(approach_samples),
            }
        )

    def plan_grasp(
        self,
        current_joints: np.ndarray,
        tcp_poses: np.ndarray,
        obstacle_points: np.ndarray,
        *,
        clearance_m: float,
        approach_m: float,
        approach_samples: int,
        scene_voxel_m: float,
        goal_joint_seed: np.ndarray | None = None,
        planner_attempts: dict[str, int] | None = None,
    ) -> dict[str, Any]:
        payload = {
            "action": "plan_grasp",
            # Keep joint-space values in float64 across the RPC boundary.
            # Official Nero limits are float64 decimal constants; converting a
            # value exactly on a limit to float32 can round it just outside the
            # accepted interval before the planner performs its safety check.
            "current_joints": np.asarray(current_joints, dtype=np.float64),
            "tcp_poses": np.asarray(tcp_poses, dtype=np.float32),
            "obstacle_points": np.asarray(obstacle_points, dtype=np.float32),
            "clearance_m": float(clearance_m),
            "approach_m": float(approach_m),
            "approach_samples": int(approach_samples),
            "scene_voxel_m": float(scene_voxel_m),
        }
        if goal_joint_seed is not None:
            payload["goal_joint_seed"] = np.asarray(
                goal_joint_seed, dtype=np.float64
            )
        _apply_planner_attempts(payload, planner_attempts)
        return self._request(payload)

    def plan_attached_lift(
        self,
        current_joints: np.ndarray,
        obstacle_points: np.ndarray,
        attached_bounds_local: np.ndarray,
        *,
        lift_m: float,
        scene_voxel_m: float,
        planner_attempts: dict[str, int] | None = None,
    ) -> dict[str, Any]:
        """Plan a constrained local-Z retreat with carried-object geometry."""

        payload = {
            "action": "plan_attached_lift",
            "current_joints": np.asarray(current_joints, dtype=np.float64),
            "obstacle_points": np.asarray(obstacle_points, dtype=np.float32),
            "attached_bounds_local": np.asarray(
                attached_bounds_local, dtype=np.float32
            ),
            "lift_m": float(lift_m),
            "scene_voxel_m": float(scene_voxel_m),
        }
        _apply_planner_attempts(payload, planner_attempts)
        return self._request(payload)

    def plan_joint_target(
        self,
        current_joints: np.ndarray,
        target_joints: np.ndarray,
        obstacle_points: np.ndarray,
        *,
        scene_voxel_m: float,
        attached_bounds_local: np.ndarray | None = None,
        planner_attempts: dict[str, int] | None = None,
    ) -> dict[str, Any]:
        """Plan a whole-robot collision-free path to an exact joint target."""

        payload = {
            "action": "plan_joint_target",
            "current_joints": np.asarray(current_joints, dtype=np.float64),
            "target_joints": np.asarray(target_joints, dtype=np.float64),
            "obstacle_points": np.asarray(obstacle_points, dtype=np.float32),
            "scene_voxel_m": float(scene_voxel_m),
        }
        if attached_bounds_local is not None:
            payload["attached_bounds_local"] = np.asarray(
                attached_bounds_local, dtype=np.float32
            )
        _apply_planner_attempts(payload, planner_attempts)
        return self._request(payload)


PLANNER_ATTEMPT_KEYS = ("max_attempts", "enable_graph_attempt", "finetune_attempts")
# World-representation options understood by the planner service.
PLANNER_SCENE_KEYS = (
    "scene_mesh",
    "scene_tile_m",
    "scene_coarse_voxel_m",
    "scene_fine_radius_m",
    "scene_crop_radius_m",
)


def _apply_planner_attempts(
    payload: dict[str, Any], planner_attempts: dict[str, Any] | None
) -> None:
    """Forward optional planner options; absent keys keep server defaults."""

    if not planner_attempts:
        return
    for key in PLANNER_ATTEMPT_KEYS:
        value = planner_attempts.get(key)
        if value is not None:
            payload[key] = int(value)
    for key in PLANNER_SCENE_KEYS:
        value = planner_attempts.get(key)
        if value is not None:
            payload[key] = str(value) if key == "scene_mesh" else float(value)
