"""Pluggable grasp-model clients for YOR manipulation.

Ported from ``YOR/Agent/agents_yor/grasp_backends.py``. The model server owns
inference. Backends in this module only normalize its output to camera-frame
gripper-base poses, confidence scores, and geometric anchor points used by
``robot/manipulation.py``'s target/IK selection code.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from pathlib import Path
from typing import Any, Protocol

import numpy as np


LOGGER = logging.getLogger(__name__)

DEFAULT_GRASP_BACKEND = "graspgenx"
DEFAULT_GRASPGENX_GRIPPER_PROFILE = "piper_hand"
_GRASPGENX_SWEEP_VECTOR_KEYS = (
    "graspgenx_sweep_extents_open_m",
    "graspgenx_sweep_offset_open_m",
    "graspgenx_sweep_extents_mid_m",
    "graspgenx_sweep_offset_mid_m",
)

_GRASPGENX_PROFILE_ROOT = (
    Path(__file__).resolve().parent
    / "assets"
    / "graspgenx"
    / "gripper_descriptions"
    / "assets"
    / "x_grippers"
)
_GRIPPER_TYPE_MAP = {"parallel_2f": 0, "revolute_2f": 1, "revolute_3f": 2}


@dataclass(frozen=True)
class GraspBackendResult:
    """Normalized output shared by all YOR grasp backends."""

    poses: np.ndarray
    scores: np.ndarray
    anchor_points: np.ndarray
    metadata: dict[str, Any]


class GraspBackend(Protocol):
    """Minimal interface consumed by ``robot/manipulation.py``."""

    name: str
    origin_to_tcp_m: float

    def __call__(
        self,
        depth: np.ndarray,
        intrinsics: np.ndarray,
        segmentation: np.ndarray,
        instance_id: int,
    ) -> GraspBackendResult: ...


class ContactGraspNetBackend:
    """Adapter around CaP-X's existing Contact-GraspNet HTTP client."""

    name = "contact_graspnet"

    def __init__(self, config: dict[str, Any]) -> None:
        self.origin_to_tcp_m = float(
            config.get("contact_graspnet_origin_to_tcp_m", 0.1034)
        )
        from .perception.contact_graspnet_client import init_contact_graspnet

        self._plan = init_contact_graspnet()

    def __call__(
        self,
        depth: np.ndarray,
        intrinsics: np.ndarray,
        segmentation: np.ndarray,
        instance_id: int,
    ) -> GraspBackendResult:
        poses, scores, contact_points = self._plan(
            depth, intrinsics, segmentation, instance_id
        )
        return GraspBackendResult(
            poses=np.asarray(poses, dtype=np.float32),
            scores=np.asarray(scores, dtype=np.float32).reshape(-1),
            anchor_points=np.asarray(contact_points, dtype=np.float32),
            metadata={"backend": self.name},
        )


def _vector3(config: dict[str, Any], key: str) -> np.ndarray:
    value = np.asarray(config.get(key), dtype=np.float32)
    if value.shape != (3,) or not np.all(np.isfinite(value)):
        raise ValueError(f"{key} must contain three finite values")
    return value


def _positive_int(config: dict[str, Any], key: str, default: int) -> int:
    value = int(config.get(key, default))
    if value <= 0:
        raise ValueError(f"{key} must be positive")
    return value


def _graspgenx_sweep_volume(
    config: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    """Resolve an official named profile or explicit parameter-only values."""

    configured_profile = config.get("graspgenx_gripper_profile")
    if configured_profile is None:
        has_explicit_sweep = any(
            config.get(key) is not None for key in _GRASPGENX_SWEEP_VECTOR_KEYS
        )
        profile = "" if has_explicit_sweep else DEFAULT_GRASPGENX_GRIPPER_PROFILE
    else:
        profile = str(configured_profile).strip()
    if profile:
        if Path(profile).name != profile:
            raise ValueError("graspgenx_gripper_profile must be a simple name")
        root = Path(
            config.get("graspgenx_gripper_profile_root", _GRASPGENX_PROFILE_ROOT)
        ).expanduser()
        profile_path = root / profile / "config.json"
        if not profile_path.is_file():
            raise FileNotFoundError(
                f"GraspGen-X gripper profile does not exist: {profile_path}"
            )
        with profile_path.open("r", encoding="utf-8") as stream:
            profile_config = json.load(stream)
        sweep = profile_config["sweep_volume"]
        params = {
            "extents_open": np.asarray(sweep["extents"], dtype=np.float32),
            "offset_open": np.asarray(sweep["offset"], dtype=np.float32),
            "extents_mid": np.asarray(sweep["extents2"], dtype=np.float32),
            "offset_mid": np.asarray(sweep["offset2"], dtype=np.float32),
            "gripper_type": _GRIPPER_TYPE_MAP.get(profile_config.get("type"), 0),
            "fingertip_depth": float(profile_config["fingertip"][-1]),
        }
        source = f"official-profile:{profile}"
    else:
        params = {
            "extents_open": _vector3(config, "graspgenx_sweep_extents_open_m"),
            "offset_open": _vector3(config, "graspgenx_sweep_offset_open_m"),
            "extents_mid": _vector3(config, "graspgenx_sweep_extents_mid_m"),
            "offset_mid": _vector3(config, "graspgenx_sweep_offset_mid_m"),
            "gripper_type": int(config.get("graspgenx_gripper_type", 0)),
            "fingertip_depth": float(
                config.get(
                    "graspgenx_fingertip_depth_m",
                    0.13,
                )
            ),
        }
        source = "explicit-parameters"

    for key in ("extents_open", "offset_open", "extents_mid", "offset_mid"):
        value = np.asarray(params[key], dtype=np.float32)
        if value.shape != (3,) or not np.all(np.isfinite(value)):
            raise ValueError(f"GraspGen-X {key} must contain three finite values")
        params[key] = value
    if not np.all(params["extents_open"] > 0.0) or not np.all(
        params["extents_mid"] > 0.0
    ):
        raise ValueError("GraspGen-X sweep-volume extents must be positive")
    if params["gripper_type"] not in (0, 1, 2):
        raise ValueError("graspgenx_gripper_type must be 0, 1, or 2")
    if not np.isfinite(params["fingertip_depth"]) or params["fingertip_depth"] <= 0:
        raise ValueError("GraspGen-X fingertip depth must be positive and finite")
    return params, source


class GraspGenXBackend:
    """yor_agent client for the official GraspGen-X ZMQ server protocol.

    This uses the server's parameter-only ``infer_scene_depth`` action.  It
    deliberately performs no collision filtering; candidate filtering remains
    limited to score/top-k selection and YOR's existing target/IK checks.
    """

    name = "graspgenx"

    _PLANNERS = frozenset({"diffusion", "graspmoe"})

    def __init__(self, config: dict[str, Any]) -> None:
        self.host = str(config.get("graspgenx_host", "127.0.0.1"))
        self.port = int(config.get("graspgenx_port", 5556))
        self.timeout_ms = _positive_int(config, "graspgenx_timeout_ms", 180_000)
        self.planner = str(config.get("graspgenx_planner", "diffusion")).lower()
        if self.planner not in self._PLANNERS:
            raise ValueError(
                f"graspgenx_planner must be one of {sorted(self._PLANNERS)}"
            )
        self.min_object_points = _positive_int(
            config, "graspgenx_min_object_points", 100
        )
        self.num_grasps = _positive_int(config, "graspgenx_num_grasps", 200)
        self.topk_num_grasps = _positive_int(
            config, "graspgenx_topk_num_grasps", 100
        )
        self.grasp_threshold = float(
            config.get("graspgenx_score_threshold", -1.0)
        )
        if not np.isfinite(self.grasp_threshold):
            raise ValueError("graspgenx_score_threshold must be finite")

        self.origin_to_tcp_m = float(
            config.get("graspgenx_origin_to_tcp_m", 0.105)
        )
        if not np.isfinite(self.origin_to_tcp_m) or self.origin_to_tcp_m <= 0.0:
            raise ValueError("graspgenx_origin_to_tcp_m must be positive and finite")

        self.sweep_volume_params, self.gripper_profile_source = (
            _graspgenx_sweep_volume(config)
        )
        self._context = None
        self._socket = None
        self.last_metadata: dict[str, Any] = {}

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
                "GraspGen-X backend requires pyzmq; install the yor-agent "
                "'manipulation' extra"
            ) from exc
        self._context = zmq.Context.instance()
        self._socket = self._context.socket(zmq.REQ)
        self._socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self._socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.connect(self.address)
        LOGGER.info("Connected to GraspGen-X server at %s", self.address)

    def _request(self, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            import msgpack_numpy
            import zmq
        except ImportError as exc:
            raise RuntimeError(
                "GraspGen-X backend requires msgpack-numpy and pyzmq; install "
                "the yor-agent 'manipulation' extra"
            ) from exc
        self._connect()
        assert self._socket is not None
        try:
            self._socket.send(msgpack_numpy.packb(payload, use_bin_type=True))
            raw = self._socket.recv()
        except zmq.error.Again as exc:
            self.close()
            raise TimeoutError(
                f"GraspGen-X server at {self.address} did not respond within "
                f"{self.timeout_ms} ms"
            ) from exc
        response = msgpack_numpy.unpackb(raw, raw=False)
        if not isinstance(response, dict):
            raise RuntimeError("GraspGen-X server returned a non-dictionary response")
        if "error" in response:
            raise RuntimeError(f"GraspGen-X server error: {response['error']}")
        return response

    def health(self) -> dict[str, Any]:
        return self._request({"action": "health"})

    def __call__(
        self,
        depth: np.ndarray,
        intrinsics: np.ndarray,
        segmentation: np.ndarray,
        instance_id: int,
    ) -> GraspBackendResult:
        depth = np.asarray(depth, dtype=np.float32)
        intrinsics = np.asarray(intrinsics, dtype=np.float64)
        mask = np.asarray(segmentation)
        if depth.ndim != 2 or mask.shape != depth.shape:
            raise ValueError("GraspGen-X depth and instance mask shapes must match")
        if intrinsics.shape != (3, 3):
            raise ValueError("GraspGen-X intrinsics must have shape (3, 3)")
        instance_mask = np.zeros(mask.shape, dtype=np.int32)
        instance_mask[mask == instance_id] = int(instance_id)

        response = self._request(
            {
                "action": "infer_scene_depth",
                "depth": depth,
                "intrinsics": intrinsics,
                "instance_mask": instance_mask,
                "sweep_volume_params": self.sweep_volume_params,
                "planner": self.planner,
                "min_object_points": self.min_object_points,
                "num_grasps": self.num_grasps,
            }
        )
        ids = np.asarray(response.get("instance_ids", [])).reshape(-1)
        matching = np.flatnonzero(ids == int(instance_id))
        if matching.size == 0:
            skipped = np.asarray(
                response.get("skipped_instance_ids", [])
            ).reshape(-1)
            reason = (
                "instance was skipped for insufficient valid points"
                if np.any(skipped == int(instance_id))
                else "server returned no grasp candidates"
            )
            raise RuntimeError(f"GraspGen-X {reason} for instance {instance_id}")
        result_index = int(matching[0])
        poses = np.asarray(response["grasps"][result_index], dtype=np.float32).reshape(
            -1, 4, 4
        )
        scores = np.asarray(
            response["confidences"][result_index], dtype=np.float32
        ).reshape(-1)
        count = min(poses.shape[0], scores.size)
        poses, scores = poses[:count], scores[:count]
        tags_by_instance = response.get("branch_tags", [])
        tags = (
            list(tags_by_instance[result_index])
            if result_index < len(tags_by_instance)
            else []
        )
        if self.grasp_threshold > 0.0:
            keep = scores >= self.grasp_threshold
            poses, scores = poses[keep], scores[keep]
            if tags:
                tags = [tag for tag, selected in zip(tags, keep) if selected]
        if poses.shape[0] > self.topk_num_grasps:
            order = np.argsort(-scores, kind="stable")[: self.topk_num_grasps]
            poses, scores = poses[order], scores[order]
            if tags:
                tags = [tags[index] for index in order]
        if poses.shape[0] == 0:
            raise RuntimeError(
                "GraspGen-X returned no candidates after score filtering"
            )

        # GraspGen-X returns the gripper-base pose.  YOR's geometric anchor is
        # the configured jaw-center TCP along the model frame's local +Z.
        local_tcp = np.asarray([0.0, 0.0, self.origin_to_tcp_m], dtype=np.float32)
        anchor_points = poses[:, :3, 3] + np.einsum(
            "nij,j->ni", poses[:, :3, :3], local_tcp
        )
        self.last_metadata = {
            "backend": self.name,
            "planner": self.planner,
            "gripper_profile_source": self.gripper_profile_source,
            "branch_tags": tags,
            "server_timing": response.get("timing", {}),
            "skipped_instance_ids": np.asarray(
                response.get("skipped_instance_ids", [])
            ).reshape(-1).tolist(),
        }
        return GraspBackendResult(
            poses=poses,
            scores=scores,
            anchor_points=anchor_points,
            metadata=dict(self.last_metadata),
        )


def create_grasp_backend(config: dict[str, Any]) -> GraspBackend:
    """Construct the configured grasp backend without loading it at API init."""

    name = str(config.get("grasp_backend", DEFAULT_GRASP_BACKEND)).strip().lower()
    if name in {"contact_graspnet", "contact-graspnet", "cgn"}:
        return ContactGraspNetBackend(config)
    if name in {"graspgenx", "graspgen-x", "ggx"}:
        return GraspGenXBackend(config)
    raise ValueError(
        "grasp_backend must be 'contact_graspnet' or 'graspgenx'; "
        f"got {name!r}"
    )
