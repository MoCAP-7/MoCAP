"""Minimal Contact-GraspNet HTTP client retained as an optional backend."""

from __future__ import annotations

import base64
import io
import os
from typing import Any

import numpy as np


DEFAULT_SERVICE_URL = "http://127.0.0.1:8115"


def _encode_array(value: np.ndarray) -> str:
    buffer = io.BytesIO()
    np.save(buffer, np.asarray(value), allow_pickle=False)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _decode_array(value: str) -> np.ndarray:
    data = base64.b64decode(value)
    return np.load(io.BytesIO(data), allow_pickle=False)


def init_contact_graspnet(
    device: str = "cuda",
    checkpoint_path: str | None = None,
    *,
    service_url: str | None = None,
    timeout_s: float = 120.0,
) -> Any:
    """Return the service-backed grasp planner expected by the backend adapter."""

    del device, checkpoint_path
    base_url = str(
        service_url
        or os.environ.get("CONTACT_GRASPNET_SERVICE_URL", DEFAULT_SERVICE_URL)
    ).rstrip("/")

    def plan(
        depth: np.ndarray,
        cam_K: np.ndarray,
        segmap: np.ndarray,
        segmap_id: int,
        local_regions: bool = True,
        filter_grasps: bool = True,
        skip_border_objects: bool = False,
        z_range: list[float] | None = None,
        forward_passes: int = 2,
        max_retries: int = 10,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        try:
            import requests
        except ImportError as exc:
            raise RuntimeError(
                "Contact-GraspNet client requires requests; install "
                "yor-agent[manipulation]"
            ) from exc
        payload = {
            "depth_base64": _encode_array(depth),
            "cam_K_base64": _encode_array(cam_K),
            "segmap_base64": _encode_array(segmap),
            "segmap_id": int(segmap_id),
            "local_regions": bool(local_regions),
            "filter_grasps": bool(filter_grasps),
            "skip_border_objects": bool(skip_border_objects),
            "z_range": [0.2, 2.0] if z_range is None else list(z_range),
            "forward_passes": int(forward_passes),
            "max_retries": int(max_retries),
        }
        response = requests.post(
            f"{base_url}/plan", json=payload, timeout=float(timeout_s)
        )
        response.raise_for_status()
        result = response.json()
        return (
            _decode_array(result["grasps_base64"]),
            _decode_array(result["scores_base64"]),
            _decode_array(result["contact_pts_base64"]),
        )

    return plan
