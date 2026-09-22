"""Small HTTP client for the YOR SAM3 service.

This intentionally contains no CaP-X imports or visualization dependencies.
"""

from __future__ import annotations

import base64
import io
import os
import time
from collections.abc import Sequence
from typing import Any

import numpy as np
from PIL import Image


DEFAULT_SERVICE_URL = "http://127.0.0.1:8114"


def _encode_image(image: np.ndarray | Image.Image) -> str:
    if isinstance(image, np.ndarray):
        values = np.asarray(image)
        if values.dtype != np.uint8:
            values = np.clip(values, 0, 255).astype(np.uint8)
        image = Image.fromarray(values).convert("RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _post_json(
    url: str,
    payload: dict[str, Any],
    *,
    timeout_s: float,
    max_retries: int,
) -> dict[str, Any]:
    try:
        import requests
    except ImportError as exc:
        raise RuntimeError(
            "SAM3 client requires requests; install yor-agent[manipulation]"
        ) from exc

    delay_s = 1.0
    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            response = requests.post(url, json=payload, timeout=timeout_s)
            response.raise_for_status()
            result = response.json()
            if not isinstance(result, dict):
                raise RuntimeError("SAM3 returned a non-object JSON response")
            return result
        except (requests.RequestException, ValueError, RuntimeError) as exc:
            last_error = exc
            if attempt + 1 < max_retries:
                time.sleep(delay_s)
                delay_s = min(delay_s * 2.0, 8.0)
    raise RuntimeError(f"SAM3 request failed after {max_retries} attempts: {last_error}")


def init_sam3(
    checkpoint_path: str | None = None,
    device: str = "cuda",
    model_type: str = "vit_l",
    *,
    service_url: str | None = None,
    timeout_s: float = 120.0,
    max_retries: int = 5,
):
    """Return the text-prompt segmentation callable used by YOR primitives.

    Model arguments are retained for source compatibility but inference is
    always performed by the standalone service.
    """

    del checkpoint_path, device, model_type
    base_url = str(
        service_url or os.environ.get("SAM3_SERVICE_URL", DEFAULT_SERVICE_URL)
    ).rstrip("/")

    def segment(
        image: np.ndarray | Image.Image,
        text_prompt: str,
        box_prompt: Sequence[float] | None = None,
    ) -> list[dict[str, Any]]:
        del box_prompt
        payload = {
            "image_base64": _encode_image(image),
            "text_prompt": str(text_prompt),
        }
        response = _post_json(
            f"{base_url}/segment",
            payload,
            timeout_s=float(timeout_s),
            max_retries=int(max_retries),
        )
        results = response.get("results", [])
        if not isinstance(results, list):
            raise RuntimeError("SAM3 response.results must be a list")
        decoded: list[dict[str, Any]] = []
        for item in results:
            if not isinstance(item, dict):
                raise RuntimeError("SAM3 result entry must be an object")
            shape = tuple(int(value) for value in item["shape"])
            mask = np.frombuffer(
                base64.b64decode(item["mask_base64"]), dtype=np.uint8
            ).reshape(shape)
            decoded.append(
                {
                    "mask": mask.astype(bool),
                    "box": item.get("box"),
                    "score": float(item.get("score", 0.0)),
                    "label": item.get("label", text_prompt),
                }
            )
        return decoded

    return segment
