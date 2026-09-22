"""Lazy lightweight open-vocabulary detector used only for coarse alignment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class FastDetection:
    """One image-space detection in ``[x0, y0, x1, y1]`` pixel coordinates."""

    box_xyxy: tuple[float, float, float, float]
    score: float
    label: str

    @property
    def center_xy(self) -> tuple[float, float]:
        x0, y0, x1, y1 = self.box_xyxy
        return 0.5 * (x0 + x1), 0.5 * (y0 + y1)


class YOLOEDetector:
    """Small YOLOE wrapper with no dependency cost until first inference.

    YOLOE is deliberately not used for manipulation masks.  Its sole job is
    to keep a text-named object inside a central horizontal image band before
    the stopped SAM3/GraspGen planning phase.
    """

    def __init__(
        self,
        *,
        model: str = "yoloe-26s-seg.pt",
        device: str = "cuda:0",
        image_size: int = 960,
        confidence: float = 0.15,
    ) -> None:
        if not isinstance(model, str) or not model.strip():
            raise ValueError("fast detector model must be a non-empty string")
        if not isinstance(device, str) or not device.strip():
            raise ValueError("fast detector device must be a non-empty string")
        if isinstance(image_size, bool) or not 160 <= int(image_size) <= 1280:
            raise ValueError("fast detector image_size must be in [160, 1280]")
        if not np.isfinite(confidence) or not 0.0 < confidence < 1.0:
            raise ValueError("fast detector confidence must be in (0, 1)")
        self.model_name = model.strip()
        self.device = device.strip()
        self.image_size = int(image_size)
        self.confidence = float(confidence)
        self._model: Any | None = None
        self._active_prompt: str | None = None

    def _load(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            from ultralytics import YOLOE
        except ImportError as exc:
            raise RuntimeError(
                "coarse manipulation alignment requires YOLOE; install "
                "yor-agent[fast-detection] in the Jetson agent environment"
            ) from exc
        self._model = YOLOE(self.model_name)
        return self._model

    @staticmethod
    def _numpy(value: Any) -> np.ndarray:
        detach = getattr(value, "detach", None)
        if callable(detach):
            value = detach()
        cpu = getattr(value, "cpu", None)
        if callable(cpu):
            value = cpu()
        return np.asarray(value)

    def __call__(
        self, image: np.ndarray, *, text_prompt: str
    ) -> list[FastDetection]:
        rgb = np.asarray(image, dtype=np.uint8)
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("fast detector expects an HxWx3 RGB image")
        prompt = str(text_prompt).strip()
        if not prompt:
            raise ValueError("text_prompt must be non-empty")
        model = self._load()
        if prompt != self._active_prompt:
            names = [prompt]
            get_text_pe = getattr(model, "get_text_pe", None)
            if callable(get_text_pe):
                model.set_classes(names, get_text_pe(names))
            else:
                model.set_classes(names)
            self._active_prompt = prompt
        results = model.predict(
            source=rgb,
            imgsz=self.image_size,
            conf=self.confidence,
            device=self.device,
            verbose=False,
        )
        if not results:
            return []
        boxes = getattr(results[0], "boxes", None)
        if boxes is None or getattr(boxes, "xyxy", None) is None:
            return []
        xyxy = self._numpy(boxes.xyxy).reshape(-1, 4)
        confidence = self._numpy(boxes.conf).reshape(-1)
        detections: list[FastDetection] = []
        for box, score in zip(xyxy, confidence, strict=False):
            values = np.asarray(box, dtype=np.float64)
            if not np.all(np.isfinite(values)) or not np.isfinite(score):
                continue
            x0, y0, x1, y1 = values.tolist()
            if x1 <= x0 or y1 <= y0:
                continue
            detections.append(
                FastDetection((x0, y0, x1, y1), float(score), prompt)
            )
        return detections


def init_fast_detector(**settings: Any) -> YOLOEDetector:
    """Construct the configured production detector."""

    return YOLOEDetector(**settings)
