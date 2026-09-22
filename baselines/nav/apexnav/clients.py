"""HTTP clients preserving ApexNav's original detector/segmentor/ITM split."""

from __future__ import annotations

import base64
from dataclasses import dataclass
import time
from typing import Any, Iterable

import cv2
import numpy as np
import requests

from .config import VLMConfig


COCO_CLASSES = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
}


def _normalized_label(value: str) -> str:
    return " ".join(str(value).strip().lower().rstrip(".").split())


@dataclass(frozen=True)
class Detection:
    box_xyxy_normalized: np.ndarray
    score: float
    phrase: str


@dataclass(frozen=True)
class SegmentedDetection:
    mask: np.ndarray
    score: float
    phrase: str
    label_index: int
    box_xyxy_normalized: np.ndarray


class ApexNavModelClients:
    def __init__(self, config: VLMConfig) -> None:
        self.config = config
        self.session = requests.Session()
        self.calls: list[dict[str, Any]] = []

    def close(self) -> None:
        self.session.close()

    def healthcheck(self) -> dict[str, bool]:
        urls = {
            "grounding_dino": self.config.grounding_dino_url,
            "blip2_itm": self.config.blip2_itm_url,
            "mobile_sam": self.config.mobile_sam_url,
            "yolov7": self.config.yolov7_url,
        }
        result: dict[str, bool] = {}
        for name, url in urls.items():
            try:
                response = self.session.get(url.rsplit("/", 1)[0], timeout=1.0)
                result[name] = response.status_code < 500
            except requests.RequestException:
                # Flask model servers expose POST-only endpoints, so a failed GET
                # does not prove that the TCP service is absent. The run preflight
                # performs a socket check; actual response shape is validated here.
                result[name] = False
        return result

    def image_text_similarity(self, rgb: np.ndarray, target: str, room: str) -> float:
        if room and room != "everywhere":
            text = f"Seems like there is a {room} or a {target} ahead?"
        else:
            text = f"Seems like there is a {target} ahead?"
        response = self._post(
            self.config.blip2_itm_url,
            "blip2_itm",
            image=self._encode_image(rgb),
            txt=text,
        )
        score = float(response["response"])
        if not np.isfinite(score):
            raise RuntimeError("BLIP-2 ITM returned a non-finite score")
        return score

    def detect_and_segment(
        self,
        rgb: np.ndarray,
        target: str,
        similar_labels: Iterable[str],
    ) -> list[SegmentedDetection]:
        target_labels = [_normalized_label(item) for item in target.split("|") if item.strip()]
        similar = [_normalized_label(item) for item in similar_labels if str(item).strip()]
        if not target_labels:
            raise ValueError("target must contain at least one label")
        if len(similar) > 4:
            raise ValueError("ApexNav object fusion supports at most four similar labels")
        all_labels = target_labels + similar
        coco_labels = [label for label in all_labels if label in COCO_CLASSES]
        dino_labels = [label for label in all_labels if label not in COCO_CLASSES]

        # Match upstream get_object(): if any requested target is open-vocabulary,
        # GroundingDINO sees all labels while YOLOv7 still handles COCO target aliases.
        if any(label not in COCO_CLASSES for label in target_labels):
            dino_labels = list(all_labels)
            coco_labels = [label for label in target_labels if label in COCO_CLASSES]

        detections: list[Detection] = []
        if coco_labels:
            response = self._post(
                self.config.yolov7_url,
                "yolov7",
                image=self._encode_image(rgb),
                agnostic_nms=True,
                conf_thres=float(self.config.yolo_confidence_threshold),
                iou_thres=float(self.config.yolo_iou_threshold),
            )
            detections.extend(
                item for item in self._parse_detections(response) if item.phrase in coco_labels
            )
        if dino_labels:
            caption = " ".join(f"{label}.  " for label in dino_labels)
            response = self._post(
                self.config.grounding_dino_url,
                "grounding_dino",
                image=self._encode_image(rgb),
                caption=caption,
                box_threshold=float(self.config.grounding_dino_box_threshold),
                text_threshold=float(self.config.grounding_dino_text_threshold),
            )
            detections.extend(
                item for item in self._parse_detections(response) if item.phrase in dino_labels
            )

        results: list[SegmentedDetection] = []
        seen: set[tuple[str, tuple[int, ...]]] = set()
        for detection in sorted(detections, key=lambda item: item.score, reverse=True):
            box_pixels = self._box_pixels(detection.box_xyxy_normalized, rgb.shape[:2])
            if (box_pixels[2] - box_pixels[0]) * (box_pixels[3] - box_pixels[1]) >= (
                rgb.shape[0] * rgb.shape[1] * 0.99
            ):
                continue
            key = (detection.phrase, tuple(box_pixels))
            if key in seen:
                continue
            seen.add(key)
            mask = self.segment_box(rgb, box_pixels)
            if self.config.mask_erosion_pixels:
                mask = cv2.erode(
                    mask.astype(np.uint8),
                    None,
                    iterations=int(self.config.mask_erosion_pixels),
                ).astype(bool)
            if not np.any(mask):
                continue
            label_index = (
                0
                if detection.phrase in target_labels
                else similar.index(detection.phrase) + 1
            )
            results.append(
                SegmentedDetection(
                    mask=mask,
                    score=detection.score,
                    phrase=detection.phrase,
                    label_index=label_index,
                    box_xyxy_normalized=detection.box_xyxy_normalized.copy(),
                )
            )
        return results

    def segment_box(self, rgb: np.ndarray, box_pixels: list[int]) -> np.ndarray:
        response = self._post(
            self.config.mobile_sam_url,
            "mobile_sam",
            image=self._encode_image(rgb),
            bbox=box_pixels,
        )
        raw = base64.b64decode(response["cropped_mask"])
        mask = np.frombuffer(raw, dtype=np.uint8)
        expected = rgb.shape[0] * rgb.shape[1]
        if mask.size != expected:
            raise RuntimeError(
                f"MobileSAM returned {mask.size} mask bytes; expected {expected}"
            )
        return mask.reshape(rgb.shape[:2]).astype(bool)

    @staticmethod
    def annotate(rgb: np.ndarray, detections: Iterable[SegmentedDetection]) -> np.ndarray:
        image = np.asarray(rgb, dtype=np.uint8).copy()
        for item in detections:
            color = (255, 0, 0) if item.label_index == 0 else (0, 255, 0)
            contours, _ = cv2.findContours(
                item.mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(image, contours, -1, color, 2)
            box = ApexNavModelClients._box_pixels(item.box_xyxy_normalized, image.shape[:2])
            cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), color, 2)
            cv2.putText(
                image,
                f"{item.phrase} {item.score:.2f}",
                (box[0], max(18, box[1] - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
            )
        return image

    @staticmethod
    def _box_pixels(box: np.ndarray, shape: tuple[int, int]) -> list[int]:
        height, width = shape
        value = np.asarray(box, dtype=np.float64).reshape(4).copy()
        if float(np.max(np.abs(value))) <= 1.5:
            value *= np.asarray([width, height, width, height], dtype=np.float64)
        value[0::2] = np.clip(value[0::2], 0, width - 1)
        value[1::2] = np.clip(value[1::2], 0, height - 1)
        x1, y1, x2, y2 = [int(round(item)) for item in value]
        return [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]

    @staticmethod
    def _parse_detections(response: dict[str, Any]) -> list[Detection]:
        boxes = np.asarray(response.get("boxes", []), dtype=np.float64)
        scores = np.asarray(response.get("logits", []), dtype=np.float64).reshape(-1)
        phrases = [_normalized_label(item) for item in response.get("phrases", [])]
        if boxes.size == 0:
            return []
        boxes = boxes.reshape(-1, 4)
        if not (len(boxes) == len(scores) == len(phrases)):
            raise RuntimeError("detector response fields have inconsistent lengths")
        return [
            Detection(box.copy(), float(score), phrase)
            for box, score, phrase in zip(boxes, scores, phrases)
            if np.all(np.isfinite(box)) and np.isfinite(score)
        ]

    def _encode_image(self, image: np.ndarray) -> str:
        contiguous = np.ascontiguousarray(image, dtype=np.uint8)
        ok, buffer = cv2.imencode(
            ".jpg",
            contiguous,
            [cv2.IMWRITE_JPEG_QUALITY, int(self.config.jpeg_quality)],
        )
        if not ok:
            raise RuntimeError("failed to JPEG-encode model input")
        return base64.b64encode(buffer).decode("ascii")

    def _post(self, url: str, model: str, **payload: Any) -> dict[str, Any]:
        started = time.monotonic()
        error: str | None = None
        try:
            response = self.session.post(
                url, json=payload, timeout=float(self.config.request_timeout_s)
            )
            response.raise_for_status()
            result = response.json()
            if not isinstance(result, dict):
                raise TypeError(f"{model} response must be a JSON object")
            return result
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            raise RuntimeError(f"{model} request failed: {error}") from exc
        finally:
            self.calls.append(
                {
                    "model": model,
                    "url": url,
                    "elapsed_s": round(time.monotonic() - started, 4),
                    "error": error,
                }
            )
