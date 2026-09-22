"""Serve the original ApexNav YOLOv7 model without installing it into ROS."""

from __future__ import annotations

import argparse
import base64
import hashlib
import os
import sys
from pathlib import Path
import types
from typing import Any


COCO_CLASSES = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
)

YOLOV7_E6E_SHA256 = "b370120a414bf32b5d65fc808e5a32c8d9b3c63902d1bc41894fc9d86eccf9cb"


def _verify_official_weights(path: Path) -> None:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != YOLOV7_E6E_SHA256:
        raise RuntimeError(f"refusing unverified YOLOv7 checkpoint: {path}")


def _decode_image(value: str):
    import cv2
    import numpy as np

    raw = base64.b64decode(value)
    image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("failed to decode YOLOv7 input image")
    return image


class YOLOv7Server:
    def __init__(self, checkout: Path, weights: Path) -> None:
        import cv2
        import numpy as np
        import torch

        if not (checkout / "models" / "experimental.py").is_file():
            raise FileNotFoundError(f"YOLOv7 checkout is incomplete: {checkout}")
        if not weights.is_file():
            raise FileNotFoundError(f"YOLOv7 weights not found: {weights}")
        _verify_official_weights(weights)
        sys.path.insert(0, str(checkout))
        # YOLOv7's inference modules import its training-only plotting helper,
        # which in turn imports seaborn. The model path never calls seaborn;
        # avoid mutating the shared VLFM environment just for that optional UI.
        sys.modules.setdefault("seaborn", types.ModuleType("seaborn"))
        from models.experimental import attempt_load
        from utils.datasets import letterbox
        from utils.general import check_img_size, non_max_suppression, scale_coords
        from utils.torch_utils import TracedModel

        self.cv2 = cv2
        self.np = np
        self.torch = torch
        self.letterbox = letterbox
        self.non_max_suppression = non_max_suppression
        self.scale_coords = scale_coords
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.half = self.device.type == "cuda"
        # PyTorch 2.6 defaults torch.load(weights_only=True), while the pinned
        # official YOLOv7 checkpoint contains its legacy model class. The hash
        # check above gates this narrowly scoped compatibility shim.
        original_torch_load = torch.load

        def legacy_torch_load(*args, **kwargs):
            kwargs.setdefault("weights_only", False)
            return original_torch_load(*args, **kwargs)

        torch.load = legacy_torch_load
        try:
            self.model = attempt_load(str(weights), map_location=self.device)
        finally:
            torch.load = original_torch_load
        stride = int(self.model.stride.max())
        self.image_size = check_img_size(640, s=stride)
        trace_directory = checkout / ".apexnav_runtime"
        trace_directory.mkdir(parents=True, exist_ok=True)
        previous_directory = Path.cwd()
        try:
            os.chdir(trace_directory)
            self.model = TracedModel(self.model, self.device, self.image_size)
        finally:
            os.chdir(previous_directory)
        if self.half:
            self.model.half()

    def predict(self, image, payload: dict[str, Any]) -> dict[str, Any]:
        original_shape = image.shape
        resized = self.cv2.resize(
            image,
            (self.image_size, int(self.image_size * 0.7)),
            interpolation=self.cv2.INTER_AREA,
        )
        resized = self.letterbox(resized, new_shape=self.image_size)[0]
        tensor = self.np.ascontiguousarray(resized.transpose(2, 0, 1))
        tensor = self.torch.from_numpy(tensor).to(self.device)
        tensor = tensor.half() if self.half else tensor.float()
        tensor /= 255.0
        if tensor.ndimension() == 3:
            tensor = tensor.unsqueeze(0)
        with self.torch.inference_mode():
            prediction = self.model(tensor)[0]
        prediction = self.non_max_suppression(
            prediction,
            float(payload.get("conf_thres", 0.30)),
            float(payload.get("iou_thres", 0.50)),
            agnostic=bool(payload.get("agnostic_nms", True)),
        )[0]
        if prediction is None or len(prediction) == 0:
            return {"boxes": [], "logits": [], "phrases": []}
        prediction[:, :4] = self.scale_coords(
            tensor.shape[2:], prediction[:, :4], original_shape
        ).round()
        prediction[:, [0, 2]] /= original_shape[1]
        prediction[:, [1, 3]] /= original_shape[0]
        return {
            "boxes": prediction[:, :4].detach().cpu().tolist(),
            "logits": prediction[:, 4].detach().cpu().tolist(),
            "phrases": [COCO_CLASSES[int(index)] for index in prediction[:, 5]],
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkout",
        default="baselines/nav/apexnav/.deps/src/yolov7",
    )
    parser.add_argument(
        "--weights",
        default="/home/yor/models/apexnav/yolov7-e6e.pt",
    )
    parser.add_argument("--port", type=int, default=12184)
    args = parser.parse_args(argv)

    from flask import Flask, jsonify, request

    model = YOLOv7Server(
        Path(args.checkout).expanduser().resolve(),
        Path(args.weights).expanduser().resolve(),
    )
    app = Flask("apexnav_yolov7")

    @app.post("/yolov7")
    def infer():
        payload = request.get_json(force=True)
        return jsonify(model.predict(_decode_image(payload["image"]), payload))

    app.run(host="127.0.0.1", port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
