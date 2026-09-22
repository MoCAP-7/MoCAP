"""VGGT-1B stance service: where the human's head was, in the robot camera frame.

The readiness prior sends one ZED capture (RGB, metric depth, the floor axes
expressed in the ZED optical frame) plus one or more first-person video frames.
VGGT predicts every camera relative to image 0 (the ZED view, OpenCV axes = the
ZED optical frame), and the depth head on image 0 alone fixes the metric scale
against the ZED depth. Each frame comes back as a camera row: forward/left on
the floor plane, height above the floor and the planar heading.

Only the aggregator, the camera head and the depth head are loaded, from the
fp16 safetensors file straight to the inference device (no fp32 copy anywhere).
``torch``, ``vggt``, ``safetensors`` and ``uvicorn`` are imported lazily inside
``load_model``, ``VggtRunner`` and ``main`` so the module and its unit tests do
not need them.
"""
from __future__ import annotations

import argparse
import base64
import binascii
import contextlib
import dataclasses
import io
import logging
import math
import pathlib
import re
import tempfile
import threading
import time
from typing import Any, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel, Field

logger = logging.getLogger("vggt_stance")

SCHEMA_VERSION = "yor-vggt-stance-v1"
DEFAULT_WEIGHTS = "/home/yor/models/vggt/vggt_1b_fp16.safetensors"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8117
DEFAULT_MAX_FRAMES = 8
# The VGGT preprocessing makes the long side 518 px; every camera here is
# landscape, so the content width of the padded square is the full 518.
MODEL_IMAGE_SIZE = 518
_SKIPPED_HEAD_PREFIXES = ("point_head.", "track_head.")

app = FastAPI(title="VGGT stance service")

_RUNNER: Any = None
_INFERENCE_LOCK = threading.Lock()
_MAX_FRAMES = DEFAULT_MAX_FRAMES
# What main() was asked to load; reported by /health while the model loads.
_STARTUP: dict[str, Any] = {"device": None, "dtype": None, "weights": None}


# --- results -----------------------------------------------------------------


@dataclasses.dataclass
class VggtRaw:
    """What one VGGT forward pass returns, as float32 numpy on the host."""

    extrinsics: np.ndarray  # (N, 3, 4) camera-from-world; world = camera 0
    intrinsics: np.ndarray  # (N, 3, 3) pixels at the model resolution
    depth0: np.ndarray  # (H, W) VGGT depth of image 0, VGGT units
    conf0: np.ndarray  # (H, W) depth confidence of image 0
    image0: np.ndarray  # (3, H, W) preprocessed image 0 (white padding = 1.0)
    model_hw: tuple[int, int]
    timing: dict[str, float]  # preprocess, aggregator, camera_head, depth_head
    depth_head_frames: int  # 1, or N when the S=1 slice fell back to the full head


@dataclasses.dataclass
class ScaleResult:
    scale: Optional[float]
    iqr_ratio: Optional[float]
    pixels: int
    conf_threshold_used: float


# --- pure numpy geometry (unit-tested without torch) -------------------------


def content_region(image0: np.ndarray) -> tuple[int, int, int, int]:
    """Rows/cols of image 0 that are not the constant padding: (r0, r1, c0, c1)."""
    image = np.asarray(image0, dtype=np.float64)
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError(f"image0 must be (3, H, W), got {image.shape}")
    rows = np.flatnonzero(image.std(axis=(0, 2)) > 1e-3)
    cols = np.flatnonzero(image.std(axis=(0, 1)) > 1e-3)
    if rows.size == 0 or cols.size == 0:
        raise ValueError("image 0 has no content region (constant image)")
    return int(rows.min()), int(rows.max()) + 1, int(cols.min()), int(cols.max()) + 1


def _resize_nearest(source: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Nearest-neighbour resize by linspace-rounded row/col indices (no cv2)."""
    height, width = source.shape
    rows = np.rint(np.linspace(0.0, height - 1, int(shape[0]))).astype(int)
    cols = np.rint(np.linspace(0.0, width - 1, int(shape[1]))).astype(int)
    return source[rows[:, None], cols[None, :]]


def align_scale(
    zed_depth_m: np.ndarray,
    depth0: np.ndarray,
    conf0: np.ndarray,
    region: tuple[int, int, int, int],
    *,
    conf_threshold: float = 2.0,
    min_depth_m: float = 0.2,
    max_depth_m: float = 8.0,
) -> ScaleResult:
    """Metric scale = median(ZED depth / VGGT depth of image 0) over confident pixels."""
    r0, r1, c0, c1 = region
    d0 = np.asarray(depth0, dtype=np.float64)[r0:r1, c0:c1]
    k0 = np.asarray(conf0, dtype=np.float64)[r0:r1, c0:c1]
    zed = np.nan_to_num(np.asarray(zed_depth_m, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    zed_small = _resize_nearest(zed, (r1 - r0, c1 - c0))
    median_conf = float(np.median(k0)) if k0.size else float("nan")
    threshold = float(conf_threshold)
    if math.isfinite(median_conf):
        threshold = min(threshold, median_conf)
    valid = (
        (zed_small > min_depth_m)
        & (zed_small < max_depth_m)
        & np.isfinite(d0)
        & (d0 > 1e-3)
        & (k0 >= threshold)
    )
    ratios = zed_small[valid] / d0[valid]
    pixels = int(ratios.size)
    if pixels == 0:
        return ScaleResult(scale=None, iqr_ratio=None, pixels=0, conf_threshold_used=threshold)
    scale = float(np.median(ratios))
    p25, p75 = (float(v) for v in np.percentile(ratios, [25, 75]))
    iqr_ratio = p75 / p25 if p25 > 0.0 and math.isfinite(p75 / p25) else None
    return ScaleResult(scale=scale, iqr_ratio=iqr_ratio, pixels=pixels, conf_threshold_used=threshold)


def camera_row(
    extri_i: np.ndarray,
    intri_i: np.ndarray,
    *,
    scale: Optional[float],
    down: np.ndarray,
    forward: np.ndarray,
    left: np.ndarray,
    camera_height_m: float,
    label: str,
    source_wh: tuple[int, int],
    model_w: int,
) -> dict[str, Any]:
    """One camera in the ZED planar frame: C = -R^T t * scale, axis = R^T [0,0,1]."""
    extri = np.asarray(extri_i, dtype=np.float64)
    intri = np.asarray(intri_i, dtype=np.float64)
    rotation = extri[:, :3]
    translation = extri[:, 3]
    centre = -rotation.T @ translation
    axis = rotation.T @ np.array([0.0, 0.0, 1.0])
    down = np.asarray(down, dtype=np.float64)
    forward = np.asarray(forward, dtype=np.float64)
    left = np.asarray(left, dtype=np.float64)
    heading_deg = math.degrees(math.atan2(float(axis @ left), float(axis @ forward)))
    focal_px_model = float(intri[0, 0])
    width, height = int(source_wh[0]), int(source_wh[1])
    focal_px = focal_px_model * width / float(model_w)
    forward_m: Optional[float] = None
    left_m: Optional[float] = None
    height_m: Optional[float] = None
    if scale is not None:
        centre_m = centre * float(scale)
        forward_m = float(centre_m @ forward)
        left_m = float(centre_m @ left)
        height_m = float(camera_height_m) - float(centre_m @ down)
    return {
        "label": label,
        "forward_m": forward_m,
        "left_m": left_m,
        "height_above_floor_m": height_m,
        "heading_deg": heading_deg,
        "focal_px": focal_px,
        "focal_px_model": focal_px_model,
        "width": width,
        "height": height,
    }


def _b64decode(value: str, what: str) -> bytes:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{what} is empty")
    try:
        data = base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError(f"{what} is not valid base64: {exc}") from exc
    if not data:
        raise ValueError(f"{what} decodes to zero bytes")
    return data


def _image_size(data: bytes, what: str) -> tuple[int, int]:
    """(width, height) from the image header; no full decode."""
    try:
        with Image.open(io.BytesIO(data)) as image:
            width, height = image.size
    except Exception as exc:  # PIL raises several unrelated types
        raise ValueError(f"{what} is not a decodable image: {exc}") from exc
    if width <= 0 or height <= 0:
        raise ValueError(f"{what} has an empty size {width}x{height}")
    return int(width), int(height)


def decode_jpeg_size(b64: str) -> tuple[int, int]:
    """(width, height) of a base64 JPEG via PIL (header only)."""
    return _image_size(_b64decode(b64, "jpeg"), "jpeg")


def decode_depth_png16(b64: str) -> np.ndarray:
    """Base64 16-bit grayscale PNG in millimetres -> float32 metres (0 -> NaN)."""
    data = _b64decode(b64, "depth_png16_base64")
    try:
        with Image.open(io.BytesIO(data)) as image:
            image.load()
            array = np.asarray(image)
    except Exception as exc:
        raise ValueError(f"depth PNG could not be decoded: {exc}") from exc
    if array.ndim != 2 or array.dtype.kind not in "iu" or array.dtype.itemsize not in (2, 4):
        raise ValueError(
            f"depth PNG must be 16-bit grayscale (mode I;16 or I), got {array.dtype} shape {array.shape}"
        )
    millimetres = array.astype(np.int64)
    depth = millimetres.astype(np.float32) / 1000.0
    depth[millimetres == 0] = np.nan
    return depth


# --- request / response models -------------------------------------------------


class RobotView(BaseModel):
    label: str = "zed"
    rgb_jpeg_base64: str
    depth_png16_base64: str
    intrinsics: list[list[float]]
    down: list[float]
    planar_forward: list[float]
    planar_left: list[float]
    ground_camera_height_m: float


class HumanFrame(BaseModel):
    label: str
    jpeg_base64: str


class StanceOptions(BaseModel):
    conf_threshold: float = 2.0
    min_depth_m: float = 0.2
    max_depth_m: float = 8.0


class StanceRequest(BaseModel):
    robot: RobotView
    frames: list[HumanFrame]
    options: StanceOptions = Field(default_factory=StanceOptions)


class CameraRow(BaseModel):
    label: str
    forward_m: Optional[float]
    left_m: Optional[float]
    height_above_floor_m: Optional[float]
    heading_deg: float
    focal_px: float
    focal_px_model: float
    width: int
    height: int


class StanceResponse(BaseModel):
    schema_version: str
    device: str
    dtype: str
    model_image_hw: list[int]
    content_region: dict[str, list[int]]
    scale: Optional[float]
    scale_iqr_ratio: Optional[float]
    scale_pixels: int
    conf_threshold: float
    robot_camera: CameraRow
    cameras: list[CameraRow]
    robot_focal_px: float
    timing_s: dict[str, float]
    depth_head_frames: int
    max_memory_mb: Optional[float]


# --- model -------------------------------------------------------------------


@contextlib.contextmanager
def _parameters_on_meta():
    """Construct a module with every parameter on the meta device.

    ``with torch.device("meta")`` cannot be used for VGGT: the DINOv2 backbone
    evaluates ``torch.linspace(...).item()`` in its constructor, which meta
    tensors refuse. Redirecting ``register_parameter`` instead (the pattern of
    accelerate's ``init_empty_weights``) leaves such scratch tensors on the CPU
    while every parameter lands on meta at registration, before its module
    initialises it: the constructor's ``torch.empty`` tensor exists on the CPU
    only until that swap, so the transient peak is one parameter, not the
    model, and no weight is ever initialised for real.
    """
    from torch import nn

    original = nn.Module.register_parameter

    def register_on_meta(module, name, param):
        original(module, name, param)
        if param is not None:
            registered = module._parameters[name]
            module._parameters[name] = nn.Parameter(
                registered.to("meta"), requires_grad=registered.requires_grad
            )

    nn.Module.register_parameter = register_on_meta  # type: ignore[assignment]
    try:
        yield
    finally:
        nn.Module.register_parameter = original  # type: ignore[assignment]


class VggtRunner:
    """Owns the loaded model; ``run`` maps image paths to a ``VggtRaw``."""

    def __init__(
        self,
        model: Any,
        *,
        device: str,
        dtype_name: str,
        weights: str,
        depth_head_full: bool = False,
    ) -> None:
        self.model = model
        self.device = device
        self.dtype_name = dtype_name
        self.weights = weights
        self.depth_head_full = depth_head_full

    def _torch_dtype(self):
        import torch

        return torch.float16 if self.dtype_name == "float16" else torch.float32

    def _is_cuda(self) -> bool:
        return self.device.startswith("cuda")

    def _synchronize(self) -> None:
        import torch

        if self._is_cuda():
            torch.cuda.synchronize()
        elif self.device == "mps":
            torch.mps.synchronize()

    def max_memory_mb(self) -> Optional[float]:
        import torch

        if self._is_cuda():
            return float(torch.cuda.max_memory_allocated()) / 2**20
        if self.device == "mps":
            return float(torch.mps.driver_allocated_memory()) / 2**20
        return None

    def _autocast(self):
        """float16 needs autocast on every device, not only CUDA.

        The DPT head adds a positional embedding that its helper returns as
        float32 (``vggt/heads/utils.py``, ``position_grid_to_embed``), so the
        fp16 convolution after it sees an fp32 input and raises without
        autocast ("Input type (MPSFloatType) and weight type (MPSHalfType)").
        Autocast re-casts that input, and runs softmax/layer norm in fp32.
        """
        import torch

        if self.dtype_name != "float16":
            return contextlib.nullcontext()
        device_type = "cuda" if self._is_cuda() else self.device
        return torch.autocast(device_type, dtype=torch.float16)

    def run(self, image_paths: list[str]) -> VggtRaw:
        import torch
        from vggt.utils.load_fn import load_and_preprocess_images
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri

        paths = [str(path) for path in image_paths]
        timing: dict[str, float] = {}
        model = self.model
        try:
            t0 = time.perf_counter()
            images = load_and_preprocess_images(paths, mode="pad")
            images = images.to(device=self.device, dtype=self._torch_dtype())[None]  # (1, N, 3, H, W)
            self._synchronize()
            timing["preprocess"] = time.perf_counter() - t0
            frame_count = int(images.shape[1])
            model_hw = (int(images.shape[-2]), int(images.shape[-1]))

            with torch.inference_mode(), self._autocast():
                t0 = time.perf_counter()
                tokens, patch_start_idx = model.aggregator(images)
                self._synchronize()
                timing["aggregator"] = time.perf_counter() - t0

                t0 = time.perf_counter()
                pose_enc = model.camera_head(tokens)[-1]
                extri, intri = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
                self._synchronize()
                timing["camera_head"] = time.perf_counter() - t0

                t0 = time.perf_counter()
                depth_head_frames = 1
                if self.depth_head_full:
                    depth, conf = model.depth_head(tokens, images=images, patch_start_idx=patch_start_idx)
                    depth_head_frames = frame_count
                else:
                    try:
                        # The DPT head is per-frame; slicing every cached token
                        # tensor to the first frame is exactly "depth on image 0".
                        tokens0 = [None if t is None else t[:, :1] for t in tokens]
                        depth, conf = model.depth_head(
                            tokens0, images=images[:, :1], patch_start_idx=patch_start_idx
                        )
                        del tokens0
                    except Exception as exc:
                        logger.warning(
                            "depth head on image 0 failed (%s: %s); running it on all %d frames",
                            type(exc).__name__,
                            exc,
                            frame_count,
                        )
                        depth, conf = model.depth_head(tokens, images=images, patch_start_idx=patch_start_idx)
                        depth_head_frames = frame_count
                self._synchronize()
                timing["depth_head"] = time.perf_counter() - t0
                del tokens

                extrinsics = extri[0].float().cpu().numpy().astype(np.float32)
                intrinsics = intri[0].float().cpu().numpy().astype(np.float32)
                depth0 = depth[0, 0, :, :, 0].float().cpu().numpy().astype(np.float32)
                conf0 = conf[0, 0].float().cpu().numpy().astype(np.float32)
                image0 = images[0, 0].float().cpu().numpy().astype(np.float32)
                del depth, conf, images, pose_enc, extri, intri
        finally:
            if self._is_cuda():
                # Jetson CPU and GPU share physical RAM: hand the allocator
                # blocks back after every request (sam3 precedent).
                torch.cuda.empty_cache()
        return VggtRaw(
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            depth0=depth0,
            conf0=conf0,
            image0=image0,
            model_hw=model_hw,
            timing=timing,
            depth_head_frames=depth_head_frames,
        )


def load_model(weights: str, device: str, dtype: str, *, depth_head_full: bool = False) -> VggtRunner:
    """Build VGGT-1B (camera + depth heads) and stream the fp16 weights to ``device``."""
    import torch
    from safetensors import safe_open
    from vggt.models.aggregator import _RESNET_MEAN, _RESNET_STD
    from vggt.models.vggt import VGGT

    dtype_by_name = {"float16": torch.float16, "float32": torch.float32}
    if dtype not in dtype_by_name:
        raise ValueError("dtype must be float16 or float32")
    torch_dtype = dtype_by_name[dtype]
    weights_path = pathlib.Path(weights)
    if not weights_path.is_file():
        raise FileNotFoundError(f"VGGT weights not found: {weights_path}")

    t0 = time.perf_counter()
    with _parameters_on_meta():
        model = VGGT(enable_point=False, enable_track=False)

    state: dict[str, Any] = {}
    skipped = 0
    with safe_open(str(weights_path), framework="pt", device=device) as handle:
        for key in handle.keys():
            if key.startswith(_SKIPPED_HEAD_PREFIXES):
                skipped += 1
                continue
            state[key] = handle.get_tensor(key).to(torch_dtype)
    loaded = len(state)
    missing, unexpected = model.load_state_dict(state, strict=False, assign=True)
    del state
    if missing or unexpected:
        raise RuntimeError(
            f"VGGT checkpoint mismatch: missing={sorted(missing)} unexpected={sorted(unexpected)}"
        )
    # Non-persistent buffers are not in the file; they were created on the CPU.
    for name, value in (("_resnet_mean", _RESNET_MEAN), ("_resnet_std", _RESNET_STD)):
        model.aggregator.register_buffer(
            name,
            torch.tensor(value, dtype=torch.float32).view(1, 1, 3, 1, 1).to(device=device, dtype=torch_dtype),
            persistent=False,
        )
    model.eval()
    model.requires_grad_(False)
    runner = VggtRunner(
        model, device=device, dtype_name=dtype, weights=str(weights_path), depth_head_full=depth_head_full
    )
    memory = runner.max_memory_mb()
    logger.info(
        "VGGT-1B loaded: %d tensors (%d point/track head tensors skipped) on %s as %s in %.1f s%s",
        loaded,
        skipped,
        device,
        dtype,
        time.perf_counter() - t0,
        "" if memory is None else f", device memory {memory:.0f} MB",
    )
    return runner


def set_runner(runner: Any) -> None:
    """Install the inference runner (a ``VggtRunner`` or a test double)."""
    global _RUNNER
    _RUNNER = runner


# --- HTTP --------------------------------------------------------------------


@app.exception_handler(RequestValidationError)
def _validation_error(_request: Request, exc: RequestValidationError) -> JSONResponse:
    parts = []
    for error in exc.errors():
        location = ".".join(str(item) for item in error.get("loc", ()))
        parts.append(f"{location}: {error.get('msg', 'invalid')}")
    return JSONResponse(status_code=400, content={"detail": "invalid request: " + "; ".join(parts)})


def _bad_request(message: str) -> HTTPException:
    return HTTPException(status_code=400, detail=message)


def _unit_vector(values: list[float], name: str) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64).reshape(-1)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise _bad_request(f"robot.{name} must be three finite numbers")
    norm = float(np.linalg.norm(vector))
    if not 0.9 <= norm <= 1.1:
        raise _bad_request(f"robot.{name} must be a unit vector (norm {norm:.3f})")
    return vector


def _intrinsics_matrix(values: list[list[float]]) -> np.ndarray:
    try:
        matrix = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise _bad_request(f"robot.intrinsics must be a 3x3 matrix: {exc}") from exc
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise _bad_request("robot.intrinsics must be a finite 3x3 matrix")
    return matrix


def _safe_label(label: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", label)[:40] or "frame"


@app.get("/health")
def health():
    runner = _RUNNER
    if runner is None:
        return JSONResponse(
            status_code=503,
            content={
                "status": "loading",
                "model_loaded": False,
                "device": _STARTUP["device"],
                "dtype": _STARTUP["dtype"],
                "weights": _STARTUP["weights"],
                "max_memory_mb": None,
            },
        )
    return {
        "status": "ok",
        "model_loaded": True,
        "device": str(runner.device),
        "dtype": str(runner.dtype_name),
        "weights": getattr(runner, "weights", _STARTUP["weights"]),
        "max_memory_mb": runner.max_memory_mb(),
    }


@app.post("/stance", response_model=StanceResponse)
def stance(request: StanceRequest) -> StanceResponse:
    runner = _RUNNER
    if runner is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    t_start = time.perf_counter()

    frames = request.frames
    if not frames:
        raise _bad_request("frames must contain at least one frame")
    if len(frames) > _MAX_FRAMES:
        raise _bad_request(f"frames has {len(frames)} entries; the limit is {_MAX_FRAMES}")
    labels = [frame.label for frame in frames]
    if any(not label.strip() for label in labels) or not request.robot.label.strip():
        raise _bad_request("frame labels must be non-empty")
    if len(set(labels)) != len(labels):
        raise _bad_request(f"frame labels must be unique, got {labels}")

    down = _unit_vector(request.robot.down, "down")
    forward = _unit_vector(request.robot.planar_forward, "planar_forward")
    left = _unit_vector(request.robot.planar_left, "planar_left")
    camera_height_m = float(request.robot.ground_camera_height_m)
    if not math.isfinite(camera_height_m) or not 0.2 < camera_height_m < 3.0:
        raise _bad_request("robot.ground_camera_height_m must be within (0.2, 3.0) m")
    intrinsics = _intrinsics_matrix(request.robot.intrinsics)
    options = request.options
    if not math.isfinite(options.conf_threshold):
        raise _bad_request("options.conf_threshold must be finite")
    if not (0.0 < options.min_depth_m < options.max_depth_m) or not math.isfinite(options.max_depth_m):
        raise _bad_request("options must satisfy 0 < min_depth_m < max_depth_m")

    try:
        rgb_bytes = _b64decode(request.robot.rgb_jpeg_base64, "robot.rgb_jpeg_base64")
        rgb_wh = _image_size(rgb_bytes, "robot.rgb_jpeg_base64")
        zed_depth = decode_depth_png16(request.robot.depth_png16_base64)
        frame_bytes: list[bytes] = []
        frame_wh: list[tuple[int, int]] = []
        for index, frame in enumerate(frames):
            data = _b64decode(frame.jpeg_base64, f"frames[{index}].jpeg_base64")
            frame_bytes.append(data)
            frame_wh.append(_image_size(data, f"frames[{index}].jpeg_base64"))
    except ValueError as exc:
        raise _bad_request(str(exc)) from exc
    if zed_depth.shape != (rgb_wh[1], rgb_wh[0]):
        raise _bad_request(
            f"depth shape {zed_depth.shape} does not match the rgb size (H, W) = {(rgb_wh[1], rgb_wh[0])}"
        )
    decode_s = time.perf_counter() - t_start

    with _INFERENCE_LOCK:
        with tempfile.TemporaryDirectory(prefix="vggt_stance_") as tmp:
            root = pathlib.Path(tmp)
            paths = [root / "00_robot.jpg"]
            paths[0].write_bytes(rgb_bytes)
            for index, (label, data) in enumerate(zip(labels, frame_bytes), start=1):
                path = root / f"{index:02d}_{_safe_label(label)}.jpg"
                path.write_bytes(data)
                paths.append(path)
            try:
                raw = runner.run([str(path) for path in paths])
                region = content_region(raw.image0)
            except Exception as exc:
                logger.exception("inference failed")
                raise HTTPException(
                    status_code=500, detail=f"inference failed: {type(exc).__name__}: {exc}"
                ) from exc
        max_memory_mb = runner.max_memory_mb()

    expected = 1 + len(frames)
    if int(raw.extrinsics.shape[0]) != expected:
        raise HTTPException(
            status_code=500,
            detail=f"inference failed: RuntimeError: {raw.extrinsics.shape[0]} cameras for {expected} images",
        )
    scale = align_scale(
        zed_depth,
        raw.depth0,
        raw.conf0,
        region,
        conf_threshold=float(options.conf_threshold),
        min_depth_m=float(options.min_depth_m),
        max_depth_m=float(options.max_depth_m),
    )
    model_w = int(raw.model_hw[1])
    row_kwargs = dict(
        scale=scale.scale,
        down=down,
        forward=forward,
        left=left,
        camera_height_m=camera_height_m,
        model_w=model_w,
    )
    robot_row = camera_row(
        raw.extrinsics[0], raw.intrinsics[0], label=request.robot.label, source_wh=rgb_wh, **row_kwargs
    )
    rows = [
        camera_row(raw.extrinsics[i + 1], raw.intrinsics[i + 1], label=labels[i], source_wh=frame_wh[i], **row_kwargs)
        for i in range(len(frames))
    ]
    timing_s = {
        "decode": decode_s,
        "preprocess": float(raw.timing.get("preprocess", 0.0)),
        "aggregator": float(raw.timing.get("aggregator", 0.0)),
        "camera_head": float(raw.timing.get("camera_head", 0.0)),
        "depth_head": float(raw.timing.get("depth_head", 0.0)),
    }
    timing_s["total"] = time.perf_counter() - t_start
    response = StanceResponse(
        schema_version=SCHEMA_VERSION,
        device=str(runner.device),
        dtype=str(runner.dtype_name),
        model_image_hw=[int(raw.model_hw[0]), int(raw.model_hw[1])],
        content_region={"rows": [region[0], region[1]], "cols": [region[2], region[3]]},
        scale=scale.scale,
        scale_iqr_ratio=scale.iqr_ratio,
        scale_pixels=scale.pixels,
        conf_threshold=scale.conf_threshold_used,
        robot_camera=CameraRow(**robot_row),
        cameras=[CameraRow(**row) for row in rows],
        robot_focal_px=float(intrinsics[0, 0]),
        timing_s=timing_s,
        depth_head_frames=int(raw.depth_head_frames),
        max_memory_mb=None if max_memory_mb is None else float(max_memory_mb),
    )
    logger.info(
        "stance: %d frame(s) %s; scale %s iqr %s over %d px (conf >= %.2f); robot %s; %s; "
        "depth head on %d frame(s); %.2f s total (aggregator %.2f s)%s",
        len(frames),
        labels,
        "none" if scale.scale is None else f"{scale.scale:.3f}",
        "none" if scale.iqr_ratio is None else f"{scale.iqr_ratio:.2f}",
        scale.pixels,
        scale.conf_threshold_used,
        _format_row(robot_row),
        "; ".join(_format_row(row) for row in rows),
        raw.depth_head_frames,
        timing_s["total"],
        timing_s["aggregator"],
        "" if max_memory_mb is None else f"; device memory {max_memory_mb:.0f} MB",
    )
    return response


def _format_row(row: dict[str, Any]) -> str:
    def fmt(value: Optional[float]) -> str:
        return "none" if value is None else f"{value:.3f}"

    return (
        f"{row['label']}: forward {fmt(row['forward_m'])} left {fmt(row['left_m'])} "
        f"height {fmt(row['height_above_floor_m'])} heading {row['heading_deg']:.1f} deg"
    )


# --- CLI ---------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="VGGT-1B stance service (human head pose relative to the ZED view)")
    parser.add_argument("--device", choices=("cuda", "mps", "cpu"), default="cuda")
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS, help="fp16 safetensors checkpoint")
    parser.add_argument("--max-frames", type=int, default=DEFAULT_MAX_FRAMES, help="frames per request limit")
    parser.add_argument(
        "--depth-head-full",
        action="store_true",
        help="run the depth head on every frame instead of image 0 only",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    global _MAX_FRAMES
    args = build_parser().parse_args(argv)
    if args.max_frames < 1:
        raise SystemExit("--max-frames must be at least 1")
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    import uvicorn

    _MAX_FRAMES = int(args.max_frames)
    _STARTUP.update({"device": args.device, "dtype": args.dtype, "weights": args.weights})
    set_runner(load_model(args.weights, args.device, args.dtype, depth_head_full=bool(args.depth_head_full)))
    uvicorn.run(app, host=args.host, port=int(args.port), log_level=str(args.log_level).lower())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
