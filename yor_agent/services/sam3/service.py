import asyncio
import base64
import functools
import gc
import io
import logging
from contextlib import nullcontext
from typing import Any, List, Tuple

import numpy as np
import torch
import tyro
import uvicorn
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel

from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Global state
_PROCESSOR: Any | None = None
_MODEL: Any | None = None
_DEVICE: str = "cuda"
_INFERENCE_DTYPE: torch.dtype = torch.float32
_RELEASE_CUDA_CACHE: bool = True

# Semaphore to serialize GPU access (prevents OOM from concurrent inference)
_GPU_SEMAPHORE = asyncio.Semaphore(1)


async def _run_on_gpu(fn, *args, **kwargs):
    """Run a blocking GPU function without blocking the event loop."""
    async with _GPU_SEMAPHORE:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, functools.partial(fn, *args, **kwargs))

# --- Helper Functions ---


def _to_numpy(tensor: Any) -> np.ndarray:
    """Convert tensor-like object to numpy array."""
    if hasattr(tensor, "detach"):
        t = tensor.detach().cpu()
        if t.dtype == torch.bfloat16:
            t = t.float()
        return t.numpy()
    if hasattr(tensor, "cpu"):
        t = tensor.cpu()
        if hasattr(t, "dtype") and t.dtype == torch.bfloat16:
            t = t.float()
        return t.numpy()
    if hasattr(tensor, "numpy"):
        return tensor.numpy()
    return np.asarray(tensor)


def _autocast_context():
    if "cuda" in _DEVICE and _INFERENCE_DTYPE in (torch.float16, torch.bfloat16):
        return torch.autocast("cuda", dtype=_INFERENCE_DTYPE)
    return nullcontext()


def _release_cuda_cache() -> None:
    if _RELEASE_CUDA_CACHE and "cuda" in _DEVICE and torch.cuda.is_available():
        # Jetson CPU and GPU share physical RAM. Returning unused allocator
        # blocks after each serialized request lets ZED and GraspNet reuse it.
        torch.cuda.empty_cache()


def _resolve_model_dtype(device: str, model_dtype: str) -> tuple[str, torch.dtype]:
    dtype_name = str(model_dtype).lower()
    if dtype_name == "auto":
        if "cuda" in device:
            dtype_name = (
                "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
            )
        else:
            dtype_name = "float32"
    dtype_by_name = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if dtype_name not in dtype_by_name:
        raise ValueError("model_dtype must be auto, float32, float16, or bfloat16")
    return dtype_name, dtype_by_name[dtype_name]


def _move_model_for_inference(model: Any, device: str, dtype: torch.dtype):
    """Move a module while compacting only real floating-point tensors.

    ``Module.to(dtype=...)`` also converts complex RoPE buffers to a real
    dtype, discarding their imaginary component. SAM3 needs those buffers to
    remain complex, while its ordinary parameters can safely use BF16/FP16.
    ``_apply`` performs the conversion one tensor at a time, which also avoids
    materializing complete FP32 and compact copies at once.
    """

    def convert(tensor: torch.Tensor) -> torch.Tensor:
        target_dtype = dtype if tensor.is_floating_point() else tensor.dtype
        return tensor.to(device=device, dtype=target_dtype)

    model = model._apply(convert)
    if dtype is not torch.float32:
        for module in model.modules():
            if getattr(module, "_sam3_force_fp32", False):
                module.to(device=device, dtype=torch.float32)
    return model


def decode_image(base64_str: str) -> Image.Image:
    try:
        image_data = base64.b64decode(base64_str)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        return image
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {e}")


def encode_mask(mask: np.ndarray) -> str:
    # Pack boolean mask to bytes (uint8) then base64
    return base64.b64encode(mask.astype(np.uint8).tobytes()).decode("utf-8")


def encode_array(arr: np.ndarray) -> str:
    """Encode a numpy array as base64 bytes."""
    return base64.b64encode(np.ascontiguousarray(arr).tobytes()).decode("utf-8")


# --- Request/Response Models ---


class SegmentRequest(BaseModel):
    image_base64: str
    text_prompt: str


class PointPromptRequest(BaseModel):
    image_base64: str
    point_coords: list[float]  # [x, y] — JSON arrays, not tuples


class PointPromptResponse(BaseModel):
    scores: list[float]
    masks_base64: str
    masks_shape: list[int]  # [num_masks, H, W]
    masks_dtype: str


class MaskData(BaseModel):
    mask_base64: str
    shape: list[int]  # [H, W]
    box: list[float]  # [x1, y1, x2, y2]
    score: float
    label: str


class SegmentResponse(BaseModel):
    results: list[MaskData]


# --- Core Logic ---


def _segment_to_cpu(pil_image: Image.Image, text_prompt: str):
    """Run inference and return only detached CPU arrays."""
    with torch.inference_mode(), _autocast_context():
        inference_state = _PROCESSOR.set_image(pil_image)
        output = _PROCESSOR.set_text_prompt(state=inference_state, prompt=text_prompt)

        masks_tensor = output.get("masks")
        boxes_tensor = output.get("boxes")
        scores_tensor = output.get("scores")

        if masks_tensor is None or boxes_tensor is None:
            return None

        return (
            _to_numpy(masks_tensor),
            _to_numpy(boxes_tensor),
            _to_numpy(scores_tensor),
        )


def _do_segment(pil_image: Image.Image, text_prompt: str):
    """Blocking SAM3 text-prompt segmentation (runs on GPU thread)."""
    try:
        arrays = _segment_to_cpu(pil_image, text_prompt)
    finally:
        # _segment_to_cpu owns all GPU inference-state references. They are
        # gone before empty_cache is called here, including on inference error.
        _release_cuda_cache()

    if arrays is None:
        return SegmentResponse(results=[])

    masks_np, boxes_np, scores_np = arrays

    # Squeeze masks if needed: (N, 1, H, W) -> (N, H, W)
    if masks_np.ndim == 4 and masks_np.shape[1] == 1:
        masks_np = masks_np.squeeze(1)

    results_data = []
    num_preds = len(scores_np)

    for i in range(num_preds):
        mask = masks_np[i] > 0  # Boolean mask
        box = boxes_np[i].tolist()  # [x1, y1, x2, y2]
        score = float(scores_np[i])

        results_data.append(
            MaskData(
                mask_base64=encode_mask(mask),
                shape=mask.shape,
                box=box,
                score=score,
                label=text_prompt,
            )
        )

    # Sort by score descending
    results_data.sort(key=lambda x: x.score, reverse=True)

    return SegmentResponse(results=results_data)


@app.post("/segment", response_model=SegmentResponse)
async def segment(req: SegmentRequest):
    if _PROCESSOR is None:
        raise HTTPException(status_code=503, detail="Model not initialized")

    pil_image = decode_image(req.image_base64)

    try:
        return await _run_on_gpu(_do_segment, pil_image, req.text_prompt)
    except Exception as e:
        logger.exception("Inference failed")
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")


def _segment_point_to_cpu(
    pil_image: Image.Image, point_coords_tuple: tuple[float, float]
):
    with torch.inference_mode(), _autocast_context():
        inference_state = _PROCESSOR.set_image(pil_image)
        point_coords = np.array([list(point_coords_tuple)], dtype=np.float32)
        point_labels = np.array([1], dtype=np.int64)  # foreground point
        masks, scores, _ = _MODEL.predict_inst(
            inference_state,
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=True,
        )

        return _to_numpy(masks), _to_numpy(scores)


def _do_segment_point(pil_image: Image.Image, point_coords_tuple: tuple[float, float]):
    """Blocking SAM3 point-prompt segmentation (runs on GPU thread)."""
    try:
        masks_np, scores_np = _segment_point_to_cpu(
            pil_image, point_coords_tuple
        )
    finally:
        _release_cuda_cache()

    if masks_np.size == 0 or scores_np.size == 0:
        return PointPromptResponse(
            scores=[],
            masks_base64="",
            masks_shape=(0, 0, 0),
            masks_dtype="float32",
        )

    # Sort by score descending
    sort_idx = np.argsort(scores_np)[::-1]
    masks_np = masks_np[sort_idx]
    scores_np = scores_np[sort_idx]

    return PointPromptResponse(
        scores=scores_np.astype(float).tolist(),
        masks_base64=encode_array(masks_np),
        masks_shape=tuple(masks_np.shape),
        masks_dtype=str(masks_np.dtype),
    )


@app.post("/segment_point", response_model=PointPromptResponse)
async def segment_point(req: PointPromptRequest):
    if _PROCESSOR is None or _MODEL is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    if getattr(_MODEL, "inst_interactive_predictor", None) is None:
        raise HTTPException(
            status_code=503,
            detail="Instance interactivity not enabled on SAM3 model",
        )

    pil_image = decode_image(req.image_base64)

    try:
        return await _run_on_gpu(_do_segment_point, pil_image, req.point_coords)
    except Exception as e:
        logger.exception("Point prompt inference failed")
        raise HTTPException(status_code=500, detail=f"Point prompt inference failed: {e}")


def main(
    device: str = "cuda",
    port: int = 8114,
    host: str = "127.0.0.1",
    confidence_threshold: float = 0.05,
    max_masks: int = 3,
    enable_inst_interactivity: bool = False,
    model_dtype: str = "auto",
    mmap_checkpoint: bool = True,
    release_cuda_cache: bool = True,
):
    global _MODEL, _PROCESSOR, _DEVICE, _INFERENCE_DTYPE, _RELEASE_CUDA_CACHE

    _DEVICE = device
    _RELEASE_CUDA_CACHE = bool(release_cuda_cache)
    if not 0.0 <= float(confidence_threshold) <= 1.0:
        raise ValueError("confidence_threshold must be in [0, 1]")
    if int(max_masks) <= 0:
        raise ValueError("max_masks must be positive")

    dtype_name, _INFERENCE_DTYPE = _resolve_model_dtype(device, model_dtype)

    # Setup torch settings for Ampere+ GPUs as recommended
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # Set the default CUDA device so all tensor allocations inside
        # build_sam3_image_model and Sam3Processor use the correct device.
        # Without this, internal tensors default to cuda:0 while model
        # weights are on the specified device, causing device mismatch errors.
        if "cuda" in device:
            device_idx = int(device.split(":")[-1]) if ":" in device else 0
            torch.cuda.set_device(device_idx)

    logger.info("Loading SAM3 model...")
    try:
        # Construct and load on CPU first, with the 3.2 GiB checkpoint kept
        # file-backed. Convert once to compact inference weights on the target
        # device. This avoids a persistent FP32 CUDA model on unified memory.
        _MODEL = build_sam3_image_model(
            device="cpu",
            enable_inst_interactivity=enable_inst_interactivity,
            mmap_checkpoint=mmap_checkpoint,
        )
    except Exception as e:
        logger.error(f"Error building SAM3 model: {e}")
        raise

    if hasattr(_MODEL, "to") and device:
        _MODEL = _move_model_for_inference(
            _MODEL,
            device=device,
            dtype=_INFERENCE_DTYPE,
        )
    _MODEL.eval()
    gc.collect()
    _release_cuda_cache()

    _PROCESSOR = Sam3Processor(
        _MODEL,
        device=device,
        confidence_threshold=float(confidence_threshold),
        max_masks=int(max_masks),
    )
    logger.info(
        "SAM3 model loaded on %s (dtype=%s, max_masks=%d, threshold=%.3f, "
        "instance_interactivity=%s). Starting Server...",
        device,
        dtype_name,
        max_masks,
        confidence_threshold,
        enable_inst_interactivity,
    )

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    tyro.cli(main)
