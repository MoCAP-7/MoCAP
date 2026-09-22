"""Episode images: the level view with CoW's localization and a top-down map."""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image, ImageDraw


def attention_overlay(rgb: np.ndarray, attention: Any) -> Image.Image:
    """Mark the pixels CoW's localizer registered (224 x 224 grid) on the level view."""

    image = Image.fromarray(np.asarray(rgb, dtype=np.uint8))
    if attention is None:
        return image
    grid = np.asarray(attention.detach().cpu() if hasattr(attention, "detach") else attention)
    rows, cols = np.nonzero(grid > 0)
    if rows.size == 0:
        return image
    scale_y = image.height / grid.shape[0]
    scale_x = image.width / grid.shape[1]
    draw = ImageDraw.Draw(image)
    for row, col in zip(rows[:64], cols[:64]):
        cx, cy = (col + 0.5) * scale_x, (row + 0.5) * scale_y
        draw.ellipse([cx - 10, cy - 10, cx + 10, cy + 10], outline=(0, 255, 0), width=4)
    return image


def map_image(exploration: Any, voxel_types: Any, *, pixels_per_voxel: int = 4) -> Image.Image | None:
    """Top-down image of CoW's voxel map, forward up and left to the left."""

    nodes = list(exploration.voxels.nodes(data=True))
    if not nodes:
        return None
    lateral = np.array([key[0] for key, _ in nodes])
    forward = np.array([key[2] for key, _ in nodes])
    agent = getattr(exploration, "agent_voxel", None)
    if agent is not None:
        lateral = np.append(lateral, agent[0])
        forward = np.append(forward, agent[2])
    width = int(lateral.max() - lateral.min() + 1)
    height = int(forward.max() - forward.min() + 1)
    canvas = np.full((height, width, 3), 40, dtype=np.uint8)
    colors = {
        voxel_types.FREE: (235, 235, 235),
        voxel_types.OCCUPIED: (200, 40, 40),
        voxel_types.FRONTIER: (240, 200, 0),
        voxel_types.DBG: (60, 170, 60),
    }

    def cell(key: tuple[int, ...]) -> tuple[int, int]:
        return int(forward.max() - key[2]), int(lateral.max() - key[0])

    for key, data in nodes:
        row, col = cell(key)
        canvas[row, col] = colors.get(data.get("voxel_type"), (120, 120, 120))
        if data.get("roi_count", 0) > 0:
            canvas[row, col] = (40, 90, 230)
    if agent is not None:
        row, col = cell(agent)
        canvas[row, col] = (0, 0, 0)
    image = Image.fromarray(canvas)
    return image.resize((width * pixels_per_voxel, height * pixels_per_voxel), Image.NEAREST)
