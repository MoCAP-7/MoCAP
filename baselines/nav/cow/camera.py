"""Render ZED RGB-D frames into the camera model CoW was evaluated with.

CoW's mapping assumes a square pinhole image from a camera whose optical axis
is horizontal, at a known height, with one field of view for both axes. The
ZED is rigidly tilted, so each frame is re-rendered about the same optical
centre into such a level virtual camera. This is a pure rotation, so no
parallax or hole filling is involved: each virtual pixel takes the nearest
source pixel, and that pixel's measured 3D point is re-expressed as depth
along the virtual optical axis. Using the measured point rather than the
virtual ray keeps the height error proportional to range even where the floor
meets the horizon and depth changes quickly between neighbouring pixels.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LevelCameraModel:
    source_width: int
    source_height: int
    intrinsics: tuple[tuple[float, float, float], ...]
    down_camera_xyz: tuple[float, float, float]
    output_size: int = 672
    fov_deg: float = 90.0


class LevelCamera:
    def __init__(self, model: LevelCameraModel) -> None:
        if model.output_size <= 0 or not 0.0 < model.fov_deg < 180.0:
            raise ValueError("output_size must be positive and fov_deg in (0, 180)")
        k = np.asarray(model.intrinsics, dtype=np.float64)
        if k.shape != (3, 3):
            raise ValueError("intrinsics must be a 3x3 matrix")
        self.model = model
        rotation_level_from_camera = level_rotation(model.down_camera_xyz)
        self.rotation_camera_from_level = rotation_level_from_camera.T

        size = model.output_size
        focal = (size / 2.0) / np.tan(np.radians(model.fov_deg) / 2.0)
        # Pixel centres follow CoW's unprojection: index i maps to (i + 0.5 - size / 2).
        centres = np.arange(size, dtype=np.float64) + 0.5 - size / 2.0
        u_level, v_level = np.meshgrid(centres / focal, centres / focal)
        rays_level = np.stack([u_level, v_level, np.ones_like(u_level)], axis=-1)
        rays_camera = rays_level @ self.rotation_camera_from_level.T
        forward = rays_camera[..., 2]
        in_front = forward > 1e-6
        safe_forward = np.where(in_front, forward, 1.0)
        u_source = k[0, 0] * rays_camera[..., 0] / safe_forward + k[0, 2]
        v_source = k[1, 1] * rays_camera[..., 1] / safe_forward + k[1, 2]
        inside = (
            in_front
            & (u_source >= 0.0)
            & (u_source <= model.source_width - 1)
            & (v_source >= 0.0)
            & (v_source <= model.source_height - 1)
        )
        self._inside = inside
        self._u = u_source
        self._v = v_source
        self._u_nearest = np.clip(np.rint(u_source), 0, model.source_width - 1).astype(np.intp)
        self._v_nearest = np.clip(np.rint(v_source), 0, model.source_height - 1).astype(np.intp)
        # A source pixel's point is depth * (x, y, 1); its level-frame depth is
        # that point projected on the level optical axis.
        source_rays = np.stack(
            [
                (self._u_nearest - k[0, 2]) / k[0, 0],
                (self._v_nearest - k[1, 2]) / k[1, 1],
                np.ones(self._u_nearest.shape, dtype=np.float64),
            ],
            axis=-1,
        )
        self._level_depth_per_source_depth = source_rays @ rotation_level_from_camera[2]

    @property
    def coverage(self) -> float:
        """Fraction of virtual pixels that see the source image."""

        return float(self._inside.mean())

    def render(self, rgb: np.ndarray, depth_m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        expected = (self.model.source_height, self.model.source_width)
        if rgb.shape[:2] != expected or depth_m.shape[:2] != expected:
            raise ValueError(f"frame shape {rgb.shape[:2]} / {depth_m.shape[:2]} != {expected}")
        return self.render_rgb(rgb), self.render_depth(depth_m)

    def render_rgb(self, rgb: np.ndarray) -> np.ndarray:
        image = np.asarray(rgb, dtype=np.float32)
        u0 = np.clip(np.floor(self._u).astype(np.intp), 0, self.model.source_width - 2)
        v0 = np.clip(np.floor(self._v).astype(np.intp), 0, self.model.source_height - 2)
        du = np.clip(self._u - u0, 0.0, 1.0)[..., None]
        dv = np.clip(self._v - v0, 0.0, 1.0)[..., None]
        blended = (
            image[v0, u0] * (1.0 - du) * (1.0 - dv)
            + image[v0, u0 + 1] * du * (1.0 - dv)
            + image[v0 + 1, u0] * (1.0 - du) * dv
            + image[v0 + 1, u0 + 1] * du * dv
        )
        blended[~self._inside] = 0.0
        return np.clip(np.rint(blended), 0, 255).astype(np.uint8)

    def render_depth(self, depth_m: np.ndarray) -> np.ndarray:
        """Nearest-neighbour depth along the virtual optical axis; invalid is 0."""

        depth = np.asarray(depth_m, dtype=np.float32)
        if depth.ndim == 3:
            depth = depth[..., 0]
        sampled = depth[self._v_nearest, self._u_nearest].astype(np.float64)
        level = sampled * self._level_depth_per_source_depth
        valid = self._inside & np.isfinite(sampled) & (sampled > 0.0) & (level > 0.0)
        return np.where(valid, level, 0.0).astype(np.float32)


def level_rotation(down_camera_xyz: tuple[float, float, float]) -> np.ndarray:
    """Rotation taking optical-frame vectors (x right, y down, z forward) to the level frame.

    The level frame keeps the camera's heading, with y along gravity.
    """

    down = np.asarray(down_camera_xyz, dtype=np.float64).reshape(3)
    norm = np.linalg.norm(down)
    if not np.isfinite(norm) or norm < 1e-6:
        raise ValueError("down_camera_xyz must be a finite nonzero vector")
    y_axis = down / norm
    optical_axis = np.array([0.0, 0.0, 1.0])
    z_axis = optical_axis - np.dot(optical_axis, y_axis) * y_axis
    z_norm = np.linalg.norm(z_axis)
    if z_norm < 1e-6:
        raise ValueError("the camera must not look straight along gravity")
    z_axis /= z_norm
    x_axis = np.cross(y_axis, z_axis)
    return np.stack([x_axis, y_axis, z_axis])
