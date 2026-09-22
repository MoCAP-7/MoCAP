"""Voxel-surface meshes with far fewer triangles than cuRobo's per-voxel quads.

``curobo.scene.Mesh.from_pointcloud`` voxelizes a point cloud and emits one quad
(two triangles) for every exposed voxel face. A table top observed at 12 mm
pitch therefore becomes tens of thousands of coplanar triangles, and cuRobo's
Warp BVH closest-point queries (one per robot sphere per optimizer step) slow
down with that count: on the Jetson a 12.7k-point table-and-wall scene planned
in 63 s where an almost empty world planned in 7 s (2026-09-04 benchmark).

``greedy_voxel_surface`` builds the *same* occupied-voxel surface, with the same
vertex lattice ``origin + index * pitch`` and the same face orientation as
cuRobo, but merges coplanar exposed faces into maximal rectangles (greedy
meshing). The represented solid is identical; only the triangle count drops.

``crop_points`` removes points that no robot sphere can ever get within the
mesh query distance of; they cannot influence any collision term.

This module is free of ``curobo``/``torch`` imports so it is unit-tested on a
development machine.
"""

from __future__ import annotations

from typing import Any

import numpy as np


# Unit-quad corner offsets per exposed-face direction, in cuRobo's order
# (types.py Mesh.from_pointcloud face_templates): -x, +x, -y, +y, -z, +z.
_FACE_TEMPLATES = np.array(
    [
        [[0, 0, 0], [0, 1, 0], [0, 1, 1], [0, 0, 1]],  # -x face
        [[1, 0, 0], [1, 0, 1], [1, 1, 1], [1, 1, 0]],  # +x face
        [[0, 0, 0], [0, 0, 1], [1, 0, 1], [1, 0, 0]],  # -y face
        [[0, 1, 0], [1, 1, 0], [1, 1, 1], [0, 1, 1]],  # +y face
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],  # -z face
        [[0, 0, 1], [0, 1, 1], [1, 1, 1], [1, 0, 1]],  # +z face
    ],
    dtype=np.int64,
)
# (axis, direction, template index) in cuRobo's iteration order.
_AXIS_SHIFTS = (
    (0, 1, 1),
    (0, -1, 0),
    (1, 1, 3),
    (1, -1, 2),
    (2, 1, 5),
    (2, -1, 4),
)


def voxelize(points: np.ndarray, pitch: float) -> tuple[np.ndarray, np.ndarray]:
    """Return cuRobo's occupancy grid and grid origin for a point cloud."""

    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) == 0:
        raise ValueError("points must have shape [N, 3] with N >= 1")
    if not np.isfinite(pitch) or pitch <= 0.0:
        raise ValueError("pitch must be positive")
    origin = pts.min(axis=0) - pitch
    ijk = np.floor((pts - origin) / pitch).astype(np.int64)
    grid_shape = ijk.max(axis=0) + 3  # +2 pad so the boundary is always empty
    occupied = np.zeros(grid_shape, dtype=bool)
    occupied[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = True
    return occupied, origin


def exposed_faces(occupied: np.ndarray, axis: int, direction: int) -> np.ndarray:
    """Boolean grid of voxels whose face in ``direction`` along ``axis`` is exposed."""

    shifted = np.roll(occupied, -direction, axis=axis)
    return occupied & ~shifted


def greedy_rectangles(mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Cover the True cells of a 2-D mask with maximal axis-aligned rectangles.

    Returns ``(row, column, height, width)`` tuples. Every True cell is covered
    exactly once and no False cell is covered.
    """

    mask = np.asarray(mask, dtype=bool)
    remaining = mask.copy()
    rectangles: list[tuple[int, int, int, int]] = []
    rows, columns = np.nonzero(remaining)
    for row, column in zip(rows.tolist(), columns.tolist()):
        if not remaining[row, column]:
            continue
        width = 1
        while column + width < remaining.shape[1] and remaining[row, column + width]:
            width += 1
        height = 1
        while row + height < remaining.shape[0] and bool(
            np.all(remaining[row + height, column : column + width])
        ):
            height += 1
        remaining[row : row + height, column : column + width] = False
        rectangles.append((row, column, height, width))
    return rectangles


def greedy_voxel_surface(
    points: np.ndarray, pitch: float
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Mesh the occupied-voxel surface of ``points`` with merged coplanar faces.

    Returns ``(vertices[V, 3], triangles[T, 3], stats)``. The surface, vertex
    lattice and face orientation equal cuRobo's ``Mesh.from_pointcloud``; the
    triangle count is what ``stats['per_voxel_triangles']`` would have been
    reduced to ``stats['triangles']``.
    """

    occupied, origin = voxelize(points, pitch)
    vertices: list[np.ndarray] = []
    triangles: list[np.ndarray] = []
    vertex_offset = 0
    per_voxel_quads = 0
    for axis, direction, template_index in _AXIS_SHIFTS:
        boundary = exposed_faces(occupied, axis, direction)
        per_voxel_quads += int(np.count_nonzero(boundary))
        template = _FACE_TEMPLATES[template_index]
        in_plane = [a for a in range(3) if a != axis]
        for slab in range(boundary.shape[axis]):
            slice_mask = np.take(boundary, slab, axis=axis)
            if not slice_mask.any():
                continue
            for row, column, height, width in greedy_rectangles(slice_mask):
                base = np.zeros(3, dtype=np.int64)
                base[axis] = slab
                base[in_plane[0]] = row
                base[in_plane[1]] = column
                scale = np.ones(3, dtype=np.int64)
                scale[in_plane[0]] = height
                scale[in_plane[1]] = width
                corners = base[None, :] + template * scale[None, :]
                vertices.append(corners.astype(np.float64) * pitch + origin)
                triangles.append(
                    np.asarray(
                        [
                            [vertex_offset, vertex_offset + 1, vertex_offset + 2],
                            [vertex_offset, vertex_offset + 2, vertex_offset + 3],
                        ],
                        dtype=np.int64,
                    )
                )
                vertex_offset += 4
    if not vertices:
        raise ValueError("point cloud produced no exposed voxel faces")
    vertex_array = np.concatenate(vertices, axis=0)
    triangle_array = np.concatenate(triangles, axis=0)
    stats = {
        "occupied_voxels": int(np.count_nonzero(occupied)),
        "per_voxel_triangles": int(2 * per_voxel_quads),
        "triangles": int(len(triangle_array)),
        "vertices": int(len(vertex_array)),
        "pitch_m": float(pitch),
    }
    return vertex_array, triangle_array, stats


def per_voxel_surface(points: np.ndarray, pitch: float) -> tuple[np.ndarray, np.ndarray]:
    """Reference re-implementation of cuRobo's per-voxel quad mesh (for tests)."""

    occupied, origin = voxelize(points, pitch)
    vertices: list[np.ndarray] = []
    triangles: list[np.ndarray] = []
    offset = 0
    for axis, direction, template_index in _AXIS_SHIFTS:
        coords = np.argwhere(exposed_faces(occupied, axis, direction))
        if len(coords) == 0:
            continue
        template = _FACE_TEMPLATES[template_index]
        quad_vertices = coords[:, None, :] + template[None, :, :]
        vertices.append(quad_vertices.reshape(-1, 3).astype(np.float64) * pitch + origin)
        index = np.arange(len(coords)) * 4 + offset
        triangles.append(
            np.column_stack([index, index + 1, index + 2, index, index + 2, index + 3]).reshape(
                -1, 3
            )
        )
        offset += len(coords) * 4
    return np.concatenate(vertices, axis=0), np.concatenate(triangles, axis=0)


def tiled_voxel_surfaces(
    points: np.ndarray, pitch: float, tile_m: float
) -> tuple[list[tuple[np.ndarray, np.ndarray]], dict[str, Any]]:
    """Split cuRobo's per-voxel surface of ``points`` into small tile meshes.

    cuRoboV2's mesh distance kernel (``curobo/_src/geom/data/data_mesh.py``,
    ``compute_local_sdf``) queries each mesh with a maximum distance of half
    its bounding-box diagonal, so one scene-sized mesh makes every robot
    sphere search most of the BVH on every optimizer step (63 s per plan on
    the Jetson for a 1.0 x 0.65 x 0.5 m table scene versus 7 s for an
    almost empty world). Splitting the *same* exposed voxel faces by tile keeps
    the represented surface identical while each tile's bounding box, and
    hence its query radius, shrinks to about ``tile_m``.

    Returns ``[(vertices, triangles), ...]`` for every non-empty tile plus
    statistics. Faces are the per-voxel quads cuRobo would emit (same lattice,
    same winding); interior faces between occupied voxels are never exposed,
    so cutting the voxel grid along tile planes adds no faces.
    """

    if not np.isfinite(tile_m) or tile_m <= 0.0:
        raise ValueError("tile_m must be positive")
    occupied, origin = voxelize(points, pitch)
    voxels_per_tile = max(1, int(round(tile_m / pitch)))
    faces_by_tile: dict[tuple[int, ...], list[tuple[np.ndarray, np.ndarray]]] = {}
    total_faces = 0
    for axis, direction, template_index in _AXIS_SHIFTS:
        coords = np.argwhere(exposed_faces(occupied, axis, direction))
        if len(coords) == 0:
            continue
        total_faces += len(coords)
        template = _FACE_TEMPLATES[template_index]
        tile_keys = coords // voxels_per_tile
        for key in np.unique(tile_keys, axis=0):
            members = coords[np.all(tile_keys == key, axis=1)]
            quad_vertices = members[:, None, :] + template[None, :, :]
            faces_by_tile.setdefault(tuple(int(k) for k in key), []).append(
                (quad_vertices.reshape(-1, 3).astype(np.float64) * pitch + origin, members)
            )
    tiles: list[tuple[np.ndarray, np.ndarray]] = []
    for key in sorted(faces_by_tile):
        vertex_blocks = [block for block, _ in faces_by_tile[key]]
        vertices = np.concatenate(vertex_blocks, axis=0)
        quad_count = len(vertices) // 4
        index = np.arange(quad_count) * 4
        triangles = np.column_stack(
            [index, index + 1, index + 2, index, index + 2, index + 3]
        ).reshape(-1, 3)
        tiles.append((vertices, triangles))
    if not tiles:
        raise ValueError("point cloud produced no exposed voxel faces")
    stats = {
        "occupied_voxels": int(np.count_nonzero(occupied)),
        "triangles": int(2 * total_faces),
        "per_voxel_triangles": int(2 * total_faces),
        "tiles": int(len(tiles)),
        "tile_m": float(voxels_per_tile * pitch),
        "pitch_m": float(pitch),
    }
    return tiles, stats


def crop_points(
    points: np.ndarray, radius_m: float, center: tuple[float, float, float] = (0.0, 0.0, 0.0)
) -> tuple[np.ndarray, int]:
    """Drop points farther than ``radius_m`` from ``center``; return kept and dropped count."""

    pts = np.asarray(points)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points must have shape [N, 3]")
    if not np.isfinite(radius_m) or radius_m <= 0.0:
        raise ValueError("radius_m must be positive")
    delta = pts - np.asarray(center, dtype=pts.dtype)
    keep = np.einsum("ij,ij->i", delta, delta) <= radius_m * radius_m
    return pts[keep], int(np.count_nonzero(~keep))
