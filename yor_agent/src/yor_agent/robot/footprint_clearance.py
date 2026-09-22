"""Footprint-swept clearance for direct base motion.

Principle: a commanded base velocity is sent only when the robot footprint,
swept along that velocity for the reaction horizon and the braking distance,
contains no observed obstacle.  Anything that has not been observed is not
assumed to be free; the caller decides what to do about unobserved space.

The footprint is a set of height layers, each a convex polygon in the base
frame (origin at the swerve center, +x forward, +y left, z above the floor).
ZED depth pixels are converted into base-frame points with the floor plane
and the camera offset, rasterized into cells per layer, and a cell counts as
occupied when enough points fall into it.  A thin chair leg therefore stops
the base with a single occupied cell, and a wall approached at an angle is
caught by the full width of the footprint rather than by a central cone.

Because the ZED cannot see close to the body, :class:`StickyOccupancy` keeps
the occupied cells seen earlier in the same primitive call in the odometry
frame and re-checks them against the current footprint pose every cycle.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import math
from typing import Any

import numpy as np

# Chassis 43 x 34.5 cm. The arms layer is the fallback used only when the live
# sphere envelope is unavailable, so it is drawn around that envelope: in the
# travel pose the collision spheres reach 0.457 m ahead of the swerve centre
# and 0.433 m to each side, i.e. 0.87 m across (measured from the deployed
# sphere model, 2026-09-10). Margins of 5 cm are already included.
#
# An earlier revision put the arms at 1.20 m across and padded every generated
# outline to match. That figure was never measured; it made the modelled robot
# 0.32 m wider than it is and refused aisles it fits through.
DEFAULT_LAYERS: tuple[dict[str, Any], ...] = (
    {
        "name": "chassis",
        "z_min": 0.10,
        "z_max": 0.27,
        "polygon_xy": [[0.22, 0.27], [-0.22, 0.27], [-0.22, -0.27], [0.22, -0.27]],
    },
    {
        "name": "arms",
        "z_min": 0.27,
        "z_max": 1.45,
        "polygon_xy": [
            [0.51, 0.49],
            [-0.20, 0.49],
            [-0.20, -0.49],
            [0.51, -0.49],
        ],
    },
)

CLEAR = "clear"
BLOCKED = "obstacle_in_swept_footprint"


def convex_hull(points: np.ndarray) -> np.ndarray:
    """Return the counter-clockwise convex hull of 2-D points (monotone chain)."""

    pts = np.unique(np.asarray(points, dtype=np.float64).reshape(-1, 2), axis=0)
    if len(pts) < 3:
        return pts
    order = np.lexsort((pts[:, 1], pts[:, 0]))
    pts = pts[order]

    def cross(o: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
        return float((a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]))

    lower: list[np.ndarray] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0.0:
            lower.pop()
        lower.append(p)
    upper: list[np.ndarray] = []
    for p in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0.0:
            upper.pop()
        upper.append(p)
    return np.asarray(lower[:-1] + upper[:-1], dtype=np.float64)


def points_in_convex_polygon(points_xy: np.ndarray, polygon_ccw: np.ndarray) -> np.ndarray:
    """Vectorized inclusion test against a counter-clockwise convex polygon."""

    pts = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
    poly = np.asarray(polygon_ccw, dtype=np.float64)
    if len(pts) == 0:
        return np.zeros(0, dtype=bool)
    if len(poly) < 3:
        return np.zeros(len(pts), dtype=bool)
    # Vectorized over edges as well as points: height-resolved footprints have
    # an order of magnitude more layers and vertices than the two hand-written
    # ones, and a Python loop per edge dominated the control cycle.
    a = poly
    edge = np.roll(poly, -1, axis=0) - poly
    edge_cross = np.subtract.outer(pts[:, 1], a[:, 1]) * edge[:, 0] - np.subtract.outer(
        pts[:, 0], a[:, 0]
    ) * edge[:, 1]
    return np.all(edge_cross >= -1e-9, axis=1)


def polygon_support_along(
    polygon_ccw: np.ndarray, direction_xy: np.ndarray, laterals: np.ndarray
) -> np.ndarray:
    """Front boundary of a convex polygon along ``direction`` per lateral line.

    For every lateral offset (perpendicular to ``direction``) this returns the
    largest coordinate along ``direction`` of the polygon on that line, or
    ``-inf`` when the line misses the polygon.  It gives the exact free
    distance of an obstacle cell from the part of the outline that will hit
    it, instead of from the outline's single most advanced vertex.
    """

    poly = np.asarray(polygon_ccw, dtype=np.float64)
    direction = np.asarray(direction_xy, dtype=np.float64)
    lateral_axis = np.asarray([-direction[1], direction[0]], dtype=np.float64)
    along = poly @ direction
    lateral = poly @ lateral_axis
    laterals = np.asarray(laterals, dtype=np.float64).reshape(-1)
    result = np.full(len(laterals), -np.inf)
    count = len(poly)
    for index in range(count):
        a_along, a_lat = along[index], lateral[index]
        b_along, b_lat = along[(index + 1) % count], lateral[(index + 1) % count]
        low, high = min(a_lat, b_lat), max(a_lat, b_lat)
        mask = (laterals >= low - 1e-9) & (laterals <= high + 1e-9)
        if high - low < 1e-12:
            values = np.full(len(laterals), max(a_along, b_along))
        else:
            t = (laterals - a_lat) / (b_lat - a_lat)
            values = a_along + t * (b_along - a_along)
        result = np.where(mask, np.maximum(result, values), result)
    return result


def distance_outside_polygon(
    points_xy: np.ndarray, polygon_ccw: np.ndarray
) -> np.ndarray:
    """How far each point lies outside a convex CCW outline, 0.0 when inside.

    Measured against the edge planes, so it is exact whenever the closest
    feature of the outline is an edge and a lower bound in the wedge beyond a
    vertex.

    Against the body outline this says how far a blocking cell is from the
    chassis the robot stands on.  It does not on its own separate a real
    obstacle from the robot seeing itself: the arms reach well past the
    chassis rectangle, so a gripper the self filter missed lands tens of
    centimetres outside it.  What distinguishes them is the pair of this and
    the cell's free distance, which is negative exactly when the band's own
    outline already covers the cell.
    """

    pts = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
    poly = np.asarray(polygon_ccw, dtype=np.float64)
    if len(pts) == 0:
        return np.zeros(0, dtype=np.float64)
    if len(poly) < 3:
        return np.full(len(pts), np.inf)
    # This number decides whether a blocking cell reads as a real obstacle or
    # as a self-filter leftover, so a polygon wound the other way must not
    # silently invert it into the opposite conclusion.
    area = float(
        np.sum(poly[:, 0] * np.roll(poly[:, 1], -1) - np.roll(poly[:, 0], -1) * poly[:, 1])
    )
    if area < 0.0:
        poly = poly[::-1]
    edge = np.roll(poly, -1, axis=0) - poly
    length = np.hypot(edge[:, 0], edge[:, 1])
    length = np.where(length < 1e-12, 1.0, length)
    # Outward normal of a CCW edge (dx, dy) is (dy, -dx).
    normal = np.stack([edge[:, 1] / length, -edge[:, 0] / length], axis=1)
    offsets = (
        np.subtract.outer(pts[:, 0], poly[:, 0]) * normal[:, 0]
        + np.subtract.outer(pts[:, 1], poly[:, 1]) * normal[:, 1]
    )
    return np.maximum(0.0, np.max(offsets, axis=1))


@dataclass(frozen=True)
class FootprintLayer:
    """One height band of the robot and its convex outline in the base frame."""

    name: str
    z_min: float
    z_max: float
    polygon_xy: np.ndarray
    # The fixed term of this layer's sweep in place of the check's margin;
    # None uses the check's margin. Set for the band a vertical pad adds
    # around the arms, whose cells lie below or above the hardware.
    sweep_margin_m: float | None = None

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "FootprintLayer":
        name = str(values.get("name", "layer")).strip() or "layer"
        z_min = float(values["z_min"])
        z_max = float(values["z_max"])
        raw_margin = values.get("sweep_margin_m")
        sweep_margin = None if raw_margin is None else float(raw_margin)
        if sweep_margin is not None and not (math.isfinite(sweep_margin) and sweep_margin >= 0.0):
            raise ValueError(f"footprint layer {name!r} sweep_margin_m must be finite and nonnegative")
        polygon = np.asarray(values["polygon_xy"], dtype=np.float64)
        if polygon.ndim != 2 or polygon.shape[1] != 2 or len(polygon) < 3:
            raise ValueError(f"footprint layer {name!r} needs at least 3 xy vertices")
        if not np.all(np.isfinite(polygon)):
            raise ValueError(f"footprint layer {name!r} vertices must be finite")
        if not (math.isfinite(z_min) and math.isfinite(z_max) and z_min < z_max):
            raise ValueError(f"footprint layer {name!r} needs z_min < z_max")
        hull = convex_hull(polygon)
        if len(hull) < 3:
            raise ValueError(f"footprint layer {name!r} polygon is degenerate")
        return cls(
            name=name,
            z_min=z_min,
            z_max=z_max,
            polygon_xy=hull,
            sweep_margin_m=sweep_margin,
        )

    def extent_along(self, direction_xy: np.ndarray) -> float:
        """Signed extent of the outline along a unit direction."""

        return float(np.max(self.polygon_xy @ np.asarray(direction_xy, dtype=np.float64)))

    def swept(self, offset_xy: np.ndarray) -> np.ndarray:
        """Convex hull of the outline and the outline translated by ``offset``."""

        translated = self.polygon_xy + np.asarray(offset_xy, dtype=np.float64)
        return convex_hull(np.vstack([self.polygon_xy, translated]))


# Where the robot body physically is (chassis, 5 cm margin). Cells inside it
# are self-filter leftovers or the floor the robot stands on and are ignored;
# everything else inside a layer outline (for example the empty space between
# the two grippers) is still real space that an obstacle can occupy.
DEFAULT_BODY_POLYGON: list[list[float]] = [
    [0.22, 0.27],
    [-0.22, 0.27],
    [-0.22, -0.27],
    [0.22, -0.27],
]


@dataclass(frozen=True)
class FootprintConfig:
    """Layered footprint plus the camera position in the base frame."""

    layers: tuple[FootprintLayer, ...]
    body_polygon_xy: np.ndarray
    base_to_camera_forward_m: float = 0.2143
    base_to_camera_left_m: float = 0.0603

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | None) -> "FootprintConfig":
        values = dict(values or {})
        raw_layers = values.get("layers")
        if raw_layers is None:
            raw_layers = DEFAULT_LAYERS
        layers = tuple(FootprintLayer.from_mapping(layer) for layer in raw_layers)
        if not layers:
            raise ValueError("footprint needs at least one layer")
        body = np.asarray(values.get("body_polygon_xy") or DEFAULT_BODY_POLYGON, dtype=np.float64)
        if body.ndim != 2 or body.shape[1] != 2 or len(body) < 3 or not np.all(np.isfinite(body)):
            raise ValueError("footprint.body_polygon_xy needs at least 3 finite xy vertices")
        body = convex_hull(body)
        if len(body) < 3:
            raise ValueError("footprint.body_polygon_xy is degenerate")
        forward = float(values.get("base_to_camera_forward_m", 0.2143))
        left = float(values.get("base_to_camera_left_m", 0.0603))
        if not (math.isfinite(forward) and math.isfinite(left)):
            raise ValueError("base_to_camera offsets must be finite")
        return cls(
            layers=layers,
            body_polygon_xy=body,
            base_to_camera_forward_m=forward,
            base_to_camera_left_m=left,
        )

    @property
    def z_min(self) -> float:
        return min(layer.z_min for layer in self.layers)

    @property
    def z_max(self) -> float:
        return max(layer.z_max for layer in self.layers)

    def camera_offset_xy(self) -> np.ndarray:
        return np.asarray(
            [self.base_to_camera_forward_m, self.base_to_camera_left_m], dtype=np.float64
        )

    def layer(self, name: str) -> FootprintLayer | None:
        for layer in self.layers:
            if layer.name == name:
                return layer
        return None

    def with_layers(self, layers: Sequence[FootprintLayer]) -> "FootprintConfig":
        """The same footprint carrying a different set of height layers."""

        layers = tuple(layers)
        if not layers:
            raise ValueError("footprint needs at least one layer")
        return FootprintConfig(
            layers=layers,
            body_polygon_xy=self.body_polygon_xy,
            base_to_camera_forward_m=self.base_to_camera_forward_m,
            base_to_camera_left_m=self.base_to_camera_left_m,
        )

    def describe(self) -> dict[str, Any]:
        return {
            "layers": [
                {
                    "name": layer.name,
                    "z_min": layer.z_min,
                    "z_max": layer.z_max,
                    "polygon_xy": layer.polygon_xy.tolist(),
                    **(
                        {}
                        if layer.sweep_margin_m is None
                        else {"sweep_margin_m": layer.sweep_margin_m}
                    ),
                }
                for layer in self.layers
            ],
            "body_polygon_xy": self.body_polygon_xy.tolist(),
            "base_to_camera_forward_m": self.base_to_camera_forward_m,
            "base_to_camera_left_m": self.base_to_camera_left_m,
        }


@dataclass(frozen=True)
class CameraGeometry:
    """Depth-image intrinsics (already scaled to the depth shape) and floor plane."""

    intrinsics: np.ndarray
    camera_height_m: float
    down_camera_xyz: np.ndarray

    def planar_axes(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Unit forward, left, and down axes of the leveled view, in optical coords."""

        down = np.asarray(self.down_camera_xyz, dtype=np.float64).reshape(3)
        norm = float(np.linalg.norm(down))
        if norm <= 1e-9:
            raise ValueError("camera down vector must be nonzero")
        down = down / norm
        optical_forward = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
        forward = optical_forward - down * float(np.dot(optical_forward, down))
        forward_norm = float(np.linalg.norm(forward))
        if forward_norm <= 1e-6:
            raise ValueError("camera forward axis is degenerate")
        forward /= forward_norm
        left = np.cross(forward, down)
        left /= float(np.linalg.norm(left))
        return forward, left, down


@dataclass
class BasePoints:
    """Depth pixels converted to base-frame points (x forward, y left, z up)."""

    xyz: np.ndarray
    sampled_pixels: int
    valid_pixels: int
    valid_fraction: float
    columns: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    image_width: int = 0


def depth_to_base_points(
    depth_m: np.ndarray,
    geometry: CameraGeometry,
    footprint: FootprintConfig,
    *,
    stride: int = 2,
    min_depth_m: float = 0.05,
    max_depth_m: float = 6.0,
    validity_rows: tuple[float, float] = (0.30, 1.0),
) -> BasePoints:
    """Convert one depth image into base-frame points.

    ``valid_fraction`` is measured over the image band below the horizon
    (``validity_rows`` as fractions of the height) so that a plain wall at
    close range, which the ZED cannot measure, is reported as mostly invalid
    instead of silently producing no obstacle points.
    """

    depth = np.asarray(depth_m, dtype=np.float32)
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth.ndim != 2 or min(depth.shape) < 4:
        raise ValueError("depth image must be a 2-D array")
    stride = max(1, int(stride))
    height, width = depth.shape
    rows, columns = np.mgrid[0:height:stride, 0:width:stride]
    z = depth[rows, columns].astype(np.float64)
    finite = np.isfinite(z)
    valid = finite & (z > float(min_depth_m)) & (z <= float(max_depth_m))

    row0 = int(np.clip(validity_rows[0] * height, 0, height))
    row1 = int(np.clip(validity_rows[1] * height, 0, height))
    band = (rows >= row0) & (rows < row1)
    band_count = int(np.count_nonzero(band))
    band_valid = int(np.count_nonzero(finite & (z > float(min_depth_m)) & band))
    valid_fraction = band_valid / band_count if band_count else 0.0

    intrinsics = np.asarray(geometry.intrinsics, dtype=np.float64)
    zs = z[valid]
    cols = columns[valid].astype(np.float64)
    rws = rows[valid].astype(np.float64)
    points_camera = np.column_stack(
        [
            (cols - intrinsics[0, 2]) * zs / intrinsics[0, 0],
            (rws - intrinsics[1, 2]) * zs / intrinsics[1, 1],
            zs,
        ]
    )
    forward, left, down = geometry.planar_axes()
    offset = footprint.camera_offset_xy()
    base_x = points_camera @ forward + offset[0]
    base_y = points_camera @ left + offset[1]
    base_z = float(geometry.camera_height_m) - points_camera @ down
    xyz = np.column_stack([base_x, base_y, base_z])
    return BasePoints(
        xyz=xyz,
        sampled_pixels=int(z.size),
        valid_pixels=int(np.count_nonzero(valid)),
        valid_fraction=float(valid_fraction),
        columns=columns[valid].astype(np.int64),
        image_width=int(width),
    )


def spheres_to_base(
    spheres_camera: np.ndarray,
    geometry: CameraGeometry,
    footprint: FootprintConfig,
) -> np.ndarray:
    """Map camera-optical collision spheres into the base frame.

    Applies exactly the transform :func:`depth_to_base_points` applies to
    depth pixels, so the arms and the obstacles they are checked against are
    placed off the same floor plane and any error in it cancels.  Returns
    ``(N, 4)`` of ``[x, y, z, radius]``.
    """

    spheres = np.asarray(spheres_camera, dtype=np.float64).reshape(-1, 4)
    if not len(spheres):
        return np.zeros((0, 4), dtype=np.float64)
    forward, left, down = geometry.planar_axes()
    offset = footprint.camera_offset_xy()
    centers = spheres[:, :3]
    return np.column_stack(
        [
            centers @ forward + offset[0],
            centers @ left + offset[1],
            float(geometry.camera_height_m) - centers @ down,
            spheres[:, 3],
        ]
    )


# Normals used to circumscribe a union of discs. The result is exact along
# each normal and over-states the outline between them, always outward. The
# excess is 1/cos(pi/k) - 1 of the radius for one disc (0.9 % at 24) but grows
# with the spread of the centers: measured on this robot's arms it reaches
# about 7 mm, and on a deliberately elongated set about 42 mm.
DISC_SUPPORT_DIRECTIONS = 24


def _semi_axes(values: Any, count: int) -> np.ndarray:
    """Normalize radii or per-axis semi-axes to ``(count, 2)``."""

    axes = np.asarray(values, dtype=np.float64)
    if axes.ndim == 1:
        axes = np.column_stack([axes, axes])
    return axes.reshape(-1, 2) if len(axes.reshape(-1, 2)) == count else np.zeros((0, 2))


def ellipse_support_polygon(
    centers_xy: np.ndarray,
    semi_axes_xy: np.ndarray,
    *,
    directions: int = DISC_SUPPORT_DIRECTIONS,
) -> np.ndarray:
    """Smallest convex polygon on ``directions`` normals containing all ellipses.

    ``semi_axes_xy`` is ``(N, 2)`` of axis-aligned semi-axes, or ``(N,)`` of
    radii for circles.  Axis-aligned ellipses are what the arm envelope needs:
    the sphere model under-states the arms sideways by about 0.11 m per side
    but not forward, so one isotropic pad would push the modelled gripper past
    the hand-measured front edge and refuse approaches that used to be legal.

    The convex hull of a union of ellipses is a smooth convex body, so every
    evenly spaced support half-plane touches it and the intersection of them is
    the polygon whose vertices are consecutive half-plane crossings.  The
    caller must still verify containment: see :func:`polygon_contains_ellipses`.
    """

    centers = np.asarray(centers_xy, dtype=np.float64).reshape(-1, 2)
    axes = _semi_axes(semi_axes_xy, len(centers))
    if not len(centers) or len(axes) != len(centers):
        return np.zeros((0, 2), dtype=np.float64)
    count = max(3, int(directions))
    angles = np.arange(count, dtype=np.float64) * (2.0 * math.pi / count)
    normals = np.column_stack([np.cos(angles), np.sin(angles)])
    # Support of an axis-aligned ellipse along u is c.u + |(a ux, b uy)|.
    reach = np.sqrt((axes[:, 0:1] * normals[:, 0]) ** 2 + (axes[:, 1:2] * normals[:, 1]) ** 2)
    support = np.max(centers @ normals.T + reach, axis=0)
    following = np.roll(normals, -1, axis=0)
    support_following = np.roll(support, -1)
    determinant = normals[:, 0] * following[:, 1] - normals[:, 1] * following[:, 0]
    if np.any(np.abs(determinant) < 1e-12):
        return np.zeros((0, 2), dtype=np.float64)
    return np.column_stack(
        [
            (support * following[:, 1] - support_following * normals[:, 1]) / determinant,
            (normals[:, 0] * support_following - following[:, 0] * support) / determinant,
        ]
    )


def polygon_contains_ellipses(
    polygon_ccw: np.ndarray,
    centers_xy: np.ndarray,
    semi_axes_xy: np.ndarray,
    *,
    tolerance_m: float = 1e-9,
) -> bool:
    """True when every ellipse lies inside the convex polygon.

    Checked per edge with the outward normal, so a polygon that would let a
    part of the robot poke outside its own outline is rejected rather than
    trusted.
    """

    poly = np.asarray(polygon_ccw, dtype=np.float64).reshape(-1, 2)
    centers = np.asarray(centers_xy, dtype=np.float64).reshape(-1, 2)
    axes = _semi_axes(semi_axes_xy, len(centers))
    if len(poly) < 3 or len(axes) != len(centers):
        return False
    if not len(centers):
        return True
    following = np.roll(poly, -1, axis=0)
    edges = following - poly
    normals = np.column_stack([edges[:, 1], -edges[:, 0]])
    lengths = np.linalg.norm(normals, axis=1)
    keep = lengths > 1e-12
    if not np.any(keep):
        return False
    normals = normals[keep] / lengths[keep][:, None]
    limits = np.sum(poly[keep] * normals, axis=1)
    stretch = np.sqrt(
        (axes[:, 0:1] * normals[:, 0]) ** 2 + (axes[:, 1:2] * normals[:, 1]) ** 2
    )
    reach = centers @ normals.T + stretch
    return bool(np.all(np.max(reach, axis=0) <= limits + float(tolerance_m)))


def sphere_layers(
    spheres_base: np.ndarray,
    *,
    z_min: float,
    z_max: float,
    band_m: float,
    forward_margin_m: float,
    lateral_margin_m: float,
    z_margin_m: float = 0.0,
    always_occupied_xy: np.ndarray | None = None,
    structure_margin_m: float = 0.0,
    name: str = "arm",
) -> tuple[FootprintLayer, ...]:
    """Height-resolved outlines of a sphere model between ``z_min`` and ``z_max``.

    One layer per ``band_m`` of height, each outlining only the spheres that
    reach into that band, grown by ``margin_m``.  A single tall prism drawn
    around the gripper's forward reach charges a table top at 0.7 m against
    hardware that is nowhere near it; resolving by height charges each
    obstacle against the part of the robot that is actually at its height.

    Each sphere is grown into an axis-aligned ellipsoid before its band
    membership is decided, with a separate margin per axis.  They cover
    different errors and must not be one number.  Sideways the pad absorbs how
    much the sphere model under-states the arms, about 0.11 m per side on this
    robot.  Forward it must not: the sphere model already reaches slightly
    further than the hand measurement there, so a sideways-sized pad would
    push the modelled gripper past the outline the robot has been navigating
    with and refuse approaches that used to be legal.  Vertically it only has
    to absorb floor-plane drift, since the bands are built once per call while
    obstacles are mapped with each cycle's live plane; a large vertical pad
    would smear the gripper over a table top's height and throw away the
    resolution this function exists to provide.

    ``always_occupied_xy`` is unioned into every band for structure the sphere
    model omits (on this robot the URDF covers the arms only, not the chassis,
    the lift column or the camera head), grown by ``structure_margin_m``.
    Adjacent bands that come out with the same outline are merged, which keeps
    the per-cycle cost near the static footprint's whenever the arms occupy
    only a few bands.  Returns an empty tuple when no outline can be built or
    verified, so the caller keeps its static layers.
    """

    spheres = np.asarray(spheres_base, dtype=np.float64).reshape(-1, 4)
    low, high = float(z_min), float(z_max)
    band = float(band_m)
    forward_margin = max(0.0, float(forward_margin_m))
    lateral_margin = max(0.0, float(lateral_margin_m))
    if not (math.isfinite(low) and math.isfinite(high)) or high <= low or band <= 0.0:
        return ()
    if len(spheres) and not np.all(np.isfinite(spheres)):
        return ()
    fixed = (
        np.asarray(always_occupied_xy, dtype=np.float64).reshape(-1, 2)
        if always_occupied_xy is not None
        else np.zeros((0, 2), dtype=np.float64)
    )
    structure = max(0.0, float(structure_margin_m))
    z_margin = max(0.0, float(z_margin_m))
    count = max(1, int(math.ceil((high - low) / band - 1e-9)))
    if len(spheres):
        horizontal = np.column_stack(
            [spheres[:, 3] + forward_margin, spheres[:, 3] + lateral_margin]
        )
        vertical = np.maximum(1e-9, spheres[:, 3] + z_margin)
    else:
        horizontal = np.zeros((0, 2))
        vertical = np.zeros(0)
    outlines: list[tuple[float, float, np.ndarray]] = []
    for index in range(count):
        band_low = low + index * band
        band_high = min(high, band_low + band)
        if band_high - band_low <= 1e-9:
            continue
        centers = np.zeros((0, 2), dtype=np.float64)
        axes = np.zeros((0, 2), dtype=np.float64)
        if len(spheres):
            # Widest horizontal section the grown ellipsoid presents here.
            gap = np.maximum(
                0.0, np.maximum(band_low - spheres[:, 2], spheres[:, 2] - band_high)
            )
            ratio = gap / vertical
            reaches = ratio < 1.0
            if np.any(reaches):
                centers = spheres[reaches, :2]
                axes = horizontal[reaches] * np.sqrt(1.0 - ratio[reaches] ** 2)[:, None]
        centers = np.vstack([centers, fixed])
        axes = np.vstack([axes, np.full((len(fixed), 2), structure)])
        polygon = ellipse_support_polygon(centers, axes)
        if len(polygon) < 3:
            return ()
        hull = convex_hull(polygon)
        if len(hull) < 3 or not polygon_contains_ellipses(hull, centers, axes):
            return ()
        if outlines and np.array_equal(outlines[-1][2], hull):
            outlines[-1] = (outlines[-1][0], band_high, outlines[-1][2])
        else:
            outlines.append((band_low, band_high, hull))
    return tuple(
        FootprintLayer(
            name=f"{name}_{band_low:.2f}".replace(".", "p"),
            z_min=band_low,
            z_max=band_high,
            polygon_xy=hull,
        )
        for band_low, band_high, hull in outlines
    )


def arm_band_layers(
    spheres_base: np.ndarray,
    *,
    z_min: float,
    z_max: float,
    band_m: float,
    forward_margin_m: float,
    lateral_margin_m: float,
    z_margin_m: float,
    always_occupied_xy: np.ndarray | None = None,
    structure_margin_m: float = 0.0,
    name: str = "arm",
    core_z_margin_m: float | None = None,
    underside_forward_margin_m: float | None = None,
    underside_sweep_margin_m: float | None = None,
) -> tuple[FootprintLayer, ...]:
    """The arm layers, split by how far a cell is from the hardware's height.

    A large vertical pad keeps the arms from passing over a support surface
    just below them, but its bands then charge that surface with the full
    horizontal pads and sweep margin as well, which refuses steps that stop
    short of the surface's edge. With ``core_z_margin_m`` below
    ``z_margin_m`` the arms get two layer sets that are both checked:

    - the core set, built with ``core_z_margin_m`` and the full pads, charges
      everything at the arms' own height exactly as a single set with that
      vertical pad would;
    - the fringe set (``<name>_fringe_*``), built with the full ``z_margin_m``,
      ``underside_forward_margin_m`` (None: ``forward_margin_m``) and the
      same lateral pad, sweeps with ``underside_sweep_margin_m`` (None: the
      check's margin). A cell only the fringe reaches is below or above the
      hardware, so it blocks only when the unpadded arms would pass over it
      within the reaction and braking travel.

    A cell blocks when any layer of either set blocks, so the core set's
    stricter check still decides at the arms' height, and the always-occupied
    structure keeps its own margin in both. Without ``core_z_margin_m``, or
    when it is not below ``z_margin_m``, this is :func:`sphere_layers`
    unchanged. Returns an empty tuple when either set cannot be built, so the
    caller keeps its static layers rather than half the model.
    """

    options = dict(
        z_min=z_min,
        z_max=z_max,
        band_m=band_m,
        lateral_margin_m=lateral_margin_m,
        always_occupied_xy=always_occupied_xy,
        structure_margin_m=structure_margin_m,
    )
    if core_z_margin_m is None or float(core_z_margin_m) >= float(z_margin_m):
        return sphere_layers(
            spheres_base,
            forward_margin_m=forward_margin_m,
            z_margin_m=z_margin_m,
            name=name,
            **options,
        )
    core = sphere_layers(
        spheres_base,
        forward_margin_m=forward_margin_m,
        z_margin_m=core_z_margin_m,
        name=name,
        **options,
    )
    fringe = sphere_layers(
        spheres_base,
        forward_margin_m=(
            forward_margin_m
            if underside_forward_margin_m is None
            else underside_forward_margin_m
        ),
        z_margin_m=z_margin_m,
        name=f"{name}_fringe",
        **options,
    )
    if not core or not fringe:
        return ()
    if underside_sweep_margin_m is not None:
        fringe = tuple(
            FootprintLayer(
                name=layer.name,
                z_min=layer.z_min,
                z_max=layer.z_max,
                polygon_xy=layer.polygon_xy,
                sweep_margin_m=max(0.0, float(underside_sweep_margin_m)),
            )
            for layer in fringe
        )
    return (*core, *fringe)


# Base-frame window that is rasterized. Anything farther cannot matter for a
# direct primitive (at most 2 m of travel) and keeping the grid fixed lets the
# per-cycle work be one ``np.bincount`` per layer instead of a sort.
GRID_X_M = (-1.5, 5.5)
GRID_Y_M = (-2.5, 2.5)


LAYER_Z_TOLERANCE_M = 0.03


def arm_forward_band_range(
    spheres_base: np.ndarray,
    *,
    z_min: float,
    z_max: float,
    band_m: float,
    structure_front_m: float,
    forward_margin_m: float = 0.0,
    z_margin_m: float = 0.0,
    point_tolerance_m: float = LAYER_Z_TOLERANCE_M,
) -> tuple[float, float] | None:
    """Heights where the arms reach ahead of the structure the body stands in.

    Nav2's collision monitor cannot resolve height, so the docking split gives
    it two polygons: the structure outline, which every point is judged
    against, and the arms envelope, which only points inside this window are
    judged against.  The window must therefore cover exactly the heights where
    the arms stick out FORWARD of the structure outline, which is the only
    place the arms envelope says anything the structure outline does not.

    "Where the arms are" is the wrong question and would undo the split: in
    the travel pose the elbows occupy z 0.67-0.82 laterally while reaching no
    further forward than the chassis (measured ``front_by_band_m`` 0.27 in
    those bands), so keying on sphere presence pulls the height of a support
    surface into the arms envelope and stalls docking exactly as an unsplit
    polygon does.

    The lower edge is the measured crossing height, not the grid line below
    it, and carries no tolerance; only the upper edge is widened. Widening
    downward charges whatever lies just under the grippers to the arms
    envelope, which is the surface the robot is deliberately reaching over
    whenever it works at one.

    Each band's forward reach is the support of the same grown ellipsoids
    :func:`sphere_layers` builds, so the two agree band for band.  The window
    is the outer hull of the qualifying bands, widened by the tolerance with
    which a depth point is charged to neighbouring bands.  Returns ``None``
    when no band reaches past the structure, so the caller can fall back to
    judging the whole cloud against the arms envelope.
    """

    spheres = np.asarray(spheres_base, dtype=np.float64).reshape(-1, 4)
    low, high = float(z_min), float(z_max)
    band = float(band_m)
    front = float(structure_front_m)
    if not (math.isfinite(low) and math.isfinite(high)) or high <= low or band <= 0.0:
        return None
    if not math.isfinite(front):
        return None
    if not len(spheres) or not np.all(np.isfinite(spheres)):
        return None
    tolerance = max(0.0, float(point_tolerance_m))
    forward = np.maximum(0.0, spheres[:, 3] + max(0.0, float(forward_margin_m)))
    vertical = np.maximum(1e-9, spheres[:, 3] + max(0.0, float(z_margin_m)))
    x_center = spheres[:, 0]
    z_center = spheres[:, 2]
    count = max(1, int(math.ceil((high - low) / band - 1e-9)))
    first: int | None = None
    last: int | None = None
    for index in range(count):
        band_low = low + index * band
        band_high = min(high, band_low + band)
        if band_high - band_low <= 1e-9:
            continue
        # The widest horizontal section the grown ellipsoid presents in this
        # band, written exactly as sphere_layers writes it so a sphere sitting
        # on a band edge lands in the same band in both.
        gap = np.maximum(0.0, np.maximum(band_low - z_center, z_center - band_high))
        ratio = gap / vertical
        reaches = ratio < 1.0
        if not np.any(reaches):
            continue
        semi = forward[reaches] * np.sqrt(1.0 - ratio[reaches] ** 2)
        if float(np.max(x_center[reaches] + semi)) <= front:
            continue
        first = index if first is None else first
        last = index
    if first is None or last is None:
        return None
    # The lower edge is where the arms actually start reaching past the
    # structure, not the grid line below it. Widening downward is not the safe
    # direction: the clearance between the grippers and a support surface the
    # robot reaches over can be smaller than one band plus the tolerance, and
    # charging that surface to the arms envelope makes every pose at it look
    # like an imminent arm collision. Below the crossing a point is still
    # judged against the structure outline and against the swept-footprint
    # gate the direct primitives apply, so excluding it here loses no
    # protection. The upper edge keeps the tolerance, where no such surface
    # competes with the arms for the same cells.
    band_low = low + first * band
    band_high = min(high, band_low + band)
    steps = max(1, int(math.ceil(band / max(1e-4, band / 25.0))))
    crossing = band_low
    for step in range(steps + 1):
        height = band_low + (band_high - band_low) * step / steps
        gap = np.abs(height - z_center)
        ratio = gap / vertical
        reaches = ratio < 1.0
        if not np.any(reaches):
            continue
        semi = forward[reaches] * np.sqrt(1.0 - ratio[reaches] ** 2)
        if float(np.max(x_center[reaches] + semi)) > front:
            crossing = height
            break
    return (
        crossing,
        min(high, low + (last + 1) * band) + tolerance,
    )


def rasterize_layer(
    points_xyz: np.ndarray,
    weights: np.ndarray | None,
    layer: FootprintLayer,
    cell_m: float,
    *,
    z_tolerance_m: float = LAYER_Z_TOLERANCE_M,
) -> tuple[np.ndarray, np.ndarray]:
    """Return cell centers and summed weights for one layer (fixed grid).

    Points within ``z_tolerance_m`` of a layer boundary count for both layers,
    so floor-plane jitter cannot move an obstacle into a narrower layer.
    """

    pts = np.asarray(points_xyz, dtype=np.float64).reshape(-1, 3)
    empty = (np.zeros((0, 2), dtype=np.float64), np.zeros(0, dtype=np.float64))
    if len(pts) == 0:
        return empty
    if weights is None:
        weights = np.ones(len(pts), dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    cell = float(cell_m)
    tolerance = max(0.0, float(z_tolerance_m))
    select = (
        (pts[:, 2] >= layer.z_min - tolerance)
        & (pts[:, 2] < layer.z_max + tolerance)
        & (pts[:, 0] >= GRID_X_M[0])
        & (pts[:, 0] < GRID_X_M[1])
        & (pts[:, 1] >= GRID_Y_M[0])
        & (pts[:, 1] < GRID_Y_M[1])
    )
    if not np.any(select):
        return empty
    nx = int(math.ceil((GRID_X_M[1] - GRID_X_M[0]) / cell))
    ny = int(math.ceil((GRID_Y_M[1] - GRID_Y_M[0]) / cell))
    ix = np.floor((pts[select, 0] - GRID_X_M[0]) / cell).astype(np.int64)
    iy = np.floor((pts[select, 1] - GRID_Y_M[0]) / cell).astype(np.int64)
    ix = np.clip(ix, 0, nx - 1)
    iy = np.clip(iy, 0, ny - 1)
    flat = ix * ny + iy
    counts = np.bincount(flat, weights=weights[select], minlength=nx * ny)
    occupied = np.nonzero(counts > 0.0)[0]
    centers = np.column_stack(
        [
            (occupied // ny + 0.5) * cell + GRID_X_M[0],
            (occupied % ny + 0.5) * cell + GRID_Y_M[0],
        ]
    )
    return centers, counts[occupied]


def occupied_cells(
    points_xyz: np.ndarray,
    footprint: FootprintConfig,
    *,
    cell_m: float,
    cell_min_points: int,
    z_tolerance_m: float = LAYER_Z_TOLERANCE_M,
) -> dict[str, np.ndarray]:
    """Occupied cell centers per layer (cells with at least ``cell_min_points``)."""

    result: dict[str, np.ndarray] = {}
    for layer in footprint.layers:
        centers, counts = rasterize_layer(
            points_xyz, None, layer, cell_m, z_tolerance_m=z_tolerance_m
        )
        result[layer.name] = centers[counts >= float(cell_min_points)]
    return result


def sweep_distance(
    speed_mps: float,
    *,
    frame_age_s: float,
    command_latency_s: float,
    lease_s: float,
    brake_accel_mps2: float,
    margin_m: float,
) -> tuple[float, float, float]:
    """Return (horizon_s, brake_distance_m, sweep_distance_m) for one speed.

    The braking distance uses the Pi's S-curve profile, pi v^2 / (4 a), which
    is longer than the constant-deceleration v^2 / (2 a).
    """

    speed = max(0.0, float(speed_mps))
    horizon = max(0.0, float(frame_age_s)) + max(0.0, float(command_latency_s)) + max(
        0.0, float(lease_s)
    )
    accel = max(1e-3, float(brake_accel_mps2))
    brake = math.pi * speed * speed / (4.0 * accel)
    sweep = speed * horizon + brake + max(0.0, float(margin_m)) if speed > 1e-9 else 0.0
    return horizon, brake, sweep


@dataclass
class VisibilityResult:
    """Per-column evidence that the area about to be entered was seen free."""

    lane_bins: int
    unknown_bins: int
    seen_free_bins: int
    far_edge_m: float

    @property
    def unknown_fraction(self) -> float:
        return self.unknown_bins / self.lane_bins if self.lane_bins else 0.0

    def summary(self) -> dict[str, Any]:
        return {
            "lane_bins": self.lane_bins,
            "unknown_bins": self.unknown_bins,
            "seen_free_bins": self.seen_free_bins,
            "unknown_fraction": self.unknown_fraction,
            "far_edge_m": self.far_edge_m,
        }


def sweep_visibility(
    points: BasePoints,
    geometry: CameraGeometry,
    footprint: FootprintConfig,
    direction_xy: Sequence[float],
    sweep_m: float,
    *,
    bins: int = 32,
    ray_rows: tuple[float, ...] = (0.35, 0.5, 0.65, 0.8, 0.95),
    image_height: int | None = None,
    floor_z_max_m: float = 0.10,
    min_floor_points: int = 3,
    min_depth_m: float = 0.05,
    depth_step_m: float = 0.05,
) -> VisibilityResult:
    """Decide, per image column bin, whether the newly swept area was seen free.

    A column bin is in the *lane* when a ray through it (cast at several rows
    of the lower image) enters the area the footprint is about to occupy: the
    swept outline minus the current outline, inside a layer's height band.
    A lane bin was *seen free* when valid returns exist in it beyond the far
    edge of the sweep (floor, or an obstacle that is still farther away):
    light reached something behind the area, so the area is free along those
    rays.  Lane bins with no return beyond the sweep are unknown.  This is
    robust to the ZED's confidence filtering, which drops many floor pixels
    at random while a plain wall at close range drops all of them.
    """

    direction = np.asarray(direction_xy, dtype=np.float64)
    norm = float(np.linalg.norm(direction))
    width = int(points.image_width)
    if norm <= 1e-9 or sweep_m <= 0.0 or width <= 0:
        return VisibilityResult(0, 0, 0, 0.0)
    direction = direction / norm
    swept = [
        (layer, layer.swept(direction * float(sweep_m)), layer.polygon_xy)
        for layer in footprint.layers
    ]
    far_edge = max(float(np.max(poly @ direction)) for _, poly, _ in swept)
    intrinsics = np.asarray(geometry.intrinsics, dtype=np.float64)
    height = int(image_height) if image_height else int(round(2 * intrinsics[1, 2] + 1))
    forward, left, down = geometry.planar_axes()
    offset = footprint.camera_offset_xy()

    # Lane bins: which column bins look into the newly swept area.
    bins = max(1, int(bins))
    bin_edges = np.linspace(0.0, width, bins + 1)
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    rows = np.asarray([float(np.clip(r * height, 0, height - 1)) for r in ray_rows])
    cols_grid, rows_grid = np.meshgrid(centers, rows)
    ray_x = (cols_grid.reshape(-1) - intrinsics[0, 2]) / intrinsics[0, 0]
    ray_y = (rows_grid.reshape(-1) - intrinsics[1, 2]) / intrinsics[1, 1]
    bin_of_ray = np.tile(np.arange(bins), len(rows))
    lane = np.zeros(bins, dtype=bool)
    far_sample = far_edge + 0.3
    samples = np.arange(float(min_depth_m) + float(depth_step_m), far_sample + 1e-9, float(depth_step_m))
    for sample in samples:
        px = ray_x * sample
        py = ray_y * sample
        pz = np.full_like(px, sample)
        base_x = px * forward[0] + py * forward[1] + pz * forward[2] + offset[0]
        base_y = px * left[0] + py * left[1] + pz * left[2] + offset[1]
        base_z = float(geometry.camera_height_m) - (px * down[0] + py * down[1] + pz * down[2])
        xy = np.column_stack([base_x, base_y])
        for layer, swept_poly, now_poly in swept:
            in_band = (base_z >= layer.z_min) & (base_z < layer.z_max)
            if not np.any(in_band):
                continue
            entering = points_in_convex_polygon(xy[in_band], swept_poly) & ~points_in_convex_polygon(
                xy[in_band], now_poly
            )
            hit_bins = bin_of_ray[in_band][entering]
            lane[hit_bins] = True
    lane_bins = int(np.count_nonzero(lane))
    if lane_bins == 0:
        return VisibilityResult(0, 0, 0, far_edge)

    # Seen-free bins: returns beyond the far edge of the sweep, at floor
    # height or anywhere up to the top of the footprint.
    xyz = np.asarray(points.xyz, dtype=np.float64).reshape(-1, 3)
    columns = np.asarray(points.columns, dtype=np.int64).reshape(-1)
    seen = np.zeros(bins, dtype=bool)
    if len(xyz) and len(columns) == len(xyz):
        along = xyz[:, :2] @ direction
        beyond = (
            (xyz[:, 2] >= -float(floor_z_max_m))
            & (xyz[:, 2] <= footprint.z_max)
            & (along >= far_edge)
        )
        if np.any(beyond):
            point_bins = np.clip((columns[beyond] * bins) // max(width, 1), 0, bins - 1)
            counts = np.bincount(point_bins, minlength=bins)
            seen = counts >= int(min_floor_points)
    seen_free = int(np.count_nonzero(lane & seen))
    unknown = int(np.count_nonzero(lane & ~seen))
    return VisibilityResult(lane_bins, unknown, seen_free, far_edge)


@dataclass
class BlockingEvidence:
    """Where the cells that blocked one height band are, and where they came from.

    ``blocking_cells`` alone says a band refused the move; it cannot say
    whether a real obstacle stood there or the robot blocked itself.  These
    are the quantities that separate the two after the run is over.
    """

    count: int
    live_cells: int
    remembered_cells: int
    nearest_free_distance_m: float
    nearest_lateral_m: float
    nearest_cell_xy: tuple[float, float]
    nearest_outside_body_m: float
    min_outside_body_m: float
    sample_cells_xy: list[tuple[float, float]] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "live_cells": self.live_cells,
            "remembered_cells": self.remembered_cells,
            "nearest_free_distance_m": self.nearest_free_distance_m,
            "nearest_lateral_m": self.nearest_lateral_m,
            "nearest_cell_xy": [
                float(self.nearest_cell_xy[0]),
                float(self.nearest_cell_xy[1]),
            ],
            "nearest_outside_body_m": self.nearest_outside_body_m,
            "min_outside_body_m": self.min_outside_body_m,
            "sample_cells_xy": [[float(x), float(y)] for x, y in self.sample_cells_xy],
        }


@dataclass
class LayerClearance:
    name: str
    occupied_cells: int
    blocking_cells: int
    min_free_distance_m: float | None
    z_min: float | None = None
    z_max: float | None = None
    blocking: BlockingEvidence | None = None
    # This layer's own sweep when its margin differs from the check's.
    sweep_distance_m: float | None = None

    def summary(self) -> dict[str, Any]:
        summary: dict[str, Any] = {
            "name": self.name,
            "z_min": self.z_min,
            "z_max": self.z_max,
            "occupied_cells": self.occupied_cells,
            "blocking_cells": self.blocking_cells,
            "min_free_distance_m": self.min_free_distance_m,
        }
        if self.sweep_distance_m is not None:
            summary["sweep_distance_m"] = self.sweep_distance_m
        if self.blocking is not None:
            summary["blocking"] = self.blocking.summary()
        return summary

    def describe(self) -> str:
        """One line naming this band and what it holds, for a truncated trace."""

        band = (
            f"[{self.z_min:.2f},{self.z_max:.2f}]"
            if self.z_min is not None and self.z_max is not None
            else "[?]"
        )
        if self.blocking is None:
            return f"{self.name} {band} 0 blocking"
        evidence = self.blocking
        return (
            f"{self.name} {band} {evidence.count} cells "
            f"@{evidence.nearest_free_distance_m:.3f} m fwd, "
            f"{evidence.nearest_lateral_m:+.3f} m lat, "
            f"{evidence.nearest_outside_body_m:.3f} m outside body, "
            f"{evidence.live_cells} live/{evidence.remembered_cells} remembered"
        )


@dataclass
class ClearanceResult:
    clear: bool
    reason: str
    velocity_xy: tuple[float, float]
    speed_mps: float
    frame_age_s: float
    horizon_s: float
    brake_distance_m: float
    sweep_distance_m: float
    min_free_distance_m: float | None
    layers: list[LayerClearance] = field(default_factory=list)
    valid_fraction: float = 1.0
    memory_cells: int = 0

    def blocking_layers(self) -> list[LayerClearance]:
        """The bands that actually refused the move, nearest obstacle first."""

        blocking = [layer for layer in self.layers if layer.blocking_cells]
        return sorted(
            blocking,
            key=lambda layer: (
                layer.blocking.nearest_free_distance_m
                if layer.blocking is not None
                else math.inf
            ),
        )

    def binding_layer(self) -> LayerClearance | None:
        """The band that supplied ``min_free_distance_m``.

        With no blocking band this is just the closest thing in the corridor;
        the gate is decided by :meth:`blocking_layers`, not by this.
        """

        candidates = [
            layer for layer in self.layers if layer.min_free_distance_m is not None
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda layer: layer.min_free_distance_m)

    def describe_block(self) -> str:
        """One line naming what refused the move, short enough to never be cut.

        The structured per-layer evidence sits several levels deep in a
        primitive's metrics, which is exactly where trace summarisation used to
        replace it with a truncated repr.  This string rides at the top of the
        summary so the band is recoverable from the trace either way.
        """

        blocking = self.blocking_layers()
        if not blocking:
            binding = self.binding_layer()
            if binding is None:
                return "clear: no occupied cell in the corridor"
            return f"clear: nearest {binding.describe()}"
        head = "; ".join(layer.describe() for layer in blocking[:3])
        if len(blocking) > 3:
            head += f"; +{len(blocking) - 3} more bands"
        return head

    def summary(self) -> dict[str, Any]:
        blocking = self.blocking_layers()
        binding = self.binding_layer()
        return {
            "mode": "swept_footprint",
            "clear": self.clear,
            "reason": self.reason,
            # Named before the per-layer list so a truncated trace still says
            # which height band stopped the robot.
            "blocked_by": self.describe_block(),
            "blocking_layer_names": [layer.name for layer in blocking],
            "binding_layer_name": binding.name if binding is not None else None,
            "velocity_xy": [float(self.velocity_xy[0]), float(self.velocity_xy[1])],
            "speed_mps": self.speed_mps,
            "frame_age_s": self.frame_age_s,
            "horizon_s": self.horizon_s,
            "brake_distance_m": self.brake_distance_m,
            "sweep_distance_m": self.sweep_distance_m,
            "min_free_distance_m": self.min_free_distance_m,
            "valid_depth_fraction": self.valid_fraction,
            "memory_cells": self.memory_cells,
            "layers": [layer.summary() for layer in self.layers],
        }


def check_translation(
    points_xyz: np.ndarray,
    weights: np.ndarray | None,
    footprint: FootprintConfig,
    velocity_xy: Sequence[float],
    *,
    frame_age_s: float,
    command_latency_s: float,
    lease_s: float,
    brake_accel_mps2: float,
    margin_m: float,
    cell_m: float,
    cell_min_points: int,
    valid_fraction: float = 1.0,
    memory_cells: int = 0,
    corridor_length_m: float = 5.0,
) -> ClearanceResult:
    """Point-based convenience wrapper around :func:`check_cells`.

    ``weights`` lets remembered cells enter with the full ``cell_min_points``
    weight; live depth points carry weight one.
    """

    cells_by_layer: dict[str, np.ndarray] = {}
    for layer in footprint.layers:
        centers, counts = rasterize_layer(points_xyz, weights, layer, cell_m)
        cells_by_layer[layer.name] = centers[counts >= float(cell_min_points)]
    return check_cells(
        cells_by_layer,
        footprint,
        velocity_xy,
        frame_age_s=frame_age_s,
        command_latency_s=command_latency_s,
        lease_s=lease_s,
        brake_accel_mps2=brake_accel_mps2,
        margin_m=margin_m,
        valid_fraction=valid_fraction,
        memory_cells=memory_cells,
        corridor_length_m=corridor_length_m,
    )


# A handful of cells is enough to see where a blocking cluster sits; the trace
# summariser replaces longer lists with a bare length.
BLOCKING_SAMPLE_LIMIT = 8


def _blocking_evidence(
    occupied: np.ndarray,
    blocks: np.ndarray,
    live_count: int,
    layer: FootprintLayer,
    body_polygon_xy: np.ndarray,
    direction: np.ndarray,
    lateral_axis: np.ndarray,
) -> BlockingEvidence:
    """Describe the cells that refused one band, nearest one first."""

    indices = np.flatnonzero(blocks)
    cells = occupied[indices]
    front = polygon_support_along(layer.polygon_xy, direction, cells @ lateral_axis)
    free = cells @ direction - front
    # A blocking cell always has a finite front to measure from: the swept
    # hull is the outline translated along the direction, so it spans the same
    # lateral range, and polygon_support_along only runs out of outline
    # outside that range. Rank any non-finite value last anyway rather than
    # let a change in the sweep's shape quietly promote it to nearest.
    ranking = np.where(np.isfinite(free), free, np.inf)
    order = np.argsort(ranking)
    outside = distance_outside_polygon(cells, body_polygon_xy)
    nearest = int(order[0])
    remembered = int(np.count_nonzero(indices >= live_count))
    return BlockingEvidence(
        count=int(len(indices)),
        live_cells=int(len(indices)) - remembered,
        remembered_cells=remembered,
        nearest_free_distance_m=float(free[nearest]),
        nearest_lateral_m=float(cells[nearest] @ lateral_axis),
        nearest_cell_xy=(float(cells[nearest][0]), float(cells[nearest][1])),
        nearest_outside_body_m=float(outside[nearest]),
        min_outside_body_m=float(np.min(outside)),
        sample_cells_xy=[
            (float(cells[index][0]), float(cells[index][1]))
            for index in order[:BLOCKING_SAMPLE_LIMIT]
        ],
    )


def check_cells(
    cells_by_layer: Mapping[str, np.ndarray],
    footprint: FootprintConfig,
    velocity_xy: Sequence[float],
    *,
    frame_age_s: float,
    command_latency_s: float,
    lease_s: float,
    brake_accel_mps2: float,
    margin_m: float,
    valid_fraction: float = 1.0,
    memory_cells: int = 0,
    corridor_length_m: float = 5.0,
    live_cell_counts: Mapping[str, int] | None = None,
) -> ClearanceResult:
    """Decide whether translating at ``velocity_xy`` is clear of occupied cells.

    The footprint is swept by ``speed * horizon + speed^2 / (2 a) + margin``
    along the velocity direction.  Cells already inside the current outline
    are ignored (they hold self-filter leftovers and the place the robot is
    standing on); any occupied cell in the newly swept area blocks.
    """

    vx, vy = float(velocity_xy[0]), float(velocity_xy[1])
    speed = math.hypot(vx, vy)
    horizon, brake, sweep = sweep_distance(
        speed,
        frame_age_s=frame_age_s,
        command_latency_s=command_latency_s,
        lease_s=lease_s,
        brake_accel_mps2=brake_accel_mps2,
        margin_m=margin_m,
    )
    if speed <= 1e-9:
        return ClearanceResult(
            clear=True,
            reason=CLEAR,
            velocity_xy=(vx, vy),
            speed_mps=0.0,
            frame_age_s=float(frame_age_s),
            horizon_s=horizon,
            brake_distance_m=0.0,
            sweep_distance_m=0.0,
            min_free_distance_m=None,
            valid_fraction=valid_fraction,
            memory_cells=memory_cells,
        )
    direction = np.asarray([vx / speed, vy / speed], dtype=np.float64)
    lateral_axis = np.asarray([-direction[1], direction[0]], dtype=np.float64)
    layers: list[LayerClearance] = []
    blocked = False
    overall_min_free: float | None = None
    for layer in footprint.layers:
        occupied = np.asarray(
            cells_by_layer.get(layer.name, np.zeros((0, 2))), dtype=np.float64
        ).reshape(-1, 2)
        # Cells enter as live depth first and remembered cells second, so one
        # count is enough to say which half a blocking cell came from.
        live_count = int((live_cell_counts or {}).get(layer.name, len(occupied)))
        live_count = max(0, min(live_count, len(occupied)))
        min_free: float | None = None
        blocking = 0
        evidence: BlockingEvidence | None = None
        # The reaction and braking travel is the same for every layer; only
        # the fixed margin on top of it may be the layer's own.
        layer_sweep: float | None = None
        if layer.sweep_margin_m is not None:
            layer_sweep = speed * horizon + brake + max(0.0, float(layer.sweep_margin_m))
        if len(occupied):
            swept_polygon = layer.swept(
                direction * (sweep if layer_sweep is None else layer_sweep)
            )
            inside_sweep = points_in_convex_polygon(occupied, swept_polygon)
            # Only cells under the physical body are ignored; an obstacle in
            # the empty space inside an outline (between the grippers) is real.
            inside_body = points_in_convex_polygon(occupied, footprint.body_polygon_xy)
            blocks = inside_sweep & ~inside_body
            blocking = int(np.count_nonzero(blocks))
            corridor = layer.swept(direction * corridor_length_m)
            in_corridor = points_in_convex_polygon(occupied, corridor) & ~inside_body
            if np.any(in_corridor):
                cells = occupied[in_corridor]
                front = polygon_support_along(
                    layer.polygon_xy, direction, cells @ lateral_axis
                )
                free = cells @ direction - front
                free = free[np.isfinite(free)]
                if len(free):
                    min_free = float(np.min(free))
            if blocking:
                evidence = _blocking_evidence(
                    occupied,
                    blocks,
                    live_count,
                    layer,
                    footprint.body_polygon_xy,
                    direction,
                    lateral_axis,
                )
        if blocking:
            blocked = True
        if min_free is not None:
            overall_min_free = min_free if overall_min_free is None else min(overall_min_free, min_free)
        layers.append(
            LayerClearance(
                name=layer.name,
                occupied_cells=int(len(occupied)),
                blocking_cells=blocking,
                min_free_distance_m=min_free,
                z_min=float(layer.z_min),
                z_max=float(layer.z_max),
                blocking=evidence,
                sweep_distance_m=layer_sweep,
            )
        )
    return ClearanceResult(
        clear=not blocked,
        reason=BLOCKED if blocked else CLEAR,
        velocity_xy=(vx, vy),
        speed_mps=speed,
        frame_age_s=float(frame_age_s),
        horizon_s=horizon,
        brake_distance_m=brake,
        sweep_distance_m=sweep,
        min_free_distance_m=overall_min_free,
        layers=layers,
        valid_fraction=valid_fraction,
        memory_cells=memory_cells,
    )


def _rotation(yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.asarray([[c, -s], [s, c]], dtype=np.float64)


def base_origin_in_odom(camera_pose_xy_yaw: Sequence[float], footprint: FootprintConfig) -> np.ndarray:
    """Swerve-center position in odom given the ZED planar pose."""

    x, y, yaw = (float(v) for v in camera_pose_xy_yaw)
    return np.asarray([x, y], dtype=np.float64) - _rotation(yaw) @ footprint.camera_offset_xy()


# Fewer valid pixels than this cannot show that a cell is empty, however far
# the few there are; a cell a few metres out still projects to more.
SEEN_THROUGH_MIN_PIXELS = 12


def seen_through_cells(
    cells_xy: np.ndarray,
    layer: FootprintLayer,
    depth_m: np.ndarray,
    geometry: CameraGeometry,
    footprint: FootprintConfig,
    *,
    cell_m: float,
    margin_m: float,
    min_valid_fraction: float,
    min_pixels: int = SEEN_THROUGH_MIN_PIXELS,
    min_depth_m: float = 0.05,
    pad_m: float | None = None,
) -> np.ndarray:
    """Which base-frame cells of one layer the depth frame looks straight through.

    Each cell is the volume ``cell_m`` square in plan, grown by ``pad_m`` on
    every side (half a cell unless given), over the layer's height band. Its
    eight corners are projected into the depth image, and the cell
    counts as seen through only when every corner is in front of the camera
    and inside the image, at least ``min_valid_fraction`` of the pixels in the
    corners' bounding box carry valid depth, and every one of those depths lies
    beyond the cell's far side by ``margin_m``.

    The bounding box is wider than the cell's silhouette, so a ray that misses
    the cell but meets something nearer beside it also keeps the cell. Every
    simplification here errs toward remembering.
    """

    cells = np.asarray(cells_xy, dtype=np.float64).reshape(-1, 2)
    seen = np.zeros(len(cells), dtype=bool)
    if not len(cells):
        return seen
    depth = np.asarray(depth_m, dtype=np.float32)
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    height, width = depth.shape
    intrinsics = np.asarray(geometry.intrinsics, dtype=np.float64)
    forward, left, down = geometry.planar_axes()
    offset = footprint.camera_offset_xy()

    # Grown by half a cell by default: a remembered centre comes back off its
    # odom grid by up to a quarter cell and is read at a pose that drifts, so
    # an object standing just beside the centre it reports has to fall inside
    # the box it is judged by.
    pad = 0.5 * float(cell_m) if pad_m is None else max(0.0, float(pad_m))
    half = 0.5 * float(cell_m) + pad
    corner_xy = np.array([[-half, -half], [-half, half], [half, -half], [half, half]])
    # Each plan corner at the bottom and the top of the band: eight corners.
    xy = np.repeat(cells[:, None, :] + corner_xy[None, :, :], 2, axis=1)
    corner_z = np.tile(np.array([layer.z_min, layer.z_max], dtype=np.float64), 4)
    along = xy[..., 0] - offset[0]
    across = xy[..., 1] - offset[1]
    below = np.broadcast_to(float(geometry.camera_height_m) - corner_z[None, :], along.shape)
    # Inverse of depth_to_base_points: forward, left and down are orthonormal.
    optical = along[..., None] * forward + across[..., None] * left + below[..., None] * down
    optical_z = optical[..., 2]

    rows = np.flatnonzero(np.all(optical_z > float(min_depth_m), axis=1))
    if not len(rows):
        return seen
    z = optical_z[rows]
    u = intrinsics[0, 0] * optical[rows, :, 0] / z + intrinsics[0, 2]
    v = intrinsics[1, 1] * optical[rows, :, 1] / z + intrinsics[1, 2]
    u0 = np.floor(u.min(axis=1)).astype(np.int64)
    u1 = np.ceil(u.max(axis=1)).astype(np.int64)
    v0 = np.floor(v.min(axis=1)).astype(np.int64)
    v1 = np.ceil(v.max(axis=1)).astype(np.int64)
    far = z.max(axis=1)
    inside = (u0 >= 0) & (v0 >= 0) & (u1 <= width - 1) & (v1 <= height - 1)

    for k in np.flatnonzero(inside):
        patch = depth[v0[k] : v1[k] + 1, u0[k] : u1[k] + 1]
        valid = np.isfinite(patch) & (patch > float(min_depth_m))
        count = int(np.count_nonzero(valid))
        if count < int(min_pixels) or count < float(min_valid_fraction) * patch.size:
            continue
        if float(np.min(patch[valid])) > float(far[k]) + float(margin_m):
            seen[rows[k]] = True
    return seen


class StickyOccupancy:
    """Occupied cells seen earlier in the same primitive call, kept in odom.

    A cell is forgotten once a later frame looks straight through it (see
    :func:`seen_through_cells`); cells the camera can no longer see are kept.

    Cells are stored per layer as odom-frame cell keys so memory stays bounded;
    each remembered cell re-enters the check as already certified because it
    met the point threshold when it was observed.
    """

    _KEY_OFFSET = 1 << 20  # keeps packed odom cell keys positive

    def __init__(
        self,
        footprint: FootprintConfig,
        *,
        cell_m: float,
        cell_min_points: int,
        max_range_m: float = 4.0,
    ) -> None:
        self.footprint = footprint
        self.cell_m = float(cell_m)
        # Remembered positions are re-quantized in odom at half the base cell
        # so the round trip adds at most a quarter cell per axis.
        self.odom_cell_m = 0.5 * float(cell_m)
        self.cell_min_points = int(cell_min_points)
        self.max_range_m = float(max_range_m)
        self._keys: dict[str, np.ndarray] = {
            layer.name: np.zeros(0, dtype=np.int64) for layer in footprint.layers
        }

    def reset(self) -> None:
        for name in self._keys:
            self._keys[name] = np.zeros(0, dtype=np.int64)

    @property
    def count(self) -> int:
        return sum(len(keys) for keys in self._keys.values())

    def forget(self, masks_by_layer: Mapping[str, np.ndarray]) -> int:
        """Drop remembered cells, per layer, by their row in :meth:`cells_base`.

        Rows follow the stored key order, so a mask is only valid until the
        next :meth:`add_cells`, which re-sorts the keys.
        """

        dropped = 0
        for name, mask in masks_by_layer.items():
            keys = self._keys[name]
            mask = np.asarray(mask, dtype=bool).reshape(-1)
            if len(mask) != len(keys):
                raise ValueError(f"forget mask for {name!r} does not match its remembered cells")
            dropped += int(np.count_nonzero(mask))
            self._keys[name] = keys[~mask]
        return dropped

    def add_cells(
        self, cells_by_layer: Mapping[str, np.ndarray], camera_pose_xy_yaw: Sequence[float]
    ) -> None:
        """Remember occupied base-frame cell centers observed at ``camera_pose``."""

        yaw = float(camera_pose_xy_yaw[2])
        rotation = _rotation(yaw)
        origin = base_origin_in_odom(camera_pose_xy_yaw, self.footprint)
        for layer in self.footprint.layers:
            occupied = np.asarray(
                cells_by_layer.get(layer.name, np.zeros((0, 2))), dtype=np.float64
            ).reshape(-1, 2)
            if not len(occupied):
                continue
            near = np.hypot(occupied[:, 0], occupied[:, 1]) <= self.max_range_m
            occupied = occupied[near]
            if not len(occupied):
                continue
            odom = occupied @ rotation.T + origin
            keys = np.floor(odom / self.odom_cell_m).astype(np.int64) + self._KEY_OFFSET
            packed = keys[:, 0] * (2 * self._KEY_OFFSET) + keys[:, 1]
            self._keys[layer.name] = np.union1d(self._keys[layer.name], packed)

    def add(self, points_xyz: np.ndarray, camera_pose_xy_yaw: Sequence[float]) -> None:
        """Rasterize ``points_xyz`` and remember the occupied cells."""

        cells = occupied_cells(
            points_xyz,
            self.footprint,
            cell_m=self.cell_m,
            cell_min_points=self.cell_min_points,
        )
        self.add_cells(cells, camera_pose_xy_yaw)

    def cells_base(self, camera_pose_xy_yaw: Sequence[float]) -> dict[str, np.ndarray]:
        """Remembered cell centers per layer in the current base frame."""

        yaw = float(camera_pose_xy_yaw[2])
        rotation = _rotation(yaw)
        origin = base_origin_in_odom(camera_pose_xy_yaw, self.footprint)
        result: dict[str, np.ndarray] = {}
        for layer in self.footprint.layers:
            packed = self._keys[layer.name]
            if not len(packed):
                result[layer.name] = np.zeros((0, 2), dtype=np.float64)
                continue
            kx = packed // (2 * self._KEY_OFFSET) - self._KEY_OFFSET
            ky = packed % (2 * self._KEY_OFFSET) - self._KEY_OFFSET
            odom = (np.column_stack([kx, ky]).astype(np.float64) + 0.5) * self.odom_cell_m
            result[layer.name] = (odom - origin) @ rotation
        return result

    def points_base(self, camera_pose_xy_yaw: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
        """Remembered cells as base-frame points with certified weights."""

        chunks: list[np.ndarray] = []
        for layer in self.footprint.layers:
            base = self.cells_base(camera_pose_xy_yaw)[layer.name]
            if not len(base):
                continue
            z = np.full(len(base), 0.5 * (layer.z_min + layer.z_max))
            chunks.append(np.column_stack([base, z]))
        if not chunks:
            return np.zeros((0, 3), dtype=np.float64), np.zeros(0, dtype=np.float64)
        xyz = np.vstack(chunks)
        weights = np.full(len(xyz), float(self.cell_min_points))
        return xyz, weights
