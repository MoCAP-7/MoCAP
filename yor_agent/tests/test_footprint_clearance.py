"""Footprint-swept clearance against synthetic ZED depth images.

The scenes below render a pitched depth camera over boxes and vertical wall
segments placed in the odometry frame, so every geometric claim of the
collision-avoidance plan is checked here before any robot time is spent.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import unittest

import numpy as np

from yor_agent.robot.footprint_clearance import (
    BLOCKED,
    CLEAR,
    CameraGeometry,
    FootprintConfig,
    StickyOccupancy,
    base_origin_in_odom,
    check_cells,
    check_translation,
    convex_hull,
    depth_to_base_points,
    ellipse_support_polygon,
    occupied_cells,
    points_in_convex_polygon,
    polygon_contains_ellipses,
    arm_forward_band_range,
    sphere_layers,
    spheres_to_base,
    sweep_visibility,
)
from yor_agent.robot.navigation_controller import (
    DEPTH_UNKNOWN_IN_SWEEP_REASON,
    NavigationConfig,
    NavigationController,
    wrap_angle,
)

WIDTH, HEIGHT = 96, 64
FX = FY = 60.0
PITCH_DEG = 18.6
CAMERA_HEIGHT = 1.05


def make_geometry() -> tuple[CameraGeometry, float]:
    cx, cy = (WIDTH - 1) / 2.0, (HEIGHT - 1) / 2.0
    intrinsics = np.array([[FX, 0.0, cx], [0.0, FY, cy], [0.0, 0.0, 1.0]])
    theta = math.radians(PITCH_DEG)
    # Gravity direction expressed in optical coordinates (x right, y down, z forward).
    down = np.array([0.0, math.cos(theta), math.sin(theta)])
    return CameraGeometry(intrinsics, CAMERA_HEIGHT, down), theta


@dataclass
class Box:
    """Axis-aligned box in the odom frame."""

    xmin: float
    xmax: float
    ymin: float
    ymax: float
    zmin: float
    zmax: float


@dataclass
class Wall:
    """Vertical wall segment in the odom frame from (x1, y1) to (x2, y2)."""

    x1: float
    y1: float
    x2: float
    y2: float
    height: float = 1.5


def render_depth(
    footprint: FootprintConfig,
    *,
    base_pose: tuple[float, float, float],
    boxes: tuple[Box, ...] = (),
    walls: tuple[Wall, ...] = (),
) -> np.ndarray:
    """Depth image (optical z) of floor, boxes and walls seen from ``base_pose``."""

    geometry, theta = make_geometry()
    bx, by, yaw = base_pose
    c, s = math.cos(yaw), math.sin(yaw)
    rotation = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    offset = footprint.camera_offset_xy()
    origin = rotation @ np.array([offset[0], offset[1], CAMERA_HEIGHT]) + np.array(
        [bx, by, 0.0]
    )
    # Camera axes in the base frame (x forward, y left, z up).
    forward_b = np.array([math.cos(theta), 0.0, -math.sin(theta)])
    right_b = np.array([0.0, -1.0, 0.0])
    down_b = np.array([-math.sin(theta), 0.0, -math.cos(theta)])
    rows, cols = np.mgrid[0:HEIGHT, 0:WIDTH]
    dx = (cols - geometry.intrinsics[0, 2]) / FX
    dy = (rows - geometry.intrinsics[1, 2]) / FY
    # Unnormalized direction whose optical-z component is 1, so the ray
    # parameter t equals the ZED depth value.
    directions = (
        dx[..., None] * (rotation @ right_b)
        + dy[..., None] * (rotation @ down_b)
        + (rotation @ forward_b)
    )
    depth = np.full((HEIGHT, WIDTH), np.inf)

    def keep(t: np.ndarray) -> None:
        nonlocal depth
        valid = np.isfinite(t) & (t > 1e-6)
        depth = np.where(valid & (t < depth), t, depth)

    dz = directions[..., 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        t_floor = np.where(dz < 0.0, -origin[2] / dz, np.inf)
    keep(t_floor)
    for box in boxes:
        lo = np.array([box.xmin, box.ymin, box.zmin])
        hi = np.array([box.xmax, box.ymax, box.zmax])
        with np.errstate(divide="ignore", invalid="ignore"):
            t1 = (lo - origin) / directions
            t2 = (hi - origin) / directions
        tmin = np.max(np.minimum(t1, t2), axis=-1)
        tmax = np.min(np.maximum(t1, t2), axis=-1)
        hit = (tmax >= np.maximum(tmin, 0.0)) & (tmin > 0.0)
        keep(np.where(hit, tmin, np.inf))
    for wall in walls:
        along = np.array([wall.x2 - wall.x1, wall.y2 - wall.y1, 0.0])
        length = float(np.linalg.norm(along))
        along /= length
        normal = np.array([-along[1], along[0], 0.0])
        p0 = np.array([wall.x1, wall.y1, 0.0])
        denominator = directions @ normal
        with np.errstate(divide="ignore", invalid="ignore"):
            t = ((p0 - origin) @ normal) / denominator
        hit_point = origin + directions * t[..., None]
        s = (hit_point - p0) @ along
        inside = (
            (t > 0.0)
            & (s >= 0.0)
            & (s <= length)
            & (hit_point[..., 2] >= 0.0)
            & (hit_point[..., 2] <= wall.height)
        )
        keep(np.where(inside, t, np.inf))
    depth[~np.isfinite(depth)] = np.nan
    return depth.astype(np.float32)


def camera_pose_for_base(footprint: FootprintConfig, base_pose):
    bx, by, yaw = base_pose
    c, s = math.cos(yaw), math.sin(yaw)
    offset = footprint.camera_offset_xy()
    return (bx + c * offset[0] - s * offset[1], by + s * offset[0] + c * offset[1], yaw)


OBLIQUE_WALL = Wall(0.9, 0.9, 0.9 + 0.866 * 2.5, 0.9 - 0.5 * 2.5)  # 30 deg to +x


class GeometryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.footprint = FootprintConfig.from_mapping(None)
        self.geometry, _ = make_geometry()

    def convert(self, depth: np.ndarray):
        return depth_to_base_points(
            depth, self.geometry, self.footprint, stride=1, max_depth_m=6.0
        )

    def test_hull_and_polygon_inclusion(self) -> None:
        hull = convex_hull(np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0.5, 0.5]]))

        self.assertEqual(len(hull), 4)
        inside = points_in_convex_polygon(np.array([[0.5, 0.5], [1.5, 0.5]]), hull)
        self.assertEqual(inside.tolist(), [True, False])

    def test_default_footprint_is_the_measured_travel_pose(self) -> None:
        arms = next(layer for layer in self.footprint.layers if layer.name == "arms")

        # The collision-sphere envelope of the travel pose plus 0.05 m: 0.433 m
        # to each side and 0.457 m ahead of the swerve centre.
        self.assertAlmostEqual(arms.extent_along(np.array([0.0, 1.0])), 0.49)
        self.assertAlmostEqual(arms.extent_along(np.array([0.0, -1.0])), 0.49)
        self.assertAlmostEqual(arms.extent_along(np.array([1.0, 0.0])), 0.51)
        self.assertAlmostEqual(arms.extent_along(np.array([-1.0, 0.0])), 0.20)

    def test_depth_points_recover_a_box_in_the_base_frame(self) -> None:
        depth = render_depth(
            self.footprint,
            base_pose=(0.0, 0.0, 0.0),
            boxes=(Box(1.0, 1.1, -0.2, 0.2, 0.0, 0.5),),
        )

        points = self.convert(depth)
        box_points = points.xyz[(points.xyz[:, 2] > 0.12) & (points.xyz[:, 2] < 0.48)]

        self.assertGreater(len(box_points), 50)
        self.assertTrue(np.all(box_points[:, 0] > 0.97))
        self.assertTrue(np.all(box_points[:, 0] < 1.13))
        self.assertTrue(np.all(np.abs(box_points[:, 1]) < 0.23))
        floor = points.xyz[points.xyz[:, 2] < 0.05]
        self.assertGreater(len(floor), 500)
        self.assertGreater(points.valid_fraction, 0.9)

    def check(self, depth, velocity=(0.18, 0.0), *, frame_age=0.30, sticky=None, pose=None):
        points = self.convert(depth)
        xyz, weights = points.xyz, None
        memory_cells = 0
        if sticky is not None:
            sticky.add(points.xyz, pose)
            memory_xyz, memory_weights = sticky.points_base(pose)
            memory_cells = len(memory_xyz)
            if memory_cells:
                xyz = np.vstack([points.xyz, memory_xyz])
                weights = np.concatenate([np.ones(len(points.xyz)), memory_weights])
        return check_translation(
            xyz,
            weights,
            self.footprint,
            velocity,
            frame_age_s=frame_age,
            command_latency_s=0.10,
            lease_s=0.25,
            brake_accel_mps2=0.30,
            margin_m=0.05,
            cell_m=0.05,
            cell_min_points=3,
            valid_fraction=points.valid_fraction,
            memory_cells=memory_cells,
        )

    def test_open_floor_is_clear(self) -> None:
        depth = render_depth(self.footprint, base_pose=(0.0, 0.0, 0.0))

        result = self.check(depth)

        self.assertTrue(result.clear)
        self.assertEqual(result.reason, CLEAR)
        self.assertAlmostEqual(
            result.sweep_distance_m, 0.18 * 0.65 + math.pi * 0.18**2 / 1.2 + 0.05
        )

    def test_obstacle_between_the_grippers_is_not_ignored(self) -> None:
        # Inside the arms outline but outside the body: the chassis would hit it.
        depth = render_depth(
            self.footprint,
            base_pose=(0.0, 0.0, 0.0),
            boxes=(Box(0.34, 0.38, -0.05, 0.05, 0.0, 0.9),),
        )

        result = self.check(depth)

        self.assertFalse(result.clear)
        self.assertLess(result.min_free_distance_m, 0.2)

    def visibility(self, depth, direction=(1.0, 0.0), sweep=0.25):
        points = self.convert(depth)
        return sweep_visibility(
            points, self.geometry, self.footprint, direction, sweep, image_height=HEIGHT
        )

    def test_open_floor_is_seen_free_even_with_random_dropouts(self) -> None:
        depth = render_depth(self.footprint, base_pose=(0.0, 0.0, 0.0))
        rng = np.random.default_rng(0)
        dropped = depth.copy()
        dropped[rng.random(depth.shape) < 0.5] = np.nan  # ZED confidence filtering

        full = self.visibility(depth)
        noisy = self.visibility(dropped)

        self.assertGreater(full.lane_bins, 8)
        self.assertEqual(full.unknown_bins, 0)
        self.assertEqual(noisy.lane_bins, full.lane_bins)
        self.assertLess(noisy.unknown_fraction, 0.2)

    def test_plain_wall_at_close_range_is_unknown(self) -> None:
        # The ZED returns nothing for a textureless wall closer than its minimum
        # range: no obstacle points, no floor behind it.
        depth = render_depth(self.footprint, base_pose=(0.0, 0.0, 0.0))
        wall_rows = depth > 0.0
        near = render_depth(
            self.footprint, base_pose=(0.0, 0.0, 0.0), walls=(Wall(0.55, -2.0, 0.55, 2.0),)
        )
        depth[(near < 0.5) & wall_rows] = np.nan

        result = self.visibility(depth)

        self.assertGreater(result.lane_bins, 8)
        self.assertGreater(result.unknown_fraction, 0.9)

    def test_visible_obstacle_ahead_does_not_count_as_unknown_when_floor_shows_behind(self) -> None:
        depth = render_depth(
            self.footprint,
            base_pose=(0.0, 0.0, 0.0),
            boxes=(Box(1.4, 1.5, -0.2, 0.2, 0.0, 0.3),),
        )

        result = self.visibility(depth)

        # The box is beyond the sweep; floor is visible beside and behind it.
        self.assertLess(result.unknown_fraction, 0.3)

    def test_oblique_wall_blocks_the_elbow_while_the_center_ray_is_far(self) -> None:
        # The left elbow (0.05, 0.65) touches the 30-degree wall when the base
        # has advanced to x = 1.28; at x = 1.15 it is 0.13 m away.
        depth = render_depth(self.footprint, base_pose=(1.15, 0.0, 0.0), walls=(OBLIQUE_WALL,))

        result = self.check(depth)
        arms = next(layer for layer in result.layers if layer.name == "arms")

        self.assertFalse(result.clear)
        self.assertEqual(result.reason, BLOCKED)
        self.assertGreater(arms.blocking_cells, 0)
        self.assertIsNotNone(result.min_free_distance_m)
        self.assertLess(result.min_free_distance_m, 0.2)
        # Straight ahead of the camera the wall is still more than a metre away,
        # which is why the old central-cone gate let the elbow hit.
        camera_ray = depth[HEIGHT // 2, int(round(self.geometry.intrinsics[0, 2]))]
        self.assertGreater(float(camera_ray), 0.9)

    def test_oblique_wall_is_clear_while_still_far(self) -> None:
        depth = render_depth(self.footprint, base_pose=(0.4, 0.0, 0.0), walls=(OBLIQUE_WALL,))

        result = self.check(depth)

        self.assertTrue(result.clear)
        # The nearest part of the wall is 0.6-0.9 m ahead of the slanted
        # front-left edge; the exact value depends on cell quantization.
        self.assertGreater(result.min_free_distance_m, 0.5)

    def test_thin_pole_in_the_footprint_lane_blocks(self) -> None:
        # A 3 cm pole at y = +0.40 is outside the old central cone but inside
        # the arm layer's lane; one occupied cell is enough.
        depth = render_depth(
            self.footprint,
            base_pose=(0.1, 0.0, 0.0),
            boxes=(Box(0.75, 0.78, 0.38, 0.41, 0.0, 1.0),),
        )

        result = self.check(depth)

        self.assertFalse(result.clear)
        self.assertLess(result.min_free_distance_m, 0.15)

    def test_object_beside_the_footprint_lane_does_not_block(self) -> None:
        depth = render_depth(
            self.footprint,
            base_pose=(0.0, 0.0, 0.0),
            boxes=(Box(0.3, 0.4, 0.85, 0.95, 0.0, 0.8),),
        )

        result = self.check(depth)

        self.assertTrue(result.clear)

    def test_low_box_is_remembered_after_it_leaves_the_view(self) -> None:
        # A 0.2 m box only concerns the chassis layer (front edge 0.22 m); at
        # base x = 0.85 it is 0.13 m ahead of that edge and far below the view.
        box = Box(1.2, 1.4, -0.15, 0.15, 0.0, 0.2)
        sticky = StickyOccupancy(self.footprint, cell_m=0.05, cell_min_points=3)
        first_pose = camera_pose_for_base(self.footprint, (0.0, 0.0, 0.0))
        second_pose = camera_pose_for_base(self.footprint, (0.85, 0.0, 0.0))

        far = self.check(
            render_depth(self.footprint, base_pose=(0.0, 0.0, 0.0), boxes=(box,)),
            sticky=sticky,
            pose=first_pose,
        )
        near_depth = render_depth(self.footprint, base_pose=(0.85, 0.0, 0.0), boxes=(box,))
        near_points = self.convert(near_depth)
        near_without_memory = self.check(near_depth)
        near_with_memory = self.check(near_depth, sticky=sticky, pose=second_pose)

        self.assertTrue(far.clear)
        self.assertGreater(sticky.count, 0)
        # The box has dropped below the camera's field of view...
        self.assertFalse(np.any((near_points.xyz[:, 2] > 0.1) & (near_points.xyz[:, 0] < 1.0)))
        self.assertTrue(near_without_memory.clear)
        # ...but the cells seen from the first pose still block from the second.
        self.assertFalse(near_with_memory.clear)
        self.assertGreater(near_with_memory.memory_cells, 0)

    def test_reverse_sweep_uses_the_rear_extent(self) -> None:
        depth = render_depth(self.footprint, base_pose=(0.0, 0.0, 0.0))
        result = self.check(depth, velocity=(-0.08, 0.0))

        self.assertTrue(result.clear)
        self.assertAlmostEqual(result.speed_mps, 0.08)

    def test_base_origin_from_camera_pose(self) -> None:
        origin = base_origin_in_odom((0.2143, 0.0603, 0.0), self.footprint)

        self.assertTrue(np.allclose(origin, [0.0, 0.0]))


@dataclass
class Pose:
    x_m: float
    y_m: float
    yaw_rad: float
    valid: bool = True


@dataclass
class Frame:
    planar_pose: Pose
    depth_m: np.ndarray
    timestamp_ns: int = 2_000_000_000
    ground_camera_height_m: float | None = None
    ground_down_camera_xyz: tuple[float, float, float] | None = None
    ground_plane_timestamp_ns: int | None = None


class SceneRobot:
    """Manual base whose ZED frames are rendered from the current pose."""

    def __init__(self, footprint: FootprintConfig, *, walls=(), boxes=(), swept=True) -> None:
        self.footprint = footprint
        self.walls = tuple(walls)
        self.boxes = tuple(boxes)
        self.navigation_config = {
            "settle_cycles": 2,
            "stop_timeout_s": 0.4,
            "swept_clearance": swept,
            # Drive at the real robot's maximum so the sweep is the longest.
            "default_linear_mps": 0.18,
        }
        geometry, theta = make_geometry()
        self.manipulation_config = {
            "camera_calibration_resolution": [WIDTH, HEIGHT],
            "camera_intrinsics": geometry.intrinsics.tolist(),
            "visible_object_docking": {
                "ground_camera_height_m": CAMERA_HEIGHT,
                "ground_down_camera_xyz": geometry.down_camera_xyz.tolist(),
                "obstacle_min_height_m": 0.10,
                "obstacle_max_height_m": 1.45,
                "obstacle_max_depth_m": 20.0,
            },
        }
        self.now = 0.0
        self.base = [0.0, 0.0, 0.0]
        self.velocity = np.zeros(3)
        self.commands: list[list[float]] = []
        self.depth_override: np.ndarray | None = None

    def clock(self) -> float:
        return self.now

    def sleep(self, duration: float) -> None:
        vx, vy, omega = self.velocity
        yaw = wrap_angle(self.base[2] + omega * duration)
        c, s = math.cos(yaw), math.sin(yaw)
        self.base[0] += (vx * c - vy * s) * duration
        self.base[1] += (vx * s + vy * c) * duration
        self.base[2] = yaw
        self.now += duration

    def navigation_frame(self, *, max_age_s: float):
        del max_age_s
        depth = (
            self.depth_override
            if self.depth_override is not None
            else render_depth(
                self.footprint, base_pose=tuple(self.base), walls=self.walls, boxes=self.boxes
            )
        )
        cx, cy, yaw = camera_pose_for_base(self.footprint, tuple(self.base))
        return Frame(planar_pose=Pose(cx, cy, yaw), depth_m=depth)

    def navigation_frame_age_s(self) -> float:
        return 0.05

    def base_status(self):
        moving = bool(np.any(self.velocity))
        return {
            "lease_active": moving,
            "lease_remaining_s": 0.25 if moving else 0.0,
            "last_velocity": self.velocity.tolist(),
            "estop_latched": False,
            "limits": {"lease_s": 0.25, "max_linear_mps": 0.18, "max_yaw_rad_s": 0.35},
        }

    def submit_base_velocity(self, velocity):
        self.velocity = np.asarray(velocity, dtype=float)
        self.commands.append(self.velocity.tolist())
        return {"accepted": True}


def make_controller(robot: SceneRobot) -> NavigationController:
    return NavigationController(
        robot,
        config=NavigationConfig.from_mapping(robot.navigation_config),
        clock=robot.clock,
        sleep=robot.sleep,
    )


class ControllerIntegrationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.footprint = FootprintConfig.from_mapping(None)

    def test_drive_straight_stops_before_the_elbow_reaches_an_oblique_wall(self) -> None:
        robot = SceneRobot(self.footprint, walls=(OBLIQUE_WALL,))

        result = make_controller(robot).drive_straight(1.5, timeout_s=40.0)

        self.assertFalse(result["success"])
        self.assertEqual(result["reason"], "obstacle_too_close")
        progress = result["metrics"]["progress_m"]
        # First contact is the front-left corner of the arms outline with the
        # 30-degree wall: the wall passes y = 0.49 at x = 1.610, so the corner
        # (0.51, 0.49) touches at a progress of 1.100 m. The sweep at 0.18 m/s
        # (age 0.05 s) is 0.18*0.40 + pi*0.18^2/1.2 + 0.05 = 0.207 m.
        contact = 1.610 - 0.51
        self.assertLess(progress, contact - 0.15)
        self.assertGreater(progress, contact - 0.35)
        self.assertEqual(result["metrics"]["clearance"]["mode"], "swept_footprint")
        self.assertLess(result["metrics"]["front_clearance_m"], 0.3)
        self.assertEqual(robot.commands[-1], [0.0, 0.0, 0.0])

    def test_legacy_gate_would_have_driven_into_the_same_wall(self) -> None:
        robot = SceneRobot(self.footprint, walls=(OBLIQUE_WALL,), swept=False)

        result = make_controller(robot).drive_straight(1.5, timeout_s=40.0)

        self.assertTrue(result["success"])
        self.assertEqual(result["reason"], "target_reached")
        self.assertEqual(result["metrics"]["clearance"]["mode"], "legacy_central_depth")

    def test_mostly_invalid_depth_fails_closed(self) -> None:
        robot = SceneRobot(self.footprint)
        depth = np.full((HEIGHT, WIDTH), np.nan, dtype=np.float32)
        depth[:4, :] = 3.0
        robot.depth_override = depth

        result = make_controller(robot).drive_straight(0.5, timeout_s=10.0)

        self.assertFalse(result["success"])
        self.assertEqual(result["reason"], "RuntimeError:depth_mostly_invalid")
        self.assertEqual(robot.commands[-1], [0.0, 0.0, 0.0])

    def test_open_floor_reaches_the_target(self) -> None:
        robot = SceneRobot(self.footprint)

        result = make_controller(robot).drive_straight(1.0, timeout_s=30.0)

        self.assertTrue(result["success"])
        self.assertEqual(result["metrics"]["clearance"]["mode"], "swept_footprint")
        free = result["metrics"]["front_clearance_m"]
        self.assertTrue(free is None or free > 1.0, free)

    def test_reverse_is_not_gated_by_the_forward_depth(self) -> None:
        robot = SceneRobot(self.footprint)
        depth = np.full((HEIGHT, WIDTH), np.nan, dtype=np.float32)
        robot.depth_override = depth

        result = make_controller(robot).drive_straight(-0.15, timeout_s=10.0)

        self.assertTrue(result["success"])
        self.assertEqual(result["reason"], "target_reached")
        self.assertEqual(result["metrics"]["clearance"]["mode"], "swept_footprint")

    def test_unknown_area_ahead_fails_closed(self) -> None:
        robot = SceneRobot(self.footprint)
        depth = render_depth(self.footprint, base_pose=(0.0, 0.0, 0.0))
        # A plain panel 0.4 m wide at 0.75 m: the ZED returns nothing on it and
        # nothing is visible behind it, while the floor beside it keeps the
        # overall valid fraction well above the mostly-invalid gate.
        near = render_depth(
            self.footprint, base_pose=(0.0, 0.0, 0.0), walls=(Wall(0.75, -0.2, 0.75, 0.2),)
        )
        depth[(near < 0.7) & (depth > 0.0)] = np.nan
        robot.depth_override = depth

        result = make_controller(robot).drive_straight(0.5, timeout_s=10.0)

        self.assertFalse(result["success"])
        # The policy reads this string, so it must name the recovery rather
        # than only the condition, and it must keep its spacing.
        self.assertEqual(result["reason"], DEPTH_UNKNOWN_IN_SWEEP_REASON)
        self.assertTrue(result["reason"].startswith("depth_unknown_in_sweep:"))
        for phrase in ("turn_relative", "drive_straight", "Do not retry"):
            self.assertIn(phrase, result["reason"])
        self.assertEqual(robot.commands[-1], [0.0, 0.0, 0.0])
        debug = result["metrics"]["clearance_debug"]
        self.assertGreater(debug["valid_depth_fraction"], 0.25)
        self.assertGreater(debug["visibility"]["unknown_fraction"], 0.3)

    def test_random_depth_dropouts_do_not_stop_the_robot(self) -> None:
        robot = SceneRobot(self.footprint)
        rng = np.random.default_rng(1)

        def noisy_frame(*, max_age_s: float):
            frame = SceneRobot.navigation_frame(robot, max_age_s=max_age_s)
            depth = frame.depth_m.copy()
            depth[rng.random(depth.shape) < 0.5] = np.nan
            frame.depth_m = depth
            return frame

        robot.navigation_frame = noisy_frame  # type: ignore[method-assign]

        result = make_controller(robot).drive_straight(1.0, timeout_s=30.0)

        self.assertTrue(result["success"], result["reason"])

    def test_planar_alignment_uses_the_same_gate(self) -> None:
        robot = SceneRobot(self.footprint, boxes=(Box(0.7, 0.8, -0.2, 0.2, 0.0, 0.8),))

        result = make_controller(robot).move_planar_relative(
            0.6, 0.0, 0.0, max_linear_mps=0.12, max_lateral_mps=0.08, timeout_s=30.0
        )

        self.assertFalse(result["success"])
        self.assertEqual(result["reason"], "obstacle_too_close")
        self.assertEqual(result["metrics"]["clearance"]["mode"], "swept_footprint")

    def test_footprint_config_rejects_bad_layers(self) -> None:
        with self.assertRaises(ValueError):
            NavigationConfig.from_mapping(
                {"footprint": {"layers": [{"name": "x", "z_min": 0.5, "z_max": 0.1, "polygon_xy": [[0, 0], [1, 0], [0, 1]]}]}}
            )
        with self.assertRaises(ValueError):
            NavigationConfig.from_mapping({"min_valid_depth_fraction": 1.5})

    def test_memory_never_shortens_a_cell_the_same_frame_measured(self) -> None:
        # Remembering a cell quantizes it onto the odom grid and reads it back
        # at the cell center, displacing it by up to half a cell. Adding this
        # cycle's cells before checking them let that copy decide the verdict:
        # on 2026-09-04 three of four obstacle_too_close stops were the ghost
        # of a cell the same frame had measured 12.1 mm farther away.
        robot = SceneRobot(self.footprint, walls=(OBLIQUE_WALL,))
        robot.base = [0.55, 0.0, 0.0]
        controller = make_controller(robot)
        frame = robot.navigation_frame(max_age_s=1.0)

        without_memory = controller._swept_clearance(frame, (0.08, 0.0), None)
        first_cycle = controller._swept_clearance(frame, (0.08, 0.0), controller._new_sticky())

        assert without_memory is not None and first_cycle is not None
        self.assertIsNotNone(without_memory.min_free_distance_m)
        self.assertEqual(first_cycle.memory_cells, 0)
        self.assertAlmostEqual(
            first_cycle.min_free_distance_m,
            without_memory.min_free_distance_m,
            places=9,
        )
        self.assertEqual(first_cycle.clear, without_memory.clear)

    def test_memory_still_blocks_on_a_later_pose(self) -> None:
        # The order fix must not cost the memory its purpose: a cell seen from
        # one pose still has to block from the next one.
        box = Box(1.2, 1.4, -0.15, 0.15, 0.0, 0.2)
        robot = SceneRobot(self.footprint, boxes=(box,))
        controller = make_controller(robot)
        sticky = controller._new_sticky()

        far = controller._swept_clearance(
            robot.navigation_frame(max_age_s=1.0), (0.18, 0.0), sticky
        )
        robot.base = [0.85, 0.0, 0.0]
        near = controller._swept_clearance(
            robot.navigation_frame(max_age_s=1.0), (0.18, 0.0), sticky
        )

        assert far is not None and near is not None
        self.assertTrue(far.clear)
        self.assertGreater(near.memory_cells, 0)
        self.assertFalse(near.clear)

    def test_the_refusal_says_which_cells_came_from_the_memory(self) -> None:
        # The provenance split is read off a row index, and the only place the
        # boundary is established is _swept_clearance capturing the live count
        # before stacking the remembered cells behind them. A unit test of
        # check_cells cannot see that wiring; this drives the real path.
        box = Box(1.2, 1.4, -0.15, 0.15, 0.0, 0.2)
        robot = SceneRobot(self.footprint, boxes=(box,))
        controller = make_controller(robot)
        sticky = controller._new_sticky()

        controller._swept_clearance(
            robot.navigation_frame(max_age_s=1.0), (0.18, 0.0), sticky
        )
        robot.base = [0.85, 0.0, 0.0]
        near = controller._swept_clearance(
            robot.navigation_frame(max_age_s=1.0), (0.18, 0.0), sticky
        )

        assert near is not None
        blocking = near.blocking_layers()
        self.assertTrue(blocking)
        for layer in blocking:
            evidence = layer.blocking
            self.assertEqual(
                evidence.live_cells + evidence.remembered_cells, evidence.count
            )
            self.assertLessEqual(evidence.remembered_cells, near.memory_cells)
        # The obstacle was seen from the earlier pose, so the memory has to be
        # carrying some of what now blocks; a split that read the wrong end of
        # the stack would report every cell as freshly measured.
        self.assertGreater(
            sum(layer.blocking.remembered_cells for layer in blocking), 0
        )

    def test_the_metrics_name_the_band_beside_the_numbers(self) -> None:
        # The flat one-liner is what survives where the structured summary is
        # given up: a motion history entry's metrics sit a level deeper than
        # trace summarising descends. It has to be emitted here to be there.
        box = Box(1.2, 1.4, -0.15, 0.15, 0.0, 0.2)
        robot = SceneRobot(self.footprint, boxes=(box,))
        controller = make_controller(robot)
        sticky = controller._new_sticky()

        controller._swept_clearance(
            robot.navigation_frame(max_age_s=1.0), (0.18, 0.0), sticky
        )
        robot.base = [0.85, 0.0, 0.0]
        blocked = controller._swept_clearance(
            robot.navigation_frame(max_age_s=1.0), (0.18, 0.0), sticky
        )

        assert blocked is not None
        metrics = controller._clearance_metrics(blocked, None)

        name = blocked.blocking_layers()[0].name
        self.assertIn(name, metrics["clearance_blocked_by"])
        self.assertLess(len(metrics["clearance_blocked_by"]), 600)
        self.assertEqual(metrics["clearance"]["blocking_layer_names"][0], name)


class ArmFootprintTest(unittest.TestCase):
    """Height-resolved outlines built from the live collision spheres."""

    def setUp(self) -> None:
        self.footprint = FootprintConfig.from_mapping(None)
        self.forward = np.array([1.0, 0.0])
        # One gripper-sized cluster reaching x = 0.45 at z 0.55-0.65, which is
        # the shape that matters: below an office desk top, ahead of the body.
        self.spheres = np.array(
            [
                [0.40, 0.10, 0.60, 0.05],
                [0.40, -0.10, 0.60, 0.05],
                [0.20, 0.30, 0.80, 0.05],
            ]
        )

    def test_support_polygon_contains_every_disc(self) -> None:
        rng = np.random.default_rng(20260904)
        centers = rng.uniform(-0.6, 0.6, size=(40, 2))
        radii = rng.uniform(0.005, 0.08, size=40)
        polygon = ellipse_support_polygon(centers, radii)
        hull = convex_hull(polygon)

        angles = np.linspace(0.0, 2.0 * math.pi, 64, endpoint=False)
        rim = np.column_stack([np.cos(angles), np.sin(angles)])
        boundary = np.vstack([center + radius * rim for center, radius in zip(centers, radii)])

        self.assertGreaterEqual(len(hull), 3)
        self.assertTrue(np.all(points_in_convex_polygon(boundary, hull)))
        self.assertTrue(polygon_contains_ellipses(hull, centers, radii))
        # Conservative, and bounded. Probing along a support normal is vacuous
        # (the polygon is exact there), so sweep every direction and bound the
        # worst one. This is what a coarser normal set would blow up.
        exact = convex_hull(boundary)
        probes = np.linspace(0.0, 2.0 * math.pi, 512, endpoint=False)
        axes = np.column_stack([np.cos(probes), np.sin(probes)])
        excess = np.max(hull @ axes.T, axis=0) - np.max(exact @ axes.T, axis=0)
        self.assertGreaterEqual(float(np.min(excess)), -1e-9)
        self.assertLess(float(np.max(excess)), 0.05)

    def bands(self, **overrides) -> tuple:
        options = dict(
            z_min=0.27,
            z_max=1.45,
            band_m=0.05,
            forward_margin_m=0.0,
            lateral_margin_m=0.0,
            always_occupied_xy=self.footprint.body_polygon_xy,
        )
        options.update(overrides)
        return sphere_layers(self.spheres, **options)

    def front_of(self, bands, height: float) -> float:
        for layer in bands:
            if layer.z_min <= height < layer.z_max:
                return layer.extent_along(self.forward)
        raise AssertionError(f"no band covers z={height}")

    def test_bands_reach_forward_only_where_the_spheres_are(self) -> None:
        bands = self.bands()
        body_front = float(np.max(self.footprint.body_polygon_xy @ self.forward))

        self.assertAlmostEqual(bands[0].z_min, 0.27)
        self.assertAlmostEqual(bands[-1].z_max, 1.45)
        for before, after in zip(bands, bands[1:]):
            self.assertAlmostEqual(before.z_max, after.z_min)
        # At gripper height the outline reaches the fingers...
        self.assertGreater(self.front_of(bands, 0.60), 0.44)
        self.assertLess(self.front_of(bands, 0.60), 0.47)
        # ...at desk-top height only the upper-arm sphere is in the band, so
        # the outline stops far short of the fingers instead of extruding them.
        self.assertLess(self.front_of(bands, 0.74), 0.30)
        # ...below and above the arms it is the body alone.
        self.assertAlmostEqual(self.front_of(bands, 0.32), body_front, places=6)
        self.assertAlmostEqual(self.front_of(bands, 1.32), body_front, places=6)

    def test_adjacent_bands_with_the_same_outline_are_merged(self) -> None:
        # Most of a 1.18 m span holds no arm at all. Keeping one layer per
        # band there would multiply the per-cycle cost for nothing.
        bands = self.bands()

        self.assertLess(len(bands), 24)
        self.assertGreater(len(bands), 3)
        for before, after in zip(bands, bands[1:]):
            self.assertFalse(np.array_equal(before.polygon_xy, after.polygon_xy))

    def test_each_margin_grows_only_its_own_axis(self) -> None:
        # The three margins cover different errors and must stay separate. The
        # sideways figure is large because the sphere model under-states the
        # arms' width; applied forward it pushed the modelled gripper past the
        # static outline and refused legal approaches, and applied vertically
        # it smeared the grippers over a table top's height.
        left = np.array([0.0, 1.0])
        # Placed outside the body outline on both axes so each margin's effect
        # is visible instead of being swallowed by the chassis rectangle.
        spheres = np.array([[0.40, 0.40, 0.60, 0.05], [0.40, -0.40, 0.60, 0.05]])
        options = dict(
            z_min=0.27,
            z_max=1.45,
            band_m=0.05,
            forward_margin_m=0.0,
            lateral_margin_m=0.0,
            always_occupied_xy=self.footprint.body_polygon_xy,
        )
        plain = sphere_layers(spheres, **options)
        wide = sphere_layers(spheres, **{**options, "lateral_margin_m": 0.16})
        deep = sphere_layers(spheres, **{**options, "forward_margin_m": 0.16})
        tall = sphere_layers(spheres, **{**options, "z_margin_m": 0.10})

        def side_of(bands, height: float) -> float:
            for layer in bands:
                if layer.z_min <= height < layer.z_max:
                    return layer.extent_along(left)
            raise AssertionError("no band covers that height")

        def highest_arm_band(bands) -> float:
            body = float(np.max(self.footprint.body_polygon_xy @ self.forward))
            return max(b.z_max for b in bands if b.extent_along(self.forward) > body + 1e-6)

        # Sideways reaches wider without reaching further forward...
        self.assertGreater(side_of(wide, 0.60), side_of(plain, 0.60) + 0.15)
        self.assertAlmostEqual(self.front_of(wide, 0.60), self.front_of(plain, 0.60))
        # ...and forward is the mirror image of that.
        self.assertGreater(self.front_of(deep, 0.60), self.front_of(plain, 0.60) + 0.15)
        self.assertAlmostEqual(side_of(deep, 0.60), side_of(plain, 0.60))
        # Neither reaches one millimetre higher; only the vertical margin does.
        self.assertAlmostEqual(highest_arm_band(wide), highest_arm_band(plain))
        self.assertAlmostEqual(highest_arm_band(deep), highest_arm_band(plain))
        self.assertGreater(highest_arm_band(tall), highest_arm_band(plain))

    def test_structure_margin_grows_the_bands_that_hold_no_arm(self) -> None:
        plain = self.bands()
        padded = self.bands(structure_margin_m=0.05)

        self.assertAlmostEqual(self.front_of(plain, 1.32), 0.22, places=6)
        self.assertAlmostEqual(self.front_of(padded, 1.32), 0.27, places=6)

    def test_unusable_spheres_fall_back_to_the_static_layer(self) -> None:
        broken = np.array([[0.4, 0.0, np.nan, 0.05]])

        self.assertEqual(sphere_layers(broken, z_min=0.27, z_max=1.45, band_m=0.05, forward_margin_m=0.0, lateral_margin_m=0.0), ())
        self.assertEqual(
            sphere_layers(self.spheres, z_min=1.45, z_max=0.27, band_m=0.05, forward_margin_m=0.0, lateral_margin_m=0.0), ()
        )
        self.assertEqual(
            sphere_layers(np.zeros((0, 4)), z_min=0.27, z_max=1.45, band_m=0.05, forward_margin_m=0.0, lateral_margin_m=0.0), ()
        )

    def test_spheres_land_in_the_same_base_frame_as_depth_points(self) -> None:
        geometry, _ = make_geometry()
        _, _, down = geometry.planar_axes()
        at_camera = spheres_to_base(np.array([[0.0, 0.0, 0.0, 0.0]]), geometry, self.footprint)
        one_below = spheres_to_base(
            np.array([[*(1.0 * down), 0.0]]), geometry, self.footprint
        )

        self.assertAlmostEqual(at_camera[0, 0], self.footprint.base_to_camera_forward_m)
        self.assertAlmostEqual(at_camera[0, 1], self.footprint.base_to_camera_left_m)
        self.assertAlmostEqual(at_camera[0, 2], CAMERA_HEIGHT)
        self.assertAlmostEqual(one_below[0, 2], CAMERA_HEIGHT - 1.0)

    def test_a_desk_top_stops_blocking_while_the_gripper_still_does(self) -> None:
        static = self.footprint
        bands = sphere_layers(
            self.spheres,
            z_min=0.27,
            z_max=1.45,
            band_m=0.05,
            forward_margin_m=0.05,
            lateral_margin_m=0.16,
            always_occupied_xy=static.body_polygon_xy,
        )
        resolved = static.with_layers(
            [layer for layer in static.layers if layer.name != "arms"] + list(bands)
        )
        sweep = dict(
            frame_age_s=0.05,
            command_latency_s=0.10,
            lease_s=0.25,
            brake_accel_mps2=0.30,
            margin_m=0.05,
        )

        def verdict(config: FootprintConfig, height: float) -> bool:
            point = np.array([[0.58, 0.0, height]])
            cells = occupied_cells(point, config, cell_m=0.05, cell_min_points=1)
            return check_cells(cells, config, (0.08, 0.0), **sweep).clear

        # The static prism charges a 0.72 m desk top against the gripper...
        self.assertFalse(verdict(static, 0.72))
        self.assertTrue(verdict(resolved, 0.72))
        # ...while a real obstacle at gripper height still stops the base, and
        # the height-resolved model stops it sooner because it is honest about
        # how far the fingers reach.
        self.assertFalse(verdict(static, 0.60))
        self.assertFalse(verdict(resolved, 0.60))


class ArmFootprintWiringTest(unittest.TestCase):
    """The controller swaps in the live outline, and falls back when it cannot."""

    class StubSelfFilter:
        def __init__(self, spheres: np.ndarray, *, arms=("left", "right")) -> None:
            self._spheres = spheres
            self._arms = tuple(arms)
            self.arm_from_camera = {"left": None, "right": None}

        def camera_spheres(self) -> np.ndarray:
            return self._spheres

        def camera_sphere_arms(self) -> tuple[str, ...]:
            return self._arms

        def filtered_depth(self, depth_m: np.ndarray) -> np.ndarray:
            return depth_m

    def setUp(self) -> None:
        self.footprint = FootprintConfig.from_mapping(None)
        self.robot = SceneRobot(self.footprint)
        self.controller = make_controller(self.robot)
        self.frame = self.robot.navigation_frame(max_age_s=1.0)

    def test_static_layers_are_kept_without_a_self_filter(self) -> None:
        self.controller._refresh_arm_footprint(self.frame)

        self.assertEqual(self.controller.last_footprint_debug["source"], "static")
        self.assertEqual(
            [layer.name for layer in self.controller._footprint_config().layers],
            ["chassis", "arms"],
        )

    def test_live_spheres_replace_the_arm_layer(self) -> None:
        geometry, _ = make_geometry()
        _, _, down = geometry.planar_axes()
        # One sphere 0.35 m ahead of the camera and 0.40 m below it: inside the
        # configured arm band, so only the bands at its own height may reach
        # forward and every other band must fall back to the body outline.
        center = np.array([0.0, 0.0, 0.35]) + 0.40 * down
        self.controller._robot_self_filter = self.StubSelfFilter(
            np.array([[*center, 0.05]])
        )

        self.controller._refresh_arm_footprint(self.frame)
        debug = self.controller.last_footprint_debug
        layers = self.controller._footprint_config().layers
        fronts = debug["front_by_band_m"]
        body_front = float(np.max(self.footprint.body_polygon_xy @ np.array([1.0, 0.0])))
        structure = self.controller.config.arm_footprint_structure_margin_m

        self.assertEqual(debug["source"], "arm_spheres")
        self.assertEqual(debug["replaced_layer"], "arms")
        self.assertNotIn("arms", [layer.name for layer in layers])
        self.assertEqual(layers[0].name, "chassis")
        self.assertGreater(len(layers), 2)
        # One band holds the sphere and reaches forward; the rest hold no
        # hardware at all and shrink to the padded body outline.
        at_body = [v for v in fronts.values() if abs(v - body_front - structure) < 1e-6]
        self.assertGreater(max(fronts.values()), body_front + structure)
        self.assertGreaterEqual(len(at_body), 2)

    def test_an_attached_payload_keeps_the_static_layers(self) -> None:
        # A grasped object is masked out of the depth image by the self filter
        # but is not in the sphere envelope, so a live outline would model it
        # nowhere. The static prism at least covered the space between the
        # grippers.
        self.controller._robot_self_filter = self.StubSelfFilter(
            np.array([[0.0, 0.0, 0.35, 0.05]])
        )
        self.robot.robot_self_filter_attached_objects = lambda: {
            "right": {"size_xyz": [0.2, 0.2, 0.2]}
        }

        self.controller._refresh_arm_footprint(self.frame)

        self.assertEqual(self.controller.last_footprint_debug["reason"], "attached_object")
        self.assertEqual(
            [layer.name for layer in self.controller._footprint_config().layers],
            ["chassis", "arms"],
        )

    def test_a_partial_arm_envelope_keeps_the_static_layers(self) -> None:
        # require_both_arms false lets the self filter report one arm; half a
        # collision model of a 1.2 m wide robot is worse than none.
        self.controller._robot_self_filter = self.StubSelfFilter(
            np.array([[0.0, 0.0, 0.35, 0.05]]), arms=("right",)
        )

        self.controller._refresh_arm_footprint(self.frame)

        self.assertEqual(
            self.controller.last_footprint_debug["reason"], "partial_arm_status"
        )
        self.assertEqual(
            [layer.name for layer in self.controller._footprint_config().layers],
            ["chassis", "arms"],
        )

    def test_disabling_the_option_keeps_the_static_layers(self) -> None:
        self.robot.navigation_config["arm_footprint_from_spheres"] = False
        controller = make_controller(self.robot)
        controller._robot_self_filter = self.StubSelfFilter(
            np.array([[0.0, 0.0, 0.35, 0.05]])
        )

        controller._refresh_arm_footprint(self.frame)

        self.assertEqual(controller.last_footprint_debug["source"], "static")
        self.assertEqual(
            [layer.name for layer in controller._footprint_config().layers],
            ["chassis", "arms"],
        )


    def test_a_core_vertical_pad_adds_fringe_layers_with_their_own_margin(self) -> None:
        self.robot.navigation_config.update(
            {
                "arm_footprint_z_margin_m": 0.10,
                "arm_footprint_core_z_margin_m": 0.01,
                "arm_footprint_underside_forward_margin_m": 0.0,
                "arm_footprint_underside_clearance_margin_m": 0.01,
            }
        )
        controller = make_controller(self.robot)
        geometry, _ = make_geometry()
        _, _, down = geometry.planar_axes()
        center = np.array([0.0, 0.0, 0.35]) + 0.40 * down
        controller._robot_self_filter = self.StubSelfFilter(np.array([[*center, 0.05]]))

        controller._refresh_arm_footprint(self.frame)
        names = [layer.name for layer in controller._footprint_config().layers]
        layers = controller._footprint_config().layers
        fringe = [layer for layer in layers if layer.name.startswith("arms_fringe_")]
        core = [
            layer
            for layer in layers
            if layer.name.startswith("arms_") and not layer.name.startswith("arms_fringe_")
        ]
        debug = controller.last_footprint_debug

        self.assertEqual(debug["source"], "arm_spheres")
        self.assertEqual(names[0], "chassis")
        self.assertTrue(core)
        self.assertTrue(fringe)
        self.assertTrue(all(layer.sweep_margin_m is None for layer in core))
        self.assertTrue(all(layer.sweep_margin_m == 0.01 for layer in fringe))
        self.assertEqual(debug["underside_clearance_margin_m"], 0.01)
        self.assertEqual(debug["underside_forward_margin_m"], 0.0)
        self.assertIn("fringe_front_by_band_m", debug)


class UndersideMarginTest(unittest.TestCase):
    """A support surface the arms pass above versus hardware at their height.

    One sphere per arm with its underside at 0.84 m and its front at 0.46 m,
    a table-top slab at 0.77 m whose near edge is 0.115 m ahead of that
    front, and a forward command at 0.10 m/s: the reaction and braking travel
    alone is about 0.064 m.
    """

    BODY = np.array([[0.22, 0.27], [-0.22, 0.27], [-0.22, -0.27], [0.22, -0.27]])
    SPLIT = dict(
        core_z_margin_m=0.01,
        underside_forward_margin_m=0.0,
        underside_sweep_margin_m=0.01,
    )

    def spheres(self, forward_m: float = 0.40) -> np.ndarray:
        return np.array(
            [[forward_m, 0.20, 0.90, 0.06], [forward_m, -0.20, 0.90, 0.06]]
        )

    def layers(self, spheres: np.ndarray, **overrides) -> tuple:
        from yor_agent.robot.footprint_clearance import arm_band_layers

        options = dict(
            z_min=0.27,
            z_max=1.45,
            band_m=0.05,
            forward_margin_m=0.05,
            lateral_margin_m=0.05,
            z_margin_m=0.10,
            always_occupied_xy=self.BODY,
            structure_margin_m=0.05,
            name="arms",
        )
        options.update(overrides)
        return arm_band_layers(spheres, **options)

    def check(self, layers: tuple, points: np.ndarray):
        from yor_agent.robot.footprint_clearance import (
            FootprintConfig,
            check_cells,
            occupied_cells,
        )

        footprint = FootprintConfig.from_mapping(
            {
                "layers": [
                    {
                        "name": "chassis",
                        "z_min": 0.10,
                        "z_max": 0.27,
                        "polygon_xy": self.BODY.tolist(),
                    }
                ],
                "body_polygon_xy": self.BODY.tolist(),
            }
        )
        footprint = footprint.with_layers([footprint.layers[0], *layers])
        cells = occupied_cells(
            points, footprint, cell_m=0.05, cell_min_points=1, z_tolerance_m=0.03
        )
        return check_cells(
            cells,
            footprint,
            (0.10, 0.0),
            frame_age_s=0.025,
            command_latency_s=0.10,
            lease_s=0.25,
            brake_accel_mps2=0.30,
            margin_m=0.05,
        )

    @staticmethod
    def slab(z_m: float, x_range=(0.58, 1.0), y_range=(-0.5, 0.5)) -> np.ndarray:
        xs = np.arange(x_range[0], x_range[1], 0.01)
        ys = np.arange(y_range[0], y_range[1], 0.01)
        grid = np.stack(np.meshgrid(xs, ys, indexing="ij"), axis=-1).reshape(-1, 2)
        return np.column_stack([grid, np.full(len(grid), z_m)])

    def test_without_a_core_pad_the_layers_are_the_single_set(self) -> None:
        from yor_agent.robot.footprint_clearance import sphere_layers

        single = sphere_layers(
            self.spheres(),
            z_min=0.27,
            z_max=1.45,
            band_m=0.05,
            forward_margin_m=0.05,
            lateral_margin_m=0.05,
            z_margin_m=0.10,
            always_occupied_xy=self.BODY,
            structure_margin_m=0.05,
            name="arms",
        )
        unsplit = self.layers(self.spheres())

        self.assertEqual([layer.name for layer in single], [layer.name for layer in unsplit])
        for expected, actual in zip(single, unsplit):
            np.testing.assert_allclose(expected.polygon_xy, actual.polygon_xy)
            self.assertIsNone(actual.sweep_margin_m)

    def test_a_step_that_stops_short_of_a_surface_below_the_arms_is_clear(self) -> None:
        table = self.slab(0.77)

        single = self.check(self.layers(self.spheres()), table)
        split = self.check(self.layers(self.spheres(), **self.SPLIT), table)

        # One padded set charges the table top with the full pads and refuses.
        self.assertFalse(single.clear)
        self.assertTrue(split.clear)

    def test_hardware_height_obstacles_keep_the_full_pads(self) -> None:
        box = self.slab(0.90, x_range=(0.58, 0.63), y_range=(-0.05, 0.05))

        result = self.check(self.layers(self.spheres(), **self.SPLIT), box)

        self.assertFalse(result.clear)
        blocking = [layer.name for layer in result.layers if layer.blocking_cells]
        self.assertTrue(any(not name.startswith("arms_fringe_") for name in blocking))

    def test_the_arms_still_may_not_pass_over_the_surface(self) -> None:
        table = self.slab(0.77)
        # The arms' unpadded front now sits 0.02 m short of the edge, inside
        # the reaction and braking travel.
        result = self.check(self.layers(self.spheres(0.50), **self.SPLIT), table)

        self.assertFalse(result.clear)
        blocking = [layer for layer in result.layers if layer.blocking_cells]
        self.assertTrue(blocking)
        self.assertTrue(all(layer.name.startswith("arms_fringe_") for layer in blocking))
        self.assertTrue(all(layer.sweep_distance_m is not None for layer in blocking))

    def test_underside_settings_may_only_relax_their_full_counterparts(self) -> None:
        NavigationConfig.from_mapping(
            {
                "arm_footprint_z_margin_m": 0.10,
                "arm_footprint_core_z_margin_m": 0.01,
                "arm_footprint_underside_forward_margin_m": 0.0,
                "arm_footprint_underside_clearance_margin_m": 0.01,
            }
        )
        for settings in (
            {"arm_footprint_core_z_margin_m": 0.2},
            {"arm_footprint_underside_forward_margin_m": 0.06},
            {"arm_footprint_underside_clearance_margin_m": -0.01},
            {"arm_footprint_underside_clearance_margin_m": "0.01"},
        ):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                NavigationConfig.from_mapping(settings)


class ArmForwardBandRangeTest(unittest.TestCase):
    """The height window Nav2's arms collision monitor is allowed to see.

    Nav2 cannot resolve height, so the ZED bridge sends its arms monitor only
    the points inside this window and every point is judged against the
    structure outline anyway. The window must therefore cover the heights
    where the arms reach FORWARD of that outline, not merely the heights the
    arms occupy: in the travel pose the elbows fill z 0.67-0.82 sideways while
    reaching no further forward than the chassis, and keying on occupancy
    would pull office-desk height into the arms envelope and re-create the
    stall this split exists to remove.
    """

    STRUCTURE_FRONT_M = 0.27

    def setUp(self) -> None:
        self.options = dict(z_min=0.27, z_max=1.45, band_m=0.05)
        self.body = np.array(
            [[0.22, 0.27], [-0.22, 0.27], [-0.22, -0.27], [0.22, -0.27]]
        )
        # The measured travel pose (trace 20260907-144711 front_by_band_m
        # {'0.67': 0.27, '0.72': 0.27, '0.77': 0.27, '0.82': 0.432,
        # '0.87': 0.4814, '0.92': 0.456, '0.97': 0.27}): elbows wide but not
        # forward under the grippers, grippers forward at 0.82-0.97, and
        # shoulders above them that are wide again.
        self.spheres = np.array(
            [
                [0.05, 0.46, 0.70, 0.04],
                [0.05, -0.46, 0.70, 0.04],
                [0.05, 0.44, 0.78, 0.04],
                [0.05, -0.44, 0.78, 0.04],
                [0.38, 0.10, 0.87, 0.03],
                [0.38, -0.10, 0.87, 0.03],
                [0.30, 0.20, 0.94, 0.03],
                [0.02, 0.30, 1.15, 0.04],
            ]
        )

    def band(self, spheres=None, **overrides):
        options = dict(
            self.options,
            structure_front_m=self.STRUCTURE_FRONT_M,
            forward_margin_m=0.05,
            z_margin_m=0.01,
            point_tolerance_m=0.03,
        )
        options.update(overrides)
        source = self.spheres if spheres is None else spheres
        return arm_forward_band_range(source, **options)

    def test_window_matches_the_bands_that_reach_past_the_structure(self) -> None:
        band = self.band(point_tolerance_m=0.0)
        layers = sphere_layers(
            self.spheres,
            **self.options,
            forward_margin_m=0.05,
            lateral_margin_m=0.05,
            z_margin_m=0.01,
            always_occupied_xy=self.body,
            structure_margin_m=0.05,
        )
        forward = np.array([1.0, 0.0])
        reaching = [
            layer
            for layer in layers
            if layer.extent_along(forward) > self.STRUCTURE_FRONT_M + 1e-9
        ]

        self.assertIsNotNone(band)
        self.assertTrue(reaching)
        first = min(l.z_min for l in reaching)
        # The upper edge is the grid line, as the layers draw it. The lower
        # edge is the measured crossing instead, so it sits inside the first
        # reaching layer rather than on its floor: everything below it reaches
        # no further forward than the structure and belongs to the structure
        # outline.
        self.assertAlmostEqual(band[1], max(l.z_max for l in reaching), places=6)
        self.assertGreaterEqual(band[0], first)
        self.assertLess(band[0], first + self.options["band_m"])
        below = self.band(point_tolerance_m=0.0, z_max=band[0] - 1e-6)
        self.assertIsNone(below, "a band below the crossing still reached forward")

    def test_elbow_height_stays_with_the_structure_outline(self) -> None:
        low, high = self.band()

        # The desk top, its depth noise, and the elbow band it shares are all
        # judged against the structure outline, which is what let
        # drive_straight advance where Nav2's flat rectangle stopped.
        for desk_z in (0.70, 0.74, 0.78):
            self.assertLess(
                desk_z, low, f"desk point at {desk_z} entered the arms band"
            )
        # The grippers, which do reach past the structure, stay inside it.
        for arm_z in (0.85, 0.94):
            self.assertTrue(low <= arm_z <= high)
        # The shoulders above them are wide, not forward, so they leave again.
        self.assertLess(high, 1.15)

    def test_the_occupancy_predicate_would_have_swallowed_the_desk(self) -> None:
        # Guard against regressing to "any sphere in this band": the elbows
        # would then open the window at 0.64 m and the desk would be charged
        # against the arms envelope again.
        occupied = self.band(structure_front_m=-10.0)
        forward = self.band()

        self.assertLess(occupied[0], 0.70)
        self.assertGreater(forward[0], 0.78)

    def test_no_forward_reach_or_invalid_input_falls_back(self) -> None:
        self.assertIsNone(arm_forward_band_range(np.zeros((0, 4)), **self.options,
                                                 structure_front_m=0.27))
        self.assertIsNone(
            arm_forward_band_range(
                np.array([[0.0, 0.0, np.nan, 0.05]]),
                **self.options,
                structure_front_m=0.27,
            )
        )
        # Arms tucked entirely inside the structure outline select no band.
        self.assertIsNone(self.band(spheres=self.spheres[:4]))

    def test_window_follows_the_arms(self) -> None:
        raised = self.spheres.copy()
        raised[:, 2] += 0.30
        low, high = self.band(spheres=raised, point_tolerance_m=0.0)
        base_low, base_high = self.band(point_tolerance_m=0.0)

        self.assertGreater(low, base_low)
        self.assertGreater(high, base_high)


if __name__ == "__main__":
    unittest.main()
