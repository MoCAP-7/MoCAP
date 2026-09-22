"""Operator-facing Viser scene for visible-object docking diagnostics."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..geometry import matrix_to_quaternion_wxyz


def _point_cloud_in_navigation_frame(
    depth_m: np.ndarray,
    rgb: np.ndarray,
    intrinsics: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    max_points: int = 100_000,
    max_depth_m: float = 20.0,
    ground_camera_height_m: float | None = None,
    ground_down_camera_xyz: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Back-project RGB-D into the docking frame: +X forward, +Y left, +Z up."""

    depth = np.asarray(depth_m, dtype=np.float64)
    color = np.asarray(rgb, dtype=np.uint8)
    camera = np.asarray(intrinsics, dtype=np.float64)
    if depth.shape != color.shape[:2]:
        raise ValueError("depth and RGB shapes do not match")
    if camera.shape != (3, 3) or not np.all(np.isfinite(camera)):
        raise ValueError("intrinsics must be a finite 3x3 matrix")
    if not math.isfinite(max_depth_m) or max_depth_m <= 0.08:
        raise ValueError("max_depth_m must be finite and greater than 0.08")
    valid = np.isfinite(depth) & (depth > 0.08) & (depth <= max_depth_m)
    if mask is not None:
        selected = np.asarray(mask, dtype=bool)
        if selected.shape != depth.shape:
            raise ValueError("mask and depth shapes do not match")
        valid &= selected
    rows, columns = np.nonzero(valid)
    if rows.size == 0:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8)
    if rows.size > max_points:
        keep = np.linspace(0, rows.size - 1, max_points, dtype=np.int64)
        rows = rows[keep]
        columns = columns[keep]
    optical_depth = depth[rows, columns]
    camera_right = (
        (columns - camera[0, 2]) * optical_depth / camera[0, 0]
    )
    camera_down = (
        (rows - camera[1, 2]) * optical_depth / camera[1, 1]
    )
    points_camera = np.column_stack(
        [camera_right, camera_down, optical_depth]
    )
    if ground_camera_height_m is None and ground_down_camera_xyz is None:
        points = np.column_stack(
            [optical_depth, -camera_right, -camera_down]
        )
    elif ground_camera_height_m is None or ground_down_camera_xyz is None:
        raise ValueError("ground height and down vector must be supplied together")
    else:
        down = np.asarray(ground_down_camera_xyz, dtype=np.float64).reshape(-1)
        if down.shape != (3,) or not np.all(np.isfinite(down)):
            raise ValueError("ground down vector must contain three finite values")
        down_norm = float(np.linalg.norm(down))
        if down_norm <= 1e-6 or not math.isfinite(ground_camera_height_m):
            raise ValueError("ground calibration must be finite and nonzero")
        down /= down_norm
        optical_forward = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
        planar_forward = optical_forward - down * float(
            np.dot(optical_forward, down)
        )
        planar_forward /= np.linalg.norm(planar_forward)
        planar_left = np.cross(planar_forward, down)
        planar_left /= np.linalg.norm(planar_left)
        points = np.column_stack(
            [
                points_camera @ planar_forward,
                points_camera @ planar_left,
                float(ground_camera_height_m) - points_camera @ down,
            ]
        )
    return points.astype(np.float32), color[rows, columns]


def _rounded_bbox_boundary(
    forward_min_m: float,
    forward_max_m: float,
    left_min_m: float,
    left_max_m: float,
    radius_m: float,
    *,
    samples_per_corner: int = 12,
) -> np.ndarray:
    """Return the rounded boundary of a 2D bbox inflated by ``radius_m``."""

    values = np.asarray(
        [forward_min_m, forward_max_m, left_min_m, left_max_m, radius_m],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(values)):
        raise ValueError("bbox boundary values must be finite")
    if forward_min_m > forward_max_m or left_min_m > left_max_m:
        raise ValueError("bbox bounds are inverted")
    if radius_m <= 0.0 or samples_per_corner < 2:
        raise ValueError("radius and corner samples must be positive")
    corners = (
        (forward_max_m, left_max_m, 0.0, math.pi / 2.0),
        (forward_min_m, left_max_m, math.pi / 2.0, math.pi),
        (forward_min_m, left_min_m, math.pi, 3.0 * math.pi / 2.0),
        (forward_max_m, left_min_m, 3.0 * math.pi / 2.0, 2.0 * math.pi),
    )
    boundary = []
    for forward, left, angle0, angle1 in corners:
        angles = np.linspace(
            angle0, angle1, samples_per_corner, endpoint=False, dtype=np.float64
        )
        boundary.append(
            np.column_stack(
                [
                    forward + radius_m * np.cos(angles),
                    left + radius_m * np.sin(angles),
                ]
            )
        )
    return np.ascontiguousarray(np.concatenate(boundary, axis=0))


def _world_xy_in_navigation_frame(
    world_xy: np.ndarray, reference_pose_xy_yaw: np.ndarray
) -> np.ndarray:
    """Express world XY samples in one ZED pose's forward/left frame."""

    points = np.asarray(world_xy, dtype=np.float64)
    reference = np.asarray(reference_pose_xy_yaw, dtype=np.float64).reshape(-1)
    if points.ndim != 2 or points.shape[1:] != (2,):
        raise ValueError("world_xy must have shape (N, 2)")
    if reference.shape != (3,) or not np.all(np.isfinite(reference)):
        raise ValueError("reference pose must contain three finite values")
    if not np.all(np.isfinite(points)):
        raise ValueError("world_xy must be finite")
    delta = points - reference[:2]
    cosine = math.cos(float(reference[2]))
    sine = math.sin(float(reference[2]))
    return np.column_stack(
        [
            cosine * delta[:, 0] + sine * delta[:, 1],
            -sine * delta[:, 0] + cosine * delta[:, 1],
        ]
    )


class NavigationViser:
    """Serve live docking geometry in the primitive's planar control frame."""

    def __init__(self, *, host: str = "0.0.0.0", port: int = 8080) -> None:
        try:
            import viser
        except ImportError as exc:
            raise RuntimeError(
                "Viser is not installed in the Jetson test environment. Install "
                "it with: python -m pip install viser"
            ) from exc
        self._server = viser.ViserServer(host=host, port=int(port))
        self.host = host
        get_port = getattr(self._server, "get_port", None)
        self.port = int(get_port()) if callable(get_port) else int(port)
        self._server.scene.set_up_direction("+z")
        self._summary_handle: Any | None = None
        self._actual_world_xy: list[np.ndarray] = []
        self._scene_reference_pose: np.ndarray | None = None
        self._server.gui.add_markdown(
            """
## YOR visible-object docking

- Scene axes: **+X forward**, **+Y left**, **+Z up**
- Magenta points: SAM3 target mask
- Light-gray points: ignored floor band below the obstacle-height threshold
- Orange rectangle: target planar bbox
- Cyan rounded line: bbox inflated by the configured docking threshold
- Green area: checked and clear straight corridor
- Red area: checked and blocked straight corridor
- Gray area: geometric preview only; path not checked before turning
- Yellow marker: target-bbox point closest to the ZED/front edge
- Green marker: next desired camera/front position
- Red points: temporally stable, above-floor cluster that blocks the path
- Blue line and marker: selected A* or known-free frontier path and waypoint
- Yellow line and marker: actual ZED pose trajectory

The grid follows the current ZED SDK floor-plane estimate. Collision checking
uses the same timestamped camera height and ground normal. Viser is diagnostic
only and does not issue motion commands.
"""
        )

    def _append_actual_pose(self, pose_xy_yaw: Any) -> None:
        pose = np.asarray(pose_xy_yaw, dtype=np.float64).reshape(-1)
        if pose.shape != (3,) or not np.all(np.isfinite(pose)):
            return
        point = pose[:2].copy()
        if not self._actual_world_xy or np.linalg.norm(
            point - self._actual_world_xy[-1]
        ) >= 0.002:
            self._actual_world_xy.append(point)

    def _render_actual_trajectory(self) -> None:
        reference = self._scene_reference_pose
        if reference is None or not self._actual_world_xy:
            return
        world = np.stack(self._actual_world_xy, axis=0)
        local = _world_xy_in_navigation_frame(world, reference)
        points = np.column_stack(
            [local, np.full(local.shape[0], 0.15, dtype=np.float64)]
        )
        name = "/docking/actual_trajectory"
        self._server.scene.remove_by_name(name)
        if points.shape[0] >= 2:
            self._server.scene.add_line_segments(
                name=name,
                points=np.stack([points[:-1], points[1:]], axis=1),
                colors=np.asarray([255, 205, 35], dtype=np.uint8),
                line_width=6.0,
            )
        marker_name = "/docking/actual_position"
        self._server.scene.remove_by_name(marker_name)
        self._server.scene.add_icosphere(
            name=marker_name,
            radius=0.05,
            color=np.asarray([255, 205, 35], dtype=np.uint8),
            position=points[-1],
        )

    def publish_actual_pose(self, pose_xy_yaw: list[float]) -> None:
        """Append one live ZED pose and refresh the actual path overlay."""

        self._append_actual_pose(pose_xy_yaw)
        self._render_actual_trajectory()

    def publish(self, debug: dict[str, Any]) -> None:
        root = "/docking"
        self._server.scene.remove_by_name(root)
        if int(debug.get("iteration", -1)) == 0:
            self._actual_world_xy.clear()
        pose = np.asarray(debug.get("pose_xy_yaw", []), dtype=np.float64).reshape(-1)
        if pose.shape == (3,) and np.all(np.isfinite(pose)):
            self._scene_reference_pose = pose.copy()
            self._append_actual_pose(pose)
        rgb = np.asarray(debug["rgb"], dtype=np.uint8)
        depth = np.asarray(debug["depth_m"], dtype=np.float64)
        intrinsics = np.asarray(debug["intrinsics"], dtype=np.float64)
        target = debug["target"]
        config = debug["config"]
        ground_plane = debug.get("ground_plane", {})
        path_status = debug.get("path_status")
        avoidance_plan_reason = debug.get("avoidance_plan_reason")
        travel_distance = max(0.0, float(debug["travel_distance_m"]))

        ground_height = float(
            ground_plane.get(
                "camera_height_m", config.ground_camera_height_m
            )
        )
        ground_down = np.asarray(
            ground_plane.get(
                "down_camera_xyz", config.ground_down_camera_xyz
            ),
            dtype=np.float64,
        )
        points, colors = _point_cloud_in_navigation_frame(
            depth,
            rgb,
            intrinsics,
            max_depth_m=float(config.target_max_depth_m),
            ground_camera_height_m=ground_height,
            ground_down_camera_xyz=ground_down,
        )
        ignored_floor = points[:, 2] < float(config.obstacle_min_height_m)
        self._server.scene.add_point_cloud(
            name=f"{root}/zed_rgbd",
            points=points[~ignored_floor],
            colors=colors[~ignored_floor],
            point_size=0.006,
            point_shape="circle",
        )
        ignored_points = points[ignored_floor]
        self._server.scene.add_point_cloud(
            name=f"{root}/ignored_floor_band",
            points=ignored_points,
            colors=np.tile(
                np.asarray([[205, 205, 205]], dtype=np.uint8),
                (ignored_points.shape[0], 1),
            ),
            point_size=0.003,
            point_shape="circle",
        )
        target_points, _ = _point_cloud_in_navigation_frame(
            depth,
            rgb,
            intrinsics,
            mask=target.mask,
            max_points=30_000,
            max_depth_m=float(config.target_max_depth_m),
            ground_camera_height_m=ground_height,
            ground_down_camera_xyz=ground_down,
        )
        self._server.scene.add_point_cloud(
            name=f"{root}/target_mask",
            points=target_points,
            colors=np.tile(
                np.asarray([[255, 70, 230]], dtype=np.uint8),
                (target_points.shape[0], 1),
            ),
            point_size=0.010,
            point_shape="circle",
        )

        self._server.scene.add_grid(
            name=f"{root}/control_plane",
            width=6.0,
            height=6.0,
            cell_size=0.10,
            section_size=0.50,
            plane="xy",
            position=np.asarray([2.0, 0.0, 0.0]),
        )
        self._server.scene.add_frame(
            name=f"{root}/zed_front",
            position=np.asarray([0.0, 0.0, ground_height]),
            wxyz=np.asarray([1.0, 0.0, 0.0, 0.0]),
            axes_length=0.18,
            axes_radius=0.005,
        )
        ground_down /= np.linalg.norm(ground_down)
        optical_forward = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
        planar_forward = optical_forward - ground_down * float(
            np.dot(optical_forward, ground_down)
        )
        planar_forward /= np.linalg.norm(planar_forward)
        planar_left = np.cross(planar_forward, ground_down)
        planar_left /= np.linalg.norm(planar_left)
        navigation_from_camera = np.vstack(
            [planar_forward, planar_left, -ground_down]
        )
        vertical_fov = 2.0 * math.atan2(
            rgb.shape[0], 2.0 * float(intrinsics[1, 1])
        )
        self._server.scene.add_camera_frustum(
            name=f"{root}/zed_camera",
            position=np.asarray([0.0, 0.0, ground_height]),
            wxyz=matrix_to_quaternion_wxyz(navigation_from_camera),
            fov=vertical_fov,
            aspect=float(rgb.shape[1]) / float(rgb.shape[0]),
            scale=0.12,
            image=rgb,
        )

        bbox_dimensions = np.asarray(
            [
                max(0.01, target.forward_max_m - target.forward_min_m),
                max(0.01, target.left_max_m - target.left_min_m),
                0.025,
            ]
        )
        bbox_position = np.asarray(
            [
                0.5 * (target.forward_min_m + target.forward_max_m),
                0.5 * (target.left_min_m + target.left_max_m),
                0.025,
            ]
        )
        self._server.scene.add_box(
            name=f"{root}/target_planar_bbox",
            color=np.asarray([255, 145, 30], dtype=np.uint8),
            dimensions=bbox_dimensions,
            position=bbox_position,
            wireframe=True,
        )
        boundary = _rounded_bbox_boundary(
            target.forward_min_m,
            target.forward_max_m,
            target.left_min_m,
            target.left_max_m,
            config.docking_distance_m,
        )
        boundary_3d = np.column_stack(
            [boundary, np.full(boundary.shape[0], 0.04, dtype=np.float64)]
        )
        boundary_segments = np.stack(
            [boundary_3d, np.roll(boundary_3d, -1, axis=0)], axis=1
        )
        self._server.scene.add_line_segments(
            name=f"{root}/docking_envelope",
            points=boundary_segments,
            colors=np.asarray([30, 220, 235], dtype=np.uint8),
            line_width=4.0,
        )

        closest = np.asarray(
            [target.closest_forward_m, target.closest_left_m, 0.07],
            dtype=np.float64,
        )
        self._server.scene.add_line_segments(
            name=f"{root}/target_distance",
            points=np.asarray([[np.asarray([0.0, 0.0, 0.07]), closest]]),
            colors=np.asarray([255, 215, 30], dtype=np.uint8),
            line_width=3.0,
        )
        self._server.scene.add_icosphere(
            name=f"{root}/target_closest_point",
            radius=0.035,
            color=np.asarray([255, 215, 30], dtype=np.uint8),
            position=closest,
        )

        target_bearing = float(target.bearing_rad)
        cosine, sine = math.cos(target_bearing), math.sin(target_bearing)
        if travel_distance > 0.0:
            goal = np.asarray(
                [travel_distance * cosine, travel_distance * sine, 0.08]
            )
            self._server.scene.add_icosphere(
                name=f"{root}/next_camera_goal",
                radius=0.04,
                color=np.asarray([40, 220, 90], dtype=np.uint8),
                position=goal,
            )

        next_step = max(
            config.minimum_forward_command_m,
            min(config.maximum_forward_step_m, travel_distance),
        )
        path_limit = next_step + config.obstacle_path_margin_m
        if (
            isinstance(path_status, dict)
            and path_status.get("path_limit_m") is not None
        ):
            path_limit = float(path_status["path_limit_m"])
        path_valid = isinstance(path_status, dict) and bool(path_status.get("valid"))
        path_clear = path_valid and bool(path_status.get("clear"))
        corridor_bearing = (
            float(path_status.get("bearing_rad", 0.0))
            if isinstance(path_status, dict)
            else target_bearing
        )
        corridor_color = (
            np.asarray([130, 130, 145], dtype=np.uint8)
            if path_status is None
            else np.asarray([40, 210, 95], dtype=np.uint8)
            if path_clear
            else np.asarray([235, 55, 55], dtype=np.uint8)
        )
        corridor_quaternion = np.asarray(
            [
                math.cos(corridor_bearing / 2.0),
                0.0,
                0.0,
                math.sin(corridor_bearing / 2.0),
            ]
        )
        self._server.scene.add_box(
            name=f"{root}/checked_corridor",
            color=corridor_color,
            dimensions=np.asarray(
                [max(0.01, path_limit), 2.0 * config.corridor_half_width_m, 0.012]
            ),
            position=np.asarray(
                [0.5 * path_limit * cosine, 0.5 * path_limit * sine, 0.012]
            ),
            wxyz=corridor_quaternion,
            opacity=0.22,
            material="toon3",
            cast_shadow=False,
            receive_shadow=False,
        )

        obstacle_points_path = (
            np.empty((0, 2), dtype=np.float64)
            if not isinstance(path_status, dict)
            else np.asarray(
                path_status.get("_obstacle_points_path_m", np.empty((0, 2))),
                dtype=np.float64,
            )
        )
        if (
            obstacle_points_path.ndim == 2
            and obstacle_points_path.shape[1:] == (2,)
            and obstacle_points_path.shape[0] > 0
        ):
            along = obstacle_points_path[:, 0]
            cross = obstacle_points_path[:, 1]
            obstacle_points = np.column_stack(
                [
                    cosine * along - sine * cross,
                    sine * along + cosine * cross,
                    np.full(along.shape, 0.09),
                ]
            )
            self._server.scene.add_point_cloud(
                name=f"{root}/blocking_cluster",
                points=obstacle_points.astype(np.float32),
                colors=np.tile(
                    np.asarray([[245, 45, 45]], dtype=np.uint8),
                    (obstacle_points.shape[0], 1),
                ),
                point_size=0.025,
                point_shape="circle",
            )

        avoidance_payload = debug.get(
            "avoidance_path_body_forward_left_m"
        )
        avoidance_path = (
            np.empty((0, 2), dtype=np.float64)
            if avoidance_payload is None
            else np.asarray(avoidance_payload, dtype=np.float64)
        )
        if (
            avoidance_path.ndim == 2
            and avoidance_path.shape[1:] == (2,)
            and avoidance_path.shape[0] >= 2
            and np.all(np.isfinite(avoidance_path))
        ):
            path_3d = np.column_stack(
                [
                    avoidance_path,
                    np.full(avoidance_path.shape[0], 0.10, dtype=np.float64),
                ]
            )
            self._server.scene.add_line_segments(
                name=f"{root}/avoidance_path",
                points=np.stack([path_3d[:-1], path_3d[1:]], axis=1),
                colors=np.asarray([35, 105, 255], dtype=np.uint8),
                line_width=5.0,
            )
            waypoint_payload = debug.get(
                "avoidance_waypoint_body_forward_left_m"
            )
            waypoint = (
                path_3d[-1]
                if waypoint_payload is None
                else np.asarray(
                    [
                        float(waypoint_payload[0]),
                        float(waypoint_payload[1]),
                        0.10,
                    ],
                    dtype=np.float64,
                )
            )
            self._server.scene.add_icosphere(
                name=f"{root}/avoidance_waypoint",
                radius=0.045,
                color=np.asarray([35, 105, 255], dtype=np.uint8),
                position=waypoint,
            )

        self._render_actual_trajectory()

        path_reason = "not checked; turn required first"
        path_details: list[str] = []
        if isinstance(path_status, dict):
            path_reason = str(path_status.get("reason", "unknown"))
            if path_status.get("temporal_frames") is not None:
                path_details.append(
                    f"- Stable depth frames: "
                    f"**{int(path_status['temporal_frames'])}**"
                )
            if path_status.get("ground_filtered_pixels") is not None:
                path_details.append(
                    f"- Floor pixels filtered: "
                    f"**{int(path_status['ground_filtered_pixels'])}**"
                )
            if path_status.get("height_candidate_pixels") is not None:
                path_details.append(
                    f"- Above-floor corridor pixels: "
                    f"**{int(path_status['height_candidate_pixels'])}**"
                )
            if path_status.get("target_mask_filtered_pixels") is not None:
                path_details.append(
                    f"- Target-mask pixels excluded: "
                    f"**{int(path_status['target_mask_filtered_pixels'])}**"
                )
            if path_status.get("obstacle_pixels") is not None:
                path_details.append(
                    f"- Blocking component pixels: "
                    f"**{int(path_status['obstacle_pixels'])}**"
                )
        previous_summary = self._summary_handle
        if previous_summary is not None:
            previous_summary.remove()
        self._summary_handle = self._server.gui.add_markdown(
            "\n".join(
                [
                    f"### Docking target: `{debug['object_name']}`",
                    "",
                    f"- Phase: **{debug['phase']}**",
                    f"- Iteration: **{int(debug['iteration'])}**",
                    f"- SAM3 score: **{float(target.score):.3f}**",
                    f"- Target distance: **{float(target.distance_m):.3f} m**",
                    f"- Bearing: **{math.degrees(target_bearing):+.1f}°**",
                    f"- Bearing tolerance: "
                    f"**{math.degrees(config.bearing_tolerance_rad):.1f}°**",
                    f"- Remaining to {config.docking_distance_m:.2f} m boundary: "
                    f"**{travel_distance:.3f} m**",
                    f"- Corridor width: "
                    f"**{2.0 * config.corridor_half_width_m:.2f} m**",
                    f"- 2-D robot radius: **{config.robot_radius_m:.2f} m**",
                    f"- Extra obstacle margin: "
                    f"**{config.obstacle_clearance_margin_m:.2f} m**",
                    f"- Occupancy resolution: "
                    f"**{config.occupancy_resolution_m:.2f} m**",
                    f"- Ignored floor band: "
                    f"**0–{float(config.obstacle_min_height_m):.2f} m**",
                    f"- Ground source: "
                    f"**{ground_plane.get('source', 'unknown')}**",
                    f"- Current ZED height: **{ground_height:.3f} m**",
                    f"- Path: **{path_reason}**",
                    *(
                        [f"- Avoidance plan: **{avoidance_plan_reason}**"]
                        if avoidance_plan_reason is not None
                        else []
                    ),
                    *path_details,
                ]
            )
        )

        look_at = np.asarray(
            [target.closest_forward_m, target.closest_left_m, 0.0],
            dtype=np.float64,
        )
        camera_position = look_at + np.asarray([-1.0, -1.2, 0.9])
        self._server.initial_camera.look_at = look_at
        self._server.initial_camera.position = camera_position
        get_clients = getattr(self._server, "get_clients", None)
        client_map = get_clients() if callable(get_clients) else {}
        clients = client_map.values() if isinstance(client_map, dict) else []
        for client in clients:
            try:
                client.camera.up_direction = np.asarray([0.0, 0.0, 1.0])
                client.camera.look_at = look_at
                client.camera.position = camera_position
            except Exception:
                pass

    def close(self) -> None:
        stop = getattr(self._server, "stop", None)
        if callable(stop):
            stop()
