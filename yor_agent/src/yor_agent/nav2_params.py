"""Derive Nav2 collision geometry from the primitive configuration."""

from __future__ import annotations

import argparse
import copy
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml


FOOTPRINT_VERTICES = 16
# Costmap inflation beyond the footprint edge. 0.30 on top of the 0.60 m
# circle put the ZED 0.9 m from anything before the cost started rising; with
# the circle at 0.30 m the band is halved so a docking goal 0.50 m from a desk
# top is not buried in inflation cost.
INFLATION_OUTSIDE_FOOTPRINT_M = 0.15


def circular_footprint(
    *, radius_m: float, center_forward_m: float, center_left_m: float
) -> list[list[float]]:
    """Return a counter-clockwise regular polygon in ``base_footprint``."""

    return [
        [
            round(
                center_forward_m
                + radius_m * math.cos(2.0 * math.pi * index / FOOTPRINT_VERTICES),
                6,
            ),
            round(
                center_left_m
                + radius_m * math.sin(2.0 * math.pi * index / FOOTPRINT_VERTICES),
                6,
            ),
        ]
        for index in range(FOOTPRINT_VERTICES)
    ]


def polygon_footprint(vertices_xy: Any) -> list[list[float]]:
    """Return a counter-clockwise convex polygon in ``base_footprint``.

    ``vertices_xy`` are metres from the swerve centre, +x forward, +y left, in
    any order; the convex hull is taken so a hand-typed rectangle cannot come
    out self-intersecting. A polygon lets Nav2 keep the arms' width and the
    chassis rear out of walls, which a circle centred on the ZED cannot do
    without also pushing its front far past the grippers.
    """

    try:
        points = [[float(x), float(y)] for x, y in vertices_xy]
    except (TypeError, ValueError) as exc:
        raise ValueError("robot_footprint_xy must be a list of [x, y] pairs") from exc
    if len(points) < 3 or not all(math.isfinite(v) for point in points for v in point):
        raise ValueError("robot_footprint_xy needs at least 3 finite [x, y] vertices")
    unique = sorted(set((x, y) for x, y in points))
    if len(unique) < 3:
        raise ValueError("robot_footprint_xy is degenerate")

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[tuple[float, float]] = []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0.0:
            lower.pop()
        lower.append(point)
    upper: list[tuple[float, float]] = []
    for point in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0.0:
            upper.pop()
        upper.append(point)
    hull = lower[:-1] + upper[:-1]
    if len(hull) < 3:
        raise ValueError("robot_footprint_xy is degenerate")
    return [[round(x, 6), round(y, 6)] for x, y in hull]


def _arm_band_parameters(robot: Mapping[str, Any]) -> dict[str, Any]:
    """Arm-height band settings, taken from the direct-motion gate's config.

    The ZED bridge must band the cloud exactly the way
    ``robot/footprint_clearance.py`` bands it for ``drive_straight``, so both
    read the same numbers rather than keeping a second copy in the Nav2
    settings.
    """

    navigation = robot.get("navigation")
    if not isinstance(navigation, Mapping):
        return {}
    footprint = navigation.get("footprint")
    layer_name = str(navigation.get("arm_footprint_layer", "arms"))
    z_min, z_max = 0.27, 1.45
    if isinstance(footprint, Mapping):
        for layer in footprint.get("layers") or ():
            if isinstance(layer, Mapping) and layer.get("name") == layer_name:
                z_min = float(layer.get("z_min", z_min))
                z_max = float(layer.get("z_max", z_max))
                break
    # The arms collision monitor gets its own full vertical pad
    # (nav2_arm_band_z_margin_m; the gate's arm_footprint_z_margin_m when
    # unset), so it sees a support surface just below the arms. Banded with
    # the core pad alone it did not, and the final turn of a docking goal
    # carried an arm into a table corner (2026-09-14). The price is that a goal
    # which would put the arms over such a surface stalls instead; docking then
    # docks only if the base already stands on the goal's bearing inside its
    # arrival boundary.
    z_margin = navigation.get("nav2_arm_band_z_margin_m")
    if z_margin is None:
        z_margin = navigation.get("arm_footprint_z_margin_m", 0.01)
    return {
        "arm_band_enabled": bool(
            navigation.get("arm_footprint_from_spheres", True)
        ),
        "arm_band_z_min_m": z_min,
        "arm_band_z_max_m": z_max,
        "arm_band_m": float(navigation.get("arm_footprint_band_m", 0.05)),
        "arm_band_z_margin_m": float(z_margin),
        "arm_band_forward_margin_m": float(
            navigation.get("arm_footprint_forward_margin_m", 0.05)
        ),
        "arm_band_point_tolerance_m": float(
            navigation.get("layer_z_tolerance_m", 0.03)
        ),
    }


def _polygon_contains(
    outer_ccw: list[list[float]], inner_ccw: list[list[float]]
) -> bool:
    """Whether every vertex of ``inner_ccw`` lies inside the convex ``outer_ccw``."""

    count = len(outer_ccw)
    for x, y in inner_ccw:
        for index in range(count):
            ax, ay = outer_ccw[index]
            bx, by = outer_ccw[(index + 1) % count]
            if (bx - ax) * (y - ay) - (by - ay) * (x - ax) < -1e-9:
                return False
    return True


def render_nav2_parameters(
    primitive_config: Mapping[str, Any],
    template: Mapping[str, Any],
    robot_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Inject primitive-owned geometry into a detached Nav2 parameter tree."""

    primitives = primitive_config.get("primitives")
    if not isinstance(primitives, Mapping):
        raise ValueError("primitive config must contain a primitives mapping")
    docking_section = primitives.get("dock_to_visible_object")
    if not isinstance(docking_section, Mapping):
        raise ValueError("dock_to_visible_object primitive config is required")
    settings = docking_section.get("settings")
    if not isinstance(settings, Mapping):
        raise ValueError("dock_to_visible_object settings are required for Nav2")
    if str(settings.get("backend", "legacy")).strip().lower() != "nav2":
        raise ValueError("dock_to_visible_object backend must be nav2")
    docking_distance_m = _finite_float(settings, "docking_distance_m")
    nav2_goal_distance_m = _finite_float(settings, "nav2_goal_distance_m")
    radius_m = _finite_float(settings, "robot_radius_m")
    center_forward_m = _finite_float(settings, "base_to_camera_forward_m")
    center_left_m = _finite_float(settings, "base_to_camera_left_m")
    if not 0.1 <= docking_distance_m <= 1.5:
        raise ValueError("docking_distance_m must be in [0.1, 1.5]")
    if not 0.1 <= nav2_goal_distance_m <= docking_distance_m:
        raise ValueError(
            "nav2_goal_distance_m must be in [0.1, docking_distance_m]"
        )
    if not 0.1 <= radius_m <= 1.5:
        raise ValueError("robot_radius_m must be in [0.1, 1.5]")
    if not 0.0 <= center_forward_m <= 1.0:
        raise ValueError("base_to_camera_forward_m must be in [0, 1]")
    if not -0.5 <= center_left_m <= 0.5:
        raise ValueError("base_to_camera_left_m must be in [-0.5, 0.5]")

    rendered = copy.deepcopy(dict(template))
    polygon_vertices = settings.get("robot_footprint_xy")
    if polygon_vertices is not None:
        footprint = polygon_footprint(polygon_vertices)
        # Nav2 inflates from the footprint edge but requires the parameter to
        # exceed the circumscribed radius about the base frame.
        circumscribed_m = max(math.hypot(x, y) for x, y in footprint)
    else:
        footprint = circular_footprint(
            radius_m=radius_m,
            center_forward_m=center_forward_m,
            center_left_m=center_left_m,
        )
        circumscribed_m = radius_m
    footprint_text = json.dumps(footprint, separators=(",", ":"))
    flat_points = [coordinate for point in footprint for coordinate in point]

    rendered["local_costmap"]["local_costmap"]["ros__parameters"][
        "footprint"
    ] = footprint_text
    rendered["global_costmap"]["global_costmap"]["ros__parameters"][
        "footprint"
    ] = footprint_text
    try:
        arms_monitor = rendered["collision_monitor_arms"]["ros__parameters"]
        structure_monitor = rendered["collision_monitor_structure"][
            "ros__parameters"
        ]
    except KeyError as exc:  # Fail closed like the footprint placeholders.
        raise ValueError(
            "Nav2 template must define collision_monitor_structure and "
            "collision_monitor_arms"
        ) from exc
    arms_monitor["ArmsStop"]["points"] = flat_points
    # The arms monitor only ever sees points the ZED bridge measured at arm
    # height; everything else is charged against the structure outline, the
    # same split robot/footprint_clearance.py applies to the direct
    # primitives. Without it the arms' 2-D envelope charges a desk top against
    # grippers that pass above it and docking stalls short (2026-09-07).
    structure_vertices = settings.get("robot_structure_footprint_xy")
    if structure_vertices is None:
        raise ValueError(
            "robot_structure_footprint_xy is required for the structure "
            "collision monitor"
        )
    structure_footprint = polygon_footprint(structure_vertices)
    if polygon_vertices is not None and not _polygon_contains(
        footprint, structure_footprint
    ):
        raise ValueError(
            "robot_structure_footprint_xy must lie inside robot_footprint_xy"
        )
    structure_flat = [
        coordinate for point in structure_footprint for coordinate in point
    ]
    structure_monitor["StructureStop"]["points"] = structure_flat

    inflation_radius = round(
        circumscribed_m + INFLATION_OUTSIDE_FOOTPRINT_M, 6
    )
    rendered["local_costmap"]["local_costmap"]["ros__parameters"][
        "inflation_layer"
    ]["inflation_radius"] = inflation_radius
    rendered["global_costmap"]["global_costmap"]["ros__parameters"][
        "inflation_layer"
    ]["inflation_radius"] = inflation_radius

    zed_parameters = rendered.setdefault("yor_zed_bridge", {}).setdefault(
        "ros__parameters", {}
    )
    zed_parameters.update(
        {
            "max_depth_m": float(settings.get("target_max_depth_m", 20.0)),
            "base_to_camera_forward_m": center_forward_m,
            "base_to_camera_left_m": center_left_m,
            # A copy, not the same list object: an alias in the rendered
            # YAML is not resolved by every ROS 2 parameter parser.
            "structure_footprint_xy": list(structure_flat),
            "structure_footprint_topic": structure_monitor["StructureApproach"][
                "footprint_topic"
            ],
            "arm_band_cloud_topic": arms_monitor["zed_points_arms"]["topic"],
            # The floor-plane gate runs in the bridge and in the docking
            # perception with the same tolerances and the same fallback plane,
            # which its envelope is centred on, taken from the same docking
            # settings so the two cannot drift apart.
            "fallback_camera_height_m": float(
                settings.get("ground_camera_height_m", 1.122339129447937)
            ),
            "fallback_ground_down_xyz": [
                float(value)
                for value in settings.get(
                    "ground_down_camera_xyz",
                    [0.013630234132681617, 0.9477663115351749, 0.31867418382495005],
                )
            ],
            "ground_plane_max_age_s": float(settings.get("ground_plane_max_age_s", 2.0)),
            "ground_plane_max_height_step_m": float(
                settings.get("ground_plane_max_height_step_m", 0.05)
            ),
            "ground_plane_max_tilt_step_deg": float(
                settings.get("ground_plane_max_tilt_step_deg", 3.0)
            ),
            "ground_plane_settle_s": float(settings.get("ground_plane_settle_s", 2.0)),
            "ground_plane_max_height_error_m": float(
                settings.get("ground_plane_max_height_error_m", 0.25)
            ),
            "ground_plane_max_tilt_error_deg": float(
                settings.get("ground_plane_max_tilt_error_deg", 15.0)
            ),
        }
    )
    if robot_config is not None:
        robot = robot_config.get("robot")
        if not isinstance(robot, Mapping):
            raise ValueError("robot config must contain a robot mapping")
        zed_parameters.update(_arm_band_parameters(robot))
        manipulation = robot.get("manipulation")
        if not isinstance(manipulation, Mapping):
            raise ValueError("robot.manipulation config is required for Nav2")
        self_filter = manipulation.get("robot_self_filter")
        if not isinstance(self_filter, Mapping):
            raise ValueError("robot_self_filter config is required for Nav2")
        intrinsics = manipulation.get("camera_intrinsics")
        resolution = manipulation.get("camera_calibration_resolution")
        if (
            not isinstance(intrinsics, list)
            or len(intrinsics) != 3
            or not isinstance(resolution, list)
            or len(resolution) != 2
        ):
            raise ValueError("Nav2 camera calibration is invalid")
        transforms: dict[str, list[float]] = {}
        for name in ("left", "right"):
            value = manipulation.get(f"{name}_arm_from_camera")
            if not isinstance(value, list) or len(value) != 4:
                raise ValueError(f"{name}_arm_from_camera is invalid")
            flattened = [float(item) for row in value for item in row]
            if len(flattened) != 16 or not all(map(math.isfinite, flattened)):
                raise ValueError(f"{name}_arm_from_camera is invalid")
            transforms[name] = flattened
        zed_parameters.update(
            {
                "camera_fx": float(intrinsics[0][0]),
                "camera_fy": float(intrinsics[1][1]),
                "camera_cx": float(intrinsics[0][2]),
                "camera_cy": float(intrinsics[1][2]),
                "calibration_width": int(resolution[0]),
                "calibration_height": int(resolution[1]),
                "self_filter_enabled": bool(self_filter.get("enabled", False)),
                "self_filter_urdf_path": str(self_filter.get("urdf_path", "")),
                "self_filter_spheres_path": str(
                    self_filter.get("spheres_path", "")
                ),
                "self_filter_sphere_erosion_m": float(
                    self_filter.get("sphere_erosion_m", 0.008)
                ),
                "self_filter_depth_tolerance_m": float(
                    self_filter.get("depth_tolerance_m", 0.010)
                ),
                "self_filter_joint_change_tolerance_rad": float(
                    self_filter.get("joint_change_tolerance_rad", 0.002)
                ),
                "self_filter_require_both_arms": bool(
                    self_filter.get("require_both_arms", True)
                ),
                "self_filter_mask_padding_m": float(
                    self_filter.get("mask_padding_m", 0.0)
                ),
                "self_filter_arm_status_poll_hz": float(
                    self_filter.get("arm_status_poll_hz", 2.0)
                ),
                "arm_rpc_host": str(robot.get("arm_rpc_host", "")),
                "arm_rpc_port": int(robot.get("arm_rpc_port", 5558)),
                "left_arm_from_camera": transforms["left"],
                "right_arm_from_camera": transforms["right"],
            }
        )
    return rendered


def render_nav2_parameter_file(
    *,
    primitive_config_path: str | Path,
    template_path: str | Path,
    robot_config_path: str | Path | None = None,
) -> dict[str, Any]:
    primitive_config = yaml.safe_load(
        Path(primitive_config_path).read_text(encoding="utf-8")
    )
    if not isinstance(primitive_config, Mapping):
        raise ValueError("primitive config must be a YAML mapping")
    template = yaml.safe_load(Path(template_path).read_text(encoding="utf-8"))
    if not isinstance(template, Mapping):
        raise ValueError("Nav2 parameter template must be a YAML mapping")
    robot_config = None
    if robot_config_path is not None:
        robot_config = yaml.safe_load(
            Path(robot_config_path).read_text(encoding="utf-8")
        )
        if not isinstance(robot_config, Mapping):
            raise ValueError("robot config must be a YAML mapping")
    return render_nav2_parameters(primitive_config, template, robot_config)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Render Nav2 parameters from primitive_config.yaml"
    )
    parser.add_argument("--primitive-config", required=True)
    parser.add_argument("--template", required=True)
    parser.add_argument("--robot-config")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    rendered = render_nav2_parameter_file(
        primitive_config_path=args.primitive_config,
        template_path=args.template,
        robot_config_path=args.robot_config,
    )
    output_path = Path(args.output)
    output_path.write_text(
        yaml.safe_dump(rendered, sort_keys=False), encoding="utf-8"
    )
    primitive_config = yaml.safe_load(
        Path(args.primitive_config).read_text(encoding="utf-8")
    )
    docking = primitive_config["primitives"]["dock_to_visible_object"]["settings"]
    print(
        "Generated Nav2 parameters from primitive config: "
        f"docking_distance_m={float(docking['docking_distance_m']):.3f}, "
        f"nav2_goal_distance_m={float(docking['nav2_goal_distance_m']):.3f}, "
        f"robot_radius_m={float(docking['robot_radius_m']):.3f}, "
        f"footprint={'polygon' if docking.get('robot_footprint_xy') else 'circle'}, "
        "collision_monitors=structure(all points)+arms(arm-height points)"
    )
    return 0


def _finite_float(values: Mapping[str, Any], key: str) -> float:
    if key not in values or isinstance(values[key], bool):
        raise ValueError(f"{key} must be configured as a finite number")
    try:
        value = float(values[key])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be configured as a finite number") from exc
    if not math.isfinite(value):
        raise ValueError(f"{key} must be configured as a finite number")
    return value


if __name__ == "__main__":
    raise SystemExit(main())
