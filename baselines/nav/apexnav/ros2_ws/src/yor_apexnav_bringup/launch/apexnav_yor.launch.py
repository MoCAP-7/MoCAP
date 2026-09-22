"""Launch the upstream ApexNav planner with only YOR embodiment parameters changed."""

from __future__ import annotations

import math
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, OpaqueFunction, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import yaml


def _nodes(context):
    config_path = LaunchConfiguration("baseline_config").perform(context)
    with open(config_path, "r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    camera = config["camera"]
    planner = config["planner"]
    robot = config["robot"]
    intrinsics = camera["intrinsics"]

    trajectory_share = get_package_share_directory("trajectory_manager")
    tsp_share = get_package_share_directory("lkh_mtsp_solver")
    planning_yaml = os.path.join(trajectory_share, "config", "planning_param.yaml")
    control_yaml = os.path.join(trajectory_share, "config", "control_param.yaml")
    tsp_dir = os.path.join(tsp_share, "resource")

    planner_parameters = {
        "is_real_world": True,
        "sdf_map.ray_mode": 0,
        "sdf_map.resolution": float(planner["map_resolution_m"]),
        "sdf_map.map_size_x": float(planner["map_size_m"]),
        "sdf_map.map_size_y": float(planner["map_size_m"]),
        "sdf_map.obstacles_inflation": float(planner["obstacle_inflation_m"]),
        "sdf_map.local_bound": 5.0,
        # Clear only map cells intersected by ApexNav's own rectangular
        # footprint at the base odometry center. The half-cell diagonal keeps
        # continuous boundary samples from indexing an adjacent UNKNOWN cell.
        "sdf_map.robot_clear_radius": 0.5
        * math.hypot(
            float(planner["footprint_length_m"]),
            float(planner["footprint_width_m"]),
        )
        + float(planner["map_resolution_m"]) / math.sqrt(2.0),
        "sdf_map.p_hit": 0.90,
        "sdf_map.p_miss": 0.48,
        "sdf_map.p_min": 0.10,
        "sdf_map.p_max": 0.98,
        "sdf_map.p_occ": 0.80,
        "sdf_map.max_ray_length": float(camera["maximum_depth_m"]) - 0.01,
        "sdf_map.optimistic": False,
        "sdf_map.signed_dist": False,
        "map_ros/cx": float(intrinsics[0][2]),
        "map_ros/cy": float(intrinsics[1][2]),
        "map_ros/fx": float(intrinsics[0][0]),
        "map_ros/fy": float(intrinsics[1][1]),
        "map_ros/depth_filter_maxdist": float(camera["maximum_depth_m"]),
        "map_ros/depth_filter_mindist": float(camera["minimum_depth_m"]),
        "map_ros/depth_filter_margin": 2,
        "map_ros/filter_min_height": float(planner["minimum_obstacle_height_m"]),
        "map_ros/filter_max_height": float(planner["maximum_obstacle_height_m"]),
        "map_ros/k_depth_scaling_factor": 65535.0,
        "map_ros/skip_pixel": 1,
        "map_ros/frame_id": "world",
        "map_ros/virtual_ground_height": float(planner["virtual_ground_height_m"]),
        # The camera is fixed with a downward tilt and the real-world FSM never
        # pitches it, so object clouds are not gated on camera pitch.
        "map_ros/min_object_camera_pitch": 0.0,
        "fsm/replan_time": 0.30,
        "fsm/replan_traj_end_threshold": 1.0,
        "fsm/replan_frontier_change_delay": 1.0,
        "fsm/replan_timeout": 3.0,
        # Planning on the robot takes far longer than replan_time, so a replan
        # made while moving is predicted as far ahead as recent plans took. The
        # robot keeps following, and being checked against, the published
        # trajectory until the new one starts.
        "fsm/adaptive_replan_time": bool(planner["adaptive_replan_time"]),
        "fsm/replan_time_margin": 0.5,
        "fsm/max_replan_time": 8.0,
        "fsm/max_replan_overrun": 0.5,
        "exploration/policy": 2,
        "exploration/sigma_threshold": 0.015,
        "exploration/max_to_mean_threshold": 1.10,
        "exploration/max_to_mean_percentage": 0.90,
        "exploration/tsp_dir": tsp_dir,
        "frontier.cluster_min": 5,
        "frontier.cluster_size_xy": 0.65,
        "frontier.min_view_finish_fraction": 0.2,
        "frontier.min_contain_unknown": 30,
        "object.min_observation_num": 2,
        "object.fusion_type": 1,
        "object.use_observation": True,
        "object.vis_cloud": True,
        "perception_utils.left_angle": math.radians(30.0),
        "perception_utils.right_angle": math.radians(30.0),
        "perception_utils.max_dist": 4.0,
        "perception_utils.vis_dist": 1.0,
        "astar.lambda_heu": 1.0,
        "astar.resolution_astar": 0.1,
        # Embodiment-only overrides; policy and semantic thresholds above are upstream.
        "max_vel": float(robot["maximum_linear_mps"]),
        "max_acc": 0.30,
        "max_domega": 0.70,
        "wheel_base": float(planner["wheel_base_m"]),
        "length": float(planner["footprint_length_m"]),
        "width": float(planner["footprint_width_m"]),
        "safe_dist": float(planner["optimizer_safe_distance_m"]),
        "kino_astar.max_vel": float(robot["maximum_linear_mps"]),
        "kino_astar.max_acc": 0.30,
        "kino_astar.max_cur": 10.0,
        "kino_astar.wheel_base": float(planner["wheel_base_m"]),
        "kino_astar.length": float(planner["footprint_length_m"]),
        "kino_astar.width": float(planner["footprint_width_m"]),
        "kino_astar.height": 1.60,
        # The footprint stays upstream's, so path search also keeps the base
        # center out of obstacles inflated by the robot's reach.
        "kino_astar.check_inflated_occupancy": True,
        "optimizer.max_vel": float(robot["maximum_linear_mps"]),
        "optimizer.max_acc": 0.30,
        "optimizer.max_domega": 0.70,
        "optimizer.wheel_base": float(planner["wheel_base_m"]),
        "optimizer.safe_dist": float(planner["optimizer_safe_distance_m"]),
        # Retime each optimized trajectory along its own path before the
        # planner stores and publishes it, so its speed and heading rate stay
        # within the reference limits and its accelerations within the limits
        # above, reaching the reference limits as soon as those allow.
        "optimizer.time_scale_to_limits": bool(planner["time_scale_to_limits"]),
        "optimizer.time_scale_max_vel": float(planner["reference_max_linear_mps"]),
        "optimizer.time_scale_max_omega": float(planner["reference_max_yaw_rad_s"]),
    }
    remappings = [
        ("/odom_world", "/apexnav/odom"),
        ("/map_ros/pose", "/apexnav/sensor_pose"),
        ("/map_ros/robot_pose", "/apexnav/odom"),
        ("/map_ros/depth", "/apexnav/depth_normalized"),
        ("/detector/clouds_with_scores", "/apexnav/detector/clouds_with_scores"),
        ("/detector/confidence_threshold", "/apexnav/detector/confidence_threshold"),
        ("/blip2/cosine_score", "/apexnav/blip2/cosine_score"),
        ("/move_base_simple/goal", "/apexnav/start"),
        ("/initialpose", "/apexnav/manual_goal"),
        ("/planning/trajectory", "/apexnav/planning/trajectory"),
        ("/traj_server/stop", "/apexnav/traj_server/stop"),
        ("/ros/state", "/apexnav/ros/state"),
        ("/ros/expl_state", "/apexnav/ros/expl_state"),
        ("/ros/expl_result", "/apexnav/ros/expl_result"),
        ("/solve_tsp", "/apexnav/solve_tsp"),
        ("/robot", "/apexnav/visualization/robot"),
    ]
    exploration = Node(
        package="exploration_manager",
        executable="exploration_node",
        name="apexnav_planner",
        output="screen",
        parameters=[planning_yaml, planner_parameters],
        remappings=remappings,
    )
    tsp = Node(
        package="lkh_mtsp_solver",
        executable="tsp_node",
        name="apexnav_tsp_solver",
        output="log",
        parameters=[{"exploration/tsp_dir": tsp_dir}],
        remappings=[("/solve_tsp", "/apexnav/solve_tsp")],
    )
    trajectory = Node(
        package="trajectory_manager",
        executable="traj_server",
        name="apexnav_trajectory_server",
        output="screen",
        parameters=[
            control_yaml,
            {
                # Rotate the forward-facing ZED once to initialize the local map
                # before UNKNOWN-space collision checking starts exploration.
                "need_init": True,
                "initial_rotation_timeout_s": float(
                    planner["initial_scan_timeout_s"]
                ),
                "max_correction_vel": float(robot["maximum_linear_mps"]),
                "max_correction_omega": float(robot["maximum_yaw_rad_s"]),
            },
        ],
        remappings=[
            ("odometry", "/apexnav/odom"),
            ("trajectory", "/apexnav/planning/trajectory"),
            ("cmd_vel", "/apexnav/cmd_vel_raw"),
            ("initial_scan_status", "/apexnav/initial_scan_status"),
            ("/traj_server/stop", "/apexnav/traj_server/stop"),
        ],
    )
    critical_exit_handlers = [
        RegisterEventHandler(
            OnProcessExit(
                target_action=node,
                on_exit=[
                    EmitEvent(
                        event=Shutdown(
                            reason=f"critical ApexNav process exited: {name}"
                        )
                    )
                ],
            )
        )
        for name, node in (
            ("exploration", exploration),
            ("tsp", tsp),
            ("trajectory", trajectory),
        )
    ]
    return [exploration, tsp, trajectory, *critical_exit_handlers]


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "baseline_config",
                description="Absolute path to baselines/nav/apexnav/config.yaml",
            ),
            OpaqueFunction(function=_nodes),
        ]
    )
