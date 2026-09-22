from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    package = FindPackageShare("yor_nav2_bridge")
    params = LaunchConfiguration("params_file")
    tf_remappings = [("/tf", "tf"), ("/tf_static", "tf_static")]
    lifecycle_nodes = [
        "controller_server",
        "planner_server",
        "behavior_server",
        "bt_navigator",
        "velocity_smoother",
    ]
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "params_file",
                default_value=PathJoinSubstitution([package, "config", "nav2_params.yaml"]),
            ),
            Node(
                package="yor_nav2_bridge",
                executable="zed_bridge",
                name="yor_zed_bridge",
                output="screen",
                parameters=[params],
            ),
            Node(
                package="yor_nav2_bridge",
                executable="base_bridge",
                name="yor_base_bridge",
                output="screen",
            ),
            # Launch only the Nav2 servers used by NavigateToPose. The stock
            # navigation_launch.py also starts smoother_server and
            # waypoint_follower, neither of which is used by this primitive.
            Node(
                package="nav2_controller",
                executable="controller_server",
                name="controller_server",
                output="screen",
                parameters=[params],
                remappings=tf_remappings + [("cmd_vel", "cmd_vel_nav")],
            ),
            Node(
                package="nav2_planner",
                executable="planner_server",
                name="planner_server",
                output="screen",
                parameters=[params],
                remappings=tf_remappings,
            ),
            Node(
                package="nav2_behaviors",
                executable="behavior_server",
                name="behavior_server",
                output="screen",
                parameters=[params],
                remappings=tf_remappings,
            ),
            Node(
                package="nav2_bt_navigator",
                executable="bt_navigator",
                name="bt_navigator",
                output="screen",
                parameters=[params],
                remappings=tf_remappings,
            ),
            Node(
                package="nav2_velocity_smoother",
                executable="velocity_smoother",
                name="velocity_smoother",
                output="screen",
                parameters=[params],
                remappings=tf_remappings
                + [("cmd_vel", "cmd_vel_nav"), ("cmd_vel_smoothed", "cmd_vel")],
            ),
            Node(
                package="nav2_lifecycle_manager",
                executable="lifecycle_manager",
                name="lifecycle_manager_navigation",
                output="screen",
                parameters=[
                    {"autostart": True, "node_names": lifecycle_nodes}
                ],
            ),
            # Chained monitors: the structure one clamps cmd_vel against every
            # point using the body outline, then the arms one clamps what is
            # left against the arms envelope using only the arm-height points
            # the ZED bridge republishes. Humble pools all of a node's sources
            # for every one of its polygons, so height resolution needs two
            # nodes. The base cannot move until both are active: nothing
            # publishes /yor/cmd_vel_safe before then.
            Node(
                package="nav2_collision_monitor",
                executable="collision_monitor",
                name="collision_monitor_structure",
                output="screen",
                parameters=[params],
            ),
            Node(
                package="nav2_collision_monitor",
                executable="collision_monitor",
                name="collision_monitor_arms",
                output="screen",
                parameters=[params],
            ),
            Node(
                package="nav2_lifecycle_manager",
                executable="lifecycle_manager",
                name="lifecycle_manager_collision_monitor",
                output="screen",
                parameters=[
                    {
                        "autostart": True,
                        "node_names": [
                            "collision_monitor_structure",
                            "collision_monitor_arms",
                        ],
                    }
                ],
            ),
        ]
    )
