"""Republish the existing atomic ZED stream as odometry, TF, and PointCloud2."""

from __future__ import annotations

import json
import math
import struct
import threading
import time
from typing import Any

from geometry_msgs.msg import Point32, PolygonStamped, TransformStamped
from nav_msgs.msg import Odometry
import numpy as np
from rcl_interfaces.msg import SetParametersResult
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2, PointField
from tf2_ros import TransformBroadcaster

from yor_agent.robot.footprint_clearance import (
    LAYER_Z_TOLERANCE_M,
    arm_forward_band_range,
)

from yor_agent.robot.ground_plane_gate import (
    ACCEPTED,
    GroundPlaneDecision,
    GroundPlaneGate,
)

from yor_agent.robot.self_filter import (
    RobotDepthSelfFilter,
    RobotSelfFilterConfig,
)


def _quaternion_from_matrix(matrix: np.ndarray) -> tuple[float, float, float, float]:
    """Return ROS xyzw quaternion for a proper 3x3 rotation matrix."""

    m = np.asarray(matrix, dtype=np.float64)
    trace = float(np.trace(m))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        return (
            float((m[2, 1] - m[1, 2]) / s),
            float((m[0, 2] - m[2, 0]) / s),
            float((m[1, 0] - m[0, 1]) / s),
            float(0.25 * s),
        )
    index = int(np.argmax(np.diag(m)))
    if index == 0:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        return (
            float(0.25 * s),
            float((m[0, 1] + m[1, 0]) / s),
            float((m[0, 2] + m[2, 0]) / s),
            float((m[2, 1] - m[1, 2]) / s),
        )
    if index == 1:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        return (
            float((m[0, 1] + m[1, 0]) / s),
            float(0.25 * s),
            float((m[1, 2] + m[2, 1]) / s),
            float((m[0, 2] - m[2, 0]) / s),
        )
    s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
    return (
        float((m[0, 2] + m[2, 0]) / s),
        float((m[1, 2] + m[2, 1]) / s),
        float(0.25 * s),
        float((m[1, 0] - m[0, 1]) / s),
    )


def _cloud_message(points: np.ndarray, stamp) -> PointCloud2:
    """One XYZ float32 cloud in the ZED optical frame."""

    message = PointCloud2()
    message.header.stamp = stamp
    message.header.frame_id = "zed_left_camera_optical_frame"
    message.height = 1
    message.width = int(points.shape[0])
    message.fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
    ]
    message.is_bigendian = struct.pack("=I", 1) == struct.pack(">I", 1)
    message.point_step = 12
    message.row_step = message.point_step * message.width
    message.data = np.ascontiguousarray(points, dtype=np.float32).tobytes()
    message.is_dense = True
    return message


class ZedBridge(Node):
    def __init__(self) -> None:
        super().__init__("yor_zed_bridge")
        self.declare_parameter("zed_host", "127.0.0.1")
        self.declare_parameter("zed_port", 6000)
        self.declare_parameter("zed_transport", "atomic")
        self.declare_parameter("published_color_order", "auto")
        self.declare_parameter("publish_hz", 8.0)
        self.declare_parameter("point_stride", 3)
        self.declare_parameter("max_depth_m", 20.0)
        self.declare_parameter("camera_fx", 267.214)
        self.declare_parameter("camera_fy", 267.150)
        self.declare_parameter("camera_cx", 337.181)
        self.declare_parameter("camera_cy", 182.288)
        self.declare_parameter("calibration_width", 672)
        self.declare_parameter("calibration_height", 376)
        self.declare_parameter("self_filter_enabled", False)
        self.declare_parameter("self_filter_urdf_path", "")
        self.declare_parameter("self_filter_spheres_path", "")
        self.declare_parameter("self_filter_sphere_erosion_m", 0.008)
        self.declare_parameter("self_filter_depth_tolerance_m", 0.010)
        self.declare_parameter(
            "self_filter_joint_change_tolerance_rad", 0.002
        )
        self.declare_parameter("self_filter_require_both_arms", True)
        self.declare_parameter("self_filter_mask_padding_m", 0.0)
        self.declare_parameter("self_filter_arm_status_poll_hz", 2.0)
        self.declare_parameter("self_filter_attached_objects_json", "{}")
        self.declare_parameter("arm_rpc_host", "")
        self.declare_parameter("arm_rpc_port", 5558)
        self.declare_parameter(
            "left_arm_from_camera", np.eye(4).reshape(-1).tolist()
        )
        self.declare_parameter(
            "right_arm_from_camera", np.eye(4).reshape(-1).tolist()
        )
        self.declare_parameter("fallback_camera_height_m", 1.122339129447937)
        # Horizontal geometry from the upstream YOR ZED/base calibration.
        # This is the left optical center relative to the swerve rotation
        # center, expressed in robot-forward / robot-left coordinates.
        self.declare_parameter("base_to_camera_forward_m", 0.2143)
        self.declare_parameter("base_to_camera_left_m", 0.0603)
        self.declare_parameter("twist_smoothing_alpha", 0.35)
        self.declare_parameter(
            "fallback_ground_down_xyz",
            [0.013630234132681617, 0.9477663115351749, 0.31867418382495005],
        )
        # Plausibility gate on the SDK floor plane, shared with the docking
        # perception (robot/ground_plane_gate.py). The camera TF, every
        # cloud point's height and the arm band below all follow that plane,
        # and the SDK's fit is wrong in bursts whenever little floor is in
        # view: a plane within one step of the plane in use is adopted at
        # once, a larger jump only after it has persisted for the settle
        # time, a plane outside the envelope around the fallback never. A
        # plane the SDK stopped refitting is not offered at all, so it
        # cannot earn persistence frame after frame.
        self.declare_parameter("ground_plane_max_age_s", 2.0)
        self.declare_parameter("ground_plane_max_height_step_m", 0.05)
        self.declare_parameter("ground_plane_max_tilt_step_deg", 3.0)
        self.declare_parameter("ground_plane_settle_s", 2.0)
        self.declare_parameter("ground_plane_max_height_error_m", 0.25)
        self.declare_parameter("ground_plane_max_tilt_error_deg", 15.0)
        # Height-resolved feed for Nav2's collision monitors. The monitors
        # cannot resolve height, so this node splits the cloud the way
        # robot/footprint_clearance.py already does for the direct primitives:
        # every point stays in /yor/zed/points, which the structure monitor
        # checks against the body outline, and only points at arm height are
        # republished here for the monitor that carries the arms' envelope.
        # Without this a 0.72 m desk top is charged against grippers that pass
        # above it and docking stalls short of every table (2026-09-07).
        self.declare_parameter("arm_band_cloud_topic", "/yor/zed/points_arms")
        self.declare_parameter("arm_band_enabled", True)
        self.declare_parameter("arm_band_z_min_m", 0.27)
        self.declare_parameter("arm_band_z_max_m", 1.45)
        self.declare_parameter("arm_band_m", 0.05)
        self.declare_parameter("arm_band_z_margin_m", 0.01)
        self.declare_parameter("arm_band_forward_margin_m", 0.05)
        self.declare_parameter("arm_band_point_tolerance_m", LAYER_Z_TOLERANCE_M)
        self.declare_parameter(
            "structure_footprint_topic", "/yor/collision_monitor/structure_footprint"
        )
        # Flat [x1, y1, x2, y2, ...] in base_footprint: the body polygon the
        # gate stands the chassis, lift column and camera head on, already
        # grown by its structure margin.
        self.declare_parameter(
            "structure_footprint_xy",
            [0.27, 0.54, -0.27, 0.54, -0.27, -0.54, 0.27, -0.54],
        )
        from navdp_deploy.sources.zed_commlink import ZEDCommlinkSource

        self._source = ZEDCommlinkSource(
            host=str(self.get_parameter("zed_host").value),
            port=int(self.get_parameter("zed_port").value),
            transport=str(self.get_parameter("zed_transport").value),
            published_color_order=str(
                self.get_parameter("published_color_order").value
            ),
        )
        self._stride = int(self.get_parameter("point_stride").value)
        self._max_depth = float(self.get_parameter("max_depth_m").value)
        self._calibration_size = (
            int(self.get_parameter("calibration_width").value),
            int(self.get_parameter("calibration_height").value),
        )
        self._intrinsics = np.asarray(
            [
                float(self.get_parameter("camera_fx").value),
                float(self.get_parameter("camera_fy").value),
                float(self.get_parameter("camera_cx").value),
                float(self.get_parameter("camera_cy").value),
            ],
            dtype=np.float64,
        )
        self._fallback_height = float(
            self.get_parameter("fallback_camera_height_m").value
        )
        self._fallback_down = np.asarray(
            self.get_parameter("fallback_ground_down_xyz").value,
            dtype=np.float64,
        )
        self._ground_gate = GroundPlaneGate(
            self._fallback_height,
            self._fallback_down,
            max_height_step_m=float(
                self.get_parameter("ground_plane_max_height_step_m").value
            ),
            max_tilt_step_deg=float(
                self.get_parameter("ground_plane_max_tilt_step_deg").value
            ),
            settle_s=float(self.get_parameter("ground_plane_settle_s").value),
            max_height_error_m=float(
                self.get_parameter("ground_plane_max_height_error_m").value
            ),
            max_tilt_error_deg=float(
                self.get_parameter("ground_plane_max_tilt_error_deg").value
            ),
        )
        self._ground_plane_max_age_s = float(
            self.get_parameter("ground_plane_max_age_s").value
        )
        if not 0.2 <= self._ground_plane_max_age_s <= 10.0:
            raise ValueError("ground_plane_max_age_s must be in [0.2, 10]")
        self._last_ground_plane_log: float | None = None
        self._base_to_camera = np.asarray(
            [
                float(self.get_parameter("base_to_camera_forward_m").value),
                float(self.get_parameter("base_to_camera_left_m").value),
            ],
            dtype=np.float64,
        )
        self._twist_alpha = float(
            self.get_parameter("twist_smoothing_alpha").value
        )
        self._self_filter: RobotDepthSelfFilter | None = None
        self._self_filter_enabled = bool(
            self.get_parameter("self_filter_enabled").value
        )
        self._arm_status_poll_hz = float(
            self.get_parameter("self_filter_arm_status_poll_hz").value
        )
        if not math.isfinite(self._arm_status_poll_hz) or not (
            0.2 <= self._arm_status_poll_hz <= 20.0
        ):
            raise ValueError(
                "self-filter arm status poll rate must be in [0.2, 20] Hz"
            )
        self._arm_status_lock = threading.Lock()
        self._arm_status: dict[str, Any] | None = None
        self._arm_status_timestamp = 0.0
        self._arm_status_error: str | None = None
        self._attached_lock = threading.Lock()
        self._attached_objects: dict[str, Any] = {}
        self._arm_poll_stop = threading.Event()
        self._arm_poll_thread: threading.Thread | None = None
        self._last_filter_warning = 0.0
        self._last_filter_rebuild_log = 0.0
        if self._self_filter_enabled:
            config = RobotSelfFilterConfig.from_mapping(
                {
                    "enabled": True,
                    "urdf_path": self.get_parameter(
                        "self_filter_urdf_path"
                    ).value,
                    "spheres_path": self.get_parameter(
                        "self_filter_spheres_path"
                    ).value,
                    "sphere_erosion_m": self.get_parameter(
                        "self_filter_sphere_erosion_m"
                    ).value,
                    "depth_tolerance_m": self.get_parameter(
                        "self_filter_depth_tolerance_m"
                    ).value,
                    "joint_change_tolerance_rad": self.get_parameter(
                        "self_filter_joint_change_tolerance_rad"
                    ).value,
                    "require_both_arms": self.get_parameter(
                        "self_filter_require_both_arms"
                    ).value,
                    "mask_padding_m": self.get_parameter(
                        "self_filter_mask_padding_m"
                    ).value,
                }
            )
            if config is None:
                raise RuntimeError("enabled robot self-filter has no config")
            calibrations = {
                name: np.asarray(
                    self.get_parameter(f"{name}_arm_from_camera").value,
                    dtype=np.float64,
                ).reshape(4, 4)
                for name in ("left", "right")
            }
            self._self_filter = RobotDepthSelfFilter(
                config, arm_from_camera=calibrations
            )
            self._set_attached_objects(
                str(
                    self.get_parameter(
                        "self_filter_attached_objects_json"
                    ).value
                )
            )
            self.add_on_set_parameters_callback(self._on_set_parameters)
            self._arm_poll_thread = threading.Thread(
                target=self._poll_arm_status,
                name="yor-zed-arm-status",
                daemon=True,
            )
            self._arm_poll_thread.start()
        self._last_pose_sample: tuple[int, float, float, float] | None = None
        self._filtered_twist = np.zeros(3, dtype=np.float64)
        self._cloud_pub = self.create_publisher(
            PointCloud2, "/yor/zed/points", qos_profile_sensor_data
        )
        self._arm_band_enabled = bool(
            self.get_parameter("arm_band_enabled").value
        )
        self._arm_band_z_range = (
            float(self.get_parameter("arm_band_z_min_m").value),
            float(self.get_parameter("arm_band_z_max_m").value),
        )
        self._arm_band_m = float(self.get_parameter("arm_band_m").value)
        self._arm_band_z_margin_m = float(
            self.get_parameter("arm_band_z_margin_m").value
        )
        self._arm_band_forward_margin_m = float(
            self.get_parameter("arm_band_forward_margin_m").value
        )
        self._arm_band_point_tolerance_m = float(
            self.get_parameter("arm_band_point_tolerance_m").value
        )
        if (
            not 0.0 <= self._arm_band_z_range[0] < self._arm_band_z_range[1] <= 3.0
            or not 0.01 <= self._arm_band_m <= 0.50
            or not 0.0 <= self._arm_band_z_margin_m <= 0.20
            or not 0.0 <= self._arm_band_forward_margin_m <= 0.30
            or not 0.0 <= self._arm_band_point_tolerance_m <= 0.20
        ):
            raise ValueError("invalid ZED bridge arm-band settings")
        self._arm_cloud_pub = self.create_publisher(
            PointCloud2,
            str(self.get_parameter("arm_band_cloud_topic").value),
            qos_profile_sensor_data,
        )
        structure_flat = np.asarray(
            self.get_parameter("structure_footprint_xy").value, dtype=np.float64
        ).reshape(-1)
        # Fail closed: the structure monitor's approach polygon is fed only by
        # the topic this node publishes, and Humble's Polygon silently leaves
        # an approach polygon empty (and therefore inert) when no footprint
        # arrives. A missing outline here would remove that guard without any
        # error, so refuse to start instead. The ZED cloud then stops too and
        # the Pi base watchdog holds the base.
        if (
            structure_flat.size < 6
            or structure_flat.size % 2 != 0
            or not np.all(np.isfinite(structure_flat))
        ):
            raise ValueError(
                "structure_footprint_xy must be a flat list of at least three "
                "finite [x, y] pairs; it is generated by yor_agent.nav2_params"
            )
        self._structure_polygon = structure_flat.reshape(-1, 2)
        self._structure_front_m = float(np.max(self._structure_polygon[:, 0]))
        # The collision monitor's approach polygon can only take a footprint
        # topic (nav2_collision_monitor Polygon::getParameters returns early
        # for APPROACH), and nav2_costmap_2d's FootprintSubscriber uses the
        # system default QoS, so publish reliably rather than as sensor data.
        self._structure_polygon_pub = self.create_publisher(
            PolygonStamped,
            str(self.get_parameter("structure_footprint_topic").value),
            10,
        )
        self._last_arm_band_log: tuple[str, float] | None = None
        self._odom_pub = self.create_publisher(Odometry, "/yor/odom", 10)
        self._tf = TransformBroadcaster(self)
        publish_hz = float(self.get_parameter("publish_hz").value)
        if (
            self._stride < 1
            or publish_hz <= 0.0
            or self._max_depth <= 0.0
            or not np.all(np.isfinite(self._base_to_camera))
            or not 0.0 < self._twist_alpha <= 1.0
        ):
            raise ValueError("invalid ZED bridge publish settings")
        self.create_timer(1.0 / publish_hz, self._publish_latest)
        self.create_timer(1.0 / publish_hz, self._publish_structure_footprint)

    def _base_pose_and_twist(self, pose) -> tuple[float, float, float, np.ndarray]:
        """Convert the left-camera pose to the swerve-center pose and twist."""

        yaw = float(pose.yaw_rad)
        cosine = math.cos(yaw)
        sine = math.sin(yaw)
        offset_forward, offset_left = self._base_to_camera
        offset_world_x = cosine * offset_forward - sine * offset_left
        offset_world_y = sine * offset_forward + cosine * offset_left
        base_x = float(pose.x_m) - offset_world_x
        base_y = float(pose.y_m) - offset_world_y
        timestamp_ns = int(pose.timestamp_ns)

        previous = self._last_pose_sample
        if previous is not None and timestamp_ns > previous[0]:
            dt = (timestamp_ns - previous[0]) * 1e-9
            if 0.01 <= dt <= 0.5:
                delta_yaw = math.atan2(
                    math.sin(yaw - previous[3]), math.cos(yaw - previous[3])
                )
                world_vx = (base_x - previous[1]) / dt
                world_vy = (base_y - previous[2]) / dt
                midpoint_yaw = previous[3] + 0.5 * delta_yaw
                midpoint_cosine = math.cos(midpoint_yaw)
                midpoint_sine = math.sin(midpoint_yaw)
                measured = np.asarray(
                    [
                        midpoint_cosine * world_vx + midpoint_sine * world_vy,
                        -midpoint_sine * world_vx + midpoint_cosine * world_vy,
                        delta_yaw / dt,
                    ],
                    dtype=np.float64,
                )
                if np.all(np.isfinite(measured)):
                    alpha = self._twist_alpha
                    self._filtered_twist = (
                        alpha * measured + (1.0 - alpha) * self._filtered_twist
                    )
            else:
                self._filtered_twist.fill(0.0)
        self._last_pose_sample = (timestamp_ns, base_x, base_y, yaw)
        return base_x, base_y, yaw, self._filtered_twist.copy()

    def _publish_latest(self) -> None:
        frame = self._source.latest_frame(max_age_s=0.5)
        if frame is None:
            return
        pose = frame.planar_pose
        if pose is None or not pose.valid:
            self.get_logger().warning("ZED tracking pose unavailable")
            return
        stamp = self.get_clock().now().to_msg()
        base_x, base_y, base_yaw, twist = self._base_pose_and_twist(pose)
        yaw_half = base_yaw / 2.0
        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = "odom"
        odom.child_frame_id = "base_footprint"
        odom.pose.pose.position.x = base_x
        odom.pose.pose.position.y = base_y
        odom.pose.pose.orientation.z = math.sin(yaw_half)
        odom.pose.pose.orientation.w = math.cos(yaw_half)
        # nav_msgs/Odometry requires Twist in child_frame_id coordinates.
        odom.twist.twist.linear.x = float(twist[0])
        odom.twist.twist.linear.y = float(twist[1])
        odom.twist.twist.angular.z = float(twist[2])
        self._odom_pub.publish(odom)
        base_tf = TransformStamped()
        base_tf.header = odom.header
        base_tf.child_frame_id = "base_footprint"
        base_tf.transform.translation.x = base_x
        base_tf.transform.translation.y = base_y
        base_tf.transform.rotation = odom.pose.pose.orientation
        self._tf.sendTransform(base_tf)

        # A frame without a fresh plane offers nothing and the gate's plane,
        # the fallback until the SDK convinces it otherwise, stays in use.
        plane_stamp = getattr(frame, "ground_plane_timestamp_ns", None)
        plane_age_s = (
            (int(frame.timestamp_ns) - int(plane_stamp)) * 1e-9
            if plane_stamp is not None
            else 0.0
        )
        if (
            frame.ground_camera_height_m is not None
            and frame.ground_down_camera_xyz is not None
            and plane_age_s <= self._ground_plane_max_age_s
        ):
            decision = self._ground_gate.offer(
                float(frame.ground_camera_height_m),
                np.asarray(frame.ground_down_camera_xyz, dtype=np.float64),
                int(frame.timestamp_ns) * 1e-9,
            )
            self._log_ground_plane(decision)
        height = self._ground_gate.camera_height_m
        down_vector = self._ground_gate.down_camera_xyz
        optical_forward = np.asarray([0.0, 0.0, 1.0])
        forward = optical_forward - down_vector * float(
            np.dot(optical_forward, down_vector)
        )
        forward /= np.linalg.norm(forward)
        left = np.cross(forward, down_vector)
        left /= np.linalg.norm(left)
        camera_from_optical = np.stack((forward, left, -down_vector), axis=0)
        qx, qy, qz, qw = _quaternion_from_matrix(camera_from_optical)
        camera_tf = TransformStamped()
        camera_tf.header.stamp = stamp
        camera_tf.header.frame_id = "base_footprint"
        camera_tf.child_frame_id = "zed_left_camera_optical_frame"
        camera_tf.transform.translation.x = float(self._base_to_camera[0])
        camera_tf.transform.translation.y = float(self._base_to_camera[1])
        camera_tf.transform.translation.z = float(height)
        camera_tf.transform.rotation.x = qx
        camera_tf.transform.rotation.y = qy
        camera_tf.transform.rotation.z = qz
        camera_tf.transform.rotation.w = qw
        self._tf.sendTransform(camera_tf)
        cloud, arm_cloud = self._point_cloud(
            frame.depth_m, stamp, float(height), camera_from_optical
        )
        if cloud is not None:
            self._cloud_pub.publish(cloud)
        if arm_cloud is not None:
            self._arm_cloud_pub.publish(arm_cloud)

    def _publish_structure_footprint(self) -> None:
        """Publish the body outline the structure collision monitor sweeps.

        On its own timer, not with the cloud: the outline is static in
        ``base_footprint`` and an approach polygon whose footprint stops
        arriving is silently inert, so it must survive a ZED stall.
        """

        message = PolygonStamped()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = "base_footprint"
        message.polygon.points = [
            Point32(x=float(x), y=float(y), z=0.0)
            for x, y in self._structure_polygon
        ]
        self._structure_polygon_pub.publish(message)

    def _arm_band_selection(
        self,
        points_optical: np.ndarray,
        camera_height_m: float,
        camera_from_optical: np.ndarray,
    ) -> tuple[np.ndarray | None, str]:
        """Mask of cloud points at arm-reach height, and why it is what it is.

        ``None`` means "publish everything", which is what the arms monitor
        saw before this split existed: any doubt about where the arms are
        keeps the conservative behaviour.
        """

        if not self._arm_band_enabled:
            return None, "disabled"
        if self._self_filter is None:
            return None, "self_filter_off"
        with self._attached_lock:
            attached = bool(self._attached_objects)
        if attached:
            # A carried object is attached to the arm spheres' envelope but is
            # not in this cloud; do not narrow what the arms monitor sees.
            return None, "attached_object"
        spheres_camera = np.asarray(
            self._self_filter.camera_spheres(), dtype=np.float64
        ).reshape(-1, 4)
        if not len(spheres_camera):
            return None, "no_spheres"
        if len(self._self_filter.camera_sphere_arms()) < 2:
            return None, "single_arm"
        # Same transform the cloud points get, off the same floor plane, so an
        # error in it moves the arms and the obstacles together.
        forward, left, down = (
            camera_from_optical[0],
            camera_from_optical[1],
            -camera_from_optical[2],
        )
        centers = spheres_camera[:, :3]
        spheres_base = np.column_stack(
            [
                centers @ forward + self._base_to_camera[0],
                centers @ left + self._base_to_camera[1],
                camera_height_m - centers @ down,
                spheres_camera[:, 3],
            ]
        )
        band = arm_forward_band_range(
            spheres_base,
            z_min=self._arm_band_z_range[0],
            z_max=self._arm_band_z_range[1],
            band_m=self._arm_band_m,
            structure_front_m=self._structure_front_m,
            forward_margin_m=self._arm_band_forward_margin_m,
            z_margin_m=self._arm_band_z_margin_m,
            point_tolerance_m=self._arm_band_point_tolerance_m,
        )
        if band is None:
            # The arms reach no further forward than the structure outline
            # anywhere, so the structure monitor already covers them.
            return np.zeros(len(points_optical), dtype=bool), "no_forward_reach"
        point_z = camera_height_m - points_optical @ down
        return (point_z >= band[0]) & (point_z <= band[1]), (
            f"band[{band[0]:.2f},{band[1]:.2f}]"
        )

    def _log_ground_plane(self, decision: GroundPlaneDecision) -> None:
        if decision.status == ACCEPTED:
            if decision.settled:
                self.get_logger().info(f"ZED floor plane {decision.reason}")
            return
        now = time.monotonic()
        if (
            self._last_ground_plane_log is not None
            and now - self._last_ground_plane_log < 30.0
        ):
            return
        self._last_ground_plane_log = now
        self.get_logger().warning(
            f"ZED floor plane {decision.status} ({decision.reason}); rejected "
            f"camera height {decision.offered_height_m:.2f} m, using "
            f"{decision.camera_height_m:.2f} m confirmed "
            f"{decision.held_for_s:.1f} s ago"
        )

    def _log_arm_band(self, reason: str, kept: int, total: int) -> None:
        now = time.monotonic()
        if (
            self._last_arm_band_log is not None
            and self._last_arm_band_log[0] == reason
            and now - self._last_arm_band_log[1] < 30.0
        ):
            return
        self._last_arm_band_log = (reason, now)
        if reason.startswith("band["):
            self.get_logger().info(
                f"arm-height cloud {reason}: {kept}/{total} points"
            )
        else:
            self.get_logger().warning(
                f"arm-height cloud unavailable ({reason}); the arms collision "
                f"monitor sees the whole cloud ({total} points)"
            )

    def _point_cloud(
        self,
        depth_m: np.ndarray,
        stamp,
        camera_height_m: float,
        camera_from_optical: np.ndarray,
    ) -> tuple[PointCloud2 | None, PointCloud2 | None]:
        depth = np.asarray(depth_m, dtype=np.float32)
        height, width = depth.shape
        calibration_width, calibration_height = self._calibration_size
        fx, fy, cx, cy = self._intrinsics
        fx *= width / calibration_width
        fy *= height / calibration_height
        cx = (cx + 0.5) * width / calibration_width - 0.5
        cy = (cy + 0.5) * height / calibration_height - 0.5
        if self._self_filter is not None:
            with self._arm_status_lock:
                status = self._arm_status
                status_age = time.monotonic() - self._arm_status_timestamp
                status_error = self._arm_status_error
            maximum_age = max(1.0, 3.0 / self._arm_status_poll_hz)
            if status is None or status_age > maximum_age:
                self._warn_filter(
                    "robot self-filter has no fresh arm status; suppressing "
                    f"point cloud ({status_error or 'waiting'})"
                )
                return None, None
            with self._attached_lock:
                attached = {
                    name: {"bounds_local": list(entry["bounds_local"])}
                    for name, entry in self._attached_objects.items()
                }
            intrinsics = np.asarray(
                [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
                dtype=np.float64,
            )
            try:
                rebuilt = self._self_filter.update(
                    status,
                    depth_shape=depth.shape,
                    intrinsics=intrinsics,
                    attached_objects=attached,
                )
                depth = self._self_filter.filtered_depth(depth)
            except Exception as exc:  # noqa: BLE001 - fail closed on geometry
                self._warn_filter(
                    f"robot self-filter failed; suppressing point cloud: {exc}"
                )
                return None, None
            if (
                rebuilt
                and time.monotonic() - self._last_filter_rebuild_log >= 1.0
            ):
                self.get_logger().info(
                    f"robot self-filter rebuilt: {self._self_filter.last_debug}"
                )
                self._last_filter_rebuild_log = time.monotonic()
        rows, columns = np.mgrid[0:height:self._stride, 0:width:self._stride]
        z = depth[rows, columns]
        valid = np.isfinite(z) & (z > 0.10) & (z <= self._max_depth)
        z = z[valid]
        x = (columns[valid].astype(np.float32) - cx) * z / fx
        y = (rows[valid].astype(np.float32) - cy) * z / fy
        points = np.column_stack((x, y, z)).astype(np.float32, copy=False)
        message = _cloud_message(points, stamp)
        try:
            selection, reason = self._arm_band_selection(
                points.astype(np.float64, copy=False),
                camera_height_m,
                camera_from_optical,
            )
        except Exception as exc:  # noqa: BLE001 - never stop feeding the monitor
            # A stale source makes Humble's collision monitor an unguarded
            # pass-through, so the arms topic must keep flowing whatever the
            # band computation does; the whole cloud is the safe content.
            selection, reason = None, f"error:{type(exc).__name__}"
        if selection is None:
            arm_message = message
            kept = int(points.shape[0])
        else:
            arm_points = points[selection]
            arm_message = _cloud_message(arm_points, stamp)
            kept = int(arm_points.shape[0])
        self._log_arm_band(reason, kept, int(points.shape[0]))
        return message, arm_message

    def _warn_filter(self, message: str) -> None:
        now = time.monotonic()
        if now - self._last_filter_warning >= 2.0:
            self.get_logger().warning(message)
            self._last_filter_warning = now

    def _set_attached_objects(self, payload_json: str) -> None:
        payload = json.loads(payload_json)
        if not isinstance(payload, dict):
            raise ValueError("self-filter attached objects must be a JSON object")
        # Validate shape and extents before accepting a live parameter update.
        RobotDepthSelfFilter._attached_key(payload)
        normalized = {
            str(name): {
                "bounds_local": np.asarray(
                    entry["bounds_local"], dtype=np.float64
                ).tolist()
            }
            for name, entry in payload.items()
        }
        with self._attached_lock:
            self._attached_objects = normalized

    def _on_set_parameters(self, parameters) -> SetParametersResult:
        for parameter in parameters:
            if parameter.name != "self_filter_attached_objects_json":
                continue
            try:
                self._set_attached_objects(str(parameter.value))
            except Exception as exc:  # noqa: BLE001 - reject malformed update
                return SetParametersResult(successful=False, reason=str(exc))
        return SetParametersResult(successful=True)

    def _poll_arm_status(self) -> None:
        from commlink import RPCClient

        host = str(self.get_parameter("arm_rpc_host").value)
        port = int(self.get_parameter("arm_rpc_port").value)
        interval = 1.0 / self._arm_status_poll_hz
        client = None
        while not self._arm_poll_stop.is_set():
            started = time.monotonic()
            try:
                if not host:
                    raise RuntimeError("arm_rpc_host is empty")
                if client is None:
                    client = RPCClient(host=host, port=port)
                    client.socket.rcvtimeo = 750
                    client.socket.sndtimeo = 750
                    client.socket.linger = 0
                status = client.get_status()
                if not isinstance(status, dict):
                    raise RuntimeError("arm RPC returned a non-mapping status")
                with self._arm_status_lock:
                    self._arm_status = status
                    self._arm_status_timestamp = time.monotonic()
                    self._arm_status_error = None
            except Exception as exc:  # noqa: BLE001 - retried, point cloud fails closed
                with self._arm_status_lock:
                    self._arm_status_error = f"{type(exc).__name__}: {exc}"
                if client is not None:
                    self._close_rpc_client(client)
                    client = None
            remaining = max(0.0, interval - (time.monotonic() - started))
            self._arm_poll_stop.wait(remaining)
        if client is not None:
            self._close_rpc_client(client)

    @staticmethod
    def _close_rpc_client(client: Any) -> None:
        attributes = getattr(client, "__dict__", {})
        socket = attributes.get("socket")
        context = attributes.get("context")
        if socket is not None:
            socket.close(linger=0)
        if context is not None:
            context.term()

    def destroy_node(self) -> bool:
        self._arm_poll_stop.set()
        if self._arm_poll_thread is not None:
            self._arm_poll_thread.join(timeout=1.5)
        self._source.close()
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ZedBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
