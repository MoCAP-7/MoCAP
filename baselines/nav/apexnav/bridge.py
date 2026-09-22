"""ROS2 bridge between ApexNav and YOR's atomic ZED / leased base services."""

from __future__ import annotations

import array
import json
import math
from pathlib import Path
import threading
import time
from typing import Any

import cv2
from geometry_msgs.msg import PoseStamped, TransformStamped, Twist
from nav_msgs.msg import Odometry
import numpy as np
from plan_env.msg import MultipleMasksWithConfidence
import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CompressedImage, Image, PointCloud2, PointField
from std_msgs.msg import Bool, Float64, Int32, String
from tf2_msgs.msg import TFMessage

from .clients import ApexNavModelClients
from .config import ApexNavConfig
from .geometry import (
    base_pose_from_camera,
    camera_rotation_world,
    masked_depth_to_world,
    relative_camera_pose,
    rotation_matrix_to_quaternion,
    scaled_intrinsics,
    wrap_angle,
)
from .priors import TargetPrior


class YorApexNavBridge(Node):
    """Own the complete robot-specific boundary of the ApexNav baseline."""

    def __init__(
        self,
        config: ApexNavConfig,
        prior: TargetPrior,
        output_dir: Path,
        *,
        source: Any | None = None,
        rpc: Any | None = None,
        models: ApexNavModelClients | None = None,
        enable_motion: bool = True,
        external_control: bool = False,
        sensor_log: Any | None = None,
    ) -> None:
        super().__init__("yor_apexnav_bridge")
        self.config = config
        self.prior = prior
        self.output_dir = output_dir
        # Receives one record per published odometry sample.
        self._sensor_log = sensor_log
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._trace_path = output_dir / "trace.jsonl"
        self._trace_lock = threading.Lock()
        self._perception_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._enable_motion = bool(enable_motion)
        self._external_control = bool(external_control)
        if self._external_control and not self._enable_motion:
            raise ValueError("external_control requires enable_motion")
        self._closed = False

        if source is None:
            from navdp_deploy.sources.zed_commlink import ZEDCommlinkSource

            source = ZEDCommlinkSource(
                host=config.robot.zed_host,
                port=config.robot.zed_port,
                transport="atomic",
                published_color_order="auto",
            )
        self._source = source
        if rpc is None and self._enable_motion and not self._external_control:
            from commlink import RPCClient

            rpc = RPCClient(
                host=config.robot.base_rpc_host,
                port=config.robot.base_rpc_port,
            )
            # commlink's default ZeroMQ REQ socket waits forever. Bound every
            # base operation so an offline Pi cannot trap startup or Ctrl-C.
            import zmq

            timeout_ms = int(round(config.robot.base_rpc_timeout_s * 1000.0))
            rpc.socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
            rpc.socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
            rpc.socket.setsockopt(zmq.LINGER, 0)
        self._rpc = rpc
        self._models = models or ApexNavModelClients(config.vlm)

        initial = self._source.next_frame(timeout_s=5.0)
        self._validate_frame(initial, require_ground=True)
        raw_pose = initial.planar_pose
        self._origin_pose = np.asarray(
            [raw_pose.x_m, raw_pose.y_m, raw_pose.yaw_rad], dtype=np.float64
        )
        self._latest_sensor_monotonic = time.monotonic()
        self._last_frame_timestamp_ns = -1
        self._last_odometry_timestamp_ns = -1
        self._last_perception_timestamp_ns = -1
        self._last_base_sample: tuple[int, np.ndarray] | None = None
        self._filtered_twist = np.zeros(3, dtype=np.float64)
        # Velocity is differenced over the mapping period, the interval its
        # filter was designed for, although odometry is published faster.
        self._twist_interval_s = 1.0 / config.camera.mapping_hz

        self._command = np.zeros(3, dtype=np.float64)
        self._command_received_at = 0.0
        self._command_active = False
        self._zero_pending = False
        self._sequence = time.time_ns()
        if self._enable_motion and not self._external_control:
            self._preflight_base()

        self.final_state: int | None = None
        self.exploration_result: int | None = None
        self.started = False
        self.trigger_acknowledged = False
        self.trigger_attempts = 0
        self.last_trigger_monotonic = 0.0
        self.consecutive_perception_errors = 0
        self.fatal_error: str | None = None
        self.command_count = 0
        self.nonzero_command_count = 0
        self._traced_first_nonzero_command = False
        self.map_free_cell_count = 0
        self.map_ready = False
        self.initial_scan_complete = False
        self.initial_scan_failed = False
        self.finished = threading.Event()

        self._depth_pub = self.create_publisher(
            Image, "/apexnav/depth_normalized", qos_profile_sensor_data
        )
        self._rgb_pub = self.create_publisher(
            Image, "/apexnav/rgb", qos_profile_sensor_data
        )
        self._camera_pose_pub = self.create_publisher(
            Odometry, "/apexnav/sensor_pose", qos_profile_sensor_data
        )
        self._odom_pub = self.create_publisher(Odometry, "/apexnav/odom", 10)
        self._confidence_pub = self.create_publisher(
            Float64, "/apexnav/detector/confidence_threshold", 10
        )
        self._itm_pub = self.create_publisher(Float64, "/apexnav/blip2/cosine_score", 10)
        self._objects_pub = self.create_publisher(
            MultipleMasksWithConfidence,
            "/apexnav/detector/clouds_with_scores",
            10,
        )
        # Viewer-only outputs. Both are built only while something subscribes,
        # and neither can end the episode or stop the base when it fails.
        self._detection_jpeg_pub = self.create_publisher(
            CompressedImage, "/apexnav/detector/detect_img/compressed", 1
        )
        self._tf_pub = (
            self.create_publisher(TFMessage, "/tf", 10)
            if config.viewer.enabled and config.viewer.publish_tf
            else None
        )
        self._label_pub = self.create_publisher(String, "/apexnav/detector/label", 1)
        self._trigger_pub = self.create_publisher(PoseStamped, "/apexnav/start", 1)
        control_group = MutuallyExclusiveCallbackGroup()
        # Odometry, the depth/pose pair, and perception each run in their own
        # mutually exclusive group: a slow callback cannot delay odometry, and
        # no timer can overlap itself and publish frames out of order.
        odometry_group = MutuallyExclusiveCallbackGroup()
        mapping_group = MutuallyExclusiveCallbackGroup()
        perception_group = MutuallyExclusiveCallbackGroup()
        if not self._external_control:
            self.create_subscription(
                Twist,
                "/apexnav/cmd_vel_raw",
                self._on_twist,
                10,
                callback_group=control_group,
            )
        self.create_subscription(Int32, "/apexnav/ros/state", self._on_state, 10)
        self.create_subscription(
            Int32, "/apexnav/ros/expl_result", self._on_result, 10
        )
        self.create_subscription(PointCloud2, "/grid_map/free", self._on_free_map, 10)
        self.create_subscription(
            Bool,
            "/apexnav/initial_scan_status",
            self._on_initial_scan_status,
            QoSProfile(
                depth=1,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
            ),
        )
        self.create_timer(
            1.0 / config.camera.odometry_hz,
            self._publish_odometry,
            callback_group=odometry_group,
        )
        self.create_timer(
            1.0 / config.camera.mapping_hz,
            self._publish_latest_sensor,
            callback_group=mapping_group,
        )
        self.create_timer(
            1.0 / config.camera.perception_hz,
            self._run_perception,
            callback_group=perception_group,
        )
        if not self._external_control:
            self.create_timer(
                1.0 / config.robot.control_hz,
                self._renew_base_lease,
                callback_group=control_group,
            )
        self._trace(
            "bridge_initialized",
            origin_pose=self._origin_pose.tolist(),
            prior=prior.as_dict(),
            enable_motion=self._enable_motion,
        )

    def start_episode(self) -> None:
        if not self.started:
            label = String()
            label.data = self.prior.target
            self._label_pub.publish(label)
            self._publish_confidence()
            self.started = True
        trigger = PoseStamped()
        trigger.header.stamp = self.get_clock().now().to_msg()
        trigger.header.frame_id = "world"
        self._trigger_pub.publish(trigger)
        self.trigger_attempts += 1
        self.last_trigger_monotonic = time.monotonic()
        self._trace(
            "episode_trigger_published",
            target=self.prior.target,
            attempt=self.trigger_attempts,
            subscription_count=self._trigger_pub.get_subscription_count(),
        )

    def stop(self, reason: str) -> None:
        self._stop_base()
        self._trace(
            "episode_stopped",
            reason=reason,
            final_state=self.final_state,
            exploration_result=self.exploration_result,
        )

    def _validate_frame(self, frame: Any, *, require_ground: bool) -> None:
        if frame is None:
            raise RuntimeError("ZED returned no frame")
        validated = getattr(frame, "validated", None)
        if callable(validated):
            validated()
        if frame.planar_pose is None or not frame.planar_pose.valid:
            raise RuntimeError("ZED tracking pose is unavailable")
        if np.asarray(frame.rgb).ndim != 3 or np.asarray(frame.depth_m).ndim != 2:
            raise RuntimeError("ZED frame does not contain RGB plus 2-D metric depth")
        if np.asarray(frame.rgb).shape[:2] != np.asarray(frame.depth_m).shape:
            raise RuntimeError("ZED RGB and depth shapes differ")
        if require_ground and self.config.camera.require_dynamic_ground_plane:
            if frame.ground_camera_height_m is None or frame.ground_down_camera_xyz is None:
                raise RuntimeError("ApexNav requires a valid dynamic ZED ground plane")
            ground_stamp = getattr(frame, "ground_plane_timestamp_ns", None)
            if ground_stamp is None:
                raise RuntimeError("dynamic ground plane has no timestamp")
            age_s = max(0.0, (int(frame.timestamp_ns) - int(ground_stamp)) * 1e-9)
            if age_s > self.config.camera.ground_plane_max_age_s:
                raise RuntimeError(f"dynamic ground plane is stale by {age_s:.2f} s")

    def _frame_geometry(
        self, frame: Any
    ) -> tuple[np.ndarray, float, np.ndarray, np.ndarray, np.ndarray]:
        pose = frame.planar_pose
        camera_pose = relative_camera_pose(
            np.asarray([pose.x_m, pose.y_m, pose.yaw_rad]), self._origin_pose
        )
        height = frame.ground_camera_height_m
        down = frame.ground_down_camera_xyz
        if height is None or down is None:
            height = self.config.camera.fallback_camera_height_m
            down = self.config.camera.fallback_ground_down_xyz
        rotation = camera_rotation_world(camera_pose[2], down)
        camera_xyz = np.asarray([camera_pose[0], camera_pose[1], float(height)])
        intrinsics = scaled_intrinsics(
            np.asarray(self.config.camera.intrinsics, dtype=np.float64),
            self.config.camera.calibration_resolution,
            np.asarray(frame.depth_m).shape,
        )
        return camera_pose, float(height), np.asarray(down), rotation, intrinsics

    def _publish_odometry(self) -> None:
        """Publish base odometry for every new ZED pose, apart from mapping data."""

        try:
            started = time.monotonic()
            frame = self._source.latest_frame(max_age_s=self.config.robot.sensor_timeout_s)
            if frame is None or int(frame.timestamp_ns) <= self._last_odometry_timestamp_ns:
                return
            frame_age = getattr(self._source, "latest_frame_age_s", None)
            arrival_age_s = frame_age() if callable(frame_age) else None
            pose = frame.planar_pose
            if pose is None or not pose.valid:
                raise RuntimeError("ZED tracking pose is unavailable")
            timestamp_ns = int(frame.timestamp_ns)
            self._last_odometry_timestamp_ns = timestamp_ns
            self._latest_sensor_monotonic = time.monotonic()
            camera_pose = relative_camera_pose(
                np.asarray([pose.x_m, pose.y_m, pose.yaw_rad]), self._origin_pose
            )
            odom = self._odom_message(camera_pose, timestamp_ns, self._capture_stamp(timestamp_ns))
            self._odom_pub.publish(odom)
            published_wall_ns = time.time_ns()
            if self._sensor_log is not None:
                ground_stamp = getattr(frame, "ground_plane_timestamp_ns", None)
                self._sensor_log.write(
                    {
                        "event": "odom",
                        "frame_timestamp_ns": timestamp_ns,
                        "publish_stamp_ns": timestamp_ns,
                        "publish_wall_ns": published_wall_ns,
                        "arrival_age_s": arrival_age_s,
                        "callback_s": time.monotonic() - started,
                        "ground_plane_age_s": None
                        if ground_stamp is None
                        else (timestamp_ns - int(ground_stamp)) * 1e-9,
                        "camera_pose": [float(value) for value in camera_pose],
                        "camera_height_m": getattr(frame, "ground_camera_height_m", None),
                        "base_xy": [
                            odom.pose.pose.position.x,
                            odom.pose.pose.position.y,
                        ],
                        "twist": [float(value) for value in self._filtered_twist],
                    }
                )
            self._publish_viewer_transform(odom)
        except Exception as exc:  # noqa: BLE001 - node must fail safe and keep logging
            self.get_logger().error(f"odometry bridge error: {exc}")
            self._trace("sensor_error", error=f"{type(exc).__name__}: {exc}")
            self._stop_base()

    def _publish_latest_sensor(self) -> None:
        """Publish the depth image and camera pose that ApexNav's map pairs by stamp."""

        try:
            started = time.monotonic()
            frame = self._source.latest_frame(max_age_s=self.config.robot.sensor_timeout_s)
            if frame is None or int(frame.timestamp_ns) <= self._last_frame_timestamp_ns:
                return
            self._validate_frame(frame, require_ground=True)
            timestamp_ns = int(frame.timestamp_ns)
            self._last_frame_timestamp_ns = timestamp_ns
            stamp = self._capture_stamp(timestamp_ns)
            camera_pose, height, _down, rotation, _intrinsics = self._frame_geometry(frame)
            depth = self._depth_message(frame.depth_m, stamp)
            built = time.monotonic()
            self._depth_pub.publish(depth)
            self._camera_pose_pub.publish(
                self._camera_pose_message(camera_pose, height, rotation, stamp)
            )
            # No ApexNav node reads RGB, so it is only built for a subscriber
            # such as a viewer.
            rgb_published = self._rgb_pub.get_subscription_count() > 0
            if rgb_published:
                self._rgb_pub.publish(self._image_message(frame.rgb, "rgb8", stamp))
            if self._sensor_log is not None:
                self._sensor_log.write(
                    {
                        "event": "mapping",
                        "frame_timestamp_ns": timestamp_ns,
                        "build_s": built - started,
                        "publish_s": time.monotonic() - built,
                        "rgb_published": rgb_published,
                    }
                )
            self._publish_confidence()
        except Exception as exc:  # noqa: BLE001 - node must fail safe and keep logging
            self.get_logger().error(f"sensor bridge error: {exc}")
            self._trace("sensor_error", error=f"{type(exc).__name__}: {exc}")
            self._stop_base()

    def _run_perception(self) -> None:
        if not self._perception_lock.acquire(blocking=False):
            return
        try:
            frame = self._source.latest_frame(max_age_s=self.config.robot.sensor_timeout_s)
            if frame is None or int(frame.timestamp_ns) == self._last_perception_timestamp_ns:
                return
            self._validate_frame(frame, require_ground=True)
            self._last_perception_timestamp_ns = int(frame.timestamp_ns)
            started = time.monotonic()
            rgb = np.ascontiguousarray(frame.rgb, dtype=np.uint8)
            detections = self._models.detect_and_segment(
                rgb, self.prior.target, self.prior.similar_labels
            )
            camera_pose, height, _down, rotation, intrinsics = self._frame_geometry(frame)
            camera_xyz = np.asarray([camera_pose[0], camera_pose[1], height])
            stamp = self.get_clock().now().to_msg()
            message = MultipleMasksWithConfidence()
            for detection in detections:
                points = masked_depth_to_world(
                    frame.depth_m,
                    detection.mask,
                    intrinsics,
                    camera_xyz,
                    rotation,
                    minimum_depth_m=self.config.camera.minimum_depth_m,
                    maximum_depth_m=self.config.camera.maximum_depth_m,
                    stride=self.config.camera.object_cloud_stride,
                    maximum_points=self.config.camera.maximum_object_points,
                )
                if len(points) == 0:
                    continue
                message.point_clouds.append(self._point_cloud_message(points, stamp))
                message.confidence_scores.append(float(detection.score))
                message.label_indices.append(int(detection.label_index))
            self._objects_pub.publish(message)
            itm = self._models.image_text_similarity(
                rgb, self.prior.target, self.prior.room
            )
            itm_message = Float64()
            itm_message.data = float(itm)
            self._itm_pub.publish(itm_message)
            annotated = None
            if self.config.experiment.save_detection_images:
                annotated = self._models.annotate(rgb, detections)
                filename = self.output_dir / f"detection_{frame.timestamp_ns}.jpg"
                cv2.imwrite(str(filename), annotated[..., ::-1])
            self._publish_detection_image(rgb, detections, annotated, stamp)
            self._trace(
                "perception",
                timestamp_ns=int(frame.timestamp_ns),
                detection_count=len(message.point_clouds),
                detections=[
                    {
                        "phrase": item.phrase,
                        "score": item.score,
                        "label_index": item.label_index,
                    }
                    for item in detections
                ],
                itm_score=itm,
                elapsed_s=round(time.monotonic() - started, 4),
                model_calls=self._models.calls[-4:],
            )
            self.consecutive_perception_errors = 0
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}"
            self.consecutive_perception_errors += 1
            self.get_logger().error(
                f"perception error ({self.consecutive_perception_errors}/"
                f"{self.config.vlm.maximum_consecutive_errors}): {exc}"
            )
            self._trace(
                "perception_error",
                error=error,
                consecutive=self.consecutive_perception_errors,
            )
            if (
                self.consecutive_perception_errors
                >= self.config.vlm.maximum_consecutive_errors
            ):
                self.fatal_error = error
                self.finished.set()
        finally:
            self._perception_lock.release()

    def _publish_detection_image(
        self, rgb: np.ndarray, detections: Any, annotated: np.ndarray | None, stamp: Any
    ) -> None:
        """Send the annotated detection image to a viewer as JPEG while one watches."""

        try:
            if self._detection_jpeg_pub.get_subscription_count() == 0:
                return
            if annotated is None:
                annotated = self._models.annotate(rgb, detections)
            encoded, jpeg = cv2.imencode(".jpg", annotated[..., ::-1])
            if not encoded:
                raise RuntimeError("JPEG encoding failed")
            message = CompressedImage()
            message.header.stamp = stamp
            message.header.frame_id = "zed_left_camera_optical_frame"
            message.format = "jpeg"
            message.data = array.array("B", jpeg.tobytes())
            self._detection_jpeg_pub.publish(message)
        except Exception as exc:  # noqa: BLE001 - not a perception error
            self.get_logger().warning(
                f"viewer detection image skipped: {exc}", throttle_duration_sec=30.0
            )

    def _publish_viewer_transform(self, odom: Odometry) -> None:
        """Publish world to base_footprint for a viewer that follows the robot.

        No ApexNav node reads TF, so this is only sent while something
        subscribes, and a failure is logged without stopping the base.
        """

        if self._tf_pub is None:
            return
        try:
            if self._tf_pub.get_subscription_count() == 0:
                return
            transform = TransformStamped()
            transform.header = odom.header
            transform.child_frame_id = odom.child_frame_id
            transform.transform.translation.x = odom.pose.pose.position.x
            transform.transform.translation.y = odom.pose.pose.position.y
            transform.transform.rotation = odom.pose.pose.orientation
            self._tf_pub.publish(TFMessage(transforms=[transform]))
        except Exception as exc:  # noqa: BLE001 - never stop the base for a viewer
            self.get_logger().warning(
                f"viewer transform skipped: {exc}", throttle_duration_sec=30.0
            )

    def _publish_confidence(self) -> None:
        message = Float64()
        message.data = float(self.prior.confidence_threshold)
        self._confidence_pub.publish(message)

    def _depth_message(self, depth_m: np.ndarray, stamp: Any) -> Image:
        depth = np.asarray(depth_m, dtype=np.float32)
        valid = np.isfinite(depth) & (depth >= self.config.camera.minimum_depth_m)
        # Reserve encoded zero for invalid ZED depth. The Humble patch makes
        # MapROS skip zero instead of interpreting missing pixels as free space.
        normalized = np.zeros_like(depth, dtype=np.float32)
        scale = self.config.camera.maximum_depth_m - self.config.camera.minimum_depth_m
        normalized[valid] = np.clip(
            (depth[valid] - self.config.camera.minimum_depth_m) / scale,
            1.0 / 65535.0,
            1.0,
        )
        return self._image_message(normalized, "32FC1", stamp)

    @staticmethod
    def _capture_stamp(timestamp_ns: int) -> Any:
        """Stamp a message with the ZED capture time of the frame it came from."""

        return Time(nanoseconds=int(timestamp_ns)).to_msg()

    @staticmethod
    def _image_message(image: np.ndarray, encoding: str, stamp: Any) -> Image:
        pixels = np.ascontiguousarray(image)
        message = Image()
        message.header.stamp = stamp
        message.header.frame_id = "world"
        message.height = int(pixels.shape[0])
        message.width = int(pixels.shape[1])
        message.encoding = encoding
        message.is_bigendian = False
        message.step = int(pixels.strides[0])
        # An array.array skips the generated setter's per-byte Python check,
        # which otherwise dominates the cost of publishing a frame.
        message.data = array.array("B", pixels.tobytes())
        return message

    @staticmethod
    def _camera_pose_message(
        camera_pose: np.ndarray, height: float, rotation: np.ndarray, stamp: Any
    ) -> Odometry:
        qx, qy, qz, qw = rotation_matrix_to_quaternion(rotation)
        message = Odometry()
        message.header.stamp = stamp
        message.header.frame_id = "world"
        message.child_frame_id = "zed_left_camera_optical_frame"
        message.pose.pose.position.x = float(camera_pose[0])
        message.pose.pose.position.y = float(camera_pose[1])
        message.pose.pose.position.z = float(height)
        message.pose.pose.orientation.x = qx
        message.pose.pose.orientation.y = qy
        message.pose.pose.orientation.z = qz
        message.pose.pose.orientation.w = qw
        return message

    def _odom_message(self, camera_pose: np.ndarray, timestamp_ns: int, stamp: Any) -> Odometry:
        base_pose = base_pose_from_camera(
            camera_pose,
            self.config.robot.base_to_camera_forward_m,
            self.config.robot.base_to_camera_left_m,
        )
        previous = self._last_base_sample
        update_reference = previous is None or timestamp_ns <= previous[0]
        if not update_reference:
            dt = (timestamp_ns - previous[0]) * 1e-9
            # A sample closer than the design interval keeps the older
            # reference, so faster odometry does not shorten the difference.
            update_reference = dt >= self._twist_interval_s
            if update_reference and dt <= 0.5:
                delta = base_pose - previous[1]
                delta[2] = wrap_angle(delta[2])
                mid_yaw = previous[1][2] + 0.5 * delta[2]
                world_vx, world_vy = delta[0] / dt, delta[1] / dt
                twist = np.asarray(
                    [
                        math.cos(mid_yaw) * world_vx + math.sin(mid_yaw) * world_vy,
                        -math.sin(mid_yaw) * world_vx + math.cos(mid_yaw) * world_vy,
                        delta[2] / dt,
                    ]
                )
                # The base is nonholonomic. Remove lateral pose-estimation
                # noise before passing odometry into ApexNav's car model.
                twist[1] = 0.0
                self._filtered_twist = 0.35 * twist + 0.65 * self._filtered_twist
                self._filtered_twist[1] = 0.0
                self._filtered_twist[0] = float(
                    np.clip(
                        self._filtered_twist[0],
                        -self.config.robot.maximum_linear_mps,
                        self.config.robot.maximum_linear_mps,
                    )
                )
                self._filtered_twist[2] = float(
                    np.clip(
                        self._filtered_twist[2],
                        -self.config.robot.maximum_yaw_rad_s,
                        self.config.robot.maximum_yaw_rad_s,
                    )
                )
        if update_reference:
            self._last_base_sample = (int(timestamp_ns), base_pose.copy())
        message = Odometry()
        message.header.stamp = stamp
        message.header.frame_id = "world"
        message.child_frame_id = "base_footprint"
        message.pose.pose.position.x = float(base_pose[0])
        message.pose.pose.position.y = float(base_pose[1])
        message.pose.pose.orientation.z = math.sin(base_pose[2] / 2.0)
        message.pose.pose.orientation.w = math.cos(base_pose[2] / 2.0)
        message.twist.twist.linear.x = float(self._filtered_twist[0])
        message.twist.twist.linear.y = float(self._filtered_twist[1])
        message.twist.twist.angular.z = float(self._filtered_twist[2])
        return message

    @staticmethod
    def _point_cloud_message(points: np.ndarray, stamp: Any) -> PointCloud2:
        xyz = np.ascontiguousarray(points, dtype=np.float32).reshape(-1, 3)
        message = PointCloud2()
        message.header.stamp = stamp
        message.header.frame_id = "world"
        message.height = 1
        message.width = len(xyz)
        message.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        message.is_bigendian = False
        message.point_step = 12
        message.row_step = 12 * len(xyz)
        message.is_dense = bool(np.all(np.isfinite(xyz)))
        message.data = array.array("B", xyz.tobytes())
        return message

    def _preflight_base(self) -> None:
        status = self._rpc.get_status()
        if not isinstance(status, dict):
            raise RuntimeError("base RPC returned an invalid status")
        if status.get("estop_latched", False):
            raise RuntimeError("Pi base emergency stop is latched")
        if status.get("lease_active", False):
            raise RuntimeError("another process owns the Pi base velocity lease")
        limits = status.get("limits")
        if not isinstance(limits, dict):
            raise RuntimeError("base RPC did not publish velocity limits")
        if float(limits.get("lease_s", 1.0)) > 0.30:
            raise RuntimeError("Pi base lease exceeds the 0.30 s safety bound")

    def _on_twist(self, message: Twist) -> None:
        command = np.asarray(
            [float(message.linear.x), 0.0, float(message.angular.z)], dtype=np.float64
        )
        if not np.all(np.isfinite(command)):
            self.get_logger().error("discarded non-finite ApexNav Twist")
            return
        command[0] = float(
            np.clip(
                command[0],
                -self.config.robot.maximum_linear_mps,
                self.config.robot.maximum_linear_mps,
            )
        )
        command[2] = float(
            np.clip(
                command[2],
                -self.config.robot.maximum_yaw_rad_s,
                self.config.robot.maximum_yaw_rad_s,
            )
        )
        command[0] = self._speed_floor(command[0], self.config.robot.minimum_linear_mps)
        command[2] = self._speed_floor(command[2], self.config.robot.minimum_yaw_rad_s)
        self.command_count += 1
        if np.any(np.abs(command) > 1e-9):
            self.nonzero_command_count += 1
            if not self._traced_first_nonzero_command:
                self._traced_first_nonzero_command = True
                self._trace("first_nonzero_command", command=command.tolist())
        with self._state_lock:
            if np.any(np.abs(command) > 1e-9):
                self._command = command
                self._command_received_at = time.monotonic()
                self._command_active = True
            else:
                self._command.fill(0.0)
                self._command_active = False
                self._zero_pending = True

    @staticmethod
    def _speed_floor(value: float, floor: float) -> float:
        if abs(value) <= 1e-9:
            return 0.0
        return math.copysign(max(abs(value), floor), value)

    def _renew_base_lease(self) -> None:
        if not self._enable_motion or self._rpc is None:
            return
        with self._state_lock:
            now = time.monotonic()
            stale_command = now - self._command_received_at > self.config.robot.command_timeout_s
            stale_sensor = now - self._latest_sensor_monotonic > self.config.robot.sensor_timeout_s
            if self._command_active and (stale_command or stale_sensor):
                self._command_active = False
                self._command.fill(0.0)
                self._zero_pending = True
            if self._command_active:
                command = self._command.copy()
            elif self._zero_pending:
                command = np.zeros(3, dtype=np.float64)
                self._zero_pending = False
            else:
                return
        self._send_base(command)

    def _send_base(self, command: np.ndarray) -> None:
        self._sequence = max(time.time_ns(), self._sequence + 1)
        reply = self._rpc.submit_velocity(command.tolist(), self._sequence)
        if not isinstance(reply, dict) or not reply.get("accepted", False):
            raise RuntimeError(f"base RPC rejected ApexNav velocity: {reply!r}")

    def _stop_base(self) -> None:
        with self._state_lock:
            self._command.fill(0.0)
            self._command_active = False
            self._zero_pending = False
        if self._enable_motion and self._rpc is not None:
            try:
                self._send_base(np.zeros(3, dtype=np.float64))
            except Exception as exc:  # noqa: BLE001
                self.get_logger().error(f"failed to stop base: {exc}")

    def _on_state(self, message: Int32) -> None:
        self.final_state = int(message.data)
        if self.started and self.final_state >= 2:
            self.trigger_acknowledged = True
        if self.final_state == 5:
            self.finished.set()

    def _on_result(self, message: Int32) -> None:
        self.exploration_result = int(message.data)

    def _on_free_map(self, message: PointCloud2) -> None:
        self.map_free_cell_count = int(message.width) * int(message.height)
        if (
            not self.map_ready
            and self.map_free_cell_count >= self.config.planner.minimum_map_free_cells
        ):
            self.map_ready = True
            self._trace("map_ready", free_cell_count=self.map_free_cell_count)

    def _on_initial_scan_status(self, message: Bool) -> None:
        if bool(message.data):
            if not self.initial_scan_complete:
                self.initial_scan_complete = True
                self._trace("initial_scan_complete")
            return
        error = "initial mapping rotation timed out before completing 360 degrees"
        self.get_logger().error(error)
        self._trace("initial_scan_failed", error=error)
        self.initial_scan_failed = True
        if self._enable_motion:
            self.fatal_error = error
        self.finished.set()

    def _trace(self, event: str, **payload: Any) -> None:
        record = {
            "monotonic_s": round(time.monotonic(), 6),
            "event": event,
            **payload,
        }
        with self._trace_lock:
            with self._trace_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, sort_keys=True) + "\n")

    def destroy_node(self) -> bool:
        if self._closed:
            return super().destroy_node()
        self._closed = True
        self._stop_base()
        try:
            self._models.close()
        finally:
            close = getattr(self._source, "close", None)
            if callable(close):
                close()
        return super().destroy_node()
