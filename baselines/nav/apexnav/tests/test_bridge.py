import array
from pathlib import Path
from types import SimpleNamespace
import tempfile
import time
import unittest

import cv2
from geometry_msgs.msg import Twist
import numpy as np
import rclpy
from std_msgs.msg import Bool, Int32

from baselines.nav.apexnav.bridge import YorApexNavBridge
from baselines.nav.apexnav.config import ApexNavConfig, ExperimentConfig, ViewerConfig
from baselines.nav.apexnav.priors import TargetPrior


class _Frame:
    def __init__(self):
        self.rgb = np.zeros((24, 32, 3), dtype=np.uint8)
        self.depth_m = np.full((24, 32), 2.0, dtype=np.float32)
        self.timestamp_ns = 10_000_000_000
        self.planar_pose = SimpleNamespace(
            x_m=1.0, y_m=2.0, yaw_rad=0.25, valid=True
        )
        self.ground_camera_height_m = 1.1
        self.ground_down_camera_xyz = (0.0, 1.0, 0.0)
        self.ground_plane_timestamp_ns = self.timestamp_ns

    def validated(self):
        return self


class _Source:
    def __init__(self):
        self.frame = _Frame()
        self.closed = False

    def next_frame(self, timeout_s):
        return self.frame

    def latest_frame(self, max_age_s):
        return self.frame

    def close(self):
        self.closed = True


class _RPC:
    def __init__(self):
        self.commands = []

    def get_status(self):
        return {
            "estop_latched": False,
            "lease_active": False,
            "limits": {"lease_s": 0.25},
        }

    def submit_velocity(self, velocity, sequence):
        self.commands.append((list(velocity), int(sequence)))
        return {"accepted": True}


class _Models:
    calls = []

    def close(self):
        return None


class _PerceptionModels(_Models):
    def __init__(self, annotate_error=None):
        self.calls = []
        self.annotated = 0
        self._annotate_error = annotate_error

    def detect_and_segment(self, rgb, target, similar_labels):
        return []

    def image_text_similarity(self, rgb, target, room):
        return 0.25

    def annotate(self, rgb, detections):
        self.annotated += 1
        if self._annotate_error is not None:
            raise self._annotate_error
        return np.full_like(rgb, 128)


class BridgeSafetyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def test_velocity_floor_watchdog_and_stop(self):
        source = _Source()
        rpc = _RPC()
        prior = TargetPrior("chair", (), 0.5, "everywhere", "test")
        with tempfile.TemporaryDirectory() as directory:
            node = YorApexNavBridge(
                ApexNavConfig(),
                prior,
                Path(directory),
                source=source,
                rpc=rpc,
                models=_Models(),
            )
            command = Twist()
            command.linear.x = 0.01
            command.angular.z = -0.02
            node._on_twist(command)
            node._renew_base_lease()
            self.assertEqual(rpc.commands[-1][0], [0.05, 0.0, -0.16])

            node._command_received_at = time.monotonic() - 1.0
            node._renew_base_lease()
            self.assertEqual(rpc.commands[-1][0], [0.0, 0.0, 0.0])

            node.start_episode()
            node.start_episode()
            self.assertTrue(node.started)
            self.assertEqual(node.trigger_attempts, 2)
            self.assertGreater(node.last_trigger_monotonic, 0.0)

            state = Int32()
            state.data = 2
            node._on_state(state)
            self.assertTrue(node.trigger_acknowledged)

            free_map = node._point_cloud_message(
                np.zeros((node.config.planner.minimum_map_free_cells, 3)),
                node.get_clock().now().to_msg(),
            )
            node._on_free_map(free_map)
            self.assertTrue(node.map_ready)

            scan_status = Bool()
            scan_status.data = True
            node._on_initial_scan_status(scan_status)
            self.assertTrue(node.initial_scan_complete)

            depth = np.asarray([[np.nan, 0.0, 0.1, 5.0, 6.0]], dtype=np.float32)
            message = node._depth_message(depth, node.get_clock().now().to_msg())
            encoded = np.frombuffer(message.data, dtype=np.float32)
            np.testing.assert_array_equal(encoded[:2], [0.0, 0.0])
            self.assertGreater(encoded[2], 0.0)
            np.testing.assert_array_equal(encoded[3:], [1.0, 1.0])

            camera_pose = np.asarray([0.4, -0.2, 0.0])
            odom = node._odom_message(
                camera_pose, source.frame.timestamp_ns, node.get_clock().now().to_msg()
            )
            self.assertEqual(odom.child_frame_id, "base_footprint")
            np.testing.assert_allclose(
                [odom.pose.pose.position.x, odom.pose.pose.position.y],
                [0.4 - 0.2143, -0.2 - 0.0603],
            )
            fast_odom = node._odom_message(
                np.asarray([0.8, 0.4, 0.2]),
                source.frame.timestamp_ns + 200_000_000,
                node.get_clock().now().to_msg(),
            )
            self.assertLessEqual(
                abs(fast_odom.twist.twist.linear.x),
                node.config.robot.maximum_linear_mps,
            )
            self.assertEqual(fast_odom.twist.twist.linear.y, 0.0)
            node.destroy_node()
        self.assertTrue(source.closed)

    def test_failed_initial_scan_is_fatal_with_motion(self):
        source = _Source()
        rpc = _RPC()
        prior = TargetPrior("chair", (), 0.5, "everywhere", "test")
        with tempfile.TemporaryDirectory() as directory:
            node = YorApexNavBridge(
                ApexNavConfig(),
                prior,
                Path(directory),
                source=source,
                rpc=rpc,
                models=_Models(),
            )
            scan_status = Bool()
            scan_status.data = False
            node._on_initial_scan_status(scan_status)
            self.assertTrue(node.finished.is_set())
            self.assertIn("timed out", node.fatal_error)
            node.destroy_node()

    def test_failed_initial_scan_is_a_no_motion_check_result(self):
        source = _Source()
        prior = TargetPrior("chair", (), 0.5, "everywhere", "test")
        with tempfile.TemporaryDirectory() as directory:
            node = YorApexNavBridge(
                ApexNavConfig(),
                prior,
                Path(directory),
                source=source,
                models=_Models(),
                enable_motion=False,
            )
            scan_status = Bool()
            scan_status.data = False
            node._on_initial_scan_status(scan_status)
            self.assertTrue(node.finished.is_set())
            self.assertTrue(node.initial_scan_failed)
            self.assertIsNone(node.fatal_error)
            node.destroy_node()

    def test_external_control_does_not_own_base_rpc(self):
        source = _Source()
        prior = TargetPrior("chair", (), 0.5, "everywhere", "test")
        with tempfile.TemporaryDirectory() as directory:
            node = YorApexNavBridge(
                ApexNavConfig(),
                prior,
                Path(directory),
                source=source,
                models=_Models(),
                enable_motion=True,
                external_control=True,
            )
            self.assertIsNone(node._rpc)
            node.destroy_node()

    def _node(self, source, *, sensor_log=None, config=None, models=None):
        return YorApexNavBridge(
            config or ApexNavConfig(),
            TargetPrior("chair", (), 0.5, "everywhere", "test"),
            Path(self._directory.name),
            source=source,
            models=models or _Models(),
            enable_motion=False,
            sensor_log=sensor_log,
        )

    def setUp(self):
        self._directory = tempfile.TemporaryDirectory()

    def tearDown(self):
        self._directory.cleanup()

    def test_odometry_is_logged_once_per_new_frame_and_never_goes_backwards(self):
        source = _Source()
        records = []
        log = SimpleNamespace(write=lambda record: records.append(dict(record)))
        node = self._node(source, sensor_log=log)
        published = []
        node._odom_pub = SimpleNamespace(publish=published.append)
        node._publish_odometry()
        # The same frame is not published or logged twice.
        node._publish_odometry()
        # An older frame is dropped instead of moving odometry backwards.
        source.frame.timestamp_ns -= 50_000_000
        node._publish_odometry()
        node.destroy_node()

        self.assertEqual(len(published), 1)
        # Odometry carries the ZED capture time of its frame.
        self.assertEqual(
            (published[0].header.stamp.sec, published[0].header.stamp.nanosec), (10, 0)
        )
        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(record["event"], "odom")
        self.assertEqual(record["frame_timestamp_ns"], 10_000_000_000)
        self.assertEqual(record["publish_stamp_ns"], record["frame_timestamp_ns"])
        self.assertGreater(record["publish_wall_ns"], 0)
        self.assertGreaterEqual(record["callback_s"], 0.0)
        self.assertEqual(record["ground_plane_age_s"], 0.0)
        self.assertEqual(len(record["camera_pose"]), 3)
        self.assertEqual(len(record["twist"]), 3)

    def test_mapping_pairs_depth_and_pose_on_the_capture_stamp_and_skips_unwatched_rgb(self):
        source = _Source()
        records = []
        log = SimpleNamespace(write=lambda record: records.append(dict(record)))
        node = self._node(source, sensor_log=log)
        depth, pose, rgb = [], [], []
        node._depth_pub = SimpleNamespace(publish=depth.append)
        node._camera_pose_pub = SimpleNamespace(publish=pose.append)
        node._rgb_pub = SimpleNamespace(publish=rgb.append, get_subscription_count=lambda: 0)
        node._publish_latest_sensor()
        node._publish_latest_sensor()
        node.destroy_node()

        self.assertEqual((len(depth), len(pose), len(rgb)), (1, 1, 0))
        self.assertEqual(depth[0].header.stamp, pose[0].header.stamp)
        self.assertEqual((depth[0].header.stamp.sec, depth[0].header.stamp.nanosec), (10, 0))
        self.assertIsInstance(depth[0].data, array.array)
        self.assertEqual([record["event"] for record in records], ["mapping"])
        self.assertFalse(records[0]["rgb_published"])

    def test_velocity_is_differenced_over_the_mapping_interval(self):
        node = self._node(_Source())
        stamp = node.get_clock().now().to_msg()
        interval_ns = int(round(1e9 / node.config.camera.mapping_hz))
        start_ns = 20_000_000_000
        node._odom_message(np.zeros(3), start_ns, stamp)
        # A sample inside the interval keeps the older reference.
        node._odom_message(np.asarray([0.005, 0.0, 0.0]), start_ns + interval_ns // 4, stamp)
        self.assertEqual(node._last_base_sample[0], start_ns)
        np.testing.assert_array_equal(node._filtered_twist, np.zeros(3))
        odom = node._odom_message(np.asarray([0.01, 0.0, 0.0]), start_ns + interval_ns, stamp)
        node.destroy_node()

        self.assertEqual(node._last_base_sample[0], start_ns + interval_ns)
        self.assertAlmostEqual(
            odom.twist.twist.linear.x, 0.35 * 0.01 / (interval_ns * 1e-9)
        )

    def test_watched_detection_image_is_published_as_jpeg_and_still_saved(self):
        models = _PerceptionModels()
        node = self._node(_Source(), models=models)
        published = []
        node._detection_jpeg_pub = SimpleNamespace(
            publish=published.append, get_subscription_count=lambda: 1
        )
        node._run_perception()
        node.destroy_node()

        self.assertEqual(node.consecutive_perception_errors, 0)
        self.assertEqual(models.annotated, 1)
        self.assertTrue(
            (Path(self._directory.name) / "detection_10000000000.jpg").is_file()
        )
        self.assertEqual(len(published), 1)
        image = published[0]
        self.assertEqual(
            (image.format, image.header.frame_id), ("jpeg", "zed_left_camera_optical_frame")
        )
        self.assertIsInstance(image.data, array.array)
        decoded = cv2.imdecode(np.frombuffer(image.data, dtype=np.uint8), cv2.IMREAD_COLOR)
        self.assertEqual(decoded.shape, (24, 32, 3))

    def test_viewer_detection_image_never_counts_as_a_perception_error(self):
        source = _Source()
        models = _PerceptionModels(annotate_error=RuntimeError("annotation failed"))
        node = self._node(
            source,
            config=ApexNavConfig(experiment=ExperimentConfig(save_detection_images=False)),
            models=models,
        )
        watchers = [0]
        published = []
        node._detection_jpeg_pub = SimpleNamespace(
            publish=published.append, get_subscription_count=lambda: watchers[0]
        )
        # Nothing is saved and nobody watches, so nothing is annotated.
        node._run_perception()
        self.assertEqual(models.annotated, 0)
        watchers[0] = 1
        for _ in range(node.config.vlm.maximum_consecutive_errors):
            source.frame.timestamp_ns += 100_000_000
            node._run_perception()
        node.destroy_node()

        self.assertEqual(models.annotated, node.config.vlm.maximum_consecutive_errors)
        self.assertEqual(published, [])
        self.assertEqual(node.consecutive_perception_errors, 0)
        self.assertIsNone(node.fatal_error)
        self.assertFalse(node.finished.is_set())

    def test_viewer_transform_follows_odometry_only_while_watched_and_never_stops_the_base(self):
        source = _Source()
        node = self._node(source)
        odometry, transforms, watchers, stops = [], [], [0], []
        node._odom_pub = SimpleNamespace(publish=odometry.append)
        node._tf_pub = SimpleNamespace(
            publish=transforms.append, get_subscription_count=lambda: watchers[0]
        )
        node._stop_base = lambda: stops.append(True)
        node._publish_odometry()
        watchers[0] = 1
        source.frame.timestamp_ns += 33_000_000
        node._publish_odometry()

        def unreachable_viewer(message):
            raise RuntimeError("viewer gone")

        node._tf_pub = SimpleNamespace(
            publish=unreachable_viewer, get_subscription_count=lambda: 1
        )
        source.frame.timestamp_ns += 33_000_000
        node._publish_odometry()
        self.assertEqual(stops, [])
        node.destroy_node()

        self.assertEqual(len(odometry), 3)
        self.assertEqual(len(transforms), 1)
        transform = transforms[0].transforms[0]
        self.assertEqual(
            (transform.header.frame_id, transform.child_frame_id), ("world", "base_footprint")
        )
        self.assertEqual(transform.header.stamp, odometry[1].header.stamp)
        self.assertEqual(
            transform.transform.translation.x, odometry[1].pose.pose.position.x
        )
        self.assertEqual(transform.transform.rotation, odometry[1].pose.pose.orientation)
        trace = (Path(self._directory.name) / "trace.jsonl").read_text(encoding="utf-8")
        self.assertNotIn("sensor_error", trace)

    def test_viewer_transform_publisher_follows_the_configuration(self):
        node = self._node(
            _Source(), config=ApexNavConfig(viewer=ViewerConfig(publish_tf=False))
        )
        node.destroy_node()
        self.assertIsNone(node._tf_pub)


if __name__ == "__main__":
    unittest.main()
