from __future__ import annotations

import base64
import dataclasses
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import io
import json
import math
import socket
import tempfile
import threading
import types
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

from yor_agent.robot import readiness_prior
from yor_agent.robot.readiness_prior import (
    EVENTS_SCHEMA,
    PnPResult,
    ReadinessPrior,
    ReadinessPriorConfig,
    ReadinessPriorEstimate,
)

HAS_CV2 = importlib.util.find_spec("cv2") is not None

# Synthetic ZED (depth resolution) and human (event frame) cameras.
ZED_W, ZED_H = 320, 240
ZED_K = np.array([[300.0, 0.0, 159.5], [0.0, 300.0, 119.5], [0.0, 0.0, 1.0]])
FRAME_W, FRAME_H = 640, 480
SOURCE_W, SOURCE_H = 1280, 960
# Sidecar for the 1280-wide source: halved for the 640-wide frame this gives
# fx = 320 (a 90 deg horizontal FOV) and cx = 319.5.
SIDECAR = {
    "model": "pinhole",
    "width": SOURCE_W,
    "height": SOURCE_H,
    "fx": 640.0,
    "fy": 640.0,
    "cx": 639.5,
    "cy": 479.5,
    "source_path": "/videos/test_undistorted.intrinsics.json",
}
HUMAN_FOCAL = 320.0
# Human camera centre in the ZED optical frame (x right, y down, z forward):
# 0.8 m to the right, 0.45 m higher, 0.6 m ahead of the ZED.
HUMAN_CENTER = np.array([0.8, -0.45, 0.6])
# ZED level: down is +y, so floor-forward is +z and floor-left is -x.
DOWN = np.array([0.0, 1.0, 0.0])
PLANAR_FORWARD = np.array([0.0, 0.0, 1.0])
PLANAR_LEFT = np.array([-1.0, 0.0, 0.0])
POSE = np.array([1.0, -2.0, 0.7])
TARGET_XY = (2.5, -1.0)

DEFAULT_EVENT = {"object": "can", "hand": "right", "t_ready_s": 16.5, "t_grasp_s": 19.0}


def _rot_x(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _rot_y(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


# Human camera axes expressed in the ZED frame: yawed 25 deg to the left
# (toward the scene) and pitched 8 deg down.
HUMAN_R_ZH = _rot_y(math.radians(-25.0)) @ _rot_x(math.radians(8.0))


def expected_world_xy(center_zed: np.ndarray) -> tuple[float, float]:
    forward = float(center_zed @ PLANAR_FORWARD)
    left = float(center_zed @ PLANAR_LEFT)
    yaw = float(POSE[2])
    return (
        float(POSE[0]) + math.cos(yaw) * forward - math.sin(yaw) * left,
        float(POSE[1]) + math.sin(yaw) * forward + math.cos(yaw) * left,
    )


def expected_heading(axis_zed: np.ndarray) -> float:
    angle = math.atan2(float(axis_zed @ PLANAR_LEFT), float(axis_zed @ PLANAR_FORWARD))
    angle += float(POSE[2])
    return math.atan2(math.sin(angle), math.cos(angle))


def wrap(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def make_events_file(
    directory: Path,
    *,
    events=None,
    intrinsics="sidecar",
    frame_size=(FRAME_W, FRAME_H),
    create_frames: bool = True,
    frame_extension: str = "jpg",
    schema: str = EVENTS_SCHEMA,
) -> Path:
    version = "20260911T000000.000000Z"
    frames_dir = f"manipulation_events_{version}_frames"
    (directory / frames_dir).mkdir(parents=True, exist_ok=True)
    payload_events = []
    for index, event in enumerate(events if events is not None else [DEFAULT_EVENT]):
        frames = {}
        for kind, time_s in (
            ("nav", max(0.0, event["t_ready_s"] - 2.0)),
            ("ready", event["t_ready_s"]),
            ("grasp", event["t_grasp_s"]),
        ):
            name = f"{frames_dir}/event{index:02d}_{kind}.{frame_extension}"
            if create_frames:
                (directory / name).write_bytes(b"not a real image")
            frames[kind] = {
                "path": name,
                "time_s": time_s,
                "width": frame_size[0],
                "height": frame_size[1],
                "scale": frame_size[0] / SOURCE_W,
            }
        payload_events.append(
            {**event, "confidence": None, "notes": "", "frames": frames}
        )
    path = directory / f"manipulation_events_{version}.json"
    payload = {
        "schema_version": schema,
        "created_at": "2026-09-11T00:00:00Z",
        "artifact_path": str(path),
        "source_video": {
            "path": "/videos/test_undistorted.mp4",
            "size_bytes": 1,
            "mtime_ns": 1,
            "width": SOURCE_W,
            "height": SOURCE_H,
            "fps": 10.0,
            "duration_s": 43.8,
        },
        "intrinsics": dict(SIDECAR) if intrinsics == "sidecar" else intrinsics,
        "motion": {
            "threshold": 1.0,
            "median_yavg": 1.3,
            "per_second_yavg": [1.0] * 44,
            "windows": [{"start_s": 0.0, "end_s": 2.0}],
        },
        "labeling": {
            "mode": "manual",
            "model": None,
            "prompt_sha256": None,
            "raw_response": None,
        },
        "events": payload_events,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def strip_event_times(events_path: Path) -> None:
    """Remove every frame ``time_s`` and the event ``t_ready_s`` from the file."""

    payload = json.loads(events_path.read_text(encoding="utf-8"))
    for event in payload["events"]:
        event.pop("t_ready_s", None)
        for frame in event["frames"].values():
            frame.pop("time_s", None)
    events_path.write_text(json.dumps(payload), encoding="utf-8")


def build_prior(events_path: Path, matcher=None, transport=None, **overrides) -> ReadinessPrior:
    config = ReadinessPriorConfig.from_mapping(
        {"enabled": True, "events_path": str(events_path), **overrides}
    )
    return ReadinessPrior(config, matcher=matcher, transport=transport)


# --------------------------------------------------------------------- vggt
# Camera rows from the validated zed + nav + ready run (PLAN.md): the ready
# frame 0.526 m ahead / 0.317 m right of the ZED, head 1.472 m above the
# floor, heading 27 deg; scale 2.244 with IQR ratio 1.16.
CAMERA_ROWS = {
    "nav": {
        "label": "nav", "forward_m": -1.554, "left_m": 0.310,
        "height_above_floor_m": 1.558, "heading_deg": -13.0,
        "focal_px": 1101.3, "focal_px_model": 445.5, "width": 1280, "height": 960,
    },
    "ready": {
        "label": "ready", "forward_m": 0.526, "left_m": -0.317,
        "height_above_floor_m": 1.472, "heading_deg": 27.0,
        "focal_px": 1096.4, "focal_px_model": 443.5, "width": 1280, "height": 960,
    },
    "grasp": {
        "label": "grasp", "forward_m": 0.402, "left_m": -0.301,
        "height_above_floor_m": 1.401, "heading_deg": 30.5,
        "focal_px": 1090.0, "focal_px_model": 440.9, "width": 1280, "height": 960,
    },
}
ROBOT_ROW = {
    "label": "zed", "forward_m": 0.004, "left_m": -0.011,
    "height_above_floor_m": 1.062, "heading_deg": -0.1,
    "focal_px": 268.1, "focal_px_model": 206.7, "width": 672, "height": 376,
}
CAMERA_HEIGHT_M = 1.054
STANCE_OFFSET_M = 0.30


def canned_response(labels, *, scale=2.244, scale_iqr_ratio=1.16, rows=None, **overrides):
    """A ``POST /stance`` 200 body with one camera row per label, in order."""

    rows = dict(CAMERA_ROWS, **(rows or {}))
    body = {
        "schema_version": "yor-vggt-stance-v1",
        "device": "mps",
        "dtype": "float32",
        "model_image_hw": [518, 518],
        "content_region": {"rows": [112, 406], "cols": [0, 518]},
        "scale": scale,
        "scale_iqr_ratio": scale_iqr_ratio,
        "scale_pixels": 53524,
        "conf_threshold": 1.0,
        "robot_camera": dict(ROBOT_ROW),
        "cameras": [dict(rows[label]) for label in labels],
        "robot_focal_px": 267.2,
        "timing_s": {
            "decode": 0.05, "preprocess": 0.2, "aggregator": 3.1,
            "camera_head": 0.4, "depth_head": 0.6, "total": 4.4,
        },
        "depth_head_frames": 1,
        "max_memory_mb": 2870.5,
    }
    body.update(overrides)
    return body


CANNED = object()


class FakeTransport:
    """Records every call; answers with a canned body built from the request."""

    def __init__(self, response=CANNED, *, error=None):
        self.calls: list[tuple[str, dict | None, float]] = []
        self.response = response
        self.error = error

    def __call__(self, url, payload, timeout_s):
        self.calls.append((url, payload, timeout_s))
        if self.error is not None:
            raise self.error
        if callable(self.response):
            return self.response(payload)
        if self.response is CANNED:
            return canned_response([frame["label"] for frame in payload["frames"]])
        return self.response


def write_jpeg_frames(events_path: Path) -> dict[str, bytes]:
    """Replace the placeholder frame files with small, distinct real JPEGs."""

    frames_dir = events_path.parent / "manipulation_events_20260911T000000.000000Z_frames"
    written = {}
    for index, label in enumerate(("nav", "ready", "grasp")):
        image = np.zeros((48, 64, 3), dtype=np.uint8)
        image[:, :, index] = 200
        image[10 + index * 5 : 20 + index * 5, 20:40, :] = 30
        buffer = io.BytesIO()
        Image.fromarray(image, "RGB").save(buffer, format="JPEG", quality=90)
        path = frames_dir / f"event00_{label}.jpg"
        path.write_bytes(buffer.getvalue())
        written[label] = buffer.getvalue()
    return written


def planar_world(forward_m: float, left_m: float) -> tuple[float, float]:
    yaw = float(POSE[2])
    return (
        float(POSE[0]) + math.cos(yaw) * forward_m - math.sin(yaw) * left_m,
        float(POSE[1]) + math.sin(yaw) * forward_m + math.cos(yaw) * left_m,
    )


def expected_stance(row, target_xy, *, offset_m=STANCE_OFFSET_M, near_m=0.30):
    """Head, heading, feet and bearing the locked stance rule gives for a row."""

    head = planar_world(row["forward_m"], row["left_m"])
    heading = wrap(math.radians(row["heading_deg"]) + float(POSE[2]))
    feet = (head[0] - offset_m * math.cos(heading), head[1] - offset_m * math.sin(heading))
    distance = math.hypot(head[0] - target_xy[0], head[1] - target_xy[1])
    if distance <= near_m:
        bearing = wrap(heading + math.pi)
    else:
        bearing = wrap(math.atan2(feet[1] - target_xy[1], feet[0] - target_xy[0]))
    return head, heading, feet, bearing


def fake_cv2_module():
    """Enough of cv2 for solve_pose when the matcher and solver are injected."""

    def resize(image, size, interpolation=None):
        width, height = size
        return np.zeros((height, width, image.shape[2]), dtype=image.dtype)

    return types.SimpleNamespace(resize=resize, INTER_AREA=3)


def rvec_from_rotation(rotation: np.ndarray) -> np.ndarray:
    """Axis-angle vector of a rotation matrix (inverse of Rodrigues)."""

    angle = math.acos(max(-1.0, min(1.0, (np.trace(rotation) - 1.0) / 2.0)))
    if angle < 1e-12:
        return np.zeros(3)
    axis = np.array(
        [
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ]
    ) / (2.0 * math.sin(angle))
    return axis * angle


class TempDirMixin:
    def setUp(self) -> None:
        super().setUp()
        self._tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tempdir.cleanup)
        self.directory = Path(self._tempdir.name)


class ReadinessPriorConfigTest(unittest.TestCase):
    def test_defaults_from_none(self) -> None:
        config = ReadinessPriorConfig.from_mapping(None)
        self.assertFalse(config.enabled)
        self.assertIsNone(config.events_path)
        self.assertEqual(config.source, "ready_frame")
        self.assertEqual(config.method, "vggt")
        self.assertEqual(config.vggt_service_url, "http://127.0.0.1:8117")
        self.assertEqual(config.vggt_timeout_s, 60.0)
        self.assertEqual(config.vggt_frames, ("nav", "ready"))
        self.assertEqual(config.stance_offset_m, 0.30)
        self.assertEqual(config.near_object_m, 0.30)
        self.assertEqual(config.min_height_m, 1.2)
        self.assertEqual(config.max_height_m, 2.1)
        self.assertEqual(config.max_scale_iqr_ratio, 2.0)
        self.assertEqual(config.min_inliers, 30)
        self.assertEqual(config.focal_sweep_fov_deg, (60, 70, 80, 90, 100, 110, 120))
        self.assertEqual(config.retry_bearing_offsets_deg, (45.0, -45.0))
        self.assertTrue(config.fallback_to_arrival_bearing)

    def test_fallback_to_arrival_bearing_is_a_boolean_switch(self) -> None:
        config = ReadinessPriorConfig.from_mapping({"fallback_to_arrival_bearing": False})
        self.assertFalse(config.fallback_to_arrival_bearing)
        with self.assertRaisesRegex(
            ValueError, "fallback_to_arrival_bearing must be a boolean"
        ):
            ReadinessPriorConfig.from_mapping({"fallback_to_arrival_bearing": "no"})

    def test_accepts_shipped_yaml_block(self) -> None:
        block = {
            "enabled": False,
            "events_path": None,
            "source": "ready_frame",
            "method": "vggt",
            "vggt_service_url": "http://127.0.0.1:8117",
            "vggt_timeout_s": 60.0,
            "vggt_frames": ["nav", "ready"],
            "stance_offset_m": 0.30,
            "near_object_m": 0.30,
            "min_height_m": 1.2,
            "max_height_m": 2.1,
            "max_scale_iqr_ratio": 2.0,
            "min_inliers": 30,
            "max_reprojection_error_px": 8.0,
            "max_bearing_error_deg": 45.0,
            "retry_bearing_offsets_deg": [45.0, -45.0],
        }
        config = ReadinessPriorConfig.from_mapping(block)
        self.assertEqual(config.retry_bearing_offsets_deg, (45.0, -45.0))
        self.assertIsInstance(config.retry_bearing_offsets_deg, tuple)
        self.assertEqual(config.max_bearing_error_deg, 45.0)
        self.assertEqual(config.method, "vggt")
        self.assertEqual(config.vggt_frames, ("nav", "ready"))
        # The pre-VGGT block (no method keys) still loads as method vggt.
        legacy = ReadinessPriorConfig.from_mapping(
            {key: block[key] for key in ("enabled", "events_path", "source", "min_inliers")}
        )
        self.assertEqual(legacy.method, "vggt")

    def test_lists_become_tuples(self) -> None:
        config = ReadinessPriorConfig.from_mapping(
            {
                "focal_sweep_fov_deg": [70, 90],
                "retry_bearing_offsets_deg": [30.0],
                "vggt_frames": ["ready", "grasp"],
            }
        )
        self.assertEqual(config.focal_sweep_fov_deg, (70, 90))
        self.assertEqual(config.retry_bearing_offsets_deg, (30.0,))
        self.assertEqual(config.vggt_frames, ("ready", "grasp"))
        self.assertIsInstance(config.vggt_frames, tuple)

    def test_service_url_trailing_slash_is_stripped(self) -> None:
        config = ReadinessPriorConfig.from_mapping(
            {"vggt_service_url": "https://10.21.97.103:8117/"}
        )
        self.assertEqual(config.vggt_service_url, "https://10.21.97.103:8117")

    def test_method_pnp_is_selectable(self) -> None:
        config = ReadinessPriorConfig.from_mapping({"method": "pnp"})
        self.assertEqual(config.method, "pnp")

    def test_rejects_bad_vggt_settings(self) -> None:
        cases = {
            "method": "sift",
            "vggt_service_url": "127.0.0.1:8117",
            "vggt_timeout_s": 0,
            "vggt_frames": [],
            "stance_offset_m": -1,
            "near_object_m": 2.5,
            "max_height_m": 3.5,
            "max_scale_iqr_ratio": 0.5,
        }
        for key, value in cases.items():
            with self.subTest(key=key):
                with self.assertRaisesRegex(ValueError, key):
                    ReadinessPriorConfig.from_mapping({key: value})
        with self.assertRaisesRegex(ValueError, "vggt_frames must not repeat"):
            ReadinessPriorConfig.from_mapping({"vggt_frames": ["nav", "nav"]})
        with self.assertRaisesRegex(ValueError, "vggt_frames must include the source frame 'ready'"):
            ReadinessPriorConfig.from_mapping({"vggt_frames": ["grasp"], "source": "ready_frame"})
        with self.assertRaisesRegex(ValueError, "vggt_frames must include the source frame 'nav'"):
            ReadinessPriorConfig.from_mapping({"vggt_frames": ["ready"], "source": "nav_frame"})
        with self.assertRaisesRegex(ValueError, "vggt_frames entries"):
            ReadinessPriorConfig.from_mapping({"vggt_frames": ["ready", "walk"]})
        with self.assertRaisesRegex(ValueError, "min_height_m must satisfy"):
            ReadinessPriorConfig.from_mapping({"min_height_m": 2.1, "max_height_m": 2.1})
        with self.assertRaisesRegex(ValueError, "min_height_m must satisfy"):
            ReadinessPriorConfig.from_mapping({"min_height_m": 0.0})
        with self.assertRaisesRegex(ValueError, "vggt_timeout_s must be in"):
            ReadinessPriorConfig.from_mapping({"vggt_timeout_s": 601.0})
        with self.assertRaisesRegex(ValueError, "vggt_service_url"):
            ReadinessPriorConfig.from_mapping({"vggt_service_url": ""})
        with self.assertRaisesRegex(ValueError, "vggt_service_url"):
            ReadinessPriorConfig.from_mapping({"vggt_service_url": 8117})

    def test_rejects_unknown_keys(self) -> None:
        with self.assertRaisesRegex(ValueError, r"unknown readiness_prior settings: \['events'\]"):
            ReadinessPriorConfig.from_mapping({"events": "x.json"})

    def test_rejects_bad_source(self) -> None:
        with self.assertRaisesRegex(ValueError, "source"):
            ReadinessPriorConfig.from_mapping({"source": "grasp_frame"})

    def test_enabled_requires_events_path(self) -> None:
        with self.assertRaisesRegex(ValueError, "events_path is required"):
            ReadinessPriorConfig.from_mapping({"enabled": True})
        config = ReadinessPriorConfig.from_mapping(
            {"enabled": True, "events_path": "/tmp/events.json"}
        )
        self.assertTrue(config.enabled)

    def test_enabled_must_be_bool(self) -> None:
        with self.assertRaisesRegex(ValueError, "enabled must be a boolean"):
            ReadinessPriorConfig.from_mapping({"enabled": "yes", "events_path": "e.json"})

    def test_event_object_needs_a_word_that_can_match(self) -> None:
        self.assertEqual(ReadinessPriorConfig.from_mapping({"event_object": "the can"}).event_object, "the can")
        self.assertIsNone(ReadinessPriorConfig.from_mapping({}).event_object)
        for value in ("", "the", 3):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "event_object"):
                    ReadinessPriorConfig.from_mapping({"event_object": value})

    def test_rejects_out_of_range_values(self) -> None:
        cases = {
            "min_inliers": 3,
            "max_reprojection_error_px": 0.0,
            "sift_features": 50,
            "ratio_test": 1.0,
            "focal_sweep_fov_deg": [],
            "max_bearing_error_deg": 0.0,
            "robot_image_max_width": 32,
        }
        for key, value in cases.items():
            with self.subTest(key=key):
                with self.assertRaisesRegex(ValueError, key):
                    ReadinessPriorConfig.from_mapping({key: value})
        with self.assertRaisesRegex(ValueError, "focal_sweep_fov_deg entries"):
            ReadinessPriorConfig.from_mapping({"focal_sweep_fov_deg": [90, 5]})
        with self.assertRaisesRegex(ValueError, "retry_bearing_offsets_deg entries"):
            ReadinessPriorConfig.from_mapping({"retry_bearing_offsets_deg": [200.0]})
        with self.assertRaisesRegex(ValueError, "min_inliers must be an integer"):
            ReadinessPriorConfig.from_mapping({"min_inliers": 30.0})


class EventsLoadingTest(TempDirMixin, unittest.TestCase):
    def test_missing_file_raises_at_construction(self) -> None:
        config = ReadinessPriorConfig(enabled=True, events_path=str(self.directory / "nope.json"))
        with self.assertRaises(FileNotFoundError):
            ReadinessPrior(config)

    def test_wrong_schema_raises_at_construction(self) -> None:
        path = make_events_file(self.directory, schema="yor-manipulation-events-v0")
        with self.assertRaisesRegex(ValueError, "unsupported schema_version"):
            build_prior(path)

    def test_malformed_json_raises_value_error(self) -> None:
        path = self.directory / "broken.json"
        path.write_text("{not json", encoding="utf-8")
        with self.assertRaises(ValueError):
            build_prior(path)

    def test_events_must_be_a_list(self) -> None:
        path = self.directory / "events.json"
        path.write_text(
            json.dumps({"schema_version": EVENTS_SCHEMA, "events": {"object": "can"}}),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "'events' list"):
            build_prior(path)

    def test_events_path_none_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "events_path"):
            ReadinessPrior(ReadinessPriorConfig())

    def test_loads_events_and_sidecar(self) -> None:
        path = make_events_file(self.directory)
        prior = build_prior(path)
        self.assertEqual(len(prior.events), 1)
        self.assertEqual(prior.events[0]["object"], "can")
        self.assertEqual(prior.video_intrinsics["fx"], 640.0)
        self.assertIsNone(prior.last_reason)

    def test_missing_sidecar_gives_no_intrinsics(self) -> None:
        path = make_events_file(self.directory, intrinsics=None)
        prior = build_prior(path)
        self.assertIsNone(prior.video_intrinsics)
        frame = prior.select_frame(prior.events[0])
        self.assertIsNone(prior.scaled_event_intrinsics(frame))


class MatchEventTest(TempDirMixin, unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        path = make_events_file(
            self.directory,
            events=[
                {"object": "can", "hand": "right", "t_ready_s": 16.5, "t_grasp_s": 19.0},
                {"object": "the red can", "hand": "left", "t_ready_s": 30.0, "t_grasp_s": 31.0},
                {"object": "a mug", "hand": "both", "t_ready_s": 35.0, "t_grasp_s": 36.0},
            ],
        )
        self.prior = build_prior(path)

    def test_article_and_token_overlap(self) -> None:
        self.assertEqual(self.prior.match_event("the can")["object"], "can")
        self.assertEqual(self.prior.match_event("red can")["object"], "can")
        self.assertEqual(self.prior.match_event("Red Can!")["object"], "can")
        self.assertEqual(self.prior.match_event("mug")["object"], "a mug")
        self.assertEqual(self.prior.match_event("red")["object"], "the red can")

    def test_no_overlap_and_articles_only(self) -> None:
        self.assertIsNone(self.prior.match_event("table"))
        self.assertIsNone(self.prior.match_event("the"))
        self.assertIsNone(self.prior.match_event("a the an"))
        self.assertIsNone(self.prior.match_event(""))

    def test_first_match_in_file_order_is_logged(self) -> None:
        with self.assertLogs("yor_agent.robot.readiness_prior", level="INFO") as logs:
            event = self.prior.match_event("the red can")
        self.assertEqual(event["object"], "can")
        self.assertTrue(any("matched event 0" in line for line in logs.output))


METRIC_KEYS = {
    "method",
    "bearing_rad",
    "bearing_deg",
    "heading_rad",
    "heading_deg",
    "hand",
    "suggested_arm",
    "inliers",
    "reprojection_error_px",
    "focal_px",
    "frame_time_s",
    "event_object",
    "human_xy",
    "stance_xy",
    "human_height_m",
    "scale",
    "scale_iqr_ratio",
    "frames",
    "service_time_s",
}


class ReadinessPriorEstimateTest(unittest.TestCase):
    def test_to_metrics_keys(self) -> None:
        estimate = ReadinessPriorEstimate(
            bearing_rad=math.pi / 2,
            heading_rad=-math.pi / 2,
            hand="right",
            suggested_arm="right",
            method="pnp",
            inliers=42,
            reprojection_error_px=1.5,
            focal_px=320.0,
            frame_time_s=16.5,
            event_object="can",
            human_xy=(1.0, 2.0),
        )
        metrics = estimate.to_metrics()
        self.assertEqual(set(metrics), METRIC_KEYS)
        self.assertAlmostEqual(metrics["bearing_deg"], 90.0)
        self.assertAlmostEqual(metrics["heading_deg"], -90.0)
        self.assertEqual(metrics["human_xy"], [1.0, 2.0])
        self.assertEqual(metrics["inliers"], 42)
        # The vggt-only fields keep their defaults for a pnp estimate.
        self.assertIsNone(metrics["stance_xy"])
        self.assertIsNone(metrics["human_height_m"])
        self.assertIsNone(metrics["scale"])
        self.assertIsNone(metrics["scale_iqr_ratio"])
        self.assertEqual(metrics["frames"], [])
        self.assertIsNone(metrics["service_time_s"])
        json.dumps(metrics, allow_nan=False)

    def test_to_metrics_for_a_vggt_estimate(self) -> None:
        estimate = ReadinessPriorEstimate(
            bearing_rad=-2.6,
            heading_rad=0.5,
            hand="left",
            suggested_arm="left",
            method="vggt",
            inliers=None,
            reprojection_error_px=None,
            focal_px=1096.4,
            frame_time_s=16.5,
            event_object="can",
            human_xy=(1.6, -1.9),
            stance_xy=(1.3, -2.0),
            human_height_m=1.472,
            scale=2.244,
            scale_iqr_ratio=1.16,
            frames=("nav", "ready"),
            service_time_s=4.4,
        )
        metrics = estimate.to_metrics()
        self.assertEqual(set(metrics), METRIC_KEYS)
        self.assertIsNone(metrics["inliers"])
        self.assertIsNone(metrics["reprojection_error_px"])
        self.assertEqual(metrics["focal_px"], 1096.4)
        self.assertEqual(metrics["stance_xy"], [1.3, -2.0])
        self.assertEqual(metrics["human_height_m"], 1.472)
        self.assertEqual(metrics["scale"], 2.244)
        self.assertEqual(metrics["scale_iqr_ratio"], 1.16)
        self.assertEqual(metrics["frames"], ["nav", "ready"])
        self.assertEqual(metrics["service_time_s"], 4.4)
        json.dumps(metrics, allow_nan=False)

    def test_to_metrics_never_emits_nan(self) -> None:
        estimate = ReadinessPriorEstimate(
            bearing_rad=0.0,
            heading_rad=0.0,
            hand=None,
            suggested_arm=None,
            method="vggt",
            inliers=None,
            reprojection_error_px=float("nan"),
            focal_px=float("inf"),
            frame_time_s=float("nan"),
            event_object="can",
            human_xy=(0.0, 0.0),
            human_height_m=float("nan"),
            scale=float("-inf"),
        )
        metrics = estimate.to_metrics()
        self.assertIsNone(metrics["reprojection_error_px"])
        self.assertIsNone(metrics["focal_px"])
        self.assertIsNone(metrics["frame_time_s"])
        self.assertIsNone(metrics["human_height_m"])
        self.assertIsNone(metrics["scale"])
        json.dumps(metrics, allow_nan=False)
        # frame_time_s is Optional: None is the value for an event without times.
        without_time = dataclasses.replace(
            estimate, reprojection_error_px=None, focal_px=None, frame_time_s=None,
            human_height_m=None, scale=None,
        )
        self.assertIsNone(without_time.frame_time_s)
        self.assertIsNone(without_time.to_metrics()["frame_time_s"])
        json.dumps(without_time.to_metrics(), allow_nan=False)
        json.dumps(dataclasses.asdict(without_time), allow_nan=False)

    def test_suggested_arm_for_hand(self) -> None:
        self.assertEqual(readiness_prior.suggested_arm_for_hand("left"), "left")
        self.assertEqual(readiness_prior.suggested_arm_for_hand("right"), "right")
        self.assertIsNone(readiness_prior.suggested_arm_for_hand("both"))
        self.assertIsNone(readiness_prior.suggested_arm_for_hand(None))


class GeometryHelpersTest(unittest.TestCase):
    def test_rodrigues_matches_axis_rotation(self) -> None:
        angle = 0.6
        np.testing.assert_allclose(
            readiness_prior._rotation_from_rvec(np.array([0.0, angle, 0.0])),
            _rot_y(angle),
            atol=1e-12,
        )
        np.testing.assert_allclose(
            readiness_prior._rotation_from_rvec(np.zeros(3)), np.eye(3), atol=1e-12
        )
        rotation = HUMAN_R_ZH.T
        np.testing.assert_allclose(
            readiness_prior._rotation_from_rvec(rvec_from_rotation(rotation)),
            rotation,
            atol=1e-9,
        )

    def test_planar_to_world_matches_docking_controller_convention(self) -> None:
        pose = np.array([1.0, -2.0, math.pi / 2])
        x, y = readiness_prior._planar_to_world(0.5, 0.25, pose)
        # Facing +y: forward adds to y, left points to -x.
        self.assertAlmostEqual(x, 0.75)
        self.assertAlmostEqual(y, -1.5)

    def test_pnp_result_camera_center_and_axis(self) -> None:
        rotation = HUMAN_R_ZH.T
        tvec = -rotation @ HUMAN_CENTER
        result = PnPResult(
            rvec=rvec_from_rotation(rotation),
            tvec=tvec,
            inliers=50,
            matches=60,
            correspondences=55,
            reprojection_error_px=0.5,
            focal_px=320.0,
            sweep=None,
        )
        np.testing.assert_allclose(result.camera_center(), HUMAN_CENTER, atol=1e-9)
        np.testing.assert_allclose(result.optical_axis(), HUMAN_R_ZH[:, 2], atol=1e-9)


class EstimateWithoutCv2Test(TempDirMixin, unittest.TestCase):
    """Every path of estimate() that does not need OpenCV, on every machine."""

    def setUp(self) -> None:
        super().setUp()
        self.events_path = make_events_file(self.directory)
        self.rgb = np.zeros((ZED_H, ZED_W, 3), dtype=np.uint8)
        self.depth = np.full((ZED_H, ZED_W), 2.5, dtype=np.float32)

    def call_estimate(self, prior: ReadinessPrior, object_name: str = "the can"):
        return prior.estimate(
            object_name,
            self.rgb,
            self.depth,
            ZED_K,
            POSE,
            down=DOWN,
            planar_forward=PLANAR_FORWARD,
            planar_left=PLANAR_LEFT,
            target_xy=TARGET_XY,
        )

    def test_unknown_object_gives_no_event_match(self) -> None:
        prior = build_prior(self.events_path)
        self.assertIsNone(self.call_estimate(prior, "table"))
        self.assertEqual(prior.last_reason, "no_event_match")

    def test_a_pinned_event_object_matches_whatever_the_docking_name_is(self) -> None:
        # The event is the can's; "bottle" shares no word with it, so without
        # the pin nothing matches, with it the estimate goes on to the frame
        # (missing here, which is the next thing it reports).
        path = make_events_file(self.directory / "missing", create_frames=False)
        self.assertIsNone(self.call_estimate(build_prior(path), "bottle"))
        pinned = build_prior(path, event_object="the can")
        with self.assertLogs("yor_agent.robot.readiness_prior", level="WARNING"):
            self.assertIsNone(self.call_estimate(pinned, "bottle"))
        self.assertEqual(pinned.last_reason, "frame_missing")

    def test_bad_frame_path_gives_frame_missing(self) -> None:
        path = make_events_file(self.directory / "missing", create_frames=False)
        prior = build_prior(path)
        with self.assertLogs("yor_agent.robot.readiness_prior", level="WARNING") as logs:
            self.assertIsNone(self.call_estimate(prior))
        self.assertEqual(prior.last_reason, "frame_missing")
        self.assertTrue(any("event frame missing" in line for line in logs.output))

    def test_absent_frame_entry_gives_frame_missing(self) -> None:
        payload = json.loads(self.events_path.read_text(encoding="utf-8"))
        del payload["events"][0]["frames"]["nav"]
        self.events_path.write_text(json.dumps(payload), encoding="utf-8")
        prior = build_prior(self.events_path, source="nav_frame")
        self.assertIsNone(self.call_estimate(prior))
        self.assertEqual(prior.last_reason, "frame_missing")

    def test_source_selects_nav_or_ready_frame(self) -> None:
        ready = build_prior(self.events_path)
        nav = build_prior(self.events_path, source="nav_frame")
        event = ready.events[0]
        self.assertTrue(ready.select_frame(event)["path"].endswith("event00_ready.jpg"))
        self.assertTrue(nav.select_frame(event)["path"].endswith("event00_nav.jpg"))
        self.assertEqual(ready.select_frame(event)["time_s"], 16.5)
        self.assertEqual(nav.select_frame(event)["time_s"], 14.5)

    def test_frame_path_resolves_relative_to_events_json(self) -> None:
        prior = build_prior(self.events_path)
        frame = prior.select_frame(prior.events[0])
        self.assertEqual(
            prior.frame_path(frame),
            self.events_path.parent / frame["path"],
        )
        absolute = {"path": "/abs/frame.jpg"}
        self.assertEqual(prior.frame_path(absolute), Path("/abs/frame.jpg"))

    def test_missing_cv2_gives_cv2_unavailable(self) -> None:
        prior = build_prior(self.events_path, method="pnp")
        with mock.patch.object(readiness_prior, "_cv2", side_effect=ImportError("no cv2")):
            self.assertIsNone(self.call_estimate(prior))
        self.assertEqual(prior.last_reason, "cv2_unavailable")

    def test_solve_pose_without_cv2_gives_cv2_unavailable(self) -> None:
        prior = build_prior(self.events_path, matcher=lambda a, b: (np.zeros((0, 2)), np.zeros((0, 2))))
        with mock.patch.object(readiness_prior, "_cv2", side_effect=ImportError("no cv2")):
            result = prior.solve_pose(
                self.rgb, self.rgb, self.depth, ZED_K, event_intrinsics=None, frame_size=(ZED_W, ZED_H)
            )
        self.assertIsNone(result)
        self.assertEqual(prior.last_reason, "cv2_unavailable")

    def test_scaled_event_intrinsics_follow_frame_scale(self) -> None:
        prior = build_prior(self.events_path)
        frame = prior.select_frame(prior.events[0])
        scaled = prior.scaled_event_intrinsics(frame)
        self.assertAlmostEqual(scaled["fx"], 320.0)
        self.assertAlmostEqual(scaled["fy"], 320.0)
        self.assertAlmostEqual(scaled["cx"], 319.5)
        self.assertAlmostEqual(scaled["cy"], 239.5)
        # A frame decoded at a different width than recorded follows the pixels.
        rescaled = prior.scaled_event_intrinsics(frame, decoded_size=(320, 240))
        self.assertAlmostEqual(rescaled["fx"], 160.0)
        self.assertAlmostEqual(rescaled["cx"], 159.5)
        # Without a recorded scale the frame/source widths give it.
        no_scale = {key: value for key, value in frame.items() if key != "scale"}
        self.assertAlmostEqual(prior.scaled_event_intrinsics(no_scale)["fx"], 320.0)

    def _prior_with_injected_pose(self, rotation_zh: np.ndarray, center: np.ndarray, **overrides):
        overrides.setdefault("method", "pnp")
        prior = build_prior(self.events_path, **overrides)
        rotation = rotation_zh.T
        result = PnPResult(
            rvec=rvec_from_rotation(rotation),
            tvec=-rotation @ center,
            inliers=48,
            matches=70,
            correspondences=60,
            reprojection_error_px=0.7,
            focal_px=320.0,
            sweep=None,
        )
        prior._load_frame_rgb = lambda path: np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
        prior.solve_pose = lambda *args, **kwargs: result
        return prior

    def test_geometry_from_injected_pose(self) -> None:
        prior = self._prior_with_injected_pose(HUMAN_R_ZH, HUMAN_CENTER)
        estimate = self.call_estimate(prior)
        self.assertIsNotNone(estimate)
        self.assertIsNone(prior.last_reason)
        expected_xy = expected_world_xy(HUMAN_CENTER)
        self.assertAlmostEqual(estimate.human_xy[0], expected_xy[0], places=9)
        self.assertAlmostEqual(estimate.human_xy[1], expected_xy[1], places=9)
        self.assertAlmostEqual(
            estimate.bearing_rad,
            wrap(math.atan2(expected_xy[1] - TARGET_XY[1], expected_xy[0] - TARGET_XY[0])),
            places=9,
        )
        self.assertAlmostEqual(estimate.heading_rad, expected_heading(HUMAN_R_ZH[:, 2]), places=9)
        self.assertEqual(estimate.hand, "right")
        self.assertEqual(estimate.suggested_arm, "right")
        self.assertEqual(estimate.method, "pnp")
        self.assertEqual(estimate.inliers, 48)
        self.assertEqual(estimate.reprojection_error_px, 0.7)
        self.assertEqual(estimate.focal_px, 320.0)
        self.assertEqual(estimate.frame_time_s, 16.5)
        self.assertEqual(estimate.event_object, "can")

    def test_identity_pose_puts_human_at_the_robot(self) -> None:
        prior = self._prior_with_injected_pose(np.eye(3), np.zeros(3))
        estimate = self.call_estimate(prior)
        self.assertAlmostEqual(estimate.human_xy[0], POSE[0])
        self.assertAlmostEqual(estimate.human_xy[1], POSE[1])
        self.assertAlmostEqual(estimate.heading_rad, POSE[2])

    def test_both_hands_suggest_no_arm(self) -> None:
        path = make_events_file(
            self.directory / "both",
            events=[{"object": "can", "hand": "both", "t_ready_s": 1.0, "t_grasp_s": 2.0}],
        )
        self.events_path = path
        prior = self._prior_with_injected_pose(HUMAN_R_ZH, HUMAN_CENTER)
        estimate = self.call_estimate(prior)
        self.assertEqual(estimate.hand, "both")
        self.assertIsNone(estimate.suggested_arm)

    def test_far_solution_is_degenerate_geometry(self) -> None:
        prior = self._prior_with_injected_pose(HUMAN_R_ZH, np.array([12.0, 0.0, 3.0]))
        self.assertIsNone(self.call_estimate(prior))
        self.assertEqual(prior.last_reason, "degenerate_geometry")

    def test_nav_frame_source_reports_nav_frame_time(self) -> None:
        prior = self._prior_with_injected_pose(HUMAN_R_ZH, HUMAN_CENTER, source="nav_frame")
        estimate = self.call_estimate(prior)
        self.assertEqual(estimate.frame_time_s, 14.5)

    def test_event_without_times_gives_frame_time_none(self) -> None:
        strip_event_times(self.events_path)
        prior = self._prior_with_injected_pose(HUMAN_R_ZH, HUMAN_CENTER)
        estimate = self.call_estimate(prior)
        self.assertIsNotNone(estimate, prior.last_reason)
        self.assertIsNone(estimate.frame_time_s)
        metrics = estimate.to_metrics()
        self.assertIsNone(metrics["frame_time_s"])
        json.dumps(metrics, allow_nan=False)
        json.dumps(dataclasses.asdict(estimate), allow_nan=False)

    def test_frame_time_falls_back_to_the_event_ready_time(self) -> None:
        payload = json.loads(self.events_path.read_text(encoding="utf-8"))
        del payload["events"][0]["frames"]["ready"]["time_s"]
        self.events_path.write_text(json.dumps(payload), encoding="utf-8")
        prior = self._prior_with_injected_pose(HUMAN_R_ZH, HUMAN_CENTER)
        self.assertEqual(self.call_estimate(prior).frame_time_s, 16.5)


class SolvePoseWithStubbedSolverTest(TempDirMixin, unittest.TestCase):
    """solve_pose bookkeeping (depth lookup, focal sweep, thresholds) without cv2."""

    def setUp(self) -> None:
        super().setUp()
        self.events_path = make_events_file(self.directory)
        self.rgb = np.zeros((ZED_H, ZED_W, 3), dtype=np.uint8)
        self.depth = np.full((ZED_H, ZED_W), 2.5, dtype=np.float32)
        self.event_rgb = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
        patcher = mock.patch.object(readiness_prior, "_cv2", return_value=fake_cv2_module())
        patcher.start()
        self.addCleanup(patcher.stop)

    @staticmethod
    def grid_matches(count: int, *, scale: float = 1.0):
        cols = np.linspace(10, ZED_W - 10, count)
        rows = np.linspace(10, ZED_H - 10, count)
        robot_px = np.stack([cols, rows], axis=1)
        event_px = robot_px * 2.0

        def matcher(event_rgb, robot_rgb):
            return event_px.copy(), robot_px * scale

        return matcher, robot_px

    def solve(self, prior, *, event_intrinsics="sidecar"):
        if event_intrinsics == "sidecar":
            event_intrinsics = prior.scaled_event_intrinsics(prior.select_frame(prior.events[0]))
        return prior.solve_pose(
            self.event_rgb,
            self.rgb,
            self.depth,
            ZED_K,
            event_intrinsics=event_intrinsics,
            frame_size=(FRAME_W, FRAME_H),
        )

    def test_too_few_matches(self) -> None:
        matcher, _ = self.grid_matches(5)
        prior = build_prior(self.events_path, matcher=matcher, min_inliers=4)
        self.assertIsNone(self.solve(prior))
        self.assertEqual(prior.last_reason, "too_few_matches")
        self.assertEqual(prior.last_diagnostics["matches"], 5)

    def test_matches_below_min_inliers_are_too_few_matches(self) -> None:
        matcher, _ = self.grid_matches(20)
        prior = build_prior(self.events_path, matcher=matcher, min_inliers=30)
        self.assertIsNone(self.solve(prior))
        self.assertEqual(prior.last_reason, "too_few_matches")

    def test_invalid_depth_gives_too_few_correspondences(self) -> None:
        matcher, robot_px = self.grid_matches(10)
        depth = self.depth.copy()
        for column, row in robot_px[:5]:
            depth[int(round(row)), int(round(column))] = np.nan
        depth[int(round(robot_px[5, 1])), int(round(robot_px[5, 0]))] = 0.0
        prior = build_prior(self.events_path, matcher=matcher, min_inliers=4)
        result = prior.solve_pose(
            self.event_rgb, self.rgb, depth, ZED_K,
            event_intrinsics=None, frame_size=(FRAME_W, FRAME_H),
        )
        self.assertIsNone(result)
        self.assertEqual(prior.last_reason, "too_few_correspondences")
        self.assertEqual(prior.last_diagnostics["correspondences"], 4)

    def test_object_points_are_back_projected_through_the_zed_intrinsics(self) -> None:
        matcher, robot_px = self.grid_matches(12)
        depth = np.full((ZED_H, ZED_W), np.nan, dtype=np.float32)
        expected = []
        for index, (column, row) in enumerate(robot_px):
            r, c = int(round(row)), int(round(column))
            z = 2.0 + 0.1 * index
            depth[r, c] = z
            expected.append([(c - 159.5) * z / 300.0, (r - 119.5) * z / 300.0, z])
        prior = build_prior(self.events_path, matcher=matcher, min_inliers=4)
        captured = {}

        def fake_solve(cv2, object_points, image_points, camera_matrix):
            captured["object_points"] = np.array(object_points)
            captured["image_points"] = np.array(image_points)
            captured["camera_matrix"] = np.array(camera_matrix)
            return 12, 0.4, np.zeros(3), np.array([0.0, 0.0, -1.0])

        prior._solve_single = fake_solve
        result = prior.solve_pose(
            self.event_rgb, self.rgb, depth, ZED_K,
            event_intrinsics={"fx": 320.0, "fy": 321.0, "cx": 319.5, "cy": 239.5},
            frame_size=(FRAME_W, FRAME_H),
        )
        self.assertIsNotNone(result)
        np.testing.assert_allclose(captured["object_points"], np.array(expected), atol=1e-9)
        np.testing.assert_allclose(captured["image_points"], robot_px * 2.0, atol=1e-9)
        np.testing.assert_allclose(
            captured["camera_matrix"],
            [[320.0, 0.0, 319.5], [0.0, 321.0, 239.5], [0.0, 0.0, 1.0]],
        )
        self.assertEqual(result.inliers, 12)
        self.assertEqual(result.matches, 12)
        self.assertEqual(result.correspondences, 12)
        self.assertEqual(result.focal_px, 320.0)
        self.assertIsNone(result.sweep)
        np.testing.assert_allclose(result.tvec, [0.0, 0.0, -1.0])

    def test_downscaled_robot_pixels_are_rescaled_before_the_depth_lookup(self) -> None:
        matcher, robot_px = self.grid_matches(12, scale=0.5)
        seen = {}

        def checking_matcher(event_rgb, robot_rgb):
            seen["shape"] = robot_rgb.shape
            return matcher(event_rgb, robot_rgb)

        prior = build_prior(
            self.events_path, matcher=checking_matcher, min_inliers=4, robot_image_max_width=160
        )
        captured = {}

        def fake_solve(cv2, object_points, image_points, camera_matrix):
            captured["object_points"] = np.array(object_points)
            return 12, 0.1, np.zeros(3), np.zeros(3)

        prior._solve_single = fake_solve
        self.assertIsNotNone(self.solve(prior))
        self.assertEqual(seen["shape"], (ZED_H // 2, ZED_W // 2, 3))
        expected = np.stack(
            [
                (np.rint(robot_px[:, 0]) - 159.5) * 2.5 / 300.0,
                (np.rint(robot_px[:, 1]) - 119.5) * 2.5 / 300.0,
                np.full(len(robot_px), 2.5),
            ],
            axis=1,
        )
        np.testing.assert_allclose(captured["object_points"], expected, atol=1e-9)

    def test_focal_sweep_keeps_most_inliers_then_lowest_error(self) -> None:
        matcher, _ = self.grid_matches(40)
        prior = build_prior(
            self.events_path, matcher=matcher, focal_sweep_fov_deg=[60, 90, 120], min_inliers=4
        )
        by_focal = {}
        expected_focals = {
            fov: (FRAME_W / 2.0) / math.tan(math.radians(fov) / 2.0) for fov in (60, 90, 120)
        }
        outcomes = {
            60: (30, 2.0),
            90: (35, 3.0),
            120: (35, 1.0),
        }

        def fake_solve(cv2, object_points, image_points, camera_matrix):
            focal = float(camera_matrix[0, 0])
            fov = min(expected_focals, key=lambda key: abs(expected_focals[key] - focal))
            self.assertAlmostEqual(camera_matrix[0, 2], (FRAME_W - 1) / 2.0)
            self.assertAlmostEqual(camera_matrix[1, 2], (FRAME_H - 1) / 2.0)
            by_focal[fov] = focal
            inliers, error = outcomes[fov]
            return inliers, error, np.array([0.0, 0.0, 0.1 * fov]), np.array([fov, 0.0, 0.0])

        prior._solve_single = fake_solve
        result = self.solve(prior, event_intrinsics=None)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result.focal_px, expected_focals[120])
        self.assertEqual(result.inliers, 35)
        self.assertEqual(result.reprojection_error_px, 1.0)
        np.testing.assert_allclose(result.tvec, [120.0, 0.0, 0.0])
        self.assertEqual([entry["fov_deg"] for entry in result.sweep], [60.0, 90.0, 120.0])
        self.assertEqual([entry["inliers"] for entry in result.sweep], [30, 35, 35])
        self.assertEqual(
            [entry["reprojection_error_px"] for entry in result.sweep], [2.0, 3.0, 1.0]
        )
        for fov, entry in zip((60, 90, 120), result.sweep):
            self.assertAlmostEqual(entry["focal_px"], expected_focals[fov])
            self.assertEqual(entry["tvec"], [float(fov), 0.0, 0.0])
        self.assertEqual(prior.last_diagnostics["sweep"], result.sweep)

    def test_sweep_with_every_solve_failing_is_pnp_failed(self) -> None:
        matcher, _ = self.grid_matches(40)
        prior = build_prior(self.events_path, matcher=matcher, min_inliers=4)
        prior._solve_single = lambda cv2, o, i, k: None
        self.assertIsNone(self.solve(prior, event_intrinsics=None))
        self.assertEqual(prior.last_reason, "pnp_failed")
        self.assertEqual(len(prior.last_diagnostics["sweep"]), 7)
        self.assertTrue(all(entry["inliers"] == 0 for entry in prior.last_diagnostics["sweep"]))
        self.assertTrue(
            all(entry["reprojection_error_px"] is None for entry in prior.last_diagnostics["sweep"])
        )

    def test_too_few_inliers_and_high_error_are_rejected(self) -> None:
        matcher, _ = self.grid_matches(40)
        prior = build_prior(self.events_path, matcher=matcher, min_inliers=30)
        prior._solve_single = lambda cv2, o, i, k: (20, 0.5, np.zeros(3), np.zeros(3))
        self.assertIsNone(self.solve(prior))
        self.assertEqual(prior.last_reason, "too_few_inliers")
        self.assertEqual(prior.last_diagnostics["inliers"], 20)
        prior._solve_single = lambda cv2, o, i, k: (35, 9.5, np.zeros(3), np.zeros(3))
        self.assertIsNone(self.solve(prior))
        self.assertEqual(prior.last_reason, "reprojection_error_too_high")


class VggtEstimateTest(TempDirMixin, unittest.TestCase):
    """method vggt through an injected transport: payload, stance rule, gates, reasons."""

    def setUp(self) -> None:
        super().setUp()
        self.events_path = make_events_file(self.directory)
        self.frame_bytes = write_jpeg_frames(self.events_path)
        rng = np.random.default_rng(11)
        self.rgb = rng.integers(0, 256, size=(ZED_H, ZED_W, 3), dtype=np.uint8)
        self.depth = np.full((ZED_H, ZED_W), 2.5, dtype=np.float32)
        self.depth[0, 0] = np.nan
        self.depth[0, 1] = np.inf
        self.depth[0, 2] = -np.inf
        self.depth[1, 0] = 0.0
        self.depth[1, 1] = 1.234

    def call_estimate(
        self,
        prior: ReadinessPrior,
        object_name: str = "the can",
        *,
        target_xy=TARGET_XY,
        camera_height_m=CAMERA_HEIGHT_M,
    ):
        return prior.estimate(
            object_name,
            self.rgb,
            self.depth,
            ZED_K,
            POSE,
            down=DOWN,
            planar_forward=PLANAR_FORWARD,
            planar_left=PLANAR_LEFT,
            target_xy=target_xy,
            camera_height_m=camera_height_m,
        )

    def test_request_payload_matches_the_service_contract(self) -> None:
        transport = FakeTransport()
        prior = build_prior(self.events_path, transport=transport)
        estimate = self.call_estimate(prior)
        self.assertIsNotNone(estimate, prior.last_reason)
        self.assertEqual(len(transport.calls), 1)
        url, payload, timeout_s = transport.calls[0]
        self.assertEqual(url, "http://127.0.0.1:8117/stance")
        self.assertEqual(timeout_s, 60.0)
        self.assertEqual(set(payload), {"robot", "frames"})
        robot = payload["robot"]
        self.assertEqual(
            set(robot),
            {
                "label",
                "rgb_jpeg_base64",
                "depth_png16_base64",
                "intrinsics",
                "down",
                "planar_forward",
                "planar_left",
                "ground_camera_height_m",
            },
        )
        self.assertEqual(robot["label"], "zed")
        self.assertEqual(robot["intrinsics"], ZED_K.tolist())
        self.assertEqual(robot["down"], DOWN.tolist())
        self.assertEqual(robot["planar_forward"], PLANAR_FORWARD.tolist())
        self.assertEqual(robot["planar_left"], PLANAR_LEFT.tolist())
        self.assertEqual(robot["ground_camera_height_m"], CAMERA_HEIGHT_M)
        # The whole payload is plain JSON.
        json.dumps(payload, allow_nan=False)
        # Frames: the config order, the JPEG file bytes verbatim.
        self.assertEqual([frame["label"] for frame in payload["frames"]], ["nav", "ready"])
        for frame in payload["frames"]:
            self.assertEqual(set(frame), {"label", "jpeg_base64"})
            self.assertEqual(
                base64.b64decode(frame["jpeg_base64"]), self.frame_bytes[frame["label"]]
            )
        # The robot JPEG decodes to the ZED shape.
        robot_image = Image.open(io.BytesIO(base64.b64decode(robot["rgb_jpeg_base64"])))
        self.assertEqual(robot_image.format, "JPEG")
        self.assertEqual(robot_image.mode, "RGB")
        self.assertEqual(robot_image.size, (ZED_W, ZED_H))
        # The depth PNG carries 16-bit millimetres with the invalid pixels at 0.
        depth_image = Image.open(io.BytesIO(base64.b64decode(robot["depth_png16_base64"])))
        self.assertEqual(depth_image.format, "PNG")
        self.assertIn(depth_image.mode, ("I;16", "I"))
        millimetres = np.asarray(depth_image)
        self.assertEqual(millimetres.shape, (ZED_H, ZED_W))
        self.assertIn(millimetres.dtype, (np.uint16, np.int32))
        self.assertEqual(int(millimetres[0, 0]), 0)
        self.assertEqual(int(millimetres[0, 1]), 0)
        self.assertEqual(int(millimetres[0, 2]), 0)
        self.assertEqual(int(millimetres[1, 0]), 0)
        self.assertEqual(int(millimetres[1, 1]), 1234)
        self.assertEqual(int(millimetres[5, 5]), 2500)
        self.assertEqual(int(millimetres.max()), 2500)

    def test_depth_encoder_clips_to_the_png_range(self) -> None:
        depth = np.array([[70.0, 65.535, -1.0, 0.0005]], dtype=np.float64)
        image = Image.open(io.BytesIO(readiness_prior._encode_depth_png16(depth)))
        values = np.asarray(image).reshape(-1).astype(np.int64).tolist()
        self.assertEqual(values, [65535, 65535, 0, 0])
        # Only the first three channels of an RGBA robot image are sent.
        rgba = np.concatenate([self.rgb, np.full((ZED_H, ZED_W, 1), 255, np.uint8)], axis=2)
        jpeg = Image.open(io.BytesIO(readiness_prior._encode_rgb_jpeg(rgba)))
        self.assertEqual(jpeg.mode, "RGB")
        self.assertEqual(jpeg.size, (ZED_W, ZED_H))

    def test_estimate_far_from_the_object_bears_from_object_to_feet(self) -> None:
        prior = build_prior(self.events_path, transport=FakeTransport())
        with self.assertLogs("yor_agent.robot.readiness_prior", level="INFO") as logs:
            estimate = self.call_estimate(prior)
        self.assertIsNotNone(estimate, prior.last_reason)
        self.assertIsNone(prior.last_reason)
        row = CAMERA_ROWS["ready"]
        head, heading, feet, bearing = expected_stance(row, TARGET_XY)
        self.assertGreater(math.hypot(head[0] - TARGET_XY[0], head[1] - TARGET_XY[1]), 0.30)
        self.assertEqual(estimate.method, "vggt")
        self.assertIsNone(estimate.inliers)
        self.assertIsNone(estimate.reprojection_error_px)
        self.assertEqual(estimate.focal_px, row["focal_px"])
        self.assertEqual(estimate.human_height_m, 1.472)
        self.assertEqual(estimate.scale, 2.244)
        self.assertEqual(estimate.scale_iqr_ratio, 1.16)
        self.assertEqual(estimate.frames, ("nav", "ready"))
        self.assertIsInstance(estimate.service_time_s, float)
        self.assertGreaterEqual(estimate.service_time_s, 0.0)
        expected_head = readiness_prior._planar_to_world(0.526, -0.317, POSE)
        self.assertAlmostEqual(estimate.human_xy[0], expected_head[0], places=12)
        self.assertAlmostEqual(estimate.human_xy[1], expected_head[1], places=12)
        self.assertAlmostEqual(estimate.human_xy[0], head[0], places=12)
        self.assertAlmostEqual(estimate.human_xy[1], head[1], places=12)
        self.assertAlmostEqual(estimate.heading_rad, wrap(math.radians(27.0) + POSE[2]), places=12)
        self.assertAlmostEqual(estimate.stance_xy[0], feet[0], places=12)
        self.assertAlmostEqual(estimate.stance_xy[1], feet[1], places=12)
        self.assertAlmostEqual(
            math.hypot(estimate.stance_xy[0] - head[0], estimate.stance_xy[1] - head[1]),
            STANCE_OFFSET_M,
            places=12,
        )
        self.assertAlmostEqual(
            estimate.bearing_rad,
            wrap(math.atan2(feet[1] - TARGET_XY[1], feet[0] - TARGET_XY[0])),
            places=12,
        )
        self.assertAlmostEqual(estimate.bearing_rad, bearing, places=12)
        self.assertEqual(estimate.hand, "right")
        self.assertEqual(estimate.suggested_arm, "right")
        self.assertEqual(estimate.frame_time_s, 16.5)
        self.assertEqual(estimate.event_object, "can")
        metrics = estimate.to_metrics()
        self.assertEqual(set(metrics), METRIC_KEYS)
        self.assertEqual(metrics["frames"], ["nav", "ready"])
        self.assertEqual(metrics["stance_xy"], [estimate.stance_xy[0], estimate.stance_xy[1]])
        json.dumps(metrics, allow_nan=False)
        self.assertTrue(any("bearing" in line and "object -> feet" in line for line in logs.output))
        # Diagnostics: the service body plus what was sent and how long it took.
        self.assertEqual(prior.last_diagnostics["request_frames"], ["nav", "ready"])
        self.assertEqual(prior.last_diagnostics["scale"], 2.244)
        self.assertEqual(prior.last_diagnostics["schema_version"], "yor-vggt-stance-v1")
        self.assertEqual(len(prior.last_diagnostics["cameras"]), 2)
        self.assertEqual(prior.last_diagnostics["service_time_s"], estimate.service_time_s)

    def test_estimate_near_the_object_bears_from_heading_plus_180(self) -> None:
        prior = build_prior(self.events_path, transport=FakeTransport())
        head = planar_world(0.526, -0.317)
        near_target = (head[0] + 0.10, head[1] - 0.05)
        with self.assertLogs("yor_agent.robot.readiness_prior", level="INFO") as logs:
            estimate = self.call_estimate(prior, target_xy=near_target)
        self.assertIsNotNone(estimate, prior.last_reason)
        _, heading, feet, bearing = expected_stance(CAMERA_ROWS["ready"], near_target)
        self.assertAlmostEqual(estimate.bearing_rad, wrap(heading + math.pi), places=12)
        self.assertAlmostEqual(estimate.bearing_rad, bearing, places=12)
        self.assertAlmostEqual(estimate.stance_xy[0], feet[0], places=12)
        self.assertAlmostEqual(estimate.stance_xy[1], feet[1], places=12)
        # The feet-based bearing would differ: the rule really switched.
        self.assertGreater(
            abs(wrap(estimate.bearing_rad - math.atan2(feet[1] - near_target[1], feet[0] - near_target[0]))),
            1e-3,
        )
        self.assertTrue(any("near object: heading + 180" in line for line in logs.output))
        # Just inside near_object_m still counts as near; just outside does not.
        inside = (head[0] + 0.299, head[1])
        estimate = self.call_estimate(prior, target_xy=inside)
        self.assertAlmostEqual(estimate.bearing_rad, wrap(heading + math.pi), places=12)
        beyond = (head[0] + 0.301, head[1])
        estimate = self.call_estimate(prior, target_xy=beyond)
        self.assertAlmostEqual(
            estimate.bearing_rad,
            wrap(math.atan2(feet[1] - beyond[1], feet[0] - beyond[0])),
            places=12,
        )

    def test_stance_and_near_thresholds_follow_the_config(self) -> None:
        prior = build_prior(
            self.events_path,
            transport=FakeTransport(),
            stance_offset_m=0.5,
            near_object_m=1.0,
        )
        head = planar_world(0.526, -0.317)
        target = (head[0] + 0.8, head[1])
        estimate = self.call_estimate(prior, target_xy=target)
        _, heading, feet, bearing = expected_stance(
            CAMERA_ROWS["ready"], target, offset_m=0.5, near_m=1.0
        )
        self.assertAlmostEqual(estimate.bearing_rad, wrap(heading + math.pi), places=12)
        self.assertAlmostEqual(estimate.stance_xy[0], feet[0], places=12)
        self.assertAlmostEqual(estimate.stance_xy[1], feet[1], places=12)
        self.assertAlmostEqual(
            math.hypot(estimate.stance_xy[0] - head[0], estimate.stance_xy[1] - head[1]), 0.5
        )

    def test_nav_frame_source_takes_the_nav_row(self) -> None:
        prior = build_prior(self.events_path, transport=FakeTransport(), source="nav_frame")
        estimate = self.call_estimate(prior)
        self.assertIsNotNone(estimate, prior.last_reason)
        row = CAMERA_ROWS["nav"]
        self.assertEqual(estimate.frames, ("nav", "ready"))
        self.assertEqual(estimate.human_height_m, row["height_above_floor_m"])
        self.assertEqual(estimate.focal_px, row["focal_px"])
        self.assertAlmostEqual(estimate.heading_rad, wrap(math.radians(-13.0) + POSE[2]), places=12)
        head = planar_world(row["forward_m"], row["left_m"])
        self.assertAlmostEqual(estimate.human_xy[0], head[0], places=12)
        self.assertAlmostEqual(estimate.human_xy[1], head[1], places=12)
        self.assertEqual(estimate.frame_time_s, 14.5)

    def test_frames_config_selects_which_event_frames_are_sent(self) -> None:
        transport = FakeTransport()
        prior = build_prior(self.events_path, transport=transport, vggt_frames=["ready"])
        estimate = self.call_estimate(prior)
        self.assertEqual(estimate.frames, ("ready",))
        self.assertEqual([f["label"] for f in transport.calls[-1][1]["frames"]], ["ready"])
        prior = build_prior(
            self.events_path, transport=transport, vggt_frames=["grasp", "ready", "nav"]
        )
        estimate = self.call_estimate(prior)
        self.assertEqual(estimate.frames, ("grasp", "ready", "nav"))
        self.assertEqual(
            [f["label"] for f in transport.calls[-1][1]["frames"]], ["grasp", "ready", "nav"]
        )
        # The stance row is the ready row wherever it sits in the request.
        self.assertEqual(estimate.human_height_m, 1.472)
        self.assertEqual(prior.last_diagnostics["request_frames"], ["grasp", "ready", "nav"])

    def test_missing_camera_height_never_calls_the_service(self) -> None:
        transport = FakeTransport()
        prior = build_prior(self.events_path, transport=transport)
        with self.assertLogs("yor_agent.robot.readiness_prior", level="WARNING") as logs:
            self.assertIsNone(self.call_estimate(prior, camera_height_m=None))
        self.assertEqual(prior.last_reason, "camera_height_missing")
        self.assertEqual(transport.calls, [])
        self.assertTrue(any("camera_height_m" in line for line in logs.output))
        with self.assertLogs("yor_agent.robot.readiness_prior", level="WARNING"):
            self.assertIsNone(self.call_estimate(prior, camera_height_m=float("nan")))
        self.assertEqual(prior.last_reason, "camera_height_missing")
        self.assertEqual(transport.calls, [])

    def test_transport_failures_give_service_unavailable(self) -> None:
        errors = [
            urllib.error.URLError("connection refused"),
            TimeoutError("timed out"),
            socket.timeout("timed out"),
            OSError("network unreachable"),
            ConnectionResetError("reset"),
            ValueError("Expecting value: line 1 column 1 (char 0)"),
        ]
        for error in errors:
            with self.subTest(error=type(error).__name__):
                transport = FakeTransport(error=error)
                prior = build_prior(self.events_path, transport=transport)
                with self.assertLogs("yor_agent.robot.readiness_prior", level="WARNING") as logs:
                    self.assertIsNone(self.call_estimate(prior))
                self.assertEqual(prior.last_reason, "vggt_service_unavailable")
                self.assertEqual(len(transport.calls), 1)
                self.assertIn(type(error).__name__, prior.last_diagnostics["error"])
                self.assertEqual(prior.last_diagnostics["url"], "http://127.0.0.1:8117/stance")
                self.assertEqual(prior.last_diagnostics["request_frames"], ["nav", "ready"])
                self.assertIn("service_time_s", prior.last_diagnostics)
                self.assertTrue(any("VGGT service call failed" in line for line in logs.output))

    def test_http_error_keeps_the_status_and_detail(self) -> None:
        error = urllib.error.HTTPError(
            "http://127.0.0.1:8117/stance",
            400,
            "Bad Request",
            {},
            io.BytesIO(b'{"detail":"depth shape (240, 320) != rgb (480, 640)"}'),
        )
        prior = build_prior(self.events_path, transport=FakeTransport(error=error))
        with self.assertLogs("yor_agent.robot.readiness_prior", level="WARNING"):
            self.assertIsNone(self.call_estimate(prior))
        self.assertEqual(prior.last_reason, "vggt_service_unavailable")
        self.assertEqual(prior.last_diagnostics["http_status"], 400)
        self.assertIn("depth shape", prior.last_diagnostics["detail"])
        self.assertIn("HTTPError", prior.last_diagnostics["error"])

    def test_malformed_responses_are_invalid(self) -> None:
        nan_heading = canned_response(["nav", "ready"])
        nan_heading["cameras"][1]["heading_deg"] = float("nan")
        none_heading = canned_response(["nav", "ready"])
        none_heading["cameras"][1]["heading_deg"] = None
        no_heading = canned_response(["nav", "ready"])
        del no_heading["cameras"][1]["heading_deg"]
        row_not_object = canned_response(["nav", "ready"])
        row_not_object["cameras"][1] = [0.5, -0.3]
        no_scale_key = canned_response(["nav", "ready"])
        del no_scale_key["scale"]
        cases = {
            "one camera for two frames": canned_response(["ready"]),
            "three cameras for two frames": canned_response(["nav", "ready", "grasp"]),
            "list body": [canned_response(["nav", "ready"])],
            "string body": "ok",
            "none body": None,
            "no cameras key": {"scale": 2.0, "scale_iqr_ratio": 1.1},
            "cameras not a list": canned_response(["nav", "ready"], cameras={"ready": {}}),
            "nan heading": nan_heading,
            "none heading": none_heading,
            "missing heading": no_heading,
            "row not an object": row_not_object,
            "missing scale key": no_scale_key,
        }
        for name, response in cases.items():
            with self.subTest(case=name):
                prior = build_prior(self.events_path, transport=FakeTransport(response))
                with self.assertLogs("yor_agent.robot.readiness_prior", level="WARNING") as logs:
                    self.assertIsNone(self.call_estimate(prior))
                self.assertEqual(prior.last_reason, "vggt_response_invalid")
                self.assertIn("error", prior.last_diagnostics)
                self.assertTrue(any("invalid VGGT response" in line for line in logs.output))

    def test_unstable_scale_is_rejected_before_the_height_gate(self) -> None:
        # The service nulls every metric field when it has no scale; that must
        # read as scale_unstable, not as an invalid response or a bad height.
        no_scale = canned_response(["nav", "ready"], scale=None, scale_iqr_ratio=None)
        for row in no_scale["cameras"]:
            row["forward_m"] = row["left_m"] = row["height_above_floor_m"] = None
        wide = canned_response(["nav", "ready"], scale_iqr_ratio=2.5)
        wide["cameras"][1]["height_above_floor_m"] = 2.9
        cases = {
            "scale null": no_scale,
            "iqr 2.5": wide,
            "iqr null": canned_response(["nav", "ready"], scale_iqr_ratio=None),
            "iqr nan": canned_response(["nav", "ready"], scale_iqr_ratio=float("nan")),
            "scale inf": canned_response(["nav", "ready"], scale=float("inf")),
            "iqr just over": canned_response(["nav", "ready"], scale_iqr_ratio=2.0001),
        }
        for name, response in cases.items():
            with self.subTest(case=name):
                prior = build_prior(self.events_path, transport=FakeTransport(response))
                with self.assertLogs("yor_agent.robot.readiness_prior", level="WARNING") as logs:
                    self.assertIsNone(self.call_estimate(prior))
                self.assertEqual(prior.last_reason, "scale_unstable")
                self.assertTrue(any("scale unstable" in line for line in logs.output))
        # At the limit the scale passes; a wider limit accepts 2.5.
        prior = build_prior(
            self.events_path,
            transport=FakeTransport(canned_response(["nav", "ready"], scale_iqr_ratio=2.0)),
        )
        self.assertIsNotNone(self.call_estimate(prior), prior.last_reason)
        prior = build_prior(
            self.events_path,
            transport=FakeTransport(canned_response(["nav", "ready"], scale_iqr_ratio=2.5)),
            max_scale_iqr_ratio=3.0,
        )
        estimate = self.call_estimate(prior)
        self.assertIsNotNone(estimate, prior.last_reason)
        self.assertEqual(estimate.scale_iqr_ratio, 2.5)

    def test_implausible_height_is_rejected(self) -> None:
        for height in (1.1, 2.2, 0.0, -0.5, 1.1999):
            with self.subTest(height=height):
                response = canned_response(["nav", "ready"])
                response["cameras"][1]["height_above_floor_m"] = height
                prior = build_prior(self.events_path, transport=FakeTransport(response))
                with self.assertLogs("yor_agent.robot.readiness_prior", level="WARNING") as logs:
                    self.assertIsNone(self.call_estimate(prior))
                self.assertEqual(prior.last_reason, "implausible_height")
                self.assertTrue(any("head height" in line for line in logs.output))
        # Only the stance row is gated; the nav row may be anything.
        response = canned_response(["nav", "ready"])
        response["cameras"][0]["height_above_floor_m"] = 0.2
        prior = build_prior(self.events_path, transport=FakeTransport(response))
        self.assertIsNotNone(self.call_estimate(prior), prior.last_reason)
        # The bounds are inclusive and follow the config.
        for height, overrides in ((1.2, {}), (2.1, {}), (1.1, {"min_height_m": 1.0})):
            with self.subTest(height=height, overrides=overrides):
                response = canned_response(["nav", "ready"])
                response["cameras"][1]["height_above_floor_m"] = height
                prior = build_prior(self.events_path, transport=FakeTransport(response), **overrides)
                estimate = self.call_estimate(prior)
                self.assertIsNotNone(estimate, prior.last_reason)
                self.assertEqual(estimate.human_height_m, height)

    def test_non_finite_metric_fields_with_a_scale_are_invalid(self) -> None:
        response = canned_response(["nav", "ready"])
        response["cameras"][1]["forward_m"] = None
        prior = build_prior(self.events_path, transport=FakeTransport(response))
        with self.assertLogs("yor_agent.robot.readiness_prior", level="WARNING"):
            self.assertIsNone(self.call_estimate(prior))
        self.assertEqual(prior.last_reason, "vggt_response_invalid")
        self.assertIn("forward_m", prior.last_diagnostics["error"])

    def test_missing_stance_frame_gives_frame_missing_without_a_call(self) -> None:
        transport = FakeTransport()
        prior = build_prior(self.events_path, transport=transport)
        ready = self.events_path.parent / "manipulation_events_20260911T000000.000000Z_frames/event00_ready.jpg"
        ready.unlink()
        with self.assertLogs("yor_agent.robot.readiness_prior", level="WARNING") as logs:
            self.assertIsNone(self.call_estimate(prior))
        self.assertEqual(prior.last_reason, "frame_missing")
        self.assertEqual(transport.calls, [])
        self.assertTrue(any("event frame missing" in line for line in logs.output))
        # No entry at all for the stance frame is the same reason.
        payload = json.loads(self.events_path.read_text(encoding="utf-8"))
        del payload["events"][0]["frames"]["ready"]
        self.events_path.write_text(json.dumps(payload), encoding="utf-8")
        prior = build_prior(self.events_path, transport=transport)
        with self.assertLogs("yor_agent.robot.readiness_prior", level="WARNING") as logs:
            self.assertIsNone(self.call_estimate(prior))
        self.assertEqual(prior.last_reason, "frame_missing")
        self.assertEqual(transport.calls, [])
        self.assertTrue(any("no ready frame entry" in line for line in logs.output))

    def test_missing_secondary_frame_is_dropped_with_a_warning(self) -> None:
        transport = FakeTransport()
        prior = build_prior(self.events_path, transport=transport)
        nav = self.events_path.parent / "manipulation_events_20260911T000000.000000Z_frames/event00_nav.jpg"
        nav.unlink()
        with self.assertLogs("yor_agent.robot.readiness_prior", level="WARNING") as logs:
            estimate = self.call_estimate(prior)
        self.assertIsNotNone(estimate, prior.last_reason)
        self.assertEqual(estimate.frames, ("ready",))
        self.assertEqual([f["label"] for f in transport.calls[0][1]["frames"]], ["ready"])
        self.assertEqual(estimate.human_height_m, 1.472)
        self.assertTrue(any("nav frame missing" in line for line in logs.output))
        # An absent entry (not just a missing file) is dropped the same way.
        payload = json.loads(self.events_path.read_text(encoding="utf-8"))
        del payload["events"][0]["frames"]["grasp"]
        self.events_path.write_text(json.dumps(payload), encoding="utf-8")
        prior = build_prior(
            self.events_path, transport=transport, vggt_frames=["ready", "grasp"]
        )
        with self.assertLogs("yor_agent.robot.readiness_prior", level="WARNING") as logs:
            estimate = self.call_estimate(prior)
        self.assertEqual(estimate.frames, ("ready",))
        self.assertTrue(any("no grasp frame entry" in line for line in logs.output))

    def test_unknown_object_never_calls_the_service(self) -> None:
        transport = FakeTransport()
        prior = build_prior(self.events_path, transport=transport)
        self.assertIsNone(self.call_estimate(prior, "table"))
        self.assertEqual(prior.last_reason, "no_event_match")
        self.assertEqual(transport.calls, [])

    def test_event_without_times_gives_frame_time_none(self) -> None:
        strip_event_times(self.events_path)
        prior = build_prior(self.events_path, transport=FakeTransport())
        estimate = self.call_estimate(prior)
        self.assertIsNotNone(estimate, prior.last_reason)
        self.assertIsNone(estimate.frame_time_s)
        metrics = estimate.to_metrics()
        self.assertIsNone(metrics["frame_time_s"])
        self.assertTrue(all(not (isinstance(v, float) and not math.isfinite(v)) for v in metrics.values()))
        json.dumps(metrics, allow_nan=False)
        json.dumps(dataclasses.asdict(estimate), allow_nan=False)

    def test_service_url_without_trailing_slash_is_joined(self) -> None:
        transport = FakeTransport()
        prior = build_prior(
            self.events_path,
            transport=transport,
            vggt_service_url="http://10.21.97.103:8117/",
            vggt_timeout_s=12.5,
        )
        self.assertIsNotNone(self.call_estimate(prior), prior.last_reason)
        self.assertEqual(transport.calls[0][0], "http://10.21.97.103:8117/stance")
        self.assertEqual(transport.calls[0][2], 12.5)

    def test_method_pnp_ignores_the_transport(self) -> None:
        transport = FakeTransport()
        prior = build_prior(self.events_path, transport=transport, method="pnp")
        with mock.patch.object(readiness_prior, "_cv2", side_effect=ImportError("no cv2")):
            self.assertIsNone(self.call_estimate(prior))
        self.assertEqual(prior.last_reason, "cv2_unavailable")
        self.assertEqual(transport.calls, [])

    def test_check_service_returns_the_health_dict(self) -> None:
        health = {
            "status": "ok", "model_loaded": True, "device": "cuda", "dtype": "float16",
            "weights": "/home/yor/models/vggt/vggt_1b_fp16.safetensors", "max_memory_mb": 2870.5,
        }
        transport = FakeTransport(health)
        prior = build_prior(self.events_path, transport=transport)
        self.assertEqual(prior.check_service(), health)
        self.assertEqual(transport.calls, [("http://127.0.0.1:8117/health", None, 5.0)])
        self.assertIsNone(prior.last_reason)
        self.assertEqual(prior.check_service(timeout_s=1.5), health)
        self.assertEqual(transport.calls[-1][2], 1.5)

    def test_check_service_failure_is_service_unavailable(self) -> None:
        transport = FakeTransport(error=urllib.error.URLError("connection refused"))
        prior = build_prior(self.events_path, transport=transport)
        with self.assertLogs("yor_agent.robot.readiness_prior", level="WARNING"):
            self.assertIsNone(prior.check_service())
        self.assertEqual(prior.last_reason, "vggt_service_unavailable")
        self.assertIn("URLError", prior.last_diagnostics["error"])
        self.assertEqual(prior.last_diagnostics["url"], "http://127.0.0.1:8117/health")
        # A 503 while the model loads is unavailable too.
        loading = urllib.error.HTTPError(
            "http://127.0.0.1:8117/health", 503, "Service Unavailable", {},
            io.BytesIO(b'{"status":"loading","model_loaded":false}'),
        )
        prior = build_prior(self.events_path, transport=FakeTransport(error=loading))
        with self.assertLogs("yor_agent.robot.readiness_prior", level="WARNING"):
            self.assertIsNone(prior.check_service())
        self.assertEqual(prior.last_reason, "vggt_service_unavailable")
        self.assertEqual(prior.last_diagnostics["http_status"], 503)
        # A body that is not an object is not a health report.
        prior = build_prior(self.events_path, transport=FakeTransport(["ok"]))
        self.assertIsNone(prior.check_service())
        self.assertEqual(prior.last_reason, "vggt_service_unavailable")
        self.assertIn("not an object", prior.last_diagnostics["error"])

    def test_construction_never_touches_the_transport(self) -> None:
        transport = FakeTransport(error=AssertionError("no network at construction"))
        build_prior(self.events_path, transport=transport)
        self.assertEqual(transport.calls, [])


class _StanceHandler(BaseHTTPRequestHandler):
    """A stand-in for the VGGT service: /health, /stance, and failure modes."""

    def log_message(self, format, *args):  # noqa: A002 - BaseHTTPRequestHandler API
        return

    def _send_json(self, status: int, body) -> None:
        data = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        self.server.requests.append(("GET", self.path, dict(self.headers), None))
        if self.path == "/health":
            self._send_json(200, {"status": "ok", "model_loaded": True, "device": "cpu"})
        elif self.path == "/loading":
            self._send_json(503, {"status": "loading", "model_loaded": False})
        elif self.path == "/broken":
            data = b"not json"
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        else:
            self._send_json(404, {"detail": "not found"})

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        self.server.requests.append(("POST", self.path, dict(self.headers), payload))
        mode = self.server.stance_mode
        if self.path != "/stance":
            self._send_json(404, {"detail": "not found"})
        elif mode == "http_error":
            self._send_json(400, {"detail": "depth shape mismatch"})
        elif mode == "crash":
            self._send_json(500, {"detail": "inference failed: RuntimeError: boom"})
        elif mode == "slow":
            self.server.release.wait(3.0)
            self._send_json(200, canned_response([f["label"] for f in payload["frames"]]))
        else:
            labels = [frame["label"] for frame in payload["frames"]]
            body = canned_response(labels)
            body["received"] = {
                "labels": labels,
                "robot_keys": sorted(payload["robot"]),
                "ground_camera_height_m": payload["robot"]["ground_camera_height_m"],
            }
            self._send_json(200, body)


class UrllibTransportTest(TempDirMixin, unittest.TestCase):
    """The default transport and the whole vggt path against a local http.server."""

    def setUp(self) -> None:
        super().setUp()
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), _StanceHandler)
        self.server.daemon_threads = True
        self.server.requests = []
        self.server.stance_mode = "ok"
        self.server.release = threading.Event()
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self._stop_server)
        host, port = self.server.server_address[:2]
        self.url = f"http://{host}:{port}"
        self.events_path = make_events_file(self.directory)
        write_jpeg_frames(self.events_path)
        self.rgb = np.zeros((ZED_H, ZED_W, 3), dtype=np.uint8)
        self.depth = np.full((ZED_H, ZED_W), 2.5, dtype=np.float32)

    def _stop_server(self) -> None:
        self.server.release.set()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5.0)

    def call_estimate(self, prior: ReadinessPrior):
        return prior.estimate(
            "the can",
            self.rgb,
            self.depth,
            ZED_K,
            POSE,
            down=DOWN,
            planar_forward=PLANAR_FORWARD,
            planar_left=PLANAR_LEFT,
            target_xy=TARGET_XY,
            camera_height_m=CAMERA_HEIGHT_M,
        )

    def test_get_returns_the_json_body(self) -> None:
        body = readiness_prior._urllib_transport(f"{self.url}/health", None, 5.0)
        self.assertEqual(body, {"status": "ok", "model_loaded": True, "device": "cpu"})
        method, path, headers, payload = self.server.requests[-1]
        self.assertEqual((method, path, payload), ("GET", "/health", None))
        self.assertEqual(headers.get("Accept"), "application/json")

    def test_post_sends_json_and_returns_the_json_body(self) -> None:
        payload = {"robot": {"ground_camera_height_m": 1.054, "label": "zed"}, "frames": [{"label": "ready", "jpeg_base64": "AA=="}]}
        body = readiness_prior._urllib_transport(f"{self.url}/stance", payload, 5.0)
        self.assertEqual(body["received"]["labels"], ["ready"])
        self.assertEqual(body["received"]["ground_camera_height_m"], 1.054)
        method, path, headers, received = self.server.requests[-1]
        self.assertEqual((method, path), ("POST", "/stance"))
        self.assertEqual(headers.get("Content-Type"), "application/json")
        self.assertEqual(received, payload)

    def test_http_errors_and_non_json_bodies_raise(self) -> None:
        with self.assertRaises(urllib.error.HTTPError) as caught:
            readiness_prior._urllib_transport(f"{self.url}/loading", None, 5.0)
        self.assertEqual(caught.exception.code, 503)
        with self.assertRaises(urllib.error.HTTPError):
            readiness_prior._urllib_transport(f"{self.url}/missing", None, 5.0)
        with self.assertRaises(ValueError):
            readiness_prior._urllib_transport(f"{self.url}/broken", None, 5.0)

    def test_estimate_end_to_end_over_http(self) -> None:
        prior = build_prior(self.events_path, vggt_service_url=self.url)
        self.assertEqual(prior.check_service(), {"status": "ok", "model_loaded": True, "device": "cpu"})
        estimate = self.call_estimate(prior)
        self.assertIsNotNone(estimate, (prior.last_reason, prior.last_diagnostics))
        self.assertEqual(estimate.method, "vggt")
        self.assertEqual(estimate.frames, ("nav", "ready"))
        _, heading, feet, bearing = expected_stance(CAMERA_ROWS["ready"], TARGET_XY)
        self.assertAlmostEqual(estimate.bearing_rad, bearing, places=12)
        self.assertAlmostEqual(estimate.heading_rad, heading, places=12)
        self.assertAlmostEqual(estimate.stance_xy[0], feet[0], places=12)
        self.assertAlmostEqual(estimate.stance_xy[1], feet[1], places=12)
        self.assertGreater(estimate.service_time_s, 0.0)
        self.assertEqual(
            prior.last_diagnostics["received"]["robot_keys"],
            sorted(
                [
                    "label", "rgb_jpeg_base64", "depth_png16_base64", "intrinsics",
                    "down", "planar_forward", "planar_left", "ground_camera_height_m",
                ]
            ),
        )
        self.assertEqual(prior.last_diagnostics["received"]["ground_camera_height_m"], CAMERA_HEIGHT_M)
        posted = [entry for entry in self.server.requests if entry[0] == "POST"]
        self.assertEqual(len(posted), 1)
        self.assertEqual(posted[0][3]["robot"]["label"], "zed")
        self.assertEqual([f["label"] for f in posted[0][3]["frames"]], ["nav", "ready"])
        json.dumps(estimate.to_metrics(), allow_nan=False)

    def test_service_400_and_500_are_service_unavailable_with_detail(self) -> None:
        prior = build_prior(self.events_path, vggt_service_url=self.url)
        self.server.stance_mode = "http_error"
        with self.assertLogs("yor_agent.robot.readiness_prior", level="WARNING"):
            self.assertIsNone(self.call_estimate(prior))
        self.assertEqual(prior.last_reason, "vggt_service_unavailable")
        self.assertEqual(prior.last_diagnostics["http_status"], 400)
        self.assertIn("depth shape mismatch", prior.last_diagnostics["detail"])
        self.server.stance_mode = "crash"
        with self.assertLogs("yor_agent.robot.readiness_prior", level="WARNING"):
            self.assertIsNone(self.call_estimate(prior))
        self.assertEqual(prior.last_reason, "vggt_service_unavailable")
        self.assertEqual(prior.last_diagnostics["http_status"], 500)
        self.assertIn("inference failed", prior.last_diagnostics["detail"])

    def test_timeout_is_service_unavailable(self) -> None:
        prior = build_prior(self.events_path, vggt_service_url=self.url, vggt_timeout_s=0.2)
        self.server.stance_mode = "slow"
        with self.assertLogs("yor_agent.robot.readiness_prior", level="WARNING") as logs:
            self.assertIsNone(self.call_estimate(prior))
        self.assertEqual(prior.last_reason, "vggt_service_unavailable")
        self.assertLess(prior.last_diagnostics["service_time_s"], 2.0)
        self.assertTrue(any("timed out" in line.lower() or "timeout" in line.lower() for line in logs.output))

    def test_unreachable_service_is_service_unavailable(self) -> None:
        probe = socket.socket()
        probe.bind(("127.0.0.1", 0))
        closed_port = probe.getsockname()[1]
        probe.close()
        prior = build_prior(self.events_path, vggt_service_url=f"http://127.0.0.1:{closed_port}")
        with self.assertLogs("yor_agent.robot.readiness_prior", level="WARNING"):
            self.assertIsNone(prior.check_service(timeout_s=2.0))
        self.assertEqual(prior.last_reason, "vggt_service_unavailable")
        with self.assertLogs("yor_agent.robot.readiness_prior", level="WARNING"):
            self.assertIsNone(self.call_estimate(prior))
        self.assertEqual(prior.last_reason, "vggt_service_unavailable")
        self.assertIn("URLError", prior.last_diagnostics["error"])


def synthetic_scene():
    """3-D points seen by the ZED (depth 1.75-4.25 m) and their human-camera pixels."""

    rows, cols = np.mgrid[6 : ZED_H - 6 : 9, 6 : ZED_W - 6 : 9]
    rows = rows.ravel().astype(np.float64)
    cols = cols.ravel().astype(np.float64)

    def depth_field(r, c):
        return 2.0 + 2.0 * c / (ZED_W - 1) + 0.25 * np.sin(r / 17.0)

    full_rows, full_cols = np.mgrid[0:ZED_H, 0:ZED_W]
    depth = depth_field(full_rows.astype(np.float64), full_cols.astype(np.float64)).astype(
        np.float32
    )
    z = depth_field(rows, cols)
    points = np.stack(
        [(cols - ZED_K[0, 2]) * z / ZED_K[0, 0], (rows - ZED_K[1, 2]) * z / ZED_K[1, 1], z],
        axis=1,
    )
    rotation = HUMAN_R_ZH.T
    tvec = -rotation @ HUMAN_CENTER
    human = np.einsum("ij,nj->ni", rotation, points) + tvec
    u = HUMAN_FOCAL * human[:, 0] / human[:, 2] + (FRAME_W - 1) / 2.0
    v = HUMAN_FOCAL * human[:, 1] / human[:, 2] + (FRAME_H - 1) / 2.0
    visible = (human[:, 2] > 0.2) & (u >= 0) & (u <= FRAME_W - 1) & (v >= 0) & (v <= FRAME_H - 1)
    event_px = np.stack([u, v], axis=1)[visible]
    robot_px = np.stack([cols, rows], axis=1)[visible]
    return depth, event_px, robot_px


def matcher_with_outliers(event_px, robot_px, *, fraction: float = 0.2, seed: int = 7):
    rng = np.random.default_rng(seed)
    corrupted = event_px.copy()
    count = int(round(fraction * len(event_px)))
    index = rng.choice(len(event_px), size=count, replace=False)
    corrupted[index] = rng.uniform([0.0, 0.0], [FRAME_W - 1.0, FRAME_H - 1.0], size=(count, 2))

    def matcher(event_rgb, robot_rgb):
        return corrupted.copy(), robot_px.copy()

    return matcher, len(event_px) - count


@unittest.skipUnless(HAS_CV2, "cv2 is not installed")
class SyntheticPnPTest(TempDirMixin, unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        import cv2

        self.cv2 = cv2
        self.depth, self.event_px, self.robot_px = synthetic_scene()
        self.assertGreaterEqual(len(self.event_px), 60)
        self.rgb = np.zeros((ZED_H, ZED_W, 3), dtype=np.uint8)

    def write_events(self, directory: Path, **kwargs) -> Path:
        path = make_events_file(directory, **kwargs)
        for frame in ("nav", "ready"):
            image = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
            self.cv2.imwrite(str(path.parent / f"manipulation_events_20260911T000000.000000Z_frames/event00_{frame}.jpg"), image)
        return path

    def call_estimate(self, prior: ReadinessPrior):
        return prior.estimate(
            "the can",
            self.rgb,
            self.depth,
            ZED_K,
            POSE,
            down=DOWN,
            planar_forward=PLANAR_FORWARD,
            planar_left=PLANAR_LEFT,
            target_xy=TARGET_XY,
        )

    def test_estimate_recovers_human_pose_with_outliers(self) -> None:
        matcher, true_count = matcher_with_outliers(self.event_px, self.robot_px)
        prior = build_prior(self.write_events(self.directory), matcher=matcher, method="pnp")
        estimate = self.call_estimate(prior)
        self.assertIsNotNone(estimate, prior.last_reason)
        expected_xy = expected_world_xy(HUMAN_CENTER)
        self.assertLess(math.hypot(estimate.human_xy[0] - expected_xy[0], estimate.human_xy[1] - expected_xy[1]), 0.01)
        self.assertLess(abs(wrap(estimate.heading_rad - expected_heading(HUMAN_R_ZH[:, 2]))), math.radians(1.0))
        self.assertGreaterEqual(estimate.inliers, true_count)
        self.assertAlmostEqual(
            estimate.bearing_rad,
            wrap(math.atan2(estimate.human_xy[1] - TARGET_XY[1], estimate.human_xy[0] - TARGET_XY[0])),
        )
        self.assertLess(estimate.reprojection_error_px, 1.0)
        self.assertAlmostEqual(estimate.focal_px, HUMAN_FOCAL)
        self.assertEqual(estimate.suggested_arm, "right")
        self.assertEqual(estimate.method, "pnp")

    def test_focal_sweep_picks_the_focal_nearest_the_truth(self) -> None:
        matcher, true_count = matcher_with_outliers(self.event_px, self.robot_px)
        prior = build_prior(
            self.write_events(self.directory, intrinsics=None), matcher=matcher, method="pnp"
        )
        estimate = self.call_estimate(prior)
        self.assertIsNotNone(estimate, prior.last_reason)
        sweep = prior.last_diagnostics["sweep"]
        self.assertEqual([entry["fov_deg"] for entry in sweep], [60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0])
        nearest = min(sweep, key=lambda entry: abs(entry["focal_px"] - HUMAN_FOCAL))
        self.assertEqual(nearest["fov_deg"], 90.0)
        self.assertAlmostEqual(estimate.focal_px, nearest["focal_px"])
        self.assertGreaterEqual(estimate.inliers, true_count)
        expected_xy = expected_world_xy(HUMAN_CENTER)
        self.assertLess(math.hypot(estimate.human_xy[0] - expected_xy[0], estimate.human_xy[1] - expected_xy[1]), 0.01)

    def test_min_inliers_above_the_scene_rejects(self) -> None:
        matcher, true_count = matcher_with_outliers(self.event_px, self.robot_px)
        prior = build_prior(
            self.write_events(self.directory),
            matcher=matcher,
            min_inliers=len(self.event_px) + 10,
            method="pnp",
        )
        self.assertIsNone(self.call_estimate(prior))
        self.assertEqual(prior.last_reason, "too_few_matches")
        prior = build_prior(
            self.write_events(self.directory / "b"),
            matcher=matcher,
            min_inliers=len(self.event_px),
            method="pnp",
        )
        self.assertIsNone(self.call_estimate(prior))
        self.assertEqual(prior.last_reason, "too_few_inliers")

    def test_default_sift_matcher_end_to_end_on_an_identical_view(self) -> None:
        # The event frame IS the ZED image: PnP must put the human camera at
        # the ZED, i.e. at the robot pose with its yaw.
        rng = np.random.default_rng(3)
        texture = rng.integers(0, 256, size=(ZED_H, ZED_W, 3), dtype=np.uint8)
        texture = self.cv2.GaussianBlur(texture, (0, 0), 1.2)
        path = make_events_file(self.directory, intrinsics=None, frame_size=(ZED_W, ZED_H), frame_extension="png")
        frame = path.parent / "manipulation_events_20260911T000000.000000Z_frames/event00_ready.png"
        self.cv2.imwrite(str(frame), np.ascontiguousarray(texture[:, :, ::-1]))
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["intrinsics"] = {
            "model": "pinhole", "width": ZED_W, "height": ZED_H,
            "fx": 300.0, "fy": 300.0, "cx": 159.5, "cy": 119.5, "source_path": "x",
        }
        payload["events"][0]["frames"]["ready"]["scale"] = 1.0
        path.write_text(json.dumps(payload), encoding="utf-8")
        prior = build_prior(path, min_inliers=30, method="pnp")
        estimate = prior.estimate(
            "can", texture, self.depth, ZED_K, POSE,
            down=DOWN, planar_forward=PLANAR_FORWARD, planar_left=PLANAR_LEFT, target_xy=TARGET_XY,
        )
        self.assertIsNotNone(estimate, (prior.last_reason, prior.last_diagnostics))
        self.assertLess(math.hypot(estimate.human_xy[0] - POSE[0], estimate.human_xy[1] - POSE[1]), 0.05)
        self.assertLess(abs(wrap(estimate.heading_rad - POSE[2])), math.radians(2.0))
        self.assertGreaterEqual(estimate.inliers, 30)


if __name__ == "__main__":
    unittest.main()
