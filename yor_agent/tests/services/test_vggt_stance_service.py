from __future__ import annotations

import base64
import contextlib
import io
import math
from pathlib import Path
import sys
import unittest
from unittest import mock

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

SERVICES_ROOT = Path(__file__).resolve().parents[2] / "services"
if str(SERVICES_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVICES_ROOT))

from vggt_stance import service  # noqa: E402
from vggt_stance.service import (  # noqa: E402
    VggtRaw,
    align_scale,
    build_parser,
    camera_row,
    content_region,
    decode_depth_png16,
    decode_jpeg_size,
)

MODEL = 518
BAND_ROWS = (112, 406)  # a 672x376 ZED frame padded to 518x518: 294 content rows

# Floor axes of desk_can_angled.npz (ZED optical frame), orthonormalised.
_DOWN = np.array([-0.00115063, 0.93348312, 0.35861949])
_FORWARD = np.array([4.42040600e-04, -3.58619215e-01, 9.33483831e-01])
_LEFT = np.array([-0.99999924, -0.00123262, 0.0])


def _orthonormal_axes() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    down = _DOWN / np.linalg.norm(_DOWN)
    forward = _FORWARD - (_FORWARD @ down) * down
    forward /= np.linalg.norm(forward)
    left = np.cross(down, forward)
    return down, forward, left


DOWN, FORWARD, LEFT = _orthonormal_axes()
CAMERA_HEIGHT = 1.054


def camera_from_world(axis: np.ndarray, down: np.ndarray) -> np.ndarray:
    """Rows = the camera's x/y/z axes in world coordinates (OpenCV: y down, z forward)."""
    z = axis / np.linalg.norm(axis)
    y = down - (down @ z) * z
    y /= np.linalg.norm(y)
    x = np.cross(y, z)
    return np.stack([x, y, z])


def extrinsic(rotation: np.ndarray, centre: np.ndarray) -> np.ndarray:
    return np.hstack([rotation, (-rotation @ centre)[:, None]])


def tilted_axis(heading_deg: float, tilt_down_deg: float) -> np.ndarray:
    h = math.radians(heading_deg)
    t = math.radians(tilt_down_deg)
    return math.cos(t) * (math.cos(h) * FORWARD + math.sin(h) * LEFT) + math.sin(t) * DOWN


def padded_image0(band: tuple[int, int] = BAND_ROWS, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    image = np.ones((3, MODEL, MODEL), dtype=np.float32)
    image[:, band[0] : band[1], :] = rng.uniform(0.1, 0.9, size=(3, band[1] - band[0], MODEL)).astype(np.float32)
    return image


def jpeg_base64(width: int, height: int, seed: int = 1) -> str:
    rng = np.random.default_rng(seed)
    rgb = rng.integers(0, 255, size=(height, width, 3), dtype=np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(rgb, "RGB").save(buffer, format="JPEG", quality=95)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def depth_png16_base64(depth_mm: np.ndarray) -> str:
    buffer = io.BytesIO()
    Image.fromarray(depth_mm.astype(np.uint16)).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


class FakeRunner:
    """Stands in for VggtRunner: N synthetic cameras, image 0 at the origin."""

    device = "cpu"
    dtype_name = "float32"
    weights = "/fake/weights.safetensors"

    def __init__(
        self,
        *,
        depth0_value: float = 1.5,
        conf_value: float = 3.0,
        depth_head_frames: int = 1,
        error: Exception | None = None,
    ) -> None:
        self.depth0_value = depth0_value
        self.conf_value = conf_value
        self.depth_head_frames = depth_head_frames
        self.error = error
        self.paths: list[str] = []
        self.rotation = camera_from_world(tilted_axis(27.0, 8.0), DOWN)
        self.centres = [np.array([0.25, -0.15, -0.2]), np.array([-0.7, 0.1, 0.4])]

    def extrinsics(self, count: int) -> np.ndarray:
        rows = [np.hstack([np.eye(3), np.zeros((3, 1))])]
        for index in range(count - 1):
            rows.append(extrinsic(self.rotation, self.centres[index % len(self.centres)]))
        return np.stack(rows).astype(np.float32)

    def max_memory_mb(self) -> float:
        return 123.5

    def run(self, paths: list[str]) -> VggtRaw:
        self.paths = list(paths)
        if self.error is not None:
            raise self.error
        count = len(paths)
        rng = np.random.default_rng(7)
        depth0 = np.zeros((MODEL, MODEL), dtype=np.float32)
        band = slice(BAND_ROWS[0], BAND_ROWS[1])
        depth0[band, :] = self.depth0_value * rng.uniform(0.99, 1.01, size=(BAND_ROWS[1] - BAND_ROWS[0], MODEL))
        conf0 = np.full((MODEL, MODEL), self.conf_value, dtype=np.float32)
        intrinsics = np.tile(np.array([[210.0, 0.0, 259.0], [0.0, 210.0, 259.0], [0.0, 0.0, 1.0]], np.float32), (count, 1, 1))
        return VggtRaw(
            extrinsics=self.extrinsics(count),
            intrinsics=intrinsics,
            depth0=depth0,
            conf0=conf0,
            image0=padded_image0(),
            model_hw=(MODEL, MODEL),
            timing={"preprocess": 0.1, "aggregator": 0.2, "camera_head": 0.03, "depth_head": 0.04},
            depth_head_frames=self.depth_head_frames,
        )


class ContentRegionTest(unittest.TestCase):
    def test_white_padded_band(self):
        self.assertEqual(content_region(padded_image0()), (112, 406, 0, MODEL))

    def test_constant_image_rejected(self):
        with self.assertRaises(ValueError):
            content_region(np.ones((3, MODEL, MODEL), dtype=np.float32))


class AlignScaleTest(unittest.TestCase):
    def setUp(self):
        self.region = (BAND_ROWS[0], BAND_ROWS[1], 0, MODEL)
        rows = BAND_ROWS[1] - BAND_ROWS[0]
        rng = np.random.default_rng(3)
        # A ZED depth already at the region size, so the nearest mapping is the identity.
        self.zed = rng.uniform(0.5, 6.0, size=(rows, MODEL))
        self.depth0 = np.zeros((MODEL, MODEL))
        self.depth0[BAND_ROWS[0] : BAND_ROWS[1], :] = self.zed / 2.0 * rng.uniform(0.99, 1.01, size=(rows, MODEL))
        self.conf0 = np.full((MODEL, MODEL), 3.0)

    def test_known_ratio(self):
        result = align_scale(self.zed, self.depth0, self.conf0, self.region)
        self.assertAlmostEqual(result.scale, 2.0, delta=0.01)
        self.assertGreater(result.iqr_ratio, 1.0)
        self.assertLess(result.iqr_ratio, 1.05)
        self.assertEqual(result.pixels, self.zed.size)
        self.assertEqual(result.conf_threshold_used, 2.0)

    def test_threshold_is_min_of_option_and_median(self):
        conf0 = np.full((MODEL, MODEL), 1.0)
        result = align_scale(self.zed, self.depth0, conf0, self.region)
        self.assertEqual(result.conf_threshold_used, 1.0)
        self.assertEqual(result.pixels, self.zed.size)
        result = align_scale(self.zed, self.depth0, self.conf0, self.region, conf_threshold=0.5)
        self.assertEqual(result.conf_threshold_used, 0.5)

    def test_invalid_zed_pixels_ignored(self):
        zed = self.zed.copy()
        zed[:10, :] = np.nan
        zed[10:20, :] = np.inf
        zed[20:30, :] = -np.inf
        zed[30:40, :] = 0.05  # below min_depth_m
        zed[40:50, :] = 9.0  # above max_depth_m
        result = align_scale(zed, self.depth0, self.conf0, self.region)
        self.assertEqual(result.pixels, self.zed.size - 50 * MODEL)
        self.assertAlmostEqual(result.scale, 2.0, delta=0.01)

    def test_empty_gives_none(self):
        result = align_scale(np.full_like(self.zed, np.nan), self.depth0, self.conf0, self.region)
        self.assertIsNone(result.scale)
        self.assertIsNone(result.iqr_ratio)
        self.assertEqual(result.pixels, 0)

    def test_zed_resized_by_nearest_index_mapping(self):
        zed = np.full((376, 672), 3.0)
        depth0 = np.zeros((MODEL, MODEL))
        depth0[BAND_ROWS[0] : BAND_ROWS[1], :] = 1.5
        result = align_scale(zed, depth0, self.conf0, self.region)
        self.assertAlmostEqual(result.scale, 2.0, places=9)
        self.assertAlmostEqual(result.iqr_ratio, 1.0, places=9)
        self.assertEqual(result.pixels, (BAND_ROWS[1] - BAND_ROWS[0]) * MODEL)


class CameraRowTest(unittest.TestCase):
    def setUp(self):
        self.intrinsics = np.array([[211.5, 0.0, 259.0], [0.0, 211.5, 259.0], [0.0, 0.0, 1.0]])
        self.kwargs = dict(down=DOWN, forward=FORWARD, left=LEFT, camera_height_m=CAMERA_HEIGHT, model_w=MODEL)

    def test_identity_is_the_robot_camera(self):
        row = camera_row(
            np.hstack([np.eye(3), np.zeros((3, 1))]), self.intrinsics, scale=2.0, label="zed", source_wh=(672, 376), **self.kwargs
        )
        self.assertEqual(row["label"], "zed")
        self.assertAlmostEqual(row["forward_m"], 0.0, places=12)
        self.assertAlmostEqual(row["left_m"], 0.0, places=12)
        self.assertAlmostEqual(row["height_above_floor_m"], CAMERA_HEIGHT, places=12)
        self.assertAlmostEqual(row["heading_deg"], 0.0, places=6)
        self.assertEqual((row["width"], row["height"]), (672, 376))

    def test_known_rotation_and_centre(self):
        centre = np.array([0.5, -0.3, -0.4])
        rotation = camera_from_world(tilted_axis(27.0, 8.0), DOWN)
        row = camera_row(extrinsic(rotation, centre), self.intrinsics, scale=1.0, label="ready", source_wh=(1280, 960), **self.kwargs)
        self.assertAlmostEqual(row["forward_m"], float(centre @ FORWARD), places=6)
        self.assertAlmostEqual(row["left_m"], float(centre @ LEFT), places=6)
        self.assertAlmostEqual(row["height_above_floor_m"], CAMERA_HEIGHT - float(centre @ DOWN), places=6)
        self.assertAlmostEqual(row["heading_deg"], 27.0, places=6)

    def test_scale_multiplies_the_centre_only(self):
        centre = np.array([0.5, -0.3, -0.4])
        rotation = camera_from_world(tilted_axis(-13.0, 5.0), DOWN)
        row = camera_row(extrinsic(rotation, centre), self.intrinsics, scale=2.0, label="nav", source_wh=(1280, 960), **self.kwargs)
        self.assertAlmostEqual(row["forward_m"], 2.0 * float(centre @ FORWARD), places=6)
        self.assertAlmostEqual(row["height_above_floor_m"], CAMERA_HEIGHT - 2.0 * float(centre @ DOWN), places=6)
        self.assertAlmostEqual(row["heading_deg"], -13.0, places=6)

    def test_focal_scaled_to_source_width(self):
        row = camera_row(np.hstack([np.eye(3), np.zeros((3, 1))]), self.intrinsics, scale=1.0, label="a", source_wh=(1280, 960), **self.kwargs)
        self.assertAlmostEqual(row["focal_px_model"], 211.5)
        self.assertAlmostEqual(row["focal_px"], 211.5 * 1280 / MODEL)

    def test_without_scale_metric_fields_are_none(self):
        rotation = camera_from_world(tilted_axis(27.0, 8.0), DOWN)
        row = camera_row(extrinsic(rotation, np.array([0.5, -0.3, -0.4])), self.intrinsics, scale=None, label="ready", source_wh=(1280, 960), **self.kwargs)
        self.assertIsNone(row["forward_m"])
        self.assertIsNone(row["left_m"])
        self.assertIsNone(row["height_above_floor_m"])
        self.assertAlmostEqual(row["heading_deg"], 27.0, places=6)
        self.assertAlmostEqual(row["focal_px"], 211.5 * 1280 / MODEL)


class DecodeTest(unittest.TestCase):
    def test_depth_png16_round_trip(self):
        depth_mm = np.array([[0, 2500, 65535], [1, 1000, 0]], dtype=np.uint16)
        depth = decode_depth_png16(depth_png16_base64(depth_mm))
        self.assertEqual(depth.dtype, np.float32)
        self.assertEqual(depth.shape, (2, 3))
        self.assertTrue(np.isnan(depth[0, 0]))
        self.assertTrue(np.isnan(depth[1, 2]))
        self.assertAlmostEqual(float(depth[0, 1]), 2.5, places=6)
        self.assertAlmostEqual(float(depth[0, 2]), 65.535, places=5)
        self.assertAlmostEqual(float(depth[1, 0]), 0.001, places=6)

    def test_depth_png_rejects_rgb_and_garbage(self):
        with self.assertRaises(ValueError):
            decode_depth_png16(jpeg_base64(8, 4))
        with self.assertRaises(ValueError):
            decode_depth_png16(base64.b64encode(b"not a png").decode("ascii"))
        with self.assertRaises(ValueError):
            decode_depth_png16("not base64!!")

    def test_jpeg_size(self):
        self.assertEqual(decode_jpeg_size(jpeg_base64(64, 36)), (64, 36))
        with self.assertRaises(ValueError):
            decode_jpeg_size(base64.b64encode(b"nope").decode("ascii"))


def make_request(frame_labels=("nav", "ready"), *, rgb_wh=(64, 36), depth_shape=None, depth_mm=3000):
    width, height = rgb_wh
    depth = np.full(depth_shape or (height, width), depth_mm, dtype=np.uint16)
    return {
        "robot": {
            "rgb_jpeg_base64": jpeg_base64(width, height, seed=11),
            "depth_png16_base64": depth_png16_base64(depth),
            "intrinsics": [[267.2, 0.0, 337.2], [0.0, 267.2, 182.3], [0.0, 0.0, 1.0]],
            "down": DOWN.tolist(),
            "planar_forward": FORWARD.tolist(),
            "planar_left": LEFT.tolist(),
            "ground_camera_height_m": CAMERA_HEIGHT,
        },
        "frames": [{"label": label, "jpeg_base64": jpeg_base64(128, 96, seed=20 + i)} for i, label in enumerate(frame_labels)],
    }


class ServiceEndpointTest(unittest.TestCase):
    def setUp(self):
        service.set_runner(None)
        self.client = TestClient(service.app)

    def tearDown(self):
        service.set_runner(None)

    def test_health_before_and_after_runner(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 503)
        body = response.json()
        self.assertEqual(body["status"], "loading")
        self.assertFalse(body["model_loaded"])
        self.assertIsNone(body["max_memory_mb"])

        service.set_runner(FakeRunner())
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "ok")
        self.assertTrue(body["model_loaded"])
        self.assertEqual(body["device"], "cpu")
        self.assertEqual(body["dtype"], "float32")
        self.assertEqual(body["weights"], "/fake/weights.safetensors")
        self.assertEqual(body["max_memory_mb"], 123.5)

    def test_stance_without_runner_is_503(self):
        response = self.client.post("/stance", json=make_request())
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["detail"], "model not loaded")

    def test_stance_rows_in_request_order(self):
        runner = FakeRunner(depth_head_frames=1)
        service.set_runner(runner)
        response = self.client.post("/stance", json=make_request(("nav", "ready")))
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["schema_version"], "yor-vggt-stance-v1")
        self.assertEqual(body["device"], "cpu")
        self.assertEqual(body["dtype"], "float32")
        self.assertEqual(body["model_image_hw"], [MODEL, MODEL])
        self.assertEqual(body["content_region"], {"rows": [112, 406], "cols": [0, MODEL]})
        self.assertAlmostEqual(body["scale"], 2.0, delta=0.02)
        self.assertLess(body["scale_iqr_ratio"], 1.05)
        self.assertGreater(body["scale_pixels"], 0)
        self.assertEqual(body["conf_threshold"], 2.0)
        self.assertEqual(body["depth_head_frames"], 1)
        self.assertEqual(body["max_memory_mb"], 123.5)
        self.assertAlmostEqual(body["robot_focal_px"], 267.2)
        for key in ("decode", "preprocess", "aggregator", "camera_head", "depth_head", "total"):
            self.assertIn(key, body["timing_s"])
        self.assertAlmostEqual(body["timing_s"]["aggregator"], 0.2)

        robot = body["robot_camera"]
        self.assertEqual(robot["label"], "zed")
        self.assertAlmostEqual(robot["forward_m"], 0.0, places=9)
        self.assertAlmostEqual(robot["left_m"], 0.0, places=9)
        self.assertAlmostEqual(robot["height_above_floor_m"], CAMERA_HEIGHT, places=9)
        self.assertAlmostEqual(robot["heading_deg"], 0.0, places=6)
        self.assertEqual((robot["width"], robot["height"]), (64, 36))
        self.assertAlmostEqual(robot["focal_px"], 210.0 * 64 / MODEL)

        self.assertEqual([row["label"] for row in body["cameras"]], ["nav", "ready"])
        extrinsics = runner.extrinsics(3)
        for index, row in enumerate(body["cameras"]):
            expected = camera_row(
                extrinsics[index + 1],
                np.array([[210.0, 0.0, 259.0], [0.0, 210.0, 259.0], [0.0, 0.0, 1.0]]),
                scale=body["scale"],
                down=DOWN,
                forward=FORWARD,
                left=LEFT,
                camera_height_m=CAMERA_HEIGHT,
                label=row["label"],
                source_wh=(128, 96),
                model_w=MODEL,
            )
            for key in ("forward_m", "left_m", "height_above_floor_m", "heading_deg", "focal_px", "focal_px_model"):
                self.assertAlmostEqual(row[key], expected[key], places=5, msg=key)
            self.assertEqual((row["width"], row["height"]), (128, 96))
        self.assertAlmostEqual(body["cameras"][0]["heading_deg"], 27.0, places=4)

        # Image 0 is the robot JPEG, then one file per frame in request order.
        names = [Path(path).name for path in runner.paths]
        self.assertEqual(names, ["00_robot.jpg", "01_nav.jpg", "02_ready.jpg"])

    def test_depth_head_frames_echoed(self):
        service.set_runner(FakeRunner(depth_head_frames=3))
        response = self.client.post("/stance", json=make_request(("nav", "ready")))
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["depth_head_frames"], 3)

    def test_scale_less_response_keeps_heading(self):
        # A degenerate VGGT depth (all zero) leaves no pixel for the alignment.
        service.set_runner(FakeRunner(depth0_value=0.0))
        response = self.client.post("/stance", json=make_request(("ready",)))
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertIsNone(body["scale"])
        self.assertIsNone(body["scale_iqr_ratio"])
        self.assertEqual(body["scale_pixels"], 0)
        row = body["cameras"][0]
        self.assertIsNone(row["forward_m"])
        self.assertIsNone(row["left_m"])
        self.assertIsNone(row["height_above_floor_m"])
        self.assertAlmostEqual(row["heading_deg"], 27.0, places=4)
        self.assertIsNone(body["robot_camera"]["height_above_floor_m"])
        self.assertAlmostEqual(body["robot_camera"]["heading_deg"], 0.0, places=6)

    def test_options_are_forwarded(self):
        service.set_runner(FakeRunner())
        payload = make_request(("ready",))
        payload["options"] = {"conf_threshold": 1.5, "min_depth_m": 0.2, "max_depth_m": 8.0}
        response = self.client.post("/stance", json=payload)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["conf_threshold"], 1.5)
        payload["options"] = {"conf_threshold": 2.0, "min_depth_m": 0.2, "max_depth_m": 2.5}
        response = self.client.post("/stance", json=payload)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertIsNone(response.json()["scale"])  # the 3.0 m ZED depth is now out of range

    def test_bad_base64_is_400(self):
        service.set_runner(FakeRunner())
        payload = make_request()
        payload["robot"]["rgb_jpeg_base64"] = "not base64!!"
        response = self.client.post("/stance", json=payload)
        self.assertEqual(response.status_code, 400)
        self.assertIn("base64", response.json()["detail"])
        payload = make_request()
        payload["frames"][1]["jpeg_base64"] = base64.b64encode(b"garbage").decode("ascii")
        response = self.client.post("/stance", json=payload)
        self.assertEqual(response.status_code, 400)
        self.assertIn("frames[1]", response.json()["detail"])

    def test_depth_shape_mismatch_is_400(self):
        service.set_runner(FakeRunner())
        response = self.client.post("/stance", json=make_request(depth_shape=(36, 60)))
        self.assertEqual(response.status_code, 400)
        self.assertIn("depth shape", response.json()["detail"])

    def test_zero_frames_is_400(self):
        service.set_runner(FakeRunner())
        response = self.client.post("/stance", json=make_request(()))
        self.assertEqual(response.status_code, 400)
        self.assertIn("at least one frame", response.json()["detail"])

    def test_too_many_frames_is_400(self):
        service.set_runner(FakeRunner())
        with mock.patch.object(service, "_MAX_FRAMES", 2):
            response = self.client.post("/stance", json=make_request(("a", "b", "c")))
            self.assertEqual(response.status_code, 400)
            self.assertIn("limit is 2", response.json()["detail"])
            response = self.client.post("/stance", json=make_request(("a", "b")))
            self.assertEqual(response.status_code, 200, response.text)

    def test_duplicate_or_empty_labels_are_400(self):
        service.set_runner(FakeRunner())
        response = self.client.post("/stance", json=make_request(("ready", "ready")))
        self.assertEqual(response.status_code, 400)
        self.assertIn("unique", response.json()["detail"])
        response = self.client.post("/stance", json=make_request(("ready", "")))
        self.assertEqual(response.status_code, 400)
        self.assertIn("non-empty", response.json()["detail"])

    def test_axis_and_height_validation(self):
        service.set_runner(FakeRunner())
        payload = make_request()
        payload["robot"]["down"] = [0.0, 2.0, 0.0]
        response = self.client.post("/stance", json=payload)
        self.assertEqual(response.status_code, 400)
        self.assertIn("unit vector", response.json()["detail"])
        payload = make_request()
        payload["robot"]["ground_camera_height_m"] = 3.5
        response = self.client.post("/stance", json=payload)
        self.assertEqual(response.status_code, 400)
        self.assertIn("ground_camera_height_m", response.json()["detail"])

    def test_missing_field_is_400(self):
        service.set_runner(FakeRunner())
        payload = make_request()
        del payload["robot"]["depth_png16_base64"]
        response = self.client.post("/stance", json=payload)
        self.assertEqual(response.status_code, 400)
        self.assertIn("depth_png16_base64", response.json()["detail"])

    def test_runner_error_is_500(self):
        service.set_runner(FakeRunner(error=RuntimeError("MPS out of memory")))
        with self.assertLogs("vggt_stance", level="ERROR"):
            response = self.client.post("/stance", json=make_request())
        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json()["detail"], "inference failed: RuntimeError: MPS out of memory")


class ParserTest(unittest.TestCase):
    def test_defaults(self):
        args = build_parser().parse_args([])
        self.assertEqual(args.device, "cuda")
        self.assertEqual(args.dtype, "float16")
        self.assertEqual(args.host, "127.0.0.1")
        self.assertEqual(args.port, 8117)
        self.assertEqual(args.weights, "/home/yor/models/vggt/vggt_1b_fp16.safetensors")
        self.assertEqual(args.max_frames, 8)
        self.assertFalse(args.depth_head_full)
        self.assertEqual(args.log_level, "INFO")

    def test_choices(self):
        args = build_parser().parse_args(["--device", "mps", "--dtype", "float32", "--depth-head-full", "--port", "9000"])
        self.assertEqual((args.device, args.dtype, args.depth_head_full, args.port), ("mps", "float32", True, 9000))
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            build_parser().parse_args(["--device", "tpu"])


if __name__ == "__main__":
    unittest.main()
