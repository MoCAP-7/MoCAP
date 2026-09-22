from __future__ import annotations

import base64
import contextlib
import hashlib
import io
import json
import shutil
import statistics
import struct
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from nav_planner import readiness
from nav_planner.prompts import MANIPULATION_EVENTS_PROMPT
from nav_planner.readiness import (
    EVENTS_SCHEMA,
    ManipulationEventsBuilder,
    ReadinessConfig,
    jpeg_dimensions,
    load_intrinsics_sidecar,
    low_motion_windows,
    normalize_event,
    parse_events_response,
    parse_manual_event,
    parse_signalstats,
    per_second_means,
    probe_video,
)
from nav_planner.readiness_cli import build_parser, main
from nav_planner.reference_frames import extract_video_sample_frames

FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")
HAS_FFMPEG = bool(FFMPEG and FFPROBE)

SIGNALSTATS_TEXT = (
    "frame:0    pts:1024    pts_time:0.1\n"
    "lavfi.signalstats.YAVG=1.37552\n"
    "frame:1    pts:2048    pts_time:0.2\n"
    "lavfi.signalstats.YAVG=0.905365\n"
    "frame:2    pts:3072    pts_time:0.3\n"
    "lavfi.signalstats.YAVG=1.56792\n"
)


def _fake_jpeg(width: int, height: int, *, progressive: bool = False) -> bytes:
    """A minimal JPEG byte string with APP0 and one SOF segment (no scan)."""

    app0 = b"\xff\xe0" + struct.pack(">H", 4) + b"\x00\x00"
    marker = 0xC2 if progressive else 0xC0
    sof = struct.pack(">BBHBHHB", 0xFF, marker, 11, 8, height, width, 1) + bytes(
        [1, 0x11, 0]
    )
    return b"\xff\xd8" + app0 + sof + b"\xff\xd9"


def _write_clip(directory: Path) -> Path:
    """A 4 s 320x240 10 fps test-pattern clip."""

    clip = directory / "clip.mp4"
    subprocess.run(
        [
            FFMPEG,
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=320x240:rate=10",
            "-t",
            "4",
            "-pix_fmt",
            "yuv420p",
            "-y",
            str(clip),
        ],
        check=True,
    )
    return clip


class _NeverClient:
    """Any attribute access proves a VLM client was touched."""

    def __getattr__(self, name: str):
        raise AssertionError(f"manual labelling must not touch the VLM client ({name})")


class ParseSignalstatsTest(unittest.TestCase):
    def test_pairs_pts_time_with_the_following_yavg_line(self) -> None:
        self.assertEqual(
            parse_signalstats(SIGNALSTATS_TEXT),
            [(0.1, 1.37552), (0.2, 0.905365), (0.3, 1.56792)],
        )

    def test_ignores_unrelated_lines_and_orphans(self) -> None:
        text = (
            "[warning] something\n"
            "lavfi.signalstats.YAVG=9.9\n"  # no preceding pts_time
            "frame:0    pts:1024    pts_time:0.1\n"
            "noise\n"
            "lavfi.signalstats.YAVG=1.0\n"
            "frame:1    pts:2048    pts_time:N/A\n"
            "lavfi.signalstats.YAVG=2.0\n"
        )
        self.assertEqual(parse_signalstats(text), [(0.1, 1.0)])
        self.assertEqual(parse_signalstats(""), [])


class PerSecondMeansTest(unittest.TestCase):
    def test_means_over_floor_of_pts_time(self) -> None:
        samples = [(0.1 * index, float(index)) for index in range(1, 25)]
        means = per_second_means(samples)
        self.assertEqual(len(means), 3)
        self.assertAlmostEqual(means[0], sum(range(1, 10)) / 9)
        self.assertAlmostEqual(means[1], sum(range(10, 20)) / 10)
        self.assertAlmostEqual(means[2], sum(range(20, 25)) / 5)

    def test_missing_second_repeats_the_previous_mean(self) -> None:
        samples = [(0.5, 2.0), (2.5, 6.0)]
        self.assertEqual(per_second_means(samples), [2.0, 2.0, 6.0])
        self.assertEqual(per_second_means([]), [])


class LowMotionWindowsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.kwargs = dict(threshold=1.0, min_window_s=1.5, margin_s=1.0, duration_s=20.0)

    def _signal(self, low_seconds: set[int], length: int = 20) -> list[float]:
        return [0.2 if second in low_seconds else 2.0 for second in range(length)]

    def test_one_second_dip_is_dropped(self) -> None:
        self.assertEqual(low_motion_windows(self._signal({5}), **self.kwargs), [])

    def test_three_second_dip_becomes_one_window_with_margin(self) -> None:
        windows = low_motion_windows(self._signal({10, 11, 12}), **self.kwargs)
        self.assertEqual(windows, [{"start_s": 9.0, "end_s": 14.0}])

    def test_adjacent_windows_merge(self) -> None:
        windows = low_motion_windows(self._signal({10, 11, 12, 14, 15}), **self.kwargs)
        self.assertEqual(windows, [{"start_s": 9.0, "end_s": 17.0}])

    def test_windows_clamp_at_zero_and_duration(self) -> None:
        windows = low_motion_windows(
            self._signal({0, 1, 18, 19}),
            threshold=1.0,
            min_window_s=1.5,
            margin_s=1.0,
            duration_s=19.7,
        )
        self.assertEqual(
            windows,
            [{"start_s": 0.0, "end_s": 3.0}, {"start_s": 17.0, "end_s": 19.7}],
        )

    def test_threshold_is_strict(self) -> None:
        signal = [1.0, 1.0, 2.0]
        self.assertEqual(low_motion_windows(signal, **self.kwargs), [])
        self.assertEqual(
            low_motion_windows(signal, **{**self.kwargs, "threshold": 1.01}),
            [{"start_s": 0.0, "end_s": 3.0}],
        )

    def test_real_video_series_keeps_the_grasp_window_at_the_default(self) -> None:
        # Per-second YAVG of test_by_Tatsuya_20260910_233039_undistorted.mp4
        # (2560x1920, 10 fps, 43.83 s).  The right hand grasps the can at
        # 18.5-19.5 s from a stance reached at 16.5 s; the wearer keeps moving
        # head and hands at the desk, so those seconds sit just below the
        # median (18.22) and a 0.75 x median threshold (13.67) pruned them.
        # Phone handling at 0-1 s and 36-39 s is low motion too.
        per_second = [
            1.5332, 4.9510, 28.0183, 27.3560, 23.3860, 17.6912, 23.5153, 27.6992,
            19.1660, 25.4795, 16.7266, 22.6194, 23.6034, 20.8224, 13.7089, 15.0062,
            18.7495, 14.8768, 14.1041, 11.8109, 15.9577, 25.6962, 19.4504, 20.8595,
            16.9990, 30.8531, 21.9032, 23.8022, 33.1134, 26.7084, 30.5135, 22.1231,
            17.5187, 17.5221, 11.0432, 16.9287, 9.1449, 5.2928, 6.9566, 8.6579,
            17.2667, 19.9656, 8.2199, 2.0184,
        ]
        config = ReadinessConfig()
        threshold = config.motion_threshold_factor * statistics.median(per_second)
        self.assertAlmostEqual(threshold, 18.2204, places=3)

        windows = low_motion_windows(
            per_second,
            threshold=threshold,
            min_window_s=config.min_window_s,
            margin_s=config.window_margin_s,
            duration_s=43.833991,
        )

        self.assertTrue(
            any(w["start_s"] <= 16.5 and w["end_s"] >= 19.0 for w in windows),
            windows,
        )
        self.assertEqual(
            windows,
            [
                {"start_s": 0.0, "end_s": 3.0},
                {"start_s": 13.0, "end_s": 22.0},
                {"start_s": 31.0, "end_s": 43.833991},
            ],
        )
        # The old 0.75 x median default is exactly what dropped the event.
        old = low_motion_windows(
            per_second,
            threshold=0.75 * statistics.median(per_second),
            min_window_s=config.min_window_s,
            margin_s=config.window_margin_s,
            duration_s=43.833991,
        )
        self.assertFalse(any(w["start_s"] <= 16.5 and w["end_s"] >= 19.0 for w in old))


class ParseManualEventTest(unittest.TestCase):
    def test_valid_event(self) -> None:
        self.assertEqual(
            parse_manual_event("can:right:16.5:19.0"),
            {
                "object": "can",
                "hand": "right",
                "t_ready_s": 16.5,
                "t_grasp_s": 19.0,
                "confidence": None,
                "notes": "",
            },
        )
        self.assertEqual(parse_manual_event(" red can : Both : 1 : 2 ")["hand"], "both")
        self.assertEqual(parse_manual_event(" red can : Both : 1 : 2 ")["object"], "red can")

    def test_malformed_event_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "OBJECT:HAND:T_READY:T_GRASP"):
            parse_manual_event("can:right:16.5")
        with self.assertRaisesRegex(ValueError, "numbers"):
            parse_manual_event("can:right:soon:later")
        with self.assertRaisesRegex(ValueError, "object"):
            parse_manual_event(":right:1:2")

    def test_hand_is_validated(self) -> None:
        with self.assertRaisesRegex(ValueError, "hand"):
            parse_manual_event("can:up:1:2")

    def test_grasp_before_ready_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "precede"):
            parse_manual_event("can:right:19.0:16.5")
        with self.assertRaisesRegex(ValueError, ">= 0"):
            parse_manual_event("can:right:-1:2")


class ParseEventsResponseTest(unittest.TestCase):
    EVENT = {
        "object": "can",
        "hand": "right",
        "t_ready_s": 16.5,
        "t_grasp_s": 19.0,
        "confidence": 0.8,
        "notes": "reaches for the can on the desk",
    }

    def test_fenced_json(self) -> None:
        text = "```json\n" + json.dumps({"events": [self.EVENT]}) + "\n```"
        self.assertEqual(parse_events_response(text, duration_s=43.8), [self.EVENT])

    def test_prose_around_json(self) -> None:
        text = (
            "Here is what I found:\n"
            + json.dumps({"events": [self.EVENT]})
            + "\nLet me know if you need more."
        )
        self.assertEqual(parse_events_response(text, duration_s=43.8), [self.EVENT])

    def test_empty_events(self) -> None:
        self.assertEqual(parse_events_response('{"events": []}', duration_s=43.8), [])

    def test_numeric_strings_and_missing_optionals_are_normalized(self) -> None:
        text = json.dumps(
            {"events": [{"object": "cup", "hand": "LEFT", "t_ready_s": "3", "t_grasp_s": 4}]}
        )
        self.assertEqual(
            parse_events_response(text, duration_s=10.0),
            [
                {
                    "object": "cup",
                    "hand": "left",
                    "t_ready_s": 3.0,
                    "t_grasp_s": 4.0,
                    "confidence": None,
                    "notes": "",
                }
            ],
        )

    def test_bad_hand_is_rejected(self) -> None:
        text = json.dumps({"events": [{**self.EVENT, "hand": "tentacle"}]})
        with self.assertRaisesRegex(ValueError, "hand"):
            parse_events_response(text, duration_s=43.8)

    def test_times_outside_the_video_are_rejected(self) -> None:
        text = json.dumps({"events": [self.EVENT]})
        with self.assertRaisesRegex(ValueError, "duration"):
            parse_events_response(text, duration_s=18.0)
        text = json.dumps({"events": [{**self.EVENT, "t_grasp_s": 16.0}]})
        with self.assertRaisesRegex(ValueError, "precede"):
            parse_events_response(text, duration_s=43.8)

    def test_non_json_and_missing_events_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "no JSON object"):
            parse_events_response("no events here", duration_s=10.0)
        with self.assertRaisesRegex(ValueError, '"events" list'):
            parse_events_response('{"result": []}', duration_s=10.0)
        with self.assertRaisesRegex(ValueError, "confidence"):
            normalize_event({**self.EVENT, "confidence": "high"})


class JpegDimensionsTest(unittest.TestCase):
    def test_baseline_and_progressive_headers(self) -> None:
        self.assertEqual(jpeg_dimensions(_fake_jpeg(1280, 960)), (1280, 960))
        self.assertEqual(
            jpeg_dimensions(_fake_jpeg(640, 480, progressive=True)), (640, 480)
        )

    def test_non_jpeg_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            jpeg_dimensions(b"\x89PNG\r\n\x1a\n")
        with self.assertRaises(ValueError):
            jpeg_dimensions(b"\xff\xd8\xff\xd9")


class IntrinsicsSidecarTest(unittest.TestCase):
    def test_default_sidecar_next_to_the_video(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            video = Path(directory) / "x_undistorted.mp4"
            video.write_bytes(b"")
            self.assertIsNone(load_intrinsics_sidecar(video))
            sidecar = Path(directory) / "x_undistorted.intrinsics.json"
            sidecar.write_text(
                json.dumps(
                    {
                        "schema_version": "yor-video-intrinsics-v1",
                        "model": "pinhole",
                        "width": 2560,
                        "height": 1920,
                        "fx": 1107.8439067,
                        "fy": 1107.8439067,
                        "cx": 1280.0,
                        "cy": 960.0,
                        "device": "Aria Gen2",
                    }
                )
            )
            self.assertEqual(
                load_intrinsics_sidecar(video),
                {
                    "model": "pinhole",
                    "width": 2560,
                    "height": 1920,
                    "fx": 1107.8439067,
                    "fy": 1107.8439067,
                    "cx": 1280.0,
                    "cy": 960.0,
                    "source_path": str(sidecar.resolve()),
                },
            )
            with self.assertRaises(FileNotFoundError):
                load_intrinsics_sidecar(video, Path(directory) / "missing.json")
            sidecar.write_text(json.dumps({"model": "fisheye", "width": 1, "height": 1}))
            with self.assertRaisesRegex(ValueError, "pinhole"):
                load_intrinsics_sidecar(video)


class _Interactions:
    def __init__(self, output_text: str) -> None:
        self.requests: list[dict] = []
        self.output_text = output_text

    def create(self, **kwargs):
        self.requests.append(kwargs)
        return SimpleNamespace(output_text=self.output_text)


class _Responses(_Interactions):
    pass


class VlmLabelingTest(unittest.TestCase):
    """The VLM path with fake clients; ffmpeg steps are patched out."""

    SOURCE = {"width": 320, "height": 240, "fps": 10.0, "duration_s": 4.0}
    RESPONSE = (
        "```json\n"
        + json.dumps(
            {
                "events": [
                    {
                        "object": "can",
                        "hand": "right",
                        "t_ready_s": 2.0,
                        "t_grasp_s": 3.0,
                        "confidence": 0.9,
                        "notes": "reaches for the can",
                    }
                ]
            }
        )
        + "\n```"
    )

    def setUp(self) -> None:
        # Seconds 0-1 moving (10.0), seconds 2-3 still (1.0): median 5.5 is
        # the default threshold, one run [2, 4) -> window [1, 4] after the margin.
        samples = [
            (round(0.1 * index, 3), 10.0 if index < 20 else 1.0)
            for index in range(1, 40)
        ]
        self.sampled_calls: list[dict] = []
        self.fake_frame_bytes = _fake_jpeg(64, 48)

        def fake_sample(video, output_dir, **kwargs):
            self.sampled_calls.append(kwargs)
            output_dir.mkdir(parents=True)
            frames = []
            for index in range(2):
                path = output_dir / f"frame_{index:06d}.jpg"
                path.write_bytes(self.fake_frame_bytes + bytes([index]))
                seconds = kwargs["start_s"] + index / kwargs["fps"]
                frames.append(
                    {
                        "timestamp": f"0:{seconds:06.3f}",
                        "time_s": seconds,
                        "path": path,
                        "mime_type": "image/jpeg",
                    }
                )
            return frames

        def fake_extract_frame(video, time_s, destination, *, max_width):
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(_fake_jpeg(160, 120))
            return 160, 120

        self.patches = [
            mock.patch.object(readiness, "probe_video", return_value=dict(self.SOURCE)),
            mock.patch.object(readiness, "ego_motion_per_frame", return_value=samples),
            mock.patch.object(readiness, "extract_video_sample_frames", side_effect=fake_sample),
            mock.patch.object(readiness, "extract_frame", side_effect=fake_extract_frame),
            mock.patch.object(
                readiness, "make_client", side_effect=AssertionError("no real client")
            ),
        ]
        for patch in self.patches:
            patch.start()
            self.addCleanup(patch.stop)
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.video = self.root / "tour.mp4"
        self.video.write_bytes(b"fake video")

    def test_gemini_request_and_artifact(self) -> None:
        interactions = _Interactions(self.RESPONSE)
        client = SimpleNamespace(interactions=interactions)
        progress: list[dict] = []
        builder = ManipulationEventsBuilder(
            ReadinessConfig(), client=client, on_progress=progress.append
        )

        result = builder.build(self.video, self.root / "outputs")

        self.assertEqual(result["motion"]["windows"], [{"start_s": 1.0, "end_s": 4.0}])
        self.assertEqual(result["motion"]["threshold"], 5.5)
        self.assertEqual(self.sampled_calls[0]["start_s"], 1.0)
        self.assertEqual(self.sampled_calls[0]["duration_s"], 3.0)
        self.assertEqual(self.sampled_calls[0]["fps"], 2.0)
        self.assertEqual(self.sampled_calls[0]["max_width"], 768)
        self.assertEqual(len(interactions.requests), 1)
        request = interactions.requests[0]
        self.assertEqual(request["model"], "gemini-3.7-flash")
        self.assertEqual(request["generation_config"], {"thinking_level": "high"})
        self.assertEqual(request["timeout"], 900.0)
        inputs = request["input"]
        self.assertEqual(inputs[0], {"type": "text", "text": MANIPULATION_EVENTS_PROMPT})
        self.assertIn("0:01.000", inputs[1]["text"])
        self.assertEqual(inputs[2], {"type": "text", "text": "VIDEO FRAME at 0:01.000:"})
        self.assertEqual(inputs[3]["type"], "image")
        self.assertEqual(inputs[3]["mime_type"], "image/jpeg")
        self.assertEqual(
            inputs[3]["data"],
            base64.b64encode(self.fake_frame_bytes + b"\x00").decode("ascii"),
        )
        self.assertEqual(inputs[4], {"type": "text", "text": "VIDEO FRAME at 0:01.500:"})
        self.assertEqual(len(inputs), 6)

        self.assertEqual(result["labeling"]["mode"], "vlm")
        self.assertEqual(result["labeling"]["model"], "gemini-3.7-flash")
        self.assertEqual(
            result["labeling"]["prompt_sha256"],
            hashlib.sha256(MANIPULATION_EVENTS_PROMPT.encode("utf-8")).hexdigest(),
        )
        self.assertIn(self.RESPONSE, result["labeling"]["raw_response"])
        self.assertEqual(len(result["events"]), 1)
        event = result["events"][0]
        self.assertEqual(event["object"], "can")
        self.assertEqual(event["hand"], "right")
        self.assertEqual(event["confidence"], 0.9)
        frames = event["frames"]
        self.assertEqual(frames["nav"]["time_s"], 0.0)
        self.assertEqual(frames["ready"]["time_s"], 2.0)
        self.assertEqual(frames["grasp"]["time_s"], 3.0)
        artifact = Path(result["artifact_path"])
        self.assertTrue(artifact.is_file())
        self.assertEqual(
            frames["ready"]["path"], f"{artifact.stem}_frames/event00_ready.jpg"
        )
        self.assertTrue((artifact.parent / frames["ready"]["path"]).is_file())
        self.assertEqual(frames["ready"]["scale"], 0.5)
        self.assertEqual(
            [entry["event"] for entry in progress][:4],
            ["video_probed", "intrinsics_loaded", "ego_motion_started", "motion_windows_found"],
        )
        self.assertIn("window_labeling_started", [entry["event"] for entry in progress])

    def test_openai_request(self) -> None:
        responses = _Responses(json.dumps({"events": []}))
        client = SimpleNamespace(responses=responses)
        builder = ManipulationEventsBuilder(
            ReadinessConfig(model="gpt-5.6-sol", thinking_level="xhigh"), client=client
        )

        result = builder.build(self.video, self.root / "outputs")

        self.assertEqual(result["events"], [])
        request = responses.requests[0]
        self.assertEqual(request["model"], "gpt-5.6-sol")
        self.assertEqual(request["reasoning"], {"effort": "xhigh"})
        self.assertEqual(request["input"][0]["role"], "user")
        content = request["input"][0]["content"]
        self.assertEqual(content[0], {"type": "input_text", "text": MANIPULATION_EVENTS_PROMPT})
        self.assertEqual(content[2], {"type": "input_text", "text": "VIDEO FRAME at 0:01.000:"})
        self.assertEqual(content[3]["type"], "input_image")
        self.assertEqual(content[3]["detail"], "high")
        self.assertTrue(content[3]["image_url"].startswith("data:image/jpeg;base64,"))
        self.assertEqual(result["labeling"]["mode"], "vlm")
        self.assertEqual(result["labeling"]["model"], "gpt-5.6-sol")

    def test_gemini_rejects_openai_only_thinking_levels(self) -> None:
        builder = ManipulationEventsBuilder(
            ReadinessConfig(thinking_level="xhigh"), client=SimpleNamespace()
        )
        with self.assertRaisesRegex(ValueError, "thinking_level for gemini"):
            builder.build(self.video, self.root / "outputs")

    def test_no_windows_means_no_request(self) -> None:
        interactions = _Interactions(self.RESPONSE)
        builder = ManipulationEventsBuilder(
            ReadinessConfig(motion_threshold=0.0),
            client=SimpleNamespace(interactions=interactions),
        )
        result = builder.build(self.video, self.root / "outputs")
        self.assertEqual(interactions.requests, [])
        self.assertEqual(result["events"], [])
        self.assertIsNone(result["labeling"]["raw_response"])
        self.assertEqual(result["labeling"]["mode"], "vlm")
        self.assertEqual(result["motion"]["threshold"], 0.0)

    def test_threshold_factor_scales_the_median(self) -> None:
        interactions = _Interactions(json.dumps({"events": []}))
        builder = ManipulationEventsBuilder(
            ReadinessConfig(motion_threshold_factor=0.1),
            client=SimpleNamespace(interactions=interactions),
        )
        result = builder.build(self.video, self.root / "outputs")
        self.assertEqual(result["motion"]["threshold"], 0.55)
        self.assertEqual(result["motion"]["windows"], [])
        self.assertEqual(interactions.requests, [])

    def test_manual_mode_never_touches_the_client(self) -> None:
        builder = ManipulationEventsBuilder(ReadinessConfig(), client=_NeverClient())
        result = builder.build(
            self.video,
            self.root / "outputs",
            label="manual",
            manual_events=[parse_manual_event("can:left:2.5:3.5")],
        )
        self.assertEqual(
            result["labeling"],
            {"mode": "manual", "model": None, "prompt_sha256": None, "raw_response": None},
        )
        self.assertEqual(result["motion"]["windows"], [{"start_s": 1.0, "end_s": 4.0}])
        self.assertEqual(result["events"][0]["frames"]["nav"]["time_s"], 0.5)
        with self.assertRaisesRegex(ValueError, "at least one event"):
            builder.build(self.video, self.root / "outputs", label="manual")


@unittest.skipUnless(HAS_FFMPEG, "ffmpeg and ffprobe are required")
class ManualBuildTest(unittest.TestCase):
    """Manual labelling end to end on a real 4 s clip."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.directory = tempfile.TemporaryDirectory()
        cls.root = Path(cls.directory.name)
        cls.video = _write_clip(cls.root)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.directory.cleanup()

    def test_probe_video(self) -> None:
        probed = probe_video(self.video)
        self.assertEqual((probed["width"], probed["height"], probed["fps"]), (320, 240, 10.0))
        self.assertAlmostEqual(probed["duration_s"], 4.0, places=2)

    def test_window_sampling_offsets_time_by_start(self) -> None:
        frames = extract_video_sample_frames(
            self.video,
            self.root / "window_frames",
            fps=2.0,
            limit=10,
            max_width=160,
            start_s=2.0,
            duration_s=1.0,
        )
        self.assertEqual([frame["time_s"] for frame in frames][:2], [2.0, 2.5])
        self.assertEqual(frames[0]["timestamp"], "0:02.000")
        self.assertTrue(all(2.0 <= frame["time_s"] <= 3.0 for frame in frames))
        self.assertEqual(jpeg_dimensions(Path(frames[0]["path"]).read_bytes()), (160, 120))

    def test_manual_build_writes_the_locked_artifact(self) -> None:
        builder = ManipulationEventsBuilder(ReadinessConfig(), client=_NeverClient())
        output_dir = self.root / "outputs"

        result = builder.build(
            self.video,
            output_dir,
            label="manual",
            manual_events=[parse_manual_event("ball:left:1.0:2.5")],
        )

        artifact = Path(result["artifact_path"])
        self.assertRegex(artifact.name, r"^manipulation_events_\d{8}T\d{6}\.\d{6}Z\.json$")
        self.assertEqual(artifact.parent, output_dir.resolve())
        saved = json.loads(artifact.read_text())
        self.assertEqual(saved, result)
        self.assertEqual(
            list(saved),
            [
                "schema_version",
                "created_at",
                "artifact_path",
                "source_video",
                "intrinsics",
                "motion",
                "labeling",
                "events",
            ],
        )
        self.assertEqual(saved["schema_version"], EVENTS_SCHEMA)
        self.assertEqual(
            list(saved["source_video"]),
            ["path", "size_bytes", "mtime_ns", "width", "height", "fps", "duration_s"],
        )
        self.assertEqual(saved["source_video"]["width"], 320)
        self.assertIsNone(saved["intrinsics"])
        self.assertEqual(
            list(saved["motion"]), ["threshold", "median_yavg", "per_second_yavg", "windows"]
        )
        self.assertEqual(len(saved["motion"]["per_second_yavg"]), 4)
        self.assertEqual(
            saved["labeling"],
            {"mode": "manual", "model": None, "prompt_sha256": None, "raw_response": None},
        )
        event = saved["events"][0]
        self.assertEqual(
            list(event),
            ["object", "hand", "t_ready_s", "t_grasp_s", "confidence", "notes", "frames"],
        )
        self.assertEqual(list(event["frames"]), ["nav", "ready", "grasp"])
        expected_times = {"nav": 0.0, "ready": 1.0, "grasp": 2.5}
        for role, frame in event["frames"].items():
            self.assertEqual(list(frame), ["path", "time_s", "width", "height", "scale"])
            self.assertEqual(frame["time_s"], expected_times[role])
            self.assertEqual(frame["path"], f"{artifact.stem}_frames/event00_{role}.jpg")
            path = artifact.parent / frame["path"]
            self.assertTrue(path.is_file())
            self.assertEqual(jpeg_dimensions(path.read_bytes()), (frame["width"], frame["height"]))
            self.assertEqual(frame["scale"], frame["width"] / 320)
            self.assertEqual(frame["height"], round(frame["width"] * 240 / 320))

    def test_sidecar_intrinsics_are_copied(self) -> None:
        sidecar = self.video.with_suffix(".intrinsics.json")
        sidecar.write_text(
            json.dumps(
                {"model": "pinhole", "width": 320, "height": 240, "fx": 300.0, "fy": 300.0, "cx": 160.0, "cy": 120.0}
            )
        )
        self.addCleanup(sidecar.unlink)
        builder = ManipulationEventsBuilder(ReadinessConfig(frame_width=160))
        result = builder.build(
            self.video,
            self.root / "outputs_intrinsics",
            label="manual",
            manual_events=[parse_manual_event("ball:right:3.0:3.9")],
        )
        self.assertEqual(result["intrinsics"]["fx"], 300.0)
        self.assertEqual(result["intrinsics"]["source_path"], str(sidecar.resolve()))
        self.assertEqual(result["events"][0]["frames"]["ready"]["scale"], 0.5)
        self.assertEqual(result["events"][0]["frames"]["ready"]["width"], 160)
        explicit = self.root / "other.json"
        explicit.write_text(
            json.dumps(
                {"model": "pinhole", "width": 2560, "height": 1920, "fx": 1.0, "fy": 1.0, "cx": 0.0, "cy": 0.0}
            )
        )
        with self.assertRaisesRegex(ValueError, "2560x1920"):
            builder.build(
                self.video,
                self.root / "outputs_intrinsics",
                label="manual",
                manual_events=[parse_manual_event("ball:right:3.0:3.9")],
                intrinsics_path=explicit,
            )

    def test_cli_manual_end_to_end(self) -> None:
        output_dir = self.root / "cli_outputs"
        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = main(
                [
                    str(self.video),
                    "--output-dir",
                    str(output_dir),
                    "--label",
                    "manual",
                    "--event",
                    "ball:left:1.0:2.5",
                    "--event",
                    "cup:both:3.0:3.5",
                    "--frame-width",
                    "160",
                ]
            )
        self.assertEqual(code, 0)
        summary = json.loads(stdout.getvalue().strip().splitlines()[-1])
        self.assertEqual(summary["events"], 2)
        self.assertIsInstance(summary["windows"], int)
        self.assertEqual(len(summary["motion_windows"]), summary["windows"])
        self.assertIsInstance(summary["motion_threshold"], float)
        for window in summary["motion_windows"]:
            self.assertEqual(list(window), ["start_s", "end_s"])
        self.assertTrue(Path(summary["artifact"]).is_file())
        self.assertEqual(Path(summary["artifact"]).parent, output_dir.resolve())
        events = [json.loads(line)["event"] for line in stderr.getvalue().splitlines()]
        self.assertEqual(events[-1], "events_saved")


class CliParserTest(unittest.TestCase):
    def test_defaults(self) -> None:
        args = build_parser().parse_args(["tour.mp4"])
        self.assertEqual(args.label, "vlm")
        self.assertEqual(args.model, "gemini-3.7-flash")
        self.assertEqual(args.frame_width, 1280)
        self.assertEqual(args.sample_fps, 2.0)
        self.assertEqual(args.sample_width, 768)
        self.assertIsNone(args.motion_threshold)
        self.assertEqual(args.motion_threshold_factor, 1.0)
        self.assertEqual(args.event, [])
        args = build_parser().parse_args(
            ["tour.mp4", "--motion-threshold-factor", "0.85", "--motion-threshold", "12"]
        )
        self.assertEqual(args.motion_threshold_factor, 0.85)
        self.assertEqual(args.motion_threshold, 12.0)

    def test_bad_threshold_factor_is_a_parser_error(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as raised:
                main(["tour.mp4", "--motion-threshold-factor", "0"])
        self.assertEqual(raised.exception.code, 2)

    def test_manual_requires_an_event(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as raised:
                main(["tour.mp4", "--label", "manual"])
        self.assertEqual(raised.exception.code, 2)

    def test_event_is_rejected_in_vlm_mode(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as raised:
                main(["tour.mp4", "--event", "can:right:1:2"])
        self.assertEqual(raised.exception.code, 2)

    def test_malformed_event_is_a_parser_error(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as raised:
                main(["tour.mp4", "--label", "manual", "--event", "can:right:2:1"])
        self.assertEqual(raised.exception.code, 2)


class ReadinessConfigTest(unittest.TestCase):
    def test_rejects_bad_values(self) -> None:
        for kwargs in (
            {"motion_threshold": -1.0},
            {"motion_threshold_factor": 0.0},
            {"motion_threshold_factor": -0.5},
            {"sample_fps": 0.0},
            {"frame_width": 0},
            {"min_window_s": -1.0},
            {"image_detail": "ultra"},
        ):
            with self.assertRaises(ValueError, msg=str(kwargs)):
                ReadinessConfig(**kwargs)


if __name__ == "__main__":
    unittest.main()
