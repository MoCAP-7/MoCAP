"""Passive-video readiness prior: where the human stood when they grasped.

The offline pipeline (``nav_planner.readiness``) turns one first-person video
into ``manipulation_events_<version>.json``: per grasp, the frame where the
human became manipulation-ready plus the hand they used. At dock time this
module recovers the human camera pose in the ZED frame from those frames.
The camera centre projected to the floor is where the person's head was; the
feet sit ``stance_offset_m`` behind it along the heading, the bearing from the
object toward that point is the side the robot should dock from, and the hand
becomes the suggested arm.

``method: vggt`` (the default) sends the live ZED view (RGB, depth, floor
axes) and the event frames of ``vggt_frames`` to the VGGT-1B stance service
(``services/vggt_stance``) over plain ``urllib`` and reads the human camera
rows it returns: metric forward/left offsets on the floor plane, height above
the floor and heading. Two gates reject a solution whose depth scale is
unstable (``scale_iqr_ratio``) or whose head height is implausible.

``method: pnp`` is the Phase 0 ablation: SIFT + Lowe ratio between the ready
frame and the ZED image, ZED matches back-projected through the depth image,
PnP-RANSAC for the human camera pose. cv2 is imported only inside
:func:`_cv2`, so the module imports on machines without OpenCV; there the pnp
method returns ``None`` with ``last_reason == "cv2_unavailable"``.
"""

from __future__ import annotations

import base64
from collections.abc import Callable, Mapping
from dataclasses import dataclass
import io
import json
import logging
import math
import os
from pathlib import Path
import re
import time
from typing import Any
import urllib.error
import urllib.request

import numpy as np

EVENTS_SCHEMA = "yor-manipulation-events-v1"

_ARTICLES = {"a", "an", "the"}
_FRAME_KEYS = {"ready_frame": "ready", "nav_frame": "nav"}
_EVENT_FRAME_LABELS = ("nav", "ready", "grasp")
_METHODS = ("vggt", "pnp")
_MIN_PNP_CORRESPONDENCES = 6
# A human camera solved farther than this from the ZED is a PnP failure mode
# (a mirror-image or scale-collapsed solution), not a person in the room.
_MAX_HUMAN_DISTANCE_M = 10.0
# Largest depth the 16-bit millimetre PNG can carry.
_MAX_DEPTH_PNG_M = 65.535

Matcher = Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]
# ``transport(url, payload, timeout_s) -> parsed JSON``; ``payload`` None is a
# GET (``/health``), a mapping is POSTed as JSON (``/stance``).
Transport = Callable[[str, "Mapping[str, Any] | None", float], Any]


def _cv2():
    """Import OpenCV on first use; raises ImportError when it is absent."""

    import cv2

    return cv2


def _pil_image():
    """Import ``PIL.Image`` on first use (the encoders need Pillow >= 9.0)."""

    from PIL import Image

    return Image


def _urllib_transport(
    url: str, payload: Mapping[str, Any] | None, timeout_s: float
) -> Any:
    """Default HTTP transport: stdlib ``urllib``, JSON in, JSON out.

    ``payload`` ``None`` sends a GET; otherwise the mapping is POSTed as a
    JSON body. ``urllib.error.URLError`` / ``HTTPError`` / ``socket.timeout``
    propagate to the caller, as does ``ValueError`` for a non-JSON body.
    """

    if payload is None:
        request = urllib.request.Request(
            url, headers={"Accept": "application/json"}, method="GET"
        )
    else:
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        body = response.read()
    return json.loads(body.decode("utf-8"))


def _encode_rgb_jpeg(rgb: np.ndarray, *, quality: int = 95) -> bytes:
    """Robot RGB (H, W, 3+) uint8 -> JPEG bytes (Pillow >= 9.0 APIs only)."""

    image_module = _pil_image()
    array = np.ascontiguousarray(np.asarray(rgb)[:, :, :3], dtype=np.uint8)
    buffer = io.BytesIO()
    image_module.fromarray(array, "RGB").save(buffer, format="JPEG", quality=quality)
    return buffer.getvalue()


def _encode_depth_png16(depth: np.ndarray) -> bytes:
    """Depth in metres (H, W) -> 16-bit grayscale PNG in millimetres, 0 = invalid."""

    image_module = _pil_image()
    metres = np.nan_to_num(
        np.asarray(depth, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0
    )
    metres = np.clip(metres, 0.0, _MAX_DEPTH_PNG_M)
    millimetres = np.rint(metres * 1000.0).astype(np.uint16)
    buffer = io.BytesIO()
    image_module.fromarray(millimetres).save(buffer, format="PNG")
    return buffer.getvalue()


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _finite_or_none(value: Any) -> float | None:
    """``float(value)`` when it is a finite number, else ``None`` (never NaN)."""

    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _frame_time_s(frame: Mapping[str, Any], event: Mapping[str, Any]) -> float | None:
    """The frame's ``time_s`` (else the event's ``t_ready_s``) as a finite float, or ``None``."""

    time_s = _finite_or_none(frame.get("time_s"))
    if time_s is None:
        time_s = _finite_or_none(event.get("t_ready_s"))
    return time_s


def _wrap_angle(angle: float) -> float:
    return float(math.atan2(math.sin(angle), math.cos(angle)))


def _planar_to_world(
    forward_m: float, left_m: float, pose_xy_yaw: np.ndarray
) -> tuple[float, float]:
    """Floor-plane (forward, left) offsets from the ZED to odom x, y."""

    cosine = math.cos(float(pose_xy_yaw[2]))
    sine = math.sin(float(pose_xy_yaw[2]))
    world_x = float(pose_xy_yaw[0]) + cosine * forward_m - sine * left_m
    world_y = float(pose_xy_yaw[1]) + sine * forward_m + cosine * left_m
    return world_x, world_y


def _rotation_from_rvec(rvec: np.ndarray) -> np.ndarray:
    """Rodrigues formula (axis-angle vector -> 3x3 rotation) without cv2."""

    vector = np.asarray(rvec, dtype=np.float64).reshape(3)
    angle = float(np.linalg.norm(vector))
    if angle < 1e-12:
        return np.eye(3)
    axis = vector / angle
    skew = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )
    return (
        np.eye(3)
        + math.sin(angle) * skew
        + (1.0 - math.cos(angle)) * (skew @ skew)
    )


def _tokens(text: str) -> set[str]:
    return {word for word in re.findall(r"\w+", str(text).lower())} - _ARTICLES


def _is_int(value: Any) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, bool)


def _is_finite_number(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        return False
    return math.isfinite(float(value))


@dataclass(frozen=True)
class ReadinessPriorConfig:
    """``dock_to_visible_object.settings.readiness_prior`` block."""

    enabled: bool = False
    events_path: str | None = None
    # The event is normally the first one sharing a word with the docking
    # object name. Set this to pick the event by these words instead, so a
    # fallback name for the same object (a "bottle" prompt for a can SAM3 no
    # longer finds at range) still docks with that object's event.
    event_object: str | None = None
    # ready_frame is the method; nav_frame is the t_ready - 2 s walking frame
    # kept for the ablation. With method vggt this is the frame whose camera
    # row gives the stance.
    source: str = "ready_frame"
    # Pose transfer backend: vggt (the VGGT-1B stance service) or pnp
    # (SIFT + PnP against the ZED depth, the Phase 0 ablation).
    method: str = "vggt"
    vggt_service_url: str = "http://127.0.0.1:8117"
    vggt_timeout_s: float = 60.0
    # Event frames sent with the ZED view, request order; must include the
    # ``source`` frame.
    vggt_frames: tuple[str, ...] = ("nav", "ready")
    # Feet = head - stance_offset_m along the heading; a head within
    # near_object_m of the object docks from heading + 180 deg.
    stance_offset_m: float = 0.30
    near_object_m: float = 0.30
    # Gates on the VGGT solution: head height above the floor and the
    # depth-scale IQR ratio.
    min_height_m: float = 1.2
    max_height_m: float = 2.1
    max_scale_iqr_ratio: float = 2.0
    # pnp only.
    min_inliers: int = 30
    max_reprojection_error_px: float = 8.0
    sift_features: int = 4000
    ratio_test: float = 0.8
    # Horizontal field of view candidates for a video without an intrinsics
    # sidecar; the focal with the most PnP inliers wins.
    focal_sweep_fov_deg: tuple[float, ...] = (60, 70, 80, 90, 100, 110, 120)
    # Consumed by the docking controller, not by the prior itself.
    max_bearing_error_deg: float = 45.0
    retry_bearing_offsets_deg: tuple[float, ...] = (45.0, -45.0)
    # After the prior bearing and its offsets all fail, dock along the
    # arrival bearing; off, the prior's bearings are the only attempts.
    fallback_to_arrival_bearing: bool = True
    robot_image_max_width: int = 1280

    @classmethod
    def from_mapping(
        cls, values: Mapping[str, Any] | None
    ) -> "ReadinessPriorConfig":
        source = dict(values or {})
        known = set(cls.__dataclass_fields__)
        unknown = sorted(set(source) - known)
        if unknown:
            raise ValueError(f"unknown readiness_prior settings: {unknown}")
        payload: dict[str, Any] = {}
        for key in known:
            if key not in source:
                continue
            value = source[key]
            if isinstance(value, list):
                value = tuple(value)
            if key == "events_path" and isinstance(value, os.PathLike):
                value = os.fspath(value)
            if key == "vggt_service_url" and isinstance(value, str):
                value = value.strip().rstrip("/")
            payload[key] = value
        config = cls(**payload)
        config._validate()
        return config

    def _validate(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("readiness_prior.enabled must be a boolean")
        if self.events_path is not None and (
            not isinstance(self.events_path, str) or not self.events_path.strip()
        ):
            raise ValueError("readiness_prior.events_path must be a non-empty path")
        if self.enabled and self.events_path is None:
            raise ValueError(
                "readiness_prior.events_path is required when readiness_prior is enabled"
            )
        if self.event_object is not None and (
            not isinstance(self.event_object, str) or not _tokens(self.event_object)
        ):
            raise ValueError(
                "readiness_prior.event_object must be a string with a word other "
                "than an article"
            )
        if self.source not in _FRAME_KEYS:
            raise ValueError(
                "readiness_prior.source must be one of "
                f"{sorted(_FRAME_KEYS)}, got {self.source!r}"
            )
        self._validate_vggt()
        if not _is_int(self.min_inliers) or self.min_inliers < 4:
            raise ValueError("readiness_prior.min_inliers must be an integer >= 4")
        if (
            not _is_finite_number(self.max_reprojection_error_px)
            or not 0.0 < float(self.max_reprojection_error_px) <= 50.0
        ):
            raise ValueError(
                "readiness_prior.max_reprojection_error_px must be in (0, 50]"
            )
        if not _is_int(self.sift_features) or self.sift_features < 100:
            raise ValueError("readiness_prior.sift_features must be an integer >= 100")
        if not _is_finite_number(self.ratio_test) or not 0.0 < float(
            self.ratio_test
        ) < 1.0:
            raise ValueError("readiness_prior.ratio_test must be in (0, 1)")
        if not isinstance(self.focal_sweep_fov_deg, tuple) or not self.focal_sweep_fov_deg:
            raise ValueError(
                "readiness_prior.focal_sweep_fov_deg must be a non-empty list"
            )
        for fov in self.focal_sweep_fov_deg:
            if not _is_finite_number(fov) or not 10.0 < float(fov) < 170.0:
                raise ValueError(
                    "readiness_prior.focal_sweep_fov_deg entries must be in (10, 170)"
                )
        if (
            not _is_finite_number(self.max_bearing_error_deg)
            or not 0.0 < float(self.max_bearing_error_deg) <= 180.0
        ):
            raise ValueError(
                "readiness_prior.max_bearing_error_deg must be in (0, 180]"
            )
        if not isinstance(self.retry_bearing_offsets_deg, tuple):
            raise ValueError("readiness_prior.retry_bearing_offsets_deg must be a list")
        for offset in self.retry_bearing_offsets_deg:
            if not _is_finite_number(offset) or abs(float(offset)) > 180.0:
                raise ValueError(
                    "readiness_prior.retry_bearing_offsets_deg entries must be in [-180, 180]"
                )
        if not isinstance(self.fallback_to_arrival_bearing, bool):
            raise ValueError(
                "readiness_prior.fallback_to_arrival_bearing must be a boolean"
            )
        if not _is_int(self.robot_image_max_width) or self.robot_image_max_width < 64:
            raise ValueError(
                "readiness_prior.robot_image_max_width must be an integer >= 64"
            )

    def _validate_vggt(self) -> None:
        if self.method not in _METHODS:
            raise ValueError(
                "readiness_prior.method must be one of "
                f"{sorted(_METHODS)}, got {self.method!r}"
            )
        url = self.vggt_service_url
        if not isinstance(url, str) or not url.strip() or not (
            url.startswith("http://") or url.startswith("https://")
        ):
            raise ValueError(
                "readiness_prior.vggt_service_url must be a non-empty http:// or https:// URL"
            )
        if (
            not _is_finite_number(self.vggt_timeout_s)
            or not 0.0 < float(self.vggt_timeout_s) <= 600.0
        ):
            raise ValueError("readiness_prior.vggt_timeout_s must be in (0, 600]")
        frames = self.vggt_frames
        if not isinstance(frames, tuple) or not frames:
            raise ValueError(
                "readiness_prior.vggt_frames must be a non-empty list of event frame labels"
            )
        for label in frames:
            if label not in _EVENT_FRAME_LABELS:
                raise ValueError(
                    "readiness_prior.vggt_frames entries must be one of "
                    f"{list(_EVENT_FRAME_LABELS)}, got {label!r}"
                )
        if len(set(frames)) != len(frames):
            raise ValueError("readiness_prior.vggt_frames must not repeat a label")
        stance_label = _FRAME_KEYS[self.source]
        if stance_label not in frames:
            raise ValueError(
                f"readiness_prior.vggt_frames must include the source frame {stance_label!r}"
            )
        if (
            not _is_finite_number(self.stance_offset_m)
            or not 0.0 <= float(self.stance_offset_m) <= 1.0
        ):
            raise ValueError("readiness_prior.stance_offset_m must be in [0, 1]")
        if (
            not _is_finite_number(self.near_object_m)
            or not 0.0 <= float(self.near_object_m) <= 2.0
        ):
            raise ValueError("readiness_prior.near_object_m must be in [0, 2]")
        if not _is_finite_number(self.max_height_m) or float(self.max_height_m) > 3.0:
            raise ValueError("readiness_prior.max_height_m must be a number <= 3.0")
        if (
            not _is_finite_number(self.min_height_m)
            or not 0.0 < float(self.min_height_m) < float(self.max_height_m)
        ):
            raise ValueError(
                "readiness_prior.min_height_m must satisfy 0 < min_height_m < max_height_m"
            )
        if (
            not _is_finite_number(self.max_scale_iqr_ratio)
            or float(self.max_scale_iqr_ratio) < 1.0
        ):
            raise ValueError("readiness_prior.max_scale_iqr_ratio must be a number >= 1.0")


@dataclass(frozen=True)
class ReadinessPriorEstimate:
    """Where the human stood, expressed for the docking controller.

    ``human_xy`` is the head (camera) floor point in odom for both methods.
    The pnp-only fields (``inliers``, ``reprojection_error_px``) are ``None``
    for vggt; the vggt-only fields (``stance_xy``, ``human_height_m``,
    ``scale``, ``scale_iqr_ratio``, ``frames``, ``service_time_s``) keep
    their defaults for pnp.
    """

    # Direction from the object toward where the human stood (odom frame).
    bearing_rad: float
    # Human optical axis projected onto the floor (odom yaw).
    heading_rad: float
    hand: str | None
    suggested_arm: str | None
    method: str
    inliers: int | None
    reprojection_error_px: float | None
    focal_px: float | None
    # Video time of the stance frame; None when the event carries no time.
    frame_time_s: float | None
    event_object: str
    human_xy: tuple[float, float]
    # Feet in odom (vggt): head minus stance_offset_m along the heading.
    stance_xy: tuple[float, float] | None = None
    # Head above the floor (vggt).
    human_height_m: float | None = None
    # VGGT -> metric depth scale and its interquartile ratio (vggt).
    scale: float | None = None
    scale_iqr_ratio: float | None = None
    # Event frame labels sent to the service, request order (vggt).
    frames: tuple[str, ...] = ()
    # Wall time of the HTTP call (vggt).
    service_time_s: float | None = None

    def to_metrics(self) -> dict[str, Any]:
        """JSON-ready dict: every field, ``None`` where not applicable, never NaN."""

        return {
            "method": self.method,
            "bearing_rad": float(self.bearing_rad),
            "bearing_deg": float(math.degrees(self.bearing_rad)),
            "heading_rad": float(self.heading_rad),
            "heading_deg": float(math.degrees(self.heading_rad)),
            "hand": self.hand,
            "suggested_arm": self.suggested_arm,
            "inliers": None if self.inliers is None else int(self.inliers),
            "reprojection_error_px": _finite_or_none(self.reprojection_error_px),
            "focal_px": _finite_or_none(self.focal_px),
            "frame_time_s": _finite_or_none(self.frame_time_s),
            "event_object": self.event_object,
            "human_xy": [float(self.human_xy[0]), float(self.human_xy[1])],
            "stance_xy": (
                None
                if self.stance_xy is None
                else [float(self.stance_xy[0]), float(self.stance_xy[1])]
            ),
            "human_height_m": _finite_or_none(self.human_height_m),
            "scale": _finite_or_none(self.scale),
            "scale_iqr_ratio": _finite_or_none(self.scale_iqr_ratio),
            "frames": [str(label) for label in self.frames],
            "service_time_s": _finite_or_none(self.service_time_s),
        }


@dataclass(frozen=True)
class PnPResult:
    """Human camera pose in the ZED optical frame: ``X_human = R(rvec) X_zed + tvec``."""

    rvec: np.ndarray
    tvec: np.ndarray
    inliers: int
    matches: int
    correspondences: int
    reprojection_error_px: float
    focal_px: float
    # One entry per swept field of view when the event had no intrinsics
    # (``fov_deg, focal_px, inliers, reprojection_error_px, rvec, tvec``).
    sweep: list[dict] | None

    def camera_center(self) -> np.ndarray:
        """Human camera centre in the ZED optical frame (``-R^T t``)."""

        rotation = _rotation_from_rvec(self.rvec)
        return -rotation.T @ np.asarray(self.tvec, dtype=np.float64).reshape(3)

    def optical_axis(self) -> np.ndarray:
        """Human optical axis (+z of the human camera) in the ZED optical frame."""

        rotation = _rotation_from_rvec(self.rvec)
        return rotation.T @ np.array([0.0, 0.0, 1.0])


def suggested_arm_for_hand(hand: str | None) -> str | None:
    """The human's grasp hand maps to the robot arm on the same side."""

    if hand in ("left", "right"):
        return hand
    return None


class ReadinessPrior:
    """Estimate the human's docking bearing for an object from the events file."""

    def __init__(
        self,
        config: ReadinessPriorConfig,
        *,
        matcher: Matcher | None = None,
        logger: logging.Logger | None = None,
        transport: Transport | None = None,
    ) -> None:
        self.config = config
        self._matcher = matcher
        self._logger = logger or logging.getLogger(__name__)
        # ``transport(url, payload, timeout_s)``; construction never touches
        # the network.
        self._transport: Transport = transport or _urllib_transport
        self._service_url = str(config.vggt_service_url).rstrip("/")
        if config.events_path is None:
            raise ValueError("readiness prior needs readiness_prior.events_path")
        self.events_path = Path(config.events_path).expanduser()
        payload = self._load_events(self.events_path)
        self.payload: dict[str, Any] = payload
        self.events: list[dict] = [dict(event) for event in payload["events"]]
        intrinsics = payload.get("intrinsics")
        self.video_intrinsics: dict[str, Any] | None = (
            dict(intrinsics) if isinstance(intrinsics, Mapping) else None
        )
        self.last_reason: str | None = None
        # pnp: counts from the most recent solve_pose call. vggt: the service
        # response plus ``request_frames`` / ``service_time_s``, or
        # ``{"error", "url"}`` when the transport failed.
        self.last_diagnostics: dict[str, Any] = {}

    # ---------------------------------------------------------------- service
    def check_service(self, timeout_s: float = 5.0) -> dict | None:
        """``GET {vggt_service_url}/health``: its JSON, or ``None`` when unreachable."""

        url = f"{self._service_url}/health"
        try:
            body = self._transport(url, None, float(timeout_s))
        except Exception as exc:  # noqa: BLE001 - any transport failure = unavailable
            self.last_reason = "vggt_service_unavailable"
            self.last_diagnostics = self._transport_error_diagnostics(exc, url)
            self._logger.warning(
                "readiness prior: VGGT service health check failed at %s (%s)",
                url,
                self.last_diagnostics["error"],
            )
            return None
        if not isinstance(body, Mapping):
            self.last_reason = "vggt_service_unavailable"
            self.last_diagnostics = {
                "error": f"health body is {type(body).__name__}, not an object",
                "url": url,
            }
            return None
        self.last_reason = None
        self.last_diagnostics = dict(body)
        return dict(body)

    @staticmethod
    def _transport_error_diagnostics(exc: BaseException, url: str) -> dict[str, Any]:
        diagnostics: dict[str, Any] = {
            "error": f"{type(exc).__name__}: {exc}",
            "url": url,
        }
        if isinstance(exc, urllib.error.HTTPError):
            diagnostics["http_status"] = int(exc.code)
            try:
                detail = exc.read(4096)
            except Exception:  # noqa: BLE001 - the body is optional context
                detail = b""
            if detail:
                diagnostics["detail"] = detail.decode("utf-8", errors="replace")
        return diagnostics

    # ----------------------------------------------------------------- events
    @staticmethod
    def _load_events(path: Path) -> dict[str, Any]:
        if not path.is_file():
            raise FileNotFoundError(f"readiness events file not found: {path}")
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, Mapping):
            raise ValueError("readiness events JSON must be an object")
        schema = payload.get("schema_version")
        if schema != EVENTS_SCHEMA:
            raise ValueError(
                f"unsupported schema_version {schema!r} in {path} "
                f"(expected {EVENTS_SCHEMA!r})"
            )
        events = payload.get("events")
        if not isinstance(events, list):
            raise ValueError("readiness events JSON must contain an 'events' list")
        for index, event in enumerate(events):
            if (
                not isinstance(event, Mapping)
                or not isinstance(event.get("object"), str)
                or not isinstance(event.get("frames"), Mapping)
            ):
                raise ValueError(
                    f"readiness event {index} must carry 'object' and 'frames'"
                )
        return dict(payload)

    def match_event(self, object_name: str) -> dict | None:
        """First event (file order) sharing a non-article word with ``object_name``."""

        wanted = _tokens(object_name)
        if not wanted:
            return None
        for index, event in enumerate(self.events):
            if wanted & _tokens(event.get("object", "")):
                self._logger.info(
                    "readiness prior: object %r matched event %d %r (hand=%s, "
                    "t_ready=%.2f s)",
                    object_name,
                    index,
                    event.get("object"),
                    event.get("hand"),
                    float(event.get("t_ready_s", float("nan"))),
                )
                return event
        return None

    def select_frame(self, event: Mapping[str, Any]) -> dict | None:
        """The event frame chosen by ``config.source`` (``ready`` or ``nav``)."""

        frames = event.get("frames")
        if not isinstance(frames, Mapping):
            return None
        frame = frames.get(_FRAME_KEYS[self.config.source])
        if not isinstance(frame, Mapping) or not frame.get("path"):
            return None
        return dict(frame)

    def frame_path(self, frame: Mapping[str, Any]) -> Path:
        """Frame file: relative paths hang off the events JSON's directory."""

        path = Path(str(frame["path"])).expanduser()
        if not path.is_absolute():
            path = self.events_path.parent / path
        return path

    def scaled_event_intrinsics(
        self,
        frame: Mapping[str, Any],
        *,
        decoded_size: tuple[int, int] | None = None,
    ) -> dict[str, float] | None:
        """Sidecar intrinsics rescaled to the frame's pixels, or None without a sidecar.

        ``decoded_size`` is ``(width, height)`` of the frame as loaded; when it
        differs from the recorded frame width the scale follows the pixels.
        """

        sidecar = self.video_intrinsics
        if not sidecar:
            return None
        scale = frame.get("scale")
        if scale is None:
            frame_width = frame.get("width")
            source_width = sidecar.get("width")
            if not frame_width or not source_width:
                return None
            scale = float(frame_width) / float(source_width)
        scale = float(scale)
        if decoded_size is not None and frame.get("width"):
            scale *= float(decoded_size[0]) / float(frame["width"])
        if not math.isfinite(scale) or scale <= 0.0:
            return None
        try:
            fx = float(sidecar["fx"])
            fy = float(sidecar["fy"])
            cx = float(sidecar["cx"])
            cy = float(sidecar["cy"])
        except (KeyError, TypeError, ValueError):
            return None
        return {
            "fx": fx * scale,
            "fy": fy * scale,
            "cx": (cx + 0.5) * scale - 0.5,
            "cy": (cy + 0.5) * scale - 0.5,
        }

    # --------------------------------------------------------------- matching
    def _load_frame_rgb(self, path: Path) -> np.ndarray | None:
        try:
            cv2 = _cv2()
        except ImportError:
            self.last_reason = "cv2_unavailable"
            return None
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None or image.ndim != 3 or image.shape[2] != 3:
            self.last_reason = "frame_missing"
            return None
        return np.ascontiguousarray(image[:, :, ::-1])

    def _default_matcher(self, cv2) -> Matcher:
        sift = cv2.SIFT_create(nfeatures=int(self.config.sift_features))
        ratio = float(self.config.ratio_test)

        def match(event_rgb: np.ndarray, robot_rgb: np.ndarray):
            event_gray = cv2.cvtColor(
                np.ascontiguousarray(event_rgb), cv2.COLOR_RGB2GRAY
            )
            robot_gray = cv2.cvtColor(
                np.ascontiguousarray(robot_rgb), cv2.COLOR_RGB2GRAY
            )
            event_keypoints, event_descriptors = sift.detectAndCompute(event_gray, None)
            robot_keypoints, robot_descriptors = sift.detectAndCompute(robot_gray, None)
            empty = np.zeros((0, 2), dtype=np.float64)
            if (
                event_descriptors is None
                or robot_descriptors is None
                or len(event_keypoints) < 2
                or len(robot_keypoints) < 2
            ):
                return empty, empty.copy()
            pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(
                event_descriptors, robot_descriptors, k=2
            )
            event_px = []
            robot_px = []
            for pair in pairs:
                if len(pair) < 2:
                    continue
                best, second = pair[0], pair[1]
                if best.distance < ratio * second.distance:
                    event_px.append(event_keypoints[best.queryIdx].pt)
                    robot_px.append(robot_keypoints[best.trainIdx].pt)
            if not event_px:
                return empty, empty.copy()
            return (
                np.asarray(event_px, dtype=np.float64).reshape(-1, 2),
                np.asarray(robot_px, dtype=np.float64).reshape(-1, 2),
            )

        return match

    def _matcher_for(self, cv2) -> Matcher:
        if self._matcher is None:
            self._matcher = self._default_matcher(cv2)
        return self._matcher

    def solve_pose(
        self,
        event_rgb: np.ndarray,
        robot_rgb: np.ndarray,
        depth: np.ndarray,
        intrinsics: np.ndarray,
        *,
        event_intrinsics: Mapping[str, float] | None,
        frame_size: tuple[int, int],
    ) -> PnPResult | None:
        """Solve the human camera pose in the ZED frame from one event frame.

        ``event_intrinsics`` holds ``fx, fy, cx, cy`` in ``event_rgb`` pixels
        (see :meth:`scaled_event_intrinsics`); ``None`` sweeps
        ``config.focal_sweep_fov_deg`` over ``frame_size = (width, height)``.
        ``intrinsics`` is the 3x3 ZED matrix at the depth image's resolution.
        On failure returns ``None`` and sets ``last_reason``.
        """

        self.last_diagnostics = {}
        try:
            cv2 = _cv2()
        except ImportError:
            self.last_reason = "cv2_unavailable"
            return None
        matcher = self._matcher_for(cv2)

        robot_rgb = np.asarray(robot_rgb)
        depth = np.asarray(depth, dtype=np.float64)
        camera = np.asarray(intrinsics, dtype=np.float64).reshape(3, 3)
        height, width = depth.shape[:2]
        robot_scale = min(1.0, float(self.config.robot_image_max_width) / float(width))
        if robot_scale < 1.0:
            scaled_size = (
                max(1, int(round(width * robot_scale))),
                max(1, int(round(height * robot_scale))),
            )
            matcher_image = cv2.resize(robot_rgb, scaled_size, interpolation=cv2.INTER_AREA)
            robot_scale = scaled_size[0] / float(width)
        else:
            matcher_image = robot_rgb

        event_px, robot_px = matcher(np.asarray(event_rgb), matcher_image)
        event_px = np.asarray(event_px, dtype=np.float64).reshape(-1, 2)
        robot_px = np.asarray(robot_px, dtype=np.float64).reshape(-1, 2) / robot_scale
        matches = int(min(len(event_px), len(robot_px)))
        self.last_diagnostics["matches"] = matches
        if matches < max(_MIN_PNP_CORRESPONDENCES, int(self.config.min_inliers)):
            self.last_reason = "too_few_matches"
            return None

        fx, fy = camera[0, 0], camera[1, 1]
        cx, cy = camera[0, 2], camera[1, 2]
        columns = np.rint(robot_px[:matches, 0]).astype(np.int64)
        rows = np.rint(robot_px[:matches, 1]).astype(np.int64)
        inside = (rows >= 0) & (rows < height) & (columns >= 0) & (columns < width)
        z = np.full(matches, np.nan)
        z[inside] = depth[rows[inside], columns[inside]]
        valid = inside & np.isfinite(z) & (z > 0.0)
        correspondences = int(np.count_nonzero(valid))
        self.last_diagnostics["correspondences"] = correspondences
        if correspondences < _MIN_PNP_CORRESPONDENCES:
            self.last_reason = "too_few_correspondences"
            return None
        object_points = np.stack(
            [
                (columns[valid] - cx) * z[valid] / fx,
                (rows[valid] - cy) * z[valid] / fy,
                z[valid],
            ],
            axis=1,
        )
        image_points = event_px[:matches][valid]

        sweep: list[dict] | None = None
        if event_intrinsics is not None:
            candidates = [
                (
                    None,
                    float(event_intrinsics["fx"]),
                    self._camera_matrix(
                        float(event_intrinsics["fx"]),
                        float(event_intrinsics["fy"]),
                        float(event_intrinsics["cx"]),
                        float(event_intrinsics["cy"]),
                    ),
                )
            ]
        else:
            frame_width, frame_height = float(frame_size[0]), float(frame_size[1])
            candidates = []
            for fov_deg in self.config.focal_sweep_fov_deg:
                focal = (frame_width / 2.0) / math.tan(math.radians(float(fov_deg)) / 2.0)
                candidates.append(
                    (
                        float(fov_deg),
                        focal,
                        self._camera_matrix(
                            focal, focal, (frame_width - 1.0) / 2.0, (frame_height - 1.0) / 2.0
                        ),
                    )
                )
            sweep = []

        best: tuple[int, float, np.ndarray, np.ndarray, float] | None = None
        for fov_deg, focal, matrix in candidates:
            solved = self._solve_single(cv2, object_points, image_points, matrix)
            if sweep is not None:
                entry = {
                    "fov_deg": fov_deg,
                    "focal_px": float(focal),
                    "inliers": 0 if solved is None else int(solved[0]),
                    "reprojection_error_px": None if solved is None else float(solved[1]),
                    "rvec": None if solved is None else [float(v) for v in solved[2].reshape(3)],
                    "tvec": None if solved is None else [float(v) for v in solved[3].reshape(3)],
                }
                sweep.append(entry)
            if solved is None:
                continue
            inliers, error, rvec, tvec = solved
            if (
                best is None
                or inliers > best[0]
                or (inliers == best[0] and error < best[1])
            ):
                best = (inliers, error, rvec, tvec, float(focal))
        self.last_diagnostics["sweep"] = sweep
        if best is None:
            self.last_reason = "pnp_failed"
            return None
        inliers, error, rvec, tvec, focal = best
        self.last_diagnostics.update(
            {"inliers": inliers, "reprojection_error_px": error, "focal_px": focal}
        )
        if inliers < int(self.config.min_inliers):
            self.last_reason = "too_few_inliers"
            return None
        if error > float(self.config.max_reprojection_error_px):
            self.last_reason = "reprojection_error_too_high"
            return None
        return PnPResult(
            rvec=np.asarray(rvec, dtype=np.float64).reshape(3),
            tvec=np.asarray(tvec, dtype=np.float64).reshape(3),
            inliers=inliers,
            matches=matches,
            correspondences=correspondences,
            reprojection_error_px=error,
            focal_px=focal,
            sweep=sweep,
        )

    @staticmethod
    def _camera_matrix(fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
        return np.array(
            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64
        )

    def _solve_single(
        self,
        cv2,
        object_points: np.ndarray,
        image_points: np.ndarray,
        camera_matrix: np.ndarray,
    ) -> tuple[int, float, np.ndarray, np.ndarray] | None:
        """PnP-RANSAC (EPNP) then LM refinement on the inliers; RMS over inliers."""

        objects = np.ascontiguousarray(object_points, dtype=np.float64).reshape(-1, 1, 3)
        images = np.ascontiguousarray(image_points, dtype=np.float64).reshape(-1, 1, 2)
        try:
            ok, rvec, tvec, inlier_index = cv2.solvePnPRansac(
                objects,
                images,
                camera_matrix,
                None,
                iterationsCount=500,
                reprojectionError=float(self.config.max_reprojection_error_px),
                confidence=0.999,
                flags=cv2.SOLVEPNP_EPNP,
            )
        except cv2.error:
            return None
        if not ok or inlier_index is None:
            return None
        index = np.asarray(inlier_index, dtype=np.int64).reshape(-1)
        if index.size < 4:
            return None
        inlier_objects = np.ascontiguousarray(objects[index])
        inlier_images = np.ascontiguousarray(images[index])
        try:
            rvec, tvec = cv2.solvePnPRefineLM(
                inlier_objects, inlier_images, camera_matrix, None, rvec, tvec
            )
        except cv2.error:
            pass
        projected, _ = cv2.projectPoints(inlier_objects, rvec, tvec, camera_matrix, None)
        residual = projected.reshape(-1, 2) - inlier_images.reshape(-1, 2)
        error = float(math.sqrt(float(np.mean(np.sum(residual * residual, axis=1)))))
        if not math.isfinite(error):
            return None
        return int(index.size), error, np.asarray(rvec, dtype=np.float64), np.asarray(
            tvec, dtype=np.float64
        )

    # --------------------------------------------------------------- geometry
    def planar_pose(
        self,
        result: PnPResult,
        pose_xy_yaw: np.ndarray,
        *,
        down: np.ndarray,
        planar_forward: np.ndarray,
        planar_left: np.ndarray,
        target_xy: tuple[float, float],
    ) -> dict[str, Any] | None:
        """Project the solved human camera onto the floor and into odom.

        Returns ``bearing_rad`` (object -> human), ``heading_rad`` (human optical
        axis as odom yaw), ``human_xy`` (odom), ``human_camera_xyz`` (ZED optical
        frame) and ``human_height_above_camera_m``; ``None`` (with
        ``last_reason == "degenerate_geometry"``) for an implausible solution.
        """

        center = result.camera_center()
        axis = result.optical_axis()
        pose = np.asarray(pose_xy_yaw, dtype=np.float64).reshape(3)
        forward = np.asarray(planar_forward, dtype=np.float64).reshape(3)
        left = np.asarray(planar_left, dtype=np.float64).reshape(3)
        down_axis = np.asarray(down, dtype=np.float64).reshape(3)
        if (
            not np.all(np.isfinite(center))
            or not np.all(np.isfinite(axis))
            or float(np.linalg.norm(center)) > _MAX_HUMAN_DISTANCE_M
        ):
            self.last_reason = "degenerate_geometry"
            return None
        forward_h = float(center @ forward)
        left_h = float(center @ left)
        axis_forward = float(axis @ forward)
        axis_left = float(axis @ left)
        if math.hypot(axis_forward, axis_left) < 1e-6:
            self.last_reason = "degenerate_geometry"
            return None
        human_x, human_y = _planar_to_world(forward_h, left_h, pose)
        bearing = _wrap_angle(
            math.atan2(human_y - float(target_xy[1]), human_x - float(target_xy[0]))
        )
        heading = _wrap_angle(math.atan2(axis_left, axis_forward) + float(pose[2]))
        return {
            "bearing_rad": bearing,
            "heading_rad": heading,
            "human_xy": (float(human_x), float(human_y)),
            "human_forward_left_m": (forward_h, left_h),
            "human_camera_xyz": [float(v) for v in center],
            "human_height_above_camera_m": float(-(center @ down_axis)),
        }

    # --------------------------------------------------------------- estimate
    def estimate(
        self,
        object_name: str,
        rgb: np.ndarray,
        depth: np.ndarray,
        intrinsics: np.ndarray,
        pose_xy_yaw: np.ndarray,
        *,
        down: np.ndarray,
        planar_forward: np.ndarray,
        planar_left: np.ndarray,
        target_xy: tuple[float, float],
        camera_height_m: float | None = None,
    ) -> ReadinessPriorEstimate | None:
        """Bearing and heading of the human's ready stance for ``object_name``.

        ``rgb``/``depth``/``intrinsics`` are the ZED view at the depth
        resolution, ``pose_xy_yaw`` the odom base pose that view was taken
        from, ``down``/``planar_forward``/``planar_left`` the calibrated floor
        axes in the ZED optical frame and ``target_xy`` the object in odom.
        ``camera_height_m`` is the ZED height above the floor; ``method: vggt``
        needs it (``camera_height_missing`` otherwise), pnp ignores it.
        Returns ``None`` and sets ``last_reason`` when no estimate is possible.
        """

        self.last_reason = None
        event = self.match_event(self.config.event_object or object_name)
        if event is None:
            self.last_reason = "no_event_match"
            return None
        if self.config.method == "pnp":
            return self._estimate_pnp(
                object_name,
                event,
                rgb,
                depth,
                intrinsics,
                pose_xy_yaw,
                down=down,
                planar_forward=planar_forward,
                planar_left=planar_left,
                target_xy=target_xy,
            )
        return self._estimate_vggt(
            object_name,
            event,
            rgb,
            depth,
            intrinsics,
            pose_xy_yaw,
            down=down,
            planar_forward=planar_forward,
            planar_left=planar_left,
            target_xy=target_xy,
            camera_height_m=camera_height_m,
        )

    # ------------------------------------------------------------------- vggt
    def _resolve_request_frames(
        self, event: Mapping[str, Any]
    ) -> tuple[list[tuple[str, Path]], dict | None]:
        """``config.vggt_frames`` resolved to files, request order kept.

        Returns ``(frames, stance_frame)``; ``stance_frame`` is ``None`` (and
        ``last_reason`` is ``frame_missing``) when the ``source`` frame has no
        entry or no file. Any other missing frame is dropped with a WARNING.
        """

        stance_label = _FRAME_KEYS[self.config.source]
        stance_frame = self.select_frame(event)
        if stance_frame is None:
            self.last_reason = "frame_missing"
            self._logger.warning(
                "readiness prior: event %r has no %s frame entry",
                event.get("object"),
                stance_label,
            )
            return [], None
        stance_path = self.frame_path(stance_frame)
        if not stance_path.is_file():
            self.last_reason = "frame_missing"
            self._logger.warning("readiness prior: event frame missing at %s", stance_path)
            return [], None
        entries = event.get("frames")
        frames: list[tuple[str, Path]] = []
        for label in self.config.vggt_frames:
            if label == stance_label:
                frames.append((label, stance_path))
                continue
            entry = entries.get(label) if isinstance(entries, Mapping) else None
            if not isinstance(entry, Mapping) or not entry.get("path"):
                self._logger.warning(
                    "readiness prior: event %r has no %s frame entry; sending %s without it",
                    event.get("object"),
                    label,
                    list(self.config.vggt_frames),
                )
                continue
            path = self.frame_path(entry)
            if not path.is_file():
                self._logger.warning(
                    "readiness prior: %s frame missing at %s; sending %s without it",
                    label,
                    path,
                    list(self.config.vggt_frames),
                )
                continue
            frames.append((label, path))
        return frames, stance_frame

    def build_stance_request(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        intrinsics: np.ndarray,
        *,
        down: np.ndarray,
        planar_forward: np.ndarray,
        planar_left: np.ndarray,
        camera_height_m: float,
        frames: list[tuple[str, bytes]],
    ) -> dict[str, Any]:
        """The ``POST /stance`` body (``yor-vggt-stance-v1`` request shape)."""

        def axis(vector: np.ndarray) -> list[float]:
            return [float(v) for v in np.asarray(vector, dtype=np.float64).reshape(3)]

        matrix = np.asarray(intrinsics, dtype=np.float64).reshape(3, 3)
        return {
            "robot": {
                "label": "zed",
                "rgb_jpeg_base64": _b64(_encode_rgb_jpeg(rgb)),
                "depth_png16_base64": _b64(_encode_depth_png16(depth)),
                "intrinsics": [[float(v) for v in row] for row in matrix],
                "down": axis(down),
                "planar_forward": axis(planar_forward),
                "planar_left": axis(planar_left),
                "ground_camera_height_m": float(camera_height_m),
            },
            "frames": [
                {"label": str(label), "jpeg_base64": _b64(data)} for label, data in frames
            ],
        }

    def _estimate_vggt(
        self,
        object_name: str,
        event: Mapping[str, Any],
        rgb: np.ndarray,
        depth: np.ndarray,
        intrinsics: np.ndarray,
        pose_xy_yaw: np.ndarray,
        *,
        down: np.ndarray,
        planar_forward: np.ndarray,
        planar_left: np.ndarray,
        target_xy: tuple[float, float],
        camera_height_m: float | None,
    ) -> ReadinessPriorEstimate | None:
        self.last_diagnostics = {}
        frames, stance_frame = self._resolve_request_frames(event)
        if stance_frame is None:
            return None
        labels = [label for label, _ in frames]
        stance_label = _FRAME_KEYS[self.config.source]
        stance_index = labels.index(stance_label)
        if camera_height_m is None or not _is_finite_number(camera_height_m):
            self.last_reason = "camera_height_missing"
            self.last_diagnostics = {"request_frames": labels}
            self._logger.warning(
                "readiness prior: method vggt needs camera_height_m (got %r)",
                camera_height_m,
            )
            return None

        payload = self.build_stance_request(
            rgb,
            depth,
            intrinsics,
            down=down,
            planar_forward=planar_forward,
            planar_left=planar_left,
            camera_height_m=float(camera_height_m),
            frames=[(label, path.read_bytes()) for label, path in frames],
        )
        url = f"{self._service_url}/stance"
        started = time.perf_counter()
        try:
            response = self._transport(url, payload, float(self.config.vggt_timeout_s))
        except Exception as exc:  # noqa: BLE001 - URLError, HTTPError, timeout, OSError, ...
            self.last_reason = "vggt_service_unavailable"
            self.last_diagnostics = self._transport_error_diagnostics(exc, url)
            self.last_diagnostics["request_frames"] = labels
            self.last_diagnostics["service_time_s"] = time.perf_counter() - started
            self._logger.warning(
                "readiness prior: VGGT service call failed for %r after %.2f s (%s)",
                object_name,
                self.last_diagnostics["service_time_s"],
                self.last_diagnostics["error"],
            )
            return None
        service_time_s = time.perf_counter() - started

        if not isinstance(response, Mapping):
            self.last_reason = "vggt_response_invalid"
            self.last_diagnostics = {
                "error": f"response body is {type(response).__name__}, not an object",
                "url": url,
                "request_frames": labels,
                "service_time_s": service_time_s,
            }
            self._logger.warning(
                "readiness prior: invalid VGGT response for %r (%s)",
                object_name,
                self.last_diagnostics["error"],
            )
            return None
        self.last_diagnostics = dict(response)
        self.last_diagnostics["request_frames"] = labels
        self.last_diagnostics["service_time_s"] = service_time_s

        row = self._validate_stance_response(response, labels, stance_index)
        if row is None:
            self._logger.warning(
                "readiness prior: invalid VGGT response for %r (%s)",
                object_name,
                self.last_diagnostics.get("error"),
            )
            return None

        # Gates, scale first: the metric fields of every row need the scale.
        scale = _finite_or_none(response.get("scale"))
        iqr_ratio = _finite_or_none(response.get("scale_iqr_ratio"))
        if (
            scale is None
            or iqr_ratio is None
            or iqr_ratio > float(self.config.max_scale_iqr_ratio)
        ):
            self.last_reason = "scale_unstable"
            self._logger.warning(
                "readiness prior: VGGT scale unstable for %r (scale=%s, iqr_ratio=%s, "
                "max %.2f, pixels=%s, frames=%s)",
                object_name,
                response.get("scale"),
                response.get("scale_iqr_ratio"),
                float(self.config.max_scale_iqr_ratio),
                response.get("scale_pixels"),
                labels,
            )
            return None
        forward_m = _finite_or_none(row.get("forward_m"))
        left_m = _finite_or_none(row.get("left_m"))
        height_m = _finite_or_none(row.get("height_above_floor_m"))
        if forward_m is None or left_m is None or height_m is None:
            self.last_reason = "vggt_response_invalid"
            self.last_diagnostics["error"] = (
                f"{stance_label} row has non-finite metric fields: "
                f"forward_m={row.get('forward_m')!r}, left_m={row.get('left_m')!r}, "
                f"height_above_floor_m={row.get('height_above_floor_m')!r}"
            )
            self._logger.warning(
                "readiness prior: invalid VGGT response for %r (%s)",
                object_name,
                self.last_diagnostics["error"],
            )
            return None
        if not float(self.config.min_height_m) <= height_m <= float(self.config.max_height_m):
            self.last_reason = "implausible_height"
            self._logger.warning(
                "readiness prior: VGGT head height %.2f m for %r outside [%.2f, %.2f] "
                "(scale=%.3f, iqr_ratio=%.2f, frames=%s)",
                height_m,
                object_name,
                float(self.config.min_height_m),
                float(self.config.max_height_m),
                scale,
                iqr_ratio,
                labels,
            )
            return None

        # Stance rule: feet = head - stance_offset_m along the heading; a head
        # within near_object_m of the object docks from heading + 180 deg,
        # otherwise the bearing runs from the object to the feet.
        pose = np.asarray(pose_xy_yaw, dtype=np.float64).reshape(3)
        head_x, head_y = _planar_to_world(forward_m, left_m, pose)
        heading = _wrap_angle(math.radians(float(row["heading_deg"])) + float(pose[2]))
        offset = float(self.config.stance_offset_m)
        feet_x = head_x - offset * math.cos(heading)
        feet_y = head_y - offset * math.sin(heading)
        target_x, target_y = float(target_xy[0]), float(target_xy[1])
        distance = math.hypot(head_x - target_x, head_y - target_y)
        near_object = distance <= float(self.config.near_object_m)
        if near_object:
            bearing = _wrap_angle(heading + math.pi)
        else:
            bearing = _wrap_angle(math.atan2(feet_y - target_y, feet_x - target_x))

        hand = event.get("hand")
        hand = hand if hand in ("left", "right", "both") else None
        estimate = ReadinessPriorEstimate(
            bearing_rad=float(bearing),
            heading_rad=float(heading),
            hand=hand,
            suggested_arm=suggested_arm_for_hand(hand),
            method="vggt",
            inliers=None,
            reprojection_error_px=None,
            focal_px=_finite_or_none(row.get("focal_px")),
            frame_time_s=_frame_time_s(stance_frame, event),
            event_object=str(event.get("object")),
            human_xy=(float(head_x), float(head_y)),
            stance_xy=(float(feet_x), float(feet_y)),
            human_height_m=float(height_m),
            scale=float(scale),
            scale_iqr_ratio=float(iqr_ratio),
            frames=tuple(labels),
            service_time_s=float(service_time_s),
        )
        self._logger.info(
            "readiness prior: %r head %.2f m ahead, %.2f m left of the ZED, %.2f m "
            "above the floor, heading %.1f deg -> feet at odom (%.2f, %.2f), bearing "
            "%.1f deg (%s, head %.2f m from the object), scale %.3f (iqr ratio %.2f), "
            "frames %s, service %.2f s, hand=%s",
            object_name,
            forward_m,
            left_m,
            height_m,
            math.degrees(heading),
            feet_x,
            feet_y,
            math.degrees(bearing),
            "near object: heading + 180" if near_object else "object -> feet",
            distance,
            scale,
            iqr_ratio,
            labels,
            service_time_s,
            hand,
        )
        return estimate

    def _validate_stance_response(
        self, response: Mapping[str, Any], labels: list[str], stance_index: int
    ) -> dict | None:
        """The stance row of a well-formed response, else ``None`` + ``vggt_response_invalid``.

        Only the structure is checked here (keys, one row per frame sent,
        finite heading); the metric fields are read after the scale gate
        because the service nulls them when it has no scale.
        """

        def invalid(message: str) -> None:
            self.last_reason = "vggt_response_invalid"
            self.last_diagnostics["error"] = message

        missing = [key for key in ("cameras", "scale", "scale_iqr_ratio") if key not in response]
        if missing:
            invalid(f"response lacks {missing}")
            return None
        cameras = response.get("cameras")
        if not isinstance(cameras, list) or len(cameras) != len(labels):
            count = len(cameras) if isinstance(cameras, list) else type(cameras).__name__
            invalid(f"response has {count} cameras for {len(labels)} frames {labels}")
            return None
        row = cameras[stance_index]
        if not isinstance(row, Mapping):
            invalid(f"camera row {stance_index} is {type(row).__name__}, not an object")
            return None
        required = ("heading_deg", "forward_m", "left_m", "height_above_floor_m")
        absent = [key for key in required if key not in row]
        if absent:
            invalid(f"camera row {stance_index} lacks {absent}")
            return None
        if _finite_or_none(row.get("heading_deg")) is None:
            invalid(f"camera row {stance_index} heading_deg is {row.get('heading_deg')!r}")
            return None
        return dict(row)

    # -------------------------------------------------------------------- pnp
    def _estimate_pnp(
        self,
        object_name: str,
        event: Mapping[str, Any],
        rgb: np.ndarray,
        depth: np.ndarray,
        intrinsics: np.ndarray,
        pose_xy_yaw: np.ndarray,
        *,
        down: np.ndarray,
        planar_forward: np.ndarray,
        planar_left: np.ndarray,
        target_xy: tuple[float, float],
    ) -> ReadinessPriorEstimate | None:
        frame = self.select_frame(event)
        if frame is None:
            self.last_reason = "frame_missing"
            return None
        path = self.frame_path(frame)
        if not path.is_file():
            self.last_reason = "frame_missing"
            self._logger.warning("readiness prior: event frame missing at %s", path)
            return None
        event_rgb = self._load_frame_rgb(path)
        if event_rgb is None:
            return None
        frame_size = (int(event_rgb.shape[1]), int(event_rgb.shape[0]))
        event_intrinsics = self.scaled_event_intrinsics(frame, decoded_size=frame_size)
        result = self.solve_pose(
            event_rgb,
            rgb,
            depth,
            intrinsics,
            event_intrinsics=event_intrinsics,
            frame_size=frame_size,
        )
        if result is None:
            self._logger.info(
                "readiness prior: no pose for %r (%s; %s)",
                object_name,
                self.last_reason,
                self.last_diagnostics,
            )
            return None
        geometry = self.planar_pose(
            result,
            pose_xy_yaw,
            down=down,
            planar_forward=planar_forward,
            planar_left=planar_left,
            target_xy=target_xy,
        )
        if geometry is None:
            return None
        hand = event.get("hand")
        hand = hand if hand in ("left", "right", "both") else None
        estimate = ReadinessPriorEstimate(
            bearing_rad=float(geometry["bearing_rad"]),
            heading_rad=float(geometry["heading_rad"]),
            hand=hand,
            suggested_arm=suggested_arm_for_hand(hand),
            method="pnp",
            inliers=int(result.inliers),
            reprojection_error_px=float(result.reprojection_error_px),
            focal_px=float(result.focal_px),
            frame_time_s=_frame_time_s(frame, event),
            event_object=str(event.get("object")),
            human_xy=geometry["human_xy"],
        )
        self._logger.info(
            "readiness prior: %r human at odom (%.2f, %.2f), bearing %.1f deg, "
            "heading %.1f deg, %d/%d inliers, rms %.2f px, focal %.0f px, "
            "eye %.2f m above the ZED, hand=%s",
            object_name,
            estimate.human_xy[0],
            estimate.human_xy[1],
            math.degrees(estimate.bearing_rad),
            math.degrees(estimate.heading_rad),
            result.inliers,
            result.correspondences,
            result.reprojection_error_px,
            result.focal_px,
            geometry["human_height_above_camera_m"],
            hand,
        )
        return estimate
