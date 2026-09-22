#!/usr/bin/env python3
"""Publish atomic ZED RGB-D and optional tracking poses for NavDP."""

from __future__ import annotations

import argparse
from collections import deque
import sys
import time
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from commlink import Publisher
import pyzed.sl as sl

from robot.utils.ndarray_wire import pack_array
from navdp.navdp_deploy.geometry import zed_pose_to_planar


TOPIC = "navdp/rgbd"
FORMAT = "navdp-rgbd-frame-v1"
DEPTH_MODES = {
    "neural-light": sl.DEPTH_MODE.NEURAL_LIGHT,
    "neural": sl.DEPTH_MODE.NEURAL,
    "neural-plus": sl.DEPTH_MODE.NEURAL_PLUS,
}


def _zed_vector(value, count: int) -> np.ndarray:
    if hasattr(value, "get"):
        value = value.get()
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.size != count:
        raise RuntimeError(
            f"ZED SDK returned {array.size} values; expected {count}"
        )
    return array


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Atomic RGB-D-only ZED publisher for NavDP"
    )
    parser.add_argument("--port", type=int, default=6000)
    parser.add_argument("--fps", type=int, choices=(15, 30, 60), default=30)
    parser.add_argument(
        "--depth-mode",
        choices=sorted(DEPTH_MODES),
        default="neural",
    )
    parser.add_argument("--depth-max-m", type=float, default=20.0)
    parser.add_argument("--confidence-threshold", type=int, default=87)
    parser.add_argument("--texture-confidence-threshold", type=int, default=95)
    parser.add_argument("--log-hz", type=float, default=1.0)
    parser.add_argument(
        "--floor-plane-hz",
        type=float,
        default=2.0,
        help="ZED SDK floor-plane updates per second (0 disables them)",
    )
    parser.add_argument(
        "--floor-plane-max-age-s",
        type=float,
        default=2.0,
        help="maximum age of a floor-plane estimate published as valid",
    )
    parser.add_argument(
        "--disable-positional-tracking",
        action="store_true",
        help="publish RGB-D without the pose feedback required by navigate-zed",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.depth_max_m <= 0:
        raise SystemExit("--depth-max-m must be positive")
    if not 1 <= args.confidence_threshold <= 100:
        raise SystemExit("--confidence-threshold must be in [1, 100]")
    if not 1 <= args.texture_confidence_threshold <= 100:
        raise SystemExit("--texture-confidence-threshold must be in [1, 100]")
    if args.log_hz < 0:
        raise SystemExit("--log-hz must be nonnegative")
    if not 0.0 <= args.floor_plane_hz <= 10.0:
        raise SystemExit("--floor-plane-hz must be in [0, 10]")
    if not 0.2 <= args.floor_plane_max_age_s <= 10.0:
        raise SystemExit("--floor-plane-max-age-s must be in [0.2, 10]")

    init = sl.InitParameters()
    init.camera_resolution = sl.RESOLUTION.VGA
    init.camera_fps = args.fps
    init.depth_mode = DEPTH_MODES[args.depth_mode]
    init.coordinate_units = sl.UNIT.METER
    init.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    init.depth_maximum_distance = float(args.depth_max_m)
    init.depth_stabilization = 1

    runtime = sl.RuntimeParameters()
    runtime.confidence_threshold = args.confidence_threshold
    runtime.texture_confidence_threshold = args.texture_confidence_threshold

    camera = sl.Camera()
    status = camera.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"ZED open failed: {status}")

    tracking_enabled = not args.disable_positional_tracking
    if tracking_enabled:
        tracking_parameters = sl.PositionalTrackingParameters()
        tracking_parameters.set_gravity_as_origin = True
        tracking_parameters.enable_area_memory = False
        status = camera.enable_positional_tracking(tracking_parameters)
        if status != sl.ERROR_CODE.SUCCESS:
            camera.close()
            raise RuntimeError(f"ZED positional tracking failed: {status}")

    info = camera.get_camera_information()
    width = int(info.camera_configuration.resolution.width)
    height = int(info.camera_configuration.resolution.height)
    serial = int(info.serial_number)
    image = sl.Mat(width, height, sl.MAT_TYPE.U8_C4, memory_type=sl.MEM.CPU)
    depth = sl.Mat(width, height, sl.MAT_TYPE.F32_C1, memory_type=sl.MEM.CPU)
    pose = sl.Pose()
    floor_plane = sl.Plane()
    floor_to_camera = sl.Transform()
    publisher = Publisher("*", port=args.port)
    sequence = 0
    next_log = 0.0
    started = time.monotonic()
    floor_enabled = tracking_enabled and args.floor_plane_hz > 0.0
    next_floor_query = 0.0
    floor_heights: deque[float] = deque(maxlen=3)
    floor_normals: deque[np.ndarray] = deque(maxlen=3)
    ground_height_m: float | None = None
    ground_down_camera: np.ndarray | None = None
    ground_timestamp_ns: int | None = None
    ground_status = "DISABLED" if not floor_enabled else "INITIALIZING"

    print(
        "[navdp-zed] started "
        f"serial={serial} size={width}x{height} fps={args.fps} "
        f"depth={args.depth_mode} max={args.depth_max_m:.1f}m "
        f"tracking={tracking_enabled} floor_hz={args.floor_plane_hz:.1f} "
        f"topic={TOPIC} port={args.port}",
        flush=True,
    )
    try:
        while True:
            if camera.grab(runtime) != sl.ERROR_CODE.SUCCESS:
                time.sleep(0.002)
                continue
            camera.retrieve_image(image, sl.VIEW.LEFT, sl.MEM.CPU)
            camera.retrieve_measure(depth, sl.MEASURE.DEPTH, sl.MEM.CPU)
            bgr = image.get_data(sl.MEM.CPU)[..., :3]
            depth_m = depth.get_data(sl.MEM.CPU).astype(np.float32, copy=False)
            sequence += 1
            timestamp_ns = int(
                camera.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_nanoseconds()
            )
            planar_pose = None
            tracking_state = "DISABLED"
            pose_valid = False
            if tracking_enabled:
                state = camera.get_position(pose, sl.REFERENCE_FRAME.WORLD)
                tracking_state = str(state).split(".")[-1]
                pose_valid = state == sl.POSITIONAL_TRACKING_STATE.OK
                quaternion = pose.get_orientation(sl.Orientation()).get()
                translation = pose.get_translation(sl.Translation()).get()
                planar_pose = zed_pose_to_planar(
                    quaternion,
                    translation,
                    timestamp_ns=timestamp_ns,
                    valid=pose_valid,
                )
            now = time.monotonic()
            if floor_enabled and pose_valid and now >= next_floor_query:
                floor_state = camera.find_floor_plane(
                    floor_plane, floor_to_camera
                )
                ground_status = str(floor_state).split(".")[-1]
                next_floor_query = now + 1.0 / args.floor_plane_hz
                if floor_state == sl.ERROR_CODE.SUCCESS:
                    equation = _zed_vector(
                        floor_plane.get_plane_equation(), 4
                    )
                    if np.all(np.isfinite(equation)):
                        normal = equation[:3]
                        normal_norm = float(np.linalg.norm(normal))
                        equation_height = (
                            abs(float(equation[3])) / normal_norm
                            if normal_norm > 1e-6
                            else float("nan")
                        )
                        closest_height = float(
                            floor_plane.get_closest_distance()
                        )
                        sample_height = (
                            closest_height
                            if closest_height > 0.0
                            else equation_height
                        )
                        if (
                            normal_norm > 1e-6
                            and np.isfinite(sample_height)
                            and 0.1 < sample_height < 3.5
                        ):
                            normal = normal / normal_norm
                            if normal[1] < 0.0:
                                normal = -normal
                            if floor_heights and abs(
                                sample_height - float(np.median(floor_heights))
                            ) > 0.25:
                                floor_heights.clear()
                                floor_normals.clear()
                            floor_heights.append(sample_height)
                            floor_normals.append(normal)
                            ground_height_m = float(np.median(floor_heights))
                            median_normal = np.median(
                                np.stack(tuple(floor_normals), axis=0), axis=0
                            )
                            median_normal_norm = float(
                                np.linalg.norm(median_normal)
                            )
                            if median_normal_norm > 1e-6:
                                median_normal /= median_normal_norm
                                ground_down_camera = median_normal
                                ground_timestamp_ns = timestamp_ns
                                ground_status = "OK"
            ground_valid = (
                ground_height_m is not None
                and ground_down_camera is not None
                and ground_timestamp_ns is not None
                and (timestamp_ns - ground_timestamp_ns) * 1e-9
                <= args.floor_plane_max_age_s
            )
            published_ground_status = "OK" if ground_valid else ground_status
            message = {
                "format": FORMAT,
                "timestamp_ns": timestamp_ns,
                "sequence": sequence,
                "color_order": "bgr",
                "rgb": pack_array(bgr),
                "depth_m": pack_array(depth_m),
                "pose_valid": pose_valid,
                "tracking_state": tracking_state,
                "ground_plane_valid": ground_valid,
                "ground_plane_status": published_ground_status,
            }
            if planar_pose is not None:
                message["pose_xy_yaw"] = [
                    planar_pose.x_m,
                    planar_pose.y_m,
                    planar_pose.yaw_rad,
                ]
            if ground_valid:
                message["ground_camera_height_m"] = ground_height_m
                message["ground_down_camera_xyz"] = (
                    ground_down_camera.tolist()
                )
                message["ground_plane_timestamp_ns"] = ground_timestamp_ns
            publisher.publish(
                TOPIC,
                message,
            )

            if args.log_hz > 0 and now >= next_log:
                valid = np.isfinite(depth_m) & (depth_m > 0)
                valid_fraction = float(np.mean(valid))
                effective_hz = sequence / max(now - started, 1e-6)
                print(
                    "[navdp-zed] "
                    f"seq={sequence} effective={effective_hz:.1f}Hz "
                    f"valid_depth={valid_fraction:.1%} tracking={tracking_state}",
                    f"ground={published_ground_status} "
                    f"height={ground_height_m if ground_valid else None}",
                    flush=True,
                )
                next_log = now + 1.0 / args.log_hz
    except KeyboardInterrupt:
        print("[navdp-zed] stopping", flush=True)
    finally:
        if tracking_enabled:
            camera.disable_positional_tracking()
        camera.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
