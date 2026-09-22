"""Viser scene for virtual base-pose and grasp reachability debugging."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..geometry import matrix_to_quaternion_wxyz
from .grasp import _add_parallel_jaw_goal_icon
from .navigation import _point_cloud_in_navigation_frame


class ManipulationReadinessViser:
    """Render SAM3 geometry, virtual bases, IK status, and the chosen grasp."""

    def __init__(self, *, host: str = "0.0.0.0", port: int = 8080) -> None:
        try:
            import viser
        except ImportError as exc:
            raise RuntimeError(
                "Viser is not installed. Install yor-agent[visualization]."
            ) from exc
        self._server = viser.ViserServer(host=host, port=int(port))
        self.host = host
        get_port = getattr(self._server, "get_port", None)
        self.port = int(get_port()) if callable(get_port) else int(port)
        self._server.scene.set_up_direction("+z")
        self._summary_handle: Any | None = None
        self._server.gui.add_markdown(
            """
## YOR manipulation-readiness search

- Axes: **+X forward**, **+Y left**, **+Z up**
- Magenta points: the SAM3 object mask used by GraspGen-X
- During depth validation, magenta pixels have no usable metric depth and
  green pixels are real ZED depth samples inside the SAM3 mask
- Green base: selected strictly manipulation-ready candidate
- Amber base: selected best-effort finite-IK candidate
- Cyan bases: strictly IK-feasible alternatives
- Yellow bases: finite-IK best-effort alternatives
- Gray bases: evaluated without a finite IK result
- U-shaped glyph: virtual grasp used only to choose the base pose

The candidates are virtual; only the final selected base pose is executed.
Viser is diagnostic-only and never sends robot commands.
"""
        )

    @staticmethod
    def _base_transform(candidate: dict[str, Any]) -> np.ndarray:
        yaw = float(candidate["yaw_rad"])
        cosine, sine = math.cos(yaw), math.sin(yaw)
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = np.asarray(
            [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]]
        )
        transform[:3, 3] = [
            float(candidate["forward_m"]),
            float(candidate["left_m"]),
            0.0,
        ]
        return transform

    def _replace_summary(self, lines: list[str]) -> None:
        previous = self._summary_handle
        if previous is not None:
            previous.remove()
        self._summary_handle = self._server.gui.add_markdown("\n".join(lines))

    def _publish_sam3_depth(self, debug: dict[str, Any]) -> None:
        root = "/manipulation_readiness"
        rgb = np.asarray(debug["rgb"], dtype=np.uint8).copy()
        depth = np.asarray(debug["depth_m"], dtype=np.float64)
        mask = np.asarray(debug["mask"], dtype=bool)
        intrinsics = np.asarray(debug["intrinsics"], dtype=np.float64)
        valid = mask & np.isfinite(depth) & (depth > 0.08) & (depth < 2.5)
        missing = mask & ~valid
        if np.any(missing):
            rgb[missing] = (
                0.25 * rgb[missing]
                + 0.75 * np.asarray([255, 70, 230], dtype=np.float64)
            ).astype(np.uint8)
        if np.any(valid):
            rgb[valid] = np.asarray([35, 255, 85], dtype=np.uint8)
        vertical_fov = 2.0 * math.atan2(
            rgb.shape[0], 2.0 * float(intrinsics[1, 1])
        )
        self._server.scene.add_camera_frustum(
            name=f"{root}/sam3_depth_camera",
            position=np.zeros(3),
            wxyz=np.asarray([1.0, 0.0, 0.0, 0.0]),
            fov=vertical_fov,
            aspect=float(rgb.shape[1]) / float(rgb.shape[0]),
            scale=0.35,
            image=rgb,
        )
        score = debug.get("sam_score")
        score_text = "n/a" if score is None else f"{float(score):.3f}"
        phase = str(debug.get("phase", "sam3_depth_validation"))
        if phase == "failed":
            phase = f"failed after {debug.get('failed_after_phase', 'unknown')}"
        summary = [
            f"### `{debug['object_name']}` with {str(debug['arm']).title()} arm",
            "",
            f"- Phase: **{phase}**",
            f"- SAM3 score: **{score_text}**",
            "- SAM3/depth attempt: "
            f"**{int(debug.get('attempt_index', 0)) + 1}/"
            f"{int(debug.get('maximum_attempts', 1))}**",
            f"- Mask pixels: **{int(debug.get('mask_pixels', np.count_nonzero(mask)))}**",
            "- Real metric depth: "
            f"**{int(debug.get('valid_depth_points', np.count_nonzero(valid)))}/"
            f"{int(debug.get('minimum_valid_depth_points', 0))} required**",
            f"- Depth accepted: **{bool(debug.get('accepted', False))}**",
        ]
        if debug.get("segmentation_error"):
            summary.append(f"- SAM3 error: **{debug['segmentation_error']}**")
        if debug.get("grasp_backend"):
            summary.append(f"- Grasp backend: **{debug['grasp_backend']}**")
        if debug.get("generated_grasp_count") is not None:
            summary.append(
                "- Target-filtered grasp candidates: "
                f"**{int(debug['generated_grasp_count'])}**"
            )
        if debug.get("failure_reason"):
            summary.append(f"- Failure: **{debug['failure_reason']}**")
        self._replace_summary(summary)
        self._server.initial_camera.look_at = np.zeros(3)
        self._server.initial_camera.position = np.asarray([-0.7, -0.7, 0.45])

    def _publish_failure_only(self, debug: dict[str, Any]) -> None:
        self._replace_summary(
            [
                f"### `{debug.get('object_name', 'unknown')}` with "
                f"{str(debug.get('arm', 'unknown')).title()} arm",
                "",
                "- Phase: **failed**",
                f"- Failed after: **{debug.get('failed_after_phase', 'unknown')}**",
                f"- Failure: **{debug.get('failure_reason', 'unknown')}**",
            ]
        )

    def publish(self, debug: dict[str, Any]) -> None:
        root = "/manipulation_readiness"
        self._server.scene.remove_by_name(root)
        phase = str(debug.get("phase", ""))
        source_phase = (
            str(debug.get("failed_after_phase", ""))
            if phase == "failed"
            else phase
        )
        if phase == "target_too_far":
            self._replace_summary(
                [
                    f"### `{debug.get('object_name', 'unknown')}` with "
                    f"{str(debug.get('arm', 'unknown')).title()} arm",
                    "",
                    "- Phase: **target too far for local manipulation**",
                    "- Measured camera distance: "
                    f"**{float(debug.get('target_distance_m', math.nan)):.3f} m**",
                    "- Local-preparation limit: "
                    f"**{float(debug.get('maximum_distance_m', math.nan)):.3f} m**",
                    "- Recovery: **move closer or call dock_to_visible_object, then retry**",
                ]
            )
            return
        if source_phase in {
            "sam3_depth_validation",
            "grasp_generation",
            "batched_ik_evaluation",
        }:
            self._publish_sam3_depth(debug)
            return
        if phase == "failed" and "selected_candidate_index" not in debug:
            self._publish_failure_only(debug)
            return
        rgb = np.asarray(debug["rgb"], dtype=np.uint8)
        depth = np.asarray(debug["depth_m"], dtype=np.float64)
        mask = np.asarray(debug["mask"], dtype=bool)
        intrinsics = np.asarray(debug["intrinsics"], dtype=np.float64)
        ground_height = float(debug["ground_camera_height_m"])
        ground_down = np.asarray(debug["ground_down_camera_xyz"], dtype=np.float64)
        navigation_from_camera = np.asarray(
            debug["navigation_from_camera"], dtype=np.float64
        )
        points, colors = _point_cloud_in_navigation_frame(
            depth,
            rgb,
            intrinsics,
            max_depth_m=2.5,
            ground_camera_height_m=ground_height,
            ground_down_camera_xyz=ground_down,
        )
        self._server.scene.add_point_cloud(
            name=f"{root}/zed_rgbd",
            points=points,
            colors=colors,
            point_size=0.004,
            point_shape="circle",
        )
        target_points, _ = _point_cloud_in_navigation_frame(
            depth,
            rgb,
            intrinsics,
            mask=mask,
            max_points=30000,
            max_depth_m=2.5,
            ground_camera_height_m=ground_height,
            ground_down_camera_xyz=ground_down,
        )
        self._server.scene.add_point_cloud(
            name=f"{root}/sam3_target",
            points=target_points,
            colors=np.tile(
                np.asarray([[255, 70, 230]], dtype=np.uint8),
                (target_points.shape[0], 1),
            ),
            point_size=0.009,
            point_shape="circle",
        )
        self._server.scene.add_grid(
            name=f"{root}/floor",
            width=2.0,
            height=2.0,
            cell_size=0.05,
            section_size=0.25,
            plane="xy",
            position=np.asarray([0.45, 0.0, 0.0]),
        )
        self._server.scene.add_frame(
            name=f"{root}/current_base",
            position=np.zeros(3),
            wxyz=np.asarray([1.0, 0.0, 0.0, 0.0]),
            axes_length=0.14,
            axes_radius=0.004,
        )
        vertical_fov = 2.0 * math.atan2(
            rgb.shape[0], 2.0 * float(intrinsics[1, 1])
        )
        self._server.scene.add_camera_frustum(
            name=f"{root}/zed_camera",
            position=np.asarray([0.0, 0.0, ground_height]),
            wxyz=matrix_to_quaternion_wxyz(navigation_from_camera),
            fov=vertical_fov,
            aspect=float(rgb.shape[1]) / float(rgb.shape[0]),
            scale=0.08,
            image=rgb,
        )

        selected_index = int(debug["selected_candidate_index"])
        candidates = list(debug.get("candidates", []))
        selected_candidate: dict[str, Any] | None = None
        for candidate in candidates:
            index = int(candidate["candidate_index"])
            selected = index == selected_index
            strict_ready = bool(candidate.get("strict_ready", False))
            strict_pi = candidate.get("selection_quality") == "strict_pi"
            finite = int(candidate.get("finite_grasp_count", 0)) > 0
            color = (
                np.asarray([35, 225, 85], dtype=np.uint8)
                if selected and strict_pi
                else np.asarray([245, 165, 35], dtype=np.uint8)
                if selected
                else np.asarray([45, 195, 235], dtype=np.uint8)
                if strict_ready
                else np.asarray([235, 205, 65], dtype=np.uint8)
                if finite
                else np.asarray([145, 145, 155], dtype=np.uint8)
            )
            transform = self._base_transform(candidate)
            self._server.scene.add_frame(
                name=f"{root}/candidates/{index}",
                position=transform[:3, 3],
                wxyz=matrix_to_quaternion_wxyz(transform[:3, :3]),
                axes_length=0.08 if selected else 0.05,
                axes_radius=0.004 if selected else 0.002,
                origin_radius=0.012 if selected else 0.007,
                origin_color=color,
            )
            final = transform[:2, 3]
            line = np.asarray(
                [
                    [[0.0, 0.0, 0.04], [*final, 0.04]],
                ],
                dtype=np.float64,
            )
            self._server.scene.add_line_segments(
                name=f"{root}/candidate_paths/{index}",
                points=line,
                colors=color,
                line_width=4.0 if selected else 1.0,
            )
            if selected:
                selected_candidate = candidate

        if selected_candidate is not None:
            selected_grasp = selected_candidate.get("selected_grasp")
            if isinstance(selected_grasp, dict):
                arm_target = np.asarray(
                    selected_grasp["arm_target"], dtype=np.float64
                )
                arm_from_camera = np.asarray(
                    debug["arm_from_camera"], dtype=np.float64
                )
                navigation_from_camera_transform = np.eye(4, dtype=np.float64)
                navigation_from_camera_transform[:3, :3] = navigation_from_camera
                navigation_from_camera_transform[:3, 3] = [0.0, 0.0, ground_height]
                navigation_from_arm = (
                    navigation_from_camera_transform @ np.linalg.inv(arm_from_camera)
                )
                current_navigation_from_grasp = (
                    self._base_transform(selected_candidate)
                    @ navigation_from_arm
                    @ arm_target
                )
                _add_parallel_jaw_goal_icon(
                    self._server.scene,
                    name=f"{root}/selected_grasp",
                    transform=current_navigation_from_grasp,
                    status_color=(
                        np.asarray([35, 225, 85], dtype=np.uint8)
                        if selected_candidate.get("selection_quality")
                        == "strict_pi"
                        else np.asarray([245, 165, 35], dtype=np.uint8)
                    ),
                )

        selected = selected_candidate or {}
        motion = debug.get("motion")
        summary = [
            f"### `{debug['object_name']}` with {str(debug['arm']).title()} arm",
            "",
            f"- Phase: **{debug['phase']}**",
            f"- SAM3 score: **{float(debug['sam_score']):.3f}**",
            f"- Batched IK queries: **{int(debug['ik_query_count'])}**",
            "- Pi-bound candidates rejected before batching: "
            f"**{int(debug.get('pi_bound_rejected_count', 0))}**",
            "- Collision-safe grasp pool: "
            f"**{int(debug.get('collision_safe_grasp_count', 0))}**",
            "- Base poses with a strict Pi grasp: "
            f"**{int(debug.get('strict_pi_candidate_count', 0))}**",
            f"- SAM3/depth attempts: **{len(debug.get('sam3_depth_attempts', [])) or 1}**",
            f"- Selected local pose: **{float(selected.get('forward_m', 0.0)):+.3f} m forward, "
            f"{float(selected.get('left_m', 0.0)):+.3f} m left, "
            f"{math.degrees(float(selected.get('yaw_rad', 0.0))):+.1f}°**",
            f"- Feasible grasps there: **{int(selected.get('feasible_grasp_count', 0))}**",
            f"- Selection quality: **{selected.get('selection_quality', 'unknown')}**",
            "- Base motion: **one absolute SE(2) target**",
            "- Next step: **call sample_grasp_pose to consume the certified goalset, "
            "then goto_grasp_pose**",
        ]
        if isinstance(motion, dict):
            summary.append(
                f"- Holonomic motion: **{motion.get('reason', 'unknown')}**"
            )
        if debug.get("failure_reason"):
            summary.append(f"- Failure: **{debug['failure_reason']}**")
        self._replace_summary(summary)

        look_at = np.asarray(
            [
                float(selected.get("forward_m", 0.25)),
                float(selected.get("left_m", 0.0)),
                0.25,
            ]
        )
        camera_position = look_at + np.asarray([-0.8, -0.9, 0.65])
        self._server.initial_camera.look_at = look_at
        self._server.initial_camera.position = camera_position

    def close(self) -> None:
        stop = getattr(self._server, "stop", None)
        if callable(stop):
            stop()
