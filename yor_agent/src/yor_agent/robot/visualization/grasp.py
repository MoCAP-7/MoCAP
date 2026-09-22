"""Operator-facing Viser scene for YOR grasp-backend diagnostics."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..geometry import matrix_to_quaternion_wxyz


def _point_cloud_in_arm_frame(
    depth_m: np.ndarray,
    rgb: np.ndarray,
    intrinsics: np.ndarray,
    arm_from_camera: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    max_points: int = 100_000,
) -> tuple[np.ndarray, np.ndarray]:
    """Back-project an RGB-D image and express finite samples in arm base."""

    depth = np.asarray(depth_m, dtype=np.float64)
    color = np.asarray(rgb, dtype=np.uint8)
    if depth.shape != color.shape[:2]:
        raise ValueError("depth and RGB shapes do not match")
    valid = np.isfinite(depth) & (depth > 0.08) & (depth < 2.5)
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool)
    rows, columns = np.nonzero(valid)
    if rows.size == 0:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8)
    if rows.size > max_points:
        # Deterministic uniform subsampling keeps repeated captures comparable.
        keep = np.linspace(0, rows.size - 1, max_points, dtype=np.int64)
        rows = rows[keep]
        columns = columns[keep]
    z = depth[rows, columns]
    x = (columns - intrinsics[0, 2]) * z / intrinsics[0, 0]
    y = (rows - intrinsics[1, 2]) * z / intrinsics[1, 1]
    points_camera = np.column_stack([x, y, z])
    points_arm = (
        points_camera @ arm_from_camera[:3, :3].T
        + arm_from_camera[:3, 3]
    )
    return points_arm.astype(np.float32), color[rows, columns]


def _candidate_status(candidate: dict[str, Any]) -> str:
    """Return the operator-facing IK state for one grasp candidate."""

    plan = candidate.get("ik_plan")
    if plan is None:
        return "not_evaluated"
    if not isinstance(plan, dict) or not plan.get("success", False):
        return "failed"
    if bool(plan.get("ik_converged", False)):
        return "converged"
    if bool(plan.get("ik_quality_acceptable", False)):
        return "acceptable"
    return "best_effort"


def _selected_candidate(debug: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    """Return the one candidate selected by grasp/IK ranking."""

    selected = [
        (rank, candidate)
        for rank, candidate in enumerate(debug.get("candidates", []), start=1)
        if bool(candidate.get("selected", False))
    ]
    if len(selected) != 1:
        raise RuntimeError(
            "grasp visualization requires exactly one selected candidate; "
            f"got {len(selected)}"
        )
    return selected[0]


def _add_parallel_jaw_goal_icon(
    scene: Any,
    *,
    name: str,
    transform: np.ndarray,
    status_color: np.ndarray,
) -> None:
    """Draw one bold pose gizmo with a schematic Nero parallel-jaw gripper."""

    transform = np.asarray(transform, dtype=np.float64)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError("selected grasp transform must be a finite 4x4 matrix")
    scene.add_frame(
        name=name,
        position=transform[:3, 3],
        wxyz=matrix_to_quaternion_wxyz(transform[:3, :3]),
        axes_length=0.12,
        axes_radius=0.006,
        origin_radius=0.011,
        origin_color=status_color,
    )
    scene.add_icosphere(
        name=f"{name}/goal_center",
        radius=0.015,
        color=status_color,
        opacity=0.35,
        material="toon3",
        cast_shadow=False,
        receive_shadow=False,
    )

    # The icon is deliberately schematic rather than a collision mesh. In the
    # Nero TCP frame the two fingers are separated along local Y (closing), and
    # extend toward the jaw-center target along local +Z (approach).
    body_color = np.asarray([235, 235, 242], dtype=np.uint8)
    accent_color = np.asarray([255, 70, 230], dtype=np.uint8)
    pieces = (
        ("palm", body_color, [0.040, 0.100, 0.024], [0.0, 0.0, -0.082]),
        (
            "finger_positive_y",
            body_color,
            [0.028, 0.016, 0.076],
            [0.0, 0.043, -0.032],
        ),
        (
            "finger_negative_y",
            body_color,
            [0.028, 0.016, 0.076],
            [0.0, -0.043, -0.032],
        ),
        (
            "tip_positive_y",
            accent_color,
            [0.032, 0.019, 0.018],
            [0.0, 0.043, 0.010],
        ),
        (
            "tip_negative_y",
            accent_color,
            [0.032, 0.019, 0.018],
            [0.0, -0.043, 0.010],
        ),
    )
    for piece_name, color, dimensions, position in pieces:
        scene.add_box(
            name=f"{name}/parallel_jaw_icon/{piece_name}",
            color=color,
            dimensions=np.asarray(dimensions, dtype=np.float64),
            position=np.asarray(position, dtype=np.float64),
            opacity=0.72,
            material="toon3",
            cast_shadow=False,
            receive_shadow=False,
        )


class GraspViser:
    """Serve static grasp scenes to a browser on the operator's computer."""

    def __init__(self, *, host: str = "0.0.0.0", port: int = 8080) -> None:
        try:
            import viser
        except ImportError as exc:
            raise RuntimeError(
                "Viser is not installed in the Jetson test environment. Install "
                "it with: python -m pip install viser"
            ) from exc
        self._server = viser.ViserServer(host=host, port=int(port))
        self.host = host
        get_port = getattr(self._server, "get_port", None)
        self.port = int(get_port()) if callable(get_port) else int(port)
        self._server.scene.set_up_direction("+z")
        self._summary_handles: dict[str, Any] = {}
        self._server.gui.add_markdown(
            """
## YOR selected grasp goal

- Axes: **X red**, **Y green**, **Z blue**
- The bold RGB gizmo is the final **commanded Nero TCP**
- The smaller, thinner RGB frame is the measured **initial Nero TCP**
- The translucent U-shaped icon shows the parallel-jaw gripper orientation
- Nero **+Y is finger closing** and **+Z is approach**
- Goal-center color reports IK quality:
  🟢 strict convergence, 🔵 within configured grasp residuals,
  🟡 finite best-effort outside those residuals, 🔴 IK failure,
  ⚪ IK not evaluated
- Only the final selected grasp is rendered; rejected candidates remain in
  diagnostic data but do not clutter the scene

The scene is expressed in the selected Nero **arm-base frame**. The gripper
icon is an orientation glyph, not collision geometry or a swept-path preview.
"""
        )

    def publish(
        self,
        debug: dict[str, Any],
        *,
        start_position: np.ndarray,
        start_quaternion_wxyz: np.ndarray,
        endpoint_name: str,
        object_name: str,
    ) -> None:
        arm = str(debug["arm"])
        backend_name = str(debug.get("grasp_backend", "unknown")).lower()
        root = f"/{arm}"
        self._server.scene.remove_by_name(root)
        rgb = np.asarray(debug["rgb"], dtype=np.uint8)
        depth = np.asarray(debug["depth_m"], dtype=np.float64)
        mask = np.asarray(debug["mask"], dtype=bool)
        intrinsics = np.asarray(debug["intrinsics"], dtype=np.float64)
        arm_from_camera = np.asarray(debug["arm_from_camera"], dtype=np.float64)

        points, colors = _point_cloud_in_arm_frame(
            depth,
            rgb,
            intrinsics,
            arm_from_camera,
        )
        self._server.scene.add_point_cloud(
            name=f"{root}/zed_rgbd",
            points=points,
            colors=colors,
            point_size=0.002,
            point_shape="circle",
        )
        object_points, _ = _point_cloud_in_arm_frame(
            depth,
            rgb,
            intrinsics,
            arm_from_camera,
            mask=mask,
            max_points=30_000,
        )
        self._server.scene.add_point_cloud(
            name=f"{root}/object_mask",
            points=object_points,
            colors=np.tile(
                np.asarray([[255, 80, 255]], dtype=np.uint8),
                (object_points.shape[0], 1),
            ),
            point_size=0.004,
            point_shape="circle",
        )

        camera_quaternion = matrix_to_quaternion_wxyz(arm_from_camera[:3, :3])
        vertical_fov = 2.0 * math.atan2(
            rgb.shape[0], 2.0 * float(intrinsics[1, 1])
        )
        self._server.scene.add_camera_frustum(
            name=f"{root}/zed_camera",
            position=arm_from_camera[:3, 3],
            wxyz=camera_quaternion,
            fov=vertical_fov,
            aspect=float(rgb.shape[1]) / float(rgb.shape[0]),
            scale=0.08,
            image=rgb,
        )
        self._server.scene.add_frame(
            name=f"{root}/arm_base",
            position=np.zeros(3),
            wxyz=np.asarray([1.0, 0.0, 0.0, 0.0]),
            axes_length=0.12,
            axes_radius=0.004,
        )
        self._server.scene.add_frame(
            name=f"{root}/initial_{endpoint_name}_pose",
            position=np.asarray(start_position, dtype=np.float64),
            wxyz=np.asarray(start_quaternion_wxyz, dtype=np.float64),
            axes_length=0.075,
            axes_radius=0.0025,
            origin_radius=0.006,
            origin_color=np.asarray([145, 145, 145], dtype=np.uint8),
        )

        marker_colors = {
            "converged": np.asarray([30, 220, 80], dtype=np.uint8),
            "acceptable": np.asarray([50, 130, 255], dtype=np.uint8),
            "best_effort": np.asarray([255, 190, 20], dtype=np.uint8),
            "failed": np.asarray([235, 45, 45], dtype=np.uint8),
            "not_evaluated": np.asarray([180, 180, 180], dtype=np.uint8),
        }
        rank, selected_candidate = _selected_candidate(debug)
        transform = np.asarray(
            selected_candidate["arm_from_grasp"], dtype=np.float64
        )
        status = _candidate_status(selected_candidate)
        _add_parallel_jaw_goal_icon(
            self._server.scene,
            name=f"{root}/grasp_goal",
            transform=transform,
            status_color=marker_colors[status],
        )

        plan = selected_candidate.get("ik_plan")
        position_error = None if not isinstance(plan, dict) else plan.get(
            "ik_position_error_m"
        )
        rotation_error = None if not isinstance(plan, dict) else plan.get(
            "ik_rotation_error_rad"
        )
        joint_travel = None if not isinstance(plan, dict) else plan.get(
            "joint_travel_l2_rad"
        )
        state_text = {
            "converged": "🟢 strictly converged",
            "acceptable": "🔵 acceptable residual",
            "best_effort": "🟡 best effort",
            "failed": "🔴 IK failed",
            "not_evaluated": "⚪ IK not evaluated",
        }[status]
        residual_text = "IK residual unavailable"
        if position_error is not None and rotation_error is not None:
            residual_text = (
                f"{float(position_error) * 1000.0:.1f} mm / "
                f"{math.degrees(float(rotation_error)):.1f}°"
            )
        wrist_text = (
            "180° symmetry branch"
            if bool(selected_candidate.get("symmetry_flipped", False))
            else "canonical orientation"
        )
        previous_summary = self._summary_handles.pop(arm, None)
        if previous_summary is not None:
            previous_summary.remove()
        summary_lines = [
            f"### {arm.title()} grasp goal: `{object_name}`",
            "",
            f"- Backend: `{backend_name}`",
            f"- Selected rank: **#{rank}** of "
            f"{len(debug.get('candidates', []))} evaluated candidates",
            f"- IK: {state_text}",
            f"- Residual: **{residual_text}**",
            f"- Backend score: **{float(selected_candidate['score']):.3f}**",
            f"- Wrist: **{wrist_text}**",
            f"- Endpoint: `{endpoint_name}`",
        ]
        if joint_travel is not None:
            summary_lines.append(
                f"- Joint travel L2: **{float(joint_travel):.2f} rad**"
            )
        summary_lines.extend(
            [
                "",
                "Only this selected goal is rendered. The U-shaped glyph is "
                "schematic: **+Y closes the fingers, +Z is approach**.",
            ]
        )
        self._summary_handles[arm] = self._server.gui.add_markdown(
            "\n".join(summary_lines)
        )

        selected_position = transform[:3, 3]
        if np.all(np.isfinite(selected_position)):
            self._server.initial_camera.look_at = selected_position
            self._server.initial_camera.position = (
                selected_position + np.asarray([0.55, -0.55, 0.35])
            )
            # initial_camera affects clients that connect after publication.
            # Operators commonly open the URL while inference is still running,
            # so also focus every already-connected browser on the new result.
            get_clients = getattr(self._server, "get_clients", None)
            client_map = get_clients() if callable(get_clients) else {}
            clients = client_map.values() if isinstance(client_map, dict) else []
            for client in clients:
                try:
                    client.camera.up_direction = np.asarray([0.0, 0.0, 1.0])
                    client.camera.look_at = selected_position
                    client.camera.position = selected_position + np.asarray(
                        [0.55, -0.55, 0.35]
                    )
                except Exception:
                    # A browser can disconnect between get_clients() and the
                    # camera update; the shared scene remains valid.
                    pass

    def close(self) -> None:
        stop = getattr(self._server, "stop", None)
        if callable(stop):
            stop()
