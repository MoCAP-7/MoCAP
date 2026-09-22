"""Cap-X's R1Pro mobile-manipulation API on the physical YOR robot.

The function names, signatures, docstrings, constants and failure semantics
are those of ``capx.integrations.r1pro.control.R1ProControlApi``, so a
generated program sees Cap-X's own vocabulary: a failed call prints and
returns ``False``/``None`` instead of raising, no clearance gate refuses a
motion, and no advice text is added. The world frame is the ZED odometry
frame of the base: x forward and y left at tracking start, z up, origin on the
floor under the camera at that moment. ``get_robot_position`` reports the
camera's floor projection in that frame. ``navigate_to_pose`` turns toward
the goal, drives straight to it and turns to the goal heading with the base
controller and no obstacle check, like Cap-X's simulator navigation without
its map. Arm targets are converted into the selected Nero arm-base frame with
the calibrated arm-from-camera transform and the ZED floor plane.

R1Pro calls without a counterpart on YOR are not provided: ``reset_torso``
and ``find_object_torso_rotate`` (no torso), ``point_prompt_molmo`` (no
Molmo service) and ``write_video`` (the episode recorder already keeps the
whole video).
"""

from __future__ import annotations

import base64
import math
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .bootstrap import configure_import_paths

configure_import_paths()

from capx.integrations.base_api import ApiBase  # noqa: E402
from yor_agent.primitive_config import primitive_settings  # noqa: E402
from yor_agent.robot.geometry import (  # noqa: E402
    matrix_to_quaternion_wxyz,
    normalize_quaternion_wxyz,
    pose_matrix,
    quaternion_wxyz_to_matrix,
    quaternion_wxyz_to_rpy,
    rpy_to_quaternion_wxyz,
)
from yor_agent.robot.perception import init_contact_graspnet, init_sam3  # noqa: E402
from yor_agent.robot.perception.sam3_client import (  # noqa: E402
    DEFAULT_SERVICE_URL,
    _encode_image,
    _post_json,
)


#: Cap-X's R1Pro functions, in the order ``R1ProControlApi.functions`` lists
#: them, without the four that have no counterpart on YOR.
R1PRO_FUNCTIONS = (
    "segment_sam3_text_prompt",
    "segment_sam3_point_prompt",
    "navigate_to_pose",
    "open_gripper",
    "close_gripper",
    "move_hand",
    "get_robot_position",
    "get_robot_relative_eef_pose",
    "move_to_joint_positions",
    "get_current_eef_pose",
    "get_current_joint_positions",
    "solve_ik",
    "lift_arm",
    "check_object_in_hand",
    "get_env_observation",
    "get_sam3_mask",
    "get_object_pose",
    "find_object_base_rotate",
    "get_navigation_pose",
    "save_current_observation",
    "grasp_object",
    "sample_grasp_pose",
)
OMITTED_R1PRO_FUNCTIONS = (
    "point_prompt_molmo",
    "reset_torso",
    "find_object_torso_rotate",
    "write_video",
)

# Cap-X's own constants (capx/integrations/r1pro/control.py and utils.py).
SAM3_SCORE_THRESHOLD = 0.1
NAVIGATION_WAYPOINTS = 5
BASE_ROTATE_STEP_RAD = 0.5
PREGRASP_BACKOFF_M = 0.1
SIMPLE_PREGRASP_OFFSET_M = 0.25
GRASP_POSITION_TOLERANCE_M = 0.03
TABLE_EDGE_BUFFER_M = 0.3
OUTLIER_NEIGHBORS = 20
OUTLIER_STD_RATIO = 2.0
#: Cap-X's top-down grasp orientation (xyzw): the tool z axis points down and
#: its x axis along world x, as on R1Pro.
TOP_DOWN_QUATERNION_XYZW = np.asarray([1.0, 0.0, 0.0, 0.0])

# YOR counterparts of R1Pro's joint-space lift and the simulator's in-hand
# check, and the joint streaming granularity of the Pi arm service.
LIFT_HEIGHT_M = 0.10
HELD_OBJECT_MIN_WIDTH_M = 0.01
JOINT_WAYPOINT_STEP_RAD = 0.05
MAX_TRAJECTORY_WAYPOINTS = 256
#: A navigation goal closer than this is reached without moving the base.
NAVIGATION_POSITION_TOLERANCE_M = 0.05
NAVIGATION_YAW_TOLERANCE_RAD = math.radians(3.0)


@dataclass(frozen=True)
class OrientedBoundingBox:
    """An object-aligned box like Open3D's ``OrientedBoundingBox``.

    ``center`` is its centre, the columns of ``R`` are its axes and
    ``extent`` holds the full side lengths, all in the world frame.
    """

    center: np.ndarray
    R: np.ndarray
    extent: np.ndarray


class CapXYorR1ProApi(ApiBase):
    """Cap-X's R1Pro control API backed by YOR's ZED, SAM3, Contact-GraspNet,
    the base controller (obstacle check off) and the Pi arm service."""

    def __init__(
        self,
        env: Any,
        *,
        segment_client_factory: Callable[[], Callable[..., Any]] | None = None,
        point_segment_client_factory: Callable[[], Callable[..., Any]] | None = None,
        grasp_client_factory: Callable[[], Callable[..., Any]] | None = None,
        sam3_score_threshold: float = SAM3_SCORE_THRESHOLD,
        arm_timeout_s: float = 60.0,
    ) -> None:
        super().__init__(env)
        # The defaults are looked up at call time so a test can replace them.
        self._segment = (segment_client_factory or init_sam3)()
        self._segment_point = (point_segment_client_factory or init_sam3_point_prompt)()
        self._plan_grasps = (grasp_client_factory or init_contact_graspnet)()
        self.sam3_score_threshold = float(sam3_score_threshold)
        self._arm_timeout_s = float(arm_timeout_s)
        #: Whether each gripper's last command closed it; an open gripper
        #: holds nothing, whatever width it reads.
        self._gripper_closed: dict[str, bool] = {}
        docking = primitive_settings(env.primitive_config, "dock_to_visible_object") or {}
        self._fallback_ground_plane = _fallback_ground_plane(docking)
        # Every arm pose here is the TCP; a flange-controlled arm service
        # would offset all of them by the flange-to-TCP transform.
        try:
            status = env.arm_status()
        except Exception:  # noqa: BLE001 - arms may be absent at construction
            status = None
        frame = status.get("end_effector_frame") if isinstance(status, dict) else None
        if frame not in (None, "tcp"):
            raise RuntimeError(
                f"the arm service controls the {frame!r} frame; this API expects the TCP"
            )

    def functions(self) -> dict[str, Callable[..., Any]]:
        return {name: getattr(self, name) for name in R1PRO_FUNCTIONS}

    # ------------------------------------------------------------------
    # Perception
    # ------------------------------------------------------------------

    def segment_sam3_text_prompt(
        self,
        rgb: np.ndarray,
        text_prompt: str,
    ) -> list[dict[str, Any]]:
        """Run SAM3 segmentation on an RGB image conditioned on a text prompt.

        Args:
            rgb:
                RGB image array of shape (H, W, 3), dtype uint8.
            text_prompt:
                Text prompt for SAM3 segmentation.

        Returns:
            masks:
                A list of dictionaries. Each dict may contain:

                  - "mask":  np.ndarray of shape (H, W), dtype bool or uint8,
                              where True/1 means the pixel belongs to the instance.
                  - "box": list [x1, y1, x2, y2] in pixel coordinates.
                  - "score": float confidence score.

        Example:
            >>> rgb, depth = get_env_observation()
            >>> masks = segment_sam3_text_prompt(rgb, text_prompt="red mug")
        """
        return self._segment(np.asarray(rgb, dtype=np.uint8), text_prompt=str(text_prompt))

    def segment_sam3_point_prompt(
        self,
        rgb: np.ndarray,
        point_coords: tuple[float, float],
    ) -> list[dict[str, Any]]:
        """Run SAM3 segmentation on an RGB image, optionally conditioned on an image coordinate point prompt.

        Args:
            rgb:
                RGB image array of shape (H, W, 3), dtype uint8.
            point_coords:
                (x, y) pixel coordinates of the point prompt.

        Returns:
            masks:
                A list of dictionaries. Each dict may contain:

                  - "mask":  np.ndarray of shape (H, W), dtype bool or uint8,
                              where True/1 means the pixel belongs to the instance.
                  - "score": float confidence score.

        Example:
            >>> rgb, depth = get_env_observation()
            >>> masks = segment_sam3_point_prompt(rgb, (100, 100))
        """
        return self._segment_point(np.asarray(rgb, dtype=np.uint8), point_coords)

    def get_env_observation(self) -> tuple[np.ndarray, np.ndarray]:
        """Get the observation of the environment.
        Args:
            None
        Returns:
            rgb: RGB image of the environment in np.ndarray format. Shape: (H, W, 3), dtype uint8.
            depth: Depth image of the environment in np.ndarray format. Shape: (H, W), dtype float32.
        """
        rgb, depth, _, _ = self._camera_view(require_pose=False)
        return rgb, depth

    def get_sam3_mask(self, object_name: str) -> np.ndarray:
        """Get the mask of an object in the environment from a natural language description and current camera view.
        Args:
            object_name: The name of the object to get the mask of.
        Returns:
            mask sum: The sum of the mask of the object in the environment, indicating the number of pixels in the mask.
        """
        rgb, _, _, _ = self._camera_view(require_pose=False)
        results = self._segment(rgb, text_prompt=object_name)
        if len(results) == 0:
            raise ValueError("No sam3 detections")
        scores = [result["score"] for result in results]
        mask = np.asarray(results[int(np.argmax(scores))]["mask"])
        return mask.sum()

    def get_object_pose(
        self, object_name: str, return_bbox_extent: bool = False
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray, OrientedBoundingBox]:
        """Get the pose of an object in the environment from a natural language description and current camera view.
        If the object is not found, return None for all return values.

        Args:
            object_name: The name of the object to get the pose of.
            return_bbox_extent:  Whether to return the extent of the oriented bounding box (oriented by quaternion_wxyz). Default is False.

        Returns:
            position: (3,) XYZ in meters, in the world frame.
            quaternion_wxyz: (4,) WXYZ unit quaternion.
            bbox_extent: (3,) XYZ in meters (full side length, not half-length extent). If return_bbox_extent is False, returns None.
            o3d_points: point cloud of the object in np.ndarray format, in the world frame.
            obb: Oriented bounding box of the object (center, R, extent).
        """
        rgb, depth, intrinsics, world_from_cam = self._camera_view()
        results = self._segment(rgb, text_prompt=object_name)
        if len(results) == 0:
            raise ValueError("No sam3 detections")
        scores = [result["score"] for result in results]
        if np.max(scores) < self.sam3_score_threshold:
            return None, None, None, None, None
        mask = np.asarray(results[int(np.argmax(scores))]["mask"], dtype=bool)
        points = _object_points(mask, depth, intrinsics, world_from_cam)
        obb = oriented_bounding_box(points)
        quaternion = matrix_to_quaternion_wxyz(obb.R)
        print(f"Object position for {object_name}: {obb.center}")
        print(f"Object quaternion wxyz for {object_name}: {quaternion}")
        if return_bbox_extent:
            print(f"Object extent for {object_name}: {obb.extent}")
            return obb.center, quaternion, obb.extent, points, obb
        return obb.center, quaternion, None, points, obb

    def sample_grasp_pose(
        self, object_name: str
    ) -> tuple[list[tuple[np.ndarray, np.ndarray]], list[tuple[np.ndarray, np.ndarray]]] | tuple[None, None]:
        """Sample a grasp pose for an object in the environment from a natural language description.
        If the object is not found or no grasp is found, return None for all return values.
        If the object is found, return a list of pregrasp and grasp poses for the object.
        The complete list of pregrasp and grasp poses are:
        - Simple pregrasp pose: The pose to execute before the grasp using a simple topdown grasp pose from the object's oriented bounding box.
        - Simple grasp pose: The pose to execute during the grasp using a simple topdown grasp pose from the object's oriented bounding box.
        - Pregrasp pose topdown: The pose to execute before the grasp using the graspnet grasp pose, but with a topdown orientation.
        - Grasp pose topdown: The pose to execute during the grasp using the graspnet grasp pose, but with a topdown orientation.
        If no graspnet detections are found, return the simple pregrasp and grasp poses.
        Args:
            object_name: The name of the object to sample a grasp pose for.
        Returns:
            if object is found:
                pregrasp_poses: List of pregrasp poses to execute, [simple_pregrasp_pose, pregrasp_pose_topdown] or [simple_pregrasp_pose] if no graspnet detections are found. Each pose is (position, quaternion_xyzw) in the world frame.
                grasp_poses: List of grasp poses to execute, [simple_grasp_pose, grasp_pose_topdown] or [simple_grasp_pose] if no graspnet detections are found. Each pose is (position, quaternion_xyzw) in the world frame.
            if object is not found:
                return None, None
        """
        rgb, depth, intrinsics, world_from_cam = self._camera_view()
        results = self._segment(rgb, text_prompt=object_name)
        if len(results) == 0:
            raise ValueError("No sam3 detections")
        scores = [result["score"] for result in results]
        if np.max(scores) < self.sam3_score_threshold:
            print(f"No sam3 detections for {object_name} and no grasp poses found")
            return None, None
        mask = np.asarray(results[int(np.argmax(scores))]["mask"], dtype=bool)
        points = _object_points(mask, depth, intrinsics, world_from_cam)
        obb = oriented_bounding_box(points)
        simple_pregrasp_pose, simple_grasp_pose = self.sample_grasp_pose_simple(
            object_name, obb
        )

        grasps, grasp_scores = self._graspnet_candidates(depth, intrinsics, mask)
        if len(grasps) == 0:
            print(f"No graspnet detections for {object_name} using simple grasp poses")
            return [simple_pregrasp_pose], [simple_grasp_pose]

        # Contact-GraspNet's best grasp, expressed at the Nero TCP through the
        # calibrated Contact-GraspNet-to-TCP transform, then in the world.
        best = grasps[int(np.argmax(grasp_scores))]
        world_from_tcp = world_from_cam @ best @ self._tcp_from_graspnet()
        grasp_pos = world_from_tcp[:3, 3].copy()
        grasp_quat = _xyzw(matrix_to_quaternion_wxyz(world_from_tcp[:3, :3]))
        print(f"Grasp sample position for {object_name}: {grasp_pos}")
        print(f"Grasp sample quaternion xyzw for {object_name}: {grasp_quat}")
        approach_dir = world_from_tcp[:3, 2]
        pregrasp_pos = grasp_pos - approach_dir * PREGRASP_BACKOFF_M

        pregrasp_pose_topdown = (pregrasp_pos, TOP_DOWN_QUATERNION_XYZW.copy())
        grasp_pose_topdown = (grasp_pos, TOP_DOWN_QUATERNION_XYZW.copy())
        return [simple_pregrasp_pose, pregrasp_pose_topdown], [
            simple_grasp_pose,
            grasp_pose_topdown,
        ]

    def sample_grasp_pose_simple(
        self, object_name: str, object_obb: OrientedBoundingBox
    ) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
        """Sample a simple topdown grasp pose for an object in the environment from the object's oriented bounding box.
        The grasp pose is the object's center position and the object's orientation.
        Args:
            object_name: Name of the object to sample a grasp pose for.
            object_obb: Oriented bounding box of the object.
        Returns:
            pregrasp_pose: Pregrasp pose to execute, (position, quaternion_xyzw).
            grasp_pose: Grasp pose to execute, (position, quaternion_xyzw).
        """
        del object_name
        grasp_pos = np.asarray(object_obb.center, dtype=np.float64).copy()
        # Top-down, turned to the object's longest horizontal axis.
        horizontal = np.hypot(object_obb.R[0], object_obb.R[1]) * object_obb.extent
        axis = object_obb.R[:, int(np.argmax(horizontal))]
        yaw = math.atan2(axis[1], axis[0]) if np.hypot(axis[0], axis[1]) > 1e-6 else 0.0
        rotation = _rotation_z(yaw) @ quaternion_wxyz_to_matrix(_wxyz(TOP_DOWN_QUATERNION_XYZW))
        grasp_quat = _xyzw(matrix_to_quaternion_wxyz(rotation))
        approach_dir = rotation[:, 2]
        pregrasp_pos = grasp_pos - approach_dir * SIMPLE_PREGRASP_OFFSET_M
        return (pregrasp_pos, grasp_quat), (grasp_pos, grasp_quat.copy())

    def get_navigation_pose(
        self, P_table: np.ndarray, P_object: np.ndarray
    ) -> tuple[float, float, float]:
        """Get the navigation pose for the robot to navigate to the object on the table.
        The navigation pose is the pose that the robot should navigate to in order to reach the object.
        since the object is on the table, we need to navigate to the closest point on the table edge to the object.
        Args:
            P_table: point cloud of the table in the environment in np.ndarray format.
            P_object: point cloud of the object in the environment in np.ndarray format.
        Returns:
            navigation_pose: Navigation pose for the robot to navigate to the object.
        """
        return get_navigation_pose(P_table, P_object)

    def save_current_observation(self, name) -> None:
        """
        Save the current observation in the environment.
        Args:
            name: Prefix of the saved image file.
        Returns:
            None.
        """
        from PIL import Image

        rgb, _, _, _ = self._camera_view(require_pose=False)
        directory = getattr(self._env, "episode_directory", None)
        directory = Path.cwd() if directory is None else Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{Path(str(name)).name}_rgb.png"
        Image.fromarray(rgb).save(path)
        print(f"Saved observation to {path}")

    # ------------------------------------------------------------------
    # Base
    # ------------------------------------------------------------------

    def get_robot_position(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get the robot position in the environment.
        Returns:
            robot_position: Robot position in the environment (x, y, 0) in the world frame.
            robot_quat: Robot quaternion in the environment in xyzw format.
            robot_yaw: Robot yaw in the environment.
        """
        pose = self._planar_pose()
        if pose is None:
            print("Robot position unavailable: no valid ZED tracking pose")
            return None, None, None
        x, y, yaw = pose
        return (
            np.asarray([x, y, 0.0]),
            np.asarray([0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)]),
            np.asarray(yaw),
        )

    def navigate_to_pose(self, pose_2d) -> bool:
        """
        Navigate to a pose in the environment.
        Args:
            pose: Pose to navigate to, xy and yaw in world frame.
            if the pose is not reachable, interpolate between the robot pose and the goal pose to find a reachable pose.
        Returns:
            success: Whether the pose was navigated to successfully.
        """
        robot_pos, robot_quat, robot_yaw = self.get_robot_position()
        if robot_pos is None:
            return False
        robot_pose = (robot_pos[0], robot_pos[1], float(robot_yaw))
        goal = np.asarray(pose_2d, dtype=np.float64).reshape(3)

        # interpolated goals between robot pose and goal
        waypoints = np.linspace(goal, robot_pose, NAVIGATION_WAYPOINTS)
        for waypoint in waypoints:
            success = self._navigate_to_pose(waypoint)
            if success:
                print("Reached waypoint", waypoint)
                return True
        return False

    def find_object_base_rotate(self, object_name: str) -> bool:
        """
        Rotate the robot base until the object is found in the current field of view.
        Args:
            object_name: Name of the object to find.
        Returns:
            success: Whether the object was found.
        """
        num_idx = int(2 * np.pi // BASE_ROTATE_STEP_RAD)
        for idx in range(num_idx):
            if idx > 0:
                print("Rotating base by", BASE_ROTATE_STEP_RAD, "rad")
                self._turn(BASE_ROTATE_STEP_RAD)
            if self.detect_object_sam3(object_name):
                print("Object found")
                return True
        return False

    def detect_object_sam3(self, object_name: str) -> bool:
        """
        Detect an object in the environment.
        Args:
            object_name: Name of the object to detect.
        Returns:
            success: Whether the object was detected.
        """
        rgb, _, _, _ = self._camera_view(require_pose=False)
        results = self._segment(rgb, text_prompt=object_name)
        if len(results) == 0:
            return False
        scores = [result["score"] for result in results]
        return bool(np.max(scores) > self.sam3_score_threshold)

    # ------------------------------------------------------------------
    # Arms
    # ------------------------------------------------------------------

    def open_gripper(self, arm=0) -> bool:
        """Open gripper fully.
        Args:
            arm: Arm to open the gripper for. 0: left arm, 1: right arm.
        """
        return self._set_gripper(arm, opened=True)

    def close_gripper(self, arm=0) -> bool:
        """Close gripper fully.

        Args:
            arm: Arm to close the gripper for. 0: left arm, 1: right arm.
        """
        return self._set_gripper(arm, opened=False)

    def move_hand(self, target_pose: tuple[np.ndarray, np.ndarray], arm=0) -> bool:
        """
        Move the hand to a target pose in the environment. This function will ignore all obstacles except the robot itself.
        Args:
            target_pose: Target pose to move the hand to, a tuple of (position, quaternion (xyzw)) in the world frame.
            arm: Arm to move the hand for. 0: left arm, 1: right arm.
        Returns:
            success: Whether the hand was moved successfully.
        """
        name = _arm_name(arm)
        try:
            position = np.asarray(target_pose[0], dtype=np.float64).reshape(3)
            quaternion = _wxyz(np.asarray(target_pose[1], dtype=np.float64).reshape(4))
            arm_from_world, _ = self._arm_frames(name)
            target = arm_from_world @ pose_matrix(position, quaternion)
            result = self._env.move_arm_pose(
                name, _transform_to_xyz_rpy(target), self._arm_timeout_s
            )
        except Exception as exc:  # noqa: BLE001 - Cap-X reports and returns False
            print("Move hand failed:", exc)
            return False
        if not _succeeded(result):
            print("Move hand failed:", _reason(result))
            return False
        return True

    def get_robot_relative_eef_pose(self, arm=0) -> tuple[np.ndarray, np.ndarray]:
        """
        Get the relative end-effector pose of the robot.
        Args:
            arm: Arm to get the end-effector pose for. 0: left arm, 1: right arm.
        Returns:
            relative_eef_pose: Relative end-effector pose of the robot in robot base frame (x forward, y left, z up from the floor under the camera), a tuple of (position, quaternion (xyzw)).
        """
        name = _arm_name(arm)
        try:
            _, world_from_arm = self._arm_frames(name)
            world_from_tcp = world_from_arm @ self._arm_from_tcp(name)
            pose = self._planar_pose()
            if pose is None:
                raise RuntimeError("no valid ZED tracking pose")
            x, y, yaw = pose
            world_from_robot = np.eye(4)
            world_from_robot[:3, :3] = _rotation_z(yaw)
            world_from_robot[:3, 3] = [x, y, 0.0]
            robot_from_tcp = np.linalg.inv(world_from_robot) @ world_from_tcp
        except Exception as exc:  # noqa: BLE001 - Cap-X reports and returns None
            print("Relative end-effector pose unavailable:", exc)
            return None, None
        return (
            robot_from_tcp[:3, 3].copy(),
            _xyzw(matrix_to_quaternion_wxyz(robot_from_tcp[:3, :3])),
        )

    def move_to_joint_positions(self, target_joint_positions, arm=0) -> bool:
        """
        Move the robot to a target joint positions in the environment.
        Joint orders: the seven joints of the selected arm, in the order returned by get_current_joint_positions(arm). The base is not moved by this call.
        Args:
            target_joint_positions: Target joint positions to move the arm to, shape (7,).
            arm: Arm to move. 0: left arm, 1: right arm.
        Returns:
            success: Whether the joint positions were reached successfully.
        """
        name = _arm_name(arm)
        try:
            target = np.asarray(target_joint_positions, dtype=np.float64).reshape(7)
            current = self._joint_positions(name)
            steps = int(math.ceil(float(np.max(np.abs(target - current))) / JOINT_WAYPOINT_STEP_RAD))
            waypoints = np.linspace(current, target, max(2, min(steps + 1, MAX_TRAJECTORY_WAYPOINTS)))
            result = self._env.execute_arm_trajectory(
                name, waypoints.tolist(), self._arm_timeout_s
            )
        except Exception as exc:  # noqa: BLE001 - Cap-X reports and returns False
            print("Move to joint positions failed:", exc)
            return False
        if not _succeeded(result):
            print("Move to joint positions failed:", _reason(result))
            return False
        return True

    def get_current_eef_pose(self, arm=0) -> tuple[np.ndarray, np.ndarray]:
        """
        Get the current end-effector pose in the environment.
        Args:
            arm: Arm to get the end-effector pose for. 0: left arm, 1: right arm.
        Returns:
            current_eef_pose: Current end-effector pose in the world frame, a tuple of (position, quaternion (xyzw)).
        """
        name = _arm_name(arm)
        try:
            _, world_from_arm = self._arm_frames(name)
            world_from_tcp = world_from_arm @ self._arm_from_tcp(name)
        except Exception as exc:  # noqa: BLE001 - Cap-X reports and returns None
            print("End-effector pose unavailable:", exc)
            return None, None
        return (
            world_from_tcp[:3, 3].copy(),
            _xyzw(matrix_to_quaternion_wxyz(world_from_tcp[:3, :3])),
        )

    def get_current_joint_positions(self, arm=0) -> np.ndarray:
        """
        Get the current joint positions in the environment.
        Args:
            arm: Arm to read. 0: left arm, 1: right arm.
        Returns:
            current_joint_positions: Current joint positions of the seven joints of the arm.
        """
        name = _arm_name(arm)
        try:
            return self._joint_positions(name)
        except Exception as exc:  # noqa: BLE001 - Cap-X reports and returns None
            print("Joint positions unavailable:", exc)
            return None

    def solve_ik(
        self,
        position: np.ndarray,
        quaternion_wxyz: np.ndarray,
        arm: int = 0,
        offset_translation: np.ndarray = np.array([0.0, 0.0, 0.0]),
    ) -> np.ndarray:
        """Solve inverse kinematics for the hand of one arm.

        Args:
            position:
                Target position in world frame.
                Shape: (3,), dtype float64.
            quaternion_wxyz:
                Target orientation as a unit quaternion in world frame.
                Shape: (4,), [w, x, y, z], dtype float64.
            arm: int = 0, Arm to solve the IK for. 0: left arm, 1: right arm.
            offset_translation: Extra translation applied in the target's own frame, in meters. Default: none.
        Returns:
            joints:
                np.ndarray of shape (7,), dtype float64.
                Joint angles of the seven joints of the arm, or None if no solution was found.
        """
        name = _arm_name(arm)
        try:
            pos = np.asarray(position, dtype=np.float64).reshape(3)
            quat = normalize_quaternion_wxyz(np.asarray(quaternion_wxyz, dtype=np.float64).reshape(4))
            offset = np.asarray(offset_translation, dtype=np.float64).reshape(3)
            world_from_target = pose_matrix(pos, quat)
            world_from_target[:3, 3] += world_from_target[:3, :3] @ offset
            arm_from_world, _ = self._arm_frames(name)
            target = arm_from_world @ world_from_target
            result = self._env.plan_arm_poses(name, [_transform_to_xyz_rpy(target)])
            plans = result.get("plans") if isinstance(result, dict) else None
            plan = plans[0] if plans else None
            if not _succeeded(plan):
                raise RuntimeError(_reason(plan if isinstance(plan, dict) else result))
            return np.asarray(plan["ik_joint_target"], dtype=np.float64).reshape(7)
        except Exception as exc:  # noqa: BLE001 - Cap-X reports the IK error
            print(f"IK solve error: {exc}")
            return None

    def lift_arm(self, arm=0) -> bool:
        """
        Lift the arm in the environment.
        Args:
            arm: Arm to lift the arm for. 0: left arm, 1: right arm.
        Returns:
            success: Whether the arm was lifted.
        """
        name = _arm_name(arm)
        try:
            arm_from_world, world_from_arm = self._arm_frames(name)
            world_from_tcp = world_from_arm @ self._arm_from_tcp(name)
            world_from_tcp[2, 3] += LIFT_HEIGHT_M
            result = self._env.move_arm_pose(
                name,
                _transform_to_xyz_rpy(arm_from_world @ world_from_tcp),
                self._arm_timeout_s,
            )
        except Exception as exc:  # noqa: BLE001 - Cap-X reports and returns False
            print("Lift arm failed:", exc)
            return False
        if not _succeeded(result):
            print("Lift arm failed:", _reason(result))
            return False
        return True

    def check_object_in_hand(self, arm=0) -> bool:
        """
        Check if the grasp was successful by checking if there is an object in the hand. Note that this function may still return True if the wrong object is in the hand.
        The hand holds an object when the gripper was closed with close_gripper and stopped wider than its closed width; an open gripper holds nothing.
        Args:
            arm: Arm to check the grasp for. 0: left arm, 1: right arm.
        Returns:
            grasped_success: Whether the grasp was successful.
        """
        name = _arm_name(arm)
        if not self._gripper_closed.get(name, False):
            print("No object in hand: the gripper is open")
            return False
        try:
            status = self._arm_status(name)
            width = float(status["gripper"]["width_m"])
            control = status.get("gripper_control") or self._env.arm_status().get(
                "gripper_control"
            ) or {}
            closed_width = float(control.get("close_width_m", 0.0))
        except Exception as exc:  # noqa: BLE001 - Cap-X reports and returns False
            print("Gripper state unavailable:", exc)
            return False
        if width <= closed_width + HELD_OBJECT_MIN_WIDTH_M:
            print("Grasp completed, but no object detected in hand after executing grasp")
            return False
        print("Done with grasp")
        return True

    def grasp_object(
        self, pregrasp_pose: np.ndarray, grasp_pose: np.ndarray, object_name: str, arm=0
    ) -> bool:
        """
        Grasp an object in the environment.
        Args:
            pregrasp_pose: Pregrasp pose to execute, (position, quaternion_xyzw).
            grasp_pose: Grasp pose to execute, (position, quaternion_xyzw).
            object_name: Name of the object to grasp.
            arm: Arm to grasp the object for. 0: left arm, 1: right arm.
        Returns:
            success: Whether the grasp sequence (open, reach, close, lift) was executed; False if a pose is missing or a motion failed.
        """
        del object_name
        if pregrasp_pose is None or grasp_pose is None:
            print("Pregrasp or grasp pose not found, grasp failed")
            return False

        self.open_gripper(arm=arm)

        grasp_position = np.asarray(grasp_pose[0], dtype=np.float64).reshape(3)
        grasp_quaternion_wxyz = _wxyz(np.asarray(grasp_pose[1], dtype=np.float64).reshape(4))
        joints = self.solve_ik(grasp_position, grasp_quaternion_wxyz, arm=arm)
        success = joints is not None and self.move_to_joint_positions(joints, arm=arm)
        current_eef_pose = self.get_current_eef_pose(arm=arm)[0]
        if current_eef_pose is not None and (
            np.linalg.norm(current_eef_pose - grasp_position) > GRASP_POSITION_TOLERANCE_M
        ):
            # Cap-X mirrors the horizontal reach error and retries. Its extra
            # descent on the retry compensates R1Pro's hand-to-fingertip
            # offset; the Nero TCP is already at the fingertips.
            diff = current_eef_pose - grasp_position
            diff[-1] = 0.0
            new_grasp_pose = grasp_position - diff
            joints = self.solve_ik(new_grasp_pose, grasp_quaternion_wxyz, arm=arm)
            success = joints is not None and self.move_to_joint_positions(joints, arm=arm)

        print("closing gripper")
        self.close_gripper(arm=arm)

        print("lifting arm")
        lifted = self.lift_arm(arm=arm)
        return bool(success and lifted)

    # ------------------------------------------------------------------
    # Hardware helpers
    # ------------------------------------------------------------------

    def _camera_view(
        self, *, require_pose: bool = True
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
        """Return the current RGB, depth (H, W), scaled intrinsics and world-from-camera."""

        observation = self._env.get_observation()
        camera = observation["robot0_robotview"]
        rgb = np.asarray(camera["images"]["rgb"], dtype=np.uint8)
        depth = np.asarray(camera["images"]["depth"], dtype=np.float32)
        if depth.ndim == 3 and depth.shape[-1] == 1:
            depth = depth[..., 0]
        if depth.shape != rgb.shape[:2]:
            raise RuntimeError("ZED RGB and depth dimensions do not match")
        intrinsics = self._scaled_intrinsics(depth.shape)
        world_from_cam = None
        if require_pose:
            world_from_cam = self._world_from_camera(observation)
        return rgb, depth, intrinsics, world_from_cam

    def _scaled_intrinsics(self, shape: tuple[int, int]) -> np.ndarray:
        calibration = self._env.manipulation_calibration("left")
        intrinsics = np.asarray(calibration["camera_intrinsics"], dtype=np.float64).copy()
        calibration_width, calibration_height = calibration["camera_calibration_resolution"]
        height, width = shape
        scale_x = width / float(calibration_width)
        scale_y = height / float(calibration_height)
        intrinsics[0, 0] *= scale_x
        intrinsics[0, 2] = scale_x * (intrinsics[0, 2] + 0.5) - 0.5
        intrinsics[1, 1] *= scale_y
        intrinsics[1, 2] = scale_y * (intrinsics[1, 2] + 0.5) - 0.5
        return intrinsics

    def _world_from_camera(self, observation: dict[str, Any]) -> np.ndarray:
        pose = _finite_pose(observation.get("base", {}).get("pose_xy_yaw"))
        if pose is None:
            raise RuntimeError("the ZED planar tracking pose is unavailable")
        ground = observation["robot0_robotview"].get("ground_plane") or {}
        if ground.get("valid"):
            height = float(ground["camera_height_m"])
            down = np.asarray(ground["down_camera_xyz"], dtype=np.float64)
        elif self._fallback_ground_plane is not None:
            height, down = self._fallback_ground_plane
        else:
            raise RuntimeError("the ZED floor plane is unavailable")
        return world_from_camera(pose, height, down)

    def _planar_pose(self) -> np.ndarray | None:
        observation = self._env.get_observation()
        return _finite_pose(observation.get("base", {}).get("pose_xy_yaw"))

    def _arm_frames(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        """Return (arm_from_world, world_from_arm) for one arm at the current base pose."""

        observation = self._env.get_observation()
        world_from_cam = self._world_from_camera(observation)
        arm_from_cam = np.asarray(
            self._env.manipulation_calibration(name)["arm_from_camera"], dtype=np.float64
        )
        arm_from_world = arm_from_cam @ np.linalg.inv(world_from_cam)
        return arm_from_world, np.linalg.inv(arm_from_world)

    def _arm_status(self, name: str) -> dict[str, Any]:
        status = self._env.arm_status()
        arm = status.get(name) if isinstance(status, dict) else None
        if not isinstance(arm, dict):
            raise RuntimeError(f"{name} arm status is unavailable")
        return arm

    def _joint_positions(self, name: str) -> np.ndarray:
        joints = np.asarray(self._arm_status(name)["joint_pos"], dtype=np.float64).reshape(-1)
        if joints.shape != (7,) or not np.all(np.isfinite(joints)):
            raise RuntimeError(f"{name} arm joint feedback is invalid")
        return joints

    def _arm_from_tcp(self, name: str) -> np.ndarray:
        pose = np.asarray(self._arm_status(name)["tcp_pose_xyz_rpy"], dtype=np.float64).reshape(6)
        return pose_matrix(pose[:3], rpy_to_quaternion_wxyz(pose[3:]))

    def _set_gripper(self, arm: int | str, *, opened: bool) -> bool:
        name = _arm_name(arm)
        settings = dict(getattr(self._env, "capx_manipulation_config", {}) or {})
        if bool(settings.get("simulate_gripper", False)):
            self._gripper_closed[name] = not opened
            return True
        try:
            result = self._env.set_gripper(name, opened, 3.0, None)
        except Exception as exc:  # noqa: BLE001 - Cap-X reports and returns False
            print("Gripper command failed:", exc)
            return False
        if not _succeeded(result):
            print("Gripper command failed:", _reason(result))
            return False
        self._gripper_closed[name] = not opened
        return True

    def _graspnet_candidates(
        self, depth: np.ndarray, intrinsics: np.ndarray, mask: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        try:
            grasps, scores, _ = self._plan_grasps(depth, intrinsics, mask.astype(np.uint8), 1)
        except Exception as exc:  # noqa: BLE001 - Cap-X falls back to the simple grasp
            print("Contact-GraspNet failed:", exc)
            return np.zeros((0, 4, 4)), np.zeros(0)
        grasps = np.asarray(grasps, dtype=np.float64)
        scores = np.asarray(scores, dtype=np.float64).reshape(-1)
        if grasps.ndim != 3 or grasps.shape[1:] != (4, 4) or len(grasps) != len(scores):
            return np.zeros((0, 4, 4)), np.zeros(0)
        finite = np.isfinite(scores) & np.all(np.isfinite(grasps), axis=(1, 2))
        return grasps[finite], scores[finite]

    def _tcp_from_graspnet(self) -> np.ndarray:
        """The calibrated Contact-GraspNet gripper frame to Nero TCP transform."""

        config = self._env.manipulation_config
        depth_m = float(config.get("contact_graspnet_origin_to_tcp_m", 0.1034))
        alignment = np.asarray(
            config.get("contact_graspnet_to_nero_tcp_rpy_rad", [0.0, 0.0, -math.pi / 2.0]),
            dtype=np.float64,
        ).reshape(3)
        graspnet_from_tcp = np.eye(4)
        graspnet_from_tcp[:3, :3] = quaternion_wxyz_to_matrix(rpy_to_quaternion_wxyz(alignment))
        graspnet_from_tcp[2, 3] = depth_m
        return graspnet_from_tcp

    def _navigate_to_pose(self, pose_2d: Sequence[float]) -> bool:
        """Turn toward the goal, drive straight to it, turn to its heading.

        Straight segments are bounded by the controller's distance limit and
        re-aimed from the measured pose, so a long goal is reached in several
        segments. There is no obstacle check.
        """

        goal_x, goal_y, goal_yaw = (float(value) for value in pose_2d)
        try:
            controller = self._env.controller
            segment_limit = float(controller.config.max_distance_m)
            for _ in range(64):
                pose = self._planar_pose()
                if pose is None:
                    print("Navigate to pose failed: no valid ZED tracking pose")
                    return False
                remaining = math.hypot(goal_x - pose[0], goal_y - pose[1])
                if remaining <= NAVIGATION_POSITION_TOLERANCE_M:
                    break
                bearing = math.atan2(goal_y - pose[1], goal_x - pose[0])
                heading_error = _wrap_angle(bearing - pose[2])
                if abs(heading_error) > NAVIGATION_YAW_TOLERANCE_RAD:
                    if not self._turn(heading_error):
                        return False
                result = controller.drive_straight(
                    min(remaining, segment_limit), obstacle_check=False
                )
                if not _succeeded(result):
                    print("Navigate to pose failed:", _reason(result))
                    return False
            else:
                print("Navigate to pose failed: goal not reached")
                return False
            pose = self._planar_pose()
            if pose is None:
                print("Navigate to pose failed: no valid ZED tracking pose")
                return False
            yaw_error = _wrap_angle(goal_yaw - pose[2])
            if abs(yaw_error) > NAVIGATION_YAW_TOLERANCE_RAD and not self._turn(yaw_error):
                return False
        except Exception as exc:  # noqa: BLE001 - Cap-X reports and returns False
            print("Navigate to pose failed:", exc)
            return False
        return True

    def _turn(self, angle_rad: float) -> bool:
        """Turn in place, in steps within the controller's turn limit."""

        try:
            controller = self._env.controller
            limit = float(controller.config.max_turn_rad)
            remaining = _wrap_angle(float(angle_rad))
            while abs(remaining) > NAVIGATION_YAW_TOLERANCE_RAD:
                step = max(-limit, min(limit, remaining))
                result = controller.turn_relative(step)
                if not _succeeded(result):
                    print("Base rotation failed:", _reason(result))
                    return False
                remaining -= step
        except Exception as exc:  # noqa: BLE001 - Cap-X ignores a failed rotation
            print("Base rotation failed:", exc)
            return False
        return True


# ----------------------------------------------------------------------
# Perception clients and geometry
# ----------------------------------------------------------------------


def init_sam3_point_prompt(
    *,
    service_url: str | None = None,
    timeout_s: float = 120.0,
    max_retries: int = 5,
) -> Callable[[np.ndarray, Sequence[float]], list[dict[str, Any]]]:
    """Return the point-prompt segmentation callable of the SAM3 service."""

    base_url = str(
        service_url or os.environ.get("SAM3_SERVICE_URL", DEFAULT_SERVICE_URL)
    ).rstrip("/")

    def segment_point(
        image: np.ndarray, point_coords: Sequence[float]
    ) -> list[dict[str, Any]]:
        x, y = (float(value) for value in point_coords)
        response = _post_json(
            f"{base_url}/segment_point",
            {"image_base64": _encode_image(image), "point_coords": [x, y]},
            timeout_s=float(timeout_s),
            max_retries=int(max_retries),
        )
        scores = [float(value) for value in response.get("scores") or []]
        shape = tuple(int(value) for value in response.get("masks_shape") or ())
        encoded = response.get("masks_base64") or ""
        if not scores or not encoded or len(shape) != 3 or shape[0] == 0:
            return []
        dtype = np.dtype(str(response.get("masks_dtype") or "float32"))
        masks = np.frombuffer(base64.b64decode(encoded), dtype=dtype).reshape(shape)
        results = []
        for index in range(min(len(scores), shape[0])):
            mask = masks[index]
            results.append(
                {
                    "mask": np.asarray(mask > 0.0 if mask.dtype.kind == "f" else mask, dtype=bool),
                    "score": scores[index],
                }
            )
        return results

    return segment_point


def _wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def _fallback_ground_plane(
    docking_settings: dict[str, Any],
) -> tuple[float, np.ndarray] | None:
    height = docking_settings.get("ground_camera_height_m")
    down = docking_settings.get("ground_down_camera_xyz")
    if height is None or down is None:
        return None
    return float(height), np.asarray(down, dtype=np.float64).reshape(3)


def floor_axes(down_camera_xyz: Sequence[float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return unit camera-frame down, floor-forward and floor-left axes."""

    down = np.asarray(down_camera_xyz, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(down))
    if norm <= 1e-9:
        raise ValueError("the floor down direction must be nonzero")
    down = down / norm
    optical_forward = np.asarray([0.0, 0.0, 1.0])
    planar_forward = optical_forward - down * float(np.dot(optical_forward, down))
    norm = float(np.linalg.norm(planar_forward))
    if norm <= 1e-6:
        raise RuntimeError("camera optical axis is parallel to the floor normal")
    planar_forward /= norm
    planar_left = np.cross(planar_forward, down)
    planar_left /= np.linalg.norm(planar_left)
    return down, planar_forward, planar_left


def world_from_camera(
    pose_xy_yaw: Sequence[float],
    camera_height_m: float,
    down_camera_xyz: Sequence[float],
) -> np.ndarray:
    """World-from-camera transform from the planar base pose and the floor plane.

    The planar pose is the camera's floor projection and heading in the ZED
    odometry frame; the floor plane gives the camera height and the world
    down direction in camera coordinates.
    """

    down, forward, left = floor_axes(down_camera_xyz)
    x, y, yaw = (float(value) for value in np.asarray(pose_xy_yaw, dtype=np.float64).reshape(3))
    # Columns: the world forward, left and up axes in camera coordinates.
    camera_from_floor = np.column_stack((forward, left, -down))
    transform = np.eye(4)
    transform[:3, :3] = _rotation_z(yaw) @ camera_from_floor.T
    transform[:3, 3] = [x, y, float(camera_height_m)]
    return transform


def backproject_depth(
    mask: np.ndarray, depth: np.ndarray, intrinsics: np.ndarray, world_from_cam: np.ndarray
) -> np.ndarray:
    """Masked depth pixels to world points (camera: x right, y down, z forward)."""

    rows, cols = np.nonzero(mask)
    z = depth[rows, cols].astype(np.float64)
    x = (cols.astype(np.float64) - intrinsics[0, 2]) * z / intrinsics[0, 0]
    y = (rows.astype(np.float64) - intrinsics[1, 2]) * z / intrinsics[1, 1]
    points = np.column_stack((x, y, z))
    return points @ world_from_cam[:3, :3].T + world_from_cam[:3, 3]


def _object_points(
    mask: np.ndarray, depth: np.ndarray, intrinsics: np.ndarray, world_from_cam: np.ndarray
) -> np.ndarray:
    if mask.shape != depth.shape:
        raise RuntimeError("SAM3 mask shape does not match the ZED image")
    valid = mask & np.isfinite(depth) & (depth > 0.0)
    points = backproject_depth(valid, depth, intrinsics, world_from_cam)
    return remove_statistical_outliers(points, OUTLIER_NEIGHBORS, OUTLIER_STD_RATIO)


def remove_statistical_outliers(
    points: np.ndarray, nb_neighbors: int, std_ratio: float
) -> np.ndarray:
    """Open3D's ``remove_statistical_outlier`` on a NumPy point cloud."""

    if len(points) <= nb_neighbors:
        return points
    try:
        from scipy.spatial import cKDTree
    except ImportError:
        return points
    distances, _ = cKDTree(points).query(points, k=nb_neighbors + 1)
    mean_distance = distances[:, 1:].mean(axis=1)
    threshold = mean_distance.mean() + std_ratio * mean_distance.std()
    return points[mean_distance <= threshold]


def oriented_bounding_box(points: np.ndarray) -> OrientedBoundingBox:
    """PCA-aligned box like Open3D's ``get_oriented_bounding_box``."""

    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 4:
        raise ValueError(f"an oriented bounding box needs at least 4 points, got {len(points)}")
    center = np.mean(points, axis=0)
    centered = points - center
    _, _, axes = np.linalg.svd(centered, full_matrices=False)
    rotation = axes.T
    if np.linalg.det(rotation) < 0.0:
        rotation[:, -1] *= -1.0
    local = centered @ rotation
    lower = np.min(local, axis=0)
    upper = np.max(local, axis=0)
    return OrientedBoundingBox(
        center=center + ((lower + upper) * 0.5) @ rotation.T,
        R=rotation,
        extent=upper - lower,
    )


def convex_hull_xy(points_xy: np.ndarray) -> np.ndarray:
    """Counter-clockwise convex hull vertices of 2-D points (monotone chain)."""

    unique = np.unique(np.asarray(points_xy, dtype=np.float64).reshape(-1, 2), axis=0)
    if len(unique) < 3:
        return unique
    ordered = unique[np.lexsort((unique[:, 1], unique[:, 0]))]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[np.ndarray] = []
    for point in ordered:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper: list[np.ndarray] = []
    for point in ordered[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return np.asarray(lower[:-1] + upper[:-1])


def closest_point_on_segment(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ab = b - a
    t = np.dot(p - a, ab) / (np.dot(ab, ab) + 1e-8)
    t = np.clip(t, 0.0, 1.0)
    return a + t * ab


def get_navigation_pose(P_table: np.ndarray, P_object: np.ndarray) -> tuple[float, float, float]:
    """Cap-X's table-edge navigation goal (capx/integrations/r1pro/utils.py)."""

    P_table = np.asarray(P_table, dtype=np.float64).reshape(-1, 3)
    P_object = np.asarray(P_object, dtype=np.float64).reshape(-1, 3)
    object_center = np.median(P_object, axis=0)
    table_polygon = convex_hull_xy(P_table[:, :2])
    if len(table_polygon) < 2:
        raise ValueError("the table point cloud does not span an edge")
    table_center_xy = table_polygon.mean(axis=0)
    object_xy = object_center[:2]

    min_dist = np.inf
    best_edge_idx = 0
    best_edge_point = table_polygon[0]
    for i in range(len(table_polygon)):
        a = table_polygon[i]
        b = table_polygon[(i + 1) % len(table_polygon)]
        cp = closest_point_on_segment(object_xy, a, b)
        d = np.linalg.norm(cp - object_xy)
        if d < min_dist:
            min_dist = d
            best_edge_idx = i
            best_edge_point = cp

    a = table_polygon[best_edge_idx]
    b = table_polygon[(best_edge_idx + 1) % len(table_polygon)]
    edge = b - a
    edge = edge / (np.linalg.norm(edge) + 1e-8)
    n1 = np.asarray([-edge[1], edge[0]])
    n2 = -n1
    to_center = table_center_xy - best_edge_point
    outward = n1 if np.dot(n1, to_center) < 0 else n2
    base_xy = best_edge_point + outward * TABLE_EDGE_BUFFER_M
    dx, dy = object_xy - base_xy
    yaw = np.arctan2(dy, dx)
    return float(base_xy[0]), float(base_xy[1]), float(yaw)


def _rotation_z(yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.asarray([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _xyzw(quaternion_wxyz: np.ndarray) -> np.ndarray:
    q = np.asarray(quaternion_wxyz, dtype=np.float64).reshape(4)
    return np.asarray([q[1], q[2], q[3], q[0]])


def _wxyz(quaternion_xyzw: np.ndarray) -> np.ndarray:
    q = np.asarray(quaternion_xyzw, dtype=np.float64).reshape(4)
    return normalize_quaternion_wxyz(np.asarray([q[3], q[0], q[1], q[2]]))


def _transform_to_xyz_rpy(transform: np.ndarray) -> list[float]:
    quaternion = matrix_to_quaternion_wxyz(transform[:3, :3])
    rpy = quaternion_wxyz_to_rpy(quaternion)
    return [float(value) for value in np.concatenate((transform[:3, 3], rpy))]


def _finite_pose(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    pose = np.asarray(value, dtype=np.float64).reshape(-1)
    if pose.shape != (3,) or not np.all(np.isfinite(pose)):
        return None
    return pose


def _arm_name(arm: int | str) -> str:
    if arm in (0, "0", "left"):
        return "left"
    if arm in (1, "1", "right"):
        return "right"
    raise ValueError("arm must be 0 (left) or 1 (right)")


def _succeeded(result: Any) -> bool:
    return isinstance(result, dict) and bool(result.get("success", False))


def _reason(result: Any) -> str:
    if isinstance(result, dict):
        return str(result.get("reason") or result.get("error") or result)
    return repr(result)


__all__ = [
    "CapXYorR1ProApi",
    "OrientedBoundingBox",
    "R1PRO_FUNCTIONS",
    "OMITTED_R1PRO_FUNCTIONS",
    "backproject_depth",
    "convex_hull_xy",
    "floor_axes",
    "get_navigation_pose",
    "init_sam3_point_prompt",
    "oriented_bounding_box",
    "remove_statistical_outliers",
    "world_from_camera",
]
