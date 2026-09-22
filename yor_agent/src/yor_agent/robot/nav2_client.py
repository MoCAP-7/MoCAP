"""Lazy ROS 2 client for the Nav2 ``NavigateToPose`` and ``ComputePathToPose`` actions.

ROS imports intentionally happen at construction time.  The agent package and
its unit tests therefore remain usable before Humble is installed or sourced.
"""

from __future__ import annotations

import json
import math
import threading
import time
from collections.abc import Callable, Mapping
from typing import Any


class _Nav2ProgressWatchdog:
    """Track meaningful Nav2 motion across planner recovery cycles."""

    def __init__(
        self,
        *,
        started: float,
        timeout_s: float,
        translation_m: float,
        yaw_rad: float,
    ) -> None:
        self.timeout_s = float(timeout_s)
        self.translation_m = float(translation_m)
        self.yaw_rad = float(yaw_rad)
        self.last_progress_at = float(started)
        self._anchor_pose: tuple[float, float, float] | None = None

    def update(self, feedback: Mapping[str, Any], *, now: float) -> None:
        pose = self._valid_pose(feedback.get("current_pose_xy_yaw"))
        progressed = False

        if self._anchor_pose is None and pose is not None:
            self._anchor_pose = pose
        elif pose is not None and self._anchor_pose is not None:
            dx = pose[0] - self._anchor_pose[0]
            dy = pose[1] - self._anchor_pose[1]
            yaw_delta = abs(
                math.atan2(
                    math.sin(pose[2] - self._anchor_pose[2]),
                    math.cos(pose[2] - self._anchor_pose[2]),
                )
            )
            progressed = (
                math.hypot(dx, dy) >= self.translation_m
                or yaw_delta >= self.yaw_rad
            )

        if progressed:
            self.last_progress_at = float(now)
            if pose is not None:
                self._anchor_pose = pose

    def stalled(self, *, now: float) -> bool:
        return float(now) - self.last_progress_at >= self.timeout_s

    def restart_timer(self, *, now: float) -> None:
        """Start the action-level deadline after Nav2 accepts the goal."""

        self.last_progress_at = float(now)

    def idle_s(self, *, now: float) -> float:
        return max(0.0, float(now) - self.last_progress_at)

    @staticmethod
    def _valid_pose(value: Any) -> tuple[float, float, float] | None:
        if not isinstance(value, (list, tuple)) or len(value) != 3:
            return None
        try:
            pose = tuple(float(item) for item in value)
        except (TypeError, ValueError):
            return None
        if not all(math.isfinite(item) for item in pose):
            return None
        return pose


class Nav2Client:
    """Own a small rclpy node and synchronous facade around Nav2 actions."""

    def __init__(
        self,
        *,
        action_name: str = "navigate_to_pose",
        stop_service_name: str = "/yor_base_bridge/stop",
        halt_service_name: str = "/yor_base_bridge/halt",
        clear_stop_service_name: str = "/yor_base_bridge/clear_stop",
        self_filter_service_name: str = "/yor_zed_bridge/set_parameters",
        planner_action_name: str = "compute_path_to_pose",
        node_name: str = "yor_agent_nav2_client",
    ) -> None:
        try:
            import rclpy
            from action_msgs.msg import GoalStatus
            from nav2_msgs.action import ComputePathToPose, NavigateToPose
            from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
            from rcl_interfaces.srv import SetParameters
            from rclpy.action import ActionClient
            from rclpy.executors import SingleThreadedExecutor
            from rclpy.signals import SignalHandlerOptions
            from std_srvs.srv import Trigger
        except ImportError as exc:
            raise RuntimeError(
                "ROS 2 Humble/Nav2 Python packages are unavailable in this "
                "agent process. In the same shell that launches yor-agent or "
                "the Web UI, source /home/yor/codefield/YOR/yor_agent/scripts/"
                "source_nav2_env.sh, then restart the process."
            ) from exc

        self._rclpy = rclpy
        self._goal_status = GoalStatus
        self._navigate_type = NavigateToPose
        self._compute_path_type = ComputePathToPose
        self._trigger_type = Trigger
        self._parameter_type = Parameter
        self._parameter_value_type = ParameterValue
        self._parameter_kind = ParameterType
        self._set_parameters_type = SetParameters
        self._owns_context = not rclpy.ok()
        if self._owns_context:
            # Keep SIGINT on the main Python thread. The docking controller can
            # then catch KeyboardInterrupt, cancel the goal, latch the bridge,
            # and issue the existing direct-RPC zero before the process exits.
            rclpy.init(
                args=None,
                signal_handler_options=SignalHandlerOptions.NO,
            )
        self._node = rclpy.create_node(node_name)
        self._action = ActionClient(self._node, NavigateToPose, action_name)
        self._planner_action = ActionClient(
            self._node, ComputePathToPose, planner_action_name
        )
        self._stop_client = self._node.create_client(Trigger, stop_service_name)
        self._halt_client = self._node.create_client(Trigger, halt_service_name)
        self._clear_stop_client = self._node.create_client(
            Trigger, clear_stop_service_name
        )
        self._self_filter_client = self._node.create_client(
            SetParameters, self_filter_service_name
        )
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)
        self._spin_thread = threading.Thread(
            target=self._executor.spin,
            name="yor-nav2-client",
            daemon=True,
        )
        self._spin_thread.start()
        self._lock = threading.Lock()
        self._active_goal: Any | None = None
        # The send future of a goal whose acceptance is still awaited, so a
        # cancel that lands in that window can still reach the goal.
        self._pending_send_future: Any | None = None
        self._stop_requested = threading.Event()
        self._closed = False

    def navigate_to_pose(
        self,
        x_m: float,
        y_m: float,
        yaw_rad: float,
        *,
        frame_id: str = "odom",
        server_timeout_s: float = 10.0,
        timeout_s: float = 300.0,
        behavior_tree: str = "",
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
        progress_interval_s: float = 1.0,
        no_progress_timeout_s: float = 10.0,
        minimum_progress_translation_m: float = 0.03,
        minimum_progress_yaw_rad: float = 0.05,
        self_filter_attached_objects: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Send one goal and wait while the background executor spins ROS.

        ``progress_callback`` receives JSON-safe lifecycle/feedback snapshots.
        It is observational only: callback failures never affect navigation.
        """

        values = (
            x_m,
            y_m,
            yaw_rad,
            server_timeout_s,
            timeout_s,
            progress_interval_s,
            no_progress_timeout_s,
            minimum_progress_translation_m,
            minimum_progress_yaw_rad,
        )
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("Nav2 goal and timeout values must be finite")
        if server_timeout_s <= 0.0 or timeout_s <= 0.0:
            raise ValueError("Nav2 timeouts must be positive")
        if progress_interval_s <= 0.0:
            raise ValueError("progress_interval_s must be positive")
        if no_progress_timeout_s <= 0.0:
            raise ValueError("no_progress_timeout_s must be positive")
        if (
            minimum_progress_translation_m <= 0.0
            or minimum_progress_yaw_rad <= 0.0
        ):
            raise ValueError("Nav2 minimum progress values must be positive")
        if not frame_id:
            raise ValueError("Nav2 frame_id must be non-empty")
        if self._closed:
            raise RuntimeError("Nav2 client is closed")
        started = time.monotonic()
        feedback: dict[str, Any] = {}
        feedback_lock = threading.Lock()
        last_progress_emit = [float("-inf")]
        watchdog = _Nav2ProgressWatchdog(
            started=started,
            timeout_s=no_progress_timeout_s,
            translation_m=minimum_progress_translation_m,
            yaw_rad=minimum_progress_yaw_rad,
        )

        def feedback_snapshot() -> dict[str, Any]:
            with feedback_lock:
                return dict(feedback)

        def emit_progress(
            state: str,
            *,
            reason: str | None = None,
            terminal: bool = False,
            snapshot: dict[str, Any] | None = None,
        ) -> None:
            if progress_callback is None:
                return
            event: dict[str, Any] = {
                "state": state,
                "terminal": bool(terminal),
                "elapsed_s": max(0.0, time.monotonic() - started),
                "timestamp": time.time(),
                "feedback": feedback_snapshot() if snapshot is None else snapshot,
            }
            if reason is not None:
                event["reason"] = reason
            try:
                progress_callback(event)
            except Exception:
                # Status reporting must never alter a physical motion result.
                pass

        if self._stop_requested.is_set():
            # A stop that landed since the caller last looked must not be
            # erased by the next goal: the caller forgets it once per
            # sequence with ``reset_stop_request``, never a goal.
            emit_progress("failed", reason="operator_stop", terminal=True)
            return self._result(False, "operator_stop", started, feedback)
        emit_progress("waiting_for_server")
        if not self._action.wait_for_server(timeout_sec=float(server_timeout_s)):
            emit_progress(
                "failed", reason="server_unavailable", terminal=True
            )
            return self._result(False, "server_unavailable", started, feedback)
        if self_filter_attached_objects is not None:
            emit_progress("updating_robot_self_filter")
            filter_result = self.update_robot_self_filter(
                self_filter_attached_objects,
                timeout_s=float(server_timeout_s),
            )
            if not filter_result["success"]:
                emit_progress(
                    "failed", reason=str(filter_result["reason"]), terminal=True
                )
                return self._result(
                    False, str(filter_result["reason"]), started, feedback
                )
        emit_progress("clearing_stop")
        clear_result = self.clear_stop(timeout_s=float(server_timeout_s))
        if not clear_result["success"]:
            emit_progress(
                "failed", reason=str(clear_result["reason"]), terminal=True
            )
            return self._result(
                False,
                str(clear_result["reason"]),
                started,
                feedback,
            )

        goal = self._navigate_type.Goal()
        goal.pose.header.frame_id = frame_id
        goal.pose.header.stamp = self._node.get_clock().now().to_msg()
        goal.pose.pose.position.x = float(x_m)
        goal.pose.pose.position.y = float(y_m)
        goal.pose.pose.orientation.z = math.sin(float(yaw_rad) / 2.0)
        goal.pose.pose.orientation.w = math.cos(float(yaw_rad) / 2.0)
        goal.behavior_tree = str(behavior_tree)

        def on_feedback(message: Any) -> None:
            value = message.feedback
            update: dict[str, Any] = {}
            update["distance_remaining_m"] = float(value.distance_remaining)
            update["navigation_time_s"] = (
                float(value.navigation_time.sec)
                + float(value.navigation_time.nanosec) * 1e-9
            )
            update["number_of_recoveries"] = int(value.number_of_recoveries)
            estimated = getattr(value, "estimated_time_remaining", None)
            if estimated is not None:
                update["estimated_time_remaining_s"] = max(
                    0.0,
                    float(estimated.sec) + float(estimated.nanosec) * 1e-9,
                )
            current_pose = getattr(value, "current_pose", None)
            if current_pose is not None:
                pose = current_pose.pose
                quaternion = pose.orientation
                update["current_pose_xy_yaw"] = [
                    float(pose.position.x),
                    float(pose.position.y),
                    math.atan2(
                        2.0
                        * (
                            float(quaternion.w) * float(quaternion.z)
                            + float(quaternion.x) * float(quaternion.y)
                        ),
                        1.0
                        - 2.0
                        * (
                            float(quaternion.y) * float(quaternion.y)
                            + float(quaternion.z) * float(quaternion.z)
                        ),
                    ),
                ]
            now = time.monotonic()
            with feedback_lock:
                feedback.update(update)
                watchdog.update(feedback, now=now)
                snapshot = dict(feedback)
                should_emit = (
                    now - last_progress_emit[0] >= float(progress_interval_s)
                )
                if should_emit:
                    last_progress_emit[0] = now
            if should_emit:
                emit_progress("navigating", snapshot=snapshot)

        emit_progress("sending_goal")
        send_future = self._action.send_goal_async(
            goal, feedback_callback=on_feedback
        )
        with self._lock:
            self._pending_send_future = send_future
        if not self._wait_future(send_future, started, timeout_s):
            # An operator stop wakes this wait too; it must read as one to a
            # caller that sequences goals. The cancel waits briefly for the
            # acceptance; a send still unanswered stays on record so a later
            # cancel can reach the goal if Nav2 accepts it after all.
            operator_stopped = self._stop_requested.is_set()
            self._cancel_goal_and_latch()
            reason = "operator_stop" if operator_stopped else "goal_response_timeout"
            emit_progress("failed", reason=reason, terminal=True)
            return self._result(False, reason, started, feedback)
        goal_handle = send_future.result()
        accepted = goal_handle is not None and bool(goal_handle.accepted)
        with self._lock:
            self._pending_send_future = None
            if accepted:
                self._active_goal = goal_handle
        if not accepted:
            emit_progress("failed", reason="goal_rejected", terminal=True)
            return self._result(False, "goal_rejected", started, feedback)
        watchdog.restart_timer(now=time.monotonic())
        emit_progress("navigating")
        result_future = goal_handle.get_result_async()
        if not self._wait_future(
            result_future,
            started,
            timeout_s,
            stop_condition=lambda: watchdog.stalled(now=time.monotonic()),
        ):
            operator_stopped = self._stop_requested.is_set()
            stalled = (
                not operator_stopped
                and watchdog.stalled(now=time.monotonic())
            )
            # The client's own cancel: it must not read as an operator stop
            # to a caller that sequences goals.
            self._cancel_goal_and_latch()
            if operator_stopped:
                reason = "operator_stop"
            elif stalled:
                reason = "stalled_no_progress"
                with feedback_lock:
                    feedback["no_progress_timeout_s"] = float(
                        no_progress_timeout_s
                    )
                    feedback["no_progress_elapsed_s"] = watchdog.idle_s(
                        now=time.monotonic()
                    )
            else:
                reason = "timeout"
            emit_progress("canceled", reason=reason, terminal=True)
            return self._result(False, reason, started, feedback)

        wrapped = result_future.result()
        with self._lock:
            self._active_goal = None
        status = int(wrapped.status)
        success = status == self._goal_status.STATUS_SUCCEEDED
        status_names = {
            self._goal_status.STATUS_SUCCEEDED: "succeeded",
            self._goal_status.STATUS_ABORTED: "aborted",
            self._goal_status.STATUS_CANCELED: "canceled",
        }
        reason = status_names.get(status, f"goal_status_{status}")
        emit_progress(
            "succeeded" if success else reason,
            reason=reason,
            terminal=True,
        )
        output = self._result(success, reason, started, feedback_snapshot())
        output["status"] = status
        return output

    def compute_path_to_pose(
        self,
        x_m: float,
        y_m: float,
        yaw_rad: float,
        *,
        frame_id: str = "odom",
        server_timeout_s: float = 10.0,
        timeout_s: float = 10.0,
    ) -> dict[str, Any]:
        """Ask the Nav2 planner whether a pose is reachable, without driving.

        A pure planning query: it neither clears nor sets the bridge stop
        latch, and an operator stop does not cut the wait short. The result
        carries ``success`` plus ``reason`` (``planned``, ``server_unavailable``,
        ``goal_response_timeout``, ``goal_rejected``, ``timeout`` or
        ``no_path``), ``path_length_m`` for a planned path, and ``elapsed_s``.
        """

        values = (x_m, y_m, yaw_rad, server_timeout_s, timeout_s)
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("Nav2 plan-check pose and timeout values must be finite")
        if server_timeout_s <= 0.0 or timeout_s <= 0.0:
            raise ValueError("Nav2 timeouts must be positive")
        if not frame_id:
            raise ValueError("Nav2 frame_id must be non-empty")
        if self._closed:
            raise RuntimeError("Nav2 client is closed")
        started = time.monotonic()
        if not self._planner_action.wait_for_server(
            timeout_sec=float(server_timeout_s)
        ):
            return self._plan_result(False, "server_unavailable", started, None)

        goal = self._compute_path_type.Goal()
        goal.goal.header.frame_id = frame_id
        goal.goal.header.stamp = self._node.get_clock().now().to_msg()
        goal.goal.pose.position.x = float(x_m)
        goal.goal.pose.position.y = float(y_m)
        goal.goal.pose.orientation.z = math.sin(float(yaw_rad) / 2.0)
        goal.goal.pose.orientation.w = math.cos(float(yaw_rad) / 2.0)
        goal.use_start = False
        goal.planner_id = ""

        send_future = self._planner_action.send_goal_async(goal)
        if not self._wait_future(
            send_future, started, timeout_s, respect_stop_requested=False
        ):
            return self._plan_result(False, "goal_response_timeout", started, None)
        goal_handle = send_future.result()
        if goal_handle is None or not goal_handle.accepted:
            return self._plan_result(False, "goal_rejected", started, None)
        result_future = goal_handle.get_result_async()
        if not self._wait_future(
            result_future, started, timeout_s, respect_stop_requested=False
        ):
            # Release the planner; the goal cancel alone leaves the bridge
            # stop latch untouched.
            try:
                goal_handle.cancel_goal_async()
            except Exception:  # noqa: BLE001 - the timeout is the result
                pass
            return self._plan_result(False, "timeout", started, None)

        wrapped = result_future.result()
        if int(wrapped.status) != self._goal_status.STATUS_SUCCEEDED:
            return self._plan_result(False, "no_path", started, None)
        poses = list(wrapped.result.path.poses)
        if not poses:
            return self._plan_result(False, "no_path", started, None)
        path_length_m = 0.0
        for previous, current in zip(poses, poses[1:]):
            path_length_m += math.hypot(
                float(current.pose.position.x) - float(previous.pose.position.x),
                float(current.pose.position.y) - float(previous.pose.position.y),
            )
        return self._plan_result(True, "planned", started, path_length_m)

    def stop_requested(self) -> bool:
        """Whether an operator stop has been requested since the last reset.

        ``cancel_and_stop`` sets the flag; ``navigate_to_pose`` refuses a goal
        while it is set and only ``reset_stop_request`` clears it, so a caller
        that sequences several goals resets it once at the start and reads it
        between them. The client's own stall and timeout cancels leave it
        unset.
        """

        return self._stop_requested.is_set()

    def reset_stop_request(self) -> None:
        """Forget a stop request, typically one latched by an earlier task's
        final ``cancel_and_stop``, before a new sequence of goals begins."""

        self._stop_requested.clear()

    def cancel_and_stop(self, *, timeout_s: float = 0.75) -> dict[str, Any]:
        """Cancel the action and synchronously latch the ROS velocity bridge.

        Waiting for the bridge's zero command to complete is important.  The
        primitive issues a second, direct-RPC zero immediately afterwards; if
        both clients generate timestamp sequences concurrently, the Pi can
        otherwise receive the newer bridge sequence first and reject the
        direct zero as stale.

        This is the operator's path: it records the stop request, which any
        wait in flight and every later goal honour until the next reset.
        """

        self._stop_requested.set()
        return self._cancel_goal_and_latch(timeout_s=timeout_s)

    def _cancel_goal_and_latch(self, *, timeout_s: float = 0.75) -> dict[str, Any]:
        """Cancel the active goal and latch the bridge; the stop request flag
        is left as it is.

        The client's own stall and timeout cancels come through here, so a
        caller sequencing goals can tell them from an operator stop. A goal
        whose acceptance is still awaited gets a short, bounded wait: once
        accepted it is cancelled rather than left driving after the latch
        is released for the next goal.
        """

        with self._lock:
            goal = self._active_goal
            pending = self._pending_send_future
        handles = [] if goal is None else [goal]
        # The pending send is waited for whether or not a handle is held: a
        # handle left from an earlier cancel must not hide the goal whose
        # acceptance is still awaited.
        if pending is not None:
            if self._wait_future(
                pending, time.monotonic(), 0.5, respect_stop_requested=False
            ):
                with self._lock:
                    if self._pending_send_future is pending:
                        self._pending_send_future = None
                handle = pending.result()
                if (
                    handle is not None
                    and bool(handle.accepted)
                    and all(handle is not held for held in handles)
                ):
                    handles.append(handle)
        for handle in handles:
            try:
                handle.cancel_goal_async()
            except Exception:  # noqa: BLE001 - the bridge stop is independent
                pass
        with self._lock:
            # A cancelled goal is no longer the active one; keeping it would
            # make a later cancel skip the goal that replaced it.
            if any(self._active_goal is handle for handle in handles):
                self._active_goal = None
        bridge = {
            "requested": False,
            "completed": False,
            "success": False,
            "message": "stop_service_unavailable",
        }
        try:
            bridge = self._call_trigger(
                self._stop_client,
                timeout_s=timeout_s,
                unavailable_message="stop_service_unavailable",
                timeout_message="stop_service_timeout",
            )
        except Exception as exc:  # noqa: BLE001 - direct zero remains fallback
            bridge["message"] = f"{type(exc).__name__}:{exc}"
        return {
            "accepted": True,
            "goal_cancel_requested": bool(handles),
            "bridge_stop_requested": bool(bridge["requested"]),
            "bridge_stop_completed": bool(bridge["completed"]),
            "bridge_stop_succeeded": bool(bridge["success"]),
            "bridge_stop_message": str(bridge["message"]),
        }

    def halt(self, *, timeout_s: float = 0.75) -> dict[str, Any]:
        """Synchronously discard cached ROS velocity and hold bridge zero."""

        bridge = {
            "requested": False,
            "completed": False,
            "success": False,
            "message": "halt_service_unavailable",
        }
        try:
            bridge = self._call_trigger(
                self._halt_client,
                timeout_s=timeout_s,
                unavailable_message="halt_service_unavailable",
                timeout_message="halt_service_timeout",
            )
        except Exception as exc:  # noqa: BLE001 - direct zero remains fallback
            bridge["message"] = f"{type(exc).__name__}:{exc}"
        return {
            "accepted": True,
            "bridge_halt_requested": bool(bridge["requested"]),
            "bridge_halt_completed": bool(bridge["completed"]),
            "bridge_halt_succeeded": bool(bridge["success"]),
            "bridge_halt_message": str(bridge["message"]),
        }

    def _call_trigger(
        self,
        client: Any,
        *,
        timeout_s: float,
        unavailable_message: str,
        timeout_message: str,
    ) -> dict[str, Any]:
        """Call one bridge Trigger and wait even after stop was requested."""

        timeout_s = float(timeout_s)
        if not math.isfinite(timeout_s) or timeout_s <= 0.0:
            raise ValueError("trigger timeout_s must be positive and finite")
        wait_s = min(0.25, timeout_s)
        if not client.wait_for_service(timeout_sec=wait_s):
            return {
                "requested": False,
                "completed": False,
                "success": False,
                "message": unavailable_message,
            }
        future = client.call_async(self._trigger_type.Request())
        started = time.monotonic()
        if not self._wait_future(
            future,
            started,
            timeout_s,
            respect_stop_requested=False,
        ):
            return {
                "requested": True,
                "completed": False,
                "success": False,
                "message": timeout_message,
            }
        response = future.result()
        return {
            "requested": True,
            "completed": True,
            "success": bool(response is not None and response.success),
            "message": (
                "no_response"
                if response is None
                else str(response.message or "trigger_completed")
            ),
        }

    def clear_stop(self, *, timeout_s: float = 10.0) -> dict[str, Any]:
        """Release a prior failure latch before an explicit new goal."""

        if not self._clear_stop_client.wait_for_service(timeout_sec=timeout_s):
            return {"success": False, "reason": "clear_stop_service_unavailable"}
        future = self._clear_stop_client.call_async(self._trigger_type.Request())
        started = time.monotonic()
        if not self._wait_future(future, started, timeout_s):
            return {"success": False, "reason": "clear_stop_timeout"}
        response = future.result()
        if response is None or not response.success:
            message = "no_response" if response is None else response.message
            return {
                "success": False,
                "reason": f"clear_stop_failed:{message}",
            }
        return {"success": True, "reason": "clear_stop_succeeded"}

    def update_robot_self_filter(
        self,
        attached_objects: Mapping[str, Any],
        *,
        timeout_s: float = 10.0,
    ) -> dict[str, Any]:
        """Atomically send held-object boxes to the ZED/Nav2 filter."""

        try:
            payload = json.dumps(
                dict(attached_objects), sort_keys=True, separators=(",", ":")
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("self-filter attachments must be JSON-safe") from exc
        if not self._self_filter_client.wait_for_service(timeout_sec=timeout_s):
            return {
                "success": False,
                "reason": "robot_self_filter_parameter_service_unavailable",
            }
        parameter = self._parameter_type(
            name="self_filter_attached_objects_json",
            value=self._parameter_value_type(
                type=self._parameter_kind.PARAMETER_STRING,
                string_value=payload,
            ),
        )
        request = self._set_parameters_type.Request(parameters=[parameter])
        future = self._self_filter_client.call_async(request)
        started = time.monotonic()
        if not self._wait_future(future, started, timeout_s):
            return {"success": False, "reason": "robot_self_filter_update_timeout"}
        response = future.result()
        results = None if response is None else response.results
        if not results or not all(result.successful for result in results):
            detail = "no_response"
            if results:
                detail = next(
                    (
                        result.reason
                        for result in results
                        if not result.successful and result.reason
                    ),
                    "rejected",
                )
            return {
                "success": False,
                "reason": f"robot_self_filter_update_failed:{detail}",
            }
        return {"success": True, "reason": "robot_self_filter_updated"}

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        # Wake a wait still in flight on another thread, then cancel and
        # latch; no goal follows a close, so the flag's value is moot after.
        self._stop_requested.set()
        self._cancel_goal_and_latch()
        self._executor.shutdown(timeout_sec=1.0)
        self._spin_thread.join(timeout=1.0)
        self._node.destroy_node()
        if self._owns_context and self._rclpy.ok():
            self._rclpy.shutdown()

    def _wait_future(
        self,
        future: Any,
        started: float,
        timeout_s: float,
        *,
        stop_condition: Callable[[], bool] | None = None,
        respect_stop_requested: bool = True,
    ) -> bool:
        while not future.done():
            if respect_stop_requested and self._stop_requested.is_set():
                return False
            if stop_condition is not None and stop_condition():
                return False
            # An externally owned context can still be shut down independently;
            # stop waiting so the caller reaches its direct-RPC zero fallback.
            if not self._rclpy.ok():
                self._stop_requested.set()
                return False
            if time.monotonic() - started >= float(timeout_s):
                return False
            if respect_stop_requested:
                self._stop_requested.wait(0.02)
            else:
                time.sleep(0.02)
        return True

    @staticmethod
    def _result(
        success: bool,
        reason: str,
        started: float,
        feedback: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "success": bool(success),
            "reason": reason,
            "elapsed_s": max(0.0, time.monotonic() - started),
            "feedback": dict(feedback),
        }

    @staticmethod
    def _plan_result(
        success: bool,
        reason: str,
        started: float,
        path_length_m: float | None,
    ) -> dict[str, Any]:
        return {
            "success": bool(success),
            "reason": reason,
            "path_length_m": None if path_length_m is None else float(path_length_m),
            "elapsed_s": max(0.0, time.monotonic() - started),
        }
