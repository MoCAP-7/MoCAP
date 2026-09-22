from __future__ import annotations

import inspect
import math
import threading
import time
from types import SimpleNamespace
import unittest

from yor_agent.robot.nav2_client import Nav2Client, _Nav2ProgressWatchdog


class _DelayedFuture:
    def __init__(self) -> None:
        self.done_calls = 0

    def done(self) -> bool:
        self.done_calls += 1
        return self.done_calls >= 2

    @staticmethod
    def result():
        return type(
            "TriggerResponse",
            (),
            {"success": True, "message": "base_bridge_stopped"},
        )()


class _TriggerClient:
    def __init__(self, future: _DelayedFuture) -> None:
        self.future = future
        self.requests = []

    @staticmethod
    def wait_for_service(*, timeout_sec: float) -> bool:
        return timeout_sec > 0.0

    def call_async(self, request):
        self.requests.append(request)
        return self.future


class _Rclpy:
    @staticmethod
    def ok() -> bool:
        return True


def make_stop_client(stop_future: _DelayedFuture | None = None) -> Nav2Client:
    """A ``Nav2Client`` with only the stop machinery wired: the flag, the
    goal bookkeeping and the bridge stop trigger."""

    client = object.__new__(Nav2Client)
    client._stop_requested = threading.Event()
    client._lock = threading.Lock()
    client._active_goal = None
    client._pending_send_future = None
    client._stop_client = _TriggerClient(stop_future or _DelayedFuture())
    client._trigger_type = type("Trigger", (), {"Request": type("Request", (), {})})
    client._rclpy = _Rclpy()
    return client


class Nav2StopTest(unittest.TestCase):
    def test_cancel_waits_for_bridge_stop_after_setting_stop_event(self) -> None:
        future = _DelayedFuture()
        client = make_stop_client(future)

        result = client.cancel_and_stop(timeout_s=0.2)

        self.assertTrue(client._stop_requested.is_set())
        self.assertGreaterEqual(future.done_calls, 2)
        self.assertEqual(len(client._stop_client.requests), 1)
        self.assertTrue(result["bridge_stop_requested"])
        self.assertTrue(result["bridge_stop_completed"])
        self.assertTrue(result["bridge_stop_succeeded"])
        self.assertEqual(result["bridge_stop_message"], "base_bridge_stopped")

    def test_stop_requested_is_readable_and_resettable_between_goals(self) -> None:
        client = make_stop_client()

        self.assertFalse(client.stop_requested())
        client.cancel_and_stop(timeout_s=0.2)
        self.assertTrue(client.stop_requested())
        # Reading does not consume the flag; only a reset clears it.
        self.assertTrue(client.stop_requested())
        client.reset_stop_request()
        self.assertFalse(client.stop_requested())
        self.assertFalse(client._stop_requested.is_set())

    def test_planner_action_name_is_a_constructor_setting(self) -> None:
        parameter = inspect.signature(Nav2Client.__init__).parameters[
            "planner_action_name"
        ]
        self.assertEqual(parameter.kind, inspect.Parameter.KEYWORD_ONLY)
        self.assertEqual(parameter.default, "compute_path_to_pose")


class _Message:
    """Attribute bag standing in for a ROS message; nested fields appear on
    first access so ``goal.goal.pose.position.x = ...`` just works."""

    def __getattr__(self, name):
        value = _Message()
        setattr(self, name, value)
        return value


class _ComputePathType:
    Goal = _Message


class _Future:
    def __init__(self, value, *, done_after: int = 1) -> None:
        self._value = value
        self._done_after = done_after
        self.done_calls = 0

    def done(self) -> bool:
        self.done_calls += 1
        return self.done_calls >= self._done_after

    def result(self):
        return self._value


class _NeverDoneFuture:
    @staticmethod
    def done() -> bool:
        return False


class _GoalHandle:
    def __init__(self, result_future, *, accepted: bool = True) -> None:
        self.accepted = accepted
        self._result_future = result_future
        self.cancel_calls = 0

    def get_result_async(self):
        return self._result_future

    def cancel_goal_async(self) -> None:
        self.cancel_calls += 1


class _PlannerAction:
    def __init__(self, send_future, *, available: bool = True) -> None:
        self.send_future = send_future
        self.available = available
        self.goals = []

    def wait_for_server(self, *, timeout_sec: float) -> bool:
        return self.available and timeout_sec > 0.0

    def send_goal_async(self, goal):
        self.goals.append(goal)
        return self.send_future


class _GoalStatus:
    STATUS_SUCCEEDED = 4
    STATUS_CANCELED = 5
    STATUS_ABORTED = 6


class _Node:
    @staticmethod
    def get_clock():
        return SimpleNamespace(now=lambda: SimpleNamespace(to_msg=lambda: "stamp"))


def _path_pose(x: float, y: float):
    return SimpleNamespace(
        pose=SimpleNamespace(position=SimpleNamespace(x=x, y=y))
    )


def _planner_result(status: int, poses):
    return SimpleNamespace(
        status=status,
        result=SimpleNamespace(path=SimpleNamespace(poses=list(poses))),
    )


def make_planner_client(
    planner: _PlannerAction, *, stop_requested: bool = False
) -> Nav2Client:
    """A ``Nav2Client`` with only the planner path wired, plus the stop
    machinery so the tests can assert it stays untouched."""

    client = make_stop_client()
    if stop_requested:
        client._stop_requested.set()
    client._closed = False
    client._node = _Node()
    client._goal_status = _GoalStatus
    client._compute_path_type = _ComputePathType
    client._planner_action = planner
    client._clear_stop_client = _TriggerClient(_DelayedFuture())
    return client


class _NavigateType:
    Goal = _Message


class _NavigateAction:
    """The ``NavigateToPose`` action client: ``on_send`` runs inside
    ``send_goal_async``, before the caller sees the returned future."""

    def __init__(self, send_future, *, on_send=None) -> None:
        self.send_future = send_future
        self.on_send = on_send
        self.goals = []

    @staticmethod
    def wait_for_server(*, timeout_sec: float) -> bool:
        return timeout_sec > 0.0

    def send_goal_async(self, goal, feedback_callback=None):
        del feedback_callback
        self.goals.append(goal)
        if self.on_send is not None:
            self.on_send()
        return self.send_future


class _TriggeringFuture(_Future):
    """A send future whose first ``done()`` poll runs ``trigger``: the
    operator's stop lands while the goal's acceptance is awaited."""

    def __init__(self, value, trigger, *, done_after: int = 2) -> None:
        super().__init__(value, done_after=done_after)
        self._trigger = trigger

    def done(self) -> bool:
        first = self.done_calls == 0
        outcome = super().done()
        if first:
            self._trigger()
        return outcome


def make_navigate_client(action: _NavigateAction) -> Nav2Client:
    """A ``Nav2Client`` with the navigate path wired over the stop machinery."""

    client = make_stop_client()
    client._closed = False
    client._node = _Node()
    client._goal_status = _GoalStatus
    client._navigate_type = _NavigateType
    client._action = action
    client._clear_stop_client = _TriggerClient(_DelayedFuture())
    return client


def _navigate(client: Nav2Client, **overrides):
    options = {
        "frame_id": "odom",
        "timeout_s": 1.0,
        "no_progress_timeout_s": 0.05,
        "progress_interval_s": 1.0,
    }
    options.update(overrides)
    return client.navigate_to_pose(1.0, 2.0, 0.0, **options)


class Nav2NavigateStopTest(unittest.TestCase):
    """Which cancels count as an operator stop, a stop already pending when a
    goal is asked for, and a stop that lands while acceptance is awaited."""

    def test_stall_cancel_leaves_the_stop_request_unset(self) -> None:
        handle = _GoalHandle(_NeverDoneFuture())
        action = _NavigateAction(_Future(handle))
        client = make_navigate_client(action)

        result = _navigate(client)

        self.assertFalse(result["success"])
        self.assertEqual(result["reason"], "stalled_no_progress")
        self.assertEqual(result["feedback"]["no_progress_timeout_s"], 0.05)
        # The goal was cancelled and the bridge latched, as before, but the
        # client's own cancel is not an operator stop.
        self.assertEqual(handle.cancel_calls, 1)
        self.assertEqual(len(client._stop_client.requests), 1)
        self.assertFalse(client.stop_requested())
        self.assertIsNone(client._pending_send_future)

    def test_goal_response_timeout_cancel_leaves_the_stop_request_unset(self) -> None:
        action = _NavigateAction(_NeverDoneFuture())
        client = make_navigate_client(action)

        result = _navigate(client, timeout_s=0.05)

        self.assertEqual(result["reason"], "goal_response_timeout")
        self.assertEqual(len(action.goals), 1)
        self.assertEqual(len(client._stop_client.requests), 1)
        self.assertFalse(client.stop_requested())
        # The unanswered send stays on record so a later cancel can still
        # reach the goal should Nav2 accept it after all.
        self.assertIs(client._pending_send_future, action.send_future)
        self.assertIsNone(client._active_goal)

    def test_operator_cancel_sets_the_stop_request(self) -> None:
        # A goal still active when the operator stops: cancelled, forgotten
        # as the active goal, and the stop recorded for later goals.
        handle = _GoalHandle(_NeverDoneFuture())
        client = make_stop_client()
        client._active_goal = handle
        self.assertFalse(client.stop_requested())

        result = client.cancel_and_stop(timeout_s=0.2)

        self.assertTrue(client.stop_requested())
        self.assertTrue(result["goal_cancel_requested"])
        self.assertEqual(handle.cancel_calls, 1)
        self.assertIsNone(client._active_goal)
        self.assertTrue(result["bridge_stop_succeeded"])

    def test_a_pending_stop_refuses_the_goal_before_the_latch_is_released(
        self,
    ) -> None:
        handle = _GoalHandle(_Future(None), accepted=False)
        action = _NavigateAction(_Future(handle))
        client = make_navigate_client(action)
        client.cancel_and_stop(timeout_s=0.2)
        events = []

        result = _navigate(client, progress_callback=events.append)

        self.assertFalse(result["success"])
        self.assertEqual(result["reason"], "operator_stop")
        self.assertEqual(result["feedback"], {})
        # Nothing was sent, the bridge latch was not released, and the stop
        # is still on record for the caller to see.
        self.assertEqual(action.goals, [])
        self.assertEqual(client._clear_stop_client.requests, [])
        self.assertEqual(handle.cancel_calls, 0)
        self.assertTrue(client.stop_requested())
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["state"], "failed")
        self.assertEqual(events[0]["reason"], "operator_stop")
        self.assertTrue(events[0]["terminal"])
        # Only the explicit reset lets a goal through again.
        client.reset_stop_request()
        self.assertEqual(_navigate(client)["reason"], "goal_rejected")
        self.assertEqual(len(action.goals), 1)

    def test_a_stop_as_the_goal_is_sent_cancels_it_once_accepted(self) -> None:
        handle = _GoalHandle(_NeverDoneFuture())
        client_box = []
        action = _NavigateAction(
            _Future(handle, done_after=2),
            on_send=lambda: client_box[0].cancel_and_stop(timeout_s=0.2),
        )
        client = make_navigate_client(action)
        client_box.append(client)

        result = _navigate(client)

        # The operator's cancel found nothing yet; the navigate call's own
        # cancel then waited for the acceptance and cancelled the goal, and
        # the stop that woke the wait is what the caller is told.
        self.assertEqual(result["reason"], "operator_stop")
        self.assertEqual(handle.cancel_calls, 1)
        self.assertEqual(len(client._stop_client.requests), 2)
        self.assertTrue(client.stop_requested())
        self.assertIsNone(client._pending_send_future)
        self.assertIsNone(client._active_goal)

    def test_a_stop_while_acceptance_is_awaited_cancels_the_accepted_goal(
        self,
    ) -> None:
        handle = _GoalHandle(_NeverDoneFuture())
        client_box = []
        send_future = _TriggeringFuture(
            handle, lambda: client_box[0].cancel_and_stop(timeout_s=0.2)
        )
        client = make_navigate_client(_NavigateAction(send_future))
        client_box.append(client)

        result = _navigate(client)

        # The operator's cancel found the pending send, waited for the
        # acceptance and cancelled the goal once; the call's own cancel
        # then had nothing left to cancel.
        self.assertEqual(result["reason"], "operator_stop")
        self.assertEqual(handle.cancel_calls, 1)
        self.assertEqual(len(client._stop_client.requests), 2)
        self.assertIsNone(client._pending_send_future)

    def test_a_cancelled_goal_does_not_hide_the_next_goal_from_a_stop(self) -> None:
        # Goal A stalls and is cancelled by the client itself; goal B is then
        # sent and the operator stops while its acceptance is awaited. B has
        # to be the goal cancelled, not A again.
        handle_a = _GoalHandle(_NeverDoneFuture())
        client = make_navigate_client(_NavigateAction(_Future(handle_a)))

        self.assertEqual(_navigate(client)["reason"], "stalled_no_progress")
        self.assertEqual(handle_a.cancel_calls, 1)
        self.assertIsNone(client._active_goal)

        handle_b = _GoalHandle(_NeverDoneFuture())
        client._action = _NavigateAction(
            _TriggeringFuture(handle_b, lambda: client.cancel_and_stop(timeout_s=0.2))
        )

        result = _navigate(client)

        self.assertEqual(result["reason"], "operator_stop")
        self.assertEqual(handle_b.cancel_calls, 1)
        self.assertEqual(handle_a.cancel_calls, 1)
        self.assertIsNone(client._pending_send_future)
        self.assertIsNone(client._active_goal)

    def test_cancel_reaches_a_pending_send_beside_a_stale_handle(self) -> None:
        stale = _GoalHandle(_NeverDoneFuture())
        pending = _GoalHandle(_NeverDoneFuture())
        client = make_stop_client()
        client._active_goal = stale
        client._pending_send_future = _Future(pending, done_after=2)

        result = client.cancel_and_stop(timeout_s=0.2)

        self.assertTrue(result["goal_cancel_requested"])
        self.assertEqual(stale.cancel_calls, 1)
        self.assertEqual(pending.cancel_calls, 1)
        self.assertIsNone(client._pending_send_future)
        self.assertIsNone(client._active_goal)

    def test_cancel_with_only_a_pending_send_waits_for_the_acceptance(self) -> None:
        handle = _GoalHandle(_NeverDoneFuture())
        client = make_stop_client()
        client._pending_send_future = _Future(handle, done_after=3)

        result = client.cancel_and_stop(timeout_s=0.2)

        self.assertTrue(result["goal_cancel_requested"])
        self.assertEqual(handle.cancel_calls, 1)
        self.assertIsNone(client._pending_send_future)
        self.assertTrue(result["bridge_stop_succeeded"])

    def test_cancel_with_an_unanswered_or_rejected_send_cancels_nothing(self) -> None:
        client = make_stop_client()
        pending = _NeverDoneFuture()
        client._pending_send_future = pending
        started = time.monotonic()

        result = client.cancel_and_stop(timeout_s=0.2)

        # Bounded: the wait for the acceptance gives up well under a second
        # and the bridge is still latched. The send is left for a later
        # cancel to try again.
        self.assertLess(time.monotonic() - started, 1.5)
        self.assertFalse(result["goal_cancel_requested"])
        self.assertTrue(result["bridge_stop_succeeded"])
        self.assertIs(client._pending_send_future, pending)

        rejected = _GoalHandle(_Future(None), accepted=False)
        client = make_stop_client()
        client._pending_send_future = _Future(rejected)

        result = client.cancel_and_stop(timeout_s=0.2)

        self.assertFalse(result["goal_cancel_requested"])
        self.assertEqual(rejected.cancel_calls, 0)
        self.assertIsNone(client._pending_send_future)


class Nav2ComputePathTest(unittest.TestCase):
    def test_planned_path_sums_consecutive_pose_distances(self) -> None:
        wrapped = _planner_result(
            _GoalStatus.STATUS_SUCCEEDED,
            [_path_pose(0.0, 0.0), _path_pose(3.0, 0.0), _path_pose(3.0, 4.0)],
        )
        handle = _GoalHandle(_Future(wrapped, done_after=2))
        planner = _PlannerAction(_Future(handle, done_after=2))
        client = make_planner_client(planner)

        result = client.compute_path_to_pose(
            1.5, -2.0, math.pi / 2.0, frame_id="odom", timeout_s=1.0
        )

        self.assertTrue(result["success"])
        self.assertEqual(result["reason"], "planned")
        self.assertAlmostEqual(result["path_length_m"], 7.0)
        self.assertGreaterEqual(result["elapsed_s"], 0.0)
        self.assertEqual(set(result), {"success", "reason", "path_length_m", "elapsed_s"})
        goal = planner.goals[0]
        self.assertEqual(goal.goal.header.frame_id, "odom")
        self.assertEqual(goal.goal.header.stamp, "stamp")
        self.assertEqual(goal.goal.pose.position.x, 1.5)
        self.assertEqual(goal.goal.pose.position.y, -2.0)
        self.assertAlmostEqual(goal.goal.pose.orientation.z, math.sin(math.pi / 4.0))
        self.assertAlmostEqual(goal.goal.pose.orientation.w, math.cos(math.pi / 4.0))
        self.assertFalse(goal.use_start)
        self.assertEqual(goal.planner_id, "")
        # A pure planning query: no stop latch, no clear_stop, no cancel.
        self.assertFalse(client._stop_requested.is_set())
        self.assertEqual(client._stop_client.requests, [])
        self.assertEqual(client._clear_stop_client.requests, [])
        self.assertEqual(handle.cancel_calls, 0)

    def test_planning_completes_despite_a_pending_operator_stop(self) -> None:
        wrapped = _planner_result(
            _GoalStatus.STATUS_SUCCEEDED, [_path_pose(0.0, 0.0), _path_pose(1.0, 0.0)]
        )
        handle = _GoalHandle(_Future(wrapped, done_after=3))
        planner = _PlannerAction(_Future(handle, done_after=3))
        client = make_planner_client(planner, stop_requested=True)

        result = client.compute_path_to_pose(0.0, 0.0, 0.0, timeout_s=1.0)

        self.assertEqual(result["reason"], "planned")
        self.assertAlmostEqual(result["path_length_m"], 1.0)
        # The latch is neither consumed nor cleared by a plan check, so the
        # caller can still see the stop before sending the next goal.
        self.assertTrue(client._stop_requested.is_set())
        self.assertTrue(client.stop_requested())
        self.assertEqual(client._stop_client.requests, [])

    def test_server_unavailable(self) -> None:
        planner = _PlannerAction(_Future(None), available=False)
        client = make_planner_client(planner)

        result = client.compute_path_to_pose(0.0, 0.0, 0.0, server_timeout_s=0.1)

        self.assertFalse(result["success"])
        self.assertEqual(result["reason"], "server_unavailable")
        self.assertIsNone(result["path_length_m"])
        self.assertEqual(planner.goals, [])

    def test_goal_response_timeout(self) -> None:
        planner = _PlannerAction(_NeverDoneFuture())
        client = make_planner_client(planner)

        result = client.compute_path_to_pose(0.0, 0.0, 0.0, timeout_s=0.05)

        self.assertEqual(result["reason"], "goal_response_timeout")
        self.assertIsNone(result["path_length_m"])
        self.assertFalse(client._stop_requested.is_set())
        self.assertEqual(client._stop_client.requests, [])

    def test_goal_rejected(self) -> None:
        handle = _GoalHandle(_Future(None), accepted=False)
        client = make_planner_client(_PlannerAction(_Future(handle)))

        result = client.compute_path_to_pose(0.0, 0.0, 0.0)

        self.assertEqual(result["reason"], "goal_rejected")
        self.assertIsNone(result["path_length_m"])

    def test_result_timeout_cancels_only_the_planning_goal(self) -> None:
        handle = _GoalHandle(_NeverDoneFuture())
        client = make_planner_client(_PlannerAction(_Future(handle)))

        result = client.compute_path_to_pose(0.0, 0.0, 0.0, timeout_s=0.05)

        self.assertEqual(result["reason"], "timeout")
        self.assertIsNone(result["path_length_m"])
        self.assertEqual(handle.cancel_calls, 1)
        self.assertFalse(client._stop_requested.is_set())
        self.assertEqual(client._stop_client.requests, [])

    def test_aborted_and_empty_paths_are_no_path(self) -> None:
        for label, wrapped in (
            ("aborted", _planner_result(_GoalStatus.STATUS_ABORTED, [])),
            ("canceled", _planner_result(_GoalStatus.STATUS_CANCELED, [])),
            ("empty", _planner_result(_GoalStatus.STATUS_SUCCEEDED, [])),
        ):
            with self.subTest(label):
                handle = _GoalHandle(_Future(wrapped))
                client = make_planner_client(_PlannerAction(_Future(handle)))

                result = client.compute_path_to_pose(0.0, 0.0, 0.0)

                self.assertFalse(result["success"])
                self.assertEqual(result["reason"], "no_path")
                self.assertIsNone(result["path_length_m"])

    def test_rejects_invalid_arguments(self) -> None:
        client = make_planner_client(_PlannerAction(_Future(None)))
        with self.assertRaisesRegex(ValueError, "finite"):
            client.compute_path_to_pose(math.nan, 0.0, 0.0)
        with self.assertRaisesRegex(ValueError, "timeouts"):
            client.compute_path_to_pose(0.0, 0.0, 0.0, timeout_s=0.0)
        with self.assertRaisesRegex(ValueError, "frame_id"):
            client.compute_path_to_pose(0.0, 0.0, 0.0, frame_id="")
        client._closed = True
        with self.assertRaisesRegex(RuntimeError, "closed"):
            client.compute_path_to_pose(0.0, 0.0, 0.0)


class Nav2ProgressWatchdogTest(unittest.TestCase):
    def make_watchdog(self) -> _Nav2ProgressWatchdog:
        return _Nav2ProgressWatchdog(
            started=0.0,
            timeout_s=10.0,
            translation_m=0.03,
            yaw_rad=0.05,
        )

    def test_stationary_feedback_stalls_at_limit(self) -> None:
        watchdog = self.make_watchdog()
        feedback = {
            "distance_remaining_m": 0.20,
            "current_pose_xy_yaw": [1.0, 2.0, 0.4],
        }

        watchdog.update(feedback, now=1.0)
        # A shorter replanned path is not physical base motion and must not
        # keep an otherwise stationary action alive.
        watchdog.update(
            {
                "distance_remaining_m": 0.05,
                "current_pose_xy_yaw": [1.0, 2.0, 0.4],
            },
            now=9.9,
        )

        self.assertFalse(watchdog.stalled(now=9.9))
        self.assertTrue(watchdog.stalled(now=10.0))

    def test_translation_resets_timer(self) -> None:
        watchdog = self.make_watchdog()
        watchdog.update(
            {
                "distance_remaining_m": 0.20,
                "current_pose_xy_yaw": [1.0, 2.0, 0.4],
            },
            now=1.0,
        )

        watchdog.update(
            {
                "distance_remaining_m": 0.20,
                "current_pose_xy_yaw": [1.031, 2.0, 0.4],
            },
            now=9.0,
        )

        self.assertFalse(watchdog.stalled(now=18.9))
        self.assertTrue(watchdog.stalled(now=19.0))

    def test_action_acceptance_restarts_timer(self) -> None:
        watchdog = self.make_watchdog()

        watchdog.restart_timer(now=8.0)

        self.assertFalse(watchdog.stalled(now=17.9))
        self.assertTrue(watchdog.stalled(now=18.0))

    def test_yaw_resets_timer(self) -> None:
        watchdog = self.make_watchdog()
        watchdog.update(
            {
                "distance_remaining_m": 0.20,
                "current_pose_xy_yaw": [1.0, 2.0, 0.4],
            },
            now=1.0,
        )
        watchdog.update(
            {
                "distance_remaining_m": 0.20,
                "current_pose_xy_yaw": [1.0, 2.0, 0.451],
            },
            now=8.0,
        )
        self.assertFalse(watchdog.stalled(now=17.9))
        self.assertTrue(watchdog.stalled(now=18.0))


if __name__ == "__main__":
    unittest.main()
