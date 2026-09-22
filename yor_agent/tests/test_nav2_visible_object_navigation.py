from __future__ import annotations

import json
import math
from types import SimpleNamespace
import unittest
from unittest import mock

from yor_agent.exceptions import PrimitiveFailed
from yor_agent.primitives.registry import PrimitiveRegistry
from yor_agent.primitives.visible_object_navigation import (
    _make_dock_to_visible_object,
    register_visible_object_navigation_primitives,
)
from yor_agent.robot.nav2_visible_object_navigation import (
    Nav2VisibleObjectDockingConfig,
    Nav2VisibleObjectDockingController,
)
from yor_agent.robot.visible_object_navigation import VisibleObjectDockingConfig

from test_visible_object_navigation import FakeDockingEnv, FakeMotion


class FakeNav2Client:
    """``plannable`` scripts ``compute_path_to_pose``: None plans everything,
    a list is consumed one entry per call (exhausted means no path) where an
    entry is True (planned), False (no path) or a failure reason string, and
    a callable of ``(x, y, yaw)`` decides per goal. A successful goal leaves
    the target at ``arrival_distance_m`` in the fake camera; a failed one
    ends with ``failure_reason``.

    The stop flag mirrors the real client: ``cancel_and_stop`` sets it,
    ``navigate_to_pose`` refuses a goal while it is set, only
    ``reset_stop_request`` clears it, and the client's own stall and timeout
    cancels leave it alone."""

    def __init__(
        self,
        env: FakeDockingEnv,
        *,
        success: bool = True,
        failure_reason: str = "aborted",
        plannable=None,
        arrival_distance_m: float = 0.60,
    ) -> None:
        self.env = env
        self.success = success
        self.failure_reason = failure_reason
        self.plannable = (
            list(plannable) if isinstance(plannable, (list, tuple)) else plannable
        )
        self.arrival_distance_m = arrival_distance_m
        self.goals: list[tuple[float, float, float, dict]] = []
        self.plan_checks: list[tuple[float, float, float, dict]] = []
        self.cancel_calls = 0
        self.halt_calls = 0
        self.reset_calls = 0
        self.stop_flag = False

    def compute_path_to_pose(self, x_m, y_m, yaw_rad, **options):
        self.plan_checks.append((float(x_m), float(y_m), float(yaw_rad), options))
        if self.plannable is None:
            outcome = True
        elif callable(self.plannable):
            outcome = bool(self.plannable(x_m, y_m, yaw_rad))
        else:
            outcome = self.plannable.pop(0) if self.plannable else False
        if outcome is True:
            return {
                "success": True,
                "reason": "planned",
                "path_length_m": 1.5,
                "elapsed_s": 0.01,
            }
        return {
            "success": False,
            "reason": "no_path" if outcome is False else str(outcome),
            "path_length_m": None,
            "elapsed_s": 0.01,
        }

    def stop_requested(self) -> bool:
        return self.stop_flag

    def reset_stop_request(self) -> None:
        self.reset_calls += 1
        self.stop_flag = False

    def navigate_to_pose(self, x_m, y_m, yaw_rad, **options):
        if self.stop_flag:
            # A stop already requested refuses the goal before it is sent.
            return {"success": False, "reason": "operator_stop", "feedback": {}}
        self.goals.append((float(x_m), float(y_m), float(yaw_rad), options))
        progress = options.get("progress_callback")
        if progress is not None:
            progress(
                {
                    "state": "navigating",
                    "terminal": False,
                    "elapsed_s": 0.1,
                    "feedback": {"distance_remaining_m": 0.5},
                }
        )
        if self.success:
            self.env.target_distance_m = self.arrival_distance_m
            if progress is not None:
                progress(
                    {
                        "state": "succeeded",
                        "terminal": True,
                        "elapsed_s": 0.2,
                        "reason": "succeeded",
                        "feedback": {"distance_remaining_m": 0.0},
                    }
                )
            return {
                "success": True,
                "reason": "succeeded",
                "feedback": {"distance_remaining_m": 0.0},
            }
        feedback = {}
        if self.failure_reason == "stalled_no_progress":
            # The client's own cancel: the goal is cancelled and the bridge
            # latched, but no operator stop is recorded.
            feedback = {
                "distance_remaining_m": 0.18,
                "navigation_time_s": 10.1,
                "number_of_recoveries": 1,
                "no_progress_timeout_s": 10.0,
            }
        return {"success": False, "reason": self.failure_reason, "feedback": feedback}

    def cancel_and_stop(self):
        self.cancel_calls += 1
        self.stop_flag = True
        return {"accepted": True}

    def halt(self):
        self.halt_calls += 1
        return {"accepted": True, "bridge_halt_requested": True}


class FloorPlaneDockingEnv(FakeDockingEnv):
    """``FakeDockingEnv`` whose ZED frame carries a fresh SDK floor plane, so
    the perception controller adopts its camera height for the observation."""

    def __init__(self, *, camera_height_m: float, **options) -> None:
        super().__init__(**options)
        self.camera_height_m = camera_height_m

    def observe(self):
        observation = super().observe()
        observation["robot0_robotview"].update(
            {
                "timestamp_ns": 2_000_000_000,
                "ground_plane": {
                    "valid": True,
                    "camera_height_m": self.camera_height_m,
                    "down_camera_xyz": [0.0, 1.0, 0.0],
                    "timestamp_ns": 1_900_000_000,
                },
            }
        )
        return observation


class ScriptedNav2Client(FakeNav2Client):
    """Nav2 outcomes per goal in order: True succeeds, False aborts, and a
    string fails with that reason. Anything past the script aborts."""

    def __init__(self, env: FakeDockingEnv, outcomes) -> None:
        super().__init__(env)
        self.outcomes = list(outcomes)

    def navigate_to_pose(self, x_m, y_m, yaw_rad, **options):
        outcome = self.outcomes.pop(0) if self.outcomes else False
        self.success = outcome is True
        self.failure_reason = outcome if isinstance(outcome, str) else "aborted"
        return super().navigate_to_pose(x_m, y_m, yaw_rad, **options)


class StoppableFakeMotion(FakeMotion):
    """``FakeMotion`` with ``NavigationController``'s operator-stop shape: a
    sticky flag ``request_stop`` sets and ``stop_requested`` reads."""

    def __init__(self, env: FakeDockingEnv) -> None:
        super().__init__(env)
        self.stop_requests = 0
        self._stop_requested = False

    def request_stop(self):
        self.stop_requests += 1
        self._stop_requested = True
        return {"accepted": True}

    def stop_requested(self) -> bool:
        return self._stop_requested


class BlindFakeMotion(StoppableFakeMotion):
    """A motion controller that takes a stop request but exposes no flag, so
    a stop before the first goal is only ever seen through the callback."""

    stop_requested = None


class StoppableDockingEnv(FakeDockingEnv):
    """``FakeDockingEnv`` with ``YorEnvironment``'s operator-stop shape: a
    registry of motion-stop callbacks that ``request_stop`` runs before the
    controller's own stop request, errors collected rather than raised."""

    def __init__(self, *, controller_flag: bool = True, **options) -> None:
        super().__init__(**options)
        self.controller = (
            StoppableFakeMotion(self) if controller_flag else BlindFakeMotion(self)
        )
        self.motion_stop_callbacks: set = set()
        self.stop_requests = 0
        # Runs at the start of every SAM3 call; the perception controller
        # caches the segment client, so a hook read per call is the way to
        # land a stop inside detection after construction.
        self.on_segment = None

    def segment(self, rgb, *, text_prompt):
        if self.on_segment is not None:
            self.on_segment()
        return super().segment(rgb, text_prompt=text_prompt)

    def register_motion_stop_callback(self, callback) -> None:
        if not callable(callback):
            raise TypeError("motion stop callback must be callable")
        self.motion_stop_callbacks.add(callback)

    def unregister_motion_stop_callback(self, callback) -> None:
        self.motion_stop_callbacks.discard(callback)

    def request_stop(self):
        self.stop_requests += 1
        errors = []
        for callback in tuple(self.motion_stop_callbacks):
            try:
                callback()
            except Exception as exc:  # noqa: BLE001 - mirrors the environment
                errors.append(f"{type(exc).__name__}: {exc}")
        result = dict(self.controller.request_stop())
        if errors:
            result["higher_level_stop_errors"] = errors
        return result


def make_prior_estimate(bearing_rad: float, **overrides):
    """Duck-typed ``ReadinessPriorEstimate`` with the attributes the controller reads.

    The defaults are pnp-shaped (inliers, reprojection error, focal length set;
    the vggt-only fields None). Pass ``method="vggt"`` overrides for the other
    shape.
    """

    values = {
        "bearing_rad": bearing_rad,
        "heading_rad": bearing_rad + math.pi,
        "hand": "right",
        "suggested_arm": "right",
        "method": "pnp",
        "inliers": 57,
        "reprojection_error_px": 2.1,
        "focal_px": 553.9,
        "frame_time_s": 16.5,
        "event_object": "can",
        "human_xy": (1.2 + math.cos(bearing_rad), math.sin(bearing_rad)),
        "stance_xy": None,
        "human_height_m": None,
        "scale": None,
        "scale_iqr_ratio": None,
        "frames": (),
        "service_time_s": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def make_vggt_prior_estimate(bearing_rad: float, **overrides):
    """The vggt shape: no PnP statistics, the stance fields filled."""

    values = {
        "method": "vggt",
        "inliers": None,
        "reprojection_error_px": None,
        "focal_px": None,
        "human_height_m": 1.47,
        "scale": 2.244,
        "scale_iqr_ratio": 1.16,
        "stance_xy": (0.3, -0.4),
        "frames": ("nav", "ready"),
        "service_time_s": 4.6,
    }
    values.update(overrides)
    return make_prior_estimate(bearing_rad, **values)


class FakePrior:
    """Duck-typed ``ReadinessPrior``: only ``config``, ``last_reason`` and
    ``estimate()`` are read by the docking controller."""

    def __init__(
        self,
        estimate=None,
        *,
        error: BaseException | None = None,
        last_reason: str | None = "no_event_match",
        max_bearing_error_deg: float = 45.0,
        retry_bearing_offsets_deg=(45.0, -45.0),
        fallback_to_arrival_bearing: bool = True,
    ) -> None:
        self.config = SimpleNamespace(
            max_bearing_error_deg=max_bearing_error_deg,
            retry_bearing_offsets_deg=retry_bearing_offsets_deg,
            fallback_to_arrival_bearing=fallback_to_arrival_bearing,
        )
        self._estimate = estimate
        self._error = error
        self.last_reason = last_reason
        self.calls: list[dict] = []

    def estimate(
        self,
        object_name,
        rgb,
        depth,
        intrinsics,
        pose_xy_yaw,
        *,
        down,
        planar_forward,
        planar_left,
        target_xy,
        camera_height_m=None,
    ):
        self.calls.append(
            {
                "object_name": object_name,
                "rgb_shape": tuple(rgb.shape),
                "depth_shape": tuple(depth.shape),
                "intrinsics_shape": tuple(intrinsics.shape),
                "pose_xy_yaw": [float(v) for v in pose_xy_yaw],
                "down": [float(v) for v in down],
                "planar_forward": [float(v) for v in planar_forward],
                "planar_left": [float(v) for v in planar_left],
                "target_xy": tuple(float(v) for v in target_xy),
                "camera_height_m": camera_height_m,
            }
        )
        if self._error is not None:
            raise self._error
        return self._estimate


class LegacyFakePrior(FakePrior):
    """A prior from before ``camera_height_m`` existed: its ``estimate``
    rejects the keyword, which the controller must absorb as an exception."""

    def estimate(  # noqa: D102 - same shape minus the new keyword
        self,
        object_name,
        rgb,
        depth,
        intrinsics,
        pose_xy_yaw,
        *,
        down,
        planar_forward,
        planar_left,
        target_xy,
    ):
        return super().estimate(
            object_name,
            rgb,
            depth,
            intrinsics,
            pose_xy_yaw,
            down=down,
            planar_forward=planar_forward,
            planar_left=planar_left,
            target_xy=target_xy,
        )


class Nav2VisibleObjectDockingTest(unittest.TestCase):
    def make_controller(self, env, client, **settings):
        return Nav2VisibleObjectDockingController(
            env,
            docking_config={"backend": "nav2", **settings},
            segment_client_factory=lambda: env.segment,
            nav2_client_factory=lambda _config: client,
        )

    def test_default_radius_is_arm_conservative_six_tenths_meter(self) -> None:
        config = Nav2VisibleObjectDockingConfig.from_mapping(None)

        self.assertEqual(config.docking_distance_m, 0.60)
        self.assertIsNone(config.nav2_goal_distance_m)
        self.assertEqual(config.effective_nav2_goal_distance_m, 0.60)
        self.assertEqual(config.robot_radius_m, 0.60)
        self.assertEqual(config.base_to_camera_forward_m, 0.2143)
        self.assertEqual(config.base_to_camera_left_m, 0.0603)
        self.assertEqual(config.nav2_frame_id, "odom")
        self.assertEqual(
            config.nav2_clear_stop_service_name,
            "/yor_base_bridge/clear_stop",
        )
        self.assertEqual(config.no_progress_timeout_s, 10.0)
        self.assertEqual(config.minimum_progress_translation_m, 0.03)
        self.assertEqual(config.minimum_progress_yaw_rad, 0.05)

    def test_one_semantic_detection_becomes_one_nav2_goal(self) -> None:
        env = FakeDockingEnv(target_distance_m=1.20)
        client = FakeNav2Client(env)
        controller = self.make_controller(
            env,
            client,
            docking_distance_m=0.80,
            nav2_goal_distance_m=0.72,
        )

        result = controller.dock_to_visible_object("table")

        self.assertTrue(result["success"])
        self.assertEqual(result["metrics"]["backend"], "nav2")
        self.assertEqual(len(client.goals), 1)
        goal_x, goal_y, goal_yaw, options = client.goals[0]
        initial_distance = result["metrics"]["target_distance_m"]
        camera_goal_distance = initial_distance - 0.72
        expected_camera_x = camera_goal_distance
        expected_camera_y = 0.0
        self.assertAlmostEqual(
            goal_x + controller.config.base_to_camera_forward_m,
            expected_camera_x,
        )
        self.assertAlmostEqual(
            goal_y + controller.config.base_to_camera_left_m,
            expected_camera_y,
        )
        self.assertAlmostEqual(goal_yaw, 0.0)
        self.assertEqual(
            result["metrics"]["desired_camera_goal_xy_yaw"],
            [expected_camera_x, expected_camera_y, 0.0],
        )
        self.assertEqual(result["metrics"]["docking_distance_m"], 0.80)
        self.assertEqual(result["metrics"]["nav2_goal_distance_m"], 0.72)
        self.assertEqual(options["frame_id"], "odom")
        self.assertEqual(options["no_progress_timeout_s"], 10.0)
        self.assertEqual(env.controller.drives, [])
        self.assertEqual(env.controller.turns, [])
        self.assertEqual(env.controller.stops, 2)
        self.assertEqual(client.halt_calls, 1)

    def test_nav2_goal_distance_cannot_exceed_success_boundary(self) -> None:
        with self.assertRaisesRegex(ValueError, "nav2_goal_distance_m"):
            Nav2VisibleObjectDockingConfig.from_mapping(
                {
                    "docking_distance_m": 0.80,
                    "nav2_goal_distance_m": 0.81,
                }
            )

    def test_already_docked_does_not_construct_ros_client(self) -> None:
        env = FakeDockingEnv(target_distance_m=0.58)
        client = FakeNav2Client(env)
        controller = self.make_controller(env, client)

        result = controller.dock_to_visible_object("table")

        self.assertTrue(result["success"])
        self.assertEqual(client.goals, [])
        self.assertEqual(env.controller.stops, 2)

    def test_progress_is_streamed_and_only_primitive_completion_is_terminal(self) -> None:
        env = FakeDockingEnv(target_distance_m=1.20)
        client = FakeNav2Client(env)
        events = []
        controller = Nav2VisibleObjectDockingController(
            env,
            docking_config={"backend": "nav2"},
            segment_client_factory=lambda: env.segment,
            nav2_client_factory=lambda _config: client,
            progress_callback=events.append,
        )

        result = controller.dock_to_visible_object("table")

        self.assertTrue(result["success"])
        self.assertIn("detecting_target", [event["state"] for event in events])
        self.assertIn("navigating", [event["state"] for event in events])
        self.assertIn("nav2_succeeded", [event["state"] for event in events])
        self.assertIn("verifying_arrival", [event["state"] for event in events])
        self.assertEqual(events[-1]["state"], "completed")
        self.assertTrue(events[-1]["terminal"])
        self.assertTrue(events[-1]["success"])
        self.assertFalse(
            next(
                event
                for event in events
                if event["state"] == "nav2_succeeded"
            )["terminal"]
        )

    def test_camera_goal_is_converted_to_rotated_base_center_goal(self) -> None:
        env = FakeDockingEnv()
        controller = self.make_controller(env, FakeNav2Client(env))

        goal_x, goal_y, goal_yaw = controller._base_goal_from_camera_goal(
            1.0, 2.0, 3.141592653589793 / 2.0
        )

        self.assertAlmostEqual(
            goal_x, 1.0 + controller.config.base_to_camera_left_m
        )
        self.assertAlmostEqual(
            goal_y, 2.0 - controller.config.base_to_camera_forward_m
        )
        self.assertAlmostEqual(goal_yaw, 3.141592653589793 / 2.0)

    def test_nav2_failure_preserves_primitive_failure_api(self) -> None:
        env = FakeDockingEnv(target_distance_m=1.20)
        client = FakeNav2Client(env, success=False)
        registry = PrimitiveRegistry()
        register_visible_object_navigation_primitives(
            registry,
            env,
            docking_config={"backend": "nav2"},
            segment_client_factory=lambda: env.segment,
            nav2_client_factory=lambda _config: client,
        )

        with self.assertRaises(PrimitiveFailed) as caught:
            registry.functions()["dock_to_visible_object"]("table")

        self.assertEqual(caught.exception.reason, "nav2_failed:aborted")
        self.assertEqual(env.controller.drives, [])
        self.assertEqual(env.controller.turns, [])
        self.assertEqual(env.controller.stops, 2)
        self.assertEqual(client.cancel_calls, 1)
        self.assertEqual(client.halt_calls, 0)

    def test_occupied_start_zone_returns_actionable_recovery_without_nav2(
        self,
    ) -> None:
        env = FakeDockingEnv(target_distance_m=1.20)
        env.controller.nav2_start_clearance_status = lambda **_options: {
            "available": True,
            "blocked": True,
            "obstacle_points": 52,
            "max_points": 3,
        }
        client = FakeNav2Client(env)
        controller = self.make_controller(env, client)

        result = controller.dock_to_visible_object("table")

        self.assertFalse(result["success"])
        self.assertIn("nav2_blocked_near_obstacle", result["reason"])
        self.assertIn("negative distance", result["reason"])
        self.assertEqual(client.goals, [])
        self.assertEqual(client.cancel_calls, 0)
        self.assertEqual(env.controller.stops, 2)
        self.assertTrue(result["metrics"]["nav2_start_clearance"]["blocked"])
        self.assertEqual(
            result["metrics"]["recommended_recovery"]["action"],
            "back_away_then_retry",
        )

    def test_final_stop_failure_does_not_hide_occupied_start_recovery(self) -> None:
        env = FakeDockingEnv(target_distance_m=1.20)
        env.controller.nav2_start_clearance_status = lambda **_options: {
            "available": True,
            "blocked": True,
            "obstacle_points": 52,
            "max_points": 3,
        }
        original_stop = env.controller.stop

        def fail_final_stop():
            if env.controller.stops == 0:
                return original_stop()
            env.controller.stops += 1
            return {
                "primitive": "stop",
                "success": False,
                "reason": "sequence_must_increase",
                "metrics": {},
            }

        env.controller.stop = fail_final_stop
        controller = self.make_controller(env, FakeNav2Client(env))

        result = controller.dock_to_visible_object("table")

        self.assertFalse(result["success"])
        self.assertIn("nav2_blocked_near_obstacle", result["reason"])
        self.assertIn("short negative distance", result["reason"])
        self.assertIn("final_stop_failed: sequence_must_increase", result["reason"])

    def test_recovery_without_progress_translates_aborted_for_policy(self) -> None:
        env = FakeDockingEnv(target_distance_m=1.20)
        client = FakeNav2Client(env, success=False)

        def abort_after_recoveries(x_m, y_m, yaw_rad, **options):
            client.goals.append((x_m, y_m, yaw_rad, options))
            return {
                "success": False,
                "reason": "aborted",
                "feedback": {
                    "distance_remaining_m": 4.39,
                    "navigation_time_s": 123.0,
                    "number_of_recoveries": 10,
                },
            }

        client.navigate_to_pose = abort_after_recoveries
        controller = self.make_controller(env, client)

        result = controller.dock_to_visible_object("trash bin")

        self.assertFalse(result["success"])
        self.assertIn("nav2_blocked_near_obstacle", result["reason"])
        self.assertIn("back away", result["reason"])
        self.assertEqual(
            result["metrics"]["recommended_recovery"]["primitive"],
            "drive_straight",
        )

    def test_client_stall_returns_actionable_forward_recovery(self) -> None:
        env = FakeDockingEnv(target_distance_m=1.20)
        client = FakeNav2Client(env, success=False)

        def stall(x_m, y_m, yaw_rad, **options):
            client.goals.append((x_m, y_m, yaw_rad, options))
            return {
                "success": False,
                "reason": "stalled_no_progress",
                "feedback": {
                    "distance_remaining_m": 0.18,
                    "navigation_time_s": 10.1,
                    "number_of_recoveries": 1,
                    "no_progress_timeout_s": 10.0,
                },
            }

        client.navigate_to_pose = stall
        controller = self.make_controller(env, client)

        result = controller.dock_to_visible_object("cardboard box")

        self.assertFalse(result["success"])
        self.assertIn("nav2_stalled_no_progress", result["reason"])
        self.assertIn("short positive distance", result["reason"])
        recovery = result["metrics"]["recommended_recovery"]
        self.assertEqual(recovery["primitive"], "drive_straight")
        self.assertEqual(recovery["distance_sign"], "positive")
        self.assertEqual(recovery["suggested_distance_m"], 0.10)

    def test_keyboard_interrupt_cancels_and_latches_before_rethrow(self) -> None:
        env = FakeDockingEnv(target_distance_m=1.20)
        client = FakeNav2Client(env)

        def interrupt(*_args, **_kwargs):
            raise KeyboardInterrupt

        client.navigate_to_pose = interrupt
        controller = self.make_controller(env, client)

        with self.assertRaises(KeyboardInterrupt):
            controller.dock_to_visible_object("table")

        self.assertEqual(client.cancel_calls, 1)
        self.assertEqual(client.halt_calls, 0)
        self.assertEqual(env.controller.stops, 2)

    def test_unknown_backend_is_rejected_during_registration(self) -> None:
        env = FakeDockingEnv()
        with self.assertRaisesRegex(ValueError, "backend"):
            register_visible_object_navigation_primitives(
                PrimitiveRegistry(),
                env,
                docking_config={"backend": "rules_v2"},
            )


class ReadinessPriorDockingTest(unittest.TestCase):
    """The passive-video prior chooses the docking bearing; Nav2 is unchanged.

    ``FakeDockingEnv`` starts at pose (0, 0, 0) with the target straight ahead,
    so the arrival bearing (object toward robot) is pi.
    """

    def make_controller(self, env, client, prior, **settings):
        return Nav2VisibleObjectDockingController(
            env,
            docking_config={"backend": "nav2", **settings},
            segment_client_factory=lambda: env.segment,
            nav2_client_factory=lambda _config: client,
            readiness_prior=prior,
        )

    @staticmethod
    def nav2_history(result):
        return [
            entry
            for entry in result["motion_history"]
            if entry["primitive"] == "nav2_navigate_to_pose"
        ]

    def assert_camera_goal(self, controller, goal, camera_x, camera_y, yaw):
        goal_x, goal_y, goal_yaw, _options = goal
        expected = controller._base_goal_from_camera_goal(camera_x, camera_y, yaw)
        self.assertAlmostEqual(goal_x, expected[0])
        self.assertAlmostEqual(goal_y, expected[1])
        yaw_difference = math.atan2(
            math.sin(goal_yaw - expected[2]), math.cos(goal_yaw - expected[2])
        )
        self.assertAlmostEqual(yaw_difference, 0.0)

    def test_prior_bearing_replaces_arrival_bearing(self) -> None:
        env = FakeDockingEnv(target_distance_m=1.20)
        client = FakeNav2Client(env)
        prior = FakePrior(make_prior_estimate(math.pi / 2.0))
        controller = self.make_controller(env, client, prior)

        result = controller.dock_to_visible_object("the can")

        self.assertTrue(result["success"])
        self.assertEqual(len(client.goals), 1)
        target_x, target_y = result["metrics"]["target_world_xy_m"]
        goal_distance = controller.config.effective_nav2_goal_distance_m
        self.assert_camera_goal(
            controller,
            client.goals[0],
            target_x,
            target_y + goal_distance,
            -math.pi / 2.0,
        )
        metrics = result["metrics"]
        self.assertEqual(metrics["bearing_source"], "prior")
        self.assertAlmostEqual(metrics["arrival_bearing_rad"], math.pi)
        self.assertEqual(
            metrics["desired_camera_goal_xy_yaw"],
            metrics["goal_attempts"][0]["camera_goal_xy_yaw"],
        )
        info = metrics["readiness_prior"]
        self.assertTrue(info["used"])
        self.assertIsNone(info["reason"])
        self.assertEqual(info["method"], "pnp")
        self.assertAlmostEqual(info["bearing_deg"], 90.0)
        self.assertAlmostEqual(info["arrival_bearing_deg"], 180.0)
        self.assertAlmostEqual(info["bearing_error_deg"], 90.0)
        self.assertEqual(info["inliers"], 57)
        self.assertEqual(info["event_object"], "can")
        self.assertEqual(info["hand"], "right")
        self.assertEqual(info["frame_time_s"], 16.5)
        self.assertEqual(len(info["human_xy"]), 2)
        history = self.nav2_history(result)
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["bearing_source"], "prior")
        self.assertAlmostEqual(history[0]["bearing_rad"], math.pi / 2.0)
        # The prior saw the same observation and target the goal was built from.
        self.assertEqual(len(prior.calls), 1)
        call = prior.calls[0]
        self.assertEqual(call["object_name"], "the can")
        self.assertEqual(call["rgb_shape"][2], 3)
        self.assertEqual(call["intrinsics_shape"], (3, 3))
        self.assertAlmostEqual(call["target_xy"][0], target_x)
        self.assertAlmostEqual(call["target_xy"][1], target_y)
        self.assertEqual(len(call["down"]), 3)
        self.assertAlmostEqual(math.hypot(*call["planar_forward"]), 1.0)
        self.assertAlmostEqual(math.hypot(*call["planar_left"]), 1.0)
        self.assertAlmostEqual(
            sum(a * b for a, b in zip(call["planar_forward"], call["down"])), 0.0
        )

    def test_prior_hand_becomes_suggested_arm(self) -> None:
        env = FakeDockingEnv(target_distance_m=1.20)
        prior = FakePrior(
            make_prior_estimate(math.pi / 2.0, hand="right", suggested_arm="right")
        )
        controller = self.make_controller(env, FakeNav2Client(env), prior)

        result = controller.dock_to_visible_object("can")

        self.assertEqual(result["suggested_arm"], "left")
        self.assertEqual(result["metrics"]["suggested_arm_source"], "target_side")
        self.assertEqual(result["metrics"]["human_hand_arm"], "right")
        self.assertEqual(result["metrics"]["readiness_prior"]["suggested_arm"], "right")

    def test_a_target_off_to_one_side_picks_that_arm_over_the_hand(self) -> None:
        env = FakeDockingEnv(target_distance_m=1.20)
        prior = FakePrior(
            make_prior_estimate(math.pi / 2.0, hand="right", suggested_arm="right")
        )
        controller = self.make_controller(env, FakeNav2Client(env), prior)
        target = SimpleNamespace(closest_left_m=-0.40)
        metrics: dict = {}

        controller._record_suggested_arm(metrics, target, "right")

        self.assertEqual(metrics["suggested_arm"], "right")
        self.assertEqual(metrics["suggested_arm_source"], "target_side")
        self.assertAlmostEqual(
            metrics["target_left_of_base_m"],
            -0.40 + controller.config.base_to_camera_left_m,
        )
        del prior

    def test_a_centred_target_lets_the_human_hand_break_the_tie(self) -> None:
        env = FakeDockingEnv(target_distance_m=1.20)
        controller = self.make_controller(env, FakeNav2Client(env), None)
        # Cancel the camera offset so the object sits on the base centre line.
        centred = SimpleNamespace(
            closest_left_m=-controller.config.base_to_camera_left_m
        )

        for hand, expected in (("right", "right"), ("left", "left")):
            metrics: dict = {}
            controller._record_suggested_arm(metrics, centred, hand)
            self.assertEqual(metrics["suggested_arm"], expected, hand)
            self.assertEqual(metrics["suggested_arm_source"], "human_hand")

        # With no hand to fall back on the narrow side still decides.
        metrics = {}
        controller._record_suggested_arm(metrics, centred, None)
        self.assertEqual(metrics["suggested_arm_source"], "target_side")

    def test_both_hands_gives_no_suggested_arm(self) -> None:
        env = FakeDockingEnv(target_distance_m=1.20)
        prior = FakePrior(
            make_prior_estimate(math.pi / 2.0, hand="both", suggested_arm=None)
        )
        controller = self.make_controller(env, FakeNav2Client(env), prior)

        result = controller.dock_to_visible_object("can")

        self.assertTrue(result["success"])
        self.assertEqual(result["suggested_arm"], "left")
        self.assertEqual(result["metrics"]["suggested_arm_source"], "target_side")
        self.assertNotIn("human_hand_arm", result["metrics"])
        self.assertEqual(result["metrics"]["readiness_prior"]["hand"], "both")

    def test_without_prior_result_has_no_suggested_arm_and_records_reason(
        self,
    ) -> None:
        env = FakeDockingEnv(target_distance_m=1.20)
        controller = self.make_controller(env, FakeNav2Client(env), None)

        result = controller.dock_to_visible_object("can")

        self.assertTrue(result["success"])
        self.assertEqual(result["suggested_arm"], "left")
        self.assertEqual(result["metrics"]["suggested_arm_source"], "target_side")
        self.assertEqual(
            result["metrics"]["readiness_prior"],
            {"used": False, "reason": "not_configured"},
        )
        self.assertEqual(result["metrics"]["bearing_source"], "arrival")
        self.assertAlmostEqual(result["metrics"]["arrival_bearing_rad"], math.pi)

    def test_aligned_prior_within_docking_distance_shortcuts(self) -> None:
        env = FakeDockingEnv(target_distance_m=0.50)
        client = FakeNav2Client(env)
        prior = FakePrior(make_prior_estimate(math.pi))
        controller = self.make_controller(env, client, prior, docking_distance_m=0.55)

        result = controller.dock_to_visible_object("can")

        self.assertTrue(result["success"])
        self.assertEqual(result["reason"], "within_docking_distance")
        self.assertEqual(result["suggested_arm"], "left")
        self.assertEqual(client.goals, [])
        self.assertNotIn("goal_attempts", result["metrics"])
        self.assertTrue(result["metrics"]["readiness_prior"]["used"])
        self.assertAlmostEqual(
            result["metrics"]["readiness_prior"]["bearing_error_deg"], 0.0
        )

    def test_misaligned_prior_within_docking_distance_redocks(self) -> None:
        env = FakeDockingEnv(target_distance_m=0.50)
        client = FakeNav2Client(env)
        prior = FakePrior(make_prior_estimate(0.0))
        controller = self.make_controller(env, client, prior, docking_distance_m=0.55)

        result = controller.dock_to_visible_object("can")

        self.assertTrue(result["success"])
        self.assertEqual(len(client.goals), 1)
        target_x, target_y = result["metrics"]["target_world_xy_m"]
        goal_distance = controller.config.effective_nav2_goal_distance_m
        self.assert_camera_goal(
            controller, client.goals[0], target_x + goal_distance, target_y, math.pi
        )
        self.assertEqual(result["metrics"]["bearing_source"], "prior")
        self.assertAlmostEqual(
            abs(result["metrics"]["readiness_prior"]["bearing_error_deg"]), 180.0
        )

    def test_bearing_error_at_the_threshold_still_shortcuts(self) -> None:
        env = FakeDockingEnv(target_distance_m=0.50)
        client = FakeNav2Client(env)
        prior = FakePrior(
            make_prior_estimate(math.pi - math.radians(30.0)),
            max_bearing_error_deg=30.0,
        )
        controller = self.make_controller(env, client, prior, docking_distance_m=0.55)

        result = controller.dock_to_visible_object("can")

        self.assertTrue(result["success"])
        self.assertEqual(client.goals, [])

    def test_prior_goal_failures_retry_offsets_then_succeed(self) -> None:
        env = FakeDockingEnv(target_distance_m=1.20)
        client = ScriptedNav2Client(env, [False, False, True])
        prior = FakePrior(make_prior_estimate(math.pi / 2.0))
        controller = self.make_controller(env, client, prior)

        result = controller.dock_to_visible_object("can")

        self.assertTrue(result["success"])
        self.assertEqual(len(client.goals), 3)
        history = self.nav2_history(result)
        self.assertEqual(
            [entry["bearing_source"] for entry in history],
            ["prior", "prior_offset", "prior_offset"],
        )
        expected_bearings = [math.pi / 2.0, 3.0 * math.pi / 4.0, math.pi / 4.0]
        for entry, expected in zip(history, expected_bearings):
            self.assertAlmostEqual(entry["bearing_rad"], expected)
        self.assertEqual([entry["success"] for entry in history], [False, False, True])
        target_x, target_y = result["metrics"]["target_world_xy_m"]
        goal_distance = controller.config.effective_nav2_goal_distance_m
        for goal, bearing in zip(client.goals, expected_bearings):
            self.assert_camera_goal(
                controller,
                goal,
                target_x + goal_distance * math.cos(bearing),
                target_y + goal_distance * math.sin(bearing),
                math.atan2(-math.sin(bearing), -math.cos(bearing)),
            )
        metrics = result["metrics"]
        self.assertEqual(metrics["bearing_source"], "prior_offset")
        self.assertEqual(len(metrics["goal_attempts"]), 3)
        self.assertEqual(
            [attempt["bearing_source"] for attempt in metrics["goal_attempts"]],
            ["prior", "prior_offset", "prior_offset"],
        )
        self.assertEqual(metrics["goal_attempts"][-1]["reason"], "succeeded")
        self.assertEqual(metrics["nav2"]["reason"], "succeeded")
        self.assertEqual(
            metrics["desired_camera_goal_xy_yaw"],
            metrics["goal_attempts"][-1]["camera_goal_xy_yaw"],
        )
        self.assertEqual(
            metrics["nav2_goal_xy_yaw"], metrics["goal_attempts"][-1]["nav2_goal_xy_yaw"]
        )
        self.assertEqual(client.halt_calls, 1)
        self.assertEqual(env.controller.stops, 2)

    def test_all_prior_attempts_fail_then_arrival_fallback_fails(self) -> None:
        env = FakeDockingEnv(target_distance_m=1.20)
        client = FakeNav2Client(env, success=False)
        prior = FakePrior(make_prior_estimate(math.pi / 2.0))
        controller = self.make_controller(env, client, prior)

        reference_env = FakeDockingEnv(target_distance_m=1.20)
        reference_client = FakeNav2Client(reference_env, success=False)
        reference = self.make_controller(reference_env, reference_client, None)
        reference.dock_to_visible_object("can")

        result = controller.dock_to_visible_object("can")

        self.assertFalse(result["success"])
        self.assertEqual(result["reason"], "nav2_failed:aborted")
        self.assertEqual(len(client.goals), 4)
        history = self.nav2_history(result)
        self.assertEqual(
            [entry["bearing_source"] for entry in history],
            ["prior", "prior_offset", "prior_offset", "arrival_fallback"],
        )
        self.assertEqual(result["metrics"]["bearing_source"], "arrival_fallback")
        self.assertEqual(len(result["metrics"]["goal_attempts"]), 4)
        # The fallback is the goal the controller without a prior would send.
        self.assertEqual(client.goals[-1][:3], reference_client.goals[0][:3])
        self.assertEqual(
            result["metrics"]["desired_camera_goal_xy_yaw"],
            reference.dock_to_visible_object("can")["metrics"][
                "desired_camera_goal_xy_yaw"
            ],
        )
        self.assertEqual(client.cancel_calls, 1)
        self.assertEqual(client.halt_calls, 0)
        self.assertEqual(env.controller.stops, 2)

    def test_last_attempt_decides_failure_reason_and_recovery(self) -> None:
        env = FakeDockingEnv(target_distance_m=1.20)
        client = FakeNav2Client(env, success=False)
        outcomes = [
            {"success": False, "reason": "stalled_no_progress", "feedback": {}},
            {"success": False, "reason": "aborted", "feedback": {}},
            {"success": False, "reason": "aborted", "feedback": {}},
            {
                "success": False,
                "reason": "aborted",
                "feedback": {"navigation_time_s": 30.0, "number_of_recoveries": 3},
            },
        ]

        def scripted(x_m, y_m, yaw_rad, **options):
            client.goals.append((x_m, y_m, yaw_rad, options))
            return outcomes.pop(0)

        client.navigate_to_pose = scripted
        prior = FakePrior(make_prior_estimate(math.pi / 2.0))
        controller = self.make_controller(env, client, prior)

        result = controller.dock_to_visible_object("can")

        self.assertFalse(result["success"])
        self.assertIn("nav2_blocked_near_obstacle", result["reason"])
        self.assertEqual(
            result["metrics"]["recommended_recovery"]["action"],
            "back_away_then_retry",
        )
        self.assertEqual(
            result["metrics"]["goal_attempts"][0]["reason"], "stalled_no_progress"
        )

    def test_blocked_prior_attempt_stops_retries_and_returns_recovery(self) -> None:
        env = FakeDockingEnv(target_distance_m=1.20)
        client = FakeNav2Client(env, success=False)

        def blocked(x_m, y_m, yaw_rad, **options):
            client.goals.append((x_m, y_m, yaw_rad, options))
            return {
                "success": False,
                "reason": "blocked_no_progress",
                "feedback": {"distance_remaining_m": 0.9},
            }

        client.navigate_to_pose = blocked
        prior = FakePrior(make_prior_estimate(math.pi / 2.0))
        controller = self.make_controller(env, client, prior)

        result = controller.dock_to_visible_object("can")

        self.assertFalse(result["success"])
        self.assertIn("nav2_blocked_near_obstacle", result["reason"])
        self.assertEqual(len(client.goals), 1)
        self.assertEqual(len(result["metrics"]["goal_attempts"]), 1)
        self.assertEqual(result["metrics"]["bearing_source"], "prior")
        self.assertEqual(
            [entry["bearing_source"] for entry in self.nav2_history(result)],
            ["prior"],
        )
        self.assertEqual(
            result["metrics"]["recommended_recovery"]["action"],
            "back_away_then_retry",
        )
        self.assertEqual(client.cancel_calls, 1)
        self.assertEqual(env.controller.stops, 2)

    def test_prior_without_estimate_reproduces_arrival_goal(self) -> None:
        env = FakeDockingEnv(target_distance_m=1.20)
        client = FakeNav2Client(env)
        prior = FakePrior(None, last_reason="no_event_match")
        controller = self.make_controller(env, client, prior)

        reference_env = FakeDockingEnv(target_distance_m=1.20)
        reference_client = FakeNav2Client(reference_env)
        reference = self.make_controller(reference_env, reference_client, None)

        result = controller.dock_to_visible_object("table")
        expected = reference.dock_to_visible_object("table")

        self.assertTrue(result["success"])
        self.assertEqual(client.goals[0][:3], reference_client.goals[0][:3])
        self.assertEqual(
            result["metrics"]["desired_camera_goal_xy_yaw"],
            expected["metrics"]["desired_camera_goal_xy_yaw"],
        )
        self.assertEqual(
            result["metrics"]["nav2_goal_xy_yaw"],
            expected["metrics"]["nav2_goal_xy_yaw"],
        )
        self.assertEqual(result["metrics"]["bearing_source"], "arrival")
        info = result["metrics"]["readiness_prior"]
        self.assertFalse(info["used"])
        self.assertEqual(info["reason"], "no_event_match")
        self.assertEqual(result["suggested_arm"], "left")
        self.assertEqual(len(prior.calls), 1)

    def test_prior_exception_reproduces_arrival_goal(self) -> None:
        env = FakeDockingEnv(target_distance_m=1.20)
        client = FakeNav2Client(env)
        prior = FakePrior(error=RuntimeError("cv2 exploded"))
        controller = self.make_controller(env, client, prior)

        reference_env = FakeDockingEnv(target_distance_m=1.20)
        reference_client = FakeNav2Client(reference_env)
        reference = self.make_controller(reference_env, reference_client, None)

        result = controller.dock_to_visible_object("table")
        expected = reference.dock_to_visible_object("table")

        self.assertTrue(result["success"])
        self.assertEqual(client.goals[0][:3], reference_client.goals[0][:3])
        self.assertEqual(
            result["metrics"]["desired_camera_goal_xy_yaw"],
            expected["metrics"]["desired_camera_goal_xy_yaw"],
        )
        self.assertEqual(result["metrics"]["bearing_source"], "arrival")
        self.assertEqual(
            result["metrics"]["readiness_prior"],
            {"used": False, "reason": "exception:RuntimeError:cv2 exploded"},
        )

    def test_prior_receives_the_config_camera_height_by_default(self) -> None:
        env = FakeDockingEnv(target_distance_m=1.20)
        prior = FakePrior(make_prior_estimate(math.pi / 2.0))
        controller = self.make_controller(env, FakeNav2Client(env), prior)

        result = controller.dock_to_visible_object("can")

        self.assertTrue(result["success"])
        self.assertFalse(result["metrics"]["ground_plane"]["dynamic"])
        self.assertEqual(len(prior.calls), 1)
        camera_height_m = prior.calls[0]["camera_height_m"]
        self.assertIsInstance(camera_height_m, float)
        self.assertEqual(camera_height_m, 1.122339129447937)

    def test_prior_receives_the_fresh_floor_plane_camera_height(self) -> None:
        # The first plane inside the gate's envelope replaces the configured
        # fallback at once, so the prior sees the live height.
        env = FloorPlaneDockingEnv(camera_height_m=1.10, target_distance_m=1.20)
        prior = FakePrior(make_prior_estimate(math.pi / 2.0))
        controller = self.make_controller(
            env,
            FakeNav2Client(env),
            prior,
            ground_down_camera_xyz=[0.0, 1.0, 0.0],
        )

        result = controller.dock_to_visible_object("can")

        self.assertTrue(result["success"])
        self.assertTrue(result["metrics"]["ground_plane"]["dynamic"])
        self.assertEqual(result["metrics"]["ground_plane"]["gate"], "accepted")
        self.assertEqual(len(prior.calls), 1)
        self.assertAlmostEqual(prior.calls[0]["camera_height_m"], 1.10)
        # The prior saw the same floor as the goal: the plane's down axis.
        self.assertEqual(prior.calls[0]["down"], [0.0, 1.0, 0.0])

    def test_query_passes_the_active_camera_height_verbatim(self) -> None:
        env = FakeDockingEnv(target_distance_m=1.20)
        prior = FakePrior(make_prior_estimate(math.pi / 2.0))
        controller = self.make_controller(env, FakeNav2Client(env), prior)
        rgb, depth, intrinsics = controller._perception._camera_input()
        pose = controller._perception._current_pose()
        controller._perception._active_ground_camera_height_m = 1.07

        outcome = controller._query_readiness_prior(
            "can",
            rgb,
            depth,
            intrinsics,
            pose,
            target_xy=(1.2, 0.0),
            arrival_bearing_rad=math.pi,
        )

        self.assertTrue(outcome.used)
        self.assertEqual(len(prior.calls), 1)
        self.assertAlmostEqual(prior.calls[0]["camera_height_m"], 1.07)

    def test_vggt_estimate_records_stance_fields_and_null_pnp_statistics(
        self,
    ) -> None:
        env = FakeDockingEnv(target_distance_m=1.20)
        client = FakeNav2Client(env)
        prior = FakePrior(make_vggt_prior_estimate(math.pi / 2.0))
        controller = self.make_controller(env, client, prior)

        pnp_env = FakeDockingEnv(target_distance_m=1.20)
        pnp_client = FakeNav2Client(pnp_env)
        pnp_prior = FakePrior(make_prior_estimate(math.pi / 2.0))
        pnp_controller = self.make_controller(pnp_env, pnp_client, pnp_prior)

        result = controller.dock_to_visible_object("can")
        pnp_result = pnp_controller.dock_to_visible_object("can")

        self.assertTrue(result["success"])
        metrics = result["metrics"]
        self.assertEqual(metrics["bearing_source"], "prior")
        # The goal depends on the bearing alone, never on the method's shape.
        self.assertEqual(client.goals[0][:3], pnp_client.goals[0][:3])
        self.assertEqual(
            metrics["desired_camera_goal_xy_yaw"],
            pnp_result["metrics"]["desired_camera_goal_xy_yaw"],
        )
        info = metrics["readiness_prior"]
        self.assertTrue(info["used"])
        self.assertIsNone(info["reason"])
        self.assertEqual(info["method"], "vggt")
        self.assertAlmostEqual(info["bearing_deg"], 90.0)
        self.assertIsNone(info["inliers"])
        self.assertIsNone(info["reprojection_error_px"])
        self.assertIsNone(info["focal_px"])
        self.assertEqual(info["human_height_m"], 1.47)
        self.assertEqual(info["scale_iqr_ratio"], 1.16)
        self.assertEqual(info["stance_xy"], [0.3, -0.4])
        self.assertEqual(info["frames"], ["nav", "ready"])
        self.assertEqual(len(info["human_xy"]), 2)
        self.assertEqual(result["suggested_arm"], "left")
        json.dumps(metrics, allow_nan=False)

    def test_pnp_estimate_still_records_its_statistics(self) -> None:
        env = FakeDockingEnv(target_distance_m=1.20)
        prior = FakePrior(make_prior_estimate(math.pi / 2.0))
        controller = self.make_controller(env, FakeNav2Client(env), prior)

        result = controller.dock_to_visible_object("can")

        info = result["metrics"]["readiness_prior"]
        self.assertTrue(info["used"])
        self.assertEqual(info["method"], "pnp")
        self.assertEqual(info["inliers"], 57)
        self.assertAlmostEqual(info["reprojection_error_px"], 2.1)
        self.assertAlmostEqual(info["focal_px"], 553.9)
        self.assertIsNone(info["human_height_m"])
        self.assertIsNone(info["scale_iqr_ratio"])
        self.assertIsNone(info["stance_xy"])
        self.assertEqual(info["frames"], [])
        json.dumps(result["metrics"], allow_nan=False)

    def test_estimate_without_the_vggt_fields_records_them_as_absent(
        self,
    ) -> None:
        # An estimate from before the vggt fields existed: the metrics still
        # carry every key, with the absent ones None (or an empty frame list).
        estimate = make_prior_estimate(math.pi / 2.0)
        for name in (
            "stance_xy",
            "human_height_m",
            "scale",
            "scale_iqr_ratio",
            "frames",
            "service_time_s",
        ):
            delattr(estimate, name)
        env = FakeDockingEnv(target_distance_m=1.20)
        prior = FakePrior(estimate)
        controller = self.make_controller(env, FakeNav2Client(env), prior)

        result = controller.dock_to_visible_object("can")

        info = result["metrics"]["readiness_prior"]
        self.assertTrue(info["used"])
        self.assertEqual(info["inliers"], 57)
        self.assertIsNone(info["human_height_m"])
        self.assertIsNone(info["scale_iqr_ratio"])
        self.assertIsNone(info["stance_xy"])
        self.assertEqual(info["frames"], [])
        json.dumps(result["metrics"], allow_nan=False)

    def test_non_finite_frame_time_is_recorded_as_none(self) -> None:
        # The prior sets frame_time_s to NaN for an event without a frame
        # time; the metrics must still serialize with allow_nan=False and the
        # bearing must still be used.
        for label, overrides in (
            ("nan", {"frame_time_s": math.nan}),
            ("inf", {"frame_time_s": math.inf}),
            ("none", {"frame_time_s": None}),
        ):
            with self.subTest(label):
                env = FakeDockingEnv(target_distance_m=1.20)
                prior = FakePrior(make_prior_estimate(math.pi / 2.0, **overrides))
                controller = self.make_controller(env, FakeNav2Client(env), prior)

                result = controller.dock_to_visible_object("can")

                self.assertTrue(result["success"])
                self.assertEqual(result["metrics"]["bearing_source"], "prior")
                info = result["metrics"]["readiness_prior"]
                self.assertTrue(info["used"])
                self.assertIsNone(info["frame_time_s"])
                self.assertEqual(info["inliers"], 57)
                json.dumps(result["metrics"], allow_nan=False)

        # An estimate without the attribute at all is recorded the same way.
        estimate = make_prior_estimate(math.pi / 2.0)
        delattr(estimate, "frame_time_s")
        env = FakeDockingEnv(target_distance_m=1.20)
        controller = self.make_controller(env, FakeNav2Client(env), FakePrior(estimate))

        result = controller.dock_to_visible_object("can")

        self.assertIsNone(result["metrics"]["readiness_prior"]["frame_time_s"])
        json.dumps(result["metrics"], allow_nan=False)

    def test_prior_without_camera_height_keyword_reproduces_arrival_goal(
        self,
    ) -> None:
        env = FakeDockingEnv(target_distance_m=1.20)
        client = FakeNav2Client(env)
        prior = LegacyFakePrior(make_prior_estimate(math.pi / 2.0))
        controller = self.make_controller(env, client, prior)

        reference_env = FakeDockingEnv(target_distance_m=1.20)
        reference_client = FakeNav2Client(reference_env)
        reference = self.make_controller(reference_env, reference_client, None)

        result = controller.dock_to_visible_object("table")
        expected = reference.dock_to_visible_object("table")

        self.assertTrue(result["success"])
        self.assertEqual(client.goals[0][:3], reference_client.goals[0][:3])
        self.assertEqual(
            result["metrics"]["desired_camera_goal_xy_yaw"],
            expected["metrics"]["desired_camera_goal_xy_yaw"],
        )
        self.assertEqual(
            result["metrics"]["nav2_goal_xy_yaw"],
            expected["metrics"]["nav2_goal_xy_yaw"],
        )
        self.assertEqual(result["metrics"]["bearing_source"], "arrival")
        info = result["metrics"]["readiness_prior"]
        self.assertFalse(info["used"])
        self.assertTrue(info["reason"].startswith("exception:TypeError"))
        self.assertIn("camera_height_m", info["reason"])
        self.assertEqual(prior.calls, [])
        self.assertEqual(result["suggested_arm"], "left")

    def test_non_finite_prior_bearing_falls_back_to_arrival(self) -> None:
        env = FakeDockingEnv(target_distance_m=1.20)
        client = FakeNav2Client(env)
        prior = FakePrior(make_prior_estimate(math.nan))
        controller = self.make_controller(env, client, prior)

        result = controller.dock_to_visible_object("table")

        self.assertTrue(result["success"])
        self.assertEqual(result["metrics"]["bearing_source"], "arrival")
        self.assertEqual(
            result["metrics"]["readiness_prior"]["reason"], "invalid_bearing"
        )
        self.assertTrue(all(math.isfinite(v) for v in client.goals[0][:3]))

    def test_blocked_start_short_circuits_before_any_prior_attempt(self) -> None:
        env = FakeDockingEnv(target_distance_m=1.20)
        env.controller.nav2_start_clearance_status = lambda **_options: {
            "available": True,
            "blocked": True,
            "obstacle_points": 52,
            "max_points": 3,
        }
        client = FakeNav2Client(env)
        prior = FakePrior(make_prior_estimate(math.pi / 2.0))
        controller = self.make_controller(env, client, prior)

        result = controller.dock_to_visible_object("can")

        self.assertFalse(result["success"])
        self.assertIn("nav2_blocked_near_obstacle", result["reason"])
        self.assertEqual(client.goals, [])
        self.assertEqual(client.cancel_calls, 0)
        self.assertEqual(result["metrics"]["goal_attempts"], [])
        # Nothing was navigated, so no goal metrics describe a goal; the
        # prior's bearing is still on record as the reference.
        self.assertNotIn("bearing_source", result["metrics"])
        self.assertNotIn("nav2_goal_xy_yaw", result["metrics"])
        self.assertAlmostEqual(
            result["metrics"]["reference_bearing_rad"], math.pi / 2.0
        )
        self.assertTrue(result["metrics"]["readiness_prior"]["used"])

    def test_keyboard_interrupt_in_first_attempt_makes_no_second_attempt(
        self,
    ) -> None:
        env = FakeDockingEnv(target_distance_m=1.20)
        client = FakeNav2Client(env)
        calls = []

        def interrupt(*args, **_kwargs):
            calls.append(args)
            raise KeyboardInterrupt

        client.navigate_to_pose = interrupt
        prior = FakePrior(make_prior_estimate(math.pi / 2.0))
        controller = self.make_controller(env, client, prior)

        with self.assertRaises(KeyboardInterrupt):
            controller.dock_to_visible_object("can")

        self.assertEqual(len(calls), 1)
        self.assertEqual(client.cancel_calls, 1)
        self.assertEqual(env.controller.stops, 2)

    def test_registration_forwards_prior_to_nav2_backend_only(self) -> None:
        env = FakeDockingEnv(target_distance_m=1.20)
        client = FakeNav2Client(env)
        prior = FakePrior(make_prior_estimate(math.pi / 2.0))
        registry = PrimitiveRegistry()
        register_visible_object_navigation_primitives(
            registry,
            env,
            docking_config={"backend": "nav2"},
            segment_client_factory=lambda: env.segment,
            nav2_client_factory=lambda _config: client,
            readiness_prior=prior,
        )

        dock = registry.functions()["dock_to_visible_object"]
        result = dock("can")

        self.assertEqual(len(prior.calls), 1)
        self.assertEqual(result["suggested_arm"], "left")
        self.assertEqual(result["metrics"]["bearing_source"], "prior")
        self.assertIn("suggested_arm", dock.__doc__)
        self.assertIn("prepare_for_manipulation", dock.__doc__)

        legacy_registry = PrimitiveRegistry()
        register_visible_object_navigation_primitives(
            legacy_registry,
            FakeDockingEnv(),
            docking_config={"backend": "legacy"},
            readiness_prior=prior,
        )
        self.assertIn("dock_to_visible_object", legacy_registry.functions())

    def test_from_mapping_accepts_readiness_prior_block(self) -> None:
        block = {
            "enabled": False,
            "events_path": None,
            "source": "ready_frame",
            "min_inliers": 30,
        }

        nav2_config = Nav2VisibleObjectDockingConfig.from_mapping(
            {"backend": "nav2", "readiness_prior": block, "docking_distance_m": 0.7}
        )
        legacy_config = VisibleObjectDockingConfig.from_mapping(
            {"readiness_prior": block, "docking_distance_m": 0.7}
        )

        self.assertEqual(nav2_config.docking_distance_m, 0.7)
        self.assertEqual(legacy_config.docking_distance_m, 0.7)
        self.assertFalse(hasattr(nav2_config, "readiness_prior"))
        with self.assertRaisesRegex(ValueError, "unknown Nav2 docking settings"):
            Nav2VisibleObjectDockingConfig.from_mapping({"readiness_priors": block})


class ExplicitBearingAndGoalDistanceTest(unittest.TestCase):
    """A scripted approach bearing and the goal-distance schedule.

    As above, ``FakeDockingEnv`` starts at (0, 0, 0) with the target straight
    ahead, so the arrival bearing is pi. Its detected target distance is a
    fixed fraction of the depth it renders (about 0.89), which the arrival
    depths below account for.
    """

    def make_controller(self, env, client, prior=None, **settings):
        return Nav2VisibleObjectDockingController(
            env,
            docking_config={"backend": "nav2", **settings},
            segment_client_factory=lambda: env.segment,
            nav2_client_factory=lambda _config: client,
            readiness_prior=prior,
        )

    @staticmethod
    def nav2_history(result):
        return [
            entry
            for entry in result["motion_history"]
            if entry["primitive"] == "nav2_navigate_to_pose"
        ]

    def assert_goal_on_bearing(self, controller, goal, result, bearing, distance):
        target_x, target_y = result["metrics"]["target_world_xy_m"]
        expected = controller._base_goal_from_camera_goal(
            target_x + distance * math.cos(bearing),
            target_y + distance * math.sin(bearing),
            math.atan2(-math.sin(bearing), -math.cos(bearing)),
        )
        goal_x, goal_y, goal_yaw = goal[:3]
        self.assertAlmostEqual(goal_x, expected[0])
        self.assertAlmostEqual(goal_y, expected[1])
        yaw_difference = math.atan2(
            math.sin(goal_yaw - expected[2]), math.cos(goal_yaw - expected[2])
        )
        self.assertAlmostEqual(yaw_difference, 0.0)

    def test_explicit_bearing_is_the_only_attempt_and_skips_the_prior(self) -> None:
        env = FakeDockingEnv(target_distance_m=1.20)
        client = FakeNav2Client(env)
        prior = FakePrior(make_prior_estimate(0.0))
        controller = self.make_controller(env, client, prior)

        result = controller.dock_to_visible_object("can", approach_bearing_deg=90.0)

        self.assertTrue(result["success"])
        self.assertEqual(result["reason"], "within_docking_distance")
        self.assertEqual(prior.calls, [])
        self.assertEqual(len(client.goals), 1)
        self.assertEqual(client.plan_checks, [])
        goal_distance = controller.config.effective_nav2_goal_distance_m
        self.assert_goal_on_bearing(
            controller, client.goals[0], result, math.pi / 2.0, goal_distance
        )
        metrics = result["metrics"]
        self.assertEqual(
            metrics["readiness_prior"], {"used": False, "reason": "explicit_bearing"}
        )
        self.assertEqual(metrics["approach_bearing_deg"], 90.0)
        self.assertAlmostEqual(metrics["reference_bearing_rad"], math.pi / 2.0)
        self.assertAlmostEqual(metrics["arrival_bearing_rad"], math.pi)
        self.assertEqual(metrics["bearing_source"], "explicit")
        self.assertEqual(metrics["goal_distance_used_m"], goal_distance)
        attempts = metrics["goal_attempts"]
        self.assertEqual(len(attempts), 1)
        self.assertEqual(attempts[0]["bearing_source"], "explicit")
        self.assertAlmostEqual(attempts[0]["bearing_rad"], math.pi / 2.0)
        self.assertEqual(attempts[0]["goal_distance_m"], goal_distance)
        self.assertEqual(attempts[0]["plan_checks"], [])
        self.assertTrue(attempts[0]["success"])
        history = self.nav2_history(result)
        self.assertEqual(history[0]["bearing_source"], "explicit")
        self.assertEqual(history[0]["goal_distance_m"], goal_distance)
        # No human hand to consult: the target's side alone names the arm.
        self.assertEqual(result["suggested_arm"], "left")
        self.assertEqual(metrics["suggested_arm_source"], "target_side")
        self.assertNotIn("human_hand_arm", metrics)
        json.dumps(metrics, allow_nan=False)

    def test_explicit_bearing_failure_has_no_offset_or_arrival_fallback(self) -> None:
        env = FakeDockingEnv(target_distance_m=1.20)
        client = FakeNav2Client(env, success=False)
        prior = FakePrior(make_prior_estimate(math.pi / 2.0))
        controller = self.make_controller(env, client, prior)

        result = controller.dock_to_visible_object("can", approach_bearing_deg=90.0)

        self.assertFalse(result["success"])
        self.assertEqual(result["reason"], "nav2_failed:aborted")
        self.assertEqual(len(client.goals), 1)
        self.assertEqual(
            [attempt["bearing_source"] for attempt in result["metrics"]["goal_attempts"]],
            ["explicit"],
        )
        self.assertEqual(client.cancel_calls, 1)
        self.assertEqual(env.controller.stops, 2)

    def test_explicit_bearing_wraps_and_must_be_finite(self) -> None:
        env = FakeDockingEnv(target_distance_m=1.20)
        controller = self.make_controller(env, FakeNav2Client(env))

        result = controller.dock_to_visible_object("can", approach_bearing_deg=450.0)

        self.assertAlmostEqual(result["metrics"]["reference_bearing_rad"], math.pi / 2.0)
        self.assertEqual(result["metrics"]["approach_bearing_deg"], 450.0)
        stops_before = env.controller.stops
        for bad in (math.nan, math.inf, "east"):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    controller.dock_to_visible_object("can", approach_bearing_deg=bad)
        # Rejected before the entry stop, like a bad object name.
        self.assertEqual(env.controller.stops, stops_before)

    def test_explicit_shortcut_needs_alignment_within_tolerance(self) -> None:
        env = FakeDockingEnv(target_distance_m=0.50)
        client = FakeNav2Client(env)
        controller = self.make_controller(env, client, docking_distance_m=0.55)

        # 20 deg from the arrival bearing: inside the 22.5 deg default.
        result = controller.dock_to_visible_object("can", approach_bearing_deg=160.0)

        self.assertTrue(result["success"])
        self.assertEqual(result["reason"], "within_docking_distance")
        self.assertEqual(client.goals, [])
        self.assertNotIn("goal_attempts", result["metrics"])
        self.assertAlmostEqual(
            result["metrics"]["reference_bearing_rad"], math.radians(160.0)
        )
        self.assertEqual(result["metrics"]["approach_bearing_deg"], 160.0)

        # 40 deg off: within distance, but on the wrong side, so re-dock.
        env = FakeDockingEnv(target_distance_m=0.50)
        client = FakeNav2Client(env)
        controller = self.make_controller(env, client, docking_distance_m=0.55)

        result = controller.dock_to_visible_object("can", approach_bearing_deg=140.0)

        self.assertTrue(result["success"])
        self.assertEqual(len(client.goals), 1)
        self.assert_goal_on_bearing(
            controller,
            client.goals[0],
            result,
            math.radians(140.0),
            controller.config.effective_nav2_goal_distance_m,
        )
        self.assertEqual(result["metrics"]["bearing_source"], "explicit")

        # A wider configured tolerance admits the same 40 deg.
        env = FakeDockingEnv(target_distance_m=0.50)
        client = FakeNav2Client(env)
        controller = self.make_controller(
            env,
            client,
            docking_distance_m=0.55,
            explicit_bearing_alignment_tolerance_deg=45.0,
        )

        result = controller.dock_to_visible_object("can", approach_bearing_deg=140.0)

        self.assertTrue(result["success"])
        self.assertEqual(client.goals, [])

    def test_plan_check_navigates_the_first_plannable_distance(self) -> None:
        env = FakeDockingEnv(target_distance_m=1.20)
        client = FakeNav2Client(env, plannable=[False, True], arrival_distance_m=1.10)
        events = []
        controller = Nav2VisibleObjectDockingController(
            env,
            docking_config={
                "backend": "nav2",
                "docking_distance_m": 0.65,
                "nav2_goal_distance_m": 0.60,
                "nav2_goal_distance_fallbacks_m": [0.90],
                "plan_check_before_goal": True,
            },
            segment_client_factory=lambda: env.segment,
            nav2_client_factory=lambda _config: client,
            progress_callback=events.append,
        )
        self.assertEqual(controller.config.nav2_goal_distance_fallbacks_m, (0.90,))

        result = controller.dock_to_visible_object("can", approach_bearing_deg=90.0)

        self.assertTrue(result["success"])
        self.assertEqual(result["reason"], "docked_at_fallback_distance")
        self.assertEqual(len(client.goals), 1)
        self.assert_goal_on_bearing(controller, client.goals[0], result, math.pi / 2.0, 0.90)
        # Both distances were checked in order; only the second was driven.
        self.assertEqual(len(client.plan_checks), 2)
        self.assert_goal_on_bearing(
            controller, client.plan_checks[0], result, math.pi / 2.0, 0.60
        )
        self.assertEqual(client.plan_checks[1][:3], client.goals[0][:3])
        self.assertEqual(client.plan_checks[0][3]["frame_id"], "odom")
        metrics = result["metrics"]
        self.assertEqual(metrics["goal_distance_used_m"], 0.90)
        self.assertEqual(metrics["nav2_goal_distance_m"], 0.60)
        attempts = metrics["goal_attempts"]
        self.assertEqual(len(attempts), 1)
        self.assertEqual(attempts[0]["goal_distance_m"], 0.90)
        self.assertTrue(attempts[0]["success"])
        self.assertEqual(
            [(check["goal_distance_m"], check["plannable"], check["reason"], check["path_length_m"])
             for check in attempts[0]["plan_checks"]],
            [(0.60, False, "no_path", None), (0.90, True, "planned", 1.5)],
        )
        self.assertEqual(
            attempts[0]["plan_checks"][1]["nav2_goal_xy_yaw"], attempts[0]["nav2_goal_xy_yaw"]
        )
        self.assertEqual(
            [event["plannable"] for event in events if event["state"] == "plan_checked"],
            [False, True],
        )
        self.assertEqual(result["suggested_arm"], "left")
        self.assertEqual(client.halt_calls, 1)
        json.dumps(metrics, allow_nan=False)

    def test_fallback_distance_moves_the_verification_boundary_out(self) -> None:
        # The same arrival depth fails the inner rule and passes the fallback
        # rule: the boundary moves out by the fallback's extra distance.
        settings = {
            "docking_distance_m": 0.65,
            "nav2_goal_distance_m": 0.60,
            "nav2_goal_distance_fallbacks_m": [0.90],
        }
        env = FakeDockingEnv(target_distance_m=1.20)
        client = FakeNav2Client(env, arrival_distance_m=1.10)
        controller = self.make_controller(env, client, **settings)
        result = controller.dock_to_visible_object("can", approach_bearing_deg=90.0)
        self.assertFalse(result["success"])
        self.assertEqual(result["reason"], "final_visual_verification_failed")
        self.assertEqual(result["metrics"]["goal_distance_used_m"], 0.60)

        env = FakeDockingEnv(target_distance_m=1.20)
        client = ScriptedNav2Client(env, [False, True])
        client.arrival_distance_m = 1.10
        controller = self.make_controller(env, client, **settings)
        result = controller.dock_to_visible_object("can", approach_bearing_deg=90.0)
        self.assertTrue(result["success"])
        self.assertEqual(result["reason"], "docked_at_fallback_distance")
        self.assertEqual(result["metrics"]["goal_distance_used_m"], 0.90)

        # Beyond the moved boundary the fallback still fails verification.
        env = FakeDockingEnv(target_distance_m=1.20)
        client = ScriptedNav2Client(env, [False, True])
        client.arrival_distance_m = 1.20
        controller = self.make_controller(env, client, **settings)
        result = controller.dock_to_visible_object("can", approach_bearing_deg=90.0)
        self.assertFalse(result["success"])
        self.assertEqual(result["reason"], "final_visual_verification_failed")
        self.assertEqual(result["metrics"]["goal_distance_used_m"], 0.90)
        self.assertNotIn("suggested_arm", result)

    def test_plan_check_off_navigates_each_distance_in_turn(self) -> None:
        env = FakeDockingEnv(target_distance_m=1.20)
        client = ScriptedNav2Client(env, [False, True])
        controller = self.make_controller(
            env,
            client,
            docking_distance_m=0.65,
            nav2_goal_distance_m=0.60,
            nav2_goal_distance_fallbacks_m=[0.90],
        )

        result = controller.dock_to_visible_object("can", approach_bearing_deg=90.0)

        self.assertTrue(result["success"])
        self.assertEqual(result["reason"], "docked_at_fallback_distance")
        self.assertEqual(client.plan_checks, [])
        self.assertEqual(len(client.goals), 2)
        self.assert_goal_on_bearing(controller, client.goals[0], result, math.pi / 2.0, 0.60)
        self.assert_goal_on_bearing(controller, client.goals[1], result, math.pi / 2.0, 0.90)
        attempts = result["metrics"]["goal_attempts"]
        self.assertEqual(
            [(a["bearing_source"], a["goal_distance_m"], a["success"], a["plan_checks"]) for a in attempts],
            [("explicit", 0.60, False, []), ("explicit", 0.90, True, [])],
        )
        self.assertEqual(
            [entry["goal_distance_m"] for entry in self.nav2_history(result)], [0.60, 0.90]
        )
        self.assertEqual(result["metrics"]["goal_distance_used_m"], 0.90)

    def test_nothing_plannable_fails_without_driving(self) -> None:
        env = FakeDockingEnv(target_distance_m=1.20)
        client = FakeNav2Client(env, plannable=[False, False])
        controller = self.make_controller(
            env,
            client,
            nav2_goal_distance_fallbacks_m=[0.90],
            plan_check_before_goal=True,
        )

        result = controller.dock_to_visible_object("can", approach_bearing_deg=90.0)

        self.assertFalse(result["success"])
        self.assertEqual(result["reason"], "no_plannable_goal_on_bearing")
        self.assertEqual(client.goals, [])
        self.assertEqual(len(client.plan_checks), 2)
        metrics = result["metrics"]
        attempts = metrics["goal_attempts"]
        self.assertEqual(len(attempts), 1)
        self.assertEqual(attempts[0]["bearing_source"], "explicit")
        self.assertFalse(attempts[0]["success"])
        self.assertEqual(attempts[0]["reason"], "no_plannable_goal_on_bearing")
        self.assertIsNone(attempts[0]["goal_distance_m"])
        self.assertIsNone(attempts[0]["nav2_goal_xy_yaw"])
        self.assertIsNone(attempts[0]["camera_goal_xy_yaw"])
        self.assertFalse(attempts[0]["navigated"])
        self.assertEqual(
            [check["plannable"] for check in attempts[0]["plan_checks"]], [False, False]
        )
        # No goal was navigated, so the goal metrics say so.
        self.assertIsNone(metrics["bearing_source"])
        self.assertIsNone(metrics["nav2_goal_xy_yaw"])
        self.assertIsNone(metrics["desired_camera_goal_xy_yaw"])
        self.assertNotIn("goal_distance_used_m", metrics)
        self.assertNotIn("recommended_recovery", metrics)
        self.assertEqual(self.nav2_history(result), [])
        # The client existed for the plan checks, so the failure path latches it.
        self.assertEqual(client.cancel_calls, 1)
        self.assertEqual(client.halt_calls, 0)
        self.assertEqual(env.controller.stops, 2)
        json.dumps(metrics, allow_nan=False)

    def test_prior_mode_moves_to_the_next_bearing_when_none_is_plannable(self) -> None:
        env = FakeDockingEnv(target_distance_m=1.20)
        client = FakeNav2Client(env, plannable=[False, False, True])
        prior = FakePrior(make_prior_estimate(math.pi / 2.0))
        controller = self.make_controller(
            env,
            client,
            prior,
            nav2_goal_distance_fallbacks_m=[0.90],
            plan_check_before_goal=True,
        )

        result = controller.dock_to_visible_object("can")

        self.assertTrue(result["success"])
        self.assertEqual(result["reason"], "within_docking_distance")
        self.assertEqual(len(client.goals), 1)
        self.assertEqual(len(client.plan_checks), 3)
        attempts = result["metrics"]["goal_attempts"]
        self.assertEqual(
            [(a["bearing_source"], a["reason"], a["goal_distance_m"], len(a["plan_checks"])) for a in attempts],
            [
                ("prior", "no_plannable_goal_on_bearing", None, 2),
                ("prior_offset", "succeeded", 0.60, 1),
            ],
        )
        self.assertEqual(result["metrics"]["bearing_source"], "prior_offset")
        self.assert_goal_on_bearing(
            controller, client.goals[0], result, 3.0 * math.pi / 4.0, 0.60
        )
        self.assertEqual(result["metrics"]["goal_distance_used_m"], 0.60)

    def test_prior_without_offsets_or_arrival_fallback_makes_one_attempt(self) -> None:
        env = FakeDockingEnv(target_distance_m=1.20)
        client = FakeNav2Client(env, success=False)
        prior = FakePrior(
            make_prior_estimate(math.pi / 2.0),
            retry_bearing_offsets_deg=(),
            fallback_to_arrival_bearing=False,
        )
        controller = self.make_controller(env, client, prior)

        result = controller.dock_to_visible_object("can")

        self.assertFalse(result["success"])
        self.assertEqual(result["reason"], "nav2_failed:aborted")
        self.assertEqual(len(client.goals), 1)
        self.assertEqual(
            [a["bearing_source"] for a in result["metrics"]["goal_attempts"]], ["prior"]
        )

        # With the switch left on, the arrival bearing still closes the list.
        env = FakeDockingEnv(target_distance_m=1.20)
        client = FakeNav2Client(env, success=False)
        prior = FakePrior(make_prior_estimate(math.pi / 2.0), retry_bearing_offsets_deg=())
        controller = self.make_controller(env, client, prior)

        result = controller.dock_to_visible_object("can")

        self.assertEqual(
            [a["bearing_source"] for a in result["metrics"]["goal_attempts"]],
            ["prior", "arrival_fallback"],
        )

    def test_reference_bearing_is_the_first_attempts_bearing(self) -> None:
        env = FakeDockingEnv(target_distance_m=1.20)
        prior = FakePrior(make_prior_estimate(math.pi / 2.0))
        controller = self.make_controller(env, FakeNav2Client(env), prior)
        result = controller.dock_to_visible_object("can")
        self.assertAlmostEqual(result["metrics"]["reference_bearing_rad"], math.pi / 2.0)

        env = FakeDockingEnv(target_distance_m=1.20)
        controller = self.make_controller(env, FakeNav2Client(env), None)
        result = controller.dock_to_visible_object("can")
        self.assertAlmostEqual(result["metrics"]["reference_bearing_rad"], math.pi)
        self.assertAlmostEqual(
            result["metrics"]["reference_bearing_rad"],
            result["metrics"]["arrival_bearing_rad"],
        )

        env = FakeDockingEnv(target_distance_m=1.20)
        prior = FakePrior(None, last_reason="no_event_match")
        controller = self.make_controller(env, FakeNav2Client(env), prior)
        result = controller.dock_to_visible_object("can")
        self.assertAlmostEqual(result["metrics"]["reference_bearing_rad"], math.pi)

        # Recorded also when already docked and no goal is sent.
        env = FakeDockingEnv(target_distance_m=0.50)
        client = FakeNav2Client(env)
        prior = FakePrior(make_prior_estimate(math.pi))
        controller = self.make_controller(env, client, prior, docking_distance_m=0.55)
        result = controller.dock_to_visible_object("can")
        self.assertEqual(client.goals, [])
        self.assertAlmostEqual(result["metrics"]["reference_bearing_rad"], math.pi)

    def test_operator_stop_ends_every_remaining_goal_and_bearing(self) -> None:
        env = FakeDockingEnv(target_distance_m=1.20)
        client = FakeNav2Client(env, success=False)

        def operator_stop(x_m, y_m, yaw_rad, **options):
            client.goals.append((x_m, y_m, yaw_rad, options))
            return {"success": False, "reason": "operator_stop", "feedback": {}}

        client.navigate_to_pose = operator_stop
        prior = FakePrior(make_prior_estimate(math.pi / 2.0))
        controller = self.make_controller(
            env, client, prior, nav2_goal_distance_fallbacks_m=[0.90]
        )

        result = controller.dock_to_visible_object("can")

        self.assertFalse(result["success"])
        self.assertEqual(result["reason"], "nav2_failed:operator_stop")
        # Neither the farther distance nor the offset and arrival bearings.
        self.assertEqual(len(client.goals), 1)
        attempts = result["metrics"]["goal_attempts"]
        self.assertEqual(
            [(a["bearing_source"], a["goal_distance_m"], a["reason"]) for a in attempts],
            [("prior", 0.60, "operator_stop")],
        )
        self.assertNotIn("recommended_recovery", result["metrics"])
        self.assertEqual(client.cancel_calls, 1)
        self.assertEqual(env.controller.stops, 2)

    def test_goal_distance_settings_are_validated(self) -> None:
        config = Nav2VisibleObjectDockingConfig.from_mapping(None)
        self.assertEqual(config.nav2_goal_distance_fallbacks_m, ())
        self.assertFalse(config.plan_check_before_goal)
        self.assertEqual(config.explicit_bearing_alignment_tolerance_deg, 22.5)
        self.assertEqual(config.nav2_goal_distance_schedule_m, (0.60,))

        config = Nav2VisibleObjectDockingConfig.from_mapping(
            {
                "docking_distance_m": 0.80,
                "nav2_goal_distance_m": 0.72,
                "nav2_goal_distance_fallbacks_m": [0.75, 1.2],
                "plan_check_before_goal": True,
                "explicit_bearing_alignment_tolerance_deg": 180.0,
            }
        )
        self.assertEqual(config.nav2_goal_distance_fallbacks_m, (0.75, 1.2))
        self.assertIsInstance(config.nav2_goal_distance_fallbacks_m, tuple)
        self.assertEqual(config.nav2_goal_distance_schedule_m, (0.72, 0.75, 1.2))
        self.assertTrue(config.plan_check_before_goal)

        for label, settings in (
            ("not above goal", {"nav2_goal_distance_fallbacks_m": [0.60]}),
            ("not ascending", {"nav2_goal_distance_fallbacks_m": [1.2, 0.9]}),
            ("repeated", {"nav2_goal_distance_fallbacks_m": [0.9, 0.9]}),
            ("too far", {"nav2_goal_distance_fallbacks_m": [3.5]}),
            ("nan", {"nav2_goal_distance_fallbacks_m": [math.nan]}),
            ("string entry", {"nav2_goal_distance_fallbacks_m": ["0.9"]}),
            ("bool entry", {"nav2_goal_distance_fallbacks_m": [True]}),
            ("scalar", {"nav2_goal_distance_fallbacks_m": 0.9}),
            ("plan check string", {"plan_check_before_goal": "yes"}),
            ("zero tolerance", {"explicit_bearing_alignment_tolerance_deg": 0.0}),
            ("wide tolerance", {"explicit_bearing_alignment_tolerance_deg": 181.0}),
            ("nan tolerance", {"explicit_bearing_alignment_tolerance_deg": math.nan}),
        ):
            with self.subTest(label):
                with self.assertRaisesRegex(ValueError, next(iter(settings))):
                    Nav2VisibleObjectDockingConfig.from_mapping(settings)

    def test_primitive_passes_the_bearing_through_and_documents_it(self) -> None:
        env = FakeDockingEnv(target_distance_m=1.20)
        client = FakeNav2Client(env)
        prior = FakePrior(make_prior_estimate(0.0))
        registry = PrimitiveRegistry()
        register_visible_object_navigation_primitives(
            registry,
            env,
            docking_config={"backend": "nav2"},
            segment_client_factory=lambda: env.segment,
            nav2_client_factory=lambda _config: client,
            readiness_prior=prior,
        )
        dock = registry.functions()["dock_to_visible_object"]

        result = dock("can", approach_bearing_deg=90.0)

        self.assertEqual(result["metrics"]["approach_bearing_deg"], 90.0)
        self.assertEqual(result["metrics"]["bearing_source"], "explicit")
        self.assertEqual(prior.calls, [])
        # Omitted, the prior chooses as before.
        env.target_distance_m = 1.20
        result = dock("can")
        self.assertEqual(result["metrics"]["bearing_source"], "prior")
        self.assertEqual(len(prior.calls), 1)
        documentation = registry.documentation()
        self.assertIn(
            "def dock_to_visible_object(object_name: str, *, "
            "approach_bearing_deg: float | None = None)",
            documentation,
        )
        self.assertIn("approach_bearing_deg: Optional object-centric", documentation)


class StopWindowAndFallbackTest(unittest.TestCase):
    """Operator stops that land while no goal is active, plan checks the
    planner could not answer, fallbacks the base already stands at, and the
    goal metrics that must only describe a navigated goal.

    ``FakeDockingEnv`` starts at (0, 0, 0) with the target straight ahead,
    so the arrival bearing is pi; its detected target distance is about
    0.89 of the rendered depth.
    """

    schedule = {
        "docking_distance_m": 0.65,
        "nav2_goal_distance_m": 0.60,
        "nav2_goal_distance_fallbacks_m": [0.75, 0.90],
    }

    def make_controller(self, env, client, prior=None, **settings):
        return Nav2VisibleObjectDockingController(
            env,
            docking_config={"backend": "nav2", **settings},
            segment_client_factory=lambda: env.segment,
            nav2_client_factory=lambda _config: client,
            readiness_prior=prior,
        )

    @staticmethod
    def attempt_rows(result):
        return [
            (a["bearing_source"], a["goal_distance_m"], a["reason"], a["navigated"])
            for a in result["metrics"]["goal_attempts"]
        ]

    def test_stop_during_the_plan_check_sends_no_goal(self) -> None:
        env = FakeDockingEnv(target_distance_m=1.20)
        client = FakeNav2Client(env)
        events = []

        def stop_while_planning(x_m, y_m, yaw_rad, **options):
            # The operator's Stop reaches the client mid-check: the plan
            # itself still succeeds, as the real client's does.
            client.cancel_and_stop()
            return FakeNav2Client.compute_path_to_pose(
                client, x_m, y_m, yaw_rad, **options
            )

        client.compute_path_to_pose = stop_while_planning
        controller = Nav2VisibleObjectDockingController(
            env,
            docking_config={
                "backend": "nav2",
                **self.schedule,
                "plan_check_before_goal": True,
            },
            segment_client_factory=lambda: env.segment,
            nav2_client_factory=lambda _config: client,
            progress_callback=events.append,
        )

        result = controller.dock_to_visible_object("can", approach_bearing_deg=90.0)

        self.assertFalse(result["success"])
        self.assertEqual(result["reason"], "nav2_failed:operator_stop")
        self.assertEqual(client.goals, [])
        self.assertEqual(len(client.plan_checks), 1)
        self.assertEqual(
            self.attempt_rows(result), [("explicit", None, "operator_stop", False)]
        )
        attempt = result["metrics"]["goal_attempts"][0]
        self.assertEqual(len(attempt["plan_checks"]), 1)
        self.assertTrue(attempt["plan_checks"][0]["plannable"])
        self.assertIsNone(attempt["nav2_goal_xy_yaw"])
        # Nothing navigated: no goal metrics, no recovery advice, no history.
        self.assertNotIn("nav2_goal_xy_yaw", result["metrics"])
        self.assertNotIn("bearing_source", result["metrics"])
        self.assertNotIn("goal_distance_used_m", result["metrics"])
        self.assertNotIn("recommended_recovery", result["metrics"])
        self.assertEqual(
            [e for e in result["motion_history"] if e["primitive"] == "nav2_navigate_to_pose"],
            [],
        )
        self.assertIn("stop_requested", [event["state"] for event in events])
        # The Stop's own cancel plus the failure path's final cancel.
        self.assertEqual(client.cancel_calls, 2)
        self.assertEqual(client.halt_calls, 0)
        self.assertEqual(env.controller.stops, 2)
        json.dumps(result["metrics"], allow_nan=False)

    def test_stop_between_two_goal_distances_sends_no_second_goal(self) -> None:
        env = FakeDockingEnv(target_distance_m=1.20)
        client = FakeNav2Client(env, success=False)

        def abort_then_stop(x_m, y_m, yaw_rad, **options):
            # The goal ends on its own; the operator's Stop lands right
            # after, before the next distance is sent.
            outcome = FakeNav2Client.navigate_to_pose(
                client, x_m, y_m, yaw_rad, **options
            )
            client.cancel_and_stop()
            return outcome

        client.navigate_to_pose = abort_then_stop
        prior = FakePrior(make_prior_estimate(math.pi / 2.0))
        controller = self.make_controller(env, client, prior, **self.schedule)

        result = controller.dock_to_visible_object("can")

        self.assertFalse(result["success"])
        self.assertEqual(result["reason"], "nav2_failed:operator_stop")
        # One goal drove; neither the farther distances nor the offset and
        # arrival bearings were sent.
        self.assertEqual(len(client.goals), 1)
        self.assertEqual(
            self.attempt_rows(result),
            [("prior", 0.60, "aborted", True), ("prior", None, "operator_stop", False)],
        )
        # The goal metrics keep describing the goal that was navigated.
        metrics = result["metrics"]
        self.assertEqual(metrics["bearing_source"], "prior")
        self.assertEqual(
            metrics["nav2_goal_xy_yaw"], metrics["goal_attempts"][0]["nav2_goal_xy_yaw"]
        )
        self.assertEqual(metrics["goal_distance_used_m"], 0.60)
        self.assertNotIn("recommended_recovery", metrics)

    def test_stale_stop_flag_from_the_previous_dock_does_not_block(self) -> None:
        env = FakeDockingEnv(target_distance_m=1.20)
        client = FakeNav2Client(env, success=False)
        controller = self.make_controller(env, client, plan_check_before_goal=True)

        first = controller.dock_to_visible_object("can", approach_bearing_deg=90.0)

        self.assertFalse(first["success"])
        self.assertEqual(first["reason"], "nav2_failed:aborted")
        # The failure path's final cancel_and_stop left the flag set.
        self.assertTrue(client.stop_flag)
        self.assertEqual(client.reset_calls, 1)

        client.success = True
        env.target_distance_m = 1.20
        second = controller.dock_to_visible_object("can", approach_bearing_deg=90.0)

        self.assertTrue(second["success"])
        self.assertEqual(len(client.goals), 2)
        self.assertEqual(client.reset_calls, 2)
        self.assertEqual(
            self.attempt_rows(second), [("explicit", 0.60, "succeeded", True)]
        )

    def test_planner_failures_that_say_nothing_about_the_goal_fall_through(
        self,
    ) -> None:
        for reason in ("server_unavailable", "goal_response_timeout", "timeout"):
            with self.subTest(reason):
                env = FakeDockingEnv(target_distance_m=1.20)
                client = FakeNav2Client(env, plannable=[reason])
                controller = self.make_controller(
                    env,
                    client,
                    **self.schedule,
                    plan_check_before_goal=True,
                )

                result = controller.dock_to_visible_object(
                    "can", approach_bearing_deg=90.0
                )

                # The inner goal was sent and reached, as Nav2 alone would.
                self.assertTrue(result["success"])
                self.assertEqual(result["reason"], "within_docking_distance")
                self.assertEqual(len(client.goals), 1)
                self.assertEqual(len(client.plan_checks), 1)
                self.assertEqual(client.plan_checks[0][:3], client.goals[0][:3])
                attempt = result["metrics"]["goal_attempts"][0]
                self.assertEqual(attempt["goal_distance_m"], 0.60)
                self.assertEqual(
                    attempt["plan_checks"],
                    [
                        {
                            "goal_distance_m": 0.60,
                            "plannable": None,
                            "reason": reason,
                            "path_length_m": None,
                            "nav2_goal_xy_yaw": attempt["nav2_goal_xy_yaw"],
                        }
                    ],
                )
                json.dumps(result["metrics"], allow_nan=False)

        # A rejected planning goal is as unreachable as no path: skipped.
        env = FakeDockingEnv(target_distance_m=1.20)
        client = FakeNav2Client(env, plannable=["goal_rejected", True])
        client.arrival_distance_m = 0.85
        controller = self.make_controller(
            env, client, **self.schedule, plan_check_before_goal=True
        )

        result = controller.dock_to_visible_object("can", approach_bearing_deg=90.0)

        self.assertTrue(result["success"])
        self.assertEqual(result["reason"], "docked_at_fallback_distance")
        self.assertEqual(len(client.goals), 1)
        self.assertEqual(result["metrics"]["goal_distance_used_m"], 0.75)
        self.assertEqual(
            [(c["goal_distance_m"], c["plannable"], c["reason"])
             for c in result["metrics"]["goal_attempts"][0]["plan_checks"]],
            [(0.60, False, "goal_rejected"), (0.75, True, "planned")],
        )

    def test_planner_action_name_reaches_the_client(self) -> None:
        env = FakeDockingEnv(target_distance_m=1.20)
        default = Nav2VisibleObjectDockingController(
            env, docking_config={"backend": "nav2"}, segment_client_factory=lambda: env.segment
        )
        custom = Nav2VisibleObjectDockingController(
            env,
            docking_config={"backend": "nav2", "nav2_planner_action_name": "plan_only"},
            segment_client_factory=lambda: env.segment,
        )
        self.assertEqual(default.config.nav2_planner_action_name, "compute_path_to_pose")
        self.assertEqual(custom.config.nav2_planner_action_name, "plan_only")

        with mock.patch(
            "yor_agent.robot.nav2_visible_object_navigation.Nav2Client"
        ) as client_type:
            default._default_client(default.config)
            custom._default_client(custom.config)

        self.assertEqual(
            [call.kwargs["planner_action_name"] for call in client_type.call_args_list],
            ["compute_path_to_pose", "plan_only"],
        )
        self.assertEqual(
            client_type.call_args_list[0].kwargs["action_name"], "navigate_to_pose"
        )
        with self.assertRaisesRegex(ValueError, "action"):
            Nav2VisibleObjectDockingConfig.from_mapping(
                {"nav2_planner_action_name": ""}
            )

    def test_fallback_the_base_already_stands_at_docks_in_place(self) -> None:
        # Detected about 0.80 m from the target on the arrival bearing.
        env = FakeDockingEnv(target_distance_m=0.90)
        client = FakeNav2Client(env, plannable=[False, False])
        events = []
        controller = Nav2VisibleObjectDockingController(
            env,
            docking_config={
                "backend": "nav2",
                **self.schedule,
                "plan_check_before_goal": True,
            },
            segment_client_factory=lambda: env.segment,
            nav2_client_factory=lambda _config: client,
            progress_callback=events.append,
        )

        result = controller.dock_to_visible_object("can", approach_bearing_deg=180.0)

        self.assertTrue(result["success"])
        self.assertEqual(result["reason"], "docked_at_fallback_distance")
        self.assertEqual(client.goals, [])
        metrics = result["metrics"]
        current_distance = metrics["target_distance_m"]
        self.assertAlmostEqual(current_distance, 0.80, delta=0.01)
        # 0.60 and 0.75 lie inside the current distance and were checked;
        # 0.90 does not, so it was neither checked nor driven to.
        self.assertEqual(len(client.plan_checks), 2)
        self.assertEqual(
            self.attempt_rows(result),
            [("explicit", current_distance, "docked_at_fallback_distance", False)],
        )
        attempt = metrics["goal_attempts"][0]
        self.assertTrue(attempt["success"])
        self.assertIsNone(attempt["nav2_goal_xy_yaw"])
        self.assertIsNone(attempt["camera_goal_xy_yaw"])
        self.assertEqual(
            [(c["goal_distance_m"], c["plannable"]) for c in attempt["plan_checks"]],
            [(0.60, False), (0.75, False)],
        )
        self.assertEqual(metrics["goal_distance_used_m"], current_distance)
        self.assertIsNone(metrics["bearing_source"])
        self.assertIsNone(metrics["nav2_goal_xy_yaw"])
        self.assertIsNone(metrics["desired_camera_goal_xy_yaw"])
        self.assertNotIn("nav2", metrics)
        # The arrival check re-detected the target from where it stood.
        self.assertAlmostEqual(metrics["final_target"]["target_distance_m"], current_distance)
        self.assertEqual(result["suggested_arm"], "left")
        self.assertEqual(
            [e for e in result["motion_history"] if e["primitive"] == "nav2_navigate_to_pose"],
            [],
        )
        states = [event["state"] for event in events]
        self.assertIn("docked_in_place", states)
        self.assertIn("verifying_arrival", states)
        self.assertEqual(client.halt_calls, 1)
        self.assertEqual(client.cancel_calls, 0)
        self.assertEqual(env.controller.stops, 2)
        json.dumps(metrics, allow_nan=False)

    def test_docking_in_place_still_verifies_the_arrival(self) -> None:
        env = FakeDockingEnv(target_distance_m=0.90)
        client = FakeNav2Client(env, plannable=[False, False])
        controller = self.make_controller(
            env, client, **self.schedule, plan_check_before_goal=True
        )
        original_detect = controller._perception._detect_target

        def detect_then_move_target(*args, **kwargs):
            target = original_detect(*args, **kwargs)
            # The next observation, the arrival check's, finds the object
            # farther than the moved boundary allows.
            env.target_distance_m = 1.20
            return target

        controller._perception._detect_target = detect_then_move_target

        result = controller.dock_to_visible_object("can", approach_bearing_deg=180.0)

        self.assertFalse(result["success"])
        self.assertEqual(result["reason"], "final_visual_verification_failed")
        self.assertEqual(client.goals, [])
        self.assertEqual(
            [a["reason"] for a in result["metrics"]["goal_attempts"]],
            ["docked_at_fallback_distance"],
        )
        self.assertNotIn("suggested_arm", result)

    def test_fallback_beyond_the_base_on_another_bearing_still_navigates(
        self,
    ) -> None:
        env = FakeDockingEnv(target_distance_m=0.90)
        client = FakeNav2Client(env, plannable=[False, False, True])
        client.arrival_distance_m = 1.00
        controller = self.make_controller(
            env, client, **self.schedule, plan_check_before_goal=True
        )

        # 90 deg off the arrival bearing: outside the alignment tolerance,
        # so the base must drive round to the far fallback.
        result = controller.dock_to_visible_object("can", approach_bearing_deg=90.0)

        self.assertTrue(result["success"])
        self.assertEqual(result["reason"], "docked_at_fallback_distance")
        self.assertEqual(len(client.goals), 1)
        self.assertEqual(len(client.plan_checks), 3)
        self.assertEqual(
            self.attempt_rows(result),
            [("explicit", 0.90, "succeeded", True)],
        )
        self.assertEqual(result["metrics"]["bearing_source"], "explicit")
        self.assertEqual(
            result["metrics"]["nav2_goal_xy_yaw"], list(client.goals[0][:3])
        )

    def test_legacy_backend_rejects_the_approach_bearing_before_moving(self) -> None:
        class LegacyController:
            config = SimpleNamespace(docking_distance_m=0.70)

            def __init__(self) -> None:
                self.calls = []

            def dock_to_visible_object(self, object_name):
                self.calls.append(object_name)
                return {"success": True, "reason": "within_docking_distance"}

        legacy = LegacyController()
        dock = _make_dock_to_visible_object(legacy)

        with self.assertRaisesRegex(ValueError, "nav2 docking backend"):
            dock("can", approach_bearing_deg=90.0)
        self.assertEqual(legacy.calls, [])
        self.assertTrue(dock("can")["success"])
        self.assertEqual(legacy.calls, ["can"])
        self.assertIn("Needs the\n                Nav2 docking backend", dock.__doc__)

        # The real legacy controller, through registration: rejected before
        # its entry stop.
        env = FakeDockingEnv(target_distance_m=1.20)
        registry = PrimitiveRegistry()
        register_visible_object_navigation_primitives(
            registry,
            env,
            docking_config={"backend": "legacy"},
            segment_client_factory=lambda: env.segment,
        )
        with self.assertRaisesRegex(ValueError, "approach_bearing_deg"):
            registry.functions()["dock_to_visible_object"]("can", approach_bearing_deg=90.0)
        self.assertEqual(env.controller.stops, 0)

        # The Nav2 backend keeps accepting it.
        env = FakeDockingEnv(target_distance_m=1.20)
        client = FakeNav2Client(env)
        nav2 = _make_dock_to_visible_object(self.make_controller(env, client))
        self.assertTrue(nav2("can", approach_bearing_deg=90.0)["success"])
        self.assertEqual(len(client.goals), 1)


class OperatorStopBeforeTheFirstGoalTest(unittest.TestCase):
    """A Web UI Stop that lands before the first goal: during SAM3 detection,
    the start-clearance check, or the moment between the last check and the
    goal. Also the client's own stall cancels, which must not pass for one.

    ``FakeDockingEnv`` starts at (0, 0, 0) with the target straight ahead.
    """

    schedule = {
        "docking_distance_m": 0.65,
        "nav2_goal_distance_m": 0.60,
        "nav2_goal_distance_fallbacks_m": [0.75, 0.90],
    }

    def make_controller(self, env, client, prior=None, **settings):
        return Nav2VisibleObjectDockingController(
            env,
            docking_config={"backend": "nav2", **self.schedule, **settings},
            segment_client_factory=lambda: env.segment,
            nav2_client_factory=lambda _config: client,
            readiness_prior=prior,
        )

    @staticmethod
    def attempt_rows(result):
        return [
            (a["bearing_source"], a["goal_distance_m"], a["reason"], a["navigated"])
            for a in result["metrics"]["goal_attempts"]
        ]

    def assert_stopped_before_any_goal(self, env, client, result) -> None:
        self.assertFalse(result["success"])
        self.assertEqual(result["reason"], "nav2_failed:operator_stop")
        self.assertEqual(client.goals, [])
        self.assertEqual(
            self.attempt_rows(result), [("explicit", None, "operator_stop", False)]
        )
        self.assertNotIn("nav2_goal_xy_yaw", result["metrics"])
        self.assertNotIn("recommended_recovery", result["metrics"])
        # One Stop reached the environment and its controller; the entry
        # and final stops are the primitive's own.
        self.assertEqual(env.stop_requests, 1)
        self.assertEqual(env.controller.stop_requests, 1)
        self.assertEqual(env.controller.stops, 2)
        # The call's callback is gone again.
        self.assertEqual(env.motion_stop_callbacks, set())
        json.dumps(result["metrics"], allow_nan=False)

    def test_stop_during_detection_sends_no_goal(self) -> None:
        # With and without a controller that exposes its own flag: without
        # one, only the call's own record of the Stop can refuse the goal,
        # since the client did not exist when the Stop landed and its flag
        # is reset at acquisition.
        for controller_flag in (False, True):
            with self.subTest(controller_flag=controller_flag):
                env = StoppableDockingEnv(
                    target_distance_m=1.20, controller_flag=controller_flag
                )
                env.on_segment = env.request_stop
                client = FakeNav2Client(env)
                controller = self.make_controller(env, client)

                result = controller.dock_to_visible_object(
                    "can", approach_bearing_deg=90.0
                )

                self.assert_stopped_before_any_goal(env, client, result)
                self.assertEqual(client.plan_checks, [])
                self.assertEqual(client.reset_calls, 1)
                # Only the failure path's final cancel: the client was not
                # there to be told when the Stop landed.
                self.assertEqual(client.cancel_calls, 1)

    def test_stop_during_the_start_clearance_check_sends_no_goal(self) -> None:
        env = StoppableDockingEnv(target_distance_m=1.20, controller_flag=False)
        client = FakeNav2Client(env)

        def stop_while_checking(**_options):
            env.request_stop()
            return {
                "available": True,
                "blocked": False,
                "obstacle_points": 0,
                "max_points": 3,
            }

        env.controller.nav2_start_clearance_status = stop_while_checking
        controller = self.make_controller(env, client)

        result = controller.dock_to_visible_object("can", approach_bearing_deg=90.0)

        self.assert_stopped_before_any_goal(env, client, result)
        self.assertFalse(result["metrics"]["nav2_start_clearance"]["blocked"])
        self.assertEqual(client.reset_calls, 1)
        self.assertEqual(client.cancel_calls, 1)

    def test_a_stop_before_the_first_goal_reaches_an_existing_client(self) -> None:
        # The client outlives a dock. A Stop during the next dock's detection
        # is remembered by the call and also cancels and latches through the
        # client, whose flag the acquisition then resets without effect.
        env = StoppableDockingEnv(target_distance_m=1.20, controller_flag=False)
        client = FakeNav2Client(env)
        controller = self.make_controller(env, client)

        first = controller.dock_to_visible_object("can", approach_bearing_deg=90.0)

        self.assertTrue(first["success"])
        self.assertEqual(client.cancel_calls, 0)
        self.assertEqual(env.motion_stop_callbacks, set())

        env.target_distance_m = 1.20
        env.on_segment = env.request_stop
        second = controller.dock_to_visible_object("can", approach_bearing_deg=90.0)

        self.assertFalse(second["success"])
        self.assertEqual(second["reason"], "nav2_failed:operator_stop")
        self.assertEqual(len(client.goals), 1)
        # The Stop's cancel through the client, then the failure path's.
        self.assertEqual(client.cancel_calls, 2)
        self.assertEqual(client.reset_calls, 2)
        self.assertEqual(env.motion_stop_callbacks, set())

    def test_the_controllers_own_stop_flag_alone_refuses_the_goal(self) -> None:
        # A stop recorded only on the motion controller, bypassing the
        # callback registry, is still honoured before the goal.
        env = StoppableDockingEnv(target_distance_m=1.20, controller_flag=True)
        env.on_segment = env.controller.request_stop
        client = FakeNav2Client(env)
        controller = self.make_controller(env, client)

        result = controller.dock_to_visible_object("can", approach_bearing_deg=90.0)

        self.assertFalse(result["success"])
        self.assertEqual(result["reason"], "nav2_failed:operator_stop")
        self.assertEqual(client.goals, [])
        self.assertEqual(env.stop_requests, 0)
        self.assertTrue(env.controller.stop_requested())
        self.assertEqual(env.motion_stop_callbacks, set())

    def test_stop_between_the_last_check_and_the_goal_is_refused_by_the_client(
        self,
    ) -> None:
        # The Stop lands after the pending-stop check and before the goal is
        # sent, while the self-filter attachments are fetched: the client's
        # own flag, set by the Stop's cancel, refuses the goal.
        env = StoppableDockingEnv(target_distance_m=1.20, controller_flag=False)
        client = FakeNav2Client(env)

        def stop_then_no_attachments():
            env.request_stop()
            return {}

        env.robot_self_filter_attached_objects = stop_then_no_attachments
        controller = self.make_controller(env, client)

        result = controller.dock_to_visible_object("can", approach_bearing_deg=90.0)

        self.assertFalse(result["success"])
        self.assertEqual(result["reason"], "nav2_failed:operator_stop")
        self.assertEqual(client.goals, [])
        self.assertEqual(
            [a["reason"] for a in result["metrics"]["goal_attempts"]],
            ["operator_stop"],
        )
        # The Stop's cancel through the client, then the failure path's;
        # nothing farther or on another bearing followed.
        self.assertEqual(client.cancel_calls, 2)
        self.assertEqual(env.stop_requests, 1)
        self.assertEqual(env.motion_stop_callbacks, set())

    def test_a_stop_during_the_first_goal_is_not_reported_as_docked_in_place(
        self,
    ) -> None:
        # The base stands inside the next fallback distance on the arrival
        # bearing. The Stop lands during the first goal, which ends with a
        # reason other than operator_stop; the fallback the base already
        # holds must then not count as a dock.
        env = StoppableDockingEnv(target_distance_m=0.81)
        client = FakeNav2Client(env)

        def stop_mid_goal(x_m, y_m, yaw_rad, **options):
            client.goals.append((x_m, y_m, yaw_rad, options))
            env.request_stop()
            return {"success": False, "reason": "goal_response_timeout", "feedback": {}}

        client.navigate_to_pose = stop_mid_goal
        controller = self.make_controller(env, client)

        result = controller.dock_to_visible_object("can", approach_bearing_deg=180.0)

        self.assertFalse(result["success"])
        self.assertEqual(result["reason"], "nav2_failed:operator_stop")
        self.assertEqual(len(client.goals), 1)
        self.assertEqual(
            self.attempt_rows(result),
            [
                ("explicit", 0.60, "goal_response_timeout", True),
                ("explicit", None, "operator_stop", False),
            ],
        )
        self.assertIn("nav2_final_cancel_and_stop", result["metrics"])
        self.assertNotIn("nav2_final_halt", result["metrics"])
        self.assertEqual(client.halt_calls, 0)
        self.assertEqual(env.motion_stop_callbacks, set())

    def test_a_stop_during_a_goal_reaches_the_client_through_the_callback(
        self,
    ) -> None:
        env = StoppableDockingEnv(target_distance_m=1.20)
        client = FakeNav2Client(env)
        registered_during_goal = []

        def stop_mid_goal(x_m, y_m, yaw_rad, **options):
            registered_during_goal.append(len(env.motion_stop_callbacks))
            client.goals.append((x_m, y_m, yaw_rad, options))
            env.request_stop()
            reason = "operator_stop" if client.stop_flag else "aborted"
            return {"success": False, "reason": reason, "feedback": {}}

        client.navigate_to_pose = stop_mid_goal
        controller = self.make_controller(env, client)

        result = controller.dock_to_visible_object("can", approach_bearing_deg=90.0)

        self.assertEqual(registered_during_goal, [1])
        self.assertEqual(result["reason"], "nav2_failed:operator_stop")
        self.assertEqual(len(client.goals), 1)
        self.assertEqual(client.cancel_calls, 2)
        self.assertEqual(env.motion_stop_callbacks, set())

    def test_the_stop_callback_is_unregistered_on_every_exit(self) -> None:
        def blocked(**_options):
            return {
                "available": True,
                "blocked": True,
                "obstacle_points": 52,
                "max_points": 3,
            }

        def interrupt(*_args, **_kwargs):
            raise KeyboardInterrupt

        cases = {
            "success": (1.20, {}, None),
            "failure": (1.20, {"success": False}, None),
            "already docked": (0.50, {}, None),
            "blocked start": (1.20, {}, "blocked"),
            "interrupt": (1.20, {}, "interrupt"),
        }
        for label, (distance, client_options, twist) in cases.items():
            with self.subTest(label):
                env = StoppableDockingEnv(target_distance_m=distance)
                client = FakeNav2Client(env, **client_options)
                if twist == "blocked":
                    env.controller.nav2_start_clearance_status = blocked
                elif twist == "interrupt":
                    client.navigate_to_pose = interrupt
                controller = self.make_controller(env, client)

                if twist == "interrupt":
                    with self.assertRaises(KeyboardInterrupt):
                        controller.dock_to_visible_object(
                            "can", approach_bearing_deg=90.0
                        )
                else:
                    controller.dock_to_visible_object("can", approach_bearing_deg=90.0)

                self.assertEqual(env.motion_stop_callbacks, set())
                self.assertEqual(env.stop_requests, 0)

    def test_a_stall_at_the_inner_distance_falls_through_to_the_next(self) -> None:
        # The client's stall cancel is not an operator stop: the next, farther
        # goal on the same bearing still runs.
        env = FakeDockingEnv(target_distance_m=1.20)
        client = ScriptedNav2Client(env, ["stalled_no_progress", True])
        controller = self.make_controller(env, client)

        result = controller.dock_to_visible_object("can", approach_bearing_deg=90.0)

        self.assertTrue(result["success"])
        self.assertEqual(result["reason"], "docked_at_fallback_distance")
        self.assertEqual(len(client.goals), 2)
        self.assertEqual(
            self.attempt_rows(result),
            [
                ("explicit", 0.60, "stalled_no_progress", True),
                ("explicit", 0.75, "succeeded", True),
            ],
        )
        self.assertFalse(client.stop_flag)
        self.assertEqual(client.cancel_calls, 0)
        self.assertEqual(client.halt_calls, 1)
        self.assertNotIn("recommended_recovery", result["metrics"])

    def test_stalls_through_the_whole_schedule_keep_the_stall_recovery(self) -> None:
        env = FakeDockingEnv(target_distance_m=1.20)
        client = ScriptedNav2Client(env, ["stalled_no_progress"] * 3)
        controller = self.make_controller(env, client)

        result = controller.dock_to_visible_object("can", approach_bearing_deg=90.0)

        self.assertFalse(result["success"])
        self.assertIn("nav2_stalled_no_progress", result["reason"])
        self.assertIn("short positive distance", result["reason"])
        self.assertEqual(len(client.goals), 3)
        self.assertEqual(
            self.attempt_rows(result),
            [
                ("explicit", 0.60, "stalled_no_progress", True),
                ("explicit", 0.75, "stalled_no_progress", True),
                ("explicit", 0.90, "stalled_no_progress", True),
            ],
        )
        recovery = result["metrics"]["recommended_recovery"]
        self.assertEqual(recovery["action"], "small_forward_step_then_reobserve")
        self.assertEqual(recovery["primitive"], "drive_straight")
        self.assertEqual(recovery["suggested_distance_m"], 0.10)
        # Every goal was sent, so no stall set the flag; the one cancel is
        # the failure path's final cancel_and_stop.
        self.assertEqual(client.cancel_calls, 1)
        self.assertEqual(env.controller.stops, 2)


class StallingNav2Client(FakeNav2Client):
    """Nav2 outcomes per goal in order: a float stalls the goal and leaves the
    fake camera's target at that depth, True succeeds, and a string fails with
    that reason. Anything past the script aborts. ``lose_target`` hides the
    target after every stall; ``after_stall`` runs after every stall."""

    def __init__(
        self, env: FakeDockingEnv, outcomes, *, lose_target: bool = False
    ) -> None:
        super().__init__(env)
        self.outcomes = list(outcomes)
        self.lose_target = lose_target
        self.after_stall = None

    def navigate_to_pose(self, x_m, y_m, yaw_rad, **options):
        outcome = self.outcomes.pop(0) if self.outcomes else False
        stall = isinstance(outcome, float)
        self.success = outcome is True
        if stall:
            self.failure_reason = "stalled_no_progress"
        else:
            self.failure_reason = outcome if isinstance(outcome, str) else "aborted"
        result = super().navigate_to_pose(x_m, y_m, yaw_rad, **options)
        if stall:
            self.env.target_distance_m = outcome
            if self.lose_target:
                self.env.target_visible = False
            if self.after_stall is not None:
                self.after_stall()
        return result


class StallArrivalCheckTest(unittest.TestCase):
    """A Nav2 stall ends the goal, not the dock: docking observes the object
    again and docks where the base stands when the camera is on the attempt's
    bearing inside that goal's arrival boundary.

    ``FakeDockingEnv`` starts at (0, 0, 0) with the target straight ahead, so
    the base stands on the 180 deg bearing; the fake camera reports the target
    about 0.10 m short of the depth the client leaves.
    """

    schedule = {
        "docking_distance_m": 0.65,
        "nav2_goal_distance_m": 0.60,
        "nav2_goal_distance_fallbacks_m": [0.75, 0.90],
    }

    def make_controller(self, env, client, **settings):
        return Nav2VisibleObjectDockingController(
            env,
            docking_config={"backend": "nav2", **self.schedule, **settings},
            segment_client_factory=lambda: env.segment,
            nav2_client_factory=lambda _config: client,
        )

    @staticmethod
    def attempt_rows(result):
        return [
            (a["bearing_source"], a["goal_distance_m"], a["reason"], a["navigated"])
            for a in result["metrics"]["goal_attempts"]
        ]

    def test_a_stall_on_the_bearing_inside_the_boundary_docks_there(self) -> None:
        env = FakeDockingEnv(target_distance_m=1.20)
        # The goal stalls with the object about 0.70 m away, inside 0.65 + 0.08.
        client = StallingNav2Client(env, [0.80])
        events = []
        controller = Nav2VisibleObjectDockingController(
            env,
            docking_config={"backend": "nav2", **self.schedule},
            segment_client_factory=lambda: env.segment,
            nav2_client_factory=lambda _config: client,
            progress_callback=events.append,
        )

        result = controller.dock_to_visible_object("can", approach_bearing_deg=180.0)

        self.assertTrue(result["success"])
        self.assertEqual(result["reason"], "docked_after_stall")
        self.assertEqual(len(client.goals), 1)
        self.assertEqual(
            self.attempt_rows(result),
            [("explicit", 0.60, "stalled_no_progress", True)],
        )
        metrics = result["metrics"]
        (check,) = metrics["stall_checks"]
        self.assertTrue(check["docked"])
        self.assertTrue(check["on_bearing"])
        self.assertFalse(check["refused_by_operator_stop"])
        self.assertIsNone(check["detection_error"])
        self.assertEqual(check["goal_distance_m"], 0.60)
        self.assertAlmostEqual(check["bearing_error_deg"], 0.0, delta=0.5)
        self.assertAlmostEqual(check["arrival_limit_m"], 0.73)
        self.assertAlmostEqual(check["target_distance_m"], 0.70, delta=0.02)
        # The stall check's observation is the arrival check.
        self.assertEqual(
            metrics["final_target"]["target_distance_m"], check["target_distance_m"]
        )
        self.assertIn("suggested_arm", result)
        self.assertNotIn("recommended_recovery", metrics)
        self.assertIn("stall_checked", [event["state"] for event in events])
        self.assertEqual(client.halt_calls, 1)
        self.assertEqual(client.cancel_calls, 0)
        json.dumps(metrics, allow_nan=False)

    def test_a_stall_outside_the_boundary_drives_on_to_the_next_goal(self) -> None:
        env = FakeDockingEnv(target_distance_m=1.20)
        # About 0.85 m is outside 0.73, so the 0.75 goal is driven; its stall
        # at about 0.83 m is inside that goal's boundary, 0.75 + 0.05 + 0.08.
        client = StallingNav2Client(env, [0.95, 0.93])
        controller = self.make_controller(env, client)

        result = controller.dock_to_visible_object("can", approach_bearing_deg=180.0)

        self.assertTrue(result["success"])
        self.assertEqual(result["reason"], "docked_after_stall")
        self.assertEqual(len(client.goals), 2)
        self.assertEqual(
            self.attempt_rows(result),
            [
                ("explicit", 0.60, "stalled_no_progress", True),
                ("explicit", 0.75, "stalled_no_progress", True),
            ],
        )
        checks = result["metrics"]["stall_checks"]
        self.assertEqual([check["docked"] for check in checks], [False, True])
        self.assertEqual([check["goal_distance_m"] for check in checks], [0.60, 0.75])
        self.assertAlmostEqual(checks[0]["arrival_limit_m"], 0.73)
        self.assertAlmostEqual(checks[1]["arrival_limit_m"], 0.88)

    def test_a_fallback_the_stalled_base_stands_inside_docks_in_place(self) -> None:
        env = FakeDockingEnv(target_distance_m=1.20)
        # About 0.70 m after the stall: outside 0.65 without a tolerance, but
        # inside the 0.75 fallback, which the base would have to back away to.
        client = StallingNav2Client(env, [0.80])
        controller = self.make_controller(
            env, client, final_distance_tolerance_m=0.0
        )

        result = controller.dock_to_visible_object("can", approach_bearing_deg=180.0)

        self.assertTrue(result["success"])
        self.assertEqual(result["reason"], "docked_at_fallback_distance")
        self.assertEqual(len(client.goals), 1)
        rows = self.attempt_rows(result)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0], ("explicit", 0.60, "stalled_no_progress", True))
        source, distance, reason, navigated = rows[1]
        self.assertEqual(
            (source, reason, navigated),
            ("explicit", "docked_at_fallback_distance", False),
        )
        (check,) = result["metrics"]["stall_checks"]
        self.assertFalse(check["docked"])
        self.assertTrue(check["on_bearing"])
        self.assertAlmostEqual(distance, check["target_distance_m"])
        self.assertAlmostEqual(
            result["metrics"]["goal_distance_used_m"], check["target_distance_m"]
        )

    def test_a_stall_off_the_bearing_does_not_dock(self) -> None:
        env = FakeDockingEnv(target_distance_m=1.20)
        client = StallingNav2Client(env, [0.80, 0.80, 0.80])
        controller = self.make_controller(env, client)

        # Inside every boundary, but 90 deg round the object from the bearing
        # asked for: the base is on the wrong side to count as docked there.
        result = controller.dock_to_visible_object("can", approach_bearing_deg=90.0)

        self.assertFalse(result["success"])
        self.assertIn("nav2_stalled_no_progress", result["reason"])
        self.assertEqual(len(client.goals), 3)
        checks = result["metrics"]["stall_checks"]
        self.assertEqual([check["on_bearing"] for check in checks], [False] * 3)
        self.assertEqual([check["docked"] for check in checks], [False] * 3)
        self.assertAlmostEqual(checks[0]["bearing_error_deg"], 90.0, delta=0.5)
        self.assertIn("recommended_recovery", result["metrics"])

    def test_a_target_lost_after_a_stall_keeps_the_stall_failure(self) -> None:
        env = FakeDockingEnv(target_distance_m=1.20)
        client = StallingNav2Client(env, [0.80, 0.80, 0.80], lose_target=True)
        controller = self.make_controller(env, client)

        result = controller.dock_to_visible_object("can", approach_bearing_deg=180.0)

        self.assertFalse(result["success"])
        self.assertIn("nav2_stalled_no_progress", result["reason"])
        self.assertEqual(len(client.goals), 3)
        checks = result["metrics"]["stall_checks"]
        self.assertEqual([check["docked"] for check in checks], [False] * 3)
        self.assertTrue(
            all("SAM3 found no instance" in check["detection_error"] for check in checks)
        )
        self.assertTrue(all(check["target_distance_m"] is None for check in checks))
        json.dumps(result["metrics"], allow_nan=False)

    def test_a_stop_during_the_stall_check_is_not_reported_as_docked(self) -> None:
        env = StoppableDockingEnv(target_distance_m=1.20)
        client = StallingNav2Client(env, [0.80])

        def stop_once():
            env.on_segment = None
            env.request_stop()

        def stop_during_the_next_detection():
            env.on_segment = stop_once

        client.after_stall = stop_during_the_next_detection
        controller = self.make_controller(env, client)

        result = controller.dock_to_visible_object("can", approach_bearing_deg=180.0)

        self.assertFalse(result["success"])
        self.assertEqual(result["reason"], "nav2_failed:operator_stop")
        self.assertEqual(len(client.goals), 1)
        self.assertEqual(
            self.attempt_rows(result),
            [
                ("explicit", 0.60, "stalled_no_progress", True),
                ("explicit", None, "operator_stop", False),
            ],
        )
        (check,) = result["metrics"]["stall_checks"]
        self.assertFalse(check["docked"])
        self.assertTrue(check["refused_by_operator_stop"])
        self.assertEqual(env.stop_requests, 1)
        self.assertEqual(client.halt_calls, 0)
        self.assertEqual(env.motion_stop_callbacks, set())


class ArmSideHeadingOffsetTest(unittest.TestCase):
    """The optional heading offset that puts the object in front of one arm.

    As above, ``FakeDockingEnv`` starts at (0, 0, 0) with the target straight
    ahead: the arrival goal faces along +x (yaw 0), a bearing of pi/2 (object
    toward goal along +y) faces the object with yaw -pi/2, and a bearing of 0
    faces it with yaw pi, where the offset has to wrap.
    """

    def make_controller(self, env, client, prior=None, **settings):
        return Nav2VisibleObjectDockingController(
            env,
            docking_config={"backend": "nav2", **settings},
            segment_client_factory=lambda: env.segment,
            nav2_client_factory=lambda _config: client,
            readiness_prior=prior,
        )

    def dock(self, *, prior=None, approach_bearing_deg=None, **settings):
        """One successful dock from the standard start; the single goal sent."""

        env = FakeDockingEnv(target_distance_m=1.20)
        client = FakeNav2Client(env)
        controller = self.make_controller(env, client, prior, **settings)
        result = controller.dock_to_visible_object(
            "can", approach_bearing_deg=approach_bearing_deg
        )
        self.assertTrue(result["success"], result["reason"])
        self.assertEqual(len(client.goals), 1)
        return controller, client.goals[0], result

    @staticmethod
    def wrap(angle):
        return math.atan2(math.sin(angle), math.cos(angle))

    def assert_yaw(self, actual, expected):
        self.assertAlmostEqual(self.wrap(actual - expected), 0.0)

    def assert_camera_goal(self, controller, goal, camera_x, camera_y, yaw):
        expected = controller._base_goal_from_camera_goal(camera_x, camera_y, yaw)
        self.assertAlmostEqual(goal[0], expected[0])
        self.assertAlmostEqual(goal[1], expected[1])
        self.assert_yaw(goal[2], expected[2])

    def test_defaults_are_off_and_favour_the_left_arm(self) -> None:
        config = Nav2VisibleObjectDockingConfig.from_mapping(None)

        self.assertEqual(config.arm_side_heading_offset_deg, 0.0)
        self.assertEqual(config.arm_side_heading_default_arm, "left")

    def test_zero_offset_keeps_the_facing_heading_on_every_bearing_source(self) -> None:
        # Arrival source: the goal is bit-identical to one with default settings.
        _, _, plain = self.dock()
        for default_arm in ("left", "right"):
            with self.subTest(source="arrival", default_arm=default_arm):
                _, goal, result = self.dock(
                    arm_side_heading_offset_deg=0.0,
                    arm_side_heading_default_arm=default_arm,
                )
                metrics = result["metrics"]
                self.assertEqual(
                    metrics["desired_camera_goal_xy_yaw"],
                    plain["metrics"]["desired_camera_goal_xy_yaw"],
                )
                self.assertEqual(
                    metrics["nav2_goal_xy_yaw"], plain["metrics"]["nav2_goal_xy_yaw"]
                )
                self.assertEqual(goal[2], 0.0)
                self.assertEqual(metrics["heading_offset_arm"], default_arm)
                self.assertEqual(metrics["heading_offset_deg"], 0.0)
        with self.subTest(source="explicit"):
            _, goal, result = self.dock(
                arm_side_heading_offset_deg=0, approach_bearing_deg=90.0
            )
            self.assert_yaw(goal[2], -math.pi / 2.0)
            self.assertEqual(result["metrics"]["heading_offset_arm"], "left")
            self.assertEqual(result["metrics"]["heading_offset_deg"], 0.0)
        with self.subTest(source="prior"):
            prior = FakePrior(make_prior_estimate(math.pi / 2.0, suggested_arm="right"))
            _, goal, result = self.dock(prior=prior, arm_side_heading_offset_deg=0.0)
            self.assert_yaw(goal[2], -math.pi / 2.0)
            self.assertEqual(result["metrics"]["bearing_source"], "prior")
            # The prior's hand names the arm even when there is nothing to apply.
            self.assertEqual(result["metrics"]["heading_offset_arm"], "right")
            self.assertEqual(result["metrics"]["heading_offset_deg"], 0.0)

    def test_left_arm_turns_the_base_right_on_an_explicit_bearing(self) -> None:
        controller, goal, result = self.dock(
            arm_side_heading_offset_deg=10.0,
            arm_side_heading_default_arm="left",
            approach_bearing_deg=90.0,
        )

        metrics = result["metrics"]
        target_x, target_y = metrics["target_world_xy_m"]
        goal_distance = controller.config.effective_nav2_goal_distance_m
        facing_yaw = -math.pi / 2.0
        expected_yaw = self.wrap(facing_yaw - math.radians(10.0))
        # Same camera position as facing the object; only the heading turns.
        self.assert_camera_goal(
            controller, goal, target_x, target_y + goal_distance, expected_yaw
        )
        self.assert_yaw(metrics["desired_camera_goal_xy_yaw"][2], expected_yaw)
        self.assert_yaw(metrics["goal_attempts"][0]["camera_goal_xy_yaw"][2], expected_yaw)
        self.assert_yaw(metrics["nav2_goal_xy_yaw"][2], goal[2])
        self.assertEqual(metrics["heading_offset_arm"], "left")
        self.assertEqual(metrics["heading_offset_deg"], -10.0)
        # The bearing bookkeeping is about bearings, not headings.
        self.assertAlmostEqual(metrics["reference_bearing_rad"], math.pi / 2.0)
        self.assertAlmostEqual(metrics["goal_attempts"][0]["bearing_rad"], math.pi / 2.0)
        self.assertEqual(metrics["bearing_source"], "explicit")
        self.assertEqual(result["suggested_arm"], "left")
        json.dumps(metrics, allow_nan=False)

    def test_left_arm_turns_the_base_right_on_the_arrival_bearing(self) -> None:
        _, _, plain = self.dock()
        _, goal, result = self.dock(arm_side_heading_offset_deg=10.0)

        metrics = result["metrics"]
        self.assertEqual(metrics["bearing_source"], "arrival")
        self.assertEqual(
            metrics["desired_camera_goal_xy_yaw"][:2],
            plain["metrics"]["desired_camera_goal_xy_yaw"][:2],
        )
        self.assert_yaw(metrics["desired_camera_goal_xy_yaw"][2], -math.radians(10.0))
        self.assert_yaw(goal[2], -math.radians(10.0))
        self.assertEqual(metrics["heading_offset_arm"], "left")
        self.assertEqual(metrics["heading_offset_deg"], -10.0)

    def test_right_arm_turns_the_base_left_and_the_heading_wraps(self) -> None:
        with self.subTest(source="arrival"):
            _, goal, result = self.dock(
                arm_side_heading_offset_deg=10.0,
                arm_side_heading_default_arm="right",
            )
            self.assert_yaw(goal[2], math.radians(10.0))
            self.assertEqual(result["metrics"]["heading_offset_arm"], "right")
            self.assertEqual(result["metrics"]["heading_offset_deg"], 10.0)
        with self.subTest(source="explicit"):
            controller, goal, result = self.dock(
                arm_side_heading_offset_deg=10.0,
                arm_side_heading_default_arm="right",
                approach_bearing_deg=0.0,
            )
            target_x, target_y = result["metrics"]["target_world_xy_m"]
            goal_distance = controller.config.effective_nav2_goal_distance_m
            expected_yaw = -math.pi + math.radians(10.0)
            self.assert_camera_goal(
                controller, goal, target_x + goal_distance, target_y, expected_yaw
            )
            camera_yaw = result["metrics"]["desired_camera_goal_xy_yaw"][2]
            self.assertTrue(-math.pi < camera_yaw <= math.pi)
            self.assertAlmostEqual(camera_yaw, expected_yaw)
            self.assertEqual(result["metrics"]["heading_offset_deg"], 10.0)

    def test_prior_hand_overrides_the_default_arm(self) -> None:
        for hand, default_arm, expected_deg in (
            ("right", "left", 10.0),
            ("left", "right", -10.0),
        ):
            with self.subTest(hand=hand, default_arm=default_arm):
                prior = FakePrior(
                    make_prior_estimate(math.pi / 2.0, hand=hand, suggested_arm=hand)
                )
                _, goal, result = self.dock(
                    prior=prior,
                    arm_side_heading_offset_deg=10.0,
                    arm_side_heading_default_arm=default_arm,
                )
                metrics = result["metrics"]
                self.assertEqual(metrics["bearing_source"], "prior")
                self.assert_yaw(
                    goal[2], -math.pi / 2.0 + math.radians(expected_deg)
                )
                self.assertEqual(metrics["heading_offset_arm"], hand)
                self.assertEqual(metrics["heading_offset_deg"], expected_deg)
                self.assertEqual(metrics["human_hand_arm"], hand)

    def test_prior_without_a_hand_uses_the_default_arm(self) -> None:
        with self.subTest(prior="used, both hands"):
            prior = FakePrior(
                make_prior_estimate(math.pi / 2.0, hand="both", suggested_arm=None)
            )
            _, goal, result = self.dock(
                prior=prior,
                arm_side_heading_offset_deg=10.0,
                arm_side_heading_default_arm="right",
            )
            self.assertEqual(result["metrics"]["bearing_source"], "prior")
            self.assert_yaw(goal[2], -math.pi / 2.0 + math.radians(10.0))
            self.assertEqual(result["metrics"]["heading_offset_arm"], "right")
            self.assertEqual(result["metrics"]["heading_offset_deg"], 10.0)
        with self.subTest(prior="no estimate"):
            # A prior that answered nothing names no hand either, even though
            # its fixture would have; the arrival goal turns for the default.
            prior = FakePrior(None)
            _, goal, result = self.dock(
                prior=prior,
                arm_side_heading_offset_deg=10.0,
                arm_side_heading_default_arm="left",
            )
            self.assertEqual(result["metrics"]["bearing_source"], "arrival")
            self.assertFalse(result["metrics"]["readiness_prior"]["used"])
            self.assert_yaw(goal[2], -math.radians(10.0))
            self.assertEqual(result["metrics"]["heading_offset_arm"], "left")
            self.assertEqual(result["metrics"]["heading_offset_deg"], -10.0)

    def test_every_prior_bearing_of_one_call_turns_the_same_way(self) -> None:
        env = FakeDockingEnv(target_distance_m=1.20)
        client = ScriptedNav2Client(env, [False, False, False, True])
        prior = FakePrior(
            make_prior_estimate(math.pi / 2.0, hand="right", suggested_arm="right")
        )
        controller = self.make_controller(
            env, client, prior, arm_side_heading_offset_deg=10.0
        )

        result = controller.dock_to_visible_object("can")

        self.assertTrue(result["success"])
        attempts = result["metrics"]["goal_attempts"]
        self.assertEqual(
            [attempt["bearing_source"] for attempt in attempts],
            ["prior", "prior_offset", "prior_offset", "arrival_fallback"],
        )
        for attempt in attempts:
            with self.subTest(attempt["bearing_source"]):
                bearing = attempt["bearing_rad"]
                facing_yaw = math.atan2(-math.sin(bearing), -math.cos(bearing))
                self.assert_yaw(
                    attempt["camera_goal_xy_yaw"][2],
                    facing_yaw + math.radians(10.0),
                )
        self.assertEqual(result["metrics"]["heading_offset_arm"], "right")
        self.assertEqual(result["metrics"]["heading_offset_deg"], 10.0)

    def test_docked_in_place_still_compares_bearings_not_headings(self) -> None:
        # Detected inside the fallback distance on the arrival bearing: once
        # the inner goal fails, the pose counts as docked without a goal, the
        # offset heading notwithstanding.
        env = FakeDockingEnv(target_distance_m=0.70)
        client = ScriptedNav2Client(env, [False])
        controller = self.make_controller(
            env,
            client,
            nav2_goal_distance_fallbacks_m=[0.75],
            arm_side_heading_offset_deg=10.0,
        )

        result = controller.dock_to_visible_object("can", approach_bearing_deg=180.0)

        self.assertTrue(result["success"])
        self.assertEqual(result["reason"], "docked_at_fallback_distance")
        self.assertEqual(len(client.goals), 1)
        self.assert_yaw(client.goals[0][2], -math.radians(10.0))
        metrics = result["metrics"]
        current_distance = metrics["target_distance_m"]
        self.assertLess(0.60, current_distance)
        self.assertLessEqual(current_distance, 0.75)
        self.assertEqual(
            [(a["goal_distance_m"], a["reason"]) for a in metrics["goal_attempts"]],
            [(0.60, "aborted"), (current_distance, "docked_at_fallback_distance")],
        )
        self.assertEqual(metrics["heading_offset_deg"], -10.0)

    def test_offset_settings_are_validated(self) -> None:
        for offset in (0, 45, 10.5):
            with self.subTest(offset=offset):
                config = Nav2VisibleObjectDockingConfig.from_mapping(
                    {"arm_side_heading_offset_deg": offset}
                )
                self.assertEqual(config.arm_side_heading_offset_deg, offset)
        config = Nav2VisibleObjectDockingConfig.from_mapping(
            {"arm_side_heading_default_arm": "right"}
        )
        self.assertEqual(config.arm_side_heading_default_arm, "right")

        for label, settings in (
            ("too wide", {"arm_side_heading_offset_deg": 46.0}),
            ("negative", {"arm_side_heading_offset_deg": -1.0}),
            ("nan", {"arm_side_heading_offset_deg": math.nan}),
            ("string", {"arm_side_heading_offset_deg": "10"}),
            ("bool", {"arm_side_heading_offset_deg": True}),
            ("none", {"arm_side_heading_offset_deg": None}),
            ("unknown arm", {"arm_side_heading_default_arm": "middle"}),
            ("upper-case arm", {"arm_side_heading_default_arm": "Left"}),
            ("empty arm", {"arm_side_heading_default_arm": ""}),
        ):
            with self.subTest(label):
                with self.assertRaisesRegex(ValueError, next(iter(settings))):
                    Nav2VisibleObjectDockingConfig.from_mapping(settings)


if __name__ == "__main__":
    unittest.main()
