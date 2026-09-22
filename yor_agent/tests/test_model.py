from __future__ import annotations

import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from yor_agent.exceptions import FormatError, ModelError
from yor_agent.models.llm import (
    CAPX_COARSE_NAVIGATION_SYSTEM_PROMPT,
    LLM,
    SYSTEM_PROMPT,
    extract_code,
)


def observation(width: int = 64) -> dict:
    return {
        "robot0_robotview": {
            "images": {
                "rgb": np.zeros((48, width, 3), dtype=np.uint8),
                "depth": np.ones((48, width, 1), dtype=np.float32),
            },
            "timestamp_ns": 123,
        },
        "base": {
            "pose_xy_yaw": np.array([1.0, 2.0, 0.5]),
            "last_velocity": np.zeros(3),
            "lease_active": False,
            "lease_remaining_s": 0.0,
            "estop_latched": False,
            "telemetry": {},
        },
        "arms": {"available": False, "estop_latched": False},
        "lift": {"available": True, "height_m": 0.42},
    }


class ExtractCodeTest(unittest.TestCase):
    def test_accepts_bare_python(self) -> None:
        self.assertEqual(extract_code("drive_straight(0.3)\n"), "drive_straight(0.3)")

    def test_accepts_one_fenced_block_and_drops_surrounding_prose(self) -> None:
        response = "First I look around.\n```python\nobserve()\n```\nThen I decide."

        self.assertEqual(extract_code(response), "observe()")

    def test_rejects_empty_output(self) -> None:
        for response in ["", "   \n", "```python\n\n```"]:
            with self.subTest(response=response), self.assertRaises(FormatError):
                extract_code(response)

    def test_rejects_multiple_blocks(self) -> None:
        response = "```python\nobserve()\n```\nand\n```python\nstop()\n```"

        with self.assertRaises(FormatError) as caught:
            extract_code(response)

        self.assertIn("2 code blocks", str(caught.exception))

    def test_rejects_non_python_block(self) -> None:
        with self.assertRaises(FormatError):
            extract_code("```bash\nls\n```")

    def test_rejects_unfenced_prose(self) -> None:
        with self.assertRaises(FormatError):
            extract_code("I will drive forward 30 centimeters and then look again.")

    def test_regenerate_substring_is_not_special(self) -> None:
        # The old CaP-X parser treated any response *without* the substring
        # "REGENERATE" as a finish signal. Here the word carries no meaning:
        # prose is still a format error, and code is still just code.
        with self.assertRaises(FormatError):
            extract_code("REGENERATE, because the robot did not reach the sofa.")
        self.assertEqual(
            extract_code("REGENERATE\n```python\nobserve()\n```"), "observe()"
        )

    def test_syntax_error_inside_a_fence_is_left_to_the_executor(self) -> None:
        self.assertEqual(extract_code("```python\ndrive(0.1\n```"), "drive(0.1")


class PromptTest(unittest.TestCase):
    def test_initial_messages_carry_task_docs_summary_and_image(self) -> None:
        model = LLM({"provider": "vertex", "name": "gemini-2.5-pro"})

        messages = model.initial_messages(
            task="Go to the orange sofa.",
            observation=observation(),
            primitive_docs="def observe():\n    ...",
        )

        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("fresh namespace", messages[0]["content"])
        self.assertIn("Never write an `import` statement", messages[0]["content"])
        self.assertIn("alias `np`", messages[0]["content"])
        self.assertIn('`arm="left"` or `arm="right"`', messages[0]["content"])
        self.assertIn("explicitly pass the", messages[0]["content"])
        self.assertIn("+X is horizontal right", messages[0]["content"])
        self.assertIn("+Y is vertical down", messages[0]["content"])
        self.assertIn("+Z is horizontal forward", messages[0]["content"])
        self.assertIn("each arm TCP is roughly 0.50 m", messages[0]["content"])
        self.assertIn("always prefer ``goto_grasp_pose``", messages[0]["content"])
        self.assertIn("When ``prepare_for_manipulation`` is available", messages[0]["content"])
        self.assertIn("already visible in the current camera view", messages[0]["content"])
        self.assertIn("does not search for or center", messages[0]["content"])
        self.assertIn("Do not move the base or arm between", messages[0]["content"])
        self.assertIn("current mobile-base pose is unfavorable", messages[0]["content"])
        self.assertIn("closer to the table or manipulation workspace", messages[0]["content"])
        self.assertIn("comfortably within arm reach", messages[0]["content"])
        self.assertIn("too far away", messages[0]["content"])
        self.assertIn("remains unusable after repeated retries", messages[0]["content"])
        self.assertNotIn("temporary simulated-gripper task", messages[0]["content"])
        self.assertIn(
            "Do not call ``prepare_for_manipulation`` on the receptacle",
            messages[0]["content"],
        )
        self.assertIn("passing them to ``goto_pose``", messages[0]["content"])
        self.assertIn("``drive_lateral``", messages[0]["content"])
        self.assertIn("positive is robot-left", messages[0]["content"])
        self.assertIn("``nav2_blocked_near_obstacle``", messages[0]["content"])
        self.assertIn("short negative ``drive_straight``", messages[0]["content"])
        self.assertIn("after any obstacle or safety failure", messages[0]["content"])
        self.assertIn("Never make a long blind approach", messages[0]["content"])
        text = messages[1]["content"][0]["text"]
        self.assertIn("Go to the orange sofa.", text)
        self.assertIn("def observe()", text)
        self.assertIn('"pose_xy_yaw": [\n      1.0,\n      2.0,\n      0.5\n    ]', text)
        self.assertNotIn("array(", text)
        url = messages[1]["content"][1]["image_url"]["url"]
        self.assertTrue(url.startswith("data:image/png;base64,"))

    def test_system_prompt_describes_mobile_footprint_and_motion_limits(
        self,
    ) -> None:
        model = LLM({"provider": "vertex", "name": "gemini-2.5-pro"})

        messages = model.initial_messages(
            task="Go to the blue trash bin.",
            observation=observation(),
            primitive_docs="def drive_straight(distance_m):\n    ...",
        )

        system_prompt = messages[0]["content"]
        self.assertIn("about 0.43 m to each side", system_prompt)
        self.assertIn("``turn_relative`` has no obstacle check", system_prompt)
        self.assertIn("enters the swept footprint ahead", system_prompt)
        self.assertIn("move the full swept footprint away", system_prompt)
        self.assertIn("do not merely point the camera away", system_prompt)
        self.assertIn("reverse only when the rear path is known clear", system_prompt)
        self.assertIn("leaving view is not proof", system_prompt)

    def test_the_coarse_navigation_prompt_names_only_the_coarse_vocabulary(self) -> None:
        model = LLM(
            {"provider": "openai", "name": "gpt-5.6-sol", "system_prompt": "capx_coarse_navigation"}
        )

        messages = model.initial_messages(
            task="Find the can.",
            observation=observation(),
            primitive_docs="def go_forward():\n    ...",
        )

        system_prompt = messages[0]["content"]
        self.assertEqual(system_prompt, CAPX_COARSE_NAVIGATION_SYSTEM_PROMPT)
        for name in (
            "go_forward",
            "turn_left_45_degrees",
            "turn_right_45_degrees",
            "goto_planar_position",
        ):
            self.assertIn(f"``{name}``", system_prompt)
        for hidden in (
            "dock_to_visible_object",
            "prepare_for_manipulation",
            "drive_straight",
            "turn_relative",
            "drive_lateral",
        ):
            self.assertNotIn(hidden, system_prompt)
        # Everything but the navigation guidance matches the default prompt.
        for rule in (
            "fresh namespace",
            "Never write an `import` statement",
            '`arm="left"` or `arm="right"`',
            "+X is horizontal right",
            "always prefer ``goto_grasp_pose``",
            "passing them to ``goto_pose``",
            "about 0.43 m to each side",
            "leaving view is not proof",
        ):
            self.assertIn(rule, SYSTEM_PROMPT)
            self.assertIn(rule, system_prompt)

    def test_the_default_prompt_plans_routes_for_the_whole_body(self) -> None:
        for rule in (
            "Plan the route for your whole body",
            "keep about 0.3 m between the arms and furniture on both sides",
            "in open floor move 0.5-1.0 m and turn 30-90 degrees",
            "do not retry the same heading with a shorter step",
            "do not creep along furniture in small steps",
        ):
            self.assertIn(rule, SYSTEM_PROMPT)
        for removed in (
            "small forward or lateral steps",
            "cautious small base motions",
            "use conservative angles and distances",
            "Move in small increments",
        ):
            self.assertNotIn(removed, SYSTEM_PROMPT)

    def test_the_system_prompt_is_the_default_unless_another_is_named(self) -> None:
        self.assertEqual(LLM({}).system_prompt, SYSTEM_PROMPT)
        self.assertEqual(LLM({}).system_prompt_name, "default")
        with self.assertRaises(ValueError):
            LLM({"system_prompt": "no_such_prompt"})

    def test_wide_images_are_downscaled(self) -> None:
        from PIL import Image
        import base64
        import io

        model = LLM({"image_max_width": 32})
        messages = model.initial_messages(
            task="t", observation=observation(width=256), primitive_docs=""
        )

        url = messages[1]["content"][1]["image_url"]["url"]
        payload = base64.b64decode(url.split(",", 1)[1])
        self.assertEqual(Image.open(io.BytesIO(payload)).width, 32)

    def test_feedback_reports_stdout_error_and_calls(self) -> None:
        model = LLM({})
        model.last_response = "```python\nobserve()\n```"

        messages = model.format_feedback(
            "observe()",
            {
                "stdout": "hello\n",
                "stderr": "",
                "error": {"type": "NameError", "traceback": "NameError: boom"},
                "interrupted_by": None,
                "primitive_calls": [{"name": "observe"}],
            },
            observation(),
        )

        self.assertEqual(messages[0]["role"], "assistant")
        self.assertEqual(messages[0]["content"], "```python\nobserve()\n```")
        text = messages[1]["content"][0]["text"]
        self.assertIn("hello", text)
        self.assertIn("raised an exception", text)
        self.assertIn("NameError: boom", text)
        self.assertIn("observe()", text)

    def test_feedback_explains_goto_grasp_goal_ik_failure(self) -> None:
        model = LLM({})
        messages = model.format_feedback(
            "goto_grasp_pose('box', position, quaternion, arm='right')",
            {
                "stdout": "",
                "stderr": "",
                "error": None,
                "interrupted_by": {
                    "primitive": "goto_grasp_pose",
                    "reason": "grasp_goal_ik_failed",
                    "result": {
                        "success": False,
                        "reason": "grasp_goal_ik_failed",
                        "status": "Goalset planning returned None.",
                        "arm": "right",
                        "goalset_candidate_count": 3,
                        "pi_goal_ik_converged": False,
                        "pi_goal_ik_position_error_m": 0.018,
                        "pi_goal_ik_rotation_error_rad": 0.047,
                        "grasp_ik_position_tolerance_m": 0.020,
                        "grasp_ik_rotation_tolerance_rad": 0.10,
                        "pi_goal_joint_seed_forwarded": False,
                        "target_distance_m": 0.006,
                        "sam_score": 0.96,
                        "planning_time_s": 1.8,
                        "terminal_check": {
                            "safe": [True, False, True],
                            "minimum_clearance_m": [0.08, 0.004, 0.06],
                            "first_collision_step": [None, 3, None],
                            "clearance_m": 0.008,
                        },
                    },
                },
                "primitive_calls": [{"name": "goto_grasp_pose"}],
            },
            observation(),
        )

        text = messages[1]["content"][0]["text"]
        self.assertIn('"failure_stage": "goal_ik"', text)
        self.assertIn('"terminal_safe_count": 2', text)
        self.assertIn('"terminal_collision_count": 1', text)
        self.assertIn("occurred before trajectory planning", text)
        self.assertIn("individually colliding alternatives", text)
        self.assertIn("small mobile-base adjustment", text)
        self.assertIn('"position_error_m": 0.018', text)
        self.assertIn('"accepted_position_error_m": 0.02', text)
        self.assertIn('"accepted_rotation_error_rad": 0.1', text)
        self.assertNotIn("first_collision_step", text)

    def test_feedback_preserves_prepare_too_far_recovery(self) -> None:
        model = LLM({})
        messages = model.format_feedback(
            "prepare_for_manipulation('box', arm='left')",
            {
                "stdout": "",
                "stderr": "",
                "error": None,
                "interrupted_by": {
                    "primitive": "prepare_for_manipulation",
                    "reason": "target_too_far_for_local_manipulation",
                    "result": {
                        "success": False,
                        "reason": "target_too_far_for_local_manipulation",
                        "target_distance_m": 1.2,
                        "maximum_distance_m": 0.95,
                        "recovery": {
                            "action": "dock_to_visible_object",
                            "message": "Move closer to the target, then retry.",
                        },
                    },
                },
                "primitive_calls": [{"name": "prepare_for_manipulation"}],
            },
            observation(),
        )

        text = messages[1]["content"][0]["text"]
        self.assertIn("target_too_far_for_local_manipulation", text)
        self.assertIn('"action": "dock_to_visible_object"', text)
        self.assertIn("Move closer to the target", text)

    def test_feedback_omits_prepare_motion_trace_diagnostics(self) -> None:
        model = LLM({})
        messages = model.format_feedback(
            "prepare_for_manipulation('box', arm='right')",
            {
                "stdout": "",
                "stderr": "",
                "error": None,
                "interrupted_by": {
                    "primitive": "prepare_for_manipulation",
                    "reason": "RuntimeError:single-stage SE(2) motion failed",
                    "result": {
                        "success": False,
                        "reason": "RuntimeError:single-stage SE(2) motion failed",
                        "arm": "right",
                        "selected_base_pose": {
                            "forward_m": 0.05,
                            "left_m": -0.10,
                            "yaw_rad": 0.17,
                        },
                        "motion_target": {
                            "target_pose_world": [1.0, 2.0, 0.3]
                        },
                        "motion": {
                            "metrics": {
                                "command_history": [[0.1, -0.1, 0.2]] * 100
                            }
                        },
                    },
                },
                "primitive_calls": [{"name": "prepare_for_manipulation"}],
            },
            observation(),
        )

        text = messages[1]["content"][0]["text"]
        self.assertIn("single-stage SE(2) motion failed", text)
        self.assertNotIn("selected_base_pose", text)
        self.assertNotIn("motion_target", text)
        self.assertNotIn("command_history", text)

    def test_feedback_summarizes_prepare_planner_diagnostics(self) -> None:
        model = LLM({})
        large_pose = np.eye(4).tolist()
        messages = model.format_feedback(
            "prepare_for_manipulation('box', arm='left')",
            {
                "stdout": "",
                "stderr": "",
                "error": None,
                "interrupted_by": {
                    "primitive": "prepare_for_manipulation",
                    "reason": "_CandidatePlanningError:" + "raw " * 1000,
                    "result": {
                        "success": False,
                        "reason": "_CandidatePlanningError:" + "raw " * 1000,
                        "arm": "left",
                        "diagnostics": {
                            "stage": "parallel_curobo_virtual_certification",
                            "batch": {
                                "planning_time_s": 57.6,
                                "batch_goalset_count": 10,
                            },
                            "planner": {
                                "status": "Goalset planning returned None.",
                                "position_tolerance_m": 0.02,
                                "rotation_tolerance_rad": 0.1,
                                "ik_calls": [{"joint_solutions": [[0.0] * 7] * 16}],
                            },
                            "attempts": [
                                {
                                    "success": False,
                                    "reason": "grasp_goal_ik_failed",
                                    "strict_pi_count": 0,
                                    "converged_pi_count": 0,
                                    "finite_pi_count": 16,
                                    "submitted_tcp_poses": [large_pose] * 16,
                                    "ik_diagnostics": {
                                        "joint_solution_rad_by_returned_seed": [
                                            [0.0] * 7
                                        ]
                                        * 16
                                    },
                                }
                            ]
                            * 16,
                        },
                    },
                },
                "primitive_calls": [{"name": "prepare_for_manipulation"}],
            },
            observation(),
        )

        text = messages[1]["content"][0]["text"]
        self.assertIn("no_local_base_pose_passed_parallel_curobo", text)
        self.assertIn('"evaluated_base_count": 16', text)
        self.assertIn('"bases_with_acceptable_grasp": 0', text)
        self.assertIn('"finite_grasp_count": 256', text)
        self.assertIn('"grasp_goal_ik_failed": 16', text)
        self.assertNotIn("submitted_tcp_poses", text)
        self.assertNotIn("joint_solution_rad_by_returned_seed", text)
        self.assertLess(len(text), 4000)

    def test_feedback_summarizes_strict_pi_readiness_failure(self) -> None:
        model = LLM({})
        messages = model.format_feedback(
            "prepare_for_manipulation('box', arm='left')",
            {
                "stdout": "",
                "stderr": "",
                "error": None,
                "interrupted_by": {
                    "primitive": "prepare_for_manipulation",
                    "reason": "_CandidatePlanningError:no strict Pi grasp",
                    "result": {
                        "success": False,
                        "reason": "_CandidatePlanningError:no strict Pi grasp",
                        "arm": "left",
                        "diagnostics": {
                            "stage": "parallel_pi_virtual_certification",
                            "evaluated_base_count": 729,
                            "ik_query_count": 46656,
                            "collision_safe_grasp_count": 64,
                            "bases_with_converged_grasp": 0,
                        },
                    },
                },
                "primitive_calls": [{"name": "prepare_for_manipulation"}],
            },
            observation(),
        )

        text = messages[1]["content"][0]["text"]
        self.assertIn(
            "no_local_base_pose_has_collision_safe_strict_pi_grasp", text
        )
        self.assertIn('"evaluated_base_count": 729', text)
        self.assertIn('"ik_query_count": 46656', text)
        self.assertIn('"collision_safe_grasp_count": 64', text)
        self.assertIn("cuRobo was intentionally not called", text)
        self.assertNotIn("tcp_poses", text)

    def test_feedback_distinguishes_partial_pi_budget_from_reachability(self) -> None:
        model = LLM({})
        messages = model.format_feedback(
            "prepare_for_manipulation('box', arm='left')",
            {
                "stdout": "",
                "stderr": "",
                "error": None,
                "interrupted_by": {
                    "primitive": "prepare_for_manipulation",
                    "reason": "_CandidatePlanningError:budget ended",
                    "result": {
                        "success": False,
                        "reason": "_CandidatePlanningError:budget ended",
                        "arm": "left",
                        "diagnostics": {
                            "stage": (
                                "parallel_pi_virtual_certification_"
                                "budget_exhausted"
                            ),
                            "evaluated_base_count": 729,
                            "ik_query_count": 160,
                            "pi_ik_requested_count": 256,
                            "collision_safe_grasp_count": 64,
                            "bases_with_converged_grasp": 0,
                        },
                    },
                },
                "primitive_calls": [{"name": "prepare_for_manipulation"}],
            },
            observation(),
        )

        text = messages[1]["content"][0]["text"]
        self.assertIn("pi_ik_budget_exhausted_without_strict_grasp", text)
        self.assertIn('"ik_query_count": 160', text)
        self.assertIn('"pi_ik_requested_count": 256', text)
        self.assertIn("inconclusive", text)
        self.assertIn("Try the other arm if appropriate", text)
        self.assertNotIn("Retry once", text)

    def test_unsupported_provider_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            LLM({"provider": "unknown"})

    def test_each_provider_has_its_own_default_model(self) -> None:
        self.assertEqual(LLM({"provider": "vertex"}).name, "gemini-2.5-pro")
        self.assertEqual(LLM({"provider": "qwen"}).name, "qwen3.7-plus")
        self.assertEqual(
            LLM({"provider": "deepseek"}).name,
            "deepseek-v4-flash-vision-exp",
        )
        self.assertEqual(LLM({"provider": "openai"}).name, "gpt-5.6-sol")

    def test_openai_uses_responses_multimodal_input_and_continues_by_id(
        self,
    ) -> None:
        model = LLM(
            {
                "provider": "openai",
                "name": "gpt-5.6-sol",
                "reasoning_effort": "high",
                "temperature": 0.3,
                "max_tokens": 8192,
            }
        )
        responses = _FakeResponses()
        model._client = SimpleNamespace(responses=responses)
        messages = model.initial_messages(
            task="Find the blue trash bin.",
            observation=observation(),
            primitive_docs="def observe(): ...",
        )
        exact_image_url = messages[-1]["content"][1]["image_url"]["url"]

        code = model.query(messages)
        messages.extend(
            model.format_feedback(
                code,
                {
                    "stdout": "",
                    "stderr": "",
                    "error": None,
                    "interrupted_by": None,
                    "primitive_calls": [{"name": "observe"}],
                },
                observation(),
            )
        )
        model.query(messages)

        first, second = responses.requests
        self.assertEqual(first["model"], "gpt-5.6-sol")
        self.assertEqual(first["reasoning"], {"effort": "high"})
        self.assertEqual(first["max_output_tokens"], 8192)
        self.assertNotIn("previous_response_id", first)
        self.assertEqual(len(first["input"]), 1)
        first_content = first["input"][0]["content"]
        self.assertEqual(first_content[0]["type"], "input_text")
        self.assertEqual(first_content[1]["type"], "input_image")
        self.assertEqual(first_content[1]["image_url"], exact_image_url)
        self.assertEqual(first_content[1]["detail"], "auto")
        self.assertEqual(second["previous_response_id"], "resp_1")
        self.assertEqual(len(second["input"]), 1)
        self.assertEqual(second["input"][0]["role"], "user")
        self.assertEqual(model.usage["prompt_tokens"], 246)
        self.assertEqual(model.usage["output_tokens"], 14)

    def test_openai_requires_its_own_api_key(self) -> None:
        model = LLM({"provider": "openai"})

        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ModelError) as caught:
                model._openai_client()

        self.assertIn("OPENAI_API_KEY", str(caught.exception))

    def test_qwen_receives_the_same_multimodal_message_and_returns_code(self) -> None:
        model = LLM(
            {
                "provider": "qwen",
                "name": "qwen3.7-plus",
                "temperature": 0.3,
                "max_tokens": 8192,
                "enable_thinking": False,
            }
        )
        completions = _FakeCompletions()
        model._client = SimpleNamespace(
            chat=SimpleNamespace(completions=completions)
        )
        messages = model.initial_messages(
            task="Go to the orange sofa.",
            observation=observation(),
            primitive_docs="def observe(): ...",
        )
        exact_image_url = messages[-1]["content"][1]["image_url"]["url"]

        code = model.query(messages)

        self.assertEqual(code, "observe()")
        request = completions.request
        self.assertEqual(request["model"], "qwen3.7-plus")
        self.assertEqual(request["extra_body"], {"enable_thinking": False})
        sent_url = request["messages"][-1]["content"][1]["image_url"]["url"]
        self.assertEqual(sent_url, exact_image_url)
        self.assertEqual(model.usage["prompt_tokens"], 123)
        self.assertEqual(model.usage["output_tokens"], 7)

    def test_qwen_requires_key_and_regional_endpoint(self) -> None:
        model = LLM({"provider": "qwen"})

        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ModelError) as caught:
                model._qwen_client()

        self.assertIn("DASHSCOPE_API_KEY", str(caught.exception))

    def test_deepseek_receives_multimodal_message_and_official_thinking_shape(
        self,
    ) -> None:
        model = LLM(
            {
                "provider": "deepseek",
                "name": "deepseek-v4-flash-vision-exp",
                "temperature": 0.3,
                "max_tokens": 8192,
                "enable_thinking": False,
            }
        )
        completions = _FakeCompletions()
        model._client = SimpleNamespace(
            chat=SimpleNamespace(completions=completions)
        )
        messages = model.initial_messages(
            task="Grasp the box.",
            observation=observation(),
            primitive_docs="def observe(): ...",
        )
        exact_image_url = messages[-1]["content"][1]["image_url"]["url"]

        code = model.query(messages)

        self.assertEqual(code, "observe()")
        request = completions.request
        self.assertEqual(request["model"], "deepseek-v4-flash-vision-exp")
        self.assertEqual(
            request["extra_body"], {"thinking": {"type": "disabled"}}
        )
        sent_url = request["messages"][-1]["content"][1]["image_url"]["url"]
        self.assertEqual(sent_url, exact_image_url)
        self.assertEqual(model.usage["prompt_tokens"], 123)
        self.assertEqual(model.usage["output_tokens"], 7)

    def test_deepseek_requires_its_own_api_key(self) -> None:
        model = LLM({"provider": "deepseek"})

        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ModelError) as caught:
                model._deepseek_client()

        self.assertIn("DEEPSEEK_API_KEY", str(caught.exception))

    def test_deepseek_defaults_to_official_base_url(self) -> None:
        model = LLM({"provider": "deepseek"})

        with patch.dict(
            os.environ, {"DEEPSEEK_API_KEY": "test-key"}, clear=True
        ), patch("openai.OpenAI") as openai_client:
            model._deepseek_client()

        self.assertEqual(
            openai_client.call_args.kwargs["base_url"],
            "https://api.deepseek.com",
        )


class _FakeCompletions:
    def __init__(self) -> None:
        self.request: dict = {}

    def create(self, **kwargs):
        self.request = kwargs
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="```python\nobserve()\n```")
                )
            ],
            usage=SimpleNamespace(prompt_tokens=123, completion_tokens=7),
        )


class _FakeResponses:
    def __init__(self) -> None:
        self.requests: list[dict] = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        return SimpleNamespace(
            id=f"resp_{len(self.requests)}",
            output_text="```python\nobserve()\n```",
            usage=SimpleNamespace(input_tokens=123, output_tokens=7),
        )


if __name__ == "__main__":
    unittest.main()
