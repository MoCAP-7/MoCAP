"""Model adapter: provider calls, prompts, code extraction, feedback formatting.

This is the only component that knows about model providers. Vertex credentials
are resolved by ``google-genai`` from Google's standard environment variables;
Qwen reads ``DASHSCOPE_API_KEY`` and ``QWEN_BASE_URL``; DeepSeek reads
``DEEPSEEK_API_KEY`` and optionally ``DEEPSEEK_BASE_URL``. Secret values are
never placed in prompts, configuration, or traces.
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
from collections.abc import Mapping
from typing import Any

import numpy as np

from ..environment import summarize_observation
from ..exceptions import FormatError, ModelError

SYSTEM_PROMPT = """\
You are the policy generator for the physical YOR mobile robot. A human operator \
is supervising the robot right now.

Each turn you write ONE short Python program. It runs on the robot computer in a \
fresh namespace that contains only the documented primitive functions plus \
`finish()`. There are no imports, no variables carried over from your previous \
program, and no access to the robot connection, hardware bridge, or credentials.

Rules:
- Reply with Python only: either bare code, or exactly one ```python fenced block.
- Never write an `import` statement. NumPy and the alias `np` are not available;
  use ordinary Python lists for numeric constants and pass primitive return values
  directly to later primitives.
- Keep each program short. Call a few primitives, then look at what came back.
- You do not need to write `if not result["success"]` blocks. If a motion \
primitive fails, the base is stopped for you, the remaining lines of your \
program are skipped, and you receive the failure reason plus a fresh \
observation on the next turn.
- Whatever you `print()` is returned to you next turn; use it to carry findings \
forward, because your Python variables are not kept.
- Angles are degrees, positive is left. Straight distances are meters, positive \
is forward. For ``drive_lateral``, positive is robot-left and negative is \
robot-right.
- YOR is a wide wheeled base: with the arms in their travel pose it spans about \
0.87 m side to side (about 0.43 m to each side of the base center) and the \
grippers reach about 0.46 m ahead of the base center. ``dock_to_visible_object`` \
protects the full footprint through Nav2. ``drive_straight`` and the forward \
part of ``prepare_for_manipulation`` stop when live depth, or anything seen \
earlier in the same call, enters the swept footprint ahead. The camera looks \
down from the mast, so an object lower than about 0.7 m that is already within \
the gripper reach, or lower than about 0.45 m within half a metre of the camera, \
is invisible unless it was seen earlier in the same call; each new call starts \
without that memory. ``turn_relative`` has no obstacle check, ``drive_lateral`` \
and the sideways part of ``prepare_for_manipulation`` have no side check, and \
reverse has no rear check. \
When a wall or furniture appears close in the current view, treat it as a \
collision concern for the whole 0.87 m span: move the full swept footprint away, \
do not merely point the camera away, \
reverse only when the rear path is known clear, and remember that an obstacle \
leaving view is not proof that it is clear.
- Plan the route for your whole body, not for the camera. The arms stick out \
about 0.43 m to each side and about 0.46 m ahead of the base center, and a \
step is refused when furniture comes within a few centimetres of them, so \
keep about 0.3 m between the arms and furniture on both sides and pass sofas, \
desks and chairs through the middle of the open floor instead of along their \
edge. Size each step to the free space in view: in open floor move 0.5-1.0 m \
and turn 30-90 degrees at a time, and use short steps only when something is \
within about 1 m ahead.
- After ``obstacle_too_close``, do not retry the same heading with a shorter \
step. The failure's ``clearance_blocked_by`` says where the blocking cells are \
(``lat`` negative is to the right, positive to the left): turn or move away \
from that side first.
- For every manipulation primitive with an ``arm`` parameter, explicitly pass the \
keyword argument `arm="left"` or `arm="right"`. You should first check \
whether the target object is on the left or right \
side of the screen, then use the corresponding arm.
- Camera-relative XYZ offsets use the leveled ZED view frame: +X is horizontal right, \
+Y is vertical down, and +Z is horizontal forward. The robot compensates camera \
pitch and calibration internally; never transform these offsets yourself.
- In the initial/home configuration, each arm TCP is roughly 0.50 m away \
from the ZED camera horizontally. Treat this only as a rough spatial prior; use \
perception and collision-aware primitives for actual motion. When navigating, prefer to use \
``dock_to_visible_object`` when the target object is visible. 
- When grasping an object, always prefer ``goto_grasp_pose`` over constructing a \
grasp motion with ``goto_pose``. A ``goto_grasp_pose`` failure often means the \
current mobile-base pose is unfavorable; make a small base-position adjustment, \
re-observe, resample the grasp if needed, and retry.
- When ``prepare_for_manipulation`` is available, call it only when the target \
object is already visible in the current camera view and nearby. Call it \
before ``sample_grasp_pose`` and ``goto_grasp_pose``, using the same object name and arm \
throughout. It does not search for or center an off-screen object. Do not move \
the base or arm between those calls. If preparation \
reports that the target is too far, move closer before retrying. If it fails too many times \
then fall back using normal position adjustment primitives to make manipulation ready.
- When attempting a grasp or other manipulation, it is often helpful to position \
the mobile base closer to the table or manipulation workspace so the target is \
comfortably within arm reach.
- For a task that asks you to place or throw a \
previously grasped object into a receptacle, after navigate successfully to the \
receptacle. Do not call ``prepare_for_manipulation`` on the receptacle. Reuse \
the numeric position and quaternion from the earlier successful \
``goto_grasp_pose`` call by passing them to ``goto_pose`` with the same arm, \
then call ``open_gripper`` with that arm to release the object.
- ``dock_to_visible_object`` can fail to detect a target that is too far away or \
not yet clearly visible. You can try to observe the target object by yourself, \
and move closer with steps sized to the free space in view, re-observe after \
each step, then you can retry docking. If you have visually confirmed the object but \
``dock_to_visible_object`` remains unusable after repeated retries, fall back to \
short ``turn_relative``, ``drive_straight``, and (when the side path is known \
clear) ``drive_lateral`` steps to navigate toward it. \
Re-observe after every step, size angles and distances to the free space in \
view, and stop or change course immediately after any obstacle or safety failure. Never make a \
long blind approach toward an unconfirmed target.
- If ``dock_to_visible_object`` returns ``nav2_blocked_near_obstacle``, do not \
retry from the same pose. When the rear path is known to be clear, first use a \
short negative ``drive_straight`` step to back away, then re-observe and retry. \
If the rear path is not known clear, re-observe and choose another cautious \
adjustment instead of reversing blindly.
- Re-observe after each motion. Prefer stopping early over guessing when the way \
ahead is unclear, but do not creep along furniture in small steps.
- Call `finish(reason="...")` when you want to end the run. It only stops the \
loop; it does not assert that the goal was achieved.
"""

#: The coarse navigation baseline's prompt (configs/capx_coarse_navigation.yaml):
#: SYSTEM_PROMPT with the guidance for docking, manipulation readiness and the
#: fine motion primitives replaced by CaP-X's coarse navigation vocabulary.
CAPX_COARSE_NAVIGATION_SYSTEM_PROMPT = """\
You are the policy generator for the physical YOR mobile robot. A human operator \
is supervising the robot right now.

Each turn you write ONE short Python program. It runs on the robot computer in a \
fresh namespace that contains only the documented primitive functions plus \
`finish()`. There are no imports, no variables carried over from your previous \
program, and no access to the robot connection, hardware bridge, or credentials.

Rules:
- Reply with Python only: either bare code, or exactly one ```python fenced block.
- Never write an `import` statement. NumPy and the alias `np` are not available;
  use ordinary Python lists for numeric constants and pass primitive return values
  directly to later primitives.
- Keep each program short. Call a few primitives, then look at what came back.
- You do not need to write `if not result["success"]` blocks. If a motion \
primitive fails, the base is stopped for you, the remaining lines of your \
program are skipped, and you receive the failure reason plus a fresh \
observation on the next turn.
- Whatever you `print()` is returned to you next turn; use it to carry findings \
forward, because your Python variables are not kept.
- The base moves only through four navigation primitives: ``go_forward`` drives \
exactly 1 meter forward, ``turn_left_45_degrees`` and ``turn_right_45_degrees`` \
turn in place by 45 degrees, and ``goto_planar_position`` moves to a position \
relative to the robot while keeping its heading. Compose exploration and the \
approach from them with Python loops and conditionals, deciding each step from \
perception results, and use ``goto_planar_position`` when a continuous offset is \
needed. Angles are degrees; for ``goto_planar_position``, ``forward_m`` is \
positive forward and ``left_m`` is positive robot-left.
- YOR is a wide wheeled base: with the arms in their travel pose it spans about \
0.87 m side to side (about 0.43 m to each side of the base center) and the \
grippers reach about 0.46 m ahead of the base center. ``go_forward`` and the \
forward part of ``goto_planar_position`` stop when live depth, or anything seen \
earlier in the same call, enters the swept footprint ahead. The camera looks \
down from the mast, so an object lower than about 0.7 m that is already within \
the gripper reach, or lower than about 0.45 m within half a metre of the camera, \
is invisible unless it was seen earlier in the same call; each new call starts \
without that memory. The 45-degree turns have no obstacle check, the sideways \
part of ``goto_planar_position`` has no side check, and the base cannot drive \
backward. When a wall or furniture appears close in the current view, treat it \
as a collision concern for the whole 0.87 m span: move the full swept footprint \
away instead of merely pointing the camera away, and remember that an obstacle \
leaving view is not proof that it is clear.
- For every manipulation primitive with an ``arm`` parameter, explicitly pass the \
keyword argument `arm="left"` or `arm="right"`. You should first check \
whether the target object is on the left or right \
side of the screen, then use the corresponding arm.
- Camera-relative XYZ offsets use the leveled ZED view frame: +X is horizontal right, \
+Y is vertical down, and +Z is horizontal forward. The robot compensates camera \
pitch and calibration internally; never transform these offsets yourself.
- In the initial/home configuration, each arm TCP is roughly 0.50 m away \
from the ZED camera horizontally. Treat this only as a rough spatial prior; use \
perception and collision-aware primitives for actual motion.
- When grasping an object, always prefer ``goto_grasp_pose`` over constructing a \
grasp motion with ``goto_pose``. A ``goto_grasp_pose`` failure often means the \
current mobile-base pose is unfavorable; adjust the base position, \
re-observe, resample the grasp if needed, and retry.
- When attempting a grasp or other manipulation, it is often helpful to position \
the mobile base closer to the table or manipulation workspace so the target is \
comfortably within arm reach.
- For a task that asks you to place or throw a previously grasped object into a \
receptacle, after navigating to the receptacle, reuse the numeric position and \
quaternion from the earlier successful ``goto_grasp_pose`` call by passing them \
to ``goto_pose`` with the same arm, then call ``open_gripper`` with that arm to \
release the object.
- Re-observe after every motion, and change course immediately after any \
obstacle or safety failure. Never drive toward a target you have not confirmed \
in the current view.
- Call `finish(reason="...")` when you want to end the run. It only stops the \
loop; it does not assert that the goal was achieved.
"""

#: System prompts selectable with ``model.system_prompt``.
SYSTEM_PROMPTS = {
    "default": SYSTEM_PROMPT,
    "capx_coarse_navigation": CAPX_COARSE_NAVIGATION_SYSTEM_PROMPT,
}

TASK_TEMPLATE = """\
Task: {task}

Primitives available to your program:

{primitive_docs}

Current observation:
{observation}

Write your first Python program."""

FEEDBACK_TEMPLATE = """\
Your program was executed on the robot.

```python
{code}
```

{outcome}
Current observation:
{observation}

Write the next Python program, or call finish(reason="...")."""

FORMAT_ERROR_TEMPLATE = """\
Your reply could not be read as one Python program: {error}

Reply with Python only: either bare code, or exactly one ```python fenced block.
Do not add explanation outside the code."""

FENCE_PATTERN = re.compile(r"```([A-Za-z0-9_+-]*)\n(.*?)```", re.DOTALL)
DEFAULT_MODEL_NAMES = {
    "vertex": "gemini-2.5-pro",
    "qwen": "qwen3.7-plus",
    "deepseek": "deepseek-v4-flash-vision-exp",
    "openai": "gpt-5.6-sol",
    # Operator-typed programs (see models/manual.py); no API backend.
    "manual": "operator",
    # The fixed readiness-prior trial (see models/scripted.py); no API backend.
    "scripted": "readiness-trial",
}
PROVIDERS = frozenset(DEFAULT_MODEL_NAMES)


class LLM:
    """Vertex, Qwen, DeepSeek, and OpenAI policy-model adapter."""

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        config = dict(config or {})
        self.provider = str(config.get("provider", "vertex")).strip().lower()
        default_name = DEFAULT_MODEL_NAMES.get(self.provider, "gemini-2.5-pro")
        self.name = str(config.get("name", default_name)).strip()
        self.temperature = float(config.get("temperature", 1.0))
        self.max_tokens = int(config.get("max_tokens", 20480))
        self.image_max_width = int(config.get("image_max_width", 1024))
        self.base_url = str(config.get("base_url", "")).strip() or None
        self.enable_thinking = bool(config.get("enable_thinking", False))
        thinking_budget = config.get("thinking_budget")
        self.thinking_budget = (
            None if thinking_budget is None else int(thinking_budget)
        )
        self.reasoning_effort = str(
            config.get("reasoning_effort", "medium")
        ).strip().lower()
        self.timeout_s = float(config.get("timeout_s", 180.0))
        self.system_prompt_name = str(config.get("system_prompt", "default")).strip().lower()
        if self.system_prompt_name not in SYSTEM_PROMPTS:
            raise ValueError(
                f"model.system_prompt must be one of {sorted(SYSTEM_PROMPTS)}, "
                f"got {self.system_prompt_name!r}"
            )
        self.system_prompt = SYSTEM_PROMPTS[self.system_prompt_name]
        if self.provider not in PROVIDERS:
            raise ValueError(
                f"unsupported model provider {self.provider!r}; "
                "supported providers are 'vertex', 'qwen', 'deepseek', "
                "'openai', 'manual', and 'scripted'"
            )
        if not self.name:
            raise ValueError("model.name must not be empty")
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError("model.temperature must be in [0, 2]")
        if self.max_tokens <= 0:
            raise ValueError("model.max_tokens must be positive")
        if self.timeout_s <= 0:
            raise ValueError("model.timeout_s must be positive")
        if self.thinking_budget is not None and self.thinking_budget <= 0:
            raise ValueError("model.thinking_budget must be positive")
        if self.reasoning_effort not in {
            "none",
            "low",
            "medium",
            "high",
            "xhigh",
            "max",
        }:
            raise ValueError(
                "model.reasoning_effort must be one of "
                "'none', 'low', 'medium', 'high', 'xhigh', or 'max'"
            )
        self.n_calls = 0
        self.usage: dict[str, int] = {"prompt_tokens": 0, "output_tokens": 0}
        self.last_response: str | None = None
        self._client: Any | None = None
        self._openai_previous_response_id: str | None = None

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def initial_messages(
        self,
        *,
        task: str,
        observation: Mapping[str, Any],
        primitive_docs: str,
    ) -> list[dict[str, Any]]:
        text = TASK_TEMPLATE.format(
            task=task,
            primitive_docs=primitive_docs,
            observation=_observation_text(observation),
        )
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self._parts(text, observation)},
        ]

    def format_feedback(
        self,
        code: str,
        execution: Mapping[str, Any],
        observation: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        """Assistant turn plus the execution/observation feedback message."""

        text = FEEDBACK_TEMPLATE.format(
            code=code,
            outcome=_execution_text(execution),
            observation=_observation_text(observation),
        )
        return [
            {"role": "assistant", "content": self.last_response or code},
            {"role": "user", "content": self._parts(text, observation)},
        ]

    def format_error_feedback(self, error: BaseException) -> list[dict[str, Any]]:
        return [
            {"role": "assistant", "content": self.last_response or ""},
            {"role": "user", "content": FORMAT_ERROR_TEMPLATE.format(error=error)},
        ]

    def _parts(
        self, text: str, observation: Mapping[str, Any] | None
    ) -> list[dict[str, Any]]:
        parts: list[dict[str, Any]] = [{"type": "text", "text": text}]
        url = observation_rgb_data_url(observation, max_width=self.image_max_width)
        if url is not None:
            parts.append({"type": "image_url", "image_url": {"url": url}})
        return parts

    # ------------------------------------------------------------------
    # Provider call
    # ------------------------------------------------------------------

    def query(self, messages: list[Mapping[str, Any]]) -> str:
        """Ask the model for one policy and normalize it to a code string.

        Raises:
            ModelError: the provider call itself failed.
            FormatError: the response is empty or is not one Python program.
        """

        response = self._generate(messages)
        self.last_response = response
        return extract_code(response)

    def _generate(self, messages: list[Mapping[str, Any]]) -> str:
        if self.provider == "manual":
            raise ModelError(
                "the 'manual' provider has no API backend; build the run with "
                "yor_agent.models.manual.ManualModel"
            )
        if self.provider == "scripted":
            raise ModelError(
                "the 'scripted' provider has no API backend; build the run with "
                "yor_agent.models.scripted.ReadinessTrialModel"
            )
        if self.provider == "openai":
            return self._generate_openai(messages)
        if self.provider == "qwen":
            return self._generate_qwen(messages)
        if self.provider == "deepseek":
            return self._generate_deepseek(messages)
        return self._generate_vertex(messages)

    def _generate_vertex(self, messages: list[Mapping[str, Any]]) -> str:
        from google.genai import types

        system_instruction, contents = _to_gemini_contents(messages)
        try:
            response = self._genai_client().models.generate_content(
                model=self.name,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=self.temperature,
                    max_output_tokens=self.max_tokens,
                ),
            )
        except Exception as exc:  # noqa: BLE001 - surfaced as one agent-level error
            raise ModelError(f"{self.provider}/{self.name} query failed: {exc}") from exc
        self.n_calls += 1
        self._record_vertex_usage(response)
        return response.text or ""

    def _generate_qwen(self, messages: list[Mapping[str, Any]]) -> str:
        """Call Qwen's OpenAI-compatible multimodal Chat Completions API."""

        extra_body: dict[str, Any] = {"enable_thinking": self.enable_thinking}
        if self.thinking_budget is not None:
            extra_body["thinking_budget"] = self.thinking_budget
        # The trajectory is intentionally OpenAI-shaped already. Round-tripping
        # through JSON detaches nested Mapping objects without touching inline
        # base64 image URLs.
        qwen_messages = json.loads(json.dumps(messages))
        try:
            response = self._qwen_client().chat.completions.create(
                model=self.name,
                messages=qwen_messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                extra_body=extra_body,
            )
        except ModelError:
            raise
        except Exception as exc:  # noqa: BLE001 - surfaced as one agent-level error
            raise ModelError(f"{self.provider}/{self.name} query failed: {exc}") from exc
        self.n_calls += 1
        self._record_openai_usage(response)
        return self._openai_response_text(response)

    def _generate_openai(self, messages: list[Mapping[str, Any]]) -> str:
        """Call OpenAI's Responses API with persisted multi-turn reasoning."""

        previous_response_id = self._openai_previous_response_id
        input_messages = messages[-1:] if previous_response_id else messages
        request: dict[str, Any] = {
            "model": self.name,
            "instructions": self.system_prompt,
            "input": _to_openai_responses_input(input_messages),
            "reasoning": {"effort": self.reasoning_effort},
            "temperature": self.temperature,
            "max_output_tokens": self.max_tokens,
        }
        if previous_response_id:
            request["previous_response_id"] = previous_response_id
        try:
            response = self._openai_client().responses.create(**request)
        except ModelError:
            raise
        except Exception as exc:  # noqa: BLE001 - surfaced as one agent-level error
            raise ModelError(f"{self.provider}/{self.name} query failed: {exc}") from exc
        self.n_calls += 1
        response_id = str(getattr(response, "id", "") or "").strip()
        if response_id:
            self._openai_previous_response_id = response_id
        self._record_responses_usage(response)
        return str(getattr(response, "output_text", "") or "")

    def _generate_deepseek(self, messages: list[Mapping[str, Any]]) -> str:
        """Call DeepSeek's official OpenAI-compatible multimodal endpoint."""

        deepseek_messages = json.loads(json.dumps(messages))
        thinking_type = "enabled" if self.enable_thinking else "disabled"
        try:
            response = self._deepseek_client().chat.completions.create(
                model=self.name,
                messages=deepseek_messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                extra_body={"thinking": {"type": thinking_type}},
            )
        except ModelError:
            raise
        except Exception as exc:  # noqa: BLE001 - surfaced as one agent-level error
            raise ModelError(f"{self.provider}/{self.name} query failed: {exc}") from exc
        self.n_calls += 1
        self._record_openai_usage(response)
        return self._openai_response_text(response)

    def _openai_response_text(self, response: Any) -> str:
        """Extract assistant text from an OpenAI-compatible chat response."""

        try:
            content = response.choices[0].message.content
        except (AttributeError, IndexError, TypeError) as exc:
            raise ModelError(
                f"{self.provider}/{self.name} returned no assistant message"
            ) from exc
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            text_parts: list[str] = []
            for part in content:
                if isinstance(part, Mapping):
                    text_parts.append(str(part.get("text", "")))
                else:
                    text_parts.append(str(getattr(part, "text", "")))
            return "".join(text_parts)
        return "" if content is None else str(content)

    def _genai_client(self) -> Any:
        if self._client is None:
            from google import genai

            # Credentials/project/location come from the standard Google
            # environment variables; nothing here reads or logs them.
            self._client = genai.Client()
        return self._client

    def _qwen_client(self) -> Any:
        if self._client is not None:
            return self._client
        api_key = os.environ.get("DASHSCOPE_API_KEY", "").strip()
        base_url = self.base_url or os.environ.get("QWEN_BASE_URL", "").strip()
        if not api_key:
            raise ModelError("qwen requires DASHSCOPE_API_KEY in the process environment")
        if not base_url:
            raise ModelError("qwen requires a region-specific QWEN_BASE_URL (or model.base_url)")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ModelError("qwen requires the 'openai' Python package; reinstall yor-agent") from exc
        self._client = OpenAI(api_key=api_key, base_url=base_url, timeout=self.timeout_s)
        return self._client

    def _deepseek_client(self) -> Any:
        if self._client is not None:
            return self._client
        api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
        base_url = (
            self.base_url
            or os.environ.get("DEEPSEEK_BASE_URL", "").strip()
            or "https://api.deepseek.com"
        )
        if not api_key:
            raise ModelError(
                "deepseek requires DEEPSEEK_API_KEY in the process environment"
            )
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ModelError(
                "deepseek requires the 'openai' Python package; reinstall yor-agent"
            ) from exc
        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=self.timeout_s,
        )
        return self._client

    def _openai_client(self) -> Any:
        if self._client is not None:
            return self._client
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise ModelError(
                "openai requires OPENAI_API_KEY in the process environment"
            )
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ModelError(
                "openai requires the 'openai' Python package; reinstall yor-agent"
            ) from exc
        kwargs: dict[str, Any] = {
            "api_key": api_key,
            "timeout": self.timeout_s,
        }
        if self.base_url:
            kwargs["base_url"] = self.base_url
        self._client = OpenAI(**kwargs)
        return self._client

    def _record_vertex_usage(self, response: Any) -> None:
        usage = getattr(response, "usage_metadata", None)
        if usage is None:
            return
        self.usage["prompt_tokens"] += int(getattr(usage, "prompt_token_count", 0) or 0)
        self.usage["output_tokens"] += int(
            getattr(usage, "candidates_token_count", 0) or 0
        )

    def _record_openai_usage(self, response: Any) -> None:
        usage = getattr(response, "usage", None)
        if usage is None:
            return
        self.usage["prompt_tokens"] += int(getattr(usage, "prompt_tokens", 0) or 0)
        self.usage["output_tokens"] += int(getattr(usage, "completion_tokens", 0) or 0)

    def _record_responses_usage(self, response: Any) -> None:
        usage = getattr(response, "usage", None)
        if usage is None:
            return
        self.usage["prompt_tokens"] += int(
            getattr(usage, "input_tokens", 0) or 0
        )
        self.usage["output_tokens"] += int(
            getattr(usage, "output_tokens", 0) or 0
        )


def _to_openai_responses_input(
    messages: list[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Translate the shared multimodal trajectory into Responses API input."""

    converted: list[dict[str, Any]] = []
    for message in messages:
        role = str(message.get("role", "user"))
        if role == "system":
            continue
        content = message.get("content", "")
        if isinstance(content, str):
            converted.append({"role": role, "content": content})
            continue
        parts: list[dict[str, Any]] = []
        for part in content if isinstance(content, list) else []:
            if not isinstance(part, Mapping):
                continue
            part_type = part.get("type")
            if part_type == "text":
                parts.append(
                    {"type": "input_text", "text": str(part.get("text", ""))}
                )
            elif part_type == "image_url":
                image = part.get("image_url")
                if isinstance(image, Mapping):
                    image_url = str(image.get("url", ""))
                    detail = str(image.get("detail", "auto"))
                else:
                    image_url = str(image or "")
                    detail = "auto"
                parts.append(
                    {
                        "type": "input_image",
                        "image_url": image_url,
                        "detail": detail,
                    }
                )
        converted.append({"role": role, "content": parts})
    return converted

# ----------------------------------------------------------------------
# Code extraction
# ----------------------------------------------------------------------


def extract_code(response: str) -> str:
    """Normalize a model response into exactly one Python code string.

    Accepts bare Python, or exactly one fenced block. Unlike the CaP-X parser
    this never falls back to substring matching, and it never silently keeps
    prose as if it were code.
    """

    if not isinstance(response, str) or not response.strip():
        raise FormatError("the response was empty")
    blocks = FENCE_PATTERN.findall(response)
    if len(blocks) > 1:
        raise FormatError(
            f"the response contained {len(blocks)} code blocks; send exactly one"
        )
    if blocks:
        language, code = blocks[0]
        if language and language.lower() not in {"python", "py"}:
            raise FormatError(f"code block language was {language!r}, expected python")
        code = code.strip()
        if not code:
            raise FormatError("the code block was empty")
        # A syntax error inside a proper code block is the model's *program*
        # being wrong, not its *format*; the executor reports it as feedback.
        return code

    code = response.strip()
    try:
        compile(code, "<policy>", "exec")
    except SyntaxError as exc:
        # Unfenced text that will not compile is prose, not a program.
        raise FormatError(f"the response is not valid Python ({exc.msg})") from exc
    return code


# ----------------------------------------------------------------------
# Rendering helpers
# ----------------------------------------------------------------------


def _observation_text(observation: Mapping[str, Any] | None) -> str:
    if not observation:
        return "(unavailable)"
    return json.dumps(summarize_observation(observation), indent=2)


def _execution_text(execution: Mapping[str, Any]) -> str:
    sections: list[str] = []
    stdout = execution.get("stdout") or ""
    sections.append(f"stdout:\n```\n{stdout.strip() or '(empty)'}\n```")
    stderr = execution.get("stderr") or ""
    if stderr.strip():
        sections.append(f"stderr:\n```\n{stderr.strip()}\n```")

    interrupted = execution.get("interrupted_by")
    error = execution.get("error")
    if interrupted:
        feedback_result = _primitive_failure_result_for_model(interrupted)
        feedback_reason = str(
            feedback_result.get("reason") or interrupted.get("reason") or "unknown"
        )
        sections.append(
            f"A primitive failed, so the base was stopped and the rest of your "
            f"program was skipped:\n```\n"
            f"{interrupted['primitive']}: {feedback_reason}\n"
            f"{json.dumps(feedback_result, indent=2)}\n```"
        )
    elif error:
        sections.append(
            f"Your program raised an exception, so the rest of it did not run:\n"
            f"```\n{error.get('traceback') or error.get('message')}\n```"
        )
    else:
        sections.append("Your program ran to completion without raising.")

    calls = execution.get("primitive_calls") or []
    if calls:
        names = ", ".join(f"{call['name']}()" for call in calls)
        sections.append(f"Primitives called, in order: {names}")
    return "\n\n".join(sections) + "\n"


def _primitive_failure_result_for_model(
    interrupted: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Keep trace detail while giving the model a concise recovery diagnosis."""

    result = interrupted.get("result")
    if not isinstance(result, Mapping):
        return {}
    primitive = interrupted.get("primitive")
    if primitive == "goto_grasp_pose":
        return _goto_grasp_pose_failure_for_model(result)
    if primitive == "prepare_for_manipulation":
        return _prepare_for_manipulation_failure_for_model(result)
    return result


def _prepare_for_manipulation_failure_for_model(
    result: Mapping[str, Any],
) -> dict[str, Any]:
    """Summarize readiness failures without copying planner tensors into a prompt.

    The executor retains the complete primitive result in the trace.  This view
    deliberately contains only bounded scalar aggregates and recovery guidance;
    per-base poses, TCP matrices, joint seeds, and per-seed arrays are debugging
    evidence for humans rather than useful policy context for the model.
    """

    raw_reason = str(result.get("reason") or "unknown")
    diagnostics = result.get("diagnostics")
    diagnostics = diagnostics if isinstance(diagnostics, Mapping) else {}
    diagnostic_stage = str(diagnostics.get("stage") or "")
    if raw_reason.startswith("_CandidatePlanningError:"):
        if diagnostic_stage == "parallel_pi_virtual_certification":
            reason = "no_local_base_pose_has_collision_safe_strict_pi_grasp"
        elif (
            diagnostic_stage
            == "parallel_pi_virtual_certification_budget_exhausted"
        ):
            reason = "pi_ik_budget_exhausted_without_strict_grasp"
        elif diagnostic_stage == "parallel_pi_ik_execution":
            reason = "parallel_pi_ik_batch_failed_or_timed_out"
        else:
            reason = "no_local_base_pose_passed_parallel_curobo"
    elif raw_reason.startswith("TimeoutError:"):
        reason = "grasp_motion_planner_timeout"
    else:
        # Bound unexpected exception text as a final guard against a service
        # embedding a large diagnostic payload in its message.
        reason = raw_reason[:240]

    summary: dict[str, Any] = {
        "primitive": "prepare_for_manipulation",
        "success": False,
        "reason": reason,
        "arm": result.get("arm"),
    }
    for key in ("target_distance_m", "maximum_distance_m"):
        value = result.get(key)
        if isinstance(value, (int, float)) and np.isfinite(value):
            summary[key] = float(value)

    recovery = result.get("recovery")
    if isinstance(recovery, Mapping):
        summary["recovery"] = {
            key: recovery.get(key) for key in ("action", "message") if recovery.get(key)
        }

    if isinstance(diagnostics, Mapping):
        attempts = diagnostics.get("attempts")
        attempts = attempts if isinstance(attempts, list) else []
        reason_counts: dict[str, int] = {}
        successful_bases = 0
        bases_with_strict_pi = 0
        strict_pi_grasps = 0
        converged_pi_grasps = 0
        finite_pi_grasps = 0
        for attempt in attempts:
            if not isinstance(attempt, Mapping):
                continue
            attempt_reason = str(attempt.get("reason") or "unknown")
            reason_counts[attempt_reason] = reason_counts.get(attempt_reason, 0) + 1
            successful_bases += int(bool(attempt.get("success", False)))
            strict_count = int(attempt.get("strict_pi_count") or 0)
            bases_with_strict_pi += int(strict_count > 0)
            strict_pi_grasps += strict_count
            converged_pi_grasps += int(attempt.get("converged_pi_count") or 0)
            finite_pi_grasps += int(attempt.get("finite_pi_count") or 0)

        batch = diagnostics.get("batch")
        batch = batch if isinstance(batch, Mapping) else {}
        planner = diagnostics.get("planner")
        planner = planner if isinstance(planner, Mapping) else {}
        evaluated_base_count = int(
            diagnostics.get("evaluated_base_count") or len(attempts)
        )
        compact_diagnostics: dict[str, Any] = {
            "stage": diagnostics.get("stage"),
            "evaluated_base_count": evaluated_base_count,
            "successful_base_count": successful_bases,
            "failure_reason_counts": reason_counts,
            "pi_ranking": {
                "bases_with_acceptable_grasp": bases_with_strict_pi,
                "acceptable_grasp_count": strict_pi_grasps,
                "converged_grasp_count": converged_pi_grasps,
                "finite_grasp_count": finite_pi_grasps,
            },
        }
        if diagnostic_stage in {
            "parallel_pi_virtual_certification",
            "parallel_pi_virtual_certification_budget_exhausted",
            "parallel_pi_ik_execution",
        }:
            compact_diagnostics["pi_ranking"] = {
                "ik_query_count": int(diagnostics.get("ik_query_count") or 0),
                "pi_ik_requested_count": int(
                    diagnostics.get("pi_ik_requested_count") or 0
                ),
                "pi_eligible_pair_count": int(
                    diagnostics.get("pi_eligible_pair_count") or 0
                ),
                "pi_shortlisted_base_count": int(
                    diagnostics.get("pi_shortlisted_base_count") or 0
                ),
                "collision_safe_grasp_count": int(
                    diagnostics.get("collision_safe_grasp_count") or 0
                ),
                "nominal_grasp_evaluated_count": int(
                    diagnostics.get("nominal_grasp_evaluated_count") or 0
                ),
                "nominal_grasp_converged_count": int(
                    diagnostics.get("nominal_grasp_converged_count") or 0
                ),
                "robustness_query_count": int(
                    diagnostics.get("robustness_query_count") or 0
                ),
                "bases_with_converged_grasp": int(
                    diagnostics.get("bases_with_converged_grasp") or 0
                ),
                "bases_with_nominal_converged_grasp": int(
                    diagnostics.get("bases_with_nominal_converged_grasp") or 0
                ),
                "robustness_enabled": bool(
                    diagnostics.get("robustness_enabled", False)
                ),
                "robustness_variant_count": int(
                    diagnostics.get("robustness_variant_count") or 0
                ),
                "robustness_preferred_converged_variants": int(
                    diagnostics.get("robustness_preferred_converged_variants")
                    or diagnostics.get("robustness_min_converged_variants")
                    or 0
                ),
            }
        else:
            compact_diagnostics["curobo"] = {
                "status": planner.get("status"),
                "planning_time_s": batch.get("planning_time_s"),
                "goalset_width": batch.get("batch_goalset_count"),
                "position_tolerance_m": planner.get("position_tolerance_m"),
                "rotation_tolerance_rad": planner.get("rotation_tolerance_rad"),
            }
        summary["diagnostics_summary"] = compact_diagnostics

    if reason == "no_local_base_pose_has_collision_safe_strict_pi_grasp":
        if bool(diagnostics.get("robustness_enabled", False)):
            summary["interpretation"] = (
                "No evaluated local base pose had a collision-safe nominal grasp "
                "that also met the configured Pi convergence count across its "
                "small base-pose perturbations; cuRobo was intentionally not called."
            )
        else:
            summary["interpretation"] = (
                "No evaluated local base pose had even one collision-safe grasp "
                "with Pi ik_converged=true; cuRobo was intentionally not called."
            )
        summary["recommended_next_actions"] = [
            "Do not retry the identical preparation from the unchanged base pose.",
            "Use the other arm if appropriate; otherwise make a small base "
            "adjustment or dock closer before retrying.",
        ]
    elif reason == "pi_ik_budget_exhausted_without_strict_grasp":
        summary["interpretation"] = (
            "The bounded Pi IK search completed only part of its shortlist and "
            "found no strict grasp in that computed prefix. This is inconclusive, "
            "not evidence that every shortlisted grasp is unreachable."
        )
        summary["recommended_next_actions"] = [
            "Try the other arm if appropriate.",
            "Otherwise inspect per-wave timing or make a small base adjustment "
            "before another preparation attempt.",
        ]
    elif reason == "parallel_pi_ik_batch_failed_or_timed_out":
        summary["interpretation"] = (
            "The exact Pi IK batch did not return within its service deadline; "
            "this is a computation failure, not evidence that all grasps are unreachable."
        )
        summary["recommended_next_actions"] = [
            "Try the other arm if appropriate.",
            "Otherwise confirm the Pi arm service reports multiple IK workers, "
            "then reduce pi_ik_candidate_limit or inspect the retained trace timing.",
        ]
    elif reason == "no_local_base_pose_passed_parallel_curobo":
        pi_summary = summary.get("diagnostics_summary", {}).get("pi_ranking", {})
        if pi_summary.get("bases_with_acceptable_grasp") == 0:
            interpretation = (
                "Neither Pi IK nor cuRobo found an acceptable grasp for the "
                "evaluated base poses with this arm; this is not a trajectory-stage failure."
            )
        else:
            interpretation = (
                "Pi IK found at least one acceptable candidate, but cuRobo rejected "
                "all evaluated base worlds."
            )
        summary["interpretation"] = interpretation
        summary["recommended_next_actions"] = [
            "Do not retry the identical preparation from the unchanged base pose.",
            "Use the other arm if the object lies on that arm's side; otherwise make "
            "a small base adjustment or dock closer before retrying.",
        ]
    elif reason == "grasp_motion_planner_timeout":
        summary["interpretation"] = (
            "The client timed out; the planner service may still be finishing the "
            "abandoned GPU request."
        )
        summary["recommended_next_actions"] = [
            "Do not immediately submit the same expensive preparation request again."
        ]

    return summary


def _goto_grasp_pose_failure_for_model(
    result: Mapping[str, Any],
) -> dict[str, Any]:
    reason = str(result.get("reason") or "unknown")
    status = str(result.get("status") or "")
    terminal = result.get("terminal_check")
    terminal = terminal if isinstance(terminal, Mapping) else {}
    safe_raw = terminal.get("safe")
    safe_values = (
        [bool(value) for value in safe_raw]
        if isinstance(safe_raw, list)
        else []
    )
    terminal_count = len(safe_values)
    terminal_safe_count = sum(safe_values)
    terminal_collision_count = terminal_count - terminal_safe_count
    clearances = terminal.get("minimum_clearance_m")
    finite_clearances = []
    if isinstance(clearances, list):
        finite_clearances = [
            float(value)
            for value in clearances
            if isinstance(value, (int, float)) and np.isfinite(value)
        ]

    if reason == "grasp_goal_ik_failed":
        failure_stage = "goal_ik"
        interpretation = (
            "No collision-aware cuRobo IK seed reached an acceptable grasp goal. "
            "This failure occurred before trajectory planning."
        )
        next_actions = [
            "Do not retry the identical pose from the unchanged base position.",
            "Make a small mobile-base adjustment, usually a few centimeters "
            "closer or a small yaw change, then re-observe and resample the grasp.",
            "Use the Pi IK residuals below to judge whether a small adjustment is "
            "likely enough; try the other arm only when its geometry is better.",
        ]
    elif reason == "all_goalset_gripper_swept_volumes_collide":
        failure_stage = "terminal_gripper_collision"
        interpretation = (
            "Every grasp candidate collided in the exact open-gripper terminal "
            "or approach-sweep check; goal IK and trajectory planning were not run."
        )
        next_actions = [
            "Adjust the base pose or viewing angle, re-observe, and resample.",
            "Do not relax collision clearance or blindly retry the same pose.",
        ]
    elif reason == "no_collision_free_grasp_path":
        failure_stage = "trajectory_planning"
        interpretation = (
            "At least one terminal grasp passed the mesh gate, but cuRobo did not "
            "find a complete collision-free approach and grasp trajectory."
        )
        next_actions = [
            "Change the base position or approach angle, then re-observe and resample.",
            "Avoid repeatedly executing the same grasp pose in the same scene.",
        ]
    elif reason == "requested_grasp_is_not_on_fresh_target_instance":
        failure_stage = "fresh_target_validation"
        interpretation = (
            "The requested grasp no longer matches the freshly segmented target."
        )
        next_actions = [
            "Re-observe and call sample_grasp_pose again before retrying execution."
        ]
    elif reason in {
        "pi_grasp_goal_joint_seed_unavailable",
        "goal_joint_seed_tcp_mismatch",
    }:
        failure_stage = "pi_ik_or_seed_validation"
        interpretation = (
            "The Pi IK result was unavailable or failed the independent TCP "
            "consistency check before execution."
        )
        next_actions = [
            "Re-observe and resample after a small base adjustment.",
            "Do not manually construct a replacement grasp with goto_pose.",
        ]
    else:
        failure_stage = "planner_validation"
        interpretation = (
            "The grasp planner or its post-plan safety validation rejected the request."
        )
        next_actions = [
            "Use the reason and scalar diagnostics below, adjust the base or view, "
            "then re-observe and resample rather than blindly retrying."
        ]

    if failure_stage == "goal_ik" and terminal_count:
        interpretation += (
            f" {terminal_safe_count}/{terminal_count} candidates passed the terminal "
            "gripper collision gate, so any individually colliding alternatives "
            "are not the stated cause of this goal-IK failure."
        )

    return {
        "primitive": "goto_grasp_pose",
        "success": False,
        "reason": reason,
        "planner_status": status or None,
        "failure_stage": failure_stage,
        "interpretation": interpretation,
        "arm": result.get("arm"),
        "goalset": {
            "candidate_count": result.get(
                "goalset_candidate_count", terminal_count or None
            ),
            "terminal_checked_count": terminal_count,
            "terminal_safe_count": terminal_safe_count,
            "terminal_collision_count": terminal_collision_count,
            "all_terminal_candidates_safe": bool(
                terminal_count and terminal_safe_count == terminal_count
            ),
            "minimum_terminal_clearance_m": (
                min(finite_clearances) if finite_clearances else None
            ),
            "required_terminal_clearance_m": terminal.get("clearance_m"),
        },
        "pi_ik": {
            "converged": result.get("pi_goal_ik_converged"),
            "position_error_m": result.get("pi_goal_ik_position_error_m"),
            "rotation_error_rad": result.get("pi_goal_ik_rotation_error_rad"),
            "accepted_position_error_m": result.get(
                "grasp_ik_position_tolerance_m"
            ),
            "accepted_rotation_error_rad": result.get(
                "grasp_ik_rotation_tolerance_rad"
            ),
            "joint_seed_forwarded": result.get("pi_goal_joint_seed_forwarded"),
        },
        "fresh_target": {
            "requested_pose_to_target_m": result.get("target_distance_m"),
            "sam_score": result.get("sam_score"),
        },
        "planning_scene": {
            "input_points": result.get("input_scene_points"),
            "planning_points": result.get("planning_scene_points"),
            "removed_robot_points": result.get("removed_robot_points"),
            "planning_time_s": result.get("planning_time_s"),
        },
        "recommended_next_actions": next_actions,
    }


def observation_rgb_data_url(
    observation: Mapping[str, Any] | None, *, max_width: int
) -> str | None:
    """Encode the exact RGB image shown to policy VLMs as a PNG data URL.

    The startup navigation planner deliberately calls this same function so it
    never receives a higher-resolution or otherwise different "raw" frame.
    """

    if not observation:
        return None
    try:
        rgb = observation["robot0_robotview"]["images"]["rgb"]
    except (KeyError, TypeError):
        return None
    if rgb is None:
        return None
    array = np.asarray(rgb)
    if array.ndim != 3 or array.shape[2] not in (3, 4):
        return None

    from PIL import Image

    image = Image.fromarray(np.ascontiguousarray(array[:, :, :3], dtype=np.uint8))
    if max_width > 0 and image.width > max_width:
        height = max(1, round(image.height * max_width / image.width))
        image = image.resize((max_width, height), Image.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def _to_gemini_contents(messages: list[Mapping[str, Any]]) -> tuple[str | None, list]:
    """Convert the OpenAI-style trajectory into genai contents."""

    from google.genai import types

    role_map = {"user": "user", "assistant": "model", "model": "model"}
    system_instruction: str | None = None
    contents = []
    for message in messages:
        content = message.get("content")
        items = (
            [{"type": "text", "text": content}] if isinstance(content, str) else content
        )
        if message.get("role") == "system":
            system_instruction = "\n".join(
                item["text"] for item in items if item.get("type") == "text"
            )
            continue
        parts = []
        for item in items:
            if item.get("type") == "text":
                parts.append(types.Part.from_text(text=item["text"]))
            elif item.get("type") == "image_url":
                header, _, payload = item["image_url"]["url"].partition(",")
                mime = header.split(";")[0].removeprefix("data:")
                parts.append(
                    types.Part.from_bytes(data=base64.b64decode(payload), mime_type=mime)
                )
        if parts:
            contents.append(
                types.Content(role=role_map.get(message.get("role"), "user"), parts=parts)
            )
    return system_instruction, contents
