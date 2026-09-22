"""One-shot Gemini or OpenAI advice loaded from offline video memory."""

from __future__ import annotations

import base64
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .llm import observation_rgb_data_url

PLANNER_PROMPT = (
    "You are a scene-aware navigation advisor for the physical YOR "
    "mobile-manipulation robot.\n\n"
    "Below is a navigation memory produced earlier by watching "
    "a passive human egocentric tour of this environment. It was intentionally "
    "built to capture navigable regions, landmarks, observed transitions, "
    "localization cues, and small-object evidence without knowing the current "
    "target. You also receive the operator's current task and the robot's "
    "current camera view. Infer the robot's likely location and give a concrete "
    "navigation instruction with useful subgoals and visual cues. Explain what "
    "the robot should look for next and how it can recognize progress. Preserve "
    "uncertainty and alternatives when the memory or current view is ambiguous."
    "\n\n"
    "The image labeled CURRENT ROBOT CAMERA IMAGE is the live VLM-ready ZED "
    "frame. Any images labeled PASSIVE-VIDEO REFERENCE FRAME are older evidence "
    "extracted from the human video at timestamps cited by the memory. Use those "
    "reference frames to verify identities, support surfaces, landmarks, and "
    "route cues, but never confuse them with the robot's current view.\n\n"
    "Embodiment constraint: YOR is a wheeled mobile base with arms extending "
    "left and right; in the travel pose it spans about 0.87 m side to side, so "
    "plan for a footprint reaching about 0.43 m to each side of the base "
    "center, with the grippers reaching about 0.46 m ahead. If a wall or "
    "furniture is "
    "close beside the robot, advise a short forward or lateral step to move the "
    "full footprint away from it before turning; do not merely point the camera "
    "away, advise reversing only when the rear path is known clear, and never "
    "treat out-of-view as proof of clearance. Once the target is visible, "
    "prefer dock_to_visible_object, whose Nav2 controller handles the full "
    "footprint.\n\n"
    "Your output is advice for another capable robot agent, not low-level "
    "control code and not a mandatory plan. The robot agent will decide how to "
    "use its own navigation and manipulation primitives. Do not claim that a "
    "target is currently visible unless the current robot image supports that "
    "claim. Do not use knowledge of this environment beyond the supplied memory "
    "and image.\n\n"
    "Keep the advice compact: at most 120 English words. Use exactly this "
    "structure:\n"
    "Location: one short sentence, including uncertainty only when needed.\n"
    "1. Two to four numbered navigation subgoals in total. Each subgoal must "
    "combine the action with one useful visual landmark or progress check.\n"
    "Target cue: one short sentence describing the final support surface or "
    "object evidence.\n"
    "Prefer viewpoint-invariant left/right landmark relationships over inferred "
    "compass directions. When useful, state one obvious wrong direction to "
    "avoid. Do not repeat the task, list every visible landmark, cite transition "
    "IDs or video timestamps, add generic safety advice, explain your reasoning, "
    "or use Markdown section headings.\n\n"
    "PASSIVE-VIDEO SCENE MEMORY:\n"
    "{memory}\n\n"
    "OPERATOR TASK:\n"
    "{task}\n\n"
    "Write the compact one-time navigation advice now."
)


SUPPORTED_MEMORY_SCHEMAS = {
    "yor-video-memory-v1",
    "yor-gemini-video-memory-v1",
}


class StartupNavigationPlanner:
    """Read saved memory and produce one advisory instruction at run startup."""

    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        client: Any | None = None,
    ) -> None:
        config = dict(config)
        memory_source = str(config.get("memory_path", "")).strip()
        if not memory_source:
            raise ValueError("navigation_planner.memory_path must not be empty")
        self.memory_path = Path(memory_source).expanduser()
        self.model = str(config.get("model", "gemini-3.7-flash")).strip()
        self.provider = _provider_for_model(self.model)
        self.thinking_level = str(
            config.get("reasoning_effort", config.get("thinking_level", "high"))
        ).strip()
        self.timeout_s = float(config.get("timeout_s", 300.0))
        self.max_memory_frames = int(config.get("max_memory_frames", 32))
        self.image_detail = str(config.get("image_detail", "high")).strip()
        allowed_levels = (
            {"none", "low", "medium", "high", "xhigh", "max"}
            if self.provider == "openai"
            else {"low", "medium", "high"}
        )
        if self.thinking_level not in allowed_levels:
            raise ValueError(
                "navigation_planner thinking level is unsupported by "
                f"{self.provider}: {self.thinking_level}"
            )
        if self.image_detail not in {"low", "high", "original", "auto"}:
            raise ValueError(
                "navigation_planner.image_detail must be low, high, original, or auto"
            )
        self._client = client

    def plan(
        self,
        *,
        task: str,
        observation: Mapping[str, Any],
        image_max_width: int,
    ) -> str:
        memory = json.loads(self.memory_path.read_text(encoding="utf-8"))
        if memory.get("schema_version") not in SUPPORTED_MEMORY_SCHEMAS:
            raise ValueError(
                f"unsupported navigation memory schema in {self.memory_path}"
            )
        memory_text = str(memory.get("memory_text", "")).strip()
        if not memory_text:
            raise ValueError(f"navigation memory is empty: {self.memory_path}")
        image_url = observation_rgb_data_url(
            observation, max_width=int(image_max_width)
        )
        if image_url is None:
            raise RuntimeError("current observation has no VLM-ready RGB image")
        prompt = PLANNER_PROMPT.format(memory=memory_text, task=task)
        reference_frames = self._load_reference_frames(memory)
        if self.provider == "openai":
            return self._plan_openai(prompt, image_url, reference_frames)
        return self._plan_gemini(prompt, image_url, reference_frames)

    def _load_reference_frames(
        self, memory: Mapping[str, Any]
    ) -> list[dict[str, str]]:
        loaded: list[dict[str, str]] = []
        for frame in list(memory.get("reference_frames") or [])[
            : self.max_memory_frames
        ]:
            if not isinstance(frame, Mapping):
                continue
            frame_path = Path(str(frame.get("path", ""))).expanduser()
            if not frame_path.is_absolute():
                frame_path = self.memory_path.parent / frame_path
            if not frame_path.is_file():
                raise FileNotFoundError(
                    f"navigation memory reference frame is missing: {frame_path}"
                )
            loaded.append(
                {
                    "timestamp": str(
                        frame.get("timestamp") or frame.get("time_s") or "unknown"
                    ),
                    "mime_type": str(frame.get("mime_type") or "image/jpeg"),
                    "data": base64.b64encode(frame_path.read_bytes()).decode("ascii"),
                }
            )
        return loaded

    def _plan_gemini(
        self,
        prompt: str,
        image_url: str,
        reference_frames: list[dict[str, str]],
    ) -> str:
        header, _, image_data = image_url.partition(",")
        mime_type = header.removeprefix("data:").split(";", 1)[0]
        inputs: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": prompt,
            },
            {"type": "text", "text": "CURRENT ROBOT CAMERA IMAGE:"},
            {
                "type": "image",
                "mime_type": mime_type,
                "data": image_data,
            },
        ]
        for frame in reference_frames:
            inputs.extend(
                [
                    {
                        "type": "text",
                        "text": (
                            "PASSIVE-VIDEO REFERENCE FRAME at "
                            f"{frame['timestamp']}:"
                        ),
                    },
                    {
                        "type": "image",
                        "mime_type": frame["mime_type"],
                        "data": frame["data"],
                    },
                ]
            )
        interaction = self._get_client().interactions.create(
            model=self.model,
            input=inputs,
            generation_config={"thinking_level": self.thinking_level},
            timeout=self.timeout_s,
        )
        advice = str(getattr(interaction, "output_text", "") or "").strip()
        if not advice:
            raise RuntimeError("Gemini returned empty startup navigation advice")
        return advice

    def _plan_openai(
        self,
        prompt: str,
        image_url: str,
        reference_frames: list[dict[str, str]],
    ) -> str:
        content: list[dict[str, Any]] = [
            {"type": "input_text", "text": prompt},
            {"type": "input_text", "text": "CURRENT ROBOT CAMERA IMAGE:"},
            {
                "type": "input_image",
                "image_url": image_url,
                "detail": self.image_detail,
            },
        ]
        for frame in reference_frames:
            content.extend(
                [
                    {
                        "type": "input_text",
                        "text": (
                            "PASSIVE-VIDEO REFERENCE FRAME at "
                            f"{frame['timestamp']}:"
                        ),
                    },
                    {
                        "type": "input_image",
                        "image_url": (
                            f"data:{frame['mime_type']};base64,{frame['data']}"
                        ),
                        "detail": self.image_detail,
                    },
                ]
            )
        response = self._get_client().responses.create(
            model=self.model,
            input=[{"role": "user", "content": content}],
            reasoning={"effort": self.thinking_level},
            timeout=self.timeout_s,
        )
        advice = str(getattr(response, "output_text", "") or "").strip()
        if not advice:
            raise RuntimeError("OpenAI returned empty startup navigation advice")
        return advice

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        if self.provider == "openai":
            api_key = os.environ.get("OPENAI_API_KEY", "").strip()
            if not api_key:
                raise RuntimeError(
                    "startup navigation planner requires OPENAI_API_KEY"
                )
            from openai import OpenAI

            self._client = OpenAI(api_key=api_key, timeout=self.timeout_s)
            return self._client
        api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError(
                "startup navigation planner requires GEMINI_API_KEY"
            )
        from google import genai

        self._client = genai.Client(api_key=api_key)
        return self._client


def _provider_for_model(model: str) -> str:
    normalized = str(model).strip().lower()
    if normalized.startswith("gpt-"):
        return "openai"
    if normalized.startswith("gemini-"):
        return "gemini"
    raise ValueError(
        f"cannot infer navigation planner provider from model {model!r}; "
        "use a gemini-* or gpt-* model"
    )


# Keep the old public name working for existing imports.
GeminiStartupNavigationPlanner = StartupNavigationPlanner


def policy_task_with_navigation_advice(task: str, advice: str) -> str:
    """Keep the operator task authoritative while exposing fallible advice."""

    return f"""\
Operator task: {task}

Fallible one-time navigation advice:
{advice}

Use current observations and primitive feedback; adapt if this advice conflicts with visible evidence or is not executable."""
