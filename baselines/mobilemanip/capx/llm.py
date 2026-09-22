"""OpenAI Responses adapter matching yor_agent's GPT-5.6 call path."""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any


class OpenAIResponsesQuery:
    """Callable replacement for Cap-X's internal model-query function."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        timeout_s: float = 180.0,
        client: Any | None = None,
    ) -> None:
        self._api_key = str(api_key or os.environ.get("OPENAI_API_KEY", "")).strip()
        if not self._api_key and client is None:
            raise RuntimeError("GPT-5.6 Sol requires OPENAI_API_KEY")
        self._timeout_s = float(timeout_s)
        self._client = client
        self._previous_response_ids: dict[int, str] = {}

    def __call__(
        self, args: Any, prompt: list[Mapping[str, Any]]
    ) -> dict[str, Any]:
        instructions = _system_instructions(prompt)
        inputs = _responses_input(prompt)
        session_key = id(args)
        previous_response_id = self._previous_response_ids.get(session_key)
        if previous_response_id:
            inputs = inputs[-1:]
        request: dict[str, Any] = {
            "model": str(args.model),
            "instructions": instructions,
            "input": inputs,
            "reasoning": {"effort": str(args.reasoning_effort)},
            "temperature": float(args.temperature),
            "max_output_tokens": int(args.max_tokens),
        }
        if previous_response_id:
            request["previous_response_id"] = previous_response_id
        response = self._get_client().responses.create(**request)
        response_id = str(getattr(response, "id", "") or "").strip()
        if response_id:
            self._previous_response_ids[session_key] = response_id
        content = str(getattr(response, "output_text", "") or "").strip()
        if not content:
            raise RuntimeError("OpenAI returned an empty GPT-5.6 Sol response")
        return {"content": content, "reasoning": None}

    def _get_client(self) -> Any:
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(
                api_key=self._api_key,
                timeout=self._timeout_s,
            )
        return self._client


def _system_instructions(messages: list[Mapping[str, Any]]) -> str:
    blocks: list[str] = []
    for message in messages:
        if str(message.get("role", "")) != "system":
            continue
        content = message.get("content", "")
        if isinstance(content, str):
            blocks.append(content)
        elif isinstance(content, list):
            blocks.extend(
                str(part.get("text", ""))
                for part in content
                if isinstance(part, Mapping) and part.get("type") == "text"
            )
    return "\n".join(block for block in blocks if block).strip()


def _responses_input(
    messages: list[Mapping[str, Any]],
) -> list[dict[str, Any]]:
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
            if part.get("type") == "text":
                parts.append(
                    {"type": "input_text", "text": str(part.get("text", ""))}
                )
            elif part.get("type") == "image_url":
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

