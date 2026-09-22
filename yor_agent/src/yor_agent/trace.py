"""Persistent run trace: messages, policies, primitive calls, observations.

The trace keeps mini-SWE-agent's ordered message trajectory and adds the
generated code, execution feedback, observation summaries, and primitive events.
Large RGB/depth artifacts are written next to the JSON and referenced by path;
no base64 image ever lands in ``trace.json``.
"""

from __future__ import annotations

import base64
import json
import os
import time
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any


class Trace:
    """Append-only run record, checkpointed to ``<output_dir>/trace.json``."""

    def __init__(
        self,
        config: Mapping[str, Any] | None = None,
        *,
        output_dir: str | os.PathLike[str] | None = None,
        run_id: str | None = None,
    ) -> None:
        config = dict(config or {})
        directory = output_dir or config.get("output_dir") or "./outputs/run"
        self.output_dir = Path(directory).expanduser()
        self.artifacts_dir = self.output_dir / "artifacts"
        self.path = self.output_dir / "trace.json"
        self.run_id = run_id or f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
        self.data: dict[str, Any] = {
            # Keep planner output at the top: it is the quickest explanation
            # of how the task was grounded before the policy trajectory.
            "navigation_plan": None,
            "run_id": self.run_id,
            "task": None,
            "created_at": _now(),
            # When the agent loop started. The runtime (and this trace) is
            # built first, which can take much longer than usual when a robot
            # service is still coming up.
            "started_at": None,
            "updated_at": _now(),
            "config": {},
            "messages": [],
            "policies": [],
            "primitive_calls": [],
            "observations": [],
            "model_usage": {},
            "finish": None,
            "stop": None,
            "error": None,
        }
        self._turn = 0
        self._artifacts = 0
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        # Written immediately so a setup failure (before the agent loop
        # starts) still leaves a trace.json instead of an empty folder.
        self.save()

    # ------------------------------------------------------------------

    def start(self, task: str, config: Mapping[str, Any] | None = None) -> None:
        self.data["started_at"] = _now()
        self.data["task"] = task
        self.data["config"] = redact_config(config or {})
        self.save()

    def record_message(self, message: Mapping[str, Any]) -> None:
        """Append one conversation message, moving inline images to artifacts."""

        self.data["messages"].append(self._redact_message(message))
        self.save()

    def record_messages(self, messages: list[Mapping[str, Any]]) -> None:
        for message in messages:
            self.data["messages"].append(self._redact_message(message))
        self.save()

    def begin_turn(self) -> int:
        self._turn += 1
        return self._turn

    @property
    def turn(self) -> int:
        """Current one-based model turn (zero before the first turn)."""

        return self._turn

    def record_policy(self, code: str, execution: Mapping[str, Any]) -> None:
        """Record one generated policy and its execution record."""

        record = dict(execution)
        # Primitive calls are already streamed into their own list.
        record.pop("primitive_calls", None)
        self.data["policies"].append(
            {
                "turn": self._turn,
                "at": _now(),
                "code": code,
                "execution": record,
            }
        )
        self.save()

    def record_primitive_call(self, event: Mapping[str, Any]) -> None:
        """Checkpoint after every primitive invocation."""

        entry = dict(event)
        entry["turn"] = self._turn
        self.data["primitive_calls"].append(entry)
        self.save()

    def record_observation(self, summary: Mapping[str, Any]) -> dict[str, Any]:
        entry = {"turn": self._turn, "at": _now(), **dict(summary)}
        self.data["observations"].append(entry)
        self.save()
        return entry

    def record_model_usage(self, usage: Mapping[str, Any]) -> None:
        self.data["model_usage"] = dict(usage)

    def record_navigation_plan(self, plan: Mapping[str, Any]) -> None:
        self.data["navigation_plan"] = {"at": _now(), **dict(plan)}
        self.save()

    def record_error(self, exc: BaseException) -> None:
        self.data["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "at": _now(),
        }
        self.save()

    def finish(self, reason: str) -> None:
        """Record the declared finish reason. This is not a success claim."""

        self.data["finish"] = {"reason": reason, "at": _now()}
        self.save()

    def stop(self, reason: str) -> None:
        """Record a supervising operator's stop request, not a task result."""

        self.data["stop"] = {"reason": reason, "at": _now()}
        self.save()

    def save(self) -> None:
        """Atomically rewrite ``trace.json``."""

        self.data["updated_at"] = _now()
        temporary = self.path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(self.data, indent=2, default=_json_default), encoding="utf-8"
        )
        temporary.replace(self.path)

    # ------------------------------------------------------------------

    def _redact_message(self, message: Mapping[str, Any]) -> dict[str, Any]:
        content = message.get("content")
        if not isinstance(content, list):
            return {"role": message.get("role"), "content": content}
        parts: list[Any] = []
        for item in content:
            if isinstance(item, Mapping) and item.get("type") == "image_url":
                url = item.get("image_url", {}).get("url", "")
                parts.append(self._store_image(url))
            else:
                parts.append(dict(item) if isinstance(item, Mapping) else item)
        return {"role": message.get("role"), "content": parts}

    def _store_image(self, url: str) -> dict[str, Any]:
        """Write an inline data URL to ``artifacts/`` and return a reference."""

        if not url.startswith("data:"):
            return {"type": "image_ref", "path": url}
        header, _, payload = url.partition(",")
        mime = header.removeprefix("data:").split(";")[0] or "image/png"
        suffix = {"image/png": ".png", "image/jpeg": ".jpg"}.get(mime, ".bin")
        self._artifacts += 1
        name = f"turn{self._turn:03d}_{self._artifacts:04d}{suffix}"
        artifact = self.artifacts_dir / name
        try:
            artifact.write_bytes(base64.b64decode(payload))
        except Exception as exc:  # noqa: BLE001 - a bad image must not kill a run
            return {"type": "image_ref", "path": None, "error": str(exc)}
        return {
            "type": "image_ref",
            "path": str(artifact.relative_to(self.output_dir)),
            "mime": mime,
            "bytes": artifact.stat().st_size,
        }


SECRET_KEYS = {
    "api_key",
    "credentials",
    "google_application_credentials",
    "key",
    "password",
    "secret",
    "token",
}


def redact_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Drop anything that looks like a credential before persisting config."""

    redacted: dict[str, Any] = {}
    for key, value in config.items():
        if str(key).lower() in SECRET_KEYS:
            redacted[key] = "<redacted>"
        elif isinstance(value, Mapping):
            redacted[key] = redact_config(value)
        else:
            redacted[key] = value
    return redacted


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _json_default(value: Any) -> Any:
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return tolist()
    return repr(value)
