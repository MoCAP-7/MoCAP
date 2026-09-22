"""Task-agnostic Gemini and OpenAI navigation-memory builders."""

from __future__ import annotations

import base64
import hashlib
import mimetypes
import os
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .json_utils import atomic_write_json
from .prompts import VIDEO_MEMORY_PROMPT
from .reference_frames import (
    extract_reference_frames,
    extract_video_sample_frames,
)

MEMORY_SCHEMA = "yor-video-memory-v1"
LEGACY_MEMORY_SCHEMA = "yor-gemini-video-memory-v1"
SUPPORTED_MEMORY_SCHEMAS = {MEMORY_SCHEMA, LEGACY_MEMORY_SCHEMA}


@dataclass(frozen=True)
class VideoMemoryConfig:
    model: str = "gemini-3.7-flash"
    thinking_level: str = "high"
    upload_poll_interval_s: float = 5.0
    upload_timeout_s: float = 1800.0
    request_timeout_s: float = 900.0
    max_reference_frames: int = 32
    reference_frame_width: int = 768
    video_frame_fps: float = 2.0
    max_video_input_frames: int = 300
    video_frame_width: int = 2048
    image_detail: str = "high"


class VideoMemoryBuilder:
    """Build memory from native Gemini video or GPT timestamped frames."""

    def __init__(
        self,
        config: VideoMemoryConfig | None = None,
        *,
        client: Any | None = None,
        on_progress: Any | None = None,
    ) -> None:
        self.config = config or VideoMemoryConfig()
        self.provider = _provider_for_model(self.config.model)
        allowed_levels = (
            {"none", "low", "medium", "high", "xhigh", "max"}
            if self.provider == "openai"
            else {"low", "medium", "high"}
        )
        if self.config.thinking_level not in allowed_levels:
            choices = ", ".join(sorted(allowed_levels))
            raise ValueError(f"thinking_level for {self.provider} must be one of {choices}")
        if self.config.image_detail not in {"low", "high", "original", "auto"}:
            raise ValueError("image_detail must be low, high, original, or auto")
        self._client = client
        self._on_progress = on_progress

    def build(
        self,
        video_path: str | Path,
        output_dir: str | Path,
    ) -> dict[str, Any]:
        video = Path(video_path).expanduser().resolve()
        destination_dir = Path(output_dir).expanduser().resolve()
        if not video.is_file():
            raise FileNotFoundError(video)
        source = self._source_metadata(video)
        created_at = datetime.now(timezone.utc)
        version = created_at.strftime("%Y%m%dT%H%M%S.%fZ")
        output = destination_dir / f"memory_{version}.json"

        client = self._get_client()
        if self.provider == "openai":
            memory_text, video_input = self._understand_with_openai(
                client, video
            )
        else:
            memory_text, video_input = self._understand_with_gemini(
                client, video, source
            )
        if not memory_text:
            raise RuntimeError(f"{self.provider} returned an empty video memory")
        memory = {
            "schema_version": MEMORY_SCHEMA,
            "created_at": created_at.isoformat(),
            "artifact_path": str(output),
            "model": self.config.model,
            "provider": self.provider,
            "thinking_level": self.config.thinking_level,
            "source_video": source,
            "video_input": video_input,
            "generation_prompt": VIDEO_MEMORY_PROMPT,
            "generation_prompt_sha256": hashlib.sha256(
                VIDEO_MEMORY_PROMPT.encode("utf-8")
            ).hexdigest(),
            "memory_text": memory_text,
        }
        frame_dir = destination_dir / f"{output.stem}_frames"
        try:
            frames = extract_reference_frames(
                video,
                memory_text,
                frame_dir,
                relative_to=destination_dir,
                limit=self.config.max_reference_frames,
                max_width=self.config.reference_frame_width,
            )
            memory["reference_frames"] = frames
            memory["reference_frame_extraction"] = {
                "status": "complete" if frames else "no_timestamps",
                "count": len(frames),
                "max_width": self.config.reference_frame_width,
            }
            self._progress("reference_frames_saved", count=len(frames))
        except Exception as exc:
            # The expensive VLM result remains usable even if local ffmpeg
            # is missing or one cited timestamp cannot be decoded.
            memory["reference_frames"] = []
            memory["reference_frame_extraction"] = {
                "status": "failed",
                "count": 0,
                "error": str(exc),
            }
            self._progress("reference_frame_extraction_failed", error=str(exc))
        atomic_write_json(output, memory)
        self._progress("memory_saved", path=str(output))
        return memory

    def _understand_with_gemini(
        self,
        client: Any,
        video: Path,
        source: dict[str, Any],
    ) -> tuple[str, dict[str, Any]]:
        self._progress(
            "video_upload_started",
            path=str(video),
            size_bytes=source["size_bytes"],
        )
        uploaded = client.files.upload(
            file=str(video),
            config={
                "http_options": {
                    "timeout": int(self.config.upload_timeout_s * 1000)
                }
            },
        )
        uploaded = self._wait_until_active(client, uploaded)
        mime_type = getattr(uploaded, "mime_type", None) or source["mime_type"]
        self._progress("video_understanding_started", model=self.config.model)
        interaction = client.interactions.create(
            model=self.config.model,
            input=[
                {
                    "type": "video",
                    "uri": getattr(uploaded, "uri"),
                    "mime_type": mime_type,
                },
                {"type": "text", "text": VIDEO_MEMORY_PROMPT},
            ],
            generation_config={"thinking_level": self.config.thinking_level},
            timeout=self.config.request_timeout_s,
        )
        memory_text = str(getattr(interaction, "output_text", "") or "").strip()
        return memory_text, {
            "mode": "native_video",
            "mime_type": mime_type,
        }

    def _understand_with_openai(
        self,
        client: Any,
        video: Path,
    ) -> tuple[str, dict[str, Any]]:
        self._progress(
            "video_frame_extraction_started",
            path=str(video),
            fps=self.config.video_frame_fps,
            max_frames=self.config.max_video_input_frames,
        )
        with tempfile.TemporaryDirectory(prefix="yor-nav-video-frames-") as directory:
            frames = extract_video_sample_frames(
                video,
                Path(directory) / "frames",
                fps=self.config.video_frame_fps,
                limit=self.config.max_video_input_frames,
                max_width=self.config.video_frame_width,
            )
            self._progress(
                "video_frames_extracted",
                count=len(frames),
                fps=self.config.video_frame_fps,
            )
            content: list[dict[str, Any]] = [
                {"type": "input_text", "text": VIDEO_MEMORY_PROMPT},
                {
                    "type": "input_text",
                    "text": (
                        "The passive video is provided below as chronological "
                        f"JPEG frames sampled at {self.config.video_frame_fps:g} "
                        "frames per second. Each image is preceded by its video "
                        "timestamp. Treat adjacent images as one temporal sequence."
                    ),
                },
            ]
            for frame in frames:
                encoded = base64.b64encode(
                    Path(frame["path"]).read_bytes()
                ).decode("ascii")
                content.extend(
                    [
                        {
                            "type": "input_text",
                            "text": f"VIDEO FRAME at {frame['timestamp']}:",
                        },
                        {
                            "type": "input_image",
                            "image_url": f"data:image/jpeg;base64,{encoded}",
                            "detail": self.config.image_detail,
                        },
                    ]
                )
            self._progress(
                "video_understanding_started", model=self.config.model
            )
            response = client.responses.create(
                model=self.config.model,
                input=[{"role": "user", "content": content}],
                reasoning={"effort": self.config.thinking_level},
                timeout=self.config.request_timeout_s,
            )
            memory_text = str(getattr(response, "output_text", "") or "").strip()
        return memory_text, {
            "mode": "sampled_frames",
            "frame_count": len(frames),
            "fps": self.config.video_frame_fps,
            "max_frames": self.config.max_video_input_frames,
            "max_width": self.config.video_frame_width,
            "image_detail": self.config.image_detail,
        }

    def _wait_until_active(self, client: Any, uploaded: Any) -> Any:
        deadline = time.monotonic() + self.config.upload_timeout_s
        while True:
            state = _state_name(getattr(uploaded, "state", None))
            if state == "ACTIVE":
                self._progress(
                    "video_upload_active", name=getattr(uploaded, "name", None)
                )
                return uploaded
            if state == "FAILED":
                raise RuntimeError(
                    "Gemini Files API failed to process the uploaded video"
                )
            if time.monotonic() >= deadline:
                raise TimeoutError("timed out waiting for Gemini to process the video")
            time.sleep(self.config.upload_poll_interval_s)
            uploaded = client.files.get(name=getattr(uploaded, "name"))

    def _get_client(self) -> Any:
        if self._client is None:
            self._client = make_client(
                self.provider, timeout_s=self.config.request_timeout_s
            )
        return self._client

    @staticmethod
    def _source_metadata(video: Path) -> dict[str, Any]:
        stat = video.stat()
        return {
            "path": str(video),
            "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "mime_type": mimetypes.guess_type(video.name)[0] or "video/mp4",
        }

    def _progress(self, event: str, **payload: Any) -> None:
        if self._on_progress is not None:
            self._on_progress({"event": event, **payload})


def make_client(provider: str, *, timeout_s: float) -> Any:
    """Build the Gemini or OpenAI SDK client from the environment's API key.

    The SDKs are imported here, lazily, so offline paths that never call a
    model (manual event labelling, tests with fake clients) stay stdlib-only.
    """

    if provider == "openai":
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError(
                "OpenAI video understanding requires OPENAI_API_KEY"
            )
        from openai import OpenAI

        return OpenAI(api_key=api_key, timeout=timeout_s)
    if provider == "gemini":
        api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("Gemini requires GEMINI_API_KEY in the environment")
        from google import genai

        return genai.Client(api_key=api_key)
    raise ValueError(f"unknown video-understanding provider {provider!r}")


def _state_name(state: Any) -> str:
    if state is None:
        return "PROCESSING"
    name = getattr(state, "name", None)
    if name:
        return str(name).upper()
    return str(state).rsplit(".", 1)[-1].upper()


def _provider_for_model(model: str) -> str:
    normalized = str(model).strip().lower()
    if normalized.startswith("gpt-"):
        return "openai"
    if normalized.startswith("gemini-"):
        return "gemini"
    raise ValueError(
        f"cannot infer video-understanding provider from model {model!r}; "
        "use a gemini-* or gpt-* model"
    )


# Backward-compatible imports for callers created before the OpenAI backend.
GeminiConfig = VideoMemoryConfig
GeminiVideoMemoryBuilder = VideoMemoryBuilder
