"""Version and attach local visual evidence to a legacy memory artifact."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .json_utils import atomic_write_json, read_json
from .memory import SUPPORTED_MEMORY_SCHEMAS
from .reference_frames import extract_reference_frames


def enrich_existing_memory(
    memory_path: str | Path,
    *,
    output_dir: str | Path | None = None,
    max_reference_frames: int = 32,
    reference_frame_width: int = 768,
) -> dict[str, Any]:
    """Create a timestamp-named, frame-backed copy without another VLM call."""

    source_artifact = Path(memory_path).expanduser().resolve()
    memory = read_json(source_artifact)
    if memory.get("schema_version") not in SUPPORTED_MEMORY_SCHEMAS:
        raise ValueError(f"unsupported memory artifact: {source_artifact}")
    created_at = datetime.fromisoformat(str(memory["created_at"]))
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=timezone.utc)
    created_at = created_at.astimezone(timezone.utc)
    version = created_at.strftime("%Y%m%dT%H%M%S.%fZ")
    destination_dir = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else source_artifact.parent
    )
    output = destination_dir / f"memory_{version}.json"
    if output.exists():
        raise FileExistsError(output)

    video = Path(str(memory["source_video"]["path"])).expanduser().resolve()
    memory_text = str(memory.get("memory_text", "")).strip()
    frame_dir = destination_dir / f"{output.stem}_frames"
    frames = extract_reference_frames(
        video,
        memory_text,
        frame_dir,
        relative_to=destination_dir,
        limit=max_reference_frames,
        max_width=reference_frame_width,
    )
    enriched = {
        **memory,
        "artifact_path": str(output),
        "legacy_source_artifact": str(source_artifact),
        "reference_frames": frames,
        "reference_frame_extraction": {
            "status": "complete" if frames else "no_timestamps",
            "count": len(frames),
            "max_width": reference_frame_width,
        },
    }
    atomic_write_json(output, enriched)
    return enriched
