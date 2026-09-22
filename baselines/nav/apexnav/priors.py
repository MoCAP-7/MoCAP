"""Target priors used by ApexNav's detector fusion and semantic value map."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


@dataclass(frozen=True)
class TargetPrior:
    target: str
    similar_labels: tuple[str, ...]
    confidence_threshold: float
    room: str
    source: str

    def validate(self) -> "TargetPrior":
        if not self.target.strip():
            raise ValueError("target must not be empty")
        if len(self.similar_labels) > 4:
            raise ValueError("ApexNav supports at most four similar-object labels")
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence threshold must be in [0, 1]")
        if not self.room.strip():
            raise ValueError("room prior must not be empty")
        return self

    def as_dict(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "similar_labels": list(self.similar_labels),
            "confidence_threshold": self.confidence_threshold,
            "room": self.room,
            "source": self.source,
        }


def _key(value: str) -> str:
    return " ".join(str(value).strip().lower().split())


def load_target_prior(
    path: str | Path,
    target: str,
    *,
    similar_labels: Sequence[str] | None = None,
    confidence_threshold: float | None = None,
    room: str | None = None,
) -> TargetPrior:
    normalized = _key(target)
    if not normalized:
        raise ValueError("target must not be empty")
    loaded = yaml.safe_load(Path(path).expanduser().read_text(encoding="utf-8"))
    if not isinstance(loaded, Mapping):
        raise ValueError("target prior file must contain a mapping")
    entry = loaded.get(normalized)
    if entry is not None and not isinstance(entry, Mapping):
        raise ValueError(f"prior for {normalized!r} must be a mapping")
    entry = dict(entry or {})
    explicit = any(value is not None for value in (similar_labels, confidence_threshold, room))
    return TargetPrior(
        target=normalized,
        similar_labels=tuple(
            _key(item)
            for item in (
                similar_labels if similar_labels is not None else entry.get("similar_labels", [])
            )
            if _key(item)
        ),
        confidence_threshold=float(
            confidence_threshold
            if confidence_threshold is not None
            else entry.get("confidence_threshold", 0.50)
        ),
        room=_key(room if room is not None else entry.get("room", "everywhere")),
        source="explicit" if explicit else ("cache" if entry else "llm_disabled_default"),
    ).validate()
