"""Convert an explicit navigation instruction into an ApexNav detector target."""

from __future__ import annotations

import re


_PREFIX = re.compile(
    r"^(?:please\s+)?(?:find|locate|navigate\s+to|go\s+to)\s+"
    r"(?:(?:the|a|an)(?:\s+|$))?",
    flags=re.IGNORECASE,
)


def instruction_to_target(instruction: str) -> str:
    normalized = " ".join(str(instruction).strip().split())
    match = _PREFIX.match(normalized)
    if match is None:
        raise ValueError(
            "ApexNav instructions must begin with Find, Locate, Navigate to, or Go to"
        )
    target = normalized[match.end() :].strip().rstrip(".!?").strip().lower()
    if not target:
        raise ValueError("navigation instruction does not contain a target")
    return target
