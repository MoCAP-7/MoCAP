"""Turn YOR's shared navigation tasks into CoW object goals."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path

from ..experiment.protocol import DEFAULT_TASK_SUITE, MANIPULATION_CLAUSE, suite_instruction

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_TASK_CONFIG = DEFAULT_TASK_SUITE

# CoW's PASTURE splits pass the whole annotated description (object plus
# appearance or spatial context) as the class text, so only the imperative
# prefix is removed.
_INSTRUCTION_PREFIX = re.compile(
    r"^(?:please\s+)?(?:navigate\s+to|go\s+to|find|locate|look\s+for)\s+(?:the|an|a)\s+",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class CowTask:
    task_id: str | None
    instruction: str
    goal: str
    config_path: str | None

    def as_dict(self) -> dict[str, str | None]:
        return asdict(self)


def instruction_to_goal(instruction: str) -> str:
    text = " ".join(str(instruction).split()).rstrip(".")
    goal = _INSTRUCTION_PREFIX.sub("", text, count=1)
    if not text or goal == text or not goal:
        raise ValueError(
            "instruction must name one object after 'Navigate to', 'Go to', 'Find', "
            f"'Locate' or 'Look for' and an article: {instruction!r}"
        )
    if MANIPULATION_CLAUSE.search(goal):
        raise ValueError(f"CoW only navigates to an object; the instruction asks for more: {instruction!r}")
    return goal


def direct_task(instruction: str) -> CowTask:
    normalized = " ".join(str(instruction).split())
    return CowTask(
        task_id=None,
        instruction=normalized,
        goal=instruction_to_goal(normalized),
        config_path=None,
    )


def load_task(task_id: str, path: str | Path = DEFAULT_TASK_CONFIG) -> CowTask:
    normalized_id = str(task_id).strip()
    instruction = suite_instruction(normalized_id, path)
    return CowTask(
        task_id=normalized_id,
        instruction=instruction,
        goal=instruction_to_goal(instruction),
        config_path=str(Path(path).expanduser().resolve()),
    )
