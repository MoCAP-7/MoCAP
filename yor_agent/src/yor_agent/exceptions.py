"""Control-flow and error types shared by the agent, executor, and primitives."""

from __future__ import annotations

from typing import Any


class YorAgentError(Exception):
    """Base class for every error raised inside ``yor_agent``."""


class Finished(YorAgentError):
    """Raised by the executor's ``finish()`` helper to stop the outer loop.

    This is a control-flow transition only. It does **not** claim that the
    physical task goal was reached; v1 has no task verifier.
    """

    def __init__(self, reason: str = "") -> None:
        reason = str(reason).strip() or "unspecified"
        super().__init__(reason)
        self.reason = reason
        # Set by :meth:`PolicyExecutor.execute` so the agent can still record the
        # stdout/primitive calls of the policy that called ``finish()``.
        self.execution: dict[str, Any] | None = None


class Stopped(YorAgentError):
    """Raised internally when the supervising operator requests a normal stop.

    Like :class:`Finished`, this is a control-flow transition rather than a
    task outcome.  The environment still performs its full safe shutdown.
    """

    def __init__(self, reason: str = "operator requested stop") -> None:
        self.reason = str(reason).strip() or "operator requested stop"
        super().__init__(self.reason)


class PrimitiveFailed(YorAgentError):
    """Raised by an effectful primitive wrapper when the controller failed.

    This is primitive-level execution safety: it interrupts the remaining
    generated policy so later motion in the same code block is not executed.
    It is not a statement about task success or failure.
    """

    def __init__(self, primitive: str, result: dict[str, Any]) -> None:
        self.primitive = str(primitive)
        self.result = dict(result)
        self.reason = str(self.result.get("reason", "unknown"))
        super().__init__(f"{self.primitive} failed: {self.reason}")


class FormatError(YorAgentError):
    """Raised when a model response cannot be normalized into one code string."""


class ModelError(YorAgentError):
    """Raised when the model provider call itself fails."""
