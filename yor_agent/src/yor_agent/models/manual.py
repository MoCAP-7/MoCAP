"""Operator-driven policy model for supervised primitive tests.

``ManualModel`` keeps the outer agent loop, prompt construction, trace, and
feedback identical to an LLM run.  The only difference is where each turn's
program comes from: instead of a provider call, :meth:`query` waits for a
program that the operator submitted through the Web UI (or any other caller
of :meth:`submit`).  Everything downstream -- ``extract_code``, the executor's
capability-limited namespace, primitive-failure handling, ``finish()``, and
the safe shutdown -- is unchanged, so an operator program is executed exactly
like a generated one and leaves the same trace.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
import queue
import threading
from typing import Any

from ..exceptions import Stopped
from .llm import LLM

DEFAULT_STOP_REASON = "operator requested stop"


class ManualModel(LLM):
    """Policy model whose programs are typed by the supervising operator."""

    def __init__(
        self,
        config: Mapping[str, Any] | None = None,
        *,
        stop_event: threading.Event | None = None,
        stop_reason: Callable[[], str] | None = None,
        poll_interval_s: float = 0.1,
    ) -> None:
        config = dict(config or {})
        config["provider"] = "manual"
        config.setdefault("name", "operator")
        super().__init__(config)
        self._programs: queue.Queue[str] = queue.Queue()
        self._stop_event = stop_event if stop_event is not None else threading.Event()
        # Optional accessor for the reason the loop owner recorded when it set
        # the stop event, so a stop while waiting reports the same reason as a
        # stop during an LLM turn.
        self.stop_reason = stop_reason
        self._poll_interval_s = float(poll_interval_s)
        if self._poll_interval_s <= 0:
            raise ValueError("poll_interval_s must be positive")
        # True while ``query`` is blocked waiting for the operator.
        self.waiting = False

    def submit(self, code: str) -> None:
        """Queue one operator program for the next model turn."""

        text = str(code)
        if not text.strip():
            raise ValueError("program must not be empty")
        self._programs.put(text)

    def pending(self) -> int:
        """Number of submitted programs not yet consumed by the loop."""

        return self._programs.qsize()

    def _generate(self, messages: list[Mapping[str, Any]]) -> str:
        """Block until the operator submits a program or requests a stop."""

        del messages  # The operator reads the same feedback in the Web UI.
        self.waiting = True
        try:
            while True:
                if self._stop_event.is_set():
                    reason = DEFAULT_STOP_REASON
                    if callable(self.stop_reason):
                        reason = str(self.stop_reason() or "").strip() or reason
                    raise Stopped(reason)
                try:
                    program = self._programs.get(timeout=self._poll_interval_s)
                except queue.Empty:
                    continue
                self.n_calls += 1
                return program
        finally:
            self.waiting = False
