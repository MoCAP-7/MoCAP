"""The outer agent loop: query, execute, observe, feed back, repeat.

The loop is deliberately domain-general. It contains no navigation-specific
branching: navigation enters only through the environment and the registered
primitives, so a manipulation provider can be added without touching this file.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
import math
import threading
from typing import Any

from ..environment import summarize_observation
from ..exceptions import Finished, FormatError, Stopped
from ..trace import Trace

#: Start of the stop reason a run ended by its time limit records.
TIME_LIMIT_STOP_REASON_PREFIX = "time limit"


class DefaultAgent:
    """Own the loop, the messages, the finish transition, and trace checkpoints."""

    def __init__(
        self,
        model: Any,
        environment: Any,
        executor: Any,
        trace: Trace,
        *,
        max_turns: int = 30,
        time_limit_s: float | None = None,
        navigation_planner: Any | None = None,
        on_event: Any | None = None,
        stop_event: threading.Event | None = None,
    ) -> None:
        self.model = model
        self.environment = environment
        self.executor = executor
        self.trace = trace
        self.max_turns = int(max_turns)
        if time_limit_s is not None:
            if isinstance(time_limit_s, bool) or not isinstance(time_limit_s, (int, float)):
                raise ValueError("time_limit_s must be a number of seconds or None")
            time_limit_s = float(time_limit_s)
            if not math.isfinite(time_limit_s) or time_limit_s <= 0.0:
                raise ValueError("time_limit_s must be positive and finite")
        self.time_limit_s = time_limit_s
        self.navigation_planner = navigation_planner
        self.messages: list[dict[str, Any]] = []
        self._on_event = on_event
        self._stop_event = stop_event or threading.Event()
        self._time_limit_lock = threading.Lock()
        self._run_active = False

    def request_stop(self, reason: str = "operator requested stop") -> None:
        """Ask the loop and any active navigation primitive to stop normally."""

        self._stop_reason = str(reason).strip() or "operator requested stop"
        self._stop_event.set()
        request_stop = getattr(self.environment, "request_stop", None)
        if callable(request_stop):
            request_stop()

    def run(self, task: str, config: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Run until the policy calls ``finish()``, the turn budget ends, or an error.

        The returned dictionary reports *why the loop stopped*. It never claims
        that the physical goal was reached; v1 has no task verifier.
        """

        self._stop_reason = "operator requested stop"
        self.trace.start(task, config)
        self._emit(
            "run_started",
            task=task,
            provider=getattr(self.model, "provider", None),
            model=getattr(self.model, "name", None),
            max_turns=self.max_turns,
            time_limit_s=self.time_limit_s,
            trace_path=str(self.trace.path),
        )
        timer = self._start_time_limit()
        try:
            self._raise_if_stopped()
            self._emit("environment_status", status="initializing")
            observation = self.environment.reset()
            summary = summarize_observation(observation)
            self.trace.record_observation(summary)
            policy_task = task
            if self.navigation_planner is not None:
                self._emit("navigation_planner_started", task=task)
                advice = self.navigation_planner.plan(
                    task=task,
                    observation=observation,
                    image_max_width=getattr(self.model, "image_max_width", 1024),
                )
                from ..models.navigation_planner import (
                    policy_task_with_navigation_advice,
                )

                policy_task = policy_task_with_navigation_advice(task, advice)
                self.trace.record_navigation_plan(
                    {
                        "model": getattr(self.navigation_planner, "model", None),
                        "provider": getattr(
                            self.navigation_planner, "provider", None
                        ),
                        "memory_path": str(
                            getattr(self.navigation_planner, "memory_path", "")
                        ),
                        "advice": advice,
                        "policy_task": policy_task,
                    }
                )
                self._emit(
                    "navigation_planner_finished", task=task, advice=advice
                )
            self.messages = self.model.initial_messages(
                task=policy_task,
                observation=observation,
                primitive_docs=self.executor.documentation(),
            )
            self.trace.record_messages(self.messages)
            self._emit("environment_status", status="ready")
            self._emit_model_input(self.messages[-1], summary, turn=1)

            while True:
                self._raise_if_stopped()
                turn = self.trace.begin_turn()
                if turn > self.max_turns:
                    reason = f"turn budget exhausted after {self.max_turns} turns"
                    self.trace.finish(reason)
                    self._emit(
                        "run_finished",
                        status="max_turns",
                        reason=reason,
                        turns=turn - 1,
                    )
                    return {"status": "max_turns", "reason": reason, "turns": turn - 1}
                try:
                    self._emit("turn_started", turn=turn)
                    self.step()
                except Finished as finished:
                    self.trace.finish(finished.reason)
                    result = {
                        "status": "finished",
                        "reason": finished.reason,
                        "turns": turn,
                    }
                    self._emit("run_finished", **result)
                    return result
                except Stopped as stopped:
                    self.trace.stop(stopped.reason)
                    result = {
                        "status": "stopped",
                        "reason": stopped.reason,
                        "turns": turn,
                    }
                    self._emit("run_finished", **result)
                    return result
        except Stopped as stopped:
            self.trace.stop(stopped.reason)
            result = {
                "status": "stopped",
                "reason": stopped.reason,
                "turns": self.trace.turn,
            }
            self._emit("run_finished", **result)
            return result
        except BaseException as exc:
            self.trace.record_error(exc)
            self._emit(
                "run_error", error_type=type(exc).__name__, message=str(exc)
            )
            raise
        finally:
            # A limit that fires from here on would stop an environment that
            # is already shutting down; it waits for a firing one to finish.
            with self._time_limit_lock:
                self._run_active = False
            if timer is not None:
                timer.cancel()
            self._record_usage()
            self.environment.safe_shutdown()
            self.trace.save()

    def step(self) -> None:
        """One model turn: generate a policy, execute it, observe, feed back."""

        turn = self.trace.turn
        self._raise_if_stopped()
        self._emit("model_query_started", turn=turn)
        try:
            code = self.model.query(self.messages)
        except FormatError as exc:
            self._emit(
                "model_format_error",
                turn=turn,
                message=str(exc),
                response=getattr(self.model, "last_response", None),
            )
            feedback = self.model.format_error_feedback(exc)
            self.messages.extend(feedback)
            self.trace.record_messages(feedback)
            return

        self._emit(
            "model_response",
            turn=turn,
            code=code,
            response=getattr(self.model, "last_response", None) or code,
        )
        self._raise_if_stopped()
        self._emit("policy_execution_started", turn=turn, code=code)

        try:
            execution = self.executor.execute(code)
        except BaseException as exc:
            # finish(), Ctrl-C, and SystemExit carry the partial record out with
            # them; the policy still belongs in the trace before it propagates.
            execution = getattr(exc, "execution", None)
            if execution is not None:
                self.trace.record_policy(code, execution)
                self._emit(
                    "policy_execution_finished",
                    turn=turn,
                    code=code,
                    execution=_event_execution(execution),
                )
            raise
        self.trace.record_policy(code, execution)
        self._emit(
            "policy_execution_finished",
            turn=turn,
            code=code,
            execution=_event_execution(execution),
        )
        self._raise_if_stopped()
        # A fresh outer-loop observation is captured after every policy, including
        # one interrupted by an exception or a failed primitive.
        observation = self.environment.observe()
        summary = summarize_observation(observation)
        self.trace.record_observation(summary)
        feedback = self.model.format_feedback(code, execution, observation)
        self.messages.extend(feedback)
        self.trace.record_messages(feedback)
        self._emit_model_input(feedback[-1], summary, turn=turn + 1)

    def _start_time_limit(self) -> threading.Timer | None:
        """Arm the wall-clock limit, which stops the run like the operator's Stop.

        The stop takes effect where an operator stop does: an active Nav2 goal
        or base motion is cancelled, and the loop ends before its next model
        query or program.
        """

        with self._time_limit_lock:
            self._run_active = True
        if self.time_limit_s is None:
            return None
        timer = threading.Timer(self.time_limit_s, self._on_time_limit)
        timer.daemon = True
        timer.start()
        return timer

    def _on_time_limit(self) -> None:
        with self._time_limit_lock:
            if not self._run_active:
                return
            reason = f"{TIME_LIMIT_STOP_REASON_PREFIX} of {self.time_limit_s:g} s reached"
            self._emit("time_limit_reached", time_limit_s=self.time_limit_s)
            self.request_stop(reason)

    def _record_usage(self) -> None:
        usage = getattr(self.model, "usage", None)
        if usage is not None:
            self.trace.record_model_usage(
                {"n_calls": getattr(self.model, "n_calls", 0), **dict(usage)}
            )

    def _raise_if_stopped(self) -> None:
        if self._stop_event.is_set():
            raise Stopped(self._stop_reason)

    def _emit_model_input(
        self, message: Mapping[str, Any], summary: Mapping[str, Any], *, turn: int
    ) -> None:
        """Emit the literal current image/text parts that the LLM will receive."""

        text = None
        image_url = None
        content = message.get("content")
        if isinstance(content, list):
            for part in content:
                if not isinstance(part, Mapping):
                    continue
                if part.get("type") == "text":
                    text = part.get("text")
                elif part.get("type") == "image_url":
                    image_url = (part.get("image_url") or {}).get("url")
        self._emit(
            "model_input",
            turn=turn,
            prompt_text=text,
            image_url=image_url,
            observation=dict(summary),
        )

    def _emit(self, event_type: str, **payload: Any) -> None:
        if self._on_event is None:
            return
        event = {
            "type": event_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **payload,
        }
        try:
            self._on_event(event)
        except Exception:
            # An observer (the Web UI) must never affect robot execution.
            pass


def _event_execution(execution: Mapping[str, Any]) -> dict[str, Any]:
    """Copy the JSON-safe, UI-relevant executor fields."""

    return {
        key: execution.get(key)
        for key in (
            "stdout",
            "stderr",
            "error",
            "interrupted_by",
            "finish_reason",
            "elapsed_s",
        )
    }
