"""FastAPI/WebSocket server for supervised ``yor_agent`` runs.

The browser is an event subscriber.  It never implements an agent loop: both
CLI and Web UI construct the same :class:`DefaultAgent` through
``launch.build_runtime``.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import contextvars
import copy
from datetime import datetime, timezone
import functools
import inspect
import io
import json
import logging
from pathlib import Path
import threading
import time
from typing import Any, Callable, Mapping

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

from ..agents.default import TIME_LIMIT_STOP_REASON_PREFIX
from ..experiment_episodes import experiment_protocol
from ..launch import Runtime, build_runtime
from ..models.llm import DEFAULT_MODEL_NAMES
from ..models.manual import ManualModel
from ..trace import redact_config

logger = logging.getLogger(__name__)

READINESS_PRIOR_EVENTS_REQUIRED = (
    "readiness prior needs dock_to_visible_object.readiness_prior.events_path "
    "(config or --readiness-events)"
)
SERVER_SHUTDOWN_STOP_REASON = "Web UI server shut down during the run"
#: How long the server's shutdown waits for a stopped run to finish its safe
#: shutdown and trace save before handing the rest to the event loop teardown.
SERVER_SHUTDOWN_WAIT_S = 30.0
REVIEW_PENDING_DETAIL = (
    "review the previous experiment episode before starting the next run"
)


def _docking_settings(config: Mapping[str, Any]) -> dict[str, Any] | None:
    """Return the live ``dock_to_visible_object.settings`` mapping, or ``None``.

    The block is read from ``config["primitive_config"]`` as ``load_config``
    normalizes it; a raw (path-valued) ``primitive_config`` or a primitive
    without settings yields ``None``.
    """

    node: Any = config
    for key in ("primitive_config", "primitives", "dock_to_visible_object", "settings"):
        if not isinstance(node, Mapping):
            return None
        node = node.get(key)
    return node if isinstance(node, dict) else None


def readiness_prior_block(config: Mapping[str, Any]) -> dict[str, Any] | None:
    """Return ``dock_to_visible_object.settings.readiness_prior`` or ``None``."""

    settings = _docking_settings(config)
    block = None if settings is None else settings.get("readiness_prior")
    return block if isinstance(block, Mapping) else None


def episode_termination(result: Mapping[str, Any]) -> str:
    """How an experiment episode ended, from the agent loop's result."""

    status = str(result.get("status") or "error")
    reason = str(result.get("reason") or "")
    if status != "stopped":
        return status
    if reason.startswith(TIME_LIMIT_STOP_REASON_PREFIX):
        return "time_limit"
    if reason == SERVER_SHUTDOWN_STOP_REASON:
        return "server_shutdown"
    return "operator_stop"


def apply_readiness_prior_toggle(
    config: Mapping[str, Any], enabled: bool
) -> dict[str, Any]:
    """Return a deep copy of ``config`` with the docking readiness prior switched.

    Off: ``enabled`` becomes false and ``events_path`` is dropped so the trace
    cannot suggest the passive-video prior was an input. A config without the
    block is returned unchanged: there is nothing to switch off, and creating
    ``settings`` would change how the docking primitive is wired. On: the
    block must already name an events JSON (``ValueError`` otherwise), mirroring
    ``--readiness-prior`` without ``--readiness-events``.
    """

    result = copy.deepcopy(dict(config))
    settings = _docking_settings(result)
    block = readiness_prior_block(result)
    if enabled:
        if settings is None or block is None or not block.get("events_path"):
            raise ValueError(READINESS_PRIOR_EVENTS_REQUIRED)
        settings["readiness_prior"] = {**block, "enabled": True}
    elif settings is not None and block is not None:
        disabled = {**block, "enabled": False}
        disabled.pop("events_path", None)
        settings["readiness_prior"] = disabled
    return result


def primitive_catalog(registry: Any) -> list[dict[str, Any]]:
    """Describe every registered primitive for the manual-policy palette.

    Each entry carries the name, the rendered signature, the parameters (so
    the browser can insert a call template with the required arguments), and
    the docstring the policy model also receives. ``finish`` is appended
    because the executor provides it outside the registry.
    """

    items: list[dict[str, Any]] = []
    for name, function in registry.functions().items():
        try:
            signature = inspect.signature(function, eval_str=True)
        except (TypeError, ValueError, NameError):
            try:
                signature = inspect.signature(function)
            except (TypeError, ValueError):
                signature = None
        params: list[dict[str, Any]] = []
        rendered = "(...)"
        if signature is not None:
            rendered = str(signature).replace("typing.", "")
            for parameter in signature.parameters.values():
                annotation = parameter.annotation
                if annotation is inspect.Parameter.empty:
                    annotation_text = ""
                elif isinstance(annotation, type):
                    annotation_text = annotation.__name__
                else:
                    annotation_text = str(annotation).replace("typing.", "")
                params.append(
                    {
                        "name": parameter.name,
                        "kind": parameter.kind.name.lower(),
                        "required": parameter.default is inspect.Parameter.empty
                        and parameter.kind
                        not in (
                            inspect.Parameter.VAR_POSITIONAL,
                            inspect.Parameter.VAR_KEYWORD,
                        ),
                        "annotation": annotation_text,
                        "default": None
                        if parameter.default is inspect.Parameter.empty
                        else repr(parameter.default),
                    }
                )
        doc = inspect.getdoc(function) or ""
        items.append(
            {
                "name": name,
                "signature": rendered,
                "params": params,
                "summary": doc.strip().splitlines()[0] if doc.strip() else "",
                "doc": doc,
            }
        )
    items.append(
        {
            "name": "finish",
            "signature": "(reason: str) -> None",
            "params": [
                {
                    "name": "reason",
                    "kind": "positional_or_keyword",
                    "required": True,
                    "annotation": "str",
                    "default": None,
                }
            ],
            "summary": "Stop the whole run and hand ``reason`` back to the operator.",
            "doc": (
                "Stop the whole run and hand ``reason`` back to the operator.\n\n"
                "This is a control transition only. It does not verify or claim "
                "that the physical goal was reached, and it does not command the "
                "robot. The outer loop always performs a safe shutdown afterwards."
            ),
        }
    )
    return items


class WebRunController:
    """Own one physical-robot run and broadcast its read-only event stream."""

    def __init__(
        self,
        config_path: Path,
        config: Mapping[str, Any],
        *,
        decorate_runtime: Callable[[Any], None] | None = None,
    ) -> None:
        self.config_path = config_path
        self.base_config = copy.deepcopy(dict(config))
        # Applied to each run's runtime before its agent loop starts, e.g. by
        # the human-video ablations that change only the initial prompt.
        self._decorate_runtime = decorate_runtime
        # The current experiment episode, its agent-loop timing, and the
        # episode still waiting for the operator's review.
        self._episode: dict[str, Any] | None = None
        self._timing: dict[str, Any] | None = None
        self.review_pending: dict[str, Any] | None = None
        self.state = "idle"
        self.history: list[str] = []
        self.clients: set[WebSocket] = set()
        self.runtime: Runtime | None = None
        self.result: dict[str, Any] | None = None
        # Provider of the current/last run; "manual" enables the Program panel.
        self.provider: str | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._worker: asyncio.Task[None] | None = None
        self._stop_event = threading.Event()
        self._sequence = 0
        self._state_lock = asyncio.Lock()

    def public_config(self) -> dict[str, Any]:
        config = redact_config(self.base_config)
        return {
            "config_path": str(self.config_path),
            "task": config.get("task") or {},
            "model": config.get("model") or {},
            "agent": config.get("agent") or {},
            "trace": config.get("trace") or {},
            "readiness_prior": self._public_readiness_prior(),
            "experiment": config.get("experiment"),
            "review": self._public_review(),
            "state": self.state,
            "provider": self.provider,
            "result": self.result,
        }

    def _public_readiness_prior(self) -> dict[str, Any]:
        """Describe the docking readiness prior for the Start form checkbox.

        ``configured`` is true only when an events JSON is set, i.e. the
        checkbox can be switched on; the shipped block (``enabled: false``,
        ``events_path: null``) and a missing block both report it false.
        """

        block = readiness_prior_block(self.base_config) or {}
        events_path = block.get("events_path")
        return {
            "configured": bool(events_path),
            "enabled": bool(block.get("enabled", False)),
            "events_path": str(events_path) if events_path else None,
            "source": str(block.get("source") or "ready_frame"),
        }

    async def start(self, values: Mapping[str, Any]) -> dict[str, Any]:
        async with self._state_lock:
            if self._worker is not None and not self._worker.done():
                raise HTTPException(status_code=409, detail="a robot run is already active")
            if self.review_pending is not None:
                raise HTTPException(status_code=409, detail=REVIEW_PENDING_DETAIL)

            config = copy.deepcopy(self.base_config)
            instruction = str(
                values.get("instruction") or config.get("task", {}).get("instruction", "")
            ).strip()
            if not instruction:
                raise HTTPException(status_code=422, detail="task instruction is required")
            config["task"] = {**dict(config.get("task") or {}), "instruction": instruction}

            model_name = str(values.get("model") or "").strip()
            provider = str(
                values.get("provider")
                or config.get("model", {}).get("provider", "vertex")
            ).strip().lower()
            if provider not in {"vertex", "qwen", "deepseek", "openai", "manual", "scripted"}:
                raise HTTPException(
                    status_code=422,
                    detail=(
                        "provider must be 'vertex', 'qwen', 'deepseek', "
                        "'openai', 'manual', or 'scripted'"
                    ),
                )
            current_model_config = dict(config.get("model") or {})
            previous_provider = str(
                current_model_config.get("provider", "vertex")
            ).strip().lower()
            model_config = {
                **current_model_config,
                "provider": provider,
            }
            if model_name:
                model_config["name"] = model_name
            elif provider != previous_provider:
                model_config["name"] = DEFAULT_MODEL_NAMES[provider]
            config["model"] = model_config
            if values.get("temperature") is not None:
                try:
                    temperature = float(values["temperature"])
                except (TypeError, ValueError) as exc:
                    raise HTTPException(
                        status_code=422, detail="temperature must be a number"
                    ) from exc
                if not 0.0 <= temperature <= 2.0:
                    raise HTTPException(
                        status_code=422, detail="temperature must be between 0 and 2"
                    )
                config["model"] = {
                    **dict(config.get("model") or {}),
                    "temperature": temperature,
                }
            if values.get("max_turns") is not None:
                try:
                    max_turns = int(values["max_turns"])
                except (TypeError, ValueError) as exc:
                    raise HTTPException(
                        status_code=422, detail="max_turns must be an integer"
                    ) from exc
                if not 1 <= max_turns <= 100:
                    raise HTTPException(
                        status_code=422, detail="max_turns must be between 1 and 100"
                    )
                config["agent"] = {
                    **dict(config.get("agent") or {}),
                    "max_turns": max_turns,
                }

            if values.get("navigation_planner") is False:
                # A debug (manual) run skips the one-shot planner call so it
                # needs no planner API key and starts immediately.
                planner_config = dict(config.get("navigation_planner") or {})
                planner_config["enabled"] = False
                planner_config.pop("memory_path", None)
                config["navigation_planner"] = planner_config

            readiness_prior = values.get("readiness_prior")
            if readiness_prior is not None:
                # The "Readiness prior" checkbox: off disables the passive-video
                # docking prior for this run, on requires a configured events
                # JSON. Absent leaves the config as loaded.
                if not isinstance(readiness_prior, bool):
                    raise HTTPException(
                        status_code=422, detail="readiness_prior must be a boolean"
                    )
                try:
                    config = apply_readiness_prior_toggle(config, readiness_prior)
                except ValueError as exc:
                    raise HTTPException(status_code=422, detail=str(exc)) from exc

            # Keep every interactive run instead of overwriting trace.json.
            trace_config = dict(config.get("trace") or {})
            experiment = config.get("experiment")
            self._episode = None
            self._timing = None
            if isinstance(experiment, Mapping) and provider != "manual":
                # An experiment episode goes to the shared episode layout and
                # is reviewed by the operator before the next run can start.
                # A manual debug run is not an episode.
                protocol = experiment_protocol()
                task_id = str(
                    (config.get("task") or {}).get("id")
                    or protocol.task_slug(instruction)
                )
                experiment_dir = protocol.experiment_directory(
                    experiment["output_root"], experiment["name"]
                )
                try:
                    episode_dir = protocol.episode_directory(
                        None, experiment_dir, task_id, experiment["start_label"]
                    )
                except (FileExistsError, ValueError) as exc:
                    raise HTTPException(status_code=409, detail=str(exc)) from exc
                trace_config["output_dir"] = str(episode_dir)
                self._episode = {
                    "experiment": dict(experiment),
                    "experiment_dir": str(experiment_dir),
                    "output_dir": str(episode_dir),
                    "task_id": task_id,
                }
            else:
                base_output = Path(trace_config.get("output_dir") or "./outputs/run")
                stamp = datetime.now().strftime("web-%Y%m%d-%H%M%S-%f")
                trace_config["output_dir"] = str(base_output / stamp)
            config["trace"] = trace_config

            self._loop = asyncio.get_running_loop()
            self._stop_event = threading.Event()
            self.runtime = None
            self.result = None
            self.provider = provider
            self.state = "starting"
            self.history.clear()
            self._sequence = 0
            await self.publish(
                {
                    "type": "session_state",
                    "state": self.state,
                    "provider": self.provider,
                    "task": instruction,
                    "trace_path": str(Path(trace_config["output_dir"]) / "trace.json"),
                }
            )
            self._worker = asyncio.create_task(self._run(config))
            return {"status": "started", "state": self.state}

    async def stop(self, reason: str | None = None) -> dict[str, Any]:
        if self._worker is None or self._worker.done():
            raise HTTPException(status_code=409, detail="no robot run is active")
        self.state = "stopping"
        await self.publish({"type": "session_state", "state": self.state})
        stop_error = await self._request_run_stop(reason)
        return {"status": "stop_requested", "stop_error": stop_error}

    async def _request_run_stop(self, reason: str | None = None) -> str | None:
        """Stop the loop and cancel active motion; return the request's error."""

        self._stop_event.set()
        runtime = self.runtime
        if runtime is None:
            return None
        arguments = () if reason is None else (reason,)
        try:
            await asyncio.wait_for(
                asyncio.to_thread(runtime.agent.request_stop, *arguments), timeout=3.0
            )
        except Exception as exc:  # the Pi lease still expires independently
            stop_error = f"{type(exc).__name__}: {exc}"
            logger.warning("normal Web UI stop request failed: %s", stop_error)
            return stop_error
        return None

    async def shutdown(self, *, timeout_s: float = SERVER_SHUTDOWN_WAIT_S) -> None:
        """Stop the active run before the server exits.

        Ctrl+C stops the server, not the run: the run lives on a worker thread
        that the event loop's teardown waits for without stopping it, so the
        process would keep driving the robot with no UI left to stop it. This
        sends the Stop button's request, then waits for the run's safe
        shutdown and trace save.
        """

        worker = self._worker
        if worker is None or worker.done():
            return
        logger.warning("Web UI server shutting down: stopping the active robot run")
        try:
            await self.stop(SERVER_SHUTDOWN_STOP_REASON)
        except HTTPException:  # the run ended in the meantime
            return
        try:
            await asyncio.wait_for(asyncio.shield(worker), timeout=timeout_s)
        except asyncio.TimeoutError:
            logger.error(
                "the robot run has not finished %.0f s after the stop request; "
                "still waiting for its safe shutdown",
                timeout_s,
            )

    async def submit_policy(self, values: Mapping[str, Any]) -> dict[str, Any]:
        """Queue one operator program for the active manual-policy run.

        The program is handed to :class:`ManualModel`, so it is executed by
        the same loop, executor, trace, and stop path as an LLM policy.
        """

        code = str(values.get("code") or "")
        if not code.strip():
            raise HTTPException(status_code=422, detail="program must not be empty")
        runtime = self.runtime
        model = None if runtime is None else runtime.model
        if (
            self._worker is None
            or self._worker.done()
            or runtime is None
            or not isinstance(model, ManualModel)
        ):
            raise HTTPException(
                status_code=409,
                detail="no manual-policy run is active; start a run with the "
                "'Manual (operator)' model first",
            )
        if self.state != "running":
            raise HTTPException(
                status_code=409,
                detail=f"the manual-policy run is {self.state}, not running",
            )
        if model.pending() > 0:
            # One program per turn is enforced here, not only in the browser,
            # so a second tab or a script cannot stack motions unseen.
            raise HTTPException(
                status_code=409,
                detail="a program is already queued for this turn; wait for its "
                "feedback before submitting another",
            )
        model.submit(code)
        pending = model.pending()
        await self.publish(
            {
                "type": "policy_submitted",
                "turn": runtime.trace.turn,
                "code": code,
                "pending": pending,
            }
        )
        return {"status": "queued", "pending": pending}

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self.clients.add(websocket)
        for event in list(self.history):
            await websocket.send_text(event)
        await websocket.send_text(
            json.dumps(
                {
                    "type": "session_state",
                    "state": self.state,
                    "provider": self.provider,
                    "result": self.result,
                    "review": self._public_review(),
                    "replay": True,
                    "timestamp": _now(),
                }
            )
        )

    async def publish(
        self, event: Mapping[str, Any], *, remember: bool = True
    ) -> None:
        payload = dict(event)
        payload.setdefault("timestamp", _now())
        self._sequence += 1
        payload["sequence"] = self._sequence
        message = json.dumps(payload, default=_json_default, allow_nan=False)
        if remember:
            self.history.append(message)
            # A run is bounded to at most 100 turns; this cap is only defensive.
            if len(self.history) > 1000:
                self.history = self.history[-1000:]

        disconnected: list[WebSocket] = []
        for websocket in list(self.clients):
            try:
                await websocket.send_text(message)
            except Exception:
                disconnected.append(websocket)
        for websocket in disconnected:
            self.clients.discard(websocket)

    def emit_from_runtime(self, event: Mapping[str, Any]) -> None:
        """Thread-safe callback passed into ``DefaultAgent`` and the executor."""

        loop = self._loop
        if loop is None or loop.is_closed():
            return
        event_type = event.get("type")
        if event_type == "run_started":
            # Reconnecting clients (e.g. a page refresh mid-run) get this as
            # the catch-up state; without it they get stuck on "starting"
            # once the one-time run_started event has scrolled out of view.
            self.state = "running"
        elif event_type == "run_finished":
            # Leave "running" as soon as the loop ends so a program submitted
            # during the safe shutdown is rejected instead of silently queued.
            self.state = str(event.get("status") or "finished")
        elif event_type == "run_error":
            self.state = "error"
        payload = dict(event)
        loop.call_soon_threadsafe(
            lambda: asyncio.create_task(self.publish(payload))
        )

    async def _run(self, config: Mapping[str, Any]) -> None:
        camera_task = asyncio.create_task(self._camera_stream())
        # A plain executor future, not a task: the loop's teardown cancels
        # every task, and this one must stay awaitable after the stop below.
        run = asyncio.get_running_loop().run_in_executor(
            None,
            functools.partial(contextvars.copy_context().run, self._run_sync, config),
        )
        try:
            try:
                result = await asyncio.shield(run)
            except asyncio.CancelledError:
                # Only the event loop's teardown cancels this task, e.g. a
                # second Ctrl+C that skips the server's shutdown. The thread
                # cannot be cancelled, so stop the run and wait for it.
                await self._request_run_stop(SERVER_SHUTDOWN_STOP_REASON)
                result = await run
            self.result = result
            self.state = result.get("status", "finished")
        except BaseException as exc:  # surfaced in the UI; agent already traced it
            self.result = {
                "status": "error",
                "reason": str(exc),
                "error_type": type(exc).__name__,
            }
            self.state = "error"
            await self.publish(
                {
                    "type": "run_error",
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                }
            )
        finally:
            camera_task.cancel()
            try:
                await camera_task
            except asyncio.CancelledError:
                pass
            episode = self._episode
            if episode is not None:
                try:
                    self.review_pending = await asyncio.to_thread(
                        self._record_episode, config, episode
                    )
                except Exception as exc:  # noqa: BLE001 - the run state must still reach the UI
                    logger.error("could not record the experiment episode: %s", exc)
            await self.publish(
                {
                    "type": "session_state",
                    "state": self.state,
                    "provider": self.provider,
                    "result": self.result,
                    "review": self._public_review(),
                }
            )

    def _run_sync(self, config: Mapping[str, Any]) -> dict[str, Any]:
        runtime = build_runtime(
            config,
            on_event=self.emit_from_runtime,
            stop_event=self._stop_event,
        )
        if self._decorate_runtime is not None:
            self._decorate_runtime(runtime)
        self.runtime = runtime
        self.publish_primitives(runtime)
        # The episode clock covers the agent loop, which starts before the
        # robot is reset, like the navigation baselines' episode clocks.
        started = time.monotonic()
        self._timing = {"started_at": _now()}
        try:
            return runtime.agent.run(config["task"]["instruction"], config)
        finally:
            self._timing["ended_at"] = _now()
            self._timing["elapsed_s"] = round(time.monotonic() - started, 3)

    async def review(self, values: Mapping[str, Any]) -> dict[str, Any]:
        """Record the operator's review of the last experiment episode."""

        async with self._state_lock:
            pending = self.review_pending
            if pending is None:
                raise HTTPException(
                    status_code=409, detail="no experiment episode is waiting for review"
                )
            task_success = values.get("task_success")
            if task_success is not None and not isinstance(task_success, bool):
                raise HTTPException(
                    status_code=422, detail="task_success must be true, false or null"
                )
            adopted = values.get("adopted")
            if not isinstance(adopted, bool):
                raise HTTPException(status_code=422, detail="adopted must be true or false")
            exclusion_reason = str(values.get("exclusion_reason") or "").strip() or None
            note = str(values.get("note") or "").strip() or None
            protocol = experiment_protocol()
            review = protocol.OperatorReview(
                task_success=task_success,
                adopted=adopted,
                exclusion_reason=None if adopted else exclusion_reason,
                note=note,
            )
            output_dir = Path(pending["output_dir"])
            result = json.loads((output_dir / "result.json").read_text(encoding="utf-8"))
            episode_dir = await asyncio.to_thread(
                protocol.record_review,
                output_dir,
                experiment_dir=Path(pending["experiment_dir"]),
                review=review,
                result=result,
                log=logger.info,
            )
            self.review_pending = None
            await self.publish(
                {
                    "type": "episode_reviewed",
                    "result_path": str(Path(episode_dir) / "result.json"),
                    "task_success": task_success,
                    "adopted": adopted,
                }
            )
            await self.publish(
                {
                    "type": "session_state",
                    "state": self.state,
                    "provider": self.provider,
                    "result": self.result,
                    "review": None,
                }
            )
            return {"status": "recorded", "episode_dir": str(episode_dir)}

    def _record_episode(
        self, config: Mapping[str, Any], episode: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Write the episode's result.json before the operator reviews it."""

        protocol = experiment_protocol()
        result = dict(self.result or {})
        timing = dict(self._timing or {})
        experiment = dict(episode["experiment"])
        planner = dict(config.get("navigation_planner") or {})
        planner_enabled = bool(planner.get("enabled", False))
        agent_config = dict(config.get("agent") or {})
        model = dict(config.get("model") or {})
        record = {
            "experiment": experiment["name"],
            "start_label": experiment["start_label"],
            "condition": experiment["condition"],
            "task_id": episode["task_id"],
            "instruction": (config.get("task") or {}).get("instruction"),
            "termination": episode_termination(result),
            "reason": result.get("reason"),
            "turns": result.get("turns"),
            "elapsed_s": timing.get("elapsed_s"),
            "started_at": timing.get("started_at"),
            "ended_at": timing.get("ended_at"),
            "time_limit_s": agent_config.get("time_limit_s"),
            "max_turns": agent_config.get("max_turns"),
            "model": {"provider": model.get("provider"), "name": model.get("name")},
            "navigation_planner": {
                "enabled": planner_enabled,
                "model": planner.get("model") if planner_enabled else None,
                "memory_path": planner.get("memory_path") if planner_enabled else None,
            },
            "readiness_prior_enabled": bool(
                (readiness_prior_block(config) or {}).get("enabled", False)
            ),
            "human_ablation": config.get("human_ablation"),
            "task_success": None,
            "task_success_source": None,
            "adopted": None,
            "exclusion_reason": None,
            "note": None,
            **protocol.yor_version(),
        }
        output_dir = Path(episode["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "result.json").write_text(
            json.dumps(record, indent=2, default=_json_default), encoding="utf-8"
        )
        return {
            **dict(episode),
            "termination": record["termination"],
            "elapsed_s": record["elapsed_s"],
            "reason": record["reason"],
        }

    def _public_review(self) -> dict[str, Any] | None:
        pending = self.review_pending
        if pending is None:
            return None
        experiment = pending["experiment"]
        return {
            "episode_dir": pending["output_dir"],
            "task_id": pending["task_id"],
            "experiment": experiment["name"],
            "condition": experiment["condition"],
            "start_label": experiment["start_label"],
            "termination": pending.get("termination"),
            "elapsed_s": pending.get("elapsed_s"),
            "reason": pending.get("reason"),
        }

    def publish_primitives(self, runtime: Any) -> None:
        """Announce the exposed primitive set once the registry exists.

        The event is remembered in the history so a reconnecting browser can
        rebuild the manual-policy palette. Thread-safe like every runtime
        event; failures never affect the run.
        """

        try:
            items = primitive_catalog(runtime.registry)
        except Exception as exc:  # noqa: BLE001 - a palette must never break a run
            logger.warning("primitive catalog unavailable: %s", exc)
            return
        self.emit_from_runtime({"type": "primitives", "items": items})

    async def _camera_stream(self, fps: float = 2.0) -> None:
        """Publish a low-rate operator preview without touching agent state."""

        while True:
            runtime = self.runtime
            if runtime is not None:
                try:
                    preview = await asyncio.to_thread(
                        runtime.environment.latest_camera_preview
                    )
                    if preview is not None:
                        image_url = await asyncio.to_thread(
                            _preview_data_url, preview["rgb"]
                        )
                        await self.publish(
                            {
                                "type": "camera_preview",
                                "image_url": image_url,
                                "frame_timestamp_ns": preview["timestamp_ns"],
                            },
                            remember=False,
                        )
                except Exception:
                    # Missing/stale preview frames never affect the agent run.
                    pass
            await asyncio.sleep(1.0 / fps)


def create_app(
    *,
    config_path: Path,
    config: Mapping[str, Any],
    decorate_runtime: Callable[[Any], None] | None = None,
) -> FastAPI:
    controller = WebRunController(
        config_path, config, decorate_runtime=decorate_runtime
    )

    @contextlib.asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        await controller.shutdown()

    app = FastAPI(title="YOR Policy Console", version="0.1.0", lifespan=lifespan)
    app.state.controller = controller

    static_dir = Path(__file__).with_name("static")
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.middleware("http")
    async def disable_console_asset_cache(request: Request, call_next):
        response = await call_next(request)
        if request.url.path == "/" or request.url.path.startswith("/static/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(static_dir / "index.html")

    @app.get("/api/config")
    async def get_config() -> dict[str, Any]:
        return controller.public_config()

    @app.post("/api/start")
    async def start(request: Request) -> dict[str, Any]:
        try:
            values = await request.json()
        except Exception as exc:
            raise HTTPException(status_code=400, detail="request body must be JSON") from exc
        if not isinstance(values, Mapping):
            raise HTTPException(status_code=400, detail="request body must be an object")
        try:
            return await controller.start(values)
        except HTTPException as exc:
            logger.warning("start request rejected: %s", exc.detail)
            raise

    @app.post("/api/stop")
    async def stop() -> dict[str, Any]:
        return await controller.stop()

    @app.post("/api/review")
    async def review(request: Request) -> dict[str, Any]:
        try:
            values = await request.json()
        except Exception as exc:
            raise HTTPException(status_code=400, detail="request body must be JSON") from exc
        if not isinstance(values, Mapping):
            raise HTTPException(status_code=400, detail="request body must be an object")
        return await controller.review(values)

    @app.post("/api/policy")
    async def submit_policy(request: Request) -> dict[str, Any]:
        try:
            values = await request.json()
        except Exception as exc:
            raise HTTPException(status_code=400, detail="request body must be JSON") from exc
        if not isinstance(values, Mapping):
            raise HTTPException(status_code=400, detail="request body must be an object")
        try:
            return await controller.submit_policy(values)
        except HTTPException as exc:
            logger.warning("policy submission rejected: %s", exc.detail)
            raise

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket) -> None:
        await controller.connect(websocket)
        try:
            while True:
                await websocket.receive_text()
        except WebSocketDisconnect:
            pass
        finally:
            controller.clients.discard(websocket)

    return app


def run_web_ui(
    *, config_path: Path, config: Mapping[str, Any], host: str, port: int
) -> None:
    """Run the browser UI without starting robot motion until Start is clicked."""

    app = create_app(config_path=config_path, config=config)
    print(f"\n  YOR Policy Console: http://localhost:{port}")
    print("  The robot remains idle until you click Start.\n")
    uvicorn.run(app, host=host, port=int(port))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_default(value: Any) -> Any:
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return tolist()
    return repr(value)


def _preview_data_url(rgb: Any, max_width: int = 1024) -> str:
    """Encode a transient low-bandwidth preview frame as JPEG."""

    import numpy as np
    from PIL import Image

    array = np.asarray(rgb)
    image = Image.fromarray(np.ascontiguousarray(array[:, :, :3], dtype=np.uint8))
    if max_width > 0 and image.width > max_width:
        height = max(1, round(image.height * max_width / image.width))
        image = image.resize((max_width, height), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=72, optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"
