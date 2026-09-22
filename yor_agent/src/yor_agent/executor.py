"""Run one LLM-generated Python policy against a capability-limited namespace.

``PolicyExecutor`` is a software component on the Jetson: a Python-policy
interpreter, not the robot. The real-robot effect boundary is ``YorEnvironment``
plus its controller and hardware bridge. The executor never sees the environment,
hardware bridge, RPC clients, or credentials -- only the callables that were
explicitly registered in a :class:`PrimitiveRegistry`.
"""

from __future__ import annotations

import builtins
import contextlib
import functools
import io
import time
import traceback
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

from .exceptions import Finished, PrimitiveFailed
from .primitives.registry import PrimitiveRegistry

# Explicit allowlist. ``__import__`` is intentionally absent, so an ``import``
# statement inside a generated policy raises ImportError instead of reaching the
# filesystem, network, or the hardware packages installed on the Jetson.
ALLOWED_BUILTINS = (
    "abs",
    "all",
    "any",
    "bool",
    "dict",
    "divmod",
    "enumerate",
    "float",
    "int",
    "isinstance",
    "len",
    "list",
    "max",
    "min",
    "print",
    "range",
    "repr",
    "reversed",
    "round",
    "set",
    "sorted",
    "str",
    "sum",
    "tuple",
    "zip",
    # Exception types a policy may legitimately catch or raise.
    "Exception",
    "RuntimeError",
    "TypeError",
    "ValueError",
    "ZeroDivisionError",
)

SAFE_BUILTINS: dict[str, Any] = {
    name: getattr(builtins, name) for name in ALLOWED_BUILTINS
}


def _warm_up_array_printing() -> None:
    """Force numpy's lazy print machinery to initialize in a normal namespace.

    ``ndarray.__str__`` performs a deferred import the first time an array is
    printed. Inside the policy namespace there is no ``__import__``, so the very
    first ``print(observation["base"]["pose_xy_yaw"])`` would die with
    "Unable to configure default ndarray.__str__". Doing it once here, at import
    time, makes array printing work for every policy without handing generated
    code an import hook.
    """

    for array in (np.zeros(3), np.zeros((2, 2), dtype=np.uint8)):
        str(array)
        repr(array)


_warm_up_array_printing()

FINISH_DOC = """\
def finish(reason: str) -> None:
    \"\"\"
    Stop the whole run and hand ``reason`` back to the operator.

    This is a control transition only. It does not verify, claim, or record
    that the physical goal was reached, and it does not command the robot.
    The outer loop always performs a safe shutdown afterwards.
    \"\"\""""

MAX_STDOUT_CHARS = 8000
MAX_REPR_CHARS = 600

# Subtrees that exist only to explain a failure. Summarising them at the usual
# depth turned the clearance gate's per-layer evidence into a repr cut off at
# MAX_REPR_CHARS, which dropped the very height band that had refused the move.
#
# Keyed on the guard block and on the per-layer list rather than on
# "clearance" itself. A drive's own gate metrics put that list at depth three,
# so every layer used to arrive as a repr string a reader had to parse. A
# motion history entry collapses at its "clearance" key before "layers" is
# reached, so this still does not expand every drive in a history, which is
# what doubled a docking trace when "clearance" itself was lifted.
DEEP_TRACE_KEYS = frozenset({"diagnostics", "last_motion_guard_block", "layers"})
DEEP_TRACE_DEPTH = 10

# Lists kept whole however long they are. The prepare search's per-query log is
# the data an IK-budget curve is replayed from, and collapsing it to its length
# left every trace unable to say when the first base certified. Its rows are
# flat scalars, so the cost is bounded by the query limit.
FULL_TRACE_LIST_KEYS = frozenset({"pi_ik_query_log"})


class PolicyExecutor:
    """Execute one generated Python policy with only registered primitives in scope."""

    def __init__(
        self,
        registry: PrimitiveRegistry,
        *,
        on_primitive_call: Callable[[dict[str, Any]], None] | None = None,
        max_stdout_chars: int = MAX_STDOUT_CHARS,
    ) -> None:
        self.registry = registry
        self._on_primitive_call = on_primitive_call
        self._max_stdout_chars = int(max_stdout_chars)

    def documentation(self) -> str:
        """Primitive documentation plus the executor-provided ``finish()``."""

        return "\n\n".join([self.registry.documentation(), FINISH_DOC])

    def execute(self, code: str) -> dict[str, Any]:
        """Run one policy and return a compact execution record.

        A fresh namespace is built for every policy: there are no globals
        carried over from the previous turn. ``Finished`` is re-raised before
        ordinary exceptions are converted into feedback, so a generated
        ``finish()`` call is never misreported as a runtime error.
        """

        calls: list[dict[str, Any]] = []
        namespace = self._build_namespace(calls)
        stdout, stderr = io.StringIO(), io.StringIO()
        record: dict[str, Any] = {
            "stdout": "",
            "stderr": "",
            "error": None,
            "interrupted_by": None,
            "primitive_calls": calls,
            "finish_reason": None,
            "elapsed_s": 0.0,
        }
        started = time.monotonic()

        try:
            compiled = compile(code, "<policy>", "exec")
        except SyntaxError as exc:
            record["elapsed_s"] = time.monotonic() - started
            record["error"] = {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": "".join(traceback.format_exception_only(type(exc), exc)),
            }
            return record

        try:
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                exec(compiled, namespace)  # noqa: S102 - trusted supervised experiment
        except Finished as finished:
            record["finish_reason"] = finished.reason
            self._finalize(record, stdout, stderr, started)
            finished.execution = record
            raise
        except PrimitiveFailed as failure:
            record["interrupted_by"] = {
                "primitive": failure.primitive,
                "reason": failure.reason,
                "result": _summarize(failure.result),
            }
        except BaseException as exc:  # noqa: BLE001 - converted into model feedback
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                # An operator Ctrl-C during motion is exactly when the partial
                # record matters most, so carry it out with the exception.
                self._finalize(record, stdout, stderr, started)
                exc.execution = record  # type: ignore[attr-defined]
                raise
            record["error"] = {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": _policy_traceback(exc),
            }

        self._finalize(record, stdout, stderr, started)
        return record

    def _finalize(
        self,
        record: dict[str, Any],
        stdout: io.StringIO,
        stderr: io.StringIO,
        started: float,
    ) -> None:
        record["stdout"] = _truncate(stdout.getvalue(), self._max_stdout_chars)
        record["stderr"] = _truncate(stderr.getvalue(), self._max_stdout_chars)
        record["elapsed_s"] = round(time.monotonic() - started, 3)

    def _build_namespace(self, calls: list[dict[str, Any]]) -> dict[str, Any]:
        namespace: dict[str, Any] = {"__builtins__": dict(SAFE_BUILTINS)}
        for name, function in self.registry.functions().items():
            namespace[name] = self._traced(name, function, calls)
        namespace["finish"] = _finish
        # Primitive signatures are typed as ``np.ndarray`` and the model is
        # shown those resolved type hints, so it reasonably expects numpy to
        # be available. Binding the module directly (rather than allowing
        # ``import numpy``) keeps the no-``__import__`` sandbox intact.
        namespace["np"] = np
        return namespace

    def _traced(
        self,
        name: str,
        function: Callable[..., Any],
        calls: list[dict[str, Any]],
    ) -> Callable[..., Any]:
        """Wrap a primitive so every invocation lands in the run trace."""

        @functools.wraps(function)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            event: dict[str, Any] = {
                "index": len(calls),
                "name": name,
                "args": [_summarize(value) for value in args],
                "kwargs": {key: _summarize(value) for key, value in kwargs.items()},
                "started_at": time.time(),
                "elapsed_s": None,
                "result": None,
                "error": None,
            }
            calls.append(event)
            started = time.monotonic()
            try:
                result = function(*args, **kwargs)
            except PrimitiveFailed as failure:
                event["result"] = _summarize(failure.result)
                event["error"] = {"type": "PrimitiveFailed", "message": failure.reason}
                raise
            except BaseException as exc:
                event["error"] = {"type": type(exc).__name__, "message": str(exc)}
                raise
            else:
                event["result"] = _summarize(result)
                return result
            finally:
                event["elapsed_s"] = round(time.monotonic() - started, 3)
                if self._on_primitive_call is not None:
                    self._on_primitive_call(event)

        return wrapper


def _finish(reason: str = "") -> None:
    """Stop the run and report ``reason`` (no physical-success claim)."""

    raise Finished(reason)


def _policy_traceback(exc: BaseException) -> str:
    """Format a traceback starting at the generated policy's own frame."""

    tb = exc.__traceback__
    if tb is not None and tb.tb_next is not None:
        # Skip PolicyExecutor.execute's own ``exec`` frame.
        tb = tb.tb_next
    return "".join(traceback.format_exception(type(exc), exc, tb))


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    omitted = len(text) - limit
    return f"{text[:limit]}\n... [{omitted} characters truncated]"


def _summarize(
    value: Any,
    *,
    depth: int = 0,
    maximum_depth: int = 4,
    keep_long_list: bool = False,
) -> Any:
    """Convert a primitive argument/result into something JSON-friendly.

    Large arrays are replaced by a shape/dtype description so the trace keeps
    references rather than raw RGB-D data.
    """

    if value is None or isinstance(value, (bool, int, float, str)):
        if isinstance(value, str) and len(value) > MAX_REPR_CHARS:
            return _truncate(value, MAX_REPR_CHARS)
        if isinstance(value, float) and value != value:  # NaN is not JSON-valid
            return "NaN"
        return value
    if depth >= maximum_depth:
        return _short_repr(value)
    if isinstance(value, Mapping):
        return {
            str(key): _summarize(
                item,
                depth=depth + 1,
                # Failure evidence is intentionally nested and is the whole
                # reason a trace gets read afterwards. Preserve those subtrees
                # while the usual shallow bound holds for all other data.
                maximum_depth=(
                    DEEP_TRACE_DEPTH if str(key) in DEEP_TRACE_KEYS else maximum_depth
                ),
                keep_long_list=str(key) in FULL_TRACE_LIST_KEYS,
            )
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        if len(value) > 32 and not keep_long_list:
            return f"<{type(value).__name__} len={len(value)}>"
        return [
            _summarize(
                item, depth=depth + 1, maximum_depth=maximum_depth
            )
            for item in value
        ]
    shape = getattr(value, "shape", None)
    if shape is not None:
        dtype = getattr(value, "dtype", None)
        tolist = getattr(value, "tolist", None)
        if getattr(value, "size", 1) <= 8 and callable(tolist):
            # Poses and velocities stay readable in the trace; RGB-D does not.
            return _summarize(
                tolist(), depth=depth + 1, maximum_depth=maximum_depth
            )
        return f"<array shape={tuple(shape)} dtype={dtype}>"
    return _short_repr(value)


def _short_repr(value: Any) -> str:
    try:
        text = repr(value)
    except Exception:  # noqa: BLE001 - repr must never break tracing
        text = f"<unreprable {type(value).__name__}>"
    return _truncate(text, MAX_REPR_CHARS)
