"""Shared catalog of named primitive callables exposed to generated policies.

The registry deliberately stores plain callables rather than ``Primitive``
objects: the current YOR code already exposes navigation and manipulation as a
name -> bound-method mapping, and signatures plus docstrings are enough to build
the policy prompt. A richer ``PrimitiveSpec`` can be added later if categories,
permissions, or schemas are genuinely needed.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Iterator
from typing import Any

PrimitiveFn = Callable[..., Any]


class PrimitiveRegistry:
    """Named callables that generated Python policies are allowed to call."""

    def __init__(self) -> None:
        self._functions: dict[str, PrimitiveFn] = {}

    def register(self, name: str, function: PrimitiveFn) -> None:
        """Register one callable under a Python-identifier name.

        Args:
            name: Name the generated policy will call. Must be a valid, unused
                Python identifier and must not shadow a keyword.
            function: Callable implementing the primitive. Its signature and
                docstring become the model-facing documentation.
        """

        if not isinstance(name, str) or not name.isidentifier():
            raise ValueError(f"primitive name must be an identifier: {name!r}")
        if name in self._functions:
            raise ValueError(f"primitive {name!r} is already registered")
        if not callable(function):
            raise TypeError(f"primitive {name!r} must be callable")
        self._functions[name] = function

    def functions(self) -> dict[str, PrimitiveFn]:
        """Return a copy of the name -> callable mapping."""

        return dict(self._functions)

    def unregister(self, name: str) -> None:
        """Remove a callable from model documentation and policy scope."""

        self._functions.pop(name, None)

    def names(self) -> list[str]:
        return list(self._functions)

    def documentation(self) -> str:
        """Render signatures and docstrings for the model prompt."""

        blocks = [
            _document_function(name, function)
            for name, function in self._functions.items()
        ]
        return "\n\n".join(blocks)

    def __contains__(self, name: object) -> bool:
        return name in self._functions

    def __len__(self) -> int:
        return len(self._functions)

    def __iter__(self) -> Iterator[str]:
        return iter(self._functions)


def _document_function(name: str, function: PrimitiveFn) -> str:
    """Format one callable as ``def name(signature):`` plus its docstring."""

    try:
        # eval_str resolves the string annotations produced by
        # ``from __future__ import annotations`` back into readable types, so the
        # model sees ``distance_m: float`` rather than ``distance_m: 'float'``.
        signature = str(inspect.signature(function, eval_str=True))
    except (TypeError, ValueError, NameError):  # builtins, C callables, odd scopes
        try:
            signature = str(inspect.signature(function))
        except (TypeError, ValueError):
            signature = "(...)"
    signature = signature.replace("typing.", "")
    doc = inspect.getdoc(function) or "(no documentation)"
    body = "\n".join(f"    {line}".rstrip() for line in doc.splitlines())
    return f"def {name}{signature}:\n    \"\"\"\n{body}\n    \"\"\""
