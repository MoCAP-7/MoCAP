"""Primitive registry and the domain providers that populate it."""

from .navigation import register_navigation_primitives
from .registry import PrimitiveFn, PrimitiveRegistry

__all__ = ["PrimitiveFn", "PrimitiveRegistry", "register_navigation_primitives"]
