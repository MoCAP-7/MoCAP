"""Navigation vocabularies exposed to Cap-X.

``CapXYorNavigationApi`` gives Cap-X the four production YOR navigation
primitives; ``CapXYorCoarseNavigationApi`` gives it only the coarse CaP-X
vocabulary (1 m forward, 45-degree turns, a relative planar position).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .bootstrap import configure_import_paths

configure_import_paths()

from capx.integrations.base_api import ApiBase  # noqa: E402
from yor_agent.primitive_config import primitive_defaults, primitive_settings  # noqa: E402
from yor_agent.primitives.coarse_navigation import (  # noqa: E402
    COARSE_NAVIGATION_PRIMITIVES,
    register_coarse_navigation_primitives,
)
from yor_agent.primitives.navigation import register_navigation_primitives  # noqa: E402
from yor_agent.primitives.registry import PrimitiveRegistry  # noqa: E402
from yor_agent.primitives.visible_object_navigation import (  # noqa: E402
    register_visible_object_navigation_primitives,
)


NAVIGATION_PRIMITIVES = (
    "drive_straight",
    "turn_relative",
    "drive_lateral",
    "dock_to_visible_object",
)


class CapXYorNavigationApi(ApiBase):
    """Cap-X API backed by the exact current YOR navigation wrappers."""

    def __init__(
        self,
        env: Any,
        *,
        registry: PrimitiveRegistry | None = None,
        docking_factories: dict[str, Callable[..., Any]] | None = None,
    ) -> None:
        super().__init__(env)
        if registry is None:
            registry = PrimitiveRegistry()
        defaults = {
            name: primitive_defaults(env.primitive_config, name)
            for name in ("turn_relative", "drive_straight", "drive_lateral")
        }
        register_navigation_primitives(
            registry,
            env.yor_environment,
            primitive_defaults=defaults,
        )
        docking_factories = dict(docking_factories or {})
        register_visible_object_navigation_primitives(
            registry,
            env.yor_environment,
            docking_config=primitive_settings(
                env.primitive_config, "dock_to_visible_object"
            ),
            **docking_factories,
        )
        registered = registry.functions()
        missing = [name for name in NAVIGATION_PRIMITIVES if name not in registered]
        if missing:
            raise RuntimeError(f"YOR navigation registration omitted {missing}")
        self._functions = {name: registered[name] for name in NAVIGATION_PRIMITIVES}

    def functions(self) -> dict[str, Callable[..., Any]]:
        return dict(self._functions)


class CapXYorCoarseNavigationApi(ApiBase):
    """Cap-X API backed by YOR's coarse CaP-X navigation vocabulary.

    The primitives are the ``yor_agent`` registrations used by the coarse
    navigation Web UI baseline, with the defaults of the loaded primitive
    configuration; no docking or fine motion primitive is registered.
    """

    def __init__(
        self,
        env: Any,
        *,
        registry: PrimitiveRegistry | None = None,
    ) -> None:
        super().__init__(env)
        if registry is None:
            registry = PrimitiveRegistry()
        register_coarse_navigation_primitives(
            registry,
            env.yor_environment,
            primitive_defaults={
                name: primitive_defaults(env.primitive_config, name)
                for name in COARSE_NAVIGATION_PRIMITIVES
            },
        )
        registered = registry.functions()
        self._functions = {
            name: registered[name] for name in COARSE_NAVIGATION_PRIMITIVES
        }

    def functions(self) -> dict[str, Callable[..., Any]]:
        return dict(self._functions)
