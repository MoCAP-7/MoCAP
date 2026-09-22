"""Small hardware/environment contracts shared by the robot layer and tests.

These protocols are the only coupling between ``NavigationController`` and the
rest of the system, so ``tests/fakes.py`` can provide a tiny stand-in with the
same surface and no moving hardware.
"""

from __future__ import annotations

from typing import Any, Protocol


class NavigationHardware(Protocol):
    """Minimum transport surface required by the navigation environment."""

    def latest_frame(self, *, max_age_s: float) -> Any | None: ...

    def next_frame(self, timeout_s: float | None = 2.0) -> Any: ...

    def get_base_status(self) -> dict[str, Any]: ...

    def submit_base_velocity(self, velocity: list[float]) -> dict[str, Any]: ...

    def get_arm_status(self) -> dict[str, Any] | None: ...

    def close(self) -> None: ...


class NavigationEnvironment(Protocol):
    """Environment methods used by :class:`NavigationController`.

    Note that ``submit_base_velocity`` lives here, inside the environment
    boundary. It is never registered as a primitive and never reaches a
    generated policy's namespace.
    """

    navigation_config: dict[str, Any]

    def navigation_frame(self, *, max_age_s: float) -> Any: ...

    def base_status(self) -> dict[str, Any]: ...

    def submit_base_velocity(self, velocity: list[float]) -> dict[str, Any]: ...
