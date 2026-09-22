"""Visible-object docking primitive exposed to generated policies.

This module owns the LLM-facing name and docstring for
``dock_to_visible_object`` (moved from
``YOR/Agent/agents_yor/visible_object_navigation.py``'s
``YorVisibleObjectNavigationApi``). The production backend turns the SAM3
metric target into one Nav2 goal; Nav2 owns planning, control, obstacle
avoidance, and rolling replanning.

``dock_to_visible_object`` is effectful (it drives the base) and always
performs its own final stop internally, so on failure this wrapper raises
:class:`PrimitiveFailed` without requesting a second stop -- same convention
as ``primitives/manipulation.py``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
import inspect
from typing import Any

from ..exceptions import PrimitiveFailed
from ..primitives.registry import PrimitiveRegistry
from ..robot.nav2_visible_object_navigation import (
    Nav2VisibleObjectDockingController,
)
from ..robot.visible_object_navigation import VisibleObjectDockingController


def register_visible_object_navigation_primitives(
    registry: PrimitiveRegistry,
    env: Any,
    *,
    docking_config: Mapping[str, Any] | None = None,
    segment_client_factory: Callable[[], Callable[..., Any]] | None = None,
    detector_factory: Callable[[Any], Any] | None = None,
    nav2_client_factory: Callable[[Any], Any] | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    readiness_prior: Any | None = None,
) -> None:
    """Register ``dock_to_visible_object`` against one ``YorEnvironment``.

    The optional factories default to the real SAM3 and lazy YOLOE clients;
    tests inject fakes here instead. ``readiness_prior`` (a duck-typed
    ``ReadinessPrior``) only reaches the Nav2 backend; the legacy backend
    ignores it.
    """

    backend = str(dict(docking_config or {}).get("backend", "legacy")).strip().lower()
    if backend == "nav2":
        controller = Nav2VisibleObjectDockingController(
            env,
            docking_config=docking_config,
            segment_client_factory=segment_client_factory,
            detector_factory=detector_factory,
            nav2_client_factory=nav2_client_factory,
            progress_callback=progress_callback,
            readiness_prior=readiness_prior,
        )
    elif backend == "legacy":
        legacy_config = dict(docking_config or {})
        legacy_config.pop("backend", None)
        controller = VisibleObjectDockingController(
            env,
            docking_config=legacy_config,
            segment_client_factory=segment_client_factory,
            detector_factory=detector_factory,
        )
    else:
        raise ValueError(
            "dock_to_visible_object backend must be 'nav2' or 'legacy', "
            f"got {backend!r}"
        )
    registry.register(
        "dock_to_visible_object", _make_dock_to_visible_object(controller)
    )


def _accepts_approach_bearing(dock: Callable[..., Any]) -> bool:
    """Whether a backend's ``dock_to_visible_object`` takes ``approach_bearing_deg``.

    Only the Nav2 backend has the keyword; the legacy controller would raise
    a TypeError after its entry stop. A callable whose signature cannot be
    read is left to the call itself.
    """

    try:
        parameters = inspect.signature(dock).parameters
    except (TypeError, ValueError):
        return True
    return "approach_bearing_deg" in parameters or any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )


def _make_dock_to_visible_object(controller: Any):
    accepts_bearing = _accepts_approach_bearing(controller.dock_to_visible_object)

    def dock_to_visible_object(
        object_name: str, *, approach_bearing_deg: float | None = None
    ) -> dict[str, Any]:
        """Dock to a currently visible object using SAM3, ZED, and Nav2.

        The caller chooses the semantic navigation target, such as ``"table"``
        for a later marker grasp. The primitive segments the target once,
        projects its closest ground-plane point into ``odom``, and asks Nav2 to
        reach the configured standoff while facing the object. Nav2 owns
        collision-aware planning, control, and replanning. A fresh SAM3
        observation verifies the final distance.

        Immediately after close-range manipulation, the robot may be too close
        to the table or another support surface for Nav2 to move. In that case
        this primitive returns ``nav2_blocked_near_obstacle``. Do not retry from
        the same base pose: if the rear path is known to be clear, first use a
        short negative ``drive_straight`` distance to back away, re-observe, and
        then retry this primitive.

        If Nav2 cannot make meaningful physical base-pose progress for the
        configured short limit, the primitive stops that goal and observes the
        object again. When the base already stands on the requested side
        within that goal's docking distance, docking succeeds there
        (``docked_after_stall``). Otherwise, once no farther goal is left, it
        returns ``nav2_stalled_no_progress`` instead of waiting through
        repeated recoveries. Re-observe; when the target remains roughly ahead and the
        forward path is clear, a short positive ``drive_straight`` adjustment
        may be used before retrying docking.

        Args:
            object_name: SAM3 prompt for an object visible in the current ZED
                image. Name the object with its short category noun, one or
                two words such as ``"can"``, ``"bottle"`` or ``"chair"``, not
                a sentence. SAM3 matches short noun phrases; a prompt that
                also describes colour, material, size or where the object
                sits usually returns no instance even with the object in
                view. Add one distinguishing word only when several objects
                of that kind are visible. If SAM3 finds no instance, retry
                with a shorter, more generic name before moving the robot,
                and keep using the name that worked in the manipulation
                primitives that follow.
            approach_bearing_deg: Optional object-centric approach direction
                in the odometry frame, in degrees: the direction from the
                object to the docking goal, the same quantity the result
                metrics report. Used by scripted experiments; omit it to let
                the human prior or the arrival direction choose. Needs the
                Nav2 docking backend: the legacy backend has no approach
                bearing and rejects the argument before moving.

        Returns:
            Result dictionary with ``success``, ``reason``, target geometry,
            path-clearance/planner metrics, and the bounded motion history.
            Continue to manipulation only when ``success`` is true. When the
            result contains ``suggested_arm``, pass it as ``arm`` to
            ``prepare_for_manipulation``: it names the arm the target ended up
            on the side of, which is the one that can reach without moving the
            base sideways first.
        """

        if not isinstance(object_name, str) or not object_name.strip():
            raise ValueError("object_name must be a non-empty string")

        if approach_bearing_deg is None:
            result = controller.dock_to_visible_object(object_name)
        elif not accepts_bearing:
            raise ValueError("approach_bearing_deg needs the nav2 docking backend")
        else:
            result = controller.dock_to_visible_object(
                object_name, approach_bearing_deg=approach_bearing_deg
            )
        if not result.get("success", False):
            raise PrimitiveFailed("dock_to_visible_object", result)
        return result

    dock_to_visible_object.__doc__ = (
        (dock_to_visible_object.__doc__ or "")
        + "\n\n        Configured success boundary: "
        + f"{controller.config.docking_distance_m:.2f} m from the target's "
        + "closest point."
    )
    return dock_to_visible_object
