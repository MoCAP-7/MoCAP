"""LLM-facing semantic primitive for last-centimeter base alignment."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Literal

from ..exceptions import PrimitiveFailed
from ..robot.manipulation import ManipulationController
from ..robot.manipulation_readiness import ManipulationReadinessController
from .registry import PrimitiveRegistry


def register_manipulation_readiness_primitive(
    registry: PrimitiveRegistry,
    env: Any,
    *,
    settings: Mapping[str, Any] | None = None,
    segment_client_factory: Callable[[], Callable[..., Any]] | None = None,
    grasp_backend_factory: Callable[[dict[str, Any]], Any] | None = None,
    manipulation_controller: ManipulationController | None = None,
    debug_callback: Callable[[dict[str, Any]], None] | None = None,
) -> ManipulationReadinessController:
    """Register the one high-level primitive and return its controller."""

    controller = ManipulationReadinessController(
        env,
        config=settings,
        segment_client_factory=segment_client_factory,
        grasp_backend_factory=grasp_backend_factory,
        manipulation_controller=manipulation_controller,
        debug_callback=debug_callback,
    )

    def prepare_for_manipulation(
        object_name: str, *, arm: Literal["left", "right"]
    ) -> dict[str, Any]:
        """Certify a nearby base pose for collision-aware grasp execution.

        Call this primitive only when the target object is currently visible in
        the camera view. It does not search for or center an absent/off-screen
        target. This primitive assumes ``dock_to_visible_object`` already
        brought YOR near the workspace. It uses SAM3 + the configured grasp
        model to evaluate a fine local SE(2) grid with batched Pi IK. A base
        pose is eligible only when at least one swept-volume collision-safe
        nominal grasp has ``ik_converged=True``; configured small base-pose
        perturbations are a soft ranking signal. Movement tracks one absolute
        SE(2) goal, permits overshoot correction, and continuously enforces the
        configured minimum chassis speeds outside the final tolerance. When
        that motion is refused, the next certified base poses are tried in the
        search's ranked order up to the configured alternative limit.
        After moving, it samples again and repeats the strict collision/Pi check
        at the actual pose; when the current pose is already selected, the
        evaluation's own collision-safe, Pi-converged goalset is certified
        without a second sampling pass. Preparation
        deliberately does not call the cuRobo trajectory planner;
        ``goto_grasp_pose`` owns final whole-arm planning. If the target is too
        far for this local search, the result instructs the caller to dock or
        move closer.
        Args:
            object_name: The object's short category noun, one or two words
                such as ``"can"``, not a sentence; SAM3 finds no instance for
                most descriptive prompts. Use the name that worked for
                ``dock_to_visible_object``.
            arm: Required Nero arm identifier.

        Returns:
            A base-readiness result with the selected relative base pose and
            strict Pi diagnostics. The next grasp sample revalidates the target
            but reuses the collision-safe, Pi-converged goalset.
            It never contains a Cartesian grasp pose. Failure stops the base
            and skips the rest of the generated policy.
        """

        if not isinstance(arm, str) or arm not in {"left", "right"}:
            raise ValueError(
                "arm must be explicitly passed as the string 'left' or 'right'"
            )
        result = controller.prepare_for_manipulation(object_name, arm)
        if not result.get("success", False):
            raise PrimitiveFailed("prepare_for_manipulation", result)
        return result

    registry.register("prepare_for_manipulation", prepare_for_manipulation)
    return controller
