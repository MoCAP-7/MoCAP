"""Loading and validation for the dedicated LLM primitive configuration."""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml


PRIMITIVE_NAMES = (
    "observe",
    "stop",
    "turn_relative",
    "drive_straight",
    "drive_lateral",
    "get_object_pose",
    "sample_grasp_pose",
    "goto_pose",
    "goto_grasp_pose",
    "open_gripper",
    "close_gripper",
    "lift_grasped_object",
    "dock_to_visible_object",
    "prepare_for_manipulation",
    # The CaP-X-style coarse navigation baseline (primitives/coarse_navigation.py).
    "go_forward",
    "turn_left_45_degrees",
    "turn_right_45_degrees",
    "goto_planar_position",
    "say_something",
)

_ALLOWED_DEFAULTS = {
    "observe": frozenset(),
    "stop": frozenset(),
    "turn_relative": frozenset({"max_yaw_deg_s", "timeout_s"}),
    "drive_straight": frozenset({"max_speed_mps", "timeout_s"}),
    "drive_lateral": frozenset({"max_speed_mps", "timeout_s"}),
    "get_object_pose": frozenset({"return_bbox_extent", "return_zed_distance"}),
    "sample_grasp_pose": frozenset(),
    "goto_pose": frozenset({"z_approach", "timeout_s"}),
    "goto_grasp_pose": frozenset({"approach_m", "timeout_s"}),
    "open_gripper": frozenset({"timeout_s", "force_n"}),
    "close_gripper": frozenset({"timeout_s", "force_n"}),
    "lift_grasped_object": frozenset({"lift_m", "timeout_s"}),
    "dock_to_visible_object": frozenset(),
    "prepare_for_manipulation": frozenset(),
    "go_forward": frozenset({"max_speed_mps", "timeout_s"}),
    "turn_left_45_degrees": frozenset({"max_yaw_deg_s", "timeout_s"}),
    "turn_right_45_degrees": frozenset({"max_yaw_deg_s", "timeout_s"}),
    "goto_planar_position": frozenset({"max_speed_mps", "timeout_s"}),
    "say_something": frozenset(),
}

#: Base motion primitives whose timeout the controller caps at 45 seconds.
_MOTION_TIMEOUT_PRIMITIVES = frozenset(
    {
        "turn_relative",
        "drive_straight",
        "drive_lateral",
        "go_forward",
        "turn_left_45_degrees",
        "turn_right_45_degrees",
        "goto_planar_position",
    }
)

_SETTING_PRIMITIVES = frozenset(
    {
        "get_object_pose",
        "sample_grasp_pose",
        "open_gripper",
        "close_gripper",
        "dock_to_visible_object",
        "prepare_for_manipulation",
    }
)


def load_primitive_config(path: str | Path) -> dict[str, Any]:
    """Load one complete primitive configuration from YAML."""

    config_path = Path(path).expanduser().resolve()
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    try:
        return normalize_primitive_config(payload, require_all=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid primitive config {config_path}: {exc}") from exc


def normalize_primitive_config(
    payload: Mapping[str, Any] | None,
    *,
    require_all: bool = False,
) -> dict[str, Any]:
    """Validate and detach a primitive configuration mapping."""

    if payload is None:
        return {"version": 1, "primitives": {}}
    if not isinstance(payload, Mapping):
        raise TypeError("primitive_config must be a path or mapping")
    unknown_top_level = sorted(set(payload) - {"version", "primitives"})
    if unknown_top_level:
        raise ValueError(f"unknown primitive config keys: {unknown_top_level}")
    version = payload.get("version", 1)
    if isinstance(version, bool) or version != 1:
        raise ValueError("primitive config version must be 1")
    sections = payload.get("primitives")
    if not isinstance(sections, Mapping):
        raise TypeError("primitive config must contain a primitives mapping")
    unknown_primitives = sorted(set(sections) - set(PRIMITIVE_NAMES))
    if unknown_primitives:
        raise ValueError(f"unknown primitives: {unknown_primitives}")
    if require_all:
        missing = sorted(set(PRIMITIVE_NAMES) - set(sections))
        if missing:
            raise ValueError(f"primitive config is missing: {missing}")

    normalized: dict[str, dict[str, Any]] = {}
    for name, raw_section in sections.items():
        if raw_section is None:
            raw_section = {}
        if not isinstance(raw_section, Mapping):
            raise TypeError(f"primitives.{name} must be a mapping")
        allowed_section_keys = {"defaults", "exposed"}
        if name in _SETTING_PRIMITIVES:
            allowed_section_keys.add("settings")
        unknown_section_keys = sorted(set(raw_section) - allowed_section_keys)
        if unknown_section_keys:
            raise ValueError(
                f"unknown primitives.{name} keys: {unknown_section_keys}"
            )
        raw_defaults = raw_section.get("defaults") or {}
        if not isinstance(raw_defaults, Mapping):
            raise TypeError(f"primitives.{name}.defaults must be a mapping")
        unknown_defaults = sorted(
            set(raw_defaults) - set(_ALLOWED_DEFAULTS[name])
        )
        if unknown_defaults:
            raise ValueError(
                f"unknown defaults for {name}: {unknown_defaults}"
            )
        defaults = dict(raw_defaults)
        _validate_defaults(name, defaults)
        exposed = raw_section.get("exposed", True)
        if type(exposed) is not bool:
            raise TypeError(f"primitives.{name}.exposed must be boolean")
        section: dict[str, Any] = {
            "defaults": defaults,
            "exposed": exposed,
        }
        if name in _SETTING_PRIMITIVES:
            raw_settings = raw_section.get("settings") or {}
            if not isinstance(raw_settings, Mapping):
                raise TypeError(
                    f"primitives.{name}.settings must be a mapping"
                )
            settings = dict(raw_settings)
            if name in {"get_object_pose", "sample_grasp_pose"}:
                unknown_settings = sorted(set(settings) - {"depth_retry_count"})
                if unknown_settings:
                    raise ValueError(
                        f"unknown settings for {name}: {unknown_settings}"
                    )
                depth_retry_count = settings.get("depth_retry_count", 0)
                if (
                    isinstance(depth_retry_count, bool)
                    or not isinstance(depth_retry_count, int)
                    or not 0 <= depth_retry_count <= 5
                ):
                    raise TypeError(
                        f"primitives.{name}.settings.depth_retry_count must be "
                        "an integer in [0, 5]"
                    )
                settings["depth_retry_count"] = depth_retry_count
            elif name in {"open_gripper", "close_gripper"}:
                unknown_settings = sorted(set(settings) - {"simulated"})
                if unknown_settings:
                    raise ValueError(
                        f"unknown settings for {name}: {unknown_settings}"
                    )
                simulated = settings.get("simulated", False)
                if type(simulated) is not bool:
                    raise TypeError(
                        f"primitives.{name}.settings.simulated must be boolean"
                    )
                settings["simulated"] = simulated
            elif name == "dock_to_visible_object":
                backend = str(settings.get("backend", "legacy")).strip().lower()
                if backend == "nav2":
                    from .robot.nav2_visible_object_navigation import (
                        Nav2VisibleObjectDockingConfig,
                    )

                    Nav2VisibleObjectDockingConfig.from_mapping(settings)
                elif backend == "legacy":
                    from .robot.visible_object_navigation import (
                        VisibleObjectDockingConfig,
                    )

                    legacy_settings = dict(settings)
                    legacy_settings.pop("backend", None)
                    VisibleObjectDockingConfig.from_mapping(legacy_settings)
                else:
                    raise ValueError(
                        "dock_to_visible_object backend must be 'nav2' or "
                        f"'legacy', got {backend!r}"
                    )
                readiness_prior = settings.get("readiness_prior")
                if readiness_prior is not None:
                    # Both backend configs pop this block; validate it here so
                    # a typo fails at config load rather than at the first
                    # dock. Imported lazily: the module is only needed when
                    # the block exists.
                    if not isinstance(readiness_prior, Mapping):
                        raise TypeError(
                            "primitives.dock_to_visible_object.settings"
                            ".readiness_prior must be a mapping"
                        )
                    from .robot.readiness_prior import ReadinessPriorConfig

                    ReadinessPriorConfig.from_mapping(readiness_prior)
            else:
                from .robot.manipulation_readiness import (
                    ManipulationReadinessConfig,
                )

                ManipulationReadinessConfig.from_mapping(settings)
            section["settings"] = settings
        normalized[name] = section
    return {"version": 1, "primitives": normalized}


def primitive_defaults(config: Mapping[str, Any], name: str) -> dict[str, Any]:
    """Return a detached defaults mapping for one primitive."""

    section = config.get("primitives", {}).get(name, {})
    return dict(section.get("defaults") or {})


def primitive_exposed(config: Mapping[str, Any], name: str) -> bool:
    """Return whether a configured primitive is visible to generated policies."""

    section = config.get("primitives", {}).get(name, {})
    return bool(section.get("exposed", True))


def primitive_settings(
    config: Mapping[str, Any], name: str
) -> dict[str, Any] | None:
    """Return settings for one primitive, or ``None`` when not configured."""

    section = config.get("primitives", {}).get(name, {})
    if "settings" not in section:
        return None
    return dict(section.get("settings") or {})


def apply_primitive_overrides(
    config: Mapping[str, Any], overrides: Mapping[str, Any] | None
) -> dict[str, Any]:
    """Return ``config`` with a few primitive sections overridden, validated.

    A task configuration changes some primitives of the complete primitive
    configuration it loads without copying it: ``exposed`` replaces the flag,
    and ``defaults`` or ``settings`` replace the keys they name.
    """

    if overrides is None:
        return normalize_primitive_config(config)
    if not isinstance(overrides, Mapping):
        raise TypeError("primitive_overrides must map primitive names to sections")
    merged = {
        name: dict(section or {})
        for name, section in (config.get("primitives") or {}).items()
    }
    for name, raw_section in overrides.items():
        if raw_section is None:
            raw_section = {}
        if not isinstance(raw_section, Mapping):
            raise TypeError(f"primitive_overrides.{name} must be a mapping")
        section = dict(merged.get(name) or {})
        for key, value in raw_section.items():
            if key in {"defaults", "settings"}:
                if not isinstance(value, Mapping):
                    raise TypeError(f"primitive_overrides.{name}.{key} must be a mapping")
                section[key] = {**dict(section.get(key) or {}), **dict(value)}
            else:
                section[key] = value
        merged[name] = section
    return normalize_primitive_config(
        {"version": config.get("version", 1), "primitives": merged}
    )


def _validate_defaults(name: str, defaults: dict[str, Any]) -> None:
    def optional_finite(key: str) -> float | None:
        value = defaults.get(key)
        if value is None:
            return None
        if isinstance(value, bool):
            raise TypeError(f"primitives.{name}.defaults.{key} must be numeric")
        value = float(value)
        if not math.isfinite(value):
            raise ValueError(f"primitives.{name}.defaults.{key} must be finite")
        defaults[key] = value
        return value

    if (
        "return_bbox_extent" in defaults
        and type(defaults["return_bbox_extent"]) is not bool
    ):
        raise TypeError(
            "primitives.get_object_pose.defaults.return_bbox_extent must be boolean"
        )
    if (
        "return_zed_distance" in defaults
        and type(defaults["return_zed_distance"]) is not bool
    ):
        raise TypeError(
            "primitives.get_object_pose.defaults.return_zed_distance must be boolean"
        )
    for key in ("max_yaw_deg_s", "max_speed_mps", "timeout_s", "force_n"):
        if key not in defaults:
            continue
        value = optional_finite(key)
        if value is None:
            continue
        if key in {"max_yaw_deg_s", "max_speed_mps", "timeout_s"} and value <= 0:
            raise ValueError(f"primitives.{name}.defaults.{key} must be positive")
        if (
            key == "timeout_s"
            and name in _MOTION_TIMEOUT_PRIMITIVES
            and value > 45
        ):
            raise ValueError(f"primitives.{name}.defaults.timeout_s must be <= 45")
        if key == "force_n" and not 0.1 <= value <= 3.0:
            raise ValueError(f"primitives.{name}.defaults.force_n must be in [0.1, 3.0]")
    if "z_approach" in defaults:
        value = optional_finite("z_approach")
        if value is None or not 0.0 <= value <= 0.25:
            raise ValueError(
                f"primitives.{name}.defaults.z_approach must be in [0, 0.25]"
            )
    if "approach_m" in defaults:
        value = optional_finite("approach_m")
        if value is None or not 0.02 <= value <= 0.25:
            raise ValueError(
                f"primitives.{name}.defaults.approach_m must be in [0.02, 0.25]"
            )
    if "lift_m" in defaults:
        value = optional_finite("lift_m")
        if value is None or not 0.02 <= value <= 0.25:
            raise ValueError(
                f"primitives.{name}.defaults.lift_m must be in [0.02, 0.25]"
            )
