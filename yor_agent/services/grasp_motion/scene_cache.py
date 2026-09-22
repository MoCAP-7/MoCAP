"""Keep cuRoboV2's name-keyed Warp mesh cache from serving a stale scene.

cuRoboV2 stores every mesh obstacle in a Warp cache keyed by ``Mesh.name``
(``curobo/_src/geom/data/data_mesh.py``, ``MeshData._load_mesh_into_cache``).
Loading a new scene through ``MotionPlanner.update_world`` clears the active
obstacle slots but keeps that cache, and a mesh whose name is already cached is
silently reused with its **old geometry**. ``MotionPlanner.clear_scene_cache``
has the same limitation. A long-lived planning service that reuses one mesh
name therefore plans every later request against the first observation.

This module is intentionally free of ``curobo``/``torch`` imports so the
eviction logic can be unit-tested on a development machine.
"""

from __future__ import annotations

from typing import Any


OBSERVED_SCENE_PREFIX = "observed_scene"


class SceneCacheError(RuntimeError):
    """The planner's mesh cache could not be reached or refused to refresh."""


def observed_scene_name(refresh_index: int) -> str:
    """Return a mesh name that is unique per world refresh."""

    if refresh_index < 0:
        raise ValueError("refresh_index must be non-negative")
    return f"{OBSERVED_SCENE_PREFIX}_{int(refresh_index):06d}"


def mesh_cache_of(planner: Any) -> Any:
    """Resolve the ``MeshData`` instance behind a cuRoboV2 ``MotionPlanner``.

    The path is ``planner.scene_collision_checker.scene_model.data.meshes``
    (``RobotSceneCollision`` -> ``SceneCollision`` -> ``SceneData`` ->
    ``MeshData``). Any missing link means the planner was built without a mesh
    world, which this service never does; fail loudly instead of planning
    against an unknown world.
    """

    candidate_paths = (
        # MotionPlanner.scene_collision_checker is a SceneCollision, which
        # owns SceneData directly (curobo/_src/geom/collision/collision_scene.py).
        ("scene_collision_checker", "data", "meshes"),
        # RobotSceneCollision wraps a SceneCollision as ``scene_model``.
        ("scene_collision_checker", "scene_model", "data", "meshes"),
    )
    for path in candidate_paths:
        node = planner
        for attribute in path:
            node = getattr(node, attribute, None)
            if node is None:
                break
        if node is not None and isinstance(getattr(node, "wp_cache", None), dict):
            return node
    raise SceneCacheError(
        "cuRobo planner exposes no MeshData with a dict wp_cache at "
        + " or ".join(".".join(path) for path in candidate_paths)
    )


def evict_cached_meshes(mesh_data: Any) -> list[str]:
    """Drop every cached Warp mesh so the next load rebuilds geometry.

    Uses ``MeshData.clear(env_idx=None, clear_warp_cache=True)`` when the
    installed cuRobo offers it and falls back to clearing the dictionary
    directly. Returns the evicted names for diagnostics.
    """

    evicted = list(mesh_data.wp_cache.keys())
    clear = getattr(mesh_data, "clear", None)
    if callable(clear):
        try:
            clear(env_idx=None, clear_warp_cache=True)
        except TypeError:
            clear(None)
    mesh_data.wp_cache.clear()
    return evicted


def verify_active_meshes(
    mesh_data: Any, expected_names: list[str], *, env_idx: int = 0
) -> dict[str, int]:
    """Confirm that the freshly loaded meshes are exactly what the world uses.

    Returns ``{name: warp_mesh_id}``. Raises ``SceneCacheError`` when the
    active mesh names differ from ``expected_names`` (order included) or the
    cache holds anything else, both of which mean a stale world.
    """

    expected = [str(name) for name in expected_names]
    if not expected:
        raise SceneCacheError("at least one fresh mesh name is required")
    names = getattr(mesh_data, "names", None)
    if names is None:
        raise SceneCacheError("cuRobo MeshData has no per-environment names")
    try:
        active = [name for name in names[env_idx] if name is not None]
    except (TypeError, IndexError, KeyError) as exc:
        raise SceneCacheError("cuRobo MeshData has no per-environment names") from exc
    if active != expected:
        raise SceneCacheError(
            f"active scene meshes {active!r} do not match the fresh meshes {expected!r}"
        )
    cached = list(mesh_data.wp_cache.keys())
    if sorted(cached) != sorted(expected):
        raise SceneCacheError(
            f"Warp mesh cache {cached!r} still holds meshes other than {expected!r}"
        )
    ids: dict[str, int] = {}
    for name in expected:
        mesh_id = getattr(mesh_data.wp_cache[name], "mesh_id", None)
        if mesh_id is None:
            raise SceneCacheError("cached Warp mesh entry has no mesh_id")
        ids[name] = int(mesh_id)
    return ids


def verify_active_mesh(mesh_data: Any, expected_name: str, *, env_idx: int = 0) -> int:
    """Single-mesh form of ``verify_active_meshes``."""

    return verify_active_meshes(mesh_data, [expected_name], env_idx=env_idx)[expected_name]
