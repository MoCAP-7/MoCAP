"""Bound cuRoboV2's sphere-mesh distance query radius to ``MeshData.max_dist``.

Why: ``curobo/_src/geom/data/data_mesh.py`` (upstream ``8e734f3``) computes
``compute_local_sdf`` / ``compute_local_sdf_with_grad`` with

    max_distance = wp.length(bounding_box_size) * 0.5

i.e. half the mesh bounding-box diagonal, "so that even very interior points
can be accounted for". ``MeshData.max_dist`` (0.1 m, ``SceneCollisionCfg.
max_distance``) is carried in the Warp struct but never used. For an observed
table scene (1.0 x 0.65 x 0.5 m) every robot sphere therefore searches the BVH
out to ~0.65 m on every optimizer step. Jetson planning-only benchmark
(2026-09-04, 12.7k-point scene): almost empty world 7 s per plan, per-voxel
scene mesh 63-70 s, 188-triangle merged mesh still 41 s.

The collision kernel only needs distances up to ``radius_adjusted`` (sphere
radius plus activation distance): a failed query returns ``max_distance``, and
``penetration = radius_adjusted - max_distance`` is then non-positive, i.e.
free. Clamping the radius to ``max(min(bbox_half_diagonal, max_dist),
query_distance)`` keeps every outside-the-surface result identical and only
changes points deeper than ``max_dist`` *inside* a closed mesh, which this
service's thin observed-voxel shells do not produce.

Usage (Jetson, before restarting the curobo service):

    .venv-grasp-motion/bin/python services/grasp_motion/patch_curobo_mesh_query.py
    .venv-grasp-motion/bin/python services/grasp_motion/patch_curobo_mesh_query.py --revert

Idempotent; verifies the expected source before and after editing. Warp
recompiles the kernel from the edited Python source at the next start.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ORIGINAL = "    max_distance = wp.length(bounding_box_size) * 0.5\n"
PATCHED = (
    "    max_distance = wp.min(wp.length(bounding_box_size) * 0.5, obs_set.max_dist)"
    "  # yor: bounded mesh query radius\n"
)
EXPECTED_SITES = 2


def default_target() -> Path:
    try:
        import curobo  # type: ignore
    except ImportError as exc:  # pragma: no cover - only on hosts without cuRobo
        raise SystemExit("curobo is not importable; pass --file explicitly") from exc
    return Path(curobo.__file__).resolve().parent / "_src" / "geom" / "data" / "data_mesh.py"


def apply_patch(source: str, *, revert: bool = False) -> tuple[str, str]:
    """Return ``(new_source, status)`` where status is applied/reverted/unchanged."""

    if revert:
        if source.count(PATCHED) == 0:
            return source, "unchanged"
        return source.replace(PATCHED, ORIGINAL), "reverted"
    if source.count(PATCHED) == EXPECTED_SITES:
        return source, "unchanged"
    if source.count(ORIGINAL) != EXPECTED_SITES:
        raise RuntimeError(
            f"expected {EXPECTED_SITES} occurrences of the bounding-box max_distance "
            f"line, found {source.count(ORIGINAL)}; cuRobo source differs from the "
            "version this patch was written for (upstream 8e734f3)"
        )
    if "s.max_dist = max_dist" not in source:
        raise RuntimeError("MeshDataWarp.max_dist is not populated in this cuRobo version")
    return source.replace(ORIGINAL, PATCHED), "applied"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--file", type=Path, default=None, help="path to data_mesh.py")
    parser.add_argument("--revert", action="store_true")
    parser.add_argument("--check", action="store_true", help="report status without writing")
    args = parser.parse_args(argv)
    target = args.file or default_target()
    source = target.read_text(encoding="utf-8")
    new_source, status = apply_patch(source, revert=args.revert)
    if args.check:
        print(f"{target}: {'patched' if source.count(PATCHED) == EXPECTED_SITES else 'original'}")
        return 0
    if status != "unchanged":
        target.write_text(new_source, encoding="utf-8")
    print(f"{target}: {status}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
