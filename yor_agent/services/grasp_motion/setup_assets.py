"""Build the cuRobo Nero model from AgileX's official collision assets.

The source checkout is deliberately kept out of git.  This script composes the
three official Nero descriptions into a plain URDF, adds the grasp-volume TCP
used by the Pi service, fits collision spheres, and writes a cuRobo robot YAML.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

import numpy as np
from scipy.ndimage import (
    binary_closing,
    binary_fill_holes,
    distance_transform_edt,
    maximum_filter,
)
import trimesh
import yaml


OFFICIAL_REPOSITORY = "https://github.com/agilexrobotics/agx_arm_urdf.git"
OFFICIAL_COMMIT = "f6642ce0d7872c686f29c99e9e10cd23d1d49313"
# Center of the official piper_hand/Nero open grasp volume. The volume spans
# local Z=[0.0775, 0.1325] m, so this frame lies midway between the fingers in
# both the closing direction and useful insertion depth instead of at the tip.
TCP_FROM_GRIPPER_BASE_Z_M = 0.105
ARM_CORE_VOXEL_PITCH_M = 0.002
ARM_CORE_RADIUS_SHRINK_M = 0.0025
ARM_CORE_SURFACE_SAMPLES = 8000
ARM_CORE_SPHERE_BUDGETS = {
    "base_link": 24,
    "link1": 12,
    "link2": 24,
    "link3": 16,
    "link4": 16,
    "link5": 12,
    "link6": 12,
    "link7": 8,
}
GRIPPER_LINK_NAMES = (
    "gripper_flange",
    "gripper_base",
    "gripper_link1",
    "gripper_link2",
)
# The sparse fitter is adequate for the arm but made the small gripper links
# very conservative after their few spheres were expanded to cover the whole
# mesh. Use a dense surface shell instead. Voxel samples are projected back to
# the exact official mesh and use half a surface-grid diagonal as their radius,
# so the envelope protrudes by at most ~5.7 mm and there is no additional
# global padding at runtime.
# Defaults chosen from the 2026-09-04 Jetson benchmark: the flange and gripper
# base use a 16 mm shell (11.3 mm protrusion, far from any grasp target) while
# the fingers keep the 8 mm shell (5.7 mm) that bounds table-top clearance.
# Total spheres 1,142 -> 680; table-scene plans 27-44 s -> 4-13 s together
# with finetune_attempts 0. Pass --gripper-surface-pitch-m 0.008 for the
# original shell.
GRIPPER_SURFACE_PITCH_M = 0.016
FINGER_SURFACE_PITCH_M = 0.008
ATTACHED_OBJECT_SPHERE_SLOTS = 64


def gripper_surface_radius_m(pitch_m: float) -> float:
    """Half a surface-grid diagonal: the shell protrudes by at most this much."""

    return float(np.sqrt(2.0) * pitch_m / 2.0)


# cuRobo's collision cost scales with the sphere count and its self-collision
# cost with the pair count. The 8 mm shell gave 1,018 gripper spheres out of
# 1,142 (2026-09-04) and dominated planning time on the Jetson, so the pitch
# is a CLI option: 12 mm -> ~8.5 mm protrusion and ~2x fewer spheres, 16 mm
# -> ~11 mm and ~4x fewer. The service accepts radii up to 15 mm.
GRIPPER_SURFACE_PITCH_MAX_M = 0.020


def _fit_arm_core_spheres(
    mesh: trimesh.Trimesh, link_name: str
) -> list[dict[str, object]]:
    """Fit a small, deliberately under-covering medial sphere set.

    Nero's official meshes include hollow shells and a few non-watertight
    seams.  A raw 3-D interior distance transform therefore follows material
    thickness instead of the external link body.  Fill each section normal to
    the link's longest axis first, then compute medial candidates in that solid
    external envelope.  The radius shrink absorbs the voxel discretization
    error and keeps the result on the core side of the physical surface.

    Candidate selection minimizes surface gap, with extra weight on the worst
    10 percent of the mesh.  This prevents a volume-only greedy fit from using
    every sphere around a joint housing while leaving a thin link end empty.
    """

    budget = ARM_CORE_SPHERE_BUDGETS[link_name]
    voxels = mesh.voxelized(ARM_CORE_VOXEL_PITCH_M).fill()
    occupied = np.asarray(voxels.matrix, dtype=bool)
    longest_axis = int(np.argmax(mesh.extents))
    envelope = occupied.copy()
    for index in range(occupied.shape[longest_axis]):
        section = np.take(occupied, index, axis=longest_axis)
        closed = binary_closing(section, structure=np.ones((3, 3), dtype=bool))
        filled = binary_fill_holes(closed)
        target = [slice(None)] * 3
        target[longest_axis] = index
        envelope[tuple(target)] |= filled

    padded = np.pad(envelope, 1, constant_values=False)
    clearance = (
        distance_transform_edt(padded)[1:-1, 1:-1, 1:-1]
        * ARM_CORE_VOXEL_PITCH_M
    )
    radii_grid = np.maximum(clearance - ARM_CORE_RADIUS_SHRINK_M, 0.0)
    maxima = envelope & (
        radii_grid >= maximum_filter(radii_grid, size=3, mode="constant")
    )
    maxima &= radii_grid >= 0.001
    indices = np.argwhere(maxima)
    centers = np.asarray(voxels.indices_to_points(indices), dtype=np.float64)
    radii = np.asarray(radii_grid[maxima], dtype=np.float64)
    if len(centers) < budget:
        raise RuntimeError(
            f"arm core fit for {link_name!r} found only {len(centers)} "
            f"candidates for a {budget}-sphere budget"
        )

    surface, _ = trimesh.sample.sample_surface(mesh, ARM_CORE_SURFACE_SAMPLES)
    candidate_gap = np.empty((len(centers), len(surface)), dtype=np.float32)
    for start in range(0, len(centers), 256):
        block = centers[start : start + 256]
        candidate_gap[start : start + len(block)] = np.maximum(
            np.linalg.norm(block[:, None, :] - surface[None, :, :], axis=2)
            - radii[start : start + len(block), None],
            0.0,
        )

    current_gap = np.full(len(surface), 0.100, dtype=np.float32)
    selected: list[int] = []
    for _ in range(budget):
        worst_decile = current_gap >= np.quantile(current_gap, 0.90)
        weights = 1.0 + 4.0 * worst_decile
        improvement = np.maximum(current_gap[None, :] - candidate_gap, 0.0)
        gain = improvement @ weights
        if selected:
            gain[selected] = -1.0
        best = int(np.argmax(gain))
        selected.append(best)
        current_gap = np.minimum(current_gap, candidate_gap[best])

    p95_gap = float(np.quantile(current_gap, 0.95))
    max_gap = float(np.max(current_gap))
    if p95_gap > 0.018 or max_gap > 0.035:
        raise RuntimeError(
            f"arm core fit for {link_name!r} is too sparse: "
            f"p95 gap={p95_gap * 1000:.1f} mm, max={max_gap * 1000:.1f} mm"
        )
    return [
        {"center": centers[index].tolist(), "radius": float(radii[index])}
        for index in selected
    ]


def _source_paths(root: Path) -> tuple[Path, Path, Path]:
    urdf = root / "nero" / "urdf"
    return (
        urdf / "nero_description.urdf",
        urdf / "nero_with_gripper_flange_description.xacro",
        urdf / "nero_with_gripper_description.xacro",
    )


def _validate_official_checkout(root: Path) -> None:
    missing = [str(path) for path in _source_paths(root) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "official AgileX Nero description is incomplete: " + ", ".join(missing)
        )
    completed = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    actual = completed.stdout.strip()
    if actual != OFFICIAL_COMMIT:
        raise RuntimeError(
            f"AgileX description must be pinned to {OFFICIAL_COMMIT}, got {actual}"
        )


def _rewrite_mesh_paths(element: ET.Element, checkout: Path) -> None:
    package_prefix = "package://agx_arm_description/agx_arm_urdf/"
    for mesh in element.iter("mesh"):
        filename = mesh.get("filename")
        if filename and filename.startswith(package_prefix):
            resolved = checkout / filename[len(package_prefix) :]
            if not resolved.is_file():
                raise FileNotFoundError(f"missing official mesh {resolved}")
            mesh.set("filename", str(resolved.resolve()))


# Joint limits that replace the official description's. The elbow's
# mechanical stop was measured at 2.20 rad by bending the arm by hand
# (2026-09-13); the official 2.14 would keep cuRobo from planning to or from
# the rest pose, which bends the elbow past it. Keep in step with
# NERO_JOINT_POSITION_LIMITS_RAD in services/nero_arm/service.py.
JOINT_LIMIT_OVERRIDES_RAD = {"joint4": (-1.01, 2.19)}


def _apply_joint_limit_overrides(root: ET.Element) -> None:
    for joint in root.iter("joint"):
        limits = JOINT_LIMIT_OVERRIDES_RAD.get(joint.get("name", ""))
        if limits is None:
            continue
        limit = joint.find("limit")
        if limit is None:
            raise ValueError(f"official joint {joint.get('name')!r} has no limit element")
        limit.set("lower", f"{limits[0]:g}")
        limit.set("upper", f"{limits[1]:g}")


def compose_urdf(checkout: Path, output_path: Path) -> None:
    """Compose the official includes without requiring a ROS installation."""

    source_paths = _source_paths(checkout)
    base_root = ET.parse(source_paths[0]).getroot()
    for include_path in source_paths[1:]:
        included_root = ET.parse(include_path).getroot()
        for child in included_root:
            if child.tag.endswith("include"):
                continue
            base_root.append(copy.deepcopy(child))
    _apply_joint_limit_overrides(base_root)

    tcp_link = ET.Element("link", {"name": "grasp_tcp"})
    tcp_joint = ET.Element("joint", {"name": "grasp_tcp_joint", "type": "fixed"})
    ET.SubElement(tcp_joint, "origin", {
        "xyz": f"0 0 {TCP_FROM_GRIPPER_BASE_Z_M:.10f}",
        "rpy": "0 0 0",
    })
    ET.SubElement(tcp_joint, "parent", {"link": "gripper_base"})
    ET.SubElement(tcp_joint, "child", {"link": "grasp_tcp"})
    base_root.extend((tcp_link, tcp_joint))
    _rewrite_mesh_paths(base_root, checkout)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(base_root, space="  ")
    ET.ElementTree(base_root).write(
        output_path, encoding="utf-8", xml_declaration=True
    )


def _neighbor_collision_ignores(urdf_path: Path) -> dict[str, list[str]]:
    """Return bidirectional parent-child ignores without CUDA parser helpers."""

    ignores: dict[str, list[str]] = {}
    for joint in ET.parse(urdf_path).getroot().findall("joint"):
        parent = joint.find("parent")
        child = joint.find("child")
        if parent is None or child is None:
            continue
        parent_name = parent.get("link")
        child_name = child.get("link")
        if not parent_name or not child_name:
            continue
        ignores.setdefault(parent_name, []).append(child_name)
        ignores.setdefault(child_name, []).append(parent_name)
    return ignores


FINGER_LINK_NAMES = ("gripper_link1", "gripper_link2")


def build_curobo_config(
    urdf_path: Path,
    output_path: Path,
    *,
    gripper_surface_pitch_m: float = GRIPPER_SURFACE_PITCH_M,
    finger_surface_pitch_m: float = FINGER_SURFACE_PITCH_M,
) -> None:
    """Fit bounded link spheres and create a seven-DOF planner config.

    ``finger_surface_pitch_m`` lets the fingers keep a fine shell, which sets
    how close to a table top they may plan, while the flange and gripper base
    use a coarser, cheaper one.
    """

    for name, pitch in (
        ("gripper_surface_pitch_m", gripper_surface_pitch_m),
        ("finger_surface_pitch_m", finger_surface_pitch_m),
    ):
        if not 0.004 <= pitch <= GRIPPER_SURFACE_PITCH_MAX_M:
            raise ValueError(f"{name} must be in [0.004, {GRIPPER_SURFACE_PITCH_MAX_M}]")

    import curobo._src.types.pose as pose_module
    from curobo._src.types.device_cfg import DeviceCfg
    from curobo.robot_builder import RobotBuilder
    import torch

    cpu_device_cfg = DeviceCfg(device=torch.device("cpu"))
    # UrdfRobotParser currently calls Pose.from_matrix without forwarding the
    # builder's DeviceCfg. Keep this offline generator CPU-only even on a host
    # where CUDA is unavailable; the generated runtime model is device-neutral.
    pose_module.DeviceCfg = lambda: cpu_device_cfg
    builder = RobotBuilder(
        str(urdf_path),
        asset_path=str(urdf_path.parent),
        tool_frames=["grasp_tcp"],
        device_cfg=cpu_device_cfg,
    )
    np.random.seed(123)
    # Build every collision link directly. RobotBuilder's generic fitter does
    # not currently forward its DeviceCfg and would initialize CUDA even though
    # both Nero fits below are deliberately CPU-only.
    builder._collision_spheres = {}
    for link_name in builder._mesh_link_names:
        geometries = builder._parser.get_link_geometry(
            link_name, use_collision_mesh=True
        )
        meshes = [
            geometry.get_trimesh_mesh(transform_with_pose=True)
            for geometry in geometries
        ]
        meshes = [mesh for mesh in meshes if mesh is not None]
        if not meshes:
            continue
        mesh = meshes[0] if len(meshes) == 1 else trimesh.util.concatenate(meshes)
        if link_name in GRIPPER_LINK_NAMES:
            link_pitch_m = (
                finger_surface_pitch_m
                if link_name in FINGER_LINK_NAMES
                else gripper_surface_pitch_m
            )
            surface_voxels = np.asarray(
                mesh.voxelized(link_pitch_m).points,
                dtype=np.float64,
            )
            if not len(surface_voxels):
                raise RuntimeError(
                    f"official gripper mesh {link_name!r} produced no surface voxels"
                )
            surface_centers = trimesh.proximity.closest_point(
                mesh, surface_voxels
            )[0]
            builder._collision_spheres[link_name] = [
                {
                    "center": center.tolist(),
                    "radius": gripper_surface_radius_m(link_pitch_m),
                }
                for center in surface_centers
            ]
            continue
        if link_name in ARM_CORE_SPHERE_BUDGETS:
            builder._collision_spheres[link_name] = _fit_arm_core_spheres(
                mesh, link_name
            )
            continue
        raise RuntimeError(f"no collision-sphere policy for mesh link {link_name!r}")
    builder._cspace_config = {  # cuRobo's builder exposes no setter yet.
        "joint_names": [f"joint{index}" for index in range(1, 8)],
        "default_joint_position": [0.0] * 7,
        "null_space_weight": [1.0] * 7,
        "cspace_distance_weight": [1.0] * 7,
        "max_acceleration": 1.0,
        "max_jerk": 10.0,
    }
    # The builder's sampled-matrix helper currently constructs an unlocked
    # eight-DOF temporary model before we can lock the gripper, which is not a
    # valid Nero planning model.  Keep the always-correct kinematic-neighbour
    # ignores and explicit gripper adjacency here; the runtime planner still
    # checks every non-ignored pair throughout each trajectory.
    builder._self_collision_ignore = _neighbor_collision_ignores(urdf_path)
    builder.add_collision_ignore(
        "gripper_base", ["gripper_link1", "gripper_link2"]
    )
    builder.add_collision_ignore("gripper_link1", ["gripper_link2"])
    # Ignore only known near-neighbour pairs around physical joints;
    # non-neighbour arm links remain checked throughout every trajectory.
    for link_name, neighbours in {
        "base_link": ["link2"],
        "link1": ["link3"],
        "link2": ["link4"],
        "link3": ["link5"],
        "link5": ["link7", "gripper_flange", "gripper_base"],
        "link6": [
            "gripper_flange",
            "gripper_base",
            "gripper_link1",
            "gripper_link2",
        ],
        "link7": ["gripper_base"],
        "gripper_flange": ["gripper_link1", "gripper_link2"],
    }.items():
        builder.add_collision_ignore(link_name, neighbours)
    config = builder.build()
    config.base_link = "base_link"
    config.lock_joints = {"gripper": 0.1}
    # Arm spheres are already shrunk inside their voxel envelope, and gripper
    # radii already include their surface-grid allowance. Any global buffer
    # would undo the bounded fit and enlarge the gripper again.
    config.collision_sphere_buffer = 0.0
    config.grasp_contact_link_names = []
    builder.save(config, str(output_path))

    payload = yaml.safe_load(output_path.read_text(encoding="utf-8"))
    payload["robot_cfg"] = payload.pop("kinematics")
    payload["robot_cfg"] = {"kinematics": payload["robot_cfg"]}
    kinematics = payload["robot_cfg"]["kinematics"]
    # Reserve a fixed, normally disabled link for cuRoboV2's official
    # AttachmentManager.  The slots start with negative radii and are populated
    # only after the gripper has closed successfully around a perceived object.
    # Keeping this as an extra link avoids inventing collision geometry in the
    # composed AgileX URDF while still making carried-object geometry part of
    # every subsequent world/self-collision query.
    kinematics["extra_collision_spheres"] = {
        "attached_object": ATTACHED_OBJECT_SPHERE_SLOTS
    }
    kinematics["extra_links"] = {
        "attached_object": {
            "fixed_transform": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            "joint_name": "attached_object_joint",
            "joint_type": "FIXED",
            "link_name": "attached_object",
            "parent_link_name": "grasp_tcp",
        }
    }
    collision_links = kinematics.setdefault("collision_link_names", [])
    if "attached_object" not in collision_links:
        collision_links.append("attached_object")
    # The observed target mask is excluded from the cuRobo obstacle world, so
    # intended finger/object contact does not require disabling collision links.
    # Keep this empty as a safe default in addition to plan_grasp explicitly
    # passing disable_collision_links=[] at runtime.
    kinematics["grasp_contact_link_names"] = []
    self_collision_ignore = kinematics.setdefault("self_collision_ignore", {})
    for link_name in (
        "link7",
        "gripper_flange",
        "gripper_base",
        "gripper_link1",
        "gripper_link2",
        "grasp_tcp",
    ):
        ignored = self_collision_ignore.setdefault(link_name, [])
        if "attached_object" not in ignored:
            ignored.append("attached_object")
    kinematics.setdefault("self_collision_buffer", {})["attached_object"] = 0.0
    output_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--gripper-surface-pitch-m",
        type=float,
        default=GRIPPER_SURFACE_PITCH_M,
        help=(
            "surface-grid pitch of the flange and gripper-base collision "
            "shell; the shell protrudes by at most pitch*sqrt(2)/2 (0.008 -> "
            "5.7 mm and 654 spheres on those links; 0.016 -> 11.3 mm and 192)"
        ),
    )
    parser.add_argument(
        "--finger-surface-pitch-m",
        type=float,
        default=FINGER_SURFACE_PITCH_M,
        help=(
            "surface-grid pitch of the two finger links only; 0.008 keeps "
            "today's table-top clearance while the flange and gripper base "
            "use --gripper-surface-pitch-m"
        ),
    )
    args = parser.parse_args()

    checkout = args.official_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    _validate_official_checkout(checkout)
    output_dir.mkdir(parents=True, exist_ok=True)
    urdf_path = output_dir / "nero_with_official_gripper.urdf"
    compose_urdf(checkout, urdf_path)
    build_curobo_config(
        urdf_path,
        output_dir / "nero_with_official_gripper.yml",
        gripper_surface_pitch_m=args.gripper_surface_pitch_m,
        finger_surface_pitch_m=args.finger_surface_pitch_m,
    )


if __name__ == "__main__":
    main()
