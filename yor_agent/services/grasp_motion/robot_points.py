"""Drop the robot's own depth returns from a planning cloud (curobo-free).

The cuRobo service plans against a mesh built from the ZED depth cloud, so
every return from the robot's own arm that survives into that cloud becomes an
obstacle the arm is already touching, and the planner reports the start state
in collision before it has searched anything.

The collision-sphere model this robot plans with is a thin surface shell, not
a volumetric fill: after the 2026-09-04 asset change the median sphere radius
is 1.5-11 mm and 657 of the 680 spheres are 11.3 mm or smaller. The previous
rule removed only points strictly inside each sphere after eroding it by
12 mm, which left almost every sphere at its 2 mm floor and removed 0-2 of the
arm's ~10,000 returns while the arm filled the camera view. Every
goto_pose('home') after a grasp then failed with "Start or End state in
collision" (2026-09-05, with and without an object in the gripper).

So the test is now the other way round: a point belongs to the robot when it
lies within the sphere radius plus a margin that covers depth noise and
hand-eye calibration error. A real obstacle that close to the arm's surface
would be inside cuRobo's own activation distance and defeat the plan anyway.
"""

from __future__ import annotations

import numpy as np

# Grown around every collision sphere. ZED neural depth noise at 0.5-1 m is
# ~5-10 mm and the hand-eye calibration's held-out residual is ~3 mm at the
# flange, larger along the arm.
ROBOT_POINT_MARGIN_M = 0.015


def remove_points_near_spheres(
    points_xyz: np.ndarray,
    spheres_xyzr: np.ndarray,
    *,
    margin_m: float = ROBOT_POINT_MARGIN_M,
) -> tuple[np.ndarray, int]:
    """Return ``(kept_points, removed_count)``.

    ``spheres_xyzr`` is ``(N, 4)`` of centre and radius in the same frame as
    ``points_xyz``. A point is removed when it lies within ``radius +
    margin_m`` of any sphere centre. Negative radii are treated as zero.
    """

    points = np.asarray(points_xyz, dtype=np.float32).reshape(-1, 3)
    spheres = np.asarray(spheres_xyzr, dtype=np.float32).reshape(-1, 4)
    margin = max(0.0, float(margin_m))
    keep = np.ones(len(points), dtype=bool)
    if not len(points) or not len(spheres):
        return points, 0
    for center_x, center_y, center_z, radius in spheres:
        reach = max(0.0, float(radius)) + margin
        delta = points - np.asarray((center_x, center_y, center_z), dtype=np.float32)
        keep &= np.einsum("ij,ij->i", delta, delta) > reach * reach
    return points[keep], int(np.count_nonzero(~keep))
