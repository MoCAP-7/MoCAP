"""Conservative static-obstacle planning on a persistent 2-D occupancy map.

The map is deliberately small and dependency-free.  RGB-D rays mark observed
space free, structural endpoints mark occupied cells, and all unobserved cells
remain unavailable to the planner.  A* therefore cannot invent a route through
space that the forward-facing camera has not measured.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
from typing import Iterable


GridCell = tuple[int, int]
PointXY = tuple[float, float]


@dataclass(frozen=True)
class OccupancyPlan:
    """Result of planning from the robot to any safe docking-ring cell."""

    success: bool
    reason: str
    path_xy: tuple[PointXY, ...] = ()
    expanded_cells: int = 0
    known_free_cells: int = 0
    occupied_cells: int = 0
    inflated_cells: int = 0

    def metrics(self) -> dict[str, object]:
        return {
            "success": self.success,
            "reason": self.reason,
            "path_xy": [list(point) for point in self.path_xy],
            "path_points": len(self.path_xy),
            "expanded_cells": self.expanded_cells,
            "known_free_cells": self.known_free_cells,
            "occupied_cells": self.occupied_cells,
            "inflated_cells": self.inflated_cells,
        }


class StaticOccupancyMap:
    """Persistent world-frame occupancy assembled from stopped RGB-D views."""

    _NEIGHBORS: tuple[tuple[int, int, float], ...] = (
        (-1, 0, 1.0),
        (1, 0, 1.0),
        (0, -1, 1.0),
        (0, 1, 1.0),
        (-1, -1, math.sqrt(2.0)),
        (-1, 1, math.sqrt(2.0)),
        (1, -1, math.sqrt(2.0)),
        (1, 1, math.sqrt(2.0)),
    )

    def __init__(self, *, resolution_m: float, inflation_radius_m: float) -> None:
        if not math.isfinite(resolution_m) or resolution_m <= 0.0:
            raise ValueError("resolution_m must be positive and finite")
        if not math.isfinite(inflation_radius_m) or inflation_radius_m < 0.0:
            raise ValueError("inflation_radius_m must be nonnegative and finite")
        self.resolution_m = float(resolution_m)
        self.inflation_radius_m = float(inflation_radius_m)
        self.free_cells: set[GridCell] = set()
        self.occupied_cells: set[GridCell] = set()
        self._inflated_cache: set[GridCell] | None = None

    def cell(self, point_xy: PointXY) -> GridCell:
        return (
            int(round(float(point_xy[0]) / self.resolution_m)),
            int(round(float(point_xy[1]) / self.resolution_m)),
        )

    def point(self, cell: GridCell) -> PointXY:
        return (
            float(cell[0]) * self.resolution_m,
            float(cell[1]) * self.resolution_m,
        )

    def integrate_rays(
        self,
        sensor_origin_xy: PointXY,
        endpoints_xy: Iterable[PointXY],
        occupied_endpoints: Iterable[bool],
    ) -> None:
        """Fuse static rays. Occupancy wins permanently over free evidence."""

        origin = self.cell(sensor_origin_xy)
        self.free_cells.add(origin)
        for endpoint_xy, endpoint_occupied in zip(
            endpoints_xy, occupied_endpoints, strict=True
        ):
            endpoint = self.cell(endpoint_xy)
            ray = self._supercover_line(origin, endpoint)
            for cell in ray[:-1]:
                if cell not in self.occupied_cells:
                    self.free_cells.add(cell)
            if endpoint_occupied:
                if endpoint not in self.occupied_cells:
                    self._inflated_cache = None
                self.occupied_cells.add(endpoint)
                self.free_cells.discard(endpoint)
            elif endpoint not in self.occupied_cells:
                self.free_cells.add(endpoint)

    def plan_to_docking_ring(
        self,
        *,
        start_xy: PointXY,
        target_xy: PointXY,
        docking_distance_m: float,
        maximum_expansions: int,
    ) -> OccupancyPlan:
        """Run A* to any observed-free cell on the target docking boundary."""

        start = self.cell(start_xy)
        inflated = self._inflated_cells()
        traversable = self.free_cells - inflated
        # The robot is known to occupy its current cell.  This exception only
        # releases the start cell, not an unobserved or occupied route around it.
        traversable.add(start)
        ring_tolerance = self.resolution_m * math.sqrt(2.0)
        goals = {
            cell
            for cell in traversable
            if docking_distance_m - ring_tolerance
            <= math.dist(self.point(cell), target_xy)
            <= docking_distance_m
        }
        common = {
            "known_free_cells": len(self.free_cells),
            "occupied_cells": len(self.occupied_cells),
            "inflated_cells": len(inflated),
        }
        if not goals:
            return OccupancyPlan(False, "no_observed_docking_goal", **common)
        if start in goals:
            return OccupancyPlan(
                True, "already_on_docking_ring", (self.point(start),), **common
            )

        queue: list[tuple[float, float, GridCell]] = []
        heapq.heappush(
            queue,
            (
                self._ring_heuristic(start, target_xy, docking_distance_m),
                0.0,
                start,
            ),
        )
        cost: dict[GridCell, float] = {start: 0.0}
        parent: dict[GridCell, GridCell] = {}
        expanded = 0
        reached: GridCell | None = None
        while queue and expanded < maximum_expansions:
            _, current_cost, current = heapq.heappop(queue)
            if current_cost > cost.get(current, math.inf) + 1e-12:
                continue
            expanded += 1
            if current in goals:
                reached = current
                break
            for dx, dy, move_cost in self._NEIGHBORS:
                neighbor = (current[0] + dx, current[1] + dy)
                if neighbor not in traversable:
                    continue
                if dx and dy and (
                    (current[0] + dx, current[1]) not in traversable
                    or (current[0], current[1] + dy) not in traversable
                ):
                    continue
                candidate_cost = current_cost + move_cost
                if candidate_cost >= cost.get(neighbor, math.inf):
                    continue
                cost[neighbor] = candidate_cost
                parent[neighbor] = current
                priority = candidate_cost + self._ring_heuristic(
                    neighbor, target_xy, docking_distance_m
                )
                heapq.heappush(queue, (priority, candidate_cost, neighbor))

        if reached is None:
            reason = "planner_expansion_limit" if queue else "no_safe_path"
            return OccupancyPlan(
                False, reason, expanded_cells=expanded, **common
            )
        cells = [reached]
        while cells[-1] != start:
            cells.append(parent[cells[-1]])
        cells.reverse()
        return OccupancyPlan(
            True,
            "path_found",
            tuple(self.point(cell) for cell in cells),
            expanded_cells=expanded,
            **common,
        )

    def plan_to_lateral_frontier(
        self,
        *,
        start_xy: PointXY,
        target_xy: PointXY,
        minimum_lateral_m: float,
        minimum_target_progress_m: float,
        maximum_expansions: int,
        preferred_side: int | None = None,
    ) -> OccupancyPlan:
        """Plan toward a known-free side frontier when the goal is occluded.

        This never makes unknown cells traversable.  It finds a reachable cell
        that is laterally displaced from the direct target ray and closer to
        the target than the current pose.  Moving one short waypoint along the
        returned path gives a forward camera a new view around an obstacle;
        the caller must stop, integrate that view, and plan again.
        """

        if minimum_lateral_m < 0.0 or not math.isfinite(minimum_lateral_m):
            raise ValueError("minimum_lateral_m must be finite and nonnegative")
        if (
            minimum_target_progress_m < 0.0
            or not math.isfinite(minimum_target_progress_m)
        ):
            raise ValueError(
                "minimum_target_progress_m must be finite and nonnegative"
            )
        if preferred_side not in (None, -1, 1):
            raise ValueError("preferred_side must be None, -1, or 1")

        start = self.cell(start_xy)
        inflated = self._inflated_cells()
        traversable = self.free_cells - inflated
        traversable.add(start)
        common = {
            "known_free_cells": len(self.free_cells),
            "occupied_cells": len(self.occupied_cells),
            "inflated_cells": len(inflated),
        }
        target_dx = float(target_xy[0]) - float(start_xy[0])
        target_dy = float(target_xy[1]) - float(start_xy[1])
        target_distance = math.hypot(target_dx, target_dy)
        if target_distance <= self.resolution_m:
            return OccupancyPlan(False, "frontier_target_too_close", **common)
        forward_x = target_dx / target_distance
        forward_y = target_dy / target_distance
        left_x = -forward_y
        left_y = forward_x

        queue: list[tuple[float, GridCell]] = [(0.0, start)]
        cost: dict[GridCell, float] = {start: 0.0}
        parent: dict[GridCell, GridCell] = {}
        expanded = 0
        best: GridCell | None = None
        best_key: tuple[float, float, float] | None = None
        while queue and expanded < maximum_expansions:
            current_cost, current = heapq.heappop(queue)
            if current_cost > cost.get(current, math.inf) + 1e-12:
                continue
            expanded += 1
            point = self.point(current)
            relative_x = point[0] - float(start_xy[0])
            relative_y = point[1] - float(start_xy[1])
            forward_progress = relative_x * forward_x + relative_y * forward_y
            lateral = relative_x * left_x + relative_y * left_y
            target_progress = target_distance - math.dist(point, target_xy)
            side_matches = (
                preferred_side is None
                or lateral * float(preferred_side) >= minimum_lateral_m
            )
            if (
                current != start
                and forward_progress >= -self.resolution_m
                and abs(lateral) >= minimum_lateral_m
                and target_progress >= minimum_target_progress_m
                and side_matches
            ):
                # Prefer the frontier that makes the most real target progress;
                # use forward projection and shorter path cost as tie-breakers.
                candidate_key = (
                    target_progress,
                    forward_progress,
                    -current_cost,
                )
                if best_key is None or candidate_key > best_key:
                    best = current
                    best_key = candidate_key

            for dx, dy, move_cost in self._NEIGHBORS:
                neighbor = (current[0] + dx, current[1] + dy)
                if neighbor not in traversable:
                    continue
                if dx and dy and (
                    (current[0] + dx, current[1]) not in traversable
                    or (current[0], current[1] + dy) not in traversable
                ):
                    continue
                candidate_cost = current_cost + move_cost
                if candidate_cost >= cost.get(neighbor, math.inf):
                    continue
                cost[neighbor] = candidate_cost
                parent[neighbor] = current
                heapq.heappush(queue, (candidate_cost, neighbor))

        if best is None:
            return OccupancyPlan(
                False,
                "no_reachable_lateral_frontier",
                expanded_cells=expanded,
                **common,
            )
        cells = [best]
        while cells[-1] != start:
            cells.append(parent[cells[-1]])
        cells.reverse()
        return OccupancyPlan(
            True,
            "lateral_frontier_found",
            tuple(self.point(cell) for cell in cells),
            expanded_cells=expanded,
            **common,
        )

    def waypoint(self, path_xy: tuple[PointXY, ...], lookahead_m: float) -> PointXY:
        """Choose the furthest line-of-sight path point within one motion step."""

        if not path_xy:
            raise ValueError("path_xy must not be empty")
        start = path_xy[0]
        inflated = self._inflated_cells()
        traversable = (self.free_cells - inflated) | {self.cell(start)}
        chosen = start
        for point in path_xy[1:]:
            if math.dist(start, point) > lookahead_m + 1e-9:
                break
            if all(
                cell in traversable
                for cell in self._supercover_line(self.cell(start), self.cell(point))
            ):
                chosen = point
        if chosen == start and len(path_xy) > 1:
            chosen = path_xy[1]
        return chosen

    def segment_status(
        self,
        start_xy: PointXY,
        end_xy: PointXY,
        *,
        allow_start_in_inflated: bool = False,
    ) -> dict[str, object]:
        """Check one swept point path against the inflated observed-free map.

        Inflation already represents the complete circular robot footprint, so
        this query must not apply another robot-radius or obstacle dilation.
        When explicitly requested, the current cell may be released so a robot
        that already drifted into the outer inflated grid shell can move out.
        Every subsequent cell remains subject to the full inflated footprint.
        """

        start = self.cell(start_xy)
        end = self.cell(end_xy)
        cells = self._supercover_line(start, end)
        inflated = self._inflated_cells()
        unknown = [
            cell
            for cell in cells
            if cell not in self.free_cells and cell not in inflated
        ]
        start_inflated = start in inflated
        blocked = [
            cell
            for cell in cells
            if cell in inflated
            and not (allow_start_in_inflated and cell == start)
        ]
        clear = not unknown and not blocked
        return {
            "clear": clear,
            "reason": (
                "segment_clear_from_inflated_start"
                if clear and start_inflated and allow_start_in_inflated
                else "segment_clear"
                if clear
                else "inflated_obstacle"
                if blocked
                else "unknown_space"
            ),
            "checked_cells": len(cells),
            "unknown_cells": len(unknown),
            "blocked_cells": len(blocked),
            "start_inflated": start_inflated,
            "allowed_inflated_start": bool(
                start_inflated and allow_start_in_inflated
            ),
            "start_xy": [float(start_xy[0]), float(start_xy[1])],
            "end_xy": [float(end_xy[0]), float(end_xy[1])],
        }

    def _inflated_cells(self) -> set[GridCell]:
        if self._inflated_cache is not None:
            return self._inflated_cache
        # Grid cells represent areas, while A* stores their centers. Include a
        # half-cell diagonal so discretization never rounds a configured robot
        # radius down (for example 0.30 m at 0.08 m resolution used to become
        # only 0.24 m along an axis).
        raster_radius_m = self.inflation_radius_m + (
            self.resolution_m * math.sqrt(2.0) / 2.0
        )
        radius_cells = int(math.ceil(raster_radius_m / self.resolution_m))
        offsets = [
            (dx, dy)
            for dx in range(-radius_cells, radius_cells + 1)
            for dy in range(-radius_cells, radius_cells + 1)
            if math.hypot(dx, dy) * self.resolution_m
            <= raster_radius_m + 1e-9
        ]
        self._inflated_cache = {
            (cell[0] + dx, cell[1] + dy)
            for cell in self.occupied_cells
            for dx, dy in offsets
        }
        return self._inflated_cache

    def _ring_heuristic(
        self, cell: GridCell, target_xy: PointXY, docking_distance_m: float
    ) -> float:
        radial_error_m = abs(
            math.dist(self.point(cell), target_xy) - docking_distance_m
        )
        return radial_error_m / self.resolution_m

    @staticmethod
    def _supercover_line(start: GridCell, end: GridCell) -> list[GridCell]:
        """Integer supercover line including both endpoints."""

        x0, y0 = start
        x1, y1 = end
        dx = x1 - x0
        dy = y1 - y0
        nx = abs(dx)
        ny = abs(dy)
        sign_x = 1 if dx > 0 else -1
        sign_y = 1 if dy > 0 else -1
        x, y = x0, y0
        cells = [(x, y)]
        ix = iy = 0
        while ix < nx or iy < ny:
            decision = (1 + 2 * ix) * ny - (1 + 2 * iy) * nx
            if decision == 0:
                # At an exact grid corner the geometric segment touches both
                # orthogonal cells. Including them prevents line-of-sight
                # smoothing from slipping diagonally between obstacles.
                cells.append((x + sign_x, y))
                cells.append((x, y + sign_y))
                x += sign_x
                y += sign_y
                ix += 1
                iy += 1
            elif decision < 0:
                x += sign_x
                ix += 1
            else:
                y += sign_y
                iy += 1
            cells.append((x, y))
        return cells
