"""Plausibility gate on the ZED SDK floor plane.

The SDK fits the floor plane again on every frame from whatever floor it can
see. With little floor in view (the camera looking across a table top, or at
a seated person) the fit is wrong in bursts: the reported camera height jumps
by half a metre or more from one frame to the next while the camera has not
moved, against a static jitter of a few centimetres. Both consumers of that
plane took it as it came: the Nav2 ZED bridge places the camera transform,
the height of every cloud point and the arm-height band with it, so one bad
frame charged a table top and whoever sat at it to the arms and the arms
collision monitor braked a docking that was clear (2026-09-13); the docking
perception projects the target and the obstacles with it.

The gate keeps the plane that is steady rather than the plane that is
latest. The configured fallback is only a placeholder: the first plane the
SDK offers inside the absolute envelope replaces it at once, so a fallback
that is a few centimetres off does not cost the first observation. After
that, a candidate within one step of the plane in use is adopted at once,
which follows the lift and the base's pitch over a threshold. A larger jump
is held as a pending candidate and adopted only after consecutive offers have
agreed with it for the settle time: a real change (a lift move) is steady and
the bursts are not. Agreement is judged against the running mean of the
pending offers, with twice the step, so jitter as wide as the step on either
side of a steady plane still counts as agreement. A candidate outside the
absolute envelope around the configured fallback is never adopted, however
steady, so the envelope has to cover the lift's travel and the fallback's own
error.

Time belongs to the caller: :meth:`GroundPlaneGate.offer` takes the frame
time in seconds and never reads a clock, so a replayed frame sequence decides
the same way every time.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

ACCEPTED = "accepted"
HELD = "held"
SETTLING = "settling"


@dataclass(frozen=True)
class GroundPlaneDecision:
    """What :meth:`GroundPlaneGate.offer` decided about one candidate plane.

    ``camera_height_m`` and ``down_camera_xyz`` are the plane to use: the
    plane the gate already held unless ``status`` is ``accepted``.
    ``held_for_s`` is the time since an accepted candidate last confirmed the
    plane in use, zero when this candidate did. ``offered_height_m`` is the
    candidate's height, for logs of a rejected one. ``settled`` marks an
    acceptance earned by a pending candidate persisting rather than by a
    small step.
    """

    camera_height_m: float
    down_camera_xyz: np.ndarray
    status: str
    reason: str
    held_for_s: float
    offered_height_m: float
    settled: bool = False


@dataclass
class _Pending:
    """A candidate too far from the plane in use, averaged over its offers."""

    height_m: float
    down: np.ndarray
    since_s: float
    count: int = 1

    def absorb(self, height_m: float, down: np.ndarray) -> None:
        total = self.count + 1
        self.height_m = (self.height_m * self.count + height_m) / total
        merged = _unit(self.down * self.count + down)
        if merged is not None:
            self.down = merged
        self.count = total


def _unit(vector: np.ndarray) -> np.ndarray | None:
    """Unit copy of a finite, nonzero 3-vector, or ``None``."""

    values = np.asarray(vector, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(values))
    if not np.all(np.isfinite(values)) or norm <= 1e-9:
        return None
    return values / norm


def _angle_deg(first: np.ndarray, second: np.ndarray) -> float:
    """Angle between two unit vectors in degrees."""

    cosine = float(np.clip(np.dot(first, second), -1.0, 1.0))
    return math.degrees(math.acos(cosine))


class GroundPlaneGate:
    """Adopt an offered floor plane only when it is a plausible change.

    The caller validates the shape and finiteness of what it offers; a
    non-finite or zero-length down vector is refused as invalid rather than
    crashing the frame loop, everything else malformed raises.
    """

    def __init__(
        self,
        fallback_camera_height_m: float,
        fallback_down_camera_xyz: np.ndarray,
        *,
        max_height_step_m: float = 0.05,
        max_tilt_step_deg: float = 3.0,
        settle_s: float = 2.0,
        max_height_error_m: float = 0.25,
        max_tilt_error_deg: float = 15.0,
    ) -> None:
        self._max_height_step_m = float(max_height_step_m)
        self._max_tilt_step_deg = float(max_tilt_step_deg)
        self._settle_s = float(settle_s)
        self._max_height_error_m = float(max_height_error_m)
        self._max_tilt_error_deg = float(max_tilt_error_deg)
        if not 0.005 <= self._max_height_step_m <= 0.5:
            raise ValueError("max_height_step_m must be in [0.005, 0.5]")
        if not 0.1 <= self._max_tilt_step_deg <= 45.0:
            raise ValueError("max_tilt_step_deg must be in [0.1, 45]")
        if not 0.1 <= self._settle_s <= 30.0:
            raise ValueError("settle_s must be in [0.1, 30]")
        if not self._max_height_step_m <= self._max_height_error_m <= 2.0:
            raise ValueError(
                "max_height_error_m must be in [max_height_step_m, 2.0]"
            )
        if not self._max_tilt_step_deg <= self._max_tilt_error_deg <= 90.0:
            raise ValueError(
                "max_tilt_error_deg must be in [max_tilt_step_deg, 90]"
            )
        self._fallback_height_m = float(fallback_camera_height_m)
        fallback_down = _unit(fallback_down_camera_xyz)
        if not math.isfinite(self._fallback_height_m) or fallback_down is None:
            raise ValueError("the fallback plane must be finite with a nonzero down")
        self._fallback_down = fallback_down
        self._height_m = self._fallback_height_m
        self._down = fallback_down.copy()
        # Whether the plane in use came from the SDK; until then it is the
        # fallback placeholder and the first plausible offer replaces it.
        self._observed = False
        # Frame time at which an accepted candidate last confirmed the plane
        # in use; the first offer stands in for the fallback's confirmation.
        self._confirmed_s: float | None = None
        self._pending: _Pending | None = None

    @property
    def camera_height_m(self) -> float:
        return self._height_m

    @property
    def down_camera_xyz(self) -> np.ndarray:
        return self._down.copy()

    def _within(
        self,
        height_m: float,
        down: np.ndarray,
        reference_height_m: float,
        reference_down: np.ndarray,
        scale: float = 1.0,
    ) -> bool:
        # A hair of slack so a difference of exactly the tolerance, which
        # floating point rounds either way, counts as agreement.
        return (
            abs(height_m - reference_height_m) <= scale * self._max_height_step_m + 1e-9
            and _angle_deg(down, reference_down) <= scale * self._max_tilt_step_deg + 1e-9
        )

    def _decision(
        self,
        status: str,
        reason: str,
        time_s: float,
        offered_height_m: float,
        *,
        settled: bool = False,
    ) -> GroundPlaneDecision:
        assert self._confirmed_s is not None
        return GroundPlaneDecision(
            camera_height_m=self._height_m,
            down_camera_xyz=self._down.copy(),
            status=status,
            reason=reason,
            held_for_s=max(0.0, time_s - self._confirmed_s),
            offered_height_m=offered_height_m,
            settled=settled,
        )

    def offer(
        self,
        camera_height_m: float,
        down_camera_xyz: np.ndarray,
        time_s: float,
    ) -> GroundPlaneDecision:
        """Judge one candidate plane offered at frame time ``time_s``."""

        time_s = float(time_s)
        if self._confirmed_s is None:
            self._confirmed_s = time_s
        height_m = float(camera_height_m)
        down = _unit(down_camera_xyz)
        if not math.isfinite(height_m) or down is None:
            self._pending = None
            return self._decision(
                HELD, "candidate plane is not finite", time_s, height_m
            )
        height_error_m = abs(height_m - self._fallback_height_m)
        tilt_error_deg = _angle_deg(down, self._fallback_down)
        if (
            height_error_m > self._max_height_error_m
            or tilt_error_deg > self._max_tilt_error_deg
        ):
            self._pending = None
            return self._decision(
                HELD,
                f"candidate {height_m:.2f} m is {height_error_m:.2f} m and "
                f"{tilt_error_deg:.1f} deg from the fallback, outside its "
                f"envelope of {self._max_height_error_m:.2f} m and "
                f"{self._max_tilt_error_deg:.1f} deg",
                time_s,
                height_m,
            )
        if not self._observed:
            previous_m = self._height_m
            self._adopt(height_m, down, time_s)
            return self._decision(
                ACCEPTED,
                f"first plane inside the envelope replaces the fallback "
                f"({previous_m:.2f} m)",
                time_s,
                height_m,
            )
        if self._within(height_m, down, self._height_m, self._down):
            self._adopt(height_m, down, time_s)
            return self._decision(
                ACCEPTED, "within one step of the plane in use", time_s, height_m
            )
        pending = self._pending
        if pending is not None and self._within(
            height_m, down, pending.height_m, pending.down, scale=2.0
        ):
            pending.absorb(height_m, down)
            persisted_s = max(0.0, time_s - pending.since_s)
            if persisted_s >= self._settle_s:
                previous_m = self._height_m
                self._adopt(pending.height_m, pending.down, time_s)
                return self._decision(
                    ACCEPTED,
                    f"settled at {self._height_m:.2f} m after {persisted_s:.1f} s "
                    f"over {pending.count} offers (was {previous_m:.2f} m)",
                    time_s,
                    height_m,
                    settled=True,
                )
            return self._decision(
                SETTLING,
                f"candidate {pending.height_m:.2f} m has persisted "
                f"{persisted_s:.1f} of {self._settle_s:.1f} s",
                time_s,
                height_m,
            )
        self._pending = _Pending(height_m, down.copy(), time_s)
        return self._decision(
            HELD,
            f"candidate {height_m:.2f} m is {height_m - self._height_m:+.2f} m "
            f"and {_angle_deg(down, self._down):.1f} deg from the plane in use; "
            f"adopted once steady for {self._settle_s:.1f} s",
            time_s,
            height_m,
        )

    def _adopt(self, height_m: float, down: np.ndarray, time_s: float) -> None:
        self._height_m = float(height_m)
        self._down = np.asarray(down, dtype=np.float64).copy()
        self._observed = True
        self._confirmed_s = time_s
        self._pending = None
