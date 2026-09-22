"""Client-side transport for the YOR services running on the Jetson and Pi.

Moved from ``YOR/Agent/agents_yor/hardware.py`` with only naming/comment edits so
``yor_agent`` owns the whole high-level-agent-to-low-level-robot path.
"""

from __future__ import annotations

import threading
import time
from typing import Any


class YorHardwareBridge:
    """Connect ``yor_agent`` to YOR's leased base RPC and atomic ZED stream.

    Imports are intentionally lazy.  This keeps the package importable on
    development machines while the real robot still uses the existing
    ``commlink`` and ``yor-navdp`` installations.
    """

    def __init__(
        self,
        *,
        base_rpc_host: str = "192.168.1.10",
        base_rpc_port: int = 5557,
        zed_host: str = "127.0.0.1",
        zed_port: int = 6000,
        zed_transport: str = "atomic",
        published_color_order: str = "auto",
        arm_rpc_host: str | None = None,
        arm_rpc_port: int = 5558,
        require_arms: bool = False,
        rpc_client: Any | None = None,
        zed_source: Any | None = None,
        arm_rpc_client: Any | None = None,
    ) -> None:
        if rpc_client is None:
            try:
                from commlink import RPCClient
            except ImportError as exc:
                raise RuntimeError(
                    "commlink is required on the YOR robot host"
                ) from exc
            rpc_client = RPCClient(host=base_rpc_host, port=int(base_rpc_port))
        if zed_source is None:
            try:
                from navdp_deploy.sources.zed_commlink import ZEDCommlinkSource
            except ImportError as exc:
                raise RuntimeError(
                    "yor-navdp must be installed so yor_agent can consume the atomic ZED stream"
                ) from exc
            zed_source = ZEDCommlinkSource(
                host=zed_host,
                port=int(zed_port),
                transport=zed_transport,
                published_color_order=published_color_order,
            )

        self._rpc = rpc_client
        self._zed = zed_source
        if arm_rpc_client is None and arm_rpc_host is not None:
            from commlink import RPCClient

            arm_rpc_client = RPCClient(host=arm_rpc_host, port=int(arm_rpc_port))
        if require_arms and arm_rpc_client is None:
            raise ValueError("require_arms=True requires arm_rpc_host or arm_rpc_client")
        self._arm_rpc = arm_rpc_client
        self._require_arms = bool(require_arms)
        self._sequence_lock = threading.Lock()
        self._last_sequence = time.time_ns()
        self._arm_sequences = {"left": time.time_ns(), "right": time.time_ns()}
        self._closed = False

    def latest_frame(self, *, max_age_s: float):
        return self._zed.latest_frame(max_age_s=max_age_s)

    def latest_frame_age_s(self) -> float | None:
        """Arrival age of the newest ZED frame, if the transport reports it."""

        getter = getattr(self._zed, "latest_frame_age_s", None)
        if not callable(getter):
            return None
        age = getter()
        return None if age is None else float(age)

    def next_frame(self, timeout_s: float | None = 2.0):
        return self._zed.next_frame(timeout_s=timeout_s)

    def get_base_status(self) -> dict[str, Any]:
        status = self._rpc.get_status()
        if not isinstance(status, dict):
            raise TypeError("YOR base RPC get_status() must return a dictionary")
        return status

    def submit_base_velocity(self, velocity: list[float]) -> dict[str, Any]:
        with self._sequence_lock:
            self._last_sequence = max(time.time_ns(), self._last_sequence + 1)
            sequence = self._last_sequence
        reply = self._rpc.submit_velocity(velocity, sequence)
        if not isinstance(reply, dict) or not reply.get("accepted", False):
            raise RuntimeError(f"base RPC rejected command: {reply!r}")
        return reply

    @staticmethod
    def _arm_name(arm: int | str) -> str:
        if arm in (0, "0", "left"):
            return "left"
        if arm in (1, "1", "right"):
            return "right"
        raise ValueError("arm must be 0/'left' or 1/'right'")

    def _next_arm_sequence(self, arm: int | str) -> tuple[str, int]:
        name = self._arm_name(arm)
        with self._sequence_lock:
            self._arm_sequences[name] = max(
                time.time_ns(), self._arm_sequences[name] + 1
            )
            return name, self._arm_sequences[name]

    def get_arm_status(self) -> dict[str, Any] | None:
        if self._arm_rpc is None:
            if self._require_arms:
                raise RuntimeError("required YOR arm RPC is not configured")
            return None
        status = self._arm_rpc.get_status()
        if not isinstance(status, dict):
            raise TypeError("YOR arm RPC get_status() must return a dictionary")
        return status

    def move_arm_pose(
        self,
        arm: int | str,
        pose_xyz_rpy: list[float],
        timeout_s: float = 60.0,
    ) -> dict[str, Any]:
        if self._arm_rpc is None:
            raise RuntimeError("YOR arm RPC is not configured")
        name, sequence = self._next_arm_sequence(arm)
        result = self._arm_rpc.move_tcp_pose(
            name, pose_xyz_rpy, sequence, float(timeout_s)
        )
        if not isinstance(result, dict):
            raise TypeError("YOR arm RPC move_tcp_pose() returned a non-dictionary")
        return result

    def plan_arm_poses(
        self,
        arm: int | str,
        poses_xyz_rpy: list[list[float]],
    ) -> dict[str, Any]:
        """Ask the Pi Mink backend to solve poses without commanding motion."""

        if self._arm_rpc is None:
            raise RuntimeError("YOR arm RPC is not configured")
        name = self._arm_name(arm)
        result = self._arm_rpc.plan_tcp_poses(name, poses_xyz_rpy)
        if not isinstance(result, dict):
            raise TypeError("YOR arm RPC plan_tcp_poses() returned a non-dictionary")
        return result

    def execute_arm_trajectory(
        self,
        arm: int | str,
        waypoints: list[list[float]],
        timeout_s: float = 60.0,
    ) -> dict[str, Any]:
        if self._arm_rpc is None:
            raise RuntimeError("YOR arm RPC is not configured")
        name, sequence = self._next_arm_sequence(arm)
        result = self._arm_rpc.execute_joint_trajectory(
            name, waypoints, sequence, float(timeout_s)
        )
        if not isinstance(result, dict):
            raise TypeError(
                "YOR arm RPC execute_joint_trajectory() returned a non-dictionary"
            )
        return result

    def set_gripper(
        self,
        arm: int | str,
        opened: bool,
        timeout_s: float = 3.0,
        force_n: float | None = None,
    ) -> dict[str, Any]:
        if self._arm_rpc is None:
            raise RuntimeError("YOR arm RPC is not configured")
        name, sequence = self._next_arm_sequence(arm)
        result = self._arm_rpc.set_gripper(
            name, bool(opened), sequence, float(timeout_s), force_n
        )
        if not isinstance(result, dict):
            raise TypeError("YOR arm RPC set_gripper() returned a non-dictionary")
        return result

    def emergency_stop(self) -> dict[str, Any]:
        """Latch the Pi-side emergency stop; this is not exposed to generated code."""

        return self._rpc.emergency_stop()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self.submit_base_velocity([0.0, 0.0, 0.0])
        except Exception:
            # The Pi lease expires independently, even if this final RPC is lost.
            pass
        close_zed = getattr(self._zed, "close", None)
        if callable(close_zed):
            try:
                close_zed()
            except Exception:
                pass
        # Closing a Jetson-side task is a client lifecycle event, not an arm
        # fault. The Pi arm service keeps the enabled arms in their last
        # controlled hold and owns explicit/fault emergency stops. Latching an
        # electronic stop here made every ordinary environment close require a
        # hazardous physical reset before the next RPC session.
        try:
            self._close_rpc_transport(self._rpc)
        except Exception:
            pass
        if self._arm_rpc is not None:
            try:
                self._close_rpc_transport(self._arm_rpc)
            except Exception:
                pass

    @staticmethod
    def _close_rpc_transport(client: Any) -> None:
        """Close a client locally without sending commlink's remote stop request.

        ``RPCClient`` implements dynamic ``__getattr__``. Probing an absent
        ``close`` method on the instance therefore makes an unwanted RPC call.
        Looking up methods on the class avoids that behavior, while the fallback
        handles the currently deployed commlink client, which has no public
        local-close method.
        """

        close_method = getattr(type(client), "close", None)
        if callable(close_method):
            close_method(client)
            return
        attributes = getattr(client, "__dict__", {})
        socket = attributes.get("socket")
        context = attributes.get("context")
        if socket is not None:
            try:
                socket.close(linger=0)
            except TypeError:
                socket.close()
        if context is not None:
            context.term()
