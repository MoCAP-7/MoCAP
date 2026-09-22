"""Record the Pi base RPC status to JSONL for the length of an ApexNav run.

The status holds the velocity the lease service last accepted, the base
controller's commanded velocity, and per-module steering angles and drive
velocities. This process only reads status; it never takes the velocity lease.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
import json
from pathlib import Path
import signal
import threading
import time
from typing import Any

from .config import load_config


def _connect(host: str, port: int, timeout_ms: int) -> Any:
    from commlink import RPCClient
    import zmq

    client = RPCClient(host=host, port=port)
    for option, value in (
        (zmq.RCVTIMEO, timeout_ms),
        (zmq.SNDTIMEO, timeout_ms),
        (zmq.LINGER, 0),
    ):
        client.socket.setsockopt(option, value)
    return client


def _close(client: Any) -> None:
    """Release a client's socket and context so reconnecting does not leak them."""

    socket = getattr(client, "socket", None)
    context = getattr(client, "context", None)
    try:
        if socket is not None:
            socket.close(linger=0)
        if context is not None:
            context.term()
    except Exception:  # noqa: BLE001 - a failed close must not stop logging
        pass


def poll_status(
    connect: Callable[[], Any],
    output: Path,
    period_s: float,
    stop: threading.Event,
    *,
    error_backoff_s: float = 1.0,
) -> int:
    """Append one status record per period until ``stop`` is set; return the count."""

    client = connect()
    written = 0
    try:
        with output.open("a", encoding="utf-8") as stream:
            while not stop.is_set():
                requested = time.time()
                try:
                    status = client.get_status()
                    record = {"t_request": requested, "t_reply": time.time(), "status": status}
                    delay_s = period_s - (time.time() - requested)
                except Exception as exc:  # noqa: BLE001 - keep recording
                    record = {"t_request": requested, "error": f"{type(exc).__name__}: {exc}"}
                    # A request socket that timed out cannot send again.
                    _close(client)
                    client = connect()
                    delay_s = error_backoff_s
                stream.write(json.dumps(record, default=str) + "\n")
                stream.flush()
                written += 1
                stop.wait(max(0.0, delay_s))
    finally:
        _close(client)
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    # The Pi RPC server answers one request at a time, so status polls queue
    # with the relay's lease renewals. Keep the default rate low; the recorded
    # request/reply times and the relay's per-submit RPC times show contention.
    parser.add_argument("--hz", type=float, default=4.0)
    parser.add_argument("--timeout-ms", type=int, default=1000)
    args = parser.parse_args(argv)
    config = load_config(Path(args.config).expanduser().resolve())
    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)

    stop = threading.Event()

    def request_stop(signum: int, frame: Any) -> None:
        del signum, frame
        stop.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    poll_status(
        lambda: _connect(
            config.robot.base_rpc_host, config.robot.base_rpc_port, args.timeout_ms
        ),
        output,
        1.0 / args.hz,
        stop,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
