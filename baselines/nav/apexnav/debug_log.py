"""Per-run debug recording for ApexNav on YOR.

Every episode keeps enough raw data under ``<output>/debug`` to debug the
integration afterwards without re-running it: a rosbag of the planner, tracker,
command, and detector topics, the Pi base status, the relay's commands, the
bridge's odometry samples, ROS node logs, the planner console, and a snapshot
of the code and services the run used.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import json
import os
from pathlib import Path
import queue
import socket
import subprocess
import sys
import threading
import time
from typing import Any
from urllib.parse import urlparse

DEBUG_DIRNAME = "debug"

# Images and depth are left out: they dominate a bag's size, and the detector
# already writes annotated images next to the trace.
BAG_TOPICS = (
    "/apexnav/cmd_vel_raw",
    "/apexnav/odom",
    "/apexnav/sensor_pose",
    "/apexnav/planning/trajectory",
    "/apexnav/traj_server/stop",
    "/apexnav/ros/state",
    "/apexnav/ros/expl_state",
    "/apexnav/ros/expl_result",
    "/apexnav/initial_scan_status",
    "/apexnav/start",
    "/apexnav/detector/clouds_with_scores",
    "/apexnav/detector/confidence_threshold",
    "/apexnav/blip2/cosine_score",
    "/current_desire",
    "/mpc_car/track_err",
)

_SERVICE_PROCESS_MARKERS = (
    "apexnav.gdino_server",
    "apexnav.blip2_server",
    "vlfm.model_server",
    "services/zed/service.py",
)


class JsonlWriter:
    """Append JSON records from any thread without blocking the caller on disk I/O."""

    def __init__(self, path: Path, max_pending: int = 20000) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._queue: queue.Queue[dict[str, Any] | None] = queue.Queue(maxsize=max_pending)
        self.dropped = 0
        self._thread = threading.Thread(
            target=self._drain, name=f"jsonl-{self.path.name}", daemon=True
        )
        self._thread.start()

    def write(self, record: Mapping[str, Any]) -> None:
        entry = {"t_wall": time.time(), "t_monotonic": time.monotonic(), **record}
        try:
            self._queue.put_nowait(entry)
        except queue.Full:
            self.dropped += 1

    def close(self, timeout_s: float = 2.0) -> None:
        try:
            self._queue.put(None, timeout=timeout_s)
        except queue.Full:
            pass
        self._thread.join(timeout=timeout_s)

    def _drain(self) -> None:
        with self.path.open("a", encoding="utf-8") as stream:
            while True:
                entry = self._queue.get()
                if entry is None:
                    stream.flush()
                    return
                stream.write(json.dumps(entry, sort_keys=True, default=_json_default) + "\n")
                if self._queue.empty():
                    stream.flush()


def _json_default(value: Any) -> Any:
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return tolist()
    return str(value)


def bag_record_command(bag_dir: Path, topics: Iterable[str] = BAG_TOPICS) -> list[str]:
    return ["ros2", "bag", "record", "-o", str(bag_dir), *topics]


def start_logged_process(command: list[str], log_path: Path) -> subprocess.Popen:
    """Start a helper in its own session with its output written to ``log_path``."""

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab") as stream:
        return subprocess.Popen(
            command, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True
        )


def start_console_tee(process: subprocess.Popen, log_path: Path) -> threading.Thread:
    """Copy a child's combined output to this terminal and to ``log_path``.

    The pipe is drained even after the terminal goes away, so the child never
    blocks on a full pipe.
    """

    output = process.stdout
    if output is None:
        raise ValueError("the process was not started with stdout=subprocess.PIPE")

    def pump() -> None:
        with log_path.open("ab") as log:
            for line in iter(output.readline, b""):
                log.write(line)
                log.flush()
                try:
                    sys.stdout.buffer.write(line)
                    sys.stdout.buffer.flush()
                except (OSError, ValueError):
                    pass

    log_path.parent.mkdir(parents=True, exist_ok=True)
    thread = threading.Thread(target=pump, name="apexnav-planner-console", daemon=True)
    thread.start()
    return thread


def environment_snapshot(
    repository_root: Path,
    config_path: Path,
    endpoints: Mapping[str, str],
) -> dict[str, Any]:
    """The code revision, reachable services, and service commands a run used."""

    return {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "hostname": socket.gethostname(),
        "python": sys.version.split()[0],
        "ros_domain_id": os.environ.get("ROS_DOMAIN_ID"),
        "config_path": str(config_path),
        "git": _git_state(repository_root),
        "endpoints_reachable": {
            name: _endpoint_reachable(url) for name, url in endpoints.items()
        },
        "service_processes": _matching_processes(_SERVICE_PROCESS_MARKERS),
    }


def _run(command: list[str], cwd: Path | None = None) -> str | None:
    try:
        completed = subprocess.run(
            command, cwd=cwd, capture_output=True, text=True, timeout=5.0, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout if completed.returncode == 0 else None


def _git_state(repository_root: Path) -> dict[str, Any]:
    head = _run(["git", "rev-parse", "HEAD"], cwd=repository_root)
    status = _run(["git", "status", "--porcelain"], cwd=repository_root)
    changed = [] if status is None else [line for line in status.splitlines() if line]
    return {
        "head": None if head is None else head.strip(),
        "changed_path_count": None if status is None else len(changed),
        "changed_paths": changed[:50],
    }


def _endpoint_reachable(url: str) -> bool:
    parsed = urlparse(url)
    if not parsed.hostname or not parsed.port:
        return False
    try:
        with socket.create_connection((parsed.hostname, parsed.port), timeout=0.5):
            return True
    except OSError:
        return False


def _matching_processes(markers: Iterable[str]) -> list[str]:
    listing = _run(["ps", "-eo", "pid,args"])
    if listing is None:
        return []
    markers = tuple(markers)
    return [
        line.strip()[:300]
        for line in listing.splitlines()[1:]
        if any(marker in line for marker in markers)
    ]
