"""Run one isolated ApexNav open-vocabulary ObjectNav episode on YOR."""

from __future__ import annotations

import argparse
import ipaddress
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.signals import SignalHandlerOptions
import yaml

from ..experiment.protocol import (
    DEFAULT_TASK_SUITE,
    add_experiment_arguments,
    episode_directory,
    experiment_directory,
    record_review,
    reject_manipulation,
    review_episode,
    suite_instruction,
    task_slug,
    yor_version,
)
from .bridge import YorApexNavBridge
from .config import ViewerConfig, load_config
from .debug_log import (
    DEBUG_DIRNAME,
    JsonlWriter,
    bag_record_command,
    environment_snapshot,
    start_console_tee,
    start_logged_process,
)
from .priors import load_target_prior
from .tasks import instruction_to_target

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
FOXGLOVE_BRIDGE = (
    Path(__file__).resolve().parent
    / ".deps/sysroot/opt/ros/humble/lib/foxglove_bridge/foxglove_bridge"
)
# Topics a live viewer may read. foxglove_bridge full-matches each pattern,
# ignoring case. Raw RGB and depth are left out for their size, and nothing
# that starts an episode or commands the robot is listed.
VIEWER_TOPIC_WHITELIST = (
    r"^/grid_map/(free|occupied|occupied_inflate|value_map|occupancy_object"
    r"|filtered_object_cloud|over_depth_object_cloud|all_object_cloud)$",
    r"^/object/clouds$",
    r"^/planning_vis/(frontier|viewpoints)$",
    r"^/apexnav/visualization/robot$",
    r"^/travel_traj$",
    r"^/trajectory/mincoPath$",
    r"^/kinoastar/FlatTraj$",
    r"^/mpc_car/(reference_path|predict_path|track_err)$",
    r"^/apexnav/detector/detect_img/compressed$",
    r"^/apexnav/(odom|cmd_vel_raw)$",
    r"^/apexnav/ros/(state|expl_state|expl_result)$",
    r"^/apexnav/initial_scan_status$",
    r"^/tf$",
)
_NEVER_MATCHES = ["(?!)"]


def _start_planner(config_path: Path, console_log: Path | None = None) -> subprocess.Popen:
    command = [
        "ros2",
        "launch",
        "yor_apexnav_bringup",
        "apexnav_yor.launch.py",
        f"baseline_config:={config_path}",
    ]
    if console_log is None:
        return subprocess.Popen(command, start_new_session=True)
    # Unbuffered so the terminal copy stays live while the log keeps everything.
    environment = dict(os.environ, PYTHONUNBUFFERED="1")
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        env=environment,
    )
    start_console_tee(process, console_log)
    return process


def _start_control_relay(
    config_path: Path, output_dir: Path, debug_dir: Path | None = None
) -> subprocess.Popen:
    command = [
        sys.executable,
        "-m",
        "baselines.nav.apexnav.control_relay",
        "--config",
        str(config_path),
        "--output-dir",
        str(output_dir.resolve()),
    ]
    if debug_dir is not None:
        command += ["--debug-dir", str(debug_dir.resolve())]
    process = subprocess.Popen(command, start_new_session=True)
    ready_path = output_dir / "control_ready"
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        if ready_path.exists():
            return process
        if process.poll() is not None:
            raise RuntimeError(
                f"ApexNav control relay exited during startup with code {process.returncode}"
            )
        time.sleep(0.05)
    _stop_process_group(process)
    raise RuntimeError("ApexNav control relay did not become ready within 10 s")


def _start_debug_recorders(
    config_path: Path, debug_dir: Path
) -> tuple[subprocess.Popen, subprocess.Popen]:
    """Start the base status logger and the rosbag recorder, in stop order."""

    base_status = start_logged_process(
        [
            sys.executable,
            "-m",
            "baselines.nav.apexnav.base_status_logger",
            "--config",
            str(config_path),
            "--output",
            str((debug_dir / "base_status.jsonl").resolve()),
        ],
        debug_dir / "base_status_logger.log",
    )
    bag = start_logged_process(
        bag_record_command((debug_dir / "bag").resolve()),
        debug_dir / "bag_record.log",
    )
    return base_status, bag


def _viewer_address(configured: str) -> str:
    """Bind the configured address, else this host's Tailscale IPv4, else loopback.

    The Tailscale address keeps the unauthenticated viewer socket off the
    robot's local network.
    """

    if configured:
        return configured
    try:
        completed = subprocess.run(
            ["tailscale", "ip", "-4"],
            capture_output=True,
            text=True,
            timeout=2.0,
            check=True,
        )
        return str(ipaddress.IPv4Address(completed.stdout.split()[0]))
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return "127.0.0.1"


def _viewer_parameters(viewer: ViewerConfig, address: str) -> dict:
    """foxglove_bridge settings that let a viewer read whitelisted topics only."""

    parameters = {
        "port": viewer.port,
        "address": address,
        "topic_whitelist": list(VIEWER_TOPIC_WHITELIST),
        # Unknown capability names are ignored, so viewers can neither
        # publish, call services, read or set parameters, nor fetch assets.
        "capabilities": ["none"],
        # Separate lists: a shared one is dumped as a YAML anchor and aliases,
        # which rcl's parameter file parser rejects.
        "service_whitelist": list(_NEVER_MATCHES),
        "param_whitelist": list(_NEVER_MATCHES),
        "client_topic_whitelist": list(_NEVER_MATCHES),
        "asset_uri_allowlist": list(_NEVER_MATCHES),
        "include_hidden": False,
        "min_qos_depth": 1,
        "max_qos_depth": viewer.max_qos_depth,
        # A message count per viewer; the oldest are dropped past it. It is the
        # only bound on data queued for a slow viewer.
        "message_backlog_size": viewer.message_backlog_size,
        "num_threads": viewer.num_threads,
        "sysinfo": False,
        "publish_client_count": False,
        "remote_access": False,
        "use_sim_time": False,
        "tls": False,
    }
    # "/**" because the executable reads num_threads on a temporary node before
    # it creates the bridge node.
    return {"/**": {"ros__parameters": parameters}}


def _start_viewer(
    viewer: ViewerConfig, log_dir: Path, executable: Path = FOXGLOVE_BRIDGE
) -> tuple[subprocess.Popen | None, str | None]:
    """Start the live viewer bridge, or report why the episode runs without it."""

    if not os.access(executable, os.X_OK):
        _print_status(
            f"ApexNav viewer skipped: {executable} is missing; "
            "run baselines/nav/apexnav/scripts/bootstrap_viewer_deps.sh"
        )
        return None, None
    address = _viewer_address(viewer.address)
    if not viewer.address and address == "127.0.0.1":
        _print_status("ApexNav viewer: no Tailscale address, binding 127.0.0.1")
    parameters_path = (log_dir / "foxglove_bridge.params.yaml").resolve()
    parameters_path.write_text(
        yaml.safe_dump(_viewer_parameters(viewer, address), sort_keys=False),
        encoding="utf-8",
    )
    process = start_logged_process(
        [str(executable), "--ros-args", "--params-file", str(parameters_path)],
        log_dir / "foxglove_bridge.log",
    )
    url = f"ws://{address}:{viewer.port}"
    _print_status(
        f"ApexNav live viewer: Lichtblick > Open connection > Foxglove WebSocket > {url}"
    )
    return process, url


def _stop_process_group(process: subprocess.Popen | None) -> None:
    if process is None:
        return

    def signal_group(value: signal.Signals) -> bool:
        try:
            os.killpg(process.pid, value)
            return True
        except ProcessLookupError:
            return False

    def wait_group(timeout_s: float) -> bool:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            # Reap the launch process as soon as it exits; a zombie group
            # leader otherwise makes killpg(..., 0) look artificially alive.
            process.poll()
            try:
                os.killpg(process.pid, 0)
            except ProcessLookupError:
                return True
            time.sleep(0.1)
        return False

    if not signal_group(signal.SIGINT):
        return
    if wait_group(5.0):
        return
    signal_group(signal.SIGTERM)
    if wait_group(3.0):
        return
    signal_group(signal.SIGKILL)
    wait_group(2.0)
    try:
        process.wait(timeout=0.5)
    except subprocess.TimeoutExpired:
        pass


def _stop_planner(process: subprocess.Popen | None) -> None:
    _stop_process_group(process)


def _stop_children(
    planner: subprocess.Popen | None,
    control_relay: subprocess.Popen | None,
    *recorders: subprocess.Popen | None,
) -> None:
    """Stop the planner, the base relay, then the recorders, even if one stop fails.

    The recorders stop last so they capture the zero commands sent on the way
    down. The viewer bridge is passed after them so it shows the stop too.
    """

    try:
        _stop_planner(planner)
    finally:
        try:
            _stop_process_group(control_relay)
        finally:
            for recorder in recorders:
                try:
                    _stop_process_group(recorder)
                except Exception as exc:  # noqa: BLE001 - stop the remaining recorders
                    _print_status(
                        f"ApexNav recorder stop warning: {type(exc).__name__}: {exc}"
                    )


def _raise_keyboard_interrupt(signum: int, frame: object) -> None:
    del signum, frame
    raise KeyboardInterrupt


def _ignore_interrupts() -> None:
    """Keep a repeated Ctrl-C, a closed terminal, or SIGTERM from aborting cleanup."""

    for value in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(value, signal.SIG_IGN)


def _print_status(text: str) -> None:
    """Print cleanup progress; a closed terminal must not abort cleanup."""

    try:
        print(text, flush=True)
    except OSError:
        pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    target_group = parser.add_mutually_exclusive_group(required=True)
    target_group.add_argument("--task-id", help="task in the shared YOR task suite, e.g. find_blue_trash_bin")
    target_group.add_argument("--target", help="open-vocabulary detector target")
    target_group.add_argument("--instruction", help="e.g. 'Find the blue trash bin'")
    parser.add_argument("--task-config", default=str(DEFAULT_TASK_SUITE), help="YOR task suite YAML for --task-id")
    parser.add_argument(
        "--config",
        default=str(Path(__file__).with_name("config.yaml")),
    )
    parser.add_argument("--similar-label", action="append", default=None)
    parser.add_argument("--confidence-threshold", type=float)
    parser.add_argument("--room")
    parser.add_argument("--output-dir", help="explicit episode directory instead of the experiment layout")
    parser.add_argument("--no-motion", action="store_true", help="publish ROS data but never command the base")
    parser.add_argument("--planner-already-running", action="store_true")
    parser.add_argument(
        "--no-debug-record",
        action="store_true",
        help="skip the per-run debug recording under <output>/debug",
    )
    parser.add_argument(
        "--no-viewer",
        action="store_true",
        help="do not start the live map viewer bridge",
    )
    add_experiment_arguments(parser)
    return parser


def resolve_task(args: argparse.Namespace) -> tuple[str | None, str | None, str]:
    """Task id, instruction and detector target from --task-id, --instruction or --target."""

    if args.task_id:
        instruction = reject_manipulation(suite_instruction(args.task_id, args.task_config))
        return args.task_id, instruction, instruction_to_target(instruction)
    if args.instruction:
        return None, args.instruction, instruction_to_target(args.instruction)
    return None, None, args.target


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    config_path = Path(args.config).expanduser().resolve()
    config = load_config(config_path)
    expected_domain = str(config.planner.ros_domain_id)
    if os.environ.get("ROS_DOMAIN_ID") != expected_domain:
        raise RuntimeError(
            f"isolated ApexNav requires ROS_DOMAIN_ID={expected_domain}; use scripts/run_episode.sh"
        )
    task_id, instruction, target = resolve_task(args)
    prior = load_target_prior(
        config.experiment.priors_path,
        target,
        similar_labels=args.similar_label,
        confidence_threshold=args.confidence_threshold,
        room=args.room,
    )
    experiment_dir = experiment_directory(config.experiment.output_dir, args.experiment)
    output_dir = episode_directory(
        args.output_dir, experiment_dir, task_id or task_slug(target), args.start_label
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    version = yor_version(REPOSITORY_ROOT)
    _print_status(
        f"ApexNav experiment {experiment_dir.name}, task {task_id or target!r}, "
        f"start {args.start_label}; episode directory {output_dir}"
    )
    if version["yor_dirty"]:
        _print_status(
            f"ApexNav warning: the YOR checkout at {version['yor_commit']} has uncommitted changes to tracked files"
        )
    (output_dir / "task.json").write_text(
        json.dumps(
            {
                "task_id": task_id,
                "experiment": experiment_dir.name,
                "start_label": args.start_label,
                "instruction": instruction,
                "target": target,
                "prior": prior.as_dict(),
                "ros_domain_id": config.planner.ros_domain_id,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (output_dir / "config.json").write_text(
        json.dumps(config.as_dict(), indent=2), encoding="utf-8"
    )
    debug_dir = None
    if config.experiment.record_debug and not args.no_debug_record:
        debug_dir = output_dir / DEBUG_DIRNAME
        debug_dir.mkdir(parents=True, exist_ok=True)
        # This run's ROS node logs, including the bridge's and the relay's, go
        # next to its other debug data instead of a directory shared by all runs.
        os.environ["ROS_LOG_DIR"] = str((debug_dir / "ros_logs").resolve())
        (debug_dir / "environment.json").write_text(
            json.dumps(
                environment_snapshot(
                    REPOSITORY_ROOT,
                    config_path,
                    {
                        "zed": f"tcp://{config.robot.zed_host}:{config.robot.zed_port}",
                        "base_rpc": (
                            f"tcp://{config.robot.base_rpc_host}:{config.robot.base_rpc_port}"
                        ),
                        "grounding_dino": config.vlm.grounding_dino_url,
                        "blip2_itm": config.vlm.blip2_itm_url,
                        "mobile_sam": config.vlm.mobile_sam_url,
                    },
                ),
                indent=2,
            ),
            encoding="utf-8",
        )

    planner = None
    control_relay = None
    recorders: tuple[subprocess.Popen, ...] = ()
    viewer = None
    viewer_url = None
    viewer_exit_reported = False
    sensor_log = None
    node = None
    executor = None
    reason = "exception"
    started = time.monotonic()
    # Keep the ROS context alive until our finally block has stopped the planner,
    # the base lease, callbacks, and model clients. Python's normal SIGINT then
    # raises KeyboardInterrupt instead of rclpy invalidating publishers first.
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    # A closed terminal or SIGTERM takes the same cleanup path as Ctrl-C, so the
    # planner and base relay, which run in their own sessions, are always stopped.
    signal.signal(signal.SIGTERM, _raise_keyboard_interrupt)
    signal.signal(signal.SIGHUP, _raise_keyboard_interrupt)
    try:
        if debug_dir is not None:
            recorders = _start_debug_recorders(config_path, debug_dir)
            sensor_log = JsonlWriter(debug_dir / "sensor.jsonl")
        if config.viewer.enabled and not args.no_viewer:
            # Started before the planner so it sees topics as they appear.
            try:
                viewer, viewer_url = _start_viewer(
                    config.viewer, output_dir if debug_dir is None else debug_dir
                )
            except Exception as exc:  # noqa: BLE001 - the viewer never blocks an episode
                _print_status(f"ApexNav viewer skipped: {type(exc).__name__}: {exc}")
        if not args.planner_already_running:
            planner = _start_planner(
                config_path,
                None if debug_dir is None else debug_dir / "planner_console.log",
            )
        if not args.no_motion:
            control_relay = _start_control_relay(config_path, output_dir, debug_dir)
        node = YorApexNavBridge(
            config,
            prior,
            output_dir,
            enable_motion=not args.no_motion,
            external_control=not args.no_motion,
            sensor_log=sensor_log,
        )
        # Odometry, mapping, perception, control, and the default subscription
        # group each keep a thread, so a long perception call cannot hold back
        # odometry.
        executor = MultiThreadedExecutor(num_threads=6)
        executor.add_node(node)
        deadline = started + config.planner.maximum_episode_s
        trigger_at = time.monotonic() + config.planner.trigger_delay_s
        while rclpy.ok():
            executor.spin_once(timeout_sec=0.1)
            if planner is not None and planner.poll() is not None:
                reason = f"planner_exit_{planner.returncode}"
                break
            if control_relay is not None and control_relay.poll() is not None:
                reason = f"control_exit_{control_relay.returncode}"
                break
            if (
                viewer is not None
                and not viewer_exit_reported
                and viewer.poll() is not None
            ):
                viewer_exit_reported = True
                _print_status(
                    f"ApexNav viewer exited with code {viewer.returncode}; "
                    "the episode continues"
                )
            # WAIT_TRIGGER is state 1 in upstream's real-world FSM. Publishing
            # before this state is reached is silently discarded.
            now = time.monotonic()
            retry_due = (
                not node.started
                or now - node.last_trigger_monotonic >= config.planner.trigger_retry_s
            )
            if (
                node.final_state == 1
                and not node.trigger_acknowledged
                and node.map_ready
                and node.initial_scan_complete
                and now >= trigger_at
                and retry_due
            ):
                node.start_episode()
            if node.finished.is_set():
                if node.fatal_error:
                    reason = "bridge_failure"
                elif (
                    args.no_motion
                    and node.initial_scan_failed
                    and node.nonzero_command_count > 0
                ):
                    reason = "no_motion_initial_scan_verified"
                else:
                    reason = "apexnav_finished"
                break
            if time.monotonic() >= deadline:
                reason = "episode_timeout"
                break
        if reason == "exception" and not rclpy.ok():
            reason = "operator_interrupt"
    except KeyboardInterrupt:
        reason = "operator_interrupt"
    finally:
        _ignore_interrupts()
        _print_status(
            "ApexNav: stopping the planner, base relay, and recorders; "
            "Ctrl-C is ignored until cleanup finishes"
        )
        # Stop new planner commands first. The Pi's short velocity lease expires
        # independently, so even an unavailable RPC cannot delay planner cleanup.
        _stop_children(planner, control_relay, *recorders, viewer)
        planner = None
        control_relay = None
        if node is not None:
            if executor is not None:
                if not executor.shutdown(timeout_sec=15.0):
                    print("ApexNav executor shutdown warning: callbacks did not finish in 15 s")
            try:
                node.stop(reason)
            except Exception as exc:  # noqa: BLE001 - cleanup must continue
                print(f"ApexNav stop warning: {type(exc).__name__}: {exc}")
            if executor is not None:
                executor.remove_node(node)
            try:
                node.destroy_node()
            except Exception as exc:  # noqa: BLE001 - result must still be written
                print(f"ApexNav destroy warning: {type(exc).__name__}: {exc}")
        if sensor_log is not None:
            sensor_log.close()
        if rclpy.ok():
            rclpy.shutdown()
        control_stats = {}
        control_result_path = output_dir / "control_result.json"
        if control_result_path.exists():
            try:
                control_stats = json.loads(control_result_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as exc:
                print(f"ApexNav control result warning: {exc}")
        node_fatal_error = None if node is None else node.fatal_error
        result = {
            "task_id": task_id,
            "experiment": experiment_dir.name,
            "start_label": args.start_label,
            "termination": reason,
            "reason": reason,
            "elapsed_s": round(time.monotonic() - started, 3),
            "final_state": None if node is None else node.final_state,
            "exploration_result": None if node is None else node.exploration_result,
            "fatal_error": node_fatal_error or control_stats.get("fatal_error"),
            "command_count": int(
                control_stats.get(
                    "command_count", 0 if node is None else node.command_count
                )
            ),
            "nonzero_command_count": int(
                control_stats.get(
                    "nonzero_command_count",
                    0 if node is None else node.nonzero_command_count,
                )
            ),
            "base_submit_count": int(control_stats.get("base_submit_count", 0)),
            "zero_submit_count": int(control_stats.get("zero_submit_count", 0)),
            "watchdog_zero_count": int(control_stats.get("watchdog_zero_count", 0)),
            "stale_command_zero_count": int(
                control_stats.get("stale_command_zero_count", 0)
            ),
            "stale_sensor_zero_count": int(
                control_stats.get("stale_sensor_zero_count", 0)
            ),
            "max_command_gap_s": float(control_stats.get("max_command_gap_s", 0.0)),
            "max_sensor_gap_s": float(control_stats.get("max_sensor_gap_s", 0.0)),
            "map_free_cell_count": 0 if node is None else node.map_free_cell_count,
            "initial_scan_complete": False
            if node is None
            else node.initial_scan_complete,
            "initial_scan_failed": False if node is None else node.initial_scan_failed,
            "debug_dir": None if debug_dir is None else str(debug_dir),
            "viewer_url": viewer_url,
            "viewer_exit_code": None if viewer is None else viewer.poll(),
            "task_success": None,
            "task_success_source": None,
            "adopted": None,
            "exclusion_reason": None,
            "note": None,
            **version,
        }
        (output_dir / "result.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
        print(json.dumps(result, indent=2))
    # Cleanup ignored further interrupts; the review may be interrupted again.
    signal.signal(signal.SIGINT, signal.default_int_handler)
    review = review_episode(
        args.adopt, no_motion=args.no_motion, interactive=sys.stdin.isatty(), log=_print_status
    )
    record_review(output_dir, experiment_dir=experiment_dir, review=review, result=result, log=_print_status)
    failed_process = reason.startswith(("planner_exit_", "control_exit_"))
    return 1 if reason == "bridge_failure" or failed_process else 0


if __name__ == "__main__":
    raise SystemExit(main())
