#!/usr/bin/env bash

set -Eeuo pipefail

usage() {
    echo "Usage: $0 <jetson|pi>" >&2
}

run_service() {
    local service_name="$1"

    # A long-lived tmux server keeps the environment from the shell that
    # created or last updated it. Do not let that leak into the independent
    # SAM3, GraspGenX, or cuRobo virtual environments. Services that need a
    # custom PYTHONPATH set it explicitly below; the nav2 branch rebuilds its
    # ROS environment in start_nav2.sh, including the NumPy-1.x-compatible
    # capx-jetson packages required by its commlink bridges.
    unset PYTHONPATH PYTHONHOME VIRTUAL_ENV CONDA_PREFIX CONDA_DEFAULT_ENV
    export PYTHONNOUSERSITE=1

    case "${service_name}" in
        zed)
            cd /home/yor/codefield/YOR
            export PYTHONPATH=/home/yor/codefield/YOR:/home/yor/codefield/YOR/navdp
            exec /home/yor/venvs/yor-nav/bin/python \
                yor_agent/services/zed/service.py \
                --fps 30 --depth-mode neural --depth-max-m 20.0 \
                --floor-plane-hz 2.0 --floor-plane-max-age-s 2.0 --port 6000
            ;;
        sam3)
            cd /home/yor/codefield/YOR
            # The weights are read from the default Hugging Face cache in
            # offline mode, so a cache location inherited from another
            # project's shell would hide them.
            unset HF_HOME XDG_CACHE_HOME
            export HF_HUB_OFFLINE=1
            exec /home/yor/venvs/capx-perception/bin/python \
                yor_agent/services/sam3/service.py \
                --device cuda --host 127.0.0.1 --port 8114 \
                --model-dtype auto --confidence-threshold 0.05 --max-masks 3
            ;;
        grasp)
            cd /home/yor/codefield/YOR
            exec /home/yor/codefield/YOR/Agent/.venv-graspgenx/bin/python \
                yor_agent/services/grasp/service.py --backend graspgenx \
                --checkpoint-root /home/yor/models/GraspGenXModel/release \
                --host 127.0.0.1 --port 5556
            ;;
        curobo)
            cd /home/yor/codefield/YOR/yor_agent
            exec .venv-grasp-motion/bin/python -u \
                services/grasp_motion/service.py --host 127.0.0.1 --port 5559
            ;;
        nav2)
            cd /home/yor/codefield/YOR
            exec yor_agent/scripts/start_nav2.sh
            ;;
        vggt)
            cd /home/yor/codefield/YOR
            exec /home/yor/venvs/capx-jetson/bin/python \
                yor_agent/services/vggt_stance/service.py \
                --device cuda --dtype float16 --host 127.0.0.1 --port 8117 \
                --weights /home/yor/models/vggt/vggt_1b_fp16.safetensors
            ;;
        base)
            cd /home/cone-e2/YOR
            export PYTHONPATH=/home/cone-e2/YOR:/home/cone-e2/YOR/navdp
            exec /home/cone-e2/miniconda3/envs/yor-nero/bin/python \
                yor_agent/services/base/service.py \
                --port 5557 --lease-ms 250 \
                --max-linear-mps 0.18 --max-yaw-rad-s 0.35 \
                --teleop-max-linear-mps 0.50 --teleop-max-yaw-rad-s 1.57 \
                --max-accel 0.30,0.30,0.80 --teleop-max-accel 1.9,1.9,6.5
            ;;
        nero)
            cd /home/cone-e2/YOR
            export PYTHONPATH=/home/cone-e2/YOR:/home/cone-e2/sdk/pyAgxArm
            local nero_python=/home/cone-e2/miniconda3/envs/nero-official/bin/python

            "${nero_python}" yor_agent/services/nero_arm/service.py \
                --recover-estop-only left \
                --confirm-reset-estop

            "${nero_python}" yor_agent/services/nero_arm/service.py \
                --recover-estop-only right \
                --confirm-reset-estop

            # YOR_NERO_ALLOW_DRAG=1 enables the operator-only zero-force drag
            # mode used by tools/nero_pose_teach.py. It is a permission gate
            # only; nothing enters drag mode without the per-call confirmation.
            local -a nero_extra=()
            if [[ "${YOR_NERO_ALLOW_DRAG:-0}" == "1" ]]; then
                nero_extra+=(--allow-calibration-drag)
            fi
            exec "${nero_python}" yor_agent/services/nero_arm/service.py \
                --port 5558 --speed-percent 20 --open-width-m 0.091 \
                --confirm-enable-and-home ${nero_extra[@]+"${nero_extra[@]}"}
            ;;
        gamepad)
            # A client of the base service's leased RPC, not a service of its
            # own: it waits for the base service and exits when no gamepad is
            # plugged in, leaving the reason in the pane.
            cd /home/cone-e2/YOR
            exec /home/cone-e2/miniconda3/envs/yor-nero/bin/python \
                yor_agent/tools/base_joystick_teleop.py
            ;;
        *)
            echo "Error: unknown internal service '${service_name}'." >&2
            exit 2
            ;;
    esac
}

kill_session_if_present() {
    local session_name="$1"

    if tmux has-session -t "=${session_name}" 2>/dev/null; then
        echo "Stopping existing tmux session: ${session_name}"
        tmux kill-session -t "=${session_name}"
    fi
}

start_session() {
    local session_name="$1"
    local invocation
    local pane_target="${session_name}:0.0"

    printf -v invocation '%q __run_service %q' "${script_path}" "${session_name}"

    echo "Starting tmux session: ${session_name}"
    # Keep an interactive shell as the pane's parent process. If a service
    # exits, crashes, or receives Ctrl-C, the pane returns to this shell instead
    # of destroying the entire tmux session. This also preserves startup errors
    # (especially ZED/PyZED failures) for inspection after the fact.
    tmux new-session -d -s "${session_name}"
    if ! tmux send-keys -t "${pane_target}" -l "${invocation}"; then
        tmux kill-session -t "=${session_name}" 2>/dev/null || true
        return 1
    fi
    if ! tmux send-keys -t "${pane_target}" Enter; then
        tmux kill-session -t "=${session_name}" 2>/dev/null || true
        return 1
    fi
}

script_path="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)/$(basename -- "${BASH_SOURCE[0]}")"

# Private entry point used by the tmux sessions created below. Keeping the
# service commands here avoids relying on tmux-specific multi-argument command
# handling and makes each service process replace its session shell via exec.
if [[ "${1:-}" == "__run_service" ]]; then
    if [[ $# -ne 2 ]]; then
        echo "Error: __run_service requires exactly one service name." >&2
        exit 2
    fi
    run_service "$2"
fi

if [[ $# -ne 1 ]]; then
    usage
    exit 2
fi

if ! command -v tmux >/dev/null 2>&1; then
    echo "Error: tmux is not installed or is not available in PATH." >&2
    exit 1
fi

device="${1,,}"

case "${device}" in
    jetson)
        for session_name in zed sam3 grasp curobo nav2 vggt; do
            kill_session_if_present "${session_name}"
        done

        start_session zed
        start_session sam3
        start_session grasp
        start_session curobo
        start_session nav2

        # The VGGT stance service (readiness prior, port 8117) was opt-in until
        # its footprint on the shared Jetson RAM was known. Measured 2026-09-12
        # with every other service up: 2953 MB peak against 5031 MB available,
        # and all eight services survived the run. Set YOR_START_VGGT=0 to skip
        # it when something else needs that memory.
        if [[ "${YOR_START_VGGT:-1}" == "1" ]]; then
            start_session vggt
        fi
        ;;

    pi)
        # "robot" is the legacy all-in-one session and must not retain hardware
        # ownership while the dedicated base and Nero services are running.
        for session_name in robot gamepad nero base; do
            kill_session_if_present "${session_name}"
        done

        start_session base
        start_session nero

        # The gamepad drives the base through the base service's leased RPC, so
        # it shares the base with the agent and needs neither service stopped
        # (see tools/base_joystick_teleop.py). It sends nothing until Start is
        # pressed and a stick is moved. Set YOR_START_GAMEPAD=0 to skip it.
        if [[ "${YOR_START_GAMEPAD:-1}" == "1" ]]; then
            start_session gamepad
        fi
        ;;

    *)
        echo "Error: unsupported device '${1}'." >&2
        usage
        exit 2
        ;;
esac

echo
echo "Requested service sessions were started for ${device}:"
if [[ "${device}" == "jetson" ]]; then
    if [[ "${YOR_START_VGGT:-1}" == "1" ]]; then
        echo "  zed, sam3, grasp, curobo, nav2, vggt"
    else
        echo "  zed, sam3, grasp, curobo, nav2"
    fi
elif [[ "${YOR_START_GAMEPAD:-1}" == "1" ]]; then
    echo "  base, nero, gamepad"
else
    echo "  base, nero"
fi
echo "Use 'tmux attach -t <session>' to inspect a service."
