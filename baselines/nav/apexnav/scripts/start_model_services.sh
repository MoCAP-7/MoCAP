#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
BASELINE_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
REPOSITORY_ROOT="$(cd -- "${BASELINE_DIR}/../../.." && pwd -P)"
MODEL_ENV="/home/yor/venvs/vlfm-models"
MODEL_ROOT="/home/yor/models/vlfm"
ALL_SESSIONS=(apexnav-gdino apexnav-blip2itm apexnav-mobile-sam apexnav-yolov7)
PROFILE="open-vocabulary"
BLIP2_DEVICE="cpu"
STOP=0

usage() {
  echo "Usage: $0 [--profile open-vocabulary|coco] [--blip2-device cpu|cuda] [--stop]" >&2
}

while (( $# )); do
  case "$1" in
    --profile)
      [[ $# -ge 2 ]] || { usage; exit 2; }
      PROFILE="$2"
      shift 2
      ;;
    --blip2-device)
      [[ $# -ge 2 ]] || { usage; exit 2; }
      BLIP2_DEVICE="$2"
      shift 2
      ;;
    --stop)
      STOP=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      usage
      exit 2
      ;;
  esac
done

if (( STOP )); then
  for session in "${ALL_SESSIONS[@]}"; do
    tmux kill-session -t "=${session}" 2>/dev/null || true
  done
  exit 0
fi

case "${BLIP2_DEVICE}" in
  cpu|cuda) ;;
  *)
    usage
    exit 2
    ;;
esac

case "${PROFILE}" in
  open-vocabulary)
    # Descriptive targets use GroundingDINO, MobileSAM, and BLIP2-ITM.
    # Loading YOLOv7 as well exceeds the 16 GB Jetson's unified-memory budget.
    # Keep the exact upstream BLIP2 model. It runs on CPU by default so its
    # CUDA/nvmap allocation cannot evict the detector or be selected by the OOM
    # killer; --blip2-device cuda is far faster once no other GPU services run.
    # GroundingDINO uses ApexNav's own server: VLFM's drops every detection for
    # ApexNav-style captions and ignores the client's thresholds.
    SERVICES=(
      "apexnav-gdino:12181:baselines.nav.apexnav.gdino_server:"
      "apexnav-mobile-sam:12183:baselines.nav.vlfm.model_server:mobile-sam"
      "apexnav-blip2itm:12182:baselines.nav.apexnav.blip2_server:--device=${BLIP2_DEVICE}"
    )
    ;;
  coco)
    SERVICES=(
      "apexnav-blip2itm:12182:baselines.nav.apexnav.blip2_server:--device=${BLIP2_DEVICE}"
      "apexnav-mobile-sam:12183:baselines.nav.vlfm.model_server:mobile-sam"
      "apexnav-yolov7:12184:baselines.nav.apexnav.model_server:"
    )
    ;;
  *)
    usage
    exit 2
    ;;
esac

if [[ ! -x "${MODEL_ENV}/bin/python" ]]; then
  echo "Missing isolated model environment: ${MODEL_ENV}" >&2
  exit 1
fi

# Refuse mixed profiles and stale model processes. They silently consume the
# same unified CPU/GPU memory even if the current target never calls them.
for session in "${ALL_SESSIONS[@]}"; do
  if tmux has-session -t "=${session}" 2>/dev/null; then
    echo "Session already exists: ${session}; use --stop first" >&2
    exit 1
  fi
done
for port in 12181 12182 12183 12184; do
  if nc -z -w1 127.0.0.1 "${port}" >/dev/null 2>&1; then
    echo "Port already in use: ${port}" >&2
    exit 1
  fi
done

launch_session() {
  local name="$1"
  shift
  local command
  printf -v command '%q ' "$@"
  tmux new-session -d -s "${name}" -c "${REPOSITORY_ROOT}"
  tmux send-keys -t "${name}:0.0" -l "${command}"
  tmux send-keys -t "${name}:0.0" Enter
}

wait_for_service() {
  local session="$1"
  local port="$2"
  local deadline=$((SECONDS + 360))
  while (( SECONDS < deadline )); do
    if nc -z -w1 127.0.0.1 "${port}" >/dev/null 2>&1; then
      echo "Ready: ${session} port ${port}"
      return 0
    fi
    if ! tmux has-session -t "=${session}" 2>/dev/null; then
      echo "Model session disappeared during startup: ${session}" >&2
      return 1
    fi
    if [[ "$(tmux display-message -p -t "=${session}" '#{pane_current_command}')" == "bash" ]]; then
      echo "Model process exited during startup: ${session}" >&2
      tmux capture-pane -p -S -40 -t "=${session}" >&2 || true
      return 1
    fi
    sleep 1
  done
  echo "Timed out waiting for ${session} port ${port}" >&2
  return 1
}

export PYTHONPATH="${REPOSITORY_ROOT}"
export TORCH_HOME="${MODEL_ROOT}/torch"
export HF_HOME="${MODEL_ROOT}/huggingface"
export TRANSFORMERS_CACHE="${MODEL_ROOT}/huggingface/hub"
export XDG_CACHE_HOME="${MODEL_ROOT}/cache"
export MPLCONFIGDIR="${MODEL_ROOT}/cache/matplotlib"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

environment=(
  env
  "PYTHONPATH=${PYTHONPATH}"
  "TORCH_HOME=${TORCH_HOME}"
  "HF_HOME=${HF_HOME}"
  "TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE}"
  "XDG_CACHE_HOME=${XDG_CACHE_HOME}"
  "MPLCONFIGDIR=${MPLCONFIGDIR}"
  "HF_HUB_OFFLINE=${HF_HUB_OFFLINE}"
  "TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE}"
)

started_sessions=()
cleanup_failed_start() {
  local session
  for session in "${started_sessions[@]}"; do
    tmux kill-session -t "=${session}" 2>/dev/null || true
  done
}
trap cleanup_failed_start ERR

# Load one model at a time. Concurrent initialization was the source of the
# observed Jetson OOM kill, even before the steady-state episode began.
for service in "${SERVICES[@]}"; do
  IFS=: read -r session port module argument <<<"${service}"
  command=("${environment[@]}" "${MODEL_ENV}/bin/python" -m "${module}")
  if [[ -n "${argument}" ]]; then
    command+=("${argument}")
  fi
  echo "Starting: ${session}"
  launch_session "${session}" "${command[@]}"
  started_sessions+=("${session}")
  wait_for_service "${session}" "${port}"
done

trap - ERR
echo "Started ${PROFILE} profile: ${started_sessions[*]}"
