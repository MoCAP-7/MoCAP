#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
BASELINE_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
REPOSITORY_ROOT="$(cd -- "${BASELINE_DIR}/../../.." && pwd -P)"
source "${SCRIPT_DIR}/source_env.sh"

PROFILE="open-vocabulary"
NO_MOTION=0
while (( $# )); do
  case "$1" in
    --profile)
      if [[ -z "${2:-}" ]]; then
        echo "--profile requires open-vocabulary or coco" >&2
        exit 2
      fi
      PROFILE="$2"
      shift 2
      ;;
    --no-motion)
      NO_MOTION=1
      shift
      ;;
    *)
      echo "Usage: $0 [--profile open-vocabulary|coco] [--no-motion]" >&2
      exit 2
      ;;
  esac
done

failures=0
check_port() {
  local name="$1"
  local port="$2"
  if nc -z -w1 127.0.0.1 "${port}" >/dev/null 2>&1; then
    echo "OK   ${name} port ${port}"
  else
    echo "FAIL ${name} port ${port}" >&2
    failures=$((failures + 1))
  fi
}

echo "ROS_DOMAIN_ID=${ROS_DOMAIN_ID}"
for package in plan_env exploration_manager trajectory_manager yor_apexnav_bringup; do
  if ros2 pkg prefix "${package}" >/dev/null 2>&1; then
    echo "OK   ROS package ${package}"
  else
    echo "FAIL ROS package ${package}" >&2
    failures=$((failures + 1))
  fi
done
check_port ZED 6000
check_port BLIP2-ITM 12182
check_port MobileSAM 12183
case "${PROFILE}" in
  open-vocabulary)
    check_port GroundingDINO 12181
    ;;
  coco)
    check_port YOLOv7 12184
    ;;
  *)
    echo "Unknown model profile: ${PROFILE}" >&2
    exit 2
    ;;
esac

if (( NO_MOTION )); then
  echo "SKIP Pi base RPC (--no-motion)"
elif ! python - "${BASELINE_DIR}/config.yaml" <<'PY'
import sys

import zmq
from commlink import RPCClient
from baselines.nav.apexnav.config import load_config

config = load_config(sys.argv[1])
client = RPCClient(
    host=config.robot.base_rpc_host,
    port=config.robot.base_rpc_port,
)
timeout_ms = int(round(config.robot.base_rpc_timeout_s * 1000.0))
client.socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
client.socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
client.socket.setsockopt(zmq.LINGER, 0)
try:
    status = client.get_status()
except Exception as exc:
    raise SystemExit(f"FAIL Pi base RPC unavailable after {config.robot.base_rpc_timeout_s:g} s: {type(exc).__name__}") from exc
if status.get("estop_latched"):
    raise SystemExit("FAIL Pi emergency stop is latched")
if status.get("lease_active"):
    raise SystemExit("FAIL another process owns the base lease")
print("OK   Pi base RPC is idle and not e-stopped")
PY
then
  failures=$((failures + 1))
fi

cd "${REPOSITORY_ROOT}"
python - <<'PY'
from baselines.nav.apexnav.config import load_config
config = load_config("baselines/nav/apexnav/config.yaml")
print(f"OK   configuration validated; isolated domain={config.planner.ros_domain_id}")
PY

# The live viewer is optional: an episode runs without it, so only warn.
read -r viewer_enabled viewer_port < <(python - <<'PY'
from baselines.nav.apexnav.config import load_config
viewer = load_config("baselines/nav/apexnav/config.yaml").viewer
print(int(viewer.enabled), viewer.port)
PY
)
if (( viewer_enabled )); then
  if [[ -x "${BASELINE_DIR}/.deps/sysroot/opt/ros/humble/lib/foxglove_bridge/foxglove_bridge" ]]; then
    echo "OK   live viewer bridge"
  else
    echo "WARN live viewer bridge missing; episodes run without it (scripts/bootstrap_viewer_deps.sh)"
  fi
  if ss -ltnH "sport = :${viewer_port}" 2>/dev/null | grep -q .; then
    echo "WARN port ${viewer_port} is already in use (a leftover foxglove_bridge?); the viewer will not start"
  fi
fi

if (( failures > 0 )); then
  echo "ApexNav preflight failed with ${failures} missing component(s)." >&2
  exit 1
fi
echo "ApexNav preflight passed."
