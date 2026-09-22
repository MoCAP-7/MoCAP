#!/usr/bin/env bash
# Source this file only in a dedicated ApexNav shell.

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "source_env.sh must be sourced, not executed" >&2
  exit 2
fi

APEXNAV_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
APEXNAV_BASELINE_DIR="$(cd -- "${APEXNAV_SCRIPT_DIR}/.." && pwd -P)"
APEXNAV_REPOSITORY_ROOT="$(cd -- "${APEXNAV_BASELINE_DIR}/../../.." && pwd -P)"
APEXNAV_DEPS_DIR="${APEXNAV_BASELINE_DIR}/.deps"

APEXNAV_NOUNSET_WAS_ON=0
if [[ "$-" == *u* ]]; then
  APEXNAV_NOUNSET_WAS_ON=1
  set +u
fi
source /home/yor/venvs/yor-nav/bin/activate
source /opt/ros/humble/setup.bash
if [[ ! -f "${APEXNAV_BASELINE_DIR}/ros2_ws/install/setup.bash" ]]; then
  echo "ApexNav overlay is not built: ${APEXNAV_BASELINE_DIR}/ros2_ws/install" >&2
  if (( APEXNAV_NOUNSET_WAS_ON )); then
    set -u
  fi
  unset APEXNAV_SCRIPT_DIR APEXNAV_REPOSITORY_ROOT APEXNAV_DEPS_DIR APEXNAV_NOUNSET_WAS_ON
  return 1
fi
source "${APEXNAV_BASELINE_DIR}/ros2_ws/install/setup.bash"
if (( APEXNAV_NOUNSET_WAS_ON )); then
  set -u
fi

export ROS_DOMAIN_ID=47
export ROS_LOG_DIR="${APEXNAV_BASELINE_DIR}/ros2_ws/log/runtime"
mkdir -p "${ROS_LOG_DIR}"
export PYTHONNOUSERSITE=1
export PYTHONPATH="${APEXNAV_REPOSITORY_ROOT}:${APEXNAV_REPOSITORY_ROOT}/navdp:${PYTHONPATH:-}"
export CMAKE_PREFIX_PATH="${APEXNAV_DEPS_DIR}/prefix:${APEXNAV_DEPS_DIR}/sysroot/usr:${APEXNAV_DEPS_DIR}/sysroot/opt/ros/humble:${CMAKE_PREFIX_PATH:-}"
export LD_LIBRARY_PATH="${APEXNAV_DEPS_DIR}/prefix/lib:${APEXNAV_DEPS_DIR}/sysroot/usr/lib/aarch64-linux-gnu:${APEXNAV_DEPS_DIR}/sysroot/opt/ros/humble/lib:${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="${APEXNAV_DEPS_DIR}/prefix/lib:${APEXNAV_DEPS_DIR}/sysroot/usr/lib/aarch64-linux-gnu:${LIBRARY_PATH:-}"
export PKG_CONFIG_PATH="${APEXNAV_DEPS_DIR}/prefix/lib/pkgconfig:${APEXNAV_DEPS_DIR}/sysroot/usr/lib/aarch64-linux-gnu/pkgconfig:${PKG_CONFIG_PATH:-}"

unset APEXNAV_SCRIPT_DIR APEXNAV_REPOSITORY_ROOT APEXNAV_DEPS_DIR APEXNAV_NOUNSET_WAS_ON
