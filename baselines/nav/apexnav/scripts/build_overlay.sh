#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
BASELINE_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
WORKSPACE="${BASELINE_DIR}/ros2_ws"
DEPS_DIR="${BASELINE_DIR}/.deps"

if [[ ! -d "${WORKSPACE}/src/apexnav_upstream" ]]; then
  "${SCRIPT_DIR}/prepare_workspace.sh"
fi
if [[ ! -d "${DEPS_DIR}/prefix" || ! -d "${DEPS_DIR}/sysroot" ]]; then
  echo "Missing isolated native dependencies." >&2
  echo "Run: ${SCRIPT_DIR}/bootstrap_local_deps.sh" >&2
  exit 1
fi

# ROS setup scripts reference optional variables and are not nounset-safe.
set +u
source /opt/ros/humble/setup.bash
set -u
export CMAKE_PREFIX_PATH="${DEPS_DIR}/prefix:${DEPS_DIR}/sysroot/usr:${DEPS_DIR}/sysroot/opt/ros/humble:${CMAKE_PREFIX_PATH:-}"
export LD_LIBRARY_PATH="${DEPS_DIR}/prefix/lib:${DEPS_DIR}/sysroot/usr/lib/aarch64-linux-gnu:${DEPS_DIR}/sysroot/opt/ros/humble/lib:${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="${DEPS_DIR}/prefix/lib:${DEPS_DIR}/sysroot/usr/lib/aarch64-linux-gnu:${LIBRARY_PATH:-}"
export PKG_CONFIG_PATH="${DEPS_DIR}/prefix/lib/pkgconfig:${DEPS_DIR}/sysroot/usr/lib/aarch64-linux-gnu/pkgconfig:${PKG_CONFIG_PATH:-}"

cd "${WORKSPACE}"
colcon --log-base log build \
  --build-base build \
  --install-base install \
  --symlink-install \
  --cmake-args -DCMAKE_BUILD_TYPE=Release
