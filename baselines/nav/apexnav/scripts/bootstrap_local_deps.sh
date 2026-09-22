#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
BASELINE_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
DEPS_DIR="${BASELINE_DIR}/.deps"
DOWNLOAD_DIR="${DEPS_DIR}/downloads"
SYSROOT="${DEPS_DIR}/sysroot"
PREFIX="${DEPS_DIR}/prefix"
SOURCE_DIR="${DEPS_DIR}/src"

mkdir -p "${DOWNLOAD_DIR}" "${SYSROOT}" "${PREFIX}" "${SOURCE_DIR}"

# ApexNav only uses these PCL components.  Do not ask apt to resolve the
# libpcl-dev metapackage: on Jammy that drags in VTK GUI, Qt WebKit and a JDK.
# The matching CMake patch also requests exactly this component set.
packages=(
  libpcl-dev
  libpcl-common1.12
  libpcl-features1.12
  libpcl-filters1.12
  libpcl-io1.12
  libpcl-kdtree1.12
  libpcl-ml1.12
  libpcl-octree1.12
  libpcl-sample-consensus1.12
  libpcl-search1.12
  libpcl-segmentation1.12
  libopenni0
  libompl-dev
  libompl16
  libode-dev
  libode8
  libccd2
  ros-humble-pcl-conversions
  ros-humble-pcl-msgs
)

download_bytes=0
download_count=0
for package in "${packages[@]}"; do
  if dpkg-query -W -f='${Status}' "${package}" 2>/dev/null | grep -q 'install ok installed'; then
    continue
  fi
  candidate="$(apt-cache policy "${package}" | awk '/Candidate:/ {print $2}')"
  if [[ -z "${candidate}" || "${candidate}" == "(none)" ]]; then
    echo "No apt candidate for required local package: ${package}" >&2
    exit 1
  fi
  size="$(apt-cache show "${package}=${candidate}" | awk '/^Size:/ {print $2; exit}')"
  download_bytes=$((download_bytes + size))
  download_count=$((download_count + 1))
done
if (( download_count > 32 || download_bytes > 1073741824 )); then
  echo "Refusing unexpected dependency set: ${download_count} packages, ${download_bytes} bytes" >&2
  exit 1
fi
echo "Local apt payload: ${download_count} packages, $(numfmt --to=iec "${download_bytes}")"

pushd "${DOWNLOAD_DIR}" >/dev/null
for package in "${packages[@]}"; do
  if dpkg-query -W -f='${Status}' "${package}" 2>/dev/null | grep -q 'install ok installed'; then
    continue
  fi
  if [[ "$(apt-cache policy "${package}" | awk '/Candidate:/ {print $2}')" == "(none)" ]]; then
    continue
  fi
  if compgen -G "${package}_*.deb" >/dev/null; then
    continue
  fi
  echo "Downloading isolated dependency: ${package}"
  apt download "${package}"
done
popd >/dev/null

for archive in "${DOWNLOAD_DIR}"/*.deb; do
  dpkg-deb -x "${archive}" "${SYSROOT}"
done

pcl_exports="${SYSROOT}/opt/ros/humble/share/pcl_conversions/cmake/ament_cmake_export_dependencies-extras.cmake"
if ! grep -q 'if(_dep STREQUAL "PCL")' "${pcl_exports}"; then
  patch -p1 -d "${SYSROOT}" < "${BASELINE_DIR}/patches/pcl_conversions_components.patch"
fi

build_cmake_dependency() {
  local name="$1"
  local repository="$2"
  local tag="$3"
  local expected_commit="$4"
  local actual_commit
  shift 4
  local source="${SOURCE_DIR}/${name}"
  local build="${DEPS_DIR}/build-${name}"
  if [[ ! -d "${source}/.git" ]]; then
    git clone --depth 1 --branch "${tag}" "${repository}" "${source}"
  fi
  actual_commit="$(git -C "${source}" rev-parse HEAD)"
  if [[ "${actual_commit}" != "${expected_commit}" ]]; then
    echo "Refusing unreviewed ${name} commit: ${actual_commit}" >&2
    exit 1
  fi
  if [[ -f "${source}/.gitmodules" ]]; then
    git -C "${source}" submodule update --init --depth 1
  fi
  cmake -S "${source}" -B "${build}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="${PREFIX}" \
    "$@"
  cmake --build "${build}" --parallel "$(nproc)"
  cmake --install "${build}"
}

build_cmake_dependency osqp https://github.com/osqp/osqp.git v0.6.3 \
  0dd00a578cf1c2691c5c379965d504c75bf6cfad \
  -DBUILD_SHARED_LIBS=ON -DOSQP_BUILD_DEMO_EXE=OFF -DENABLE_MKL_PARDISO=OFF

export CMAKE_PREFIX_PATH="${PREFIX}:${SYSROOT}/usr:${SYSROOT}/opt/ros/humble:${CMAKE_PREFIX_PATH:-}"
build_cmake_dependency osqp-eigen https://github.com/robotology/osqp-eigen.git v0.8.1 \
  85c37623774c682db396505f0d4ea677040c2557 \
  -DBUILD_TESTING=OFF

echo "Local ApexNav dependencies installed below ${DEPS_DIR}"
echo "No dpkg database or /opt/ros files were modified."
