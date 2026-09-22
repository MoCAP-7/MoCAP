#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
BASELINE_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
DEPS_DIR="${BASELINE_DIR}/.deps"
DOWNLOAD_DIR="${DEPS_DIR}/downloads"
SYSROOT="${DEPS_DIR}/sysroot"
BRIDGE="${SYSROOT}/opt/ros/humble/lib/foxglove_bridge/foxglove_bridge"

# The live viewer needs only this package at runtime: its other dependencies
# are headers or static libraries, or already part of ROS Humble. The ROS
# repository keeps only the newest build, so `apt download` fails once the
# robot's apt lists fall behind; fetch the reviewed build from a dated snapshot
# instead. upstream.lock records the same pin.
DEB_URL="http://snapshots.ros.org/humble/2026-08-07/ubuntu/pool/main/r/ros-humble-foxglove-bridge/ros-humble-foxglove-bridge_3.4.3-2jammy.20260725.213251_arm64.deb"
DEB_SIZE=11611234
DEB_SHA256="dc70642916382bc9d12065ad04373938c69800490ceecbaed068dfa81e5d708b"
ARCHIVE="${DOWNLOAD_DIR}/$(basename "${DEB_URL}")"

matches_pin() {
  [[ -f "$1" && "$(stat -c %s "$1")" == "${DEB_SIZE}" ]] \
    && echo "${DEB_SHA256}  $1" | sha256sum --check --status -
}

mkdir -p "${DOWNLOAD_DIR}" "${SYSROOT}"
if [[ -f "${ARCHIVE}" ]] && ! matches_pin "${ARCHIVE}"; then
  echo "Discarding ${ARCHIVE}: size or SHA256 differs from the pin" >&2
  rm -f "${ARCHIVE}"
fi
if [[ ! -f "${ARCHIVE}" ]]; then
  echo "Downloading pinned live viewer bridge: $(basename "${ARCHIVE}")"
  curl --fail --location --silent --show-error --max-filesize "${DEB_SIZE}" \
    -o "${ARCHIVE}.part" "${DEB_URL}"
  if ! matches_pin "${ARCHIVE}.part"; then
    rm -f "${ARCHIVE}.part"
    echo "Refusing live viewer bridge download: size or SHA256 differs from the pin" >&2
    exit 1
  fi
  mv "${ARCHIVE}.part" "${ARCHIVE}"
fi

dpkg-deb -x "${ARCHIVE}" "${SYSROOT}"
if [[ ! -x "${BRIDGE}" ]]; then
  echo "The unpacked package has no executable at ${BRIDGE}" >&2
  exit 1
fi
echo "Live viewer bridge installed: ${BRIDGE}"
echo "No dpkg database or /opt/ros files were modified."
