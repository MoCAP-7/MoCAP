#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
BASELINE_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
UPSTREAM_DIR="${APEXNAV_UPSTREAM_DIR:-/home/yor/codefield/ApexNav}"
DESTINATION="${BASELINE_DIR}/ros2_ws/src/apexnav_upstream"
EXPECTED_COMMIT="1ec9d155fc972fb8a139bd57f43b65247237601e"

if [[ ! -d "${UPSTREAM_DIR}/.git" ]]; then
  echo "ApexNav checkout not found: ${UPSTREAM_DIR}" >&2
  exit 1
fi
actual_commit="$(git -C "${UPSTREAM_DIR}" rev-parse origin/ros2-jazzy)"
if [[ "${actual_commit}" != "${EXPECTED_COMMIT}" ]]; then
  echo "Refusing unreviewed ApexNav ROS2 commit: ${actual_commit}" >&2
  echo "Expected: ${EXPECTED_COMMIT}" >&2
  exit 1
fi
if [[ -e "${DESTINATION}" ]]; then
  echo "Workspace source already exists: ${DESTINATION}" >&2
  echo "It is immutable by design. Move it aside manually before re-preparing." >&2
  exit 1
fi

temporary="$(mktemp -d "${BASELINE_DIR}/ros2_ws/src/.apexnav-export.XXXXXX")"
cleanup() {
  if [[ -d "${temporary}" ]]; then
    rm -rf -- "${temporary}"
  fi
}
trap cleanup EXIT

git -C "${UPSTREAM_DIR}" archive "${EXPECTED_COMMIT}" src/planner \
  | tar -x -C "${temporary}" --strip-components=2
patch --no-backup-if-mismatch --dry-run -p1 -d "${temporary}" < "${BASELINE_DIR}/patches/yor_humble.patch"
patch --no-backup-if-mismatch -p1 -d "${temporary}" < "${BASELINE_DIR}/patches/yor_humble.patch"
patch --no-backup-if-mismatch --dry-run -p1 -d "${temporary}" < "${BASELINE_DIR}/patches/yor_robot_runtime.patch"
patch --no-backup-if-mismatch -p1 -d "${temporary}" < "${BASELINE_DIR}/patches/yor_robot_runtime.patch"
printf '%s\n' "${EXPECTED_COMMIT}" > "${temporary}/.yor_apexnav_upstream_commit"
mv -- "${temporary}" "${DESTINATION}"
trap - EXIT

echo "Prepared isolated ApexNav source: ${DESTINATION}"
