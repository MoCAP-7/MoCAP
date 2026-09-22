#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
BASELINE_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
CHECKOUT="${BASELINE_DIR}/.deps/src/yolov7"
EXPECTED_COMMIT="a207844b1ce82d204ab36d87d496728d3d2348e7"
MODEL_DIR="/home/yor/models/apexnav"
WEIGHTS="${MODEL_DIR}/yolov7-e6e.pt"
EXPECTED_SHA256="b370120a414bf32b5d65fc808e5a32c8d9b3c63902d1bc41894fc9d86eccf9cb"

mkdir -p "$(dirname -- "${CHECKOUT}")" "${MODEL_DIR}"
if [[ ! -d "${CHECKOUT}/.git" ]]; then
  git clone https://github.com/WongKinYiu/yolov7.git "${CHECKOUT}"
  git -C "${CHECKOUT}" checkout "${EXPECTED_COMMIT}"
fi
actual_commit="$(git -C "${CHECKOUT}" rev-parse HEAD)"
if [[ "${actual_commit}" != "${EXPECTED_COMMIT}" ]]; then
  echo "Refusing unreviewed YOLOv7 commit: ${actual_commit}" >&2
  exit 1
fi
if [[ ! -f "${WEIGHTS}" ]]; then
  curl --fail --location --retry 3 \
    --output "${WEIGHTS}.partial" \
    https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-e6e.pt
  mv -- "${WEIGHTS}.partial" "${WEIGHTS}"
fi
actual_sha256="$(sha256sum "${WEIGHTS}" | awk '{print $1}')"
if [[ "${actual_sha256}" != "${EXPECTED_SHA256}" ]]; then
  echo "Refusing unverified YOLOv7 weights: ${actual_sha256}" >&2
  exit 1
fi
echo "YOLOv7 checkout: ${CHECKOUT}"
echo "YOLOv7 weights:  ${WEIGHTS}"
