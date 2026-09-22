#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../../.." && pwd -P)"
PYTHON="${COW_PYTHON:-/home/yor/venvs/capx-jetson/bin/python}"

export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/navdp:${REPO_ROOT}/yor_agent/src${PYTHONPATH:+:${PYTHONPATH}}"
cd "${REPO_ROOT}"
exec "${PYTHON}" -m baselines.nav.cow.run "$@"
