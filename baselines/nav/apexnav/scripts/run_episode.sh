#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
BASELINE_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
REPOSITORY_ROOT="$(cd -- "${BASELINE_DIR}/../../.." && pwd -P)"

unset PYTHONHOME CONDA_PREFIX CONDA_DEFAULT_ENV
source "${SCRIPT_DIR}/source_env.sh"
cd "${REPOSITORY_ROOT}"
exec python -m baselines.nav.apexnav.run \
  --config "${BASELINE_DIR}/config.yaml" \
  "$@"
