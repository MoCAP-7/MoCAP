#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
BASELINE_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
UPSTREAM_DIR="${COW_UPSTREAM_DIR:-/home/yor/codefield/cow}"
lock_value() {
  python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])' \
    "${BASELINE_DIR}/upstream.lock" "$1"
}
REPOSITORY="$(lock_value repository)"
EXPECTED_COMMIT="$(lock_value commit)"

if [[ ! -d "${UPSTREAM_DIR}/.git" ]]; then
  git clone --quiet "${REPOSITORY}" "${UPSTREAM_DIR}"
fi
if [[ -n "$(git -C "${UPSTREAM_DIR}" status --porcelain --untracked-files=no)" ]]; then
  echo "CoW checkout has local modifications: ${UPSTREAM_DIR}" >&2
  echo "The baseline runs CoW unmodified; restore the checkout before continuing." >&2
  exit 1
fi
if ! git -C "${UPSTREAM_DIR}" cat-file -e "${EXPECTED_COMMIT}^{commit}" 2>/dev/null; then
  git -C "${UPSTREAM_DIR}" fetch --quiet origin
fi
git -C "${UPSTREAM_DIR}" -c advice.detachedHead=false checkout --quiet "${EXPECTED_COMMIT}"
echo "CoW checkout ready at ${EXPECTED_COMMIT}: ${UPSTREAM_DIR}"
