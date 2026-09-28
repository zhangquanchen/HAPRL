#!/usr/bin/env bash
set -euo pipefail

CODE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_KIND="${1:-}"

case "${ENV_KIND}" in
  sft)
    python -m pip install --no-build-isolation -r "${CODE_ROOT}/requirements/sft.txt"
    ;;
  rl)
    python -m pip install --no-build-isolation -r "${CODE_ROOT}/requirements/rl.txt"
    python -m pip install --no-deps -e "${CODE_ROOT}/third_party/Mini-o3"
    ;;
  *)
    echo "usage: $0 {sft|rl}" >&2
    echo "Run each option in a separate Python 3.11 environment." >&2
    exit 2
    ;;
esac
