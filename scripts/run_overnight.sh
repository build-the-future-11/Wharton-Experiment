#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source .venv/bin/activate

HOURS=8
RESUME=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --hours) HOURS="$2"; shift 2 ;;
    --resume) RESUME="--resume"; shift ;;
    *) echo "Unknown arg: $1"; exit 2 ;;
  esac
done

python -m wharton_lab.orchestration.cli overnight --hours "$HOURS" $RESUME
