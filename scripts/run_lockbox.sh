#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source .venv/bin/activate
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
# Freeze once (no-op if already frozen — freeze script errors if exists; handle)
if [[ ! -f protocol/LOCKBOX_FREEZE.yaml ]]; then
  python -m wharton_lab.orchestration.lockbox freeze
fi
python -m wharton_lab.orchestration.lockbox run
python -m wharton_lab.reports.build
