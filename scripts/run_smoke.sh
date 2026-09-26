#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source .venv/bin/activate

pytest tests/unit tests/causal tests/leaky_quarantine -q
python -m wharton_lab.orchestration.cli smoke
bash scripts/build_reports.sh

echo "Smoke profile finished."
