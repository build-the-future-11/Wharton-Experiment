# Reproducibility Audit

## Environment

- Host: Apple M4, 16 GB RAM, MPS available (orchestration defaults CPU for determinism)
- Python: project `.venv` via `scripts/bootstrap.sh`
- OS: darwin (see host)

## Determinism controls

- Seeds: smoke `[11]`; pilot `[11,23,47]`; full `[11,23,47,89,131]` (`protocol/DECISIONS.md`)
- Manifest rows carry `run_id`, `config_hash` (when written), `result_path`
- Resume skips `COMPLETED` only

## Reproduce smoke

```bash
cd /Volumes/PRO-BLADE/Wharton-Experiments
bash scripts/bootstrap.sh
bash scripts/run_smoke.sh
bash scripts/build_reports.sh
```

## Reproduce one real-data comparison + one ablation

```bash
# Ridge vs M04 on ETF track (if Yahoo available) — from saved pilot receipts:
python - <<'PY'
import json
from pathlib import Path
for rid in [
  "pilot_B_RIDGE_default_etf_track_a_f0_s11_h20",
  "pilot_M04_default_etf_track_a_f0_s11_h20",
  "pilot_M01_ablation_no_regime_synthetic_track_c_f0_s11_h20",
]:
    p = Path("runs/pilot")/rid/"receipt.json"
    print(rid, json.loads(p.read_text())["metrics"])
PY
```

Ablation example: `pilot_M01_ablation_no_regime_synthetic_track_c_f0_s11_h20` vs `pilot_M01_default_synthetic_track_c_f0_s11_h20`.

## Gaps

- No git commit hash frozen yet (repo has no commits).
- Upstream lineage trees not vendored; mechanism reimplementation disclosed in `protocol/DECISIONS.md`.
- Full matrix not executed → cannot claim bit-for-bit FULL reproduction.
