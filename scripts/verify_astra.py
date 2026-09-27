"""Verify preserved evidence and reproduce the exploratory baseline in isolation."""
import hashlib
import json
from pathlib import Path

import numpy as np

from wharton_lab.evaluation.repaired import run_repaired_eval
from wharton_lab.models.m11.mpc import _scenario_cash_paths

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "audit/astra_2026-09-27"


def main():
    start = json.loads((OUT / "START_STATE.json").read_text())
    preserved = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() == sha
                 for p, sha in start["tracked_sha256"].items()
                 if p.startswith(("runs/", "src/wharton_lab/models/", "protocol/LOCKBOX", "protocol/EXPERIMENT_MANIFEST"))}
    assert all(preserved.values()), "Historical evidence or frozen model changed"
    scratch = ROOT / ".local_tmp/astra/repaired_replay"
    (scratch / "reports/evidence_bank").mkdir(parents=True, exist_ok=True)
    if not (scratch / "data").exists():
        (scratch / "data").symlink_to(ROOT / "data", target_is_directory=True)
    original = json.loads((ROOT / "runs/repaired/REPAIRED_RESULTS.json").read_text())
    replay = run_repaired_eval(scratch)
    differences = []
    for a, b in zip(original["contrasts"], replay["contrasts"]):
        for key in a:
            if isinstance(a[key], (float, int)):
                if not np.isclose(a[key], b[key], atol=1e-12, rtol=1e-9, equal_nan=True):
                    differences.append([a["dataset"], a["challenger"], key, a[key], b[key]])
            elif a[key] != b[key]:
                differences.append([a["dataset"], key])
    assert len(original["contrasts"]) == len(replay["contrasts"])
    assert not differences, differences
    # Diagnostic only: preserve the frozen M11 implementation and exhibit its defect.
    legacy_cash = _scenario_cash_paths(np.zeros((2, 2)), np.zeros((1, 2, 2)),
                                      1.0, np.zeros(2), np.array([.1, .1]), 0.0)
    corrected_cash = np.array([[.9, .8]])
    out = {"source_head_at_start": start["head"], "historical_files_preserved": len(preserved),
           "all_preserved": all(preserved.values()), "repaired_contrasts_reproduced": len(replay["contrasts"]),
           "differences": differences, "scope": "same-machine repeatability only; not independent reproduction",
           "m11_accounting_diagnostic": {"zero_returns_all_cash_payments": [.1, .1],
                                         "legacy_reported_cash": legacy_cash.tolist(),
                                         "cash_after_single_deduction": corrected_cash.tolist(),
                                         "legacy_matches": bool(np.allclose(legacy_cash, corrected_cash))}}
    (OUT / "VERIFICATION.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
