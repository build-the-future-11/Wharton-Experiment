import csv
import json

import numpy as np

from wharton_lab.evaluation.multiplicity import benjamini_hochberg, paired_vs_persistence
from wharton_lab.evaluation.repaired import diebold_mariano


def _write_manifest(root, cells):
    runs = root / "runs" / "full"
    rows = []
    for i, (ds, fold, seed, mid, mse) in enumerate(cells):
        rp = runs / f"r{i}" / "receipt.json"
        rp.parent.mkdir(parents=True)
        rp.write_text(json.dumps({"metrics": {"primary_value": mse}}))
        rows.append(
            {
                "model_id": mid,
                "dataset": ds,
                "fold": fold,
                "seed": seed,
                "horizon": 20,
                "profile": "full",
                "primary_metric": "mse",
                "status": "COMPLETED",
                "result_path": str(rp.relative_to(root)),
            }
        )
    (root / "protocol").mkdir()
    with (root / "protocol" / "EXPERIMENT_MANIFEST.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def test_duplicate_cells_collapse_and_become_untestable(tmp_path):
    cells = []
    for fold in range(4):
        for seed in (11, 23, 47, 89, 131):
            cells.append(("etf_track_a", fold, seed, "B_RIDGE", 1.0))
            cells.append(("etf_track_a", fold, seed, "B_PERSISTENCE", 2.0))
    _write_manifest(tmp_path, cells)
    (t,) = paired_vs_persistence(tmp_path, challengers=("B_RIDGE",))
    assert t.n_raw_pairs == 20
    assert t.n_unique_pairs == 1
    assert not t.testable and t.wilcoxon_p is None and not t.significant_at_0_05


def test_bh_is_monotone_and_bounded():
    q = benjamini_hochberg([0.01, 0.04, 0.03, 0.5])
    assert all(0 <= x <= 1 for x in q)
    assert q[0] <= q[2] <= q[1] <= q[3]


def test_diebold_mariano_detects_shift_and_accepts_null():
    rng = np.random.default_rng(0)
    _, p_null = diebold_mariano(rng.normal(size=2000))
    _, p_alt = diebold_mariano(rng.normal(loc=0.3, size=2000))
    assert p_null > 0.01
    assert p_alt < 1e-6
