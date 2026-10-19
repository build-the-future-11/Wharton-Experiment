import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from frem_s11.admission import MemorySample
from frem_s11.artifacts import write_development_bundle
from frem_s11.backtest import VARIANTS, BacktestResult
from frem_s11.data import ReturnPanel


def _result() -> BacktestResult:
    return BacktestResult(
        months=np.asarray([200001, 200002]),
        scores=np.ones((2, len(VARIANTS))),
        stack_weights=np.full((2, 4), 0.25),
        allocations=np.zeros((2, len(VARIANTS), 7)),
        wealth=np.ones((2, len(VARIANTS))),
        benchmark_wealth=np.ones((2, 2)),
        intervals={"equal_minus_stacked": (0.0, 0.0), "recent_minus_stacked": (0.0, 0.0)},
        relative_improvements={"equal": 0.0, "recent": 0.0},
        primary_supported=False,
        secondary_supported=False,
    )


def test_bundle_retains_arrays_hashes_and_negative_disposition(tmp_path: Path) -> None:
    months = np.asarray(
        [1998 * 100 + month for month in range(1, 13)] + [1999 * 100 + month for month in range(1, 13)]
    )
    panel = ReturnPanel(months, np.zeros((24, 6)), 200001, hashlib.sha256(b"source").hexdigest())
    sample = MemorySample("now", 16384, 40000, 1000, 2)
    admission = {"samples": [{}, {}, {"pageouts": 2}], "status": "PASS"}
    output = tmp_path / "bundle"
    summary = write_development_bundle(
        output,
        panel,
        ("A",),
        (_result(),),
        protocol_bytes=b"protocol",
        admission=admission,
        post_sample=sample,
        provenance={},
    )
    assert summary["scientific_status"] == "DEVELOPMENT_PRIMARY_NOT_SUPPORTED_STOP_S11"
    assert not (output / "INCOMPLETE.json").exists()
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert "fold_A_scores.npy" in manifest["files"]
    with pytest.raises(FileExistsError):
        write_development_bundle(
            output,
            panel,
            ("A",),
            (_result(),),
            protocol_bytes=b"protocol",
            admission=admission,
            post_sample=sample,
            provenance={},
        )
