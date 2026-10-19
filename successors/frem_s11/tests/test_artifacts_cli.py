import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from frem_s11.admission import MemorySample
from frem_s11.artifacts import write_development_bundle
from frem_s11.backtest import VARIANTS, BacktestResult
from frem_s11.data import ReturnPanel


def _result() -> BacktestResult:
    months = np.asarray(
        [2000 * 100 + month for month in range(1, 13)] + [2001 * 100 + month for month in range(1, 13)]
    )
    count = len(months)
    allocations = np.zeros((count, len(VARIANTS), 7))
    allocations[:, :, 0] = 1
    return BacktestResult(
        months,
        np.full((count, len(VARIANTS)), 0.1),
        np.full((count, 4), 0.25),
        allocations,
        np.ones((count, len(VARIANTS))),
        np.ones((count, 2)),
        {"equal_minus_stacked": (-0.01, 0.01), "recent_minus_stacked": (-0.01, 0.01)},
        {"equal": 0.0, "recent": 0.0},
        False,
        False,
    )


def test_bundle_retains_allocations_hashes_and_resource_disposition(tmp_path: Path) -> None:
    result = _result()
    panel = ReturnPanel(
        result.months,
        np.zeros((len(result.months), 6)),
        200201,
        hashlib.sha256(b"panel").hexdigest(),
    )
    sample = MemorySample("2026-10-18T00:00:00+00:00", 16384, 40000, 0, 7)
    admission = {"samples": [{"pageouts": 7}, {"pageouts": 7}, {"pageouts": 7}]}
    summary = write_development_bundle(
        tmp_path / "bundle",
        panel,
        ("A",),
        (result,),
        protocol_bytes=b"generated protocol",
        admission=admission,
        post_sample=sample,
        provenance={"fixture": True},
    )
    assert summary["scientific_status"] == "DEVELOPMENT_PRIMARY_NOT_SUPPORTED_STOP_S11"
    allocation_path = tmp_path / "bundle/fold_A_allocations.npy"
    np.testing.assert_array_equal(np.load(allocation_path, allow_pickle=False), result.allocations)
    manifest = json.loads((tmp_path / "bundle/manifest.json").read_text())
    assert "fold_A_allocations.npy" in manifest["files"]
    assert not (tmp_path / "bundle/INCOMPLETE.json").exists()


def test_cli_import_does_not_preload_numpy() -> None:
    value = subprocess.check_output(
        [sys.executable, "-c", "import sys; import frem_s11.cli; print('numpy' in sys.modules)"],
        text=True,
    ).strip()
    assert value == "False"
