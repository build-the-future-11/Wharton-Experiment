"""One generated cash-only M11-v2 ledger, with exact-rational accounting replay."""
from __future__ import annotations

import csv
from fractions import Fraction
import hashlib
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time

RECORD = Path(__file__).resolve().parent
ROOT = RECORD.parents[2]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def worker():
    import numpy as np
    import scipy
    from wharton_lab.models.m11.mpc import PortfolioState
    from wharton_lab.models.m11 import mpc_v2

    destination = RECORD / "fixture"
    destination.mkdir(exist_ok=False)
    (destination / "executed_driver.py").write_bytes(Path(__file__).read_bytes())
    cash, obligations = 1e308, np.array([1e307, 2e307])
    holdings, returns = np.zeros(1), np.zeros((5, 2, 1))
    np.savez_compressed(destination / "inputs.npz", holdings=holdings, scenario_returns=returns,
                        obligations=obligations, cash=np.asarray(cash))
    config = {"implementation": "explicit wharton_lab.models.m11.mpc_v2", "cash": cash,
              "min_cash_buffer": 1.0, "terminal_wealth_weight": 0.5, "tcost_bps": 100.0,
              "scenarios": 5, "periods": 2, "assets": 1, "optimizer_permitted": False}
    write_json(destination / "config.json", config)
    optimizer_invocations = 0

    def reject_optimizer(*args, **kwargs):
        nonlocal optimizer_invocations
        optimizer_invocations += 1
        raise AssertionError("contract forbids an optimizer in this cash-only fixture")

    # A process-local fail-before-optimization guard; the unchanged cash-only
    # branch returns directly and must never invoke this hook.
    mpc_v2.minimize = reject_optimizer
    with np.errstate(over="raise", invalid="raise"):
        result = mpc_v2.solve_mpc_v2(
            state=PortfolioState(holdings, cash), scenario_returns=returns,
            obligations=obligations, min_cash_buffer=1,
            terminal_wealth_weight=.5, tcost_bps=100,
        )
    assert result.success and optimizer_invocations == 0
    arrays = vars(result.ledger)
    np.savez_compressed(destination / "ledger.npz", **arrays)
    expected_cash = Fraction(cash)
    exact_replay = []
    for period, obligation in enumerate(obligations):
        expected_cash -= Fraction(float(obligation))
        expected = float(expected_cash)
        for scenario in range(5):
            assert abs(result.ledger.cash[scenario, period] - expected) <= 2 * math.ulp(expected)
            assert result.ledger.wealth[scenario, period] == result.ledger.cash[scenario, period]
            assert result.ledger.payments[scenario, period] == obligation
        exact_replay.append({"period": period, "cash_numerator": str(expected_cash.numerator),
                             "cash_denominator": str(expected_cash.denominator)})
    for name in ("holdings", "arrears", "fees", "turnover", "asset_gains"):
        assert not np.any(arrays[name])
    expected_objective = float(-Fraction(.5) * expected_cash)
    assert abs(result.objective - expected_objective) <= 2 * math.ulp(expected_objective)
    write_json(destination / "exact_replay.json", exact_replay)

    def table(payload):
        # Hash/load consume this same bounded snapshot; no path can be swapped
        # between the provenance check and NPZ decoding.
        stream = io.StringIO(newline="")
        fields = ("cash", "wealth", "payments", "arrears", "fees", "turnover", "asset_gains")
        writer = csv.writer(stream)
        writer.writerow(("scenario", "period", *fields, "holdings_0"))
        with np.load(io.BytesIO(payload), allow_pickle=False) as raw:
            assert set(raw.files) == set(arrays)
            for name, value in arrays.items():
                assert raw[name].dtype == value.dtype and raw[name].shape == value.shape
                np.testing.assert_array_equal(raw[name], value)
            for scenario in range(5):
                for period in range(2):
                    writer.writerow((scenario, period, *(float(raw[name][scenario, period]) for name in fields),
                                     float(raw["holdings"][scenario, period, 0])))
        return stream.getvalue().encode()

    payload = (destination / "ledger.npz").read_bytes()
    ledger_sha256 = hashlib.sha256(payload).hexdigest()
    csv_path = destination / "ledger.csv"
    csv_path.write_bytes(table(payload))
    original = csv_path.read_bytes()
    retained_table = destination / "ledger.before_regeneration.csv"
    retained_table.write_bytes(original)
    csv_path.unlink()
    replay_payload = (destination / "ledger.npz").read_bytes()
    assert hashlib.sha256(replay_payload).hexdigest() == ledger_sha256
    csv_path.write_bytes(table(replay_payload))
    assert csv_path.read_bytes() == original
    summary = {
        "status": "COMPLETED_ENGINEERING_FIXTURE", "scenarios": 5, "periods": 2, "assets": 1,
        "optimizer_invocations": optimizer_invocations, "training_steps": 0,
        "cash_and_objective_oracle": "exact binary64-input Fraction equation; at most 2 ULP after sequential ledger arithmetic",
        "table_deleted_and_recreated_byte_identically": True,
        "ledger_sha256": ledger_sha256, "inputs_sha256": digest(destination / "inputs.npz"),
        "table_sha256": digest(csv_path), "driver_sha256": digest(Path(__file__)),
        "original_table_sha256": digest(retained_table),
        "source_sha256": {str(p.relative_to(ROOT)): digest(p) for p in sorted((ROOT / "src").rglob("*.py"))},
        "numpy": str(np.__version__), "scipy": str(scipy.__version__),
        "numeric_thread_environment": {name: os.environ[name] for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
        "claim_boundary": "fictional zero-return software ledger; no market, funding reliability, forecasting or scientific result",
    }
    write_json(destination / "result.json", summary)
    write_json(RECORD / "fixture_result.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "source_sha256"}, sort_keys=True))


def execute():
    receipt = RECORD / "fixture_receipt.json"
    if receipt.exists():
        raise SystemExit("refusing to overwrite retained fixture receipt")
    final = json.loads((RECORD / "final_receipt.json").read_text())
    assert final["returncode"] == 0
    paths = list(final["source_sha256_after"])
    before = {path: digest(ROOT / path) for path in paths}
    assert before == final["source_sha256_after"]
    env = dict(os.environ, PYTHONPATH="src", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               MKL_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
    command = [sys.executable, str(Path(__file__).resolve()), "--worker"]
    started = time.perf_counter()
    try:
        outcome = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, timeout=30)
        code, stdout, stderr, timed_out = outcome.returncode, outcome.stdout, outcome.stderr, False
    except subprocess.TimeoutExpired as error:
        code, stdout, stderr, timed_out = None, error.stdout or b"", error.stderr or b"", True
    seconds = time.perf_counter() - started
    (RECORD / "fixture.stdout.log").write_bytes(stdout)
    (RECORD / "fixture.stderr.log").write_bytes(stderr)
    archive = RECORD / "fixture_artifacts.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        if (RECORD / "fixture").exists():
            handle.add(RECORD / "fixture", arcname="fixture")
    after = {path: digest(ROOT / path) for path in paths}
    result = {"argv": command, "returncode": code, "timed_out": timed_out,
              "child_command_wall_seconds": seconds, "hard_timeout_seconds": 30,
              "source_sha256_before": before, "source_sha256_after": after,
              "sources_unchanged_during_command": before == after,
              "driver_sha256": digest(Path(__file__)), "archive_sha256": digest(archive),
              "optimizer_calls": 0, "training_steps": 0, "paid_calls": 0}
    receipt.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if not k.startswith("source_")}))
    print(stdout.decode(errors="replace")[-1000:])
    print(stderr.decode(errors="replace")[-1000:])


if __name__ == "__main__":
    if sys.argv[1:] == ["--worker"]:
        worker()
    elif not sys.argv[1:]:
        execute()
    else:
        raise SystemExit("unexpected arguments")
