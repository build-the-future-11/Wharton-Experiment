from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from frem_s12.campaign import FAMILIES, search_space
from frem_s12.errors import ContractError
from frem_s12.evidence import atomic_json, package_identity, read_json, sha256_file
from frem_s12.generated import generated_panel
from frem_s12.scores import Law, energy_score
from frem_s12.search import (
    SEARCH_SEEDS,
    SearchResult,
    config_from_selection,
    read_search_result,
    run_search_cell,
    select_search,
    write_search_result,
)


def _provenance(trial: int, family: str, seed: int) -> dict[str, object]:
    identity = f"{trial}:{family}:{seed}".encode()
    unique = hashlib.sha256(identity).hexdigest()
    family_digest = hashlib.sha256(family.encode()).hexdigest()
    return {
        "environment_receipt_sha256": "1" * 64,
        "data_receipt_sha256": family_digest,
        "acquisition_receipt_sha256": hashlib.sha256((family + ":acquisition").encode()).hexdigest(),
        "archive_sha256": hashlib.sha256((family + ":archive").encode()).hexdigest(),
        "source_identity": package_identity()["sha256"],
        "consumption": {
            "path": f"ledger/search/trial={trial}/family={family}/seed={seed}/claim.json",
            "sha256": unique,
            "resumed": False,
        },
    }


def test_search_reads_fold_a_validation_only() -> None:
    base = generated_panel(months=432, seed=5)
    panel = type(base)("size_bm", base.months, base.returns)
    result = run_search_cell(panel, trial=0, seed=11)
    assert result.months[0] == 199801
    assert result.months[-1] == 199912
    assert result.scores.shape == (24,)
    assert result.outcomes.shape == (24, 6)
    for row in range(24):
        law = Law(result.forecast_atoms[row], result.forecast_probabilities[row])
        assert energy_score(law, result.outcomes[row]) == pytest.approx(result.scores[row], abs=1e-15)


def test_complete_search_selects_lowest_mean_then_trial(tmp_path: Path) -> None:
    months = np.asarray(
        [year * 100 + month for year in (1998, 1999) for month in range(1, 13)],
        dtype=np.int64,
    )
    outcomes = np.zeros((24, 6))
    weights = np.full((24, 5), 0.2)
    probabilities = np.full((24, 2), 0.5)
    for trial in range(len(search_space())):
        for family in FAMILIES:
            for seed in SEARCH_SEEDS:
                score = 0.5 if trial == 7 else 1.0 + trial / 100
                atoms = np.zeros((24, 2, 6))
                atoms[:, :, 0] = score
                result = SearchResult(
                    trial,
                    family,
                    seed,
                    months,
                    outcomes,
                    np.full(24, score),
                    weights,
                    atoms,
                    probabilities,
                )
                output = tmp_path / f"trial={trial}" / f"family={family}" / f"seed={seed}"
                write_search_result(output, result, provenance=_provenance(trial, family, seed))
    selection = select_search(tmp_path)
    assert selection["selected_trial"] == 7
    receipt = tmp_path / "selection.json"
    atomic_json(receipt, selection)
    assert config_from_selection(receipt).reservoir_width == search_space()[7]["reservoir_width"]
    with pytest.raises(ContractError, match="inventory"):
        select_search(tmp_path)


def test_incomplete_search_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(ContractError, match="incomplete"):
        select_search(tmp_path)


def test_search_reader_reconstructs_summary_from_raw_laws(tmp_path: Path) -> None:
    base = generated_panel(months=432, seed=5)
    panel = type(base)("size_bm", base.months, base.returns)
    result = run_search_cell(panel, trial=0, seed=11)
    output = tmp_path / "cell"
    write_search_result(output, result, provenance={"generated": True})
    summary = read_json(output / "summary.json")
    summary["mean_energy"] = float(summary["mean_energy"]) + 0.01
    (output / "summary.json").write_text(json.dumps(summary))
    manifest = read_json(output / "manifest.json")
    manifest["files"]["summary.json"] = sha256_file(output / "summary.json")
    (output / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ContractError, match="differs from reconstructed"):
        read_search_result(output)
