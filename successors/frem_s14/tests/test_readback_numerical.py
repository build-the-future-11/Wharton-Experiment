"""Generated end-to-end readback and adversarial retained-evidence tests."""

import json
import os
import shutil
from dataclasses import replace

import numpy as np
import pytest

from frem_s14.cli import campaign, qualification_config
from frem_s14.evaluation import canonical_mean, date_nll
from frem_s14.runtime import read_json
from frem_s14.verification import verify_semantics


@pytest.fixture(scope="module")
def reference_bundle(tmp_path_factory):
    path = tmp_path_factory.mktemp("readback") / "reference"
    cfg = replace(
        qualification_config(os.environ.get("FREM_S14_TEST_DEVICE", "cpu")), arms=("zero",)
    )
    campaign(path, cfg)
    return path


def test_generated_input_reconstruction_passes(reference_bundle):
    assert verify_semantics(reference_bundle)["cells"] == 1


def test_canonical_score_reductions_do_not_depend_on_numpy_reduction_order():
    values = np.array([1e16, 1.0, -1e16])
    assert canonical_mean(values) == pytest.approx(1 / 3)
    np.testing.assert_allclose(
        date_nll(values, np.array([10, 10, 20], dtype=np.int64)),
        np.array([5e15, -1e16]),
    )


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("scaler", "preprocessing"),
        ("training_labels", "input reconstruction"),
        ("evaluation_labels", "input reconstruction"),
        ("split", "splits differ"),
        ("binding", "binding differs"),
        ("policy", "execution contract"),
        ("shape", "invalid shape"),
        ("margin", "paired inference"),
        ("scores", "score artifact differs"),
        ("duplicate_json", "duplicate JSON"),
    ],
)
def test_readback_rejects_tampering(reference_bundle, tmp_path, mutation, match):
    root = tmp_path / "altered"
    shutil.copytree(reference_bundle, root)
    cell = root / "zero-s11"
    if mutation in ("scaler", "training_labels", "evaluation_labels", "shape", "scores"):
        path = {
            "scaler": root / "preprocessing.npz",
            "training_labels": root / "training_risk_sample.npz",
            "evaluation_labels": cell / "predictions.npz",
            "shape": cell / "predictions.npz",
            "scores": cell / "scores.npz",
        }[mutation]
        with np.load(path, allow_pickle=False) as saved:
            arrays = {k: saved[k] for k in saved.files}
        if mutation == "scaler":
            arrays["center"][0] += 1
        elif mutation == "shape":
            arrays["mean"] = arrays["mean"][:, None]
        elif mutation == "scores":
            arrays["nll"][0] += 1e-2
        else:
            arrays["y"][0] += 0.001
        np.savez(path, **arrays)
    else:
        path = {
            "split": root / "splits.json",
            "binding": cell / "started.json",
            "policy": cell / "portfolio_cost1.json",
            "margin": root / "comparisons.json",
            "duplicate_json": cell / "metrics.json",
        }[mutation]
        saved = read_json(path)
        if mutation == "split":
            saved["validation"]["last_decision"] += 1
        elif mutation == "binding":
            saved["binding"]["seed"] = 29
        elif mutation == "policy":
            saved["policy"] = "cash"
        elif mutation == "margin":
            saved["practical_nll_margin"] = 0
        else:
            path.write_text('{"rows":1,' + path.read_text().lstrip()[1:])
        if mutation != "duplicate_json":
            path.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match=match):
        verify_semantics(root)
