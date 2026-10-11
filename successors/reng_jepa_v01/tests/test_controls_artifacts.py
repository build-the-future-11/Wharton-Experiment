import json

import numpy as np
import pytest
import torch

from reng.artifacts import ExperimentManifest, write_receipt
from reng.baselines import (
    DirectCorrelatedMixture,
    DLinear,
    PatchTransformer,
    RidgeForecast,
    RollingMeanShrinkage,
    SupervisedGRU,
    TemporalJEPAAdaptation,
    ZeroHistoricalRisk,
    parameter_match_error,
)
from reng.checkpoint import create_freeze, load_checkpoint, save_checkpoint


def test_classical_baselines_fit_common_data() -> None:
    rng = np.random.default_rng(11)
    returns = rng.normal(size=(50, 3))
    assert ZeroHistoricalRisk().fit(returns).predict().mean.shape == (3,)
    assert RollingMeanShrinkage().fit(returns).predict().covariance.shape == (3, 3)
    ridge = RidgeForecast().fit(rng.normal(size=(50, 5)), returns)
    mean, covariance = ridge.predict(rng.normal(size=(2, 5)))
    assert mean.shape == (2, 3)
    assert covariance.shape == (3, 3)


def test_learned_baseline_shapes() -> None:
    sequence = torch.randn(3, 32, 18)
    gru_mean, gru_scale = SupervisedGRU(assets=2)(sequence)
    assert gru_mean.shape == gru_scale.shape == (3, 2)
    patch_mean, patch_scale = PatchTransformer()(sequence)
    assert patch_mean.shape == patch_scale.shape == (3,)
    dlinear = DLinear(32, 2)
    assert dlinear(torch.randn(3, 2, 32)).shape == (3, 2)
    for variant in ("ts_jepa", "cf_jepa"):
        prediction, target, scale = TemporalJEPAAdaptation(variant)(
            torch.randn(3, 32, 18), torch.randn(3, 21, 18), 1
        )
        assert prediction.shape == target.shape == (3, 32)
        assert scale.shape == (3,)


def test_direct_control_is_an_active_correlated_distribution() -> None:
    control = DirectCorrelatedMixture(hidden_dim=96)
    logits, components = control(torch.randn(2, 4, 3, 200))
    assert logits.shape == (2, 3)
    assert len(components) == 2 and len(components[0]) == 3
    assert components[0][0].factors.shape == (4, 8)
    assert parameter_match_error(control, control) == 0


def test_receipts_are_immutable_and_synthetic_claims_are_false(tmp_path) -> None:
    manifest = ExperimentManifest.create(
        source_tree_sha256="a" * 64,
        protocol={"id": "p"},
        configuration={"x": 1},
        dataset_sha256="b" * 64,
        dataset_label="synthetic_engineering:test",
        dataset_synthetic=True,
        seeds=(11,),
        scientific_claims_allowed=False,
    )
    target = tmp_path / "run"
    write_receipt(target, manifest, {"terminal_status": "ENGINEERING_PASS"})
    payload = json.loads((target / "manifest.json").read_text())
    assert payload["scientific_claims_allowed"] is False
    with pytest.raises(FileExistsError):
        write_receipt(target, manifest, {"terminal_status": "ENGINEERING_PASS"})


def test_checkpoint_is_hash_bound(tmp_path) -> None:
    model = torch.nn.Linear(3, 2)
    metadata = {
        "model_version": "0.1.0",
        "source_tree_sha256": "a" * 64,
        "dataset_sha256": "b" * 64,
        "split_sha256": "c" * 64,
        "configuration_sha256": "d" * 64,
        "root_seed": 11,
        "ensemble_seed": 12,
        "horizon": 21,
    }
    path = save_checkpoint(tmp_path / "checkpoint", model, metadata=metadata)
    load_checkpoint(path, torch.nn.Linear(3, 2), expected={"horizon": 21})
    with (path / "state.pt").open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(ValueError, match="hash"):
        load_checkpoint(path, torch.nn.Linear(3, 2))


def test_freeze_requires_complete_hashes_and_is_immutable(tmp_path) -> None:
    output = tmp_path / "freeze.json"
    digest = create_freeze(
        output,
        protocol={"id": "p"},
        configuration={"id": "c"},
        source_tree_sha256="a" * 64,
        dataset_sha256="b" * 64,
        split_sha256="c" * 64,
        universe_sha256="d" * 64,
        mandate_sha256="e" * 64,
        environment_sha256="f" * 64,
        protected_boundary="2027-01-01T00:00:00Z",
        rationale="All prerequisites were independently checked before outcomes.",
    )
    assert len(digest) == 64
    with pytest.raises(FileExistsError):
        create_freeze(
            output,
            protocol={"id": "p"},
            configuration={"id": "c"},
            source_tree_sha256="a" * 64,
            dataset_sha256="b" * 64,
            split_sha256="c" * 64,
            universe_sha256="d" * 64,
            mandate_sha256="e" * 64,
            environment_sha256="f" * 64,
            protected_boundary="2027-01-01T00:00:00Z",
            rationale="same freeze",
        )
