"""Generated weekly archival, retention and fail-stop evidence."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from test_experiment_audit import generated_panel
from test_model_audit import first_transition, record, request

from frem_s10.data.calendar import build_calendar_schedule
from frem_s10.data.features import build_origin_records, fit_burnin_transform
from frem_s10.errors import ArtifactIOError, IntegrityError
from frem_s10.experiment.forecasts import build_forecast_tape
from frem_s10.experiment.matrix import MODEL_VARIANTS
from frem_s10.experiment.model_archive import (
    ModelArchiveWriter,
    _payload,
    _unpack,
    decode_transition,
    verify_model_archive,
)
from frem_s10.identities import JSONValue, object_json
from frem_s10.model.checkpoint import dump_state
from frem_s10.model.engine import advance
from frem_s10.model.records import ForecastTransition
from frem_s10.runtime.artifacts import read_verified_json


def writer(path: Path) -> ModelArchiveWriter:
    return ModelArchiveWriter(
        path,
        experiment_id="generated-only",
        evidence_scope="GENERATED_ENGINEERING",
        binding_id="constructed-freeze",
        variants=("primary",),
    )


def test_all_weekly_generations_restore_exact_maturities_and_issued_laws(tmp_path: Path) -> None:
    archive = writer(tmp_path)
    step = first_transition()
    original: list[ForecastTransition] = [step]
    archive.sink("primary", step)
    for index in range(71, 77):
        step = advance(step.state, request(index), (record(index - 5),))
        original.append(step)
        archive.sink("primary", step)
    digest = archive.finish()
    manifest = verify_model_archive(tmp_path, digest)
    rows = manifest["files"]
    assert isinstance(rows, list) and len(rows) == 7
    for raw, expected in zip(rows, original, strict=True):
        row = object_json(raw)
        assert isinstance(row["name"], str) and isinstance(row["sha256"], str)
        payload = object_json(
            read_verified_json(tmp_path / row["name"], row["sha256"], maximum_bytes=16777216)
        )
        restored = decode_transition(object_json(payload["transition"]))
        assert dump_state(restored.state) == dump_state(expected.state)
        assert [x.record_id for x in restored.matured_feedback] == [
            x.record_id for x in expected.matured_feedback
        ]
        for actual, issued in zip(restored.batch.forecasts, expected.batch.forecasts, strict=True):
            assert actual.record_id == issued.record_id
            np.testing.assert_array_equal(actual.probabilities, issued.probabilities)
    assert original[-1].matured_feedback
    with pytest.raises(IntegrityError):
        archive.sink("primary", step)


def test_solver_certificates_survive_compact_exact_roundtrip() -> None:
    step = first_transition()
    for index in range(71, 102):
        step = advance(step.state, request(index), (record(index - 5),))
    restored = decode_transition(_payload(step))
    certificate = restored.batch.mixtures.global_certificate
    assert certificate is not None and len(certificate.faces) == 7
    expected = step.batch.mixtures.global_certificate
    assert expected is not None
    assert certificate.faces == expected.faces
    assert certificate.objective == expected.objective


def test_failed_generation_retains_prefix_and_forbids_automatic_continuation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = writer(tmp_path)
    first = first_transition()
    archive.sink("primary", first)
    original_bytes = (tmp_path / "weekly-0000.json").read_bytes()

    def fail(directory: Path, name: str, value: JSONValue) -> str:
        raise ArtifactIOError("constructed disk-full boundary")

    monkeypatch.setattr("frem_s10.experiment.model_archive.publish_json", fail)
    next_step = advance(first.state, request(71), (record(66),))
    with pytest.raises(ArtifactIOError):
        archive.sink("primary", next_step)
    with pytest.raises(IntegrityError):
        archive.sink("primary", next_step)
    with pytest.raises(IntegrityError):
        archive.finish()
    assert (tmp_path / "weekly-0000.json").read_bytes() == original_bytes
    assert not (tmp_path / "manifest.json").exists()


@pytest.mark.parametrize("defect", ("state", "mixture", "maturity", "compression"))
def test_rejects_mismatched_saved_state_mixture_and_maturity(defect: str) -> None:
    step = first_transition()
    for index in range(71, 76):
        step = advance(step.state, request(index), (record(index - 5),))
    if defect == "state":
        payload = _payload(step)
        payload["state_id"] = "wrong"
        with pytest.raises(IntegrityError):
            decode_transition(payload)
    elif defect == "mixture":
        bundle = replace(step.batch.mixtures, contextual_alpha=np.array([1.0, 0.0, 0.0]))
        with pytest.raises(IntegrityError):
            _payload(replace(step, batch=replace(step.batch, mixtures=bundle)))
    elif defect == "maturity":
        changed = replace(step.matured_feedback[0], a=step.matured_feedback[0].a + 1)
        with pytest.raises(IntegrityError):
            _payload(replace(step, matured_feedback=(changed,)))
    else:
        with pytest.raises(IntegrityError):
            _unpack("not-a-valid-compressed-record")


def test_callback_retains_nonselected_origins_and_failure_stops_next_issue() -> None:
    panel, _ = generated_panel(116)
    records = build_origin_records(panel, calendar=build_calendar_schedule(panel.dates))
    transform = fit_burnin_transform(records)
    eligible = tuple(record.anchor_index for record in records if record.origin_us >= 1199145600000000)
    retained = (eligible[0], eligible[3])
    seen: list[tuple[str, int]] = []

    def sink(variant: str, transition: ForecastTransition) -> None:
        seen.append((variant, transition.batch.get("contextual_stack").request.anchor_index))

    result = build_forecast_tape(records, transform, retained_anchor_indices=retained, transition_sink=sink)
    assert result.weekly_origins == 4
    assert seen == [(variant, anchor) for anchor in eligible[:4] for variant in MODEL_VARIANTS]
    assert len(result.frames) == 12 and len(seen) == 24
    calls = 0

    def fail(variant: str, transition: ForecastTransition) -> None:
        nonlocal calls
        calls += 1
        raise ArtifactIOError("constructed archive failure")

    with pytest.raises(ArtifactIOError):
        build_forecast_tape(records, transform, retained_anchor_indices=retained, transition_sink=fail)
    assert calls == 1
