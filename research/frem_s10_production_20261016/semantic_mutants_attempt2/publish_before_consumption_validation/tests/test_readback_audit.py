"""Saved evidence round trips and hostile semantic payloads, without fitting."""

from pathlib import Path

import numpy as np
import pytest
from test_experiment_audit import campaign as campaign
from test_experiment_audit import inputs as inputs
from test_experiment_audit import tape as tape

from frem_s10.errors import IntegrityError
from frem_s10.experiment.artifacts import (
    PublicationReceipt,
    _decode_frame,
    _encode,
    _verify_frame_axes,
    publish_campaign,
)
from frem_s10.experiment.forecasts import ForecastTape
from frem_s10.experiment.readback import _decode, load_campaign
from frem_s10.experiment.replay import CampaignResult, CellFailure, ReplayInputs
from frem_s10.identities import (
    JSONObject,
    JSONValue,
    canonical_bytes,
    digest_bytes,
    object_json,
    parse_json,
    scientific_digest,
)
from frem_s10.portfolio.ledger import LedgerState, initial_ledger


@pytest.fixture(scope="module")
def saved(
    campaign: CampaignResult,
    inputs: ReplayInputs,
    tape: ForecastTape,
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, PublicationReceipt]:
    directory = tmp_path_factory.mktemp("saved-generated-campaign")
    receipt = publish_campaign(
        directory, inputs, tape, experiment_id="generated-readback", steps=campaign.results
    )
    return directory, receipt


def test_full192_readonly_roundtrip_preserves_arrays_accounting_and_scope(
    saved: tuple[Path, PublicationReceipt],
    campaign: CampaignResult,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("readback must never execute fitting or policy selection")

    monkeypatch.setattr("frem_s10.model.engine.advance", forbidden)
    monkeypatch.setattr("frem_s10.experiment.replay.select_policy", forbidden)
    directory, receipt = saved
    before = {path.name: digest_bytes(path.read_bytes()) for path in directory.iterdir()}
    restored = load_campaign(directory, receipt.manifest_sha256)
    assert restored.complete and len(restored.results) == 192
    assert restored.evidence_scope == campaign.evidence_scope == "GENERATED_ENGINEERING"
    assert restored.forecast_tape_id == campaign.forecast_tape_id
    for original, decoded in zip(campaign.results, restored.results, strict=True):
        assert original.spec == decoded.spec
        assert original.final_state.record_id == decoded.final_state.record_id
        assert original.maximum_drawdown.hex() == decoded.maximum_drawdown.hex()
        np.testing.assert_array_equal(original.log_increments, decoded.log_increments)
        np.testing.assert_array_equal(original.final_state.price_pnl, decoded.final_state.price_pnl)
        np.testing.assert_array_equal(original.final_state.dividends, decoded.final_state.dividends)
        for left, right in zip(original.events, decoded.events, strict=True):
            assert (left.event_id, left.before_id, left.after_id, left.fee) == (
                right.event_id,
                right.before_id,
                right.after_id,
                right.fee,
            )
            np.testing.assert_array_equal(left.shares_after, right.shares_after)
        for left_score, right_score in zip(original.scores, decoded.scores, strict=True):
            assert left_score.energy.hex() == right_score.energy.hex()
            np.testing.assert_array_equal(left_score.interval_coverage, right_score.interval_coverage)
        with pytest.raises(ValueError):
            decoded.final_state.shares.setflags(write=True)
    assert before == {path.name: digest_bytes(path.read_bytes()) for path in directory.iterdir()}


def test_failed_prefix_roundtrip_retains_failure_and_partial(
    campaign: CampaignResult,
    inputs: ReplayInputs,
    tape: ForecastTape,
    tmp_path: Path,
) -> None:
    failure = CellFailure(
        campaign.results[1].spec.cell_id,
        "GENERATED_FAILURE",
        "failure after cell evaluation",
        campaign.results[1],
    )
    receipt = publish_campaign(
        tmp_path,
        inputs,
        tape,
        experiment_id="generated-failed-readback",
        steps=(campaign.results[0], failure),
    )
    restored = load_campaign(tmp_path, receipt.manifest_sha256)
    assert not restored.complete and len(restored.results) == 1
    assert restored.failure is not None and restored.failure.partial is not None
    assert restored.failure.code == failure.code
    assert restored.failure.partial.final_state.record_id == campaign.results[1].final_state.record_id


def test_scalar_numeric_representation_survives_semantic_hash_roundtrip() -> None:
    state = initial_ledger(cell_id="generated-integer-capital", start_us=1, initial_cash=100000)
    restored = _decode(_encode(state))
    assert isinstance(restored, LedgerState)
    assert type(restored.initial_cash) is int
    assert restored.record_id == state.record_id


@pytest.mark.parametrize(
    "field,bad", (("dtype", "<f4"), ("unit", "probability"), ("shape", [1, 3]), ("context", "Action.weights"))
)
def test_typed_array_readback_rejects_retagging(
    saved: tuple[Path, PublicationReceipt], field: str, bad: JSONValue
) -> None:
    payload = object_json(parse_json((saved[0] / "cell-000.json").read_bytes()))
    cell = object_json(payload["payload"])
    content = object_json(cell["fields"])
    state = object_json(content["final_state"])
    values = object_json(state["fields"])
    shares = object_json(values["shares"])
    shares[field] = bad
    with pytest.raises(IntegrityError):
        _decode(cell)


def _resign_cell(directory: Path, manifest: JSONObject, name: str, payload: JSONObject) -> str:
    data = canonical_bytes(payload) + b"\n"
    (directory / name).write_bytes(data)
    inventory = manifest["files"]
    assert isinstance(inventory, list)
    row = next(object_json(item) for item in inventory if object_json(item)["name"] == name)
    row["sha256"] = digest_bytes(data)
    manifest_data = canonical_bytes(manifest) + b"\n"
    (directory / "manifest.json").write_bytes(manifest_data)
    return digest_bytes(manifest_data)


def test_resigned_bytes_cannot_hide_broken_semantic_state_identity(
    campaign: CampaignResult,
    inputs: ReplayInputs,
    tape: ForecastTape,
    tmp_path: Path,
) -> None:
    receipt = publish_campaign(
        tmp_path, inputs, tape, experiment_id="generated-semantic-tamper", steps=(campaign.results[0],)
    )
    manifest = object_json(parse_json((tmp_path / "manifest.json").read_bytes()))
    payload = object_json(parse_json((tmp_path / "cell-000.json").read_bytes()))
    content = object_json(object_json(payload["payload"])["fields"])
    state = object_json(object_json(content["final_state"])["fields"])
    state["cursor_us"] = int(str(state["cursor_us"])) + 1
    changed_hash = _resign_cell(tmp_path, manifest, "cell-000.json", payload)
    with pytest.raises(IntegrityError, match="state identity"):
        load_campaign(tmp_path, changed_hash)
    assert receipt.manifest_sha256 != changed_hash


def test_resigned_partial_cannot_be_claimed_complete(
    campaign: CampaignResult,
    inputs: ReplayInputs,
    tape: ForecastTape,
    tmp_path: Path,
) -> None:
    failure = CellFailure(campaign.results[0].spec.cell_id, "FAILED", "generated failure", None)
    publish_campaign(tmp_path, inputs, tape, experiment_id="generated-false-complete", steps=(failure,))
    manifest = object_json(parse_json((tmp_path / "manifest.json").read_bytes()))
    payload = object_json(parse_json((tmp_path / "cell-000.json").read_bytes()))
    payload["status"] = "COMPLETE"
    changed_hash = _resign_cell(tmp_path, manifest, "cell-000.json", payload)
    with pytest.raises(IntegrityError):
        load_campaign(tmp_path, changed_hash)


@pytest.mark.parametrize("field", ("transform_id", "weekly_origins", "final_state_ids"))
def test_resigned_tape_metadata_must_match_semantic_identity(
    campaign: CampaignResult,
    inputs: ReplayInputs,
    tape: ForecastTape,
    tmp_path: Path,
    field: str,
) -> None:
    publish_campaign(
        tmp_path, inputs, tape, experiment_id="generated-index-tamper", steps=(campaign.results[0],)
    )
    manifest = object_json(parse_json((tmp_path / "manifest.json").read_bytes()))
    index = object_json(parse_json((tmp_path / "forecasts.json").read_bytes()))
    if field == "weekly_origins":
        index[field] = 1
    elif field == "transform_id":
        index[field] = "tampered-transform"
    else:
        states = index[field]
        assert isinstance(states, list) and isinstance(states[0], list)
        states[0][1] = "tampered-final-state"
    changed_hash = _resign_cell(tmp_path, manifest, "forecasts.json", index)
    with pytest.raises(IntegrityError, match="tape metadata"):
        load_campaign(tmp_path, changed_hash)


def test_manifest_experiment_identity_cannot_diverge_from_header(
    campaign: CampaignResult, inputs: ReplayInputs, tape: ForecastTape, tmp_path: Path
) -> None:
    publish_campaign(
        tmp_path, inputs, tape, experiment_id="generated-experiment-identity", steps=(campaign.results[0],)
    )
    manifest = object_json(parse_json((tmp_path / "manifest.json").read_bytes()))
    manifest["experiment_id"] = "different-attempt"
    encoded = canonical_bytes(manifest) + b"\n"
    (tmp_path / "manifest.json").write_bytes(encoded)
    with pytest.raises(IntegrityError, match="experiment identity"):
        load_campaign(tmp_path, digest_bytes(encoded))


@pytest.mark.parametrize("bad", (True, "1.0"))
def test_forecast_arrays_reject_coerced_boolean_or_numeric_string(tape: ForecastTape, bad: JSONValue) -> None:
    payload: JSONObject = {"payload": _encode(tape.frames[0])}
    frame = object_json(object_json(payload["payload"])["fields"])
    laws = frame["laws"]
    assert isinstance(laws, list) and isinstance(laws[0], list)
    law = object_json(object_json(laws[0][1])["fields"])
    values = object_json(law["g"])["values"]
    assert isinstance(values, list) and isinstance(values[0], list)
    values[0][0] = bad
    with pytest.raises(IntegrityError, match="nonnumeric or boolean"):
        _decode_frame(payload)


def test_resigned_header_axes_cannot_relabel_saved_values(
    campaign: CampaignResult, inputs: ReplayInputs, tape: ForecastTape, tmp_path: Path
) -> None:
    publish_campaign(
        tmp_path, inputs, tape, experiment_id="generated-axis-tamper", steps=(campaign.results[0],)
    )
    manifest = object_json(parse_json((tmp_path / "manifest.json").read_bytes()))
    header = object_json(parse_json((tmp_path / "header.json").read_bytes()))
    header["asset_axis"] = ["TLT", "SPY", "GLD"]
    changed_hash = _resign_cell(tmp_path, manifest, "header.json", header)
    with pytest.raises(IntegrityError, match="header axes"):
        load_campaign(tmp_path, changed_hash)


@pytest.mark.parametrize("defect", ("duplicate", "order", "missing", "origin", "time"))
def test_frame_structure_enforces_complete_common_origin_variants(tape: ForecastTape, defect: str) -> None:
    axes = [(frame.anchor_index, frame.variant, frame.origin_id, frame.decision_us) for frame in tape.frames]
    if defect == "duplicate":
        axes[1] = axes[0]
    elif defect == "order":
        axes[0], axes[1] = axes[1], axes[0]
    elif defect == "missing":
        axes.pop(0)
    else:
        anchor, variant, origin, clock = axes[1]
        axes[1] = (
            anchor,
            variant,
            origin + "changed" if defect == "origin" else origin,
            clock + (defect == "time"),
        )
    with pytest.raises(IntegrityError):
        _verify_frame_axes(tuple(axes))


def test_fully_resigned_tape_envelope_cannot_admit_duplicate_variant(
    campaign: CampaignResult, inputs: ReplayInputs, tape: ForecastTape, tmp_path: Path
) -> None:
    publish_campaign(
        tmp_path, inputs, tape, experiment_id="generated-structure-tamper", steps=(campaign.results[0],)
    )
    manifest = object_json(parse_json((tmp_path / "manifest.json").read_bytes()))
    inventory = manifest["files"]
    assert isinstance(inventory, list)
    rows = [object_json(item) for item in inventory]
    original = object_json(parse_json((tmp_path / "forecast-frame-0000.json").read_bytes()))
    rows[3].update({name: rows[2][name] for name in ("frame_identity", "anchor_index", "variant")})
    _resign_cell(tmp_path, manifest, "forecast-frame-0001.json", original)
    index = object_json(parse_json((tmp_path / "forecasts.json").read_bytes()))
    frame_rows = [row for row in rows if "frame_identity" in row]
    index["frames"] = list(frame_rows)
    frame_ids = [row["frame_identity"] for row in frame_rows]
    tape_id = scientific_digest(
        {
            "schema": "s10-forecast-tape-v2",
            "transform": index["transform_id"],
            "scope": index["evidence_scope"],
            "weekly_origins": index["weekly_origins"],
            "final_states": index["final_state_ids"],
            "frames": frame_ids,
        }
    )
    index["tape_identity"] = tape_id
    _resign_cell(tmp_path, manifest, "forecasts.json", index)
    header = object_json(parse_json((tmp_path / "header.json").read_bytes()))
    header["tape_identity"], header["forecast_frame_identities"] = tape_id, frame_ids
    _resign_cell(tmp_path, manifest, "header.json", header)
    for row in rows:
        if "cell_id" in row:
            name = str(row["name"])
            payload = object_json(parse_json((tmp_path / name).read_bytes()))
            payload["tape_identity"] = tape_id
            _resign_cell(tmp_path, manifest, name, payload)
    changed_hash = digest_bytes((tmp_path / "manifest.json").read_bytes())
    with pytest.raises(IntegrityError, match="duplicate or out of order"):
        load_campaign(tmp_path, changed_hash)
