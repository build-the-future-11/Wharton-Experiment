"""Read verified saved evidence into exact dataclasses without executing a model."""

from collections.abc import Callable
from dataclasses import fields
from itertools import pairwise
from pathlib import Path

import numpy as np

from frem_s10.arrays import FloatArray, frozen_f64
from frem_s10.errors import IntegrityError
from frem_s10.identities import JSONObject, JSONValue, object_json
from frem_s10.math.fees import FeeSolution
from frem_s10.metadata import finite_float, strict_int
from frem_s10.portfolio.actions import Action
from frem_s10.portfolio.ledger import LedgerEvent, LedgerState, MarkKind, NavMark, initial_ledger
from frem_s10.portfolio.policy import DecisionCertificate, PolicyDecision
from frem_s10.runtime.artifacts import read_verified_json

from .artifacts import _ARRAY_UNITS, verify_publication
from .matrix import CellSpec, campaign_matrix
from .replay import CampaignResult, CellFailure, CellResult, DecisionRecord, ForecastScore


def _typed[T](value: object, expected: type[T]) -> T:
    if not isinstance(value, expected):
        raise IntegrityError(f"expected saved {expected.__name__}, got {type(value).__name__}")
    return value


def _items[T](value: object, expected: type[T]) -> tuple[T, ...]:
    if not isinstance(value, tuple):
        raise IntegrityError("saved sequence must decode to an immutable tuple")
    return tuple(_typed(item, expected) for item in value)


def _text(value: object) -> str:
    text = _typed(value, str)
    if not text:
        raise IntegrityError("saved string identity cannot be empty")
    return text


def _boolean(value: object) -> bool:
    if type(value) is not bool:
        raise IntegrityError("saved switch must be boolean")
    return value


def _number(value: object) -> float:
    finite_float(value, name="saved finite scalar")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value
    raise IntegrityError("saved scalar must retain its JSON numeric type")


def _integer(value: object) -> int:
    return strict_int(value, name="saved integer")


def _optional_number(value: object) -> float | None:
    return None if value is None else _number(value)


def _optional_text(value: object) -> str | None:
    return None if value is None else _text(value)


def _vector(value: object) -> FloatArray:
    if not isinstance(value, np.ndarray):
        raise IntegrityError("saved vector must use the typed-array schema")
    return frozen_f64(value, shape=(3,), name="saved vector")


def _spec(v: dict[str, object]) -> CellSpec:
    return CellSpec(
        _text(v["fold_id"]), _text(v["kind"]), _text(v["arm"]), _integer(v["cost_bps"]), _text(v["policy"])
    )


def _action(v: dict[str, object]) -> Action:
    return Action(_vector(v["weights"]), _boolean(v["is_hold"]))


def _fees(v: dict[str, object]) -> FeeSolution:
    return FeeSolution(
        _number(v["post_fee_nav"]),
        _vector(v["risky_dollars"]),
        _number(v["cash"]),
        _number(v["fee"]),
        _number(v["turnover"]),
        _number(v["residual"]),
        _number(v["bracket_width"]),
        _integer(v["iterations"]),
        _vector(v["target_weights"]),
    )


def _certificate(v: dict[str, object]) -> DecisionCertificate:
    return DecisionCertificate(
        _text(v["action_set_id"]),
        _integer(v["candidate_count"]),
        _integer(v["selected_index"]),
        _text(v["criterion"]),
        _text(v["direction"]),
        _number(v["best_objective"]),
        _number(v["selected_objective"]),
        _number(v["optimality_gap"]),
        _integer(v["objective_tie_count"]),
        _integer(v["turnover_tie_count"]),
        _number(v["max_asset_weight"]),
        _number(v["total_risky_weight"]),
        _number(v["constraint_residual"]),
        _optional_text(v["statistic_id"]),
        _optional_number(v["projection_multiplier"]),
        _optional_number(v["projection_sum_residual"]),
        _optional_number(v["projection_kkt_residual"]),
    )


def _policy(v: dict[str, object]) -> PolicyDecision:
    return PolicyDecision(
        _typed(v["action"], Action),
        _typed(v["fee_solution"], FeeSolution),
        _number(v["objective"]),
        _number(v["nominal_utility"]),
        _number(v["robust_utility"]),
        _number(v["worst_utility"]),
        _integer(v["worst_atom_index"]),
        _optional_number(v["hold_objective"]),
        _optional_number(v["cash_objective"]),
        _typed(v["certificate"], DecisionCertificate),
        _text(v["forecast_id"]),
    )


def _state(v: dict[str, object]) -> LedgerState:
    ids = _items(v["consumed_event_ids"], str)
    if len(ids) != len(set(ids)):
        raise IntegrityError("saved state duplicates consumed event identities")
    return LedgerState(
        _text(v["cell_id"]),
        _number(v["initial_cash"]),
        _integer(v["start_us"]),
        _vector(v["shares"]),
        _number(v["cash"]),
        None if v["marks"] is None else _vector(v["marks"]),
        _vector(v["price_pnl"]),
        _vector(v["dividends"]),
        _number(v["fees"]),
        _number(v["turnover"]),
        _number(v["cumulative_residual"]),
        _integer(v["cursor_us"]),
        None if v["open_us"] is None else _integer(v["open_us"]),
        _text(v["phase"]),
        frozenset(ids),
        _boolean(v["forced_liquidation_enabled"]),
        None if v["forced_sale_us"] is None else _integer(v["forced_sale_us"]),
    )


def _event(v: dict[str, object]) -> LedgerEvent:
    return LedgerEvent(
        _text(v["event_id"]),
        _integer(v["timestamp_us"]),
        _text(v["kind"]),
        _text(v["before_id"]),
        _text(v["after_id"]),
        _vector(v["price_pnl"]),
        _vector(v["dividends"]),
        _number(v["fee"]),
        _number(v["turnover"]),
        _number(v["nav_before"]),
        _number(v["nav_after"]),
        _number(v["residual"]),
        _vector(v["shares_before"]),
        _vector(v["shares_after"]),
        _number(v["cash_before"]),
        _number(v["cash_after"]),
        _integer(v["phase_order"]),
    )


def _mark(v: dict[str, object]) -> NavMark:
    return NavMark(
        _integer(v["timestamp_us"]), MarkKind(_text(v["kind"])), _number(v["value"]), _text(v["state_id"])
    )


def _decision(v: dict[str, object]) -> DecisionRecord:
    return DecisionRecord(
        _integer(v["anchor_index"]),
        _text(v["origin_id"]),
        _text(v["forecast_id"]),
        _integer(v["forecast_decision_us"]),
        _integer(v["decision_us"]),
        _text(v["valuation_state_id"]),
        _typed(v["selection"], PolicyDecision),
        _typed(v["fill"], LedgerEvent),
    )


def _score(v: dict[str, object]) -> ForecastScore:
    arrays = (
        "marginal_crps",
        "loss_probabilities",
        "loss_outcomes",
        "raw_marginal_crps",
        "raw_mean_squared_errors",
        "interval_lower_raw",
        "interval_upper_raw",
        "interval_coverage",
        "interval_outcome_boundary_ties",
    )
    decoded = {name: _typed(v[name], np.ndarray) for name in arrays}
    return ForecastScore(
        _text(v["origin_id"]),
        _text(v["forecast_id"]),
        _integer(v["anchor_index"]),
        _number(v["energy"]),
        _number(v["variogram"]),
        decoded["marginal_crps"],
        _number(v["mean_squared_error"]),
        decoded["loss_probabilities"],
        decoded["loss_outcomes"],
        decoded["raw_marginal_crps"],
        decoded["raw_mean_squared_errors"],
        decoded["interval_lower_raw"],
        decoded["interval_upper_raw"],
        decoded["interval_coverage"],
        decoded["interval_outcome_boundary_ties"],
    )


def _cell(v: dict[str, object]) -> CellResult:
    return CellResult(
        _typed(v["spec"], CellSpec),
        _typed(v["final_state"], LedgerState),
        _items(v["events"], LedgerEvent),
        _items(v["decisions"], DecisionRecord),
        _items(v["scores"], ForecastScore),
        _items(v["daily_marks"], NavMark),
        _items(v["boundary_marks"], NavMark),
    )


def _failure(v: dict[str, object]) -> CellFailure:
    return CellFailure(
        _text(v["cell_id"]),
        _text(v["code"]),
        _text(v["message"]),
        None if v["partial"] is None else _typed(v["partial"], CellResult),
    )


_BUILDERS: dict[str, Callable[[dict[str, object]], object]] = {
    "CellSpec": _spec,
    "Action": _action,
    "FeeSolution": _fees,
    "DecisionCertificate": _certificate,
    "PolicyDecision": _policy,
    "LedgerState": _state,
    "LedgerEvent": _event,
    "NavMark": _mark,
    "DecisionRecord": _decision,
    "ForecastScore": _score,
    "CellResult": _cell,
    "CellFailure": _failure,
}
_FIELDS = {
    record.__name__: frozenset(field.name for field in fields(record))
    for record in (
        CellSpec,
        Action,
        FeeSolution,
        DecisionCertificate,
        PolicyDecision,
        LedgerState,
        LedgerEvent,
        NavMark,
        DecisionRecord,
        ForecastScore,
        CellResult,
        CellFailure,
    )
}


def _array(value: JSONObject, owner: str, name: str) -> FloatArray:
    expected = {"array_schema", "dtype", "shape", "unit", "context", "values"}
    unit = _ARRAY_UNITS.get(owner, {}).get(name)
    if (
        set(value) != expected
        or unit is None
        or (value["array_schema"], value["dtype"], value["unit"], value["context"])
        != ("s10-typed-array-v1", np.dtype("float64").str, unit, f"{owner}.{name}")
    ):
        raise IntegrityError("saved array schema, dtype, unit or owner field is inconsistent")
    shape = (
        (3, 3)
        if name.startswith("interval_")
        else (4,)
        if name in ("loss_probabilities", "loss_outcomes")
        else (3,)
    )
    if value["shape"] != list(shape):
        raise IntegrityError("saved array declares the wrong shape for its field")
    _validate_numeric_array(value["values"])
    return frozen_f64(np.asarray(value["values"], dtype=np.float64), shape=shape, name=f"{owner}.{name}")


def _validate_numeric_array(value: JSONValue) -> None:
    if isinstance(value, list):
        for item in value:
            _validate_numeric_array(item)
    elif type(value) not in (int, float):
        raise IntegrityError("typed float array contains nonnumeric or boolean values")


def _decode(value: JSONValue, *, owner: str = "", name: str = "") -> object:
    if isinstance(value, list):
        return tuple(_decode(item) for item in value)
    if not isinstance(value, dict):
        return value
    if "array_schema" in value:
        return _array(value, owner, name)
    kind = _text(value.get("record_type"))
    content = object_json(value.get("fields"))
    if kind not in _BUILDERS or set(value) != {"record_type", "fields"} or set(content) != _FIELDS[kind]:
        raise IntegrityError("unknown saved record or missing/unexpected scientific fields")
    decoded = {field: _decode(item, owner=kind, name=field) for field, item in content.items()}
    return _BUILDERS[kind](decoded)


def _validate_cell(cell: CellResult, *, complete: bool) -> None:
    state = cell.final_state
    if state.cell_id != cell.spec.cell_id or (complete and state.phase != "FINAL"):
        raise IntegrityError("saved cell state identity or terminal phase mismatch")
    events = cell.events
    if len({event.event_id for event in events}) != len(events):
        raise IntegrityError("saved cell contains duplicate events")
    for left, right in pairwise(events):
        if left.after_id != right.before_id or (left.timestamp_us, left.phase_order) >= (
            right.timestamp_us,
            right.phase_order,
        ):
            raise IntegrityError("saved event hash chain or phase chronology is broken")
    if events and events[-1].after_id != state.record_id:
        raise IntegrityError("saved last event does not match reconstructed state identity")
    _validate_accounting(cell)
    _validate_decisions(cell, complete=complete)
    _validate_marks(cell)


def _validate_accounting(cell: CellResult) -> None:
    state = cell.final_state
    for event in cell.events:
        residual = (
            event.nav_after
            - event.nav_before
            - float(event.price_pnl.sum() + event.dividends.sum())
            + event.fee
        )
        if abs(residual - event.residual) > 1e-10 * max(1.0, event.nav_before, event.nav_after):
            raise IntegrityError("saved event violates its dollar accounting certificate")
    for name, expected in (("price_pnl", state.price_pnl), ("dividends", state.dividends)):
        actual = sum((getattr(event, name) for event in cell.events), np.zeros(3))
        if not np.allclose(actual, expected, atol=1e-8, rtol=1e-10):
            raise IntegrityError("saved event attribution does not match final state")
    if abs(sum(event.fee for event in cell.events) - state.fees) > 1e-10 * max(1.0, state.nav):
        raise IntegrityError("saved event fees do not match final state")


def _validate_decisions(cell: CellResult, *, complete: bool) -> None:
    events = {event.event_id: event for event in cell.events}
    if complete and len(cell.decisions) != len(cell.scores):
        raise IntegrityError("completed saved cell has unpaired decisions and scores")
    if len(cell.scores) > len(cell.decisions):
        raise IntegrityError("saved partial cell contains scores without issued decisions")
    for decision in cell.decisions:
        event = events.get(decision.fill.event_id)
        if (
            event is None
            or event.after_id != decision.fill.after_id
            or decision.valuation_state_id != event.before_id
        ):
            raise IntegrityError("saved decision fill does not link to its executed ledger event")
        _validate_fill(decision, event)
        if (
            decision.forecast_id != decision.selection.forecast_id
            or decision.forecast_decision_us > decision.decision_us
        ):
            raise IntegrityError("saved decision forecast identity or causal issue time is invalid")
    for decision, score in zip(cell.decisions, cell.scores, strict=False):
        if (decision.origin_id, decision.anchor_index, decision.forecast_id) != (
            score.origin_id,
            score.anchor_index,
            score.forecast_id,
        ):
            raise IntegrityError("saved score does not match its issued decision")
    if complete:
        _validate_complete_marks(cell)


def _validate_complete_marks(cell: CellResult) -> None:
    if len(cell.log_increments) != len(cell.decisions):
        raise IntegrityError("saved boundary series differs from the decision block count")
    if not np.isclose(
        cell.log_increments.sum(),
        np.log(cell.final_state.nav / cell.final_state.initial_cash),
        atol=1e-10,
        rtol=1e-10,
    ):
        raise IntegrityError("saved boundary increments do not telescope to final liquidation")
    _ = cell.maximum_drawdown


def _validate_fill(decision: DecisionRecord, event: LedgerEvent) -> None:
    selection, fee = decision.selection, decision.selection.fee_solution
    tolerance = 1e-10 * max(1.0, event.nav_before, event.nav_after)
    if not np.array_equal(selection.action.weights, fee.target_weights):
        raise IntegrityError("saved action differs from its fee-solver target")
    pairs = (
        (event.fee, fee.fee),
        (event.turnover, fee.turnover),
        (event.cash_after, fee.cash),
        (event.nav_after, fee.post_fee_nav),
    )
    if any(abs(left - right) > tolerance for left, right in pairs):
        raise IntegrityError("saved chosen fee solution differs from actual executed fill")
    certificate = selection.certificate
    if not 0 <= certificate.selected_index < certificate.candidate_count:
        raise IntegrityError("saved finite-action certificate index is invalid")
    if selection.objective != certificate.selected_objective:
        raise IntegrityError("saved selected objective differs from its certificate")


def _validate_marks(cell: CellResult) -> None:
    state = cell.final_state
    initial = initial_ledger(
        cell_id=cell.spec.cell_id,
        start_us=state.start_us,
        initial_cash=state.initial_cash,
        forced_liquidation_enabled=state.forced_liquidation_enabled,
    )
    if cell.events and cell.events[0].before_id != initial.record_id:
        raise IntegrityError("saved event chain does not begin at its declared all-cash state")
    events = {event.after_id: event for event in cell.events}
    for mark in (*cell.daily_marks, *cell.boundary_marks):
        if mark.kind == MarkKind.INITIAL_CAPITAL:
            if (mark.state_id, mark.timestamp_us, mark.value) != (
                initial.record_id,
                initial.start_us,
                initial.initial_cash,
            ):
                raise IntegrityError("saved initial capital mark differs from actual initial state")
            continue
        event = events.get(mark.state_id)
        if event is None or event.timestamp_us != mark.timestamp_us:
            raise IntegrityError("saved mark does not identify its actual event clock and state")
        expected = event.nav_after
        if mark.kind == MarkKind.LIQUIDATABLE_BOUNDARY:
            expected = event.cash_after + (1 - cell.spec.one_way_cost) * (event.nav_after - event.cash_after)
        if abs(mark.value - expected) > 1e-10 * max(1.0, expected):
            raise IntegrityError(
                "saved NAV mark differs from its actual or hypothetical valuation convention"
            )


def load_campaign(directory: Path, manifest_sha256: str) -> CampaignResult:
    """Reconstruct a complete or failed-prefix campaign only after every hash verifies.

    This is a read-only decoder. It invokes record validation and accounting
    checks, never fitting, policy selection, source readers or execution.
    Generated evidence retains its explicit scope and gains no source authority.
    """
    manifest = verify_publication(directory, manifest_sha256)
    rows = manifest["files"]
    if not isinstance(rows, list):
        raise IntegrityError("verified inventory files must remain an ordered list")
    header_row = object_json(rows[0])
    header = object_json(
        read_verified_json(directory / _text(header_row["name"]), _text(header_row["sha256"]))
    )
    results: list[CellResult] = []
    failure: CellFailure | None = None
    for raw in rows:
        row = object_json(raw)
        if "cell_id" not in row:
            continue
        payload = object_json(
            read_verified_json(directory / _text(row["name"]), _text(row["sha256"]), maximum_bytes=134217728)
        )
        decoded = _decode(payload["payload"])
        if isinstance(decoded, CellFailure):
            failure = decoded
            if decoded.partial is not None:
                if (
                    decoded.partial.spec.cell_id != decoded.cell_id
                    or payload["last_state_id"] != decoded.partial.final_state.record_id
                ):
                    raise IntegrityError("saved failure identity differs from its retained partial ledger")
                _validate_cell(decoded.partial, complete=False)
            break
        cell = _typed(decoded, CellResult)
        _validate_cell(cell, complete=True)
        if payload["last_state_id"] != cell.final_state.record_id or row["cell_id"] != cell.spec.cell_id:
            raise IntegrityError("saved publication identity differs from reconstructed cell")
        results.append(cell)
    result = CampaignResult(
        tuple(results),
        failure,
        tuple(cell.cell_id for cell in campaign_matrix()),
        _text(header["tape_identity"]),
        _text(header["evidence_scope"]),
    )
    if result.complete != manifest["complete"] or len(results) != manifest["completed_cells"]:
        raise IntegrityError("decoded campaign completeness differs from verified manifest")
    return result
