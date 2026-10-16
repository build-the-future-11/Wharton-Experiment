"""Explicit bounded JSON schema for engineering-only forecast-state replay."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import fields, is_dataclass
from typing import cast

import numpy as np
from numpy.typing import NDArray

from frem_s10.data.source_reader import strict_json_loads
from frem_s10.domain import ForecastDistribution, ScenarioLibrary, Transform
from frem_s10.errors import ContractError, IntegrityError
from frem_s10.math.simplex import CertifiedMixture, FaceCertificate
from frem_s10.model.config import ForecastConfig
from frem_s10.model.engine import state_identity
from frem_s10.model.records import (
    ExpertSet,
    ForecastState,
    IssuedForecast,
    IssueRequest,
    MaturedFeedback,
    MixtureBundle,
)

_TYPES = {
    kind.__name__: kind
    for kind in (
        Transform,
        ScenarioLibrary,
        ForecastDistribution,
        ForecastConfig,
        IssueRequest,
        ExpertSet,
        MaturedFeedback,
        IssuedForecast,
        ForecastState,
        CertifiedMixture,
        FaceCertificate,
        MixtureBundle,
    )
}


def _encode(value: object) -> object:
    if isinstance(value, np.ndarray):
        dtype = "<i8" if value.dtype.kind in "iu" else "<f8"
        return {
            "array": dtype,
            "shape": list(value.shape),
            "hex": value.astype(dtype, copy=False).tobytes(order="C").hex(),
        }
    if is_dataclass(value) and not isinstance(value, type):
        return {
            "record": type(value).__name__,
            "fields": {field.name: _encode(getattr(value, field.name)) for field in fields(value)},
        }
    if isinstance(value, (tuple, frozenset)):
        elements = sorted(value) if isinstance(value, frozenset) else value
        return {
            "container": "set" if isinstance(value, frozenset) else "tuple",
            "items": [_encode(item) for item in elements],
        }
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise ContractError("unsupported checkpoint value type")


def _decode_record(value: dict[str, object], depth: int) -> object:
    kind, members = value["record"], value["fields"]
    if not isinstance(kind, str) or kind not in _TYPES or not isinstance(members, dict):
        raise ContractError("unknown checkpoint record")
    cls = _TYPES[kind]
    if set(members) != {field.name for field in fields(cls)}:
        raise ContractError("checkpoint fields differ from exact record schema")
    try:
        constructor = cast(Callable[..., object], cls)
        return constructor(**{key: _decode(item, depth + 1) for key, item in members.items()})
    except (AttributeError, TypeError, ValueError, OverflowError) as exc:
        raise ContractError("checkpoint record constructor rejected values") from exc


def _decode_array(value: dict[str, object]) -> NDArray[np.float64] | NDArray[np.int64]:
    dtype, shape, hex_data = value["array"], value["shape"], value["hex"]
    if (
        not isinstance(dtype, str)
        or dtype not in ("<f8", "<i8")
        or not isinstance(shape, list)
        or any(type(dim) is not int or dim < 0 for dim in shape)
        or not isinstance(hex_data, str)
    ):
        raise ContractError("checkpoint array schema rejected")
    try:
        array = np.frombuffer(bytes.fromhex(hex_data), dtype=dtype).reshape(tuple(shape))
    except (ValueError, OverflowError) as exc:
        raise ContractError("checkpoint array bytes rejected") from exc
    return array


def _decode(value: object, depth: int = 0) -> object:
    if depth > 32:
        raise ContractError("checkpoint nesting exceeds schema limit")
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if not isinstance(value, dict):
        raise ContractError("unsupported checkpoint node")
    if set(value) == {"record", "fields"}:
        return _decode_record(value, depth)
    if set(value) == {"array", "shape", "hex"}:
        return _decode_array(value)
    if set(value) == {"container", "items"}:
        return _decode_collection(value, depth)
    raise ContractError("unrecognized checkpoint schema node")


def _decode_collection(value: dict[str, object], depth: int) -> tuple[object, ...] | frozenset[object]:
    if value["container"] not in ("tuple", "set") or not isinstance(value["items"], list):
        raise ContractError("checkpoint collection schema rejected")
    items = tuple(_decode(item, depth + 1) for item in value["items"])
    if value["container"] == "tuple":
        return items
    try:
        result = frozenset(items)
    except TypeError as exc:
        raise ContractError("checkpoint identity collection contains an unhashable member") from exc
    if len(result) != len(items):
        raise ContractError("checkpoint identity collection contains duplicate members")
    return result


def dump_state(state: ForecastState) -> str:
    """Serialize exact bytes without pickle; no file or market resume operation."""
    if state.record_id != state_identity(state):
        raise IntegrityError("cannot serialize a state with a mismatched identity")
    payload = json.dumps(_encode(state), sort_keys=True, separators=(",", ":"), allow_nan=False)
    envelope = {
        "schema": "S10-FORECAST-ENGINEERING-CHECKPOINT-1",
        "market_resume_allowed": False,
        "payload": payload,
        "sha256": hashlib.sha256(payload.encode()).hexdigest(),
    }
    return json.dumps(envelope, sort_keys=True, separators=(",", ":"), allow_nan=False)


def load_state(
    text: str, *, expected_state_id: str, expected_transform_id: str, expected_config: ForecastConfig
) -> ForecastState:
    """Validate exact identities before enabling constructed engineering replay."""
    if not isinstance(text, str) or len(text.encode()) > 16 * 1024 * 1024:
        raise ContractError("checkpoint exceeds 16 MiB byte bound")
    envelope = _parse_document(text)
    if (
        set(envelope) != {"schema", "market_resume_allowed", "payload", "sha256"}
        or envelope["schema"] != "S10-FORECAST-ENGINEERING-CHECKPOINT-1"
        or envelope["market_resume_allowed"] is not False
        or not isinstance(envelope["payload"], str)
    ):
        raise ContractError("checkpoint envelope rejected")
    payload = envelope["payload"]
    if hashlib.sha256(payload.encode()).hexdigest() != envelope["sha256"]:
        raise IntegrityError("checkpoint payload hash mismatch")
    state = _decode(_parse_document(payload))
    if not isinstance(state, ForecastState):
        raise ContractError("checkpoint root is not a forecast state")
    if (
        state.record_id != expected_state_id
        or state.record_id != state_identity(state)
        or state.transform.record_id != expected_transform_id
        or state.config != expected_config
    ):
        raise IntegrityError("checkpoint scientific identity mismatch")
    return state


def _parse_document(text: str) -> dict[str, object]:
    """Map malformed JSON resource/type failures to a rejected checkpoint."""
    try:
        return strict_json_loads(text)
    except (RecursionError, TypeError, ValueError, OverflowError) as exc:
        raise ContractError("checkpoint JSON structure exceeds its supported schema") from exc
