"""Canonical finite JSON and scientific identities without filesystem access."""

from __future__ import annotations

import hashlib
import json
import math

from frem_s10.errors import ContractError

type JSONValue = bool | int | float | str | list[JSONValue] | dict[str, JSONValue] | None
type JSONObject = dict[str, JSONValue]


def checked_json(value: object) -> JSONValue:
    """Validate recursively, rejecting nonfinite scalars and arbitrary objects."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ContractError("JSON contains a nonfinite number")
        return value
    if isinstance(value, list):
        return [checked_json(item) for item in value]
    if isinstance(value, dict):
        result: JSONObject = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ContractError("JSON object keys must be strings")
            result[key] = checked_json(item)
        return result
    raise ContractError(f"Unsupported JSON value type: {type(value).__name__}")


def _unique_object(pairs: list[tuple[str, object]]) -> JSONObject:
    result: JSONObject = {}
    for key, value in pairs:
        if key in result:
            raise ContractError(f"Duplicate JSON key: {key}")
        result[key] = checked_json(value)
    return result


def parse_json(data: str | bytes) -> JSONValue:
    """Parse untrusted JSON with duplicate-key and nonfinite rejection."""
    try:
        value: object = json.loads(data, object_pairs_hook=_unique_object)
    except (ValueError, UnicodeError) as exc:
        raise ContractError("Malformed JSON") from exc
    return checked_json(value)


def object_json(value: JSONValue) -> JSONObject:
    """Require an object at a schema boundary."""
    if not isinstance(value, dict):
        raise ContractError("Expected a JSON object")
    return value


def canonical_bytes(value: JSONValue) -> bytes:
    """Serialize finite metadata in deterministic UTF-8 key order."""
    checked_json(value)
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def digest_bytes(data: bytes) -> str:
    """Return the SHA256 content identity of exact bytes."""
    return hashlib.sha256(data).hexdigest()


def digest_json(value: JSONValue) -> str:
    """Hash canonical metadata bytes; timing remains in its own payload."""
    return digest_bytes(canonical_bytes(value))


def scientific_json(value: JSONValue) -> JSONValue:
    """Encode every JSON node with a type tag, avoiding user-object collisions."""
    checked_json(value)
    if value is None:
        return ["null"]
    if isinstance(value, bool):
        return ["bool", value]
    if isinstance(value, int):
        return ["int", str(value)]
    if isinstance(value, float):
        return ["binary64", value.hex()]
    if isinstance(value, str):
        return ["string", value]
    if isinstance(value, list):
        return ["array", [scientific_json(item) for item in value]]
    if isinstance(value, dict):
        return ["object", [[key, scientific_json(value[key])] for key in sorted(value)]]
    raise ContractError("Unsupported scientific identity value")


def scientific_digest(value: JSONValue) -> str:
    """Hash explicit finite binary64 representations and ordered metadata."""
    return digest_json(scientific_json(value))
