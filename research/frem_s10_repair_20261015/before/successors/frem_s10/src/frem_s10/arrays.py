"""Immutable, shape-checked float64 and int64 scientific array boundaries."""

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from .errors import ContractError
from .metadata import Unit

FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]
Shape: TypeAlias = tuple[int | None, ...]


def _check_shape(actual: tuple[int, ...], expected: Shape, name: str) -> None:
    if len(actual) != len(expected) or any(e is not None and a != e for a, e in zip(actual, expected)):
        raise ContractError(f"{name}: expected shape {expected}, got {actual}", field=name)


def _reject_bool_sequence(value: object, name: str) -> None:
    if isinstance(value, (bool, np.bool_)):
        raise ContractError(f"{name}: booleans are not numeric observations", field=name)
    if isinstance(value, (list, tuple)):
        for item in value:
            _reject_bool_sequence(item, name)


def frozen_f64(value: object, *, shape: Shape, name: str) -> FloatArray:
    """Return finite float64 data whose bytes backing cannot become writable.

    Explicit NumPy arrays must already be float64. Numeric Python sequences are
    an explicit intake conversion; object, complex, bool and float32 arrays fail.
    Empty axes are allowed only where the caller's record contract permits them.
    """
    _reject_bool_sequence(value, name)
    if isinstance(value, np.ndarray) and value.dtype != np.dtype(np.float64):
        raise ContractError(f"{name}: float64 array required", field=name)
    try:
        raw = np.asarray(value)
        if raw.dtype.kind not in "iuf" or raw.dtype.kind == "b":
            raise ContractError(f"{name}: real numeric array required", field=name)
        converted = np.asarray(value, dtype=np.float64, order="C")
    except (TypeError, ValueError, OverflowError) as exc:
        raise ContractError(f"{name}: numeric array conversion failed", field=name) from exc
    _check_shape(converted.shape, shape, name)
    if not np.all(np.isfinite(converted)):
        raise ContractError(f"{name}: all entries must be finite", field=name)
    result = np.frombuffer(converted.astype("<f8", copy=False).tobytes(), dtype="<f8")
    return result.reshape(converted.shape)


def frozen_i64(value: object, *, shape: Shape, name: str) -> IntArray:
    """Return immutable integer64 data without float truncation or bool coercion."""
    _reject_bool_sequence(value, name)
    raw = np.asarray(value)
    if raw.dtype.kind not in "iu" or (isinstance(value, np.ndarray) and value.dtype != np.dtype(np.int64)):
        raise ContractError(f"{name}: int64 integer array required", field=name)
    _check_shape(raw.shape, shape, name)
    if raw.size and (np.any(raw < -(2**63)) or np.any(raw > 2**63 - 1)):
        raise ContractError(f"{name}: int64 range exceeded", field=name)
    converted = np.asarray(raw, dtype="<i8", order="C")
    return np.frombuffer(converted.tobytes(), dtype="<i8").reshape(converted.shape)


@dataclass(frozen=True, slots=True)
class QuantityArray:
    """Named-unit immutable array for APIs where shape alone is insufficient."""

    values: FloatArray
    unit: Unit

    def __post_init__(self) -> None:
        if not isinstance(self.unit, Unit):
            raise ContractError("declared Unit enum required", field="unit")
        values = frozen_f64(self.values, shape=self.values.shape, name="quantity")
        if self.unit in (Unit.PRICE_GROSS, Unit.PRICE_USD) and np.any(values <= 0):
            raise ContractError("strictly positive price quantity required")
        if self.unit in (Unit.DIVIDEND_PER_START_DOLLAR, Unit.USD, Unit.SHARES,
                         Unit.WEIGHT, Unit.PROBABILITY) and np.any(values < 0):
            raise ContractError("nonnegative quantity required")
        if self.unit in (Unit.WEIGHT, Unit.PROBABILITY) and np.any(values > 1 + 1e-12):
            raise ContractError("probability/weight exceeds one")
        object.__setattr__(self, "values", values)

    def require(self, unit: Unit, *, shape: Shape) -> FloatArray:
        """Expose the immutable array only under its matching unit and shape."""
        if self.unit is not unit:
            raise ContractError(f"unit mismatch: expected {unit}, got {self.unit}", field="unit")
        _check_shape(self.values.shape, shape, "quantity")
        return self.values
