import hashlib

import numpy as np
import pytest

from frem_s11.data import ReturnPanel
from frem_s11.errors import DataContractError
from frem_s11.revision import compare_vintages


def _panel(months: np.ndarray, value: float, name: bytes) -> ReturnPanel:
    returns = np.full((len(months), 6), value, dtype=np.float64)
    return ReturnPanel(months, returns, 201701, hashlib.sha256(name).hexdigest())


def _months(year: int, month: int, length: int) -> np.ndarray:
    rows = []
    for _ in range(length):
        rows.append(year * 100 + month)
        month += 1
        if month == 13:
            year, month = year + 1, 1
    return np.asarray(rows)


def test_revision_audit_aligns_exact_common_cutoff() -> None:
    months = np.asarray(
        [2000 * 100 + month for month in range(1, 13)] + [2001 * 100 + month for month in range(1, 13)]
    )
    older = _panel(months, 0.01, b"older")
    newer = _panel(months, 0.02, b"newer")
    audit = compare_vintages(older, newer, through=200112)
    assert audit.observations == 24 and audit.changed_values == 144
    assert audit.maximum_absolute_revision == pytest.approx(0.01)


def test_revision_audit_rejects_different_month_axes() -> None:
    left = _months(2000, 1, 30)
    right = _months(1999, 12, 31)
    with pytest.raises(DataContractError, match="identical"):
        compare_vintages(_panel(left, 0.01, b"left"), _panel(right, 0.01, b"right"), through=200206)
