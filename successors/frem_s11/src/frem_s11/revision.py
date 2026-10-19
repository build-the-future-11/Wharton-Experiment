"""Common-cutoff release-vintage revision diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from frem_s11.data import ReturnPanel
from frem_s11.errors import DataContractError


@dataclass(frozen=True, slots=True)
class RevisionAudit:
    """Exact aligned differences between two release vintages."""

    observations: int
    changed_values: int
    maximum_absolute_revision: float
    mean_absolute_revision: float
    older_sha256: str
    newer_sha256: str


def compare_vintages(older: ReturnPanel, newer: ReturnPanel, *, through: int) -> RevisionAudit:
    """Compare only months available in both releases through the declared cutoff."""
    left = older.through(through)
    right = newer.through(through)
    if not np.array_equal(left.months, right.months):
        raise DataContractError("release vintages do not have identical common-cutoff months")
    difference = np.abs(left.returns - right.returns)
    return RevisionAudit(
        len(left.months),
        int(np.count_nonzero(difference)),
        float(difference.max()),
        float(difference.mean()),
        left.source_sha256,
        right.source_sha256,
    )
