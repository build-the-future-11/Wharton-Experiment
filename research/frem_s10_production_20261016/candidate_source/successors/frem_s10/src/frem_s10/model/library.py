"""Strictly matured bounded scenario membership and declared A1 atom mapping."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from frem_s10.domain import OriginRecord, ScenarioLibrary
from frem_s10.errors import ChronologyError, ContractError
from frem_s10.model.config import ForecastConfig


def empty_library() -> ScenarioLibrary:
    """Construct the zero-member library before causal warm-up."""
    return ScenarioLibrary(
        (),
        np.empty(0, dtype=np.int64),
        np.empty(0, dtype=np.int64),
        np.empty((0, 3), dtype=np.float64),
        np.empty((0, 3), dtype=np.float64),
        np.empty((0, 12), dtype=np.float64),
    )


def update_library(
    previous: ScenarioLibrary,
    matured: tuple[OriginRecord, ...],
    *,
    decision_us: int,
    config: ForecastConfig,
) -> ScenarioLibrary:
    """Append only newly admitted mature records; eviction follows stable origin order."""
    identities = set(previous.record_ids)
    rows: list[tuple[int, str, int, NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]] = [
        (
            int(previous.origin_us[i]),
            record_id,
            int(previous.available_us[i]),
            previous.g[i],
            previous.d[i],
            previous.contexts[i],
        )
        for i, record_id in enumerate(previous.record_ids)
    ]
    for record in matured:
        if record.available_us >= decision_us or record.feature_available_us >= record.origin_us + 1:
            raise ChronologyError("scenario record is not causally available")
        if record.record_id in identities:
            raise ChronologyError("scenario identity already present")
        identities.add(record.record_id)
        g, d = record.g, record.d
        if config.repeat_h1:
            if record.h1_g is None or record.h1_d is None:
                raise ContractError("A1 requires same-origin weekly price/dividend components")
            g1, d1 = record.h1_g, record.h1_d
            with np.errstate(over="raise", invalid="raise"):
                g, d = g1**4, d1 * (1 + g1 + g1**2 + g1**3)
        rows.append((record.origin_us, record.record_id, record.available_us, g, d, record.x))
    rows.sort(key=lambda row: (row[0], row[1]))
    rows = rows[-config.library_window :]
    if not rows:
        return empty_library()
    return ScenarioLibrary(
        tuple(row[1] for row in rows),
        np.array([row[0] for row in rows], dtype=np.int64),
        np.array([row[2] for row in rows], dtype=np.int64),
        np.stack([row[3] for row in rows]),
        np.stack([row[4] for row in rows]),
        np.stack([row[5] for row in rows]),
    )
