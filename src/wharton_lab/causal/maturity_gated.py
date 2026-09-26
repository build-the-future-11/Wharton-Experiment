"""Forecaster that only trains on labels matured as-of a fixed clock."""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np

from wharton_lab.splits.clock import InformationClock


class MaturityGatedLinearForecaster:
    """Ridge-style linear model with explicit label maturity gating."""

    def __init__(self, *, horizon_days: int = 5, reporting_lag_days: int = 0, alpha: float = 1.0):
        self.horizon_days = horizon_days
        self.reporting_lag_days = reporting_lag_days
        self.alpha = alpha
        self._coef: np.ndarray | None = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        decision_times: list[datetime],
        as_of: datetime,
    ) -> MaturityGatedLinearForecaster:
        clk = InformationClock(decision_times[0], self.horizon_days, self.reporting_lag_days)
        mature_idx = clk.filter_matured(decision_times, as_of)
        if not mature_idx:
            raise ValueError("No matured labels as-of clock")
        Xm = np.asarray(X, dtype=float)[mature_idx]
        ym = np.asarray(y, dtype=float).ravel()[mature_idx]
        d = Xm.shape[1]
        self._coef = np.linalg.solve(
            Xm.T @ Xm + self.alpha * np.eye(d),
            Xm.T @ ym,
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._coef is None:
            raise RuntimeError("Not fitted")
        return np.asarray(X, dtype=float) @ self._coef


def decision_time_series(start: datetime, n: int, step_days: int = 1) -> list[datetime]:
    return [start + timedelta(days=i * step_days) for i in range(n)]
