"""Paired date-block inference. Seeds are repetitions, not new market histories."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class BlockBootstrapResult:
    estimate: float
    ci_lower: float
    ci_upper: float
    p_two_sided: float
    dates: int
    block_length: int
    n_resamples: int
    seed: int
    confidence: float
    estimand: str = "mean of equally weighted date means"
    method: str = "circular moving-date-block percentile CI; centered bootstrap test"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _date_means(frame: pd.DataFrame, value_column: str, time_column: str) -> np.ndarray:
    if {value_column, time_column} - set(frame):
        raise ValueError("Missing value/time columns")
    values = pd.to_numeric(frame[value_column], errors="raise").to_numpy(float)
    if not len(values) or not np.all(np.isfinite(values)):
        raise ValueError("Bootstrap inputs must be nonempty and finite; failed rows cannot disappear")
    # Every asset on the same UTC date stays in the same resampled block.
    dates = pd.to_datetime(frame[time_column], utc=True).dt.normalize()
    if dates.isna().any() or ("asset_id" in frame and frame["asset_id"].isna().any()):
        raise ValueError("Missing date/asset keys cannot disappear from a bootstrap group")
    rows = pd.DataFrame({"date": dates, "value": values})
    if "asset_id" in frame:
        rows["asset_id"] = frame["asset_id"].to_numpy()
        grouped = rows.groupby(["date", "asset_id"], sort=True)["value"].mean().groupby(level="date").mean()
    else:
        grouped = rows.groupby("date", sort=True)["value"].mean()
    return grouped.to_numpy(float)


def date_block_bootstrap(
    frame: pd.DataFrame,
    value_column: str,
    block_length: int,
    n_resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
    time_column: str = "time",
    minimum_block_length: int = 1,
) -> BlockBootstrapResult:
    """Resample date means with circular contiguous blocks, preserving peers.

    Block length must be preregistered and at least the target overlap length in
    calendar-date groups. It may need to be longer for serial dependence.
    """
    values = _date_means(frame, value_column, time_column)
    n = len(values)
    if block_length < minimum_block_length or block_length < 1 or block_length > n:
        raise ValueError("Invalid block length or below the predeclared overlap bound")
    if n < 2 * block_length:
        raise ValueError("Fewer than two blocks of observed dates: inference is not informative")
    if n_resamples < 199 or not 0 < confidence < 1:
        raise ValueError("At least 199 resamples and confidence in (0,1) are required")
    rng = np.random.default_rng(seed)
    blocks = int(np.ceil(n / block_length))
    means = np.empty(n_resamples)
    estimate = float(values.mean())
    offsets = np.arange(block_length)
    # Batch resamples to remove the Python-level bootstrap loop while bounding
    # peak memory for long date histories.
    batch_size = 256
    for begin in range(0, n_resamples, batch_size):
        stop = min(begin + batch_size, n_resamples)
        starts = rng.integers(0, n, size=(stop - begin, blocks))
        indices = ((starts[..., None] + offsets) % n).reshape(stop - begin, -1)[:, :n]
        means[begin:stop] = values[indices].mean(axis=1)
    centered_means = means - estimate
    alpha = (1 - confidence) / 2
    lower, upper = np.quantile(means, [alpha, 1 - alpha])
    p = float((1 + np.sum(np.abs(centered_means) >= abs(estimate))) / (n_resamples + 1))
    return BlockBootstrapResult(
        estimate, float(lower), float(upper), p, n, block_length, n_resamples, seed, confidence
    )


def paired_date_test(
    left: pd.DataFrame,
    right: pd.DataFrame,
    value_column: str,
    block_length: int,
    keys: tuple[str, ...] = ("time", "asset_id"),
    n_resamples: int = 2000,
    seed: int = 0,
    minimum_block_length: int = 1,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """A negative left-minus-right difference favors left for losses.

    Requiring exactly matching keys prevents silent favorable sample selection.
    Multiple seeds should be averaged within the same date/asset/model before
    this test, with seed spread reported separately.
    """
    for name, frame in (("left", left), ("right", right)):
        missing = (set(keys) | {value_column}) - set(frame.columns)
        if missing:
            raise ValueError(f"{name} missing {sorted(missing)}")
        if frame[list(keys)].isna().any().any():
            raise ValueError(f"{name} contains missing paired keys")
        if frame.duplicated(list(keys)).any():
            raise ValueError(f"{name} has duplicate paired keys; average seeds separately")
    merged = left[list(keys) + [value_column]].merge(
        right[list(keys) + [value_column]],
        on=list(keys),
        how="outer",
        validate="one_to_one",
        suffixes=("_left", "_right"),
        indicator=True,
    )
    if not merged["_merge"].eq("both").all():
        raise ValueError(
            "Model samples differ; preserve/report the failure instead of dropping unmatched rows"
        )
    merged["difference"] = merged[value_column + "_left"] - merged[value_column + "_right"]
    result = date_block_bootstrap(
        merged,
        "difference",
        block_length,
        n_resamples=n_resamples,
        seed=seed,
        minimum_block_length=minimum_block_length,
        confidence=confidence,
    )
    return {**result.to_dict(), "difference": "left minus right", "paired_rows": len(merged)}


def minimum_overlap_block_length(
    frame: pd.DataFrame, time_column: str = "time", label_time_column: str = "label_time"
) -> int:
    """Conservative lower bound from all decision-date groups before maturity.

    This is a lower bound, not an automatic statistically optimal block selector.
    Choose a larger frozen block length if training-only diagnostics warrant it.
    """
    if {time_column, label_time_column} - set(frame) or not len(frame):
        raise ValueError("Nonempty decision and label-time columns are required")
    decision = pd.to_datetime(frame[time_column], utc=True).dt.normalize().astype("int64").to_numpy()
    maturity = pd.to_datetime(frame[label_time_column], utc=True).dt.normalize().astype("int64").to_numpy()
    if np.any(maturity < decision):
        raise ValueError("Label maturity precedes decision")
    dates = np.unique(decision)
    begin = np.searchsorted(dates, decision, side="left")
    end = np.searchsorted(dates, maturity, side="right")
    return max(1, int(np.max(end - begin)))


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    """Holm step-down familywise adjustment over the declared complete family."""
    if not p_values:
        raise ValueError("No hypothesis family supplied")
    if any(not np.isfinite(p) or p < 0 or p > 1 for p in p_values.values()):
        raise ValueError("P-values must be finite numbers in [0,1]")
    ordered = sorted(p_values, key=p_values.get)
    maximum = 0.0
    adjusted: dict[str, float] = {}
    m = len(ordered)
    for rank, name in enumerate(ordered):
        maximum = max(maximum, min(1.0, (m - rank) * p_values[name]))
        adjusted[name] = maximum
    return {name: adjusted[name] for name in p_values}


def average_seeds(
    frame: pd.DataFrame,
    value_columns: list[str],
    keys: tuple[str, ...] = ("time", "asset_id"),
    seed_column: str = "seed",
) -> pd.DataFrame:
    """Average matched seed repetitions; missing runs invalidate aggregation."""
    required = set(keys) | set(value_columns) | {seed_column}
    if required - set(frame):
        raise ValueError(f"Missing columns {sorted(required - set(frame))}")
    if not len(frame) or frame[list(keys) + [seed_column]].isna().any().any():
        raise ValueError("Seed aggregation requires nonempty observations and complete nonmissing keys")
    if frame.duplicated(list(keys) + [seed_column]).any():
        raise ValueError("Duplicate seed/observation rows")
    for column in value_columns:
        if not np.isfinite(pd.to_numeric(frame[column], errors="raise").to_numpy(float)).all():
            raise ValueError("Nonfinite seed scores; a failed run cannot be averaged away")
    expected = set(frame[seed_column].unique())
    for _, rows in frame.groupby(list(keys), sort=False):
        if set(rows[seed_column]) != expected:
            raise ValueError("Not every observation has every seed; retain the failed-run record")
    output = frame.groupby(list(keys), sort=True)[value_columns].mean().reset_index()
    output.attrs["seed_count"] = len(expected)
    output.attrs["seed_is_independent_market_history"] = False
    return output
