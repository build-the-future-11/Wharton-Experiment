"""Point-in-time market contract, feature construction, and bounded fixtures.

Synthetic generators in this module are integration fixtures only. They cannot
be admitted to the manuscript's historical or prospective scientific study.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = {
    "permanent_id",
    "timestamp",
    "available_at",
    "open",
    "high",
    "low",
    "close",
    "distribution",
    "split_factor",
    "volume",
    "currency",
    "fx_to_usd",
    "fx_available_at",
    "eligible",
    "category",
}
CATEGORIES = ("us_equity", "international_equity", "usd_fixed_income", "other")


@dataclass(frozen=True)
class PanelManifest:
    provider: str
    product: str
    license: str
    retrieved_at: str
    timezone: str
    revision_policy: str
    known_gaps: tuple[str, ...]
    source_sha256: str
    synthetic: bool = False

    @classmethod
    def from_json(cls, path: str | Path) -> PanelManifest:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        missing = {
            "provider",
            "product",
            "license",
            "retrieved_at",
            "timezone",
            "revision_policy",
            "known_gaps",
            "source_sha256",
        } - payload.keys()
        if missing:
            raise ValueError(f"Manifest missing fields: {sorted(missing)}")
        return cls(**payload)


@dataclass(frozen=True)
class RobustScaler:
    median: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray, mask: np.ndarray) -> RobustScaler:
        """Fit the six continuous primitive channels on a fitting prefix only."""
        if values.shape != mask.shape or values.shape[-1] != 6:
            raise ValueError("values and mask must share shape with six primitive channels")
        work = np.where(mask, values, np.nan)
        median = np.nanmedian(work, axis=tuple(range(work.ndim - 1)))
        mad = np.nanmedian(np.abs(work - median), axis=tuple(range(work.ndim - 1)))
        median = np.nan_to_num(median, nan=0.0)
        scale = np.maximum(np.nan_to_num(1.4826 * mad, nan=1.0), 1e-6)
        return cls(median=median, scale=scale)

    def transform(self, values: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if values.shape != mask.shape or values.shape[-1] != 6:
            raise ValueError("values and mask must share shape with six primitive channels")
        normalized = (values - self.median) / self.scale
        return np.where(mask, np.clip(normalized, -10.0, 10.0), 0.0)

    def transform_feature_channels(self, channels: np.ndarray) -> np.ndarray:
        if channels.shape[-1] != 18:
            raise ValueError("Expected the 18-channel feature contract")
        output = np.asarray(channels, dtype=float).copy()
        observed = output[..., 6:12] > 0.5
        output[..., :6] = self.transform(output[..., :6], observed)
        return output


def fit_feature_scaler(fitting_channels: np.ndarray) -> RobustScaler:
    values = np.asarray(fitting_channels, dtype=float)
    if values.shape[-1] != 18:
        raise ValueError("Expected the 18-channel feature contract")
    return RobustScaler.fit(values[..., :6], values[..., 6:12] > 0.5)


@dataclass(frozen=True)
class ExampleBatch:
    decision_times: np.ndarray
    asset_ids: tuple[str, ...]
    context: np.ndarray
    asset_mask: np.ndarray
    future: dict[int, np.ndarray]
    future_mask: dict[int, np.ndarray]
    log_returns: dict[int, np.ndarray]
    scales: dict[int, np.ndarray]
    outcome_available_at: dict[int, np.ndarray]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_panel(frame: pd.DataFrame, manifest: PanelManifest | None = None) -> dict[str, object]:
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Panel missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("Panel is empty")
    panel = frame.copy()
    for column in ("timestamp", "available_at", "fx_available_at"):
        panel[column] = pd.to_datetime(panel[column], utc=True, errors="raise")
    if panel.duplicated(["permanent_id", "timestamp"]).any():
        raise ValueError("Duplicate permanent_id/timestamp rows")
    if (panel["available_at"] < panel["timestamp"]).any():
        raise ValueError("Information cannot be available before its observation")
    if (panel["fx_available_at"] < panel["timestamp"]).any():
        raise ValueError("FX cannot be available before its observation")
    numeric = [
        "open",
        "high",
        "low",
        "close",
        "distribution",
        "split_factor",
        "volume",
        "fx_to_usd",
    ]
    if not np.isfinite(panel[numeric].to_numpy(dtype=float)).all():
        raise ValueError("Panel contains non-finite accounting values")
    if (panel[["open", "high", "low", "close", "split_factor", "fx_to_usd"]].to_numpy() <= 0).any():
        raise ValueError("Prices, split factors, and FX rates must be positive")
    if (panel["distribution"] < 0).any() or (panel["volume"] < 0).any():
        raise ValueError("Distributions and volume cannot be negative")
    if (
        (panel["high"] < panel[["open", "close"]].max(axis=1))
        | (panel["low"] > panel[["open", "close"]].min(axis=1))
    ).any():
        raise ValueError("OHLC envelope is inconsistent")
    if not set(panel["category"]).issubset(CATEGORIES):
        raise ValueError("Unknown research-universe category")
    if manifest is not None:
        if not manifest.provider or not manifest.license or len(manifest.source_sha256) != 64:
            raise ValueError("Manifest provenance is incomplete")
        if manifest.synthetic and "synthetic" not in manifest.provider.lower():
            raise ValueError("Synthetic manifests must be unmistakably labeled")
    ordered = panel.sort_values(["permanent_id", "timestamp"])
    if not ordered.index.equals(panel.index):
        panel = ordered
    return {
        "rows": int(len(panel)),
        "assets": int(panel["permanent_id"].nunique()),
        "first_timestamp": panel["timestamp"].min().isoformat(),
        "last_timestamp": panel["timestamp"].max().isoformat(),
        "synthetic": bool(manifest.synthetic) if manifest else None,
    }


def select_fold_universe(
    frame: pd.DataFrame,
    freeze_time: str | pd.Timestamp,
    *,
    slots_per_category: int = 10,
    volume_lookback: int = 60,
) -> tuple[str, ...]:
    """Select without future returns or survival, breaking ties by permanent id."""
    freeze = pd.Timestamp(freeze_time)
    if freeze.tzinfo is None:
        freeze = freeze.tz_localize("UTC")
    else:
        freeze = freeze.tz_convert("UTC")
    panel = frame.copy()
    panel["timestamp"] = pd.to_datetime(panel["timestamp"], utc=True)
    panel["available_at"] = pd.to_datetime(panel["available_at"], utc=True)
    visible = panel[
        (panel["timestamp"] <= freeze)
        & (panel["available_at"] <= freeze)
        & panel["eligible"].astype(bool)
    ]
    chosen: list[str] = []
    for category in CATEGORIES:
        group = visible[visible["category"] == category]
        rows = []
        for asset, history in group.groupby("permanent_id", sort=True):
            tail = history.sort_values("timestamp").tail(volume_lookback)
            if len(tail) < volume_lookback:
                continue
            dollar_volume = tail["close"] * tail["fx_to_usd"] * tail["volume"]
            rows.append((str(asset), float(np.median(dollar_volume))))
        rows.sort(key=lambda item: (-item[1], item[0]))
        chosen.extend(asset for asset, _ in rows[:slots_per_category])
    return tuple(chosen)


def _primitive_features(history: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    h = history.sort_values("timestamp")
    open_usd = h["open"].to_numpy(float) * h["fx_to_usd"].to_numpy(float)
    close_usd = h["close"].to_numpy(float) * h["fx_to_usd"].to_numpy(float)
    high_usd = h["high"].to_numpy(float) * h["fx_to_usd"].to_numpy(float)
    low_usd = h["low"].to_numpy(float) * h["fx_to_usd"].to_numpy(float)
    split = h["split_factor"].to_numpy(float)
    dist = h["distribution"].to_numpy(float) * h["fx_to_usd"].to_numpy(float)
    gross = np.ones(len(h), dtype=float)
    gross[1:] = (split[1:] * open_usd[1:] + dist[1:]) / open_usd[:-1]
    if (gross <= 0).any():
        raise ValueError("Non-positive wealth transition cannot be represented as a log return")
    total_log_return = np.log(gross)
    gap = np.zeros(len(h), dtype=float)
    gap[1:] = np.log((split[1:] * open_usd[1:] + dist[1:]) / close_usd[:-1])
    log_dollar_volume = np.log1p(h["volume"].to_numpy(float) * close_usd)
    volume_change = np.zeros(len(h), dtype=float)
    volume_change[1:] = np.diff(log_dollar_volume)
    primitive = np.column_stack(
        [
            total_log_return,
            np.log(close_usd / open_usd),
            np.log(high_usd / low_usd),
            gap,
            log_dollar_volume,
            volume_change,
        ]
    )
    observed = np.ones_like(primitive, dtype=bool)
    observed[0, [0, 3, 5]] = False
    return primitive, observed


def _feature_channels(primitive: np.ndarray, observed: np.ndarray) -> np.ndarray:
    if primitive.shape != observed.shape or primitive.ndim != 2:
        raise ValueError("Primitive values and observation mask must be aligned matrices")
    rows = np.arange(len(primitive), dtype=int)[:, None]
    last = np.maximum.accumulate(np.where(observed, rows, -1), axis=0)
    age = np.where(last < 0, 1.0, np.minimum((rows - last) / 5.0, 1.0))
    return np.concatenate([primitive, observed.astype(float), age], axis=1)


def opening_log_return(
    history: pd.DataFrame, start_index: int, horizon: int
) -> tuple[float, pd.Timestamp]:
    """Return from next executable open through the open after ``horizon`` intervals."""
    h = history.sort_values("timestamp").reset_index(drop=True)
    entry = start_index + 1
    exit_ = entry + horizon
    if entry < 1 or exit_ >= len(h):
        raise IndexError("Insufficient future executable observations")
    current = h.iloc[entry + 1 : exit_ + 1]
    previous = h.iloc[entry:exit_]
    next_open = current["open"].to_numpy(float) * current["fx_to_usd"].to_numpy(float)
    prev_open = previous["open"].to_numpy(float) * previous["fx_to_usd"].to_numpy(float)
    gross = (
        current["split_factor"].to_numpy(float) * next_open
        + current["distribution"].to_numpy(float) * current["fx_to_usd"].to_numpy(float)
    ) / prev_open
    if np.any(gross <= 0):
        raise ValueError("Non-positive terminal wealth requires a point-mass loss model")
    available = max(
        pd.Timestamp(h.loc[exit_, "available_at"]), pd.Timestamp(h.loc[exit_, "fx_available_at"])
    )
    return float(np.log(gross).sum()), available


def build_examples(
    frame: pd.DataFrame,
    asset_ids: Iterable[str],
    *,
    context_length: int = 128,
    horizons: tuple[int, ...] = (5, 21, 63),
    decision_times: Iterable[pd.Timestamp] | None = None,
) -> ExampleBatch:
    """Construct causal contexts and training-only future blocks.

    The caller must split decision dates before fitting a ``RobustScaler``. This
    routine intentionally returns raw primitive channels for transparent tests.
    """
    panel = frame.copy()
    for column in ("timestamp", "available_at", "fx_available_at"):
        panel[column] = pd.to_datetime(panel[column], utc=True)
    assets = tuple(str(asset) for asset in asset_ids)
    histories = {
        asset: panel[panel["permanent_id"] == asset].sort_values("timestamp").reset_index(drop=True)
        for asset in assets
    }
    if any(history.empty for history in histories.values()):
        raise ValueError("Every selected asset requires history")
    common = sorted(
        set.intersection(*(set(history["timestamp"]) for history in histories.values()))
    )
    if decision_times is not None:
        requested = set()
        for value in decision_times:
            timestamp = pd.Timestamp(value)
            timestamp = (
                timestamp.tz_localize("UTC")
                if timestamp.tzinfo is None
                else timestamp.tz_convert("UTC")
            )
            requested.add(timestamp)
        common = [value for value in common if value in requested]
    max_horizon = max(horizons)
    rows: list[tuple[pd.Timestamp, list[int]]] = []
    for decision in common:
        indices = [
            int(history.index[history["timestamp"] == decision][0])
            for history in histories.values()
        ]
        if min(indices) >= context_length - 1 and all(
            index + max_horizon + 1 < len(histories[asset])
            for asset, index in zip(assets, indices, strict=True)
        ):
            if all(
                histories[asset].loc[index, "available_at"] <= decision
                for asset, index in zip(assets, indices, strict=True)
            ):
                rows.append((decision, indices))
    if not rows:
        raise ValueError("No complete causal examples for the requested contract")
    context = np.zeros((len(rows), len(assets), context_length, 18), dtype=np.float32)
    future = {h: np.zeros((len(rows), len(assets), h, 18), dtype=np.float32) for h in horizons}
    future_mask = {h: np.ones((len(rows), len(assets), h), dtype=bool) for h in horizons}
    returns = {h: np.zeros((len(rows), len(assets)), dtype=np.float64) for h in horizons}
    scales = {h: np.zeros((len(rows), len(assets)), dtype=np.float64) for h in horizons}
    mature = {h: np.empty((len(rows), len(assets)), dtype="datetime64[ns]") for h in horizons}
    cached = {
        asset: _feature_channels(*_primitive_features(history))
        for asset, history in histories.items()
    }
    primitive = {asset: _primitive_features(history)[0] for asset, history in histories.items()}
    for batch_index, (_, indices) in enumerate(rows):
        for asset_index, (asset, origin) in enumerate(zip(assets, indices, strict=True)):
            context[batch_index, asset_index] = cached[asset][
                origin - context_length + 1 : origin + 1
            ]
            daily = primitive[asset][:, 0]
            sigma = max(float(np.std(daily[max(0, origin - 62) : origin + 1], ddof=1)), 0.005)
            for horizon in horizons:
                future[horizon][batch_index, asset_index] = cached[asset][
                    origin + 1 : origin + horizon + 1
                ]
                value, available = opening_log_return(histories[asset], origin, horizon)
                returns[horizon][batch_index, asset_index] = value
                scales[horizon][batch_index, asset_index] = max(0.005, np.sqrt(horizon) * sigma)
                mature[horizon][batch_index, asset_index] = available.tz_localize(
                    None
                ).to_datetime64()
    return ExampleBatch(
        decision_times=np.asarray([value.tz_localize(None).to_datetime64() for value, _ in rows]),
        asset_ids=assets,
        context=context,
        asset_mask=np.ones((len(rows), len(assets)), dtype=bool),
        future=future,
        future_mask=future_mask,
        log_returns=returns,
        scales=scales,
        outcome_available_at=mature,
    )


def annual_split_indices(
    dates: np.ndarray, test_year: int, fitting_origins: int = 504
) -> dict[str, np.ndarray]:
    """Exact conservative offsets from manuscript Section 6.2."""
    timeline = pd.DatetimeIndex(pd.to_datetime(dates, utc=True)).sort_values().unique()
    test = np.flatnonzero(timeline.year == test_year)
    if len(test) == 0:
        raise ValueError(f"No decision dates in test year {test_year}")
    t0 = int(test[0])
    fit_end = t0 - 319
    fit_start = fit_end - fitting_origins + 1
    stopping = np.arange(t0 - 254, t0 - 191)
    calibration = np.arange(t0 - 127, t0 - 64)
    if fit_start < 0 or min(stopping.min(), calibration.min()) < 0:
        raise ValueError("Insufficient pre-fold history for prescribed split")
    return {
        "fit": np.arange(fit_start, fit_end + 1),
        "stopping": stopping,
        "calibration": calibration,
        "test": test,
    }


def synthetic_panel(
    kind: str, *, sessions: int = 220, assets: int = 4, seed: int = 11
) -> tuple[pd.DataFrame, PanelManifest]:
    """Deterministic engineering panels with different failure-relevant regimes."""
    allowed = {"calm_trend", "mean_reversion", "correlation_break", "missing_actions"}
    if kind not in allowed:
        raise ValueError(f"Unknown synthetic engineering dataset: {kind}")
    if sessions < 160 or assets < 2:
        raise ValueError("Engineering panels require >=160 sessions and >=2 assets")
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-02", periods=sessions, tz="UTC")
    common = rng.normal(0.0002, 0.007, sessions)
    records: list[dict[str, object]] = []
    for asset in range(assets):
        noise = rng.normal(0, 0.006 + asset * 0.0005, sessions)
        if kind == "calm_trend":
            returns = 0.0003 + 0.55 * common + noise
        elif kind == "mean_reversion":
            returns = np.zeros(sessions)
            for t in range(1, sessions):
                returns[t] = -0.35 * returns[t - 1] + noise[t]
        elif kind == "correlation_break":
            returns = 0.7 * common + noise
            returns[sessions // 2 :] = -0.7 * common[sessions // 2 :] + noise[sessions // 2 :]
            returns[sessions * 3 // 4 : sessions * 3 // 4 + 5] -= 0.035
        else:
            returns = 0.4 * common + noise
        opens = 100.0 * np.exp(np.cumsum(returns))
        distribution = np.zeros(sessions)
        split = np.ones(sessions)
        if kind == "missing_actions":
            distribution[80] = 0.75
            split[120] = 2.0
            opens[120:] /= 2.0
        close = opens * np.exp(rng.normal(0, 0.003, sessions))
        high = np.maximum(opens, close) * (1 + np.abs(rng.normal(0.003, 0.001, sessions)))
        low = np.minimum(opens, close) * (1 - np.abs(rng.normal(0.003, 0.001, sessions)))
        volume = rng.lognormal(13 + asset * 0.1, 0.25, sessions)
        for t, timestamp in enumerate(dates):
            # The fixture's timestamp is its declared decision-availability time.
            # Real providers require their own independently verified publication lag.
            available = timestamp
            records.append(
                {
                    "permanent_id": f"SYN{asset:02d}",
                    "timestamp": timestamp,
                    "available_at": available,
                    "open": opens[t],
                    "high": high[t],
                    "low": low[t],
                    "close": close[t],
                    "distribution": distribution[t],
                    "split_factor": split[t],
                    "volume": volume[t],
                    "currency": "USD",
                    "fx_to_usd": 1.0,
                    "fx_available_at": available,
                    "eligible": True,
                    "category": CATEGORIES[asset % len(CATEGORIES)],
                }
            )
    frame = (
        pd.DataFrame.from_records(records)
        .sort_values(["permanent_id", "timestamp"])
        .reset_index(drop=True)
    )
    payload = frame.to_csv(index=False).encode()
    manifest = PanelManifest(
        provider=f"synthetic engineering fixture: {kind}",
        product="deterministic generated panel",
        license="internal engineering fixture only",
        retrieved_at="2026-10-11T00:00:00Z",
        timezone="UTC",
        revision_policy="immutable generated fixture",
        known_gaps=(),
        source_sha256=hashlib.sha256(payload).hexdigest(),
        synthetic=True,
    )
    return frame, manifest
