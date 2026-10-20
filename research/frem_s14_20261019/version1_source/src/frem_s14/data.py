"""Timestamp-audited balanced panels; training-only scaling and purged splits."""

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .runtime import digest, read_json, timestamp


@dataclass
class Panel:
    times: np.ndarray
    available: np.ndarray
    returns: np.ndarray
    features: np.ndarray
    assets: tuple
    provenance: dict

    def validate(self):
        t, a, _ = self.features.shape
        if self.times.shape != (t,) or self.available.shape != (t, a):
            raise ValueError("panel timestamp dimensions disagree")
        if self.returns.shape != (t, a) or len(self.assets) != a or len(set(self.assets)) != a:
            raise ValueError("panel assets/returns disagree")
        if not np.issubdtype(self.times.dtype, np.integer) or not np.issubdtype(
            self.available.dtype, np.integer
        ):
            raise ValueError("timestamps must be integer UTC microseconds")
        if not np.all(np.diff(self.times) > 0) or np.any(self.available < self.times[:, None]):
            raise ValueError("times must increase and availability cannot precede observation")
        if not np.isfinite(self.features).all() or not np.isfinite(self.returns).all():
            raise ValueError("nonfinite features or returns are forbidden; no implicit imputation")
        if np.any(self.returns < -1):
            raise ValueError("simple total returns cannot be below -100 percent")
        if not np.array_equal(self.features[..., 0], self.returns):
            raise ValueError("first feature must be the backward-looking simple total return")
        return self


def load_csv(path, protected_start):
    """CSV rows on/after the boundary are discarded before numerical parsing.

    Source manifest is an explicit supplier assertion, not independent rights verification.
    return is the backward period's total return, including terminal distributions.
    """
    path = Path(path)
    meta = read_json(Path(str(path) + ".manifest.json"))
    required = {
        "schema",
        "sha256",
        "source",
        "retrieved_at",
        "license",
        "universe_policy",
        "return_definition",
        "feature_availability",
        "purpose",
    }
    if (
        set(meta) != required
        or type(meta["schema"]) is not int
        or meta["schema"] != 1
        or meta["sha256"] != digest(path)
    ):
        raise ValueError("exact provenance manifest / input checksum required")
    if any(not isinstance(meta[k], str) or not meta[k].strip() for k in required - {"schema"}):
        raise ValueError("provenance strings must be nonempty")
    timestamp(meta["retrieved_at"])
    proxy = meta["purpose"] == "historical_proxy"
    expected_return = (
        "source_adjusted_close_ratio_not_verified_total_return"
        if proxy
        else "simple_total_return_including_terminal_outcomes"
    )
    if meta["return_definition"] != expected_return:
        raise ValueError(
            "yield, price changes and unexplained adjusted prices are not total returns"
        )
    expected_availability = (
        "historical_order_only_unverified_point_in_time"
        if proxy
        else "available_at_bounds_all_features"
    )
    if meta["feature_availability"] != expected_availability:
        raise ValueError("feature availability declaration required")
    if meta["purpose"] not in ("development", "historical_proxy"):
        raise ValueError("S14 runner accepts development data only")
    boundary = timestamp(protected_start)
    rows = {}
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames or []
        if len(fields) != len(set(fields)):
            raise ValueError("duplicate CSV header")
        feature_names = sorted(x for x in fields if x.startswith("feature_"))
        if set(fields) != {"timestamp", "asset", "available_at", "return", *feature_names}:
            raise ValueError("unexpected CSV schema")
        for row in reader:
            instant = timestamp(row["timestamp"])
            if instant >= boundary:
                continue
            known = timestamp(row["available_at"])
            if known >= boundary:
                continue
            key = (instant, row["asset"])
            if not key[1].strip() or key in rows:
                raise ValueError("duplicate date/asset or empty asset")
            rows[key] = (known, [float(row["return"]), *[float(row[x]) for x in feature_names]])
    dates = sorted({k[0] for k in rows})
    assets = tuple(sorted({k[1] for k in rows}))
    if not dates or len(rows) != len(dates) * len(assets):
        raise ValueError(
            "balanced observed panel required; supply audited universe, never forward fill"
        )
    x = np.array([[rows[d, a][1] for a in assets] for d in dates], dtype=np.float64)
    available = np.array([[rows[d, a][0] for a in assets] for d in dates], dtype=np.int64)
    return Panel(
        np.array(dates, dtype=np.int64), available, x[..., 0].copy(), x, assets, meta
    ).validate()


def generated_panel(length=180, assets=4, seed=391):
    """Causal regime-switching generated fixture. Never a market benchmark."""
    rng = np.random.default_rng(seed)
    market = np.zeros(length)
    for t in range(1, length):
        market[t] = (0.35 if (t // 40) % 2 else -0.15) * market[t - 1] + rng.normal(0, 0.006)
    r = np.clip(market[:, None] + rng.normal(0, 0.008, (length, assets)), -0.8, 0.8)
    momentum = np.zeros_like(r)
    vol = np.zeros_like(r)
    for t in range(length):
        momentum[t] = r[max(0, t - 9) : t + 1].mean(axis=0)
        vol[t] = r[max(0, t - 9) : t + 1].std(axis=0)
    features = np.stack([r, momentum, vol, np.broadcast_to(market[:, None], r.shape)], -1)
    times = timestamp("2000-01-01T00:00:00Z") + np.arange(length, dtype=np.int64) * 86400000000
    return Panel(
        times,
        np.broadcast_to(times[:, None], r.shape).copy(),
        r,
        features,
        tuple(f"generated_{i}" for i in range(assets)),
        {"purpose": "generated_engineering", "generator_seed": seed},
    ).validate()


class Samples:
    """Lazy windows; no materialized N x context x channel dataset on GPU."""

    def __init__(self, panel, config):
        panel.validate()
        self.panel, self.config = panel, config
        c, h = config.model.context, config.model.horizon
        if panel.features.shape[-1] != config.model.input_dim:
            raise ValueError("input_dim must equal return plus supplied feature columns")
        t, a = panel.returns.shape
        origins = np.arange(c - 1, t - h)
        if len(origins) < 20:
            raise ValueError("insufficient chronological history")
        # The forecast is issued after all contemporaneous features are available.
        decisions = panel.available.max(axis=1)
        if not np.all(np.diff(decisions) > 0):
            raise ValueError("out-of-order availability requires an explicit as-of adapter")
        self.origin = np.repeat(origins, a)
        self.asset = np.tile(np.arange(a), len(origins))
        self.time = np.repeat(decisions[origins], a)
        self.end = np.repeat(decisions[origins + h], a)
        self.y = np.array(
            [
                np.prod(1 + panel.returns[o + 1 : o + h + 1, k]) - 1
                for o, k in zip(self.origin, self.asset, strict=True)
            ]
        )
        if not np.isfinite(self.y).all():
            raise ValueError("compound return overflow")
        first = int(len(origins) * config.train_fraction)
        second = int(len(origins) * (config.train_fraction + config.validation_fraction))
        if first + config.embargo >= second or second + config.embargo >= len(origins):
            raise ValueError("embargo exhausts chronological splits")
        validation_start = decisions[origins[first + config.embargo]]
        evaluation_start = decisions[origins[second + config.embargo]]
        self.train = np.flatnonzero((self.origin < origins[first]) & (self.end < validation_start))
        self.validation = np.flatnonzero(
            (self.time >= validation_start)
            & (self.origin < origins[second])
            & (self.end < evaluation_start)
        )
        self.test = np.flatnonzero(self.time >= evaluation_start)
        if min(map(len, (self.train, self.validation, self.test))) < 2:
            raise ValueError("empty or singleton purged split")
        # Only feature rows used as training contexts fit the scaler, never the future block.
        last = int(self.origin[self.train].max())
        prefix = panel.features[: last + 1].reshape(-1, panel.features.shape[-1])
        self.center = np.median(prefix, axis=0)
        mad = np.median(np.abs(prefix - self.center), axis=0) * 1.4826
        self.scale = np.where(mad > 1e-8, mad, 1.0)
        self.target_scale = max(float(np.std(self.y[self.train])), 1e-4)
        self.x = np.clip((panel.features - self.center) / self.scale, -20, 20).astype(np.float32)

    def batch(self, ix, future=False):
        c = self.config.model.context
        contexts = np.stack(
            [
                self.x[o - c + 1 : o + 1, a]
                for o, a in zip(self.origin[ix], self.asset[ix], strict=True)
            ]
        )
        target = None
        if future:
            # Target encoder sees a full shifted trailing window, only during training.
            target = np.stack(
                [
                    np.stack(
                        [
                            self.x[o + offset - c + 1 : o + offset + 1, a]
                            for offset in self.config.model.target_offsets
                        ]
                    )
                    for o, a in zip(self.origin[ix], self.asset[ix], strict=True)
                ]
            )
        return contexts, target

    def batches(self, ix):
        # No date may straddle batches: simultaneous assets never consume one another's labels.
        dates = np.unique(self.time[ix])
        per_batch = max(1, self.config.batch_size // len(self.panel.assets))
        for start in range(0, len(dates), per_batch):
            selected = dates[start : start + per_batch]
            yield ix[(self.time[ix] >= selected[0]) & (self.time[ix] <= selected[-1])]

    def tabular(self, ix):
        arrays = []
        for batch in self.batches(ix):
            x, _ = self.batch(batch)
            arrays.append(np.concatenate([x[:, -1], x.mean(1), x.std(1), x[:, -1] - x[:, 0]], 1))
        return np.concatenate(arrays)

    def split_receipt(self):
        return {
            name: {
                "rows": len(ix),
                "first_decision": int(self.time[ix].min()),
                "last_decision": int(self.time[ix].max()),
                "last_label": int(self.end[ix].max()),
            }
            for name, ix in (
                ("train", self.train),
                ("validation", self.validation),
                ("test", self.test),
            )
        }
