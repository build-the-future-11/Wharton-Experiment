"""Track A ETF prices — yfinance when available, else synthetic proxies."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

DEFAULT_ETF_TICKERS: tuple[str, ...] = ("SPY", "AGG", "GLD", "VNQ", "EFA")

MANIFEST_PATH = Path("data/manifests/DATA_MANIFEST.yaml")


@dataclass
class ETFTrackResult:
    tickers: list[str]
    returns: np.ndarray
    source: str
    synthetic_proxy: bool
    file_path: Optional[Path] = None


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _synthetic_etf_returns(
    tickers: Sequence[str],
    n_days: int = 504,
    seed: int = 42,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    k = len(tickers)
    chol = np.linalg.cholesky(np.eye(k) * 0.012**2 + np.ones((k, k)) * 0.003**2)
    z = rng.normal(size=(n_days, k))
    return z @ chol.T


def _rectangular_returns(close) -> tuple[np.ndarray, list[str]] | None:
    """Force a dense (T, K) float panel; drop all-NaN columns; require T>=30."""
    try:
        import pandas as pd  # type: ignore
    except Exception:
        arr = np.asarray(close, dtype=float)
        if arr.ndim == 1:
            arr = arr[:, None]
        if arr.size == 0 or arr.shape[0] < 30:
            return None
        return np.nan_to_num(arr, nan=0.0), [f"A{i}" for i in range(arr.shape[1])]

    if not isinstance(close, pd.DataFrame):
        close = pd.DataFrame(close)
    # Drop columns that never traded
    close = close.dropna(axis=1, how="all")
    if close.shape[1] == 0:
        return None
    rets = close.pct_change(fill_method=None)
    rets = rets.replace([np.inf, -np.inf], np.nan)
    # Require contemporaneous coverage across remaining assets
    rets = rets.dropna(axis=0, how="any")
    if len(rets) < 30:
        return None
    tickers = [str(c) for c in rets.columns]
    arr = np.asarray(rets.values, dtype=float)
    if not np.all(np.isfinite(arr)):
        return None
    return arr, tickers


def _cache_columns(cache_file: Path, n_cols: int) -> list[str]:
    """Column order of a cached panel: sidecar, else DATA_MANIFEST entry matching sha256."""
    sidecar = cache_file.with_suffix(".columns.json")
    if sidecar.exists():
        cols = json.loads(sidecar.read_text())
        if len(cols) == n_cols:
            return [str(c) for c in cols]
    if MANIFEST_PATH.exists():
        try:
            import yaml  # type: ignore

            entry = (yaml.safe_load(MANIFEST_PATH.read_text()) or {}).get("etf_track") or {}
            if entry.get("sha256") == _file_sha256(cache_file) and len(entry.get("columns", [])) == n_cols:
                return [str(c) for c in entry["columns"]]
        except Exception:
            pass
    return [f"UNKNOWN_COL{i}" for i in range(n_cols)]


def load_etf_track(
    tickers: Sequence[str] | None = None,
    *,
    start: str = "2020-01-01",
    end: str | None = None,
    cache_dir: Path | str = Path("data/raw/etf"),
    seed: int = 42,
) -> ETFTrackResult:
    tickers = list(tickers or DEFAULT_ETF_TICKERS)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{'_'.join(tickers)}_returns.npy"

    # Prefer a known-good rectangular cache over flaky live Yahoo pulls.
    if cache_file.exists():
        try:
            cached = np.asarray(np.load(cache_file), dtype=float)
            if cached.ndim == 2 and cached.shape[0] >= 30 and cached.shape[1] >= 1 and np.all(
                np.isfinite(cached)
            ):
                cols = _cache_columns(cache_file, cached.shape[1])
                return ETFTrackResult(
                    tickers=cols,
                    returns=cached,
                    source="cache",
                    synthetic_proxy=any("SYNTHETIC" in c for c in cols),
                    file_path=cache_file,
                )
        except Exception:
            pass

    try:
        import yfinance as yf  # type: ignore

        data = yf.download(
            list(tickers),
            start=start,
            end=end,
            progress=False,
            auto_adjust=True,
            threads=False,
        )
        if data is not None and getattr(data, "empty", True) is False:
            close = data["Close"] if "Close" in getattr(data, "columns", []) else data
            # yfinance returns multi-ticker columns alphabetically, not in request order.
            if hasattr(close, "reindex"):
                close = close.reindex(columns=[t for t in tickers if t in close.columns])
            parsed = _rectangular_returns(close)
            if parsed is not None:
                rets, used = parsed
                # Pad/truncate to requested ticker count with synthetic extras if needed
                if rets.shape[1] < len(tickers):
                    missing = [t for t in tickers if t not in used]
                    pad = _synthetic_etf_returns(missing, n_days=rets.shape[0], seed=seed)
                    # correlate pad lightly with first column
                    pad = 0.3 * rets[:, :1] + 0.7 * pad
                    rets = np.concatenate([rets, pad], axis=1)
                    used = list(used) + [f"{t}_SYNTHETIC_PAD" for t in missing]
                    source = "yfinance+synthetic_pad"
                    synthetic = True
                else:
                    source = "yfinance"
                    synthetic = False
                np.save(cache_file, rets)
                cache_file.with_suffix(".columns.json").write_text(json.dumps(list(used)))
                return ETFTrackResult(
                    tickers=list(used),
                    returns=np.asarray(rets, dtype=float),
                    source=source,
                    synthetic_proxy=synthetic,
                    file_path=cache_file,
                )
    except Exception:
        pass

    rets = _synthetic_etf_returns(tickers, seed=seed)
    np.save(cache_file, rets)
    proxy_cols = [f"{t}_SYNTHETIC_PROXY" for t in tickers]
    cache_file.with_suffix(".columns.json").write_text(json.dumps(proxy_cols))
    return ETFTrackResult(
        tickers=proxy_cols,
        returns=np.asarray(rets, dtype=float),
        source="SYNTHETIC_PROXY",
        synthetic_proxy=True,
        file_path=cache_file,
    )


def write_data_manifest(
    track: ETFTrackResult,
    manifest_path: Path | str | None = None,
    extra: dict | None = None,
) -> Path:
    manifest_path = Path(manifest_path or MANIFEST_PATH)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    entries = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "etf_track": {
            "columns": track.tickers,
            "source": track.source,
            "synthetic_proxy": track.synthetic_proxy,
            "shape": list(track.returns.shape),
        },
    }
    if track.file_path and track.file_path.exists():
        entries["etf_track"]["sha256"] = _file_sha256(track.file_path)
        entries["etf_track"]["path"] = str(track.file_path)
    if extra:
        entries.update(extra)

    try:
        import yaml  # type: ignore

        text = yaml.safe_dump(entries, sort_keys=False)
    except Exception:
        text = json.dumps(entries, indent=2)
    manifest_path.write_text(text)
    return manifest_path
