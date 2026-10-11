"""Generate a synthetic FREM panel calibrated to a retained public Yahoo panel.

The output is engineering-only. It does not convert adjusted closes into an
admitted point-in-time dataset and cannot support market-performance claims.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_SOURCE = Path(
    "/Volumes/PRO-BLADE/GitHubRepos/Old-Stuff/Everything Else/GitHub-Every-Repo/"
    "EigenFinance/data/public_yahoo_v1/prices.csv"
)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/engineering/yahoo_calibrated_v1/panel.parquet"),
    )
    parser.add_argument("--sessions", type=int, default=600)
    parser.add_argument("--assets", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261011)
    args = parser.parse_args()
    if args.output.exists() or Path(str(args.output) + ".manifest.json").exists():
        raise FileExistsError("Refusing to overwrite an existing engineering dataset")
    if args.sessions < 128 or not 2 <= args.assets <= 8:
        raise ValueError("Use at least 128 sessions and between 2 and 8 assets")

    source = pd.read_csv(args.source)
    required = {"date", "asset", "adjusted_close"}
    if required - set(source):
        raise ValueError(f"Calibration source lacks columns: {sorted(required - set(source))}")
    pivot = source.pivot(index="date", columns="asset", values="adjusted_close").sort_index()
    log_returns = np.log(pivot).diff().dropna()
    if log_returns.shape[1] < args.assets or not np.isfinite(log_returns.to_numpy()).all():
        raise ValueError("Calibration source cannot support the requested synthetic fixture")

    selected = list(log_returns.columns[: args.assets])
    values = log_returns[selected].to_numpy()
    mean = values.mean(axis=0)
    covariance = np.cov(values, rowvar=False)
    covariance += np.eye(args.assets) * 1e-10
    rng = np.random.default_rng(args.seed)
    simulated = rng.multivariate_normal(mean, covariance, size=args.sessions)
    synthetic_index = 100.0 * np.exp(np.cumsum(simulated, axis=0))
    dates = pd.bdate_range("2018-01-02", periods=args.sessions, tz="UTC") + np.timedelta64(21, "h")
    frozen_at = pd.Timestamp("2017-12-31T00:00:00Z")

    records: list[pd.DataFrame] = []
    for column in range(args.assets):
        timestamp = dates
        records.append(
            pd.DataFrame(
                {
                    "timestamp": timestamp,
                    "available_at": timestamp + np.timedelta64(15, "m"),
                    "asset": f"SYN_YC_{column:02d}",
                    "close": synthetic_index[:, column],
                    "total_return_index": synthetic_index[:, column],
                    "currency": "USD",
                    "asset_type": "etf",
                    "price_kind": "total_return_index",
                    "eligible": True,
                    "eligibility_known_at": frozen_at,
                    "universe_source": "deterministic synthetic engineering universe",
                    "executable": True,
                    "source": "synthetic Gaussian path calibrated to retained Yahoo adjusted-close moments",
                    "corporate_action_source": "synthetic total-return process; no real corporate actions",
                }
            )
        )
    panel = pd.concat(records, ignore_index=True).sort_values(["asset", "timestamp"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(args.output, index=False)
    manifest = {
        "name": "yahoo_calibrated_synthetic_engineering_v1",
        "dataset_synthetic": True,
        "scientific_claims_allowed": False,
        "base_currency": "USD",
        "universe_status": "fixed_ex_ante",
        "universe_frozen_at": frozen_at.isoformat(),
        "delistings_included": False,
        "availability_policy": "synthetic 21:00 UTC event plus deterministic 15-minute delay",
        "total_return_methodology": (
            "Positive synthetic total-return indices from a Gaussian process calibrated to the full retained "
            "Yahoo adjusted-close sample; engineering only, not a point-in-time market reconstruction"
        ),
        "feature_provenance": {},
        "source_license": "Calibration source records Yahoo terms; generated fixture is local engineering data",
        "calibration_source": str(args.source),
        "calibration_source_sha256": file_sha256(args.source),
        "calibration_assets": selected,
        "calibration_uses_full_source_sample": True,
        "seed": args.seed,
        "sessions": args.sessions,
        "assets": args.assets,
        "expected_file_sha256": file_sha256(args.output),
        "prohibition": "Do not report this fixture as market, protected, predictive, investment, or competition evidence.",
    }
    Path(str(args.output) + ".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"panel": str(args.output), "manifest": str(args.output) + ".manifest.json"}))


if __name__ == "__main__":
    main()
