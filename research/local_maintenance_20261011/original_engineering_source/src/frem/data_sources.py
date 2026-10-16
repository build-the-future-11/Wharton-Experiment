"""Explicit opt-in data acquisition. Importing this module never downloads data."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

FRED_OBSERVATIONS_DOC = "https://fred.stlouisfed.org/docs/api/fred/series_observations.html"
ALFRED_VINTAGE_DOC = "https://alfred.stlouisfed.org/help/downloaddata"


@dataclass(frozen=True)
class FredVintageConfig:
    series_id: str
    observation_start: str
    observation_end: str
    realtime_start: str = "1776-07-04"
    realtime_end: str = "9999-12-31"
    api_key_env: str = "FRED_API_KEY"
    availability_delay_days: int = 2
    timeout_seconds: float = 30.0
    page_size: int = 100000


def fetch_fred_vintages(config: FredVintageConfig, output_path: str | Path, overwrite: bool = False) -> dict:
    """Download all returned real-time periods and retain vintage provenance.

    ALFRED dates are date-granular, not intraday release timestamps. By default
    availability is delayed two calendar days to 00:00 UTC; this conservative
    policy must be replaced by verified release timestamps for intraday work.
    API keys are read from the environment and never written to the manifest.
    """
    output = Path(output_path)
    manifest_path = Path(str(output) + ".manifest.json")
    if (output.exists() or manifest_path.exists()) and not overwrite:
        raise FileExistsError("Output already exists; use an explicit new path or overwrite=True")
    if output.suffix.lower() not in {".csv", ".parquet", ".pq"}:
        raise ValueError("Output must be CSV or parquet")
    if config.availability_delay_days < 2:
        raise ValueError("Date-only vintages require >=2 days conservatism; exact-release imports use CSV")
    if not 1 <= config.page_size <= 100000 or config.timeout_seconds <= 0:
        raise ValueError("Invalid API pagination or timeout settings")
    if not config.series_id.strip() or not config.series_id.replace("_", "").isalnum():
        raise ValueError("Invalid FRED series identifier")
    key = os.environ.get(config.api_key_env)
    if not key:
        raise ValueError(f"Set environment variable {config.api_key_env} to your registered FRED API key")
    for name in ("observation_start", "observation_end", "realtime_start", "realtime_end"):
        datetime.strptime(getattr(config, name), "%Y-%m-%d")
    params = {
        "series_id": config.series_id,
        "api_key": key,
        "file_type": "json",
        "observation_start": config.observation_start,
        "observation_end": config.observation_end,
        "realtime_start": config.realtime_start,
        "realtime_end": config.realtime_end,
        "output_type": 1,
        "units": "lin",
        "sort_order": "asc",
        "limit": config.page_size,
    }
    observations: list[dict] = []
    response_hashes: list[str] = []
    offset = 0
    while True:
        params["offset"] = offset
        request = Request(
            "https://api.stlouisfed.org/fred/series/observations?" + urlencode(params),
            headers={"User-Agent": "FREM-prospective-research/0.1"},
        )
        try:
            with urlopen(request, timeout=config.timeout_seconds) as response:
                payload = response.read()
        except HTTPError as exc:
            # Do not include the URL: it contains the API key query parameter.
            raise RuntimeError(f"FRED request failed with HTTP {exc.code}") from None
        except URLError:
            raise RuntimeError("FRED request failed due to network access or timeout") from None
        response_hashes.append(sha256(payload).hexdigest())
        body = json.loads(payload)
        if "error_code" in body or "observations" not in body:
            raise RuntimeError("FRED returned an error or unexpected schema")
        page = body["observations"]
        observations.extend(page)
        offset += len(page)
        total = int(body.get("count", offset))
        if offset >= total:
            break
        if not page:
            raise RuntimeError("FRED pagination ended before the declared observation count")
    if not observations:
        raise ValueError("The selected series/date range returned no observations")
    rows = pd.DataFrame(observations)
    required = {"date", "value", "realtime_start", "realtime_end"}
    if required - set(rows):
        raise ValueError("FRED response lacks real-time interval fields")
    # Pandas Timestamp cannot represent 9999-12-31; retain real-time end as text.
    table = pd.DataFrame(
        {
            "series": config.series_id,
            "observation_time": pd.to_datetime(rows["date"], utc=True),
            "available_at": pd.to_datetime(rows["realtime_start"], utc=True)
            + pd.Timedelta(days=config.availability_delay_days),
            "value": pd.to_numeric(rows["value"].replace(".", float("nan")), errors="raise"),
            "vintage_date": rows["realtime_start"],
            "vintage_end_date": rows["realtime_end"],
            "source": "Federal Reserve Bank of St. Louis / ALFRED",
        }
    )
    if table.duplicated(["series", "observation_time", "available_at"]).any():
        raise ValueError("FRED returned duplicate vintage keys; preserve response hashes and investigate")
    table = table.sort_values(["series", "available_at", "observation_time"])
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() == ".csv":
        table.to_csv(output, index=False)
    else:
        table.to_parquet(output, index=False)
    manifest = {
        "provider": "ALFRED via FRED API version 1",
        "request": asdict(config),
        "series_universe": [config.series_id],
        "downloaded_at": datetime.now(UTC).isoformat(),
        "observations": len(table),
        "pages": len(response_hashes),
        "response_sha256": response_hashes,
        "expected_file_sha256": sha256(output.read_bytes()).hexdigest(),
        "availability_policy": f"vintage date plus {config.availability_delay_days} calendar days at 00:00 UTC",
        "intraday_release_times_verified": False,
        "citations": [FRED_OBSERVATIONS_DOC, ALFRED_VINTAGE_DOC],
        "use": "macro covariates only; yields are not investable bond total returns",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def load_macro(path: str | Path, column_map: dict[str, str] | None = None) -> pd.DataFrame:
    """Load a previously acquired vintage table; macro values are never panel prices."""
    from .data import DataValidationError, _utc, data_fingerprint

    source = Path(path)
    if source.suffix.lower() == ".csv":
        table = pd.read_csv(source)
    elif source.suffix.lower() in {".parquet", ".pq"}:
        table = pd.read_parquet(source)
    else:
        raise DataValidationError("Macro input must be CSV or parquet")
    if column_map:
        table = table.rename(columns=column_map)
    needed = {"series", "observation_time", "available_at", "value"}
    if needed - set(table):
        raise DataValidationError(f"Macro input missing {sorted(needed - set(table))}")
    table["observation_time"] = _utc(table["observation_time"], "observation_time")
    table["available_at"] = _utc(table["available_at"], "available_at")
    table["value"] = pd.to_numeric(table["value"].replace(".", float("nan")), errors="raise")
    manifest_path = Path(str(source) + ".manifest.json")
    if not manifest_path.exists():
        raise DataValidationError("Macro vintage input needs a provenance manifest describing availability")
    table.attrs.update(json.loads(manifest_path.read_text(encoding="utf-8")))
    if not table.attrs.get("availability_policy"):
        raise DataValidationError("Macro manifest lacks availability_policy")
    actual_hash = sha256(source.read_bytes()).hexdigest()
    if table.attrs.get("expected_file_sha256") not in {None, actual_hash}:
        raise DataValidationError("Macro file hash disagrees with its manifest")
    table.attrs["file_sha256"] = actual_hash
    table.attrs.setdefault("series_universe", sorted(table["series"].astype(str).unique().tolist()))
    table.attrs["fingerprint"] = data_fingerprint(table)
    return table
