"""Bounded official public-source acquisition, not an investable stock panel.

No third-party scientific imports. Dates are filtered before numeric outcomes
are parsed. Raw archives are retained opaquely and must not be redistributed
without checking the provider's terms. Existing output directories are refused.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import re
import urllib.error
import urllib.request
import zipfile
from datetime import UTC, datetime
from pathlib import Path


def parse_monthly(text: str, before: str) -> list[dict]:
    rows = []
    seen = set()
    for fields in csv.reader(io.StringIO(text)):
        if not fields or not re.fullmatch(r"\d{6}", fields[0].strip()):
            continue
        date = fields[0].strip()
        if date >= before:
            continue  # Do not convert or inspect protected numeric outcomes.
        if len(fields) != 5 or date in seen or not 1 <= int(date[4:]) <= 12:
            raise ValueError("Duplicate month or unexpected monthly schema")
        values = [float(x.strip()) for x in fields[1:]]
        if any(not math.isfinite(v) or v in (-99.99, -999) for v in values):
            raise ValueError("Missing or nonfinite factor observation")
        seen.add(date)
        mkt, smb, hml, rf = (v / 100 for v in values)
        rows.append({"month": date, "market_excess": mkt, "smb": smb,
                     "hml": hml, "rf": rf, "market_total": mkt + rf})
    if not rows or [r["month"] for r in rows] != sorted(seen):
        raise ValueError("Empty or unordered monthly data")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config_bytes = args.config.read_bytes()
    config = json.loads(config_bytes)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "request.json").write_bytes(config_bytes)
    raw = args.output / "raw"
    raw.mkdir()
    manifest = {"status": "ACQUIRING", "created_at": datetime.now(UTC).isoformat(),
                "request_sha256": hashlib.sha256(config_bytes).hexdigest(),
                "scope": config["scope"], "limitations": config["limitations"],
                "normalized_before": config["before"], "sources": [], "model_cells": 0}
    all_rows = []
    for source in config["sources"]:
        record = {"name": source["name"], "url": source["url"], "status": "STARTED"}
        try:
            if not source["url"].startswith("https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"):
                raise ValueError("Only declared official data-library endpoints are supported")
            request = urllib.request.Request(source["url"], headers={"User-Agent": "FREM-Research/0.2"})
            with urllib.request.urlopen(request, timeout=30) as response:
                payload = response.read(2_000_001)
                record["http_status"] = response.status
                record["last_modified"] = response.headers.get("Last-Modified")
            if len(payload) > 2_000_000:
                raise ValueError("Download exceeds declared two-megabyte per-source ceiling")
            filename = source["name"] + ".zip"
            (raw / filename).write_bytes(payload)
            record.update(raw_file="raw/" + filename, sha256=hashlib.sha256(payload).hexdigest(),
                          retrieved_at=datetime.now(UTC).isoformat(), bytes=len(payload))
            with zipfile.ZipFile(io.BytesIO(payload)) as archive:
                members = [m for m in archive.infolist() if m.filename.lower().endswith(".csv")]
                if len(members) != 1 or members[0].file_size > 5_000_000:
                    raise ValueError("Unexpected archive members or expanded size")
                text = archive.read(members[0]).decode("utf-8-sig")
            rows = parse_monthly(text, config["before"])
            all_rows.extend({"region": source["name"], **row} for row in rows)
            record.update(status="ACQUIRED", rows=len(rows), first_month=rows[0]["month"],
                          last_month=rows[-1]["month"], contains_2008=any(r["month"] == "200801" for r in rows))
        except (OSError, ValueError, UnicodeError, zipfile.BadZipFile, urllib.error.URLError) as exc:
            record.update(status="FAILED", error=f"{type(exc).__name__}: {exc}")
        manifest["sources"].append(record)
        (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(json.dumps({k: v for k, v in record.items() if k not in ("sha256", "url")}), flush=True)
    table = args.output / "monthly.csv"
    with table.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["region", "month", "market_excess", "smb", "hml", "rf",
                                                   "market_total"])
        writer.writeheader()
        writer.writerows(all_rows)
    manifest.update(status="COMPLETE" if all(s["status"] == "ACQUIRED" for s in manifest["sources"])
                    else "INCOMPLETE_RETAINED_FAILURES", normalized_rows=len(all_rows),
                    normalized_sha256=hashlib.sha256(table.read_bytes()).hexdigest())
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return 0 if manifest["status"] == "COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
