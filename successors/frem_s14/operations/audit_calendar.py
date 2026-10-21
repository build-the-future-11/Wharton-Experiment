"""Read-only, pre-GPU calendar audit; never repairs, intersects or fills prices."""

import argparse
import csv
import hashlib
import json
from datetime import date, datetime, timezone
from pathlib import Path

ASSETS = ("spy", "qqq", "iwm", "dia", "tlt", "ief", "gld", "slv")


def audit(input_dir, expected_hashes=None, start="2007-01-01", stop="2018-01-01"):
    if not "2007-01-01" <= start < stop <= "2025-01-01":
        raise ValueError("invalid or protected interval")
    calendars, files = {}, {}
    for asset in ASSETS:
        path = Path(input_dir) / (asset + ".us.txt")
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"regular source file required: {asset}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if expected_hashes is not None and digest != expected_hashes[asset]:
            raise ValueError(f"executed-source hash mismatch: {asset}")
        days, previous = [], None
        with path.open(newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != ["Date", "Open", "High", "Low", "Close", "Volume", "OpenInt"]:
                raise ValueError(f"unexpected source schema: {asset}")
            for row in reader:
                day = row["Date"]
                if date.fromisoformat(day).isoformat() != day:
                    raise ValueError(f"noncanonical date: {asset}")
                if day >= stop:
                    continue
                if previous is not None and day <= previous:
                    raise ValueError(f"duplicate or unordered date: {asset}")
                previous = day
                if start <= day:
                    days.append(day)
        calendars[asset] = set(days)
        files[asset] = {"sha256": digest, "bytes": path.stat().st_size, "rows": len(days)}
    union = set.union(*calendars.values())
    common = set.intersection(*calendars.values())
    for asset, days in calendars.items():
        files[asset].update(
            missing_from_union=sorted(union - days), outside_common=sorted(days - common)
        )
    compatible = bool(union) and union == common and len(common) >= 200
    return {
        "schema": 1,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "CALENDARS_COMPATIBLE" if compatible else "BLOCKED_SOURCE_CALENDARS",
        "interval": [start, stop],
        "files": files,
        "union_dates": len(union),
        "common_dates_diagnostic_only": len(common),
        "input_hashes_match_executed_run": expected_hashes is not None,
        "data_modified": False,
        "scientific_execution": False,
        "scope": "Date-only prerequisite. Does not admit prices, data rights, point-in-time validity, or scientific execution.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--executed-intake-receipt")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    expected = None
    if args.executed_intake_receipt:
        saved = json.loads(Path(args.executed_intake_receipt).read_text())
        expected = {asset: saved["files"][asset]["sha256"] for asset in ASSETS}
    result = audit(args.input_dir, expected)
    with Path(args.output).open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "output": args.output}))
    return 0 if result["status"] == "CALENDARS_COMPATIBLE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
