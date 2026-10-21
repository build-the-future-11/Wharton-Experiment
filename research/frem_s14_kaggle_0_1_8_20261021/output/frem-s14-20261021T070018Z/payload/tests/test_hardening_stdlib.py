"""Adversarial source, input and campaign contracts without numerical imports."""

import csv
import json
import math
import os
import random
import tempfile
import unittest
from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch

from frem_s14.cli import campaign, qualification_config
from frem_s14.config import RunConfig
from frem_s14.intake import ASSETS, asof_values, prepare, read_prices
from frem_s14.runtime import (
    admit,
    cell_inventory,
    digest,
    manifest,
    read_json,
    resume_seconds,
    timestamp,
    validate_forecast_clock,
    verify,
    verify_completion,
    write_json,
)
from frem_s14.suite import run_suite, suite_config


class HardeningTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_nested_manifest_is_bound(self):
        child = self.root / "fold"
        write_json(child / "value.json", {"x": 1})
        write_json(child / "manifest.json", manifest(child))
        write_json(self.root / "manifest.json", manifest(self.root))
        self.assertIn("fold/manifest.json", manifest(self.root)["files"])
        (child / "manifest.json").write_text("{}")
        with self.assertRaises(ValueError):
            verify(self.root)

    def test_special_file_and_root_symlink_rejected(self):
        os.mkfifo(self.root / "fifo")
        with self.assertRaisesRegex(ValueError, "special"):
            manifest(self.root)
        (self.root / "fifo").unlink()
        link = self.root / "link"
        link.symlink_to(self.root, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "real directory"):
            manifest(link)

    def test_resource_samples_must_all_be_valid_and_stable(self):
        for values, pages in [
            ([600, math.nan, 600], [1, 1, 1]),
            ([600, 600, 600], [1, 2, 1]),
            ([600, math.inf, 600], [1, 1, 1]),
            ([600, 600, 600], [-1, -1, -1]),
        ]:
            with (
                self.subTest(values=values, pages=pages),
                patch(
                    "frem_s14.runtime.memory_snapshot",
                    side_effect=[
                        {"available_mib": a, "pageouts": p}
                        for a, p in zip(values, pages, strict=True)
                    ],
                ),
                patch("frem_s14.runtime.time.sleep"),
            ):
                self.assertFalse(admit()["passed"])

    def test_timestamp_microseconds_are_exact(self):
        for year in (1901, 2024, 2500):
            base = f"{year}-01-01T00:00:00"
            self.assertEqual(timestamp(base + ".000002Z") - timestamp(base + ".000001Z"), 1)
        self.assertEqual(timestamp("1969-12-31T23:59:59.999999Z"), -1)

    def test_protected_boundary_cannot_be_moved(self):
        with self.assertRaisesRegex(ValueError, "protected boundary"):
            replace(RunConfig(), protected_start="2026-01-01T00:00:00Z")
        with self.assertRaisesRegex(ValueError, "protected price"):
            read_prices(self.root / "not_opened", boundary="2026-01-01")

    def test_delayed_clock_cannot_include_already_observed_target(self):
        validate_forecast_clock([10, 20, 30, 40], [10, 21, 32, 40], [0, 1, 2])
        for decisions in ([10, 30, 32, 40], [10, 31, 32, 40], [10, 19, 32, 40]):
            with self.assertRaises(ValueError):
                validate_forecast_clock([10, 20, 30, 40], decisions, [0, 1, 2])

    def test_complete_family_and_failure_denominator(self):
        cfg = suite_config("cpu")
        complete = [{"arm": cfg.arms[0], "seed": cfg.seeds[0]}]
        rows = cell_inventory(cfg, complete, (cfg.arms[0], cfg.seeds[1]))
        self.assertEqual(len(rows), 57)
        self.assertEqual([r["status"] for r in rows[:3]], ["COMPLETE", "FAILED", "NOT_STARTED"])
        with self.assertRaises(ValueError):
            cell_inventory(cfg, complete + complete)

    def test_completion_rejects_missing_or_duplicate_cells(self):
        cfg = qualification_config("cpu")
        completed = [{"arm": a, "seed": s} for a in cfg.arms for s in cfg.seeds]
        status = {
            "status": "COMPLETE",
            "protected_access": False,
            "expected_cells": len(completed),
            "complete_cells": completed,
        }
        write_json(self.root / "status.json", status)
        write_json(self.root / "cells.json", cell_inventory(cfg, completed))
        self.assertEqual(verify_completion(self.root, cfg), status)
        status["complete_cells"][-1] = status["complete_cells"][0]
        (self.root / "status.json").write_text(json.dumps(status))
        with self.assertRaises(ValueError):
            verify_completion(self.root, cfg)

    def test_blocked_run_keeps_every_not_started_cell(self):
        root = self.root / "blocked"
        with patch("frem_s14.cli.admit", return_value={"passed": False}):
            with self.assertRaises(RuntimeError):
                campaign(root, qualification_config("cpu"))
        rows = read_json(root / "cells.json")
        self.assertEqual(len(rows), 19)
        self.assertEqual({r["status"] for r in rows}, {"NOT_STARTED"})
        self.assertTrue(verify(root)["verified"])

    def test_invalid_suite_input_retains_171_cell_denominator(self):
        root = self.root / "suite"
        with self.assertRaises(FileNotFoundError):
            run_suite(self.root / "missing.csv", self.root / "qualification", root, "cpu")
        rows = read_json(root / "cells.json")
        self.assertEqual(len(rows), 171)
        self.assertEqual({r["status"] for r in rows}, {"NOT_STARTED"})
        self.assertTrue((root / "failure.json").is_file())
        self.assertTrue(verify(root)["verified"])

    def test_resume_budget_is_hash_bound_and_finite(self):
        for i, seconds in enumerate((2.0, -1, 10, True, "NaN")):
            root = self.root / str(i)
            write_json(root / "failure.json", {"wall_seconds": seconds})
            write_json(root / "manifest.json", manifest(root))
            checkpoint = root / "full-s11/epoch_000.pt"
            if seconds == 2.0:
                self.assertEqual(resume_seconds(checkpoint, 10), 2)
                (root / "failure.json").write_text('{"wall_seconds": 0}')
            with self.assertRaises(ValueError):
                resume_seconds(checkpoint, 10)

    def test_resume_rejects_nonfinite_even_with_matching_inventory(self):
        for i, value in enumerate(("NaN", "Infinity", "-Infinity")):
            root = self.root / str(i)
            root.mkdir()
            (root / "failure.json").write_text('{"wall_seconds": ' + value + "}")
            write_json(root / "manifest.json", manifest(root))
            with self.assertRaises(ValueError):
                resume_seconds(root / "full-s11/epoch_000.pt", 10)

    def prices(self, asset, count=225):
        path = self.root / "input" / "Data" / "ETFs" / (asset + ".us.txt")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["Date", "Open", "High", "Low", "Close", "Volume", "OpenInt"])
            for i in range(count):
                close = 100 + i * 0.1
                writer.writerow(
                    [
                        (date(2010, 1, 1) + timedelta(days=i)).isoformat(),
                        close * 0.99,
                        close * 1.02,
                        close * 0.97,
                        close,
                        1000 + i,
                        0,
                    ]
                )
        return path

    def test_intake_end_to_end_values_hashes_and_calendar(self):
        paths = {asset: self.prices(asset) for asset in ASSETS}
        out = self.root / "prepared"
        panel = prepare(self.root / "input", out)
        with panel.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), (225 - 20) * 8)
        self.assertEqual({r["asset"] for r in rows[:8]}, {a.upper() for a in ASSETS})
        self.assertAlmostEqual(float(rows[0]["return"]), 102 / 101.9 - 1)
        self.assertAlmostEqual(float(rows[0]["feature_00_intraday"]), 1 / 0.99 - 1)
        meta = read_json(str(panel) + ".manifest.json")
        self.assertEqual(meta["sha256"], digest(panel))
        self.assertEqual(meta["purpose"], "historical_proxy")
        receipt = read_json(out / "intake.json")
        self.assertEqual(receipt["status"], "COMPLETE")
        self.assertEqual(receipt["files"]["spy"]["sha256"], digest(paths["spy"]))
        with self.assertRaises(FileExistsError):
            prepare(self.root / "input", out)

    def test_intake_calendar_mismatch_is_retained_failure(self):
        for asset in ASSETS:
            self.prices(asset, 224 if asset == "slv" else 225)
        out = self.root / "failed"
        with self.assertRaisesRegex(ValueError, "calendar mismatch"):
            prepare(self.root / "input", out)
        self.assertEqual(read_json(out / "intake.json")["status"], "FAILED")
        self.assertFalse((out / "panel.csv").exists())

    def test_csv_extra_missing_and_invalid_open_interest_rejected(self):
        path = self.prices("spy", 1)
        header = path.read_text().splitlines()[0]
        for row in (
            "2010-01-01,1,2,1,2,100,0,extra",
            "2010-01-01,1,2,1,2,100",
            "2010-01-01,1,2,1,2,100,nan",
            "2010-01-01,1,2,1,2,100,-1",
            "2010-01-01,1,2,1,2,100,0.5",
        ):
            path.write_text(header + "\n" + row + "\n")
            with self.subTest(row=row), self.assertRaises(ValueError):
                read_prices(path)

    def test_vintage_join_matches_bruteforce_for_random_revisions(self):
        rng = random.Random(1901)
        epoch = date(2000, 1, 1)

        def stamp(d):
            return (epoch + timedelta(days=d)).isoformat() + "T00:00:00Z"

        observations, seen = [], set()
        for _ in range(150):
            period = rng.randrange(50)
            known = period + rng.randrange(1, 20)
            if (known, period) not in seen:
                seen.add((known, period))
                observations.append((stamp(known), stamp(period), rng.uniform(-3, 3)))
        decisions = [stamp(i) for i in range(70)]
        expected = []
        for decision in decisions:
            eligible = [r for r in observations if r[0] <= decision]
            latest = max(eligible, key=lambda r: (r[1], r[0])) if eligible else None
            expected.append(None if latest is None else (latest[2], timestamp(latest[0])))
        rng.shuffle(observations)
        self.assertEqual(asof_values(decisions, observations), expected)


if __name__ == "__main__":
    unittest.main()
