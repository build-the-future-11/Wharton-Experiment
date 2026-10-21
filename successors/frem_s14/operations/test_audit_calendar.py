import csv
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path

from audit_calendar import ASSETS, audit


class CalendarAuditTests(unittest.TestCase):
    def fixture(self, root, extra=False):
        for asset in ASSETS:
            with (root / (asset + ".us.txt")).open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["Date", "Open", "High", "Low", "Close", "Volume", "OpenInt"])
                for i in range(201 if extra and asset == "qqq" else 200):
                    writer.writerow(
                        [(date(2007, 1, 1) + timedelta(days=i)).isoformat(), 1, 1, 1, 1, 1, 0]
                    )

    def test_reports_every_asset_gap_without_modifying_inputs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.fixture(root, extra=True)
            before = {p.name: p.read_bytes() for p in root.iterdir()}
            result = audit(root)
            self.assertEqual(result["status"], "BLOCKED_SOURCE_CALENDARS")
            self.assertEqual(result["union_dates"], 201)
            self.assertEqual(result["common_dates_diagnostic_only"], 200)
            for asset in ASSETS:
                self.assertEqual(
                    len(result["files"][asset]["missing_from_union"]), int(asset != "qqq")
                )
            self.assertEqual(before, {p.name: p.read_bytes() for p in root.iterdir()})

    def test_matching_dates_do_not_claim_scientific_admission(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.fixture(root)
            result = audit(root)
            self.assertEqual(result["status"], "CALENDARS_COMPATIBLE")
            self.assertFalse(result["scientific_execution"])
            self.assertFalse(result["input_hashes_match_executed_run"])

    def test_changed_source_and_duplicate_dates_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.fixture(root)
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                audit(root, dict.fromkeys(ASSETS, "0" * 64))
            with (root / "spy.us.txt").open("a") as stream:
                stream.write("2007-01-01,1,1,1,1,1,0\n")
            with self.assertRaisesRegex(ValueError, "duplicate or unordered"):
                audit(root)


if __name__ == "__main__":
    unittest.main()
