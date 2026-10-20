"""Small dependency-free hostile-input and vintage-join checks."""

import csv
import tempfile
import unittest
from pathlib import Path

from frem_s14.intake import asof_values, read_prices


class IntakeTests(unittest.TestCase):
    def prices(self, rows):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        path = Path(temp.name) / "prices.txt"
        with path.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["Date", "Open", "High", "Low", "Close", "Volume", "OpenInt"])
            writer.writerows(rows)
        return path

    def test_protected_values_discarded_before_parsing(self):
        path = self.prices(
            [
                ["2010-01-01", 1, 2, 1, 2, 100, 0],
                ["2025-01-01", "DO_NOT_PARSE", "x", "x", "x", "x", 0],
            ]
        )
        self.assertEqual(len(read_prices(path)), 1)

    def test_duplicate_date_rejected(self):
        row = ["2010-01-01", 1, 2, 1, 2, 100, 0]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            read_prices(self.prices([row, row]))

    def test_bad_envelope_zero_nan_negative_volume_rejected(self):
        for values in (
            [1, 1, 1, 2, 100],
            [0, 2, 0, 2, 100],
            [1, 2, 1, "nan", 100],
            [1, 2, 1, 2, -1],
        ):
            with self.subTest(values=values), self.assertRaises(ValueError):
                read_prices(self.prices([["2010-01-01", *values, 0]]))

    def test_missing_history_not_backfilled(self):
        self.assertEqual(
            asof_values(
                ["2000-01-01T00:00:00Z"], [("2000-02-01T00:00:00Z", "2000-01-01T00:00:00Z", 5)]
            ),
            [None],
        )

    def test_old_revision_does_not_replace_newer_observation(self):
        result = asof_values(
            ["2000-05-01T00:00:00Z"],
            [
                ("2000-02-01T00:00:00Z", "2000-01-01T00:00:00Z", 1),
                ("2000-03-01T00:00:00Z", "2000-02-01T00:00:00Z", 2),
                ("2000-04-01T00:00:00Z", "2000-01-01T00:00:00Z", 99),
            ],
        )
        self.assertEqual(result[0][0], 2)

    def test_latest_available_revision_only(self):
        result = asof_values(
            ["2000-03-01T00:00:00Z", "2000-05-01T00:00:00Z"],
            [
                ("2000-02-01T00:00:00Z", "2000-01-01T00:00:00Z", 1),
                ("2000-04-01T00:00:00Z", "2000-01-01T00:00:00Z", 2),
            ],
        )
        self.assertEqual([x[0] for x in result], [1, 2])

    def test_ambiguous_vintage_rejected(self):
        row = ("2000-02-01T00:00:00Z", "2000-01-01T00:00:00Z", 1)
        with self.assertRaises(ValueError):
            asof_values([], [row, row])

    def test_timezone_required(self):
        with self.assertRaises(ValueError):
            asof_values(["2000-01-01"], [])


if __name__ == "__main__":
    unittest.main()
