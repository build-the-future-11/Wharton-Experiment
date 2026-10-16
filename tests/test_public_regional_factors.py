"""Offline source-parser checks; no network or scientific libraries."""

import unittest

from scripts.acquire_public_regional_factors import parse_monthly


class RegionalFactorTests(unittest.TestCase):
    def test_percent_units_and_market_return(self):
        row = parse_monthly("header\n200801,-5,1,2,0.2\n", "202501")[0]
        self.assertAlmostEqual(row["market_total"], -0.048)
        self.assertAlmostEqual(row["rf"], 0.002)

    def test_excluded_values_not_parsed(self):
        rows = parse_monthly("202412,1,2,3,0.1\n202501,DO,NOT,PARSE,OUTCOMES\n", "202501")
        self.assertEqual(len(rows), 1)

    def test_annual_section_ignored(self):
        rows = parse_monthly("202412,1,2,3,0.1\nAnnual Returns\n2024,8,6,4,3\n", "202501")
        self.assertEqual(len(rows), 1)

    def test_missing_duplicate_bad_date_or_nonfinite_rejected(self):
        for text in ("200801,-99.99,1,2,0", "200813,1,2,3,0", "200801,nan,2,3,0",
                     "200801,1,2,3,0\n200801,1,2,3,0", "200802,1,2,3,0\n200801,1,2,3,0"):
            with self.assertRaises(ValueError):
                parse_monthly(text, "202501")


if __name__ == "__main__":
    unittest.main()
