"""Exact input retention and chronological-prefix rejection contracts."""

import tempfile
import unittest
from pathlib import Path

from frem_s14.runtime import digest, read_json, retain_input, write_json
from frem_s14.suite import verify_fold_prefix


class InputReadbackTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "input.csv"
        self.source.write_text(
            "timestamp,asset,return\n"
            "2010-12-31T23:00:00Z,A,0.01\n"
            "2011-01-01T01:00:00+02:00,B,0.02\n"
            "2011-01-01T00:00:00Z,A,0.03\n"
        )
        self.meta = {"sha256": digest(self.source), "purpose": "generated fixture"}
        write_json(str(self.source) + ".manifest.json", self.meta)
        self.fold = self.root / "fold.csv"
        self.fold.write_text("\n".join(self.source.read_text().splitlines()[:3]) + "\n")

    def test_retained_input_bytes_and_metadata_are_exact(self):
        target = retain_input(self.source, self.root / "retained.csv")
        self.assertEqual(target.read_bytes(), self.source.read_bytes())
        self.assertEqual(read_json(str(target) + ".manifest.json"), self.meta)

    def test_retention_never_overwrites(self):
        with self.assertRaises(FileExistsError):
            retain_input(self.source, self.fold)

    def test_corrupt_supplier_checksum_is_rejected(self):
        self.source.write_text(self.source.read_text() + "x")
        with self.assertRaisesRegex(ValueError, "checksum"):
            retain_input(self.source, self.root / "retained.csv")
        self.assertFalse((self.root / "retained.csv").exists())

    def test_symlink_input_is_rejected(self):
        link = self.root / "link.csv"
        link.symlink_to(self.source)
        with self.assertRaisesRegex(ValueError, "regular"):
            retain_input(link, self.root / "retained.csv")

    def test_prefix_uses_utc_instead_of_date_substring(self):
        verify_fold_prefix(self.source, self.fold, "2011-01-01")

    def test_prefix_rejects_changed_removed_added_or_reordered_rows(self):
        lines = self.fold.read_text().splitlines(True)
        for content in (
            "".join(lines).replace("0.01", "0.99"),
            "".join(lines[:2]),
            "".join(lines + [lines[1]]),
            "".join([lines[0], lines[2], lines[1]]),
        ):
            with self.subTest(content=content):
                self.fold.write_text(content)
                with self.assertRaisesRegex(ValueError, "prefix"):
                    verify_fold_prefix(self.source, self.fold, "2011-01-01")

    def test_prefix_rejects_changed_header(self):
        self.fold.write_text(self.fold.read_text().replace("asset", "security"))
        with self.assertRaisesRegex(ValueError, "header"):
            verify_fold_prefix(self.source, self.fold, "2011-01-01")

    def test_prefix_rejects_malformed_source_width(self):
        text = self.fold.read_text().replace("A,0.01", "A,0.01,extra")
        self.source.write_text(text)
        self.fold.write_text(text)
        with self.assertRaisesRegex(ValueError, "prefix"):
            verify_fold_prefix(self.source, self.fold, "2011-01-01")
