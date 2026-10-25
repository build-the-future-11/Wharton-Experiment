"""Transport regressions independent of numerical research dependencies."""

import importlib.util
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location(
    "pack_results", Path(__file__).resolve().parents[1] / "scripts/pack_results.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class PackResultsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.run = self.root / "failed-run"
        self.run.mkdir()
        (self.run / "failure.json").write_text('{"status":"FAILED"}\n')
        (self.run / "partial.bin").write_bytes(b"\x00\xff" * 100)
        self.output = self.root / "results.zip"

    def test_failed_run_retained_without_claim_promotion(self):
        before = {p.name: p.read_bytes() for p in self.run.iterdir()}
        result = MODULE.pack(self.run, self.output)
        self.assertEqual(result["status"], "PASS_TRANSPORT_ONLY")
        self.assertEqual(result["scientific_validity"], "NOT_ASSESSED")
        with zipfile.ZipFile(self.output) as archive:
            self.assertEqual(archive.read("failed-run/failure.json"), before["failure.json"])
            self.assertEqual(archive.read("failed-run/partial.bin"), before["partial.bin"])
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.run.iterdir()})

    def test_existing_zip_is_preserved(self):
        self.output.write_bytes(b"old evidence")
        with self.assertRaises(FileExistsError):
            MODULE.pack(self.run, self.output)
        self.assertEqual(self.output.read_bytes(), b"old evidence")

    def test_existing_receipt_is_preserved(self):
        receipt = Path(str(self.output) + ".receipt.json")
        receipt.write_text("old receipt")
        with self.assertRaises(FileExistsError):
            MODULE.pack(self.run, self.output)
        self.assertFalse(self.output.exists())

    def test_output_inside_run_rejected(self):
        with self.assertRaises(ValueError):
            MODULE.pack(self.run, self.run / "recursive.zip")

    def test_input_symlink_rejected(self):
        (self.run / "link").symlink_to(self.run / "failure.json")
        with self.assertRaises(ValueError):
            MODULE.pack(self.run, self.output)
        self.assertFalse(self.output.exists())

    def test_output_symlink_rejected(self):
        self.output.symlink_to(self.run / "failure.json")
        with self.assertRaises(FileExistsError):
            MODULE.pack(self.run, self.output)

    def test_empty_run_rejected(self):
        empty = self.root / "empty"
        empty.mkdir()
        with self.assertRaises(ValueError):
            MODULE.pack(empty, self.output)

    def test_changed_input_preserves_failure_receipt(self):
        original = MODULE.regular_tree
        count = 0

        def change(root):
            nonlocal count
            count += 1
            if count == 2:
                (root / "partial.bin").write_bytes(b"changed")
            return original(root)

        with (
            patch.object(MODULE, "regular_tree", side_effect=change),
            self.assertRaisesRegex(ValueError, "changed during packing"),
        ):
            MODULE.pack(self.run, self.output)
        receipt = json.loads(Path(str(self.output) + ".receipt.json").read_text())
        self.assertEqual(receipt["status"], "FAILED")
        self.assertTrue(self.output.exists())


if __name__ == "__main__":
    unittest.main()
