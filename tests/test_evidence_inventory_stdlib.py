"""Dependency-free regression tests for preservation inventory tooling."""

import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[1] / "scripts/inventory_untracked_evidence.py"
SPEC = importlib.util.spec_from_file_location("evidence_inventory", SOURCE)
inventory = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(inventory)


class PreservationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.group = next(iter(inventory.GROUPS))
        self.relative = self.group + "/sample.json"
        self.file = self.root / self.relative
        self.file.parent.mkdir(parents=True)
        self.file.write_text("evidence")

    def baseline(self):
        return {"path_count": 1, "entries": inventory.inspect_entries(self.root, [self.relative])}

    def test_hashes_match_and_new_paths_are_not_added_to_baseline(self):
        baseline = self.baseline()
        self.assertEqual(
            inventory.inspect_entries(self.root, ["new-unclassified"], baseline), baseline["entries"]
        )

    def test_same_size_content_change_is_rejected(self):
        baseline = self.baseline()
        self.file.write_text("tampered")
        with self.assertRaisesRegex(ValueError, "digest differs"):
            inventory.inspect_entries(self.root, [], baseline)

    def test_missing_evidence_is_rejected(self):
        baseline = self.baseline()
        self.file.unlink()
        with self.assertRaises(FileNotFoundError):
            inventory.inspect_entries(self.root, [], baseline)

    def test_legacy_baseline_still_checks_sizes(self):
        baseline = self.baseline()
        del baseline["entries"][0]["sha256"]
        self.assertIn("sha256", inventory.inspect_entries(self.root, [], baseline)[0])
        self.file.write_text("changed size")
        with self.assertRaisesRegex(ValueError, "metadata differs"):
            inventory.inspect_entries(self.root, [], baseline)

    def test_duplicate_baseline_path_is_rejected(self):
        baseline = self.baseline()
        baseline["entries"].append(baseline["entries"][0])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            inventory.inspect_entries(self.root, [], baseline)

    def test_baseline_count_is_checked(self):
        baseline = self.baseline()
        baseline["path_count"] = 0
        with self.assertRaisesRegex(ValueError, "path_count"):
            inventory.inspect_entries(self.root, [], baseline)

    def test_symlink_input_and_parent_are_rejected(self):
        link = self.file.parent / "link.json"
        link.symlink_to(self.file)
        with self.assertRaisesRegex(ValueError, "symlink"):
            inventory.inspect_entries(self.root, [self.group + "/link.json"])
        link.unlink()
        directory = self.file.parent / "alias"
        directory.symlink_to(self.file.parent, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "symlink"):
            inventory.inspect_entries(self.root, [self.group + "/alias/sample.json"])

    def test_nonregular_input_and_traversal_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "nonregular"):
            inventory.regular_path(self.root, self.group)
        for path in ("../outside", "/absolute", self.group + "/../sample.json", self.group + "//sample.json"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                inventory.regular_path(self.root, path)

    def test_unclassified_path_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unclassified"):
            inventory.inspect_entries(self.root, ["unexpected.json"])

    def test_receipt_creation_uses_exclusive_open(self):
        output = self.root / "fresh.json"
        with (
            patch.object(sys, "argv", ["inventory", "--output", str(output)]),
            patch.object(inventory, "git_untracked", return_value=[]),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            inventory.main()
        self.assertEqual(json.loads(output.read_text())["path_count"], 0)

    def test_existing_receipt_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "receipt.json"
            output.write_text("original evidence\n")
            with (
                patch.object(sys, "argv", ["inventory", "--output", str(output)]),
                patch.object(inventory, "git_untracked", return_value=[]),
                contextlib.redirect_stdout(io.StringIO()),
                self.assertRaises(FileExistsError),
            ):
                inventory.main()
            self.assertEqual(output.read_text(), "original evidence\n")

    def test_output_symlink_does_not_overwrite_target(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "old.json"
            target.write_text("original evidence\n")
            output = Path(temporary) / "link.json"
            output.symlink_to(target)
            with (
                patch.object(sys, "argv", ["inventory", "--output", str(output)]),
                patch.object(inventory, "git_untracked", return_value=[]),
                contextlib.redirect_stdout(io.StringIO()),
                self.assertRaises((FileExistsError, ValueError)),
            ):
                inventory.main()
            self.assertEqual(target.read_text(), "original evidence\n")


if __name__ == "__main__":
    unittest.main()
