"""Tamper tests for the portable handoff validator; no research execution."""

import importlib.util
import json
import shutil
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("validate_package", ROOT / "scripts/validate_kaggle_package.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class HandoffValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name).resolve() / "handoff"
        self.folder.mkdir()
        for name in (
            MODULE.NOTEBOOK,
            MODULE.ARCHIVE,
            "RELEASE-0.2.4.json",
            "kernel-metadata-0.2.4.json",
            "input_manifest.json",
        ):
            shutil.copyfile(ROOT / "kaggle" / name, self.folder / name)

    def edit_json(self, name, key, value):
        p = self.folder / name
        data = json.loads(p.read_text())
        data[key] = value
        p.write_text(json.dumps(data))

    def test_exact_package_only_claims_static_pass(self):
        result = MODULE.validate(self.folder)
        self.assertEqual(result["payload_members"], 17)
        self.assertFalse(result["runtime_verified"])
        self.assertFalse(result["manual_launch_ready"])

    def test_notebook_tamper_rejected(self):
        p = self.folder / MODULE.NOTEBOOK
        p.write_bytes(p.read_bytes() + b" ")
        with self.assertRaises(ValueError):
            MODULE.validate(self.folder)

    def test_archive_tamper_rejected(self):
        p = self.folder / MODULE.ARCHIVE
        p.write_bytes(p.read_bytes() + b" ")
        with self.assertRaises(ValueError):
            MODULE.validate(self.folder)

    def test_public_visibility_rejected(self):
        self.edit_json("kernel-metadata-0.2.4.json", "is_private", False)
        with self.assertRaises(ValueError):
            MODULE.validate(self.folder)

    def test_changed_dataset_hash_rejected(self):
        self.edit_json("input_manifest.json", "sha256", {})
        with self.assertRaises(ValueError):
            MODULE.validate(self.folder)

    def test_changed_trust_record_rejected(self):
        self.edit_json("RELEASE-0.2.4.json", "archive_sha256", "0" * 64)
        with self.assertRaises(ValueError):
            MODULE.validate(self.folder)


if __name__ == "__main__":
    unittest.main()
