"""Offline regression cases for immutable inventories and honest verification scope."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.build_release_manifests import build_manifest, inventory, verify_manifest, write_snapshots
from scripts.validate_source import inspect_source, strict_json


class ReleaseToolingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        for relative in (".", "successors/rejepa_v1", "successors/reng_jepa_v01"):
            scope = self.root / relative
            (scope / "src/frem").mkdir(parents=True)
            (scope / "src/frem/__init__.py").write_text("")
            (scope / "src/frem/model.py").write_text("VALUE = 1\n")
            (scope / "pyproject.toml").write_text('[project]\nname = "fixture"\nversion = "9.8.7"\n')
        (self.root / "RESEARCH_STATE.json").write_text('{"scientific_status": "NOT_EVALUATED"}')
        self.output = self.root / "snapshots"

    def manifest(self):
        return build_manifest(self.root, root_release=True, output_dir=self.output)

    def test_inventory_never_claims_runtime_verification(self):
        dist = self.root / "dist"
        dist.mkdir()
        (dist / "old.whl").write_bytes(b"old build")
        manifest = self.manifest()
        self.assertEqual(manifest["version"], "9.8.7")
        self.assertEqual(manifest["release_status"], "SOURCE_INVENTORY_ONLY")
        self.assertTrue(all(value.startswith("NOT_") for value in manifest["verification"].values()))
        self.assertEqual(len(manifest["build_artifacts"]), 1)
        self.assertNotIn("dist/old.whl", [item["path"] for item in manifest["files"]])

    def test_excluded_trees_are_pruned_before_reading(self):
        for name in ("data/raw", "outputs", "runs", "tmp", ".venv", "legacy", ".local_tmp"):
            path = self.root / name
            path.mkdir(parents=True)
            # An excluded symlink would fail if the inventory entered this tree.
            (path / "opaque").symlink_to(self.root / "src/frem/model.py")
        (self.root / ".env").write_text("NOT_A_RELEASE_INPUT=example")
        (self.root / ".gitignore").write_text(".env\n")
        names = [item["path"] for item in self.manifest()["files"]]
        self.assertEqual(names, [".gitignore", "RESEARCH_STATE.json", "pyproject.toml",
                                 "src/frem/__init__.py", "src/frem/model.py"])

    def test_snapshot_is_fresh_and_previous_manifests_are_preserved(self):
        previous = self.root / "RELEASE_MANIFEST.json"
        previous.write_text('{"historical": true}\n')
        paths = write_snapshots(self.root, self.output)
        before = {path: path.read_bytes() for path in paths}
        with self.assertRaises(FileExistsError):
            write_snapshots(self.root, self.output)
        self.assertEqual(previous.read_text(), '{"historical": true}\n')
        for path in paths:
            self.assertEqual(path.read_bytes(), before[path])
        payload = json.loads(paths[0].read_text())
        self.assertFalse(any(item["path"].startswith("snapshots/") for item in payload["files"]))
        self.assertEqual(payload["previous_manifest"]["path"], "RELEASE_MANIFEST.json")

    def test_hashing_does_not_read_whole_files(self):
        with patch.object(Path, "read_bytes", side_effect=AssertionError("whole-file read")):
            self.assertEqual(verify_manifest(self.root, self.manifest())["status"], "PASS")

    def test_missing_and_changed_files_fail_verification(self):
        manifest = self.manifest()
        (self.root / "src/frem/model.py").write_text("VALUE = 2\n")
        (self.root / "src/frem/__init__.py").unlink()
        result = verify_manifest(self.root, manifest)
        self.assertEqual(result["status"], "FAIL")
        self.assertEqual({item["type"] for item in result["errors"]}, {"identity_mismatch", "missing_file"})

    def test_count_and_duplicate_errors_are_not_ignored(self):
        manifest = self.manifest()
        manifest["files"].append(manifest["files"][0])
        errors = verify_manifest(self.root, manifest)["errors"]
        self.assertEqual({item["type"] for item in errors}, {"count_mismatch", "duplicate_path"})

    def test_empty_inventory_does_not_pass(self):
        result = verify_manifest(self.root, {"files": [], "file_count_excluding_manifest": 0})
        self.assertEqual(result["status"], "FAIL")

    def test_previous_manifest_identity_is_verified(self):
        previous = self.root / "RELEASE_MANIFEST.json"
        previous.write_text('{"old": true}')
        manifest = self.manifest()
        previous.write_text('{"old": false}')
        result = verify_manifest(self.root, manifest)
        self.assertIn({"path": "RELEASE_MANIFEST.json", "type": "identity_mismatch"}, result["errors"])

    def test_path_escape_is_rejected_before_read(self):
        for name in ("../outside", "/outside", ""):
            with self.subTest(name=name):
                manifest = self.manifest()
                manifest["files"][0]["path"] = name
                with self.assertRaisesRegex(ValueError, "escapes"):
                    verify_manifest(self.root, manifest)

    def test_source_file_symlinks_are_rejected(self):
        path = self.root / "src/frem/model.py"
        path.unlink()
        path.symlink_to(self.root / "pyproject.toml")
        with self.assertRaisesRegex(ValueError, "Symlink"):
            self.manifest()

    def test_source_directory_symlinks_are_rejected(self):
        (self.root / "docs").symlink_to(self.root / "src", target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "Symlink directory"):
            self.manifest()

    def test_verifier_rejects_symlink_substitution(self):
        manifest = self.manifest()
        path = self.root / "src/frem/model.py"
        path.unlink()
        path.symlink_to(self.root / "pyproject.toml")
        with self.assertRaisesRegex(ValueError, "symlink"):
            verify_manifest(self.root, manifest)

    def test_output_cannot_contain_source(self):
        with self.assertRaisesRegex(ValueError, "must not contain"):
            write_snapshots(self.root, self.root.parent)

    def test_empty_source_package_is_rejected(self):
        for path in (self.root / "src/frem").glob("*.py"):
            path.unlink()
        with self.assertRaisesRegex(ValueError, "No Python source"):
            self.manifest()

    def test_static_validation_passes_valid_fixture(self):
        self.assertEqual(inspect_source(self.root)["static_status"], "PASS")

    def test_static_validation_rejects_missing_package(self):
        (self.root / "src/frem/__init__.py").unlink()
        self.assertIn("missing_package", [error["type"] for error in inspect_source(self.root)["errors"]])

    def test_static_validation_rejects_nonfinite_canonical_state(self):
        (self.root / "RESEARCH_STATE.json").write_text('{"metric": NaN}')
        self.assertEqual(inspect_source(self.root)["static_status"], "FAIL")

    def test_static_validation_rejects_duplicate_keys(self):
        (self.root / "configs").mkdir()
        (self.root / "configs/protocol.json").write_text('{"seed": 11, "seed": 29}')
        self.assertEqual(inspect_source(self.root)["static_status"], "FAIL")

    def test_json_rejects_nonfinite_and_duplicate_nested_values(self):
        for text in ('{"x": Infinity}', '{"x": -Infinity}', '{"x": 1e400}',
                     '{"a": {"x": 1, "x": 2}}'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                strict_json(text)

    def test_inventory_order_is_repeatable(self):
        first = inventory(self.root, self.root / "RELEASE_MANIFEST.json", root_release=True)
        second = inventory(self.root, self.root / "RELEASE_MANIFEST.json", root_release=True)
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
