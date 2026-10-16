"""Historical source restoration must preserve the original run identity."""

import hashlib
import tempfile
import unittest
from pathlib import Path

from scripts.audit_engineering_evidence import verify_source_archive


class SourceArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.archive = self.root / "archive"
        self.archive.mkdir()
        self.original = b"original source\n"
        self.expected = {"model.py": hashlib.sha256(self.original).hexdigest()}
        (self.archive / "model.py").write_bytes(self.original)
        (self.root / "model.py").write_text("new successor source\n")

    def test_original_archive_matches_even_when_live_source_changed(self):
        result = verify_source_archive(self.root, self.archive, self.expected)
        self.assertEqual(result, {"archive/model.py": self.expected["model.py"]})
        with self.assertRaisesRegex(ValueError, "Hash mismatch"):
            verify_source_archive(self.root, self.root, self.expected)

    def test_changed_archive_is_rejected(self):
        (self.archive / "model.py").write_text("modified historical source")
        with self.assertRaisesRegex(ValueError, "Hash mismatch"):
            verify_source_archive(self.root, self.archive, self.expected)

    def test_missing_archive_file_is_rejected(self):
        (self.archive / "model.py").unlink()
        with self.assertRaises(FileNotFoundError):
            verify_source_archive(self.root, self.archive, self.expected)

    def test_source_root_outside_repository_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "escapes repository"):
            verify_source_archive(self.root, self.root.parent, self.expected)

    def test_path_escaping_archive_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "escapes archive"):
            verify_source_archive(self.root, self.archive, {"../model.py": self.expected["model.py"]})


if __name__ == "__main__":
    unittest.main()
