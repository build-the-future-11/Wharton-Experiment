"""Archived result tampering tests; no extraction or numerical imports."""

import io
import json
import stat
import tempfile
import unittest
import zipfile
from hashlib import sha256
from pathlib import Path

from frem_s14.archive import inventory, verify_download
from frem_s14.runtime import decode_json, digest


def encoded(value):
    return json.dumps(value, sort_keys=True).encode()


def pack(files):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    return buffer.getvalue()


def spec(data):
    return {"sha256": sha256(data).hexdigest(), "bytes": len(data)}


class ArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = b"__version__ = 'test_fixture'\n"
        wheel = pack({"frem_s14/__init__.py": self.source})
        self.payload = {"frem_s14-test.whl": wheel, "tests/test_fixture.py": b"pass\n"}
        self.payload["PAYLOAD.json"] = encoded({k: spec(v) for k, v in self.payload.items()})
        release = pack(self.payload)
        self.release = self.root / "release.zip"
        self.release.write_bytes(release)
        self.output = {
            "run/release.zip": release,
            **{"run/payload/" + k: v for k, v in self.payload.items()},
            "run/installed/frem_s14/__init__.py": self.source,
            "run/delivery.json": encoded(
                {
                    "archive_sha256": digest(self.release),
                    "wheel_sha256": spec(wheel)["sha256"],
                    "mode": "historical_suite",
                }
            ),
            "run/tests.json": encoded(
                {"returncode": 1, "wall_seconds": 2, "command": ["-m", "pytest"]}
            ),
        }

    def check(self):
        path = self.root / "download.zip"
        path.write_bytes(pack(self.output))
        return verify_download(path, self.release, digest(self.release))

    def test_failed_run_integrity_does_not_become_numerical_pass(self):
        receipt = self.check()
        self.assertTrue(receipt["integrity_verified"])
        self.assertFalse(receipt["numerical_semantics_verified"])
        self.assertEqual(receipt["stages"]["tests"]["returncode"], 1)
        self.assertNotIn("qualify", receipt["stages"])

    def test_installed_source_tampering_rejected(self):
        self.output["run/installed/frem_s14/__init__.py"] += b"modified\n"
        with self.assertRaisesRegex(ValueError, "installed Python"):
            self.check()

    def test_extra_installed_module_rejected(self):
        self.output["run/installed/frem_s14/unexpected.py"] = b"pass\n"
        with self.assertRaisesRegex(ValueError, "installed Python"):
            self.check()

    def test_payload_tampering_rejected(self):
        self.output["run/payload/tests/test_fixture.py"] += b"modified\n"
        with self.assertRaisesRegex(ValueError, "payload differs"):
            self.check()

    def test_missing_retained_release_rejected(self):
        del self.output["run/release.zip"]
        with self.assertRaisesRegex(ValueError, "release ZIP"):
            self.check()

    def test_wrong_trusted_release_anchor_rejected(self):
        path = self.root / "download.zip"
        path.write_bytes(pack(self.output))
        with self.assertRaisesRegex(ValueError, "trusted release checksum"):
            verify_download(path, self.release, "0" * 64)

    def test_nested_result_manifest_and_mutation(self):
        result = b'{"status": "FAILED"}'
        self.output["run/qualification/failure.json"] = result
        self.output["run/qualification/manifest.json"] = encoded(
            {"schema": 1, "files": {"failure.json": spec(result)}}
        )
        self.assertEqual(self.check()["manifests_verified"], ["qualification/manifest.json"])
        self.output["run/qualification/failure.json"] = b"{}"
        with self.assertRaisesRegex(ValueError, "result manifest"):
            self.check()

    def test_traversal_and_noncanonical_paths_rejected(self):
        for name in (
            "../outside",
            "/absolute",
            "run/../outside",
            "run\\outside",
            "run//double",
            "./run/value",
        ):
            with self.subTest(name=name):
                with zipfile.ZipFile(io.BytesIO(pack({name: b"x"}))) as archive:
                    with self.assertRaises(ValueError):
                        inventory(archive)

    def test_zip_symlinks_rejected(self):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            entry = zipfile.ZipInfo("run/link")
            entry.external_attr = (stat.S_IFLNK | 0o777) << 16
            archive.writestr(entry, "outside")
        with zipfile.ZipFile(buffer) as archive:
            with self.assertRaises(ValueError):
                inventory(archive)

    def test_multiple_runs_rejected(self):
        self.output["other_run/value"] = b"x"
        with self.assertRaisesRegex(ValueError, "one run directory"):
            self.check()

    def test_ambiguous_nonfinite_and_overflow_json_rejected(self):
        for text in ('{"x": 1e999}', '{"x": -1e999}', '{"x": NaN}', '{"x": 1, "x": 2}'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                decode_json(text)


if __name__ == "__main__":
    unittest.main()
