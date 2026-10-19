"""Stdlib-only replay-driver contracts; tiny child processes, no model execution.

Run from the checkout: python -B tests/test_kaggle_driver_contract.py -v
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest import mock
import zipfile


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_integrity_kaggle.py"
SPEC = importlib.util.spec_from_file_location("integrity_driver_under_test", SCRIPT)
DRIVER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DRIVER)  # Driver module-level imports are standard-library only.
GIB, MIB = 1024**3, 1024**2


class KaggleDriverContractTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="kaggle_driver_contract_")
        self.addCleanup(temporary.cleanup)
        self.output = Path(temporary.name)
        self.environment = os.environ.copy()
        self.admitted = {"admitted": True, "available_bytes": 2 * GIB,
                         "required_bytes": GIB + 512 * MIB}

    def run_child(self, name, command):
        with mock.patch.object(DRIVER, "memory_admission", return_value=self.admitted):
            return DRIVER.run_command(name, command, self.output, self.environment,
                                      deadline=time.perf_counter() + 10)

    def retained_receipt(self, name, expected):
        directory = self.output / "commands" / name
        receipt = json.loads((directory / "receipt.json").read_text())
        self.assertEqual(receipt, expected)
        for filename in ("stdout.log", "stderr.log"):
            actual = hashlib.sha256((directory / filename).read_bytes()).hexdigest()
            self.assertEqual(receipt["logs_sha256"][filename], actual)
        return directory

    @staticmethod
    def probe_memory(files):
        def read_text(path, *args, **kwargs):
            try:
                return files[str(path)]
            except KeyError:
                raise FileNotFoundError(str(path)) from None

        with mock.patch.object(DRIVER.Path, "read_text", read_text), \
             mock.patch.object(DRIVER.Path, "is_file", lambda path: str(path) in files), \
             mock.patch.object(DRIVER.Path, "resolve", lambda path: path):
            return DRIVER.memory_admission()

    def test_memory_formula_boundary_and_dynamic_host_limit(self):
        files = {
            "/proc/meminfo": f"MemAvailable: {4 * GIB // 1024} kB\n",
            "/proc/self/cgroup": "0::/\n",
            "/sys/fs/cgroup/memory.max": str(3 * GIB),
            "/sys/fs/cgroup/memory.current": str(2 * GIB),
            "/sys/fs/cgroup/memory.stat": (
                f"inactive_file {GIB}\nfile_dirty {256 * MIB}\n"
                f"file_writeback {128 * MIB}\nshmem {128 * MIB}\n"),
        }
        boundary = self.probe_memory(files)
        self.assertEqual(boundary["available_bytes"], 1536 * MIB)
        self.assertEqual(boundary["required_bytes"], 1536 * MIB)
        self.assertTrue(boundary["admitted"])
        files["/sys/fs/cgroup/memory.current"] = str(2 * GIB + 1)
        self.assertFalse(self.probe_memory(files)["admitted"])
        files["/proc/meminfo"] = f"MemAvailable: {GIB // 1024} kB\n"
        self.assertEqual(self.probe_memory(files)["available_bytes"], GIB)

    def test_absent_cgroup_uses_host_memory(self):
        result = self.probe_memory({
            "/proc/meminfo": f"MemAvailable: {2 * GIB // 1024} kB\n",
            "/proc/self/cgroup": "0::/\n",
        })
        self.assertEqual(result["cgroups"], [])
        self.assertEqual(result["available_bytes"], 2 * GIB)
        self.assertTrue(result["admitted"])

    def test_memory_denial_prevents_launch_and_retains_unstarted_receipt(self):
        denied = dict(self.admitted, admitted=False, available_bytes=GIB)
        with mock.patch.object(DRIVER, "memory_admission", return_value=denied), \
             mock.patch.object(DRIVER.subprocess, "Popen", side_effect=AssertionError("must not launch")) as launch:
            receipt = DRIVER.run_command("memory_denied", ["never-execute"], self.output,
                                         self.environment, deadline=time.perf_counter() + 10)
        launch.assert_not_called()
        self.assertEqual(receipt["status"], "not_started_memory")
        self.assertIsNone(receipt["exit_status"])
        self.assertEqual(receipt["memory_admission"], denied)
        directory = self.retained_receipt("memory_denied", receipt)
        self.assertIn("Memory admission denied", (directory / "stderr.log").read_text())
        plan = [{"name": "memory_denied", "artifact_directory": str(directory / "artifacts"),
                 "cells": [{"cell_id": "never_started"}]}]
        cells = DRIVER.matrix_receipts(plan, [receipt])
        self.assertEqual(cells["cells"][0]["status"], "not_started")

    def test_failed_child_preserves_exit_logs_and_nested_artifacts_in_bundle(self):
        artifact = self.output / "commands/failed_child/artifacts/partial.txt"
        program = """from pathlib import Path
import sys
artifact = Path(sys.argv[1])
artifact.parent.mkdir(parents=True)
artifact.write_text('completed work before failure', encoding='utf-8')
print('stdout before failure', flush=True)
print('stderr before failure', file=sys.stderr, flush=True)
raise SystemExit(23)
"""
        command = [sys.executable, "-I", "-c", program, str(artifact)]
        receipt = self.run_child("failed_child", command)
        self.assertEqual(receipt["status"], "failed")
        self.assertEqual(receipt["exit_status"], 23)
        self.assertEqual(receipt["command"], command)
        directory = self.retained_receipt("failed_child", receipt)
        self.assertEqual((directory / "stdout.log").read_text(), "stdout before failure\n")
        self.assertEqual((directory / "stderr.log").read_text(), "stderr before failure\n")
        self.assertEqual(artifact.read_text(), "completed work before failure")
        DRIVER.bundle(self.output)
        relative = str(artifact.relative_to(self.output))
        manifest = json.loads((self.output / "artifact_manifest.json").read_text())
        self.assertEqual(manifest["files"][relative]["sha256"], hashlib.sha256(artifact.read_bytes()).hexdigest())
        with zipfile.ZipFile(self.output / "replay_bundle.zip") as archive:
            self.assertEqual(archive.read(relative), artifact.read_bytes())
            saved = json.loads(archive.read("commands/failed_child/receipt.json"))
            self.assertEqual(saved["exit_status"], 23)
        checksum = hashlib.sha256((self.output / "replay_bundle.zip").read_bytes()).hexdigest()
        self.assertEqual((self.output / "replay_bundle.sha256").read_text(),
                         f"{checksum}  replay_bundle.zip\n")
        self.assertFalse((self.output / "replay_bundle.zip.partial").exists())
        self.assertFalse((self.output / "replay_bundle.sha256.partial").exists())

    def test_archive_hash_failure_leaves_only_partial_bundle(self):
        (self.output / "retained.txt").write_text("retain completed output")
        original_digest = DRIVER.digest

        def failing_digest(path):
            if path.name == "replay_bundle.zip.partial":
                raise OSError("injected archive hash failure")
            return original_digest(path)

        with mock.patch.object(DRIVER, "digest", side_effect=failing_digest):
            with self.assertRaisesRegex(OSError, "injected archive hash failure"):
                DRIVER.bundle(self.output)
        self.assertFalse((self.output / "replay_bundle.zip").exists())
        self.assertFalse((self.output / "replay_bundle.sha256").exists())
        self.assertTrue((self.output / "replay_bundle.zip.partial").is_file())
        with zipfile.ZipFile(self.output / "replay_bundle.zip.partial") as archive:
            self.assertEqual(archive.read("retained.txt"), b"retain completed output")

    def test_checksum_write_failure_cannot_publish_success_bundle(self):
        (self.output / "retained.txt").write_text("retain completed output")
        original_write_text = DRIVER.Path.write_text

        def failing_write_text(path, text, *args, **kwargs):
            if path.name == "replay_bundle.sha256.partial":
                original_write_text(path, "interrupted checksum write", *args, **kwargs)
                raise OSError("injected checksum write failure")
            return original_write_text(path, text, *args, **kwargs)

        with mock.patch.object(DRIVER.Path, "write_text", failing_write_text):
            with self.assertRaisesRegex(OSError, "injected checksum write failure"):
                DRIVER.bundle(self.output)
        self.assertFalse((self.output / "replay_bundle.zip").exists())
        self.assertFalse((self.output / "replay_bundle.sha256").exists())
        self.assertTrue((self.output / "replay_bundle.zip.partial").is_file())
        self.assertEqual((self.output / "replay_bundle.sha256.partial").read_text(),
                         "interrupted checksum write")
        manifest = json.loads((self.output / "artifact_manifest.json").read_text())
        self.assertNotIn("replay_bundle.zip.partial", manifest["files"])
        self.assertNotIn("replay_bundle.sha256.partial", manifest["files"])

    def test_timeout_is_bounded_and_retains_receipt(self):
        # Even a broken timeout cannot leave this artificial child sleeping indefinitely.
        command = [sys.executable, "-I", "-c", "import time; time.sleep(3)"]
        with mock.patch.object(DRIVER, "COMMAND_SECONDS", 1):
            receipt = self.run_child("timeout", command)
        self.assertEqual(receipt["status"], "timed_out")
        self.assertTrue(receipt["timed_out"])
        self.assertEqual(receipt["timeout_seconds"], 1)
        self.assertIsNotNone(receipt["exit_status"])
        self.assertNotEqual(receipt["exit_status"], 0)
        self.assertLess(receipt["wall_seconds"], 5)
        self.retained_receipt("timeout", receipt)

    def test_launch_error_retains_errno_class_and_exit_status(self):
        receipt = self.run_child("launch_error", [str(self.output / "nonexistent_executable")])
        self.assertEqual(receipt["status"], "launch_failed")
        self.assertEqual(receipt["exit_status"], 127)
        self.assertFalse(receipt["timed_out"])
        directory = self.retained_receipt("launch_error", receipt)
        self.assertIn("FileNotFoundError", (directory / "stderr.log").read_text())


if __name__ == "__main__":
    unittest.main()
