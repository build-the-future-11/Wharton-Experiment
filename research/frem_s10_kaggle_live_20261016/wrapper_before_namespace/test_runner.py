"""Stdlib fixtures only: no maintained numerical modules or host execution."""
import importlib.util
import io
import contextlib
import json
import os
import signal
import stat
import subprocess
import sys
import tempfile
import time
import unittest
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("tonight", HERE / "runner.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class SafetyTests(unittest.TestCase):
    def test_exited_leader_descendants_receive_cleanup(self):
        with tempfile.TemporaryDirectory() as d:
            r = m.Runner(Path(d), Path(d) / "session")
            child_code = "import subprocess,sys; p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)']); print(p.pid,flush=True)"
            real_kill = os.killpg
            with patch.object(m, "verify_source"), patch.object(m, "deadline_check"), contextlib.redirect_stdout(io.StringIO()), patch.object(m.os, "killpg", wraps=real_kill) as kill:
                with self.assertRaises(subprocess.TimeoutExpired):
                    r.run("orphan-fixture", ["-c", child_code], timeout=0.5, raw=True)
                self.assertTrue(any(call.args[1] == signal.SIGTERM for call in kill.call_args_list))
                self.assertTrue(any(call.args[1] == signal.SIGKILL for call in kill.call_args_list))

    def test_duplicate_json_keys_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "bad.json"
            path.write_text('{"accepted":false,"accepted":true}')
            with self.assertRaisesRegex(ValueError, "duplicate"):
                m.read_json(path)

    def test_snapshot_marks_unfinished_invocation_nonresumable(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            r = m.Runner(root, root / "session")
            m.publish(r.control / "invocation-one-start.json", {})
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertFalse(r.snapshot()["resumable"])
                m.publish(r.control / "invocation-one-end.json", {"charged_seconds": 1})
                self.assertTrue(r.snapshot()["resumable"])

    def test_declared_probe_and_all_five_stages_fit_command_cap(self):
        plan = m.declared_command_reservation()
        self.assertEqual(plan["probe_seconds"], 78)
        self.assertEqual(plan["smoke_seconds"], 515)
        self.assertEqual(plan["total_seconds"], 593)
        self.assertLessEqual(plan["total_seconds"], plan["cap_seconds"])

    def test_no_numerical_imports_on_module_load(self):
        import sys
        self.assertNotIn("numpy", sys.modules)
        self.assertNotIn("torch", sys.modules)

    def test_deadline_expired_and_insufficient_cleanup(self):
        for now in (datetime(2026, 10, 16, 16, 30, tzinfo=timezone.utc),
                    datetime(2026, 10, 16, 16, 29, 58, tzinfo=timezone.utc)):
            with self.assertRaises(RuntimeError):
                m.deadline_check(now=now)
        m.deadline_check(60, datetime(2026, 10, 16, 16, 20, tzinfo=timezone.utc))

    def test_archive_traversal_duplicate_symlink_rejected(self):
        for name, symlink in (("../outside", False), ("/outside", False), ("link", True)):
            with self.subTest(name=name):
                data = io.BytesIO()
                with zipfile.ZipFile(data, "w") as z:
                    info = zipfile.ZipInfo(name)
                    if symlink:
                        info.external_attr = (stat.S_IFLNK | 0o777) << 16
                    z.writestr(info, "x")
                with zipfile.ZipFile(data) as z, self.assertRaises(RuntimeError):
                    m.safe_archive_members(z)

    def test_archive_inventory_and_idempotent_extract(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            archive = root / "a.zip"
            payload = b"hello"
            with zipfile.ZipFile(archive, "w") as z:
                z.writestr("tree/file.txt", payload)
                z.writestr("inventory.json", m.encoded({"tree/file.txt": m.sha(payload)}))
            digest = m.sha(archive.read_bytes())
            target = root / "tree"
            m.extract_verified(archive, target, digest, inventory_name="inventory.json")
            m.extract_verified(archive, target, digest, inventory_name="inventory.json")
            (target / "tree/file.txt").write_text("changed")
            with self.assertRaises(RuntimeError):
                m.extract_verified(archive, target, digest, inventory_name="inventory.json")

    def test_sealed_source_matches_and_modified_inventory_refused(self):
        sealed = HERE.parents[1] / "research/frem_s10_kaggle_20261016/source_bundle.zip"
        with tempfile.TemporaryDirectory() as d:
            dest = Path(d) / "source"
            m.extract_verified(sealed, dest, m.BUNDLE_SHA256, inventory_name="SOURCE_INVENTORY.json")
            m.verify_source(dest)
            inventory_path = dest / "SOURCE_INVENTORY.json"
            inventory = json.loads(inventory_path.read_bytes())
            name = "successors/frem_s10/src/frem_s10/cli.py"
            (dest / name).write_text("raise RuntimeError('tampered')")
            inventory[name] = m.sha((dest / name).read_bytes())
            inventory_path.write_bytes(m.encoded(inventory))
            with self.assertRaisesRegex(RuntimeError, "inventory bytes"):
                m.verify_source(dest)

    def test_exact_stage_semantics(self):
        valid = {"status": "INTERRUPTED", "completed": 3, "planned": 12, "scientific_status": "NOT_EVALUATED"}
        m.validate_stage_result("interrupt", valid)
        for bad in (dict(valid, completed=0), dict(valid, completed=4), dict(valid, planned=13), dict(valid, status="COMPLETE")):
            with self.assertRaises(RuntimeError):
                m.validate_stage_result("interrupt", bad)
        with self.assertRaises(RuntimeError):
            m.validate_stage_result("completed-reuse", {"status": "COMPLETE_REUSED", "completed": 12,
                                                        "planned": 12, "scientific_status": "NOT_EVALUATED"})

    def test_final_cli_json_after_live_logs(self):
        value = {"status": "COMPLETE", "nested": {"a": 1}}
        self.assertEqual(m.final_cli_result('progress {"step":1}\n' + json.dumps(value, indent=2) + '\n'), value)
        with self.assertRaises(RuntimeError):
            m.final_cli_result('{"status":"COMPLETE"}\ntrailing garbage')

    def test_failed_or_abandoned_commands_block_restart(self):
        with tempfile.TemporaryDirectory() as d:
            r = m.Runner(Path(d), Path(d) / "session")
            m.publish(r.control / "command-a-start.json", {"reserved_seconds": 10})
            with self.assertRaisesRegex(RuntimeError, "abandoned"):
                r.reserved()
            m.publish(r.control / "command-a-end.json", {"accepted": False})
            with self.assertRaisesRegex(RuntimeError, "prior failed"):
                r.reserved()

    def test_abandoned_wrapper_and_charges(self):
        with tempfile.TemporaryDirectory() as d:
            r = m.Runner(Path(d), Path(d) / "session")
            m.publish(r.control / "invocation-a-start.json", {})
            with self.assertRaisesRegex(RuntimeError, "abandoned wrapper"):
                r.wall_spent()
            m.publish(r.control / "invocation-a-end.json", {"charged_seconds": 35})
            m.publish(r.control / "inherited-reservation.json", {"seconds": 45, "wall_seconds": 120})
            self.assertEqual(r.wall_spent(), 215)
            self.assertEqual(r.reserved(), 45)

    def test_gpu_request_refused(self):
        with self.assertRaisesRegex(RuntimeError, "accelerator None"):
            m.cpu_only("T4x2")

    def test_receipt_requires_external_attestation(self):
        with self.assertRaises(RuntimeError):
            m.validate_receipt({}, helper_sha="x")
        with self.assertRaises(RuntimeError):
            m.attest(Path("unread"), "owner/slug/1", Path("unwritten"), False)

    def test_metadata_block_does_not_import_or_run_numerical_identity(self):
        with tempfile.TemporaryDirectory() as d:
            r = m.Runner(Path(d), Path(d) / "session")
            log = Path(d) / "preflight.json"
            m.publish(log, {"status": "BLOCKED", "reasons": ["PYTHON_3_12_REQUIRED"]})
            with patch.object(m, "compatibility_metadata", return_value={"status": "BLOCKED"}), patch.object(r, "run", return_value=log) as run:
                preflight, identity = r.metadata()
                self.assertEqual(preflight["status"], "BLOCKED")
                self.assertIsNone(identity)
                self.assertEqual(run.call_count, 0)

    def test_repeat_bootstrap_reservation_accumulates(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            source = root / "source"
            source.mkdir()
            m.publish(root / "bootstrap-first.json", {"reserved_seconds": 60})
            r = m.Runner(source, root / "session")
            self.assertEqual(r.wall_spent(), 60)
            m.publish(root / "bootstrap-second.json", {"reserved_seconds": 60})
            r = m.Runner(source, root / "session")
            self.assertEqual(r.wall_spent(), 120)

    def test_no_command_on_exhausted_cap(self):
        with tempfile.TemporaryDirectory() as d:
            r = m.Runner(Path(d), Path(d) / "session")
            with patch.object(m, "verify_source"), patch.object(m, "deadline_check"), patch.object(r, "reserved", return_value=599), patch.object(m.subprocess, "Popen") as popen:
                with self.assertRaises(RuntimeError):
                    r.run("dummy", ["preflight"], timeout=10)
                popen.assert_not_called()

    def test_real_stdlib_subprocess_streams_and_pairs_command_receipts(self):
        with tempfile.TemporaryDirectory() as d:
            r = m.Runner(Path(d), Path(d) / "session")
            output = io.StringIO()
            with patch.object(m, "verify_source"), patch.object(m, "deadline_check"), contextlib.redirect_stdout(output):
                log = r.run("stdlib-only", ["-c", "print('fixture progress', flush=True)"], timeout=2, raw=True)
            self.assertIn("fixture progress", output.getvalue())
            self.assertEqual(log.read_text(), "fixture progress\n")
            starts = list(r.control.glob("command-*-start.json"))
            self.assertEqual(len(starts), 1)
            self.assertTrue(starts[0].with_name(starts[0].name.replace("-start", "-end")).exists())
            self.assertEqual(r.reserved(), 7)

    def test_subprocess_timeout_is_failed_and_cannot_be_retried(self):
        with tempfile.TemporaryDirectory() as d:
            r = m.Runner(Path(d), Path(d) / "session")
            with patch.object(m, "verify_source"), patch.object(m, "deadline_check"), contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(m.subprocess.TimeoutExpired):
                    r.run("timeout-fixture", ["-c", "import time; time.sleep(5)"], timeout=0.02, raw=True)
            with self.assertRaisesRegex(RuntimeError, "prior failed"):
                r.reserved()

    def test_final_export_failure_cannot_claim_overall_completion(self):
        with tempfile.TemporaryDirectory() as d:
            r = m.Runner(Path(d), Path(d) / "session")
            with patch.object(r, "snapshot", side_effect=TimeoutError("fixture")):
                with self.assertRaises(TimeoutError):
                    r.finalize({"status": "GENERATED_CPU_STAGES_COMPLETE_WITH_FINAL_EXPORT_PENDING"})
            self.assertFalse((r.session / "FINALIZATION_RECEIPT.json").exists())

    def test_sigterm_cleans_up_separate_child_process_group(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            code = "\n".join([
                "import importlib.util, pathlib, signal",
                f"spec=importlib.util.spec_from_file_location('tonight', {str(HERE / 'runner.py')!r})",
                "m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)",
                "m.verify_source=lambda _: None; m.deadline_check=lambda *a: None",
                "signal.signal(signal.SIGTERM,m.termination_requested)",
                f"r=m.Runner(pathlib.Path({d!r}), pathlib.Path({str(root / 'session')!r}))",
                "r.run('signal-fixture',['-c', 'import os,time; print(os.getpid(),flush=True); time.sleep(10)'],timeout=12,raw=True)",
            ])
            with (root / "outer.log").open("wb") as output:
                parent = subprocess.Popen([sys.executable, "-B", "-c", code], stdout=output, stderr=subprocess.STDOUT)
                try:
                    limit = time.monotonic() + 3
                    pid = None
                    while time.monotonic() < limit:
                        logs = list((root / "session/control").glob("command-*.log"))
                        if logs and logs[0].read_text().strip():
                            pid = int(logs[0].read_text().strip())
                            break
                        time.sleep(0.02)
                    self.assertIsNotNone(pid)
                    parent.send_signal(signal.SIGTERM)
                    self.assertNotEqual(parent.wait(timeout=5), 0)
                    with self.assertRaises(ProcessLookupError):
                        os.kill(pid, 0)
                    end = list((root / "session/control").glob("command-*-end.json"))
                    self.assertEqual(len(end), 1)
                    self.assertFalse(json.loads(end[0].read_text())["accepted"])
                finally:
                    if parent.poll() is None:
                        if pid is not None:
                            try:
                                os.killpg(pid, signal.SIGTERM)
                            except ProcessLookupError:
                                pass
                        parent.kill()
                        parent.wait(timeout=2)


if __name__ == "__main__":
    unittest.main()
