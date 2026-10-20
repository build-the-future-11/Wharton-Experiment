"""Real lightweight child-process tests; no numerical libraries or market data."""

import hashlib
import io
import json
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from run_stage import _publish, run_stage


class ProcessRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.output = self.root / "stage"

    def run_python(self, code, **kwargs):
        return run_stage(
            [sys.executable, "-B", "-c", code],
            self.output,
            kwargs.pop("max_seconds", 5),
            echo=False,
            **kwargs,
        )

    def test_success_exact_binary_log_and_receipt(self):
        result = self.run_python("import os;os.write(1,b'one\\n\\xff');os.write(2,b'two\\n')")
        log = (self.output / "stage.log").read_bytes()
        self.assertEqual(log, b"one\n\xfftwo\n")
        self.assertEqual(result["status"], "COMPLETE")
        self.assertEqual(result["returncode"], 0)
        self.assertEqual(result["log_sha256"], hashlib.sha256(log).hexdigest())
        self.assertEqual(result["log_bytes"], len(log))
        self.assertFalse(result["scientific_qualification"])
        self.assertEqual(json.loads((self.output / "finished.json").read_text()), result)

    def test_nonzero_failure_retains_output(self):
        result = self.run_python("import sys;print('failure preserved');sys.exit(7)")
        self.assertEqual((result["status"], result["returncode"]), ("FAILED", 7))
        self.assertIn(b"failure preserved", (self.output / "stage.log").read_bytes())

    def test_missing_executable_is_recorded(self):
        result = run_stage([str(self.root / "missing")], self.output, 1, echo=False)
        self.assertEqual(result["status"], "LAUNCH_FAILED")
        self.assertIsNone(result["returncode"])
        self.assertIn("FileNotFoundError", result["error"])
        self.assertTrue((self.output / "finished.json").exists())

    def test_deadline_kills_child_and_preserves_partial_log(self):
        # The test exercises a real interpreter under a loaded host; allow it
        # enough time to reach its first write before testing deadline cleanup.
        result = self.run_python("import time;print('partial');time.sleep(10)", max_seconds=2)
        self.assertEqual(result["status"], "TIMED_OUT")
        self.assertLess(result["termination_wall_seconds"], 3)
        self.assertIn(b"partial", (self.output / "stage.log").read_bytes())
        self.assertLess(result["returncode"], 0)

    def test_total_deadline_shortens_stage_budget(self):
        result = self.run_python("import time;time.sleep(10)", deadline=time.monotonic() + 2)
        self.assertEqual(result["status"], "TIMED_OUT")
        self.assertLessEqual(result["effective_seconds"], 2)
        self.assertLess(result["termination_wall_seconds"], 3)

    def test_exhausted_budget_does_not_start_command(self):
        marker = self.root / "must_not_exist"
        result = self.run_python(
            f"open({str(marker)!r},'w').close()", deadline=time.monotonic() - 1
        )
        self.assertEqual(result["status"], "NOT_STARTED_BUDGET_EXHAUSTED")
        self.assertFalse(marker.exists())
        self.assertIsNone(result["returncode"])

    def test_success_cannot_extend_global_deadline(self):
        result = self.run_python("print('ok')", deadline=time.monotonic() + 20, max_seconds=1)
        self.assertEqual(result["status"], "COMPLETE")
        self.assertLessEqual(result["effective_seconds"], 1)

    def test_existing_evidence_is_never_replaced(self):
        self.run_python("print('original')")
        before = {p.name: p.read_bytes() for p in self.output.iterdir()}
        with self.assertRaises(FileExistsError):
            self.run_python("print('replacement')")
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.output.iterdir()})

    def test_invalid_arguments_create_no_evidence(self):
        bad = [
            ([], 1, None),
            ("echo", 1, None),
            ([None], 1, None),
            ([""], 1, None),
            (["x\x00"], 1, None),
            (["x"], True, None),
            (["x"], 0, None),
            (["x"], float("nan"), None),
            (["x"], float("inf"), None),
            (["x"], 1, float("nan")),
        ]
        for command, seconds, deadline in bad:
            with self.subTest(command=command, seconds=seconds, deadline=deadline):
                with self.assertRaises(ValueError):
                    run_stage(command, self.output, seconds, deadline=deadline, echo=False)
                self.assertFalse(self.output.exists())

    def test_command_is_argv_not_shell(self):
        marker = self.root / "injected"
        argument = f"$(touch {marker}); a b"
        result = run_stage(
            [sys.executable, "-c", "import sys;print(sys.argv[1])", argument],
            self.output,
            5,
            echo=False,
        )
        self.assertEqual(result["status"], "COMPLETE")
        self.assertEqual((self.output / "stage.log").read_text(), argument + "\n")
        self.assertFalse(marker.exists())

    def test_cwd_and_eof_stdin(self):
        result = self.run_python(
            "import os,sys;print(os.getcwd());print(repr(sys.stdin.read()))", cwd=self.root
        )
        self.assertEqual(result["status"], "COMPLETE")
        self.assertEqual(
            (self.output / "stage.log").read_text(), str(self.root.resolve()) + "\n''\n"
        )

    def test_output_and_heartbeat_exist_before_process_finishes(self):
        results = []
        worker = threading.Thread(
            target=lambda: results.append(
                self.run_python("import time;print('live');time.sleep(0.6)", heartbeat_seconds=0.05)
            )
        )
        worker.start()
        self.addCleanup(worker.join, 5)
        until = time.monotonic() + 3
        while time.monotonic() < until:
            log, events = self.output / "stage.log", self.output / "events.jsonl"
            if log.exists() and b"live" in log.read_bytes() and '"RUNNING"' in events.read_text():
                break
            time.sleep(0.01)
        self.assertTrue(worker.is_alive())
        self.assertFalse((self.output / "finished.json").exists())
        worker.join(5)
        self.assertEqual(results[0]["status"], "COMPLETE")

    def test_no_output_child_still_has_heartbeat(self):
        result = self.run_python("import time;time.sleep(0.25)", heartbeat_seconds=0.05)
        self.assertEqual(result["status"], "COMPLETE")
        events = [
            json.loads(line) for line in (self.output / "events.jsonl").read_text().splitlines()
        ]
        self.assertEqual(events[0]["event"], "STARTED")
        self.assertEqual(events[-1]["event"], "FINISHED")
        self.assertIn("RUNNING", [e["event"] for e in events])

    def descendant_probe(self, timeout):
        marker = self.root / "child_alive"
        child = (
            "import time\n"
            f"with open({str(marker)!r},'ab',buffering=0) as f:\n"
            " for _ in range(250):\n  f.write(b'x');time.sleep(0.02)\n"
        )
        code = (
            "import subprocess,sys,time\nfrom pathlib import Path\n"
            f"subprocess.Popen([sys.executable,'-c',{child!r}])\n"
            f"while not Path({str(marker)!r}).exists():time.sleep(0.005)\n"
            + ("time.sleep(10)\n" if timeout else "")
        )
        result = self.run_python(code, max_seconds=5 if timeout else 5)
        self.assertEqual(result["status"], "TIMED_OUT" if timeout else "COMPLETE")
        self.assertTrue(marker.exists(), "descendant must start before termination is tested")
        before = marker.read_bytes()
        time.sleep(0.15)
        self.assertEqual(marker.read_bytes(), before, "background descendant kept executing")

    def test_background_descendant_is_terminated(self):
        self.descendant_probe(False)

    def test_timeout_terminates_descendants(self):
        self.descendant_probe(True)

    def test_broken_console_does_not_lose_evidence(self):
        class BrokenConsole(io.StringIO):
            def write(self, value):
                raise BrokenPipeError("closed viewer")

        with patch("sys.stdout", BrokenConsole()):
            result = run_stage([sys.executable, "-c", "print('retained')"], self.output, 5)
        self.assertEqual(result["status"], "COMPLETE")
        self.assertIn("BrokenPipeError", result["console_error"])
        self.assertEqual((self.output / "stage.log").read_text(), "retained\n")

    def test_atomic_publication_does_not_replace_existing(self):
        path = self.root / "receipt.json"
        _publish(path, {"original": 1})
        with self.assertRaises(FileExistsError):
            _publish(path, {"replacement": 2})
        self.assertEqual(json.loads(path.read_text()), {"original": 1})
        self.assertTrue(path.with_suffix(".json.pending").exists())

    def test_cli_retains_nonzero_exit(self):
        result = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).with_name("run_stage.py")),
                "--output",
                str(self.output),
                "--max-seconds",
                "5",
                "--quiet",
                "--",
                sys.executable,
                "-c",
                "raise SystemExit(3)",
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertEqual(json.loads((self.output / "finished.json").read_text())["returncode"], 3)

    def test_large_output_is_drained_without_truncation(self):
        result = self.run_python("import os;os.write(1,b'z'*1048576)")
        self.assertEqual(result["status"], "COMPLETE")
        self.assertEqual((self.output / "stage.log").read_bytes(), b"z" * 1048576)
        self.assertEqual(result["log_bytes"], 1048576)

    def test_child_can_close_output_before_exiting(self):
        result = self.run_python("import os,time;os.close(1);os.close(2);time.sleep(0.15)")
        self.assertEqual(result["status"], "COMPLETE")
        self.assertEqual(result["log_bytes"], 0)

    def test_interrupt_records_failure_and_stops_child(self):
        runner = subprocess.Popen(
            [
                sys.executable,
                str(Path(__file__).with_name("run_stage.py")),
                "--output",
                str(self.output),
                "--max-seconds",
                "10",
                "--quiet",
                "--",
                sys.executable,
                "-c",
                "import time;print('started');time.sleep(10)",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        try:
            until = time.monotonic() + 3
            log = self.output / "stage.log"
            while time.monotonic() < until:
                if log.exists() and b"started" in log.read_bytes():
                    break
                time.sleep(0.01)
            self.assertTrue(log.exists() and b"started" in log.read_bytes())
            runner.send_signal(signal.SIGINT)
            self.assertEqual(runner.wait(timeout=3), 1)
            result = json.loads((self.output / "finished.json").read_text())
            self.assertEqual(result["status"], "INTERRUPTED")
            self.assertEqual(result["error"], "KeyboardInterrupt")
            self.assertLess(result["returncode"], 0)
        finally:
            if runner.poll() is None:
                runner.send_signal(signal.SIGINT)
                runner.wait(timeout=3)


if __name__ == "__main__":
    unittest.main()
