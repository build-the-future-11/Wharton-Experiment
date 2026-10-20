"""Executable low-memory contracts; these do not import a scientific library."""

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from frem_s14 import __version__
from frem_s14.cli import main
from frem_s14.config import ModelConfig, RunConfig
from frem_s14.runtime import admit, identity, manifest, read_json, timestamp, verify, write_json


class Contracts(unittest.TestCase):
    def test_exported_version_and_plan_cli_match_declared_suite_wrapper(self):
        self.assertEqual(__version__, "0.1.6")
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "suite.json"
            path.write_text(json.dumps({"config": RunConfig().to_dict(), "fold_ends": []}))
            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(main(["plan", "--config", str(path)]), 0)
            plan = json.loads(output.getvalue())
            self.assertEqual(plan["configuration"], json.loads(json.dumps(RunConfig().to_dict())))
            self.assertEqual(plan["cells"], len(RunConfig().arms) * len(RunConfig().seeds))

    def test_json_rejects_ambiguous_identity(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "ambiguous.json"
            for content in ('{"x": 1, "x": 2}', '{"x": NaN}', '{"x": Infinity}'):
                path.write_text(content)
                with self.assertRaises(ValueError):
                    read_json(path)

    def test_roundtrip(self):
        cfg = RunConfig()
        self.assertEqual(
            identity(cfg.to_dict()),
            identity(RunConfig.from_dict(json.loads(json.dumps(cfg.to_dict()))).to_dict()),
        )

    def test_unknown_key_rejected(self):
        with self.assertRaises(TypeError):
            RunConfig.from_dict({"typo": 12})

    def test_dimensions_rejected(self):
        for field in ("input_dim", "context", "horizon", "latent", "reservoir", "experts"):
            for value in (0, -1, True, 1.5):
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    replace(ModelConfig(), **{field: value})

    def test_numerical_config_bounds(self):
        for kwargs in (
            {"contraction": 1},
            {"ema": 0},
            {"student_df": 2},
            {"router_cap": 0.1},
            {"memory_temperature": 0},
            {"growth_threshold": float("nan")},
            {"leaks": [0.2, float("nan")]},
            {"initial_neurons": 999},
            {"bands": 99},
            {"arm": "invented"},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                replace(ModelConfig(), **kwargs)

    def test_run_bounds(self):
        for kwargs in (
            {"seeds": [11, 11]},
            {"arms": ["full", "full"]},
            {"device": "mps"},
            {"device": "cpu", "amp": True},
            {"cost_bps": 10000},
            {"max_weight": 0},
            {"train_fraction": 0.9},
            {"batch_size": 1},
            {"learning_rate": float("nan")},
            {"protected_start": "2025-01-01"},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                replace(RunConfig(), **kwargs)

    def test_timezone_required_and_equivalent(self):
        with self.assertRaises(ValueError):
            timestamp("2025-01-01T00:00:00")
        self.assertEqual(timestamp("2025-01-01T00:00:00Z"), timestamp("2025-01-01T05:30:00+05:30"))

    def test_no_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "x.json"
            write_json(path, {"x": 1})
            with self.assertRaises(FileExistsError):
                write_json(path, {"x": 2})
            self.assertEqual(json.loads(path.read_text()), {"x": 1})

    def test_manifest_tampering(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_json(root / "result.json", {"x": 1})
            write_json(root / "manifest.json", manifest(root))
            self.assertTrue(verify(root)["verified"])
            (root / "result.json").write_text("{}")
            with self.assertRaises(ValueError):
                verify(root)

    def test_manifest_extra_file_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_json(root / "manifest.json", manifest(root))
            (root / "extra").write_text("x")
            with self.assertRaises(ValueError):
                verify(root)

    def test_manifest_symlink_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "link").symlink_to("/tmp")
            with self.assertRaises(ValueError):
                manifest(root)

    def test_resource_admission(self):
        cases = [
            ([600, 600, 600], [1, 1, 1], True),
            ([600, 511, 600], [1, 1, 1], False),
            ([600, 600, 600], [1, 1, 2], False),
        ]
        for available, pages, passed in cases:
            snapshots = [
                {"available_mib": a, "pageouts": p} for a, p in zip(available, pages, strict=True)
            ]
            with (
                patch("frem_s14.runtime.memory_snapshot", side_effect=snapshots),
                patch("frem_s14.runtime.time.sleep"),
            ):
                self.assertEqual(admit()["passed"], passed)

    def test_blocked_campaign_retains_failure_before_import(self):
        from frem_s14.cli import campaign, qualification_config

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "blocked"
            with patch("frem_s14.cli.admit", return_value={"passed": False}):
                with self.assertRaises(RuntimeError):
                    campaign(root, qualification_config("cpu"))
            self.assertEqual(json.loads((root / "failure.json").read_text())["complete_cells"], [])
            self.assertTrue(verify(root)["verified"])


if __name__ == "__main__":
    unittest.main()
