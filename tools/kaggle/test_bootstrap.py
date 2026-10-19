"""Regression checks for the Python-3.13 notebook / Python-3.12 child split."""
import importlib.util
import json
import os
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location('bootstrap', HERE / 'bootstrap.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class Clock(datetime):
    @classmethod
    def now(cls, tz=None):
        return cls(2026, 10, 16, 8, tzinfo=timezone.utc)


class BootstrapTests(unittest.TestCase):
    def test_environment_removes_kernel_and_installer_overrides(self):
        with patch.dict(os.environ, {'PYTHONPATH': '/bad', 'PYTHONHOME': '/bad',
                                    'UV_INDEX': 'bad', 'PIP_INDEX_URL': 'bad', 'VIRTUAL_ENV': '/bad'}):
            env = m.clean_environment(Path('/fixture'))
        for key in ('PYTHONPATH', 'PYTHONHOME', 'UV_INDEX', 'PIP_INDEX_URL', 'VIRTUAL_ENV'):
            self.assertNotIn(key, env)
        self.assertEqual(env['UV_PYTHON_INSTALL_DIR'], '/fixture/python')

    def test_metadata_rejects_notebook_runtime_and_wrong_venv(self):
        for value in ({'python': [3, 13], 'numpy': '2.5.4', 'prefix': '/v'},
                      {'python': [3, 12], 'numpy': '2.1.3', 'prefix': '/v'},
                      {'python': [3, 12], 'numpy': '2.5.4', 'prefix': '/base'}):
            with self.assertRaises(RuntimeError):
                m.verify_metadata(json.dumps(value), Path('/v'))

    def test_setup_selects_isolated_interpreter_and_pinned_hashes(self):
        with tempfile.TemporaryDirectory() as d, patch.object(m.platform, 'system', return_value='Linux'), patch.object(m, 'datetime', Clock):
            root = Path(d)
            metadata = json.dumps({'python': [3, 12], 'numpy': '2.5.4', 'prefix': str(root / 'venv')})
            with patch.object(m, 'run_command', return_value=metadata) as run:
                python = m.prepare(root, 'fixture requirements')
            self.assertEqual(python, root / 'venv/bin/python')
            commands = [c.args[0] for c in run.call_args_list]
            self.assertEqual(len(commands), 4)
            self.assertIn('--managed-python', commands[1])
            self.assertIn('--require-hashes', commands[2])
            self.assertIn(str(python), commands[2])
            self.assertEqual(commands[3][0], str(python))
            self.assertFalse(json.loads((root / 'READY.json').read_text())['numerical_imports'])
            with patch.object(m.subprocess, 'check_output', return_value=metadata), patch.object(m, 'run_command') as run:
                self.assertEqual(m.prepare(root, 'fixture requirements'), python)
                run.assert_not_called()
            with self.assertRaisesRegex(RuntimeError, 'requirements changed'):
                m.prepare(root, 'different')

    def test_failure_preserved_and_partial_setup_never_overwritten(self):
        with tempfile.TemporaryDirectory() as d, patch.object(m.platform, 'system', return_value='Linux'), patch.object(m, 'datetime', Clock):
            with patch.object(m, 'run_command', side_effect=RuntimeError('fixture network failure')):
                with self.assertRaises(RuntimeError):
                    m.prepare(Path(d), 'fixture')
            self.assertTrue((Path(d) / 'FAILED.json').exists())
            self.assertFalse((Path(d) / 'READY.json').exists())
            with patch.object(m, 'run_command') as run, self.assertRaises(FileExistsError):
                m.prepare(Path(d), 'fixture')
            run.assert_not_called()

    def test_expired_setup_stops_before_any_write(self):
        class Expired(Clock):
            @classmethod
            def now(cls, tz=None):
                return cls(2026, 10, 17, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as d, patch.object(m.platform, 'system', return_value='Linux'), patch.object(m, 'datetime', Expired):
            root = Path(d) / 'new'
            with self.assertRaisesRegex(RuntimeError, 'deadline'):
                m.prepare(root, 'fixture')
            self.assertFalse(root.exists())

    def test_no_numerical_import_at_bootstrap(self):
        import sys
        self.assertNotIn('numpy', sys.modules)
        self.assertNotIn('torch', sys.modules)


if __name__ == '__main__':
    unittest.main()
