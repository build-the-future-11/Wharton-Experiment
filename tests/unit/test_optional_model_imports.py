"""The declared optional Torch backend stays optional for CPU accounting."""
import os
import subprocess
import sys
import unittest


class OptionalModelImportTests(unittest.TestCase):
    def test_accounting_and_linear_paths_work_without_torch(self):
        program = r'''
import importlib.abc
import sys
class NoTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise ModuleNotFoundError("Torch intentionally unavailable in this contract test")
sys.meta_path.insert(0, NoTorch())
from wharton_lab import models
from wharton_lab.models.m11.accounting import simulate_ledger
from wharton_lab.models import BaseModel, M11Model, M01RegimeDistributionalGBM
from wharton_lab.models.base import BaseModel as DirectBase
assert BaseModel is DirectBase
assert "torch" not in sys.modules
assert "FIJEPAModel" in dir(models)
assert models.M11Model is M11Model
try:
    models.not_a_model
except AttributeError:
    pass
else:
    raise AssertionError("unknown export must fail")
'''
        result = subprocess.run([sys.executable, "-c", program], env=os.environ.copy(),
                                text=True, capture_output=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
