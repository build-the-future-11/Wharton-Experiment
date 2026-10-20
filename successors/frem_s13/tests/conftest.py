"""Numerical tests require fresh admission before importing test modules."""

import os

from frem_s13.runtime import admit

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"


def pytest_sessionstart(session):
    if all("test_stdlib.py" in x for x in session.config.args):
        return
    receipt = admit()
    if not receipt["passed"]:
        import pytest

        pytest.exit(f"Numerical test admission failed: {receipt}", returncode=2)
