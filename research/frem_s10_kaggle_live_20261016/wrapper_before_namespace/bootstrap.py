"""Prepare isolated, hash-pinned CPU dependencies without importing NumPy.

The notebook kernel may be Python 3.13. Model commands must use the returned
Python 3.12 executable. Never resolve its venv symlink to the base interpreter.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

UV_VERSION = "0.12.5"
DEADLINE = "2026-10-16T16:30:00+00:00"
SETUP_SECONDS = 300
METADATA_CODE = (
    "import sys,json,importlib.metadata; "
    "print(json.dumps({'python':list(sys.version_info[:2]),"
    "'numpy':importlib.metadata.version('numpy'),'prefix':sys.prefix}))"
)


def clean_environment(root):
    """Keep notebook paths and ambient uv settings out of the isolated runtime."""
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(('PYTHON', 'UV_', 'PIP_')) and k != 'VIRTUAL_ENV'}
    env.update(PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1',
               UV_CACHE_DIR=str(root / 'cache'), UV_PYTHON_INSTALL_DIR=str(root / 'python'),
               UV_PYTHON_BIN_DIR=str(root / 'bin'), UV_NO_CONFIG='1',
               UV_NO_PROGRESS='1', UV_LINK_MODE='copy')
    return env


def run_command(argv, root, env, end, index):
    """Bound the entire child group and retain failed setup logs for diagnosis."""
    seconds = min(end - time.monotonic(),
                  (datetime.fromisoformat(DEADLINE) - datetime.now(timezone.utc)).total_seconds() - 10)
    if seconds <= 0:
        raise TimeoutError('setup budget or campaign deadline exhausted')
    log = root / f'setup-{index:02}.log'
    with log.open('xb') as stream:
        child = subprocess.Popen(argv, env=env, cwd=root, stdout=stream,
                                 stderr=subprocess.STDOUT, start_new_session=True)
        try:
            rc = child.wait(timeout=seconds)
        except BaseException:
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                child.wait(timeout=2)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=2)
            raise
    if rc:
        raise RuntimeError(f'environment setup exited {rc}; inspect {log}')
    return log.read_text()


def verify_metadata(raw, venv):
    value = json.loads(raw)
    if (value.get('python') != [3, 12] or value.get('numpy') != '2.5.4'
            or Path(value.get('prefix', '')).absolute() != venv.absolute()):
        raise RuntimeError('isolated Python 3.12 / NumPy 2.5.4 required')
    return value


def prepare(root, requirements):
    """Create once, validate on reuse, and leave incomplete environments untouched."""
    if platform.system() != 'Linux':
        raise RuntimeError('this setup targets Linux Kaggle CPU sessions')
    if Path('/dev/nvidia0').exists():
        raise RuntimeError('select accelerator None before setup')
    if (datetime.fromisoformat(DEADLINE) - datetime.now(timezone.utc)).total_seconds() <= SETUP_SECONDS + 10:
        raise RuntimeError('deadline cannot fit bounded environment setup')
    root = root.absolute()
    if root.is_symlink():
        raise RuntimeError('runtime root must not be a symlink')
    root.mkdir(parents=True, exist_ok=True)
    venv = root / 'venv'
    python = venv / 'bin/python'  # Do NOT resolve: venv location controls site-packages.
    env = clean_environment(root)
    digest = hashlib.sha256(requirements.encode()).hexdigest()
    receipt = root / 'READY.json'
    if receipt.exists():
        saved = json.loads(receipt.read_text())
        if saved.get('requirements_sha256') != digest:
            raise RuntimeError('runtime requirements changed; preserve this environment')
        raw = subprocess.check_output([str(python), '-I', '-B', '-c', METADATA_CODE],
                                      env=env, cwd=root, text=True, timeout=10)
        verify_metadata(raw, venv)
        return python
    marker = root / 'STARTED.json'
    with marker.open('x') as stream:
        json.dump({'requirements_sha256': digest, 'setup_seconds': SETUP_SECONDS,
                   'started_utc': datetime.now(timezone.utc).isoformat()}, stream)
    requirement_path = root / 'requirements.txt'
    requirement_path.write_text(requirements)
    tools = root / 'tools'
    end = time.monotonic() + SETUP_SECONDS
    commands = [
        [sys.executable, '-I', '-m', 'pip', '--isolated', 'install', '--disable-pip-version-check',
         '--no-input', '--no-deps', '--only-binary=:all:', '--target', str(tools),
         '--index-url', 'https://pypi.org/simple', f'uv=={UV_VERSION}'],
        [str(tools / 'bin/uv'), 'venv', '--python', '3.12', '--managed-python', str(venv)],
        [str(tools / 'bin/uv'), 'pip', 'install', '--python', str(python),
         '--require-hashes', '--only-binary', ':all:', '--no-deps',
         '--index-url', 'https://pypi.org/simple', '-r', str(requirement_path)],
        [str(python), '-I', '-B', '-c', METADATA_CODE],
    ]
    try:
        raw = ''
        for index, command in enumerate(commands):
            raw = run_command(command, root, env, end, index)
        metadata = verify_metadata(raw, venv)
        with receipt.open('x') as stream:
            json.dump({'requirements_sha256': digest, 'runtime': metadata,
                       'uv_version': UV_VERSION, 'numerical_imports': False,
                       'elapsed_seconds': time.monotonic() - (end - SETUP_SECONDS)}, stream, indent=2)
    except BaseException as exc:
        with (root / 'FAILED.json').open('x') as stream:
            json.dump({'error_type': type(exc).__name__, 'reason': str(exc)}, stream)
        raise
    return python
