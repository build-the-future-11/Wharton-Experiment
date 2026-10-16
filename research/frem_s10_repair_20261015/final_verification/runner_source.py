"""Explicit local engineering verification; never loads market sources."""
from pathlib import Path
import hashlib
import json
import os
import resource
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = HERE / 'final_verification'
OUT.mkdir(exist_ok=False)
PYTHON = ROOT / 'successors/frem_s10/.venv/bin/python'
ENV = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
           PYTEST_DISABLE_PLUGIN_AUTOLOAD='1', PYTHONDONTWRITEBYTECODE='1',
           UV_CACHE_DIR=str(ROOT/'research/frem_s10_implementation_20261015/uv_cache'))
ENV.pop('PYTHONPATH', None)
RESULTS = []

def run(name, cmd, cwd=ROOT, timeout=120):
    start = time.monotonic()
    result = subprocess.run(cmd, cwd=cwd, env=ENV, text=True, capture_output=True, timeout=timeout)
    log = OUT / (name + '.log')
    log.write_text(result.stdout + result.stderr)
    receipt = {'name': name, 'command': cmd, 'cwd': str(cwd), 'exit_code': result.returncode,
               'wall_seconds': time.monotonic() - start, 'log': str(log.relative_to(ROOT)),
               'log_sha256': hashlib.sha256(log.read_bytes()).hexdigest(),
               'cumulative_max_child_rss_bytes_macos': resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss}
    RESULTS.append(receipt)
    (OUT/'commands.json').write_text(json.dumps(RESULTS, indent=2)+'\n')
    print(name, result.returncode, result.stdout[-350:], flush=True)
    if result.returncode:
        raise SystemExit(result.returncode)

run('tests', [str(PYTHON), '-B', '-m', 'pytest', '-c', 'successors/frem_s10/pyproject.toml',
              'successors/frem_s10/tests', '-q', '-W', 'error', '-p', 'no:cacheprovider'], timeout=180)
run('lint', [str(PYTHON.parent/'ruff'), 'check', '--config', 'successors/frem_s10/pyproject.toml',
             'successors/frem_s10/src/frem_s10', 'successors/frem_s10/tests'])
run('format', [str(PYTHON.parent/'ruff'), 'format', '--check', '--config',
               'successors/frem_s10/pyproject.toml', 'successors/frem_s10/src/frem_s10', 'successors/frem_s10/tests'])
run('source_types', [str(PYTHON.parent/'mypy'), '--config-file', 'successors/frem_s10/pyproject.toml',
                    '--cache-dir', str(HERE/'mypy_cache'), 'successors/frem_s10/src/frem_s10', 'successors/frem_s10/tests'])
run('plan', [str(PYTHON.parent/'s10-research'), 'plan', '--manifest',
             'successors/frem_s10/src/frem_s10/science.json', '--output', str(OUT/'plan.json')])
run('build', [str(PYTHON), '-m', 'build', '--no-isolation', '--outdir', str(OUT/'dist'),
              'successors/frem_s10'])
run('clean_environment', ['uv', 'venv', '--python', str(PYTHON), str(OUT/'clean_env')])
wheel = next((OUT/'dist').glob('*.whl'))
run('wheel_install', ['uv', 'pip', 'install', '--python', str(OUT/'clean_env/bin/python'),
                      '--offline', str(wheel)])
smoke = "import sys,json,numpy; import frem_s10; import frem_s10.math.scores; from frem_s10.cli import plan; from pathlib import Path; p=Path(frem_s10.__file__); assert 'site-packages' in str(p); assert 'frem' not in sys.modules; print(json.dumps({'module':str(p),'numpy':numpy.__version__,'status':'PASS'}))"
run('wheel_smoke', [str(OUT/'clean_env/bin/python'), '-I', '-c', smoke], cwd=OUT)
run('wheel_help', [str(OUT/'clean_env/bin/s10-research'), '--help'], cwd=OUT)
print('ENGINEERING_VERIFICATION_COMPLETE_NO_MARKET_EXECUTION', flush=True)
