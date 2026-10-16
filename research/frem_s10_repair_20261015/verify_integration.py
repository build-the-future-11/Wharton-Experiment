"""Explicit local engineering verification; never loads market sources."""
from pathlib import Path
import hashlib
import json
import os
import resource
import subprocess
import time
import tarfile
import zipfile

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = HERE / 'integrated_verification_attempt2'
OUT.mkdir(exist_ok=False)
(OUT/'runner_source.py').write_text(Path(__file__).read_text())
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
              'successors/frem_s10/tests', '-q', '-W', 'error', '-p', 'no:cacheprovider', '--basetemp', str(OUT/'pytest_artifacts')], timeout=300)
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
numpy_wheel = HERE/'package_verification_attempt4/numpy-2.5.4-cp312-cp312-macosx_14_0_arm64.whl'
assert numpy_wheel.is_file()
assert hashlib.sha256(numpy_wheel.read_bytes()).hexdigest() == '9cb18a327b49c5c337f972b03682f6a49855525faaf3c0d3e9c96cd0fd8880a8'
requirements = OUT/'requirements.txt'
requirements.write_text(f"{wheel.resolve().as_uri()} --hash=sha256:{hashlib.sha256(wheel.read_bytes()).hexdigest()}\n{numpy_wheel.resolve().as_uri()} --hash=sha256:{hashlib.sha256(numpy_wheel.read_bytes()).hexdigest()}\n")
run('wheel_install', ['uv', 'pip', 'install', '--python', str(OUT/'clean_env/bin/python'),
                      '--offline', '--no-index', '--require-hashes', '-r', str(requirements)])
smoke = "import sys,json,numpy; import frem_s10; import frem_s10.experiment.precision; import frem_s10.experiment.artifacts; from importlib.resources import files; p=files('frem_s10'); s=json.loads(p.joinpath('science.json').read_text()); assert s['design_id']=='FREM-S10-DESIGN-4-CL16'; assert 'site-packages' in str(p); assert 'frem' not in sys.modules; print(json.dumps({'module':str(p),'numpy':numpy.__version__,'design_id':s['design_id'],'status':'PASS'}))"
run('wheel_smoke', [str(OUT/'clean_env/bin/python'), '-I', '-c', smoke], cwd=OUT)
run('wheel_help', [str(OUT/'clean_env/bin/s10-research'), '--help'], cwd=OUT)
run('wheel_plan', [str(OUT/'clean_env/bin/s10-research'), 'plan', '--manifest', str(ROOT/'successors/frem_s10/src/frem_s10/science.json'), '--output', str(OUT/'wheel_plan.json')], cwd=OUT)
source_dir = ROOT/'successors/frem_s10/src'
inspection = []
with zipfile.ZipFile(wheel) as archive:
    for name in archive.namelist():
        assert not name.startswith('/') and '..' not in Path(name).parts
        assert not ({'.venv', '.env', 'research', 'outputs'} & set(Path(name).parts))
        assert Path(name).name not in {'SPY.json', 'TLT.json', 'GLD.json'}
        assert (archive.getinfo(name).external_attr >> 16) & 0o170000 != 0o120000
        if name.startswith('frem_s10/') and not name.endswith('/'):
            source = source_dir/name
            assert source.is_file() and source.read_bytes() == archive.read(name), name
            inspection.append({'path':name,'sha256':hashlib.sha256(archive.read(name)).hexdigest()})
with tarfile.open(next((OUT/'dist').glob('*.tar.gz'))) as archive:
    for member in archive.getmembers():
        assert not member.issym() and not member.islnk() and not member.name.startswith('/') and '..' not in Path(member.name).parts
        assert not ({'.venv', '.env', 'research', 'outputs'} & set(Path(member.name).parts))
        assert Path(member.name).name not in {'SPY.json', 'TLT.json', 'GLD.json'}
(OUT/'archive_inspection.json').write_text(json.dumps({'status':'PASS','wheel_source_payloads':inspection,'wheel_sha256':hashlib.sha256(wheel.read_bytes()).hexdigest()},indent=2)+'\n')
print('ENGINEERING_VERIFICATION_COMPLETE_NO_MARKET_EXECUTION', flush=True)
