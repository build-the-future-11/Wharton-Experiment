"""Read-only source and retained-history checks; no market value calculation."""
from pathlib import Path
import ast
import hashlib
import json

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
BASE = ROOT/'successors/frem_s10/src'
files = sorted((BASE/'frem_s10').rglob('*.py'))
modules = {str(p.relative_to(BASE)).removesuffix('.py').replace('/', '.').removesuffix('.__init__'): p for p in files}
edges = {m: set() for m in modules}
violations = []
for module, path in modules.items():
    for node in ast.walk(ast.parse(path.read_text())):
        imports = [alias.name for alias in node.names] if isinstance(node, ast.Import) else [node.module or ''] if isinstance(node, ast.ImportFrom) else []
        for target in imports:
            if target.split('.')[0] in {'frem', 'torch', 'jax', 'tensorflow'} or (module.startswith('frem_s10.analysis') and target.startswith(('frem_s10.model', 'frem_s10.experiment', 'frem_s10.data.source_reader'))):
                violations.append({'file': str(path.relative_to(ROOT)), 'import': target})
            if target in modules:
                edges[module].add(target)
cycles = []
visited = set()
active = []
def walk(module):
    if module in active:
        cycles.append(active[active.index(module):] + [module]); return
    if module in visited: return
    active.append(module)
    for target in sorted(edges[module]): walk(target)
    active.pop(); visited.add(module)
for module in modules: walk(module)
result = {'status': 'PASS' if not violations and not cycles else 'FAIL', 'source_files':len(files), 'prohibited_imports': violations, 'cycles': cycles, 'scope':'static import graph; not behavioral or empirical validation'}
(HERE/'import_boundary_final.json').write_text(json.dumps(result, indent=2)+'\n')
assert result['status'] == 'PASS', result
old = json.loads((HERE/'predecessor_check_corrected.json').read_text())
records = []
for item in old['records']:
    checked = item.get('archived_path', item['path'])
    actual = hashlib.sha256((ROOT/checked).read_bytes()).hexdigest()
    records.append({'manifest':item['manifest'], 'path':item['path'], 'checked_path': checked, 'sha256':actual, 'expected':item['expected'], 'match':actual==item['expected']})
prefix = (HERE/'before/research/events.jsonl').read_bytes()
prefix_preserved = (ROOT/'research/events.jsonl').read_bytes().startswith(prefix)
result = {'checked':len(records),'mismatches':[r for r in records if not r['match']], 'journal_before_prefix_preserved':prefix_preserved,'records':records}
(HERE/'predecessor_check_final.json').write_text(json.dumps(result, indent=2)+'\n')
assert not result['mismatches'] and prefix_preserved
print(json.dumps({'import_graph':'PASS','source_files':len(files),'historical_identities':len(records),'journal_prefix_preserved':prefix_preserved}))
