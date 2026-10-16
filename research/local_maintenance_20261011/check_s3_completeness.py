"""One-off stdlib check of the completed, frozen S3 matrix; no training or selection."""
import ast
import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'outputs/frem_s3_20261011_attempt1'
OUT = Path(__file__).with_name('s3_completeness.json')
constants = {}
for node in ast.parse((ROOT / 'scripts/run_frem_s3.py').read_text()).body:
    if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
        if node.targets[0].id in {'FAMILIES', 'PATH_SEEDS', 'SEEDS'}:
            constants[node.targets[0].id] = ast.literal_eval(node.value)
arms = {'off', 'frozen', 'recent', 'random', 'mean', 'risk', 'combined', 'no_confidence',
        'balanced', 'larger', 'dual', 'cross_asset', 'learned', 'utility'}
config = json.loads((ROOT / 'configs/frem_s3_public.json').read_text())
completion = json.loads((RUN / 'completion.json').read_text())
assert completion['status'] == 'COMPLETE_EXPLORATORY'
assert completion['market_learned_fits'] == 24 and completion['constant_cells'] == 12
public = {(f['name'], s, m) for f in config['folds'] for s in constants['SEEDS']
          for m in [*config['models'], 'zero', 'historical_mean']}
mechanism = {(f, s) for f in constants['FAMILIES'] for s in constants['PATH_SEEDS']}
assert {tuple(p.parts[-4:-2]) + (p.parts[-2],) for p in RUN.glob('public/*/*/*/result.json')} == {
    (f, str(s), m) for f, s, m in public}
assert {(p.parts[-3], int(p.parts[-2])) for p in RUN.glob('mechanism/*/*/result.json')} == mechanism
references = {}
files = rows_checked = 0
scores = {}
for kind, cells in [('public', public), ('mechanism', mechanism)]:
    for cell in sorted(cells):
        directory = RUN.joinpath(kind, *map(str, cell))
        receipt = json.loads((directory / 'result.json').read_text())
        process = json.loads((directory / 'process.json').read_text())
        assert process['returncode'] == 0 and process['termination'] is None
        expected = arms if kind == 'mechanism' or cell[-1] == 'frem' else {'off'}
        actual = set(receipt['scores']) - {'oracle'} if kind == 'mechanism' else set(receipt['results'])
        assert actual == expected
        for arm in sorted(expected):
            path = directory / f'{arm}.csv'
            with path.open() as stream:
                rows = list(csv.DictReader(stream))
            assert rows
            identity = {(int(r['time']), r['asset']): (float(r['actual']), int(r['label_time']), int(r['asset_id']))
                        for r in rows}
            assert len(identity) == len(rows)
            assert all(math.isfinite(float(r['nll'])) for r in rows)
            assert all(int(r['time']) < int(r['label_time']) for r in rows)
            group = (kind, cell[0]) if kind == 'public' else (kind, *cell)
            if group in references:
                assert identity == references[group]
            references[group] = identity
            by_date = {}
            for row in rows:
                by_date.setdefault(row['time'], []).append(float(row['nll']))
            score = math.fsum(math.fsum(v) / len(v) for v in by_date.values()) / len(by_date)
            if kind == 'public':
                split = json.loads((directory / 'split.json').read_text())
                assert len(rows) == split['row_counts'][2] == receipt['results'][arm]['rows']
                assert split['max_training_label_ns'] < split['validation_start_ns']
                assert split['max_validation_label_ns'] < split['evaluation_decision_start_ns']
                assert all(int(r['label_time']) < 1735689600000000000 for r in rows)
                assert {int(r['seed']) for r in rows} == {cell[1]}
                assert {r['model'] for r in rows} == {cell[2]}
                assert {r['arm'] for r in rows} == {arm}
                expected_score = receipt['results'][arm]['nll']
                scores.setdefault((cell[0], cell[2] + '/' + arm), []).append(score)
            else:
                expected_score = receipt['scores'][arm]
            assert math.isclose(score, expected_score, rel_tol=0, abs_tol=1e-10)
            files += 1
            rows_checked += len(rows)
summary = json.loads((RUN / 'public_summary.json').read_text())
for (fold, arm), values in scores.items():
    assert len(values) == len(constants['SEEDS'])
    assert math.isclose(math.fsum(values) / len(values), summary['comparisons'][fold]['nll'][arm],
                        rel_tol=0, abs_tol=1e-10)
assert summary['base_cells'] == len(public)
result = {'status':'PASS_COMPLETE_PAIRED_S3_MATRIX','public_base_cells':len(public),
          'mechanism_cells':len(mechanism),'variant_prediction_files':files,
          'prediction_rows':rows_checked,'complete_seeds_and_arms':True,'finite_saved_nll':True,
          'public_labels_before_2025':True,'paired_outcomes_and_purges_verified':True,
          'receipt_and_summary_scores_recomputed':True,'new_fits':0,
          'scope':'Completed exploratory evidence checks; no bootstrap repetition or scientific promotion'}
with OUT.open('x') as stream:
    json.dump(result,stream,indent=2,allow_nan=False)
    stream.write('\n')
print(json.dumps(result))
