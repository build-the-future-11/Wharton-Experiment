import json, tempfile, math
from pathlib import Path
from unittest.mock import patch
from dataclasses import replace
from frem_s14.runtime import manifest, read_json, timestamp, admit
from frem_s14.intake import read_prices
from frem_s14.config import RunConfig
checks={}
with tempfile.TemporaryDirectory() as t:
 p=Path(t);(p/'fold').mkdir();(p/'fold/manifest.json').write_text('{}')
 checks['nested_manifest_bound']='fold/manifest.json' in manifest(p)['files']
 csv=p/'prices';csv.write_text('Date,Open,High,Low,Close,Volume,OpenInt\n2010-01-01,1,2,1,2,100,0,extra\n')
 try: read_prices(csv); checks['extra_csv_field_rejected']=False
 except ValueError: checks['extra_csv_field_rejected']=True
 j=p/'overflow.json';j.write_text('{"x": 1e999}')
 try: read_json(j);checks['overflow_json_rejected']=False
 except ValueError: checks['overflow_json_rejected']=True
try: replace(RunConfig(),protected_start='2026-01-01T00:00:00Z');checks['protected_boundary_enforced']=False
except ValueError: checks['protected_boundary_enforced']=True
checks['exact_microsecond_clock']=(timestamp('2500-01-01T00:00:00.000002Z')-timestamp('2500-01-01T00:00:00.000001Z'))==1
with patch('frem_s14.runtime.memory_snapshot',side_effect=[{'available_mib':600,'pageouts':1},{'available_mib':float('nan'),'pageouts':1},{'available_mib':600,'pageouts':1}]),patch('frem_s14.runtime.time.sleep'):
 checks['invalid_resource_sample_rejected']=not admit()['passed']
print(json.dumps(checks))
