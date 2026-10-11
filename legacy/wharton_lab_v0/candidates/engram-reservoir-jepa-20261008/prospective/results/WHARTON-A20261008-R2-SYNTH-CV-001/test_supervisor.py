"""Non-scientific controls, no model imports or generated data."""
import json, pathlib, sys, time
from supervisor import supervise
root=pathlib.Path(__file__).resolve().parent
batch=root/('guard_tests_'+str(time.time_ns()))
cases=[
 ('normal', 'print("normal child exited")',3,100*1024*1024,None),
 ('wall', 'import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(30)',.6,100*1024*1024,'wall_limit'),
 ('rss', 'import time;payload=bytearray(80*1024*1024);time.sleep(30)',5,50*1024*1024,'rss_limit'),
 ('error', 'raise RuntimeError("deliberate guard fixture")',3,100*1024*1024,'worker_error'),
]
results=[]
for name,code,wall,rss,expected in cases:
    result=supervise([sys.executable,'-c',code],root,batch/name,wall,rss,{'PYTHONDONTWRITEBYTECODE':'1'})
    assert result['stop_reason']==expected,(name,result)
    assert not result['remaining_group_pids']
    assert not result['signal_errors']
    assert result['cleanup_seconds']<5
    assert result['wall_seconds']<wall+3
    if name=='wall':assert result['exit_code']==-9
    if name=='rss':assert result['sampled_peak_group_rss_bytes']>rss
    if len(result['rss_samples'])>1:assert max(b['elapsed_seconds']-a['elapsed_seconds'] for a,b in zip(result['rss_samples'],result['rss_samples'][1:]))<=1
    results.append({k:v for k,v in result.items() if k!='rss_samples'})
import hashlib
receipt={'status':'PASS','supervisor_sha256':hashlib.sha256((root/'supervisor.py').read_bytes()).hexdigest(),'test_script_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'tests':results,'scientific_data_generated':False,'model_imported':False}
(root/'GUARD_TEST_RESULTS.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2))
