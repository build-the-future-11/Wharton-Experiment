from pathlib import Path
from datetime import datetime, timezone
import json, os, platform, shutil, subprocess, hashlib
out=Path('/kaggle/working/s10-production-capability.json')
if out.exists(): raise RuntimeError('preserve prior capability receipt')
receipt={'schema':'s10-kaggle-production-capability-v1','observed_utc':datetime.now(timezone.utc).isoformat(),'platform':platform.platform(),'machine':platform.machine(),'numerical_imports':False,'market_data_access':False,'scientific_attempt_consumed':False}
receipt['tools']={name:shutil.which(name) for name in ('bwrap','unshare','uv','ps')}
receipt['namespace']={name:os.readlink('/proc/self/ns/'+name) for name in ('user','mnt','net','cgroup')}
receipt['resources']={p:Path(p).read_text() for p in ('/proc/self/cgroup','/sys/fs/cgroup/memory.max','/sys/fs/cgroup/memory.current','/sys/fs/cgroup/memory.events')}
receipt['capabilities']=[line for line in Path('/proc/self/status').read_text().splitlines() if line.startswith(('Cap','NoNewPrivs','Seccomp'))]
commands=[]
if receipt['tools']['unshare']:
    commands.append(('required_namespaces',[receipt['tools']['unshare'],'--user','--map-root-user','--mount','--pid','--fork','--net','--ipc','--uts','--','/bin/true']))
if receipt['tools']['bwrap']:
    commands.append(('bubblewrap',[receipt['tools']['bwrap'],'--unshare-user','--unshare-pid','--unshare-ipc','--unshare-net','--unshare-uts','--die-with-parent','--ro-bind','/','/','--proc','/proc','--dev','/dev','--','/bin/true']))
receipt['probes']=[]
for name,command in commands:
    result=subprocess.run(command,capture_output=True,text=True,timeout=10,check=False)
    receipt['probes'].append({'name':name,'command':command,'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr})
receipt['status']='CAPABILITY_OBSERVED_NOT_PRODUCTION_QUALIFICATION'
raw=json.dumps(receipt,indent=2)+'\n'
with out.open('x') as f:f.write(raw)
print(raw)
print('RECEIPT_SHA256',hashlib.sha256(raw.encode()).hexdigest())
