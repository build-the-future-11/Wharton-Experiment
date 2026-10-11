"""External one-worker process-group wall/RSS watchdog (stdlib only)."""
import argparse, datetime, hashlib, json, os, signal, subprocess, sys, time, threading
from pathlib import Path

def rss_group(pgid):
    # Read OS process accounting; retain only the group owned by this supervisor.
    result = subprocess.run(['/bin/ps','-axo','pid=,pgid=,rss='], capture_output=True, text=True, timeout=.75, check=True)
    rows = [list(map(int,line.split())) for line in result.stdout.splitlines() if line.strip()]
    own = [row for row in rows if row[1] == pgid]
    return sum(row[2]*1024 for row in own), [row[0] for row in own]

def supervise(command, cwd, destination, wall, rss_limit, environment):
    out=Path(destination); out.mkdir(parents=True,exist_ok=False)
    rss_group(os.getpgrp()) # A failed OS monitor must prevent worker launch.
    receipt={'status':'running','command':command,'cwd':str(cwd),'wall_limit_seconds':wall,'rss_limit_bytes':rss_limit,
             'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'environment_overrides':environment,
             'sampling_target_seconds':.2,'monitor':'sum RSS of worker process group from macOS ps; wait4 worker high-water RSS also retained'}
    env=os.environ.copy();env.update(environment)
    start=time.monotonic(); samples=[]; reason=None; term_at=None; status=None; usage=None
    with (out/'stdout.log').open('wb') as stdout,(out/'stderr.log').open('wb') as stderr:
        worker=subprocess.Popen(command,cwd=cwd,env=env,stdout=stdout,stderr=stderr,start_new_session=True)
        pgid=worker.pid
        signal_lock=threading.Lock();sent_signals=set();signal_errors=[]
        def stop(sig):
            with signal_lock:
                if sig in sent_signals:return
                sent_signals.add(sig)
                try:os.killpg(pgid,sig)
                except ProcessLookupError:pass
                except PermissionError:
                    # macOS can return EPERM for a just-reaped group. Only accept
                    # that race if fresh OS accounting proves no group members.
                    _,members=rss_group(pgid)
                    if members:raise
        exited=threading.Event()
        def deadline_guard():
            nonlocal reason,term_at
            try:
                if not exited.wait(max(0,start+wall-time.monotonic())):
                    if reason is None:reason='wall_limit';term_at=time.monotonic()
                    stop(signal.SIGTERM)
                    if not exited.wait(2):stop(signal.SIGKILL)
            except Exception as error:signal_errors.append(repr(error))
        deadline_thread=threading.Thread(target=deadline_guard,daemon=True);deadline_thread.start()
        try:
            while True:
                now=time.monotonic()
                if now-start>=wall and reason is None:reason='wall_limit';term_at=now;stop(signal.SIGTERM)
                if term_at is not None and now-term_at>=2:stop(signal.SIGKILL)
                try:
                    rss,pids=rss_group(pgid)
                    samples.append({'elapsed_seconds':time.monotonic()-start,'rss_bytes':rss,'pids':pids})
                    if rss>rss_limit and reason is None:reason='rss_limit';term_at=time.monotonic();stop(signal.SIGTERM)
                except Exception as error:
                    if reason is None:reason='monitor_failed';term_at=time.monotonic();stop(signal.SIGTERM)
                    receipt['monitor_error']=repr(error)
                pid,status_now,usage_now=os.wait4(worker.pid,os.WNOHANG)
                if pid:
                    status=status_now;usage=usage_now;worker.returncode=os.waitstatus_to_exitcode(status);break
                if term_at is not None and time.monotonic()-term_at>4.5:
                    stop(signal.SIGKILL)
                    raise RuntimeError('worker did not reap within cleanup budget')
                deadline = start+wall if term_at is None else term_at+2
                time.sleep(max(0,min(.2,deadline-time.monotonic())))
        except Exception as error:
            if reason is None:reason='supervisor_error'
            receipt['supervisor_error']=repr(error)
            raise
        finally:
            exit_at=time.monotonic();exited.set();stop(signal.SIGKILL)
            if worker.returncode is None:
                _,status,usage=os.wait4(worker.pid,0);worker.returncode=os.waitstatus_to_exitcode(status)
            _,remaining=rss_group(pgid)
            cleanup=time.monotonic()-exit_at
            peak_sample=max((s['rss_bytes'] for s in samples),default=0)
            peak_worker=int(usage.ru_maxrss) if usage else None # macOS bytes, not Linux KiB
            if reason is None and peak_worker is not None and peak_worker>rss_limit:reason='worker_peak_rss_limit'
            if reason is None and remaining:reason='orphan_processes'
            if reason is None and cleanup>5:reason='cleanup_limit'
            if reason is None and worker.returncode:reason='worker_error'
            if signal_errors:reason='signal_error'
            receipt.update(status='completed' if reason is None else 'failed',stop_reason=reason,exit_code=worker.returncode,
                worker_pid=worker.pid,worker_group=pgid,wall_seconds=exit_at-start,cleanup_seconds=cleanup,
                sampled_peak_group_rss_bytes=peak_sample,worker_peak_rss_bytes=peak_worker,
                cpu_user_seconds=usage.ru_utime if usage else None,cpu_system_seconds=usage.ru_stime if usage else None,
                remaining_group_pids=remaining,signal_errors=signal_errors,rss_samples=samples,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
            (out/'supervisor_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--cwd',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--wall',required=True,type=float);parser.add_argument('--rss',required=True,type=int);parser.add_argument('command',nargs=argparse.REMAINDER)
    args=parser.parse_args();command=args.command[1:] if args.command[:1]==['--'] else args.command
    env={key:'1' for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','PYTHONDONTWRITEBYTECODE')};env['PYTHONPATH']='.deps:.'
    result=supervise(command,args.cwd,args.output,args.wall,args.rss,env)
    print(json.dumps({k:v for k,v in result.items() if k!='rss_samples'},indent=2))
    sys.exit(0 if result['status']=='completed' else 1)
