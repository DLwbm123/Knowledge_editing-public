"""Shared follow-on GPU process-wall-clock budget; failed attempts remain charged."""
import os,json,sys,time,fcntl,subprocess,signal
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.medtrace.stage19_fasttrack_budget import read,write,remaining
from scripts.medtrace.stage17_prepare import digest


LIMITS={'22A':1800,'22B':8400,'22C':5400,'23':9600,'recovery':3600}


def validate(auth,cfg):
    assert auth['gpu_seconds_limit']==28800 and auth['new_judgment_limit']==2000
    assert auth['approved'] and auth['gpu_uuid']==cfg['gpu_uuid'] and str(auth['physical_gpu'])==str(cfg['gpu'])
    assert digest(auth)==cfg['authorization_binding'] and cfg['phase'] in LIMITS
    assert sum(LIMITS.values())==28800


def supervise(cfg):
    root=Path(cfg['run']);validate(read(root/'public/RUN_AUTHORIZATION.json'),cfg)
    with (Path(cfg['gpu_ledger']).parent.parent/'private/supervisor.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        path=Path(cfg['gpu_ledger']);ledger=read(path)
        assert read(Path(cfg['stage21B_run'])/'public/EXIT_FACT_FIXED_L31_DOWN.json')['exit_code']==0
        assert ledger['limit_seconds']==28800 and not any('seconds' not in s for s in ledger['sessions'])
        available=min(remaining(ledger),LIMITS[cfg['phase']]-sum(x['seconds'] for x in ledger['sessions'] if x['phase']==cfg['phase']))
        if available<60:raise TimeoutError('Follow-on cumulative or stage hard cap')
        start=time.time();session=dict(started_epoch=start,phase=cfg['phase'],code=cfg['code_commit']);ledger['sessions'].append(session);write(path,ledger)
        worker=dict(cfg,gpu_deadline_epoch=start+available-10,campaign_epoch=start,train_seconds=available-10)
        dispatch=root/'private'/('DISPATCH_'+cfg['phase']+'.json');write(dispatch,worker)
        process=None;code=None
        try:
            process=subprocess.Popen([cfg['python'],cfg['entrypoint'],'run'],env=dict(os.environ,JOB_CONFIG=str(dispatch),JOB_ARGV=json.dumps([cfg['worker']])),start_new_session=True)
            session['pid']=process.pid;write(path,ledger)
            try:code=process.wait(timeout=available-10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGTERM)
                try:code=process.wait(timeout=5)
                except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);code=process.wait()
                session['hard_deadline_reached']=True
        finally:
            session.update(ended_epoch=time.time(),seconds=time.time()-start,exit_code=code);write(path,ledger)
            write(root/'public'/('EXIT_'+cfg['phase']+'.json'),dict(exit_code=code,remaining_seconds=remaining(ledger)))
        return code


def selfcheck():
    assert sum(LIMITS.values())==28800
    ledger=dict(limit_seconds=28800,sessions=[dict(started_epoch=0,seconds=100),dict(started_epoch=100)])
    assert remaining(ledger,200)==28600 and remaining(ledger,30000)==0
    a=dict(approved=True,gpu_seconds_limit=28800,new_judgment_limit=2000,gpu_uuid='u',physical_gpu=0)
    c=dict(gpu_uuid='u',gpu=0,phase='22B',authorization_binding=digest(a));validate(a,c)
    try:validate(dict(a,gpu_seconds_limit=30000),c)
    except AssertionError:pass
    else:raise AssertionError('Budget expansion accepted')


if __name__=='__main__':
    if os.environ.get('JOB_SELFCHECK'):selfcheck()
    else:sys.exit(supervise(read(os.environ['JOB_CONFIG'])))
