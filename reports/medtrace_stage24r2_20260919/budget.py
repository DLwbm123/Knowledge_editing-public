"""Append Stage24 sessions to the existing shared ledger, never initialize/reset it."""
import sys,os,time,json,fcntl,subprocess,signal
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.medtrace.stage19_fasttrack_budget import read,write,remaining
from scripts.medtrace.stage17_prepare import digest


def supervise(cfg):
    root=Path(cfg['run']);path=Path(cfg['gpu_ledger']);reg=read(root/'public/PREREGISTRATION.json')
    assert digest({k:v for k,v in reg.items() if k!='binding'})==reg['binding']==cfg['registration_binding']
    assert reg['cumulative_GPU_limit']==28800 and cfg['phase'] in reg['phase_GPU_caps']
    with (path.parent.parent/'private/supervisor.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);ledger=read(path)
        assert ledger['limit_seconds']==28800 and all('seconds' in x for x in ledger['sessions'])
        available=min(remaining(ledger),reg['phase_GPU_caps'][cfg['phase']]-sum(x['seconds'] for x in ledger['sessions'] if x['phase']==cfg['phase']))
        if available<60:raise TimeoutError('Phase or cumulative budget unavailable')
        stamp=time.time();session=dict(started_epoch=stamp,phase=cfg['phase'],code=cfg['code_commit'],registration=reg['binding']);ledger['sessions'].append(session);write(path,ledger)
        worker=dict(cfg,gpu_deadline_epoch=stamp+available-10,campaign_epoch=stamp,train_seconds=available-10)
        dispatch=root/'private'/('DISPATCH_'+cfg['phase']+'.json');write(dispatch,worker);rc=None;proc=None
        try:
            proc=subprocess.Popen([cfg['python'],cfg['entrypoint'],'run'],env=dict(os.environ,JOB_CONFIG=str(dispatch),JOB_ARGV=json.dumps([cfg['worker']])),start_new_session=True);session['pid']=proc.pid;write(path,ledger)
            try:rc=proc.wait(timeout=available-10)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid,signal.SIGTERM)
                try:rc=proc.wait(timeout=5)
                except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);rc=proc.wait()
                session['hard_deadline_reached']=True
        finally:
            session.update(ended_epoch=time.time(),seconds=time.time()-stamp,exit_code=rc);write(path,ledger)
            write(root/'public'/('EXIT_'+cfg['phase']+'.json'),dict(exit_code=rc,seconds=session['seconds'],remaining_seconds=remaining(ledger)))
    return rc

if __name__=='__main__':sys.exit(supervise(read(os.environ['JOB_CONFIG'])))
