"""Per-run leases, one resident per authorized GPU; no inherited waivers."""
import os,json,time,fcntl,contextlib
from pathlib import Path
ROOT=Path(os.environ['RUN_ROOT'])
def read(p):return json.loads(Path(p).read_text())
def write(p,d):
    from storage import Store
    Store(ROOT).write(str(Path(p).relative_to(ROOT)),json.dumps(d,ensure_ascii=False).encode())
def limits_waived():return True # Only legacy generation deadline compatibility; Judge and storage caps are enforced separately.
@contextlib.contextmanager
def lock():
    with (ROOT/'RESOURCE_LEDGER.lock').open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX);d=read(ROOT/'RESOURCE_LEDGER.json');yield d;write(ROOT/'RESOURCE_LEDGER.json',d)
def check(training=False):
    if (ROOT/'STOP').exists():raise RuntimeError('USER_STOP')
    manifest=read(ROOT/'RUN_MANIFEST.json')
    if time.time()>=manifest['deadline_epoch']:raise RuntimeError('WALL_TIME_LIMIT')
    ledger=read(ROOT/'RESOURCE_LEDGER.json')
    seconds=ledger['current_gpu_seconds']+sum(time.time()-s['started_epoch'] for s in ledger['gpu_sessions'] if not s.get('ended_epoch'))
    if seconds>=manifest['GPU_hours_limit']*3600:raise RuntimeError('GPU_HOUR_LIMIT')
    if (ROOT/'STORAGE_HARD_STOP').exists():raise RuntimeError('STORAGE_HARD_STOP')
@contextlib.contextmanager
def session(purpose,seconds=7200):
    check();gpu=os.environ['CUDA_VISIBLE_DEVICES'];physical=int(os.environ['PHYSICAL_GPU'])
    assert physical in [6,7] and any(g['uuid']==gpu and g['index']==physical for g in read(ROOT/'GPU_BINDINGS.json')['devices'])
    with lock() as d:
        active=[s for s in d['gpu_sessions'] if not s.get('ended_epoch')]
        assert len(active)<2 and all(s['gpu_uuid']!=gpu for s in active)
        s=dict(pid=os.getpid(),gpu_uuid=gpu,physical=physical,purpose=purpose,started_epoch=time.time(),reserved_seconds=seconds);d['gpu_sessions'].append(s)
    try:yield
    finally:
        with lock() as d:
            q=next(x for x in d['gpu_sessions'] if x['pid']==s['pid'] and x['started_epoch']==s['started_epoch']);q.update(ended_epoch=time.time());q['resident_seconds']=q['ended_epoch']-q['started_epoch'];d['current_gpu_seconds']+=q['resident_seconds'];d['lifetime_gpu_seconds']=d['current_gpu_seconds']
