"""Four-shard GPU guard with fixed original clock and independent storage checks."""
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import time
import traceback

root=Path(os.environ['EXPERIMENT_ROOT'])
deadline=float(os.environ['EXPERIMENT_DEADLINE'])
phase=os.environ['PHASE']
arm=os.environ['ARM']
if arm not in ('D','DU'): raise ValueError('invalid arm')
expected=101 if arm=='DU' else 25
generations=int(os.environ['EXPECTED_GENERATIONS'])
if phase != 'ENDPOINT':raise ValueError('invalid phase')


def write(name,value):
    path=root/name;tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value));tmp.replace(path)


def used_bytes():
    total=0
    for p in root.parent.rglob('*'):
        try:
            if p.is_file() and not p.is_symlink():total+=p.stat().st_blocks*512
        except FileNotFoundError:pass
    return total


def launch():
    with (root/f'{phase}_GUARD_ONCE').open('x'):pass
    if json.loads((root/'SCORING_CHAIN_ADMISSION.json').read_text()).get('status')!='SCORING_CHAIN_ADMITTED_WITH_UNKNOWN_PRESERVED':
        raise ValueError('score chain admission missing')
    if json.loads((root/'BANK_INPUTS_PRIVATE.json').read_text()).get('status')!='BANK_INPUTS_BOUND':
        raise ValueError('bank binding missing')
    prior=Path(os.environ['PREVIOUS_TRAIN_ROOT'])
    if json.loads((prior/'TRAIN_GUARD_EXIT.json').read_text()).get('code') != 0:
        raise ValueError('prior binding run failed')
    if json.loads((prior/'TRAIN_FROZEN.json').read_text()).get('owners') != 101:
        raise ValueError('prior binding incomplete')
    if json.loads((root/'ROUTES_FROZEN.json').read_text()).get('generated') != 1513:
        raise ValueError('candidate routes not frozen')
    gpu=[]
    text=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],text=True)
    for line in text.splitlines():
        index,uuid,free=(x.strip() for x in line.split(','))
        if int(free)>=45056:gpu.append((index,uuid))
    if len(gpu)<4:raise RuntimeError('four GPUs with sufficient free memory unavailable')
    gpu=gpu[:4]
    (root/'weights').mkdir(exist_ok=True)
    write(f'{phase}_GUARD_HEALTH.json',{'time':time.time(),'hard_end':deadline,'bytes':used_bytes()})
    children=[]
    def limit():resource.setrlimit(resource.RLIMIT_FSIZE,(1024**3,1024**3))
    try:
        for i,(idx,uuid) in enumerate(gpu):
            env=dict(os.environ,SHARD=str(i),CUDA_VISIBLE_DEVICES=idx,M3BENCH_FORMAL_EXPECTED_GPU_UUID=uuid,
                M3BENCH_FORMAL_ALLOWED_CUDA_VISIBLE_DEVICES=idx,M3BENCH_FORMAL_AUTHORIZED_CUDA_VISIBLE_DEVICES=idx)
            with (root/f'{phase.lower()}_{i}.log').open('x') as log:
                p=subprocess.Popen([os.environ['TRAIN_PYTHON'],os.environ[f'{phase}_NEUTRAL_WORKER']],env=env,
                    stdout=log,stderr=subprocess.STDOUT,cwd='/data/bmw',start_new_session=True,preexec_fn=limit)
            children.append(p)
        write(f'{phase}_IDENTITIES.private.json',{'pids':[p.pid for p in children],'devices':[x[0] for x in gpu]})
        while any(p.poll() is None for p in children):
            used=used_bytes();fs=os.statvfs(root);free=fs.f_bavail*fs.f_frsize
            write(f'{phase}_GUARD_HEALTH.json',{'time':time.time(),'hard_end':deadline,'bytes':used,'free':free})
            if any(p.poll() not in (None,0) for p in children):raise RuntimeError('shard failure')
            if used>=30_000_000_000 or free<=8*1024**3 or time.time()>=deadline-120:
                raise RuntimeError('resource or original deadline stop')
            time.sleep(5)
        codes=[p.wait() for p in children]
        if codes!=[0,0,0,0]:raise RuntimeError('nonzero shard exit')
        totals=[json.loads((root/f'ENDPOINT_COMPLETE_{i}.json').read_text()) for i in range(4)]
        owners=[row['owner'] for shard in totals for row in shard['results']]
        if len(owners)!=101 or len(set(owners))!=101:
            raise ValueError('endpoint owner coverage differs')
        if sum(shard['counts']['forwards'] for shard in totals)!=101:
            raise ValueError('endpoint forward counts differ')
        if any(row['endpoint_checks'] != 2 or row['midpoint_checks'] != 1 for shard in totals for row in shard['results']):
            raise ValueError('endpoint output checks missing')
        write('ENDPOINT_FROZEN.json',{'status':'REAL_LAYER_ENDPOINTS_PASS','owners':101,'forwards':101})
        write(f'{phase}_GUARD_EXIT.json',{'code':0,'time':time.time()})
    except BaseException:
        write(f'{phase}_STOP.json',{'time':time.time(),'error':traceback.format_exc()})
        for p in children:
            if p.poll() is None:os.killpg(p.pid,signal.SIGTERM)
        for p in children:
            if p.poll() is None:
                try:p.wait(timeout=15)
                except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
        raise


if __name__=='__main__':
    try:launch()
    except BaseException:
        write(f'{phase}_GUARD_FAILURE.json',{'time':time.time(),'error':traceback.format_exc()})
        write(f'{phase}_GUARD_EXIT.json',{'code':1,'time':time.time()})
        raise
