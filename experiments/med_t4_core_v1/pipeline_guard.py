"""Own-process-only guard for parallel training followed by the actual final bank."""
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(os.environ['T4_REMOTE_ROOT'])


def read(name):
    return json.loads((ROOT/name).read_text())


def write(name, value):
    p = ROOT/name
    temp = p.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2))
    temp.replace(p)


def size(directory):
    total = 0
    for p in directory.rglob('*'):
        try:
            if p.is_file() and not p.is_symlink():
                total += p.stat().st_blocks*512
        except FileNotFoundError:
            pass
    return total


def run_phase(phase, workers, clock):
    children = []
    try:
        for cfg in workers:
            env = dict(os.environ, **cfg['env'])
            with (ROOT/(phase.lower()+'_'+str(len(children))+'.log')).open('x') as log:
                child = subprocess.Popen([sys.executable, cfg['entry']], env=env, cwd='/data/bmw',
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                    preexec_fn=lambda:resource.setrlimit(resource.RLIMIT_FSIZE,(1024**3,1024**3)))
            children.append(child)
        write(phase+'_IDENTITIES.private.json', dict(pids=[p.pid for p in children],time=time.time()))
        while any(p.poll() is None for p in children):
            used = size(ROOT)+size(Path(os.environ['T4_NEUTRAL_ROOT']))
            fs = os.statvfs(ROOT)
            health = dict(time=time.time(),bytes=used,free_bytes=fs.f_bavail*fs.f_frsize,hard_end=clock['hard_end'])
            write(phase+'_GUARD_HEALTH.json',health)
            if phase == 'TRAIN':
                for cfg in workers:
                    p=Path(cfg['env']['EXPERIMENT_ROOT'])/'TRAIN_GUARD_HEALTH.json'
                    temp=p.with_suffix('.tmp');temp.write_text(json.dumps(health));temp.replace(p)
            assert used+1024**3<4_000_000_000 and health['free_bytes']>8*1024**3
            assert time.time()<clock['hard_end']-120 and not any(p.poll() not in (None,0) for p in children)
            time.sleep(5)
        assert all(p.wait()==0 for p in children)
        write(phase+'_GUARD_EXIT.json',dict(code=0,time=time.time()))
    except BaseException:
        write(phase+'_STOP.json',dict(time=time.time(),error=traceback.format_exc()))
        for p in children:
            if p.poll() is None:
                os.killpg(p.pid,signal.SIGTERM)
        for p in children:
            if p.poll() is None:
                try:p.wait(timeout=20)
                except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
        write(phase+'_GUARD_EXIT.json',dict(code=1,time=time.time()))
        raise


def main():
    with (ROOT/'PIPELINE_ONCE').open('x'):pass
    cfg, clock = read('PIPELINE_CONFIG.private.json'), read('CLOCK.json')
    assert read('CPU_ADMISSION.json')['status']=='PASS' and time.time()<clock['hard_end']-660
    for gpu in cfg['gpus']:
        row=subprocess.check_output(['nvidia-smi','-i',str(gpu['index']),'--query-gpu=uuid,memory.free',
            '--format=csv,noheader,nounits'],text=True).strip().split(',')
        assert row[0].strip()==gpu['uuid'] and int(row[1])>=45056
    run_phase('TRAIN',cfg['train_workers'],clock)
    for item in cfg['train_workers']:
        p=Path(item['env']['EXPERIMENT_ROOT'])
        assert json.loads((p/'TRAIN_WORKER_EXIT.json').read_text())['code']==0
        assert json.loads((p/'TRAIN_COMPLETE.json').read_text())['status']=='T4_C_MED_MDU_SHARD_COMPLETE'
    write('TRAIN_FROZEN.json',dict(status='ALL_ADMITTED_EXPERTS_TRAINED',time=time.time()))
    run_phase('EVAL',[cfg['eval_worker']],clock)
    assert read('EVAL_GENERATION_COMPLETE.json')['status']=='ALL_CONSUMERS_FROZEN'
    write('PIPELINE_COMPLETE.json',dict(status='GENERATION_COMPLETE',time=time.time()))


if __name__=='__main__':
    code=0
    try:main()
    except BaseException:
        code=1;write('PIPELINE_FAILURE.json',dict(time=time.time(),error=traceback.format_exc()));raise
    finally:write('GUARD_EXIT.json',dict(code=code,time=time.time()))
