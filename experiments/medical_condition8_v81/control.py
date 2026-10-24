"""Independent resource guard and dependency-ordered finite controller."""
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(os.environ['V81_ROOT'])


def read(p):
    return json.loads(Path(p).read_text())


def write(p, v):
    p = Path(p)
    t = p.with_suffix('.'+str(os.getpid())+'.tmp')
    t.write_text(json.dumps(v, ensure_ascii=False, indent=2)+'\n')
    t.replace(p)


def identity(t):
    try:
        tail = Path(f'/proc/{t["pid"]}/stat').read_text().rsplit(')',1)[1].split()
        return tail[19] == str(t['start_ticks']) and os.getsid(t['pid']) == t['pid']
    except (FileNotFoundError, ProcessLookupError):
        return False


def guard():
    with (ROOT/'GUARD_ONCE').open('x'):
        pass
    limits = read(ROOT/'RESOURCE_LIMITS.json')
    trigger = None
    last_scan = 0
    size = 0
    while True:
        now = time.time()
        targets = read(ROOT/'RESOURCE_TARGETS.json')['targets']
        live = [t for t in targets if identity(t)]
        if not live and (ROOT/'RESOURCE_SCOPE_CLOSED.json').exists():
            write(ROOT/'RESOURCE_GUARD_EXIT.json', dict(status='ALL_REGISTERED_PROCESSES_EXITED',
                  time=now, trigger=trigger))
            return
        if now-last_scan >= 30:
            size = sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file() and not p.is_symlink())
            size += sum(Path(p).stat().st_size for p in limits['external_owned_paths'] if Path(p).exists())
            last_scan = now
            if size >= limits['soft_stop_bytes'] or shutil.disk_usage(ROOT).free < limits['free_reserve_bytes']:
                trigger = trigger or dict(reason='STORAGE', time=now, bytes=size)
        gpu_seconds = sum((t.get('ended_epoch') or now)-t['started_epoch'] for t in targets if t.get('gpu') is not None)
        if gpu_seconds >= limits['gpu_process_seconds']:
            trigger = trigger or dict(reason='GPU_PROCESS_SECONDS', time=now)
        totals={k:0 for k in limits['counts']}
        for f in ROOT.glob('PROGRESS_*.json'):
            for k,v in read(f)['counts'].items():totals[k]+=v
        if any(totals[k]>limits['counts'][k] for k in totals):
            trigger = trigger or dict(reason='GLOBAL_OPERATION_BUDGET', time=now, counts=totals)
        if now >= limits['hard_end']-300:
            trigger = trigger or dict(reason='ORIGINAL_WALL_DEADLINE', time=now)
        if (ROOT/'STOP').exists():
            trigger = trigger or dict(reason='STOP_MARKER', time=now)
        if trigger:
            if not (ROOT/'STOP').exists():
                write(ROOT/'STOP', trigger)
            write(ROOT/'RESOURCE_STOP_REQUEST.json', trigger)
            for t in live:
                if identity(t) and now-trigger['time'] >= 30:
                    os.killpg(t['pid'], signal.SIGKILL if now-trigger['time']>=60 else signal.SIGTERM)
        write(ROOT/'RESOURCE_GUARD_HEALTH.json', dict(time=now, status='STOPPING' if trigger else 'HEALTHY',
              owned_bytes=size, gpu_seconds=gpu_seconds, hard_end=limits['hard_end'], live_targets=live))
        time.sleep(10)


def launch(phase, part, gpu):
    env=dict(os.environ, PHASE=phase, PARTITION=str(part), GPU=str(gpu), CUDA_VISIBLE_DEVICES=str(gpu))
    log=(ROOT/'logs'/f'{phase}_{part}.log').open('ab')
    child=subprocess.Popen([sys.executable, '-u', os.environ['NEUTRAL_WORKER']], env=env,
                           stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    log.close()
    t=dict(pid=child.pid, start_ticks=Path(f'/proc/{child.pid}/stat').read_text().split()[21],
           phase=phase, gpu=gpu, started_epoch=time.time(), deadline=read(ROOT/'RESOURCE_LIMITS.json')['hard_end'])
    d=read(ROOT/'RESOURCE_TARGETS.json'); d['targets'].append(t); write(ROOT/'RESOURCE_TARGETS.json',d)
    write(ROOT/f'START_{phase}_{part}.json',t)
    write(ROOT/f'GO_{phase}_{part}.json',dict(time=time.time()))
    return child


def stage(phase):
    jobs=[launch(phase,p,g) for p,g in enumerate(read(ROOT/'GPUS.json'))]
    codes=[]
    for child in jobs:
        code=child.wait();codes.append(code)
        d=read(ROOT/'RESOURCE_TARGETS.json')
        next(t for t in d['targets'] if t['pid']==child.pid)['ended_epoch']=time.time()
        write(ROOT/'RESOURCE_TARGETS.json',d)
    assert codes==[0,0], (phase,codes)
    assert all(read(ROOT/f'COMPLETE_{phase}_{p}.json')['status']=='PASS' for p in range(2))


def controller():
    with (ROOT/'CONTROLLER_ONCE').open('x'):
        pass
    stage('teacher')
    write(ROOT/'TEACHER_COMPLETE.json',dict(status='WAITING_ASSISTANT_BLIND_TARGET_REVIEW',time=time.time(),count=120))
    while not (ROOT/'TARGET_REVIEW_COMPLETE.json').exists():
        assert time.time()<read(ROOT/'RESOURCE_LIMITS.json')['hard_end']-300
        assert not (ROOT/'STOP').exists()
        time.sleep(15)
    assert read(ROOT/'TARGET_REVIEW_COMPLETE.json')['status']=='PASS'
    stage('train')
    assert len(list((ROOT/'training').glob('*/*.json')))==32
    write(ROOT/'ALL_CANDIDATES_FROZEN.json',dict(count=32,steps=160,time=time.time(),selection=False))
    stage('evaluate')
    expected=4*len(read(ROOT/'EVAL_PLAN.json'))
    actual=list((ROOT/'outputs').glob('*/*/*.json'))
    assert len(actual)==expected==620
    totals={k:0 for k in read(ROOT/'RESOURCE_LIMITS.json')['counts']}
    for f in ROOT.glob('EXIT_*.json'):
        d=read(f);assert d['exit_code']==0
        for k,v in d['counts'].items():totals[k]+=v
    assert totals['updates']==5120
    assert all(totals[k]<=v for k,v in read(ROOT/'RESOURCE_LIMITS.json')['counts'].items())
    write(ROOT/'MAIN_COMPLETE.json',dict(status='MODEL_CONSUMERS_COMPLETE_SCORING_PENDING',
          outputs=expected,counts=totals,time=time.time()))
    # Scoring is a small frozen-answer consumer. Weights remain until its
    # lifecycle checklist has explicitly closed all registered dependencies.
    write(ROOT/'RESOURCE_SCOPE_CLOSED.json',dict(time=time.time()))


if __name__=='__main__':
    action=os.environ['CONTROL_ACTION']
    try:
        (guard if action=='guard' else controller)()
    except BaseException as e:
        write(ROOT/f'FAILURE_{action}.json',dict(error=repr(e),traceback=traceback.format_exc(),time=time.time()))
        raise
