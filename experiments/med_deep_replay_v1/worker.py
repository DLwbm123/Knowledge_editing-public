"""Train a fixed shard after independent role, source and resource admission.

Only the new run's admitted source/FIT files are read. This worker does not
read evaluation inputs, references, predictions or scores.
"""
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback

ROOT = Path(os.environ['EXPERIMENT_ROOT'])
PARENT = Path(os.environ['PARENT_RUN'])
PART = int(os.environ['PARTITION'])
PARTITIONS = int(os.environ['PARTITIONS'])
DEADLINE = float(os.environ['EXPERIMENT_DEADLINE'])
COUNTS = dict(forwards=0, backwards=0, updates=0, generations=0, vision_forwards=0)


def write(name, value):
    target = ROOT/name
    tmp = target.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2)+'\n')
    tmp.replace(target)


def budget():
    if time.time() >= DEADLINE-60 or (ROOT/'STOP.json').exists():
        raise TimeoutError('Independent deadline or guard stop')
    health = json.loads((ROOT/'GUARD_HEALTH.json').read_text())
    if time.time()-health['time']>45 or health['hard_end']!=DEADLINE:
        raise RuntimeError('Independent guard unavailable or deadline mismatch')
    if COUNTS['updates']>2560 or COUNTS['backwards']>7680:
        raise RuntimeError('Frozen experiment count exceeded')


def main():
    if PARTITIONS not in (4,5) or PART not in range(PARTITIONS):
        raise ValueError('Invalid shard')
    with (ROOT/f'WORKER_{PART}_ONCE').open('x'):
        pass
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(InterruptedError('Guard stop')))
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('Deadline')))
    if time.time() >= DEADLINE-60:
        raise TimeoutError('No remaining time')
    signal.alarm(max(1,int(DEADLINE-time.time()-60)))
    sys.path[:0] = [str(ROOT/'tools'),str(PARENT/'private/tools'),str(PARENT/'private/source'),
                    str(PARENT/'private/official_llava')]
    import torch
    import module_run as r
    import batches
    import train
    import runtime_binding
    from types import SimpleNamespace
    from scripts.medtrace.run_realmodel_core import load_real_runtime
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    r.c.budget = budget
    r.COUNTS = COUNTS
    admission=json.loads((ROOT/'ROLE_ADMISSION.json').read_text())
    identity=json.loads((ROOT/'REPLAY_IDENTITY_LOCK_PRIVATE.json').read_text())
    rows=json.loads((ROOT/'REPLAY_CONTENT_PRIVATE.json').read_text())
    sources=json.loads((ROOT/'TRAIN_SOURCE_BOUND_PRIVATE.json').read_text())
    if admission['status']!='FIT_CONTENT_AUTHORIZED' or len(admission['owners'])!=16:
        raise ValueError('Independent role admission required')
    if identity['status']!='IDENTITY_FROZEN_ONLY' or len(rows)!=len(identity['records']):
        raise ValueError('Replay source lock mismatch')
    for row, locked in zip(rows,identity['records']):
        if any(row[k]!=locked[k] for k in ('source_group','query_id','role','source_qid')):
            raise ValueError('Replay content differs from frozen identities')
    if [x['task']['order'] for x in sources]!=admission['owners']:
        raise ValueError('Fixed panel owners differ')
    assigned=sources[PART::PARTITIONS]
    if not assigned:
        raise ValueError('Empty shard')
    with (PARENT/'private/EVAL_BINDINGS.json').open('rb') as f:
        config=runtime_binding.extract(f)['configuration_only']
    runtime=load_real_runtime(SimpleNamespace(cpu_gate=PARENT/'private/cpu_gate'))
    runtime.model.eval()
    r.old.fp32(runtime)
    if runtime.generation_config!=config['generation']:
        raise ValueError('Generation configuration differs')
    def count(_module,_args,_kwargs):
        budget()
        COUNTS['forwards']+=1
    handle=runtime.model.register_forward_pre_hook(count,with_kwargs=True)
    results=[]
    try:
        for entry in assigned:
            task, source_rows=entry['task'],entry['source_rows']
            if len(source_rows)!=5 or [x['audit_role'] for x in source_rows]!=['NATIVE','FIT','FIT','FIT','FIT']:
                raise ValueError('Original source inputs mismatch')
            runtime.model.half()
            try:
                prepared=r.s.prepare(runtime,{'configuration_only':config},task,source_rows,True)
            finally:
                r.old.fp32(runtime)
            source_batches=[r.old.batch_on(runtime,item[1]) for item in prepared]
            replay_batches=batches.prepare_replay(runtime,r,task,rows,admission,budget)
            result=train.train(runtime,r,task,source_batches,replay_batches,admission,
                               ROOT/'weights'/f"{task['order']}.pt",budget)
            results.append(dict(result,owner=task['order']))
            write(f'PROGRESS_{PART}.json',dict(time=time.time(),completed_owners=len(results),counts=COUNTS))
        if COUNTS['updates']!=160*len(assigned) or COUNTS['backwards']!=480*len(assigned):
            raise ValueError('Actual training count mismatch')
        write(f'COMPLETE_{PART}.json',dict(time=time.time(),counts=COUNTS,results=results))
    finally:
        handle.remove()


if __name__=='__main__':
    code=0
    try:
        main()
    except BaseException:
        code=1
        write(f'FAILURE_{PART}.json',dict(time=time.time(),error=traceback.format_exc(),counts=COUNTS))
        raise
    finally:
        write(f'EXIT_{PART}.json',dict(time=time.time(),code=code,counts=COUNTS))
