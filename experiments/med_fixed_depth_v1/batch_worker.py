"""Train one fixed shard after the first pair's engineering evaluation."""
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
DEADLINE = 1793447820
COUNTS = dict(forwards=0, backwards=0, operator_backwards=0, updates=0, generations=0, vision_forwards=0)


def write(name, value):
    p = ROOT / name
    tmp = p.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(p)


def budget():
    if time.time() >= DEADLINE - 60 or (ROOT / 'BATCH_STOP.json').exists():
        raise TimeoutError('Frozen deadline or independent guard stop')
    health = json.loads((ROOT / 'BATCH_GUARD_HEALTH.json').read_text())
    if time.time() - health['time'] > 45:
        raise RuntimeError('Independent guard heartbeat stale')
    if COUNTS['updates'] > 1600 or COUNTS['backwards'] > 3200:
        raise RuntimeError('Shard update budget exceeded')


def main():
    if PARTITIONS not in (4, 5) or PART not in range(PARTITIONS):
        raise ValueError('Invalid fixed shard')
    with (ROOT / f'BATCH_WORKER_{PART}_ONCE').open('x'):
        pass
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(InterruptedError('Guard stop')))
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('Frozen deadline')))
    signal.alarm(max(1, int(DEADLINE - time.time() - 60)))
    sys.path[:0] = [str(ROOT / 'tools'), str(PARENT / 'private/tools'), str(PARENT / 'private/source'),
                    str(PARENT / 'private/official_llava')]
    import torch
    import module_run as r
    import train_pair
    import runtime_binding
    from types import SimpleNamespace
    from scripts.medtrace.run_realmodel_core import load_real_runtime
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    r.c.budget = budget
    r.COUNTS = COUNTS
    if json.loads((ROOT / 'FIRST_PAIR_EVAL_COMPLETE.json').read_text())['engineering_status'] != 'PASS':
        raise ValueError('First pair generation and binding not complete')
    admission = json.loads((ROOT / 'ROLE_ADMISSION.json').read_text())
    lock = json.loads((ROOT / 'PANEL_IDENTITY_LOCK.json').read_text())
    tasks = json.loads((ROOT / 'TRAIN_SOURCE_BOUND_PRIVATE.json').read_text())
    if admission['owners'] != lock['index']['owners'] or [x['task']['order'] for x in tasks] != admission['owners']:
        raise ValueError('Fixed owner identity mismatch')
    assigned = tasks[1:][PART::PARTITIONS]
    if not assigned:
        raise ValueError('Empty shard')
    with (PARENT / 'private/EVAL_BINDINGS.json').open('rb') as f:
        config = runtime_binding.extract(f)['configuration_only']
    runtime = load_real_runtime(SimpleNamespace(cpu_gate=PARENT / 'private/cpu_gate'))
    runtime.model.eval()
    r.old.fp32(runtime)
    if runtime.generation_config != config['generation']:
        raise ValueError('Original generation configuration mismatch')
    def count(_module, _args, _kwargs):
        budget()
        COUNTS['forwards'] += 1
    handle = runtime.model.register_forward_pre_hook(count, with_kwargs=True)
    results = []
    try:
        for entry in assigned:
            results.extend(train_pair.train_pair(runtime, r, entry['task'], entry['source_rows'],
                                                {'configuration_only': config}, admission, ROOT / 'batch', budget))
            write(f'BATCH_PROGRESS_{PART}.json', dict(time=time.time(), completed_owners=len(results) // 2,
                                                      assigned_owners=len(assigned), counts=COUNTS))
        if COUNTS['updates'] != 320 * len(assigned) or COUNTS['backwards'] != 640 * len(assigned):
            raise ValueError('Shard update counts mismatch')
        write(f'BATCH_COMPLETE_{PART}.json', dict(time=time.time(), counts=COUNTS, results=results))
    finally:
        handle.remove()


if __name__ == '__main__':
    code = 0
    try:
        main()
    except BaseException:
        code = 1
        write(f'BATCH_FAILURE_{PART}.json', dict(time=time.time(), error=traceback.format_exc(), counts=COUNTS))
        raise
    finally:
        write(f'BATCH_EXIT_{PART}.json', dict(time=time.time(), code=code, counts=COUNTS))
