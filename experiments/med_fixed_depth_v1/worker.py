"""First fixed pair with source-only inputs and the original real-model runtime."""
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback

ROOT = Path(os.environ['EXPERIMENT_ROOT'])
PARENT = Path(os.environ['PARENT_RUN'])
DEADLINE = 1793447820
COUNTS = dict(forwards=0, backwards=0, operator_backwards=0, updates=0, generations=0, vision_forwards=0)


def write(name, value):
    p = ROOT / name
    tmp = p.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(p)


def budget():
    if time.time() >= DEADLINE - 60 or (ROOT / 'STOP.json').exists():
        raise TimeoutError('Frozen deadline or independent guard stop')
    health = json.loads((ROOT / 'GUARD_HEALTH.json').read_text())
    if time.time() - health['time'] > 45:
        raise RuntimeError('Independent guard heartbeat stale')
    if COUNTS['updates'] > 320 or COUNTS['backwards'] > 640:
        raise RuntimeError('First-pair scientific budget exceeded')


def main():
    with (ROOT / 'WORKER_ONCE').open('x'):
        pass
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(InterruptedError('Independent guard stop')))
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
    admission = json.loads((ROOT / 'ROLE_ADMISSION.json').read_text())
    lock = json.loads((ROOT / 'PANEL_IDENTITY_LOCK.json').read_text())
    tasks = json.loads((ROOT / 'TRAIN_SOURCE_BOUND_PRIVATE.json').read_text())
    if admission['owners'] != lock['index']['owners'] or [t['task']['order'] for t in tasks] != admission['owners']:
        raise ValueError('Fixed owner identity mismatch')
    with (PARENT / 'private/EVAL_BINDINGS.json').open('rb') as f:
        config = runtime_binding.extract(f)['configuration_only']
    runtime = load_real_runtime(SimpleNamespace(cpu_gate=PARENT / 'private/cpu_gate'))
    runtime.model.eval()
    r.old.fp32(runtime)
    if runtime.generation_config != config['generation']:
        raise ValueError('Original generation configuration mismatch')
    bindings = {'configuration_only': config}
    def count(_module, _args, _kwargs):
        budget()
        COUNTS['forwards'] += 1
    handle = runtime.model.register_forward_pre_hook(count, with_kwargs=True)
    try:
        first = tasks[0]
        results = train_pair.train_pair(runtime, r, first['task'], first['source_rows'], bindings,
                                        admission, ROOT / 'first_pair', budget)
        if COUNTS['updates'] != 320 or COUNTS['backwards'] != 640 or len(results) != 2:
            raise ValueError('First-pair actual update counts mismatch')
        write('FIRST_PAIR_TRAIN_COMPLETE.json', dict(time=time.time(), counts=COUNTS, results=results,
              evaluation_pending=True, ordinary_adam=True, old_curvature_used=False))
    finally:
        handle.remove()


if __name__ == '__main__':
    code = 0
    try:
        main()
    except BaseException:
        code = 1
        write('WORKER_FAILURE.json', dict(time=time.time(), error=traceback.format_exc(), counts=COUNTS))
        raise
    finally:
        write('WORKER_EXIT.json', dict(time=time.time(), code=code, counts=COUNTS))
