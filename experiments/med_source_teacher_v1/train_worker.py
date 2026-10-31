"""Single bounded worker: freeze all teachers, then train the fixed panel."""
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback

ROOT = Path(os.environ['EXPERIMENT_ROOT'])
PARENT = Path(os.environ['PARENT_RUN'])
FIXED = Path(os.environ['FIXED_DEPTH_ROOT'])
MED = Path(os.environ['RAW_MED_ROOT'])
DEADLINE = float(os.environ['EXPERIMENT_DEADLINE'])
COUNTS = dict(forwards=0, backwards=0, operator_backwards=0, updates=0, generations=0, vision_forwards=0)


def write(name, value):
    path = ROOT / name
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def budget():
    if time.time() >= DEADLINE - 60 or (ROOT / 'STOP.json').exists():
        raise TimeoutError('Deadline or independent guard stop')
    health = json.loads((ROOT / 'GUARD_HEALTH.json').read_text())
    if time.time() - health['time'] > 45 or health['hard_end'] != DEADLINE:
        raise RuntimeError('Independent guard unavailable')
    if COUNTS['updates'] > 2560 or COUNTS['backwards'] > 5120:
        raise RuntimeError('Training budget exceeded')


def main():
    with (ROOT / 'TRAIN_WORKER_ONCE').open('x'):
        pass
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(InterruptedError('Guard stop')))
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('Deadline')))
    if time.time() >= DEADLINE - 60:
        raise TimeoutError('No remaining time')
    signal.alarm(max(1, int(DEADLINE - time.time() - 60)))
    sys.path[:0] = [str(ROOT / 'tools'), str(PARENT / 'private/tools'), str(PARENT / 'private/source'),
                    str(PARENT / 'private/official_llava')]
    import torch
    import module_run as r
    import runtime_binding
    import teacher_producer
    import teacher_cache
    import train
    from train_pair import state_digest
    from types import SimpleNamespace
    from scripts.medtrace.run_realmodel_core import load_real_runtime
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    r.c.budget, r.COUNTS = budget, COUNTS
    admission = json.loads((FIXED / 'ROLE_ADMISSION.json').read_text())
    source = json.loads((FIXED / 'TRAIN_SOURCE_BOUND_PRIVATE.json').read_text())
    library = json.loads((MED / 'EXPERT_LIBRARY_FROZEN.json').read_text())
    if admission['status'] != 'SOURCE_AND_EVALUATION_ROLE_ADMITTED' or len(admission['owners']) != 16:
        raise ValueError('Fixed role admission invalid')
    if [entry['task']['order'] for entry in source] != admission['owners'] or library['experts'] != 146:
        raise ValueError('Fixed owner or original library mismatch')
    expected = {entry['task']['order']: library['states'][entry['task']['edit_id']] for entry in source}
    if len(expected) != 16:
        raise ValueError('Duplicate owner')
    with (PARENT / 'private/EVAL_BINDINGS.json').open('rb') as f:
        config = runtime_binding.extract(f)['configuration_only']
    runtime = load_real_runtime(SimpleNamespace(cpu_gate=PARENT / 'private/cpu_gate'))
    runtime.model.eval(); r.old.fp32(runtime)
    if runtime.generation_config != config['generation']:
        raise ValueError('Original generation configuration differs')
    def count(_module, _args, _kwargs):
        budget(); COUNTS['forwards'] += 1
    handle = runtime.model.register_forward_pre_hook(count, with_kwargs=True)
    try:
        teacher_results = []
        for entry in source:
            task = entry['task']; owner = task['order']
            state = torch.load(MED / 'candidates' / f'MED_{owner}.pt', map_location='cpu', weights_only=True)
            if state_digest(state) != expected[owner]:
                raise ValueError('Original teacher state mismatch')
            result = teacher_producer.produce(runtime, r, task, entry['source_rows'], config, state,
                                               expected[owner], ROOT / 'teachers' / f'{owner}.pt',
                                               1_000_000_000, budget)
            teacher_results.append(result)
            write('TEACHER_PROGRESS.json', dict(completed=len(teacher_results), counts=COUNTS))
        packets = {entry['task']['order']: torch.load(ROOT / 'teachers' / f"{entry['task']['order']}.pt",
                                                       map_location='cpu', weights_only=True) for entry in source}
        vocab = int(runtime.model.config.vocab_size)
        frozen = teacher_cache.freeze(packets, expected, vocab, 20_000_000_000)
        write('TEACHERS_FROZEN.json', frozen)
        results = []
        for entry in source:
            task, rows = entry['task'], entry['source_rows']; owner = task['order']
            runtime.model.half()
            try:
                prepared = r.s.prepare(runtime, {'configuration_only': config}, task, rows, True)
            finally:
                r.old.fp32(runtime)
            batches = [r.old.batch_on(runtime, item[1]) for item in prepared]
            result = train.train(runtime, r, task, batches, packets[owner]['cache'], frozen,
                                 ROOT / 'weights' / f'{owner}.pt', budget)
            results.append(dict(result, owner=owner))
            write('TRAIN_PROGRESS.json', dict(completed=len(results), counts=COUNTS))
        if COUNTS['updates'] != 2560 or COUNTS['backwards'] != 5120:
            raise ValueError('Training counts do not match fixed panel')
        write('TRAIN_COMPLETE.json', dict(results=results, counts=COUNTS))
    finally:
        handle.remove()


if __name__ == '__main__':
    code = 0
    try:
        main()
    except BaseException:
        code = 1
        write('TRAIN_FAILURE.json', dict(error=traceback.format_exc(), counts=COUNTS))
        raise
    finally:
        write('TRAIN_EXIT.json', dict(code=code, counts=COUNTS))
