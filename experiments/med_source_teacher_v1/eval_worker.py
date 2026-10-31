"""Generate the fixed panel only after the training worker and guard succeed."""
import hashlib
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
COUNTS = dict(forwards=0, generations=0)


def write(name, value, once=False):
    path = ROOT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    if once:
        with path.open('x') as f: json.dump(value, f, ensure_ascii=False)
        path.chmod(0o600)
    else:
        tmp = path.with_suffix('.tmp'); tmp.write_text(json.dumps(value)); tmp.replace(path)


def budget():
    if time.time() >= DEADLINE - 60 or (ROOT / 'STOP.json').exists():
        raise TimeoutError('Deadline or guard stop')
    health = json.loads((ROOT / 'GUARD_HEALTH.json').read_text())
    if time.time() - health['time'] > 45 or health['hard_end'] != DEADLINE:
        raise RuntimeError('Independent guard unavailable')
    if COUNTS['generations'] > 508:
        raise RuntimeError('Generation budget exceeded')


def main():
    with (ROOT / 'EVAL_WORKER_ONCE').open('x'): pass
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(InterruptedError('Guard stop')))
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('Deadline')))
    if time.time() >= DEADLINE - 60: raise TimeoutError('No remaining time')
    signal.alarm(max(1, int(DEADLINE - time.time() - 60)))
    sys.path[:0] = [str(ROOT / 'tools'), str(PARENT / 'private/tools'), str(PARENT / 'private/source'),
                    str(PARENT / 'private/official_llava')]
    import torch
    import module_run as r
    import runtime_binding
    import evaluate
    from first_eval import rng_digest
    from types import SimpleNamespace
    from scripts.medtrace.run_realmodel_core import load_real_runtime
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True); r.c.budget = budget
    if json.loads((ROOT / 'TRAIN_EXIT.json').read_text())['code'] != 0 or json.loads((ROOT / 'TRAIN_GUARD_EXIT.json').read_text())['code'] != 0:
        raise ValueError('Training dependency failed')
    results = json.loads((ROOT / 'TRAIN_COMPLETE.json').read_text())['results']
    states = {x['owner']: x for x in results}
    admission = json.loads((FIXED / 'ROLE_ADMISSION.json').read_text())
    tasks = json.loads((FIXED / 'TASK_IDENTITIES_PRIVATE.json').read_text())
    rows = json.loads((FIXED / 'EVAL_INPUTS_PRIVATE.json').read_text())
    prior = {(x['owner'], x['query_id']): x for x in json.loads((FIXED / 'PRIOR_INPUT_BINDINGS_PRIVATE.json').read_text())}
    if len(states) != 16 or set(states) != set(admission['owners']) or [t['order'] for t in tasks] != admission['owners']:
        raise ValueError('Candidate owner coverage mismatch')
    if len(rows) != 508 or len(prior) != 508:
        raise ValueError('Fixed event coverage mismatch')
    tasks = {t['order']: t for t in tasks}
    with (PARENT / 'private/EVAL_BINDINGS.json').open('rb') as f:
        config = runtime_binding.extract(f)['configuration_only']
    runtime = load_real_runtime(SimpleNamespace(cpu_gate=PARENT / 'private/cpu_gate'))
    runtime.model.eval(); r.old.fp32(runtime)
    if runtime.generation_config != config['generation']: raise ValueError('Generation configuration differs')
    rng = torch.load(MED / 'GENERATION_RNG.pt', map_location='cpu', weights_only=False)
    reference = json.loads((MED / 'PROTOCOL_LOCK.json').read_text())['source_reference']
    expected_runtime = dict(inherited_runtime=config['runtime'], actual_precision='MATH_FP32_FROZEN_FP16_PREFILL',
                            frozen_FP16_prefill=True)
    def count(_module, _args, _kwargs): budget(); COUNTS['forwards'] += 1
    handle = runtime.model.register_forward_pre_hook(count, with_kwargs=True)
    try:
        for row in rows:
            owner, query = row['owner'], row['query']; original = prior[(owner, query['query_id'])]
            if original['input']['runtime'] != expected_runtime or original['prediction_binding']['model_source'] != reference:
                raise ValueError('Model or runtime binding differs')
            if original['prediction_binding']['generation_rng_binding'] != rng_digest(rng):
                raise ValueError('Generation RNG binding differs')
            key = hashlib.sha256(json.dumps([owner, query['query_id'], 'SOURCE_TEACHER'], separators=(',', ':')).encode()).hexdigest()
            write('eval_attempts/' + key + '.json', dict(time=time.time()), once=True)
            COUNTS['generations'] += 1
            value = evaluate.one(runtime, r, tasks[owner], query, original, ROOT / 'weights' / f'{owner}.pt', states[owner], rng, budget)
            write('eval_outputs/' + key + '.json', value, once=True)
            write('EVAL_PROGRESS.json', dict(completed=COUNTS['generations'], counts=COUNTS))
        if COUNTS['generations'] != 508: raise ValueError('Event count mismatch')
        write('EVAL_COMPLETE.json', dict(status='GENERATION_COMPLETE', events=508, counts=COUNTS))
    finally:
        handle.remove()


if __name__ == '__main__':
    code = 0
    try: main()
    except BaseException:
        code = 1; write('EVAL_FAILURE.json', dict(error=traceback.format_exc(), counts=COUNTS)); raise
    finally: write('EVAL_EXIT.json', dict(code=code, counts=COUNTS))
