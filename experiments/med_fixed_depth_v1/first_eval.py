"""Evaluate the already trained first pair on fixed original single events."""
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
OLD = Path(os.environ['RAW_MED_ROOT'])
DEADLINE = 1793447820
COUNTS = dict(forwards=0, generations=0, predictions=0, reused_off=0)


def write(name, value):
    p = ROOT / name
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix('.tmp')
    if p.exists() or tmp.exists():
        raise FileExistsError(p)
    tmp.write_text(json.dumps(value, ensure_ascii=False) + '\n')
    tmp.replace(p)


def budget():
    if time.time() >= DEADLINE - 60 or (ROOT / 'FIRST_EVAL_STOP.json').exists():
        raise TimeoutError('Frozen deadline or guard stop')
    health = json.loads((ROOT / 'FIRST_EVAL_GUARD_HEALTH.json').read_text())
    if time.time() - health['time'] > 45:
        raise RuntimeError('Independent guard heartbeat stale')
    if COUNTS['generations'] > 1016:
        raise RuntimeError('Generation budget exceeded')


def rng_digest(state):
    import torch
    h = hashlib.sha256()
    for key in sorted(state['rng']):
        value = state['rng'][key]
        h.update(key.encode())
        h.update(value.cpu().numpy().tobytes() if isinstance(value, torch.Tensor) else repr(value).encode())
    value = state['numpy_rng']
    h.update(value[0].encode())
    h.update(value[1].tobytes())
    h.update(repr(value[2:]).encode())
    return h.hexdigest()


def main():
    with (ROOT / 'FIRST_EVAL_WORKER_ONCE').open('x'):
        pass
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(InterruptedError('Guard stop')))
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('Frozen deadline')))
    signal.alarm(max(1, int(DEADLINE - time.time() - 60)))
    sys.path[:0] = [str(ROOT / 'tools'), str(PARENT / 'private/tools'), str(PARENT / 'private/source'),
                    str(PARENT / 'private/official_llava')]
    import torch
    import module_run as r
    import generate
    import train_pair
    import runtime_binding
    from types import SimpleNamespace
    from scripts.medtrace.run_realmodel_core import load_real_runtime
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    r.c.budget = budget
    first = json.loads((ROOT / 'FIRST_PAIR_TRAIN_COMPLETE.json').read_text())
    if first['counts']['updates'] != 320 or json.loads((ROOT / 'WORKER_EXIT.json').read_text())['code'] != 0:
        raise ValueError('First pair training incomplete')
    if json.loads((ROOT / 'GUARD_EXIT.json').read_text())['code'] != 0:
        raise ValueError('Training guard failed')
    admission = json.loads((ROOT / 'ROLE_ADMISSION.json').read_text())
    sources = json.loads((ROOT / 'TRAIN_SOURCE_BOUND_PRIVATE.json').read_text())
    owner = admission['owners'][0]
    task = sources[0]['task']
    if task['order'] != owner:
        raise ValueError('First owner mismatch')
    rows = [x for x in json.loads((ROOT / 'EVAL_INPUTS_PRIVATE.json').read_text()) if x['owner'] == owner]
    prior = {(x['owner'], x['query_id']): x for x in json.loads((ROOT / 'PRIOR_INPUT_BINDINGS_PRIVATE.json').read_text())}
    if not rows or any((owner, x['query']['query_id']) not in prior for x in rows):
        raise ValueError('Original input binding incomplete')
    with (PARENT / 'private/EVAL_BINDINGS.json').open('rb') as f:
        config = runtime_binding.extract(f)['configuration_only']
    runtime = load_real_runtime(SimpleNamespace(cpu_gate=PARENT / 'private/cpu_gate'))
    runtime.model.eval()
    r.old.fp32(runtime)
    if runtime.generation_config != config['generation']:
        raise ValueError('Generation configuration differs')
    rng = torch.load(OLD / 'GENERATION_RNG.pt', map_location='cpu', weights_only=False)
    digest = rng_digest(rng)
    reference = json.loads((OLD / 'PROTOCOL_LOCK.json').read_text())['source_reference']
    expected_runtime = dict(inherited_runtime=config['runtime'], actual_precision='MATH_FP32_FROZEN_FP16_PREFILL',
                            frozen_FP16_prefill=True)
    states = {v['arm']: v for v in first['results']}
    def count(_module, _args, _kwargs):
        budget()
        COUNTS['forwards'] += 1
    handle = runtime.model.register_forward_pre_hook(count, with_kwargs=True)
    try:
        for row in rows:
            q = row['query']
            original = prior[(owner, q['query_id'])]
            if original['input']['runtime'] != expected_runtime or original['prediction_binding']['model_source'] != reference:
                raise ValueError('Original model or runtime binding differs')
            if original['prediction_binding']['generation_rng_binding'] != digest:
                raise ValueError('Original RNG binding differs')
            old_route = original['route']
            route = dict(activated=old_route['activated'], logical_edit_id=old_route['logical_edit_id'] if old_route['activated'] else None)
            binding = {k: original['input'][k] for k in ('question', 'image_sha256', 'prompt_ids', 'attention_mask', 'generation')}
            off_answer = None
            for arm in ('C21', 'CDEEP'):
                budget()
                state_path = ROOT / 'first_pair' / f'{owner}_{arm}.pt'
                state = torch.load(state_path, map_location='cpu', weights_only=True)
                if train_pair.state_digest(state['state']) != states[arm]['final_digest']:
                    raise ValueError('Frozen state identity mismatch')
                key = hashlib.sha256(json.dumps([owner, q['query_id'], arm], separators=(',', ':')).encode()).hexdigest()
                marker = ROOT / 'first_eval_attempts' / (key + '.json')
                write(str(marker.relative_to(ROOT)), dict(time=time.time(), arm=arm))
                if not route['activated'] and off_answer is not None:
                    value = dict(off_answer, arm=arm, module=state['module'], state_file=state_path.name,
                                 prediction_reused_from='C21_OFF')
                    COUNTS['reused_off'] += 1
                else:
                    COUNTS['generations'] += 1
                    value = generate.generate_one(runtime, r, task, q, route, binding, route, state_path, arm, rng, budget)
                    value['prediction_reused_from'] = None
                    if not route['activated']:
                        off_answer = value
                COUNTS['predictions'] += 1
                value.update(owner=owner, original_consumer_id=original['consumer_id'],
                             original_execution_binding=original['prediction_binding'], original_route=old_route)
                write('first_eval_outputs/' + key + '.json', value)
            progress = ROOT / 'FIRST_EVAL_PROGRESS.json'
            temporary = progress.with_suffix('.tmp')
            temporary.write_text(json.dumps(dict(time=time.time(), completed=COUNTS['predictions'], total=2 * len(rows))))
            temporary.replace(progress)
        if COUNTS['predictions'] != 2 * len(rows):
            raise ValueError('Generation event count mismatch')
        write('FIRST_PAIR_EVAL_COMPLETE.json', dict(time=time.time(), engineering_status='PASS',
              owners=1, source_events=len(rows), predictions=COUNTS['predictions'], generated=COUNTS['generations'], counts=COUNTS,
              reference_answers_read=False, scores_read=False))
    finally:
        handle.remove()


if __name__ == '__main__':
    code = 0
    try:
        main()
    except BaseException:
        code = 1
        write('FIRST_EVAL_FAILURE.json', dict(time=time.time(), error=traceback.format_exc(), counts=COUNTS))
        raise
    finally:
        write('FIRST_EVAL_EXIT.json', dict(time=time.time(), code=code, counts=COUNTS))
