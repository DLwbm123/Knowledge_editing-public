"""Generate a fixed shard after all fifteen remaining experts are trained."""
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
PART = int(os.environ['PARTITION'])
PARTITIONS = int(os.environ['PARTITIONS'])
DEADLINE = 1793447820
COUNTS = dict(forwards=0, generations=0, predictions=0, reused_off=0)


def write_once(name, value):
    p = ROOT / name
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix('.tmp')
    if p.exists() or tmp.exists():
        raise FileExistsError(p)
    tmp.write_text(json.dumps(value, ensure_ascii=False) + '\n')
    tmp.replace(p)


def budget():
    if time.time() >= DEADLINE - 60 or (ROOT / 'BATCH_EVAL_STOP.json').exists():
        raise TimeoutError('Frozen deadline or independent guard stop')
    health = json.loads((ROOT / 'BATCH_EVAL_GUARD_HEALTH.json').read_text())
    if time.time() - health['time'] > 45 or COUNTS['generations'] > 1016:
        raise RuntimeError('Guard or generation budget failed')


def main():
    if PARTITIONS != 4 or PART not in range(4):
        raise ValueError('Invalid fixed shard')
    with (ROOT / f'BATCH_EVAL_WORKER_{PART}_ONCE').open('x'):
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
    from first_eval import rng_digest
    from types import SimpleNamespace
    from scripts.medtrace.run_realmodel_core import load_real_runtime
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    r.c.budget = budget
    if json.loads((ROOT / 'BATCH_GUARD_EXIT.json').read_text())['codes'] != [0, 0, 0, 0]:
        raise ValueError('Four training shards not complete')
    admission = json.loads((ROOT / 'ROLE_ADMISSION.json').read_text())
    tasks = json.loads((ROOT / 'TASK_IDENTITIES_PRIVATE.json').read_text())
    if [t['order'] for t in tasks] != admission['owners']:
        raise ValueError('Fixed owner identity mismatch')
    assigned = {t['order']: t for t in tasks[1:][PART::4]}
    rows = [x for x in json.loads((ROOT / 'EVAL_INPUTS_PRIVATE.json').read_text()) if x['owner'] in assigned]
    prior = {(x['owner'], x['query_id']): x for x in json.loads((ROOT / 'PRIOR_INPUT_BINDINGS_PRIVATE.json').read_text())}
    if not rows or any((x['owner'], x['query']['query_id']) not in prior for x in rows):
        raise ValueError('Original input binding incomplete')
    states = {}
    for i in range(4):
        for item in json.loads((ROOT / f'BATCH_COMPLETE_{i}.json').read_text())['results']:
            key = (item['owner'], item['arm'])
            if key in states:
                raise ValueError('Duplicate frozen state')
            states[key] = item
    if len(states) != 30 or any((o, a) not in states for o in assigned for a in ('C21', 'CDEEP')):
        raise ValueError('Frozen batch state coverage incomplete')
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
    def count(_module, _args, _kwargs):
        budget()
        COUNTS['forwards'] += 1
    handle = runtime.model.register_forward_pre_hook(count, with_kwargs=True)
    try:
        for row in rows:
            owner, q = row['owner'], row['query']
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
                state_path = ROOT / 'batch' / f'{owner}_{arm}.pt'
                state = torch.load(state_path, map_location='cpu', weights_only=True)
                if train_pair.state_digest(state['state']) != states[(owner, arm)]['final_digest']:
                    raise ValueError('Frozen state identity mismatch')
                key = hashlib.sha256(json.dumps([owner, q['query_id'], arm], separators=(',', ':')).encode()).hexdigest()
                write_once('batch_eval_attempts/' + key + '.json', dict(time=time.time(), arm=arm))
                if not route['activated'] and off_answer is not None:
                    value = dict(off_answer, arm=arm, module=state['module'], state_file=state_path.name,
                                 prediction_reused_from='C21_OFF')
                    COUNTS['reused_off'] += 1
                else:
                    COUNTS['generations'] += 1
                    value = generate.generate_one(runtime, r, assigned[owner], q, route, binding, route, state_path, arm, rng, budget)
                    value['prediction_reused_from'] = None
                    if not route['activated']:
                        off_answer = value
                COUNTS['predictions'] += 1
                value.update(owner=owner, original_consumer_id=original['consumer_id'],
                             original_execution_binding=original['prediction_binding'], original_route=old_route)
                write_once('batch_eval_outputs/' + key + '.json', value)
            progress = ROOT / f'BATCH_EVAL_PROGRESS_{PART}.json'
            temporary = progress.with_suffix('.tmp')
            temporary.write_text(json.dumps(dict(time=time.time(), completed=COUNTS['predictions'], total=2 * len(rows))))
            temporary.replace(progress)
        if COUNTS['predictions'] != 2 * len(rows):
            raise ValueError('Prediction coverage mismatch')
        write_once(f'BATCH_EVAL_COMPLETE_{PART}.json', dict(time=time.time(), counts=COUNTS, owners=len(assigned), events=len(rows)))
    finally:
        handle.remove()


if __name__ == '__main__':
    code = 0
    try:
        main()
    except BaseException:
        code = 1
        write_once(f'BATCH_EVAL_FAILURE_{PART}.json', dict(time=time.time(), error=traceback.format_exc(), counts=COUNTS))
        raise
    finally:
        write_once(f'BATCH_EVAL_EXIT_{PART}.json', dict(time=time.time(), code=code, counts=COUNTS))
