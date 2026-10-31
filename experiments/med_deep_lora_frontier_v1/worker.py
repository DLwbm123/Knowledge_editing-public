"""Fixed first edit, engineering prediction, remaining edits, then four-point panel."""
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
COUNTS = dict(forwards=0, backwards=0, updates=0, generations=0, vision_forwards=0, operator_backwards=0)


def write(path, value, once=False):
    path = ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    if once:
        with path.open('x') as f:
            json.dump(value, f, ensure_ascii=False)
        path.chmod(0o600)
    else:
        tmp = path.with_suffix('.tmp')
        tmp.write_text(json.dumps(value, ensure_ascii=False))
        tmp.replace(path)


def budget():
    if time.time() >= DEADLINE - 60 or (ROOT / 'STOP.json').exists():
        raise TimeoutError('Original deadline or independent guard stop')
    health = json.loads((ROOT / 'GUARD_HEALTH.json').read_text())
    if time.time() - health['time'] > 45 or health['hard_end'] != DEADLINE:
        raise RuntimeError('Independent guard unavailable')
    if COUNTS['updates'] > 2560 or COUNTS['backwards'] > 5120 or COUNTS['generations'] > 2032:
        raise RuntimeError('Fixed budget exceeded')


def main():
    with (ROOT / 'WORKER_ONCE').open('x'):
        pass
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(InterruptedError('Guard stop')))
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('Deadline')))
    signal.alarm(max(1, int(DEADLINE-time.time()-60)))
    sys.path[:0] = [str(ROOT/'tools'), str(PARENT/'private/tools'), str(PARENT/'private/source'),
                    str(PARENT/'private/official_llava')]
    import torch
    import module_run as r
    import runtime_binding
    import lora_train
    import generate
    from first_eval import rng_digest
    from types import SimpleNamespace
    from scripts.medtrace.run_realmodel_core import load_real_runtime
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    r.c.budget, r.COUNTS = budget, COUNTS
    admission = json.loads((FIXED/'ROLE_ADMISSION.json').read_text())
    source = json.loads((FIXED/'TRAIN_SOURCE_BOUND_PRIVATE.json').read_text())
    tasks = json.loads((FIXED/'TASK_IDENTITIES_PRIVATE.json').read_text())
    rows = json.loads((FIXED/'EVAL_INPUTS_PRIVATE.json').read_text())
    prior = {(x['owner'], x['query_id']): x for x in json.loads((FIXED/'PRIOR_INPUT_BINDINGS_PRIVATE.json').read_text())}
    owners = admission['owners']
    if (admission['status'] != 'SOURCE_AND_EVALUATION_ROLE_ADMITTED' or len(owners) != 16
        or [x['task']['order'] for x in source] != owners or [x['order'] for x in tasks] != owners
        or len(rows) != 508 or len(prior) != 508):
        raise ValueError('Fixed panel identity or admission mismatch')
    taskmap = {x['order']: x for x in tasks}
    with (PARENT/'private/EVAL_BINDINGS.json').open('rb') as f:
        config = runtime_binding.extract(f)['configuration_only']
    runtime = load_real_runtime(SimpleNamespace(cpu_gate=PARENT/'private/cpu_gate'))
    runtime.model.eval()
    r.old.fp32(runtime)
    if runtime.generation_config != config['generation']:
        raise ValueError('Original generation config differs')
    rng = torch.load(MED/'GENERATION_RNG.pt', map_location='cpu', weights_only=False)
    rng_id = rng_digest(rng)
    model_source = json.loads((MED/'PROTOCOL_LOCK.json').read_text())['source_reference']
    expected_runtime = dict(inherited_runtime=config['runtime'], actual_precision='MATH_FP32_FROZEN_FP16_PREFILL',
                            frozen_FP16_prefill=True)
    def count(_module, _args, _kwargs):
        budget()
        COUNTS['forwards'] += 1
    handle = runtime.model.register_forward_pre_hook(count, with_kwargs=True)
    completed = []
    try:
        def fit(entry):
            result = lora_train.train(runtime, r, entry['task'], entry['source_rows'], {'configuration_only': config},
                                 admission, ROOT/'weights', budget)
            completed.append(result)
            write('TRAIN_PROGRESS.json', dict(completed=len(completed), counts=COUNTS))

        def predict(row, gamma):
            owner, query = row['owner'], row['query']
            old = prior[(owner, query['query_id'])]
            if (old['input']['runtime'] != expected_runtime or old['prediction_binding']['model_source'] != model_source
                or old['prediction_binding']['generation_rng_binding'] != rng_id):
                raise ValueError('Original runtime or RNG binding differs')
            original = old['route']
            route = dict(activated=original['activated'], logical_edit_id=original['logical_edit_id'] if original['activated'] else None)
            binding = {k: old['input'][k] for k in ('question','image_sha256','prompt_ids','attention_mask','generation')}
            key = hashlib.sha256(json.dumps([owner, query['query_id'], gamma], separators=(',', ':')).encode()).hexdigest()
            write('attempts/'+key+'.json', dict(time=time.time(), gamma=gamma), once=True)
            COUNTS['generations'] += 1
            value = generate.generate_one(runtime, r, taskmap[owner], query, route, binding, route,
                ROOT/'weights'/f'{owner}.pt', 'DEEP_LORA_R4', rng, budget, gamma)
            value.update(owner=owner, original_consumer_id=old['consumer_id'], original_execution_binding=old['prediction_binding'],
                         original_route=original, prediction_reused_from=None)
            write('predictions/'+key+'.json', value, once=True)

        fit(source[0])
        first = next(x for x in rows if x['owner'] == owners[0])
        predict(first, 1.0)
        write('FIRST_ENGINEERING_COMPLETE.json', dict(status='PASS', counts=COUNTS))
        for entry in source[1:]:
            fit(entry)
        if COUNTS['updates'] != 2560 or COUNTS['backwards'] != 5120:
            raise ValueError('Training count mismatch')
        write('TRAIN_FROZEN.json', dict(status='FROZEN', results=completed, counts=COUNTS))
        for row in rows:
            for gamma in (.25, .5, .75, 1.):
                if row is first and gamma == 1.0:
                    continue
                predict(row, gamma)
            write('EVAL_PROGRESS.json', dict(predictions=COUNTS['generations'], expected=2032))
        if COUNTS['generations'] != 2032:
            raise ValueError('Prediction count mismatch')
        write('EVAL_COMPLETE.json', dict(status='GENERATION_COMPLETE', events=508,
                                          predictions=2032, counts=COUNTS))
    finally:
        handle.remove()


if __name__ == '__main__':
    code = 0
    try:
        main()
    except BaseException:
        code = 1
        write('WORKER_FAILURE.json', dict(error=traceback.format_exc(), counts=COUNTS))
        raise
    finally:
        write('WORKER_EXIT.json', dict(code=code, counts=COUNTS))
