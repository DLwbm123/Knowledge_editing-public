"""Maximum coverage radius under the unchanged explicit calibration negatives."""
import json
import math
import os
from pathlib import Path
import random
import signal
import sys
import time
import traceback

ROOT = Path(os.environ['V110_ROOT'])
PARENT = Path(os.environ['PARENT_RUN'])
SELECTED = Path(os.environ['PARENT_SELECTED_ROOT'])
SCOPED = Path(os.environ['PARENT_SCOPED_ROOT'])
START = time.time()
counts = dict(forwards=0, feature_forwards=0, generations=0, vision_forwards=0, backwards=0, updates=0)


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def maximal_radius(positive, negative):
    assert positive and negative and all(math.isfinite(v) for v in positive + negative)
    value = min(negative) - 1e-6
    return dict(admitted=max(positive) <= value, positive_max=max(positive),
                negative_min=min(negative), radius=value, numeric_margin=1e-6)


lock = read(ROOT / 'PROTOCOL_LOCK.json')


def budget():
    assert time.time() < lock['hard_end'] - 15 and not (ROOT / 'STOP.json').exists()
    assert time.time() - read(ROOT / 'GUARD_HEALTH.json')['time'] < 30
    for key, maximum in lock['maximum_counts'].items():
        assert counts[key] <= maximum, (key, counts[key], maximum)


def main():
    with (ROOT / 'WORKER_ONCE').open('x'):
        pass
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('Frozen wall limit')))
    signal.alarm(max(1, int(lock['hard_end'] - time.time() - 15)))
    old = read(SCOPED / 'SCOPE_ADMISSION_FROZEN.json')
    assert old['heldout_inputs_used'] is False
    assert len(old['positive_LOO_distances']) == 40 and len(old['negative_distances']) == 18
    admission = maximal_radius(old['positive_LOO_distances'], old['negative_distances'])
    # Freeze the calibration solution before loading any remaining query distance.
    write(ROOT / 'RADIUS_FROZEN.json', dict(**admission, calibration_only=True,
          parent_scope_protocol='193a12ea', time=time.time()))
    if not admission['admitted']:
        write(ROOT / 'COMPLETE.json', dict(status='RADIUS_CONSTRAINT_INFEASIBLE', counts=counts, time=time.time()))
        return
    routes = read(SCOPED / 'ALL_ROUTING_FROZEN.json')['routes']
    tasks = read(ROOT / 'TASKS.json')['tasks']
    for task in tasks:
        for row in task['train'] + task['heldout'] + task['protect']:
            value = routes[task['task_id']][row['query_id']]
            value['ON'] = value['distance'] <= admission['radius']
    write(ROOT / 'ALL_ROUTING_FROZEN.json', dict(routes=routes, admission=admission,
          cached_Base_features=True, inference_answers_used=False, time=time.time()))
    sys.path[:0] = [str(PARENT / 'private/tools'), str(PARENT / 'private/source'), str(PARENT / 'private/official_llava')]
    import torch
    import module_run as r
    from scripts.medtrace.run_realmodel_core import load_real_runtime
    from types import SimpleNamespace
    r.c.budget = budget
    random.seed(lock['seed'])
    torch.manual_seed(lock['seed'])
    torch.cuda.manual_seed_all(lock['seed'])
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    runtime = load_real_runtime(SimpleNamespace(cpu_gate=PARENT / 'private/cpu_gate'))
    runtime.model.eval()
    r.old.fp32(runtime)
    assert not any(v.requires_grad for v in runtime.model.parameters())
    bindings = read(PARENT / 'private/EVAL_BINDINGS.json')
    assert runtime.generation_config == next(iter(bindings.values()))['generation']
    template = read(PARENT / 'private/M13_SOURCE_PACKET.json')['original_tasks'][4]
    def count(*_):
        budget()
        counts['forwards'] += 1
    fh = runtime.model.register_forward_pre_hook(count)
    vh = runtime.model.get_vision_tower().register_forward_pre_hook(
        lambda *_: counts.__setitem__('vision_forwards', counts['vision_forwards'] + 1))
    def prepare(rows):
        runtime.model.half()
        try:
            return r.s.prepare(runtime, bindings, template, rows, False)
        finally:
            r.old.fp32(runtime)
    def emit(item, hook, on, destination):
        counts['generations'] += 1
        if not on:
            hook.clear_request_routing()
            assert not hook.enabled
        answer, _ = r.s.fit.short.generate(runtime, hook if on else None, item[2], item[3])
        if not on:
            assert not hook.enabled
        tokens = list(answer.raw_token_ids)
        value = dict(binding=item[4], raw_answer=answer.decoded_text, raw_token_ids=tokens,
                     EOS=bool(tokens) and tokens[-1] == runtime.adapter.tokenizer.eos_token_id,
                     at_cap=len(tokens) >= 1024, edit_enabled=on)
        write(destination, value)
        return value
    for task in tasks:
        dest = ROOT / task['task_id']
        rows = task['train'] + task['heldout'] + task['protect']
        evaluation = prepare(rows)
        state = torch.load(SELECTED / task['task_id'] / 'delta.pt', map_location=runtime.device, weights_only=True)
        expert = r.p.expert(state, lock['seed'], runtime.device)
        hook = r.q.scoped.replay.MedTraceLayerHook(runtime.get_module(r.c.LAYER), expert)
        hook.attach()
        try:
            smoke = emit(evaluation[0], hook, False, dest / 'OFF_SMOKE.json')
            assert smoke['raw_answer'] == task['Base_target']['raw_answer']
            assert smoke['raw_token_ids'] == task['Base_target']['raw_token_ids']
            for item in evaluation:
                emit(item, hook, routes[task['task_id']][item[0]['query_id']]['ON'],
                     dest / 'outputs' / (item[0]['query_id'] + '.json'))
            write(dest / 'GENERATION_COMPLETE.json', dict(rows=len(evaluation), OFF_Base_exact=True, time=time.time()))
            write(ROOT / 'PROGRESS.json', dict(task=task['task_id'], counts=dict(counts), time=time.time()))
        finally:
            hook.detach()
    assert counts['generations'] == 140 and counts['feature_forwards'] == counts['backwards'] == counts['updates'] == 0
    write(ROOT / 'COMPLETE.json', dict(status='BLIND_REVIEW_PENDING', counts=dict(counts), time=time.time()))
    fh.remove()
    vh.remove()


if __name__ == '__main__':
    code = 0
    try:
        main()
    except BaseException:
        code = 1
        write(ROOT / 'FAILURE.json', dict(error=traceback.format_exc(), counts=counts, time=time.time()))
        raise
    finally:
        write(ROOT / 'WORKER_EXIT.json', dict(code=code, counts=counts, seconds=time.time() - START, time=time.time()))
