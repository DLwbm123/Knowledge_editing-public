"""One frozen prefix-prototype radius; no TT or model updates."""
import json
import os
from pathlib import Path
import random
import signal
import sys
import time
import traceback

ROOT = Path(os.environ['V109_ROOT'])
PARENT = Path(os.environ['PARENT_RUN'])
SELECTED = Path(os.environ['PARENT_SELECTED_ROOT'])
START = time.time()
counts = dict(forwards=0, feature_forwards=0, generations=0, vision_forwards=0, backwards=0, updates=0)


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def radius(positive, negative):
    upper, lower = max(positive), min(negative)
    admitted = lower - upper > 1e-6
    return dict(admitted=admitted, positive_max=upper, negative_min=lower,
                gap=lower - upper, radius=(upper + lower) / 2 if admitted else None)


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
    tasks = read(ROOT / 'TASKS.json')['tasks']
    assert len(tasks) == 14
    runtime = load_real_runtime(SimpleNamespace(cpu_gate=PARENT / 'private/cpu_gate'))
    runtime.model.eval()
    r.old.fp32(runtime)
    assert not any(v.requires_grad for v in runtime.model.parameters())
    bindings = read(PARENT / 'private/EVAL_BINDINGS.json')
    assert runtime.generation_config == next(iter(bindings.values()))['generation']
    template = read(PARENT / 'private/M13_SOURCE_PACKET.json')['original_tasks'][4]
    module = runtime.get_module(r.c.LAYER)
    def count(*_):
        budget()
        counts['forwards'] += 1
    fh = runtime.model.register_forward_pre_hook(count)
    vh = runtime.model.get_vision_tower().register_forward_pre_hook(
        lambda *_: counts.__setitem__('vision_forwards', counts['vision_forwards'] + 1))

    def prepare(rows, training=False):
        runtime.model.half()
        try:
            return r.s.prepare(runtime, bindings, template, rows, training)
        finally:
            r.old.fp32(runtime)

    def prefix(batch):
        position = int(torch.where(batch.labels[0] != -100)[0][0])
        kwargs = {k: v for k, v in batch.forward_kwargs().items() if k != 'labels'}
        for name in ('inputs_embeds', 'attention_mask', 'position_ids'):
            if isinstance(kwargs.get(name), torch.Tensor):
                kwargs[name] = kwargs[name][:, :position]
        kwargs['use_cache'] = False
        return kwargs

    features = {}
    def feature(row):
        identity = row['query_id']
        if identity in features:
            return features[identity]
        item = prepare([row], True)[0]
        batch = r.old.batch_on(runtime, item[1])
        kwargs = prefix(batch)
        if not features:
            alternative = prepare([dict(row, query_id='REFERENCE_EXCLUSION_CHECK', reference='unrelated')], True)[0]
            alternate = prefix(r.old.batch_on(runtime, alternative[1]))
            for name in ('inputs_embeds', 'attention_mask', 'position_ids'):
                if isinstance(kwargs.get(name), torch.Tensor):
                    assert torch.equal(kwargs[name], alternate[name]), 'Reference changed causal prefix'
            write(ROOT / 'REFERENCE_EXCLUSION_ADMISSION.json', dict(prefix_inputs_exact=True,
                targets_not_forwarded=True, extra_language_forwards=0))
        captured = []
        handle = module.register_forward_pre_hook(lambda _, args: captured.append(args[0][0, -1].detach().cpu().double().clone()))
        try:
            counts['feature_forwards'] += 1
            with torch.no_grad():
                output = runtime.model(**kwargs)
            del output
        finally:
            handle.remove()
        assert len(captured) == 1 and captured[0].numel() == 14336
        value = captured[0]
        assert torch.isfinite(value).all() and value.norm() > 0
        features[identity] = value / value.norm()
        write(ROOT / 'PROGRESS.json', dict(phase='FEATURES', completed=len(features), counts=dict(counts), time=time.time()))
        return features[identity]

    def distance(value, anchors):
        return float(torch.stack([(value - anchor).norm() for anchor in anchors]).min())

    positives, negatives, calibration = [], [], []
    for task in tasks:
        if task['role'] != 'DEV_CANDIDATE':
            continue
        anchors = [feature(row) for row in task['train']]
        for index, value in enumerate(anchors):
            positives.append(distance(value, anchors[:index] + anchors[index + 1:]))
        for row in task['protect']:
            assert row['relation'] != task['target']['relation']
            negatives.append(distance(feature(row), anchors))
        calibration.append(task['task_id'])
    assert len(positives) == 40 and len(negatives) == 18 and len(features) == 58
    admission = radius(positives, negatives)
    write(ROOT / 'SCOPE_ADMISSION_FROZEN.json', dict(**admission, calibration_tasks=calibration,
        positive_LOO_distances=positives, negative_distances=negatives,
        heldout_inputs_used=False, time=time.time(), counts=dict(counts)))
    torch.save(features, ROOT / 'FEATURES.pt')
    if not admission['admitted']:
        write(ROOT / 'COMPLETE.json', dict(status='SCALAR_PREFIX_SCOPE_NOT_SEPARABLE',
            generated=False, counts=dict(counts), time=time.time()))
        return
    routes = {}
    for task in tasks:
        anchors = [feature(row) for row in task['train']]
        rows = task['train'] + task['heldout'] + task['protect']
        routes[task['task_id']] = {row['query_id']: dict(distance=distance(feature(row), anchors)) for row in rows}
        for value in routes[task['task_id']].values():
            value['ON'] = value['distance'] <= admission['radius']
    assert len(features) == counts['feature_forwards'] == 126
    write(ROOT / 'ALL_ROUTING_FROZEN.json', dict(routes=routes, admission=admission,
        counts=dict(counts), time=time.time()))
    torch.save(features, ROOT / 'FEATURES.pt')

    def emit(item, hook, on, dest):
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
        write(dest, value)
        return value

    for task in tasks:
        dest = ROOT / task['task_id']
        rows = task['train'] + task['heldout'] + task['protect']
        evaluation = prepare(rows)
        state = torch.load(SELECTED / task['task_id'] / 'delta.pt', map_location=runtime.device, weights_only=True)
        expert = r.p.expert(state, lock['seed'], runtime.device)
        hook = r.q.scoped.replay.MedTraceLayerHook(module, expert)
        hook.attach()
        try:
            smoke = emit(evaluation[0], hook, False, dest / 'OFF_SMOKE.json')
            assert smoke['raw_answer'] == task['Base_target']['raw_answer']
            assert smoke['raw_token_ids'] == task['Base_target']['raw_token_ids']
            for item in evaluation:
                emit(item, hook, routes[task['task_id']][item[0]['query_id']]['ON'],
                     dest / 'outputs' / (item[0]['query_id'] + '.json'))
            write(dest / 'GENERATION_COMPLETE.json', dict(rows=len(evaluation), OFF_Base_exact=True, time=time.time()))
        finally:
            hook.detach()
    assert counts['generations'] == 140 and counts['backwards'] == counts['updates'] == 0
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
