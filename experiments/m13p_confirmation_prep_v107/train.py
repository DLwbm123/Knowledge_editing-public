"""Frozen fourteen-task source calibration and matched B/P/S, then generation."""
import copy
import json
import os
from pathlib import Path
import random
import signal
import sys
import time
import traceback

ROOT = Path(os.environ['V107_ROOT'])
PARENT = Path(os.environ['PARENT_RUN'])
START = time.time()
counts = dict(forwards=0, backwards=0, operator_backwards=0, updates=0,
              generations=0, vision_forwards=0, cache_parity_backwards=0)


def read(path):
    return json.loads(Path(path).read_text())


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2) + '\n')
    tmp.replace(path)


lock = read(ROOT / 'PROTOCOL_LOCK.json')


def budget():
    assert time.time() < lock['hard_end'] - 15 and not (ROOT / 'STOP.json').exists()
    assert time.time() - read(ROOT / 'GUARD_HEALTH.json')['time'] < 30
    for key, maximum in lock['maximum_counts'].items():
        assert counts[key] <= maximum, (key, counts[key], maximum)


def main():
    with (ROOT / 'WORKER_ONCE').open('x'):
        pass
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('Original hard end')))
    signal.alarm(max(1, int(min(86400 - 15, lock['hard_end'] - time.time() - 15))))
    sys.path[:0] = [str(ROOT / 'tools'), str(PARENT / 'private/tools'),
                   str(PARENT / 'private/source'), str(PARENT / 'private/official_llava')]
    import torch
    import module_run as r
    import matched_math
    from prefix_cache import PrefixCache
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
    cfg = read(ROOT / 'TASKS.json')
    tasks = cfg['tasks']
    assert len(tasks) == 14 and sum(t['role'] == 'CONFIRM_CANDIDATE' for t in tasks) == 6
    basis, _ = r.q.split()
    assert not {t['source_group'] for t in tasks} & {v['source_group'].split(':')[-1] for v in basis}
    runtime = load_real_runtime(SimpleNamespace(cpu_gate=PARENT / 'private/cpu_gate'))
    runtime.model.eval()
    r.old.fp32(runtime)
    assert not any(p.requires_grad for p in runtime.model.parameters())
    bindings = read(PARENT / 'private/EVAL_BINDINGS.json')
    assert runtime.generation_config == next(iter(bindings.values()))['generation']
    template = read(PARENT / 'private/M13_SOURCE_PACKET.json')['original_tasks'][4]
    cache = r.p.load_state(PARENT / 'private/OPERATOR_FISHER.pt')
    cache = {k: v.to(runtime.device) if isinstance(v, torch.Tensor) else v for k, v in cache.items()}
    assert cache['x'].shape == (1514, 14336) and cache['y'].shape == (1514, 32, 4096)
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

    def fresh(canonical):
        initial = r.p.tr.TT4(lock['seed'], 8, 8).state_dict()
        if canonical:
            initial = r.mm.canonicalize(initial)[0]
        expert = r.p.expert(initial, lock['seed'], runtime.device)
        assert sum(p.numel() for p in expert.parameters()) == 7168
        hook = r.q.scoped.replay.MedTraceLayerHook(runtime.get_module(r.c.LAYER), expert)
        hook.attach()
        return expert, hook

    def emit(item, hook, path):
        budget()
        counts['generations'] += 1
        answer, _ = r.s.fit.short.generate(runtime, hook, item[2], item[3])
        tokens = list(answer.raw_token_ids)
        value = dict(binding=item[4], raw_answer=answer.decoded_text, raw_token_ids=tokens,
                     EOS=bool(tokens) and tokens[-1] == runtime.adapter.tokenizer.eos_token_id,
                     at_cap=len(tokens) >= 1024)
        write(path, value)
        return value

    def loss(batch, hook):
        hook.set_teacher_routing(batch.labels)
        value = runtime.compute_loss(batch)
        assert torch.isfinite(value)
        return value

    def gradients(expert, hook, batch):
        value = loss(batch, hook)
        gs = torch.autograd.grad(value, tuple(expert.parameters()))
        counts['backwards'] += 1
        counts['cache_parity_backwards'] += 1
        return float(value.detach()), [v.detach().clone() for v in gs]

    for task_index, task in enumerate(tasks):
        dest = ROOT / task['task_id']
        dest.mkdir()
        assert len(task['train']) == 5 and len(task['heldout']) == 2
        prepared = prepare(task['train'], True)
        batches = [r.old.batch_on(runtime, item[1]) for item in prepared]
        native = prepare([task['target']])[0]
        base = task['Base_target']
        assert native[4] == base['binding']
        targets = coefficient = None
        for arm in lock['arms']:
            arm_dir = dest / arm
            arm_dir.mkdir()
            expert, hook = fresh(arm != 'C')
            initial = r.old.snapshot(expert)
            curve = []
            scalar = dict(step=0, second_moments=[0., 0.]) if arm != 'C' else None
            optimizer = r.p.optimizer(expert, runtime.model) if arm == 'C' else None
            try:
                smoke = emit(native, hook, arm_dir / 'ZERO_SMOKE.json')
                assert smoke['raw_token_ids'] == base['raw_token_ids'] and smoke['raw_answer'] == base['raw_answer']
                if task_index == 0 and arm in ('C', 'M13B'):
                    for nonzero in (False, True):
                        expert.load_state_dict(initial)
                        if nonzero:
                            with torch.no_grad():
                                expert.G1.fill_(.001)
                        refs = [gradients(expert, hook, b) for b in batches]
                        with PrefixCache(runtime, batches):
                            for b, expected in zip(batches, refs):
                                actual = gradients(expert, hook, b)
                                assert actual[0] == expected[0]
                                assert all(torch.equal(a, b) for a, b in zip(actual[1], expected[1]))
                    expert.load_state_dict(initial)
                    write(arm_dir / 'CACHE_ADMISSION.json', dict(zero_and_nonzero_exact=True,
                        all_five_source_gradients=True, counts=dict(counts)))
                with torch.no_grad():
                    initial_losses = [float(loss(b, hook)) for b in batches]
                with PrefixCache(runtime, batches) as prefix:
                    for step in range(1, 161):
                        budget()
                        before = r.old.snapshot(expert)
                        expert.zero_grad(set_to_none=True)
                        values = []
                        for index in (0, 1 + (step - 1) % 4):
                            value = loss(batches[index], hook)
                            (.5 * value).backward()
                            counts['backwards'] += 1
                            values.append(float(value.detach()))
                        source = {k: v.grad.detach().clone() for k, v in expert.named_parameters()}
                        if arm == 'C':
                            selected, audit = matched_math.raw_step(r, expert, optimizer, source, None, 0., None, 'C')
                        else:
                            selected, scalar, audit = matched_math.step(r, before, source, cache,
                                coefficient, scalar, targets[step - 1], arm)
                            expert.load_state_dict(selected)
                        counts['updates'] += 1
                        counts['operator_backwards'] += audit['operator_backwards']
                        curve.append(dict(step=step, source_NLL=values,
                            operator_step_norm=r.fm.map_norm(before, selected), **audit))
                        if step % 20 == 0:
                            saved = dict(state=r.old.snapshot(expert), scalar=copy.deepcopy(scalar),
                                optimizer=copy.deepcopy(optimizer.state_dict()) if optimizer else None,
                                step=step, curve=curve, counts=dict(counts), task=task['task_id'], arm=arm,
                                seed=lock['seed'], rng=r.c.rng(), source_commit=lock['source_commit'])
                            torch.save(saved, arm_dir / 'resume.tmp')
                            (arm_dir / 'resume.tmp').replace(arm_dir / 'resume.pt')
                            write(ROOT / 'PROGRESS.json', dict(phase='TRAIN', task=task['task_id'],
                                role=task['role'], arm=arm, step=step, counts=dict(counts), time=time.time()))
                    assert prefix.misses == 5 and prefix.hits == 315
                with torch.no_grad():
                    final_losses = [float(loss(b, hook)) for b in batches]
                torch.save(r.old.snapshot(expert), arm_dir / 'delta.pt')
                loaded = torch.load(arm_dir / 'delta.pt', map_location=runtime.device, weights_only=True)
                assert all(torch.equal(v, loaded[k]) for k, v in r.old.snapshot(expert).items())
                expert.load_state_dict(loaded)
                if arm == 'C':
                    gain = .5 * (initial_losses[0] - final_losses[0]) + .125 * sum(
                        a - b for a, b in zip(initial_losses[1:], final_losses[1:]))
                    penalty = float(r.mm.operator_penalty(loaded, cache))
                    coefficient = matched_math.coefficient(gain, penalty)
                    targets = [v['operator_step_norm'] for v in curve]
                    write(dest / 'CALIBRATION.json', dict(C_source_gain=gain, C_operator_penalty=penalty,
                        coefficient=coefficient, targets=targets, heldout_used=False))
                write(arm_dir / 'TRAIN_COMPLETE.json', dict(steps=160, curve=curve,
                    initial_losses=initial_losses, final_losses=final_losses, save_load_exact=True,
                    counts=dict(counts), time=time.time()))
            finally:
                hook.detach()
        del batches, prepared, native
    assert counts['updates'] == 8960 and counts['backwards'] == 17960
    assert counts['operator_backwards'] == 6720 and counts['cache_parity_backwards'] == 40
    write(ROOT / 'ALL_TRAINING_FROZEN.json', dict(tasks=14, steps_per_arm=160,
        arms=lock['arms'], counts=dict(counts), time=time.time()))
    # No trained-arm answer is decoded before all trajectories are frozen.
    for task in tasks:
        rows = task['train'] + task['heldout'] + task['protect']
        evaluation = prepare(rows)
        for item in evaluation[5:7]:
            emit(item, None, ROOT / task['task_id'] / 'BASE' / (item[0]['query_id'] + '.json'))
        for arm in lock['arms'][1:]:
            arm_dir = ROOT / task['task_id'] / arm
            expert, hook = fresh(True)
            expert.load_state_dict(torch.load(arm_dir / 'delta.pt', map_location=runtime.device, weights_only=True))
            try:
                for item in evaluation:
                    emit(item, hook, arm_dir / 'outputs' / (item[0]['query_id'] + '.json'))
                write(arm_dir / 'GENERATION_COMPLETE.json', dict(rows=len(rows), counts=dict(counts), time=time.time()))
            finally:
                hook.detach()
        write(ROOT / 'PROGRESS.json', dict(phase='GENERATE', task=task['task_id'], counts=dict(counts), time=time.time()))
    assert counts['generations'] == 462
    write(ROOT / 'COMPLETE.json', dict(status='BLIND_REVIEW_PENDING', counts=counts, time=time.time()))
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
