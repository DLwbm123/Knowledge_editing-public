"""One fixed source-generated stopping rule for the original P update."""
import json
import os
from pathlib import Path
import random
import signal
import sys
import time
import traceback

ROOT = Path(os.environ['V108_ROOT'])
PARENT = Path(os.environ['PARENT_RUN'])
START = time.time()
counts = dict(forwards=0, backwards=0, operator_backwards=0, updates=0,
              generations=0, vision_forwards=0)


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def normalize(text):
    return ' '.join(text.strip().lower().split()).rstrip('.')


def source_pass(answer, reference):
    return answer['EOS'] and not answer['at_cap'] and normalize(answer['raw_answer']) == normalize(reference)


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
    tasks = read(ROOT / 'TASKS.json')['tasks']
    assert len(tasks) == 14
    runtime = load_real_runtime(SimpleNamespace(cpu_gate=PARENT / 'private/cpu_gate'))
    runtime.model.eval()
    r.old.fp32(runtime)
    assert not any(v.requires_grad for v in runtime.model.parameters())
    bindings = read(PARENT / 'private/EVAL_BINDINGS.json')
    assert runtime.generation_config == next(iter(bindings.values()))['generation']
    template = read(PARENT / 'private/M13_SOURCE_PACKET.json')['original_tasks'][4]
    fisher = {k: v.to(runtime.device) if isinstance(v, torch.Tensor) else v
              for k, v in r.p.load_state(PARENT / 'private/OPERATOR_FISHER.pt').items()}
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

    def fresh():
        state = r.mm.canonicalize(r.p.tr.TT4(lock['seed'], 8, 8).state_dict())[0]
        expert = r.p.expert(state, lock['seed'], runtime.device)
        hook = r.q.scoped.replay.MedTraceLayerHook(runtime.get_module(r.c.LAYER), expert)
        hook.attach()
        return expert, hook

    def emit(item, hook, path, smoke=False):
        budget()
        counts['generations'] += 1
        if smoke:
            with hook.generation_request():
                answer, _ = r.s.fit.short.generate(runtime, None, item[2], item[3])
            assert hook.last_generation_trace and all(v['active_residual_norm'] == 0 for v in hook.last_generation_trace)
        else:
            answer, _ = r.s.fit.short.generate(runtime, hook, item[2], item[3])
        tokens = list(answer.raw_token_ids)
        value = dict(binding=item[4], raw_answer=answer.decoded_text, raw_token_ids=tokens,
                     EOS=bool(tokens) and tokens[-1] == runtime.adapter.tokenizer.eos_token_id,
                     at_cap=len(tokens) >= 1024)
        write(path, value)
        return value

    selected_steps = {}
    for task in tasks:
        dest = ROOT / task['task_id']
        dest.mkdir()
        prepared = prepare(task['train'], True)
        batches = [r.old.batch_on(runtime, v[1]) for v in prepared]
        selection = prepare(task['train'])
        calibration = read(ROOT / 'reference' / task['task_id'] / 'CALIBRATION.json')
        original = read(ROOT / 'reference' / task['task_id'] / 'P_TRAIN_COMPLETE.json')['curve']
        expert, hook = fresh()
        scalar = dict(step=0, second_moments=[0., 0.])
        curve, checks = [], []
        passed = False
        try:
            smoke = emit(selection[0], hook, dest / 'ZERO_SMOKE.json', True)
            assert smoke['binding'] == task['Base_target']['binding']
            assert smoke['raw_answer'] == task['Base_target']['raw_answer']
            assert smoke['raw_token_ids'] == task['Base_target']['raw_token_ids']
            with PrefixCache(runtime, batches) as prefix:
                for step in range(1, 161):
                    budget()
                    before = r.old.snapshot(expert)
                    expert.zero_grad(set_to_none=True)
                    values = []
                    for index in (0, 1 + (step - 1) % 4):
                        hook.set_teacher_routing(batches[index].labels)
                        loss = runtime.compute_loss(batches[index])
                        assert torch.isfinite(loss)
                        (.5 * loss).backward()
                        counts['backwards'] += 1
                        values.append(float(loss.detach()))
                    assert values == original[step - 1]['source_NLL'], 'P prefix source gradient path changed'
                    gradients = {k: v.grad.detach().clone() for k, v in expert.named_parameters()}
                    state, scalar, audit = matched_math.step(r, before, gradients, fisher,
                        calibration['coefficient'], scalar, calibration['targets'][step - 1], 'M13P')
                    expert.load_state_dict(state)
                    actual = r.fm.map_norm(before, state)
                    old_norm = original[step - 1]['operator_step_norm']
                    assert abs(actual - old_norm) <= 1e-12 + 1e-7 * old_norm
                    counts['updates'] += 1
                    counts['operator_backwards'] += audit['operator_backwards']
                    curve.append(dict(step=step, source_NLL=values, operator_step_norm=actual, **audit))
                    if step % 20 == 0:
                        current = r.old.snapshot(expert)
                        rng = r.c.rng()
                        outputs = [emit(item, hook, dest / 'source_checks' / str(step) /
                                        (item[0]['query_id'] + '.json')) for item in selection]
                        r.c.restore_rng(rng)
                        assert all(torch.equal(v, current[k]) for k, v in r.old.snapshot(expert).items())
                        passes = [source_pass(v, row['reference']) for v, row in zip(outputs, task['train'])]
                        checks.append(dict(step=step, pass_by_source=passes, all_source_pass=all(passes)))
                        passed = all(passes)
                        saved = dict(state=current, scalar=scalar, rng=rng, curve=curve, source_checks=checks,
                                     counts=dict(counts), source_commit=lock['source_commit'], task_id=task['task_id'])
                        torch.save(saved, dest / 'resume.tmp')
                        (dest / 'resume.tmp').replace(dest / 'resume.pt')
                        write(ROOT / 'PROGRESS.json', dict(phase='TRAIN', task=task['task_id'],
                            step=step, source_gate=passed, counts=dict(counts), time=time.time()))
                        if passed:
                            break
                assert prefix.misses == 5 and prefix.hits == 2 * step - 5
            torch.save(r.old.snapshot(expert), dest / 'delta.pt')
            loaded = torch.load(dest / 'delta.pt', map_location=runtime.device, weights_only=True)
            assert all(torch.equal(v, loaded[k]) for k, v in r.old.snapshot(expert).items())
            selected_steps[task['task_id']] = step
            write(dest / 'TRAIN_COMPLETE.json', dict(step=step, source_gate_met=passed, source_checks=checks,
                curve=curve, P_prefix_source_NLL_exact=True, P_prefix_step_norm_tight=True,
                source_only_selection=True, save_load_exact=True, counts=dict(counts), time=time.time()))
        finally:
            hook.detach()
        del batches, prepared, selection
    assert counts['updates'] == sum(selected_steps.values())
    assert counts['backwards'] == 2 * counts['updates'] and counts['operator_backwards'] == counts['updates']
    write(ROOT / 'ALL_TRAINING_FROZEN.json', dict(selected_steps=selected_steps, counts=dict(counts), time=time.time()))
    for task in tasks:
        dest = ROOT / task['task_id']
        evaluation = prepare(task['train'] + task['heldout'] + task['protect'])
        expert, hook = fresh()
        expert.load_state_dict(torch.load(dest / 'delta.pt', map_location=runtime.device, weights_only=True))
        try:
            for item in evaluation:
                emit(item, hook, dest / 'outputs' / (item[0]['query_id'] + '.json'))
            write(dest / 'GENERATION_COMPLETE.json', dict(rows=len(evaluation), time=time.time()))
        finally:
            hook.detach()
    assert counts['generations'] == 14 + counts['updates'] // 20 * 5 + 126
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
