"""Matched double-precision TT control versus a fixed Base-path null input map."""
import json
import os
from pathlib import Path
import random
import signal
import sys
import time
import traceback

ROOT = Path(os.environ['V118_ROOT'])
PARENT = Path(os.environ['PARENT_RUN'])
START = time.time()
ARMS = ('DOUBLE_CONTROL', 'PATH_NULL')
counts = dict(forwards=0, backwards=0, operator_backwards=0, updates=0,
              generations=0, vision_forwards=0, path_positions=0)


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
    import kernel
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
    write(ROOT / 'KERNEL_SELF_CHECK.json', kernel.selfcheck())
    tasks = read(ROOT / 'TASKS.json')['tasks']
    assert len(tasks) == 14 and sum(len(t['protect_extra']) for t in tasks) == 56
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

    def fresh(q):
        state = r.mm.canonicalize(r.p.tr.TT4(lock['seed'], 8, 8).state_dict())[0]
        expert = r.p.expert(state, lock['seed'], runtime.device)
        kernel.install(expert, q)
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

    # Collect actual Base trajectories before any TT is attached. New paraphrases never enter Q.
    audits = {}
    for task in tasks:
        dest = ROOT / task['task_id']
        dest.mkdir()
        vectors = []
        for item in prepare(task['protect']):
            captured = []
            def capture(_module, args):
                assert args[0].shape[0] == 1
                captured.append(args[0][0, -1:].detach().cpu().double())
            handle = runtime.get_module(r.c.LAYER).register_forward_pre_hook(capture)
            try:
                answer = emit(item, None, dest / 'BASE' / (item[0]['query_id'] + '.json'))
            finally:
                handle.remove()
            previous = task['Base_protect'][item[0]['query_id']]
            assert answer['raw_answer'] == previous['raw_answer'] and answer['raw_token_ids'] == previous['raw_token_ids']
            assert answer['EOS'] and not answer['at_cap'] and len(captured) == len(answer['raw_token_ids'])
            counts['path_positions'] += len(captured)
            budget()
            vectors.extend(captured)
        q, audit = kernel.basis(vectors)
        assert sum(v['rank'] for v in audits.values()) + audit['rank'] <= 2048
        audits[task['task_id']] = audit
        torch.save(q, dest / 'basis.pt')
        write(dest / 'BASIS_ADMISSION.json', dict(**audit, old_Base_tokens_exact=True,
            new_paraphrases_used=False, counts=dict(counts), time=time.time()))
        del q, vectors
        for item in prepare(task['protect_extra']):
            emit(item, None, dest / 'BASE' / (item[0]['query_id'] + '.json'))
        write(ROOT / 'PROGRESS.json', dict(phase='BASE_PATHS', task=task['task_id'],
            completed=len(audits), counts=dict(counts), time=time.time()))
    assert counts['generations'] == 84
    write(ROOT / 'ALL_BASES_FROZEN.json', dict(audits=audits, counts=dict(counts), time=time.time()))

    selected_steps = {}
    for task in tasks:
        task_root = ROOT / task['task_id']
        q = torch.load(task_root / 'basis.pt', map_location=runtime.device, weights_only=True)
        prepared = prepare(task['train'], True)
        batches = [r.old.batch_on(runtime, v[1]) for v in prepared]
        selection = prepare(task['train'])
        calibration = read(ROOT / 'reference' / task['task_id'] / 'CALIBRATION.json')
        for arm in ARMS:
            dest = task_root / arm
            dest.mkdir()
            expert, hook = fresh(q if arm == 'PATH_NULL' else q[:, :0])
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
                        gradients = {k: v.grad.detach().clone() for k, v in expert.named_parameters()}
                        state, scalar, audit = matched_math.step(r, before, gradients, fisher,
                            calibration['coefficient'], scalar, calibration['targets'][step - 1], 'M13P')
                        expert.load_state_dict(state)
                        counts['updates'] += 1
                        counts['operator_backwards'] += audit['operator_backwards']
                        curve.append(dict(step=step, source_NLL=values,
                            latent_operator_step_norm=r.fm.map_norm(before, state), **audit))
                        if step % 20 == 0:
                            curve[-1]['effective_operator_step_norm'] = kernel.map_norm(r, before, state,
                                q if arm == 'PATH_NULL' else q[:, :0])
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
                                counts=dict(counts), source_commit=lock['source_commit'], task_id=task['task_id'],
                                arm=arm, fixed_basis=str(task_root / 'basis.pt'))
                            torch.save(saved, dest / 'resume.tmp')
                            (dest / 'resume.tmp').replace(dest / 'resume.pt')
                            write(ROOT / 'PROGRESS.json', dict(phase='TRAIN', task=task['task_id'], arm=arm,
                                step=step, source_gate=passed, counts=dict(counts), time=time.time()))
                            if passed:
                                break
                    assert prefix.misses == 5 and prefix.hits == 2 * step - 5
                torch.save(r.old.snapshot(expert), dest / 'delta.pt')
                loaded = torch.load(dest / 'delta.pt', map_location=runtime.device, weights_only=True)
                assert all(torch.equal(v, loaded[k]) for k, v in r.old.snapshot(expert).items())
                selected_steps[task['task_id'] + '/' + arm] = step
                write(dest / 'TRAIN_COMPLETE.json', dict(step=step, source_gate_met=passed, source_checks=checks,
                    curve=curve, fixed_basis_rank=q.shape[1] if arm == 'PATH_NULL' else 0,
                    source_only_selection=True, save_load_exact=True, counts=dict(counts), time=time.time()))
            finally:
                hook.detach()
        del q, batches, prepared, selection
    assert counts['updates'] == sum(selected_steps.values())
    assert counts['backwards'] == 2 * counts['updates'] and counts['operator_backwards'] == counts['updates']
    write(ROOT / 'ALL_TRAINING_FROZEN.json', dict(selected_steps=selected_steps, counts=dict(counts), time=time.time()))
    mechanism = []
    for task in tasks:
        task_root = ROOT / task['task_id']
        q = torch.load(task_root / 'basis.pt', map_location=runtime.device, weights_only=True)
        evaluation = prepare(task['train'] + task['heldout'] + task['protect'] + task['protect_extra'])
        for arm in ARMS:
            dest = task_root / arm
            expert, hook = fresh(q if arm == 'PATH_NULL' else q[:, :0])
            expert.load_state_dict(torch.load(dest / 'delta.pt', map_location=runtime.device, weights_only=True))
            try:
                for item in evaluation:
                    answer = emit(item, hook, dest / 'outputs' / (item[0]['query_id'] + '.json'))
                    if arm == 'PATH_NULL' and item[0]['query_id'] in task['Base_protect']:
                        old = task['Base_protect'][item[0]['query_id']]
                        mechanism.append(dict(task_id=task['task_id'], query_id=item[0]['query_id'],
                            Base_tokens_exact=answer['raw_token_ids'] == old['raw_token_ids'],
                            Base_fulltext_exact=answer['raw_answer'] == old['raw_answer'],
                            maximum_active_residual_norm=max(v['active_residual_norm'] for v in hook.last_generation_trace)))
                write(dest / 'GENERATION_COMPLETE.json', dict(rows=len(evaluation), time=time.time()))
            finally:
                hook.detach()
    assert len(mechanism) == 28
    write(ROOT / 'PATH_MECHANISM_AUDIT.json', mechanism)
    assert counts['generations'] == 84 + 28 + counts['updates'] // 20 * 5 + 364
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
