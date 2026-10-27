"""Rebuild three fixed V126 trajectories, then probe uncommitted finite TRAIN steps."""
import copy
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback

ROOT = Path(os.environ['V129_ROOT'])
PARENT = Path(os.environ['PARENT_RUN'])
START = time.time()
counts = dict(forwards=0, backwards=0, operator_backwards=0, updates=0,
              generations=0, vision_forwards=0, path_positions=0, finite_probes=0)


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


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
    import priority_step
    import completion
    import project
    from prefix_cache import PrefixCache
    from scripts.medtrace.run_realmodel_core import load_real_runtime
    from types import SimpleNamespace
    r.c.budget = budget
    torch.manual_seed(lock['seed'])
    torch.cuda.manual_seed_all(lock['seed'])
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    tasks = read(ROOT / 'TASKS.json')['tasks']
    assert [t['task_id'] for t in tasks] == ['T07', 'T12', 'T13']
    assert sum(len(t['train']) for t in tasks) == 15
    assert sum(len(t['protect']) for t in tasks) == 6
    assert sum(len(t['replay_protect']) for t in tasks) == 18
    assert not any(k in t for t in tasks for k in ('heldout', 'protect_new'))
    labels = read(ROOT / 'TRAIN_CORRECTNESS_FROZEN.json')['rows']
    failed = {(v['task_id'], v['query_id']) for v in labels if not v['core_correct']}
    assert len(failed) == 3
    contexts = {(v['task_id'], v['query_id']): v for v in read(ROOT / 'PREFIX_CONTEXTS_FROZEN.json')['contexts']}
    assert len(contexts) == 15
    write(ROOT / 'PROJECT_SELF_CHECK.json', project.selfcheck())
    write(ROOT / 'KERNEL_SELF_CHECK.json', kernel.selfcheck())
    write(ROOT / 'PRIORITY_SELF_CHECK.json', priority_step.selfcheck())
    runtime = load_real_runtime(SimpleNamespace(cpu_gate=PARENT / 'private/cpu_gate'))
    runtime.model.eval()
    r.old.fp32(runtime)
    assert not any(v.requires_grad for v in runtime.model.parameters())
    bindings = read(PARENT / 'private/EVAL_BINDINGS.json')
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

    def emit(item, hook=None):
        counts['generations'] += 1
        answer, _ = r.s.fit.short.generate(runtime, hook, item[2], item[3])
        return dict(raw_answer=answer.decoded_text, raw_token_ids=list(answer.raw_token_ids))

    def state_grad(expert):
        return {k: v.grad.detach().clone() for k, v in expert.named_parameters()}

    def sum_grad(gradients, weights):
        return {k: sum(w * g[k] for w, g in zip(weights, gradients)) for k in gradients[0]}

    # These consumers rebuild the retired Base paths; old answers supply only parity checks.
    for task in tasks:
        tid = task['task_id']
        dest = ROOT / tid
        dest.mkdir()
        vectors = []
        for item in prepare(task['protect']):
            captured = []
            def capture(_module, args):
                captured.append(args[0][0, -1:].detach().cpu().double())
            handle = runtime.get_module(r.c.LAYER).register_forward_pre_hook(capture)
            try:
                answer = emit(item)
            finally:
                handle.remove()
            old = task['Base_protect'][item[0]['query_id']]
            assert answer['raw_answer'] == old['raw_answer'] and answer['raw_token_ids'] == old['raw_token_ids']
            assert answer['raw_token_ids'][-1] == runtime.adapter.tokenizer.eos_token_id
            assert len(captured) == len(answer['raw_token_ids']) < 1024
            write(dest / 'Base_paths' / (item[0]['query_id'] + '.json'), answer)
            vectors.extend(captured)
        counts['path_positions'] += len(vectors)
        q, audit = kernel.basis(vectors)
        old = read(ROOT / 'reference' / tid / 'BASIS_ADMISSION.json')
        assert audit['rank'] == old['rank'] and audit['path_positions'] == old['path_positions']
        assert max(abs(a - b) for a, b in zip(audit['singular_values'], old['singular_values'])) <= 1e-7
        torch.save(q, dest / 'basis.pt')
        write(dest / 'BASIS_ADMISSION.json', dict(**audit, Base_tokens_exact=True, time=time.time()))
    assert counts['generations'] == 6
    write(ROOT / 'ALL_BASES_FROZEN.json', dict(counts=dict(counts), time=time.time()))

    # Reproduce only the old fixed160 algorithm, never apply the new probe direction here.
    for task in tasks:
        tid = task['task_id']
        dest = ROOT / tid
        q = torch.load(dest / 'basis.pt', map_location=runtime.device, weights_only=True)
        prepared = prepare(task['train'] + task['replay_protect'], True)
        batches = [r.old.batch_on(runtime, v[1]) for v in prepared]
        eos_batches, eos_masks = [], []
        for item in prepared[:5]:
            context = contexts[(tid, item[0]['query_id'])]
            assert context['original_binding'] == item[4]
            batch, mask = completion.make_batch(runtime, item[3], context['prefix_token_ids'])
            eos_batches.append(batch)
            eos_masks.append(mask)
        calibration = read(ROOT / 'reference' / tid / 'CALIBRATION.json')
        old_curve = read(ROOT / 'reference' / tid / 'TRAIN_COMPLETE.json')['curve']
        state = r.mm.canonicalize(r.p.tr.TT4(lock['seed'], 8, 8).state_dict())[0]
        expert = r.p.expert(state, lock['seed'], runtime.device)
        kernel.install(expert, q)
        hook = r.q.scoped.replay.MedTraceLayerHook(runtime.get_module(r.c.LAYER), expert)
        hook.attach()
        scalar = dict(step=0, second_moments=[0., 0.])
        curve = []
        try:
            item = prepare([task['target']])[0]
            with hook.generation_request():
                smoke = emit(item)
            assert smoke['raw_answer'] == task['Base_target']['raw_answer']
            assert smoke['raw_token_ids'] == task['Base_target']['raw_token_ids']
            with PrefixCache(runtime, batches + eos_batches):
                for step in range(1, 161):
                    before = r.old.snapshot(expert)
                    expert.zero_grad(set_to_none=True)
                    source_values, eos_values = [], []
                    for i in (0, 1 + (step - 1) % 4):
                        hook.set_teacher_routing(batches[i].labels)
                        loss = runtime.compute_loss(batches[i])
                        (.25 * loss).backward()
                        counts['backwards'] += 1
                        source_values.append(float(loss.detach()))
                    gc = state_grad(expert)
                    expert.zero_grad(set_to_none=True)
                    for i in (0, 1 + (step - 1) % 4):
                        hook.set_request_routing(eos_masks[i])
                        loss = runtime.compute_loss(eos_batches[i])
                        (.25 * loss).backward()
                        counts['backwards'] += 1
                        eos_values.append(float(loss.detach()))
                    ge = state_grad(expert)
                    expert.zero_grad(set_to_none=True)
                    index = (step - 1) % len(task['replay_protect'])
                    hook.set_teacher_routing(batches[5 + index].labels)
                    loss = runtime.compute_loss(batches[5 + index])
                    loss.backward()
                    counts['backwards'] += 1
                    gr = state_grad(expert)
                    state, scalar, audit = priority_step.step(r, before, gc,
                        {k: ge[k] + gr[k] for k in gc}, fisher, calibration['coefficient'],
                        scalar, calibration['targets'][step - 1])
                    expert.load_state_dict(state)
                    counts['updates'] += 1
                    counts['operator_backwards'] += 1
                    current = dict(step=step, source_NLL=source_values, replay_NLL=float(loss.detach()),
                                   completion_EOS_NLL=eos_values)
                    # Recreated trajectories have no retained weight identity; require measured loss parity.
                    old = old_curve[step - 1]
                    for key in ('source_NLL', 'completion_EOS_NLL'):
                        assert max(abs(a - b) for a, b in zip(current[key], old[key])) <= 1e-6, (tid, step, key)
                    assert abs(current['replay_NLL'] - old['replay_NLL']) <= 1e-6, (tid, step, 'replay')
                    curve.append(current)
                    if step % 20 == 0:
                        torch.save(dict(state=r.old.snapshot(expert), scalar=scalar), dest / 'resume.tmp')
                        (dest / 'resume.tmp').replace(dest / 'resume.pt')
                        write(ROOT / 'PROGRESS.json', dict(phase='REBUILD_FIXED_V126', task=tid,
                            step=step, counts=dict(counts), time=time.time()))
            torch.save(r.old.snapshot(expert), dest / 'delta.pt')
            loaded = torch.load(dest / 'delta.pt', map_location=runtime.device, weights_only=True)
            assert all(torch.equal(v, loaded[k]) for k, v in r.old.snapshot(expert).items())
            write(dest / 'REBUILD_COMPLETE.json', dict(curve=curve, all160_loss_parity=True,
                save_load_exact=True, no_claim_of_deleted_weight_identity=True, time=time.time()))
        finally:
            hook.detach()
        del batches, eos_batches, eos_masks, prepared, q
    assert counts['updates'] == 480 and counts['backwards'] == 2400 and counts['generations'] == 9
    write(ROOT / 'ALL_REBUILDS_FROZEN.json', dict(counts=dict(counts), time=time.time()))

    audits = []
    for task in tasks:
        tid = task['task_id']
        dest = ROOT / tid
        q = torch.load(dest / 'basis.pt', map_location=runtime.device, weights_only=True)
        prepared = prepare(task['train'] + task['replay_protect'], True)
        batches = [r.old.batch_on(runtime, v[1]) for v in prepared]
        eos_batches, eos_masks = [], []
        for item in prepared[:5]:
            context = contexts[(tid, item[0]['query_id'])]
            batch, mask = completion.make_batch(runtime, item[3], context['prefix_token_ids'])
            eos_batches.append(batch)
            eos_masks.append(mask)
        saved = torch.load(dest / 'resume.pt', map_location=runtime.device, weights_only=True)
        before, scalar = saved['state'], saved['scalar']
        expert = r.p.expert(before, lock['seed'], runtime.device)
        kernel.install(expert, q)
        hook = r.q.scoped.replay.MedTraceLayerHook(runtime.get_module(r.c.LAYER), expert)
        hook.attach()
        try:
            with PrefixCache(runtime, batches + eos_batches):
                gradients, baseline, eos_gradients = [], [], []
                for batch in batches:
                    expert.zero_grad(set_to_none=True)
                    hook.set_teacher_routing(batch.labels)
                    loss = runtime.compute_loss(batch)
                    assert torch.isfinite(loss)
                    loss.backward()
                    counts['backwards'] += 1
                    gradients.append(state_grad(expert))
                    baseline.append(float(loss.detach()))
                for batch, mask in zip(eos_batches, eos_masks):
                    expert.zero_grad(set_to_none=True)
                    hook.set_request_routing(mask)
                    loss = runtime.compute_loss(batch)
                    loss.backward()
                    counts['backwards'] += 1
                    eos_gradients.append(state_grad(expert))
                gc = sum_grad(gradients[:5], [.25, .0625, .0625, .0625, .0625])
                ge = sum_grad(eos_gradients, [.25, .0625, .0625, .0625, .0625])
                n = len(task['replay_protect'])
                gr = sum_grad(gradients[5:], [1 / n] * n)
                calibration = read(ROOT / 'reference' / tid / 'CALIBRATION.json')
                target = calibration['targets'][-1]
                current, _, _ = priority_step.step(r, before, gc,
                    {k: ge[k] + gr[k] for k in gc}, fisher, calibration['coefficient'], scalar, target)
                counts['operator_backwards'] += 1
                # Pure protection supplies a local witness objective, not a new training loss recipe.
                params = [v.clone() for v in gr.values()]
                for value in params:
                    value.grad = value.detach().clone()
                torch.nn.utils.clip_grad_norm_(params, 1.)
                proposal, _ = r.mm.canonical_rms_step(before,
                    dict(zip(gr, [v.grad for v in params])), copy.deepcopy(scalar))
                proposal, _ = r.match_proposal(before, proposal, target)
                proposal = r.mm.canonicalize(proposal)[0]
                raw = {k: proposal[k] - before[k] for k in before}
                displacement, projection = project.project(raw, gradients)
                assert projection['converged'], (tid, projection)
                # Match the old latent step before the common fixed backtracking ladder.
                projected = r.mm.canonicalize({k: before[k] + displacement[k] for k in before})[0]
                if r.fm.map_norm(before, projected) > 1e-14:
                    projected, _ = r.match_proposal(before, projected, target)
                    projected = r.mm.canonicalize(projected)[0]
                else:
                    projected = {k: v.clone() for k, v in before.items()}
                candidates = {'CURRENT_V126_EXPECTED': current, 'PROJECTED_PROTECTION': projected}
                rows = []
                for name, proposal in candidates.items():
                    for exponent in range(8):
                        fraction = 2. ** -exponent
                        candidate = r.mm.canonicalize({k: before[k] + fraction * (proposal[k] - before[k]) for k in before})[0]
                        expert.load_state_dict(candidate)
                        values = []
                        with torch.no_grad():
                            for batch in batches:
                                hook.set_teacher_routing(batch.labels)
                                loss = runtime.compute_loss(batch)
                                assert torch.isfinite(loss)
                                values.append(float(loss))
                        counts['finite_probes'] += 1
                        tolerances = [1e-7 + 1e-6 * max(v, .001) for v in baseline]
                        all_nonincrease = all(a <= b + tol for a, b, tol in zip(values, baseline, tolerances))
                        replay_drop = sum(baseline[5:]) / n - sum(values[5:]) / n
                        significant_drop = replay_drop >= 1e-5 * max(1., sum(baseline[5:]) / n)
                        failed_indices = [5 + i for i, row in enumerate(task['replay_protect']) if (tid, row['query_id']) in failed]
                        failed_improve = all(baseline[i] - values[i] >= 1e-5 * max(1., baseline[i]) for i in failed_indices)
                        rows.append(dict(arm=name, fraction=fraction, losses=values,
                            all_source_and_protection_nonincrease=all_nonincrease,
                            replay_mean_drop=replay_drop, failed_requests_significantly_improve=failed_improve,
                            witness=all_nonincrease and significant_drop and failed_improve,
                            latent_operator_step_norm=r.fm.map_norm(before, candidate),
                            effective_operator_step_norm=kernel.map_norm(r, before, candidate, q)))
                expert.load_state_dict(before)
                assert all(torch.equal(v, before[k]) for k, v in r.old.snapshot(expert).items())
                # Reload guarantees probes did not consume the persistent V126 optimizer state.
                persistent = torch.load(dest / 'resume.pt', map_location=runtime.device, weights_only=True)
                assert all(torch.equal(v, persistent['state'][k]) for k, v in before.items()) and scalar == persistent['scalar']
                result = dict(task_id=tid, baseline_losses=baseline, projection=projection,
                    rows=rows, TT_and_optimizer_unchanged_after_probes=True, time=time.time())
                audits.append(result)
                write(dest / 'FINITE_AUDIT.json', result)
                write(ROOT / 'PROGRESS.json', dict(phase='FINITE_AUDIT', completed=len(audits),
                    counts=dict(counts), time=time.time()))
        finally:
            hook.detach()
    assert counts['backwards'] == 2448 and counts['operator_backwards'] == 483
    assert counts['updates'] == 480 and counts['finite_probes'] == 48 and counts['generations'] == 9
    projected_pass = [a['task_id'] for a in audits if any(v['witness'] for v in a['rows'] if v['arm'] == 'PROJECTED_PROTECTION')]
    current_pass = [a['task_id'] for a in audits if any(v['witness'] for v in a['rows'] if v['arm'] == 'CURRENT_V126_EXPECTED')]
    budget()
    write(ROOT / 'AUDIT_FROZEN.json', dict(audits=audits, time=time.time()))
    write(ROOT / 'COMPLETE.json', dict(status='FINITE_TRAIN_WITNESS_AUDIT_COMPLETE',
        projected_groups_with_witness=projected_pass, current_groups_with_witness=current_pass,
        bounded_function_guard_test_admissible=len(projected_pass) == 3 and any(t not in current_pass for t in ('T07', 'T12')),
        no_probe_committed=True, no_new_training_admitted_automatically=True, counts=dict(counts), time=time.time()))
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
