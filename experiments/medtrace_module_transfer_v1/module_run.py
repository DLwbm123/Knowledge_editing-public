"""Finite six-arm TT module study; parent runs remain immutable."""
import math
import os
import subprocess
import time
import traceback
from pathlib import Path
import torch
import scope as s
import function_math as fm
import fisher_math
import module_math as mm

c, p, q, old, RUN = s.c, s.p, s.q, s.old, s.RUN
GPUS = (2, 3, 4)
REFERENCE = Path(os.environ['REFERENCE_PARENT'])
SCOPE = Path(os.environ['SCOPE_PARENT'])
COUNTS = dict(forwards=0, non_generation_forwards=0, backwards=0, operator_backwards=0,
              updates=0, generation_attempts=0)
GENERATING = False
_baseline, _emit = s.baseline, s.emit


def baseline(task, row):
    path = _baseline(task, row)
    return SCOPE / path.relative_to(RUN) if path.is_relative_to(RUN) else path
s.baseline = baseline


def emit(*args, **kwargs):
    global GENERATING
    assert not GENERATING
    GENERATING = True
    try:
        return _emit(*args, **kwargs)
    finally:
        GENERATING = False
s.emit = emit


def count(*_):
    c.budget()
    COUNTS['forwards'] += 1
    if not GENERATING:
        COUNTS['non_generation_forwards'] += 1
        limit = {'module_reference': 5000, 'module_fisher': 128, 'module_train': 78000,
                 'module_evaluate': 100, 'module_route': 512}[os.environ['ACTION']]
        assert COUNTS['non_generation_forwards'] <= limit, 'Registered forward cap reached'


def load(gpu):
    assert gpu in c.read(RUN / 'RUN_MANIFEST.json')['physical_GPUs']
    free = int(subprocess.check_output(['nvidia-smi', '-i', str(gpu), '--query-gpu=memory.free',
                '--format=csv,noheader,nounits'], text=True).strip())
    assert free >= 45 * 1024, '45 GiB admission reserve unavailable'
    runtime, bindings = c.load(gpu)
    handle = runtime.model.register_forward_pre_hook(count)
    return runtime, bindings, handle


def save_counts():
    c.write(RUN / 'private' / f"COUNTS_{os.environ['ACTION']}_{os.environ.get('PARTITION', '0')}.json", COUNTS)


def set_gradient(expert, vector):
    offset = 0
    parameters = dict(expert.named_parameters())
    for key in mm.KEYS:
        parameter = parameters[key]
        size = parameter.numel()
        parameter.grad = vector[offset:offset + size].reshape_as(parameter).to(parameter).clone()
        offset += size
    assert offset == vector.numel()


def source_gradients(runtime, hook, expert, batches, step, individual):
    chosen = ((0, .25), (1 + (step - 1) % 4, .25), (5 + (step - 1) % 2, .5))
    expert.zero_grad(set_to_none=True)
    if not individual:
        losses = []
        for index, weight in chosen:
            hook.set_teacher_routing(batches[index].labels)
            loss = runtime.compute_loss(batches[index])
            assert torch.isfinite(loss)
            (weight * loss).backward()
            COUNTS['backwards'] += 1
            losses.append(float(loss.detach()))
        return None, losses
    rows, losses = [], []
    params = [dict(expert.named_parameters())[key] for key in mm.KEYS]
    for batch in batches:
        hook.set_teacher_routing(batch.labels)
        loss = runtime.compute_loss(batch)
        assert torch.isfinite(loss)
        gradient = torch.autograd.grad(loss, params)
        COUNTS['backwards'] += 1
        rows.append(torch.cat([v.detach().cpu().double().flatten() for v in gradient]))
        losses.append(float(loss.detach()))
    matrix = torch.stack(rows)
    gradient = sum(weight * matrix[index] for index, weight in chosen)
    set_gradient(expert, gradient)
    return -matrix, losses


def losses_at(runtime, hook, expert, state, batches):
    expert.load_state_dict(state)
    losses = []
    with torch.no_grad():
        for batch in batches:
            hook.set_teacher_routing(batch.labels)
            value = runtime.compute_loss(batch)
            assert torch.isfinite(value)
            losses.append(float(value))
    return losses


def match_proposal(before, proposed, target):
    initial = fm.map_norm(before, proposed)
    assert initial > 0
    # Normalize the trial direction before the existing bounded scalar solve.
    proposed = q.add(before, (target / initial) * (q.flatten(proposed) - q.flatten(before)))
    value, audit = fm.match(before, proposed, target)
    return value, dict(audit, initial_operator_norm=initial, direction_normalization=target / initial)


def finite_candidate(runtime, hook, expert, before, proposed, batches, initial_losses, canonical):
    checks = []
    displacement = q.flatten(proposed) - q.flatten(before)
    for fraction in (1., .5, .25, .125):
        candidate = q.add(before, fraction * displacement)
        if canonical:
            candidate, _ = mm.canonicalize(candidate)
        values = losses_at(runtime, hook, expert, candidate, batches)
        violations = [after - prior for after, prior in zip(values, initial_losses)]
        accepted = max(violations) <= 1e-6
        checks.append(dict(fraction=fraction, accepted=accepted, maximum_NLL_increase=max(violations)))
        if accepted:
            return candidate, dict(status='ACCEPTED', fraction=fraction, checks=checks)
    return before, dict(status='SKIPPED_FINITE_CONFLICT', fraction=0., checks=checks)


def finish_candidate(runtime, hook, expert, task, arm, base, batches, refs, curve, start, scalar_state, mechanism=None):
    final = old.snapshot(expert)
    gain, values, native, _ = s.measure(runtime, hook, expert, final, batches, refs, COUNTS)
    terminal = s.fit.prefix_check(runtime, hook, batches[0], native)
    c.save(s.point(task, arm), dict(states=dict(BASE=s.previous.weights(base), FINAL=s.previous.weights(final)),
        arm=arm, order=task['order'], scalar_optimizer_state=scalar_state,
        lock=c.digest(c.read(RUN / 'private/SCOPE_LOCK.json'))))
    saved = p.load_state(s.point(task, arm))
    assert all(torch.equal(saved['states']['FINAL'][k], final[k].cpu()) for k in mm.KEYS)
    c.write(RUN / 'private/results' / arm / f"{task['order']}.json", dict(status='COMPLETE', arm=arm,
        owner=task['order'], curve=curve, gain=gain.tolist(), values=values, terminal=terminal,
        save_load_exact=True, scalar_optimizer_state=scalar_state, mechanism=mechanism,
        counts={k: COUNTS[k] - start[k] for k in COUNTS}))


def reference():
    from m3bench_repro.editors.llava_runtime import seed_everything
    part = int(os.environ['PARTITION']); gpu = int(os.environ.get('GPU', GPUS[part]))
    assigned = s.tasks()[part::len(GPUS)]
    with p.lease(gpu):
        runtime, bindings, handle = load(gpu)
        frozen = {t['order']: s.prepare(runtime, bindings, t, s.source(t), True) for t in assigned}
        old.fp32(runtime)
        try:
            for task in assigned:
                assert not s.point(task, 'C').exists(), 'No implicit reference restart'
                seed_everything(task['seed'])
                retained = p.load_state(REFERENCE / 'private/candidates/C' / f"{task['order']}.pt")['states']
                expert = p.expert(retained['BASE'], task['seed'], runtime.device)
                base = old.snapshot(expert)
                hook = q.scoped.replay.MedTraceLayerHook(runtime.get_module(c.LAYER), expert); hook.attach()
                items = frozen[task['order']]
                batches = [old.batch_on(runtime, item[1]) for item in items]
                start, curve = dict(COUNTS), []
                try:
                    refs = [q.logits(runtime, hook, batch, verify_base=i == 0) for i, batch in enumerate(batches)]
                    opt = p.optimizer(expert, runtime.model)
                    for step in range(1, 161):
                        c.budget(); before = old.snapshot(expert)
                        _, losses = source_gradients(runtime, hook, expert, batches, step, False)
                        norm = torch.nn.utils.clip_grad_norm_(expert.parameters(), 1.)
                        assert torch.isfinite(norm)
                        opt.step(); q.d.apply_direction(opt, {id(v): before[k] for k, v in expert.named_parameters()}, 'RAW')
                        after = old.snapshot(expert); COUNTS['updates'] += 1
                        curve.append(dict(step=step, source_NLL=losses, operator_step_norm=fm.map_norm(before, after)))
                        if step % 20 == 0:
                            save_resume(task, 'C', expert, opt, None, step, curve)
                    assert all(torch.equal(expert.state_dict()[k].cpu(), retained['FINAL'][k]) for k in mm.KEYS), 'C reconstruction changed; stop before new arms'
                    finish_candidate(runtime, hook, expert, task, 'C', base, batches, refs, curve, start, None)
                    for item in items:
                        s.emit(runtime, hook, task, 'C', item, COUNTS)
                        actual = c.read(s.output('C', task['order'], item[0]))
                        prior = c.read(SCOPE / 'private/outputs/C' / str(task['order']) / (item[0]['query_id'] + '.json'))
                        assert actual['binding'] == prior['binding'] and actual['R0'] == prior['R0'], 'C source generation changed'
                    retire_resume(task, 'C')
                    print('REFERENCE_COMPLETE', task['order'], flush=True)
                finally:
                    hook.detach()
        finally:
            handle.remove(); save_counts()


def save_resume(task, arm, expert, opt, scalar_state, step, curve):
    c.save(RUN / 'private/active' / f"{arm}_{task['order']}.pt", dict(completed_step=step,
        state=s.previous.weights(expert.state_dict()), optimizer=opt.state_dict() if opt else None,
        scalar_state=scalar_state, curve=curve, counts=dict(COUNTS), rng=c.rng()))
    c.write(RUN / 'private/progress' / arm / f"{task['order']}.json", dict(step=step, counts=dict(COUNTS)))
    print('TRAIN', arm, task['order'], step, flush=True)


def retire_resume(task, arm):
    path = RUN / 'private/active' / f"{arm}_{task['order']}.pt"
    assert path.is_file() and not path.is_symlink()
    c.write(RUN / 'private/retired' / f"{arm}_{task['order']}.json", dict(bytes=path.stat().st_size,
        reason='Verified final and complete trajectory saved; resume consumer completed'))
    path.unlink()


def fisher():
    assert (RUN / 'private/REFERENCE_EXACT.json').exists()
    assert not (RUN / 'private/OPERATOR_FISHER.pt').exists(), 'No implicit Fisher reconstruction'
    task = s.tasks()[0]; basis, held = q.split()
    assert len(basis) == 61 and len(held) == 96
    assert not {x['source_group'] for x in basis} & {x['source_group'] for x in held}
    with p.lease(GPUS[0]):
        runtime, _, handle = load(GPUS[0])
        frozen = [old.cpu(q.protection_batch(runtime, row, task)) for row in basis]
        old.fp32(runtime)
        saved = p.load_state(s.point(task, 'C'))['states']
        expert = p.expert(saved['BASE'], task['seed'], runtime.device)
        hook = q.scoped.replay.MedTraceLayerHook(runtime.get_module(c.LAYER), expert); hook.attach()
        captured = {}
        def capture(_module, args, output):
            x = args[0].detach()[hook.token_mask]
            captured['x'] = x / (x.square().mean(-1, keepdim=True).sqrt() + expert.epsilon)
            captured['output'] = output
        tap = runtime.get_module(c.LAYER).register_forward_hook(capture)
        xs, ys, token_counts = [], [], []
        try:
            for index, prepared in enumerate(frozen):
                batch = old.batch_on(runtime, prepared); hook.set_teacher_routing(batch.labels)
                output = runtime.model(**batch.forward_kwargs())
                mask = batch.labels[:, 1:] != -100; logits = output.logits[:, :-1][mask].float()
                n = len(logits); probability = logits.detach().softmax(-1)
                generator = torch.Generator(device=logits.device).manual_seed(20261009 + index)
                vectors = []
                for sample in range(32):
                    c.budget()
                    signs = torch.randint(0, 2, logits.shape, device=logits.device, dtype=torch.int8, generator=generator).float() * 2 - 1
                    vector = fisher_math.categorical_probe(probability, signs)
                    assert vector.sum(-1).abs().max() <= 2e-6
                    value = (logits * vector).sum() / math.sqrt(61 * 32 * n)
                    parameters = list(expert.parameters()) if index == sample == 0 else []
                    gradients = torch.autograd.grad(value, [captured['output']] + parameters, retain_graph=sample != 31)
                    y = gradients[0].detach()[hook.token_mask]
                    COUNTS['backwards'] += 1
                    if parameters:
                        b, a = expert.factors()
                        proxy = ((captured['x'] @ a.T) * (y @ b)).sum()
                        pulled = torch.autograd.grad(proxy, parameters)
                        for direct, composed in zip(gradients[1:], pulled):
                            torch.testing.assert_close(direct, composed, rtol=1e-4, atol=1e-7)
                    vectors.append(y.cpu())
                xs.append(captured['x'].cpu()); ys.append(torch.stack(vectors, 1)); token_counts.append(n)
                captured.clear()
                del batch, output, logits, probability, gradients, value, vector, signs
                if (index + 1) % 12 == 0:
                    print('OPERATOR_FISHER', index + 1, flush=True)
            assert sum(token_counts) == 1514 and COUNTS['backwards'] == 1952
            cache = dict(x=torch.cat(xs), y=torch.cat(ys), token_counts=token_counts)
            assert cache['y'].shape == (1514, 32, 4096) and cache['x'].shape == (1514, 14336)
            assert all(torch.isfinite(cache[k]).all() for k in ('x', 'y'))
            c.save(RUN / 'private/OPERATOR_FISHER.pt', cache)
            c.write(RUN / 'public/FISHER_ADMISSION.json', dict(status='PASS', questions=61, tokens=1514,
                full_vocabulary=True, probes_per_question=32, operator_directions=1952,
                zero_TT_pullback_parity=True, base_only=True, heldout_used=False,
                bytes=sum(cache[k].numel() * cache[k].element_size() for k in ('x', 'y'))))
        finally:
            tap.remove(); hook.detach(); handle.remove(); save_counts()


def train():
    from m3bench_repro.editors.llava_runtime import seed_everything
    assert (RUN / 'private/REFERENCE_EXACT.json').exists() and (RUN / 'public/FISHER_ADMISSION.json').exists()
    part = int(os.environ['PARTITION']); gpu = int(os.environ.get('GPU', GPUS[part]))
    assigned = s.tasks()[part::len(GPUS)]
    with p.lease(gpu):
        runtime, bindings, handle = load(gpu)
        frozen = {t['order']: [old.cpu(runtime.build_edit_batch(s.record(t, row))) for row in s.source(t)] for t in assigned}
        old.fp32(runtime)
        cache = p.load_state(RUN / 'private/OPERATOR_FISHER.pt')
        cache = {k: v.to(runtime.device) if isinstance(v, torch.Tensor) else v for k, v in cache.items()}
        try:
            for task in assigned:
                saved = p.load_state(s.point(task, 'C'))['states']
                reference = c.read(RUN / 'private/results/C' / f"{task['order']}.json")
                target_norms = [row['operator_step_norm'] for row in reference['curve']]
                refstate = {k: v.to(runtime.device) for k, v in saved['FINAL'].items()}
                denominator = float(mm.operator_penalty(refstate, cache))
                source_gain = -.25 * reference['values'][0]['NLL_change'] - .0625 * sum(v['NLL_change'] for v in reference['values'][1:5]) - .25 * sum(v['NLL_change'] for v in reference['values'][5:])
                assert math.isfinite(denominator) and denominator > 0 and source_gain > 0
                coefficient = .1 * source_gain / denominator
                c.write(RUN / 'private/calibration' / f"{task['order']}.json", dict(C_operator_penalty=denominator,
                    C_source_gain=source_gain, coefficient=coefficient, fraction=.1, heldout_used=False))
                # Rotate only execution order; every registered arm is run for every owner.
                arms = list(mm.ARMS[1:]); offset = (task['order'] - 1) % len(arms); arms = arms[offset:] + arms[:offset]
                for arm in arms:
                    assert not s.point(task, arm).exists(), 'No implicit trajectory restart'
                    seed_everything(task['seed']); m1, m2, m3 = mm.flags(arm)
                    initial = {k: v.to(runtime.device) for k, v in saved['BASE'].items()}
                    if m1:
                        initial, _ = mm.canonicalize(initial)
                    expert = p.expert(initial, task['seed'], runtime.device)
                    base = old.snapshot(expert)
                    hook = q.scoped.replay.MedTraceLayerHook(runtime.get_module(c.LAYER), expert); hook.attach()
                    batches = [old.batch_on(runtime, value) for value in frozen[task['order']]]
                    start, curve = dict(COUNTS), []
                    scalar_state = dict(step=0, second_moments=[0., 0.]) if m1 else None
                    opt = None if m1 else p.optimizer(expert, runtime.model)
                    scale = torch.cat([torch.full((initial[k].numel(),), 1e-3 if k in mm.GROUPS[0] else 1e-4, dtype=torch.float64) for k in mm.KEYS])
                    try:
                        refs = [q.logits(runtime, hook, batch, verify_base=i == 0) for i, batch in enumerate(batches)]
                        for step in range(1, 161):
                            c.budget(); before = old.snapshot(expert)
                            gradients, source_losses = source_gradients(runtime, hook, expert, batches, step, m2)
                            penalty = None
                            if m3:
                                penalty = mm.operator_penalty(dict(expert.named_parameters()), cache)
                                assert torch.isfinite(penalty)
                                (coefficient * penalty).backward(); COUNTS['operator_backwards'] += 1
                            norm = torch.nn.utils.clip_grad_norm_(expert.parameters(), 1.)
                            assert torch.isfinite(norm) and all(v.grad is not None and torch.isfinite(v.grad).all() for v in expert.parameters())
                            if m1:
                                proposed, scalar_state = mm.canonical_rms_step(before, {k: v.grad.detach() for k, v in expert.named_parameters()}, scalar_state)
                            else:
                                opt.step(); q.d.apply_direction(opt, {id(v): before[k] for k, v in expert.named_parameters()}, 'RAW')
                                proposed = old.snapshot(expert)
                            projection = None
                            if m2:
                                delta = (q.flatten(proposed) - q.flatten(before)) / scale
                                delta, projection = mm.project_seven(gradients * scale, delta)
                                proposed = q.add(before, delta * scale)
                            if fm.map_norm(before, proposed) <= 1e-14:
                                chosen, matching = before, dict(status='ZERO_DIRECTION', target=target_norms[step - 1], actual=0.)
                                finite = dict(status='ZERO_DIRECTION', fraction=0., checks=[])
                            else:
                                proposed, matching = match_proposal(before, proposed, target_norms[step - 1])
                                if m2:
                                    chosen, finite = finite_candidate(runtime, hook, expert, before, proposed, batches, source_losses, m1)
                                else:
                                    chosen = mm.canonicalize(proposed)[0] if m1 else proposed
                                    finite = dict(status='NOT_APPLICABLE', fraction=1., checks=[])
                            expert.load_state_dict(chosen); COUNTS['updates'] += 1
                            actual = fm.map_norm(before, chosen)
                            if not m2 and matching.get('status') != 'ZERO_DIRECTION':
                                assert abs(actual - target_norms[step - 1]) <= 1e-10 + .002 * target_norms[step - 1], 'Canonicalization changed the matched function step'
                            curve.append(dict(step=step, source_NLL=source_losses, gradient_norm=float(norm),
                                projection=projection, finite=finite, matching=matching, operator_step_norm=actual,
                                penalty_before=float(penalty.detach()) if penalty is not None else None))
                            if step % 20 == 0:
                                save_resume(task, arm, expert, opt, scalar_state, step, curve)
                        final_penalty = float(mm.operator_penalty(old.snapshot(expert), cache))
                        finish_candidate(runtime, hook, expert, task, arm, base, batches, refs, curve, start, scalar_state,
                            dict(final_operator_penalty=final_penalty, C_operator_penalty=denominator,
                                 coefficient=coefficient if m3 else 0., C_source_gain=source_gain))
                        retire_resume(task, arm)
                    finally:
                        hook.detach()
        finally:
            handle.remove(); save_counts()


def evaluate():
    assert c.read(RUN / 'private/CANDIDATES_FROZEN.json')['candidates'] == 48
    part = int(os.environ['PARTITION']); gpu = int(os.environ.get('GPU', GPUS[part]))
    tasks = s.tasks()[part::len(GPUS)]
    with p.lease(gpu):
        runtime, bindings, handle = load(gpu)
        frozen = {t['order']: s.prepare(runtime, bindings, t, s.source(t) + s.events(t) + s.previous.data.held()) for t in tasks}
        old.fp32(runtime)
        try:
            for task in tasks:
                for arm in mm.ARMS[1:]:
                    saved = p.load_state(s.point(task, arm))
                    expert = p.expert(saved['states']['FINAL'], task['seed'], runtime.device).requires_grad_(False)
                    hook = q.scoped.replay.MedTraceLayerHook(runtime.get_module(c.LAYER), expert); hook.attach()
                    try:
                        for item in frozen[task['order']]:
                            s.emit(runtime, hook, task, arm, item, COUNTS)
                    finally:
                        hook.detach()
                    print('EVALUATED', arm, task['order'], flush=True)
        finally:
            handle.remove(); save_counts()


def aliases():
    for arm in ('C', 'NATURAL_C'):
        for path in sorted((SCOPE / 'private/outputs' / arm).glob('*/*.json')):
            dest = RUN / 'private/outputs' / arm / path.relative_to(SCOPE / 'private/outputs' / arm)
            if dest.exists():
                assert arm == 'C'
                continue
            saved = c.read(path)
            c.write(dest, dict(saved, alias_of=str(path), forwards=0))
    assert len(list((RUN / 'private/outputs/C').glob('*/*.json'))) == 939
    assert len(list((RUN / 'private/outputs/NATURAL_C').glob('*/*.json'))) == 219


def controller():
    import pipeline
    assert not (RUN / 'private/MODULE_START.json').exists(), 'No implicit controller restart'
    c.write(RUN / 'private/MODULE_START.json', dict(epoch=time.time(), pid=os.getpid()))
    c.write(RUN / 'public/PROGRESS.json', dict(status='REFERENCE_RECONSTRUCTION_RUNNING', complete=False))
    pipeline.wait([pipeline.launch('module_run.py', 'module_reference', gpu, part) for part, gpu in enumerate(GPUS)])
    assert all(s.point(t, 'C').exists() for t in s.tasks())
    c.write(RUN / 'private/REFERENCE_EXACT.json', dict(status='PASS', candidates=8, exact_source_outputs=56,
        exact_retained_final_tensors=True, epoch=time.time()))
    aliases()
    c.write(RUN / 'public/PROGRESS.json', dict(status='REFERENCE_EXACT_OPERATOR_FISHER_RUNNING', complete=False))
    pipeline.wait([pipeline.launch('module_run.py', 'module_fisher', GPUS[0], 0)])
    c.write(RUN / 'public/PROGRESS.json', dict(status='MODULE_TRAINING_RUNNING', complete=False))
    pipeline.wait([pipeline.launch('module_run.py', 'module_train', gpu, part) for part, gpu in enumerate(GPUS)])
    finish()


def finish():
    import pipeline
    from module_recovery import admitted_stage
    paths = [s.point(t, arm) for t in s.tasks() for arm in mm.ARMS]
    assert len(paths) == 48 and all(path.exists() for path in paths)
    assert not (RUN / 'private/CANDIDATES_FROZEN.json').exists(), 'Do not repeat evaluation'
    for task in s.tasks():
        for arm in mm.ARMS:
            result = c.read(RUN / 'private/results' / arm / f"{task['order']}.json")
            assert result['status'] == 'COMPLETE' and len(result['curve']) == 160 and result['save_load_exact']
    training_counts = [c.read(path) for path in (RUN / 'private').glob('COUNTS_*.json')]
    assert sum(row['updates'] for row in training_counts) == 7680
    assert sum(row['backwards'] for row in training_counts) == 40352
    assert sum(row['operator_backwards'] for row in training_counts) == 2560
    c.write(RUN / 'private/CANDIDATES_FROZEN.json', dict(candidates=48, epoch=time.time(), selection=False))
    c.write(RUN / 'public/PROGRESS.json', dict(status='ALL_CANDIDATES_FROZEN_EVALUATION_RUNNING', complete=False))
    cache = RUN / 'private/OPERATOR_FISHER.pt'
    assert cache.is_file() and not cache.is_symlink()
    c.write(RUN / 'private/FISHER_CONSUMED.json', dict(bytes=cache.stat().st_size(),
        reason='All M3 and M123 trajectories complete; factorized Fisher has no remaining consumer'))
    cache.unlink()
    admitted_stage('module_run.py', 'module_evaluate', range(len(GPUS)))
    admitted_stage('module_route.py', 'module_route', [0])
    outputs = list((RUN / 'private/outputs').glob('*/*/*.json'))
    counts = [c.read(path) for path in (RUN / 'private').glob('COUNTS_*.json')]
    assert len(outputs) == 6948
    assert sum(row['updates'] for row in counts) == 7680
    assert sum(row['backwards'] for row in counts) == 40352
    assert sum(row['operator_backwards'] for row in counts) == 2560
    assert sum(row['generation_attempts'] for row in counts) <= 5846
    assert sum(row['non_generation_forwards'] for row in counts) <= 170000
    c.write(RUN / 'private/MODULE_GENERATION_COMPLETE.json', dict(status='GENERATED_SCORING_PENDING',
        outputs=6948, counts=counts, epoch=time.time()))
    c.write(RUN / 'public/PROGRESS.json', dict(status='GENERATED_SCORING_PENDING', complete=False))
    pipeline.wait([pipeline.launch('module_queue.py', 'module_ingest')])
    deletion = []
    for path in paths:
        assert not path.is_symlink()
        deletion.append(dict(arm=path.parent.name, owner=int(path.stem), bytes=path.stat().st_size()))
    c.write(RUN / 'private/CANDIDATES_CONSUMED.json', dict(candidates=deletion,
        reason='All forced/natural generations, save/load checks and durable scoring bindings complete; rebuilding requires retraining'))
    for path in paths:
        path.unlink()
    root = RUN / 'private/judge_module_astra_medium'
    while not (root / 'ALL_WORKERS_COMPLETE.json').exists():
        c.budget()
        assert not list((root / 'workers').glob('*/SCORER_FAILURE.json'))
        time.sleep(30)
    pipeline.wait([pipeline.launch('module_report.py', 'module_report')])


if __name__ == '__main__':
    try:
        {'module_reference': reference, 'module_fisher': fisher, 'module_train': train,
         'module_evaluate': evaluate, 'module_controller': controller,
         'module_finish': finish}[os.environ['ACTION']]()
    except BaseException as error:
        c.write(RUN / 'private' / f"FAILURE_{os.environ.get('ACTION')}_{os.environ.get('PARTITION')}.json",
            dict(error=repr(error), traceback=traceback.format_exc(), counts=COUNTS, epoch=time.time()))
        if os.environ.get('ACTION') == 'module_controller':
            c.write(RUN / 'public/PROGRESS.json', dict(status='STOPPED_WITH_RETAINED_EVIDENCE', complete=False,
                retry=False, epoch=time.time()))
        raise
