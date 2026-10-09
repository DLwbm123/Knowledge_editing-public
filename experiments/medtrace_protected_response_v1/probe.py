"""One-step response-preservation diagnostic; no trained checkpoint is produced."""
import os
import time
import traceback
from dataclasses import replace
from pathlib import Path
import torch
import direction as d
import function_math as fm
import projection_math as pm

p, c, RUN, scoped = d.p, d.c, d.RUN, d.scoped
GPUS = (2, 3, 4, 5, 6, 7)
ARMS = ('RAW', 'PROJECTED', 'MATCHED_RAW')
KEYS = ('G1', 'G2', 'G3', 'G4')


def split():
    rows = c.read(Path(os.environ['REPLAY_PARENT']) / 'private/REPLAY_FIT.json')
    assert len(rows) == 192 and all(x['role'] == 'REPLAY_FIT' for x in rows)
    groups = list(dict.fromkeys(x['source_group'] for x in rows))
    assert len(groups) == 64 and all(sum(x['source_group'] == g for x in rows) == 3 for g in groups)
    first = set(groups[:32])
    return [x for x in rows if x['source_group'] in first], [x for x in rows if x['source_group'] not in first]


def flatten(state):
    return torch.cat([state[k].detach().cpu().double().flatten() for k in KEYS])


def add(before, delta):
    result, offset = {}, 0
    for k in KEYS:
        n = before[k].numel()
        result[k] = before[k] + delta[offset:offset+n].reshape_as(before[k]).to(before[k])
        offset += n
    assert offset == len(delta)
    return result


def plan():
    basis, held = split()
    groups = {x['source_group'] for x in basis + held}
    excluded = {x['source_group'] for x in c.read(RUN/'private/EVAL_LEDGER.json')['queries'].values()}
    excluded.update(x['source_group'] for rows in c.read(Path(os.environ['BASE_ROOT'])/'private/U_ROLES.json').values() for x in rows)
    assert not groups & excluded
    admission = c.read(Path(os.environ['REPLAY_PARENT'])/'public/DATA_ADMISSION.json')
    assert admission['source_group_overlap'] == 0 and admission['previous_evaluation_and_U_group_overlap'] == 0
    assert all(Path(c.local_path(x['image_path'])).is_file() for x in basis + held)
    for t in d.selected():
        state = d.start_state(t)
        assert {k: tuple(v.shape) for k, v in state.items()} == fm.SHAPES
    lock = dict(experts=8, basis_groups=32, basis_questions=96, holdout_groups=32,
        holdout_questions=96, dimensions=7168, forward_calls=4024, backward_calls=784,
        candidate_optimizer_steps=8, new_generations=0, new_Judge=0, persistent_checkpoints=0,
        arms=ARMS, source_roles=['REPLAY_FIT'], CHECK_used=False, replay_CE_optimization=False,
        initial_state='STAGED_220', RANK_RTOL=pm.RANK_RTOL,
        split_binding=c.digest([basis, held]), code=c.read(RUN/'private/GPU_SOURCE_VERSION.json'))
    c.write(RUN/'private/PROBE_LOCK.json', lock)
    c.write(RUN/'public/ADMISSION.json', dict(status='PASS', **{k:v for k,v in lock.items() if k not in ('split_binding','code')},
        projection_selfcheck=pm.selfcheck(), function_selfcheck=fm.selfcheck(),
        source_group_overlap=0, historical_source_exposure=True, independent_confirmation=False))
    p.done('PLAN_COMPLETE')


def logits(runtime, hook, batch):
    hook.set_teacher_routing(batch.labels)
    with torch.inference_mode():
        out = runtime.model(**batch.forward_kwargs())
    mask = batch.labels[:, 1:] != -100
    return out.logits[:, :-1][mask].detach().cpu().double(), batch.labels[:, 1:][mask].cpu()


def compare(reference, candidate, target):
    lp, lq = reference.log_softmax(-1), candidate.log_softmax(-1)
    kl = float((lp.exp() * (lp - lq)).sum(-1).mean())
    assert kl >= -1e-10 and torch.isfinite(lq).all()
    ix = torch.arange(len(target))
    old, new = reference.argmax(-1), candidate.argmax(-1)
    return dict(KL=max(kl, 0.), NLL=float(-lq[ix, target].mean()),
        NLL_change=float((lp[ix, target] - lq[ix, target]).mean()),
        tokens=len(target), correct_tokens=int((new == target).sum()),
        lost_correct_tokens=int(((old == target) & (new != target)).sum()),
        all_argmax_correct=bool((new == target).all()))


def one(runtime, t):
    from m3bench_repro.editors.llava_runtime import seed_everything
    seed_everything(t['seed'])
    expert = p.expert(d.start_state(t), t['seed'], runtime.device)
    before = {k:v.detach().clone() for k,v in expert.state_dict().items()}
    hook = scoped.replay.MedTraceLayerHook(runtime.get_module(c.LAYER), expert)
    hook.attach()
    rows = []
    basis, held = split()
    params = dict(expert.named_parameters())
    try:
        for i, row in enumerate(basis):
            c.budget()
            batch = runtime.build_edit_batch(scoped.replay.record(row, t))
            hook.set_teacher_routing(batch.labels)
            loss = runtime.compute_loss(batch)
            gradients = torch.autograd.grad(loss, [params[k] for k in KEYS])
            rows.append(torch.cat([g.detach().cpu().double().flatten() for g in gradients]))
            assert torch.isfinite(rows[-1]).all()
            del batch, loss, gradients
            if (i+1) % 24 == 0:print('BASIS', t['order'], i+1, flush=True)
        matrix = torch.stack(rows)
        native = [runtime.build_edit_batch(replace(c.record(t), question=q))
                  for q in [t['native']['question']] + t['fit_questions']]
        assert len(native) == 5 and all(b.target_token_ids[-1] == runtime.adapter.tokenizer.eos_token_id for b in native)
        # Match the original tail's seed/reset immediately before the candidate step.
        seed_everything(t['seed'])
        opt = p.optimizer(expert, runtime.model)
        opt.zero_grad(set_to_none=True)
        for b in native[:2]:
            hook.set_teacher_routing(b.labels)
            loss = runtime.compute_loss(b)
            assert torch.isfinite(loss)
            (.5 * loss).backward()
        norm = torch.nn.utils.clip_grad_norm_(expert.parameters(), 1.)
        assert torch.isfinite(norm) and all(v.grad is not None and torch.isfinite(v.grad).all() for v in expert.parameters())
        opt.step()
        groups = d.apply_direction(opt, {id(v): before[k] for k,v in expert.named_parameters()}, 'RAW')
        raw = {k:v.detach().clone() for k,v in expert.state_dict().items()}
        delta = flatten(raw) - flatten(before)
        projected, geometry = pm.project(matrix, delta)
        candidate = add(before, projected)
        actual = flatten(candidate) - flatten(before)
        matched, matching = fm.match(before, raw, fm.map_norm(before, candidate))
        geometry.update(actual_predicted_norm=float((matrix @ actual).norm()),
            raw_predicted_norm=float((matrix @ delta).norm()),
            roundoff_parameter_norm=float((actual - projected).norm()))
        states = dict(RAW=raw, PROJECTED=candidate, MATCHED_RAW=matched)
        diagnostics = []
        held_groups = list(dict.fromkeys(x['source_group'] for x in held))
        for i in range(101):
            c.budget()
            is_edit = i < 5
            source = None if is_edit else held[i-5]
            batch = native[i] if is_edit else runtime.build_edit_batch(scoped.replay.record(source, t))
            expert.load_state_dict(before)
            reference, target = logits(runtime, hook, batch)
            if i == 0:
                repeat, target2 = logits(runtime, hook, batch)
                assert torch.equal(reference, repeat) and torch.equal(target, target2), 'Baseline repeat mismatch'
            values = {}
            for label, state in states.items():
                expert.load_state_dict(state)
                value, target2 = logits(runtime, hook, batch)
                assert torch.equal(target, target2)
                values[label] = compare(reference, value, target)
            base = compare(reference, reference, target)
            diagnostics.append(dict(role='EDIT' if is_edit else 'HELDOUT_FIT', index=i if is_edit else i-5,
                group=None if is_edit else held_groups.index(source['source_group']),
                kind=None if is_edit else source['answer_kind'], baseline=base, candidates=values))
            if (i+1) % 24 == 0:print('DIAGNOSTIC', t['order'], i+1, flush=True)
        expert.load_state_dict(before)
        assert all(torch.equal(expert.state_dict()[k], v) for k,v in before.items())
        assert not any(v.grad is not None for v in runtime.model.parameters())
        return dict(status='COMPLETE', expert_order=t['order'], geometry=geometry,
            function_matching=matching, map_norms={k:fm.map_norm(before,v) for k,v in states.items()},
            RAW_groups=groups, diagnostics=diagnostics, forward_calls=503, backward_calls=98,
            candidate_optimizer_steps=1, state_restored_exact=True, baseline_repeat_exact=True,
            Base_gradient=False, lock=c.digest(c.read(RUN/'private/PROBE_LOCK.json')))
    finally:
        expert.load_state_dict(before)
        hook.detach()


def worker():
    assert (RUN/'private/PLAN_COMPLETE.json').exists()
    part = int(os.environ['PARTITION'])
    with p.lease(GPUS[part]):
        runtime, _ = c.load(GPUS[part])
        for task in d.selected()[part::len(GPUS)]:
            dest = RUN/'private/results'/f"{task['order']}.json"
            assert not dest.exists(), 'No implicit rerun of completed expert'
            c.write(dest, one(runtime, task))
            print('EXPERT_COMPLETE', task['order'], flush=True)
    p.done('WORKER_'+str(part))


def controller():
    import pipeline
    pipeline.wait([pipeline.launch('probe.py', 'response_worker', gpu, i) for i,gpu in enumerate(GPUS)])
    pipeline.wait([pipeline.launch('probe_report.py', 'response_report')])


if __name__ == '__main__':
    try:
        {'response_plan':plan, 'response_worker':worker, 'response_controller':controller}[os.environ['ACTION']]()
    except BaseException as error:
        c.write(RUN/'private'/('FAILURE_'+os.environ.get('ACTION','unknown')+'_'+os.environ.get('PARTITION','none')+'.json'),
                dict(error=repr(error), traceback=traceback.format_exc(), epoch=time.time()))
        raise
