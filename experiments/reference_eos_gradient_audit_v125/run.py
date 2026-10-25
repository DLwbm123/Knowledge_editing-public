"""Read-only fixed endpoint gradients for full-reference versus prefix-EOS losses."""
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback

ROOT = Path(os.environ['V125_ROOT'])
PARENT = Path(os.environ['PARENT_RUN'])
TRAIN_ROOT = Path(os.environ['V124_PARENT'])
Q_ROOT = Path(os.environ['PARENT_Q_ROOT'])
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
    import completion
    import module_run as r
    from balanced_fisher import cap_gradient, remove_opposing
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
    assert len(tasks) == 14 and sum(len(t['train']) for t in tasks) == 70
    assert sum(len(t['replay_protect']) for t in tasks) == 84
    contexts = {(v['task_id'], v['query_id']): v for v in read(ROOT / 'PREFIX_CONTEXTS_FROZEN.json')['contexts']}
    assert len(contexts) == 70
    runtime = load_real_runtime(SimpleNamespace(cpu_gate=PARENT / 'private/cpu_gate'))
    runtime.model.eval()
    r.old.fp32(runtime)
    assert not any(p.requires_grad for p in runtime.model.parameters())
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

    def prepare(rows):
        runtime.model.half()
        try:
            return r.s.prepare(runtime, bindings, template, rows, True)
        finally:
            r.old.fp32(runtime)

    def add(*values):
        return {k: sum(value[k] for value in values) for k in values[0]}

    def metric(a, b, group):
        aa = sum(float(a[k].double().square().sum()) for k in group)
        bb = sum(float(b[k].double().square().sum()) for k in group)
        ab = sum(float((a[k].double() * b[k].double()).sum()) for k in group)
        na, nb = aa ** .5, bb ** .5
        return dict(reference_norm=na, other_norm=nb, dot=ab,
                    cosine=ab / (na * nb) if na and nb else None,
                    norm_ratio=nb / na if na else None,
                    eligible_nonzero=na > 1e-8 and nb > 1e-8)

    audits = []
    for task in tasks:
        tid = task['task_id']
        q = torch.load(Q_ROOT / tid / 'basis.pt', map_location=runtime.device, weights_only=True)
        prepared = prepare(task['train'] + task['replay_protect'])
        batches = [r.old.batch_on(runtime, item[1]) for item in prepared]
        eos_batches, eos_masks = [], []
        for item in prepared[:5]:
            context = contexts[(tid, item[0]['query_id'])]
            assert context['original_binding'] == item[4]
            batch, mask = completion.make_batch(runtime, item[3], context['prefix_token_ids'])
            eos_batches.append(batch)
            eos_masks.append(mask)
        n_replay = len(task['replay_protect'])
        assert len(batches) == 5 + n_replay
        coefficient = read(ROOT / 'reference' / tid / 'CALIBRATION.json')['coefficient']
        with PrefixCache(runtime, batches + eos_batches) as prefix:
            for endpoint in ('INITIAL', 'TERMINAL'):
                if endpoint == 'INITIAL':
                    state = r.mm.canonicalize(r.p.tr.TT4(lock['seed'], 8, 8).state_dict())[0]
                else:
                    dest = TRAIN_ROOT / tid / 'SOURCE_PREFIX_COMPLETION'
                    assert read(dest / 'TRAIN_COMPLETE.json')['save_load_exact']
                    state = torch.load(dest / 'delta.pt', map_location=runtime.device, weights_only=True)
                expert = r.p.expert(state, lock['seed'], runtime.device)
                kernel.install(expert, q)
                hook = r.q.scoped.replay.MedTraceLayerHook(runtime.get_module(r.c.LAYER), expert)
                hook.attach()
                before = r.old.snapshot(expert)
                try:
                    gradients, losses = {}, {}
                    for name in ('reference', 'prefix_EOS', 'replay'):
                        expert.zero_grad(set_to_none=True)
                        losses[name] = []
                        if name == 'replay':
                            chosen = [(batches[i], None, 1 / n_replay) for i in range(5, len(batches))]
                        else:
                            source = eos_batches if name == 'prefix_EOS' else batches[:5]
                            masks = eos_masks if name == 'prefix_EOS' else [None] * 5
                            chosen = list(zip(source, masks, (.25, .0625, .0625, .0625, .0625)))
                        for batch, mask, weight in chosen:
                            if mask is None:
                                hook.set_teacher_routing(batch.labels)
                            else:
                                hook.set_request_routing(mask)
                            loss = runtime.compute_loss(batch)
                            assert torch.isfinite(loss)
                            (weight * loss).backward()
                            counts['backwards'] += 1
                            losses[name].append(float(loss.detach()))
                        gradients[name] = {k: p.grad.detach().clone() for k, p in expert.named_parameters()}
                    params = {k: v.detach().clone().requires_grad_(True) for k, v in before.items()}
                    penalty = r.mm.operator_penalty(params, fisher)
                    assert torch.isfinite(penalty)
                    gf = dict(zip(params, torch.autograd.grad(coefficient * penalty, tuple(params.values()))))
                    counts['operator_backwards'] += 1
                    gc, ge, gr = (gradients[k] for k in ('reference', 'prefix_EOS', 'replay'))
                    assert all(torch.isfinite(v).all() for gradient in (gc, ge, gr, gf) for v in gradient.values())
                    old_primary = add(gc, ge)
                    old_aux, old_cap = cap_gradient(old_primary, add(gr, gf), r.mm.GROUPS)
                    old_aux, old_direction = remove_opposing(old_primary, old_aux, r.mm.GROUPS)
                    old_total = add(old_primary, old_aux)
                    raw_aux = add(ge, gr, gf)
                    new_aux, new_cap = cap_gradient(gc, raw_aux, r.mm.GROUPS)
                    new_aux, new_direction = remove_opposing(gc, new_aux, r.mm.GROUPS)
                    new_total = add(gc, new_aux)
                    blocks = []
                    for index, group in enumerate(r.mm.GROUPS):
                        square = sum(float(gc[k].double().square().sum()) for k in group)
                        counterfactual = metric(gc, new_total, group)
                        assert counterfactual['dot'] >= square - 1e-6 * square - 1e-20
                        blocks.append(dict(index=index, keys=list(group), reference_vs_prefix_EOS=metric(gc, ge, group),
                            reference_vs_replay=metric(gc, gr, group), reference_vs_raw_fisher=metric(gc, gf, group),
                            reference_vs_raw_combined_aux=metric(gc, raw_aux, group),
                            reference_vs_current_unclipped_total=metric(gc, old_total, group),
                            reference_vs_counterfactual_unclipped_total=counterfactual))
                    assert all(torch.equal(v, before[k]) for k, v in r.old.snapshot(expert).items())
                    row = dict(task_id=tid, role=task['role'], endpoint=endpoint, losses=losses,
                        blocks=blocks, coefficient=coefficient, fisher_penalty=float(penalty.detach()),
                        old_cap=old_cap, old_direction=old_direction, counterfactual_cap=new_cap,
                        counterfactual_direction=new_direction, weights_unchanged_exact=True,
                        counts=dict(counts), time=time.time())
                    audits.append(row)
                    write(ROOT / 'states' / tid / (endpoint + '.json'), row)
                    write(ROOT / 'PROGRESS.json', dict(phase='REFERENCE_EOS_GRADIENT_AUDIT',
                        completed=len(audits), task=tid, endpoint=endpoint, counts=dict(counts), time=time.time()))
                finally:
                    hook.detach()
            assert prefix.misses == 10 + n_replay and prefix.hits == 10 + n_replay
        del batches, eos_batches, eos_masks, prepared, q
    assert len(audits) == 28 and counts['forwards'] == counts['backwards'] == 448
    assert counts['operator_backwards'] == 28 and counts['updates'] == counts['generations'] == 0
    budget()
    eligible, conflicts = [], []
    for row in audits:
        if row['endpoint'] != 'TERMINAL':
            continue
        for block in row['blocks']:
            eos = block['reference_vs_prefix_EOS']
            combined = block['reference_vs_raw_combined_aux']
            if eos['eligible_nonzero'] and combined['eligible_nonzero']:
                item = dict(task_id=row['task_id'], block=block['index'], reference_vs_EOS=eos,
                            reference_vs_combined_aux=combined)
                eligible.append(item)
                if eos['cosine'] < -1e-6 and combined['cosine'] < -1e-6:
                    conflicts.append(item)
    write(ROOT / 'AUDIT_FROZEN.json', dict(states=audits, time=time.time()))
    write(ROOT / 'COMPLETE.json', dict(status='REFERENCE_EOS_AUDIT_COMPLETE', states=28,
        eligible_terminal_blocks=len(eligible), terminal_dual_conflict_blocks=conflicts,
        reference_priority_mechanism_test_admissible=bool(conflicts),
        weights_unchanged_exact=True, counts=dict(counts), time=time.time()))
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
