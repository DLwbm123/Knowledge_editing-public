"""Read-only per-request protection gradients under fixed Q and empty-Q controls."""
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback

ROOT = Path(os.environ['V127_ROOT'])
PARENT = Path(os.environ['PARENT_RUN'])
TRAIN_ROOT = Path(os.environ['V126_PARENT'])
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

    audits, groups = [], []
    correctness = {(v['task_id'], v['query_id']): v['core_correct']
        for v in read(ROOT / 'TRAIN_CORRECTNESS_FROZEN.json')['rows']}
    assert len(correctness) == 84
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
        dest = TRAIN_ROOT / tid / 'REFERENCE_PRIORITY_COMPLETION'
        assert read(dest / 'TRAIN_COMPLETE.json')['save_load_exact']
        state = torch.load(dest / 'delta.pt', map_location=runtime.device, weights_only=True)
        expert = r.p.expert(state, lock['seed'], runtime.device)
        before = r.old.snapshot(expert)
        hook = r.q.scoped.replay.MedTraceLayerHook(runtime.get_module(r.c.LAYER), expert)
        hook.attach()
        try:
            with PrefixCache(runtime, batches + eos_batches) as prefix:
                gradients, losses = {}, {}
                kernel.install(expert, q)
                for name, source, masks in (
                        ('reference', batches[:5], [None] * 5), ('prefix_EOS', eos_batches, eos_masks)):
                    expert.zero_grad(set_to_none=True)
                    losses[name] = []
                    for batch, mask, weight in zip(source, masks, (.25, .0625, .0625, .0625, .0625)):
                        hook.set_teacher_routing(batch.labels) if mask is None else hook.set_request_routing(mask)
                        loss = runtime.compute_loss(batch)
                        assert torch.isfinite(loss)
                        (weight * loss).backward()
                        counts['backwards'] += 1
                        losses[name].append(float(loss.detach()))
                    gradients[name] = {k: p.grad.detach().clone() for k, p in expert.named_parameters()}
                requests = []
                for index, row in enumerate(task['replay_protect'], 5):
                    observed, vectors = {}, {}
                    batch = batches[index]
                    for condition in ('Q', 'EMPTY_Q'):
                        kernel.install(expert, q if condition == 'Q' else q[:, :0])
                        expert.zero_grad(set_to_none=True)
                        hook.set_teacher_routing(batch.labels)
                        loss = runtime.compute_loss(batch)
                        assert torch.isfinite(loss)
                        loss.backward()
                        counts['backwards'] += 1
                        vector = {k: p.grad.detach().clone() for k, p in expert.named_parameters()}
                        assert all(torch.isfinite(v).all() for v in vector.values())
                        vectors[condition] = vector
                        norms = [sum(float(vector[k].double().square().sum()) for k in group) ** .5
                                 for group in r.mm.GROUPS]
                        observed[condition] = dict(NLL=float(loss.detach()), gradient_norm_by_block=norms,
                            gradient_norm=sum(v * v for v in norms) ** .5)
                    qnorm, empty = (observed[c]['gradient_norm'] for c in ('Q', 'EMPTY_Q'))
                    requests.append(dict(query_id=row['query_id'], conditions=observed, vectors=vectors,
                        Q_to_empty_gradient_norm_ratio=qnorm / empty if empty else None,
                        locally_Q_gradient_zero_with_nonzero_empty=qnorm <= 1e-8 and empty > 1e-8,
                        old_training_core_correct=correctness[(tid, row['query_id'])]))
                assert prefix.misses == 10 + n_replay and prefix.hits == n_replay
            kernel.install(expert, q)
            gc, ge = gradients['reference'], gradients['prefix_EOS']
            gr = {k: sum(v['vectors']['Q'][k] for v in requests) / n_replay for k in gc}
            params = {k: v.detach().clone().requires_grad_(True) for k, v in before.items()}
            penalty = r.mm.operator_penalty(params, fisher)
            assert torch.isfinite(penalty)
            gf = dict(zip(params, torch.autograd.grad(coefficient * penalty, tuple(params.values()))))
            counts['operator_backwards'] += 1
            assert all(torch.isfinite(v).all() for gradient in (gc, ge, gr, gf) for v in gradient.values())
            raw_aux = add(ge, gr, gf)
            auxiliary, cap = cap_gradient(gc, raw_aux, r.mm.GROUPS)
            auxiliary, direction = remove_opposing(gc, auxiliary, r.mm.GROUPS)
            current_total = add(gc, auxiliary)
            for block in r.mm.GROUPS:
                square = sum(float(gc[k].double().square().sum()) for k in block)
                dot = sum(float((gc[k].double() * current_total[k].double()).sum()) for k in block)
                assert dot >= square - 1e-6 * square - 1e-20
            for v in requests:
                row = {k: value for k, value in v.items() if k != 'vectors'}
                row.update(task_id=tid, role=task['role'], blocks=[dict(index=i,
                    reference_vs_Q_request=metric(gc, v['vectors']['Q'], group),
                    Q_request_vs_current_expected_total=metric(v['vectors']['Q'], current_total, group))
                    for i, group in enumerate(r.mm.GROUPS)])
                audits.append(row)
                write(ROOT / 'requests' / tid / (v['query_id'] + '.json'), row)
            assert all(torch.equal(v, before[k]) for k, v in r.old.snapshot(expert).items())
            group = dict(task_id=tid, losses=losses, cap=cap, direction=direction,
                         weights_unchanged_exact=True, time=time.time())
            groups.append(group)
            write(ROOT / 'groups' / (tid + '.json'), group)
            write(ROOT / 'PROGRESS.json', dict(phase='PROTECTION_REACHABILITY_AUDIT',
                completed=len(audits), completed_groups=len(groups), counts=dict(counts), time=time.time()))
        finally:
            hook.detach()
        del prepared, batches, eos_batches, eos_masks, requests, q, expert
    assert len(audits) == 84 and len(groups) == 14
    assert counts['forwards'] == counts['backwards'] == 308 and counts['operator_backwards'] == 14
    assert counts['updates'] == counts['generations'] == 0
    budget()
    write(ROOT / 'AUDIT_FROZEN.json', dict(requests=audits, groups=groups, time=time.time()))
    write(ROOT / 'COMPLETE.json', dict(status='PROTECTION_REACHABILITY_AUDIT_COMPLETE', requests=84,
        groups=14, counts=dict(counts), weights_unchanged_exact=True,
        local_Q_zero_nonzero_empty=sum(v['locally_Q_gradient_zero_with_nonzero_empty'] for v in audits),
        next_training_preregistered=False, time=time.time()))
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
