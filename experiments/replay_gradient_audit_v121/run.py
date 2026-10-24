"""Fixed endpoint expected-gradient audit; no optimizer step or generated answers."""
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback

ROOT = Path(os.environ['V121_ROOT'])
PARENT = Path(os.environ['PARENT_RUN'])
TRAIN_ROOT = Path(os.environ['V120_PARENT'])
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
    assert len(tasks) == 14
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

    def add(a, b):
        return {k: a[k] + b[k] for k in a}

    def metric(a, b, group):
        aa = sum(float(a[k].double().square().sum()) for k in group)
        bb = sum(float(b[k].double().square().sum()) for k in group)
        ab = sum(float((a[k].double() * b[k].double()).sum()) for k in group)
        na, nb = aa ** .5, bb ** .5
        return dict(source_norm=na, other_norm=nb, dot=ab,
                    cosine=ab / (na * nb) if na and nb else None,
                    norm_ratio=nb / na if na else None,
                    eligible_nonzero=na > 1e-8 and nb > 1e-8)

    audits = []
    for task in tasks:
        q = torch.load(Q_ROOT / task['task_id'] / 'basis.pt', map_location=runtime.device, weights_only=True)
        prepared = prepare(task['train'] + task['replay_protect'])
        batches = [r.old.batch_on(runtime, item[1]) for item in prepared]
        n_replay = len(task['replay_protect'])
        assert len(batches) == 5 + n_replay
        coefficient = read(ROOT / 'reference' / task['task_id'] / 'CALIBRATION.json')['coefficient']
        with PrefixCache(runtime, batches) as prefix:
            for arm in ('REPLAY_DOUBLE', 'PATH_REPLAY'):
                for endpoint in ('INITIAL', 'TERMINAL'):
                    if endpoint == 'INITIAL':
                        state = r.mm.canonicalize(r.p.tr.TT4(lock['seed'], 8, 8).state_dict())[0]
                    else:
                        assert read(TRAIN_ROOT / task['task_id'] / arm / 'TRAIN_COMPLETE.json')['save_load_exact']
                        state = torch.load(TRAIN_ROOT / task['task_id'] / arm / 'delta.pt',
                                           map_location=runtime.device, weights_only=True)
                    expert = r.p.expert(state, lock['seed'], runtime.device)
                    kernel.install(expert, q if arm == 'PATH_REPLAY' else q[:, :0])
                    hook = r.q.scoped.replay.MedTraceLayerHook(runtime.get_module(r.c.LAYER), expert)
                    hook.attach()
                    before = r.old.snapshot(expert)
                    try:
                        expert.zero_grad(set_to_none=True)
                        source_losses = []
                        for index, weight in enumerate((.5, .125, .125, .125, .125)):
                            hook.set_teacher_routing(batches[index].labels)
                            loss = runtime.compute_loss(batches[index])
                            assert torch.isfinite(loss)
                            (weight * loss).backward()
                            counts['backwards'] += 1
                            source_losses.append(float(loss.detach()))
                        gs = {k: p.grad.detach().clone() for k, p in expert.named_parameters()}
                        expert.zero_grad(set_to_none=True)
                        replay_losses = []
                        for index in range(5, len(batches)):
                            hook.set_teacher_routing(batches[index].labels)
                            loss = runtime.compute_loss(batches[index])
                            assert torch.isfinite(loss)
                            (loss / n_replay).backward()
                            counts['backwards'] += 1
                            replay_losses.append(float(loss.detach()))
                        gr = {k: p.grad.detach().clone() for k, p in expert.named_parameters()}
                        params = {k: v.detach().clone().requires_grad_(True) for k, v in before.items()}
                        penalty = r.mm.operator_penalty(params, fisher)
                        assert torch.isfinite(penalty)
                        gf = dict(zip(params, torch.autograd.grad(coefficient * penalty, tuple(params.values()))))
                        counts['operator_backwards'] += 1
                        assert all(torch.isfinite(v).all() for gradients in (gs, gr, gf) for v in gradients.values())
                        combined = add(gr, gf)
                        source_replay = add(gs, gr)
                        old_fisher, _ = cap_gradient(source_replay, gf, r.mm.GROUPS)
                        old_fisher, _ = remove_opposing(source_replay, old_fisher, r.mm.GROUPS)
                        old_total = add(source_replay, old_fisher)
                        priority_aux, cap_audit = cap_gradient(gs, combined, r.mm.GROUPS)
                        priority_aux, direction_audit = remove_opposing(gs, priority_aux, r.mm.GROUPS)
                        priority_total = add(gs, priority_aux)
                        blocks = []
                        for index, group in enumerate(r.mm.GROUPS):
                            s2 = sum(float(gs[k].double().square().sum()) for k in group)
                            priority = metric(gs, priority_total, group)
                            assert priority['dot'] >= s2 - 1e-6 * s2 - 1e-20
                            blocks.append(dict(index=index, keys=list(group), source_vs_replay=metric(gs, gr, group),
                                source_vs_raw_fisher=metric(gs, gf, group),
                                source_vs_raw_combined_aux=metric(gs, combined, group),
                                source_vs_original_unclipped_total=metric(gs, old_total, group),
                                source_vs_priority_unclipped_total=priority,
                                priority_combined_aux=metric(gs, priority_aux, group)))
                        assert all(torch.equal(p, before[k]) for k, p in r.old.snapshot(expert).items())
                        row = dict(task_id=task['task_id'], role=task['role'], arm=arm, endpoint=endpoint,
                            source_NLL=source_losses, expected_source_NLL=.5 * source_losses[0] + .125 * sum(source_losses[1:]),
                            replay_NLL=replay_losses, expected_replay_NLL=sum(replay_losses) / n_replay,
                            fisher_penalty=float(penalty.detach()), coefficient=coefficient,
                            blocks=blocks, priority_cap=cap_audit, priority_direction=direction_audit,
                            weights_unchanged_exact=True, counts=dict(counts), time=time.time())
                        audits.append(row)
                        write(ROOT / 'states' / task['task_id'] / (arm + '_' + endpoint + '.json'), row)
                        write(ROOT / 'PROGRESS.json', dict(phase='GRADIENT_AUDIT', completed=len(audits),
                            task=task['task_id'], arm=arm, endpoint=endpoint, counts=dict(counts), time=time.time()))
                    finally:
                        hook.detach()
            assert prefix.misses == len(batches) and prefix.hits == 3 * len(batches)
        del batches, prepared, q
    assert len(audits) == 56 and counts['forwards'] == counts['backwards'] == 616
    assert counts['operator_backwards'] == 56 and counts['updates'] == counts['generations'] == 0
    eligible = []
    conflicts = []
    for row in audits:
        if row['arm'] == 'PATH_REPLAY' and row['endpoint'] == 'TERMINAL':
            for block in row['blocks']:
                m = block['source_vs_raw_combined_aux']
                if m['eligible_nonzero']:
                    item = dict(task_id=row['task_id'], block=block['index'], **m)
                    eligible.append(item)
                    if m['cosine'] < -1e-6:
                        conflicts.append(item)
    write(ROOT / 'GRADIENT_AUDIT_FROZEN.json', dict(states=audits, time=time.time()))
    write(ROOT / 'COMPLETE.json', dict(status='GRADIENT_AUDIT_COMPLETE',
        source_priority_test_admissible=bool(conflicts), eligible_terminal_PATH_blocks=len(eligible),
        conflicting_terminal_PATH_blocks=conflicts, counts=dict(counts), time=time.time()))
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
