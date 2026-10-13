"""Matched C/M1/REFINE160 trajectories and real incremental original R0 banks."""
import copy
import math
import os
import subprocess
from pathlib import Path
import time
import traceback
import numpy as np
import torch
import full_run as finite
import full146_contract as contract
from followup146 import gate

r, c, RUN, g = finite.r, finite.c, finite.RUN, finite.g
UP = Path(os.environ['UPSTREAM_ROOT'])
r.COUNTS['local_solves'] = 0


def tasks():
    return c.read(RUN/'private/ORIGINAL146_TASKS.json')


def anchors():
    return c.read(RUN/'private/FROZEN_ANCHORS146.json')


def source(task):
    rows = [dict(task['native'], query_id=f"edit_{task['order']}_{i}", question=q,
                 audit_role='NATIVE' if i == 0 else 'FIT')
            for i, q in enumerate([task['native']['question']] + task['fit_questions'])]
    return rows + [dict(row, audit_role='GFIT') for row in anchors() if row['owner'] == task['order']]


def point(task, arm):
    return RUN/'private/candidates'/arm/f"{task['order']}.pt"


def dest(row):
    directory = f"{row['method']}_{row['mode']}_p{row['prefix']}"
    return RUN/'private/outputs'/directory/str(row['owner'])/(c.digest(row['query']['query_id'])+'.json')


def base_path(query):
    return dest(dict(method='BASE', mode='base', prefix=0, owner=0, query=query))


def count(*_):
    c.budget()
    r.COUNTS['forwards'] += 1
    if not r.GENERATING:
        r.COUNTS['non_generation_forwards'] += 1
        # Disjoint stage reservations sum to less than the frozen 3.2M ceiling.
        limit = {'full146_train': 2900000//3, 'full146_single': 100000//3,
                 'full146_bank': 200000//3}[os.environ['ACTION']]
        assert r.COUNTS['non_generation_forwards'] <= limit


r.count = count
r.s.tasks, r.s.source, r.s.point = tasks, source, point
r.s.baseline = lambda task, row: base_path(row)
for name, module in (('joint', finite.j.jm), ('repair', finite.rm)):
    original = module.solve
    def counted(*args, _solve=original, **kwargs):
        r.COUNTS['local_solves'] += 1
        cap = len(tasks()[int(os.environ['PARTITION'])::3])*160*16
        assert r.COUNTS['local_solves'] <= cap
        return _solve(*args, **kwargs)
    module.solve = counted


def require_ready():
    assert gate(c.read(UP/'public/FULL_RESULTS.json'))
    assert c.read(UP/'private/FULL_COMPLETE.json')['status'] == 'COMPLETE'
    assert c.read(UP/'private/FULL_HANDOFF_EXIT1.json')['exit_code'] == 0
    assert c.read(UP/'public/FULL_EXECUTION_AUDIT.json')['status'] == 'PASS'
    assert c.read(UP/'private/FULL_DELIVERY.json')['remote_SHA_match']
    assert c.read(RUN/'private/FULL146_READY.json')['status'] == 'PASS'
    review = c.read(RUN/'private/SOURCE_REVIEW146.json')
    result = contract.source_review(tasks(), c.read(RUN/'private/FROZEN_TASKS146.json'),
        anchors(), review['owners'], set(c.read(RUN/'private/FORBIDDEN_SOURCE_GROUPS146.json')))
    assert result['status'] == 'PASS'
    assert c.read(RUN/'private/FULL146_BINDING.json')['source'] == c.digest([tasks(), anchors(), review])


def snapshot_numpy():
    s = np.random.get_state()
    return dict(kind=s[0], keys=s[1].tolist(), position=s[2], has_gauss=s[3], cached_gaussian=s[4])


def save_resume(task, arm, expert, optimizer, scalar, step, curve, budget):
    c.save(RUN/'private/resume146'/f"{arm}_{task['order']}.pt", dict(
        state=r.s.previous.weights(expert.state_dict()), optimizer=optimizer.state_dict() if optimizer else None,
        scalar=copy.deepcopy(scalar), step=step, seed=task['seed'], curve=curve, rng=c.rng(),
        numpy_rng=snapshot_numpy(), source_order=[v['query_id'] for v in source(task)],
        scientific_lock=c.digest(c.read(RUN/'private/SCOPE_LOCK.json')), budget=budget, counts=dict(r.COUNTS)))
    r.save_counts()
    g.memory_record(int(os.environ['GPU']), 'FULL146_CHECKPOINT', prefix='full146_train')


def reused(task, arm):
    entry = c.read(RUN/'private/REUSE146.json').get(f"{arm}:{task['order']}")
    if entry is None:
        return False
    assert entry['source'] == contract.source_binding(source(task)) and entry['task'] == contract.task_binding(task)
    saved = r.p.load_state(Path(entry['weight']))
    cold = r.q.d.start_state(task)
    expected = cold if arm == 'C' else r.mm.canonicalize(cold)[0]
    assert all(torch.equal(expected[k], saved['states']['BASE'][k]) for k in r.mm.KEYS)
    result = c.read(Path(entry['result']))
    assert result['save_load_exact'] and len(result['curve']) == 160
    assert arm == 'C' or result['scalar_optimizer_state']['step'] == 160
    c.save(point(task, arm), saved)
    c.write(RUN/'private/results'/arm/f"{task['order']}.json", dict(result, reused=True,
            source_binding=c.digest(source(task)), scientific_lock=c.digest(c.read(RUN/'private/SCOPE_LOCK.json'))))
    return True


def train():
    from m3bench_repro.editors.llava_runtime import seed_everything
    require_ready()
    c.write(RUN/'public/PROGRESS.json', dict(status='FULL146_TRAINING_ADMITTED_WAITING_RESOURCES', complete=False))
    part, gpu = int(os.environ['PARTITION']), int(os.environ['GPU'])
    assigned = tasks()[part::3]
    with r.p.lease(gpu):
        runtime, _, handle = r.load(gpu)
        assert all(v.dtype == torch.float16 for v in runtime.model.parameters() if v.is_floating_point())
        g.memory_record(gpu, 'FULL146_LOADED', prefix='full146_train')
        cache = {k: v.to(runtime.device) if isinstance(v, torch.Tensor) else v
                 for k, v in r.p.load_state(RUN/'private/inherited/OPERATOR_FISHER.pt').items()}
        try:
            for task in assigned:
                runtime.model.half()
                frozen = [r.old.cpu(runtime.build_edit_batch(r.s.record(task, row))) for row in source(task)]
                r.old.fp32(runtime)
                batches = [r.old.batch_on(runtime, b) for b in frozen]
                assert len(batches) == 7
                budget = None
                for arm in contract.ARMS:
                    assert not point(task, arm).exists(), 'No implicit training restart'
                    if reused(task, arm):
                        continue
                    seed_everything(task['seed'])
                    initial = r.q.d.start_state(task)
                    assert sum(v.numel() for v in initial.values()) == 7168 and torch.count_nonzero(initial['G1']) == 0
                    if arm != 'C':
                        initial = r.mm.canonicalize(initial)[0]
                    expert = r.p.expert(initial, task['seed'], runtime.device)
                    base = r.old.snapshot(expert)
                    hook = r.q.scoped.replay.MedTraceLayerHook(runtime.get_module(c.LAYER), expert)
                    hook.attach()
                    opt = r.p.optimizer(expert, runtime.model) if arm == 'C' else None
                    scalar = None if arm == 'C' else dict(step=0, second_moments=[0., 0.])
                    start, curve = dict(r.COUNTS), []
                    try:
                        refs = [r.q.logits(runtime, hook, b, verify_base=i == 0) for i, b in enumerate(batches)]
                        if arm != 'C':
                            reference = c.read(RUN/'private/results/C'/f"{task['order']}.json")
                            saved = r.p.load_state(point(task, 'C'))
                            penalty = float(r.mm.operator_penalty({k: v.to(runtime.device) for k, v in saved['states']['FINAL'].items()}, cache))
                            assert math.isfinite(penalty) and penalty > 0
                            inherited_C = c.read(RUN/'private/REUSE146.json').get(f"C:{task['order']}")
                            if inherited_C:
                                previous_penalty = c.read(Path(os.environ['MODULE_PARENT'])/'private/calibration'/f"{task['order']}.json")['C_operator_penalty']
                                assert abs(penalty-previous_penalty) <= 1e-12+1e-4*abs(previous_penalty)
                                c.write(RUN/'private/C_FISHER_PARITY146'/f"{task['order']}.json",
                                    dict(status='PASS', actual=penalty, expected=previous_penalty))
                            budget = .1*penalty
                            previous = RUN/'private/calibration146'/f"{task['order']}.json"
                            if previous.exists():
                                assert abs(c.read(previous)['C_operator_penalty']-penalty) <= 1e-12+1e-4*penalty
                            else:
                                c.write(previous, dict(C_operator_penalty=penalty, budget=budget, fraction=.1,
                                    source_binding=c.digest(source(task)), heldout_used=False))
                        for step in range(1, 161):
                            c.budget()
                            before = r.old.snapshot(expert)
                            if opt:
                                opt.zero_grad(set_to_none=True)
                            _, losses = r.source_gradients(runtime, hook, expert, batches, step, False)
                            norm = torch.nn.utils.clip_grad_norm_(expert.parameters(), 1.)
                            assert torch.isfinite(norm) and all(v.grad is not None and torch.isfinite(v.grad).all() for v in expert.parameters())
                            if arm == 'C':
                                opt.step()
                                r.q.d.apply_direction(opt, {id(v): before[k] for k, v in expert.named_parameters()}, 'RAW')
                                chosen, audit = r.old.snapshot(expert), dict(kind='C_RAW')
                            else:
                                pure, scalar = r.mm.canonical_rms_step(before, {k: v.grad.detach() for k, v in expert.named_parameters()}, scalar)
                                assert scalar['step'] == step
                                target = reference['curve'][step-1]['operator_step_norm']
                                if target == 0 or r.fm.map_norm(before, pure) <= 1e-14:
                                    chosen, audit = before, dict(kind='ZERO_DIRECTION')
                                else:
                                    pure = g.canonical_match(before, pure, target)
                                    if arm == 'M1':
                                        chosen, audit = pure, dict(kind='M1_MATCHED')
                                    else:
                                        chosen, audit = finite.choose(runtime, hook, expert, before, pure, step,
                                            batches, losses, cache, budget)
                            expert.load_state_dict(chosen)
                            r.COUNTS['updates'] += 1
                            curve.append(dict(step=step, source_NLL=losses, decision=audit,
                                              operator_step_norm=r.fm.map_norm(before, chosen)))
                            assert r.COUNTS['backwards'] <= len(assigned)*160*57
                            if step % 20 == 0:
                                save_resume(task, arm, expert, opt, scalar, step, curve, budget)
                        r.finish_candidate(runtime, hook, expert, task, arm, base, batches, refs, curve, start, scalar,
                            mechanism=dict(budget=budget, final_penalty=float(r.mm.operator_penalty(chosen, cache))))
                        path = RUN/'private/resume146'/f"{arm}_{task['order']}.pt"
                        c.write(RUN/'private/retired146'/f"{arm}_{task['order']}.json", dict(bytes=path.stat().st_size,
                            reason='Complete160 final/save-load/state counters retained', deleted_resume=True))
                        path.unlink()
                    finally:
                        hook.detach()
                print('FULL146_OWNER_COMPLETE', task['order'], flush=True)
            g.memory_record(gpu, 'FULL146_PARTITION_COMPLETE', complete=True, prefix='full146_train')
        finally:
            handle.remove()
            r.save_counts()


def evaluate():
    from m3bench_repro.editors.routing import MemoryRouter, decision_as_json
    require_ready()
    assert c.read(RUN/'private/FULL146_CANDIDATES_FROZEN.json')['candidates'] == 438
    part, gpu = int(os.environ['PARTITION']), int(os.environ['GPU'])
    action = os.environ['ACTION']
    rows = c.read(RUN/'private/EVAL_PLAN146.json')
    assigned = [v for i, v in enumerate(rows) if i % 3 == part and
                ((v['mode'] in ('base', 'single')) if action == 'full146_single' else (v['mode'] in ('bank', 'insertion')))]
    byid = {t['edit_id']: t for t in tasks()}
    byorder = {t['order']: t for t in tasks()}
    with r.p.lease(gpu):
        runtime, bindings, handle = r.load(gpu)
        assert all(v.dtype == torch.float16 for v in runtime.model.parameters() if v.is_floating_point()), 'Original FP16 prefill parameters must survive chunk conversion exactly'
        route_layer = runtime.target_lock['balancedit']['targets'][0]
        entries = []
        for task in tasks():
            path = Path(c.read(RUN/'private/ROUTER_BINDINGS146.json')[str(task['order'])])
            saved = r.p.load_state(path)
            assert len(saved['entries']) == 1 and saved['entries'][0]['logical_edit_id'] == task['edit_id']
            entries.append(saved['entries'][0])
        routers, expert, hook = {}, None, None
        try:
            for begin in range(0, len(assigned), 64):
                runtime.model.half()
                prepared = []
                for row in assigned[begin:begin+64]:
                    assert not dest(row).exists(), 'No implicit generation retry'
                    task = byorder[row['owner']] if row['owner'] else tasks()[0]
                    query = dict(row['query'], audit_role=row['query'].get('audit_role', 'EVAL'))
                    route, selected = None, None
                    if row['method'] != 'BASE':
                        key = (row['mode'], row['prefix'], row['owner'] if row['mode'] == 'single' else 0)
                        if key not in routers:
                            bank = [entries[row['owner']-1]] if row['mode'] == 'single' else entries[:row['prefix']]
                            routers[key] = MemoryRouter.from_state(dict(distance='euclidean', entries=bank), device=runtime.device)
                        if hook is not None:
                            hook.detach(); hook = None
                        batch = runtime.build_question_batch(r.s.record(task, query))
                        with torch.inference_mode():
                            value = runtime.extract_layer_input_key(batch, module_path=route_layer, pooling='mean')
                        decision = routers[key].route(value)
                        route = decision_as_json(decision)
                        selected = decision.logical_edit_id if decision.activated else None
                    # Prepare prompt embeddings with editing disabled and the original FP16 prefill.
                    if hook is not None:
                        hook.detach(); hook = None
                    item = r.s.prepare(runtime, bindings, task, [query])[0]
                    prepared.append((row, task, query, item, route, selected))
                r.old.fp32(runtime)
                for row, task, query, item, route, selected in prepared:
                    if selected is not None:
                        assert selected in byid and (row['mode'] == 'single' or byid[selected]['order'] <= row['prefix'])
                        selected_task = byid[selected]
                        state = r.p.load_state(point(selected_task, row['method']))['states']['FINAL']
                        expert = r.p.expert(state, selected_task['seed'], runtime.device).requires_grad_(False)
                        hook = r.q.scoped.replay.MedTraceLayerHook(runtime.get_module(c.LAYER), expert); hook.attach()
                    r.s.emit(runtime, hook, task, row['method'], item, r.COUNTS, dest=dest(row), route=route)
                    data = c.read(dest(row))
                    active = {(t['native']['image_sha256'], t['native']['question']) for t in tasks()[:row['prefix']]}
                    data.update(method=row['method'], mode=row['mode'], prefix=row['prefix'],
                        eval_owner=row['owner'], consumer_id=contract.consumer_id(row),
                        effective_expert=selected, original_router=True,
                        active_target=row['mode'] in ('bank', 'insertion') and
                            (query.get('image_sha256'), query['question']) in active)
                    c.write(dest(row), data)
                    if hook is not None:
                        hook.detach(); hook = None
                    r.save_counts()
            c.write(RUN/'private'/f"DONE_{action}_{part}.json", dict(status='COMPLETE', outputs=len(assigned), epoch=time.time()))
        finally:
            if hook is not None:
                hook.detach()
            handle.remove(); r.save_counts()


def controller():
    import pipeline
    with (RUN/'private/FULL146_ONCE').open('x'):
        pass
    require_ready()
    c.write(RUN/'public/PROGRESS.json', dict(status='FULL146_TRAINING_ADMITTED_WAITING_RESOURCES', complete=False))
    def stage(action):
        jobs, pending = {}, list(range(3))
        devices = c.read(RUN/'RUN_MANIFEST.json')['physical_GPUs']
        while pending or jobs:
            c.budget()
            for p in list(jobs):
                if p.poll() is not None:
                    assert p.returncode == 0, 'Actual worker failure; retain state, no blind retry'
                    del jobs[p]
            for gpu in devices:
                if not pending or len(jobs) >= 3:
                    break
                active = [v for v in c.read(RUN/'RESOURCE_LEDGER.json')['gpu_sessions'] if not v.get('ended_epoch')]
                uuid = c.read(RUN/'PLAN_CONFIG.json')['hardware']['UUIDs'][str(gpu)]
                if gpu in jobs.values() or any(v['gpu_uuid'] == uuid for v in active):
                    continue
                observed, free = subprocess.check_output(['nvidia-smi', '-i', str(gpu),
                    '--query-gpu=uuid,memory.free', '--format=csv,noheader,nounits'], text=True).strip().split(', ')
                assert observed == uuid
                if int(free) >= 45*1024:
                    jobs[pipeline.launch('full146_run.py', action, gpu, pending.pop(0))] = gpu
            if pending or jobs:
                c.write(RUN/'public/RESOURCE_WAIT146.json', dict(action=action, waiting_parts=pending,
                    active_workers=len(jobs), required_free_MiB=45*1024, epoch=time.time()))
                time.sleep(5)
    stage('full146_train')
    for task in tasks():
        for arm in contract.ARMS:
            result = c.read(RUN/'private/results'/arm/f"{task['order']}.json")
            assert result['save_load_exact'] and len(result['curve']) == 160
    c.write(RUN/'private/FULL146_CANDIDATES_FROZEN.json', dict(candidates=438, epoch=time.time(), selection=False))
    c.write(RUN/'public/PROGRESS.json', dict(status='FULL146_FROZEN_SINGLE_AND_BASE_GENERATING', complete=False))
    stage('full146_single')
    c.write(RUN/'public/PROGRESS.json', dict(status='FULL146_REAL_INCREMENTAL_BANK_GENERATING', complete=False))
    stage('full146_bank')
    files = list((RUN/'private/outputs').glob('*/*/*.json'))
    assert len(files) == 17373
    expected = {contract.consumer_id(v) for v in c.read(RUN/'private/EVAL_PLAN146.json')}
    assert {c.read(p)['consumer_id'] for p in files} == expected
    counts = [c.read(p) for p in (RUN/'private').glob('COUNTS_full146_*.json')]
    for key, cap in contract.CAPS.items():
        if key != 'Judge_attempts':
            assert sum(v[key] for v in counts) <= cap
    c.write(RUN/'private/MODULE_GENERATION_COMPLETE.json', dict(outputs=17373, epoch=time.time(), counts=counts))
    pipeline.wait([pipeline.launch('full146_queue.py', 'full146_ingest')])
    c.write(RUN/'public/PROGRESS.json', dict(status='FULL146_GENERATED_SCORING_PENDING', complete=False))
    while not (RUN/'private/judge_full146_astra_medium/ALL_WORKERS_COMPLETE.json').exists():
        c.budget()
        assert not list((RUN/'private/judge_full146_astra_medium/workers').glob('*/SCORER_FAILURE.json'))
        time.sleep(20)
    pipeline.wait([pipeline.launch('full146_report.py', 'full146_report')])
    paths = [point(t, arm) for t in tasks() for arm in contract.ARMS]
    paths += list((RUN/'private/inherited').rglob('*.pt'))
    assert not list((RUN/'private/resume146').glob('*.pt'))
    assert all(p.is_file() and not p.is_symlink() and RUN.resolve() in p.resolve().parents for p in paths)
    c.write(RUN/'private/FULL146_CONSUMED.json', dict(paths=list(map(str, paths)),
        reason='All registered source/training/single/insertion/bank/scoring/report consumers complete'))
    for p in paths:
        p.unlink()
    c.write(RUN/'public/FULL146_EXECUTION_AUDIT.json', dict(status='PASS', deleted_files=len(paths),
        owned_weight_files_remaining=0, original_order=True, original_R0=True, counts=counts))
    c.write(RUN/'private/FULL146_COMPLETE.json', dict(status='COMPLETE', epoch=time.time()))
    c.write(RUN/'public/PROGRESS.json', dict(status='FULL146_COMPLETE_REVIEW_PUBLICATION_PENDING',
        execution_complete=True, publication_complete=False))


if __name__ == '__main__':
    try:
        action = os.environ['ACTION']
        if action == 'full146_train': train()
        elif action in ('full146_single', 'full146_bank'): evaluate()
        elif action == 'full146_controller': controller()
        else: raise ValueError(action)
    except BaseException as error:
        r.save_counts()
        c.write(RUN/'private'/f"FAILURE_{os.environ['ACTION']}_{os.environ.get('PARTITION')}.json",
            dict(error=repr(error), traceback=traceback.format_exc(), counts=r.COUNTS, epoch=time.time()))
        raise
