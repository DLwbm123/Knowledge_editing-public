"""Original single/insertion/prefix R0 evaluation, using actual selected experts."""
import os
from pathlib import Path
import time
import torch
import contract
from runtime146 import r,c,RUN,tasks,point,dest,require_ready


def bank_entries(entries,row):
    assert len(entries)==146 and len({e['logical_edit_id'] for e in entries})==146
    assert row['method'] in ('C','M1','M3','M13','M13B','FISHER_MIN')
    if row['mode']=='single':
        assert 1<=row['owner']<=146;return [entries[row['owner']-1]]
    assert row['mode'] in ('bank','insertion') and 1<=row['prefix']<=146
    return entries[:row['prefix']]

def evaluate(expected_candidates=584,partitions=3,admit=require_ready,
        frozen_file='M13_146_CANDIDATES_FROZEN.json',scope_file='SCOPE_LOCK.json',source_file='GPU_SOURCE_VERSION.json',
    plan_file='EVAL_PLAN146.json', runtime_check=None):
    from m3bench_repro.editors.routing import MemoryRouter, decision_as_json
    admit()
    assert c.read(RUN/'private'/frozen_file)['candidates'] == expected_candidates
    part, gpu = int(os.environ['PARTITION']), int(os.environ['GPU'])
    action = os.environ['ACTION']
    rows = c.read(RUN/'private'/plan_file)
    assert partitions in (2,3) and 0<=part<partitions
    assigned = [v for i, v in enumerate(rows) if i % partitions == part and
                ((v['mode'] in ('base', 'single')) if action == 'm13_146_single' else (v['mode'] in ('bank', 'insertion')))]
    byid = {t['edit_id']: t for t in tasks()}
    byorder = {t['order']: t for t in tasks()}
    with r.p.lease(gpu):
        runtime, bindings, handle = r.load(gpu)
        assert all(v.dtype == torch.float16 for v in runtime.model.parameters() if v.is_floating_point()), 'Original FP16 prefill parameters must survive chunk conversion exactly'
        if runtime_check is not None:
            runtime_check(runtime, bindings)
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
                            bank = bank_entries(entries,row)
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
                        owner=row['owner'], eval_owner=row['owner'], consumer_id=contract.consumer_id(row),
                        effective_expert=selected, original_router=True,
                        lock=c.digest(c.read(RUN/'private'/scope_file)),execution=c.read(RUN/'private'/source_file),
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
