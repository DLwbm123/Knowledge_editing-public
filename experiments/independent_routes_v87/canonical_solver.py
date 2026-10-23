"""One bounded old/corrected solver comparison on immutable V93 endpoints."""
import time
import torch
import route_common as c
import canonedit as old
import tt_native_canon as native
import convex_correction
v = c.v
if (v.ROOT/'ENGINEERING_CARRIED_COUNTS.json').exists():
    v.counts.update(v.read(v.ROOT/'ENGINEERING_CARRIED_COUNTS.json'))
ARMS = ['CE', 'BAL_CE', 'OLD', 'FIXED', 'FIXED_SOURCE']


def numpy(x):
    return x.detach().cpu().double().numpy()


def corrected(expert, new, previous, protected, owner):
    with torch.no_grad():
        before, baseline, _, _ = old.values(expert, new, previous, protected)
    gamma = torch.tensor([.25, .1, .1, .1, .1], device=before.device)
    M0 = native.gram(expert).detach()
    curve, accepted = [], False
    for iteration in range(5):
        ms, scores, A, C = old.values(expert, new, previous, protected, True)
        d = scores - baseline
        row = dict(iteration=iteration, z=expert.z.detach().tolist(), margin=ms.tolist(),
                   protected_MAE=float(d.abs().mean()), protected_MSE=float(d.square().mean()))
        if old.gates(ms, scores, baseline):
            curve.append(dict(row, status='ACCEPT')); accepted = True; break
        if iteration == 4:
            curve.append(dict(row, status='CORRECTION_LIMIT')); break
        M = native.gram(expert).detach()
        B = C.T @ C / 5 + .04 * M + 1e-10 * torch.eye(2, device=M.device)
        g = C.T @ d / 5
        solution = convex_correction.solve(*map(numpy, (A, gamma-ms, B, g, M, M0, expert.z, C, d)))
        row.update(solver=solution, A=A.tolist(), C=C.tolist(), d=d.tolist(), M=M.tolist(), M0=M0.tolist())
        curve.append(row)
        if solution['delta'] is None:
            break
        with torch.no_grad():
            expert.z.add_(torch.tensor(solution['delta'], device=expert.z.device, dtype=expert.z.dtype))
        v.counts['updates'] += 1
        v.save_state(v.ROOT/'proposals'/f'{owner}_{iteration}.pt',
                     dict(state={k:t.detach().clone() for k,t in expert.state().items()}, z=expert.z.detach().clone()))
    if not accepted:
        with torch.no_grad():expert.z.zero_()
    with torch.no_grad():
        ms, scores, _, _ = old.values(expert, new, previous, protected)
    return dict(status='ACCEPT' if accepted else 'NO_VALID_WRITE_NOOP',
                before=before.tolist(), after=ms.tolist(), z=expert.z.detach().tolist(), curve=curve,
                protected_MAE=float((scores-baseline).abs().mean()),
                protected_MSE=float((scores-baseline).square().mean()))


def verify_old(actual, expected):
    assert actual['status'] == expected['status']
    assert len(actual['curve']) == len(expected['curve'])
    for key in ('before', 'after', 'z'):
        torch.testing.assert_close(torch.tensor(actual[key]), torch.tensor(expected[key]), rtol=1e-5, atol=5e-4)
    for a, b in zip(actual['curve'], expected['curve']):
        assert a.get('status') == b.get('status')
        for key in ('z', 'margin', 'protected_MAE', 'protected_MSE', 'proposal'):
            if a.get(key) is None or b.get(key) is None:
                assert a.get(key) == b.get(key)
            else:
                torch.testing.assert_close(torch.tensor(a[key]), torch.tensor(b[key]), rtol=1e-5, atol=5e-4)


def evaluate_new(tasks):
    import canon_evaluate_recovery as ev
    for task in tasks:
        owner = task['order']
        plans = [p for p in v.read(v.ROOT/'EVAL_PLAN.json') if p['owner'] == owner]
        items = v.prepare(task, [p['query'] for p in plans])
        routes = v.read(v.ROOT/'ROUTE_BINDINGS_FROZEN.json')[str(owner)]
        for arm in ('FIXED', 'FIXED_SOURCE'):
            saved = v.r.p.load_state(v.ROOT/'candidates'/arm/f'{owner}.pt')
            expert = v.r.p.expert(saved['state'], task['seed'], v.runtime.device)
            for index, (plan, item, route) in enumerate(zip(plans, items, routes)):
                assert route['question'] == item[4]['question'] and route['query_id'] == plan['query']['query_id']
                assert route['image_sha256'] == item[4]['image_sha256']
                hook = None
                if route['effective_expert'] is not None:
                    hook = v.r.q.scoped.replay.MedTraceLayerHook(v.runtime.get_module(v.r.c.LAYER), expert); hook.attach()
                try:answer = c.generate(hook, item)
                finally:
                    if hook:hook.detach()
                tokens = list(answer.raw_token_ids)
                if ev.zero(expert) or route['effective_expert'] is None:
                    assert tokens == v.read(v.ROOT/'outputs/Base'/str(owner)/f'{index}.json')['raw_token_ids']
                if saved.get('fallback') == 'CE':
                    assert tokens == v.read(v.ROOT/'outputs/CE'/str(owner)/f'{index}.json')['raw_token_ids']
                v.write(v.ROOT/'outputs'/arm/str(owner)/f'{index}.json',
                        dict(owner=owner, arm=arm, query=plan['query'], binding=item[4], raw_answer=answer.decoded_text,
                             raw_token_ids=tokens, EOS=tokens[-1] == v.runtime.adapter.tokenizer.eos_token_id,
                             at_cap=len(tokens)>=1024, original_route_preserved=True))
                v.progress(stage='evaluate', owner=owner, arm=arm, index=index)


def main():
    recovery = v.read(v.ROOT/'ENGINEERING_RECOVERY2.json') if (v.ROOT/'ENGINEERING_RECOVERY2.json').exists() else {}
    tasks, items, by_owner, new_all = c.prepare()
    Chart = native.chart_class(); old_by, old_all = {}, []
    for task in tasks:
        owner = task['order']; answer = v.read(v.ROOT/'OLD_SOURCE_ANSWERS.json')[str(owner)]['raw_answer']
        rows = [dict(x, reference=answer) for x in v.read(v.ROOT/'SOURCE_ROWS.json')[str(owner)]]
        batches = [v.r.old.batch_on(v.runtime, x[1]) for x in v.prepare(task, rows, True)]
        old_by[owner] = batches; old_all += batches
    probe_state = v.r.p.load_state(v.ROOT/'candidates/CE/1.pt')['state']
    probe = v.r.p.expert(probe_state, 1, v.runtime.device); batch = by_owner[1][0]
    ph = v.r.q.scoped.replay.MedTraceLayerHook(v.runtime.get_module(v.r.c.LAYER), probe)
    ph.attach(); ph.set_teacher_routing(batch.labels)
    try:
        with torch.no_grad():full = v.runtime.model(**batch.forward_kwargs()).logits.detach().cpu()
    finally:ph.detach()
    with c.CachedSource(new_all+old_all) as cache:
        ph.attach(); ph.set_teacher_routing(batch.labels); cache.cache.current=id(batch)
        try:
            with torch.no_grad():cached = v.runtime.model(**batch.forward_kwargs()).logits.detach().cpu()
            torch.testing.assert_close(full, cached, rtol=1e-5, atol=5e-4)
        finally:cache.cache.current=None; ph.detach()
        v.write(v.ROOT/'CACHED_LOGITS_PARITY.json', dict(status='PASS', rtol=1e-5, atol=5e-4))
        del probe, full, cached
        for task in tasks:
            owner = task['order']; new = by_owner[owner]
            if owner in recovery.get('completed_owners', []):
                assert (v.ROOT/'training'/f'{owner}.json').exists()
                assert (v.ROOT/'candidates/FIXED'/f'{owner}.pt').exists()
                continue
            initial = v.r.p.load_state(v.PARENT/'private/candidates/C'/f'{owner}.pt')['states']['BASE']
            initial = {k:x.to(v.runtime.device) for k,x in v.r.mm.canonicalize(initial)[0].items()}
            endpoints = [{k:x.to(v.runtime.device) for k,x in v.r.p.load_state(v.ROOT/'candidates'/arm/f'{owner}.pt')['state'].items()}
                         for arm in ('CE', 'BAL_CE')]
            expert = Chart(initial, *endpoints)
            owners = [t['order'] for t in tasks]; pos = owners.index(owner)
            protected = [old_by[owners[(pos+j)%8]][0] for j in range(1,6)]
            with torch.no_grad():_, baseline, _, _ = old.values(expert,new,old_by[owner],protected)
            M0 = native.gram(expert).detach(); endpoint_audit = {}
            for arm, coordinate in [('CE',[1.,0.]), ('BAL_CE',[0.,1.])]:
                with torch.no_grad():
                    expert.z.copy_(torch.tensor(coordinate, device=expert.z.device))
                    ms, ss, _, _ = old.values(expert,new,old_by[owner],protected)
                    radius=float(expert.z@M0@expert.z)
                endpoint_audit[arm] = dict(margin=ms.tolist(), protected_MAE=float((ss-baseline).abs().mean()),
                    protected_MSE=float((ss-baseline).square().mean()), total_radius_squared=radius,
                    nonlinear_gate=old.gates(ms,ss,baseline), radius_pass=radius<=36+1e-6)
            if owner in recovery.get('completed_old_replay', []):
                expected_endpoint = v.read(v.ROOT/'endpoint_audit'/f'{owner}.json')
                assert endpoint_audit == expected_endpoint
            else:
                v.write(v.ROOT/'endpoint_audit'/f'{owner}.json', endpoint_audit)
            with torch.no_grad():expert.z.zero_()
            if owner in recovery.get('completed_old_replay', []):
                assert v.read(v.ROOT/'old_replay'/f'{owner}.json')['status'] == 'PASS'
            else:
                audit = native.solve(expert,new,old_by[owner],protected)
                verify_old(audit,v.read(v.ROOT/'old_training'/f'{owner}.json'))
                expected = v.r.p.load_state(v.ROOT/'candidates/OLD'/f'{owner}.pt')['state']
                for k,t in expert.state().items():torch.testing.assert_close(t.cpu(),expected[k].cpu(),rtol=1e-5,atol=5e-4)
                v.write(v.ROOT/'old_replay'/f'{owner}.json',dict(status='PASS',audit=audit))
            with torch.no_grad():expert.z.zero_()
            audit=corrected(expert,new,old_by[owner],protected,owner)
            state={k:t.detach().clone() for k,t in expert.state().items()}
            path=v.ROOT/'candidates/FIXED'/f'{owner}.pt';v.save_state(path,dict(state=state,owner=owner,arm='FIXED'))
            loaded=v.r.p.load_state(path)['state'];assert all(torch.equal(state[k].cpu(),loaded[k].cpu()) for k in state)
            v.write(v.ROOT/'training'/f'{owner}.json',audit);v.progress(stage='CORRECTED_SOLVER',owner=owner,status=audit['status'])
    c.restore()
    import canon_evaluate_recovery as ev
    for task in tasks:
        owner=task['order'];state=v.r.p.load_state(v.ROOT/'candidates/FIXED'/f'{owner}.pt')['state']
        expert=v.r.p.expert(state,task['seed'],v.runtime.device)
        batch=v.r.old.batch_on(v.runtime,items[owner][0][1])
        hook=v.r.q.scoped.replay.MedTraceLayerHook(v.runtime.get_module(v.r.c.LAYER),expert);hook.attach()
        try:
            mech=ev.mechanical(hook,batch,True) if ev.zero(expert) else v.r.s.fit.short.mechanical(v.runtime,hook,batch)
            v.write(v.ROOT/'mechanical'/f'FIXED_{owner}.json',mech)
        finally:hook.detach()
        for view,item in enumerate(items[owner]):
            hook=v.r.q.scoped.replay.MedTraceLayerHook(v.runtime.get_module(v.r.c.LAYER),expert);hook.attach()
            try:answer=c.generate(hook,item)
            finally:hook.detach()
            tokens=list(answer.raw_token_ids)
            v.write(v.ROOT/'qualification'/f'{owner}_{view}.json',dict(key=f'{owner}:{view}',source=item[0],raw_answer=answer.decoded_text,
                raw_token_ids=tokens,EOS=tokens[-1]==v.runtime.adapter.tokenizer.eos_token_id,at_cap=len(tokens)>=1024))
    # CE/BAL source answers are exact inherited consumers, not additional sampling.
    for task in tasks:
        owner=task['order']
        for arm in ('CE','BAL_CE'):
            outputs=[v.read(p) for p in (v.ROOT/'outputs'/arm/str(owner)).glob('*.json')]
            for view,source in enumerate(v.read(v.ROOT/'SOURCE_ROWS.json')[str(owner)]):
                matches=[x for x in outputs if all(x['query'][k]==source[k] for k in ('question','reference','image_sha256'))]
                assert len(matches)==1
                v.write(v.ROOT/'endpoint_qualification'/f'{arm}_{owner}_{view}.json',dict(key=f'{arm}:{owner}:{view}',source=source,
                    **{k:matches[0][k] for k in ('raw_answer','raw_token_ids','EOS','at_cap')}))
    v.write(v.ROOT/'SOURCE_REVIEW_READY.json',dict(fixed_inputs=40,inherited_endpoint_inputs=80,time=time.time()))
    while not (v.ROOT/'SOURCE_REVIEW_FROZEN.json').exists():v.budget();time.sleep(3)
    labels=v.read(v.ROOT/'SOURCE_REVIEW_FROZEN.json')['pass'];assert set(labels)=={f'{o}:{j}' for o in range(1,9) for j in range(5)}
    for task in tasks:
        owner=task['order'];passed=all(labels[f'{owner}:{j}'] for j in range(5));arm='FIXED' if passed else 'CE'
        state=v.r.p.load_state(v.ROOT/'candidates'/arm/f'{owner}.pt')['state']
        path=v.ROOT/'candidates/FIXED_SOURCE'/f'{owner}.pt'
        v.save_state(path,dict(state=state,source_gate_pass=passed,fallback=None if passed else 'CE'))
        loaded=v.r.p.load_state(path)['state'];assert all(torch.equal(state[k].cpu(),loaded[k].cpu()) for k in state)
    v.write(v.ROOT/'ALL_CANDIDATES_FROZEN.json',dict(count=40,inherited_outputs=620,new_formal_consumers=310,time=time.time()))
    evaluate_new(tasks)
    c.finish(ARMS)


if __name__=='__main__':c.run(main)
