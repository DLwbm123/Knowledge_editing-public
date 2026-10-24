"""Frozen source-scope comparison; shared initializers, budgets and R0 decisions."""
import gc
import signal
import time
from dataclasses import replace
from pathlib import Path
import route_common as c

v = c.v
ARMS = ('TT_CE', 'TT_BAL', 'LoRA_BAL')


def main():
    with (v.ROOT/'WORKER_ONCE').open('x'):
        pass
    v.allocations = dict(v.limits['counts'])
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('Original window')))
    signal.alarm(max(1, int(v.limits['hard_end']-time.time()-300)))
    v.setup()
    torch, r, runtime = v.torch, v.r, v.runtime
    from alignment import TT, snapshot
    from v62_structure import LoRA, lora_step
    from token_hook import TokenHook, layout
    from path_v18 import proposal, construct
    from v51_cache import PrefixCache
    from output_init import OutputProbe, output_core
    from m3bench_repro.editors.methods import BalanceEditPaperSpecEditor
    from m3bench_repro.editors.routing import MemoryRouter, balanced_radius, decision_as_json

    template = v.read(v.PARENT/'private/M13_SOURCE_PACKET.json')['original_tasks'][4]
    tasks = v.read(v.ROOT/'TASKS.json')
    assert len(tasks) == 8 and sum(len(t['evaluation']) for t in tasks) == 93
    module = runtime.get_module('model.layers.21.mlp.down_proj')
    eos = runtime.adapter.tokenizer.eos_token_id

    def prepare(rows, training=False):
        return v.prepare(template, rows, training)

    def fresh(arm):
        ex = (LoRA(225034559) if arm == 'LoRA_BAL' else TT(225034559)).to(runtime.device)
        if arm != 'LoRA_BAL':
            ex.load_state_dict(r.mm.canonicalize(snapshot(ex))[0])
        return ex

    def value(ex, h, b, lay):
        h.set_layout(lay)
        h.teacher(b.labels, b.forward_kwargs().get('attention_mask'))
        loss = runtime.compute_loss(b)
        assert torch.isfinite(loss)
        gs = torch.autograd.grad(loss, tuple(ex.parameters()))
        v.counts['backwards'] += 1
        gradient = torch.cat([g.detach().cpu().double().flatten() for g in gs])
        assert torch.isfinite(gradient).all()
        return float(loss.detach()), gradient

    def emit(h, item, arm, group):
        dest = v.ROOT/'outputs'/arm/group/(item[0]['query_id']+'.json')
        assert not dest.exists(), 'No output resampling'
        if h:
            h.set_layout(layout(item[2], item[3]))
        assert v.counts['evaluation_generations'] < v.allocations['evaluation_generations']
        answer = c.generate(h, item)
        tokens = list(answer.raw_token_ids)
        v.write(dest, dict(query=item[0], binding=item[4], raw_answer=answer.decoded_text,
            raw_token_ids=tokens, EOS=tokens[-1] == eos, at_cap=len(tokens)>=1024,
            forced_writer=arm != 'Base'))

    # Original R0 anchors and keys are computed with Base before training.
    routes = {}
    editor = BalanceEditPaperSpecEditor(runtime)
    try:
        for task in tasks:
            group = task['group']
            source = task['training'][0]
            record = replace(r.c.record(template), record_id=group,
                question=source['question'], target=source['reference'],
                image_path=Path(r.c.local_path(source['image_path'])),
                official_rephrase='Please answer this question: '+source['question'],
                router_positive_source='fixed native-only prefix')
            folder = v.ROOT/'anchors'/group
            folder.mkdir(parents=True, exist_ok=True)
            key, positive, negative, _ = editor._anchors(record, folder)
            radius = balanced_radius(key, positive, negative, alpha=.2, distance='euclidean')
            router = MemoryRouter('euclidean')
            router.add(group, key, radius)
            routes[group] = {row['query_id']: decision_as_json(router.route(
                editor._question_key(r.s.record(template, row)))) for row in task['evaluation']}
            assert routes[group][source['query_id']]['logical_edit_id'] == group
            v.progress(stage='freeze_R0', group=group)
    finally:
        runtime.replace_module(editor.target, editor.wrapper.base)
    del editor
    v.write(v.ROOT/'ROUTES_FROZEN.json', dict(single_edit=True, alpha=.2, routes=routes, time=time.time()))

    for task in tasks:
        group = task['group']
        items = prepare(task['training'], True)
        batches = [r.old.batch_on(runtime, item[1]) for item in items]
        layouts = [layout(item[2], item[3]) for item in items]
        targets = [b.labels[0][b.labels[0] != -100].detach().cpu() for b in batches]
        assert len(batches) == 5 and all(torch.equal(targets[0], t) for t in targets)
        assert targets[0][-1] == eos and int((targets[0] == eos).sum()) == 1
        assert all(lay['prefix_length'] == int(torch.where(b.labels[0] != -100)[0][0])
                   for b, lay in zip(batches, layouts))
        with torch.no_grad():
            base_losses = [float(runtime.compute_loss(b)) for b in batches]
        ex = fresh('TT_CE')
        probe = OutputProbe(ex.factors()[1]).to(runtime.device)
        hook = TokenHook(module, probe, 'ALL_TOKENS')
        hook.attach()
        try:
            gradients = [value(probe, hook, b, lay) for b, lay in zip(batches, layouts)]
        finally:
            hook.detach()
        assert [x[0] for x in gradients] == base_losses, 'Probe zero parity'
        g = .5*gradients[0][1] + .125*sum(x[1] for x in gradients[1:])
        state = snapshot(ex)
        core, info = output_core(g.reshape(4096, 4), state['G2'])
        state['G2'] = core.to(state['G2'])
        initial = r.mm.canonicalize(state)[0]
        assert initial['G1'].count_nonzero() == 0
        v.write(v.ROOT/'initialization'/(group+'.json'), dict(info, source_losses=base_losses,
            zero_probe_base_exact=True, extra_backwards=5, time=time.time()))
        del probe, hook, gradients, ex

        # A single reusable cache is exact at zero and at a nonzero writer.
        for family in ('TT_CE', 'LoRA_BAL'):
            ex = fresh(family)
            if family == 'TT_CE':
                ex.load_state_dict(initial)
            saved = snapshot(ex)
            hook = TokenHook(module, ex, 'ALL_TOKENS')
            hook.attach()
            try:
                for nonzero in (False, True):
                    ex.load_state_dict(saved)
                    if nonzero:
                        with torch.no_grad():
                            (ex.B if family == 'LoRA_BAL' else ex.G1).fill_(.001)
                    refs = [value(ex, hook, b, lay) for b, lay in zip(batches, layouts)]
                    if not nonzero:
                        assert [x[0] for x in refs] == base_losses, 'Zero Base parity'
                    with PrefixCache(runtime, batches, 21) as cache:
                        for b, lay, ref in zip(batches, layouts, refs):
                            for _ in range(2):
                                got = value(ex, hook, b, lay)
                                assert got[0] == ref[0] and torch.equal(got[1], ref[1]), 'Cache parity'
                    assert cache.misses == cache.hits == 5
            finally:
                hook.detach()
            v.write(v.ROOT/'admission'/group/(family+'.json'), dict(status='PASS',
                zero_and_nonzero_cache_exact=True, all_five_bindings=True, time=time.time()))
            del ex, hook

        with PrefixCache(runtime, batches, 21) as cache:
            for arm in ARMS:
                ex = fresh(arm)
                if arm != 'LoRA_BAL':
                    ex.load_state_dict(initial)
                hook = TokenHook(module, ex, 'ALL_TOKENS')
                hook.attach()
                scalar = dict(step=0, second_moments=[0., 0.])
                curve = []
                began = time.time()
                try:
                    for step in range(160):
                        first = value(ex, hook, batches[0], layouts[0])
                        j = 1+step%4
                        second = value(ex, hook, batches[j], layouts[j])
                        weight = .5 if arm == 'TT_CE' else .2
                        gradient = weight*first[1]+(1-weight)*second[1]
                        before = snapshot(ex)
                        if arm == 'LoRA_BAL':
                            state, scalar = lora_step(before, gradient, scalar)
                        else:
                            raw, scalar, delta = proposal(r, before, gradient, scalar, list(before))
                            state = construct(r, before, raw, delta, delta, 1., list(before))
                        ex.load_state_dict(state)
                        v.counts['updates'] += 1
                        curve.append(dict(step=step+1, native=first[0], fit=second[0],
                                          objective=weight*first[0]+(1-weight)*second[0]))
                        if (step+1)%20 == 0:
                            v.progress(stage='train', group=group, arm=arm, step=step+1)
                finally:
                    hook.detach()
                dest = v.ROOT/'states'/group/(arm+'.pt')
                dest.parent.mkdir(parents=True, exist_ok=True)
                torch.save({k:x.cpu() for k,x in snapshot(ex).items()}, dest)
                loaded = torch.load(dest, map_location=runtime.device, weights_only=True)
                assert all(torch.equal(x, loaded[k]) for k,x in snapshot(ex).items())
                v.write(v.ROOT/'training'/group/(arm+'.json'), dict(curve=curve, save_load_exact=True,
                    seconds=time.time()-began, counts=dict(v.counts)))
                del ex, hook, loaded
        assert cache.misses == 5 and cache.hits == 955
        del batches, items, initial
        gc.collect()
        torch.cuda.empty_cache()

    assert v.counts['updates'] == 3840 and v.counts['backwards'] == 8200
    v.write(v.ROOT/'ALL_CANDIDATES_FROZEN.json', dict(states=24, counts=dict(v.counts), time=time.time()))
    for task in tasks:
        group = task['group']
        items = prepare(task['evaluation'])
        for item in items:
            if item[0]['evaluation_role'] in ('FIT', 'T2G'):
                emit(None, item, 'Base', group)
            else:
                previous = v.read(v.ROOT.parent/'outputs'/(str(item[0]['qid'])+'.json'))
                assert previous['query']['question'] == item[0]['question']
                assert previous['binding']['image_sha256'] == item[4]['image_sha256']
                v.write(v.ROOT/'outputs'/'Base'/group/(item[0]['query_id']+'.json'),
                    dict(previous, query=item[0], reused_phase_a=True))
        for arm in ARMS:
            ex = fresh(arm)
            dest = v.ROOT/'states'/group/(arm+'.pt')
            ex.load_state_dict(torch.load(dest, map_location=runtime.device, weights_only=True))
            hook = TokenHook(module, ex, 'ALL_TOKENS')
            hook.attach()
            try:
                for item in items:
                    emit(hook, item, arm, group)
            finally:
                hook.detach()
            size = dest.stat().st_size
            dest.unlink()
            v.write(v.ROOT/'lifecycle'/group/(arm+'.json'), dict(last_consumer='all_frozen_generations',
                retired_bytes=size, time=time.time()))
            del ex, hook
            v.progress(stage='evaluate', group=group, arm=arm)
        for row in task['evaluation']:
            qid = row['query_id']
            route = routes[group][qid]
            selected = route['logical_edit_id']
            assert selected in (None, group)
            base = v.read(v.ROOT/'outputs'/'Base'/group/(qid+'.json'))
            v.write(v.ROOT/'retrieval'/group/(qid+'.json'), dict(query=row, route=route,
                raw_answer=task['training'][0]['reference'] if selected else base['raw_answer'],
                Base_fallback=selected is None, model_calls=0, EOS=selected is not None or base['EOS'],
                at_cap=False if selected else base['at_cap']))
        del items
        gc.collect()
    assert v.counts['evaluation_generations'] == 343
    assert len(list((v.ROOT/'outputs').glob('*/*/*.json'))) == 372
    v.write(v.ROOT/'MAIN_COMPLETE.json', dict(status='MODEL_CONSUMERS_COMPLETE_SCORING_PENDING',
        outputs=372, retrieval=93, counts=dict(v.counts), time=time.time()))


if __name__ == '__main__':
    c.run(main)
