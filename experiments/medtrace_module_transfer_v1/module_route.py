"""Original R0 routing, identical for all five new frozen expert banks."""
import os
import time
import traceback
import torch
import module_run as r
from m3bench_repro.editors.routing import MemoryRouter, decision_as_json

s, c, RUN = r.s, r.c, r.RUN


def main():
    assert c.read(RUN / 'private/CANDIDATES_FROZEN.json')['candidates'] == 48
    tasks = s.tasks()
    original = c.read(s.PARENT / 'private/BENCHMARK146_QUEUE.json')['tasks'][:8]
    rows = [(t, s.source(t)[0]) for t in tasks] + [(t, row) for t in tasks for row in s.events(t)]
    rows += [(tasks[0], row) for row in s.previous.data.held()]
    assert len(rows) == 219
    gpu = int(os.environ.get('GPU', r.GPUS[0]))
    with r.p.lease(gpu):
        runtime, bindings, handle = r.load(gpu)
        entries = []
        route_layer = runtime.target_lock['balancedit']['targets'][0]
        for before, after in zip(original, tasks):
            state = r.p.load_state(r.p.initial(before).parent.parent / 'ROUTER.pt')
            assert len(state['entries']) == 1
            entry = dict(state['entries'][0]); assert entry['logical_edit_id'] == before['edit_id']
            entry['logical_edit_id'] = after['edit_id']; entries.append(entry)
        router = MemoryRouter.from_state(dict(distance='euclidean', entries=entries), device=runtime.device)
        prepared = []
        try:
            for task, row in rows:
                batch = runtime.build_question_batch(s.record(task, row))
                with torch.inference_mode():
                    key = runtime.extract_layer_input_key(batch, module_path=route_layer, pooling='mean')
                assert key.numel() == router.keys[0].numel() == 4096
                decision = decision_as_json(router.route(key))
                prior = c.read(r.SCOPE / 'private/outputs/NATURAL_C/0' / (row['query_id'] + '.json'))['route']
                assert decision == prior, 'Original R0 route changed'
                prepared.append((task, s.prepare(runtime, bindings, task, [row])[0], decision))
            c.write(RUN / 'private/ROUTES.json', dict(rows=[dict(query_id=item[0]['query_id'], route=d) for _, item, d in prepared],
                original_routes_exact=219, router_changed=False))
            r.old.fp32(runtime)
            for arm in r.mm.ARMS[1:]:
                for task, item, route in prepared:
                    row = item[0]; dest = s.output('NATURAL_' + arm, 0, row)
                    selected = next((t for t in tasks if t['edit_id'] == route['logical_edit_id']), None)
                    reusable = s.output(arm, selected['order'], row) if selected else s.baseline(task, row)
                    if reusable.exists():
                        saved = c.read(reusable); current = item[4]
                        assert saved['binding']['judge_input'] == current
                        c.write(dest, dict(saved, arm='NATURAL_' + arm, route=route, alias_of=str(reusable),
                            forwards=0, baseline_path=str(s.baseline(task, row))))
                        continue
                    assert selected is not None
                    saved = r.p.load_state(s.point(selected, arm))
                    expert = r.p.expert(saved['states']['FINAL'], selected['seed'], runtime.device).requires_grad_(False)
                    hook = r.q.scoped.replay.MedTraceLayerHook(runtime.get_module(c.LAYER), expert); hook.attach()
                    try:
                        s.emit(runtime, hook, task, 'NATURAL_' + arm, item, r.COUNTS, dest, route)
                    finally:
                        hook.detach()
        finally:
            handle.remove(); r.save_counts()


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        c.write(RUN / 'private/FAILURE_module_route.json', dict(error=repr(error), traceback=traceback.format_exc(), epoch=time.time()))
        raise
