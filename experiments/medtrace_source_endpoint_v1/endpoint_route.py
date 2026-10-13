"""Evaluate the unchanged original R0 router against the two revised expert banks."""
import time
import traceback
import torch
from endpoint import s
from m3bench_repro.editors.routing import MemoryRouter,decision_as_json


def main():
    gpu=s.q.GPUS[0];s.old.calibration.require_memory(gpu)
    counts=dict(forwards=0,backwards=0,updates=0,generation_attempts=0)
    ts=s.tasks();original=s.c.read(s.PARENT/'private/BENCHMARK146_QUEUE.json')['tasks'][:8]
    rows=[(t,s.source(t)[0]) for t in ts]+[(t,r) for t in ts for r in s.events(t)]+[(ts[0],r) for r in s.previous.data.held()]
    assert len(rows)==219
    with s.p.lease(gpu):
        runtime,bindings=s.c.load(gpu);entries=[]
        route_layer=runtime.target_lock['balancedit']['targets'][0]
        for old,new in zip(original,ts):
            state=s.p.load_state(s.p.initial(old).parent.parent/'ROUTER.pt')
            assert len(state['entries'])==1
            entry=dict(state['entries'][0]);assert entry['logical_edit_id']==old['edit_id']
            entry['logical_edit_id']=new['edit_id'];entries.append(entry)
        router=MemoryRouter.from_state(dict(distance='euclidean',entries=entries),device=runtime.device)
        h=runtime.model.register_forward_pre_hook(lambda *_:counts.__setitem__('forwards',counts['forwards']+1))
        prepared=[]
        try:
            for t,row in rows:
                batch=runtime.build_question_batch(s.record(t,row))
                with torch.inference_mode():key=runtime.extract_layer_input_key(batch,module_path=route_layer,pooling='mean')
                assert key.numel()==router.keys[0].numel()==4096
                decision=decision_as_json(router.route(key));prepared.append((t,s.prepare(runtime,bindings,t,[row])[0],decision))
            s.c.write(s.RUN/'private/ROUTES.json',dict(rows=[dict(query_id=item[0]['query_id'],route=d) for _,item,d in prepared],
                owner8_key='ORIGINAL_V0_KEY_AND_RADIUS_RETAINED; ONLY_LOGICAL_ID_RENAMED',new_anchor_calibration=False))
            s.old.fp32(runtime)
            for arm in ('E',):
                for t,item,route in prepared:
                    row=item[0];dest=s.output('NATURAL_'+arm,0,row)
                    selected=next((x for x in ts if x['edit_id']==route['logical_edit_id']),None)
                    reusable=s.output(arm,selected['order'],row) if selected else s.baseline(t,row)
                    if reusable.exists():
                        saved=s.c.read(reusable);b=saved['binding']['judge_input'];current=item[4]
                        assert all(b[k]==current[k] for k in ('question','reference','image_sha256','prompt_ids','attention_mask','generation'))
                        assert b['runtime']==current['runtime']
                        s.c.write(dest,dict(saved,arm='NATURAL_'+arm,route=route,alias_of=str(reusable),forwards=0,baseline_path=str(s.baseline(t,row))))
                        continue
                    assert selected is not None
                    saved=torch.load(s.point(selected,arm),map_location='cpu',weights_only=True)
                    expert=s.p.expert(saved['states']['FINAL'],selected['seed'],runtime.device).requires_grad_(False)
                    moved=any(not torch.equal(saved['states']['FINAL'][k],saved['states']['BASE'][k]) for k in s.q.KEYS)
                    hook=s.q.scoped.replay.MedTraceLayerHook(runtime.get_module(s.c.LAYER),expert)
                    if moved:hook.attach()
                    try:s.emit(runtime,hook if moved else None,t,'NATURAL_'+arm,item,counts,dest,route)
                    finally:hook.detach()
        finally:h.remove();s.c.write(s.RUN/'private/COUNTS_scope_route_0.json',counts)


if __name__=='__main__':
    try:main()
    except BaseException as error:
        s.c.write(s.RUN/'private/FAILURE_scope_route.json',dict(error=repr(error),traceback=traceback.format_exc(),epoch=time.time()))
        raise
