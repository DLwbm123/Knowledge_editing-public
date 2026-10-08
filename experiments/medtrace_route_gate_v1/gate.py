"""Frozen TT experts: separate admission, candidate ranking and write response."""
import json
import math
import os
import time
import traceback
from pathlib import Path
import torch
import paper as p
import structure as s
from methods.medtrace.core import MedTraceLayerHook
from m3bench_repro.editors.routing import MemoryRouter, decision_as_json, euclidean_distances

c,RUN=p.c,p.RUN
PREVIOUS=Path(os.environ['GATE_PREVIOUS'])
ARMS={'G':('G_R0','G_OLD_MODAL','G_MODAL_R0_GATE','G_MODAL_R0_CANDIDATES'),
      'P':('P_MEAN_SUBSPACE_R0_CANDIDATES','P_LAST_SUBSPACE_R0_CANDIDATES','P_LAST_RESIDUAL_R0_CANDIDATES')}

def choose(on,nearest,candidates,scores,restricted=True):
    if not on:return []
    assert nearest in candidates, 'Original accepted expert must remain a candidate'
    allowed=candidates if restricted else range(len(scores))
    assert all(math.isfinite(float(x)) for x in scores)
    return [max(allowed,key=lambda i:(float(scores[i]),-i))]

def functional_score(expert,h,y):
    return expert.residual(h).norm()/y.norm().clamp_min(1e-12)

def cpucheck():
    assert choose(False,0,[],[1.,2.])==[]
    assert choose(True,0,[0],[1.,2.])==[0]
    assert choose(True,0,[0],[1.,2.],False)==[1]
    assert choose(True,0,[0,1],[2.,2.])==[0]
    e=p.tr.TT4(3,8,8)
    with torch.no_grad():e.G1.normal_(0,.01)
    h=torch.randn(14336);y=torch.randn(4096);b,a=e.factors()
    expected=((h/(h.square().mean().sqrt()+e.epsilon))@a.T)@b.T
    assert torch.allclose(functional_score(e,h,y),expected.norm()/y.norm())
    assert torch.isfinite(s.projection(a,h))
    return dict(status='PASS',checks=['OFF remains OFF','candidate restriction','unrestricted gate-only ranking','stable tie order','factorized functional response parity','finite last-position projection'])

def read_origin(index,qid,label):return c.read(index[qid][label])

def prepare():
    import sqlite3
    ts=p.tasks();assert len(ts)==146
    expected={t['edit_id']:dict(path=str(p.initial(t)),hash=p.load_state(p.initial(t))['state_hash']) for t in ts}
    db=sqlite3.connect('file:'+str(PREVIOUS/'private/judge_paper_astra_medium/queue.sqlite')+'?mode=ro',uri=True)
    index={}
    for arm,qid,path in db.execute("SELECT method,query_id,path FROM consumer WHERE method LIKE 'A_%' AND prefix=146"):
        index.setdefault(qid,{})[arm]=path
    assert len(index)==1513 and all(len(v)==5 for v in index.values())
    for label in ('A_W0_R0','A_W0_MODAL','A_W0_NORM_MEAN','A_W0_INTRINSIC_ALL','A_W0_INTRINSIC_MODAL'):
        sample=read_origin(index,next(iter(index)),label)
        assert sample['binding']['phase']['weights']==expected
    c.write(RUN/'private/ORIGIN_INDEX.json',index)
    c.write(RUN/'private/REFERENCE_WEIGHTS.json',expected)
    rows={str(p.feature_path(row)):row for row in p.queries(146).values()}
    c.write(RUN/'private/FEATURE_ROWS.json',list(rows.values()))
    result=cpucheck();result.update(feature_inputs=len(rows),consumers=10591,new_training=0,arms=ARMS,CP_enabled=False)
    c.write(RUN/'public/ADMISSION.json',result)

def router_setup(device):
    entries=[]
    for t in p.tasks():entries+=p.load_state(p.initial(t).parent.parent/'ROUTER.pt')['entries']
    router=MemoryRouter.from_state(dict(distance='euclidean',entries=entries),device=device)
    assert router.logical_ids==[t['edit_id'] for t in p.tasks()]
    return router

def check_original(router,key,old):
    decision=decision_as_json(router.route(key))
    assert decision['activated']==old['activated'] and decision['logical_edit_id']==old['logical_edit_id']
    assert decision['nearest_logical_edit_id']==old['nearest_logical_edit_id']
    assert math.isclose(decision['nearest_distance'],old['nearest_distance'],rel_tol=1e-5,abs_tol=1e-5)
    return decision

def cache():
    gpu,part=int(os.environ['GPU']),int(os.environ['PARTITION'])
    index=c.read(RUN/'private/ORIGIN_INDEX.json');rows=c.read(RUN/'private/FEATURE_ROWS.json')
    with p.lease(gpu):
        runtime,_=c.load(gpu);router=router_setup(runtime.device)
        route_layer=runtime.target_lock['balancedit']['targets'][0]
        for i,row in enumerate(rows[part::3]):
            dest=p.feature_path(row)
            if dest.exists():continue
            raw=runtime.adapter.prepare_inputs(Path(c.local_path(row['image_path'])),row['question'],None)
            assert raw['image_sha256']==row['image_sha256']
            with torch.inference_mode():
                embeds,att,pos,_=runtime._expand_multimodal(raw_input_ids=raw['input_ids'],attention_mask=raw['attention_mask'],labels=torch.full_like(raw['input_ids'],-100),images=raw['images'])
                valid=torch.ones(embeds.shape[1],dtype=torch.bool,device=embeds.device) if att is None else att[0].bool()
                last=int(valid.nonzero()[-1]);assert last==embeds.shape[1]-1
                rk=[];write=[]
                h1=runtime.get_module(route_layer).register_forward_pre_hook(lambda _,args:rk.append(args[0][0][valid].float().mean(0)))
                h2=runtime.get_module(c.LAYER).register_forward_hook(lambda _,args,out:write.append((args[0][0,last].float().clone(),out[0,last].float().clone())))
                try:runtime.model(inputs_embeds=embeds,attention_mask=att,position_ids=pos,labels=None,use_cache=False,return_dict=True)
                finally:h1.remove();h2.remove()
            assert len(rk)==len(write)==1 and rk[0].shape==(4096,)
            h,y=write[0];assert h.shape==(14336,) and y.shape==(4096,)
            original=read_origin(index,row['query_id'],'A_W0_R0')
            check_original(router,rk[0],original['route'])
            c.save(dest,dict(route_vector=rk[0].cpu(),last_input=h.cpu(),last_output=y.cpu(),binding=dict(query_input=[row['image_sha256'],row['question']],prompt_ids=raw['input_ids'][0].tolist(),last_position=last,all_edits_off=True,answer_used=False,route_layer=route_layer,write_layer=c.LAYER)))
            if i==0:p.done('CACHE_NATIVE_TEST_'+str(part),dict(original_route_equal=True,answer_used=False,Base_gradient=any(x.grad is not None for x in runtime.model.parameters())))
            if i%30==0:print('FEATURE',part,i,flush=True)
    p.done('FEATURE_'+str(part))

def evaluate():
    gpu,part=int(os.environ['GPU']),int(os.environ['PARTITION']);stage=os.environ['GATE_STAGE']
    ts=p.tasks();index=c.read(RUN/'private/ORIGIN_INDEX.json');rows=p.queries(146)
    weights=c.read(RUN/'private/REFERENCE_WEIGHTS.json');lock=c.digest(c.read(RUN/'private/GATE_LOCK.json'))
    active={(t['native']['image_sha256'],t['native']['question']) for t in ts}
    with p.lease(gpu):
        runtime,bindings=c.load(gpu);router=router_setup(runtime.device)
        es=[p.expert(p.load_state(p.initial(t))['expert'],t['seed'],runtime.device).requires_grad_(False) for t in ts]
        aa=[e.factors()[1].detach() for e in es]
        mixture=p.Mixture(es);hook=MedTraceLayerHook(runtime.get_module(c.LAYER),mixture);hook.attach()
        teachers=p.tr.teachers_for(runtime,c.read(p.BASE/'private/U_ROLES.json')['CHECK']);tm={'U_'+c.digest(x[4]):x for x in teachers}
        try:
            for qid,row in list(rows.items())[part::6]:
                c.budget();cached=p.load_state(p.feature_path(row));assert cached['binding']['query_input']==[row['image_sha256'],row['question']]
                origins={label:c.read(path) for label,path in index[qid].items()}
                for d in origins.values():assert d['binding']['phase']['weights']==weights
                original=origins['A_W0_R0'];key=cached['route_vector'].to(runtime.device)
                decision=check_original(router,key,original['route']);on=decision['activated']
                dist=euclidean_distances(router._key_matrix(),key)
                candidates=[i for i,(v,radius) in enumerate(zip(dist,router.radii)) if float(v)<=radius]
                nearest=router.logical_ids.index(decision['nearest_logical_edit_id'])
                if on:assert nearest in candidates
                h,y=cached['last_input'].to(runtime.device),cached['last_output'].to(runtime.device)
                available={d['effective_expert']:d for d in origins.values()}
                if stage=='P':
                    for label in ARMS['G']:
                        d=c.read(RUN/'private/outputs'/label/(c.digest(qid)+'.json'));available[d['effective_expert']]=d
                for label in ARMS[stage]:
                    phase=dict(mode='bank',prefix=146,arm=label,node=0,slot=0,weights=weights,execution=c.read(RUN/'private/GPU_SOURCE_VERSION.json'),gate_lock=lock)
                    dest=RUN/'private/outputs'/label/(c.digest(qid)+'.json')
                    if dest.exists():assert c.read(dest)['binding']['phase']==phase;continue
                    score=None
                    if label in ('G_R0','G_OLD_MODAL'):
                        old=origins['A_W0_R0' if label=='G_R0' else 'A_W0_MODAL']
                        ids=[router.logical_ids.index(old['effective_expert'])] if old['effective_expert'] else []
                    else:
                        if label=='G_MODAL_R0_GATE':
                            old=origins['A_W0_MODAL'];threshold=old['binding']['phase']['thresholds']
                            score=[(v-t)/max(1-t,1e-6) for v,t in zip(old['route']['scores'],threshold)]
                        elif label=='G_MODAL_R0_CANDIDATES':score=origins['A_W0_MODAL']['route']['scores']
                        elif label=='P_MEAN_SUBSPACE_R0_CANDIDATES':score=origins['A_W0_INTRINSIC_MODAL']['route']['scores']
                        elif label=='P_LAST_SUBSPACE_R0_CANDIDATES':
                            with torch.inference_mode():score=[float(s.projection(a,h)) for a in aa]
                        elif label=='P_LAST_RESIDUAL_R0_CANDIDATES':
                            with torch.inference_mode():score=[float(functional_score(e,h,y)) for e in es]
                        else:raise ValueError(label)
                        ids=choose(on,nearest,candidates,score,label!='G_MODAL_R0_GATE')
                        assert bool(ids)==on
                    selected=router.logical_ids[ids[0]] if ids else None
                    mixture.ids=ids;mixture.weights=[1.] if ids else [];hook.clear_request_routing()
                    reused=old if label in ('G_R0','G_OLD_MODAL') else available.get(selected)
                    seconds=0.;kl=same=None
                    b=original['binding']['judge_input']
                    if reused:
                        assert reused['binding']['judge_input']==b
                        out=reused['R0'];kl=reused['U_KL'];same=reused['Base_token_consistency']
                    else:
                        raw=runtime.adapter.prepare_inputs(Path(c.local_path(row['image_path'])),row['question'],None)
                        assert raw['image_sha256']==b['image_sha256'] and raw['input_ids'].tolist()==[b['prompt_ids']] and raw['attention_mask'].tolist()==[b['attention_mask']]
                        assert runtime.generation_config==b['generation']
                        began=time.time()
                        with torch.inference_mode():
                            if ids:
                                with hook.generation_request():g=runtime.adapter.generate_prepared_with_result(raw,runtime.generation_config)
                            else:g=runtime.adapter.generate_prepared_with_result(raw,runtime.generation_config)
                        out=dict(raw_answer=g.decoded_text,raw_token_ids=list(g.raw_token_ids));seconds=time.time()-began
                        if qid in tm:
                            kwargs,labels,mask,logp,_,teacher=tm[qid];hook.clear_request_routing()
                            if ids:hook.set_teacher_routing(labels)
                            with torch.no_grad():kl=float(p.tr.full_vocab_kl(runtime.model(**kwargs).logits[mask],logp))
                            same=out['raw_token_ids']==teacher['tokens'];hook.clear_request_routing()
                    route=dict(activated=bool(ids),logical_edit_id=selected,original_R0=decision,candidate_count=len(candidates),scores=score,rule=label)
                    bind=dict(input=row,judge_input=b,phase=phase,arm=label,mode='bank_R0',prefix=146,owner_order=ts[-1]['order'],panel=row.get('role','PANEL'))
                    result=dict(binding=bind,R0=out,route=route,effective_expert=selected,selected_experts=[selected] if selected else [],mix_weights=[1.] if selected else [],research_lock=lock,seconds=seconds,U_KL=kl,Base_token_consistency=same,active_target=(row['image_sha256'],row['question']) in active,reused_output_binding=c.digest(reused) if reused else None,diagnostic_only=False)
                    c.write(dest,result);available[selected]=result
                print('QUERY',stage,part,qid[:8],flush=True)
        finally:hook.detach()
    p.done('EVAL_'+stage+'_'+str(part))

def controller():
    import pipeline
    p.progress('QUERY_FEATURES')
    pipeline.wait([pipeline.launch('gate.py','gate_cache',5+i,i) for i in range(3)])
    for stage in ARMS:
        os.environ['GATE_STAGE']=stage;p.progress(stage+'_EVALUATION')
        pipeline.wait([pipeline.launch('gate.py','gate_eval',5+i%3,i) for i in range(6)])
        pipeline.wait([pipeline.launch('gate_queue.py','gate_ingest')])
    p.done('GENERATION_COMPLETE');pipeline.wait([pipeline.launch('gate_queue.py','gate_ingest')])
    removed=[]
    for name in ('features','teacher'):
        for path in (RUN/'private'/name).rglob('*.pt'):
            assert not path.is_symlink() and path.resolve().is_relative_to((RUN/'private'/name).resolve())
            removed.append(dict(path=str(path),bytes=path.stat().st_size));path.unlink()
    p.done('DELETION',dict(files=removed,all_consumers_complete=True,bindings_durable=True,historical_weights_untouched=True))
    p.progress('ASTRA_SCORING')
    root=RUN/'private/judge_gate_astra_medium'
    while not (root/'ALL_WORKERS_COMPLETE.json').exists():
        c.budget();assert not list((root/'workers').glob('*/SCORER_FAILURE.json'));time.sleep(30)
    pipeline.wait([pipeline.launch('gate_report.py','gate_report')])

if __name__=='__main__':
    try:{'gate_prepare':prepare,'gate_cache':cache,'gate_eval':evaluate,'gate_controller':controller}[os.environ['ACTION']]()
    except BaseException as error:
        c.write(RUN/'private'/('FAILURE_'+os.environ['ACTION']+'_'+os.environ.get('PARTITION','all')+'.json'),dict(error=repr(error),traceback=traceback.format_exc(),epoch=time.time(),retry=False));raise
