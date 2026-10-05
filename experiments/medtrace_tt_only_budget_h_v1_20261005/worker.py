"""Fixed TT-only seeds, budget allocations and bounded H updates."""
from contextlib import contextmanager
from collections import defaultdict
from dataclasses import replace
import copy
import fcntl
import gc
import json
import os
from pathlib import Path
import random
import signal
import shutil
import sys
import time
import traceback

RUN=Path(os.environ['RUN_ROOT']);sys.path.insert(0,str(RUN/'private/tools'))
import legacy_worker as legacy
from audit import read,write,sha,digest,OLD

from protocol import ARMS,arms_for,decode
from structures import RANKS,COUNTS
import updates
legacy.GPUS={int(k):v for k,v in read(RUN/'PLAN_CONFIG.json')['hardware']['UUIDs'].items()}
PARENT=Path(read(RUN/'PLAN_CONFIG.json')['parent_run'])
H=read(RUN/'private/H_AVAILABLE.json')
LAYER=legacy.LAYER


def budget(*_):
    legacy_original_budget()
    files=[p for p in (RUN/'private').rglob('*.pt') if not p.is_symlink()]
    used=0
    for p in files:
        try:used+=p.stat().st_size
        except FileNotFoundError:
            # Another worker may finish and delete its owned temporary checkpoint.
            # Only ENOENT is benign; permission/I/O errors must still stop the run.
            continue
    with legacy.locked_ledger() as ledger:
        ledger['weights_observed_peak_bytes']=max(ledger.get('weights_observed_peak_bytes',0),used)
    if used>=2*1024**3:raise OSError('Frozen2GiB generated-weight cap reached')


legacy_original_budget=legacy.budget
legacy.budget=budget


def clone(init,seed,device):
    from structures import TT4
    assert set(init)=={'G1','G2','G3','G4'}
    e=TT4(seed,init['G1'].shape[2],init['G4'].shape[0]).to(device);e.load_state_dict(init);return e


def grad_vector(expert):
    import torch
    return torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).detach().flatten().clone() for p in expert.parameters()])


def logged_update(runtime,hook,expert,optimizer,native,fit,teacher,extra,weight,diagnose):
    import torch
    from scripts.medtrace import stage18_cfact as cf
    vectors=[];original=cf.backward_term
    def observed(e,loss,w,tokens,sample):
        before=grad_vector(e) if diagnose else None
        item=original(e,loss,w,tokens,sample)
        if diagnose:vectors.append(grad_vector(e)-before)
        return item
    cf.backward_term=observed
    try:item=cf.update(runtime,hook,expert,optimizer,native,fit,teacher,extra,extra_weight=weight)
    finally:cf.backward_term=original
    total=item['gradient_norm'];scale=min(1.,1./(total+1e-6))
    item.update(preclip_total_norm=total,clip_applied=total>1,clip_scale=scale,
        postclip_norm=float(grad_vector(expert).norm()),diagnostic=None)
    if diagnose:
        common=vectors[0]+vectors[1]+vectors[2];extra_v=vectors[3] if len(vectors)==4 else None
        cosine=lambda a,b:float(torch.nn.functional.cosine_similarity(a[None].float(),b[None].float()).item()) if a.norm()>0 and b.norm()>0 else None
        item['diagnostic']=dict(weighted_term_norms=[float(v.norm()) for v in vectors],weighted_extra_to_common_norm_ratio=float(extra_v.norm()/common.norm()) if extra_v is not None and common.norm()>0 else None,
            extra_native_fit_cosine=cosine(extra_v,vectors[0]+vectors[1]) if extra_v is not None else None,
            extra_U_cosine=cosine(extra_v,vectors[2]) if extra_v is not None else None,
            extra_common_cosine=cosine(extra_v,common) if extra_v is not None else None,
            extraction='actual successive backward gradient increments; no additional forward/RNG draw')
    return item


def reset_rng(state):
    import torch
    torch.set_rng_state(state['torch_rng'].cpu());random.setstate(state['python_rng'])
    if state['cuda_rng'] is not None:torch.cuda.set_rng_state(state['cuda_rng'].cpu())


def continuation(runtime,t,init,record,cfg,arm,mechanical=False):
    import torch
    from methods.medtrace import MedTraceLayerHook
    from structures import optimizer_for
    from scripts.medtrace import stage18_cfact as cf
    from scripts.medtrace.run_selective_write import save
    from m3bench_repro.editors.llava_runtime import seed_everything
    structure,condition,slot=decode(arm);hasH=condition!='NO_H';scale=.25 if condition=='SMALL_LR_H' else 1.
    directory=RUN/'private/edits'/f"e{t['order']:03d}"/arm;directory.mkdir(parents=True,exist_ok=True)
    final=directory/'final.pt';point=directory/'latest.pt';rows=H[t['edit_id']]
    task=dict(t,U_fit=[legacy.local_row(u) for u in t['U_fit']],H_fit=rows)
    with (RUN/'private/teacher/WRITE.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);teachers=cf.teachers_for(runtime,RUN,cfg,task,record)
    batches=[runtime.build_edit_batch(record)]+[runtime.build_edit_batch(replace(record,question=q)) for q in t['fit_questions']]
    hb=[cf.source_batch(runtime,record,h) for h in rows]
    eos=runtime.adapter.tokenizer.eos_token_id
    assert len(batches)==5 and all(eos in x.target_token_ids for x in batches+hb) and all(len(x.target_token_ids)<=128 for x in hb)
    fit=list(range(1,5));random.Random(t['seed']).shuffle(fit);order=cf.extra_schedule(task)
    e=clone(init,t['seed'],runtime.device).requires_grad_(True);w0=cf.state_hash(e)
    binding=dict(input=t['native'],fit=t['fit_questions'],U=t['U_fit'],H=rows if hasH else [],diagnostic_H=rows,arm=arm,seed=t['seed'],seed_slot=slot,W_init=w0,steps=320,fit_order=fit,H_order=order,lr_scale=scale,structure=structure,condition=condition,source=cfg['code_commit'],execution_source=cfg['execution_source'],runtime=cfg['runtime_lock'],generation=runtime.generation_config)
    if final.exists():
        state=torch.load(final,map_location='cpu',weights_only=True);assert state['step']==320 and state['binding']==binding and state['state_hash']==read(directory/'TRAINING.json')['final_state_hash'];return final
    seed_everything(t['seed']);opt=optimizer_for(e,runtime.model,scale);assert not opt.state
    hook=MedTraceLayerHook(runtime.get_module(LAYER),e);hook.attach();began=time.time();torch.cuda.reset_peak_memory_stats()
    curve=[];start=0
    try:
        baseline=updates.protection(runtime,hook,batches,teachers);thresholds=updates.limits(baseline)
        activations=updates.capture_training_activations(runtime,hook,batches,teachers,LAYER)
        d0,f0=updates.diagnostic(runtime,hook,e,[batches[0],batches[fit[0]]],teachers,hb[order[0]],1. if hasH else 0.,activations)
        write(directory/'W0_DIAGNOSTIC.json',dict(diagnostic=d0,protection=baseline,thresholds=thresholds,training_activation_shape=list(activations.shape),temporary_AB_parameters=0))
        if point.exists():
            start,curve=cf.resume(point,binding,e,opt);loaded=torch.load(point,map_location='cpu',weights_only=True);updates.restore_hook(hook,loaded['hook_state'])
        else:save(point,dict(binding=binding,expert=e.state_dict(),optimizer=opt.state_dict(),step=0,curve=[],hook_state=updates.hook_state(hook),**cf.rng_state()))
        cumulative_parameter=curve[-1]['cumulative_parameter_displacement'] if curve else 0.
        cumulative_function=curve[-1]['cumulative_function_displacement'] if curve else 0.
        for step in range(start+1,321):
            budget();fi=fit[(step-1)%4];ei=order[step-1];extra=(hb[ei],rows[ei]) if hasH else None
            before={k:p.detach().clone() for k,p in e.named_parameters()}
            with torch.no_grad():fb=e.residual(activations).detach().clone()
            diagnose=step in (1,20,80,160,320)
            diag,_=updates.diagnostic(runtime,hook,e,[batches[0],batches[fi]],[teachers[(step-1)%len(teachers)]],hb[ei],1. if hasH else 0.,activations) if diagnose else (None,None)
            def candidate():return logged_update(runtime,hook,e,opt,batches[0],batches[fi],teachers[(step-1)%len(teachers)],extra,1.,diagnose)
            item=updates.guarded_candidate(e,opt,candidate,lambda:updates.protection(runtime,hook,batches,teachers),thresholds) if condition=='GUARDED_H' else candidate()
            with torch.no_grad():fa=e.residual(activations).detach();fn=float((fa-fb).norm());netfn=float((fa-f0).norm())
            core={k:dict(parameter_norm=float(before[k].norm()),gradient_norm=float(p.grad.norm()),actual_update_norm=float((p-before[k]).norm()),relative_update_norm=float((p-before[k]).norm()/before[k].norm()) if before[k].norm()>0 else None) for k,p in e.named_parameters()}
            pn=sum(v['actual_update_norm']**2 for v in core.values())**.5;cumulative_parameter+=pn;cumulative_function+=fn
            item.update(step=step,fit_index=fi,extra_index=ei if hasH else None,diagnostic_H_index=ei,extra_role='H_fit' if hasH else None,cores=core,function_update_norm=fn,net_function_displacement=netfn,cumulative_parameter_displacement=cumulative_parameter,cumulative_function_displacement=cumulative_function,training_diagnostic=diag)
            item['terms']['native']['sample']=digest(t['native']);item['terms']['fit']['sample']=digest([t['native'],t['fit_questions'][fi-1]])
            if diagnose:
                post,postf=updates.diagnostic(runtime,hook,e,[batches[0],batches[fi]],[teachers[(step-1)%len(teachers)]],hb[ei],1. if hasH else 0.,activations);item['post_update_diagnostic']=post
            curve.append(item)
            if step%20==0:
                save(point,dict(binding=binding,expert=e.state_dict(),optimizer=opt.state_dict(),step=step,curve=curve,hook_state=updates.hook_state(hook),**cf.rng_state()))
                write(directory/'PROGRESS.json',dict(step=step,steps=320,accepted=sum(x.get('guard',{}).get('accepted',True) for x in curve)));print('UPDATE',t['order'],arm,step,flush=True)
        saved=torch.load(point,map_location='cpu',weights_only=True);assert all(torch.equal(v.cpu(),saved['expert'][k]) for k,v in e.state_dict().items())
        assert not hasH or any(c['terms']['extra']['weighted_gradient_norm']>0 for c in curve)
        peak_allocated=torch.cuda.max_memory_allocated();peak_reserved=torch.cuda.max_memory_reserved()
        with torch.no_grad(),updates.preserve(hook):
            torch.cuda.synchronize();timing_start=time.perf_counter()
            for _ in range(30):e.residual(activations)
            torch.cuda.synchronize();residual_seconds=(time.perf_counter()-timing_start)/30
        save(final,dict(binding=binding,expert=e.state_dict(),step=320,state_hash=cf.state_hash(e)))
        write(directory/'TRAINING.json',dict(status='COMPLETE',binding=binding,steps=320,seconds=time.time()-began,curve=curve,tokens={k:sum(c['terms'][k]['tokens'] for c in curve) for k in curve[0]['terms']},final_state_hash=cf.state_hash(e),final_sha256=sha(final),retained_for_deployment=True,training_peak_allocated_bytes=peak_allocated,training_peak_reserved_bytes=peak_reserved,TT_contraction_and_residual_seconds=residual_seconds,timing_repeats=30,timing_training_activation_shape=list(activations.shape),temporary_AB_bytes=294912,independent_AB_parameters=0,device_continuous_peak_unknown=True,guard_extra_forwards=sum(c.get('guard',{}).get('extra_forwards',0) for c in curve),accepted_updates=sum(c.get('guard',{}).get('accepted',True) for c in curve),diagnostic_extra_forwards=4*(1+2*5)+2*(5+len(teachers))))
        point.unlink()
    finally:hook.detach()
    del e,opt;gc.collect();torch.cuda.empty_cache();return final


def route_zero(route,seed,device):
    from structures import TT4
    return TT4(seed,*RANKS[route.split('_')[0]]).to(device)


def warmup(runtime,t,record,native_steps,cfg,route):
    import torch
    from methods.medtrace import MedTraceLayerHook
    from structures import optimizer_for
    from scripts.medtrace.run_selective_write import save
    from scripts.medtrace import stage18_cfact as cf
    from m3bench_repro.editors.llava_runtime import seed_everything
    d=RUN/'private/edits'/f"e{t['order']:03d}"/(route+'_WARMUP');d.mkdir(parents=True,exist_ok=True)
    e=route_zero(route,t['seed'],runtime.device).requires_grad_(True)
    start_state=cf.state_hash(e);batches=[runtime.build_edit_batch(record)]+[runtime.build_edit_batch(replace(record,question=q)) for q in t['fit_questions']]
    assert len(batches)==5 and all(runtime.adapter.tokenizer.eos_token_id in b.target_token_ids for b in batches)
    hook=MedTraceLayerHook(runtime.get_module(LAYER),e);hook.attach()
    try:
        for phase,steps in [('native',native_steps),('A2',80),('W0',320)]:
            seed_everything(t['seed'])
            opt=torch.optim.AdamW(e.parameters(),lr=1e-3,weight_decay=0) if phase!='W0' else optimizer_for(e,runtime.model)
            assert not opt.state
            binding=dict(route=route,phase=phase,steps=steps,frozen_native_steps=native_steps,input=t['native'],fit=t['fit_questions'],seed=t['seed'],input_state=cf.state_hash(e),source=cfg['code_commit'],execution_source=cfg['execution_source'])
            final=d/(phase+'.pt');point=d/(phase+'_latest.pt')
            if final.exists():
                saved=torch.load(final,map_location='cpu',weights_only=True);assert saved['binding']==binding and saved['step']==steps;e.load_state_dict(saved['expert']);continue
            start=0;curve=[]
            if point.exists():start,curve=cf.resume(point,binding,e,opt)
            began=time.time()
            for step in range(start+1,steps+1):
                budget();opt.zero_grad(set_to_none=True);indices=[0] if phase=='native' else [0,1+(step-1)%4];values=[]
                for i in indices:
                    batch=batches[i];hook.set_teacher_routing(batch.labels);loss=runtime.compute_loss(batch);(loss/len(indices)).backward();values.append(float(loss.detach()))
                norm=torch.nn.utils.clip_grad_norm_(e.parameters(),1.)
                assert torch.isfinite(norm) and all(p.grad is not None and torch.isfinite(p.grad).all() for p in e.parameters())
                opt.step();e.normalize_factors_();assert all(torch.isfinite(p).all() for p in e.parameters())
                curve.append(dict(step=step,indices=indices,losses=values,gradient_norm=float(norm),target_tokens=sum(len(batches[i].target_token_ids) for i in indices)))
                if step==1 or step%20==0 or step==steps:
                    save(point,dict(binding=binding,expert=e.state_dict(),optimizer=opt.state_dict(),step=step,curve=curve,**cf.rng_state()))
                    if step==1:
                        checked=clone(e.state_dict(),t['seed'],runtime.device);co=torch.optim.AdamW(checked.parameters(),lr=1e-3,weight_decay=0) if phase!='W0' else optimizer_for(checked,runtime.model)
                        after=cf.rng_state();cs,cc=cf.resume(point,binding,checked,co);assert cs==1 and cc==curve and cf.state_hash(checked)==cf.state_hash(e);reset_rng(after);del checked,co
                    print('WARMUP',t['order'],phase,step,steps,flush=True)
            save(final,dict(binding=binding,expert=e.state_dict(),step=steps,state_hash=cf.state_hash(e)))
            write(d/(phase+'_TRAINING.json'),dict(status='COMPLETE',binding=binding,steps=steps,curve=curve,seconds=time.time()-began,initial_random_state=start_state,final_state_hash=cf.state_hash(e),fresh_optimizer=True,save_load_step1=True))
            point.unlink()
    finally:hook.detach()
    e.requires_grad_(False);return e


def train_unit(runtime,bindings,ledger,canonical,cfg,structure,slot):
    import torch
    from scripts.medtrace import stage18_cfact as cf
    from scripts.medtrace.run_selective_write import save
    t=dict(canonical,seed=canonical['seed']+slot*1000003)
    arms=arms_for(structure,slot);root=RUN/'private/edits'/f"e{t['order']:03d}";root.mkdir(parents=True,exist_ok=True)
    unit=root/f'{structure}_s{slot}_COMPLETE.json'
    if unit.exists():assert read(unit)['arms']==list(arms);return
    frozen=[(p,p._version,p.data_ptr(),p.requires_grad) for p in runtime.model.parameters()];hooks=set(runtime.get_module(LAYER)._forward_hooks)
    record=legacy.record_for(t);runtime.run_root=root/'work';runtime.run_root.mkdir(exist_ok=True)
    legacy.check_input(runtime,t['native'],bindings);began=time.time()
    n=read(RUN/'PLAN_CONFIG.json')['initialization']['native_steps_by_order'][str(t['order'])]
    route=structure+'_s'+str(slot);e=warmup(runtime,t,record,n,cfg,route);init={k:v.detach().cpu().clone() for k,v in e.state_dict().items()};del e
    assert sum(v.numel() for v in init.values())==COUNTS[structure]
    points={}
    for arm in arms:
        points[arm]=continuation(runtime,t,init,record,cfg,arm)
        saved=torch.load(points[arm],map_location='cpu',weights_only=True);checked=clone(saved['expert'],t['seed'],'cpu');assert cf.state_hash(checked)==saved['state_hash']
        write(points[arm].parent/'STRUCTURE_CHECK.json',dict(status='PASS',saved_keys=sorted(saved['expert']),parameters=sum(v.numel() for v in saved['expert'].values()),bytes=points[arm].stat().st_size,state_hash=saved['state_hash'],deployment='TT_CORES_ONLY',temporary_AB_not_persisted=True))
    if not (root/'ROUTER.pt').exists():
        router=legacy.build_router(runtime,canonical,record);router['entries'][0]['label']=[];save(root/'ROUTER.pt',router)
    else:router=torch.load(root/'ROUTER.pt',map_location='cpu',weights_only=True)
    evaluate(runtime,ledger,bindings,legacy.query_ids(t),router['entries'],{t['edit_id']:points},1,'single',t['order'])
    evaluate_H(runtime,t,record,router['entries'],points,cfg)
    assert all(p._version==v and p.data_ptr()==ptr and p.requires_grad==req==False for p,v,ptr,req in frozen)
    assert set(runtime.get_module(LAYER)._forward_hooks)==hooks
    write(unit,dict(status='GENERATED_NOT_SCORED',order=t['order'],seed_slot=slot,seed=t['seed'],structure=structure,arms=list(points),warmup_once=True,Base_frozen=True,seconds=time.time()-began))
    print('UNIT_COMPLETE',t['order'],structure,slot,flush=True)


def evaluate(runtime,ledger,bindings,ids,bank,points,prefix,mode,order,forced=None):
    import torch
    from methods.medtrace import MedTraceLayerHook
    from m3bench_repro.editors.methods import BalanceEditPaperSpecEditor
    from m3bench_repro.editors.routing import MemoryRouter,decision_as_json,distances
    arms=tuple(next(iter(points.values())))
    first=next(iter(next(iter(points.values())).values()))
    expert=clone(torch.load(first,map_location='cpu',weights_only=True)['expert'],20260912,runtime.device).requires_grad_(False)
    hook=MedTraceLayerHook(runtime.get_module(LAYER),expert);hook.attach()
    editor=BalanceEditPaperSpecEditor(runtime);editor.router=MemoryRouter.from_state(dict(distance='euclidean',entries=bank),device=runtime.device)
    captured={};route=editor.router.route
    def capture(query):captured['query']=query.detach().clone();return route(query)
    editor.router.route=capture
    phase=dict(mode=mode,prefix=prefix,inserted=[x['logical_edit_id'] for x in bank],generation=runtime.generation_config,router=digest([{k:v.tolist() if hasattr(v,'tolist') else v for k,v in x.items()} for x in bank]),forced_original=forced,execution_source=read(RUN/'private/GPU_SOURCE_VERSION.json'))
    # Explicit complete artifact bindings; no sharing by answer string.
    cp_bind={eid:{a:read(p.parent/'TRAINING.json')['final_state_hash'] for a,p in ps.items()} for eid,ps in points.items()}
    phase['weight_ancestry']=digest(cp_bind);write(RUN/'private/bank_bindings'/f'{mode}_{order:03d}_{digest([phase,cp_bind])}.json',dict(phase=phase,points={eid:{a:str(p) for a,p in ps.items()} for eid,ps in points.items()},weights=cp_bind))
    loaded=None;common_checked=False
    try:
        for qid in ids:
            budget();q=ledger['queries'][qid];raw,b=legacy.check_input(runtime,q,bindings);hook.clear_request_routing()
            folder='native' if mode=='native' else 'panel'
            destinations={arm:RUN/'private/outputs'/arm/mode/f'e{order:03d}'/folder/(digest(qid)+'.json') for arm in arms}
            old_outputs={a:read(p) for a,p in destinations.items() if p.exists()}
            for a,d in old_outputs.items():
                assert d['binding']['input']==q and d['binding']['phase']==phase and d['binding']['arm']==a
            if len(old_outputs)==len(arms):continue
            query=replace(legacy.record_for(next(t for t in ledger['tasks'] if t['edit_id']==ledger['main_T0'][0])),record_id='query',question=q['question'],target='',official_rephrase='',image_path=Path(legacy.local_row(q)['image_path']))
            with torch.inference_mode():decision=editor._route(query)
            normal=decision_as_json(decision);on=decision.activated;selected=forced if forced else decision.logical_edit_id
            selected_on=bool(forced) or on
            alternatives=[]
            if not on:
                ds=distances(editor.router._key_matrix(runtime.device),captured['query'],'euclidean')
                alternatives=[dict(edit_id=e,distance=float(ds[i]),radius=editor.router.radii[i]) for i,e in enumerate(editor.router.logical_ids) if float(ds[i])<=editor.router.radii[i]]
            outputs={};actual_arms=arms if selected_on and selected in H else ['SHARED_NO_H']
            for a in actual_arms:
                reusable=old_outputs.get(a) or (next(iter(old_outputs.values()),None) if a=='SHARED_NO_H' else None)
                if reusable is not None:
                    outputs[a]=reusable['R0'];continue
                if selected_on:
                    p=points[selected][a];identity=(str(p),cp_bind[selected][a])
                    if loaded!=identity:
                        saved=torch.load(p,map_location='cpu',weights_only=True);assert saved['step']==320 and saved['state_hash']==cp_bind[selected][a]
                        expert.load_state_dict(saved['expert']);loaded=identity
                    with torch.inference_mode(),hook.generation_request():g=runtime.adapter.generate_prepared_with_result(raw,runtime.generation_config)
                    out=dict(raw_answer=g.decoded_text,raw_token_ids=list(g.raw_token_ids))
                    if a=='SHARED_NO_H' and not common_checked:
                        repeats=[]
                        for _ in range(len(arms)-1):
                            with torch.inference_mode(),hook.generation_request():g2=runtime.adapter.generate_prepared_with_result(raw,runtime.generation_config)
                            repeats.append(dict(raw_answer=g2.decoded_text,raw_token_ids=list(g2.raw_token_ids)))
                        assert all(x==out for x in repeats),'Common deterministic inference differed across six consumer checks'
                        common_checked=True
                else:out=dict(raw_answer=b['output']['model_answer_raw'],raw_token_ids=b['output']['raw_generated_token_ids'])
                outputs[a]=out
            for arm in arms:
                out=outputs[arm] if arm in outputs else outputs['SHARED_NO_H'];weight=(cp_bind[selected].get(arm) or cp_bind[selected].get('SHARED_NO_H')) if selected_on else None
                bind=dict(input=q,prefix=prefix,phase=phase,arm=arm,weight=weight,owner_order=order,selected_original=forced)
                dest=destinations[arm]
                if dest.exists():assert read(dest)['binding']==bind and read(dest)['R0']==out;continue
                write(dest,dict(binding=bind,Base_cache_id=q['opaque_Base_id'],R0=out,route=normal,effective_expert=selected if selected_on else None,other_own_radius_candidates=alternatives,
                    status='GENERATED_NOT_SCORED',execution_key=digest([bind,out,normal]),sharing='six bound consumers of common Base/nonH8 inference' if arm not in outputs else None))
    finally:
        hook.detach();target,base=editor.target,editor.wrapper.base;editor.reset_editor_state();runtime.replace_module(target,base)


def evaluate_H(runtime,t,record,bank,points,cfg):
    """Original single R0 on H_fit; explicitly not independent H_eval."""
    import torch
    from methods.medtrace import MedTraceLayerHook
    from m3bench_repro.editors.methods import BalanceEditPaperSpecEditor
    from m3bench_repro.editors.routing import MemoryRouter,decision_as_json
    expert=clone(torch.load(next(iter(points.values())),map_location='cpu',weights_only=True)['expert'],t['seed'],runtime.device).requires_grad_(False)
    hook=MedTraceLayerHook(runtime.get_module(LAYER),expert);hook.attach()
    editor=BalanceEditPaperSpecEditor(runtime);editor.router=MemoryRouter.from_state(dict(distance='euclidean',entries=bank),device=runtime.device)
    runtime_binding=next(iter(read(RUN/'private/legacy_stage17/BINDINGS.json').values()))['runtime']
    try:
        for i,h in enumerate(H[t['edit_id']]):
            raw=runtime.adapter.prepare_inputs(Path(h['image_path']),h['question'],None)
            q=dict(h,query_id='H_fit_'+digest([raw['image_sha256'],h['question'],h['reference']]),dataset='VQA-RAD',image_sha256=raw['image_sha256'])
            b=dict(question=q['question'],reference=q['reference'],image_sha256=q['image_sha256'],image_path=q['image_path'],prompt_ids=raw['input_ids'].tolist()[0],attention_mask=raw['attention_mask'].tolist()[0],runtime=runtime_binding,generation=runtime.generation_config)
            hook.clear_request_routing()
            query=replace(record,question=q['question'],image_path=Path(q['image_path']),target='',official_rephrase='')
            with torch.inference_mode():decision=editor._route(query)
            base_out=None
            for arm,p in points.items():
                dest=RUN/'private/outputs'/arm/'H_fit'/f"e{t['order']:03d}"/'panel'/(digest(q['query_id'])+'.json')
                state=torch.load(p,map_location='cpu',weights_only=True)
                bind=dict(input=q,prefix=1,arm=arm,weight=state['state_hash'],owner_order=t['order'],diagnostic='TRAINING_FIT_R0_NOT_H_EVAL',judge_input=b)
                if dest.exists():assert read(dest)['binding']==bind;continue
                if decision.activated:
                    expert.load_state_dict(state['expert'])
                    with torch.inference_mode(),hook.generation_request():g=runtime.adapter.generate_prepared_with_result(raw,runtime.generation_config)
                    out=dict(raw_answer=g.decoded_text,raw_token_ids=list(g.raw_token_ids))
                else:
                    if base_out is None:
                        with torch.inference_mode():g=runtime.adapter.generate_prepared_with_result(raw,runtime.generation_config)
                        base_out=dict(raw_answer=g.decoded_text,raw_token_ids=list(g.raw_token_ids))
                    out=base_out
                write(dest,dict(binding=bind,R0=out,route=decision_as_json(decision),effective_expert=t['edit_id'] if decision.activated else None,diagnostic_only=True))
    finally:
        hook.detach();target,base=editor.target,editor.wrapper.base;editor.reset_editor_state();runtime.replace_module(target,base)


def main():
    import torch
    assert os.environ.get('CUBLAS_WORKSPACE_CONFIG')==':4096:8'
    assert read(RUN/'RUNTIME_REPRODUCIBILITY_AMENDMENT.json')['deterministic_algorithms'] is True
    torch.use_deterministic_algorithms(True)
    gpu=int(os.environ['GPU']);assert gpu in (3,4,5,6)
    with legacy.lease(gpu):
        runtime,bindings=legacy.load(gpu);ledger=read(RUN/'private/legacy_stage17/COHORT_AND_SUPPORT_LEDGER.json');by={t['edit_id']:t for t in ledger['tasks']};tasks=[by[e] for e in ledger['main_T0'] if e in H]
        assert [t['order'] for t in tasks]==read(RUN/'PLAN_CONFIG.json')['H8_orders']
        cfg=dict(code_commit=read(RUN/'private/SOURCE_COMMIT.json')['commit'],execution_source=read(RUN/'private/GPU_SOURCE_VERSION.json'),runtime_lock=read(RUN/'private/cpu_gate/locks/CANONICAL_LLVAMED_RUNTIME_LOCK.json'),campaign_epoch=read(RUN/'RUN_MANIFEST.json')['starting_epoch'],train_seconds=24*3600)
        action=os.environ['ACTION']
        if action=='mechanical':
            from qualification import gpu_check
            gpu_check(runtime,tasks[0],cfg);write(RUN/'private/GPU_MECHANICAL.json',dict(status='PASS',epoch=time.time(),all_four_shapes=True,formal_training_not_started=True));return
        assert action=='p1' and read(RUN/'private/GPU_MECHANICAL.json')['status']=='PASS'
        phase=os.environ['PHASE'];structures=('TT44','TT88') if phase=='primary' else ('TT84','TT48')
        selected=tasks[int(os.environ['PARTITION'])::4]
        for slot in range(3):
            for t in selected:
                for structure in structures:train_unit(runtime,bindings,ledger,t,cfg,structure,slot)
        write(RUN/'private'/f"{phase}_PART_{os.environ['PARTITION']}_COMPLETE.json",dict(status='GENERATED_NOT_SCORED',edits=len(selected),seed_slots=3,structures=list(structures)))

if __name__=='__main__':
    def stop(signum,frame):raise InterruptedError('Own bounded worker interrupted; checkpoint retained')
    signal.signal(signal.SIGTERM,stop)
    try:main()
    except Exception as e:
        write(RUN/'private'/f"FAILURE_{os.environ.get('ACTION')}_{os.environ.get('GPU')}.json",dict(error=repr(e),traceback=traceback.format_exc(),epoch=time.time(),pid=os.getpid()));raise
