"""Dependency-ordered TT study, reusing the frozen PR27 runtime and update."""
from contextlib import nullcontext
from dataclasses import replace
import copy
import fcntl
import gc
import os
from pathlib import Path
import random
import signal
import sys
import time
import traceback

RUN=Path(os.environ['RUN_ROOT']);sys.path.insert(0,str(RUN/'private/tools'))
import baseline_worker as base
from audit import read,write,digest
import updates
from direction import clone,embed,step as direction_step
from geometry import spectrum,flat
PLAN=read(RUN/'PLAN_CONFIG.json');PARENT=Path(PLAN['parent_run'])
legacy=base.legacy;H=base.H;LAYER=base.LAYER
base.clone=clone


def path_for(t,structure,slot,warm=False,condition=None):
    root=PARENT/'private/edits'/f"e{t['order']:03d}"
    return root/(f'{structure}_s{slot}_WARMUP/W0.pt' if warm else f'{structure}_{condition}_s{slot}/final.pt')


def load_state(p):
    import torch
    from scripts.medtrace.stage18_cfact import state_hash
    saved=torch.load(p,map_location='cpu',weights_only=True)
    assert set(saved['expert'])=={'G1','G2','G3','G4'} and saved['step']==320
    e=clone(saved['expert'],0,'cpu');assert state_hash(e)==saved['state_hash'];return saved


def training_inputs(runtime,t,cfg,record):
    from scripts.medtrace import stage18_cfact as cf
    task=dict(t,U_fit=[legacy.local_row(u) for u in t['U_fit']],H_fit=H[t['edit_id']])
    with (RUN/'private/teacher/WRITE.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);teachers=cf.teachers_for(runtime,RUN,cfg,task,record)
    batches=[runtime.build_edit_batch(record)]+[runtime.build_edit_batch(replace(record,question=q)) for q in t['fit_questions']]
    hb=[cf.source_batch(runtime,record,h) for h in H[t['edit_id']]]
    eos=runtime.adapter.tokenizer.eos_token_id
    assert all(eos in b.target_token_ids and len(b.target_token_ids)<=128 for b in batches+hb)
    fit=list(range(1,5));random.Random(t['seed']).shuffle(fit)
    return teachers,batches,hb,fit,cf.extra_schedule(task)


def generate(runtime,hook,raw,on):
    import torch
    hook.clear_request_routing();hook.last_generation_trace=()
    torch.cuda.synchronize();began=time.perf_counter()
    with torch.inference_mode(),hook.generation_request() if on else nullcontext():
        g=runtime.adapter.generate_prepared_with_result(raw,runtime.generation_config)
    torch.cuda.synchronize();seconds=time.perf_counter()-began
    trace=list(hook.last_generation_trace);assert bool(trace)==on and not hook.enabled
    return dict(raw_answer=g.decoded_text,raw_token_ids=list(g.raw_token_ids)),dict(seconds=seconds,scope='complete_prepared_VQA_generation_excludes_input_preparation_and_route',hook_generation_trace=trace,actual_hook_active=bool(trace),post_request_hook_enabled=hook.enabled)


def nll(runtime,hook,batch,on):
    import torch
    with updates.preserve(hook),torch.no_grad():
        hook.clear_request_routing()
        if on:hook.set_teacher_routing(batch.labels)
        loss=float(runtime.compute_loss(batch))
    updates.finite([loss]);return loss


def base_H(runtime,hook,raw,b):
    # Generation identity excludes reference: one generation per image/question.
    identity={k:v for k,v in b.items() if k!='reference'};key=digest(identity)
    root=RUN/'private/base_H';root.mkdir(exist_ok=True);dest=root/(key+'.json')
    with (root/(key+'.lock')).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if dest.exists():d=read(dest);assert d['binding']==identity;return d['output'],d['diagnostic']
        out,diag=generate(runtime,hook,raw,False);write(dest,dict(binding=identity,output=out,diagnostic=diag));return out,diag


def evaluate(runtime,bindings,ledger,t,points,phase,h_only=False):
    """One R0 decision; measure forced ON. Training never reads these outputs."""
    import torch
    from methods.medtrace import MedTraceLayerHook
    from m3bench_repro.editors.methods import BalanceEditPaperSpecEditor
    from m3bench_repro.editors.routing import MemoryRouter,decision_as_json
    from scripts.medtrace import stage18_cfact as cf
    record=legacy.record_for(t);router=torch.load(PARENT/'private/edits'/f"e{t['order']:03d}"/'ROUTER.pt',map_location='cpu',weights_only=True)
    loaded={arm:load_state(path) for arm,path in points.items()}
    first=next(iter(loaded.values()));expert=clone(first['expert'],t['seed'],runtime.device).requires_grad_(False)
    hook=MedTraceLayerHook(runtime.get_module(LAYER),expert);hook.attach()
    editor=BalanceEditPaperSpecEditor(runtime);editor.router=MemoryRouter.from_state(dict(distance='euclidean',entries=router['entries']),device=runtime.device)
    runtime_binding=next(iter(bindings.values()))['runtime']
    entries=[]
    if not h_only:
        for qid in legacy.query_ids(t):
            q=ledger['queries'][qid];raw,b=legacy.check_input(runtime,q,bindings)
            entries.append(('panel',q,raw,b))
    for h in H[t['edit_id']]:
        raw=runtime.adapter.prepare_inputs(Path(h['image_path']),h['question'],None)
        q=dict(h,query_id='H_fit_'+digest([raw['image_sha256'],h['question'],h['reference']]),dataset='VQA-RAD',image_sha256=raw['image_sha256'])
        b=dict(question=q['question'],reference=q['reference'],image_sha256=q['image_sha256'],image_path=q['image_path'],prompt_ids=raw['input_ids'].tolist()[0],attention_mask=raw['attention_mask'].tolist()[0],runtime=runtime_binding,generation=runtime.generation_config)
        entries.append(('H',q,raw,b))
    try:
        for category,q,raw,b in entries:
            base.budget();hook.clear_request_routing();query=replace(record,question=q['question'],image_path=Path(legacy.local_row(q)['image_path']),target='',official_rephrase='')
            with torch.inference_mode():decision=editor._route(query)
            route=decision_as_json(decision);on=bool(decision.activated)
            assert not on or decision.logical_edit_id==t['edit_id']
            batch=cf.source_batch(runtime,record,legacy.local_row(q))
            offout,offdiag=base_H(runtime,hook,raw,b) if category=='H' else (dict(raw_answer=b['output']['model_answer_raw'],raw_token_ids=b['output']['raw_generated_token_ids']),dict(actual_hook_active=False,inherited_frozen_Base=True,seconds=None))
            offnll=nll(runtime,hook,batch,False)
            for arm,state in loaded.items():
                expert_state=state['expert']
                if any(expert.state_dict()[k].shape!=v.shape for k,v in expert_state.items()):
                    # Rebind the same attached hook for mixed P2 ranks; never train A/B.
                    expert=clone(expert_state,t['seed'],runtime.device).requires_grad_(False);hook.expert=expert
                else:expert.load_state_dict(expert_state)
                output,diag=generate(runtime,hook,raw,True);onnll=nll(runtime,hook,batch,True)
                if category=='panel' and q['query_id']==t['edit_id']:
                    # End-to-end warm inference includes image preparation and R0.
                    # The resident expert is deployed as TT cores, with transient A/B.
                    hook.clear_request_routing();torch.cuda.reset_peak_memory_stats();memory_before=torch.cuda.memory_allocated();torch.cuda.synchronize();tick=time.perf_counter()
                    raw2=runtime.adapter.prepare_inputs(Path(legacy.local_row(q)['image_path']),q['question'],None)
                    with torch.inference_mode():decision2=editor._route(query)
                    output2,_=generate(runtime,hook,raw2,bool(decision2.activated));torch.cuda.synchronize()
                    elapsed=time.perf_counter()-tick
                    assert decision_as_json(decision2)==route and output2==(output if on else offout)
                    write(RUN/'private/inference'/f"{arm}_e{t['order']:03d}.json",dict(arm=arm,anonymous_edit=f"E{t['order']:03d}",seconds=elapsed,scope='warm resident Base+TT; input preprocessing+original R0+complete autoregressive VQA; excludes initial model/expert disk loading',repeats=1,peak_allocated=torch.cuda.max_memory_allocated(),incremental_peak_allocated=torch.cuda.max_memory_allocated()-memory_before,temporary_AB_bytes=sum(f.numel()*f.element_size() for f in expert.factors())))
                for mode,value,loss,gd in [('ON',output,onnll,diag),('R0',output if on else offout,onnll if on else offnll,diag if on else offdiag),('OFF',offout,offnll,offdiag)]:
                    if category=='panel' and mode=='OFF':continue
                    folder={'panel':{'R0':'single','ON':'forced'},'H':{'R0':'H_fit','ON':'H_ON','OFF':'H_OFF'}}[category][mode]
                    dest=RUN/'private/outputs'/arm/folder/f"e{t['order']:03d}"/'panel'/(digest(q['query_id'])+'.json')
                    binding=dict(input=q,arm=arm,prefix=1,owner_order=t['order'],judge_input=b,weight=state['state_hash'],source_weight=str(points[arm]),phase=phase,mode=mode,execution_source=read(RUN/'private/GPU_SOURCE_VERSION.json'))
                    historical=PARENT/'private/outputs'/arm/('single' if category=='panel' else 'H_fit')/f"e{t['order']:03d}"/'panel'/(digest(q['query_id'])+'.json')
                    reuse=None
                    if mode=='R0' and historical.exists():
                        previous=read(historical)
                        assert previous['binding']['input']==q and previous['R0']==value and previous['route']==route
                        if category=='H':assert previous['binding']['judge_input']==b and previous['binding']['weight']==state['state_hash']
                        elif on:assert previous['binding']['weight']==state['state_hash']
                        reuse=dict(path=str(historical),binding=digest(previous),verified_new_forced_ON_equal_to_R0_when_on=True)
                    obj=dict(binding=binding,R0=value,route=route,effective_expert=t['edit_id'] if mode=='ON' or mode=='R0' and on else None,reference_NLL=loss,generation_diagnostic=gd,historical_reuse=reuse,status='GENERATED_NOT_SCORED',diagnostic_only=category=='H' or mode!='R0')
                    if dest.exists():assert read(dest)['binding']==binding and read(dest)['R0']==value
                    else:write(dest,obj)
            print('EVALUATED',phase,t['order'],category,flush=True)
    finally:
        hook.detach();target,original=editor.target,editor.wrapper.base;editor.reset_editor_state();runtime.replace_module(target,original)
    del expert;gc.collect();torch.cuda.empty_cache()


def continue_arm(runtime,t,record,cfg,init,arm,kind,match_path=None):
    import torch
    from methods.medtrace import MedTraceLayerHook
    from structures import optimizer_for
    from scripts.medtrace import stage18_cfact as cf
    from scripts.medtrace.run_selective_write import save
    from m3bench_repro.editors.llava_runtime import seed_everything
    root=RUN/'private/edits'/f"e{t['order']:03d}"/arm;root.mkdir(parents=True,exist_ok=True)
    final=root/'final.pt';latest=root/'latest.pt'
    teachers,batches,hb,fit,schedule=training_inputs(runtime,t,cfg,record)
    e=clone(init,t['seed'],runtime.device).requires_grad_(True);seed_everything(t['seed']);opt=optimizer_for(e,runtime.model)
    binding=dict(arm=arm,kind=kind,seed=t['seed'],W0=cf.state_hash(e),fit_order=fit,H_order=schedule,steps=320,input=t['native'],H=H[t['edit_id']],U=t['U_fit'],config=digest(PLAN),execution=cfg['execution_source'],generation=runtime.generation_config)
    if final.exists():s=load_state(final);assert s['binding']==binding;return final
    assert not opt.state
    targets=read(match_path) if match_path else None
    if targets:
        assert targets['training_only'] and targets['W0']==binding['W0'] and targets['seed']==t['seed'] and len(targets['steps'])==320
    hook=MedTraceLayerHook(runtime.get_module(LAYER),e);hook.attach();began=time.time();curve=[];start=0
    torch.cuda.reset_peak_memory_stats()
    try:
        initial=updates.protection(runtime,hook,batches,teachers);thresholds=updates.limits(initial)
        activations=updates.capture_training_activations(runtime,hook,batches,teachers,LAYER)
        with torch.no_grad():f0=e.residual(activations).detach().clone()
        write(root/'W0_DIAGNOSTIC.json',dict(protection=initial,thresholds=thresholds,spectrum=spectrum(e),activation_shape=list(activations.shape)))
        if latest.exists():
            start,curve=base.resume(latest,binding,e,opt);updates.restore_hook(hook,torch.load(latest,map_location='cpu',weights_only=True)['hook_state'])
        else:save(latest,dict(binding=binding,expert=e.state_dict(),optimizer=opt.state_dict(),step=0,curve=[],hook_state=updates.hook_state(hook),**cf.rng_state()))
        for index in range(start+1,321):
            base.budget();fi=fit[(index-1)%4];hi=schedule[index-1];hasH=kind!='NO_H';diagnose=index in (1,20,80,160,320)
            diag,_=updates.diagnostic(runtime,hook,e,[batches[0],batches[fi]],[teachers[(index-1)%len(teachers)]],hb[hi],float(hasH),activations) if diagnose else (None,None)
            def candidate():return base.logged_update(runtime,hook,e,opt,batches[0],batches[fi],teachers[(index-1)%len(teachers)],(hb[hi],H[t['edit_id']][hi]) if hasH else None,1.,diagnose)
            item=direction_step(runtime,hook,e,opt,candidate,batches,teachers,hb[hi],thresholds,activations,kind if kind in ('DIR','MATCH') else 'JOINT',targets['steps'][index-1] if targets else None)
            with torch.no_grad():net=float((e.residual(activations)-f0).norm())
            item.update(step=index,fit_index=fi,H_index=hi,diagnostic=diag,net_function_displacement=net,spectrum=spectrum(e) if diagnose else None)
            if diagnose:
                post,_=updates.diagnostic(runtime,hook,e,[batches[0],batches[fi]],[teachers[(index-1)%len(teachers)]],hb[hi],float(hasH),activations);item['post_diagnostic']=post
            curve.append(item)
            if index%20==0:
                save(latest,dict(binding=binding,expert=e.state_dict(),optimizer=opt.state_dict(),step=index,curve=curve,hook_state=updates.hook_state(hook),**cf.rng_state()))
                write(root/'PROGRESS.json',dict(step=index,steps=320,accepted=sum(c['accepted'] for c in curve),H_descent=sum(c['H_descent'] for c in curve)));print('UPDATE',t['order'],arm,index,flush=True)
        total=sum(c['function_update_norm'] for c in curve);match_error=abs(total-sum(targets['steps']))/max(sum(targets['steps']),1e-12) if targets else None
        status='MATCH_FAILED' if match_error is not None and (match_error>.05 or any((c['function_update_norm']==0)!=(target==0) for c,target in zip(curve,targets['steps']))) else 'COMPLETE'
        save(final,dict(binding=binding,expert=e.state_dict(),step=320,state_hash=cf.state_hash(e)))
        write(root/'TRAINING.json',dict(status=status,binding=binding,steps=320,curve=curve,seconds=time.time()-began,final_state_hash=cf.state_hash(e),parameters=sum(p.numel() for p in e.parameters()),TT_bytes=final.stat().st_size,temporary_AB_bytes=sum(f.numel()*f.element_size() for f in e.factors()),independent_AB_parameters=0,function_path=total,net_function_displacement=net,match_relative_error=match_error,accepted=sum(c['accepted'] for c in curve),nonzero=sum(c['nonzero'] for c in curve),H_descent=sum(c['H_descent'] for c in curve),H_learning_status='H_LEARNING_STALLED' if not any(c['H_descent'] for c in curve) else 'TRAINING_LOSS_DESCENT_OBSERVED_NOT_TASK_SUCCESS',peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved()))
        if kind=='DIR':write(root/'FUNCTION_TARGETS.json',dict(training_only=True,W0=binding['W0'],seed=t['seed'],activation_source='W0 native/fit/U first16 predictor activations per batch',steps=[c['function_update_norm'] for c in curve]))
        checked=load_state(final);assert checked['state_hash']==cf.state_hash(e);latest.unlink()
    finally:hook.detach()
    del e,opt;gc.collect();torch.cuda.empty_cache();return final


def p0(runtime,bindings,ledger,t,cfg):
    import torch
    points={}
    for slot in range(3):
        for structure in ('TT88','TT84'):
            points[f'{structure}_FROZEN_s{slot}']=path_for(t,structure,slot,warm=True)
            audit_geometry(runtime,dict(t,seed=t['seed']+slot*1000003),cfg,structure,slot)
        for condition in ('NO_H','H1'):points[f'TT84_{condition}_s{slot}']=path_for(t,'TT84',slot,condition=condition)
        for condition in ('NO_H','H1','GUARDED_H'):points[f'TT88_{condition}_s{slot}']=path_for(t,'TT88',slot,condition=condition)
    evaluate(runtime,bindings,ledger,t,points,'P0')
    write(RUN/'private/P0'/f"e{t['order']:03d}.json",dict(status='GENERATED_NOT_SCORED',arms=list(points),formal_continuation=False))


def audit_geometry(runtime,t,cfg,structure,slot):
    import torch
    from methods.medtrace import MedTraceLayerHook
    from structures import optimizer_for
    from direction import linearization
    from m3bench_repro.editors.llava_runtime import seed_everything
    state=load_state(path_for(t,structure,slot,warm=True))['expert'];record=legacy.record_for(t)
    teachers,batches,hb,fit,schedule=training_inputs(runtime,t,cfg,record)
    e=clone(state,t['seed'],runtime.device).requires_grad_(True);opt=optimizer_for(e,runtime.model);seed_everything(t['seed'])
    hook=MedTraceLayerHook(runtime.get_module(LAYER),e);hook.attach()
    try:
        activations=updates.capture_training_activations(runtime,hook,batches,teachers,LAYER)
        g,c,gh,h=linearization(runtime,hook,e,batches,teachers,hb[schedule[0]])
        diag,fb=updates.diagnostic(runtime,hook,e,[batches[0],batches[fit[0]]],teachers,hb[schedule[0]],1.,activations)
        before=flat(e);spec=spectrum(e)
        item=base.logged_update(runtime,hook,e,opt,batches[0],batches[fit[0]],teachers[0],(hb[schedule[0]],H[t['edit_id']][schedule[0]]),1.,True)
        d=flat(e)-before;after=updates.protection(runtime,hook,batches,teachers)
        with torch.no_grad():fn=float((e.residual(activations)-fb).norm())
        result=dict(anonymous_edit=f"E{t['order']:03d}",structure=structure,seed_slot=slot,diagnostic=diag,spectrum=spec,actual_Adam_norm=float(d.norm()),H_dot_Adam=float(gh@d),protection_dot_Adam=(g@d).tolist(),protection_gradient_norms=g.norm(dim=1).tolist(),H_raw_gradient_norm=float(gh.norm()),function_displacement=fn,protection_before=c,protection_candidate=after,thresholds=updates.limits(c),candidate_clipping=dict(preclip=item['preclip_total_norm'],scale=item['clip_scale']),candidate_only=True,formal_updates=0)
        write(RUN/'private/P0_geometry'/f"{structure}_e{t['order']:03d}_s{slot}.json",result)
    finally:hook.detach()
    del e,opt;gc.collect();torch.cuda.empty_cache()


def p1(runtime,bindings,ledger,canonical,cfg,structure='TT88'):
    for slot in range(3):
        t=dict(canonical,seed=canonical['seed']+slot*1000003);record=legacy.record_for(t);initial=load_state(path_for(t,structure,slot,warm=True))['expert'];points={}
        arm=f'{structure}_DIR_s{slot}';points[arm]=continue_arm(runtime,t,record,cfg,initial,arm,'DIR')
        if structure=='TT88':
            other=f'{structure}_MATCH_s{slot}';points[other]=continue_arm(runtime,t,record,cfg,initial,other,'MATCH',points[arm].parent/'FUNCTION_TARGETS.json')
        # All training for the paired block is complete before any generation.
        evaluate(runtime,bindings,ledger,t,points,'P1' if structure=='TT88' else 'P2A')
        write(RUN/'private/blocks'/f"{structure}_e{t['order']:03d}_s{slot}.json",dict(status='GENERATED_NOT_SCORED',arms=list(points)))


def p2b(runtime,bindings,ledger,t,cfg):
    import torch
    from qualification_direction import expansion_check
    initial=load_state(path_for(t,'TT44',0,warm=True))['expert'];record=legacy.record_for(t);points={}
    for kind in ('OUTER','MIDDLE'):
        e=embed(initial,kind,t['seed']);expansion_check(runtime,record,initial,e,t['seed'],RUN/'private/embedding'/f"e{t['order']:03d}_{kind}.json")
        new={k:v.detach().clone() for k,v in e.state_dict().items()};del e
        for condition in ('NO_H','H1'):
            arm=f'{kind}_{condition}_s0';points[arm]=continue_arm(runtime,t,record,cfg,new,arm,condition)
    evaluate(runtime,bindings,ledger,t,points,'P2B')
    write(RUN/'private/blocks'/f"P2B_e{t['order']:03d}.json",dict(status='GENERATED_NOT_SCORED',arms=list(points)))


def main():
    import torch
    torch.use_deterministic_algorithms(True);assert os.environ['CUBLAS_WORKSPACE_CONFIG']==':4096:8'
    gpu=int(os.environ['GPU']);action=os.environ['ACTION'];legacy.GPUS={int(k):v for k,v in PLAN['hardware']['UUIDs'].items()}
    with legacy.lease(gpu):
        runtime,bindings=legacy.load(gpu);ledger=read(RUN/'private/legacy_stage17/COHORT_AND_SUPPORT_LEDGER.json');by={t['edit_id']:t for t in ledger['tasks']};tasks=[by[e] for e in ledger['main_T0'] if e in H]
        cfg=dict(code_commit=read(RUN/'private/SOURCE_COMMIT.json')['commit'],execution_source=read(RUN/'private/GPU_SOURCE_VERSION.json'),runtime_lock=read(RUN/'private/cpu_gate/locks/CANONICAL_LLVAMED_RUNTIME_LOCK.json'))
        if action=='mechanical':
            from qualification_direction import check
            check(runtime,tasks[0],cfg);return
        assert read(RUN/'private/GPU_MECHANICAL.json')['status']=='PASS'
        for t in tasks[int(os.environ['PARTITION'])::4]:
            if action=='P0':p0(runtime,bindings,ledger,t,cfg)
            elif action=='P1':p1(runtime,bindings,ledger,t,cfg)
            elif action=='P2A':p1(runtime,bindings,ledger,t,cfg,'TT84')
            elif action=='P2B':p2b(runtime,bindings,ledger,t,cfg)
            else:raise ValueError(action)
        write(RUN/'private'/f"{action}_PART_{os.environ['PARTITION']}_COMPLETE.json",dict(status='GENERATED_NOT_SCORED',action=action))


if __name__=='__main__':
    def stop(*_):raise InterruptedError('Bounded worker stopped; latest state preserved')
    signal.signal(signal.SIGTERM,stop)
    try:main()
    except Exception as error:
        write(RUN/'private'/f"FAILURE_{os.environ.get('ACTION')}_{os.environ.get('GPU')}.json",dict(error=repr(error),traceback=traceback.format_exc(),epoch=time.time(),automatic_retry=False));raise
