"""Four frozen continuations and real paired banks; original writer/R0 unchanged."""
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

ARMS=('A_NO_H','B_H1','C_H025','D_EXTRA_FIT')
H=read(RUN/'private/H_AVAILABLE.json')
LAYER=legacy.LAYER


def budget(*_):
    legacy_original_budget()
    files=[p for p in (RUN/'private').rglob('*.pt') if not p.is_symlink()]
    used=sum(p.stat().st_size for p in files);ledger=read(RUN/'RESOURCE_LEDGER.json')
    with legacy.locked_ledger() as ledger:
        ledger['weights_observed_peak_bytes']=max(ledger.get('weights_observed_peak_bytes',0),used)
    if used>=4*1024**3:raise OSError('Frozen4GiB generated-weight cap reached')


legacy_original_budget=legacy.budget
legacy.budget=budget


def clone(init,seed,device):
    from methods.medtrace import AsymmetricCPExpert
    from methods.medtrace.selective_write import LowRankExpert
    import torch
    # No CP training here; this constructor only supplies the unchanged shape.
    expert=LowRankExpert(AsymmetricCPExpert(14336,4096,4).to(device),seed,rank=4).to(device)
    expert.load_state_dict(init);return expert


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


def continuation(runtime,t,init,record,cfg,arm,mechanical):
    import torch
    from methods.medtrace import MedTraceLayerHook
    from methods.medtrace.selective_write import optimizer_for
    from scripts.medtrace import stage18_cfact as cf
    from scripts.medtrace.run_selective_write import save
    from m3bench_repro.editors.llava_runtime import seed_everything
    directory=RUN/'private/edits'/f"e{t['order']:03d}"/arm;directory.mkdir(parents=True,exist_ok=True)
    final=directory/'final.pt';point=directory/'latest.pt';rows=H[t['edit_id']] if arm in ('B_H1','C_H025') else []
    task=dict(t,U_fit=[legacy.local_row(u) for u in t['U_fit']],H_fit=rows)
    with (RUN/'private/teacher/WRITE.lock').open('a') as teacher_lock:
        fcntl.flock(teacher_lock,fcntl.LOCK_EX)
        teachers=cf.teachers_for(runtime,RUN,cfg,task,record)
    batches=[runtime.build_edit_batch(record)]+[runtime.build_edit_batch(replace(record,question=q)) for q in t['fit_questions']]
    hb=[cf.source_batch(runtime,record,h) for h in rows]
    eos=runtime.adapter.tokenizer.eos_token_id
    assert len(batches)==5 and all(eos in x.target_token_ids for x in batches+hb)
    assert all(len(x.target_token_ids)<=128 for x in hb)
    fit=list(range(1,5));random.Random(t['seed']).shuffle(fit)
    order=cf.extra_schedule(task) if rows else []
    weight=.25 if arm=='C_H025' else 1.
    expert=clone(init,t['seed'],runtime.device);W0=cf.state_hash(expert)
    binding=dict(input=t['native'],fit=t['fit_questions'],U=t['U_fit'],H=rows,arm=arm,seed=t['seed'],W_init=W0,steps=320,fit_order=fit,H_order=order,extra_weight=weight if arm!='A_NO_H' else None,D_rule='next ordinary fit in frozen cyclic fit_order',source=cfg['code_commit'],execution_source=cfg['execution_source'],runtime=cfg['runtime_lock'],generation=runtime.generation_config)
    if final.exists():
        state=torch.load(final,map_location='cpu',weights_only=True);assert state['step']==320
        if state['binding']!=binding:
            approval=read(RUN/'private/INITIALIZATION_REUSE_APPROVAL.json')
            assert t['order'] in approval['orders'] and arm=='A_NO_H'
            assert state['binding']['execution_source'] in approval['approved_execution_versions']
            assert dict(state['binding'],execution_source=binding['execution_source'])==binding
            done=read(directory/'TRAINING.json');check=read(directory/'ONE_STEP_CHECK.json')
            assert done['status']=='COMPLETE' and done['steps']==320 and len(done['curve'])==320
            assert check['status']=='PASS' and check['diagnostics_parity'] and check['save_load_resume']
            assert done['final_sha256']==sha(final) and done['final_state_hash']==state['state_hash']
            write(directory/'FINAL_REUSE_BINDING.json',dict(status='PASS',original=state['binding'],current=binding,reason='mechanical fixture repair only; completed original NO_H branch retained without rewriting or retraining'))
        del expert;return final
    # Build both mechanical control objects before fixing the formal RNG state.
    # This removes constructor/setup state from the ON/OFF comparison.
    reference=clone(init,t['seed'],runtime.device).requires_grad_(True) if mechanical else None
    ropt=optimizer_for(reference,runtime.model) if mechanical else None
    if mechanical:
        assert cf.state_hash(reference)==W0
        assert all(p.data_ptr()!=q.data_ptr() for p,q in zip(expert.parameters(),reference.parameters()))
        assert not ropt.state
    seed_everything(t['seed']);expert.requires_grad_(True);opt=optimizer_for(expert,runtime.model);assert not opt.state
    curve=[];start=0
    if point.exists():start,curve=cf.resume(point,binding,expert,opt)
    hook=MedTraceLayerHook(runtime.get_module(LAYER),expert);hook.attach();began=time.time()
    try:
        for step in range(start+1,321):
            budget();fi=fit[(step-1)%4]
            if rows:ei=order[step-1];extra=(hb[ei],rows[ei])
            elif arm=='D_EXTRA_FIT':ei=fit[step%4];extra=(batches[ei],dict(question=t['fit_questions'][ei-1],source='already-frozen-native-fit'))
            else:ei=None;extra=None
            rng=cf.rng_state() if mechanical and step==1 else None
            item=logged_update(runtime,hook,expert,opt,batches[0],batches[fi],teachers[(step-1)%len(teachers)],extra,weight,step in (1,20,80,160,320))
            item['terms']['native']['sample']=digest(t['native'])
            item['terms']['fit']['sample']=digest([t['native'],t['fit_questions'][fi-1]])
            item.update(step=step,fit_index=fi,extra_index=ei,extra_role='H_fit' if rows else 'EXTRA_FIT' if extra else None,
                H_gradient=None if not rows else item['terms']['extra']['weighted_gradient_norm'])
            if mechanical and step==1:
                after=cf.rng_state()
                hook.detach();rh=MedTraceLayerHook(runtime.get_module(LAYER),reference);rh.attach();reset_rng(rng)
                try:reference_item=cf.update(runtime,rh,reference,ropt,batches[0],batches[fi],teachers[0],extra,extra_weight=weight)
                finally:rh.detach();reset_rng(after);hook.attach()
                write(directory/'ONE_STEP_RAW_EVIDENCE.json',dict(W_init=W0,logged=item,unlogged=reference_item,max_parameter_difference=[float((p-q).abs().max()) for p,q in zip(expert.parameters(),reference.parameters())]))
                assert all(torch.equal(p,q) for p,q in zip(expert.parameters(),reference.parameters())), 'Diagnostic ON/OFF changed actual one-step update'
                assert arm not in ('B_H1','C_H025') or item['H_gradient']>0
                assert arm!='D_EXTRA_FIT' or (not rows and len(hb)==0 and item['extra_role']=='EXTRA_FIT')
                # The verified logged step is the formal step1; it is saved and reused.
                save(point,dict(binding=binding,expert=expert.state_dict(),optimizer=opt.state_dict(),step=step,curve=[item],**cf.rng_state()))
                checked=clone(init,t['seed'],runtime.device);copt=optimizer_for(checked,runtime.model)
                loaded_step,loaded_curve=cf.resume(point,binding,checked,copt)
                assert loaded_step==1 and loaded_curve==[item] and all(torch.equal(p,q) for p,q in zip(expert.parameters(),checked.parameters()))
                reset_rng(after);del checked,copt,reference,ropt
                write(directory/'ONE_STEP_CHECK.json',dict(status='PASS',W_init=W0,optimizer_fresh=True,diagnostics_parity=True,reference='original stage18.update with NO_H' if arm=='A_NO_H' else 'original stage18.update with C_FACT weight1' if arm=='B_H1' else 'original update with authorized extra_weight/fit slot',formal_step1_reused=True,save_load_resume=True,H_read=bool(rows),H_real_backward=None if not rows else True))
            curve.append(item)
            if step%20==0:
                save(point,dict(binding=binding,expert=expert.state_dict(),optimizer=opt.state_dict(),step=step,curve=curve,**cf.rng_state()))
                write(directory/'PROGRESS.json',dict(step=step,steps=320));print('UPDATE',t['order'],arm,step,flush=True)
        loaded=torch.load(point,map_location='cpu',weights_only=True)
        assert loaded['binding']==binding and all(torch.equal(v.cpu(),loaded['expert'][k]) for k,v in expert.state_dict().items())
        if rows:assert any(c['H_gradient']>0 for c in curve)
        save(final,dict(binding=binding,expert=expert.state_dict(),step=320,state_hash=cf.state_hash(expert)))
        write(directory/'TRAINING.json',dict(status='COMPLETE',binding=binding,steps=320,seconds=time.time()-began,curve=curve,tokens={k:sum(c['terms'][k]['tokens'] for c in curve) for k in curve[0]['terms']},final_state_hash=cf.state_hash(expert),final_sha256=sha(final),resume_point_retained=False))
        point.unlink() # Final compact expert is now the registered consumer artifact.
    finally:hook.detach()
    expert.requires_grad_(False);del expert,opt;gc.collect();torch.cuda.empty_cache();return final


def train_edit(runtime,bindings,ledger,t,cfg,paired=False,mechanical=False):
    import torch
    from methods.medtrace import AsymmetricCPExpert,MedTraceLayerHook
    from methods.medtrace.selective_write import LowRankExpert
    from scripts.medtrace import stage15,stage18_cfact as cf
    from scripts.medtrace.run_selective_write import save
    d=RUN/'private/edits'/f"e{t['order']:03d}";d.mkdir(parents=True,exist_ok=True)
    if (d/'COMPLETE.json').exists():assert read(d/'COMPLETE.json')['arms']==list(ARMS if paired else ['SHARED_NO_H']);return
    frozen=[(p,p._version,p.data_ptr(),p.requires_grad) for p in runtime.model.parameters()];hooks=set(runtime.get_module(LAYER)._forward_hooks)
    record=legacy.record_for(t);runtime.run_root=d/'work';runtime.run_root.mkdir(exist_ok=True)
    raw,b=legacy.check_input(runtime,t['native'],bindings);initpoint=d/'W_init.pt';began=time.time()
    init_binding=dict(input=t['native'],seed=t['seed'],fit=t['fit_questions'],U=t['U_fit'],source=cfg['code_commit'],execution_source=cfg['execution_source'],runtime=cfg['runtime_lock'],generation=runtime.generation_config,procedure='nativeCP-A2_80-CPW0_320-freeR4-before-continuation')
    if initpoint.exists():
        saved=torch.load(initpoint,map_location='cpu',weights_only=True)
        if saved['binding']!=init_binding:
            approval=read(RUN/'private/INITIALIZATION_REUSE_APPROVAL.json')
            assert saved['binding']['execution_source'] in approval['approved_execution_versions']
            assert dict(saved['binding'],execution_source=init_binding['execution_source'])==init_binding
            write(d/'INITIALIZATION_REUSE_BINDING.json',dict(status='PASS',original=saved['binding'],current=init_binding,reason='mechanical-validator correction only; exact same original CP procedure/data/runtime; initialization never repeated'))
        init=saved['expert']
    else:
        if mechanical:
            rng=cf.rng_state();zero=LowRankExpert(AsymmetricCPExpert(14336,4096,4).to(runtime.device),t['seed'],rank=4).to(runtime.device)
            with torch.no_grad():zero.B.zero_()
            h=MedTraceLayerHook(runtime.get_module(LAYER),zero);h.attach()
            try:
                with torch.inference_mode():
                    off=runtime.adapter.generate_prepared_with_result(raw,runtime.generation_config)
                    with h.generation_request():z=runtime.adapter.generate_prepared_with_result(raw,runtime.generation_config)
                assert list(off.raw_token_ids)==list(z.raw_token_ids)==b['output']['raw_generated_token_ids']
            finally:h.detach();reset_rng(rng)
            del zero
        adapted=dict(canonical_edit_id=t['edit_id'],order=t['order'],seed=t['seed'],probes=[legacy.local_row(t['native'])],U=[legacy.local_row(u) for u in t['U_fit']],fit_questions=t['fit_questions'])
        cp=stage15.initialize(runtime,RUN,cfg,adapted,record=record,seed_base=20260912)
        expert=LowRankExpert(cp,t['seed'],rank=4).to(runtime.device)
        with torch.no_grad():
            x=torch.linspace(-1,1,14336,device=runtime.device).reshape(1,14336)
            assert torch.allclose(cp.residual(x),expert.residual(x),rtol=2e-4,atol=2e-5)
        if mechanical:
            out=[]
            for e in [cp,expert]:
                h=MedTraceLayerHook(runtime.get_module(LAYER),e);h.attach()
                try:
                    with torch.inference_mode(),h.generation_request():g=runtime.adapter.generate_prepared_with_result(raw,runtime.generation_config)
                    out.append(list(g.raw_token_ids))
                finally:h.detach()
            assert out[0]==out[1]
        init={k:v.detach().cpu().clone() for k,v in expert.state_dict().items()};W0=cf.state_hash(expert)
        save(initpoint,dict(expert=init,binding=init_binding,state_hash=W0))
        write(d/'INITIALIZATION.json',dict(seconds=time.time()-began,W_init=W0,CP_count=1,transfer_checks='PASS',compact_bytes=initpoint.stat().st_size))
        del cp,expert,x
        # Registered W_init is the sole parent of every continuation branch.
        saved=torch.load(initpoint,map_location='cpu',weights_only=True);assert saved['binding']==init_binding;init=saved['expert']
    points={}
    for arm in ARMS if paired else ['SHARED_NO_H']:
        points[arm]=continuation(runtime,t,init,record,cfg,arm,mechanical)
    if paired:assert len({read(p.parent/'TRAINING.json')['binding']['W_init'] for p in points.values()})==1
    from scripts.medtrace.run_selective_write import save
    if not (d/'ROUTER.pt').exists():
        router=legacy.build_router(runtime,t,record);router['entries'][0]['label']=[];save(d/'ROUTER.pt',router)
    else:router=torch.load(d/'ROUTER.pt',map_location='cpu',weights_only=True)
    evaluate(runtime,ledger,bindings,legacy.query_ids(t),router['entries'],{t['edit_id']:points},1,'single',t['order'])
    if paired:evaluate_H(runtime,t,record,router['entries'],points,cfg)
    assert all(p._version==v and p.data_ptr()==ptr and p.requires_grad==req==False for p,v,ptr,req in frozen)
    assert set(runtime.get_module(LAYER)._forward_hooks)==hooks
    write(d/'COMPLETE.json',dict(status='GENERATED_NOT_SCORED',order=t['order'],arms=list(points),points={k:str(v) for k,v in points.items()},seconds=time.time()-began,init_once=True,Base_frozen=True,hook_lifecycle='PASS'))
    # CP/A2 intermediate states have no remaining consumers after compact transfer.
    deleted=[]
    for folder in ['initial','CP_W0']:
        for p in (d/folder).rglob('*.pt'):
            assert p.resolve().is_relative_to(d.resolve()) and not p.is_symlink();deleted.append(dict(path=str(p),bytes=p.stat().st_size));p.unlink()
    write(d/'TEMPORARY_CLEANUP.json',dict(items=deleted,compact_final_weights_retained=True))
    print('EDIT_COMPLETE',t['order'],list(points),flush=True)


def evaluate(runtime,ledger,bindings,ids,bank,points,prefix,mode,order,forced=None):
    import torch
    from methods.medtrace import AsymmetricCPExpert,MedTraceLayerHook
    from methods.medtrace.selective_write import LowRankExpert
    from m3bench_repro.editors.methods import BalanceEditPaperSpecEditor
    from m3bench_repro.editors.routing import MemoryRouter,decision_as_json,distances
    expert=LowRankExpert(AsymmetricCPExpert(14336,4096,4).to(runtime.device),20260912,rank=4).to(runtime.device).requires_grad_(False)
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
            destinations={arm:RUN/'private/outputs'/arm/mode/f'e{order:03d}'/folder/(digest(qid)+'.json') for arm in ARMS}
            old_outputs={a:read(p) for a,p in destinations.items() if p.exists()}
            for a,d in old_outputs.items():
                assert d['binding']['input']==q and d['binding']['phase']==phase and d['binding']['arm']==a
            if len(old_outputs)==4:continue
            query=replace(legacy.record_for(next(t for t in ledger['tasks'] if t['edit_id']==ledger['main_T0'][0])),record_id='query',question=q['question'],target='',official_rephrase='',image_path=Path(legacy.local_row(q)['image_path']))
            with torch.inference_mode():decision=editor._route(query)
            normal=decision_as_json(decision);on=decision.activated;selected=forced if forced else decision.logical_edit_id
            selected_on=bool(forced) or on
            alternatives=[]
            if not on:
                ds=distances(editor.router._key_matrix(runtime.device),captured['query'],'euclidean')
                alternatives=[dict(edit_id=e,distance=float(ds[i]),radius=editor.router.radii[i]) for i,e in enumerate(editor.router.logical_ids) if float(ds[i])<=editor.router.radii[i]]
            outputs={};actual_arms=ARMS if selected_on and selected in H else ['SHARED_NO_H']
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
                        for _ in range(3):
                            with torch.inference_mode(),hook.generation_request():g2=runtime.adapter.generate_prepared_with_result(raw,runtime.generation_config)
                            repeats.append(dict(raw_answer=g2.decoded_text,raw_token_ids=list(g2.raw_token_ids)))
                        assert all(x==out for x in repeats),'Common deterministic inference differed across four consumer checks'
                        common_checked=True
                else:out=dict(raw_answer=b['output']['model_answer_raw'],raw_token_ids=b['output']['raw_generated_token_ids'])
                outputs[a]=out
            for arm in ARMS:
                out=outputs[arm] if arm in outputs else outputs['SHARED_NO_H'];weight=(cp_bind[selected].get(arm) or cp_bind[selected].get('SHARED_NO_H')) if selected_on else None
                bind=dict(input=q,prefix=prefix,phase=phase,arm=arm,weight=weight,owner_order=order,selected_original=forced)
                dest=destinations[arm]
                if dest.exists():assert read(dest)['binding']==bind and read(dest)['R0']==out;continue
                write(dest,dict(binding=bind,Base_cache_id=q['opaque_Base_id'],R0=out,route=normal,effective_expert=selected if selected_on else None,other_own_radius_candidates=alternatives,
                    status='GENERATED_NOT_SCORED',execution_key=digest([bind,out,normal]),sharing='four bound consumers of common Base/nonH8 inference' if arm not in outputs else None))
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
    gpu=int(os.environ['GPU']);action=os.environ['ACTION'];assert gpu in (4,5)
    with legacy.lease(gpu):
        runtime,bindings=legacy.load(gpu);ledger=read(RUN/'private/legacy_stage17/COHORT_AND_SUPPORT_LEDGER.json');by={t['edit_id']:t for t in ledger['tasks']};tasks=[by[e] for e in ledger['main_T0']]
        cfg=dict(code_commit=read(RUN/'private/SOURCE_COMMIT.json')['commit'],execution_source=read(RUN/'private/GPU_SOURCE_VERSION.json'),runtime_lock=read(RUN/'private/cpu_gate/locks/CANONICAL_LLVAMED_RUNTIME_LOCK.json'),campaign_epoch=read(RUN/'RUN_MANIFEST.json')['starting_epoch'],train_seconds=48*3600)
        if action in ('p1','mechanical'):
            selected=[t for t in tasks if t['edit_id'] in H];part=int(os.environ['PARTITION'])
            if action=='mechanical':selected=[t for t in selected if t['order'] in (31,35)]
            for t in selected[part::2]:train_edit(runtime,bindings,ledger,t,cfg,paired=True,mechanical=t['order'] in (31,35))
            write(RUN/f'private/{action.upper()}_PART_{part}_COMPLETE.json',dict(status='GENERATED_NOT_SCORED',edits=len(selected[part::2]),branches=len(selected[part::2])*4))
        elif action=='shared':
            selected=[t for t in tasks if t['edit_id'] not in H];part=int(os.environ['PARTITION'])
            for t in selected[part::2]:train_edit(runtime,bindings,ledger,t,cfg)
            write(RUN/f'private/SHARED_PART_{part}_COMPLETE.json',dict(status='GENERATED_NOT_SCORED',edits=len(selected[part::2])))
        elif action=='bank':
            bank=[];points={}
            for i,t in enumerate(tasks,1):
                d=RUN/'private/edits'/f"e{t['order']:03d}";done=read(d/'COMPLETE.json');points[t['edit_id']]={a:Path(p) for a,p in done['points'].items()}
                state=__import__('torch').load(d/'ROUTER.pt',map_location='cpu',weights_only=True);bank.extend(state['entries'])
                evaluate(runtime,ledger,bindings,[t['edit_id']],bank,points,i,'native',i)
                if i in (1,50,100,146):
                    ids=list(dict.fromkeys(q for x in tasks[:i] for q in legacy.query_ids(x)));evaluate(runtime,ledger,bindings,ids,bank,points,i,'sequential',i)
                write(RUN/'public/PROGRESS.json',dict(status='P2_BANK_RUNNING',completed=i,N=146))
            write(RUN/'private/FULL_BANK_GENERATION_COMPLETE.json',dict(status='GENERATED_NOT_SCORED',N=146,arms=list(ARMS)))
        elif action=='oracle':
            bank=[];points={}
            for t in tasks:
                d=RUN/'private/edits'/f"e{t['order']:03d}";done=read(d/'COMPLETE.json');points[t['edit_id']]={a:Path(p) for a,p in done['points'].items()};bank.extend(__import__('torch').load(d/'ROUTER.pt',map_location='cpu',weights_only=True)['entries'])
            wanted=defaultdict(set)
            for x in read(RUN/'private/FROZEN_ORACLE_PRIORITY.json')['items']:wanted[x['edit_id']].add(x['query_id'])
            for t in tasks:
                for ev in t['events']:
                    for qid in ev['all_probe_query_ids']:
                        files=[RUN/'private/outputs'/a/'sequential/e146/panel'/(digest(qid)+'.json') for a in ARMS];outs=[read(f) for f in files]
                        changed=len({digest(o['R0']) for o in outs})>1
                        if changed and ((t['edit_id'] in H and ev['task']=='T2G') or outs[0]['effective_expert'] in H):wanted[t['edit_id']].add(qid)
            write(RUN/'private/ORACLE_SELECTION.json',dict(items={k:sorted(v) for k,v in wanted.items()},source='frozen historical priorities plus authorized new H8 output-change diagnoses'))
            for t in tasks:
                if wanted[t['edit_id']]:evaluate(runtime,ledger,bindings,sorted(wanted[t['edit_id']]),bank,points,146,'oracle',t['order'],forced=t['edit_id'])
            write(RUN/'private/ORACLE_GENERATION_COMPLETE.json',dict(status='GENERATED_NOT_SCORED',diagnostic_queries=sum(map(len,wanted.values())),deployed_method=False))
        else:raise ValueError(action)


if __name__=='__main__':
    def stop(signum,frame):raise InterruptedError('Bounded own worker interrupted; accepted checkpoints preserved')
    signal.signal(signal.SIGTERM,stop)
    try:main()
    except Exception as e:
        write(RUN/'private'/f"FAILURE_{os.environ.get('ACTION')}_{os.environ.get('GPU')}.json",dict(error=repr(e),traceback=traceback.format_exc(),epoch=time.time(),pid=os.getpid()));raise
