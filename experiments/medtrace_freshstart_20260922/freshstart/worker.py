"""One real updater and event journal for canary and fresh formal experiments."""
from collections import defaultdict
import copy
from dataclasses import asdict
import json
from pathlib import Path
import random
import subprocess
import time

import torch
from freshstart.runtime import ROOT,LAYER,read,write,check_budget
from freshstart.pipeline import record,input_key,key_for,score_request,initialize
from freshstart.plan import compile_plan
from methods.medtrace import AsymmetricCPExpert,MedTraceLayerHook
from methods.medtrace.selective_write import LowRankExpert,optimizer_for,full_vocab_kl,balanced_schedule
from m3bench_repro.editors.routing import MemoryRouter
from m3bench_repro.editors.llava_runtime import seed_everything
from scripts.medtrace import stage15
from scripts.medtrace.stage17_prepare import digest
from scripts.medtrace.stage18_cfact import state_hash,source_batch,backward_term,rng_state,assert_base_off,check_cache
from scripts.medtrace.run_selective_write import save,teacher_batch


class InjectedCrash(RuntimeError):pass


def versions(runtime):
    return [(p.data_ptr(),p._version,p.requires_grad) for p in runtime.model.parameters()]


def teacher(runtime,row):
    assert_base_off(runtime)
    bases=read(ROOT/'private/preparation/BASE_OUTPUTS.json')
    base=bases[digest([input_key(row),row['reference']])]['output']
    kwargs,labels,mask,inputs=teacher_batch(runtime,dict(row,eqkey=input_key(row)),base['raw_token_ids'])
    binding=dict(input=inputs,source_input=input_key(row),generation=runtime.generation_config,
        model_revision='91bb16c122001ddc9cf1fd36ce1dae09448943a2',vision_revision='ce19dc912ca5cd21c8a653c79e251e808ccabcd1',
        runtime_commit=subprocess.check_output(['git','-C',str(ROOT/'source'),'rev-parse','HEAD'],text=True).strip(),
        backend='official_native_torch2.6.0_transformers4.51.3',tokenizer=type(runtime.adapter.tokenizer).__name__,
        teacher='FRESH_BASE_OFF_FULL_VOCAB_FP32',direction='Base||student',temperature=1.)
    key=digest(binding);path=ROOT/'private/fresh_teachers'/f'{key}.pt'
    if path.exists():cached=torch.load(path,map_location='cpu',weights_only=True)
    else:
        with torch.no_grad():logp=runtime.model(**kwargs).logits[mask].float().log_softmax(-1).cpu()
        cached=dict(binding=binding,logp=logp);save(path,cached)
    logp=check_cache(cached,binding)
    if logp.shape!=(int(mask.sum()),runtime.model.config.vocab_size):raise ValueError('Teacher shape mismatch')
    return kwargs,labels,mask,logp,key


def update(runtime,hook,expert,optimizer,native,fit,h_batch,u_teacher,h_weight,base_versions):
    optimizer.zero_grad(set_to_none=True);terms={}
    for name,batch,weight in [('native',native,.5),('fit',fit,.5)]+([('H',h_batch,h_weight)] if h_batch is not None and h_weight else []):
        hook.set_teacher_routing(batch.labels)
        terms[name]=backward_term(expert,runtime.compute_loss(batch),weight,len(batch.target_token_ids),name)
    if u_teacher is not None:
        kwargs,labels,mask,logp,key=u_teacher
        hook.set_teacher_routing(labels)
        terms['U']=backward_term(expert,full_vocab_kl(runtime.model(**kwargs).logits[mask],logp),.01,int(mask.sum()),key)
    # Masked U has no teacher lookup, forward, or gradient contribution.
    grad=torch.nn.utils.clip_grad_norm_(expert.parameters(),1.)
    if not torch.isfinite(grad) or any(p.grad is None for p in expert.parameters()):raise FloatingPointError('Invalid gradient')
    optimizer.step();expert.normalize_factors_(verify_dense=False)
    if any(not torch.isfinite(p).all() for p in expert.parameters()):raise FloatingPointError('Nonfinite expert')
    if versions(runtime)!=base_versions:raise RuntimeError('Frozen Base parameter changed')
    return dict(terms=terms,gradient_norm=float(grad))


def role_schedule(rows,steps,seed):
    if not rows:return [None]*steps
    groups=defaultdict(list)
    for i,row in enumerate(rows):groups[row['source_group']].append(dict(index=i))
    return [x['index'] for x in balanced_schedule(dict(sorted(groups.items())),steps,seed)]


def event(runtime,run,task,expert,arm,prefix,event_id,h_rows,u_rows,steps,plan_hash,*,p0=False,fail_at=None,masked_u=0):
    folder=run/'private/events'/digest(event_id);folder.mkdir(parents=True,exist_ok=True)
    before=state_hash(expert)
    binding=dict(run=str(run.resolve()),arm=arm,prefix=prefix,event_id=event_id,plan_hash=plan_hash,
        task=task,steps=steps,H=h_rows,U=u_rows,masked_u=masked_u,layer=LAYER,
        code=subprocess.check_output(['git','-C',str(ROOT/'source'),'rev-parse','HEAD'],text=True).strip())
    latest,final,journal=folder/'active.pt',folder/'version.pt',folder/'EVENT.json'
    if final.exists():
        state=torch.load(final,map_location=runtime.device,weights_only=True)
        if state['binding']!=binding:raise ValueError('Cross-run/arm/code resume forbidden')
        expert.load_state_dict(state['expert'])
        expert.requires_grad_(False)
        if state_hash(expert)!=state['parameter_hash']:raise ValueError('Immutable version mismatch')
        write(journal,dict(state='WRITTEN',binding=binding,parameter_hash=state['parameter_hash'],actual_steps=steps,recovered_commit=True))
        if fail_at=='COMMIT':raise InjectedCrash('AFTER_COMMIT')
        return state
    seed_everything(task['seed']);expert.requires_grad_(True);opt=optimizer_for(expert,runtime.model)
    if latest.exists():
        state=torch.load(latest,map_location=runtime.device,weights_only=True)
        if state['binding']!=binding:raise ValueError('Active recovery binding mismatch')
        expert.load_state_dict(state['expert']);opt.load_state_dict(state['optimizer'])
        torch.set_rng_state(state['torch_rng'].cpu());torch.cuda.set_rng_state(state['cuda_rng'].cpu());random.setstate(state['python_rng'])
        cursor,curve=state['step'],state['curve'];before=state['before']
    else:
        cursor,curve=0,[]
        save(latest,dict(binding=binding,expert=expert.state_dict(),optimizer=opt.state_dict(),step=0,curve=[],before=before,**rng_state()))
    write(journal,dict(state='PREPARED',binding=binding,before_hash=before,step=cursor))
    if fail_at=='PREPARED':raise InjectedCrash('AFTER_PREPARED_BEFORE_UPDATE')
    rec=record(task);native=runtime.build_edit_batch(rec)
    from dataclasses import replace
    fits=[runtime.build_edit_batch(replace(rec,question=q)) for q in task['fit_questions']]
    if len(fits)!=4:raise ValueError('Exactly four fit wrappers required')
    hs=[source_batch(runtime,rec,row) for row in h_rows]
    us=[teacher(runtime,row) for row in u_rows]
    h_order=role_schedule(h_rows,steps,task['seed']);u_order=role_schedule(u_rows,steps,task['seed'])
    fit_order=list(range(4));random.Random(task['seed']).shuffle(fit_order)
    guard=versions(runtime);hook=MedTraceLayerHook(runtime.get_module(LAYER),expert);hook.attach();started=time.time()
    try:
        for step in range(cursor,steps):
            check_budget(training=True,p0=p0)
            values=update(runtime,hook,expert,opt,native,fits[fit_order[step%4]],
                hs[h_order[step]] if h_order[step] is not None else None,
                us[u_order[step]] if u_order[step] is not None else None,0. if arm=='F0' else .25,guard)
            curve.append(dict(step=step+1,fit_index=fit_order[step%4],**values))
            if (step+1)%20==0 or step+1==steps:
                save(latest,dict(binding=binding,expert=expert.state_dict(),optimizer=opt.state_dict(),step=step+1,
                    curve=curve,before=before,**rng_state()))
                write(run/'STATUS.json',dict(arm=arm,prefix=prefix,event_id=event_id,step=step+1,epoch=time.time(),
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),scoring_pending=len(list((ROOT/'private/judge/pending').glob('*.json')))-len(list((ROOT/'private/judge/scores').glob('*.json')))))
                print('UPDATE',arm,prefix,step+1,'of',steps,flush=True)
    finally:hook.detach();expert.requires_grad_(False)
    if not runtime.base_guard.verify()['unchanged']:raise RuntimeError('Base guard changed')
    result=dict(binding=binding,expert=expert.state_dict(),parameter_hash=state_hash(expert),before_hash=before,
        curve=curve,actual_steps=len(curve),seconds=time.time()-started,teacher_keys=[x[-1] for x in us],masked_u=masked_u,
        optimizer=opt.state_dict(),**rng_state())
    if len(curve)!=steps:raise ValueError('Actual optimizer count mismatch')
    save(final,result)
    restored=torch.load(final,map_location=runtime.device,weights_only=True);copy_expert=copy.deepcopy(expert);copy_expert.load_state_dict(restored['expert'])
    if state_hash(copy_expert)!=result['parameter_hash']:raise ValueError('Event real save/reload failed')
    if fail_at=='WRITE':raise InjectedCrash('AFTER_VERSION_BEFORE_COMMIT')
    write(journal,dict(state='WRITTEN',binding=binding,parameter_hash=result['parameter_hash'],actual_steps=steps))
    if fail_at=='COMMIT':raise InjectedCrash('AFTER_COMMIT')
    return result


def observe(runtime,run,bank,router,task,row,arm,prefix,event_id):
    fp=input_key(row);key=key_for(runtime,record(dict(task,native=row)))
    route=asdict(router.route(key));selected=route['logical_edit_id'];expert=bank.get(selected)
    if selected and expert is None:raise ValueError('Router points to absent expert')
    raw,_,binding=stage15.prepared(runtime,row,record(task))
    version=state_hash(expert) if expert is not None else None
    execution=digest(dict(base='91bb16c122001ddc9cf1fd36ce1dae09448943a2',expert=version,input=binding,
        tokenizer=type(runtime.adapter.tokenizer).__name__,backend='official_native_torch2.6.0',route_on=bool(expert)))
    hook=MedTraceLayerHook(runtime.get_module(LAYER),expert) if expert is not None else None
    if hook:hook.attach()
    try:output=stage15.generate(runtime,raw,binding,hook)
    finally:
        if hook:hook.detach()
    state_path=ROOT/'private/execution_states'/f'{execution}.json'
    if state_path.exists() and read(state_path)['raw_token_ids']!=output['raw_token_ids']:raise RuntimeError('Same execution state produced different tokens')
    if not state_path.exists():write(state_path,output)
    consumer=digest([str(run),arm,prefix,event_id,fp]);score=score_request(row,output,consumer)
    if row['role'].startswith('U'):
        base=read(ROOT/'private/preparation/BASE_OUTPUTS.json')[digest([fp,row['reference']])]['output']
        teacher_score=score_request(dict(row,reference=base['raw_answer']),output,consumer+'/teacher','TEACHER_AGREEMENT')
        score['teacher_judge_key']=teacher_score['judge_key']
    write(run/'private/consumers'/f'{consumer}.json',dict(arm=arm,prefix=prefix,event=event_id,row=row,route=route,
        execution_state_id=execution,parameter_hash=version,output=output,**score))


def run_arm(runtime,run,tasks,plan,arm,seed,*,p0=False):
    n=len(tasks);bank={};router=MemoryRouter('euclidean')
    entries={e['logical_edit_id']:e for e in torch.load(ROOT/'private/preparation/router.pt',weights_only=True)['entries']}
    byid={t['canonical_edit_id']:t for t in tasks}
    evaluation={p['training']['canonical_edit_id']:p for p in read(ROOT/'private/FRESH_EVALUATION.json')['candidate_packages']}
    completed=[];faults=[];costs=[]
    for prefix,task in enumerate(tasks,1):
        prefix_started=time.time()
        check_budget(training=True,p0=p0)
        initial=run/'private/edits'/f'e{task["order"]:03d}'/'SHARED_W0.pt'
        if initial.exists():
            expert=LowRankExpert(AsymmetricCPExpert(14336,4096,4).to(runtime.device),task['seed'],rank=4).to(runtime.device)
            saved=torch.load(initial,map_location=runtime.device,weights_only=True)
            if saved['binding']['run']!=str(run) or saved['binding']['seed']!=seed:raise ValueError('Foreign W0 forbidden')
            expert.load_state_dict(saved['expert'])
            if state_hash(expert)!=saved['binding']['parameter_hash']:raise ValueError('W0 parameter binding mismatch')
        else:expert=initialize(runtime,task,run,seed,p0=p0)
        sid=lambda role,row:digest([role,row])
        roles=plan['roles'][str(prefix)]
        u=[r for r in task['U_fit'] if roles[sid('U',r)]['state']=='active']
        h=task['H_fit'] if arm!='F0' else []
        eid=f'{arm}/p{prefix}/writer';kwargs=dict(p0=p0,masked_u=len(task['U_fit'])-len(u))
        untouched={k:state_hash(v) for k,v in bank.items()}
        if p0 and arm=='F0' and prefix==1:
            for boundary in ['PREPARED','WRITE','COMMIT']:
                try:event(runtime,run,task,expert,arm,prefix,eid,h,u,320,plan['plan_hash'],fail_at=boundary,**kwargs)
                except InjectedCrash as error:faults.append(str(error))
                else:raise RuntimeError('Failure injection was not reached')
        result=event(runtime,run,task,expert,arm,prefix,eid,h,u,320,plan['plan_hash'],**kwargs)
        if result['actual_steps']!=320:raise ValueError('Incomplete writer')
        if untouched!={k:state_hash(v) for k,v in bank.items()}:raise ValueError('Non-target expert changed')
        bank[task['canonical_edit_id']]=expert
        e=entries[task['canonical_edit_id']];router.add(e['logical_edit_id'],e['key'].to(runtime.device),e['radius'])
        observe(runtime,run,bank,router,task,task['native'],arm,prefix,eid)
        expected=[s for s in plan['slots'] if s['prefix']==prefix] if arm in ['FR','FA'] else []
        for slot in expected:
            old=byid[slot['expert']];eid=arm+'/'+slot['slot_key'];rows=slot[arm]
            observe(runtime,run,bank,router,old,old['native'],arm,prefix,eid+'/before')
            h=[plan['supports'][s]['row'] for s in rows['H']];u=[plan['supports'][s]['row'] for s in rows['U']]
            unaffected={k:state_hash(v) for k,v in bank.items() if k!=slot['expert']}
            event(runtime,run,old,bank[slot['expert']],arm,prefix,eid,h,u,slot['steps'],plan['plan_hash'],p0=p0)
            if unaffected!={k:state_hash(v) for k,v in bank.items() if k!=slot['expert']}:raise ValueError('Maintenance wrote another expert')
            observe(runtime,run,bank,router,old,old['native'],arm,prefix,eid+'/after');completed.append(eid)
        if set(x for x in completed if x.startswith(arm+'/'+plan['plan_hash']+'/'+str(prefix)+'/'))!={arm+'/'+s['slot_key'] for s in expected}:
            raise ValueError('Prefix planned/completed slot mismatch')
        if not p0 and prefix in {5,19,32,45,n}:
            for i,t in enumerate(tasks[:prefix]):
                for row in [t['native']]+evaluation[t['canonical_edit_id']]['evaluation']:
                    observe(runtime,run,bank,router,t,row,arm,prefix,f'panel/{i}/{row["role"]}')
                if prefix in {19,45,n}:
                    for row in t['H_fit']+t['U_fit']:
                        observe(runtime,run,bank,router,t,row,arm,prefix,f'training_diagnostic/{i}/{row["role"]}')
        save(run/'private/banks'/f'{arm}-p{prefix}.pt',dict(run=str(run),arm=arm,prefix=prefix,router=router.export_state(),
            experts={k:v.state_dict() for k,v in bank.items()},hashes={k:state_hash(v) for k,v in bank.items()}))
        if prefix in {19,45,n}:
            restored=torch.load(run/'private/banks'/f'{arm}-p{prefix}.pt',map_location=runtime.device,weights_only=True)
            if restored['run']!=str(run) or restored['arm']!=arm:raise ValueError('Wrong bank restore')
            for name,parameters in restored['experts'].items():
                clone=copy.deepcopy(bank[name]);clone.load_state_dict(parameters)
                if state_hash(clone)!=restored['hashes'][name]:raise ValueError('Bank reload parameter hash mismatch')
                del clone
            restored_router=MemoryRouter.from_state(restored['router'],device=runtime.device)
            if asdict(restored_router.route(router.keys[-1]))!=asdict(router.route(router.keys[-1])):raise ValueError('Bank router reload mismatch')
            write(run/'private/banks'/f'{arm}-p{prefix}-RELOAD.json',dict(status='PASS',experts=len(bank),prefix=prefix,parameter_hashes=restored['hashes']))
        write(run/f'{arm}_STATUS.json',dict(status='RUNNING',prefix=prefix,N=n,completed_slots=completed,fault_injections=faults))
        costs.append(time.time()-prefix_started)
        if prefix==3:
            write(run/f'COST_SAMPLE_{arm}.json',dict(first_three_prefix_seconds=costs,mean_seconds=sum(costs)/3,
                includes='initialization when not already shared, teachers, writer, native generation, planned maintenance',
                reserve_factor=1.3,estimated_remaining_arm_seconds=(n-3)*sum(costs)/3*1.3,
                full_panel_generation_not_in_first_three=True))
    write(run/f'{arm}_STATUS.json',dict(status='TRAINING_AND_GENERATION_COMPLETE_SCORING_PENDING',prefix=n,N=n,completed_slots=completed,
        fault_injections=faults,missing_independent_panels=['heldout_rephrase','positive_image'],base_unchanged=runtime.base_guard.verify()['unchanged']))
    return dict(arm=arm,N=n,slots=len(completed),fault_injections=faults)
