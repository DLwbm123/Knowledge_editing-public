"""One full-TT protected RAW160 arm with source-only feasibility checks."""
import os
import time
import traceback
from pathlib import Path
import torch
import optimizer160 as fit
import damage as data
import protection_math as pm
q,c,p,RUN,old=fit.q,fit.c,fit.p,fit.RUN,fit.old
PARENT=Path(os.environ['DAMAGE_PARENT'])
REFRESH=(1,41,81,121)


def raw_path(t):return RUN/'private/raw_reference'/f"{t['order']}.pt"
def point(t):return RUN/'private/protected_candidates'/f"{t['order']}.pt"
def weights(state):return {k:v.detach().cpu().clone() for k,v in state.items()}


def scale_for(expert,opt):
    groups={id(v):g['lr'] for g in opt.param_groups for v in g['params']}
    params=dict(expert.named_parameters())
    assert set(params)==set(q.KEYS) and set(groups.values())=={1e-4,1e-3}
    return torch.cat([torch.full((params[k].numel(),),groups[id(params[k])],dtype=torch.float64) for k in q.KEYS])


def plan():
    pm.selfcheck();old.finite_math.selfcheck();fit.selfcheck()
    assert c.read(PARENT/'public/REVIEW_AUDIT.json')['status']=='PASS'
    assert c.read(PARENT/'public/RESULTS.json')['decision']=='QUALIFIED_UNPROTECTED_REFERENCE_AUDITED'
    assert c.read(RUN/'public/ATTRIBUTION.json')['training_geometry_uses_these_errors'] is False
    basis,held=q.split();tasks=q.d.selected();assert len(tasks)==8 and len(basis)==61 and len(held)==96
    assert not {r['source_group'] for r in basis}&{r['source_group'] for r in held}
    baseline=c.read(Path(os.environ['FP32_BASELINE_PARENT'])/'private/FP32_SEMANTIC_QUALIFICATION.json')['rows']
    assert sum(r['correct'] for r in baseline if r['role']=='BASIS')==61
    c.write(RUN/'private/HELD_BASE.json',c.read(PARENT/'private/HELD_BASE.json'))
    c.write(RUN/'private/OPTIMIZER160_LOCK.json',c.read(PARENT/'private/OPTIMIZER160_LOCK.json'))
    oldresult=c.read(PARENT/'public/RESULTS.json')
    lock=dict(arms=['PROTECTED'],reference_arms=['RAW','ADAM'],experts=8,steps=160,trainable_parameters=7168,
        initial_state='ORIGINAL_SEEDED_ZERO_TT88',source_loss='.5 native + .5 cyclicFIT',source_fraction=.9,
        geometry='full7168 optimizer-coordinate Fisher trust ball',Fisher_refresh=REFRESH,Fisher_questions=61,Fisher_probes=32,Fisher_rows=1952,
        Fisher_target='original fixed Base response prefixes; local current-state Fisher',Fisher_seed_rule='20261009+original_BASIS_question_index',
        ridge_relative=pm.RIDGE,rank_truncation=False,trust_radius='norm of own current RAW proposal in fixed learning-rate coordinates',
        step_restoration_bisections=16,endpoint_restoration_bisections=16,endpoint_fraction=.9,
        raw_rebuild_updates=1280,protected_updates=1280,total_updates=2560,maximum_backwards=73984,
        maximum_non_generation_forwards=138108,maximum_forwards=138108+963*1024,
        new_generations=963,raw_rebuild_generations=40,protected_source_generations=40,protected_event_generations=115,protected_held_generations=768,
        maximum_new_Judge=923,held_prefix_forwards=1344,source_maximum_non_generation_forwards=136764,
        generation='MATH_FP32_FROZEN_FP16_PREFILL_GREEDY_1024',source_rebuild_gate='40RAW exact accepted input/text/tokens',
        baseline_T2G_by_owner=[next(v['candidate_correct'] for v in oldresult['per_expert'] if v['arm']=='RAW' and v['role']=='T2G' and v['owner']==i) for i in range(1,9)],
        source_semantic_gate='native8/8,FIT32/32,T1G32/32,T2G no owner below fixed RAW reference',
        held_evaluated_for_all_frozen_candidates=True,all_candidates_frozen_before_eval=True,heldout_geometry=False,
        primary='original63; current FP32correct62 and all96 also fixed',selection=False,automatic_extension=False,
        BASIS_binding=c.digest(basis),HELD_binding=c.digest(held),epoch=time.time())
    c.write(RUN/'private/PROTECTION_LOCK.json',lock);c.write(RUN/'public/ADMISSION.json',dict(status='PASS',**lock,solver_selfcheck=pm.selfcheck()))
    p.done('PROTECTION_PLAN_COMPLETE')


def retain_raw(t,arm,expert,base):
    assert arm=='RAW' and not raw_path(t).exists()
    c.save(raw_path(t),dict(BASE=weights(base),RAW=weights(expert.state_dict()),source_lock=c.digest(c.read(RUN/'private/OPTIMIZER160_LOCK.json'))))


def rebuild():
    fit.jobs=lambda:[(t,'RAW') for t in q.d.selected()]
    fit.worker(retain_raw)


def raw_barrier():
    paths=list((RUN/'private/outputs/RAW').glob('*/*.json'));assert len(paths)==40
    for path in paths:
        a=c.read(path);b=c.read(PARENT/'private/outputs/RAW'/path.relative_to(RUN/'private/outputs/RAW'))
        assert a['binding']==b['binding'] and a['R0']==b['R0'],'RAW reconstruction drift'
    counts=[c.read(RUN/'private'/f'OPTIMIZER160_COUNTS_{i}.json') for i in range(6)]
    assert sum(v['updates'] for v in counts)==1280 and sum(v['backwards'] for v in counts)==2560
    assert sum(v['forwards'] for v in counts)==3046+sum(c.read(x)['forwards'] for x in paths)
    c.write(RUN/'private/RAW_REFERENCE_READY.json',dict(status='PASS',source_exact=40,counts=counts,epoch=time.time()))


def measure(runtime,hook,expert,state,batches,refs,full=False):
    expert.load_state_dict(state);values=[];native=None
    for i,(b,(ref,target)) in enumerate(zip(batches,refs)):
        logits,y=q.logits(runtime,hook,b);assert torch.equal(y,target)
        value=q.compare(ref,logits,target)
        if full:value['token_diagnostics']=fit.token_stats(logits,target)
        values.append(value)
        if i==0:native=(logits,target)
    return old.progress(values),values,native


def source_gradient(runtime,hook,expert,batches,refs,counts):
    params=[dict(expert.named_parameters())[k] for k in q.KEYS];gradient=torch.zeros(7168,dtype=torch.float64);values=[]
    for i,(batch,(ref,target)) in enumerate(zip(batches,refs)):
        hook.set_teacher_routing(batch.labels);output=runtime.model(**batch.forward_kwargs())
        mask=batch.labels[:,1:]!=-100;logits=output.logits[:,:-1][mask]
        labels=batch.labels[:,1:][mask]
        loss=torch.nn.functional.cross_entropy(logits.double(),labels)
        assert torch.isfinite(loss)
        weight=.5 if i==0 else .125
        g=torch.autograd.grad(-weight*loss,params);counts['backwards']+=1
        gradient+=torch.cat([v.detach().cpu().double().flatten() for v in g])
        values.append(q.compare(ref,logits.detach().cpu().double(),target))
        del output,logits,loss,g
    return gradient,old.progress(values)


def emit(runtime,hook,t,row,raw,expanded,binding,counts,role,index,baseline):
    dest=RUN/'private/outputs/PROTECTED'/str(t['order'])/(row['query_id']+'.json');assert not dest.exists(),'No implicit generation retry'
    c.budget();start=counts['forwards'];answer,trace=fit.short.generate(runtime,hook,raw,expanded);counts['generations']+=1
    c.write(dest,dict(arm='PROTECTED',expert_order=t['order'],role=role,index=index,
        original_primary=row.get('original_primary',False),same_reference=row.get('same_reference',False),
        binding=dict(input=dict(row,image_sha256=raw['image_sha256']),judge_input=binding),
        R0=dict(raw_answer=answer.decoded_text,raw_token_ids=list(answer.raw_token_ids)),baseline_path=str(baseline),
        generation_trace=trace,forwards=counts['forwards']-start,EOS=answer.raw_token_ids[-1]==runtime.adapter.tokenizer.eos_token_id,
        at_cap=len(answer.raw_token_ids)>=1024,lock=c.digest(c.read(RUN/'private/PROTECTION_LOCK.json')),execution=c.read(RUN/'private/GPU_SOURCE_VERSION.json')))


def protected_worker():
    from m3bench_repro.editors.llava_runtime import seed_everything
    assert c.read(RUN/'private/RAW_REFERENCE_READY.json')['status']=='PASS'
    part=int(os.environ['PARTITION']);gpu=q.GPUS[part];old.calibration.require_memory(gpu)
    counts=dict(forwards=0,backwards=0,updates=0,generations=0);lock=c.read(RUN/'private/PROTECTION_LOCK.json')
    with p.lease(gpu):
        runtime,_=c.load(gpu);tasks=q.d.selected()[part::6];basis=q.split()[0]
        frozen=fit.short.prepare(runtime,tasks)
        frozen_basis={r['query_id']:old.cpu(q.protection_batch(runtime,r,tasks[0])) for r in basis}
        old.fp32(runtime);q.protection_batch=lambda runtime,row,task:old.batch_on(runtime,frozen_basis[row['query_id']])
        handle=runtime.model.register_forward_pre_hook(lambda *_:counts.__setitem__('forwards',counts['forwards']+1))
        try:
            for t in tasks:
                assert not point(t).exists(),'No implicit trajectory restart'
                seed_everything(t['seed']);saved=torch.load(raw_path(t),map_location='cpu',weights_only=True)
                expert=p.expert(saved['BASE'],t['seed'],runtime.device);base=old.snapshot(expert);raw_endpoint={k:v.to(runtime.device) for k,v in saved['RAW'].items()}
                hook=q.scoped.replay.MedTraceLayerHook(runtime.get_module(c.LAYER),expert);hook.attach()
                resume=RUN/'private/active'/f"{t['order']}_resume.pt";metric_file=RUN/'private/active'/f"{t['order']}_geometry.pt"
                start=dict(counts);curve=[];refreshes=[];diagnostics=[]
                try:
                    batches=[old.batch_on(runtime,x[1]) for x in frozen[t['order']]]
                    refs=[q.logits(runtime,hook,b,verify_base=i==0) for i,b in enumerate(batches)]
                    opt=p.optimizer(expert,runtime.model);scale=scale_for(expert,opt);params=dict(expert.named_parameters())
                    for step in range(1,161):
                        c.budget();before=old.snapshot(expert)
                        if step in REFRESH:
                            matrix,backwards,audit=old.prior.basis_gradients(runtime,hook,basis,t,params);counts['backwards']+=backwards
                            geometry=pm.Geometry(matrix*scale);del matrix
                            refreshes.append(dict(step=step,**audit,geometry=geometry.audit))
                            c.save(metric_file,dict(step=step,v=geometry.v,values=geometry.values,audit=geometry.audit,lock=c.digest(lock)))
                        opt.zero_grad(set_to_none=True);losses={}
                        for i in (0,1+(step-1)%4):
                            hook.set_teacher_routing(batches[i].labels);loss=runtime.compute_loss(batches[i]);assert torch.isfinite(loss)
                            losses[str(i)]=float(loss.detach());(.5*loss).backward();counts['backwards']+=1
                        norm=torch.nn.utils.clip_grad_norm_(expert.parameters(),1.);assert torch.isfinite(norm)
                        assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in expert.parameters())
                        opt.step();groups=q.d.apply_direction(opt,{id(v):before[k] for k,v in expert.named_parameters()},'RAW')
                        proposed=old.snapshot(expert);expert.load_state_dict(before)
                        gradient,current_gain=source_gradient(runtime,hook,expert,batches,refs,counts)
                        raw_gain,raw_values,_=measure(runtime,hook,expert,proposed,batches,refs)
                        raw_delta=q.flatten(proposed)-q.flatten(before);z=raw_delta/scale;g=gradient*scale
                        linear=float(g@z);actual=raw_gain-current_gain;restoration=[];alpha=1.;solver=None
                        if linear>0 and actual>1e-8 and z.norm()>0:
                            value,solver=geometry.solve(g,z);initial=q.add(before,scale*value)
                            gain,_,_=measure(runtime,hook,expert,initial,batches,refs)
                            def state(a):return proposed if a==1. else initial if a==0. else q.add(initial,a*(q.flatten(proposed)-q.flatten(initial)))
                            def evaluate(a):return measure(runtime,hook,expert,state(a),batches,refs)[0]-current_gain
                            alpha,restoration=old.finite_math.restore(evaluate,gain-current_gain,actual)
                            chosen=state(alpha);accepted,values,_=measure(runtime,hook,expert,chosen,batches,refs)
                            assert accepted-current_gain>=.9*actual,'Actual source-step feasibility failed'
                            mode='PROTECTED_OR_RESTORED'
                        else:
                            chosen=proposed;accepted=raw_gain;values=raw_values;mode='RAW_NONPOSITIVE_GAIN_FALLBACK'
                        expert.load_state_dict(chosen);counts['updates']+=1
                        dz=(q.flatten(chosen)-q.flatten(before))/scale
                        roundoff=8*torch.finfo(torch.float32).eps*float((q.flatten(before)/scale).norm())
                        assert dz.norm()<=z.norm()+roundoff+1e-10
                        curve.append(dict(step=step,mode=mode,pre_update_loss=losses,groups=groups,solver=solver,
                            current_gain=current_gain,RAW_actual_step_gain=actual,accepted_actual_step_gain=accepted-current_gain,
                            restoration_alpha=alpha,restoration_trace=restoration,coordinate_step_norm=float(dz.norm()),RAW_coordinate_step_norm=float(z.norm()),roundoff_bound=roundoff,
                            core_update_norms={k:float((chosen[k]-before[k]).norm()) for k in q.KEYS}))
                        if step==1:mechanical=fit.short.mechanical(runtime,hook,batches[0])
                        if step in fit.NODES:
                            _,values,_=measure(runtime,hook,expert,chosen,batches,refs,full=True);diagnostics.append(dict(step=step,values=values))
                            c.write(RUN/'private/progress/PROTECTED'/f"{t['order']}.json",dict(step=step,curve=curve,diagnostics=diagnostics,refreshes=refreshes))
                            print('PROTECTION_TRAIN',t['order'],step,accepted,mode,alpha,flush=True)
                        if step%20==0:
                            c.save(resume,dict(completed_step=step,state=weights(chosen),optimizer=opt.state_dict(),counts=dict(counts),curve=curve,lock=c.digest(lock)))
                    final=old.snapshot(expert);raw_norm=float(((q.flatten(raw_endpoint)-q.flatten(base))/scale).norm())
                    final_norm=float(((q.flatten(final)-q.flatten(base))/scale).norm());radial=min(1.,raw_norm/final_norm) if final_norm else 1.
                    if radial<1.:final=q.add(base,radial*(q.flatten(final)-q.flatten(base)))
                    gain,_,_=measure(runtime,hook,expert,final,batches,refs)
                    # RAW endpoint gain was measured during the independently reproduced trajectory.
                    raw_gain=c.read(RUN/'private/results/RAW'/f"{t['order']}.json")['edit_gain'];assert raw_gain>1e-8
                    def endpoint(a):return raw_endpoint if a==1. else final if a==0. else q.add(final,a*(q.flatten(raw_endpoint)-q.flatten(final)))
                    alpha,trace=old.finite_math.restore(lambda a:measure(runtime,hook,expert,endpoint(a),batches,refs)[0],gain,raw_gain)
                    chosen=endpoint(alpha);gain,values,native=measure(runtime,hook,expert,chosen,batches,refs,full=True)
                    assert gain>=.9*raw_gain
                    actual_norm=float(((q.flatten(chosen)-q.flatten(base))/scale).norm())
                    assert actual_norm<=raw_norm+8*torch.finfo(torch.float32).eps*float((q.flatten(base)/scale).norm())+1e-10
                    terminal=fit.prefix_check(runtime,hook,batches[0],native)
                    state=weights(chosen);package=dict(states=dict(BASE=weights(base),FINAL=state),order=t['order'],lock=c.digest(lock))
                    c.save(point(t),package);loaded=torch.load(point(t),map_location='cpu',weights_only=True)
                    assert all(torch.equal(state[k],loaded['states']['FINAL'][k]) for k in state)
                    result=dict(status='COMPLETE',order=t['order'],curve=curve,refreshes=refreshes,diagnostics=diagnostics,final_values=values,
                        endpoint=dict(RAW_gain=raw_gain,protected_gain=gain,gain_ratio=gain/raw_gain,radial=radial,restoration_alpha=alpha,restoration_trace=trace,
                            RAW_coordinate_norm=raw_norm,protected_coordinate_norm=actual_norm,exact_RAW=all(torch.equal(state[k],saved['RAW'][k]) for k in state)),
                        mechanical=mechanical,terminal_prefix_check=terminal,all_four_cores_updated=all(any(s['core_update_norms'][k]>0 for s in curve) for k in q.KEYS),
                        updates=160,backwards=counts['backwards']-start['backwards'],non_generation_forwards=counts['forwards']-start['forwards'],lock=c.digest(lock))
                    assert result['all_four_cores_updated'] and result['backwards']==8928
                    c.write(RUN/'private/source_results'/f"{t['order']}.json",result)
                    for i,(row,_,raw,expanded,b,parent) in enumerate(frozen[t['order']]):emit(runtime,hook,t,row,raw,expanded,b,counts,'NATIVE' if i==0 else 'FIT',i,parent)
                    expert.load_state_dict(base);assert all(torch.equal(v,base[k]) for k,v in expert.state_dict().items())
                    # Optimizer/metric last consumer completed; final candidate remains for evaluation.
                    retired=[x for x in (resume,metric_file) if x.exists()]
                    receipt=dict(paths=list(map(str,retired)),bytes=sum(x.stat().st_size for x in retired),reason='source training and generation complete; final candidate retained')
                    c.write(RUN/'private/retired_active'/f"{t['order']}.json",receipt)
                    for path in retired:assert not path.is_symlink();path.unlink()
                finally:hook.detach()
            assert not any(v.grad is not None for v in runtime.model.parameters())
        finally:handle.remove();c.write(RUN/'private'/f'PROTECTION_COUNTS_{part}.json',counts)
    p.done('PROTECTION_WORKER_'+str(part))


def freeze():
    rows=[c.read(RUN/'private/source_results'/f'{i}.json') for i in range(1,9)]
    assert all(r['endpoint']['gain_ratio']>=.9 and r['all_four_cores_updated'] for r in rows)
    counts=[c.read(RUN/'private'/f'PROTECTION_COUNTS_{i}.json') for i in range(6)]
    raw=c.read(RUN/'private/RAW_REFERENCE_READY.json')['counts']
    assert sum(r['updates'] for r in counts+raw)==2560 and sum(r['backwards'] for r in counts+raw)==73984
    assert sum(r['non_generation_forwards'] for r in rows)+3046<=136764
    assert len(list((RUN/'private/outputs/PROTECTED').glob('*/*.json')))==40
    c.write(RUN/'private/CANDIDATES_FROZEN.json',dict(status='PASS',candidates=8,counts=counts,source_numerical_gate=True,epoch=time.time()))


def eval_worker():
    assert c.read(RUN/'private/CANDIDATES_FROZEN.json')['status']=='PASS'
    part=int(os.environ['PARTITION']);gpu=q.GPUS[part];old.calibration.require_memory(gpu)
    tasks=q.d.selected()[part::6];h=data.held();hb=c.read(RUN/'private/HELD_BASE.json');counts=dict(forwards=0,generations=0,prefix_forwards=0)
    with p.lease(gpu):
        runtime,bindings=c.load(gpu);original=runtime.llava_model().prepare_inputs_labels_for_multimodal;frozen={}
        for row in h+[r for t in tasks for r in data.rows(t)]:
            if row['query_id'] in frozen:continue
            raw=runtime.adapter.prepare_inputs(Path(c.local_path(row['image_path'])),row['question'],None);assert raw['image_sha256']==row['image_sha256']
            with torch.no_grad():expanded=original(raw['input_ids'],None,raw['attention_mask'],None,None,raw['images'],image_sizes=None)
            assert expanded[4].dtype==torch.float16
            frozen[row['query_id']]=(raw,tuple(x.detach().cpu() if isinstance(x,torch.Tensor) else x for x in expanded))
        batches=[old.cpu(q.protection_batch(runtime,r,tasks[0])) for r in h];old.fp32(runtime)
        batches=[old.batch_on(runtime,b) for b in batches]
        handle=runtime.model.register_forward_pre_hook(lambda *_:counts.__setitem__('forwards',counts['forwards']+1))
        try:
            refs=[old.unedited(runtime,b) for b in batches];counts['prefix_forwards']+=96
            for t in tasks:
                saved=torch.load(point(t),map_location='cpu',weights_only=True);assert saved['lock']==c.digest(c.read(RUN/'private/PROTECTION_LOCK.json'))
                expert=p.expert(saved['states']['FINAL'],t['seed'],runtime.device).requires_grad_(False)
                hook=q.scoped.replay.MedTraceLayerHook(runtime.get_module(c.LAYER),expert);hook.attach()
                try:
                    values=[]
                    for batch,(ref,target) in zip(batches,refs):
                        v,y=q.logits(runtime,hook,batch);assert torch.equal(target,y);values.append(q.compare(ref,v,y));counts['prefix_forwards']+=1
                    c.write(RUN/'private/held_diagnostics'/f"{t['order']}.json",dict(values=values,original_primary=[r['original_primary'] for r in h]))
                    for index,row in enumerate(data.rows(t)+h):
                        raw,expanded=frozen[row['query_id']]
                        binding=dict(question=row['question'],reference=row['reference'],image_sha256=raw['image_sha256'],image_path=row['image_path'],
                            prompt_ids=raw['input_ids'][0].tolist(),attention_mask=raw['attention_mask'][0].tolist(),
                            runtime=dict(inherited_runtime=next(iter(bindings.values()))['runtime'],actual_precision='MATH_FP32_FROZEN_FP16_PREFILL',frozen_FP16_prefill=True),generation=runtime.generation_config)
                        baseline=hb[row['query_id']]['path'] if row['audit_role']=='HELDOUT' else PARENT/'private/audit_outputs/BASE'/str(t['order'])/(row['query_id']+'.json')
                        emit(runtime,hook,t,row,raw,expanded,binding,counts,row['audit_role'],index,baseline)
                finally:hook.detach()
                expert.load_state_dict(saved['states']['BASE']);assert all(torch.equal(v.cpu(),saved['states']['BASE'][k]) for k,v in expert.state_dict().items())
                print('PROTECTION_EVAL_COMPLETE',t['order'],flush=True)
        finally:handle.remove();c.write(RUN/'private'/f'PROTECTION_EVAL_COUNTS_{part}.json',counts)
    p.done('PROTECTION_EVAL_WORKER_'+str(part))


def finish():
    paths=list((RUN/'private/outputs/PROTECTED').glob('*/*.json'));assert len(paths)==923
    counts=[c.read(RUN/'private'/f'PROTECTION_EVAL_COUNTS_{i}.json') for i in range(6)]
    assert sum(v['generations'] for v in counts)==883 and sum(v['prefix_forwards'] for v in counts)==1344
    train=[c.read(RUN/'private'/f'PROTECTION_COUNTS_{i}.json') for i in range(6)];raw=c.read(RUN/'private/RAW_REFERENCE_READY.json')['counts']
    assert sum(x['generations'] for x in train+raw+counts)==963
    source=[c.read(RUN/'private/source_results'/f'{i}.json') for i in range(1,9)]
    total=sum(x['forwards'] for x in raw+train+counts)
    rawoutputs=[c.read(p) for p in (RUN/'private/outputs/RAW').glob('*/*.json')]
    assert total==3046+sum(r['non_generation_forwards'] for r in source)+1344+sum(c.read(x)['forwards'] for x in paths)+sum(x['forwards'] for x in rawoutputs)
    assert total<=c.read(RUN/'private/PROTECTION_LOCK.json')['maximum_forwards'] and not list((RUN/'private').glob('FAILURE*'))
    c.write(RUN/'private/PROTECTION_GENERATION_COMPLETE.json',dict(status='COMPLETE',forwards=total,new_generations=963,protected_outputs=923,epoch=time.time()))
    retired=[f for t in q.d.selected() for f in (raw_path(t),point(t))]
    assert all(f.is_file() and not f.is_symlink() and RUN.resolve() in f.resolve().parents for f in retired)
    receipt=dict(paths=list(map(str,retired)),bytes=sum(f.stat().st_size for f in retired),reason='all registered source/evaluation consumers durable')
    c.write(RUN/'private/DELETION_PLAN.json',receipt)
    for f in retired:f.unlink()
    active=[c.read(p) for p in (RUN/'private/retired_active').glob('*.json')];assert len(active)==8
    c.write(RUN/'public/LIFECYCLE.json',dict(status='COMPLETE',candidate_packages_deleted=16,candidate_bytes_deleted=receipt['bytes'],
        optimizer_metric_files_deleted=sum(len(r['paths']) for r in active),optimizer_metric_final_bytes_deleted=sum(r['bytes'] for r in active),retained_copies=0,historical_deletions=0,reconstruction_requires_training=True))


def controller():
    import pipeline
    pipeline.wait([pipeline.launch('protection.py','protection_rebuild',g,i) for i,g in enumerate(q.GPUS)])
    raw_barrier()
    pipeline.wait([pipeline.launch('protection.py','protection_worker',g,i) for i,g in enumerate(q.GPUS)])
    freeze()
    pipeline.wait([pipeline.launch('protection.py','protection_eval',g,i) for i,g in enumerate(q.GPUS)])
    finish();pipeline.wait([pipeline.launch('protection_queue.py','protection_ingest')])
    while not (RUN/'private/judge_protection_astra_medium/ALL_WORKERS_COMPLETE.json').exists():time.sleep(10)
    pipeline.wait([pipeline.launch('protection_queue.py','protection_report')])


if __name__=='__main__':
    try:{'protection_plan':plan,'protection_rebuild':rebuild,'protection_worker':protected_worker,'protection_eval':eval_worker,'protection_controller':controller}[os.environ['ACTION']]()
    except BaseException as error:
        c.write(RUN/'private'/('FAILURE_'+os.environ.get('ACTION','unknown')+'_'+os.environ.get('PARTITION','none')+'.json'),dict(error=repr(error),traceback=traceback.format_exc(),epoch=time.time()))
        raise
