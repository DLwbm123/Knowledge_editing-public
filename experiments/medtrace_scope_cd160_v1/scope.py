"""Frozen two-group comparison; existing runtime, optimizer and generation reused."""
import os
import time
import traceback
from pathlib import Path
from dataclasses import replace
import torch
import protection as previous
import scope_math as sm
q,c,p,RUN,old = previous.q,previous.c,previous.p,previous.RUN,previous.old
fit=previous.fit
q.GPUS=(5,6,7)
SUP=Path(os.environ['SUPPLEMENT_PARENT'])
PARENT=Path(os.environ['PROTECTION_PARENT'])


def tasks():return c.read(SUP/'private/FROZEN_TASKS.json')
def source(t):
    rows=[dict(t['native'],query_id=f"edit_{t['order']}_{i}",question=question,audit_role='NATIVE' if i==0 else 'FIT')
          for i,question in enumerate([t['native']['question']]+t['fit_questions'])]
    return rows+[dict(r,audit_role='GFIT') for r in c.read(SUP/'private/FROZEN_ANCHORS.json') if r['owner']==t['order']]
def events(t):
    original=c.read(PARENT/'private/BENCHMARK146_QUEUE.json')['tasks'][t['order']-1]
    rows=previous.data.rows(original)
    if t['order']==8:
        rows=[r for r in rows if r['audit_role'] not in ('T1G','T2G')]+c.read(SUP/'private/REVISED_EVENT_ROWS.json')
    return rows

def jobs(protected=None):
    arms=('C','D') if protected is None else ('D',) if protected else ('C',)
    out=[(t,a) for t in tasks() for a in arms]
    return out+[(tasks()[7],a) for a in (('A8','B8') if protected is None else ('B8',) if protected else ('A8',))]
def point(t,arm):return RUN/'private/candidates'/arm/f"{t['order']}.pt"
def output(arm,owner,row):return RUN/'private/outputs'/arm/str(owner)/(row['query_id']+'.json')
def baseline(t,row):
    if row['audit_role']=='HELDOUT':return Path(c.read(RUN/'private/HELD_BASE.json')[row['query_id']]['path'])
    if row['audit_role']=='GFIT' or t['order']==8 and row['audit_role'] in ('NATIVE','FIT','T1G','T2G'):
        return output('BASE',t['order'],row)
    if row['audit_role'] in ('NATIVE','FIT'):
        return Path(os.environ['GENERATION_PARENT'])/'private/outputs'/str(t['order'])/'BASE'/(row['query_id']+'.json')
    return Path(os.environ['DAMAGE_PARENT'])/'private/audit_outputs/BASE'/str(t['order'])/(row['query_id']+'.json')

def record(t,row):
    return replace(c.record(t),question=row['question'],target=row['reference'],image_path=Path(c.local_path(row['image_path'])))
def prepare(runtime,bindings,t,rows,training=False):
    original=runtime.llava_model().prepare_inputs_labels_for_multimodal;out=[]
    for row in rows:
        raw=runtime.adapter.prepare_inputs(Path(c.local_path(row['image_path'])),row['question'],None)
        if row.get('image_sha256'):assert raw['image_sha256']==row['image_sha256']
        with torch.no_grad():expanded=original(raw['input_ids'],None,raw['attention_mask'],None,None,raw['images'],image_sizes=None)
        assert expanded[4].dtype==torch.float16
        expanded=tuple(x.detach().cpu() if isinstance(x,torch.Tensor) else x for x in expanded)
        binding=dict(question=row['question'],reference=row['reference'],image_sha256=raw['image_sha256'],image_path=row['image_path'],
            prompt_ids=raw['input_ids'][0].tolist(),attention_mask=raw['attention_mask'][0].tolist(),
            runtime=dict(inherited_runtime=next(iter(bindings.values()))['runtime'],actual_precision='MATH_FP32_FROZEN_FP16_PREFILL',frozen_FP16_prefill=True),generation=runtime.generation_config)
        b=old.cpu(runtime.build_edit_batch(record(t,row))) if training else None
        out.append((row,b,raw,expanded,binding))
    return out

def emit(runtime,hook,t,arm,item,counts,dest=None,route=None):
    row,_,raw,expanded,binding=item;dest=dest or output(arm,t['order'],row)
    assert not dest.exists(),'No implicit generation retry'
    c.budget();start=counts['forwards'];counts['generation_attempts']+=1
    answer,trace=fit.short.generate(runtime,hook,raw,expanded)
    c.write(dest,dict(arm=arm,expert_order=t['order'],role=row['audit_role'],original_primary=row.get('original_primary',False),
        same_reference=row.get('same_reference',False),binding=dict(input=dict(row,image_sha256=raw['image_sha256']),judge_input=binding),
        R0=dict(raw_answer=answer.decoded_text,raw_token_ids=list(answer.raw_token_ids)),baseline_path=str(baseline(t,row)),
        generation_trace=trace,forwards=counts['forwards']-start,at_cap=len(answer.raw_token_ids)>=1024,route=route,
        lock=c.digest(c.read(RUN/'private/SCOPE_LOCK.json')),execution=c.read(RUN/'private/GPU_SOURCE_VERSION.json')))


def weights(n):
    s=[.5,.125,.125,.125,.125]+([0.,0.] if n==7 else [])
    return torch.tensor([s]+([[0.,0.,0.,0.,0.,.5,.5]] if n==7 else []),dtype=torch.float64)

def measure(runtime,hook,expert,state,batches,refs,counts,gradient=False):
    expert.load_state_dict(state);values=[];grads=[];native=None
    for i,(batch,(ref,target)) in enumerate(zip(batches,refs)):
        if gradient:
            hook.set_teacher_routing(batch.labels);result=runtime.model(**batch.forward_kwargs())
            mask=batch.labels[:,1:]!=-100;logits=result.logits[:,:-1][mask];y=batch.labels[:,1:][mask]
            loss=torch.nn.functional.cross_entropy(logits.double(),y);assert torch.isfinite(loss)
            g=torch.autograd.grad(-loss,[dict(expert.named_parameters())[k] for k in q.KEYS]);counts['backwards']+=1
            grads.append(torch.cat([x.detach().cpu().double().flatten() for x in g]));logits=logits.detach().cpu().double();y=y.cpu()
            del result,loss,g
        else:logits,y=q.logits(runtime,hook,batch)
        assert torch.equal(y,target);values.append(q.compare(ref,logits,target))
        if i==0:native=(logits,target)
    w=weights(len(batches));gain=w@torch.tensor([-v['NLL_change'] for v in values],dtype=torch.float64)
    return gain,values,native,w@torch.stack(grads) if gradient else None


def restore(before,proposed,initial,scale,radius,linear_g,required_linear,current,required,measure_gain):
    def accepted(state):
        z=(q.flatten(state)-q.flatten(before))/scale
        # Float32 storage may round the final displacement by one small ulp.
        tol=8*torch.finfo(torch.float32).eps*float((q.flatten(before)/scale).norm())+1e-10
        return bool(z.norm()<=radius+tol and sm.feasible(linear_g@z,required_linear)) and sm.feasible(measure_gain(state)-current,required)
    trace=[]
    if initial is not None and accepted(initial):return initial,'PROTECTED',trace
    raw_ok=accepted(proposed)
    if raw_ok:
        start=initial if initial is not None else before;lo,hi=0.,1.;chosen=proposed
        for _ in range(16):
            a=(lo+hi)/2;state=q.add(start,a*(q.flatten(proposed)-q.flatten(start)));ok=accepted(state);trace.append(dict(alpha=a,feasible=ok))
            if ok:hi,chosen=a,state
            else:lo=a
        return chosen,'RESTORED_TO_RAW',trace
    if initial is not None:
        for k in range(1,17):
            a=2.**(-k);state=q.add(before,a*(q.flatten(initial)-q.flatten(before)));ok=accepted(state);trace.append(dict(scale=a,feasible=ok))
            if ok:return state,'SCALED_PROTECTED',trace
    return before,'SKIPPED_INFEASIBLE',trace


def train():
    from m3bench_repro.editors.llava_runtime import seed_everything
    protected=os.environ['ACTION']=='scope_protected';part=int(os.environ['PARTITION']);gpu=q.GPUS[part]
    assigned=jobs(protected)[part::len(q.GPUS)];old.calibration.require_memory(gpu)
    counts=dict(forwards=0,backwards=0,updates=0,generation_attempts=0)
    with p.lease(gpu):
        runtime,bindings=c.load(gpu);frozen={t['order']:prepare(runtime,bindings,t,source(t),True) for t,a in assigned}
        basis=q.split()[0]
        basis_frozen={r['query_id']:old.cpu(q.protection_batch(runtime,r,assigned[0][0])) for r in basis} if protected else {}
        old.fp32(runtime)
        if protected:q.protection_batch=lambda runtime,row,task:old.batch_on(runtime,basis_frozen[row['query_id']])
        handle=runtime.model.register_forward_pre_hook(lambda *_:counts.__setitem__('forwards',counts['forwards']+1))
        try:
            for t,arm in assigned:
                assert not point(t,arm).exists(),'No implicit trajectory restart'
                seed_everything(t['seed']);expert=p.expert(q.d.start_state(t),t['seed'],runtime.device);base=old.snapshot(expert)
                assert sum(x.numel() for x in expert.parameters())==7168 and torch.count_nonzero(expert.G1)==0
                hook=q.scoped.replay.MedTraceLayerHook(runtime.get_module(c.LAYER),expert);hook.attach()
                resume=RUN/'private/active'/f'{arm}_{t["order"]}.pt';metric=resume.with_name(resume.stem+'_geometry.pt')
                items=frozen[t['order']][:5] if arm in ('A8','B8') else frozen[t['order']]
                batches=[old.batch_on(runtime,v[1]) for v in items];start=dict(counts);curve=[];refreshes=[]
                try:
                    refs=[q.logits(runtime,hook,b,verify_base=i==0) for i,b in enumerate(batches)]
                    opt=p.optimizer(expert,runtime.model);scale=previous.scale_for(expert,opt);params=dict(expert.named_parameters())
                    mechanical=None
                    for step in range(1,161):
                        c.budget();before=old.snapshot(expert)
                        if protected and step in previous.REFRESH:
                            matrix,n,audit=old.prior.basis_gradients(runtime,hook,basis,t,params);counts['backwards']+=n
                            geometry=sm.Geometry(matrix*scale);del matrix
                            refreshes.append(dict(step=step,**audit,geometry=geometry.audit));c.save(metric,dict(v=geometry.v,values=geometry.values,step=step))
                        opt.zero_grad(set_to_none=True)
                        chosen_indices=[(0,.5),(1+(step-1)%4,.5)] if len(batches)==5 else [(0,.25),(1+(step-1)%4,.25),(5+(step-1)%2,.5)]
                        for i,w in chosen_indices:
                            hook.set_teacher_routing(batches[i].labels);loss=runtime.compute_loss(batches[i]);assert torch.isfinite(loss)
                            (w*loss).backward();counts['backwards']+=1
                        norm=torch.nn.utils.clip_grad_norm_(expert.parameters(),1.);assert torch.isfinite(norm)
                        assert all(x.grad is not None and torch.isfinite(x.grad).all() for x in expert.parameters())
                        opt.step();q.d.apply_direction(opt,{id(v):before[k] for k,v in expert.named_parameters()},'RAW')
                        proposed=old.snapshot(expert);chosen=proposed;mode='RAW';solver=None;trace=[];gains=None
                        if protected:
                            current,_,_,g=measure(runtime,hook,expert,before,batches,refs,counts,True)
                            raw_gain,_,_,_=measure(runtime,hook,expert,proposed,batches,refs,counts)
                            z=(q.flatten(proposed)-q.flatten(before))/scale;lg=g*scale
                            delta,solver=sm.solve(geometry,lg,z);initial=q.add(before,delta*scale) if delta is not None else None
                            required=.9*(raw_gain-current).clamp_min(0);required_linear=.9*(lg@z).clamp_min(0)
                            evaluator=lambda s:measure(runtime,hook,expert,s,batches,refs,counts)[0]
                            chosen,mode,trace=restore(before,proposed,initial,scale,float(z.norm()),lg,required_linear,current,required,evaluator)
                            accepted=evaluator(chosen);gains=dict(current=current.tolist(),raw=(raw_gain-current).tolist(),accepted=(accepted-current).tolist(),required=required.tolist())
                        expert.load_state_dict(chosen);counts['updates']+=1
                        curve.append(dict(step=step,mode=mode,solver=solver,restoration=trace,gains=gains,
                            core_update_norms={k:float((chosen[k]-before[k]).norm()) for k in q.KEYS}))
                        if step==1 and any(x>0 for x in curve[-1]['core_update_norms'].values()):mechanical=fit.short.mechanical(runtime,hook,batches[0])
                        if step%20==0:
                            c.save(resume,dict(completed_step=step,state=previous.weights(chosen),optimizer=opt.state_dict(),curve=curve,counts=dict(counts),rng=c.rng()))
                            c.write(RUN/'private/progress'/arm/f'{t["order"]}.json',dict(step=step,mode=mode,counts=counts));print('TRAIN',arm,t['order'],step,mode,flush=True)
                    final=old.snapshot(expert);endpoint=dict(qualified=True)
                    if protected:
                        refarm='C' if arm=='D' else 'A8';saved=torch.load(point(t,refarm),map_location='cpu',weights_only=True)
                        reference={k:v.to(runtime.device) for k,v in saved['states']['FINAL'].items()}
                        rg=measure(runtime,hook,expert,reference,batches,refs,counts)[0];radius=float(((q.flatten(reference)-q.flatten(base))/scale).norm())
                        n=float(((q.flatten(final)-q.flatten(base))/scale).norm());radial=min(1.,radius/n) if n else 1.
                        final=q.add(base,radial*(q.flatten(final)-q.flatten(base)))
                        required=.9*rg.clamp_min(0);fg=measure(runtime,hook,expert,final,batches,refs,counts)[0];trace=[]
                        if not sm.feasible(fg,required) and sm.feasible(rg,required):
                            lo,hi=0.,1.;chosen=reference
                            for _ in range(16):
                                a=(lo+hi)/2;candidate=q.add(final,a*(q.flatten(reference)-q.flatten(final)))
                                ok=sm.feasible(measure(runtime,hook,expert,candidate,batches,refs,counts)[0],required);trace.append(dict(alpha=a,feasible=ok))
                                if ok:hi,chosen=a,candidate
                                else:lo=a
                            final=chosen
                        fg=measure(runtime,hook,expert,final,batches,refs,counts)[0]
                        endpoint=dict(qualified=sm.feasible(fg,required),reference_gain=rg.tolist(),gain=fg.tolist(),required=required.tolist(),radial=radial,restoration=trace)
                        actual=float(((q.flatten(final)-q.flatten(base))/scale).norm())
                        assert actual<=radius+8*torch.finfo(torch.float32).eps*float((q.flatten(base)/scale).norm())+1e-10
                    gain,values,native,_=measure(runtime,hook,expert,final,batches,refs,counts)
                    moved=any(float((final[k]-base[k]).norm())>0 for k in q.KEYS)
                    terminal=fit.prefix_check(runtime,hook,batches[0],native) if moved else dict(status='ZERO_RESIDUAL_SKIPPED')
                    c.save(point(t,arm),dict(states=dict(BASE=previous.weights(base),FINAL=previous.weights(final)),arm=arm,order=t['order'],lock=c.digest(c.read(RUN/'private/SCOPE_LOCK.json'))))
                    c.write(RUN/'private/results'/arm/f'{t["order"]}.json',dict(status='COMPLETE',curve=curve,refreshes=refreshes,endpoint=endpoint,
                        gain=gain.tolist(),values=values,mechanical=mechanical,terminal=terminal,counts={k:counts[k]-start[k] for k in counts},
                        all_four_cores_updated=all(any(v['core_update_norms'][k]>0 for v in curve) for k in q.KEYS)))
                    if not moved:hook.detach()
                    for item in items:emit(runtime,hook if moved else None,t,arm,item,counts)
                    hook.detach()
                    receipt=dict(paths=[str(x) for x in (resume,metric) if x.exists()],reason='training completed; final retained')
                    c.write(RUN/'private/retired'/f'{arm}_{t["order"]}.json',receipt)
                    for path in map(Path,receipt['paths']):assert not path.is_symlink();path.unlink()
                finally:hook.detach()
        finally:handle.remove();c.write(RUN/'private'/f'COUNTS_{os.environ["ACTION"]}_{part}.json',counts)


def base_worker():
    part=int(os.environ['PARTITION']);gpu=q.GPUS[part];old.calibration.require_memory(gpu)
    assigned=[(t,r) for t in tasks() for r in source(t)+events(t) if baseline(t,r)==output('BASE',t['order'],r)][part::3]
    counts=dict(forwards=0,backwards=0,updates=0,generation_attempts=0)
    with p.lease(gpu):
        runtime,bindings=c.load(gpu);prepared=[(t,prepare(runtime,bindings,t,[r])[0]) for t,r in assigned];old.fp32(runtime)
        h=runtime.model.register_forward_pre_hook(lambda *_:counts.__setitem__('forwards',counts['forwards']+1))
        try:
            for t,item in prepared:emit(runtime,None,t,'BASE',item,counts)
        finally:h.remove();c.write(RUN/'private'/f'COUNTS_scope_base_{part}.json',counts)


def evaluate():
    assert c.read(RUN/'private/CANDIDATES_FROZEN.json')['candidates']==18
    part=int(os.environ['PARTITION']);gpu=q.GPUS[part];old.calibration.require_memory(gpu)
    assigned=jobs()[part::3];counts=dict(forwards=0,backwards=0,updates=0,generation_attempts=0)
    with p.lease(gpu):
        runtime,bindings=c.load(gpu);held=previous.data.held()
        frozen={t['order']:prepare(runtime,bindings,t,events(t)+held) for t,a in assigned};old.fp32(runtime)
        h=runtime.model.register_forward_pre_hook(lambda *_:counts.__setitem__('forwards',counts['forwards']+1))
        try:
            for t,arm in assigned:
                saved=torch.load(point(t,arm),map_location='cpu',weights_only=True)
                expert=p.expert(saved['states']['FINAL'],t['seed'],runtime.device).requires_grad_(False)
                moved=any(not torch.equal(saved['states']['FINAL'][k],saved['states']['BASE'][k]) for k in q.KEYS)
                hook=q.scoped.replay.MedTraceLayerHook(runtime.get_module(c.LAYER),expert);hook.attach()
                try:
                    if not moved:hook.detach()
                    for item in frozen[t['order']]:emit(runtime,hook if moved else None,t,arm,item,counts)
                finally:hook.detach()
                print('EVAL_COMPLETE',arm,t['order'],flush=True)
        finally:h.remove();c.write(RUN/'private'/f'COUNTS_scope_eval_{part}.json',counts)


def plan():
    checks=sm.selfcheck();assert c.read(SUP/'public/DATA_ADMISSION.json')['status']=='SOURCE_EVIDENCE_GATE_PASS'
    ts=tasks();assert len(jobs())==18 and [t['order'] for t in ts]==list(range(1,9))
    assert sum(len(events(t)) for t in ts)==115 and sum(len(source(t)) for t in ts)==56
    assert sum(baseline(t,r)==output('BASE',t['order'],r) for t in ts for r in source(t)+events(t))==29
    c.write(RUN/'private/HELD_BASE.json',c.read(PARENT/'private/HELD_BASE.json'))
    lock=dict(arms=['C','D','A8','B8'],trajectories=18,steps=160,updates=2880,GPUS=list(q.GPUS),
        maximum_generations=2567,maximum_Judge=2567,maximum_backwards=88352,maximum_non_generation_forwards=220000,
        source_fraction=.9,Fisher_refresh=list(previous.REFRESH),ridge=sm.RIDGE,seed_selection=False,
        source_roles=['NATIVE','FIT','GFIT'],geometry_heldout=False,all_candidates_frozen_before_eval=True,
        source_binding=c.digest(c.read(SUP/'private/FROZEN_ANCHORS.json')),tasks_binding=c.digest(ts),selfcheck=checks)
    c.write(RUN/'private/SCOPE_LOCK.json',lock);c.write(RUN/'public/ADMISSION.json',dict(status='PASS',**lock))


def controller():
    import pipeline
    for action in ('scope_base','scope_raw','scope_protected'):
        pipeline.wait([pipeline.launch('scope.py',action,g,i) for i,g in enumerate(q.GPUS)])
    assert all(point(t,a).exists() for t,a in jobs())
    c.write(RUN/'private/CANDIDATES_FROZEN.json',dict(candidates=18,epoch=time.time(),selection=False))
    pipeline.wait([pipeline.launch('scope.py','scope_eval',g,i) for i,g in enumerate(q.GPUS)])
    pipeline.wait([pipeline.launch('scope_route.py','scope_route',q.GPUS[0],0)])
    finish()


def finish():
    counts=[c.read(f) for f in (RUN/'private').glob('COUNTS_*.json')]
    assert sum(x['updates'] for x in counts)==2880 and sum(x['backwards'] for x in counts)<=88352
    paths=list((RUN/'private/outputs').glob('*/*/*.json'));outputs=[c.read(f) for f in paths]
    assert sum(x['generation_attempts'] for x in counts)<=2567
    non_generation=sum(x['forwards'] for x in counts)-sum(x['forwards'] for x in outputs if not x.get('alias_of'))
    assert non_generation<=220000 and len(outputs)==2567
    failures={x.name for x in (RUN/'private').glob('FAILURE*')}
    resolved=c.read(RUN/'private/ROUTE_RECOVERY_COMPLETE.json')['resolved_failures'] if (RUN/'private/ROUTE_RECOVERY_COMPLETE.json').exists() else []
    assert failures==set(resolved), 'Unresolved engineering failure'
    c.write(RUN/'private/SCOPE_GENERATION_COMPLETE.json',dict(status='GENERATED_SCORING_PENDING',counts=counts,non_generation_forwards=non_generation,outputs=len(outputs),epoch=time.time()))
    c.write(RUN/'public/PROGRESS.json',dict(status='GENERATED_SCORING_PENDING',training_complete=True,scoring_complete=False,publication_complete=False))


if __name__=='__main__':
    try:{'scope_plan':plan,'scope_base':base_worker,'scope_raw':train,'scope_protected':train,'scope_eval':evaluate,'scope_controller':controller,'scope_finish':finish}[os.environ['ACTION']]()
    except BaseException as error:
        c.write(RUN/'private'/f'FAILURE_{os.environ.get("ACTION")}_{os.environ.get("PARTITION")}.json',dict(error=repr(error),traceback=traceback.format_exc(),epoch=time.time()))
        raise
