"""Fixed five-arm pure-W attribution experiment; parent runs stay read-only."""
import os
import json
import time
import traceback
from pathlib import Path
import torch
import torch.nn.functional as F
import base_run as v1
import journal as j

c, RUN, data, old = v1.c, v1.RUN, v1.data, v1.old
PARENT = Path(os.environ['PUREW_PARENT'])
ARMS = ('A_LOW', 'B_NORM', 'C_FULL_FIXED', 'D_FULL_REFRESH', 'E_RLS')
GPUS = (5,) * 5
v1.ARMS = ARMS
v1.GPUS = GPUS
c.write = j.write


def binding(): return c.digest(c.read(RUN/'private/LOCK.json'))
def key(row): return str(row['owner']) + ':' + row['query_id']
def done(name, **value):
    c.write(RUN/'private'/name,dict(value,lock=binding(),epoch=time.time()))
def is_done(name): return j.completed(RUN/'private'/name,binding())


def plan():
    import diagnostic_math as sm
    if (RUN/'private/LOCK.json').exists() and (RUN/'private/PLAN_COMPLETE.json').exists():
        assert c.read(RUN/'private/LOCK.json')['epoch']=='PUREW_SCOPE_V3_20261010'
        assert c.read(RUN/'private/LOCK.json')==c.read(RUN/'public/PROTOCOL.json')
        assert c.read(RUN/'private/LOCK.json')==c.read(RUN/'private/FROZEN_PROTOCOL.json')
        return
    if (RUN/'private/LOCK.json').exists():
        assert c.read(RUN/'private/LOCK.json')['epoch']=='PUREW_SCOPE_V3_20261010'
    else:v1.plan()
    from protocol import specification
    lock=specification(c.read(RUN/'private/LOCK.json'),sm.selfcheck(),j.selfcheck())
    assert lock==c.read(RUN/'private/FROZEN_PROTOCOL.json'),'remote admission differs from preregistration'
    c.write(RUN/'private/LOCK.json',lock);c.write(RUN/'public/PROTOCOL.json',lock)
    assert c.read(PARENT/'public/RESULTS.json')['status']=='COMPLETE'
    assert c.read(PARENT/'private/ROWS.json')==v1.rows()
    c.write(RUN/'public/ADMISSION.json',dict(status='CPU_AND_DATA_ADMITTED_NOT_GPU_RESULTS',**lock))
    # Preserve answers and their original generation/scoring provenance; only consumer binding is new.
    for row in v1.rows():
        src=v1.output('BASE',row['owner'],row).relative_to(RUN)
        prior=c.read(PARENT/src)
        assert prior['binding']['input']==row and prior['lock']==c.digest(c.read(PARENT/'private/LOCK.json'))
        dst=RUN/src
        inherited=dict(prior,lock=binding(),inherited_generation=dict(path=str(PARENT/src),execution=prior['execution'],lock=prior['lock']))
        if dst.exists():assert c.read(dst)==inherited
        else:c.write(dst,inherited)
    base=c.read(PARENT/'private/BASE_COMPLETE.json')
    if not is_done('BASE_COMPLETE.json'):done('BASE_COMPLETE.json',**dict(base,inherited=True,new_generations=0))
    done('PLAN_COMPLETE.json',status='PASS',inherited_Base=267)


def source_items(runtime,bindings):
    selected=[r for r in v1.rows() if r['audit_role'] in ('NATIVE','FIT','GFIT')]
    items=v1.prepare(runtime,bindings,selected,False)
    tasks={t['order']:t for t in data.tasks()}
    result=[]
    for row,_,raw,expanded,bind in items:
        batch=runtime.build_edit_batch(data.record(tasks[row['owner']],row))
        result.append((row,v1.cpu_batch(batch),raw,expanded,bind,(batch.image_token_start,batch.image_token_end)))
    return result


def tracked(runtime,name):
    count=dict(forwards=0,backwards=0,updates=0,generations=0,generation_attempts=0)
    path=RUN/'private/attempts'/f'{name}_{os.getpid()}_{time.time_ns()}.json'
    def persist():c.write(path,dict(count,action=name,pid=os.getpid(),lock=binding()))
    def before(*_):count['forwards']+=1;persist()
    handle=runtime.model.register_forward_pre_hook(before);persist()
    return count,handle,persist


def emit(runtime,item,stage,count,persist):
    row=item[0];path=v1.output(stage,row['owner'],row)
    if path.exists():
        saved=c.read(path)
        assert saved['lock']==binding() and saved['binding']['input']==row
        assert saved['stage']==stage and saved['owner']==row['owner']
        return path
    count['generation_attempts']+=1;persist()
    result=v1.emit(runtime,item[:5],stage,count);persist();return result


def admission():
    from purew_math import target_logits,preservation_kl
    import propagation_math as sm
    if is_done('MECHANICAL_COMPLETE.json'):return
    c.available=v1.admission
    with c.lease(GPUS[0]):
        runtime,bindings=c.load(GPUS[0]);count,h,persist=tracked(runtime,'admission')
        try:
            v1.fp32_native(runtime);weight=runtime.get_module(v1.LAYER).weight
            items=source_items(runtime,bindings)
            geom=RUN/'private/GEOMETRY.pt'
            if not geom.exists():
                bases,audit=sm.make_basis(weight)
                c.save(geom,dict(bases=tuple(x.cpu() for x in bases),audit=audit,lock=binding()))
            geometry=torch.load(geom,map_location='cpu',weights_only=True);assert geometry['lock']==binding()
            bases=tuple(x.to(runtime.device) for x in geometry['bases'])
            cache={};captured={};meta=[None]
            def capture(_,args):
                if meta[0] is not None:
                    captured['keys']=sm.pool_keys(args[0].detach(),meta[0][0],meta[0][1])
            observer=runtime.get_module(v1.LAYER).register_forward_pre_hook(capture)
            try:
                for item in items:
                    row,value,_,_,_,span=item;path=RUN/'private/cache/source'/f"{row['owner']}_{row['query_id']}.pt"
                    if not path.exists():
                        batch=old.batch_on(runtime,value);meta[0]=(batch.labels,span)
                        with torch.no_grad():target_logits(runtime.model,batch)
                        tk,vk,mk=captured['keys'];cp=bases[0]@tk;st=bases[1]@tk;sv=bases[1]@vk;sc=bases[1]@mk
                        gamma,g=sm.gate(st,sv)
                        c.save(path,dict(private=cp.cpu(),text=st.cpu(),image=sv.cpu(),shared=sc.cpu(),gate=g,lock=binding()))
                    saved=torch.load(path,map_location='cpu',weights_only=True);assert saved['lock']==binding();cache[key(row)]=saved
                meta[0]=None
                basis=data.q.split()[0]
                for index,row in enumerate(basis):
                    path=RUN/'private/cache/protection'/f'{index}.pt'
                    if path.exists():continue
                    batch=data.q.protection_batch(runtime,row,data.tasks()[0]);meta[0]=(batch.labels,None)
                    with torch.no_grad():logits,_=target_logits(runtime.model,batch)
                    tk,_,_=captured['keys']
                    c.save(path,dict(reference=logits.cpu(),private=(bases[0]@tk).cpu(),lock=binding()))
                # Same real gradient, five proposals; restore original W after every probe.
                subset=items[:7];base=weight.detach().clone();weight.requires_grad_(True)
                import diagnostic_math as dm
                from purew_math import row_basis
                gradient=torch.zeros_like(weight)
                pp=torch.eye(448,device=runtime.device);ps=torch.eye(64,device=runtime.device)
                for i,fraction in [(0,.25),(1,.25),(5,.5)]:
                    batch=old.batch_on(runtime,subset[i][1]);meta[0]=(batch.labels,subset[i][5]);weight.grad=None
                    logits,labels=target_logits(runtime.model,batch)
                    count['backwards']+=1;persist();(fraction*F.cross_entropy(logits,labels)).backward()
                    gradient.add_(weight.grad)
                meta[0]=None;pb=data.q.protection_batch(runtime,basis[0],data.tasks()[0]);weight.grad=None
                ref=torch.load(RUN/'private/cache/protection/0.pt',map_location='cpu',weights_only=True)['reference'].to(runtime.device)
                logits,_=target_logits(runtime.model,pb);kl=preservation_kl(logits,ref)
                assert abs(float(kl.detach()))<1e-5
                count['backwards']+=1;persist();kl.backward();gradient.add_(weight.grad)
                q=row_basis(base);full_gradient=dm.full(gradient,q)
                reports=[]
                before={n:p._version for n,p in runtime.model.named_parameters() if p is not weight}
                for arm in ARMS:
                    proposal=dm.direction(gradient,arm,bases,(pp,ps),q)
                    step,receipt=dm.step(proposal,arm,full_gradient)
                    merged=base+step if isinstance(step,torch.Tensor) else sm.merged(base,step,bases)
                    with torch.no_grad():weight.copy_(merged)
                    assert torch.isfinite(weight).all() and not torch.equal(weight,base)
                    assert float((weight-base).norm())<=.0501
                    # The sole edited down_proj cannot alter its own input features.
                    batch=old.batch_on(runtime,subset[0][1]);meta[0]=(batch.labels,subset[0][5])
                    with torch.no_grad():target_logits(runtime.model,batch)
                    tk,vk,mk=captured['keys']
                    assert torch.equal(bases[0]@tk,cache[key(subset[0][0])]['private'].to(runtime.device))
                    assert torch.equal(bases[1]@vk,cache[key(subset[0][0])]['image'].to(runtime.device))
                    assert torch.equal(bases[1]@tk,cache[key(subset[0][0])]['text'].to(runtime.device))
                    assert torch.equal(bases[1]@mk,cache[key(subset[0][0])]['shared'].to(runtime.device))
                    if not isinstance(step,torch.Tensor):assert torch.equal(merged,sm.merged(base,tuple(x.clone() for x in step),bases))
                    reports.append(dict(arm=arm,gamma=1.,actual_step_norm=float((weight-base).norm()),**receipt))
                    with torch.no_grad():weight.copy_(base)
                    assert torch.equal(weight,base)
                assert before=={n:p._version for n,p in runtime.model.named_parameters() if p is not weight}
                weight.grad=None;weight.requires_grad_(False)
            finally:observer.remove()
            assert not runtime.get_module(v1.LAYER)._forward_pre_hooks and not runtime.get_module(v1.LAYER)._forward_hooks
            # Two fixed source-only cross-round identity probes; not fresh formal Base consumers.
            for owner in (1,8):
                item=next(x for x in items if x[0]['owner']==owner and x[0]['audit_role']=='NATIVE')
                probe=emit(runtime,item,'BASE_IDENTITY',count,persist)
                assert c.read(probe)['R0']==c.read(v1.output('BASE',owner,item[0]))['R0']
            schema=v1.state_schema(runtime.model)
            assert schema==c.read(RUN/'private/BASE_COMPLETE.json')['schema']
            c.write(RUN/'public/GATE_ADMISSION.json',dict(basis=geometry['audit'],
                rows=[dict(owner=x[0]['owner'],role=x[0]['audit_role'],within_edit=i%7,
                           **cache[key(x[0])]['gate']) for i,x in enumerate(items)],
                annotation='training evidence only; cosine gate is not a clinical validity judge'))
            done('MECHANICAL_COMPLETE.json',status='PASS',reports=reports,geometry=geometry['audit'],
                 exact_native_Base_identity=True,feature_invariance_exact=True,
                 parameters=sum(p.numel() for p in runtime.model.parameters()),added_parameters=0,
                 trainable_parameter_count=weight.numel(),counts=count)
            c.write(RUN/'public/MECHANICAL_COMPLETE.json',c.read(RUN/'private/MECHANICAL_COMPLETE.json'))
            print('MECHANICAL_PASS',reports,flush=True)
        finally:h.remove();persist()


def train():
    from purew_math import target_logits,preservation_kl,row_basis
    import propagation_math as sm
    import diagnostic_math as dm
    part=int(os.environ['PARTITION']);arm=ARMS[part];gpu=GPUS[part]
    if is_done(f'TRAIN_COMPLETE_{arm}.json'):return
    assert is_done('MECHANICAL_COMPLETE.json');c.available=v1.admission
    with j.exclusive(RUN/'private/locks'/f'{arm}.lock'),c.lease(gpu):
        runtime,bindings=c.load(gpu);count,h,persist=tracked(runtime,'train_'+arm)
        try:
            v1.fp32_native(runtime);weight=runtime.get_module(v1.LAYER).weight
            base=weight.detach().clone();schema=v1.state_schema(runtime.model);identity=id(weight)
            parameters=sum(p.numel() for p in runtime.model.parameters())
            assert parameters==7566219264 and schema==c.read(RUN/'private/BASE_COMPLETE.json')['schema']
            frozen={n:p._version for n,p in runtime.model.named_parameters() if p is not weight}
            buffers={n:b.detach().clone() for n,b in runtime.model.named_buffers()}
            geometry=torch.load(RUN/'private/GEOMETRY.pt',map_location='cpu',weights_only=True);assert geometry['lock']==binding()
            bases=tuple(x.to(runtime.device) for x in geometry['bases']);q0=row_basis(base)
            prepared=source_items(runtime,bindings)
            features={key(x[0]):torch.load(RUN/'private/cache/source'/f"{x[0]['owner']}_{x[0]['query_id']}.pt",map_location='cpu',weights_only=True) for x in prepared}
            protected=[v1.cpu_batch(data.q.protection_batch(runtime,row,data.tasks()[0])) for row in data.q.split()[0]]
            refs=[torch.load(RUN/'private/cache/protection'/f'{i}.pt',map_location='cpu',weights_only=True) for i in range(61)]
            assert all(x['lock']==binding() for x in list(features.values())+refs)
            low=arm in ('A_LOW','B_NORM','E_RLS')
            coefficients=tuple(torch.zeros(weight.shape[0],a.shape[0],device=runtime.device) for a in bases)
            inverses=tuple(torch.eye(a.shape[0],dtype=torch.float64,device=runtime.device) for a in bases)
            active=RUN/'private/active'/f'{arm}.pt';events=RUN/'private/training_diagnostics'/arm
            state=dict(completed_edits=0,active_owner=1,completed_step=0,committed_updates=0,last_event=None,lock=binding(),boundary=None)
            if active.exists():
                state=torch.load(active,map_location='cpu',weights_only=True);assert state['lock']==binding()
                coefficients=tuple(x.to(runtime.device) for x in state.get('coefficients',coefficients));inverses=tuple(x.to(runtime.device) for x in state.get('inverses',inverses))
                with torch.no_grad():weight.copy_(sm.merged(base,coefficients,bases) if low else state['weight'].to(runtime.device))
                j.ensure_event(events,state)
            weight.requires_grad_(True)
            def verify():
                assert id(weight)==identity and v1.state_schema(runtime.model)==schema
                assert frozen=={n:p._version for n,p in runtime.model.named_parameters() if p is not weight}
                assert all(torch.equal(v,buffers[n]) for n,v in runtime.model.named_buffers())
                assert [n for n,p in runtime.model.named_parameters() if p.requires_grad]==[v1.LAYER+'.weight']
                assert not runtime.get_module(v1.LAYER)._forward_pre_hooks and not runtime.get_module(v1.LAYER)._forward_hooks
            def save_state():
                if low:state.update(coefficients=tuple(x.cpu() for x in coefficients),inverses=tuple(x.cpu() for x in inverses))
                else:state['weight']=weight.detach().cpu()
                c.save(active,state)
                with active.open('rb') as f:os.fsync(f.fileno())
                fd=os.open(active.parent,os.O_RDONLY)
                try:os.fsync(fd)
                finally:os.close(fd)
                j.ensure_event(events,state)
            for task in data.tasks():
                owner=task['order']
                if owner<=state['completed_edits']:continue
                assert state['active_owner']==owner
                if arm=='D_FULL_REFRESH':
                    if state['boundary'] is None:state['boundary']=weight.detach().cpu()
                    q=row_basis(state['boundary'].to(runtime.device))
                else:q=q0
                items=[x for x in prepared if x[0]['owner']==owner];assert len(items)==7
                batches=[old.batch_on(runtime,x[1]) for x in items]
                for step in range(state['completed_step']+1,161):
                    c.budget();verify();sample=step in (1,40,80,160)
                    total=torch.zeros_like(weight);parts=tuple(torch.zeros_like(x) for x in coefficients) if low else None;losses={}
                    selected=[(0,.25),(1+(step-1)%4,.25),(5+(step-1)%2,.5)]
                    for index,fraction in selected:
                        weight.grad=None;logits,labels=target_logits(runtime.model,batches[index])
                        loss=F.cross_entropy(logits,labels);assert torch.isfinite(loss)
                        count['backwards']+=1;persist();(fraction*loss).backward();total.add_(weight.grad)
                        if low:
                            ds=dm.direction(weight.grad,arm,bases,inverses,q)
                            for target,d in zip(parts,ds):target.add_(d)
                        losses[str(index)]=float(loss.detach());del logits,loss
                    ce=total.clone() if sample else None
                    index=((owner-1)*160+step-1)%61;weight.grad=None
                    logits,_=target_logits(runtime.model,old.batch_on(runtime,protected[index]))
                    kl=preservation_kl(logits,refs[index]['reference'].to(runtime.device));assert torch.isfinite(kl)
                    count['backwards']+=1;persist();kl.backward();total.add_(weight.grad)
                    if low:
                        ds=dm.direction(weight.grad,arm,bases,inverses,q)
                        for target,d in zip(parts,ds):target.add_(d)
                    losses['BASIS_KL']=float(kl.detach());del logits,kl
                    assert all(p.grad is None for p in runtime.model.parameters() if p is not weight)
                    projected=dm.full(total,q) if not low or arm=='B_NORM' or sample else None
                    if not low:parts=projected
                    delta,receipt=dm.step(parts,arm,projected)
                    diagnostics={}
                    if sample:
                        raw=sm.coordinates(total,bases,tuple(torch.eye(a.shape[0],device=a.device,dtype=a.dtype) for a in bases),1.)
                        attenuated=sm.coordinates(total,bases,inverses,1.)
                        low_norm=torch.sqrt(sum(x.square().sum() for x in raw));attenuated_norm=torch.sqrt(sum(x.square().sum() for x in attenuated))
                        dense_delta=sum(d@a for d,a in zip(delta,bases)) if low else delta
                        reference_full=dm.full(total,q0) if arm=='D_FULL_REFRESH' else projected
                        diagnostics=dict(full_null_reference='initial_Base',raw_gradient_norm=float(total.norm()),full_null_gradient_norm=float(reference_full.norm()),low_gradient_norm=float(low_norm),
                            low_over_full_energy=float(low_norm.square()/reference_full.norm().square().clamp_min(1e-30)),
                            RLS_gradient_norm_ratio=float(attenuated_norm/low_norm.clamp_min(1e-30)),CE_gradient_norm=float(ce.norm()),KL_gradient_norm=float(weight.grad.norm()),
                            delta_dot_CE=float((dense_delta*ce).sum()),delta_dot_KL=float((dense_delta*weight.grad).sum()))
                    if low:
                        coefficients=tuple(x+d for x,d in zip(coefficients,delta));current=sm.merged(base,coefficients,bases)
                    else:current=weight.detach()+delta
                    actual=float((current-weight.detach()).norm());assert torch.isfinite(current).all() and actual<=.0501
                    with torch.no_grad():weight.copy_(current)
                    weight.grad=None
                    state.update(completed_step=step,committed_updates=state['committed_updates']+1)
                    event=dict(update=state['committed_updates'],owner=owner,step=step,losses=losses,source_indices=[i for i,f in selected],actual_step_norm=actual,diagnostics=diagnostics,**receipt)
                    j.write(RUN/'private/step_attempts'/arm/f'{event["update"]:04d}_{os.getpid()}_{time.time_ns()}.json',event)
                    state['last_event']=event;save_state();count['updates']+=1;persist()
                    del delta,parts,total,current,projected,ce
                    if sample:del dense_delta,raw,attenuated
                    if step%20==0:
                        c.write(RUN/'private/progress'/f'{arm}.json',dict(event,committed_updates=state['committed_updates'],counts=count));print('TRAIN',arm,owner,step,actual,flush=True)
                verify()
                for item in prepared:
                    row=item[0]
                    if row['owner']<=owner and row['audit_role'] in ('NATIVE','GFIT'):emit(runtime,item,f'{arm}_PREFIX_{owner}',count,persist)
                if arm=='E_RLS':
                    fractions=(.25,.0625,.0625,.0625,.0625,.25,.25)
                    for item,fraction in zip(items,fractions):
                        f=features[key(item[0])];inverses=(sm.recurse(inverses[0],f['private'].to(runtime.device),fraction),sm.recurse(inverses[1],f['shared'].to(runtime.device),fraction))
                    index=((owner-1)*160)%61;inverses=(sm.recurse(inverses[0],refs[index]['private'].to(runtime.device)),inverses[1])
                eigen=[torch.linalg.eigvalsh(x) for x in inverses]
                assert all(float(x.min())>0 and float(x.max())<=1.+1e-8 for x in eigen)
                c.write(RUN/'private/geometry_diagnostics'/f'{arm}_{owner}.json',dict(owner=owner,private_trace=float(inverses[0].trace()),shared_trace=float(inverses[1].trace()),minimum_eigenvalues=[float(x.min()) for x in eigen],symmetry_errors=[float((x-x.T).abs().max()) for x in inverses],lock=binding()))
                state.update(completed_edits=owner,active_owner=owner+1,completed_step=0,boundary=None);save_state()
            verify();weight.requires_grad_(False)
            assert state['committed_updates']==1280 and len(list(events.glob('*.json')))==1280
            assert not any(p.requires_grad or p.grad is not None for p in runtime.model.parameters())
            dest=RUN/'private/candidates'/f'{arm}.pt'
            if not dest.exists():c.save(dest,dict(weight=weight.detach().cpu(),schema=schema,parameter_count=parameters,layer=v1.LAYER,lock=binding()))
            else:
                saved=torch.load(dest,map_location='cpu',weights_only=True);assert saved['lock']==binding() and saved['schema']==schema and torch.equal(saved['weight'],weight.detach().cpu())
            done(f'TRAIN_COMPLETE_{arm}.json',ordered_edits=8,committed_updates=1280,trace_updates=1280,diagnostic_events=1280,parameters_before=parameters,parameters_after=parameters,added_parameters=0,noneditable_versions_verified=True,history_replay=False)
        finally:h.remove();persist()


def evaluate():
    import sys
    assert 'diagnostic_math' not in sys.modules
    # Reuse the already verified native loader; its globals address this new run only.
    import native_eval
    native_eval.ARMS=ARMS;native_eval.GPUS=GPUS;v1.ARMS=ARMS;v1.GPUS=GPUS
    native_eval.evaluate()


def retire_active(arm):
    assert is_done(f'TRAIN_COMPLETE_{arm}.json')
    assert (RUN/'private/candidates'/f'{arm}.pt').is_file()
    path=RUN/'private/active'/f'{arm}.pt';receipt=f'ACTIVE_RETIRED_{arm}.json'
    if is_done(receipt):return
    if not is_done(f'ACTIVE_RETIRE_REQUEST_{arm}.json'):done(f'ACTIVE_RETIRE_REQUEST_{arm}.json',path=str(path),bytes=path.stat().st_size,consumers=['committed_bound_candidate','train_receipt','prefixes','step_journal'])
    assert path.resolve().is_relative_to(RUN.resolve()) and not path.is_symlink()
    if path.exists():path.unlink()
    done(receipt,bytes=c.read(RUN/'private'/f'ACTIVE_RETIRE_REQUEST_{arm}.json')['bytes'],retained_copy=False)


def controller():
    import pipeline
    with j.exclusive(RUN/'private/locks/controller.lock'):
        if is_done('CONTROLLER_COMPLETE.json'):return
        def stage(action,part,receipt):
            if is_done(receipt):return
            start=RUN/'private'/f'START_CHAIN_{action}_{part}.json'
            if start.exists() and j.alive(c.read(start)):child=None
            else:
                with c.resources() as ledger:
                    for s in ledger['gpu_sessions']:
                        if not s.get('ended_epoch') and s['action'] in ('scope_admission','scope_train','scope_eval') and not j.alive(s):
                            stop=time.time();s.update(ended_epoch=stop,resident_seconds=stop-s['started_epoch'],abnormal_exit=True);ledger['gpu_seconds_used']+=s['resident_seconds']
                child=pipeline.launch('run.py',action,5,part)
            while not is_done(receipt):
                c.budget();assert j.alive(c.read(start)),'Owned stage stopped; inspect evidence before engineering resume';time.sleep(5)
            if child is not None:assert child.wait()==0
            else:
                while j.alive(c.read(start)):time.sleep(1)
        stage('scope_admission',0,'MECHANICAL_COMPLETE.json')
        for part,arm in enumerate(ARMS):stage('scope_train',part,f'TRAIN_COMPLETE_{arm}.json');retire_active(arm)
        if not is_done('CANDIDATES_FROZEN.json'):done('CANDIDATES_FROZEN.json',candidates=5,selection=False)
        for part,arm in enumerate(ARMS):stage('scope_eval',part,f'EVAL_COMPLETE_{arm}.json')
        if not is_done('GENERATION_COMPLETE.json'):
            paths=[p for p in (RUN/'private/outputs').glob('*/*/*.json') if p.parts[-3]!='BASE_IDENTITY'];assert len(paths)==2147 and all(c.read(p)['lock']==binding() for p in paths)
            for arm in ARMS:
                for row in v1.rows():
                    if row['audit_role'] in ('NATIVE','GFIT'):assert c.read(v1.output(arm,row['owner'],row))['R0']==c.read(v1.output(f'{arm}_PREFIX_8',row['owner'],row))['R0']
            attempts=[c.read(p) for p in (RUN/'private/attempts').glob('*.json')]
            assert sum(len(list((RUN/'private/training_diagnostics'/a).glob('*.json'))) for a in ARMS)==6400
            done('GENERATION_COMPLETE.json',outputs=2147,committed_updates=6400,backwards=sum(x['backwards'] for x in attempts),counts=attempts,new_generations=sum(x['generations'] for x in attempts),inherited_Base=267,mechanical_temporary_updates=5)
        if not is_done('DELETION.json'):
            if not is_done('DELETION_REQUEST.json'):
                paths=[RUN/'private/candidates'/f'{a}.pt' for a in ARMS]+[RUN/'private/GEOMETRY.pt']+list((RUN/'private/cache').glob('*/*.pt'))
                assert all(p.is_file() and not p.is_symlink() and p.resolve().is_relative_to(RUN.resolve()) for p in paths)
                done('DELETION_REQUEST.json',paths=list(map(str,paths)),files=len(paths),bytes=sum(p.stat().st_size for p in paths),retained_copy=False,diagnostics_retained=True,consumers=['all5train','allprefix','all5final','native_reload','durable_Judge_inputs'])
            request=c.read(RUN/'private/DELETION_REQUEST.json')
            for value in request['paths']:
                path=Path(value);assert path.resolve().is_relative_to(RUN.resolve()) and not path.is_symlink()
                if path.exists():path.unlink()
            done('DELETION.json',**{k:v for k,v in request.items() if k not in ('lock','epoch')})
        done('CONTROLLER_COMPLETE.json',status='GPU_CHAIN_COMPLETE_SCORING_PENDING')


if __name__=='__main__':
    try:
        {'scope_plan':plan,'scope_admission':admission,'scope_train':train,'scope_eval':evaluate,'scope_controller':controller}[os.environ['ACTION']]()
    except BaseException as error:
        c.write(RUN/'private/failures'/f'{os.environ.get("ACTION")}_{os.getpid()}_{time.time_ns()}.json',dict(error=repr(error),traceback=traceback.format_exc(),epoch=time.time()));raise
