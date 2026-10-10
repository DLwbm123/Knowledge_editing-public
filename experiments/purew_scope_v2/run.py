"""Fixed two-arm pure-W ScopeEdit-core experiment, isolated from round one."""
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
ARMS = ('GATED', 'ALWAYS_ON')
GPUS = (3, 4)
v1.ARMS = ARMS
v1.GPUS = GPUS
c.write = j.write


def binding(): return c.digest(c.read(RUN/'private/LOCK.json'))
def key(row): return str(row['owner']) + ':' + row['query_id']
def done(name, **value):
    c.write(RUN/'private'/name,dict(value,lock=binding(),epoch=time.time()))
def is_done(name): return j.completed(RUN/'private'/name,binding())


def plan():
    import scope_math as sm
    if (RUN/'private/LOCK.json').exists() and (RUN/'private/PLAN_COMPLETE.json').exists():
        assert c.read(RUN/'private/LOCK.json')['epoch']=='PUREW_SCOPE_V2_20261010'
        assert c.read(RUN/'private/LOCK.json')==c.read(RUN/'public/PROTOCOL.json')
        assert c.read(RUN/'private/LOCK.json')==c.read(RUN/'private/FROZEN_PROTOCOL.json')
        return
    if (RUN/'private/LOCK.json').exists():
        assert c.read(RUN/'private/LOCK.json')['epoch']=='PUREW_SCOPE_V2_20261010'
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
    import scope_math as sm
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
                # Same real gradient, two scope proposals; both native states are restored.
                subset=items[:7];base=weight.detach().clone();weight.requires_grad_(True)
                proposals={arm:(torch.zeros(weight.shape[0],448,device=runtime.device),
                                torch.zeros(weight.shape[0],64,device=runtime.device)) for arm in ARMS}
                gammas={arm:[sm.gate(cache[key(x[0])]['text'].to(runtime.device),
                                    cache[key(x[0])]['image'].to(runtime.device),arm=='ALWAYS_ON')[0]
                             for x in subset] for arm in ARMS}
                pp=torch.eye(448,device=runtime.device);ps=torch.eye(64,device=runtime.device)
                for i,fraction in [(0,.25),(1,.25),(5,.5)]:
                    batch=old.batch_on(runtime,subset[i][1]);meta[0]=(batch.labels,subset[i][5]);weight.grad=None
                    logits,labels=target_logits(runtime.model,batch)
                    count['backwards']+=1;persist();(fraction*F.cross_entropy(logits,labels)).backward()
                    for arm in ARMS:
                        parts=sm.coordinates(weight.grad,bases,(pp,ps),gammas[arm][i])
                        for total,part in zip(proposals[arm],parts):total.add_(part)
                meta[0]=None;pb=data.q.protection_batch(runtime,basis[0],data.tasks()[0]);weight.grad=None
                ref=torch.load(RUN/'private/cache/protection/0.pt',map_location='cpu',weights_only=True)['reference'].to(runtime.device)
                logits,_=target_logits(runtime.model,pb);kl=preservation_kl(logits,ref)
                assert abs(float(kl.detach()))<1e-5
                count['backwards']+=1;persist();kl.backward()
                weighted={arm:sum(f*gammas[arm][i] for i,f in [(0,.25),(1,.25),(5,.5)]) for arm in ARMS}
                for arm in ARMS:
                    parts=sm.coordinates(weight.grad,bases,(pp,ps),weighted[arm])
                    for total,part in zip(proposals[arm],parts):total.add_(part)
                reports=[]
                before={n:p._version for n,p in runtime.model.named_parameters() if p is not weight}
                for arm in ARMS:
                    gamma=weighted[arm]
                    step,receipt=sm.clipped_step(proposals[arm])
                    merged=sm.merged(base,step,bases)
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
                    assert torch.equal(merged,sm.merged(base,tuple(x.clone() for x in step),bases))
                    reports.append(dict(arm=arm,gamma=gamma,actual_step_norm=float((weight-base).norm()),**receipt))
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
    from purew_math import target_logits,preservation_kl
    import scope_math as sm
    part=int(os.environ['PARTITION']);arm=ARMS[part];gpu=GPUS[part]
    if is_done(f'TRAIN_COMPLETE_{arm}.json'):return
    assert is_done('MECHANICAL_COMPLETE.json')
    c.available=v1.admission
    with j.exclusive(RUN/'private/locks'/f'{arm}.lock'),c.lease(gpu):
        runtime,bindings=c.load(gpu);count,h,persist=tracked(runtime,'train_'+arm)
        try:
            v1.fp32_native(runtime);weight=runtime.get_module(v1.LAYER).weight
            base=weight.detach().clone();schema=v1.state_schema(runtime.model);identity=id(weight)
            parameters_before=sum(p.numel() for p in runtime.model.parameters())
            assert schema==c.read(RUN/'private/BASE_COMPLETE.json')['schema']
            assert parameters_before==c.read(RUN/'private/BASE_COMPLETE.json')['parameters']==7566219264
            frozen={n:p._version for n,p in runtime.model.named_parameters() if p is not weight}
            buffers={n:b.detach().clone() for n,b in runtime.model.named_buffers()}
            geom=torch.load(RUN/'private/GEOMETRY.pt',map_location='cpu',weights_only=True);assert geom['lock']==binding()
            bases=tuple(x.to(runtime.device) for x in geom['bases'])
            prepared=source_items(runtime,bindings)
            features={key(x[0]):torch.load(RUN/'private/cache/source'/f"{x[0]['owner']}_{x[0]['query_id']}.pt",map_location='cpu',weights_only=True) for x in prepared}
            for f in features.values():assert f['lock']==binding()
            protected=[v1.cpu_batch(data.q.protection_batch(runtime,row,data.tasks()[0])) for row in data.q.split()[0]]
            refs=[torch.load(RUN/'private/cache/protection'/f'{i}.pt',map_location='cpu',weights_only=True) for i in range(61)]
            assert all(r['lock']==binding() for r in refs)
            active=RUN/'private/active'/f'{arm}.pt';diagnostics=RUN/'private/training_diagnostics'/arm
            state=dict(coefficients=tuple(torch.zeros(weight.shape[0],a.shape[0]) for a in bases),
                       inverses=tuple(torch.eye(a.shape[0],dtype=torch.float64) for a in bases),completed_edits=0,
                       active_owner=1,completed_step=0,committed_updates=0,last_event=None,lock=binding())
            if active.exists():state=torch.load(active,map_location='cpu',weights_only=True)
            assert state['lock']==binding();j.ensure_event(diagnostics,state)
            coefficients=tuple(x.to(runtime.device) for x in state['coefficients'])
            inverses=tuple(x.to(runtime.device) for x in state['inverses'])
            with torch.no_grad():weight.copy_(sm.merged(base,coefficients,bases))
            weight.requires_grad_(True)
            def verify():
                assert id(weight)==identity and v1.state_schema(runtime.model)==schema
                assert frozen=={n:p._version for n,p in runtime.model.named_parameters() if p is not weight}
                assert all(torch.equal(v,buffers[n]) for n,v in runtime.model.named_buffers())
                assert [n for n,p in runtime.model.named_parameters() if p.requires_grad]==[v1.LAYER+'.weight']
                assert not runtime.get_module(v1.LAYER)._forward_pre_hooks and not runtime.get_module(v1.LAYER)._forward_hooks
            def save_state():
                state['coefficients']=tuple(x.cpu() for x in coefficients)
                state['inverses']=tuple(x.cpu() for x in inverses)
                c.save(active,state)
                # Persist completed numeric state before publishing its diagnostic commit.
                with active.open('rb') as f:os.fsync(f.fileno())
                fd=os.open(active.parent,os.O_RDONLY)
                try:os.fsync(fd)
                finally:os.close(fd)
                j.ensure_event(diagnostics,state)
            for task in data.tasks():
                owner=task['order']
                if owner<=state['completed_edits']:continue
                assert state['active_owner']==owner
                items=[x for x in prepared if x[0]['owner']==owner];assert len(items)==7
                batches=[old.batch_on(runtime,x[1]) for x in items]
                gates=[sm.gate(features[key(x[0])]['text'].to(runtime.device),features[key(x[0])]['image'].to(runtime.device),arm=='ALWAYS_ON')[0] for x in items]
                for step in range(state['completed_step']+1,161):
                    c.budget();verify()
                    directions=tuple(torch.zeros_like(x) for x in coefficients);losses={}
                    selected=[(0,.25),(1+(step-1)%4,.25),(5+(step-1)%2,.5)]
                    gamma=sum(f*gates[i] for i,f in selected)
                    for i,fraction in selected:
                        weight.grad=None;logits,labels=target_logits(runtime.model,batches[i])
                        loss=F.cross_entropy(logits,labels);assert torch.isfinite(loss)
                        count['backwards']+=1;persist();(fraction*loss).backward()
                        parts=sm.coordinates(weight.grad,bases,inverses,gates[i])
                        for total,part in zip(directions,parts):total.add_(part)
                        losses[str(i)]=float(loss.detach());del logits,loss,parts
                    index=((owner-1)*160+step-1)%61
                    weight.grad=None;pb=old.batch_on(runtime,protected[index])
                    logits,_=target_logits(runtime.model,pb)
                    kl=preservation_kl(logits,refs[index]['reference'].to(runtime.device));assert torch.isfinite(kl)
                    count['backwards']+=1;persist();kl.backward()
                    parts=sm.coordinates(weight.grad,bases,inverses,gamma)
                    for total,part in zip(directions,parts):total.add_(part)
                    losses['BASIS_KL']=float(kl.detach());del logits,kl,parts,pb
                    assert all(p.grad is None for p in runtime.model.parameters() if p is not weight)
                    delta,receipt=sm.clipped_step(directions)
                    coefficients=tuple(x+d for x,d in zip(coefficients,delta))
                    current=sm.merged(base,coefficients,bases)
                    actual_norm=float((current-weight.detach()).norm())
                    assert torch.isfinite(current).all() and actual_norm<=.0501
                    with torch.no_grad():weight.copy_(current)
                    weight.grad=None;del current,delta,directions
                    state.update(completed_step=step,committed_updates=state['committed_updates']+1)
                    event=dict(update=state['committed_updates'],owner=owner,step=step,losses=losses,
                               source_indices=[i for i,f in selected],source_gammas=[gates[i] for i,f in selected],
                               weighted_gamma=gamma,actual_step_norm=actual_norm,**receipt)
                    c.write(RUN/'private/step_attempts'/arm/f'{event["update"]:04d}_{os.getpid()}_{time.time_ns()}.json',event)
                    state['last_event']=event;save_state();count['updates']+=1;persist()
                    if step%20==0:
                        c.write(RUN/'private/progress'/f'{arm}.json',dict(event,committed_updates=state['committed_updates'],counts=count))
                        print('TRAIN',arm,owner,step,actual_norm,gamma,flush=True)
                verify()
                for item in prepared:
                    row=item[0]
                    if row['owner']<=owner and row['audit_role'] in ('NATIVE','GFIT'):
                        emit(runtime,item,f'{arm}_PREFIX_{owner}',count,persist)
                # Only previous edits influence this edit's writes. Admit its keys for the NEXT edit.
                fractions=(.25,.0625,.0625,.0625,.0625,.25,.25)
                for item,fraction,gamma_i in zip(items,fractions,gates):
                    f=features[key(item[0])]
                    inverses=(sm.recurse(inverses[0],f['private'].to(runtime.device),fraction),
                              sm.recurse(inverses[1],f['shared'].to(runtime.device),fraction*gamma_i))
                locality_index=((owner-1)*160)%61
                inverses=(sm.recurse(inverses[0],refs[locality_index]['private'].to(runtime.device)),inverses[1])
                eigenvalues=[torch.linalg.eigvalsh(p) for p in inverses]
                assert all(float(v.min())>0 and float(v.max())<=1.+1e-8 for v in eigenvalues)
                c.write(RUN/'private/geometry_diagnostics'/f'{arm}_{owner}.json',dict(
                    owner=owner,source_gammas=gates,locality_index=locality_index,
                    private_trace=float(inverses[0].trace()),shared_trace=float(inverses[1].trace()),
                    minimum_eigenvalues=[float(v.min()) for v in eigenvalues],
                    symmetry_errors=[float((p-p.T).abs().max()) for p in inverses],lock=binding()))
                state.update(completed_edits=owner,active_owner=owner+1,completed_step=0);save_state()
            verify();weight.requires_grad_(False)
            assert state['committed_updates']==1280 and not any(p.requires_grad or p.grad is not None for p in runtime.model.parameters())
            events=[c.read(p) for p in sorted(diagnostics.glob('*.json'))]
            assert len(events)==1280 and [e['update'] for e in events]==list(range(1,1281))
            dest=RUN/'private/candidates'/f'{arm}.pt'
            if not dest.exists():c.save(dest,dict(weight=weight.detach().cpu(),schema=schema,
                parameter_count=sum(p.numel() for p in runtime.model.parameters()),layer=v1.LAYER,lock=binding()))
            else:
                saved=torch.load(dest,map_location='cpu',weights_only=True)
                assert saved['lock']==binding() and saved['schema']==schema and saved['layer']==v1.LAYER
                assert saved['parameter_count']==sum(p.numel() for p in runtime.model.parameters())
                assert torch.equal(saved['weight'],weight.detach().cpu())
            done(f'TRAIN_COMPLETE_{arm}.json',status='TRAINED_NOT_EVALUATED',ordered_edits=8,
                 trace_updates=1280,committed_updates=1280,diagnostic_events=1280,counts=count,
                 trainable_parameters=weight.numel(),parameters_before=parameters_before,
                 parameters_after=sum(p.numel() for p in runtime.model.parameters()),
                 added_parameters=0,noneditable_versions_verified=True,history_replay=False,
                 deployment='original W only',fixed_dual_geometry=True)
        finally:h.remove();persist()


def evaluate():
    part=int(os.environ['PARTITION']);arm=ARMS[part];gpu=GPUS[part]
    if is_done(f'EVAL_COMPLETE_{arm}.json'):return
    assert is_done('CANDIDATES_FROZEN.json') and 'scope_math' not in __import__('sys').modules
    c.available=v1.admission
    with j.exclusive(RUN/'private/locks'/f'{arm}.lock'),c.lease(gpu):
        runtime,bindings=c.load(gpu);count,h,persist=tracked(runtime,'eval_'+arm)
        try:
            v1.fp32_native(runtime);items=v1.prepare(runtime,bindings,v1.rows())
            saved=torch.load(RUN/'private/candidates'/f'{arm}.pt',map_location='cpu',weights_only=True)
            assert saved['lock']==binding() and v1.state_schema(runtime.model)==saved['schema']
            assert sum(p.numel() for p in runtime.model.parameters())==saved['parameter_count']
            with torch.no_grad():runtime.get_module(saved['layer']).weight.copy_(saved['weight'].to(runtime.device))
            assert not any(p.requires_grad for p in runtime.model.parameters())
            assert not runtime.get_module(v1.LAYER)._forward_pre_hooks and not runtime.get_module(v1.LAYER)._forward_hooks
            item=next(x for x in items if x[0]['owner']==8 and x[0]['audit_role']=='NATIVE')
            probe=emit(runtime,item,f'{arm}_RELOAD',count,persist)
            assert c.read(probe)['R0']==c.read(v1.output(f'{arm}_PREFIX_8',8,item[0]))['R0']
            for item in items:emit(runtime,item,arm,count,persist)
            done(f'EVAL_COMPLETE_{arm}.json',outputs=267,counts=count,native_fresh_reload_tokens_equal=True,
                 added_parameters=0,editor_math_imported=False,inference_structure_unchanged=True)
        finally:h.remove();persist()


def controller():
    import pipeline
    with j.exclusive(RUN/'private/locks/controller.lock'):
        if is_done('CONTROLLER_COMPLETE.json'):return
        def stage(action,gpu,part,receipt):
            if is_done(receipt):return None
            start=RUN/'private'/f'START_CHAIN_{action}_{part}.json'
            if start.exists() and j.alive(c.read(start)):
                return dict(receipt=receipt,start=c.read(start))
            # Dead owned leases do not grant permission to touch any other GPU process.
            with c.resources() as ledger:
                for session in ledger['gpu_sessions']:
                    if session.get('ended_epoch') or session['action'] not in ('scope_admission','scope_train','scope_eval'):continue
                    if not j.alive(session):
                        stop=time.time();session.update(ended_epoch=stop,resident_seconds=stop-session['started_epoch'],abnormal_exit=True)
                        ledger['gpu_seconds_used']+=session['resident_seconds']
            child=pipeline.launch('run.py',action,gpu,part)
            return dict(receipt=receipt,start=c.read(start),child=child)
        def wait(jobs):
            while jobs:
                remaining=[]
                for job in jobs:
                    if is_done(job['receipt']):
                        if 'child' in job:assert job['child'].wait()==0
                        else:
                            while j.alive(job['start']):time.sleep(1)
                        continue
                    assert j.alive(job['start']),'Owned stage stopped; retain evidence and resume unfinished stage after diagnosis'
                    remaining.append(job)
                jobs=remaining
                if jobs:c.budget();time.sleep(5)
        wait([x for x in [stage('scope_admission',GPUS[0],0,'MECHANICAL_COMPLETE.json')] if x])
        wait([x for x in [stage('scope_train',gpu,part,f'TRAIN_COMPLETE_{arm}.json') for part,(arm,gpu) in enumerate(zip(ARMS,GPUS))] if x])
        if not is_done('CANDIDATES_FROZEN.json'):done('CANDIDATES_FROZEN.json',candidates=2,selection=False)
        wait([x for x in [stage('scope_eval',gpu,part,f'EVAL_COMPLETE_{arm}.json') for part,(arm,gpu) in enumerate(zip(ARMS,GPUS))] if x])
        if not is_done('GENERATION_COMPLETE.json'):
            paths=[p for p in (RUN/'private/outputs').glob('*/*/*.json') if p.parts[-3]!='BASE_IDENTITY']
            assert len(paths)==1019 and all(c.read(p)['lock']==binding() for p in paths)
            for arm in ARMS:
                for row in v1.rows():
                    if row['audit_role'] in ('NATIVE','GFIT'):
                        assert c.read(v1.output(arm,row['owner'],row))['R0']==c.read(v1.output(f'{arm}_PREFIX_8',row['owner'],row))['R0']
            attempts=[c.read(p) for p in (RUN/'private/attempts').glob('*.json')]
            done('GENERATION_COMPLETE.json',status='GENERATED_SCORING_PENDING',outputs=1019,
                 committed_updates=2560,updates=2560,backwards=sum(x['backwards'] for x in attempts),
                 counts=attempts,new_generations=sum(x['generations'] for x in attempts),inherited_Base=267,
                 mechanical_temporary_updates=2)
            c.write(RUN/'public/PROGRESS.json',dict(status='GENERATED_SCORING_PENDING',training_complete=True,
                    scoring_complete=False,publication_complete=False))
        # Small diagnostics remain; only owned consumed matrices/reference caches retire.
        if not is_done('DELETION.json'):
            if not is_done('DELETION_REQUEST.json'):
                retired=[RUN/'private'/kind/f'{arm}.pt' for kind in ('active','candidates') for arm in ARMS]
                retired+=[RUN/'private/GEOMETRY.pt']+list((RUN/'private/cache').glob('*/*.pt'))
                assert all(p.is_file() and not p.is_symlink() and p.resolve().is_relative_to(RUN.resolve()) for p in retired)
                done('DELETION_REQUEST.json',paths=list(map(str,retired)),files=len(retired),bytes=sum(p.stat().st_size for p in retired),
                     retained_copy=False,reconstruction='rerun frozen sequence; no deployed matrix retained',
                     diagnostics_retained=True,consumers=['training','prefix','final','native_reload','durable_Judge_outputs'])
            request=c.read(RUN/'private/DELETION_REQUEST.json')
            for value in request['paths']:
                path=Path(value)
                assert path.resolve().is_relative_to(RUN.resolve()) and not path.is_symlink()
                if path.exists():path.unlink()
            done('DELETION.json',**{k:v for k,v in request.items() if k not in ('lock','epoch')})
        done('CONTROLLER_COMPLETE.json',status='GPU_CHAIN_COMPLETE_SCORING_PENDING')


if __name__=='__main__':
    try:
        {'scope_plan':plan,'scope_admission':admission,'scope_train':train,
         'scope_eval':evaluate,'scope_controller':controller}[os.environ['ACTION']]()
    except BaseException as error:
        c.write(RUN/'private/failures'/f'{os.environ.get("ACTION")}_{os.getpid()}_{time.time_ns()}.json',
                dict(error=repr(error),traceback=traceback.format_exc(),epoch=time.time()))
        raise
