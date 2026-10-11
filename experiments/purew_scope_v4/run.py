"""Frozen eight-edit pure-W comparison; C is inherited, only U/H train."""
import os,json,time,traceback
from pathlib import Path
import torch
import torch.nn.functional as F
import base_run as v1
import journal as j
c,RUN,data,old=v1.c,v1.RUN,v1.data,v1.old
PARENT=Path(os.environ['PUREW_PARENT'])
REFERENCE=Path(os.environ['REFERENCE_PARENT'])
ARMS=('C','U','H');NEW_ARMS=('U','H');GPUS=(5,5,5)
v1.ARMS=ARMS;v1.GPUS=GPUS;c.write=j.write

def binding():return c.digest(c.read(RUN/'private/LOCK.json'))
def key(row):return str(row['owner'])+':'+row['query_id']
def done(name,**value):c.write(RUN/'private'/name,dict(value,lock=binding(),epoch=time.time()))
def is_done(name):return j.completed(RUN/'private'/name,binding())

def plan():
    if is_plan_complete():return
    oldlock=c.read(REFERENCE/'private/LOCK.json')
    frozen=c.read(RUN/'private/FROZEN_PROTOCOL.json')
    assert frozen['epoch']=='PUREW_SCOPE_V4_20261011'
    for field in ('supervision','evaluation','source_binding','anchors_binding'):
        if field in oldlock:assert frozen[field]==oldlock[field]
    if not (RUN/'private/LOCK.json').exists():v1.plan()
    initial=c.read(RUN/'private/LOCK.json')
    for field in ('source_binding','anchors_binding','BASIS_binding','HELD_binding'):
        if field in initial:assert frozen[field]==initial[field]
    assert c.read(REFERENCE/'private/ROWS.json')==v1.rows()==c.read(PARENT/'private/ROWS.json')
    assert c.read(REFERENCE/'private/REPORT_COMPLETE.json')['status'] in ('COMPLETE','INCOMPLETE_SCORING')
    c.write(RUN/'private/LOCK.json',frozen);c.write(RUN/'public/PROTOCOL.json',frozen)
    ref=frozen['norm_reference'];events=sorted([c.read(p) for p in (REFERENCE/'private/training_diagnostics/C_FULL_FIXED').glob('*.json')],key=lambda e:e['update'])
    assert len(events)==1280
    assert ref['source_commit']==c.read(REFERENCE/'private/GPU_SOURCE_VERSION.json')['commit']
    assert ref['steps']==[dict(update=e['update'],owner=e['owner'],step=e['step'],norm=e['actual_step_norm']) for e in events]
    assert all((e['update'],e['owner'],e['step'])==(i+1,i//160+1,i%160+1) for i,e in enumerate(events))
    for e in events:j.ensure_event(RUN/'private/training_diagnostics/C',dict(last_event=e))
    oldbinding=c.digest(oldlock)
    for path in sorted((REFERENCE/'private/outputs').glob('*/*/*.json')):
        prior=c.read(path);stage=prior['stage']
        if stage!='BASE' and not (stage=='C_FULL_FIXED' or stage.startswith('C_FULL_FIXED_')):continue
        assert prior['lock']==oldbinding
        currentstage=stage.replace('C_FULL_FIXED','C',1)
        row=prior['binding']['input'];assert row in v1.rows()
        dst=v1.output(currentstage,prior['owner'],row)
        inherited=dict(prior,stage=currentstage,lock=binding(),inherited_generation=dict(path=str(path),execution=prior['execution'],lock=prior['lock']))
        if dst.exists():assert c.read(dst)==inherited
        else:c.write(dst,inherited)
    for src,dst in [('BASE_COMPLETE.json','BASE_COMPLETE.json'),('TRAIN_COMPLETE_C_FULL_FIXED.json','TRAIN_COMPLETE_C.json'),('EVAL_COMPLETE_C_FULL_FIXED.json','EVAL_COMPLETE_C.json')]:
        d=c.read(REFERENCE/'private'/src);d.pop('lock');d.pop('epoch');d.update(inherited=True,new_updates=0,new_generations=0,source_round=3)
        done(dst,**d)
    done('PLAN_COMPLETE.json',status='PASS',inherited_Base=267,inherited_C=376)
    c.write(RUN/'public/ADMISSION.json',dict(status='FROZEN_DATA_AND_REFERENCE_ADMITTED_NOT_GPU_RESULTS',added_parameters=0,ordered_edits=8,new_arms=list(NEW_ARMS)))

def is_plan_complete():
    if not (RUN/'private/PLAN_COMPLETE.json').exists():return False
    assert c.read(RUN/'private/LOCK.json')==c.read(RUN/'private/FROZEN_PROTOCOL.json')==c.read(RUN/'public/PROTOCOL.json')
    return is_done('PLAN_COMPLETE.json')

def source_items(runtime,bindings):
    selected=[r for r in v1.rows() if r['audit_role'] in ('NATIVE','FIT','GFIT')]
    items=v1.prepare(runtime,bindings,selected,False)
    tasks={t['order']:t for t in data.tasks()}
    result=[]
    for row,_,raw,expanded,bind in items:
        batch=runtime.build_edit_batch(data.record(tasks[row['owner']],row))
        result.append((row,v1.cpu_batch(batch),raw,expanded,bind,(batch.image_token_start,batch.image_token_end)))
    assert all(x[0] in c.read(REFERENCE/'private/ROWS.json') for x in result)
    assert c.read(REFERENCE/'private/MECHANICAL_COMPLETE.json')['status']=='PASS'
    return result


def tracked(runtime,name):
    count=dict(forwards=0,backwards=0,updates=0,generations=0,generation_attempts=0)
    path=RUN/'private/attempts'/f'{name}_{os.getpid()}_{time.time_ns()}.json'
    def persist():
        c.write(path,dict(count,action=name,pid=os.getpid(),lock=binding()))
        attempts=[c.read(p) for p in (RUN/'private/attempts').glob('*.json')]
        assert sum(x['backwards'] for x in attempts)<=12490,'Frozen backward budget exhausted; retain state'
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


def admitted_history(prepared,owner,step):
    import shared_math as sm
    # Only parent-bound training rows can receive the admission annotation.
    rows=[dict(x[0],source_admission='PASS') for x in prepared]
    assert all(x[0]['audit_role'] in ('NATIVE','FIT','GFIT') for x in prepared)
    return [next(x for x in prepared if key(x[0])==key(r)) for r in sm.history(rows,owner,step-1)]


def admission():
    from purew_math import target_logits,preservation_kl
    import shared_math as sm
    if is_done('MECHANICAL_COMPLETE.json'):return
    c.available=v1.admission
    with c.lease(5):
        runtime,bindings=c.load(5);count,h,persist=tracked(runtime,'admission')
        try:
            v1.fp32_native(runtime);weight=runtime.get_module(v1.LAYER).weight
            items=source_items(runtime,bindings);base=weight.detach().clone()
            before={n:p._version for n,p in runtime.model.named_parameters() if p is not weight}
            basis=data.q.split()[0];assert len(basis)==61
            for index,row in enumerate(basis):
                path=RUN/'private/cache/protection'/f'{index}.pt'
                if path.exists():assert torch.load(path,map_location='cpu',weights_only=True)['lock']==binding();continue
                batch=data.q.protection_batch(runtime,row,data.tasks()[0])
                with torch.no_grad():logits,_=target_logits(runtime.model,batch)
                c.save(path,dict(reference=logits.cpu(),lock=binding()))
            reports=[]
            for arm,owner in [('U',1),('H',2)]:
                weight.requires_grad_(True);gradient=torch.zeros_like(weight)
                subset=[x for x in items if x[0]['owner']==owner]
                selected=[(subset[i],f) for i,f in [(0,.25),(1,.25),(5,.5)]]
                if arm=='H':selected += [(x,.25) for x in admitted_history(items,owner,1)]
                for item,fraction in selected:
                    weight.grad=None;logits,labels=target_logits(runtime.model,old.batch_on(runtime,item[1]))
                    count['backwards']+=1;persist();(fraction*F.cross_entropy(logits,labels)).backward();gradient.add_(weight.grad)
                weight.grad=None;pb=data.q.protection_batch(runtime,basis[0],data.tasks()[0])
                ref=torch.load(RUN/'private/cache/protection/0.pt',map_location='cpu',weights_only=True)['reference'].to(runtime.device)
                logits,_=target_logits(runtime.model,pb);kl=preservation_kl(logits,ref);assert abs(float(kl.detach()))<1e-5
                count['backwards']+=1;persist();kl.backward();gradient.add_(weight.grad)
                feature={}
                def capture(_,args):feature['value']=args[0].detach().clone()
                observer=runtime.get_module(v1.LAYER).register_forward_pre_hook(capture)
                try:
                    batch=old.batch_on(runtime,subset[0][1])
                    with torch.no_grad():target_logits(runtime.model,batch)
                    original_feature=feature['value']
                    target=c.read(RUN/'private/LOCK.json')['norm_reference']['steps'][(owner-1)*160]['norm']
                    delta=sm.shared_step(gradient,target);merged,actual=sm.apply_step(base,delta,target)
                    with torch.no_grad():weight.copy_(merged);target_logits(runtime.model,batch)
                    assert torch.equal(feature['value'],original_feature) and not torch.equal(weight,base)
                    assert before=={n:p._version for n,p in runtime.model.named_parameters() if p is not weight}
                    reports.append(dict(arm=arm,owner=owner,real_gradient_norm=float(gradient.norm()),shared_target_norm=target,actual_step_norm=actual,history_rows=len(selected)-3))
                finally:
                    observer.remove()
                    with torch.no_grad():weight.copy_(base)
                    assert torch.equal(weight,base);weight.grad=None;weight.requires_grad_(False)
            for owner in (1,8):
                item=next(x for x in items if x[0]['owner']==owner and x[0]['audit_role']=='NATIVE')
                probe=emit(runtime,item,'BASE_IDENTITY',count,persist)
                assert c.read(probe)['R0']==c.read(v1.output('BASE',owner,item[0]))['R0']
            assert v1.state_schema(runtime.model)==c.read(RUN/'private/BASE_COMPLETE.json')['schema']
            assert not any(p.requires_grad or p.grad is not None for p in runtime.model.parameters())
            done('MECHANICAL_COMPLETE.json',status='PASS',reports=reports,exact_native_Base_identity=True,feature_invariance_exact=True,
                 parameters=sum(p.numel() for p in runtime.model.parameters()),added_parameters=0,counts=count)
            c.write(RUN/'public/MECHANICAL_COMPLETE.json',c.read(RUN/'private/MECHANICAL_COMPLETE.json'))
            print('MECHANICAL_PASS',reports,flush=True)
        finally:h.remove();persist()


def train():
    from purew_math import target_logits,preservation_kl,row_basis
    import shared_math as sm
    part=int(os.environ['PARTITION']);arm=ARMS[part];assert arm in NEW_ARMS
    if is_done(f'TRAIN_COMPLETE_{arm}.json'):return
    assert is_done('MECHANICAL_COMPLETE.json');c.available=v1.admission
    with j.exclusive(RUN/'private/locks'/f'{arm}.lock'),c.lease(5):
        runtime,bindings=c.load(5);count,h,persist=tracked(runtime,'train_'+arm)
        try:
            v1.fp32_native(runtime);weight=runtime.get_module(v1.LAYER).weight;base=weight.detach().clone();q0=row_basis(base)
            schema=v1.state_schema(runtime.model);identity=id(weight);parameters=sum(p.numel() for p in runtime.model.parameters())
            assert parameters==7566219264 and schema==c.read(RUN/'private/BASE_COMPLETE.json')['schema']
            frozen={n:p._version for n,p in runtime.model.named_parameters() if p is not weight}
            buffers={n:b.detach().clone() for n,b in runtime.model.named_buffers()}
            prepared=source_items(runtime,bindings)
            protected=[v1.cpu_batch(data.q.protection_batch(runtime,row,data.tasks()[0])) for row in data.q.split()[0]]
            refs=[torch.load(RUN/'private/cache/protection'/f'{i}.pt',map_location='cpu',weights_only=True) for i in range(61)]
            assert all(x['lock']==binding() for x in refs)
            targets=c.read(RUN/'private/LOCK.json')['norm_reference']['steps'];norm_binding=c.digest(targets)
            active=RUN/'private/active'/f'{arm}.pt';events=RUN/'private/training_diagnostics'/arm
            state=dict(completed_edits=0,active_owner=1,completed_step=0,committed_updates=0,last_event=None,lock=binding())
            if active.exists():
                state=torch.load(active,map_location='cpu',weights_only=True);assert state['lock']==binding()
                with torch.no_grad():weight.copy_(state['weight'].to(runtime.device))
                j.ensure_event(events,state)
            weight.requires_grad_(True)
            def verify():
                assert id(weight)==identity and v1.state_schema(runtime.model)==schema
                assert frozen=={n:p._version for n,p in runtime.model.named_parameters() if p is not weight}
                assert all(torch.equal(v,buffers[n]) for n,v in runtime.model.named_buffers())
                assert [n for n,p in runtime.model.named_parameters() if p.requires_grad]==[v1.LAYER+'.weight']
                assert not runtime.get_module(v1.LAYER)._forward_pre_hooks and not runtime.get_module(v1.LAYER)._forward_hooks
            def save_state():
                state['weight']=weight.detach().cpu();c.save(active,state)
                with active.open('rb') as f:os.fsync(f.fileno())
                fd=os.open(active.parent,os.O_RDONLY)
                try:os.fsync(fd)
                finally:os.close(fd)
                j.ensure_event(events,state)
            for task in data.tasks():
                owner=task['order']
                if owner<=state['completed_edits']:continue
                assert state['active_owner']==owner
                items=[x for x in prepared if x[0]['owner']==owner];assert len(items)==7
                batches=[old.batch_on(runtime,x[1]) for x in items]
                for step in range(state['completed_step']+1,161):
                    c.budget();verify();sample=step in (1,40,80,160)
                    total=torch.zeros_like(weight);losses={};selected=[(0,.25),(1+(step-1)%4,.25),(5+(step-1)%2,.5)]
                    for index,fraction in selected:
                        weight.grad=None;logits,labels=target_logits(runtime.model,batches[index]);loss=F.cross_entropy(logits,labels);assert torch.isfinite(loss)
                        count['backwards']+=1;persist();(fraction*loss).backward();total.add_(weight.grad)
                        losses[str(index)]=float(loss.detach());del logits,loss
                    ce=total.clone() if sample else None
                    index=((owner-1)*160+step-1)%61;weight.grad=None
                    logits,_=target_logits(runtime.model,old.batch_on(runtime,protected[index]));kl=preservation_kl(logits,refs[index]['reference'].to(runtime.device));assert torch.isfinite(kl)
                    count['backwards']+=1;persist();kl.backward();total.add_(weight.grad)
                    kg=weight.grad.clone() if sample else None;losses['BASIS_KL']=float(kl.detach());del logits,kl
                    historical=admitted_history(prepared,owner,step) if arm=='H' else []
                    hg=torch.zeros_like(weight) if sample else None
                    for item in historical:
                        weight.grad=None;logits,labels=target_logits(runtime.model,old.batch_on(runtime,item[1]));loss=F.cross_entropy(logits,labels);assert torch.isfinite(loss)
                        count['backwards']+=1;persist();(.25*loss).backward();total.add_(weight.grad)
                        if sample:hg.add_(weight.grad)
                        losses['HISTORY_'+item[0]['audit_role']]=float(loss.detach());del logits,loss
                    assert all(p.grad is None for p in runtime.model.parameters() if p is not weight)
                    reference=targets[(owner-1)*160+step-1];assert (reference['owner'],reference['step'],reference['update'])==(owner,step,state['committed_updates']+1)
                    delta=sm.shared_step(total,reference['norm']);current,actual=sm.apply_step(weight.detach(),delta,reference['norm'])
                    diagnostics={}
                    if sample:
                        rowspace=(total@q0)@q0.T
                        cosine=lambda a,b:float((a*b).sum()/(a.norm()*b.norm()).clamp_min(1e-30))
                        diagnostics=dict(raw_gradient_norm=float(total.norm()),rowspace_gradient_norm=float(rowspace.norm()),CE_gradient_norm=float(ce.norm()),KL_gradient_norm=float(kg.norm()),history_gradient_norm=float(hg.norm()),history_CE_cosine=cosine(hg,ce),history_KL_cosine=cosine(hg,kg),delta_dot_CE=float((delta*ce).sum()),delta_dot_KL=float((delta*kg).sum()),delta_dot_history=float((delta*hg).sum()))
                    with torch.no_grad():weight.copy_(current)
                    weight.grad=None;state.update(completed_step=step,committed_updates=state['committed_updates']+1)
                    event=dict(update=state['committed_updates'],owner=owner,step=step,losses=losses,source_indices=[i for i,f in selected],actual_step_norm=actual,reference_step_norm=reference['norm'],norm_reference_binding=norm_binding,history=[dict(owner=x[0]['owner'],role=x[0]['audit_role'],index=next(i for i,y in enumerate(prepared) if key(y[0])==key(x[0]))) for x in historical],diagnostics=diagnostics)
                    j.write(RUN/'private/step_attempts'/arm/f'{event["update"]:04d}_{os.getpid()}_{time.time_ns()}.json',event)
                    state['last_event']=event;save_state();count['updates']+=1;persist()
                    del delta,total,current,ce,kg,hg
                    if step%20==0:
                        c.write(RUN/'private/progress'/f'{arm}.json',dict(event,committed_updates=state['committed_updates'],counts=count));print('TRAIN',arm,owner,step,actual,flush=True)
                verify()
                for item in prepared:
                    if item[0]['owner']<=owner and item[0]['audit_role'] in ('NATIVE','GFIT'):emit(runtime,item,f'{arm}_PREFIX_{owner}',count,persist)
                state.update(completed_edits=owner,active_owner=owner+1,completed_step=0);save_state()
            verify();weight.requires_grad_(False)
            assert state['committed_updates']==1280 and len(list(events.glob('*.json')))==1280
            assert not any(p.requires_grad or p.grad is not None for p in runtime.model.parameters())
            dest=RUN/'private/candidates'/f'{arm}.pt'
            if not dest.exists():c.save(dest,dict(weight=weight.detach().cpu(),schema=schema,parameter_count=parameters,layer=v1.LAYER,lock=binding()))
            else:
                saved=torch.load(dest,map_location='cpu',weights_only=True);assert saved['lock']==binding() and saved['schema']==schema and torch.equal(saved['weight'],weight.detach().cpu())
            done(f'TRAIN_COMPLETE_{arm}.json',ordered_edits=8,committed_updates=1280,trace_updates=1280,diagnostic_events=1280,parameters_before=parameters,parameters_after=parameters,added_parameters=0,noneditable_versions_verified=True,history_replay=arm=='H')
        finally:h.remove();persist()




def evaluate():
    import sys
    assert 'shared_math' not in sys.modules
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
        for part,arm in enumerate(ARMS):
            if arm in NEW_ARMS:stage('scope_train',part,f'TRAIN_COMPLETE_{arm}.json');retire_active(arm)
        if not is_done('CANDIDATES_FROZEN.json'):done('CANDIDATES_FROZEN.json',candidates=2,inherited_candidates=1,selection=False)
        for part,arm in enumerate(ARMS):
            if arm in NEW_ARMS:stage('scope_eval',part,f'EVAL_COMPLETE_{arm}.json')
        if not is_done('GENERATION_COMPLETE.json'):
            paths=[p for p in (RUN/'private/outputs').glob('*/*/*.json') if p.parts[-3]!='BASE_IDENTITY'];assert len(paths)==1395 and all(c.read(p)['lock']==binding() for p in paths)
            for arm in ARMS:
                for row in v1.rows():
                    if row['audit_role'] in ('NATIVE','GFIT'):assert c.read(v1.output(arm,row['owner'],row))['R0']==c.read(v1.output(f'{arm}_PREFIX_8',row['owner'],row))['R0']
            attempts=[c.read(p) for p in (RUN/'private/attempts').glob('*.json')]
            assert sum(len(list((RUN/'private/training_diagnostics'/a).glob('*.json'))) for a in NEW_ARMS)==2560
            done('GENERATION_COMPLETE.json',outputs=1395,committed_updates=2560,backwards=sum(x['backwards'] for x in attempts),counts=attempts,new_generations=sum(x['generations'] for x in attempts),inherited_Base=267,mechanical_temporary_updates=2,inherited_C_updates=1280,inherited_C_outputs=376)
        if not is_done('DELETION.json'):
            if not is_done('DELETION_REQUEST.json'):
                paths=[RUN/'private/candidates'/f'{a}.pt' for a in NEW_ARMS]+list((RUN/'private/cache').glob('*/*.pt'))
                assert all(p.is_file() and not p.is_symlink() and p.resolve().is_relative_to(RUN.resolve()) for p in paths)
                done('DELETION_REQUEST.json',paths=list(map(str,paths)),files=len(paths),bytes=sum(p.stat().st_size for p in paths),retained_copy=False,diagnostics_retained=True,consumers=['both_train','allprefix','both_final','native_reload','durable_Judge_inputs'])
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
