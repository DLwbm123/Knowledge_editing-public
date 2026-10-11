"""Finite native-W source/history comparison; original runtime remains unchanged."""
import os,json,time,traceback,sys
from pathlib import Path
import torch
import torch.nn.functional as F
import base_run as v
import journal as j
import cohort
c,RUN,data,old=v.c,v.RUN,v.data,v.old
c.write=j.write
v.ARMS=('U','H_SOURCE');v.GPUS=(5,5)
PARENT=Path(os.environ['REFERENCE_PARENT'])

def lock():return c.digest(c.read(RUN/'private/LOCK.json'))
def done(name,**value):j.write(RUN/'private'/name,dict(value,lock=lock(),epoch=time.time()))
def complete(name):return j.completed(RUN/'private'/name,lock())
def task(row):return dict(order=row['owner'] or 1,edit_id=row['query_id'],native=row,fit_questions=[row['question']])
def set_tasks(rows):
    native=[x for x in rows if x['audit_role']=='NATIVE']
    if not native:native=[dict(rows[0],owner=1)]
    data.tasks=lambda:[task(r) for r in native]

def tracked(runtime,name):
    count=dict(forwards=0,backwards=0,updates=0,generations=0,generation_attempts=0)
    path=RUN/'private/attempts'/f'{name}_{os.getpid()}_{time.time_ns()}.json'
    def persist():
        j.write(path,dict(count,action=name,pid=os.getpid(),lock=lock()))
        counts=[c.read(x) for x in path.parent.glob('*.json')]
        protocol=c.read(RUN/'private/FROZEN_PROTOCOL.json')
        assert sum(x['backwards'] for x in counts)<=protocol['maximum_backwards']
        assert sum(x['generation_attempts'] for x in counts)<=protocol['maximum_generation_attempts']
    def before(*_):count['forwards']+=1;persist()
    handle=runtime.model.register_forward_pre_hook(before)
    persist();return count,handle,persist

def emit(runtime,item,stage,count,persist):
    row=item[0];p=v.output(stage,row['owner'],row)
    if p.exists():
        x=c.read(p);assert x['lock']==lock() and x['binding']['input']==row
        return p
    count['generation_attempts']+=1;persist()
    p=v.emit(runtime,item[:5],stage,count);persist();return p

def qualify():
    if complete('QUAL_GENERATION_COMPLETE.json'):return
    rows=c.read(RUN/'private/COHORT_CANDIDATES.json')['candidates'];set_tasks(rows)
    c.available=v.admission
    with c.lease(5):
        runtime,bindings=c.load(5);count,h,persist=tracked(runtime,'source_qualification')
        try:
            v.fp32_native(runtime)
            for row in rows:
                # Qualifications use owner0; their answers alone may determine T0 admission.
                item=v.prepare(runtime,bindings,[row])[0];emit(runtime,item,'QUAL_BASE',count,persist)
            done('QUAL_GENERATION_COMPLETE.json',outputs=84,schema=v.state_schema(runtime.model),parameters=sum(p.numel() for p in runtime.model.parameters()))
        finally:h.remove();persist()

def rows():return c.read(RUN/'private/ROWS.json')
def source_items(runtime,bindings):return v.prepare(runtime,bindings,cohort.training_rows(rows()),True)

def base():
    if complete('BASE_COMPLETE.json'):return
    set_tasks(rows());c.available=v.admission
    with c.lease(5):
        runtime,bindings=c.load(5);count,h,persist=tracked(runtime,'base')
        try:
            v.fp32_native(runtime)
            for row in rows():
                if v.output('BASE',row['owner'],row).exists():continue
                item=v.prepare(runtime,bindings,[row])[0];emit(runtime,item,'BASE',count,persist)
            assert len(list((RUN/'private/outputs/BASE').glob('*/*.json')))==261
            done('BASE_COMPLETE.json',outputs=261,schema=v.state_schema(runtime.model),parameters=sum(p.numel() for p in runtime.model.parameters()))
        finally:h.remove();persist()

def admission():
    from purew_math import target_logits,preservation_kl
    import shared_math as sm
    if complete('MECHANICAL_COMPLETE.json'):return
    set_tasks(rows());c.available=v.admission
    with c.lease(5):
        runtime,bindings=c.load(5);count,h,persist=tracked(runtime,'admission')
        try:
            v.fp32_native(runtime);weight=runtime.get_module(v.LAYER).weight;original=weight.detach().clone()
            schema=v.state_schema(runtime.model);other={n:p._version for n,p in runtime.model.named_parameters() if p is not weight}
            source=source_items(runtime,bindings);basis=data.q.split()[0];assert len(basis)==61
            for i,row in enumerate(basis):
                path=RUN/'private/cache/protection'/f'{i}.pt'
                if path.exists():assert torch.load(path,weights_only=True)['lock']==lock();continue
                b=data.q.protection_batch(runtime,row,data.tasks()[0])
                with torch.no_grad():logits,_=target_logits(runtime.model,b)
                c.save(path,dict(reference=logits.cpu(),lock=lock()))
            reports=[]
            for arm,index in [('U',0),('H_SOURCE',1)]:
                weight.requires_grad_(True);gradient=torch.zeros_like(weight)
                selected=[(source[index],1.)]+([(source[0],.5)] if arm=='H_SOURCE' else [])
                for item,f in selected:
                    weight.grad=None;z,y=target_logits(runtime.model,old.batch_on(runtime,item[1]));count['backwards']+=1;persist();(f*F.cross_entropy(z,y)).backward();gradient.add_(weight.grad)
                weight.grad=None
                b=data.q.protection_batch(runtime,basis[0],data.tasks()[0]);z,_=target_logits(runtime.model,b)
                ref=torch.load(RUN/'private/cache/protection/0.pt',weights_only=True)['reference'].to(runtime.device)
                kl=preservation_kl(z,ref);assert abs(float(kl.detach()))<1e-5
                count['backwards']+=1;persist();kl.backward();gradient.add_(weight.grad)
                feature={};observer=runtime.get_module(v.LAYER).register_forward_pre_hook(lambda _,a:feature.update(value=a[0].detach().clone()))
                try:
                    b=old.batch_on(runtime,source[index][1])
                    with torch.no_grad():target_logits(runtime.model,b)
                    before=feature['value'];target=min(.01*float(gradient.norm()),.05)
                    delta=sm.shared_step(gradient,target);new,actual=sm.apply_step(original,delta,target)
                    with torch.no_grad():weight.copy_(new);target_logits(runtime.model,b)
                    assert actual>0 and torch.equal(before,feature['value'])
                    assert other=={n:p._version for n,p in runtime.model.named_parameters() if p is not weight}
                    reports.append(dict(arm=arm,real_gradient_norm=float(gradient.norm()),actual_write_norm=actual))
                finally:
                    observer.remove()
                    with torch.no_grad():weight.copy_(original)
                    weight.grad=None;weight.requires_grad_(False);assert torch.equal(weight,original)
            for index in (0,23):
                item=source[index];p=emit(runtime,item,'BASE_IDENTITY',count,persist)
                assert c.read(p)['R0']==c.read(v.output('BASE',item[0]['owner'],item[0]))['R0']
            assert schema==v.state_schema(runtime.model) and not any(p.requires_grad or p.grad is not None for p in runtime.model.parameters())
            done('MECHANICAL_COMPLETE.json',status='PASS',feature_invariance_exact=True,exact_native_Base_identity=True,added_parameters=0,parameters=sum(p.numel() for p in runtime.model.parameters()),reports=reports)
        finally:h.remove();persist()

def train():
    from purew_math import target_logits,preservation_kl
    import shared_math as sm
    arm=v.ARMS[int(os.environ['PARTITION'])]
    if complete(f'TRAIN_COMPLETE_{arm}.json'):return
    assert complete('MECHANICAL_COMPLETE.json')
    if arm=='H_SOURCE':assert complete('TRAIN_COMPLETE_U.json')
    set_tasks(rows());c.available=v.admission
    with j.exclusive(RUN/'private/locks'/f'{arm}.lock'),c.lease(5):
        runtime,bindings=c.load(5);count,h,persist=tracked(runtime,'train_'+arm)
        try:
            v.fp32_native(runtime);weight=runtime.get_module(v.LAYER).weight;schema=v.state_schema(runtime.model)
            parameters=sum(p.numel() for p in runtime.model.parameters());assert parameters==7566219264
            frozen={n:p._version for n,p in runtime.model.named_parameters() if p is not weight}
            buffers={n:b.detach().clone() for n,b in runtime.model.named_buffers()}
            items=source_items(runtime,bindings);by_owner={x[0]['owner']:x for x in items}
            evaluated=v.prepare(runtime,bindings,[r for r in rows() if r['audit_role'] in ('NATIVE','T2G')])
            protected=[v.cpu_batch(data.q.protection_batch(runtime,r,data.tasks()[0])) for r in data.q.split()[0]]
            refs=[torch.load(RUN/'private/cache/protection'/f'{i}.pt',map_location='cpu',weights_only=True) for i in range(61)]
            assert all(x['lock']==lock() for x in refs)
            active=RUN/'private/active'/f'{arm}.pt';events=RUN/'private/training_diagnostics'/arm
            state=dict(completed_edits=0,active_owner=1,completed_step=0,committed_updates=0,last_event=None,lock=lock())
            if active.exists():
                state=torch.load(active,map_location='cpu',weights_only=True);assert state['lock']==lock()
                with torch.no_grad():weight.copy_(state['weight'].to(runtime.device))
                j.ensure_event(events,state)
            weight.requires_grad_(True)
            def verify():
                assert schema==v.state_schema(runtime.model) and frozen=={n:p._version for n,p in runtime.model.named_parameters() if p is not weight}
                assert all(torch.equal(b,buffers[n]) for n,b in runtime.model.named_buffers())
                assert [n for n,p in runtime.model.named_parameters() if p.requires_grad]==[v.LAYER+'.weight']
                assert not runtime.get_module(v.LAYER)._forward_hooks and not runtime.get_module(v.LAYER)._forward_pre_hooks
            def commit():
                state['weight']=weight.detach().cpu();c.save(active,state)
                with active.open('rb') as f:os.fsync(f.fileno())
                fd=os.open(active.parent,os.O_RDONLY)
                try:os.fsync(fd)
                finally:os.close(fd)
                j.ensure_event(events,state)
            for owner in range(1,25):
                if owner<=state['completed_edits']:continue
                assert owner==state['active_owner']
                for step in range(state['completed_step']+1,161):
                    c.budget();verify();sample=step in (1,40,80,160)
                    total=torch.zeros_like(weight);losses={};gradients={}
                    history=cohort.past_source(rows(),owner,step) if arm=='H_SOURCE' else None
                    selected=[('CURRENT',by_owner[owner],1.)]+([('HISTORY',by_owner[history['owner']],.5)] if history else [])
                    for label,item,f in selected:
                        assert item[0]['audit_role']=='NATIVE' and item[0]['source_admission']=='PASS'
                        weight.grad=None;z,y=target_logits(runtime.model,old.batch_on(runtime,item[1]));loss=F.cross_entropy(z,y);assert torch.isfinite(loss)
                        count['backwards']+=1;persist();(f*loss).backward();total.add_(weight.grad)
                        losses[label]=float(loss.detach())
                        if sample:gradients[label]=weight.grad.clone()
                        del z,loss
                    index=((owner-1)*160+step-1)%61;weight.grad=None
                    z,_=target_logits(runtime.model,old.batch_on(runtime,protected[index]));kl=preservation_kl(z,refs[index]['reference'].to(runtime.device));assert torch.isfinite(kl)
                    count['backwards']+=1;persist();kl.backward();total.add_(weight.grad);losses['BASIS_KL']=float(kl.detach())
                    if sample:gradients['KL']=weight.grad.clone()
                    del z,kl
                    update=state['committed_updates']+1
                    if arm=='U':target=min(.01*float(total.norm()),.05);reference=None
                    else:
                        reference=c.read(RUN/'private/training_diagnostics/U'/f'{update:04d}.json')
                        assert (reference['owner'],reference['step'],reference['update'])==(owner,step,update)
                        target=reference['proposal_norm'] if owner==1 else reference['actual_step_norm']
                    delta=sm.shared_step(total,target);new,actual=sm.apply_step(weight.detach(),delta,target)
                    diagnostic={label:dict(gradient_norm=float(g.norm()),delta_dot_gradient=float((delta*g).sum())) for label,g in gradients.items()}
                    with torch.no_grad():weight.copy_(new)
                    weight.grad=None;state.update(completed_step=step,committed_updates=update)
                    if reference:
                        import math
                        assert math.isclose(actual,reference['actual_step_norm'],rel_tol=5e-4,abs_tol=1e-6)
                        if owner==1:assert losses==reference['losses'] and actual==reference['actual_step_norm']
                    event=dict(update=update,owner=owner,step=step,losses=losses,actual_step_norm=actual,proposal_norm=target,reference_step_norm=reference['actual_step_norm'] if reference else actual,U_reference_binding=c.digest(reference) if reference else None,history_owner=history['owner'] if history else None,diagnostics=diagnostic)
                    j.write(RUN/'private/step_attempts'/arm/f'{update:06d}_{os.getpid()}_{time.time_ns()}.json',event)
                    state['last_event']=event;commit();count['updates']+=1;persist()
                    del delta,new,total,gradients
                    if step%20==0:print('TRAIN',arm,owner,step,actual,flush=True);j.write(RUN/'private/progress'/f'{arm}.json',dict(event,committed_updates=update))
                for item in evaluated:
                    if item[0]['owner']<=owner:emit(runtime,item,f'{arm}_PREFIX_{owner}',count,persist)
                state.update(completed_edits=owner,active_owner=owner+1,completed_step=0);commit()
            verify();weight.requires_grad_(False)
            assert state['committed_updates']==3840 and len(list(events.glob('*.json')))==3840
            c.save(RUN/'private/candidates'/f'{arm}.pt',dict(weight=weight.detach().cpu(),schema=schema,parameter_count=parameters,layer=v.LAYER,lock=lock()))
            done(f'TRAIN_COMPLETE_{arm}.json',committed_updates=3840,ordered_edits=24,added_parameters=0,parameters_before=parameters,parameters_after=parameters,history_replay=arm=='H_SOURCE')
        finally:h.remove();persist()

def evaluate():
    assert 'shared_math' not in sys.modules and 'purew_math' not in sys.modules
    arm=v.ARMS[int(os.environ['PARTITION'])];assert complete('CANDIDATES_FROZEN.json')
    if complete(f'EVAL_COMPLETE_{arm}.json'):return
    set_tasks(rows());c.available=v.admission
    with c.lease(5):
        runtime,bindings=c.load(5);count,h,persist=tracked(runtime,'eval_'+arm)
        try:
            v.fp32_native(runtime);saved=torch.load(RUN/'private/candidates'/f'{arm}.pt',map_location='cpu',weights_only=True)
            assert saved['lock']==lock() and v.state_schema(runtime.model)==saved['schema'] and sum(p.numel() for p in runtime.model.parameters())==saved['parameter_count']
            with torch.no_grad():runtime.get_module(v.LAYER).weight.copy_(saved['weight'].to(runtime.device))
            for row in rows():
                path=v.output(arm,row['owner'],row)
                if path.exists():assert c.read(path)['lock']==lock();continue
                if row['audit_role'] in ('NATIVE','T2G'):
                    prior=c.read(v.output(f'{arm}_PREFIX_24',row['owner'],row));j.write(path,dict(prior,stage=arm,reused_generation='FINAL_PREFIX'))
                else:emit(runtime,v.prepare(runtime,bindings,[row])[0],arm,count,persist)
            row=next(r for r in rows() if r['owner']==24 and r['audit_role']=='NATIVE');probe=emit(runtime,v.prepare(runtime,bindings,[row])[0],f'{arm}_RELOAD',count,persist)
            assert c.read(probe)['R0']==c.read(v.output(arm,24,row))['R0']
            assert not any(p.requires_grad or p.grad is not None for p in runtime.model.parameters())
            done(f'EVAL_COMPLETE_{arm}.json',outputs=261,native_fresh_reload_tokens_equal=True,inference_structure_unchanged=True,editor_math_imported=False,added_parameters=0)
        finally:h.remove();persist()

def controller():
    import pipeline
    with j.exclusive(RUN/'private/locks/controller.lock'):
        if complete('GENERATION_COMPLETE.json'):return
        def stage(action,part,receipt):
            if complete(receipt):return
            start=RUN/'private'/f'START_CHAIN_{action}_{part}.json'
            if start.exists() and j.alive(c.read(start)):child=None
            else:
                with c.resources() as ledger:
                    for session in ledger['gpu_sessions']:
                        if not session.get('ended_epoch') and session['action'].startswith('source_') and not j.alive(session):
                            stop=time.time();session.update(ended_epoch=stop,resident_seconds=stop-session['started_epoch'],abnormal_exit=True);ledger['gpu_seconds_used']+=session['resident_seconds']
                child=pipeline.launch('run.py',action,5,part)
            while not complete(receipt):
                c.budget();assert j.alive(c.read(start)),'Owned stage failed; preserve state, no blind retry';time.sleep(5)
            if child is not None:assert child.wait()==0
            else:
                while j.alive(c.read(start)):c.budget();time.sleep(1)
        stage('source_base',0,'BASE_COMPLETE.json');stage('source_admission',0,'MECHANICAL_COMPLETE.json')
        for i,a in enumerate(v.ARMS):stage('source_train',i,f'TRAIN_COMPLETE_{a}.json')
        if not complete('CANDIDATES_FROZEN.json'):done('CANDIDATES_FROZEN.json',candidates=2,selection=False)
        for i,a in enumerate(v.ARMS):stage('source_eval',i,f'EVAL_COMPLETE_{a}.json')
        paths=[p for p in (RUN/'private/outputs').glob('*/*/*.json') if p.parts[-3] not in ('QUAL_BASE','BASE_IDENTITY','U_RELOAD','H_SOURCE_RELOAD')]
        assert len(paths)==3783
        counts=[c.read(p) for p in (RUN/'private/attempts').glob('*.json')]
        generated=[c.read(p) for p in (RUN/'private/outputs').glob('*/*/*.json') if not c.read(p).get('reused_generation')]
        non_generation=sum(x['forwards'] for x in counts)-sum(x['forwards'] for x in generated)
        committed=sum(len(list((RUN/'private/training_diagnostics'/a).glob('*.json'))) for a in v.ARMS)
        assert 0<=non_generation<=32000 and committed==7680
        done('GENERATION_COMPLETE.json',outputs=len(paths),committed_updates=committed,attempt_record_updates=sum(x['updates'] for x in counts),non_generation_forwards=non_generation)
        owned=[RUN/'private'/folder/f'{a}.pt' for folder in ('active','candidates') for a in v.ARMS]+list((RUN/'private/cache/protection').glob('*.pt'))
        assert all(p.is_file() and not p.is_symlink() and p.resolve().is_relative_to(RUN.resolve()) for p in owned)
        done('DELETION_REQUEST.json',paths=list(map(str,owned)),files=len(owned),bytes=sum(p.stat().st_size for p in owned),consumers=['allprefix','final','fresh_native_reload','durable_judge_inputs'],retained_copy=False)
        for p in owned:p.unlink()
        done('DELETION.json',files=len(owned),bytes=c.read(RUN/'private/DELETION_REQUEST.json')['bytes'],retained_copy=False)
        done('CONTROLLER_COMPLETE.json',status='GPU_CHAIN_COMPLETE_SCORING_PENDING')

if __name__=='__main__':
    try:{'source_qualify':qualify,'source_base':base,'source_admission':admission,'source_train':train,'source_eval':evaluate,'source_controller':controller}[os.environ['ACTION']]()
    except BaseException as e:
        j.write(RUN/'private/failures'/f'{os.environ.get("ACTION")}_{os.getpid()}_{time.time_ns()}.json',dict(error=repr(e),traceback=traceback.format_exc(),epoch=time.time()));raise
