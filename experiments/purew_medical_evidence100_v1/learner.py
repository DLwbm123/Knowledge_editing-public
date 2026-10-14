"""Current-only continuous pure-W learner; old samples never enter this process."""
import gc
import math
import os
from pathlib import Path
import random
import signal
import time
import traceback
import numpy as np
import torch
import torch.nn.functional as F
from runtime import R,ARM,WINDOW,CONTROL,c,b,load_learner,track,backward,wait,lease,peak_memory
from curvature import Risk,native_candidate
from access import install
import checkpoint


def batches(runtime,packet):
    task=packet['task'];qs=[None]+task['fit_questions']
    full=[b.batch_cpu(runtime,task,q) for q in qs]
    removed=[]
    if packet['gate']:
        from PIL import Image
        image=Image.open(task['native']['image_path']).convert('RGB')
        assert list(image.size)==packet['image_size']
        a=np.array(image)
        for x,y,w,h in packet['boxes']:
            assert 0<=x<x+w<=image.width and 0<=y<y+h<=image.height
            a[math.floor(y):math.ceil(y+h),math.floor(x):math.ceil(x+w),:]=128
        target=WINDOW/'removed.png';Image.fromarray(a).save(target)
        other=dict(task,native=dict(task['native'],image_path=str(target)))
        removed=[b.batch_cpu(runtime,other,q) for q in qs]
        assert all(torch.equal(x[1],y[1]) for x,y in zip(full,removed))
    return full,removed


def meanlogp(runtime,batch):
    z,y=b.logits(runtime,batch)
    return F.log_softmax(z,dim=-1).gather(-1,y[:,None]).mean()


def current_anchors(runtime,full,removed):
    if not removed:return None
    with torch.no_grad():
        return dict(full=[float(meanlogp(runtime,x)) for x in full],removed=[float(meanlogp(runtime,x)) for x in removed])


def source_gradient(runtime,weight,master,full,removed,anchors,slot):
    index=1+(slot-1)%4
    gradient=torch.zeros_like(master);ce=scope=0.
    for k in (0,index):
        z,y=b.logits(runtime,full[k]);loss=.5*F.cross_entropy(z,y)
        ce+=float(loss.detach());gradient+=backward(loss,weight)
        if removed:
            loss=.5*(meanlogp(runtime,removed[k])-anchors['removed'][k]).square()
            scope+=float(loss.detach());gradient+=backward(loss,weight)
    norm=float(gradient.norm());assert math.isfinite(norm)
    gradient*=min(1.,5/max(norm,1e-30))
    return gradient,ce,scope,norm


def optimize(runtime,weight,master,risk,full,removed,anchors,slot):
    gradient,ce,scope,norm=source_gradient(runtime,weight,master,full,removed,anchors,slot)
    master,native,event=native_candidate(master,weight,gradient,risk)
    if event['accepted']:
        with torch.no_grad():weight.copy_(native)
        b.COUNTS['committed_updates']+=1;b.call('C')
    event.update(source_CE=ce,medical_scope_loss=scope,unclipped_gradient_norm=norm,risk_compression_envelope=risk.eq)
    b.counts();return master,event


def probe_rows(runtime,weight,full,removed,seed):
    # Five disjoint weighted blocks, four Gaussian VJPs. Probe RNG does not advance training RNG.
    gen=torch.Generator(device=runtime.device);gen.manual_seed(seed)
    rows=torch.zeros((4,weight.numel()),device=runtime.device,dtype=torch.float64)
    for j in range(4):
        for k,coefficient in enumerate((.5,.125,.125,.125,.125)):
            z,y=b.logits(runtime,full[k]);r=(z-z.mean(-1,keepdim=True))/math.sqrt(len(y))
            if removed:
                zr,yr=b.logits(runtime,removed[k]);assert torch.equal(y,yr)
                r-= (zr-zr.mean(-1,keepdim=True))/math.sqrt(len(yr))
            noise=torch.randn(r.shape,device=r.device,dtype=r.dtype,generator=gen)
            rows[j]+=backward((r*noise).sum()*math.sqrt(coefficient)/2,weight).flatten().double()
    return rows


def validate_state(state):
    assert 1<=state['owner']<=100 and 1<=state['slot']<=160
    assert state['completed_slots']==(state['owner']-1)*160+state['slot']
    assert state['clocks']==dict(FIT=(state['slot']-1)%4,answer_probe=20260912+state['owner'],evidence_probe=20260912+100000+state['owner'])
    assert state['native'].dtype==torch.float16 and state['master'].dtype==torch.float32
    assert torch.equal(state['native'],state['master'].half())
    assert not state['boundary'] or (state['slot']==160 and state['current_anchors'] is None)
    if state['current_anchors'] is not None:
        assert set(state['current_anchors'])=={'full','removed'} and all(len(x)==5 for x in state['current_anchors'].values())
    assert state['risk']['q']<=.01 and state['risk']['rho']>=0 and len(state['risk']['B'])<=4
    return state


def save(master,weight,risk,owner,slot,completed,event,edit_initial,boundary=False,anchors=None):
    state=dict(master=master.cpu(),native=weight.detach().cpu(),risk=risk.state(),owner=owner,slot=slot,
        completed_slots=completed,event=event,edit_initial=edit_initial.cpu(),boundary=boundary,current_anchors=anchors,rng=c.rng(),
        numpy_rng=(np.random.get_state()[0],torch.from_numpy(np.random.get_state()[1].astype('int64')),*np.random.get_state()[2:]),
        clocks=dict(FIT=(slot-1)%4,answer_probe=20260912+owner,evidence_probe=20260912+100000+owner),arm=ARM)
    validate_state(state)
    checkpoint.write(R/'private/active'/ARM/'state.pt',state)
    c.write(CONTROL/'CHECKPOINT.json',dict(owner=owner,slot=slot,completed_slots=completed,boundary=boundary))
    return state


def admission(runtime,weight,full,removed,shape,frozen):
    saved=weight.detach().clone();features={}
    h=runtime.get_module(c.LAYER).register_forward_pre_hook(lambda m,a:features.update(value=a[0].detach().clone()))
    try:
        with torch.no_grad():b.logits(runtime,full[0])
        before=features['value'].clone();weight.requires_grad_(True)
        risk=Risk(weight);master,event=optimize(runtime,weight,saved.float(),risk,full,[],None,1)
        assert event['accepted'] and event['changed_elements']>0
        with torch.no_grad():b.logits(runtime,full[0])
        assert torch.equal(before,features['value'])
        probes=probe_rows(runtime,weight,full,[],20260913);risk.consolidate(probes)
        if removed:
            contrasts=probe_rows(runtime,weight,full,removed,20360913)
            assert torch.isfinite(contrasts).all() and float(contrasts.norm())>0
        with torch.no_grad():weight.copy_(saved)
    finally:h.remove();weight.requires_grad_(False)
    b.guard(runtime,weight,shape,frozen)
    c.write(R/'public'/('REAL_VLM_ADMISSION_'+ARM+'.json'),dict(status='PASS',upstream_feature_exact=True,
        real_native_W_gradient_write=True,added_parameters=0,schema_exact=True,temporary_updates=1,
        Gaussian_VJP_native_path=True,eligible_intervention_finite=bool(removed),parameters=sum(p.numel() for p in runtime.model.parameters())))


def main():
    signal.signal(signal.SIGTERM,b.request_stop);signal.signal(signal.SIGINT,b.request_stop)
    with lease():
        runtime=load_learner();h=track(runtime)
        try:
            install(R,WINDOW,os.environ['DATA_ROOT']+'/knowledge_editing/data/m3bench')
            # The learner cannot read the benchmark, evaluation, operator source pack or original images.
            for path in (R/'private/EVAL_LEDGER.json',R/'private/operator/source.json'):
                try:path.read_text()
                except PermissionError:pass
                else:raise AssertionError('Historical access was allowed')
            weight=runtime.get_module(c.LAYER).weight;shape=b.schema(runtime.model)
            frozen={n:p._version for n,p in runtime.model.named_parameters() if p is not weight}
            master=weight.detach().float().clone();risk=Risk(weight);completed=0;resume=None;start_owner=1
            active=R/'private/active'/ARM/'state.pt'
            if active.exists():
                plan=c.read(R/'private/CHECKPOINT_LAYOUT_RECOVERY.json')
                assert ARM=='M' and plan['status']=='VERIFIED_ONE_REJECTED_SLOT_AFTER_BOUNDARY'
                resume=validate_state(checkpoint.read(active))
                assert resume['owner']==1 and resume['slot']==160 and resume['boundary'] and resume['completed_slots']==160
                assert c.read(R/'public/REAL_VLM_ADMISSION_M.json')['status']=='PASS'
                with torch.no_grad():weight.copy_(resume['native'].to(runtime.device))
                master=resume['master'].to(runtime.device);risk.restore(resume['risk'],runtime.device)
                completed=161;start_owner=2
            for owner in range(start_owner,101):
                c.write(CONTROL/'NEED_SOURCE.json',dict(owner=owner))
                packet=wait(WINDOW/'current.json');assert packet['task']['order']==owner
                task=packet['task'];random.seed(task['seed']);np.random.seed(task['seed']);torch.manual_seed(task['seed']);torch.cuda.manual_seed_all(task['seed'])
                full,removed=batches(runtime,packet)
                if owner==1:
                    admission(runtime,weight,full,removed,shape,frozen)
                    master=weight.detach().float().clone();risk=Risk(weight)
                    c.write(CONTROL/'LEARNER_ADMITTED.json',dict(status='PASS'))
                    wait(CONTROL/'BASE_ADMITTED.json')
                actual_removed=removed if ARM!='M' else []
                anchors=current_anchors(runtime,full,actual_removed)
                weight.requires_grad_(True);edit_initial=weight.detach().float().clone();start_slot=1
                if resume is not None:
                    assert owner==2 and ARM=='M'
                    # The lost slot made no native/master/risk change, proven by persisted write counters.
                    g,ce,scope,norm=source_gradient(runtime,weight,master,full,[],None,1)
                    diagnostic=Risk(weight);diagnostic.restore(risk.state(),runtime.device)
                    _,_,zero=native_candidate(master,weight,g,diagnostic)
                    assert not zero['accepted'] and zero['actual_native_norm']==0 and diagnostic.q==risk.q
                    zero.update(owner=2,step=1,slot=1,arm=ARM,completed_slots=161,source_CE=ce,medical_scope_loss=scope,
                        unclipped_gradient_norm=norm,risk_compression_envelope=risk.eq,edit_native_net_displacement=0.,FIT_clock=0,
                        optimization_seconds=None,engineering_reconstructed_original_rejected_slot=True,
                        reconstruction_additional_calls_counted=True,original_native_write_proven_zero=True)
                    c.write(R/'private/steps'/ARM/'2/1.json',zero);save(master,weight,risk,2,1,161,zero,edit_initial)
                    c.write(R/'private/CHECKPOINT_LAYOUT_RECOVERY_VALIDATED.json',dict(status='PASS',no_replayed_W_update=True,
                        preserved_consumed_slot=1,additional_diagnostic_backwards=2,compressed_state_exact=True))
                    start_slot=2;resume=None;del g,diagnostic
                for slot in range(start_slot,161):
                    c.budget();start=time.perf_counter()
                    master,event=optimize(runtime,weight,master,risk,full,actual_removed,anchors,slot);completed+=1
                    b.COUNTS['attempt_slots']=b.COUNTS.get('attempt_slots',0)+1;b.counts()
                    event.update(owner=owner,step=slot,slot=slot,arm=ARM,completed_slots=completed,
                        edit_native_net_displacement=float((weight.detach().float()-edit_initial).norm()),
                        FIT_clock=(slot-1)%4,optimization_seconds=time.perf_counter()-start)
                    if slot%20==0 or b.STOP_REQUESTED:save(master,weight,risk,owner,slot,completed,event,edit_initial,anchors=anchors)
                    c.write(R/'private/steps'/ARM/str(owner)/(str(slot)+'.json'),event)
                    if b.STOP_REQUESTED:raise SystemExit('Cooperative stop saved; no automatic continuation')
                    if slot%20==0:
                        b.guard(runtime,weight,shape,frozen);b.write_progress(status='TRAINING',arm=ARM,owner=owner,slot=slot,attempt_slots=completed,last_slot_accepted=event['accepted'],risk=risk.q)
                        print('TRAIN',ARM,owner,slot,completed,event['accepted'],risk.q,flush=True)
                marker=R/'private/consolidation'/ARM/(str(owner)+'.STARTED.json')
                assert not marker.exists(),'Probe attempt may not be blindly replayed'
                c.write(marker,dict(owner=owner,epoch=time.time()))
                rows=probe_rows(runtime,weight,full,[],20260912+owner);answer=risk.consolidate(rows);del rows
                evidence=None
                if ARM=='M-E' and removed:
                    rows=probe_rows(runtime,weight,full,removed,20260912+100000+owner);evidence=risk.consolidate(rows);del rows
                c.write(marker.with_name(str(owner)+'.json'),dict(answer=answer,evidence=evidence,rho=risk.rho,risk=risk.q,
                    error_envelope=risk.eq,retained_rows=len(risk.B),candidate_rows_retired=True))
                weight.requires_grad_(False);b.guard(runtime,weight,shape,frozen)
                save(master,weight,risk,owner,160,completed,event,edit_initial,boundary=True)
                # Only a native matrix and owner number cross to the evaluator. No score/output returns.
                b.save(CONTROL/'native.pt',dict(native=weight.detach().cpu()))
                del full,removed,actual_removed,anchors,packet,task,edit_initial
                gc.collect();torch.cuda.empty_cache()
                if b.STOP_REQUESTED:raise SystemExit('Cooperative boundary stop: full curvature/RNG checkpoint durable')
                c.write(CONTROL/'EVALUATE.json',dict(owner=owner))
                wait(CONTROL/(str(owner)+'.EVAL_DONE.json'))
                c.write(R/'private/units'/ARM/(str(owner)+'.json'),dict(status='COMPLETE',slots=160,completed_slots=completed))
                peak_memory()
                wait(CONTROL/(str(owner)+'.RETIRED.json'))
            b.save(R/'private/final'/ARM/'native.pt',dict(native=weight.detach().cpu(),schema=shape,
                   parameters=sum(p.numel() for p in runtime.model.parameters())))
            c.write(R/'private'/('TRAINING_COMPLETE_'+ARM+'.json'),dict(status='COMPLETE',attempt_slots=16000))
        finally:h.close();b.counts()


if __name__=='__main__':
    try:
        main();c.write(R/'private'/('EXIT_'+os.environ['ACTION']+'.json'),dict(exit_code=0,epoch=time.time()))
    except BaseException:
        b.counts();c.write(R/'private/failures'/(os.environ['ACTION']+'_'+str(os.getpid())+'.json'),dict(traceback=traceback.format_exc(),epoch=time.time(),counts=b.COUNTS));raise
