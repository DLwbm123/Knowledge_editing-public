"""Current-only source stage and serial output-only evaluator for B/C/D."""
import math,os,signal,random,time,traceback
import numpy as np
import torch
import torch.nn.functional as F
from contextlib import nullcontext
import runtime
from runtime import R,c,b,load_learner,track,backward,lease,peak_memory
from core import Memory,dot
from source_loss import loss,scope_anchors
from access import install
from base_learner import batches
ARM=runtime.ARM;WINDOW=runtime.WINDOW;CONTROL=runtime.CONTROL
class ScientificStop(RuntimeError):pass


def probes(model,weight,memory,full,removed,order,block,count=4):
    gen=torch.Generator(device=model.device);gen.manual_seed(20260912+order+(100000 if block=='evidence' else 0))
    rows=torch.zeros((count,weight.numel()),dtype=torch.float64)
    for j in range(count):
        for k,coefficient in enumerate((.5,.125,.125,.125,.125)):
            batch=removed[k] if block=='removed' else full[k]
            z,y=b.logits(model,batch)
            if block!='evidence':
                p=z.detach().softmax(-1);root=p.sqrt();noise=torch.randn(z.shape,device=z.device,dtype=z.dtype,generator=gen)
                u=root*noise-p*(root*noise).sum(-1,keepdim=True);scalar=(z*u).sum()*math.sqrt(coefficient/len(y))/math.sqrt(count)
            else:
                zr,yr=b.logits(model,removed[k]);assert torch.equal(y,yr)
                difference=F.log_softmax(z.double(),-1).gather(-1,y[:,None]).mean()-F.log_softmax(zr.double(),-1).gather(-1,yr[:,None]).mean()
                p=difference.detach().sigmoid();noise=torch.randn((),device=z.device,generator=gen)
                scalar=difference*noise*torch.sqrt(p*(1-p))*math.sqrt(coefficient)/2
            rows[j]+=backward(scalar,weight).flatten().double().cpu()
    event=memory.consolidate(rows,'removed' if block=='removed' else 'history')
    del rows
    return event


def measure(model,full,removed,anchors):
    with torch.no_grad():
        ce=sum(float(loss(model,full,[],[],slot)[0])/4 for slot in range(1,5))
        scope=sum(float(loss(model,full,removed,anchors,slot)[0])/4 for slot in range(1,5))-ce if removed else 0.
    return dict(weighted_all_FIT_CE=ce,scope_KL=scope)


def attempt(model,weight,master,memory,full,removed,anchors,slot,hard_removed=False):
    from medical import direct_KLs
    before=weight.detach().clone();total,ce,scope=loss(model,full,removed,anchors,slot)
    g=backward(total,weight);norm=float(g.norm());assert math.isfinite(norm);g*=min(1.,5/max(norm,1e-30))
    delta,event=memory.solve(g);direction=delta.reshape_as(master).to(master.device,dtype=torch.float32);baseline=float(total.detach());del total,delta
    event.update(source_CE=ce,scope_loss=scope,gradient_norm=norm,current_risk=memory.q,candidates=[])
    accepted=False;numeric_only=False
    if event['solver_reason']=='PASS':
        for index in range(8):
            candidate_master=master+direction*(.5**index);candidate_native=candidate_master.half()
            actual=candidate_native.double()-before.double();risks=memory.inspect(actual);changed=int(torch.count_nonzero(actual));reasons=[]
            if not changed:reasons.append('ZERO_NATIVE')
            for name,risk in risks.items():
                if risk['native_risk_bound']>.01:reasons.append('RISK_'+name)
            slope=dot(g,actual);actual_loss=None;direct=None
            if not reasons:
                with torch.no_grad():
                    weight.copy_(candidate_native)
                    try:
                        actual_loss=float(loss(model,full,removed,anchors,slot)[0]);assert math.isfinite(actual_loss)
                        if hard_removed:
                            direct=direct_KLs(b,model,removed,anchors)
                            assert len(direct)==5 and all(math.isfinite(x) for x in direct)
                            if any(x>.01 for x in direct):reasons.append('CURRENT_REMOVED_HARD_KL')
                    finally:weight.copy_(before)
                if slope>=0 or actual_loss>baseline+1e-4*slope:reasons.append('ACTUAL_SOURCE_ARMIJO')
            numerical_reasons={'RISK_'+name for name,r in risks.items() if r['native_risk_bound']>.01 and r['sampled_native_risk']<=.01 and r['numeric_envelope_dominates']}
            numeric_caused=bool(changed and reasons and set(reasons)<=numerical_reasons)
            event['candidates'].append(dict(backtrack=index,changed_elements=changed,actual_loss=actual_loss,slope=slope,reasons=reasons,risks=risks,current_removed_KLs=direct,numeric_caused_rejection=numeric_caused))
            numeric_only |= numeric_caused
            if not reasons:
                memory.advance(actual)
                with torch.no_grad():weight.copy_(candidate_native)
                master=candidate_master;accepted=True;b.COUNTS['committed_updates']+=1;b.call('C')
                event.update(accepted_loss=actual_loss,risks=risks,actual_native_norm=float(actual.norm()),changed_elements=changed,quantization_residual_norm=float((actual-direction.double()*(.5**index)).norm()),native_master_discrepancy=float((candidate_master-candidate_native.float()).norm()));break
    else:event['rejection_reason']='FINITE_DUAL_REJECT'
    if not accepted:
        assert torch.equal(weight,before);event.update(actual_native_norm=0.,changed_elements=0)
    nonzero=[z for z in event['candidates'] if z['changed_elements']]
    numeric_only=bool(nonzero) and all(z['numeric_caused_rejection'] for z in nonzero)
    event.update(accepted=accepted,slot=slot,zero_native=not accepted,numeric_caused_rejection=numeric_only and not accepted);b.counts();return master,event


def validate_state(state):
    assert state['arm'] in ('B','C','D') and 1<=state['owner']<=3 and 1<=state['slot']<=20
    assert not state['boundary'] or state['slot']==20
    assert state['clocks']==dict(FIT=(state['slot']-1)%4,answer_probe=20260912+state['owner'],evidence_probe=20260912+100000+state['owner'])
    assert state['native'].dtype==torch.float16 and state['master'].dtype==torch.float32 and torch.equal(state['native'],state['master'].half())
    assert state['memory']['arm']==state['arm']
    assert all(v['q']+v['eq']<=.010000000001 for v in state['memory']['blocks'].values())
    assert state['current_anchors'] is None if state['boundary'] else True
    return state

def save(weight,master,memory,owner,slot,boundary,event=None,anchors=None):
    assert torch.equal(weight,master.half())
    state=dict(arm=ARM,native=weight.detach().cpu(),master=master.detach().cpu(),memory=memory.state(),owner=owner,slot=slot,boundary=boundary,
        rng=c.rng(),event=event,counts=dict(b.COUNTS),clocks=dict(FIT=(slot-1)%4,answer_probe=20260912+owner,evidence_probe=20260912+100000+owner),current_anchors=None if boundary else anchors,numpy_rng=(np.random.get_state()[0],torch.from_numpy(np.random.get_state()[1].astype('int64')),*np.random.get_state()[2:]))
    validate_state(state);b.save(R/'private/active'/ARM/'state.pt',state);c.write(CONTROL/'CHECKPOINT.json',dict(owner=owner,slot=slot,boundary=boundary))


def learner():
    signal.signal(signal.SIGTERM,b.request_stop);signal.signal(signal.SIGINT,b.request_stop)
    owner=int(os.environ['OWNER']);model=load_learner();hooks=track(model)
    try:
        install(R,WINDOW,os.environ['DATA_ROOT']+'/knowledge_editing/data/m3bench')
        for name in ['EVAL_LEDGER.json','operator/source.json']:
            try:(R/'private'/name).read_text()
            except PermissionError:pass
            else:raise AssertionError('Evaluation/operator leakage')
        weight=model.get_module(c.LAYER).weight;shape=b.schema(model.model);frozen={n:p._version for n,p in model.model.named_parameters() if p is not weight}
        original=weight.detach().clone();packet=c.read(WINDOW/'current.json');assert packet['quick_owner']==owner
        if owner>1:
            state=validate_state(torch.load(R/'private/active'/ARM/'state.pt',map_location='cpu',weights_only=True))
            assert state['owner']==owner-1 and state['slot']==20 and state['boundary']
            with torch.no_grad():weight.copy_(state['native'].to(model.device))
            master=state['master'].to(model.device);memory=Memory.restore(state['memory'],model.device);del state
        else:master=weight.detach().float().clone();memory=Memory.restore(torch.load(R/'private/calibration/Base.pt',map_location='cpu',weights_only=True));memory.arm=ARM
        seed=packet['task']['seed'];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        full,removed=batches(model,packet);anchors=scope_anchors(model,removed);memory.prepare();weight.requires_grad_(True)
        initial=measure(model,full,removed,anchors);source_initial=weight.detach().float().clone();source_path0=memory.path
        if owner==1 and os.environ['ACTION'].startswith('admit_'):
            features={};hook=model.get_module(c.LAYER).register_forward_pre_hook(lambda m,a:features.update(value=a[0].detach().clone()));old_rng=c.rng();old_state=memory.state();mechanical=Memory(weight,ARM)
            with torch.no_grad():b.logits(model,full[0]);before=features['value'].clone()
            trial,event=attempt(model,weight,master,mechanical,full,[],[],1)
            assert event['accepted'] and event['actual_native_norm']>0,'Scientific native admission rejected; no bypass'
            with torch.no_grad():b.logits(model,full[0]);assert torch.equal(before,features['value']);weight.copy_(original)
            hook.remove();memory=Memory.restore(old_state,model.device);c.restore_rng(old_rng);b.guard(model,weight,shape,frozen)
            c.write(R/'public'/('REAL_VLM_ADMISSION_'+ARM+'.json'),dict(status='PASS',real_native_write=True,upstream_exact=True,parameters_unchanged=True,temporary_updates=1))
            return
        assert all(c.read(R/'public'/('REAL_VLM_ADMISSION_'+a+'.json'))['status']=='PASS' and c.read(R/'public'/('BASE_IDENTITY_'+a+'.json'))['status']=='PASS' for a in ('B','C','D'))
        from medical import concept_indices,source_concepts,contrasts,validate_margins
        indices=None
        if ARM=='C' and owner==1:
            assert removed and len(removed)==5
            indices=[concept_indices(model.adapter.tokenizer,b.logits(model,batch)[1],source_concepts(packet['task']['native']['reference'])) for batch in full]
            pre=probes(model,weight,memory,full,removed,packet['task']['order'],'removed')
            c.write(R/'public/C_REMOVED_BIRTH.json',dict(status='PASS',anchor_before_first_write=True,rows=4,statistics=pre))
        slots=20;events=[];numeric_streak=0
        for slot in range(1,slots+1):
            c.budget();master,event=attempt(model,weight,master,memory,full,removed,anchors,slot,hard_removed=ARM=='C' and owner==1)
            events.append(event)
            numeric_streak=numeric_streak+1 if event['numeric_caused_rejection'] and event['source_CE']>=.1 else 0
            b.COUNTS['attempt_slots']=b.COUNTS.get('attempt_slots',0)+1
            b.counts();event.update(owner=owner,medical_gate=bool(removed));c.write(R/'private/steps'/ARM/str(owner)/(str(slot)+'.json'),event)
            if slot%20==0 or b.STOP_REQUESTED:save(weight,master,memory,owner,slot,False,event,anchors)
            c.write(R/'public/PROGRESS.json',dict(status='BCD_TRAINING',arm=ARM,owner=owner,slot=slot,accepted=event['accepted'],epoch=time.time()))
            if b.STOP_REQUESTED:raise RuntimeError('Cooperative stop saved; no implicit replay')
            if numeric_streak>=5:
                save(weight,master,memory,owner,slot,False,event,anchors)
                raise ScientificStop('NUMERIC_DOMINANCE_FIVE_HIGH_CE_REJECTIONS')
        final=measure(model,full,removed,anchors);answer=probes(model,weight,memory,full,[],packet['task']['order'],'answer',16 if ARM=='D' else 4)
        evidence=probes(model,weight,memory,full,removed,packet['task']['order'],'evidence') if removed else None
        margin_event=None
        if ARM=='C' and owner==1:
            with torch.no_grad():values=contrasts(b,model,full,removed,indices)
            margins=validate_margins(values);del values
            rows=torch.zeros((10,weight.numel()),dtype=torch.float64)
            for j,m in enumerate(margins):
                # Only the current source remains; recreate a single component graph.
                k=j//2;z,y=b.logits(model,full[k]);zr,yr=b.logits(model,removed[k]);ix=indices[k][j%2]
                contrast=(F.log_softmax(z.double(),-1).gather(-1,y[:,None]).flatten()[ix]-F.log_softmax(zr.double(),-1).gather(-1,yr[:,None]).flatten()[ix]).mean()
                rows[j]=backward(contrast/m,weight).flatten().double().cpu()
            margin_event=memory.consolidate(rows,'margin');margin_event.update(minimum_birth_margin=min(margins),maximum_birth_margin=max(margins));del rows,indices,margins
            c.write(R/'public/C_MARGIN_BIRTH.json',dict(status='PASS',ten_positive_finite=True,rows=10,statistics=margin_event,clinical_diagnosis_certificate=False))
        high_loss_frozen=final['weighted_all_FIT_CE']>=.1 and all(e['zero_native'] for e in events[-10:])
        c.write(R/'public'/('SOURCE_'+ARM+'_'+str(owner)+'.json'),dict(status='MECHANICAL_ONLY',initial=initial,final=final,answer=answer,evidence=evidence,margin=margin_event,high_loss_frozen=high_loss_frozen,added_parameters=0))
        c.write(R/'public'/('PATH_'+ARM+'_'+str(owner)+'.json'),dict(cumulative_native_path=memory.path,source_native_path=memory.path-source_path0,source_native_net=float((weight.detach().float()-source_initial).norm()),birth_path_moments=memory.birth_moments,native_master_discrepancy=float((master-weight.detach().float()).norm())))
        weight.requires_grad_(False);b.guard(model,weight,shape,frozen);peak_memory();save(weight,master,memory,owner,slots,True,event)
        b.save(CONTROL/'native.pt',dict(native=weight.detach().cpu(),schema=shape,parameters=sum(p.numel() for p in model.model.parameters())))
        if owner==3:b.save(R/'private/final'/ARM/'native.pt',dict(native=weight.detach().cpu(),schema=shape,parameters=sum(p.numel() for p in model.model.parameters())))
        del full,removed,anchors,packet
    finally:hooks.close();b.counts()


def evaluate():
    from panels import expected_ids
    import base_evaluator as old
    old.expected_ids=expected_ids
    model,bindings=c.load(2);hooks=track(model)
    try:
        tasks=c.read(R/'private/BENCHMARK146_QUEUE.json')['tasks'];ledger=c.read(R/'private/EVAL_LEDGER.json');phase=os.environ['EVAL_PHASE'];owner=int(os.environ['OWNER'])
        if phase=='base':
            from contract import base_completion
            for i in (1,50,100):
                row=tasks[i-1]['native'];d=b.generate(model,bindings,row,R/'private/evaluation'/ARM/'base'/(str(i)+'.json'),dict(stage='BASE_IDENTITY',arm=ARM));assert d['R0']==base_completion(bindings[row['opaque_Base_id']]['output'])
            c.write(R/'public'/('BASE_IDENTITY_'+ARM+'.json'),dict(status='PASS',outputs=3));return
        native=torch.load(R/'private/final'/ARM/'native.pt' if phase=='reload' else CONTROL/'native.pt',map_location='cpu',weights_only=True)
        assert b.schema(model.model)==native['schema']
        with torch.no_grad():model.get_module(c.LAYER).weight.copy_(native['native'].to(model.device))
        if phase=='reload':
            task=tasks[2];ids=[task['edit_id'],next(e['all_probe_query_ids'][0] for e in task['events'] if e['task']=='T2G')]
            for q in ids:
                d=b.generate(model,bindings,ledger['queries'][q],R/'private/reload'/ARM/(q+'.json'),dict(stage='FRESH_NATIVE_RELOAD',arm=ARM));assert d['R0']==c.read(R/'private/outputs'/ARM/'bank_R0/0/3'/(q+'.json'))['R0']
            c.write(R/'public'/('NATIVE_RELOAD_'+ARM+'.json'),dict(status='PASS',two_outputs_exact=True));c.write(R/'private'/('FINAL_CONSUMED_'+ARM+'.json'),dict(status='PASS'))
        else:
            old.panel(model,bindings,tasks,ledger,'insertion_R0',owner,owner)
            if owner in (1,3):old.panel(model,bindings,tasks,ledger,'bank_R0',0,owner)
            if owner==3:old.panel(model,bindings,tasks,ledger,'protect100_R0',0,100)
        peak_memory()
    finally:hooks.close();b.counts()


def main():
    assert os.environ['CUDA_VISIBLE_DEVICES']=='2'
    with lease():
        assert b.available(2);reserve=torch.empty(43*1024**3,dtype=torch.uint8,device='cuda');del reserve
        runtime.lease=lambda:nullcontext();torch.cuda.empty_cache=lambda:None
        (evaluate if os.environ['ACTION'].startswith('eval_') else learner)()
    c.write(R/'private'/('EXIT_'+os.environ['ACTION']+'.json'),dict(exit_code=0,epoch=time.time()))

if __name__=='__main__':
    try:main()
    except BaseException:b.counts();c.write(R/'private/failures'/(os.environ['ACTION']+'_'+str(os.getpid())+'.json'),dict(epoch=time.time(),traceback=traceback.format_exc()));raise
