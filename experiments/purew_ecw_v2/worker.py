"""Current-only source stage and serial output-only evaluator for ECW revision2."""
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


def probes(model,weight,memory,full,removed,order,block):
    gen=torch.Generator(device=model.device);gen.manual_seed(20260912+order+(100000 if block=='evidence' else 0));rows=[]
    for j in range(4):
        row=torch.zeros_like(weight,dtype=torch.float64)
        for k,coefficient in enumerate((.5,.125,.125,.125,.125)):
            z,y=b.logits(model,full[k])
            if block=='answer':
                p=z.detach().softmax(-1);root=p.sqrt();noise=torch.randn(z.shape,device=z.device,dtype=z.dtype,generator=gen)
                u=root*noise-p*(root*noise).sum(-1,keepdim=True);scalar=(z*u).sum()*math.sqrt(coefficient/len(y))/2
            else:
                zr,yr=b.logits(model,removed[k]);assert torch.equal(y,yr)
                difference=F.log_softmax(z.double(),-1).gather(-1,y[:,None]).mean()-F.log_softmax(zr.double(),-1).gather(-1,yr[:,None]).mean()
                p=difference.detach().sigmoid();noise=torch.randn((),device=z.device,generator=gen)
                scalar=difference*noise*torch.sqrt(p*(1-p))*math.sqrt(coefficient)/2
            row+=backward(scalar,weight).double()
        rows.append(row)
    event=memory.consolidate(torch.stack(rows),block);del rows;return event


def measure(model,full,removed,anchors):
    with torch.no_grad():
        ce=sum(float(loss(model,full,[],[],slot)[0])/4 for slot in range(1,5))
        scope=sum(float(loss(model,full,removed,anchors,slot)[0])/4 for slot in range(1,5))-ce if removed else 0.
    return dict(weighted_all_FIT_CE=ce,scope_KL=scope)


def attempt(model,weight,master,memory,full,removed,anchors,slot):
    before=weight.detach().clone();total,ce,scope=loss(model,full,removed,anchors,slot)
    g=backward(total,weight);norm=float(g.norm());assert math.isfinite(norm);g*=min(1.,5/max(norm,1e-30))
    delta,event=memory.solve(g);direction=delta.reshape_as(master).float();baseline=float(total.detach());del total,delta
    event.update(source_CE=ce,scope_loss=scope,gradient_norm=norm,current_risk=memory.q,candidates=[])
    accepted=False
    for index in range(8):
        candidate_master=master+direction*(.5**index);candidate_native=candidate_master.half()
        actual=candidate_native.double()-before.double();risk=memory.inspect(actual);changed=int(torch.count_nonzero(actual));reasons=[]
        if not changed:reasons.append('ZERO_NATIVE')
        if risk['native_risk_bound']>.01:reasons.append('SAMPLED_NATIVE_RISK')
        slope=dot(g,actual);actual_loss=None
        if not reasons:
            with torch.no_grad():
                weight.copy_(candidate_native)
                try:actual_loss=float(loss(model,full,removed,anchors,slot)[0])
                finally:weight.copy_(before)
            if slope>=0 or actual_loss>baseline+1e-4*slope:reasons.append('ACTUAL_SOURCE_ARMIJO')
        event['candidates'].append(dict(backtrack=index,changed_elements=changed,actual_loss=actual_loss,slope=slope,reasons=reasons,**risk))
        if not reasons:
            memory.advance(actual)
            with torch.no_grad():weight.copy_(candidate_native)
            master=candidate_master;accepted=True;b.COUNTS['committed_updates']+=1;b.call('C')
            event.update(accepted_loss=actual_loss,**risk);break
    if not accepted:
        assert torch.equal(weight,before);event.update(actual_native_norm=0.,native_risk_bound=memory.q+memory.eq)
    event.update(accepted=accepted,slot=slot,zero_native=not accepted);b.counts();return master,event


def save(weight,master,memory,owner,slot,boundary):
    assert torch.equal(weight,master.half())
    state=dict(native=weight.detach().cpu(),master=master.detach().cpu(),memory=memory.state(),owner=owner,slot=slot,boundary=boundary,
        rng=c.rng(),numpy_rng=(np.random.get_state()[0],torch.from_numpy(np.random.get_state()[1].astype('int64')),*np.random.get_state()[2:]))
    b.save(R/'private/active'/ARM/'state.pt',state);c.write(CONTROL/'CHECKPOINT.json',dict(owner=owner,slot=slot,boundary=boundary))


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
        original=weight.detach().clone();restricted=ARM.startswith('X');packet=c.read(WINDOW/'current.json');assert packet['quick_owner']==owner
        if owner>1:
            state=torch.load(R/'private/active'/ARM/'state.pt',map_location='cpu',weights_only=True)
            assert state['owner']==owner-1 and state['slot']==20 and state['boundary']
            with torch.no_grad():weight.copy_(state['native'].to(model.device))
            master=state['master'].to(model.device);memory=Memory.restore(state['memory'],model.device);del state
        else:master=weight.detach().float().clone();memory=Memory(weight,restricted)
        seed=packet['task']['seed'];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        full,removed=batches(model,packet);anchors=scope_anchors(model,removed);memory.prepare();weight.requires_grad_(True)
        initial=measure(model,full,removed,anchors)
        if owner==1:
            features={};hook=model.get_module(c.LAYER).register_forward_pre_hook(lambda m,a:features.update(value=a[0].detach().clone()));old_rng=c.rng();old_state=memory.state()
            with torch.no_grad():b.logits(model,full[0]);before=features['value'].clone()
            trial,event=attempt(model,weight,master,memory,full,[],[],1)
            assert event['accepted'] and event['actual_native_norm']>0,'Scientific native admission rejected; no bypass'
            with torch.no_grad():b.logits(model,full[0]);assert torch.equal(before,features['value']);weight.copy_(original)
            hook.remove();memory=Memory.restore(old_state,model.device);c.restore_rng(old_rng);b.guard(model,weight,shape,frozen)
            c.write(R/'public'/('REAL_VLM_ADMISSION_'+ARM+'.json'),dict(status='PASS',real_native_write=True,upstream_exact=True,parameters_unchanged=True,temporary_updates=1))
        slots=20
        for slot in range(1,slots+1):
            c.budget();master,event=attempt(model,weight,master,memory,full,removed,anchors,slot)
            b.COUNTS['attempt_slots']=b.COUNTS.get('attempt_slots',0)+1
            b.counts();event.update(owner=owner,medical_gate=bool(removed));c.write(R/'private/steps'/ARM/str(owner)/(str(slot)+'.json'),event)
            if slot%20==0 or b.STOP_REQUESTED:save(weight,master,memory,owner,slot,False)
            c.write(R/'public/PROGRESS.json',dict(status='REVISION2_TRAINING',arm=ARM,owner=owner,slot=slot,accepted=event['accepted'],epoch=time.time()))
            if b.STOP_REQUESTED:raise RuntimeError('Cooperative stop saved; no implicit replay')
        final=measure(model,full,removed,anchors);answer=probes(model,weight,memory,full,[],packet['task']['order'],'answer')
        evidence=probes(model,weight,memory,full,removed,packet['task']['order'],'evidence') if removed else None
        c.write(R/'public'/('SOURCE_'+ARM+'_'+str(owner)+'.json'),dict(status='MECHANICAL_ONLY',initial=initial,final=final,answer=answer,evidence=evidence,added_parameters=0))
        weight.requires_grad_(False);b.guard(model,weight,shape,frozen);peak_memory();save(weight,master,memory,owner,slots,True)
        b.save(CONTROL/'native.pt',dict(native=weight.detach().cpu(),schema=shape,parameters=sum(p.numel() for p in model.model.parameters())))
        if owner==3:b.save(R/'private/final'/ARM/'native.pt',dict(native=weight.detach().cpu(),schema=shape,parameters=sum(p.numel() for p in model.model.parameters())))
        del full,removed,anchors,packet
    finally:hooks.close();b.counts()


def evaluate():
    from panels import expected_ids
    import base_evaluator as old
    old.expected_ids=expected_ids
    model,bindings=c.load(4);hooks=track(model)
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
    assert os.environ['CUDA_VISIBLE_DEVICES']=='4'
    with lease():
        assert b.available(4);reserve=torch.empty(43*1024**3,dtype=torch.uint8,device='cuda');del reserve
        runtime.lease=lambda:nullcontext();torch.cuda.empty_cache=lambda:None
        (evaluate if os.environ['ACTION'].startswith('eval_') else learner)()
    c.write(R/'private'/('EXIT_'+os.environ['ACTION']+'.json'),dict(exit_code=0,epoch=time.time()))

if __name__=='__main__':
    try:main()
    except BaseException:b.counts();c.write(R/'private/failures'/(os.environ['ACTION']+'_'+str(os.getpid())+'.json'),dict(epoch=time.time(),traceback=traceback.format_exc()));raise
