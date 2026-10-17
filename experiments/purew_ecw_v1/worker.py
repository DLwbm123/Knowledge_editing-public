"""Current-source ECW quick validation; no evaluation labels enter the learner."""
import math
import os
from pathlib import Path
import signal
import random
import time
import traceback
from contextlib import nullcontext
import numpy as np
import torch
import torch.nn.functional as F
import runtime
from runtime import R,c,b,load_learner,track,backward,lease,peak_memory
from core import Memory,basis,inner
from access import install
from base_learner import batches

WINDOW=runtime.WINDOW
CONTROL=runtime.CONTROL


def loss(runtime_model,full,removed,anchors,slot):
    value=0.;ce=0.;scope=0.
    for k in (0,1+(slot-1)%4):
        z,y=b.logits(runtime_model,full[k]);part=.5*F.cross_entropy(z.double(),y)
        value=value+part;ce+=float(part.detach())
        if removed:
            zr,yr=b.logits(runtime_model,removed[k]);assert torch.equal(y,yr)
            part=.5*F.kl_div(F.log_softmax(zr.double(),-1),anchors[k],reduction='batchmean')
            value=value+part;scope+=float(part.detach())
    return value,ce,scope


def scope_anchors(model,removed):
    with torch.no_grad():return [b.logits(model,batch)[0].double().softmax(-1).detach() for batch in removed]


def probes(model,weight,memory,full,removed,owner,block):
    gen=torch.Generator(device=model.device);gen.manual_seed(20260912+owner+(100000 if block=='evidence' else 0))
    events=[]
    for j in range(4):
        row=torch.zeros_like(weight,dtype=torch.float64)
        for k,coefficient in enumerate((.5,.125,.125,.125,.125)):
            z,y=b.logits(model,full[k])
            if block=='answer':
                p=z.detach().softmax(-1);root=p.sqrt()
                noise=torch.randn(z.shape,device=z.device,dtype=z.dtype,generator=gen)
                u=root*noise-p*(root*noise).sum(-1,keepdim=True)
                scalar=(z*u).sum()*math.sqrt(coefficient/len(y))/2
            else:
                zr,yr=b.logits(model,removed[k]);assert torch.equal(y,yr)
                difference=F.log_softmax(z.double(),-1).gather(-1,y[:,None]).mean()-F.log_softmax(zr.double(),-1).gather(-1,yr[:,None]).mean()
                p=difference.detach().sigmoid()
                noise=torch.randn((),device=z.device,generator=gen)
                scalar=difference*noise*torch.sqrt(p*(1-p))*math.sqrt(coefficient)/2
            row+=backward(scalar,weight).double()
        events.append(memory.consolidate([row],block));del row
    return dict(block=block,rows=4,projected_trace=events[-1]['projected_trace'],full_probe_mass=sum(e['full_probe_mass'] for e in events))


def attempt(model,weight,master,memory,full,removed,anchors,slot):
    before=weight.detach().float().clone()
    total,ce,scope=loss(model,full,removed,anchors,slot)
    gradient=backward(total,weight);norm=float(gradient.norm());assert math.isfinite(norm)
    gradient*=min(1.,5/max(norm,1e-30))
    projected=memory.project(gradient);memory.N+=torch.outer(projected,projected)
    coordinate,event=memory.solve(gradient)
    direction=memory.embed(coordinate).float();event.update(source_CE=ce,scope_loss=scope,
        gradient_norm=norm,projected_gradient_norm=math.sqrt(inner(memory.project(gradient),memory.project(gradient))),
        current_risk=memory.q,current_residual_energy=memory.tail(memory.residual),candidates=[])
    baseline=float(total.detach());del total
    accepted=False
    for backtrack in range(8):
        candidate_master=master+direction*(.5**backtrack);candidate_native=candidate_master.half()
        actual=candidate_native.float()-before
        _,_,risk=memory.inspect(actual);changed=int(torch.count_nonzero(actual))
        reasons=[]
        if not changed:reasons.append('ZERO_NATIVE')
        if risk['coordinate_risk']>.01:reasons.append('COORDINATE_RISK')
        if risk['native_risk_bound']>.01:reasons.append('NATIVE_RESIDUAL_ENVELOPE')
        slope=inner(gradient,actual)
        actual_loss=None
        if not reasons:
            with torch.no_grad():
                weight.copy_(candidate_native)
                try:actual_loss=float(loss(model,full,removed,anchors,slot)[0])
                finally:weight.copy_(before.half())
            if slope>=0 or actual_loss>baseline+1e-4*slope:reasons.append('ACTUAL_SOURCE_ARMIJO')
        event['candidates'].append(dict(backtrack=backtrack,changed_elements=changed,
            actual_loss=actual_loss,slope=slope,reasons=reasons,**risk))
        if not reasons:
            memory.advance(actual,gradient)
            with torch.no_grad():weight.copy_(candidate_native)
            master=candidate_master;accepted=True;b.COUNTS['committed_updates']+=1;b.call('C')
            event.update(actual_native_norm=float(actual.norm()),accepted_loss=actual_loss,**risk)
            break
    if not accepted:
        assert torch.equal(weight,before.half())
        event.update(actual_native_norm=0.,native_risk_bound=.5*(math.sqrt(max(0.,2*memory.q))+math.sqrt(memory.tail(memory.residual)))**2)
    event.update(accepted=accepted,slot=slot,zero_native=not accepted)
    b.counts();return master,event


def save(weight,master,memory,owner,slot,boundary):
    state=dict(native=weight.detach().cpu(),master=master.detach().cpu(),memory=memory.state(),owner=owner,slot=slot,
        boundary=boundary,rng=c.rng(),numpy_rng=(np.random.get_state()[0],torch.from_numpy(np.random.get_state()[1].astype('int64')),*np.random.get_state()[2:]))
    b.save(R/'private/active/ECW/state.pt',state)
    c.write(CONTROL/'CHECKPOINT.json',dict(owner=owner,slot=slot,boundary=boundary))


def learner():
    signal.signal(signal.SIGTERM,b.request_stop);signal.signal(signal.SIGINT,b.request_stop)
    medical=os.environ['ACTION']=='medical'
    model=load_learner();hooks=track(model)
    try:
        install(R,WINDOW,os.environ['DATA_ROOT']+'/knowledge_editing/data/m3bench')
        for name in ['EVAL_LEDGER.json','operator/source.json']:
            try:(R/'private'/name).read_text()
            except PermissionError:pass
            else:raise AssertionError('Evaluator/operator material readable')
        weight=model.get_module(c.LAYER).weight;shape=b.schema(model.model)
        frozen={n:p._version for n,p in model.model.named_parameters() if p is not weight}
        original=weight.detach().clone();left,right=basis(weight)
        memory=Memory(left,right,weight.shape);master=weight.detach().float().clone();weight.requires_grad_(True)
        for owner in ([1] if medical else [1,2,3]):
            while not (WINDOW/'current.json').exists() or c.read(WINDOW/'current.json')['quick_owner']!=owner:c.budget();time.sleep(.2)
            packet=c.read(WINDOW/'current.json');seed=packet['task']['seed']
            random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
            full,removed=batches(model,packet);anchors=scope_anchors(model,removed)
            if owner==1:
                # Statistics-only current-source initialization, not extra training or old-case warm-up.
                for k in range(5):
                    z,y=b.logits(model,full[k]);g=backward(F.cross_entropy(z,y),weight);z=memory.project(g);memory.N+=torch.outer(z,z)
            memory.prepare()
            if owner==1 and not medical:
                features={};hook=model.get_module(c.LAYER).register_forward_pre_hook(lambda m,a:features.update(value=a[0].detach().clone()))
                old_rng=c.rng();old_state=memory.state()
                with torch.no_grad():b.logits(model,full[0]);before=features['value'].clone()
                trial,event=attempt(model,weight,master,memory,full,[],[],1)
                assert event['accepted'] and event['actual_native_norm']>0,'Real native admission rejected; do not bypass'
                with torch.no_grad():b.logits(model,full[0]);assert torch.equal(before,features['value']);weight.copy_(original)
                hook.remove();memory=Memory.restore(old_state,model.device);memory.prepare();c.restore_rng(old_rng)
                b.guard(model,weight,shape,frozen)
                c.write(R/'public/REAL_VLM_ADMISSION.json',dict(status='PASS',real_native_write=True,upstream_exact=True,parameters_unchanged=True,temporary_updates=1))
            slots=1 if medical else 20
            for slot in range(1,slots+1):
                c.budget();master,event=attempt(model,weight,master,memory,full,removed,anchors,slot)
                if not medical:b.COUNTS['attempt_slots']=b.COUNTS.get('attempt_slots',0)+1
                b.counts();peak_memory();event.update(owner=owner,medical=medical)
                c.write(R/'private/steps'/('medical' if medical else 'ECW')/str(owner)/(str(slot)+'.json'),event)
                c.write(R/'public/PROGRESS.json',dict(status='QUICK_DIAGNOSTIC',owner=owner,slot=slot,accepted=event['accepted'],medical=medical,epoch=time.time()))
                if b.STOP_REQUESTED:save(weight,master,memory,owner,slot,False);raise RuntimeError('Cooperative stop saved; no automatic replay')
            answer=probes(model,weight,memory,full,[],packet['task']['order'],'answer')
            evidence=probes(model,weight,memory,full,removed,packet['task']['order'],'evidence') if removed else None
            c.write(R/'public'/('MEDICAL_MECHANICAL.json' if medical else 'SOURCE_'+str(owner)+'.json'),dict(status='MECHANICAL_ONLY',answer=answer,evidence=evidence,parameters=sum(p.numel() for p in model.model.parameters()),added_parameters=0))
            del full,removed,anchors
            if medical:
                with torch.no_grad():weight.copy_(original)
            else:save(weight,master,memory,owner,slots,True)
            c.write(CONTROL/(str(owner)+'.DONE.json'),dict(owner=owner,slots=slots,outputs='NO_SCIENTIFIC_SCORING'))
        weight.requires_grad_(False);b.guard(model,weight,shape,frozen);peak_memory()
        if not medical:b.save(R/'private/final/native.pt',dict(native=weight.detach().cpu(),schema=shape,parameters=sum(p.numel() for p in model.model.parameters())))
    finally:hooks.close();b.counts()


def evaluate():
    model,bindings=c.load(4);hooks=track(model)
    try:
        tasks=c.read(R/'private/BENCHMARK146_QUEUE.json')['tasks'];phase=os.environ['EVAL_PHASE']
        if phase=='base':ids=[tasks[i-1]['edit_id'] for i in (1,50,100)]
        else:
            native=torch.load(R/'private/final/native.pt',map_location='cpu',weights_only=True)
            assert b.schema(model.model)==native['schema']
            with torch.no_grad():model.get_module(c.LAYER).weight.copy_(native['native'].to(model.device))
            ids=[tasks[2]['edit_id'],next(e['all_probe_query_ids'][0] for e in tasks[2]['events'] if e['task']=='T2G')]
        ledger=c.read(R/'private/EVAL_LEDGER.json')
        for q in ids:
            row=ledger['queries'][q];d=b.generate(model,bindings,row,R/'private/evaluation'/phase/(q+'.json'),dict(stage='ECW_QUICK_'+phase))
            if phase=='base':
                from contract import base_completion
                assert d['R0']==base_completion(bindings[row['opaque_Base_id']]['output'])
            if phase=='reload':assert d['R0']==c.read(R/'private/evaluation/final'/(q+'.json'))['R0']
        c.write(R/'public'/('BASE_IDENTITY.json' if phase=='base' else 'NATIVE_RELOAD.json' if phase=='reload' else 'FINAL_OUTPUTS.json'),dict(status='PASS',outputs=len(ids),scientific_scored=False))
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
    except BaseException:
        b.counts();c.write(R/'private/failures'/(os.environ['ACTION']+'_'+str(os.getpid())+'.json'),dict(epoch=time.time(),traceback=traceback.format_exc()));raise
