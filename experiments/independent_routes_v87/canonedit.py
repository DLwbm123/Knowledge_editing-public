"""Source-only two-basis CanonEdit migration and explicit TT deployment consumer."""
import itertools
import os
import signal
import traceback
import time
from pathlib import Path
import route_common as c
v=c.v
ARMS=['CE','RO','CANON','CANON_TT']


def classes():
    torch=v.torch
    class Residual(torch.nn.Module):
        def __init__(self,key,vectors):
            super().__init__();self.register_buffer('key',key);self.register_buffer('vectors',vectors)
            self.z=torch.nn.Parameter(torch.zeros(len(vectors),device=key.device))
        def residual(self,x):
            x=x/(x.square().mean(-1,keepdim=True).sqrt()+1e-8)
            scale=(x@self.key)/(self.key.square().sum()+1e-8)
            return scale[...,None]*(self.z@self.vectors)
    class Injection(torch.nn.Module):
        def __init__(self):
            super().__init__();self.r=torch.nn.Parameter(torch.zeros(4096,device=v.runtime.device))
        def residual(self,x):return self.r.expand(*x.shape[:-1],4096)
    return Residual,Injection


def score(hook,b,first_only=False):
    if hook:
        if first_only:
            mask=v.torch.zeros_like(b.labels,dtype=v.torch.bool)
            mask[0,int(v.torch.where(b.labels[0]!=-100)[0][0])-1]=True
            hook.set_request_routing(mask)
        else:hook.set_teacher_routing(b.labels)
    loss=v.runtime.compute_loss(b)
    return -loss*(b.labels!=-100).sum()


def margin(hook,new,old,first_only=False):return score(hook,new,first_only)-score(hook,old,first_only)


def ro(new,old,balanced):
    torch=v.torch;expert=classes()[1]();hook=v.r.q.scoped.replay.MedTraceLayerHook(v.runtime.get_module(v.r.c.LAYER),expert);hook.attach()
    opt=torch.optim.Adam(expert.parameters(),lr=2.)
    curve=[]
    try:
        for step in range(8):
            opt.zero_grad(set_to_none=True);ms=[]
            # Full sequence sums, same old answer for all paraphrases of this request.
            for j,(n,o) in enumerate(zip(new,old)):
                m=margin(hook,n,o,True);ms.append(float(m.detach()))
                weight=1. if balanced else (4. if j==0 else .5/4)
                loss=weight*((.25 if j==0 else 0.)-m).clamp_min(0).square()
                loss.backward();v.counts['backwards']+=1
            (.002*expert.r.square().sum()).backward();v.counts['operator_backwards']+=1
            opt.step();v.counts['updates']+=1
            with torch.no_grad():expert.r.mul_(min(1.,40/max(float(expert.r.norm()),1e-30)))
            curve.append(ms)
            if step>=3 and ms[0]>=.25 and min(ms[1:])>=0:break
    finally:hook.detach()
    return expert.r.detach().clone(),curve


def qp(a,b,B,M):
    """Enumerate active target halfspaces in the two-dimensional chart."""
    import numpy as np
    a,b,B,M=[x.detach().cpu().double().numpy() for x in (a,b,B,M)]
    candidates=[]
    if (b<=1e-8).all():candidates.append(np.zeros(a.shape[1]))
    inv=np.linalg.inv(B)
    for k in range(1,min(a.shape)+1):
        for ids in itertools.combinations(range(len(b)),k):
            A=a[list(ids)];rhs=b[list(ids)];gram=A@inv@A.T
            lam=np.linalg.lstsq(gram,rhs,rcond=1e-10)[0];z=inv@A.T@lam
            if np.min(lam)>=-1e-8 and np.min(a@z-b)>=-1e-6:candidates.append(z)
    candidates=[z for z in candidates if z@M@z<=36+1e-8 and np.isfinite(z).all()]
    return None if not candidates else v.torch.tensor(min(candidates,key=lambda z:z@B@z),device=v.runtime.device,dtype=v.torch.float32)


def values(expert,new,old,protected,grad=False):
    torch=v.torch;hook=v.r.q.scoped.replay.MedTraceLayerHook(v.runtime.get_module(v.r.c.LAYER),expert);hook.attach()
    ms=[];ss=[];aa=[];cc=[]
    try:
        for n,o in zip(new,old):
            m=margin(hook,n,o);ms.append(m.detach())
            if grad:aa.append(torch.autograd.grad(m,expert.z)[0]);v.counts['backwards']+=int(grad)
        for b in protected:
            s=score(hook,b);ss.append(s.detach())
            if grad:cc.append(torch.autograd.grad(s,expert.z)[0]);v.counts['backwards']+=int(grad)
    finally:hook.detach()
    return torch.stack(ms),torch.stack(ss),torch.stack(aa) if grad else None,torch.stack(cc) if grad else None


def gates(ms,ss,base):
    torch=v.torch;gamma=torch.tensor([.25,.10,.10,.10,.10],device=ms.device)
    change=ss-base
    return bool((ms>=gamma-1e-6).all() and change.abs().mean()<=1.5 and change.square().mean()<=2.25)


def canonical(expert,new,old,protected):
    torch=v.torch
    # Dense rank-one Frobenius Gram, normalized by mean diagonal; no held-out inputs.
    raw=expert.vectors@expert.vectors.T/(expert.key.square().sum()+1e-8)
    M=raw/raw.diag().mean().clamp_min(1e-12)
    M=M+1e-8*torch.eye(2,device=M.device)
    with torch.no_grad():before,baseline,_,_=values(expert,new,old,protected)
    gamma=torch.tensor([.25,.10,.10,.10,.10],device=before.device);curve=[];accepted=[]
    for iteration in range(5):
        ms,ss,a,C=values(expert,new,old,protected,True)
        if gates(ms,ss,baseline):accepted.append((float((ss-baseline).square().mean()),expert.z.detach().clone()));break
        if iteration==4:break
        B=C.T@C/len(C)+.04*M+1e-10*torch.eye(2,device=M.device)
        # Correction is constrained against the current signed deficits, not a clipped
        # deficit that would spuriously forbid safe reductions of satisfied margins.
        delta=qp(a,gamma-ms,B,M)
        curve.append(dict(iteration=iteration,margin=ms.tolist(),proposal=None if delta is None else delta.tolist()))
        if delta is None:break
        with torch.no_grad():
            proposal=expert.z+delta
            if float(proposal@M@proposal)>36+1e-6:break
            expert.z.copy_(proposal)
        v.counts['updates']+=1
    with torch.no_grad():
        expert.z.copy_(min(accepted,key=lambda x:x[0])[1] if accepted else torch.zeros_like(expert.z))
        after,ss,_,_=values(expert,new,old,protected)
    return dict(status='ACCEPT' if accepted else 'NO_VALID_WRITE_NOOP',before=before.tolist(),after=after.tolist(),protected_MSE=float((ss-baseline).square().mean()),protected_MAE=float((ss-baseline).abs().mean()),curve=curve),baseline


def compress(task,original,source,data,canonical):
    torch=v.torch;expert=v.r.p.expert(original,task['seed'],v.runtime.device)
    expert.load_state_dict(v.r.mm.canonicalize(original)[0]);opt=torch.optim.Adam(expert.parameters(),lr=.01)
    x=torch.cat([data[id(b)]['x'] for b in source]);target=canonical.residual(x).detach();denom=target.square().mean().clamp_min(1e-8)
    if float(target.norm())==0:
        with torch.no_grad():expert.G1.zero_()
        return expert
    for step in range(256):
        opt.zero_grad(set_to_none=True);loss=(expert.residual(x)-target).square().mean()/denom
        assert torch.isfinite(loss);loss.backward();opt.step();v.counts['operator_backwards']+=1;v.counts['updates']+=1
    return expert


def main():
    part=int(os.environ['PARTITION'])
    with (v.ROOT/f'WORKER_ONCE_{part}').open('x'):pass
    v.allocations={k:n//2 for k,n in v.limits['counts'].items()}
    signal.signal(signal.SIGALRM,lambda *_: (_ for _ in ()).throw(TimeoutError('Original deadline')))
    signal.alarm(max(1,int(v.limits['hard_end']-time.time()-300)))
    v.setup();tasks=v.read(v.ROOT/'TASKS.json');by_owner={};new_all=[]
    for task in tasks:
        its=v.prepare(task,v.read(v.ROOT/'SOURCE_ROWS.json')[str(task['order'])],True)
        bs=[v.r.old.batch_on(v.runtime,x[1]) for x in its];by_owner[task['order']]=bs;new_all+=bs
    torch=v.torch;R=classes()[0]
    old_by={};old_all=[]
    for task in tasks:
        owner=task['order'];rows=v.read(v.ROOT/'SOURCE_ROWS.json')[str(owner)];answer=v.read(v.ROOT/'OLD_SOURCE_ANSWERS.json')[str(owner)]['raw_answer']
        assert answer.strip()
        olditems=v.prepare(task,[dict(row,reference=answer) for row in rows],True)
        # Every FIT uses the same frozen source-native model answer as old contrast.
        bs=[v.r.old.batch_on(v.runtime,x[1]) for x in olditems]
        assert all(torch.equal(bs[0].labels[bs[0].labels!=-100],b.labels[b.labels!=-100]) for b in bs)
        old_by[owner]=bs;old_all+=bs
    probe_b=by_owner[tasks[part]['order']][0]
    probe_key=torch.ones(14336,device=v.runtime.device)
    probe=R(probe_key,torch.full((2,4096),1e-3,device=v.runtime.device));probe.z.data.fill_(1.)
    ph=v.r.q.scoped.replay.MedTraceLayerHook(v.runtime.get_module(v.r.c.LAYER),probe);ph.attach();ph.set_teacher_routing(probe_b.labels)
    try:
        with torch.no_grad():full=v.runtime.model(**probe_b.forward_kwargs()).logits.detach().cpu()
    finally:ph.detach()
    with c.CachedSource(new_all+old_all) as cache:
        ph.attach();ph.set_teacher_routing(probe_b.labels)
        try:
            cache.cache.current=id(probe_b)
            try:
                with torch.no_grad():cached=v.runtime.model(**probe_b.forward_kwargs()).logits.detach().cpu()
            finally:cache.cache.current=None
            torch.testing.assert_close(cached,full,rtol=1e-5,atol=5e-4)
            v.write(v.ROOT/f'CACHED_LOGITS_PARITY_{part}.json',dict(status='PASS',nonzero_rank_one=True,rtol=1e-5,atol=5e-4))
        finally:ph.detach()
        del probe,full,cached
        for task in tasks[part::2]:
            owner=task['order'];new=by_owner[owner];old=old_by[owner]
            # Fixed cyclic five other source-native prompts; model-behavior protection only.
            owners=[t['order'] for t in tasks];pos=owners.index(owner)
            protected=[old_by[owners[(pos+j)%8]][0] for j in range(1,6)]
            original=v.r.p.load_state(v.PARENT/'private/candidates/C'/f'{owner}.pt')['states']['BASE']
            from m3bench_repro.editors.llava_runtime import seed_everything
            seed_everything(task['seed']);ce=v.r.p.expert(original,task['seed'],v.runtime.device);ce.load_state_dict(v.r.mm.canonicalize(original)[0])
            hook=v.r.q.scoped.replay.MedTraceLayerHook(v.runtime.get_module(v.r.c.LAYER),ce);hook.attach()
            try:c.ce_train(ce,hook,new)
            finally:hook.detach()
            states={'CE':v.r.old.snapshot(ce)};del ce
            vectors=[];curves=[]
            for balanced in [False,True]:
                residual,curve=ro(new,old,balanced);vectors.append(residual);curves.append(curve)
            key=cache.data[id(new[0])]['key'];key=key/(key.square().mean().sqrt()+1e-8)
            expert=R(key,torch.stack(vectors));raw=dict(key=key.detach(),vectors=expert.vectors.detach(),z=torch.tensor([1.,0.],device=key.device))
            states['RO']=raw
            audit,baseline=canonical(expert,new,old,protected)
            states['CANON']=dict(key=key.detach(),vectors=expert.vectors.detach(),z=expert.z.detach().clone())
            tt=compress(task,original,new,cache.data,expert)
            th=v.r.q.scoped.replay.MedTraceLayerHook(v.runtime.get_module(v.r.c.LAYER),tt);th.attach()
            try:
                with torch.no_grad():ms=torch.stack([margin(th,n,o) for n,o in zip(new,old)]);ss=torch.stack([score(th,b) for b in protected])
            finally:th.detach()
            tt_pass=gates(ms,ss,baseline)
            if not tt_pass:
                with torch.no_grad():tt.G1.zero_()
            states['CANON_TT']=v.r.old.snapshot(tt)
            audit.update(TT_nonlinear_pass=tt_pass,TT_margin=ms.tolist(),RO_curves=curves,scope='SOURCE_ONLY_MODEL_OUTPUT_REPLACEMENT_SINGLE_LAYER_TWO_BASIS_RMS_CHART')
            v.write(v.ROOT/'training'/f'{owner}.json',audit)
            for arm,state in states.items():
                path=v.ROOT/'candidates'/arm/f'{owner}.pt';v.save_state(path,dict(state=state,owner=owner,arm=arm))
                saved=v.r.p.load_state(path)['state'];assert all(torch.equal(saved[k].cpu(),state[k].cpu()) for k in state)
            v.progress(stage='FULL_METHOD_TRAINING_COMPLETE',owner=owner,canon_status=audit['status'],TT_pass=tt_pass)
    v.write(v.ROOT/f'TRAIN_COMPLETE_{part}.json',dict(count=16,time=time.time()))
    while not all((v.ROOT/f'TRAIN_COMPLETE_{p}.json').exists() for p in range(2)):
        v.budget();time.sleep(2)
    if part==0:v.write(v.ROOT/'ALL_CANDIDATES_FROZEN.json',dict(count=32,time=time.time()))
    def loader(arm,owner):
        s=v.r.p.load_state(v.ROOT/'candidates'/arm/f'{owner}.pt')['state']
        if arm in ['CE','CANON_TT']:return v.r.p.expert(s,owner,v.runtime.device)
        e=R(s['key'].to(v.runtime.device),s['vectors'].to(v.runtime.device));e.z.data.copy_(s['z']);return e
    c.evaluate(tasks[part::2],ARMS,loader)
    v.write(v.ROOT/f'EVAL_COMPLETE_{part}.json',dict(counts=v.counts,time=time.time()))


if __name__=='__main__':
    code=0
    try:main()
    except BaseException as e:
        code=1;v.write(v.ROOT/f'FAILURE_worker_{v.PART}.json',dict(error=repr(e),traceback=traceback.format_exc(),counts=v.counts,time=time.time()))
        v.write(v.ROOT/'STOP',dict(reason='CANON_WORKER_FAILURE',time=time.time()));raise
    finally:
        v.progress(exit_code=code);v.write(v.ROOT/f'EXIT_worker_{v.PART}.json',dict(exit_code=code,counts=v.counts,time=time.time()))
