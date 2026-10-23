"""Fixed-gauge TT-native chart and source-answer safety qualification."""
import time
import route_common as c
import canonedit as old
v=c.v
ARMS=['CE','BAL_CE','TT_MARGIN','TT_SOURCE']


def chart_class():
    torch=v.torch
    class Chart(torch.nn.Module):
        def __init__(self,base,first,second):
            super().__init__()
            for k in v.r.mm.KEYS:
                self.register_buffer('base_'+k,base[k]);self.register_buffer('d1_'+k,first[k]-base[k]);self.register_buffer('d2_'+k,second[k]-base[k])
            self.z=torch.nn.Parameter(torch.zeros(2,device=v.runtime.device))
        def state(self,z=None):
            z=self.z if z is None else z
            return {k:getattr(self,'base_'+k)+z[0]*getattr(self,'d1_'+k)+z[1]*getattr(self,'d2_'+k) for k in v.r.mm.KEYS}
        def residual(self,x):
            b,a=v.r.mm.factors(self.state());x=x/(x.square().mean(-1,keepdim=True).sqrt()+1e-6)
            return (x@a.T)@b.T
    return Chart


def gram(expert):
    torch=v.torch;z=expert.z.detach();b,a=v.r.mm.factors(expert.state(z));factors=[]
    for i in range(2):
        direction=torch.zeros_like(z);direction[i]=1
        _,(db,da)=torch.autograd.functional.jvp(lambda q:v.r.mm.factors(expert.state(q)),z,direction)
        factors.append((torch.cat([db,b],dim=1),torch.cat([a,da],dim=0)))
    M=torch.stack([torch.stack([((li.T@lj)*(ri@rj.T)).sum() for lj,rj in factors]) for li,ri in factors])
    v.counts['operator_backwards']+=4
    M=(M+M.T)/2;M=M/M.diag().mean().clamp_min(1e-12)
    return M+1e-8*torch.eye(2,device=z.device)


def solve(expert,new,previous,protected):
    torch=v.torch
    with torch.no_grad():before,baseline,_,_=old.values(expert,new,previous,protected)
    gamma=torch.tensor([.25,.10,.10,.10,.10],device=before.device);curve=[];accepted=False
    M0=gram(expert).detach()
    for iteration in range(5):
        ms,ss,A,C=old.values(expert,new,previous,protected,True)
        delta_score=ss-baseline
        row=dict(iteration=iteration,z=expert.z.detach().tolist(),margin=ms.tolist(),protected_MAE=float(delta_score.abs().mean()),protected_MSE=float(delta_score.square().mean()))
        if old.gates(ms,ss,baseline):accepted=True;curve.append(dict(row,status='ACCEPT'));break
        if iteration==4:curve.append(dict(row,status='CORRECTION_LIMIT'));break
        M=gram(expert).detach()
        B=C.T@C/len(C)+.04*M+1e-10*torch.eye(2,device=M.device)
        proposal=old.qp(A,gamma-ms,B,M)
        row['proposal']=None if proposal is None else proposal.tolist();curve.append(row)
        if proposal is None:break
        with torch.no_grad():
            z=expert.z+proposal
            if float(z@M0@z)>36+1e-6:curve[-1]['status']='TOTAL_RADIUS_REJECT';break
            expert.z.copy_(z)
        v.counts['updates']+=1
    if not accepted:
        with torch.no_grad():expert.z.zero_()
    with torch.no_grad():ms,ss,_,_=old.values(expert,new,previous,protected)
    return dict(status='ACCEPT' if accepted else 'NO_VALID_WRITE_NOOP',before=before.tolist(),after=ms.tolist(),z=expert.z.detach().tolist(),curve=curve)


def balanced(expert,hook,batches):
    torch=v.torch
    opt=torch.optim.Adam([{'params':[p for k,p in expert.named_parameters() if k in ('G1','G2')],'lr':1e-3},{'params':[p for k,p in expert.named_parameters() if k in ('G3','G4')],'lr':1e-4}],betas=(.9,.999),eps=1e-8)
    for step in range(160):
        opt.zero_grad(set_to_none=True)
        for view,weight in [(0,.2),(1+step%4,.8)]:
            hook.set_teacher_routing(batches[view].labels);loss=v.runtime.compute_loss(batches[view]);assert torch.isfinite(loss)
            (weight*loss).backward();v.counts['backwards']+=1
        torch.nn.utils.clip_grad_norm_(expert.parameters(),1.);opt.step();v.counts['updates']+=1


def main():
    tasks,items,by_owner,new_all=c.prepare();torch=v.torch;Chart=chart_class();old_by={};old_all=[]
    for task in tasks:
        owner=task['order'];answer=v.read(v.ROOT/'OLD_SOURCE_ANSWERS.json')[str(owner)]['raw_answer']
        rows=[dict(x,reference=answer) for x in v.read(v.ROOT/'SOURCE_ROWS.json')[str(owner)]]
        batches=[v.r.old.batch_on(v.runtime,x[1]) for x in v.prepare(task,rows,True)];old_by[owner]=batches;old_all+=batches
    probe_state=v.r.p.load_state(v.ROOT/'candidates/CE/1.pt')['state']
    probe=v.r.p.expert(probe_state,1,v.runtime.device);batch=by_owner[1][0]
    ph=v.r.q.scoped.replay.MedTraceLayerHook(v.runtime.get_module(v.r.c.LAYER),probe);ph.attach();ph.set_teacher_routing(batch.labels)
    try:
        with torch.no_grad():full=v.runtime.model(**batch.forward_kwargs()).logits.detach().cpu()
    finally:ph.detach()
    with c.CachedSource(new_all+old_all) as cache:
        ph.attach();ph.set_teacher_routing(batch.labels)
        try:
            cache.cache.current=id(batch)
            try:
                with torch.no_grad():cached=v.runtime.model(**batch.forward_kwargs()).logits.detach().cpu()
            finally:cache.cache.current=None
            torch.testing.assert_close(full,cached,rtol=1e-5,atol=5e-4)
            v.write(v.ROOT/'CACHED_LOGITS_PARITY.json',dict(status='PASS',nonzero_matched_CE=True,rtol=1e-5,atol=5e-4))
        finally:ph.detach()
        del probe,full,cached
        for task in tasks:
            owner=task['order'];new=by_owner[owner]
            initial=v.r.p.load_state(v.PARENT/'private/candidates/C'/f'{owner}.pt')['states']['BASE'];initial=v.r.mm.canonicalize(initial)[0]
            ce=v.r.p.load_state(v.ROOT/'candidates/CE'/f'{owner}.pt')['state'];ce={k:x.to(v.runtime.device) for k,x in ce.items()};initial={k:x.to(v.runtime.device) for k,x in initial.items()}
            from m3bench_repro.editors.llava_runtime import seed_everything
            seed_everything(task['seed']);e=v.r.p.expert(initial,task['seed'],v.runtime.device);e.load_state_dict(initial)
            hook=v.r.q.scoped.replay.MedTraceLayerHook(v.runtime.get_module(v.r.c.LAYER),e);hook.attach()
            try:balanced(e,hook,new)
            finally:hook.detach()
            bal=v.r.old.snapshot(e);bal={k:x.to(v.runtime.device) for k,x in bal.items()};expert=Chart(initial,ce,bal)
            # Endpoints must exactly reproduce the actual deployable TT, not an RO proxy.
            x=torch.ones(3,14336,device=v.runtime.device)
            for z,state in [(torch.tensor([1.,0.],device=x.device),ce),(torch.tensor([0.,1.],device=x.device),bal)]:
                with torch.no_grad():
                    expert.z.copy_(z);e.load_state_dict(state);torch.testing.assert_close(expert.residual(x),e.residual(x),rtol=1e-5,atol=5e-4)
            with torch.no_grad():expert.z.zero_()
            owners=[t['order'] for t in tasks];pos=owners.index(owner);protected=[old_by[owners[(pos+j)%8]][0] for j in range(1,6)]
            audit=solve(expert,new,old_by[owner],protected)
            state={k:t.detach().clone() for k,t in expert.state().items()}
            for arm,s in [('BAL_CE',bal),('TT_MARGIN',state)]:
                p=v.ROOT/'candidates'/arm/f'{owner}.pt';v.save_state(p,dict(state=s,owner=owner,arm=arm));loaded=v.r.p.load_state(p)['state'];assert all(torch.equal(s[k].cpu(),loaded[k].cpu()) for k in s)
            v.write(v.ROOT/'training'/f'{owner}.json',audit);v.progress(stage='TT_NATIVE_TRAINING',owner=owner,status=audit['status'])
    c.restore()
    for task in tasks:
        owner=task['order'];state=v.r.p.load_state(v.ROOT/'candidates/TT_MARGIN'/f'{owner}.pt')['state'];expert=v.r.p.expert(state,task['seed'],v.runtime.device)
        for view,item in enumerate(items[owner]):
            hook=v.r.q.scoped.replay.MedTraceLayerHook(v.runtime.get_module(v.r.c.LAYER),expert);hook.attach()
            try:answer=c.generate(hook,item)
            finally:hook.detach()
            tokens=list(answer.raw_token_ids)
            v.write(v.ROOT/'qualification'/f'{owner}_{view}.json',dict(key=f'{owner}:{view}',source=item[0],raw_answer=answer.decoded_text,raw_token_ids=tokens,EOS=tokens[-1]==v.runtime.adapter.tokenizer.eos_token_id,at_cap=len(tokens)>=1024))
    v.write(v.ROOT/'SOURCE_REVIEW_READY.json',dict(inputs=40,source_only=True,time=time.time()))
    while not (v.ROOT/'SOURCE_REVIEW_FROZEN.json').exists():v.budget();time.sleep(3)
    labels=v.read(v.ROOT/'SOURCE_REVIEW_FROZEN.json')['pass'];assert set(labels)=={f'{o}:{j}' for o in range(1,9) for j in range(5)}
    for task in tasks:
        owner=task['order'];passed=all(labels[f'{owner}:{j}'] for j in range(5));arm='TT_MARGIN' if passed else 'CE'
        state=v.r.p.load_state(v.ROOT/'candidates'/arm/f'{owner}.pt')['state'];v.save_state(v.ROOT/'candidates/TT_SOURCE'/f'{owner}.pt',dict(state=state,source_gate_pass=passed,fallback=None if passed else 'CE'))
    v.write(v.ROOT/'ALL_CANDIDATES_FROZEN.json',dict(count=32,source_checks=40,time=time.time()))
    # Reuse the zero-compatible exact evaluation path; full control tokens checked afterward.
    import canon_evaluate_recovery as ev
    original_mechanical=v.r.s.fit.short.mechanical
    def checked(runtime,hook,batch):
        if ev.zero(hook.expert):return ev.mechanical(hook,batch,True)
        return original_mechanical(runtime,hook,batch)
    v.r.s.fit.short.mechanical=checked
    def loader(arm,owner):return v.r.p.expert(v.r.p.load_state(v.ROOT/'candidates'/arm/f'{owner}.pt')['state'],owner,v.runtime.device)
    c.evaluate(tasks,ARMS,loader)
    expected=v.read(v.ROOT/'CE_REFERENCE_TOKENS.json')
    for rel,tokens in expected.items():assert v.read(v.ROOT/rel)['raw_token_ids']==tokens,'Matched CE control must exactly reproduce frozen V92 tokens'
    v.write(v.ROOT/'CE_CONTROL_EXACT.json',dict(status='PASS',consumers=155))
    c.finish(ARMS)


if __name__=='__main__':c.run(main)
