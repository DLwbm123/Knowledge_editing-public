"""Full source training and routed free-answer evaluation of two TT adaptations."""
import copy
import os
import signal
import time
import traceback
import v81 as v

ARMS=('S0','U','CAP','REL','TNC')
STEPS=160


def pack(d):return v.torch.cat([d[k].reshape(-1) for k in v.r.mm.KEYS])


def unpack(x,state):
    result={};offset=0
    for k in v.r.mm.KEYS:
        n=state[k].numel();result[k]=x[offset:offset+n].reshape_as(state[k]);offset+=n
    assert offset==x.numel()
    return result


def cap_responses(state,cache):
    b,a=v.r.mm.factors(state)
    xa=cache['x']@a.T
    yb=(cache['y'].flatten(0,1)@b).reshape(cache['y'].shape[0],cache['y'].shape[1],b.shape[1])
    products=(xa[:,None]*yb).sum(-1)
    return v.torch.stack([x.sum(0) for x in products.split(cache['token_counts'])])


def curvature(expert,hook,batches,cache,task):
    """Fixed diagonal TT pullback; NOT the paper's matrix K-FAC implementation."""
    torch=v.torch;params=dict(expert.named_parameters())
    edit={k:torch.zeros_like(p) for k,p in params.items()}
    for view,b in enumerate(batches):
        hook.set_teacher_routing(b.labels)
        out=v.runtime.model(**b.forward_kwargs())
        mask=b.labels[:,1:]!=-100;logits=out.logits[:,:-1][mask].float()
        prob=logits.detach().softmax(-1)
        rng=torch.Generator(device=logits.device).manual_seed(task['seed']+860001+view)
        for j in range(8):
            signs=torch.randint(0,2,logits.shape,generator=rng,device=logits.device,dtype=torch.int8).float()*2-1
            z=prob.sqrt()*signs;z=z-prob*z.sum(-1,keepdim=True)
            weight=.5 if view==0 else .125
            value=(logits*z).sum()*(weight/(8*len(logits)))**.5
            g=torch.autograd.grad(value,tuple(params.values()),retain_graph=j<7)
            for (k,_),x in zip(params.items(),g):edit[k]+=x.detach().square()
            v.counts['backwards']+=1
        del out,logits,prob,z,g
    state={k:p.detach().clone().requires_grad_(True) for k,p in params.items()}
    values=cap_responses(state,cache)
    torch.testing.assert_close(values.square().sum(),v.r.mm.operator_penalty(state,cache))
    rng=torch.Generator(device=values.device).manual_seed(task['seed']+861001)
    cap={k:torch.zeros_like(p) for k,p in state.items()}
    for j in range(16):
        signs=torch.randint(0,2,values.shape,generator=rng,device=values.device,dtype=torch.int8).float()*2-1
        g=torch.autograd.grad((values*signs).sum()*(2/16)**.5,tuple(state.values()),retain_graph=j<15)
        for (k,_),x in zip(state.items(),g):cap[k]+=x.detach().square()
        v.counts['operator_backwards']+=1
    assert all(torch.isfinite(x).all() and (x>=0).all() for x in [*edit.values(),*cap.values()])
    mean=float(pack(edit).mean());ridge=max(1e-12,1e-4*mean)
    return edit,cap,ridge


def ce_gradient(expert,hook,batches,step):
    expert.zero_grad(set_to_none=True);losses=[]
    for view in (0,1+(step-1)%4):
        b=batches[view];hook.set_teacher_routing(b.labels)
        loss=v.runtime.compute_loss(b);assert v.torch.isfinite(loss)
        (.5*loss).backward();v.counts['backwards']+=1;losses.append(float(loss.detach()))
    return {k:p.grad.detach().clone() for k,p in expert.named_parameters()},losses


def filtered(g,edit,cap,ridge,coefficient,arm):
    raw={}
    for k in g:
        denominator=edit[k]+ridge if arm=='REL' else v.torch.full_like(edit[k],ridge)
        f=(1/(1+coefficient*cap[k]/denominator)).clamp(min=.05,max=1.)
        raw[k]=g[k]*f
    # Same group gradient norm: compare directional allocation, not scalar shrinkage.
    for group in v.r.mm.GROUPS:
        before=sum(float(g[k].double().square().sum()) for k in group)**.5
        after=sum(float(raw[k].double().square().sum()) for k in group)**.5
        scale=before/after if after else 1.
        for k in group:raw[k]=raw[k]*scale
    return raw


def losses_at(expert,hook,batches,state):
    expert.load_state_dict(state);values=[]
    with v.torch.inference_mode():
        for b in batches:
            hook.set_teacher_routing(b.labels);loss=v.runtime.compute_loss(b)
            assert v.torch.isfinite(loss);values.append(float(loss))
    return values


def terminal(expert,hook,batches,base,cache,target):
    torch=v.torch;r=v.r;state={k:x.clone() for k,x in base.items()}
    baseline=losses_at(expert,hook,batches,state);curve=[]
    for index in range(16):
        expert.load_state_dict(state);rows=[]
        for b in batches:
            expert.zero_grad(set_to_none=True);hook.set_teacher_routing(b.labels)
            loss=v.runtime.compute_loss(b);loss.backward();v.counts['backwards']+=1
            rows.append(pack({k:p.grad.detach().clone() for k,p in expert.named_parameters()}))
        j=torch.stack(rows).double();j=j/j.norm(dim=1,keepdim=True).clamp_min(1e-30)
        params={k:x.detach().clone().requires_grad_(True) for k,x in state.items()}
        penalty=r.mm.operator_penalty(params,cache)
        h=pack(dict(zip(params,torch.autograd.grad(penalty,tuple(params.values()))))).double()
        v.counts['operator_backwards']+=1
        solution=torch.linalg.lstsq((j@j.T).cpu(),(j@h).cpu(),rcond=1e-10).solution.to(j)
        direction=-h+j.T@solution
        if float(direction.norm())<=1e-20:
            curve.append(dict(iteration=index+1,status='ZERO_NULL_DIRECTION'));break
        residual=float((j@direction).norm())
        assert residual<=1e-6*float(direction.norm())+1e-12
        direction/=direction.norm()
        delta=unpack(direction,state)
        proposal={k:state[k]+delta[k].to(state[k]) for k in state}
        if r.fm.map_norm(state,proposal)<=1e-14:
            curve.append(dict(iteration=index+1,status='NO_REAL_OPERATOR_DIRECTION'));break
        candidate,matching=r.match_proposal(state,proposal,target)
        candidate=r.mm.canonicalize(candidate)[0]
        after=losses_at(expert,hook,batches,candidate)
        with torch.no_grad():new_penalty=float(r.mm.operator_penalty(candidate,cache))
        accepted=all(a<=b+1e-7+1e-4*b for a,b in zip(after,baseline)) and new_penalty<float(penalty)-max(1e-12,1e-7*float(penalty))
        v.counts['updates']+=1
        curve.append(dict(iteration=index+1,status='ACCEPT' if accepted else 'FIRST_REJECTION_STOP',
            source_NLL=after,penalty_before=float(penalty),penalty_after=new_penalty,
            null_residual=residual,matching=matching))
        if not accepted:break
        state=candidate
    return state,curve


def train(task,cache):
    from m3bench_repro.editors.llava_runtime import seed_everything
    torch=v.torch;r=v.r;owner=task['order']
    original=r.p.load_state(v.PARENT/'private/candidates/C'/f'{owner}.pt')['states']['BASE']
    sizes=[x['operator_step_norm'] for x in v.read(v.PARENT/'private/results/C'/f'{owner}.json')['curve']]
    coefficient=v.read(v.PARENT/'private/calibration146'/f'{owner}.json')['coefficient']
    items=v.prepare(task,v.read(v.ROOT/'SOURCE_ROWS.json')[str(owner)],True)
    batches=[r.old.batch_on(v.runtime,x[1]) for x in items]
    diagonal=None;warm_control=None
    for arm in ARMS[:-1]:
        seed_everything(task['seed']);expert=r.p.expert(original,task['seed'],v.runtime.device)
        expert.load_state_dict(r.mm.canonicalize(original)[0])
        hook=r.q.scoped.replay.MedTraceLayerHook(v.runtime.get_module(r.c.LAYER),expert);hook.attach()
        scalar=dict(step=0,second_moments=[0.,0.]);curve=[]
        try:
            with v.cache_class(v.runtime,batches):
                for step in range(1,161):
                    before=r.old.snapshot(expert);g,loss=ce_gradient(expert,hook,batches,step)
                    if arm in ('CAP','REL') and step>1:g=filtered(g,*diagonal,coefficient,arm)
                    chosen,scalar,audit=v.training_math.step(r,before,g,cache,
                        0. if arm=='U' else coefficient,scalar,sizes[step-1],'M1' if arm=='U' else 'M13P')
                    v.counts['operator_backwards']+=audit['operator_backwards'];v.counts['updates']+=1
                    expert.load_state_dict(chosen)
                    if step==1:
                        if arm=='S0':warm_control={k:x.clone() for k,x in chosen.items()}
                        else:assert all(torch.equal(chosen[k],warm_control[k]) for k in chosen),'Common warm step drift'
                    curve.append(dict(step=step,source_NLL=loss,operator_step_norm=r.fm.map_norm(before,chosen),**audit))
                    if arm=='S0' and step==1:
                        diagonal=curvature(expert,hook,batches,cache,task)
                        v.write(v.ROOT/'curvature'/f'{owner}.json',dict(status='PASS',warmup_step=1,ridge=diagonal[2],
                            edit_trace=float(pack(diagonal[0]).sum()),cap_trace=float(pack(diagonal[1]).sum()),
                            estimate='DIAGONAL_CATEGORICAL_GGN_TT_PULLBACK_AND_ORIGINAL_CAP_SKETCH',source_probes=40,cap_probes=16))
                        v.write(v.ROOT/'mechanical'/f'{owner}.json',r.s.fit.short.mechanical(v.runtime,hook,batches[0]))
                    if step%20==0:
                        v.save_state(v.ROOT/'resume'/arm/f'{owner}.pt',dict(state=r.old.snapshot(expert),scalar=scalar,step=step,curve=curve,counts=dict(v.counts),rng=r.c.rng()))
                        v.progress(owner=owner,arm=arm,step=step)
            final=r.old.snapshot(expert)
            if arm=='S0':
                old=v.read(v.PARENT/'private/results/M13P'/f'{owner}.json')['curve']
                for a,b in zip(old,curve):assert a['source_NLL']==b['source_NLL'] and a['operator_step_norm']==b['operator_step_norm'],'Original control drift'
            v.save_state(v.ROOT/'candidates'/arm/f'{owner}.pt',dict(state=final,scalar=scalar,owner=owner,arm=arm))
            loaded=r.p.load_state(v.ROOT/'candidates'/arm/f'{owner}.pt')['state']
            assert all(torch.equal(loaded[k].cpu(),final[k].cpu()) for k in final)
            v.write(v.ROOT/'training'/arm/f'{owner}.json',dict(status='PASS',steps=160,curve=curve,save_load_exact=True))
            if arm=='S0':
                corrected,trace=terminal(expert,hook,batches,final,cache,.1*sizes[-1])
                v.save_state(v.ROOT/'candidates/TNC'/f'{owner}.pt',dict(state=corrected,owner=owner,arm='TNC',source='S0'))
                loaded=r.p.load_state(v.ROOT/'candidates/TNC'/f'{owner}.pt')['state']
                assert all(torch.equal(loaded[k].cpu(),corrected[k].cpu()) for k in corrected)
                v.write(v.ROOT/'training/TNC'/f'{owner}.json',dict(status='PASS',curve=trace,full_source_anchor=True))
        finally:hook.detach()
        del expert
    del batches,items,diagonal


def evaluate(task):
    torch=v.torch;owner=task['order']
    plans=[x for x in v.read(v.ROOT/'EVAL_PLAN.json') if x['owner']==owner]
    routes=v.read(v.ROOT/'ROUTE_BINDINGS_FROZEN.json')[str(owner)]
    items=v.prepare(task,[p['query'] for p in plans])
    for arm in ARMS:
        state=v.r.p.load_state(v.ROOT/'candidates'/arm/f'{owner}.pt')['state']
        expert=v.r.p.expert(state,task['seed'],v.runtime.device)
        for index,(plan,item,binding) in enumerate(zip(plans,items,routes)):
            assert binding['query_id']==plan['query']['query_id'] and binding['question']==item[4]['question']
            assert binding['image_sha256']==item[4]['image_sha256']
            hook=None
            if binding['effective_expert'] is not None:
                assert binding['effective_expert']==task['edit_id']
                hook=v.r.q.scoped.replay.MedTraceLayerHook(v.runtime.get_module(v.r.c.LAYER),expert);hook.attach()
            try:
                v.generating=True;v.counts['evaluation_generations']+=1
                with torch.inference_mode():answer,trace=v.r.s.fit.short.generate(v.runtime,hook,item[2],item[3])
                v.generating=False;tokens=list(answer.raw_token_ids)
                out=dict(owner=owner,arm=arm,query=plan['query'],binding=item[4],raw_answer=answer.decoded_text,
                    raw_token_ids=tokens,EOS=tokens[-1]==v.runtime.adapter.tokenizer.eos_token_id,at_cap=len(tokens)>=1024,
                    route=binding['route'],effective_expert=binding['effective_expert'],original_route_preserved=True,generation_trace=trace)
                v.write(v.ROOT/'outputs'/arm/str(owner)/f'{index}.json',out)
                if arm=='S0':assert v.read(plan['historical_path'])['R0']['raw_token_ids']==tokens,'Control generation drift'
                v.progress(owner=owner,arm=arm,evaluation_index=index)
            finally:
                if hook:hook.detach()
        del expert


def main():
    with (v.ROOT/f'ONCE_{v.PHASE}_{v.PART}').open('x'):pass
    v.allocations={k:x//4 for k,x in v.limits['counts'].items()}
    if v.PHASE=='evaluate':
        planned=5*sum(x['owner'] in {t['order'] for t in v.read(v.ROOT/'TASKS.json')[v.PART::4]} for x in v.read(v.ROOT/'EVAL_PLAN.json'))
        v.allocations['evaluation_generations']=planned+5
        v.allocations['forwards']=(planned+5)*1024+4000
    signal.signal(signal.SIGALRM,lambda *_: (_ for _ in ()).throw(TimeoutError('Original route experiment deadline')))
    signal.alarm(max(1,int(v.limits['hard_end']-time.time()-300)));v.setup()
    if v.PHASE=='train':
        cache={k:x.to(v.runtime.device) if isinstance(x,v.torch.Tensor) else x for k,x in v.r.p.load_state(v.PARENT/'private/OPERATOR_FISHER.pt').items()}
    else:assert v.read(v.ROOT/'ALL_CANDIDATES_FROZEN.json')['count']==40
    for task in v.read(v.ROOT/'TASKS.json')[v.PART::4]:
        if v.PHASE=='train':train(task,cache)
        else:evaluate(task)
    v.write(v.ROOT/f'COMPLETE_{v.PHASE}_{v.PART}.json',dict(status='PASS',counts=v.counts,time=time.time()))


if __name__=='__main__':
    code=0
    try:main()
    except BaseException as exc:
        code=1;v.write(v.ROOT/f'FAILURE_{v.PHASE}_{v.PART}.json',dict(error=repr(exc),traceback=traceback.format_exc(),counts=v.counts,time=time.time()))
        if not (v.ROOT/'STOP').exists():v.write(v.ROOT/'STOP',dict(reason='FULL_ROUTE_FAILURE',time=time.time()))
        raise
    finally:
        v.progress(exit_code=code);v.write(v.ROOT/f'EXIT_{v.PHASE}_{v.PART}.json',dict(exit_code=code,counts=v.counts,time=time.time()))
