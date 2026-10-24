"""Finite source-only protection/TT mechanism panel; no teacher or heldout access."""
import copy
import os
from pathlib import Path
import signal
import time
import traceback
import v81 as v


def flat(values):
    return v.torch.cat([values[k].detach().double().flatten() for k in v.r.mm.KEYS])


def source_gradients(expert, hook, batches):
    gradients=[];losses=[];keys=[]
    module=v.runtime.get_module(v.r.c.LAYER)
    for batch in batches:
        captured=[]
        def capture(_module,args):
            # At hidden position t-1, predict the answer label at t.
            labels=batch.labels
            mask=v.torch.cat([labels[:,1:]!=-100, v.torch.zeros_like(labels[:,:1],dtype=v.torch.bool)],1)
            x=args[0]
            assert x.shape[:2]==mask.shape
            captured.append(x.detach()[mask].float().mean(0).cpu())
        handle=module.register_forward_pre_hook(capture)
        expert.zero_grad(set_to_none=True)
        try:
            hook.set_teacher_routing(batch.labels)
            loss=v.runtime.compute_loss(batch)
            assert v.torch.isfinite(loss)
            loss.backward();v.counts['backwards']+=1
            gradients.append({k:p.grad.detach().clone() for k,p in expert.named_parameters()})
            losses.append(float(loss.detach()))
            assert len(captured)==1
            keys.append(captured[0])
        finally:handle.remove()
    weights=[.5,.125,.125,.125,.125]
    aggregate={k:sum(w*g[k] for w,g in zip(weights,gradients)) for k in gradients[0]}
    return aggregate,gradients,losses,keys


def evaluate(hook,batches):
    losses=[]
    with v.torch.inference_mode():
        for b in batches:
            hook.set_teacher_routing(b.labels)
            loss=v.runtime.compute_loss(b)
            assert v.torch.isfinite(loss)
            losses.append(float(loss))
    return losses


def weighted(losses):return .5*losses[0]+.125*sum(losses[1:])


def main():
    torch=v.torch if hasattr(v,'torch') else None
    with (v.ROOT/f'ONCE_{v.PHASE}_{v.PART}').open('x'):pass
    signal.signal(signal.SIGALRM,lambda *_: (_ for _ in ()).throw(TimeoutError('Original panel deadline')))
    signal.alarm(max(1,int(v.limits['hard_end']-time.time()-300)))
    v.allocations={k:x//2 for k,x in v.limits['counts'].items()}
    v.setup();torch=v.torch;r=v.r
    cache={k:x.to(v.runtime.device) if isinstance(x,torch.Tensor) else x for k,x in r.p.load_state(v.PARENT/'private/OPERATOR_FISHER.pt').items()}
    for task in v.read(v.ROOT/'TASKS.json')[v.PART::2]:
        owner=task['order']
        rows=v.read(v.ROOT/'SOURCE_ROWS.json')[str(owner)]
        items=v.prepare(task,rows,True)
        batches=[r.old.batch_on(v.runtime,x[1]) for x in items]
        regularization=v.read(v.PARENT/'private/calibration146'/f'{owner}.json')['coefficient']
        for anchor in ('M1','M13'):
            saved=r.p.load_state(v.PARENT/'private/candidates'/anchor/f'{owner}.pt')
            before={k:x.to(v.runtime.device) for k,x in saved['states']['FINAL'].items()}
            scalar=copy.deepcopy(saved['scalar_optimizer_state'])
            assert scalar['step']==160 and sum(x.numel() for x in before.values())==7168
            expert=r.p.expert(before,task['seed'],v.runtime.device);expert.load_state_dict(before)
            hook=r.q.scoped.replay.MedTraceLayerHook(v.runtime.get_module(r.c.LAYER),expert);hook.attach()
            try:
                if owner==1 and anchor=='M1':
                    v.write(v.ROOT/'MECHANICAL.json',r.s.fit.short.mechanical(v.runtime,hook,batches[0]))
                    expert.load_state_dict(before)
                source,individual,losses,keys=source_gradients(expert,hook,batches)
                # U's natural finite RMS step defines the shared size. No searched radius.
                params={k:x.detach().clone().requires_grad_(True) for k,x in before.items()}
                for k,p in params.items():p.grad=source[k].detach().to(p).clone()
                torch.nn.utils.clip_grad_norm_(list(params.values()),1.)
                u,us=r.mm.canonical_rms_step(before,{k:p.grad for k,p in params.items()},copy.deepcopy(scalar))
                u=r.mm.canonicalize(u)[0];target=r.fm.map_norm(before,u)
                assert target>1e-14,'Degenerate source step: record failure, no retuning'
                p,ps,audit=v.training_math.step(r,before,source,cache,regularization,copy.deepcopy(scalar),target,'M13P')
                v.counts['operator_backwards']+=audit['operator_backwards']
                assert us['step']==ps['step']==161
                records={}
                for name,state in [('U',u),('P',p)]:
                    expert.load_state_dict(state)
                    after=evaluate(hook,batches)
                    with torch.no_grad():penalty=float(r.mm.operator_penalty(state,cache))
                    records[name]=dict(source_NLL=after,weighted_source_gain=weighted(losses)-weighted(after),
                        operator_norm=r.fm.map_norm(before,state),Fisher_penalty=penalty)
                    v.counts['updates']+=1
                with torch.no_grad():penalty_before=float(r.mm.operator_penalty(before,cache))
                nu,np=records['U']['operator_norm'],records['P']['operator_norm']
                separation=r.fm.map_norm(u,p)
                operator_cos=(nu*nu+np*np-separation*separation)/(2*nu*np) if nu*np else None
                matrix=torch.stack([flat(g).cpu() for g in individual])
                singular=torch.linalg.svdvals(matrix)
                k_matrix=torch.stack(keys).double()
                k_norm=k_matrix/k_matrix.norm(dim=1,keepdim=True).clamp_min(1e-30)
                # Derived source material only; not a checkpoint or performance endpoint.
                (v.ROOT/'source_geometry').mkdir(exist_ok=True)
                v.save_state(v.ROOT/'source_geometry'/f'{owner}_{anchor}.pt',dict(
                    gradients=matrix.float(),keys=k_matrix.float(),source_only=True,
                    semantic_groups=1,views_are_paraphrases_not_distinct_relations=True))
                v.write(v.ROOT/'cells'/f'{owner}_{anchor}.json',dict(owner=owner,anchor=anchor,
                    status='PASS',source_NLL_before=losses,weighted_source_NLL_before=weighted(losses),
                    Fisher_penalty_before=penalty_before,regularization=regularization,
                    operator_cosine_P_U=operator_cos,operator_separation=separation,updates=records,
                    gradient_singular_values=singular.tolist(),key_cosines=k_norm.matmul(k_norm.T).tolist(),
                    protection_audit=audit,optimizer_history_from_anchor=True,
                    counterfactual_only=True,not_RhoEdit_or_DPC_reproduction=True))
                v.progress(owner=owner,anchor=anchor,completed_cells=len(list((v.ROOT/'cells').glob('*.json'))))
            finally:hook.detach()
            del expert,individual,matrix,keys,params
        del batches,items
    assert v.counts['updates']==16 and v.counts['backwards']==40
    v.write(v.ROOT/f'COMPLETE_{v.PHASE}_{v.PART}.json',dict(status='PASS',counts=v.counts,time=time.time()))


if __name__=='__main__':
    code=0
    try:main()
    except BaseException as exc:
        code=1
        v.write(v.ROOT/f'FAILURE_{v.PHASE}_{v.PART}.json',dict(error=repr(exc),traceback=traceback.format_exc(),time=time.time()))
        if not (v.ROOT/'STOP').exists():v.write(v.ROOT/'STOP',dict(reason='MECHANISM_PANEL_FAILURE',time=time.time()))
        raise
    finally:
        v.progress(exit_code=code)
        v.write(v.ROOT/f'EXIT_{v.PHASE}_{v.PART}.json',dict(exit_code=code,counts=v.counts,time=time.time()))
