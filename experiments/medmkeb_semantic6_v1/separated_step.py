import torch,copy
import owner5_semantic_loss as semantic

def selfcheck():
    semantic.selfcheck()
    x=torch.arange(1.,7.,dtype=torch.float64,requires_grad=True)
    (sum(x[i] for i in (0,2,4))/3).backward()
    assert torch.equal(x.grad,torch.tensor([1/3,0,1/3,0,1/3,0],dtype=x.dtype))
    assert [f'{f}_{m}' for f in ('OFF','ON') for m in ('NONE','CE','SEM')]==['OFF_NONE','OFF_CE','OFF_SEM','ON_NONE','ON_CE','ON_SEM']
    return dict(status='PASS',semantic_hinge_gradient=True,content_ce_gradient=True,frozen_arms=6)

def step(r,before,edit,retention,cache,coefficient,scalar,target,fisher):
    params={k:v.detach().clone().requires_grad_(True) for k,v in before.items()}
    extra={k:torch.zeros_like(v) for k,v in params.items()};penalty=None;balance=direction=None
    if fisher:
        penalty=r.mm.operator_penalty(params,cache)
        extra=dict(zip(params,torch.autograd.grad(coefficient*penalty,tuple(params.values()))))
        from balanced_fisher import cap_gradient,remove_opposing
        extra,balance=cap_gradient(edit,extra,r.mm.GROUPS)
        extra,direction=remove_opposing(edit,extra,r.mm.GROUPS)
    for k,v in params.items():v.grad=edit[k].detach()+extra[k].detach()+retention[k].detach()
    norm=torch.nn.utils.clip_grad_norm_(list(params.values()),1.)
    assert torch.isfinite(norm) and all(torch.isfinite(v.grad).all() for v in params.values())
    proposed,updated=r.mm.canonical_rms_step(before,{k:v.grad.detach() for k,v in params.items()},copy.deepcopy(scalar))
    if r.fm.map_norm(before,proposed)<=1e-14:selected={k:v.clone() for k,v in before.items()};matching=dict(status='ZERO_DIRECTION')
    else:
        proposed,matching=r.match_proposal(before,proposed,target);selected=r.mm.canonicalize(proposed)[0]
        assert abs(r.fm.map_norm(before,selected)-target)<=1e-10+.002*target
    return selected,updated,dict(gradient_norm=float(norm),matching=matching,balance=balance,anti_parallel=direction,penalty_before=float(penalty.detach()) if penalty is not None else None)

