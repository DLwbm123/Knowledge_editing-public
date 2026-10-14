"""Original canonical M1/M13 update, reusing the frozen math and matching code."""
import copy
import math
import torch


def coefficient(C_source_gain,C_penalty,prior=None):
    assert math.isfinite(C_source_gain) and C_source_gain>0
    assert math.isfinite(C_penalty) and C_penalty>0
    value=.1*C_source_gain/C_penalty
    if prior is not None:
        assert abs(value-prior)<=1e-12+1e-4*abs(prior),'Frozen M13 calibration drift'
        return prior
    return value


def step(r,before,source_gradients,cache,regularization,scalar,target,arm='M13'):
    """source_gradients must be the un-clipped cyclic .25/.25/.5 aggregate."""
    assert arm in ('M1','M13') and scalar['step']>=0 and len(scalar['second_moments'])==2
    assert math.isfinite(target) and target>=0 and math.isfinite(regularization) and regularization>=0
    assert arm!='M1' or regularization==0
    params={k:v.detach().clone().requires_grad_(True) for k,v in before.items()}
    penalty=None;extra={k:torch.zeros_like(v) for k,v in params.items()}
    if arm=='M13':
        assert regularization>0
        penalty=r.mm.operator_penalty(params,cache)
        assert torch.isfinite(penalty)
        extra=dict(zip(params,torch.autograd.grad(regularization*penalty,tuple(params.values()))))
    for k,v in params.items():v.grad=source_gradients[k].detach().to(v)+extra[k].detach()
    norm=torch.nn.utils.clip_grad_norm_(list(params.values()),1.)
    assert torch.isfinite(norm) and all(torch.isfinite(v.grad).all() for v in params.values())
    proposed,updated=r.mm.canonical_rms_step(before,{k:v.grad.detach() for k,v in params.items()},copy.deepcopy(scalar))
    assert updated['step']==scalar['step']+1
    if r.fm.map_norm(before,proposed)<=1e-14:
        selected={k:v.clone() for k,v in before.items()}
        matching=dict(status='ZERO_DIRECTION',target=target,actual=0.)
    else:
        proposed,matching=r.match_proposal(before,proposed,target)
        selected=r.mm.canonicalize(proposed)[0]
        assert abs(r.fm.map_norm(before,selected)-target)<=1e-10+.002*target
    return selected,updated,dict(gradient_norm=float(norm),matching=matching,projection=None,
        finite=dict(status='NOT_APPLICABLE',fraction=1.,checks=[]),
        penalty_before=float(penalty.detach()) if penalty is not None else None,
        operator_backwards=int(arm=='M13'),source_RMS_steps=1)
