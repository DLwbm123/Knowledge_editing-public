"""Shared finite update construction for replay, projection and zero-constraint checks."""
import copy
import torch

def flatten(state,keys):return torch.cat([state[k].detach().cpu().double().reshape(-1) for k in keys])

def proposal(r,before,gradient,scalar,keys):
    local={k:v.detach().clone().requires_grad_(True) for k,v in before.items()};offset=0
    for k in keys:
        n=local[k].numel();local[k].grad=gradient[offset:offset+n].reshape(local[k].shape).to(local[k]);offset+=n
    assert offset==gradient.numel()
    torch.nn.utils.clip_grad_norm_(list(local.values()),1.)
    raw,updated=r.mm.canonical_rms_step(before,{k:local[k].grad.detach() for k in keys},copy.deepcopy(scalar))
    direction=torch.cat([(raw[k]-before[k]).detach().cpu().double().reshape(-1) for k in keys])
    return raw,updated,direction

def construct(r,before,raw,direction,delta,scale,keys):
    if scale==0 or not bool(delta.any()):return {k:v.clone() for k,v in before.items()}
    if torch.equal(delta,direction):candidate=raw
    else:
        candidate={};offset=0
        for k in keys:
            n=before[k].numel();candidate[k]=before[k]+delta[offset:offset+n].reshape(before[k].shape).to(before[k]);offset+=n
        assert offset==delta.numel()
    return r.mm.canonicalize({k:before[k]+scale*(candidate[k]-before[k]) for k in keys})[0]

def match(r,before,state,target):
    if target==0:return {k:v.clone() for k,v in before.items()}
    return r.mm.canonicalize(r.match_proposal(before,state,target)[0])[0]

def violation(initial,values):
    assert len(initial)==len(values) and len(initial)%2==0
    return max(max((initial[j+1]-initial[j])-(values[j+1]-values[j]) for j in range(0,len(initial),2)),max(values[j]-initial[j] for j in range(0,len(initial),2)))

def same(a,b):return a.keys()==b.keys() and all(torch.equal(a[k].cpu(),b[k].cpu()) for k in a)

def first_feasible(rows):
    for row in rows:
        if row['max_support_regression']<=1e-4 and row['source_gain']>=1e-6 and row['actual_operator_norm']>0:return row
    return None

def selfcheck():
    base=[1.,2.]*27;changed=base.copy();changed[0]+=.0002;assert violation(base,base)==0 and violation(base,changed)>1e-4
    a=dict(max_support_regression=.001,source_gain=.1,actual_operator_norm=.1);b=dict(max_support_regression=.00001,source_gain=.01,actual_operator_norm=.01)
    assert first_feasible([a,b]) is b and first_feasible([dict(b,actual_operator_norm=0)]) is None
    assert same({'x':torch.tensor([1.])},{'x':torch.tensor([1.])})
    return dict(status='PASS',absolute_Base_violation=True,first_fixed_feasible=True,zero_not_learning=True,no_model_calls=True)
