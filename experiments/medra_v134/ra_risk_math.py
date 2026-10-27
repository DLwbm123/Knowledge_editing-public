"""Fixed expert-level finite-operator scaling; no evaluation labels are inputs."""
import math
import torch

GRID=(1.10,1.,.90,.75,.50)
EPSILON=1e-20
SOURCE_ATOL=1e-7
RISK_RTOL=3e-5


def scaled(state,alpha):
    return {k:v*alpha if k=='G1' else v.clone() for k,v in state.items()}


def norm2(state,factors):
    b,a=factors({k:v.detach().cpu().double() for k,v in state.items()})
    value=float(((b.T@b)*(a@a.T).T).sum())
    assert math.isfinite(value) and value>0
    return value


def projections(state,cache,factors):
    b,a=factors(state);rows=[]
    for part in cache['parts']:
        x,y=part['x'],part['y']
        xa=x@a.T;yb=(y.flatten(0,1)@b).reshape(y.shape[0],y.shape[1],b.shape[1])
        products=(xa[:,None]*yb).sum(-1)
        rows.extend(v.sum(0) for v in products.split(part['token_counts']))
    return rows


def total_trace(cache,curvature):
    result=0.
    for part in cache['parts']:
        for x,y in zip(part['x'].split(part['token_counts']),part['y'].split(part['token_counts'])):
            result+=curvature.energy(dict(x=x,y=y))
    assert math.isfinite(result) and result>0
    return result


def source_ok(base,raw,candidate,expanded=False):
    assert len(base)==len(raw)==len(candidate)==5
    assert all(math.isfinite(x) for x in base+raw+candidate)
    floor=all(c<=b-.9*max(b-r,0)+SOURCE_ATOL for b,r,c in zip(base,raw,candidate))
    return floor and (not expanded or all(c<=r+SOURCE_ATOL for c,r in zip(candidate,raw)))


def select(q,q0,base,raw,evaluate):
    assert math.isfinite(q) and math.isfinite(q0) and q>=0 and q0>=0
    rows=[]
    for alpha in GRID:
        feasible=alpha*alpha*q<=q0*(1+RISK_RTOL)+EPSILON and (alpha!=1.1 or q<q0)
        if not feasible:
            rows.append(dict(alpha=alpha,risk_feasible=False,source_evaluated=False));continue
        losses=raw if alpha==1. else evaluate(alpha)
        ok=source_ok(base,raw,losses,expanded=alpha>1.)
        rows.append(dict(alpha=alpha,risk_feasible=True,source_evaluated=True,source_feasible=ok,losses=losses))
        if ok:return alpha,rows,False
    losses=evaluate(.5)
    return .5,rows,True


def selfcheck(factors):
    g=torch.Generator().manual_seed(134)
    state={k:torch.randn(shape,generator=g,dtype=torch.float64) for k,shape in zip(
        ('G1','G2','G3','G4'),((1,2,2),(2,3,2),(2,2,2),(2,4,1)))}
    raw=norm2(state,factors)
    for a in GRID:
        assert abs(norm2(scaled(state,a),factors)/(raw*a*a)-1)<1e-12
    base=[3.]*5;end=[.1]*5
    assert source_ok(base,end,[.3]*5) and not source_ok(base,end,[.5]*5)
    assert not source_ok(base,end,[.2]*5,expanded=True)
    alpha,_,violation=select(1.,1.,base,end,lambda a:[.09]*5)
    assert alpha==1. and not violation
    alpha,_,violation=select(4.,1.,base,end,lambda a:[.2]*5)
    assert alpha==.5 and not violation
    alpha,_,violation=select(4.,1.,base,end,lambda a:[2.]*5)
    assert alpha==.5 and violation
    return dict(status='PASS',finite_operator_scaling=True,source_floor=True,
                expanded_no_regression=True,backtracking=True,violation_fallback=True)
