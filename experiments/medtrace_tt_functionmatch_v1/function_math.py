"""Match a fixed TT map-change norm along a candidate parameter direction."""
import math
import torch
from mechanism_math import map_delta, lowrank_inner
from scoped_math import SHAPES

RTOL = 1e-3
ATOL = 1e-10
MAX_SCALE = 16.
MAX_BISECTIONS = 32


def map_norm(before, after):
    left, right = map_delta(before, after)
    squared = lowrank_inner(left, right, left, right)
    assert math.isfinite(squared) and squared >= -1e-10
    return max(0., squared) ** .5


def match(before, candidate, target):
    assert math.isfinite(target) and target >= 0
    delta = {k: candidate[k] - before[k] for k in before}
    def state(scale):
        return {k: v.clone() for k,v in candidate.items()} if scale == 1 else {k:before[k]+scale*delta[k] for k in before}
    if target == 0:
        return {k:v.clone() for k,v in before.items()}, dict(scale=0.,target=0.,actual=0.,relative_error=0.,evaluations=0)
    tolerance = ATOL + RTOL * target
    lo, hi, evaluations = 0., 1., 0
    while True:
        value = state(hi); norm = map_norm(before,value); evaluations += 1
        if abs(norm-target) <= tolerance:break
        if norm >= target:
            for _ in range(MAX_BISECTIONS):
                alpha = (lo+hi)/2; value = state(alpha); norm = map_norm(before,value); evaluations += 1
                if abs(norm-target) <= tolerance:
                    hi = alpha
                    break
                if norm < target:lo = alpha
                else:hi = alpha
            else:raise AssertionError('Function match did not meet frozen precision')
            break
        assert hi < MAX_SCALE, 'No registered positive scale bracket; retain state and stop'
        lo, hi = hi, min(MAX_SCALE,2*hi)
    assert abs(norm-target) <= tolerance and all(torch.isfinite(v).all() for v in value.values())
    return value, dict(scale=hi,target=target,actual=norm,relative_error=abs(norm-target)/target,evaluations=evaluations)


def selfcheck():
    g=torch.Generator().manual_seed(20261009)
    before={k:torch.randn(shape,generator=g)*.02 for k,shape in SHAPES.items()}
    candidate={k:v+torch.randn(v.shape,generator=g)*.001 for k,v in before.items()}
    for scale in (.4,1.,2.5):
        expected={k:v+scale*(candidate[k]-v) for k,v in before.items()}
        target=map_norm(before,expected);value,row=match(before,candidate,target)
        assert abs(map_norm(before,value)-target)<=ATOL+RTOL*target
        assert 0<=row['scale']<=MAX_SCALE
    value,row=match(before,candidate,0.)
    assert all(torch.equal(value[k],v) for k,v in before.items())
    try:match(before,before,1.)
    except AssertionError:pass
    else:raise AssertionError('Unreachable target accepted')
    return dict(status='PASS',shrink_and_expand=True,zero_target=True,unreachable_target_stops=True,RTOL=RTOL,ATOL=ATOL)
