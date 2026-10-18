"""Fixed construction-only acceptance; never reads check data."""
import math

FRACTIONS = tuple(2. ** -i for i in range(16))
TOLERANCE = 1e-6

def select(before, proposed, canonicalize, measure, base_nll):
    assert math.isfinite(base_nll)
    trials=[]
    for fraction in FRACTIONS:
        value=proposed if fraction==1 else canonicalize({k:before[k]+fraction*(proposed[k]-before[k]) for k in before})
        nll=float(measure(value))
        assert math.isfinite(nll)
        accepted=nll<=base_nll+TOLERANCE
        trials.append(dict(fraction=fraction,nll=nll,accepted=accepted))
        if accepted:return value,dict(fraction=fraction,trials=trials,skipped=False)
    nll=float(measure(before))
    assert math.isfinite(nll) and nll<=base_nll+TOLERANCE
    return before,dict(fraction=0.,trials=trials,skipped=True)

def selfcheck():
    # A binding constraint shrinks, an infeasible direction skips; no check data.
    v,a=select({'x':0.},{'x':1.},lambda x:x,lambda x:x['x'],.3)
    assert v['x']==.25 and len(a['trials'])==3
    v,a=select({'x':0.},{'x':1.},lambda x:x,lambda x:float(x['x']>0),0.)
    assert a['skipped'] and len(a['trials'])==16 and v['x']==0.
    v,a=select({'x':0.},{'x':1.},lambda x:x,lambda x:0.,0.)
    assert v['x']==1 and len(a['trials'])==1
    try:select({'x':0.},{'x':1.},lambda x:x,lambda x:float('nan'),0.)
    except AssertionError:pass
    else:raise AssertionError('nonfinite admitted')
    return dict(status='PASS',binding_shrink=True,infeasible_skip=True,nonbinding_identity=True,nonfinite_rejected=True)

if __name__=='__main__':print(selfcheck())
