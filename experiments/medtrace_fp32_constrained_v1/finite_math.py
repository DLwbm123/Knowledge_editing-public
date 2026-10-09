"""Bounded feasibility restoration on the protected-to-RAW segment."""
STEPS = 16


def restore(evaluate, initial, raw, fraction=.9):
    assert raw > 1e-8
    threshold = fraction * raw
    trace = [dict(alpha=0., gain=initial), dict(alpha=1., gain=raw)]
    if initial >= threshold:
        return 0., trace
    lo, hi = 0., 1.
    for _ in range(STEPS):
        mid = (lo + hi) / 2
        gain = evaluate(mid)
        assert __import__('math').isfinite(gain)
        trace.append(dict(alpha=mid, gain=gain))
        if gain >= threshold:
            hi = mid
        else:
            lo = mid
    # Keep an evaluated feasible endpoint; no monotonicity or global-optimality claim.
    assert next(x['gain'] for x in reversed(trace) if x['alpha'] == hi) >= threshold
    return hi, trace


def selfcheck():
    a,t = restore(lambda x: .8 + .2*x, .8, 1.)
    assert .8 + .2*a >= .9 and a-.5 <= 2**-STEPS and len(t)==STEPS+2
    a,t = restore(lambda _: (_ for _ in ()).throw(AssertionError('unneeded evaluation')), .95, 1.)
    assert a==0. and len(t)==2
    f=lambda x: .8+.2*x+.04*__import__('math').sin(20*x)
    a,t=restore(f,f(0.),f(1.));assert f(a)>=.9*f(1.)
    try:restore(lambda x:x,0.,0.)
    except AssertionError:pass
    else:raise AssertionError('Zero RAW progress accepted')
    return dict(status='PASS',fixed_bisections=STEPS,feasible_endpoint=True,nonmonotone_example=True)
