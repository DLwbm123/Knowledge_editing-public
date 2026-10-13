"""Matched FP32 proposal length; Adam changes direction, never native architecture."""
def update(g, master, target, optimizer, state=None):
    import torch
    assert g.dtype == master.dtype == torch.float32 and g.shape == master.shape
    assert torch.isfinite(g).all() and torch.isfinite(master).all()
    assert 0 < target <= .050001 and optimizer in ('sgd', 'adam_direction')
    if optimizer == 'sgd':
        direction = g
        next_state = None
    else:
        state = state or dict(step=0, m=torch.zeros_like(g), v=torch.zeros_like(g))
        t = state['step'] + 1
        m = .9 * state['m'] + .1 * g
        v = .999 * state['v'] + .001 * g.square()
        direction = (m / (1 - .9**t)) / ((v / (1 - .999**t)).sqrt() + 1e-8)
        next_state = dict(step=t,m=m,v=v)
    length = direction.norm()
    assert torch.isfinite(length) and length > 0, 'Zero/nonfinite direction cannot spend frozen budget; stop, no substitute direction'
    delta = -direction * (target / length)
    after = master + delta
    assert torch.isfinite(after).all()
    proposed = float(delta.norm())
    assert abs(proposed-target) <= max(1e-8,1e-5*target)
    return after, next_state, dict(gradient_norm=float(g.norm()),target_proposal_norm=target,
        proposal_norm=proposed,master_materialized_norm=float((after-master).norm()))


def selfcheck():
    import torch
    import io
    g=torch.tensor([[3.,4.],[1.,-2.]],dtype=torch.float32)
    w=torch.zeros_like(g)
    a,_,ev=update(g,w,.03,'sgd')
    assert torch.allclose(a,-.03*g/g.norm())
    b,state,_=update(g,w,.03,'adam_direction')
    assert abs(float(b.norm())-.03)<1e-7 and not torch.allclose(a,b)
    data=io.BytesIO();torch.save(state,data);data.seek(0)
    restored=torch.load(data,weights_only=True)
    c,s,_=update(g*.5,b,.002,'adam_direction',state)
    d,t,_=update(g*.5,b,.002,'adam_direction',restored)
    assert torch.equal(c,d) and torch.equal(s['m'],t['m']) and torch.equal(s['v'],t['v']) and s['step']==t['step']==2
    try:update(torch.zeros_like(g),w,.03,'sgd')
    except AssertionError:pass
    else:raise AssertionError('Zero direction silently passed')
    return dict(status='PASS',sgd_direction=True,matched_adam_length=True,exact_adam_state_reload=True,zero_direction_hard_stop=True)
