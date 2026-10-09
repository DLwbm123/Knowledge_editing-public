"""Same-state moment/coordinate decomposition, with stable low-rank map norms."""
import torch
from scoped_math import factors

MODES=('RAW','MOMENT','PRECOND','ADAM')


def candidate_states(expert,opt,before,adam):
    names={id(x):k for k,x in expert.named_parameters()};states={m:{} for m in MODES};audit=[]
    for group in opt.param_groups:
        ps=group['params'];keys=[names[id(x)] for x in ps]
        actual=torch.cat([(adam[k]-before[k]).double().flatten() for k in keys]);dn=float(actual.norm())
        directions={m:[] for m in MODES[:-1]}
        for x in ps:
            st=opt.state[x];step=int(st['step']);b1,b2=group['betas']
            mh=st['exp_avg']/(1-b1**step);vh=st['exp_avg_sq']/(1-b2**step)
            directions['RAW'].append(-x.grad.detach());directions['MOMENT'].append(-mh);directions['PRECOND'].append(-x.grad.detach()/(vh.sqrt()+group['eps']))
        for mode in MODES:
            if mode=='ADAM':
                for k in keys:states[mode][k]=adam[k].clone()
            else:
                norm=float(torch.cat([x.double().flatten() for x in directions[mode]]).norm())
                for k,vector in zip(keys,directions[mode]):states[mode][k]=before[k]+vector*(dn/norm) if norm else adam[k].clone()
            delta=torch.cat([(states[mode][k]-before[k]).double().flatten() for k in keys])
            raw=torch.cat([(states['RAW'][k]-before[k]).double().flatten() for k in keys])
            bound=4*torch.finfo(ps[0].dtype).eps*(float(torch.cat([before[k].double().flatten() for k in keys]).norm())+dn)
            assert torch.isfinite(delta).all() and abs(float(delta.norm())-dn)<=bound+1e-14
            audit.append(dict(mode=mode,cores=keys,candidate_norm=dn,actual_norm=float(delta.norm()),roundoff_bound=bound,
                cosine_to_RAW=float(torch.dot(delta,raw)/(delta.norm()*raw.norm())) if delta.norm() and raw.norm() else None,
                cosine_to_Adam=float(torch.dot(delta,actual)/(delta.norm()*actual.norm())) if delta.norm() and actual.norm() else None,
                zero_direction_Adam_fallback=mode!='ADAM' and norm==0))
    return states,audit


def lowrank_inner(left,right,other_left,other_right):
    return float(((left.T@other_left)*(right@other_right.T)).sum())


def map_delta(before,after):
    l0,r0=factors({k:x.double() for k,x in before.items()});l1,r1=factors({k:x.double() for k,x in after.items()})
    return torch.cat([l1-l0,l1],1),torch.cat([r0,r1-r0],0)


def selfcheck():
    g=torch.Generator().manual_seed(1)
    l=torch.randn(7,3,generator=g,dtype=torch.float64);r=torch.randn(3,11,generator=g,dtype=torch.float64)
    l2=torch.randn(7,4,generator=g,dtype=torch.float64);r2=torch.randn(4,11,generator=g,dtype=torch.float64)
    assert abs(lowrank_inner(l,r,l2,r2)-float(((l@r)*(l2@r2)).sum()))<1e-10
    e=torch.nn.Linear(3,2);opt=torch.optim.Adam(e.parameters(),lr=.001)
    for step in range(1,4):
        for x in e.parameters():x.grad=torch.arange(1,x.numel()+1,dtype=x.dtype).reshape_as(x)*(step if step!=2 else -1)
        before={k:x.detach().clone() for k,x in e.named_parameters()};opt.step();adam={k:x.detach().clone() for k,x in e.named_parameters()}
        states,audit=candidate_states(e,opt,before,adam)
        assert all(torch.equal(states['ADAM'][k],adam[k]) for k in adam)
        if step==1:
            assert all(torch.allclose(states['RAW'][k],states['MOMENT'][k],atol=1e-7) for k in adam)
            assert all(torch.allclose(states['PRECOND'][k],states['ADAM'][k],atol=1e-7) for k in adam)
    for x in e.parameters():x.grad=torch.zeros_like(x)
    before={k:x.detach().clone() for k,x in e.named_parameters()};opt.step();adam={k:x.detach().clone() for k,x in e.named_parameters()}
    states,audit=candidate_states(e,opt,before,adam)
    assert all(torch.equal(states['RAW'][k],adam[k]) and torch.equal(states['PRECOND'][k],adam[k]) for k in adam)
    return dict(status='PASS',lowrank_norm_and_cross_inner=True,group_norm_bound=True,first_step_moment_degeneracy=True,zero_direction_fallback=True)
