"""Convex minimum-response step with fixed gain and exact functional norm bound."""
import torch


def solve(rows, gain, required):
    rows,gain=rows.double(),gain.double()
    assert rows.ndim==2 and gain.shape==(rows.shape[1],)
    assert torch.isfinite(rows).all() and torch.isfinite(gain).all()
    assert abs(float(gain.norm())-1)<1e-10 and 0<required<1
    values,vectors=torch.linalg.eigh(rows.T@rows)
    assert float(values[0])>0, 'Nonpositive response metric; no implicit ridge or rank truncation'
    values=values/values[-1]
    coordinates=vectors.T@gain
    def candidate(mu):
        v=coordinates/(values+mu)
        return required*(vectors@v)/(coordinates@v)
    mu=0.;value=candidate(mu)
    if value.norm()>1:
        lo,hi=0.,1.
        for _ in range(64):
            if candidate(hi).norm()<=1:break
            hi*=2
        else:raise AssertionError('No norm multiplier bracket')
        for _ in range(80):
            mid=(lo+hi)/2
            if candidate(mid).norm()>1:lo=mid
            else:hi=mid
        mu=hi;value=candidate(mu)
    multiplier=required/(coordinates@(coordinates/(values+mu)))
    residual=vectors@(values*(vectors.T@value))+mu*value-multiplier*gain
    assert value.norm()<=1+1e-10 and abs(float(gain@value)-required)<1e-10
    assert residual.norm()<1e-9 and abs(mu*float(value.norm().square()-1))<1e-9
    return value,dict(norm=float(value.norm()),gain=float(gain@value),required=required,
        norm_multiplier=mu,KKT_residual=float(residual.norm()),
        metric_condition=float(values[-1]/values[0]),rank=rows.shape[1],
        objective=float((rows@value).square().sum()))


def metric_factor(state):
    g2,g3,g4=(state[k].detach().cpu().double() for k in ('G2','G3','G4'))
    a=torch.einsum('aib,bj->aij',g3,g4[:,:,0]).reshape(4,14336)
    metric=torch.einsum('ajk,bjl,kl->ab',g2,g2,a@a.T)
    return torch.linalg.cholesky(metric)


def selfcheck():
    rows=torch.diag(torch.tensor([1.,2.],dtype=torch.float64))
    value,audit=solve(rows,torch.tensor([1.,0.],dtype=torch.float64),.72)
    assert torch.allclose(value,torch.tensor([.72,0.],dtype=torch.float64),atol=1e-12)
    gain=torch.ones(2,dtype=torch.float64)/(2**.5)
    value,audit=solve(torch.diag(torch.tensor([.1,1.],dtype=torch.float64)),gain,.9)
    assert audit['norm_multiplier']>0 and abs(float(value.norm())-1)<1e-9
    assert audit['objective']<=float((torch.tensor([.1,1.])*gain*.9).square().sum())
    for rows,required in [(torch.eye(2),1.1),(torch.diag(torch.tensor([1.,0.])),.9)]:
        try:solve(rows,gain,required)
        except AssertionError:pass
        else:raise AssertionError('Invalid or singular problem accepted')
    rng=torch.Generator().manual_seed(20261009)
    state={k:torch.randn(shape,generator=rng,dtype=torch.float64)*.1 for k,shape in
        {'G2':(8,64,4),'G3':(4,112,8),'G4':(8,128,1)}.items()}
    delta=torch.randn((64,8),generator=rng,dtype=torch.float64)
    factor=metric_factor(state)
    a=torch.einsum('aib,bj->aij',state['G3'],state['G4'][:,:,0]).reshape(4,14336)
    b=torch.einsum('ia,ajb->ijb',delta,state['G2']).reshape(4096,4)
    exact=torch.trace((b.T@b)@(a@a.T))
    assert torch.allclose((delta@factor).square().sum(),exact,rtol=1e-12,atol=1e-12)
    gradient=torch.randn((64,8),generator=rng,dtype=torch.float64)
    white_gradient=gradient@torch.linalg.inv(factor).T
    assert torch.allclose((gradient*delta).sum(),(white_gradient*(delta@factor)).sum(),atol=1e-10)
    return dict(status='PASS',inactive_and_active_norm_bound=True,infeasible_and_singular_stop=True,
        TT_function_metric_and_gradient_whitening=True)


if __name__=='__main__':print(selfcheck())
