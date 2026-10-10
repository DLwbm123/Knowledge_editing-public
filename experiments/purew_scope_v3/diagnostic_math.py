"""Five editing-side interventions; deployed models contain only original W."""
import torch
from purew_math import row_basis
import propagation_math as sm

ARMS = ('A_LOW', 'B_NORM', 'C_FULL_FIXED', 'D_FULL_REFRESH', 'E_RLS')


def full(gradient, q):
    return gradient - (gradient @ q) @ q.T


def direction(gradient, arm, bases, inverses, q):
    if arm in ('C_FULL_FIXED', 'D_FULL_REFRESH'):
        return full(gradient, q)
    ps = inverses if arm == 'E_RLS' else tuple(torch.eye(a.shape[0],device=a.device,dtype=a.dtype) for a in bases)
    return sm.coordinates(gradient, bases, ps, 1.)


def step(directions, arm, full_gradient=None):
    if isinstance(directions, torch.Tensor):
        delta, receipt = sm.clipped_step((directions,))
        return delta[0], receipt
    if arm != 'B_NORM':return sm.clipped_step(directions)
    norm = torch.sqrt(sum(x.square().sum() for x in directions))
    target = .01 * full_gradient.norm().clamp_max(5.)
    scale = torch.where(norm > 0, -target / norm.clamp_min(torch.finfo(norm.dtype).tiny), torch.zeros_like(norm))
    delta = tuple(scale * x for x in directions)
    assert all(torch.isfinite(x).all() for x in delta)
    return delta, dict(coordinate_gradient_norm=float(norm),proposed_step_norm=float(target) if norm>0 else 0.,full_null_gradient_norm=float(full_gradient.norm()))


def selfcheck():
    torch.manual_seed(19);dtype=torch.float64
    w=torch.randn(3,9,dtype=dtype);q=row_basis(w);bases,_=sm.make_basis(w,3,1,19)
    g=torch.randn_like(w);p=tuple(torch.eye(a.shape[0],dtype=dtype) for a in bases)
    low=direction(g,'A_LOW',bases,p,q);dense=sum(x@a for x,a in zip(low,bases))
    assert torch.allclose(dense,g@torch.cat(bases).T@torch.cat(bases),atol=1e-12)
    projected=full(g,q);assert (projected@w.T).norm()<1e-12
    a,_=step(low,'A_LOW');b,receipt=step(low,'B_NORM',projected)
    assert torch.allclose(torch.sqrt(sum(x.square().sum() for x in b)),.01*projected.norm().clamp_max(5.),atol=1e-12)
    assert torch.allclose(b[0]/b[0].norm(),a[0]/a[0].norm(),atol=1e-12)
    assert all(torch.equal(x,y) for x,y in zip(direction(g,'E_RLS',bases,p,q),low))
    z,_=step(tuple(torch.zeros_like(x) for x in low),'B_NORM',projected);assert all(x.count_nonzero()==0 for x in z)
    for arm in ARMS:
        d=direction(g,arm,bases,p,q);delta,_=step(d,arm,projected)
        dw=delta if isinstance(delta,torch.Tensor) else sum(x@a for x,a in zip(delta,bases))
        assert dw.norm()<=.05+1e-12 and (g*dw).sum()<0
    key=torch.randn(3,dtype=dtype);pp=sm.recurse(p[0],key)
    assert (pp@key).norm()<key.norm()
    return dict(status='PASS',full_projection=True,coordinate_dense_parity=True,norm_match=True,zero_gradient_skips=True,initial_RLS_identity=True,trust_cap=True,real_descent=True)
