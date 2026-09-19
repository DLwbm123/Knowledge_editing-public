"""Executable, synthetic CPU tests. These are NOT the private GPU cases.

Usage: python cpu_validation.py
Requires PyTorch; writes CPU_VALIDATION_RESULTS.json next to this script.
The normalized-CCA formula mirrors the documented Stage24 computation.
"""
from __future__ import annotations
import json
from pathlib import Path
import torch
from scaled_regularizer import scaled_regularizer_gradients, accumulate_scaled_regularizer

torch.set_num_threads(2)


def kernel(x):
    x = x.double(); n, d = x.shape
    d2 = (x.square().sum(1)[:, None] + x.square().sum(1)[None, :] - 2*x@x.T).abs()
    k = torch.exp(-d2 / (2*d))
    c = torch.eye(n, device=x.device, dtype=x.dtype) - torch.ones(n,n,device=x.device,dtype=x.dtype)/n
    return k@c


def dependence(x,y):
    a,b = kernel(x),kernel(y)
    eye = torch.eye(len(x),device=x.device,dtype=a.dtype)*len(x)*1e-5
    ra = torch.linalg.solve((a+eye).T,a.T).T
    rb = torch.linalg.solve((b+eye).T,b.T).T
    return (ra*rb.T).sum()


def vector(gs):
    return torch.cat([g.detach().double().flatten() for g in gs])


def grad(v,params,scale=1.0):
    return vector(scaled_regularizer_gradients(v,params,scale=scale,retain_graph=True))


def check_scalar_underflow():
    p=torch.nn.Parameter(torch.ones(64))
    h=p.half();q=h.float().mean();a=4e-6*q;b=-3e-6*q
    gx,gy,g = grad(a,[p]),grad(b,[p]),grad(a+b,[p])
    recovered = grad(a+b,[p],65536.)
    reference = torch.full_like(recovered,1e-6/64)
    err=float((recovered-reference).norm()/reference.norm())
    assert gx.norm()>0 and gy.norm()>0 and g.norm()==0
    assert recovered.norm()>0 and err<1e-3
    return dict(x_norm=float(gx.norm()),y_norm=float(gy.norm()),combined_unscaled_norm=float(g.norm()),
                recovered_norm=float(recovered.norm()),analytic_norm=float(reference.norm()),relative_error=err)


def synthetic_graph(seed):
    gen=torch.Generator().manual_seed(seed)
    m,n,d,r=5,600,128,4
    a=torch.nn.Parameter(.05*torch.randn(r,d,generator=gen))
    b=torch.nn.Parameter(.01*torch.randn(d,r,generator=gen))
    u=torch.randn(m,3,d,generator=gen)
    template=torch.randn(1,n,d,generator=gen)*.1
    observations=torch.randn(m,1,d,generator=gen)*.0005
    base=(template+observations).half()
    delta=u@a.T@b.T
    out=base.clone();out[:,-3:,:]=out[:,-3:,:]+delta.half()
    z=out.float().mean(1)
    f=(out+(torch.randn(m,1,d,generator=gen)*.002).half()).half().float().mean(1)
    x=torch.randn(m,d,generator=gen)*.002
    return [a,b],delta,z,f,x


def check_hsic(seed):
    params,delta,z,f,x = synthetic_graph(seed)
    dx,dy=dependence(x,z),dependence(z,f)
    reg=.001*dx-.001*dy
    gx,gy,g0=grad(.001*dx,params),grad(-.001*dy,params),grad(reg,params)
    gs={s:grad(reg,params,float(s)) for s in (256,4096,65536)}
    # High-precision loss gradient at IDENTICAL, already rounded pooled values.
    # The toy tail is addition + mean; analytically its derivative is 1/600.
    zl=z.detach().double().requires_grad_(True);fl=f.detach().double().requires_grad_(True)
    ref_loss=.001*dependence(x.double(),zl)-.001*dependence(zl,fl)
    gz,gf=torch.autograd.grad(ref_loss,(zl,fl))
    gdelta=((gz+gf)/600.)[:,None,:].expand_as(delta).float()
    reference=vector(torch.autograd.grad(delta,params,grad_outputs=gdelta,retain_graph=True))
    relative_error=float((gs[65536]-reference).norm()/reference.norm())
    plateau_error=float((gs[4096]-gs[65536]).norm()/gs[65536].norm())
    assert gx.norm()>0 and gy.norm()>0 and g0.norm()==0
    assert relative_error<.01 and plateau_error<.01
    # .grad preservation and correct isolated accumulation.
    base_grads=[torch.full_like(p,1e-4) for p in params]
    for p,g in zip(params,base_grads):p.grad=g.clone()
    again=scaled_regularizer_gradients(reg,params,scale=65536.,retain_graph=True)
    assert all(torch.equal(p.grad,b) for p,b in zip(params,base_grads))
    stats=accumulate_scaled_regularizer(reg,params,scale=65536.,diagnostic=True)
    assert all(torch.equal(p.grad,b+g) for p,b,g in zip(params,base_grads,again))
    assert stats['accumulated_gradient_changed_elements']>0
    return dict(seed=seed,observations=5,valid_tokens=600,active_tokens=3,feature_width=128,rank=4,
                D_x=float(dx.detach()),D_y=float(dy.detach()),x_norm=float(gx.norm()),y_norm=float(gy.norm()),
                combined_unscaled_norm=float(g0.norm()),
                recovered_norms={str(s):float(g.norm()) for s,g in gs.items()},
                analytic_local_vjp_norm=float(reference.norm()),relative_error=relative_error,
                scale_plateau_relative_error=plateau_error,accumulation=stats)


def other_tests():
    p=torch.nn.Parameter(torch.tensor([.2,-.3],dtype=torch.float64));p.grad=torch.tensor([1.,2.],dtype=torch.float64)
    before=p.grad.clone();reg=p.square().sum()*1e-4
    accumulate_scaled_regularizer(reg,[p],diagnostic=True)
    assert torch.allclose(p.grad,before+2e-4*p.detach(),atol=1e-15,rtol=1e-14)
    before=p.grad.clone();accumulate_scaled_regularizer(0*p.sum(),[p])
    assert torch.equal(before,p.grad)
    p2=torch.nn.Parameter(torch.tensor([1.]));p2.grad=torch.tensor([.5]);before=p2.grad.clone()
    try:accumulate_scaled_regularizer(p2.sum()*float('inf'),[p2])
    except FloatingPointError:pass
    else:raise AssertionError('Nonfinite must fail')
    assert torch.equal(before,p2.grad)
    q=torch.nn.Parameter(torch.tensor([1.]))
    try:scaled_regularizer_gradients(q.sum(),[p2])
    except RuntimeError:pass
    else:raise AssertionError('Disconnected parameter must fail')
    try:scaled_regularizer_gradients(p2.half().sum(),[p2])
    except TypeError:pass
    else:raise AssertionError('Half loss scalar must fail')
    return dict(float64_equivalence=True,zero_lambda_no_effect=True,
                nonfinite_atomic_failure=True,disconnected_rejected=True,half_loss_rejected=True)


if __name__=='__main__':
    result=dict(scope='CPU_SYNTHETIC_ONLY_NOT_PRIVATE_GPU_VALIDATION',torch_version=torch.__version__,
                cuda_available=torch.cuda.is_available(),scalar=check_scalar_underflow(),
                hsic=[check_hsic(i) for i in (1,2,3)],checks=other_tests(),status='PASS')
    path=Path(__file__).with_name('CPU_VALIDATION_RESULTS.json')
    path.write_text(json.dumps(result,indent=2,ensure_ascii=False))
    print(json.dumps(result,indent=2,ensure_ascii=False))
