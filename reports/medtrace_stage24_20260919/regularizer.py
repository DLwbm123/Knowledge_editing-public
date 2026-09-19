"""Differentiable normalized CCA, LOKI bbb5819 formula; no selection-code reuse.
Gaussian K=exp(-abs(||xi||²+||xj||²-2xi.xj)/(2*sigma²*d)); right-center K H.
R=Kc(Kc+n*epsilon*I)^-1; D=trace(Rx Ry). No division by (n-1)^2.
Float64 solves and kernels; backward returns gradients to original FP32/FP16 path.
"""
import torch


def kernel(x, sigma=1.):
    if x.ndim!=2 or min(x.shape)<1 or not torch.isfinite(x).all() or sigma<=0:raise ValueError('finite observation x feature matrix required')
    x=x.double();n,d=x.shape
    distance=(x.square().sum(1)[:,None]+x.square().sum(1)[None,:]-2*x@x.T).abs()
    k=torch.exp(-distance/(2*sigma*sigma*d))
    h=torch.eye(n,device=x.device,dtype=x.dtype)-torch.ones((n,n),device=x.device,dtype=x.dtype)/n
    return k@h,k


def dependence(x,y):
    if len(x)!=len(y):raise ValueError('paired observations required')
    a,_=kernel(x);b,_=kernel(y);ridge=torch.eye(len(x),device=x.device,dtype=a.dtype)*(len(x)*1e-5)
    ra=torch.linalg.solve((a+ridge).T,a.T).T;rb=torch.linalg.solve((b+ridge).T,b.T).T
    value=(ra*rb.T).sum()
    if not torch.isfinite(value):raise FloatingPointError('non-finite normalized CCA')
    return value


def capture(runtime,hook,batches):
    """Five separate teacher-forced graphs; pool valid tokens without changing writer mask."""
    if len(batches)!=5:raise ValueError('native and exact four wrappers required')
    features={k:[] for k in ('input','patch','final')};masks=[]
    for batch in batches:
        hook.set_teacher_routing(batch.labels)
        mask=batch.attention_mask.bool() if batch.attention_mask is not None else torch.ones_like(batch.labels,dtype=torch.bool)
        one={};handles=[]
        def pool(z):
            if z.shape[:2]!=mask.shape:raise ValueError('expanded multimodal mask mismatch')
            return z[mask].float().mean(0)
        def first(_m,args):one['input']=pool(args[0]).detach()
        def before(_m,args,out):one['unpatched']=out.detach()
        def patch(_m,args,out):
            one['patch']=pool(out);one['delta_norm']=float((out.detach()-one.pop('unpatched')).float().norm())
        def final(_m,args,out):one['final']=pool(out[0] if isinstance(out,tuple) else out)
        try:
            handles.append(runtime.get_module('model.layers.0').register_forward_pre_hook(first))
            module=runtime.get_module('model.layers.30.mlp.down_proj')
            handles.append(module.register_forward_hook(before,prepend=True))
            handles.append(module.register_forward_hook(patch)) # Registered AFTER the active writer.
            handles.append(runtime.get_module('model.layers.31').register_forward_hook(final))
            output=runtime.model(**batch.forward_kwargs());del output
        finally:
            for h in handles:h.remove()
        if not one['patch'].requires_grad or not one['final'].requires_grad:raise RuntimeError('Detached regularizer path')
        for name in features:features[name].append(one[name])
        masks.append(dict(valid=int(mask.sum()),input_mask=mask.cpu().tolist(),patch_mask=mask.cpu().tolist(),final_mask=mask.cpu().tolist(),writer_mask=hook.token_mask.cpu().tolist(),active=int(hook.token_mask.sum()),delta_norm=one['delta_norm']))
    return {k:torch.stack(v) for k,v in features.items()},masks


def loss(features):
    dx=dependence(features['input'],features['patch']);dy=dependence(features['patch'],features['final'])
    return .001*dx-.001*dy,dx,dy


def kernel_diagnostics(features):
    result={}
    for name,x in features.items():
        c,k=kernel(x.detach());off=k[~torch.eye(len(k),device=k.device,dtype=torch.bool)];ridge=len(k)*1e-5
        result[name]=dict(observations=len(x),width=x.shape[1],offdiag_min=float(off.min()),offdiag_max=float(off.max()),offdiag_mean=float(off.mean()),condition=float(torch.linalg.cond(c+torch.eye(len(k),device=k.device)*ridge)),ridge=ridge,centered_norm=float(c.norm()))
    return result


def selfcheck():
    torch.manual_seed(24001);x=torch.randn(5,7,dtype=torch.float64,requires_grad=True);y=torch.randn(5,9,dtype=torch.float64,requires_grad=True)
    # Independent inverse reference mirrors author's formula; gradcheck tests our solve path.
    a,_=kernel(x);b,_=kernel(y);eye=torch.eye(5,dtype=x.dtype)*5e-5
    reference=((a@torch.inverse(a+eye))*(b@torch.inverse(b+eye)).T).sum()
    assert torch.allclose(dependence(x,y),reference,atol=1e-9,rtol=1e-9)
    assert torch.autograd.gradcheck(dependence,(x,y),eps=1e-6,atol=2e-5,rtol=2e-3)
    for z in (torch.ones(1,4),torch.ones(5,4),torch.ones(5,4)*1e-10,torch.eye(5)*1e5):
        z=z.double().requires_grad_();v=dependence(z,z);g=torch.autograd.grad(v,z)[0];assert torch.isfinite(v) and torch.isfinite(g).all()
    assert float(dependence(torch.ones(1,4),torch.ones(1,7)))==0
    duplicate=torch.cat([x[:1],x[:1],x[2:]]);assert torch.isfinite(dependence(duplicate,y))
    return dict(status='PASS',reference='LOKI bbb5819 normalized CCA inverse formula',float64_gradcheck=True,degenerate_and_saturated_finite=True,batch1='zero dependence',kernel_dtype='float64')

if __name__=='__main__':
    import json
    print(json.dumps(selfcheck()))
