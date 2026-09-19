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


ACTIVE_CACHE=None

def capture(runtime,hook,batches,diagnostic=False,path=None,boundaries=None):
    """Same five graphs and full-token means; omit LM head and diagnostic syncs."""
    if len(batches)!=5:raise ValueError('native plus four wrappers required')
    path=path or ('cached' if ACTIVE_CACHE is not None else 'hidden')
    features={k:[] for k in ('input','patch','final')};masks=[]
    for index,batch in enumerate(batches):
        hook.set_teacher_routing(batch.labels)
        mask=batch.attention_mask.bool() if batch.attention_mask is not None else torch.ones_like(batch.labels,dtype=torch.bool)
        one={};handles=[]
        def pool(z):
            if z.shape[:2]!=mask.shape:raise ValueError('expanded mask mismatch')
            return z[mask].float().mean(0)
        def first(_m,args):one['input']=pool(args[0]).detach()
        def before(_m,args,out):one['unpatched']=out.detach()
        def patch(_m,args,out):
            one['patch']=pool(out)
            if diagnostic:one['delta_norm']=float((out.detach()-one.pop('unpatched')).double().norm())
            if boundaries is not None:
                one['patch_tokens']=out;one['writer_inputs']=args[0].detach()
        def final(_m,args,out):
            z=out[0] if isinstance(out,tuple) else out;one['final']=pool(z)
            if boundaries is not None:one['final_tokens']=z
        try:
            handles.append(runtime.get_module('model.layers.0').register_forward_pre_hook(first))
            module=runtime.get_module('model.layers.30.mlp.down_proj')
            if diagnostic:handles.append(module.register_forward_hook(before,prepend=True))
            handles.append(module.register_forward_hook(patch))
            handles.append(runtime.get_module('model.layers.31').register_forward_hook(final))
            kwargs=batch.forward_kwargs()
            if path=='full':output=runtime.model(**kwargs)
            elif path=='hidden':
                kwargs.pop('labels',None);output=runtime.model.model(**kwargs)
            elif path=='cached':
                item=ACTIVE_CACHE[index]
                if item['batch_id']!=id(batch):raise ValueError('Cross-edit cache reuse')
                one['input']=item['input']
                z=runtime.get_module('model.layers.30')(item['hidden'],**item['kwargs'])[0]
                output=runtime.get_module('model.layers.31')(z,**item['kwargs'])
            else:raise ValueError(path)
            del output
        finally:
            for h in handles:h.remove()
        if not one['patch'].requires_grad or not one['final'].requires_grad:raise RuntimeError('Detached regularizer path')
        for name in features:features[name].append(one[name])
        if diagnostic:masks.append(dict(valid=int(mask.sum()),input_mask=mask.cpu().tolist(),patch_mask=mask.cpu().tolist(),final_mask=mask.cpu().tolist(),writer_mask=hook.token_mask.cpu().tolist(),active=int(hook.token_mask.sum()),delta_norm=one['delta_norm']))
        if boundaries is not None:boundaries.append(one)
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


def build_prefix_cache(runtime,batches):
    """Cache only the frozen prefix, at the input of the entire L30 block."""
    class Captured(Exception):pass
    result=[]
    def detached(value):
        if torch.is_tensor(value):return value.detach().clone()
        if isinstance(value,tuple):return tuple(detached(v) for v in value)
        if isinstance(value,dict):return {k:detached(v) for k,v in value.items()}
        if value is None or isinstance(value,(bool,int,float)):return value
        raise TypeError('Unexpected non-tensor prefix state')
    for batch in batches:
        item={'batch_id':id(batch)};mask=batch.attention_mask.bool()
        def first(_m,args):item['input']=args[0][mask].float().mean(0).detach().clone()
        def boundary(_m,args,kwargs):
            item['hidden']=args[0].detach().clone();item['kwargs']=detached(kwargs)
            raise Captured()
        h0=runtime.get_module('model.layers.0').register_forward_pre_hook(first)
        h30=runtime.get_module('model.layers.30').register_forward_pre_hook(boundary,with_kwargs=True)
        try:
            kwargs=batch.forward_kwargs();kwargs.pop('labels',None)
            with torch.no_grad():runtime.model.model(**kwargs)
            raise RuntimeError('Prefix boundary never reached')
        except Captured:pass
        finally:h0.remove();h30.remove()
        if item['kwargs'].get('use_cache') or item['kwargs'].get('past_key_value') is not None:raise ValueError('KV state is not a frozen prefix')
        result.append(item)
    return result
