"""One frozen CPU check of the production coefficient and nll subpath."""
def check_interface():
    import resource
    import time
    from types import SimpleNamespace
    import torch
    import torch.nn.functional as F
    started=time.time();torch.set_num_threads(1)
    assert not torch.cuda.is_initialized()
    counts={'forward':0,'gradient_paths':0};checks=[]
    def check(name,condition):
        checks.append({'name':name,'passed':bool(condition)})
        if not condition:raise AssertionError(name)
    class Model(torch.nn.Module):
        def __init__(self,dtype):
            super().__init__();self.down_proj=torch.nn.Linear(9,6,bias=False,dtype=dtype)
            with torch.no_grad():self.down_proj.weight.copy_((torch.arange(54).reshape(6,9).sin()*.05).to(dtype))
        def forward(self,x):
            counts['forward']+=1
            if counts['forward']>64:raise RuntimeError('Forward budget')
            return SimpleNamespace(logits=self.down_proj(x))
    def grad(loss,a):
        counts['gradient_paths']+=1
        if counts['gradient_paths']>16:raise RuntimeError('Gradient budget')
        return torch.autograd.grad(loss,a)[0]
    model=Model(torch.float64);layer=model.down_proj;base=layer.weight.detach().clone()
    u=torch.linalg.qr(torch.arange(12,dtype=torch.float64).reshape(6,2).cos())[0]
    v=torch.linalg.qr(torch.arange(18,dtype=torch.float64).reshape(9,2).sin())[0]
    x=torch.arange(36,dtype=torch.float64).reshape(1,4,9).cos()
    labels=torch.tensor([[-100,1,2,3]])
    a=torch.tensor([[.03,-.01],[.02,.04]],dtype=torch.float64,requires_grad=True)
    b=torch.tensor([[-.02,.04],[.01,-.03]],dtype=torch.float64,requires_grad=True)
    class Batch:
        def __init__(self,inp):self.labels=labels;self.inp=inp
        def forward_kwargs(self):return {'x':self.inp}
    original=(model.training,layer.training,layer.weight.requires_grad,layer.weight.dtype)
    def restored():return torch.equal(layer.weight,base) and (model.training,layer.training,layer.weight.requires_grad,layer.weight.dtype)==original and layer.weight.grad is None and 'forward' not in layer.__dict__
    with coefficient_forward(model,layer,u,a,v):
        actual=model(x).logits
        expected=F.linear(x,compose(base,u,a,v))
        check('continuous_composition',torch.allclose(actual,expected,atol=1e-12,rtol=0))
        objective=F.cross_entropy(actual[:,:-1].reshape(-1,6),labels[:,1:].reshape(-1))
        analytic=grad(objective,a)
        fd=torch.zeros_like(a)
        for i in range(2):
            for j in range(2):
                values=[]
                for sign in [-1,1]:
                    with torch.no_grad():a[i,j]+=sign*1e-6
                    z=model(x).logits
                    values.append(float(F.cross_entropy(z[:,:-1].reshape(-1,6),labels[:,1:].reshape(-1))))
                    with torch.no_grad():a[i,j]-=sign*1e-6
                fd[i,j]=(values[1]-values[0])/2e-6
        error=float((fd-analytic).norm()/analytic.norm())
        check('double_coefficient_FD',error<=1e-5)
        real=nll(model,Batch(x));explicit=F.cross_entropy(model(x).logits[:,:-1].float().reshape(-1,6),labels[:,1:].reshape(-1),ignore_index=-100)
        check('production_nll_causal_mask',torch.equal(real,explicit))
        ga=grad(real,a);gn=grad(nll(model,Batch(x*.7+.1)),a)
        check('overlapping_visual_null_both_sensitive',ga.norm()>0 and gn.norm()>0)
        check('other_endpoint_unconnected',b.grad is None and layer.weight.grad is None)
    check('normal_restoration',restored())
    with coefficient_forward(model,layer,u,b,v):gb=grad(nll(model,Batch(x)),b)
    check('second_endpoint_nonzero_and_no_base_grad',gb.norm()>0 and a.grad is None and restored())
    try:
        with coefficient_forward(model,layer,u,a,v):raise RuntimeError('injected')
    except RuntimeError:pass
    check('exception_restoration',restored())
    bad=a.detach().clone();bad[0,0]=float('nan')
    try:
        with coefficient_forward(model,layer,u,bad,v):pass
    except ValueError:pass
    else:raise AssertionError('Nonfinite accepted')
    check('nonfinite_restoration',restored())
    h=layer.register_forward_hook(lambda *args:None)
    try:
        try:
            with coefficient_forward(model,layer,u,a,v):pass
        except ValueError:pass
        else:raise AssertionError('Hook accepted')
    finally:h.remove()
    model.alias=layer
    try:
        try:
            with coefficient_forward(model,layer,u,a,v):pass
        except ValueError:pass
        else:raise AssertionError('Alias accepted')
    finally:del model.alias
    layer.weight.grad=torch.ones_like(layer.weight)
    try:
        try:
            with coefficient_forward(model,layer,u,a,v):pass
        except ValueError:pass
        else:raise AssertionError('Pending grad accepted')
    finally:layer.weight.grad=None
    check('guards_restore_existing_state',restored())
    model=Model(torch.float16);layer=model.down_proj;base=layer.weight.detach().clone()
    original=(model.training,layer.training,layer.weight.requires_grad,layer.weight.dtype)
    candidate=compose(base,u,a.detach(),v).half()
    with native_weight(model,layer,candidate):
        check('native_prefill',torch.equal(model(x.half()).logits,F.linear(x.half(),candidate)))
        check('native_one_token_shape',torch.equal(model(x[:,-1:].half()).logits,F.linear(x[:,-1:].half(),candidate)))
        check('interface_removed_before_native', 'forward' not in layer.__dict__)
    check('native_normal_restore',restored())
    try:
        with native_weight(model,layer,candidate):raise RuntimeError('injected')
    except RuntimeError:pass
    check('native_exception_restore',restored())
    try:
        with native_weight(model,layer,torch.full_like(candidate,float('inf'))):pass
    except ValueError:pass
    else:raise AssertionError('Inf accepted')
    check('native_nonfinite_restore_and_duplicate',restored() and torch.equal(candidate,candidate.clone()))
    return {'status':'CPU_SUBPATH_PASS_GPU_NOT_ADMITTED','checks':checks,'counters':counts,'FD_relative_error':error,'seconds':time.time()-started,'RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'VLM':0,'GPU':0,'Judge':0,'optimization_steps':0,'real_KV_cache_tested':False}
