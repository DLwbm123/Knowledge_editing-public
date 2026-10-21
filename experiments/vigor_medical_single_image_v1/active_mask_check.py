"""Frozen active-target-mask CPU regression, no optimization or VLM."""
def active_mask_check():
    import time
    import resource
    from types import SimpleNamespace
    import torch
    import torch.nn.functional as F
    started=time.time();torch.set_num_threads(1)
    assert not torch.cuda.is_initialized()
    counts={'forward':0,'gradient_paths':0};checks={}
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__();self.down_proj=torch.nn.Linear(9,6,bias=False,dtype=torch.float64);self.bump=False
            with torch.no_grad():self.down_proj.weight.copy_(torch.arange(54).reshape(6,9).double().sin()*.05)
        def forward(self,x):
            counts['forward']+=1;assert counts['forward']<=8
            z=self.down_proj(x)
            if self.bump:
                offset=torch.zeros_like(z);offset[:,1,0]=100;offset[:,3,5]=-100;z=z+offset
            return SimpleNamespace(logits=z)
    model=Model();layer=model.down_proj;base=layer.weight.detach().clone()
    u=torch.linalg.qr(torch.arange(12,dtype=torch.float64).reshape(6,2).cos())[0]
    v=torch.linalg.qr(torch.arange(18,dtype=torch.float64).reshape(9,2).sin())[0]
    a=torch.tensor([[.03,-.01],[.02,.04]],dtype=torch.float64,requires_grad=True)
    x=torch.arange(45,dtype=torch.float64).reshape(1,5,9).cos()
    class Batch:
        labels=torch.tensor([[-100,1,-100,3,-100]])
        def forward_kwargs(self):return {'x':x}
    batch=Batch()
    def gradient(loss):
        counts['gradient_paths']+=1;assert counts['gradient_paths']<=3
        return torch.autograd.grad(loss,a)[0]
    flags=[p.requires_grad for p in model.parameters()];modes=[m.training for m in model.modules()]
    def restored():return torch.equal(layer.weight,base) and layer.weight.dtype==base.dtype and flags==[p.requires_grad for p in model.parameters()] and modes==[m.training for m in model.modules()] and all(p.grad is None for p in model.parameters()) and 'forward' not in layer.__dict__
    with coefficient_forward(model,layer,u,a,v):
        actual=nll(model,batch);ga=gradient(actual)
        logits=model(x).logits
        reference=F.cross_entropy(logits[0,[0,2]].float(),torch.tensor([1,3]));gr=gradient(reference)
        checks['two_valid_shifted_targets']=int((batch.labels[:,1:]!=-100).sum())==2
        checks['loss_valid_token_mean']=bool(torch.allclose(actual,reference,atol=1e-7,rtol=1e-6))
        checks['gradient_valid_token_mean']=bool(torch.allclose(ga,gr,atol=1e-8,rtol=1e-5) and ga.norm()>0)
        model.bump=True
        masked=nll(model,batch);gm=gradient(masked)
        checks['ignored_logits_loss_invariant']=bool(torch.equal(actual,masked))
        checks['ignored_logits_gradient_invariant']=bool(torch.equal(ga,gm))
        model.bump=False;before=counts['forward'];batch.labels=torch.full((1,5),-100)
        try:nll(model,batch)
        except ValueError:checks['empty_mask_rejected_before_forward']=counts['forward']==before
        else:checks['empty_mask_rejected_before_forward']=False
    checks['normal_restore']=restored()
    try:
        with coefficient_forward(model,layer,u,a,v):raise RuntimeError('injected')
    except RuntimeError:pass
    checks['exception_restore']=restored()
    assert all(checks.values())
    return {'status':'ACTIVE_MASK_CPU_PASS_DEVICE_NOT_ADMITTED','checks':checks,'counters':counts,'loss':float(actual.detach()),'gradient_norm':float(ga.norm()),'CPU_seconds':time.time()-started,'RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'optimization':0,'VLM':0,'GPU':0,'Judge':0}
