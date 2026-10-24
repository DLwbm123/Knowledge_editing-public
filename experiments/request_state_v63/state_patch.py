"""Single-position, training-only residual intervention; no persistent model state."""
from contextlib import contextmanager
import numpy as np
import torch


def modify(hidden, position, direction):
    assert hidden.ndim == 3 and hidden.shape[0] == 1
    assert 0 <= position < hidden.shape[1]
    assert direction.shape == (hidden.shape[2],) and not direction.requires_grad
    result = hidden.clone()
    result[0, position] = hidden[0, position] + direction.to(hidden)
    return result


@contextmanager
def intervention(layer, position, direction, explicit=False):
    if direction is None:
        yield
        return
    assert not direction.requires_grad
    if explicit:
        original = layer.forward
        def forward(hidden_states, *args, **kwargs):
            return original(modify(hidden_states, position, direction), *args, **kwargs)
        layer.forward = forward
        try:
            yield
        finally:
            layer.forward = original
    else:
        def pre(module, args):
            return (modify(args[0], position, direction), *args[1:])
        handle = layer.register_forward_pre_hook(pre)
        try:
            yield
        finally:
            handle.remove()


def random_direction(direction, owner, view):
    d = direction.detach().cpu().float()
    norm = float(d.double().norm())
    assert norm > 0 and torch.isfinite(d).all()
    z = np.random.Generator(np.random.PCG64(630022 + 100*owner + view)).standard_normal(d.numel(), dtype=np.float64)
    result = torch.from_numpy((norm*z/np.linalg.norm(z)).astype(np.float32))
    torch.testing.assert_close(result.double().norm(), d.double().norm(), rtol=1e-5, atol=1e-7)
    return result


def fixture():
    """CPU causal-attention fixture; never counts as real-model admission."""
    from torch import nn
    torch.manual_seed(630022)
    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = nn.MultiheadAttention(16, 2, dropout=0., batch_first=True)
            self.proj = nn.Linear(16, 13)
            for p in self.parameters(): p.requires_grad_(False)
            self.expert = nn.Parameter(torch.zeros(16, 13))
        def forward(self, x):
            n = x.shape[1]
            mask = torch.triu(torch.ones(n,n,dtype=torch.bool), 1)
            h = x + self.attn(x,x,x,attn_mask=mask,need_weights=False)[0]
            return self.proj(h)+h@self.expert
    layer=Block(); h=torch.randn(1,7,16); saved=h.clone(); d=torch.randn(16); p=3
    y=torch.tensor([1,2,3]); rng=torch.get_rng_state().clone()
    def measure(direction=None, explicit=False):
        with intervention(layer,p,direction,explicit):
            out=layer(h)
            loss=nn.functional.cross_entropy(out[0,p:p+3],y)
            g=torch.autograd.grad(loss,layer.expert)[0].double()
        return out.detach(),float(loss),g
    clean=measure(); zero=measure(torch.zeros_like(d))
    assert all(torch.equal(a,b) if isinstance(a,torch.Tensor) else a==b for a,b in zip(clean,zero))
    changed=modify(h,p,d); assert torch.equal(h,saved)
    assert torch.equal(changed[:,:p],h[:,:p]) and torch.equal(changed[:,p+1:],h[:,p+1:])
    assert torch.equal(changed[0,p],h[0,p]+d) and changed.data_ptr()!=h.data_ptr()
    a=measure(d); b=measure(d,True)
    assert all(torch.equal(x,y) if isinstance(x,torch.Tensor) else x==y for x,y in zip(a,b))
    assert torch.isfinite(a[2]).all() and a[2].norm()>0
    # Prefix/recomputed stepwise decoding, including the actual intervention boundary.
    for n in (p+1,p+2,7):
        with intervention(layer,p,d): hooked=layer(h[:,:n]).detach()
        explicit=layer(modify(h[:,:n],p,d)).detach()
        assert torch.equal(hooked,explicit)
    repeat=measure(); assert torch.equal((clean[2]+repeat[2])/2,clean[2])
    before=layer.expert.detach().clone()
    assert torch.equal(before-.01*clean[2],before-.01*((clean[2]+repeat[2])/2))
    assert torch.equal(rng,torch.get_rng_state()) and torch.equal(h,saved)
    r=random_direction(d,1,1); assert torch.equal(r,random_direction(d,1,1))
    assert not layer._forward_pre_hooks and not d.requires_grad
    assert torch.equal(measure()[0],clean[0])
    # Exercise the production cache with three distinct batch identities.
    from types import SimpleNamespace
    from v51_cache import PrefixCache
    class TupleLayer(nn.Module):
        def __init__(self, trainable):
            super().__init__(); self.weight=nn.Parameter(torch.randn(16,16),requires_grad=trainable)
        def forward(self,x,**kwargs):return (x@self.weight,)
    layers=[TupleLayer(False),TupleLayer(True)]
    class Batch:
        def __init__(self,x):self.x=x
        def forward_kwargs(self):return dict(inputs_embeds=self.x,use_cache=False)
    batch_list=[Batch(h.clone()+k) for k in range(3)]
    def loss(batch):
        x=batch.x
        for block in layers:x=block(x,use_cache=False)[0]
        return x.square().mean()
    runtime=SimpleNamespace(model=SimpleNamespace(model=SimpleNamespace(layers=layers),training=False,config=SimpleNamespace(output_hidden_states=False,output_attentions=False)),compute_loss=loss)
    def cg(batch):
        loss=runtime.compute_loss(batch);g=torch.autograd.grad(loss,layers[1].weight)[0]
        return float(loss),g
    refs=[cg(b) for b in batch_list]
    with PrefixCache(runtime,batch_list,1) as cache:
        for _ in range(2):
            for b,ref in zip(batch_list,refs):
                val=cg(b);assert val[0]==ref[0] and torch.equal(val[1],ref[1])
        assert cache.misses==3 and cache.hits==3
    return dict(status='PASS',cache_three_batch_loss_gradient_exact=True,kind='CPU_CAUSAL_FIXTURE_ONLY',zero_exact=True,single_position=True,no_alias=True,explicit_hook_exact=True,prefix_stepwise_exact=True,duplicate_gradient_update_exact=True,RNG_unchanged=True,detached_direction=True,random_recipe=True,real_model_calls=0)

if __name__ == '__main__':
    import json
    print(json.dumps(fixture(),indent=2))
