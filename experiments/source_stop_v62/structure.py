"""Fixed normalized rank-one reference and source-only stop predicates."""
import copy, math, random
from contextlib import contextmanager
import numpy as np
import torch
from torch import nn

class LoRA(nn.Module):
    def __init__(self, seed):
        super().__init__()
        rng = torch.Generator().manual_seed(seed)
        a = torch.randn(1, 14336, generator=rng)
        self.A = nn.Parameter(a/a.norm(dim=1, keepdim=True))
        self.B = nn.Parameter(torch.zeros(4096, 1))
        self.epsilon = 1e-6
        assert sum(p.numel() for p in self.parameters()) == 18432
    def residual(self, activation):
        x = activation.to(self.A.dtype)
        x = x/(x.square().mean(-1, keepdim=True).sqrt()+self.epsilon)
        return (x@self.A.T)@self.B.T

def lora_step(before, gradient, scalar):
    local = {k:v.detach().clone().requires_grad_(True) for k,v in before.items()}
    offset = 0
    for k,v in local.items():
        n=v.numel();v.grad=gradient[offset:offset+n].reshape(v.shape).to(v);offset+=n
    assert offset==gradient.numel()
    torch.nn.utils.clip_grad_norm_(list(local.values()), 1.)
    step=scalar['step']+1;moments=[];out={}
    for index,(key,lr) in enumerate((('B',1e-3),('A',1e-4))):
        g=local[key].grad
        moment=.999*scalar['second_moments'][index]+.001*float(g.double().square().mean())
        denom=math.sqrt(moment/(1-.999**step))+1e-8
        out[key]=before[key]-(lr/denom)*g;moments.append(moment)
    # Match factor-role scalar-RMS rates, not TT's structure-specific canonicalization.
    assert all(torch.isfinite(v).all() for v in out.values())
    return {k:out[k] for k in before},dict(step=step,second_moments=moments)

@contextmanager
def preserve_rng():
    state=(random.getstate(),np.random.get_state(),torch.get_rng_state(),torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])
    try:yield
    finally:
        random.setstate(state[0]);np.random.set_state(state[1]);torch.set_rng_state(state[2])
        if state[3]:torch.cuda.set_rng_state_all(state[3])

def target_margin(logits, labels):
    wanted=labels[:,1:];mask=wanted!=-100;pred=logits[:,:-1][mask].float();target=wanted[mask]
    assert len(target)>0 and torch.isfinite(pred).all()
    best=pred.topk(2,dim=-1);other=torch.where(best.indices[:,0]==target,best.values[:,1],best.values[:,0])
    gap=pred.gather(1,target[:,None])[:,0]-other
    return float(gap.min()),target.detach().cpu().tolist()

def exact_source_pass(tokens,target,eos,at_cap):
    return tokens==target and bool(tokens) and tokens[-1]==eos and not at_cap

def selfcheck():
    e=LoRA(225034559);assert e.residual(torch.ones(1,2,14336)).count_nonzero()==0
    before={k:v.detach().clone() for k,v in e.state_dict().items()}
    out=e.residual(torch.ones(1,2,14336));loss=(out-1).square().mean();grads=torch.autograd.grad(loss,tuple(e.parameters()))
    assert grads[1].norm()>0
    g=torch.cat([v.flatten().double() for v in grads]);s=dict(step=0,second_moments=[0.,0.]);new,ns=lora_step(before,g,s)
    assert s==dict(step=0,second_moments=[0.,0.]) and ns['step']==1 and not torch.equal(new['B'],before['B'])
    logits=torch.zeros(1,4,5);labels=torch.tensor([[-100,-100,3,2]]);logits[0,1,3]=2;logits[0,2,2]=1
    margin,target=target_margin(logits,labels);assert margin==1 and target==[3,2]
    logits[0,2,4]=1;assert target_margin(logits,labels)[0]==0
    assert exact_source_pass([3,2],[3,2],2,False)
    assert not exact_source_pass([3,2],[3,2],2,True) and not exact_source_pass([3],[3,2],2,False)
    rng=torch.get_rng_state().clone()
    with preserve_rng():torch.rand(5);random.random();np.random.rand()
    assert torch.equal(rng,torch.get_rng_state())
    return dict(status='PASS',real_model_calls=0,normalized_LoRA_parameters=18432,zero_residual=True,gradient_and_scalar_step=True,strict_margin_with_EOS=True,RNG_restored=True)
if __name__=='__main__':
    import json;print(json.dumps(selfcheck(),indent=2))
