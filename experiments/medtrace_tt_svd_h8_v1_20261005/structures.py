"""Independent TT and orthogonal SVD rank4 experts; no pretrained Base factorization."""
import torch
from torch import nn


def orthogonal(x):
    q,r=torch.linalg.qr(x,mode='reduced')
    if not torch.isfinite(r).all() or torch.any(torch.diag(r).abs()<1e-8):
        raise FloatingPointError('SVD coordinate QR lost full column rank')
    sign=torch.where(torch.diag(r)<0,-torch.ones_like(torch.diag(r)),torch.ones_like(torch.diag(r)))
    return q*sign


class TT4(nn.Module):
    def __init__(self,seed):
        super().__init__();self.d_in,self.d_out,self.rank,self.epsilon=14336,4096,4,1e-6
        rng=torch.Generator().manual_seed(seed)
        self.G1=nn.Parameter(torch.zeros(1,64,4))
        for name,shape in [('G2',(4,64,4)),('G3',(4,112,4)),('G4',(4,128,1))]:
            a=torch.randn(shape[1]*shape[2],shape[0],generator=rng)
            setattr(self,name,nn.Parameter(orthogonal(a).T.reshape(shape).contiguous()))
    def factors(self):
        b=torch.einsum('ia,ajb->ijb',self.G1[0],self.G2).reshape(4096,4)
        a=torch.einsum('aib,bj->aij',self.G3,self.G4[:,:,0]).reshape(4,14336)
        return b,a
    def residual(self,activation):
        b,a=self.factors();x=activation.to(a.dtype);x=x/(x.square().mean(-1,keepdim=True).sqrt()+self.epsilon)
        return (x@a.T)@b.T
    def normalize_factors_(self,**_):pass
    def parameter_groups(self):return [self.G3,self.G4],[self.G1,self.G2]


class SVD4(nn.Module):
    def __init__(self,seed):
        super().__init__();self.d_in,self.d_out,self.rank,self.epsilon=14336,4096,4,1e-6
        rng=torch.Generator().manual_seed(seed)
        self.U=nn.Parameter(orthogonal(torch.randn(4096,4,generator=rng)))
        self.V=nn.Parameter(orthogonal(torch.randn(14336,4,generator=rng)))
        self.s=nn.Parameter(torch.zeros(4))
    def factors(self):return orthogonal(self.U)*self.s,orthogonal(self.V).T
    def residual(self,activation):
        b,a=self.factors();x=activation.to(a.dtype);x=x/(x.square().mean(-1,keepdim=True).sqrt()+self.epsilon)
        return (x@a.T)@b.T
    def normalize_factors_(self,**_):pass
    def parameter_groups(self):return [self.V],[self.U,self.s]


def optimizer_for(expert,base):
    if not isinstance(expert,(TT4,SVD4)):
        from methods.medtrace.selective_write import optimizer_for as original
        return original(expert,base)
    assert not any(p.requires_grad for p in base.parameters())
    assert all(p.dtype==torch.float32 for p in expert.parameters())
    inputs,outputs=expert.parameter_groups()
    return torch.optim.Adam([{'params':inputs,'lr':1e-4},{'params':outputs,'lr':1e-3}],betas=(.9,.999),eps=1e-8,weight_decay=0)


def free_state(expert):
    b,a=expert.factors();return {'A':a.detach().clone(),'B':b.detach().clone()}


def selfcheck():
    import io
    from methods.medtrace import AsymmetricCPExpert
    from methods.medtrace.selective_write import LowRankExpert
    result=[];gen=torch.Generator().manual_seed(77)
    for typ,count in [(TT4,3584),(SVD4,73732)]:
        e=typ(19);f=typ(19);assert all(torch.equal(p,q) for p,q in zip(e.parameters(),f.parameters()))
        assert sum(p.numel() for p in e.parameters())==count
        x=torch.randn(3,14336,generator=gen,requires_grad=True)
        assert torch.count_nonzero(e.residual(x))==0
        target=torch.randn(3,4096,generator=gen);(e.residual(x)-target).square().mean().backward()
        first=e.G1 if typ is TT4 else e.s;assert first.grad.norm()>0
        with torch.no_grad():first.normal_(0,.1)
        e.zero_grad(set_to_none=True)
        free=LowRankExpert(AsymmetricCPExpert(14336,4096,4),19,rank=4);free.load_state_dict(free_state(e))
        a,b=e.residual(x),free.residual(x);torch.testing.assert_close(a,b,rtol=2e-5,atol=2e-6)
        torch.testing.assert_close(torch.autograd.grad(a.sum(),x,retain_graph=True)[0],torch.autograd.grad(b.sum(),x)[0],rtol=2e-5,atol=2e-6)
        (a-target).square().mean().backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.norm()>0 for p in e.parameters())
        base=nn.Linear(1,1).requires_grad_(False);opt=optimizer_for(e,base);opt.step()
        stream=io.BytesIO();torch.save(e.state_dict(),stream);stream.seek(0);f.load_state_dict(torch.load(stream,weights_only=True));torch.testing.assert_close(e.residual(x),f.residual(x),rtol=0,atol=0)
        if typ is SVD4:
            for raw in [e.U,e.V]:q=orthogonal(raw);torch.testing.assert_close(q.T@q,torch.eye(4),rtol=2e-5,atol=2e-6)
        result.append({'structure':typ.__name__,'parameters':count,'zero_start_nonzero_grad_free_parity_save_load':'PASS'})
    return result
