"""Fixed source-pair midpoint; no learned gate, threshold, or extra parameter."""
import torch
from alignment import TT


def normalize(x):
    return x/(x.square().mean(-1,keepdim=True).sqrt()+1e-6)


class CenteredTT(TT):
    def __init__(self, seed, center):
        super().__init__(seed)
        assert center.shape==(14336,) and torch.isfinite(center).all()
        self.register_buffer('center',center.detach().clone(),persistent=False)

    def residual(self, activation):
        b,a=self.factors()
        x=normalize(activation.to(a.dtype))-self.center.to(a)
        return (x@a.T)@b.T


def selfcheck():
    rng=torch.Generator().manual_seed(105024)
    x=torch.randn(2,14336,generator=rng)
    normalized=normalize(x);center=normalized.mean(0)
    torch.testing.assert_close((normalized-center).sum(0),torch.zeros_like(center),atol=3e-7,rtol=0)
    ex=CenteredTT(225034559,center)
    assert ex.residual(x).count_nonzero()==0
    assert sum(p.numel() for p in ex.parameters())==7168 and set(ex.state_dict())=={'G1','G2','G3','G4'}
    with torch.no_grad():ex.G1.normal_(0,.01,generator=rng)
    b,a=ex.factors();expected=(normalized-center)@(b@a).T
    torch.testing.assert_close(ex.residual(x),expected,atol=1e-6,rtol=2e-5)
    torch.testing.assert_close(ex.residual(x)[0],-ex.residual(x)[1],atol=1e-6,rtol=2e-5)
    grads=torch.autograd.grad(ex.residual(x).square().sum(),tuple(ex.parameters()))
    assert all(torch.isfinite(g).all() for g in grads) and ex.center.grad is None
    plain=TT(225034559);plain.load_state_dict(ex.state_dict())
    zero=CenteredTT(225034559,torch.zeros_like(center));zero.load_state_dict(ex.state_dict())
    torch.testing.assert_close(zero.residual(x),plain.residual(x),atol=0,rtol=0)
    return dict(status='PASS',parameters=7168,zero_writer=True,midpoint_antisymmetry=True,
                dense_formula=True,zero_center_original_exact=True,frozen_center=True,models=0)
