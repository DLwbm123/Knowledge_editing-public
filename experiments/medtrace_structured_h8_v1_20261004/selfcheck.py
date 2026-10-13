"""CPU only: nonzero function, input-gradient, storage and one-step Tucker checks."""
import copy
import io
import json
import torch
from methods.medtrace.core import AsymmetricCPExpert
from structures import TuckerC4, optimizer_for, free_state


def main():
    torch.manual_seed(20260912)
    cp = AsymmetricCPExpert(14336, 4096, 4).float()
    with torch.no_grad():
        cp.rho.copy_(torch.tensor([.1, -.2, .3, -.4]))
    tk = TuckerC4(cp)
    assert sum(p.numel() for p in tk.parameters()) == 1600
    assert sum(p.numel() for p in cp.parameters()) == 1476
    x = torch.randn(2, 14336, requires_grad=True)
    ycp, ytk = cp.residual(x), tk.residual(x)
    torch.testing.assert_close(ycp, ytk, rtol=2e-5, atol=2e-6)
    vector = torch.randn_like(ycp)
    gcp = torch.autograd.grad((ycp*vector).sum(), x, retain_graph=True)[0]
    gtk = torch.autograd.grad((ytk*vector).sum(), x)[0]
    torch.testing.assert_close(gcp, gtk, rtol=2e-5, atol=2e-6)
    packed = io.BytesIO(); torch.save(tk.state_dict(), packed); packed.seek(0)
    restored = TuckerC4(cp)
    restored.load_state_dict(torch.load(packed, weights_only=True))
    assert all(torch.equal(v, restored.state_dict()[k]) for k,v in tk.state_dict().items())
    opt = optimizer_for(tk, torch.nn.Identity()); assert not opt.state
    (tk.residual(x)*vector).sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.norm()>0 for p in tk.parameters())
    opt.step()
    state = free_state(tk)
    xn = x / (x.square().mean(-1,keepdim=True).sqrt()+tk.epsilon)
    torch.testing.assert_close(tk.residual(x), (xn@state['A'].T)@state['B'].T, rtol=2e-5, atol=2e-6)
    assert set(tk.state_dict()) == {'u1','u2','u3','u4','L','R'}
    print(json.dumps({'status':'PASS_CPU_ONLY','CP_parameters':1476,'Tucker_parameters':1600,
                      'FREE_parameters':sum(v.numel() for v in state.values()),
                      'checks':['nonzero CP embedding','input gradient','structured save-load','all factor gradients','post-step free export'],
                      'GPU_loads':0,'Judge_attempts':0,'actual_model_mechanical':'NOT_RUN'}))

if __name__ == '__main__': main()
