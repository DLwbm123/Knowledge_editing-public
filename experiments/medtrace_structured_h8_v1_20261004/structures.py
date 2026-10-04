"""Fixed CP -> TuckerC4 embedding; persisted structure and free inference are separate."""
import math
import torch
from torch import nn


class TuckerC4(nn.Module):
    def __init__(self, cp):
        super().__init__()
        if cp.rank != 4 or (cp.d_in, cp.d_out) != (14336, 4096):
            raise ValueError('Frozen CP4 dimensions required')
        self.d_in, self.d_out, self.rank = cp.d_in, cp.d_out, 4
        self.epsilon = cp.epsilon
        for name, source in [('u1', cp.u_out), ('u2', cp.v_out),
                             ('u3', cp.u_in), ('u4', cp.v_in)]:
            setattr(self, name, nn.Parameter(source.detach().clone()))
        left = cp.rho.new_zeros(16, 4)
        right = cp.rho.new_zeros(16, 4)
        for k in range(4):
            left[5*k, k] = cp.rho.detach()[k] * cp.beta / math.sqrt(4)
            right[5*k, k] = 1
        self.L, self.R = nn.Parameter(left), nn.Parameter(right)

    def factors(self):
        return (torch.kron(self.u1, self.u2) @ self.L,
                (torch.kron(self.u3, self.u4) @ self.R).T)

    def residual(self, activation):
        x = activation.to(self.L.dtype)
        x = x / (x.square().mean(-1, keepdim=True).sqrt() + self.epsilon)
        b, a = self.factors()
        return (x @ a.T) @ b.T

    def normalize_factors_(self, **_):
        # Historical Tucker recipe: no rescaling of factors or Adam state.
        pass

    def parameter_groups(self):
        return [self.u3, self.u4, self.R], [self.u1, self.u2, self.L]


def optimizer_for(expert, base):
    """Keep original CP/FREE grouping, add only the frozen Tucker grouping."""
    from methods.medtrace.selective_write import optimizer_for as original
    if not isinstance(expert, TuckerC4):
        return original(expert, base)
    assert not any(p.requires_grad for p in base.parameters())
    assert all(p.dtype == torch.float32 for p in expert.parameters())
    inputs, outputs = expert.parameter_groups()
    return torch.optim.Adam([{'params': inputs, 'lr': 1e-4},
                             {'params': outputs, 'lr': 1e-3}],
                            betas=(.9, .999), eps=1e-8, weight_decay=0)


def free_state(expert):
    """Export an independent inference state; never overwrite the structured expert."""
    b, a = expert.factors()
    return {'A': a.detach().clone(), 'B': b.detach().clone()}
