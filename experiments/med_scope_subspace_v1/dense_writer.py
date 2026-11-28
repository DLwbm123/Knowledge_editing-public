"""Dense rank-four read-in with the existing TT write-out."""
import torch
from torch import nn


class DenseReadIn(nn.Module):
    def __init__(self, tt):
        super().__init__()
        self.G1 = nn.Parameter(tt.G1.detach().clone())
        self.G2 = nn.Parameter(tt.G2.detach().clone())
        self.A = nn.Parameter(tt.factors()[1].detach().clone())
        self.epsilon = tt.epsilon

    def parameter_groups(self):
        return [self.A], [self.G1, self.G2]

    def factors(self):
        b = torch.einsum('ia,ajb->ijb', self.G1[0], self.G2).reshape(-1, self.A.size(0))
        return b, self.A

    def residual(self, activation):
        b, a = self.factors()
        x = activation.to(a.dtype)
        x = x / (x.square().mean(-1, keepdim=True).sqrt() + self.epsilon)
        return (x @ a.T) @ b.T


def input_scale(a, fits):
    fits = fits.to(a)
    median = torch.quantile((fits @ a.T).norm(dim=-1), .5)
    if not torch.isfinite(median) or not median > 0:
        raise ValueError('FIT amplitude must be finite and positive')
    return median


def dense_initialization(solution, p4):
    from .scope_solver import row_basis

    fits = solution['fits'].double()
    target = input_scale(solution['K-P4']['A'].double(), fits)
    ku = solution['K-U']['A'].double()
    med = solution['K-MED']['A'].double()
    ku_scale = float(target / input_scale(ku, fits))
    med_scale = float(target / input_scale(med, fits))
    ku = ku * ku_scale
    med = med * med_scale
    cores = torch.cat([p4[k].double().flatten() for k in ('G3', 'G4')])
    core_rms = cores.square().mean().sqrt()
    lr = float(1e-3 * ku.square().mean().sqrt() / core_rms)
    if not 0 < lr < float('inf'):
        raise ValueError('Invalid RMS matched dense learning rate')
    return dict(ku=ku, med=med, ku_basis=row_basis(ku), ku_scale=ku_scale,
                med_scale=med_scale, lr=lr, fit_target=float(target))


def principal_angles(a, basis):
    qa = torch.linalg.qr(a.detach().double().T, mode='reduced').Q
    cosine = torch.linalg.svdvals(qa.T @ basis.to(qa).T).clamp(0, 1)
    return torch.rad2deg(torch.acos(cosine)).tolist()
