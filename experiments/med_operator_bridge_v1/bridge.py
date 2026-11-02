"""Output-space mixture of two frozen same-layer MedTrace expert residuals."""
import torch
from torch import nn

class OperatorBridge(nn.Module):
    def __init__(self, med, du, coefficient=0.5):
        super().__init__()
        if coefficient not in (0.0, 0.5, 1.0):
            raise ValueError('only endpoint checks and the preregistered midpoint are permitted')
        self.med = med.requires_grad_(False)
        self.du = du.requires_grad_(False)
        self.coefficient = coefficient

    def residual(self, activation):
        if self.coefficient == 0.0:
            return self.med.residual(activation)
        if self.coefficient == 1.0:
            return self.du.residual(activation)
        a, b = self.med.residual(activation), self.du.residual(activation)
        if a.shape != b.shape or not torch.isfinite(a).all() or not torch.isfinite(b).all():
            raise ValueError('operator residual mismatch or nonfinite value')
        return a.mul(0.5).add(b, alpha=0.5)
