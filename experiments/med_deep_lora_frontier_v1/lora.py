"""Fixed rank-four expert used by the existing answer-position hook."""
import math
import torch
from torch import nn


class DeepLoRA(nn.Module):
    def __init__(self, in_features, out_features, gamma=1.0):
        super().__init__()
        if (in_features, out_features) != (14336, 4096) or gamma not in (0., .25, .5, .75, 1.):
            raise ValueError('Unexpected dimensions or strength')
        self.A = nn.Parameter(torch.empty(4, in_features, dtype=torch.float32))
        self.B = nn.Parameter(torch.zeros(out_features, 4, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.gamma = gamma

    def forward(self, x):
        return self.gamma * torch.nn.functional.linear(torch.nn.functional.linear(x.float(), self.A), self.B)

    def residual(self, x):
        return self(x)


def create(module, device, gamma=1.0):
    expert = DeepLoRA(module.in_features, module.out_features, gamma).to(device)
    if sum(p.numel() for p in expert.parameters()) != 73728:
        raise ValueError('Unexpected expert parameter count')
    return expert
