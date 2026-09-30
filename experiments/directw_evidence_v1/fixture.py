"""Synthetic CPU-only three-token multimodal model, never medical data/results."""
from __future__ import annotations
import torch
from torch import Tensor, nn


class TinyModel(nn.Module):
    def __init__(self, dtype: torch.dtype = torch.float64):
        super().__init__()
        self.mlp = nn.Module()
        self.mlp.down_proj = nn.Linear(4, 3, bias=True, dtype=dtype)
        self.head = nn.Linear(3, 3, bias=False, dtype=dtype)
        self.register_buffer("fixed", torch.tensor(1., dtype=dtype))
        with torch.no_grad():
            self.mlp.down_proj.weight.zero_()
            self.mlp.down_proj.bias.zero_()
            self.head.weight.copy_(torch.eye(3, dtype=dtype))
        self.eval()

    def forward(self, tokens: Tensor, image: Tensor) -> Tensor:
        expanded = image[:, None, :].expand(-1, tokens.shape[1], -1)
        features = torch.cat((tokens, expanded), -1)
        return self.head(torch.tanh(self.mlp.down_proj(features))) * self.fixed

    def generate(self, tokens: Tensor, image: Tensor, steps: int = 3) -> Tensor:
        """Fixture free generation; predictions recursively feed the next input."""
        generated = []
        current = tokens[:, -1:, :].clone()
        for _ in range(steps):
            chosen = self(current, image)[:, -1].argmax(-1)
            generated.append(chosen)
            current = torch.nn.functional.one_hot(chosen, 3)[:, None, :2].to(tokens)
        return torch.stack(generated, -1)
