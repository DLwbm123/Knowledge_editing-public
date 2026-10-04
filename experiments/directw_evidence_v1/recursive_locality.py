"""M-ORE Eqs. 12--13, independently implemented in caller-chosen coordinates.

This numerical component does not choose editable weights, add a low-rank
adapter, load a model, or authorize an experiment. Keep one inverse per module.
The current proposal uses P[t-1]; commit the returned P[t] only after acceptance.
Source: https://arxiv.org/html/2605.20273 (ICML 2026).
"""
from __future__ import annotations

import math
import torch
from torch import Tensor


def initial_inverse(width: int, ridge: float, *, dtype=torch.float32,
                    device=None) -> Tensor:
    """P[0] = (I + ridge I)^-1. Width is the chosen coordinate dimension."""
    if type(width) is not int or width < 1 or not math.isfinite(ridge) or ridge <= 0:
        raise ValueError("positive coordinate width and ridge required")
    if dtype not in (torch.float32, torch.float64):
        raise ValueError("recursive statistics require FP32 or FP64")
    return torch.eye(width, dtype=dtype, device=device) / (1 + ridge)


@torch.no_grad()
def propose(gradient: Tensor, inverse: Tensor, pooled_key: Tensor,
            learning_rate: float) -> tuple[Tensor, Tensor]:
    """Return direction and next inverse without modifying input state.

The caller supplies a masked pooled key, in the same coordinates as gradient.
For original W these are actual W inputs; for paper-style B these are A @ key.
Rejected or rolled-back edits discard BOTH the direction and the next inverse.
No reset, seed generation, pooling, precision conversion or truncation is hidden.
"""
    if inverse.ndim != 2 or inverse.shape[0] != inverse.shape[1] or not inverse.numel():
        raise ValueError("square nonempty inverse required")
    width = inverse.shape[0]
    if gradient.ndim != 2 or not gradient.numel() or gradient.shape[1] != width or pooled_key.shape != (width,):
        raise ValueError("gradient/key coordinate mismatch")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("positive finite learning rate required")
    if inverse.dtype not in (torch.float32, torch.float64) or any(
        x.dtype != inverse.dtype or x.device != inverse.device
        for x in (gradient, pooled_key)
    ):
        raise ValueError("explicit same-device FP32/FP64 coordinates required")
    if any(not torch.isfinite(x).all() for x in (gradient, inverse, pooled_key)):
        raise ValueError("nonfinite recursive input")
    if not torch.allclose(inverse, inverse.T):
        raise ValueError("symmetric inverse required")
    direction = -learning_rate * (gradient @ inverse)
    projected = inverse @ pooled_key
    denominator = 1 + pooled_key @ projected
    if not torch.isfinite(denominator) or float(denominator) < 1:
        raise ValueError("invalid positive-definite recursion")
    next_inverse = inverse - torch.outer(projected, projected) / denominator
    next_inverse = (next_inverse + next_inverse.T) * .5
    if not torch.isfinite(direction).all() or not torch.isfinite(next_inverse).all():
        raise ValueError("nonfinite recursive output")
    return direction, next_inverse


def self_check() -> dict:
    """Small CPU oracle: dense solve, temporal order, rollback, module isolation."""
    rng = torch.Generator().manual_seed(20261004)
    dtype = torch.float64
    width, ridge, rate = 9, .3, .02
    inverse = initial_inverse(width, ridge, dtype=dtype)
    gram = torch.eye(width, dtype=dtype) * (1 + ridge)
    other_module = inverse.clone()
    worst = 0.
    for step in range(12):
        gradient = torch.randn(4, width, generator=rng, dtype=dtype)
        key = torch.randn(width, generator=rng, dtype=dtype)
        old = inverse.clone()
        direction, candidate = propose(gradient, inverse, key, rate)
        expected = -rate * torch.linalg.solve(gram, gradient.T).T
        error = float((direction - expected).abs().max())
        worst = max(worst, error)
        torch.testing.assert_close(direction, expected, rtol=1e-11, atol=1e-12)
        assert torch.equal(old, inverse), "proposal mutated committed history"
        assert float(torch.linalg.eigvalsh(candidate).min()) > 0
        if step in (3, 7):
            continue  # Deliberate rejection: neither W nor history is committed.
        gram = gram + torch.outer(key, key)
        torch.testing.assert_close(candidate, torch.linalg.inv(gram), rtol=1e-11, atol=1e-12)
        inverse = candidate
    assert torch.equal(other_module, initial_inverse(width, ridge, dtype=dtype))
    # A pooled zero key carries no history; it must not damp a later update.
    _, same = propose(gradient, inverse, torch.zeros_like(key), rate)
    torch.testing.assert_close(same, inverse, rtol=0, atol=0)
    try:
        propose(gradient, inverse, torch.full_like(key, float('nan')), rate)
    except ValueError:
        pass
    else:
        raise AssertionError("nonfinite key accepted")
    return dict(status="PASS", scope="CPU_SYNTHETIC_RECURSION_ONLY",
                proposals=12, committed=10, rejected=2,
                max_direction_absolute_error=worst,
                native_model_loaded=False, editing_quality_measured=False)
