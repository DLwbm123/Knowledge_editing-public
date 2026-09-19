"""Isolated loss-scaled VJP for an FP32 writer through an FP16 backbone.

This is a numerical-path repair, not a change to the HSIC coefficient.
Contract: existing .grad buffers are UNSCALED and the parameters are FP32
(or FP64 for reference tests). Call once before clipping and optimizer.step().
Do not subsequently call reg.backward(), unscale the whole optimizer, or add
these gradients a second time. No automatic scale search or semantic retries.

Locally validated on CPU synthetic graphs only; real-model CUDA acceptance is
still required. Intended for the Stage24R namespace, not historical Stage24.
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence
import math
import torch
from torch import Tensor


def _validate(reg: Tensor, parameters: Iterable[Tensor], scale: float) -> tuple[Tensor, ...]:
    params = tuple(parameters)
    if reg.numel() != 1 or not reg.requires_grad:
        raise ValueError("Regularizer must be a scalar connected to an autograd graph")
    if reg.dtype not in (torch.float32, torch.float64):
        raise TypeError("Compute the regularizer scalar in FP32 or FP64")
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be finite and positive")
    if not params or len({id(p) for p in params}) != len(params):
        raise ValueError("Provide a nonempty, unique set of writer parameters")
    for p in params:
        if not p.is_leaf or not p.requires_grad:
            raise ValueError("Only trainable writer leaf parameters are supported")
        if p.dtype not in (torch.float32, torch.float64):
            raise TypeError("Unscale at FP32/FP64 master parameters, not at FP16 activations")
        if p.grad is not None and (p.grad.is_sparse or p.grad.dtype != p.dtype):
            raise TypeError("Expected dense, unscaled master-parameter gradients")
    if not bool(torch.isfinite(reg.detach()).all()):
        raise FloatingPointError("Nonfinite regularizer; do not step optimizer")
    return params


def scaled_regularizer_gradients(
    reg: Tensor,
    parameters: Iterable[Tensor],
    *,
    scale: float = 16777216.0,
    retain_graph: bool = False,
) -> tuple[Tensor, ...]:
    """Return UNscaled gradients without changing existing .grad buffers.

    The complete signed regularizer is differentiated as a single objective.
    Disconnected parameters are errors; they are not silently replaced by zero.
    Overflow is an error before any caller-owned gradient is modified.
    """
    params = _validate(reg, parameters, scale)
    raw = torch.autograd.grad(
        reg * scale, params,
        retain_graph=retain_graph, create_graph=False, allow_unused=False,
    )
    if not all(bool(torch.isfinite(g).all()) for g in raw):
        raise FloatingPointError("Scaled regularizer VJP overflow; no optimizer step allowed")
    gradients = tuple(g.detach().to(dtype=p.dtype) / scale for p, g in zip(params, raw))
    if not all(bool(torch.isfinite(g).all()) for g in gradients):
        raise FloatingPointError("Nonfinite unscaled regularizer VJP")
    return gradients


def _norm64(values: Sequence[Tensor]) -> float:
    # Avoid a tiny-vector norm itself being lost to lower precision.
    return math.sqrt(sum(float(v.detach().double().square().sum()) for v in values))


def accumulate_scaled_regularizer(
    reg: Tensor,
    parameters: Iterable[Tensor],
    *,
    scale: float = 16777216.0,
    diagnostic: bool = False,
) -> dict[str, float | int]:
    """Add the UNscaled regularizer VJP to already accumulated CE/H/U gradients.

    No clipping, optimizer step, graph mutation or change of scientific weights.
    Inputs must not be buffers managed by another still-scaled GradScaler step.
    """
    params = tuple(parameters)
    gradients = scaled_regularizer_gradients(reg, params, scale=scale)
    before = [torch.zeros_like(p) if p.grad is None else p.grad.detach().clone() for p in params] if diagnostic else None

    # Compute/validate every result before modifying any .grad buffer.
    new_grads = tuple(g.clone() if p.grad is None else p.grad.detach() + g for p, g in zip(params, gradients))
    if not all(bool(torch.isfinite(g).all()) for g in new_grads):
        raise FloatingPointError("Gradient accumulation overflow; original .grad buffers unchanged")
    with torch.no_grad():
        for p, g in zip(params, new_grads):
            if p.grad is None:
                p.grad = g
            else:
                p.grad.copy_(g)

    stats: dict[str, float | int] = {"regularizer_grad_scale": scale}
    if diagnostic:
        assert before is not None
        actual = [p.grad - old for p, old in zip(params, before)]
        stats.update(
            applied_regularizer_gradient_norm=_norm64(gradients),
            applied_regularizer_nonzero_elements=sum(int(torch.count_nonzero(g)) for g in gradients),
            accumulated_gradient_increment_norm=_norm64(actual),
            accumulated_gradient_changed_elements=sum(int(torch.count_nonzero(g)) for g in actual),
        )
    return stats
