"""Differentiable coefficients for a single bias-free down projection.

CPU subpath qualified separately; not yet admitted for a VLM launch.
"""
from contextlib import contextmanager
import torch
import torch.nn.functional as F


def validate(model, layer):
    if not isinstance(layer, torch.nn.Linear) or layer.bias is not None:
        raise ValueError('Expected bias-free Linear')
    if layer._forward_hooks or layer._forward_pre_hooks or 'forward' in layer.__dict__:
        raise ValueError('Existing hooks or forward override')
    named = list(model.named_parameters(remove_duplicate=False))
    if sum(p is layer.weight for _, p in named) != 1:
        raise ValueError('Editable weight missing or aliased')
    if any(p.grad is not None for p in model.parameters()):
        raise ValueError('Pending gradients')


def compose(base, u, coefficient, v):
    if u.shape != (base.shape[0], coefficient.shape[0]) or v.shape != (base.shape[1], coefficient.shape[1]):
        raise ValueError('Basis shape mismatch')
    weight = base.to(coefficient.dtype) + u.to(coefficient.dtype) @ coefficient @ v.to(coefficient.dtype).T
    if not torch.isfinite(weight).all():
        raise ValueError('Nonfinite composition')
    return weight


@contextmanager
def coefficient_forward(model, layer, u, coefficient, v):
    validate(model, layer)
    parameters = list(model.parameters())
    flags = [p.requires_grad for p in parameters]
    modes = [(m, m.training) for m in model.modules()]
    base = layer.weight.detach()
    compose(base, u, coefficient, v)
    def forward(x):
        weight = compose(base, u, coefficient, v)
        return F.linear(x.to(weight.dtype), weight).to(x.dtype)
    try:
        model.eval()
        for p in parameters:
            p.requires_grad_(False)
        layer.forward = forward
        yield
    finally:
        layer.__dict__.pop('forward', None)
        for p, flag in zip(parameters, flags):
            p.requires_grad_(flag)
        for module, mode in modes:
            module.training = mode


@contextmanager
def native_weight(model, layer, candidate):
    validate(model, layer)
    if candidate.shape != layer.weight.shape or not torch.isfinite(candidate).all():
        raise ValueError('Invalid native candidate')
    native = candidate.detach().to(layer.weight.dtype)
    if not torch.isfinite(native).all():
        raise ValueError('Native overflow')
    original = layer.weight.detach().clone()
    try:
        with torch.no_grad():
            layer.weight.copy_(native)
        yield
    finally:
        with torch.no_grad():
            layer.weight.copy_(original)
