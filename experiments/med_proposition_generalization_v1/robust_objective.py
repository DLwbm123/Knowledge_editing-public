"""Mathematical primitives for a frozen two-view span objective.

Callers must bind spans to the actual shifted target mask in each view and
recompute both views with unchanged parameters and RNG before backward.
"""
import torch


def weights_and_value(first, second, tau):
    if first.ndim != 1 or second.shape != first.shape or first.numel() == 0 or tau <= 0:
        raise ValueError('Two nonempty matching span vectors and positive temperature required')
    pair = torch.stack((first.detach(), second.detach()))
    if not torch.isfinite(pair).all():
        raise ValueError('Nonfinite span loss')
    scaled = pair / tau
    weights = torch.softmax(scaled, dim=0)
    value = tau * (torch.logsumexp(scaled, dim=0) - torch.log(pair.new_tensor(2.0)))
    return weights, value.mean()


def view_loss(ce, spans, weights, strength):
    if ce.ndim != 0 or spans.ndim != 1 or spans.shape != weights.shape or spans.numel() == 0:
        raise ValueError('CE, span and frozen weight shapes do not match')
    if weights.requires_grad or strength < 0:
        raise ValueError('Weights must be detached; strength must be nonnegative')
    return 0.5 * ce + strength * (weights * spans).mean()
