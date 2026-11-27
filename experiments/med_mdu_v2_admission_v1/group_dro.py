"""Matched equal-weight and entropic group/answer-unit objectives."""
import math

import torch


def aggregation_weights(group_losses, tau=None):
    """Return detached unit weights, aggregate loss and per-group effective count."""
    assert group_losses and all(v.ndim == 1 and v.numel() for v in group_losses)
    units = torch.cat([v.detach() for v in group_losses])
    assert torch.isfinite(units).all()
    groups = len(group_losses)
    log_prior = torch.cat([torch.full_like(v, -math.log(groups * v.numel())) for v in group_losses])
    if tau is None:
        weights = log_prior.exp()
        value = (weights * units).sum()
    else:
        assert math.isfinite(tau) and tau > 0
        scaled = log_prior + units / tau
        weights = scaled.softmax(dim=0)
        value = tau * scaled.logsumexp(dim=0)
    per_unit = weights.split([v.numel() for v in group_losses])
    per_group = torch.stack([v.sum() for v in per_unit])
    effective_groups = per_group.square().sum().reciprocal()
    return per_unit, value, per_group, effective_groups


def check():
    regular = [torch.tensor([1., 1.]), torch.tensor([1.]), torch.tensor([1., 1.]), torch.tensor([1.])]
    uniform, value, groups, count = aggregation_weights(regular, 0.25)
    assert torch.allclose(groups, torch.full((4,), 0.25))
    assert torch.isclose(value, torch.tensor(1.)) and torch.isclose(count, torch.tensor(4.))
    hard = regular.copy()
    hard[2] = torch.tensor([4., 4.])
    weighted, value, groups, count = aggregation_weights(hard, 0.25)
    assert groups.argmax() == 2 and groups[2] > 0.99 and count < 1.01
    detached = [v.clone().requires_grad_(True) for v in hard]
    flat = torch.cat(detached)
    prior = torch.cat([v.log() for v in uniform])
    true_loss = 0.25 * torch.logsumexp(prior + flat / 0.25, dim=0)
    grads = torch.autograd.grad(true_loss, detached)
    assert all(torch.allclose(a, b) for a, b in zip(grads, weighted, strict=True))
    equal, _, _, count = aggregation_weights(hard)
    assert all(torch.allclose(a, b) for a, b in zip(equal, uniform, strict=True))
    print('GROUP_DRO_OBJECTIVE_PASS')


if __name__ == '__main__':
    check()
