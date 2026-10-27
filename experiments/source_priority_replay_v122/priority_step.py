"""Treat replay and Fisher jointly as source-anchored auxiliary gradients."""
import copy
import torch
from balanced_fisher import cap_gradient, remove_opposing


def total_gradient(source, replay, fisher, groups):
    combined = {k: replay[k].detach() + fisher[k].detach() for k in source}
    capped, cap = cap_gradient(source, combined, groups)
    auxiliary, direction = remove_opposing(source, capped, groups)
    total = {k: source[k].detach() + auxiliary[k] for k in source}
    for group in groups:
        square = sum(float(source[k].double().square().sum()) for k in group)
        dot = sum(float((source[k].double() * total[k].double()).sum()) for k in group)
        assert dot >= square - 1e-6 * square - 1e-20
    return total, dict(priority_cap=cap, priority_direction=direction)


def step(r, before, source, replay, cache, coefficient, scalar, target):
    params = {k: v.detach().clone().requires_grad_(True) for k, v in before.items()}
    penalty = r.mm.operator_penalty(params, cache)
    assert torch.isfinite(penalty) and coefficient > 0
    fisher = dict(zip(params, torch.autograd.grad(coefficient * penalty, tuple(params.values()))))
    total, audit = total_gradient(source, replay, fisher, r.mm.GROUPS)
    for k, value in params.items():
        value.grad = total[k].detach().to(value)
    norm = torch.nn.utils.clip_grad_norm_(list(params.values()), 1.)
    assert torch.isfinite(norm) and all(torch.isfinite(v.grad).all() for v in params.values())
    proposed, updated = r.mm.canonical_rms_step(before,
        {k: v.grad.detach() for k, v in params.items()}, copy.deepcopy(scalar))
    if r.fm.map_norm(before, proposed) <= 1e-14:
        selected = {k: v.clone() for k, v in before.items()}
        matching = dict(status='ZERO_DIRECTION', target=target, actual=0.)
    else:
        proposed, matching = r.match_proposal(before, proposed, target)
        selected = r.mm.canonicalize(proposed)[0]
        assert abs(r.fm.map_norm(before, selected) - target) <= 1e-10 + .002 * target
    return selected, updated, dict(gradient_norm=float(norm), matching=matching,
        penalty_before=float(penalty.detach()), operator_backwards=1, source_RMS_steps=1, **audit)


def selfcheck():
    for dtype in (torch.float32, torch.float64):
        source = dict(a=torch.tensor([1., 0.], dtype=dtype), b=torch.tensor([2.], dtype=dtype))
        replay = dict(a=torch.tensor([-4., 3.], dtype=dtype), b=torch.tensor([0.], dtype=dtype))
        fisher = {k: torch.zeros_like(v) for k, v in source.items()}
        total, audit = total_gradient(source, replay, fisher, (('a',), ('b',)))
        torch.testing.assert_close(total['a'], torch.tensor([1., .6], dtype=dtype))
        assert abs(audit['priority_cap']['scale'] - .2) < 1e-12
        zero = {k: torch.zeros_like(v) for k, v in source.items()}
        result, audit = total_gradient(zero, replay, fisher, (('a',), ('b',)))
        assert all(torch.count_nonzero(v) == 0 for v in result.values())
        assert audit['priority_cap']['scale'] == 0
    return dict(status='PASS', combined_auxiliary_cap=True, opposing_component_removed=True,
                zero_source_closes_auxiliary=True, source_inner_product_bound=True)
