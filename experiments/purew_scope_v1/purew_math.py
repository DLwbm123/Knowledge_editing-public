"""Editing-side geometry only; no model parameters or inference modules."""
import torch
import torch.nn.functional as F


@torch.no_grad()
def row_basis(weight):
    if weight.ndim != 2 or weight.shape[0] >= weight.shape[1]:
        raise ValueError('expected a wide down-projection matrix')
    if not torch.isfinite(weight).all():
        raise ValueError('nonfinite weight')
    q, r = torch.linalg.qr(weight.T, mode='reduced')
    diagonal = r.diag().abs()
    if diagonal.min() <= 1e-6 * diagonal.max():
        raise ValueError('full row-rank admission failed; do not silently change geometry')
    return q


@torch.no_grad()
def step_direction(gradient, basis=None, lr=.01, max_gradient_norm=5.):
    if not torch.isfinite(gradient).all():
        raise ValueError('nonfinite gradient')
    direction = gradient.clone()
    if basis is not None:
        direction.sub_((direction @ basis) @ basis.T)
    norm = direction.norm()
    delta = -lr * direction * (max_gradient_norm / norm.clamp_min(max_gradient_norm))
    if not torch.isfinite(delta).all():
        raise ValueError('nonfinite update')
    return delta, dict(raw_gradient_norm=float(gradient.norm()),
                       projected_gradient_norm=float(norm), step_norm=float(delta.norm()))


def target_logits(model, batch):
    labels = batch.labels[:, 1:]
    mask = labels != -100
    if not mask.any():
        raise ValueError('empty target')
    logits = model(**batch.forward_kwargs()).logits[:, :-1][mask].float()
    if not torch.isfinite(logits).all():
        raise ValueError('nonfinite logits')
    return logits, labels[mask]


def preservation_kl(logits, reference):
    if logits.shape != reference.shape:
        raise ValueError('reference shape mismatch')
    return F.kl_div(F.log_softmax(logits, -1), F.softmax(reference.detach(), -1), reduction='batchmean')


def selfcheck():
    torch.manual_seed(7)
    w = torch.randn(3, 7, dtype=torch.float64)
    g = torch.randn_like(w)
    q = row_basis(w)
    projected, receipt = step_direction(g, q)
    p = torch.eye(7, dtype=w.dtype) - q @ q.T
    expected = g @ p
    expected *= -.01 * min(1., 5. / float(expected.norm()))
    assert torch.allclose(projected, expected, atol=1e-12)
    assert (projected @ w.T).norm() < 1e-12
    assert projected.norm() <= .05 + 1e-12
    assert float((g * projected).sum()) < 0
    assert step_direction(torch.zeros_like(w), q)[0].count_nonzero() == 0
    raw, _ = step_direction(g)
    assert torch.allclose(raw, -.01 * g * min(1., 5. / float(g.norm())))
    x = torch.randn(4, 7, dtype=w.dtype)
    parameter = torch.nn.Parameter(w.clone())
    loss = (x @ parameter.T - 1.).square().mean()
    loss.backward()
    before = parameter.detach().clone()
    with torch.no_grad():
        parameter.add_(step_direction(parameter.grad, q, lr=.001)[0])
    assert (x @ parameter.T - 1.).square().mean() < loss
    assert not torch.equal(parameter, before)
    reference = torch.randn(5, 11, dtype=w.dtype, requires_grad=True)
    current = reference.detach().clone().requires_grad_(True)
    divergence = preservation_kl(current, reference)
    divergence.backward()
    assert abs(float(divergence)) < 1e-12 and current.grad.abs().max() < 1e-12
    assert reference.grad is None
    try:
        row_basis(torch.ones_like(w))
    except ValueError:
        pass
    else:
        raise AssertionError('rank deficient admission must fail')
    try:
        step_direction(torch.full_like(w, float('nan')))
    except ValueError:
        pass
    else:
        raise AssertionError('nonfinite admission must fail')
    return dict(status='PASS',projection=True,trust_cap=True,zero_gradient=True,
                real_weight_autograd=True,reference_detached=True,invalid_input_rejected=True)


if __name__ == '__main__':
    print(selfcheck())
