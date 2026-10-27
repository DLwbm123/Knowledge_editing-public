"""Project one optimizer displacement onto measured TRAIN loss halfspaces."""
import torch


def project(displacement, gradients):
    keys = tuple(displacement)
    shapes = {k: displacement[k].shape for k in keys}
    d = torch.cat([displacement[k].detach().double().cpu().flatten() for k in keys])
    rows = torch.stack([torch.cat([g[k].detach().double().cpu().flatten() for k in keys]) for g in gradients])
    norms = rows.norm(dim=1)
    rows = rows[norms > 1e-12] / norms[norms > 1e-12, None]
    assert len(rows) and torch.isfinite(rows).all() and torch.isfinite(d).all()
    gram, rhs = rows @ rows.T, rows @ d
    dual = torch.zeros(len(rows), dtype=torch.float64)
    tolerance = 1e-10 * max(float(d.norm()), 1e-12)
    converged = False
    for iteration in range(2000):
        largest = 0.
        for i in range(len(rows)):
            new = max(0., float(dual[i] + (rhs[i] - gram[i] @ dual) / gram[i, i]))
            largest = max(largest, abs(new - float(dual[i])))
            dual[i] = new
        residual = rhs - gram @ dual
        if float(residual.max()) <= tolerance and largest <= tolerance:
            converged = True
            break
    result = d - rows.T @ dual
    assert torch.isfinite(result).all()
    audit = dict(converged=converged, iterations=iteration + 1, constraints=len(rows),
                 maximum_linear_violation=float((rows @ result).max()),
                 displacement_norm_before=float(d.norm()), displacement_norm_after=float(result.norm()),
                 nonnegative_dual=bool((dual >= 0).all()), tolerance=tolerance)
    output, offset = {}, 0
    for k in keys:
        size = displacement[k].numel()
        output[k] = result[offset:offset + size].reshape(shapes[k]).to(displacement[k])
        offset += size
    assert offset == len(result)
    return output, audit


def selfcheck():
    d = dict(x=torch.tensor([2., -3.], dtype=torch.float64))
    rows = [dict(x=torch.tensor([1., 0.], dtype=torch.float64)),
            dict(x=torch.tensor([0., -1.], dtype=torch.float64))]
    value, audit = project(d, rows)
    assert audit['converged'] and torch.allclose(value['x'], torch.zeros(2, dtype=torch.float64), atol=1e-12)
    value, audit = project(d, rows[:1])
    torch.testing.assert_close(value['x'], torch.tensor([0., -3.], dtype=torch.float64))
    value, audit = project(dict(x=torch.tensor([-2., -3.], dtype=torch.float64)), rows[:1])
    torch.testing.assert_close(value['x'], torch.tensor([-2., -3.], dtype=torch.float64))
    return dict(status='PASS', orthant_solution=True, feasible_unchanged=True)
