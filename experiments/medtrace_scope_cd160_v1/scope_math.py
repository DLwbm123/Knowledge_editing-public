"""At most two gain halfspaces intersected with the existing Fisher trust ball."""
import itertools
import torch
from protection_math import Geometry, RIDGE


def solve(geometry, gradients, raw):
    gradients, raw = gradients.double().cpu(), raw.double().cpu()
    radius = float(raw.norm())
    if radius == 0:
        return raw.clone(), dict(status='ZERO_PROPOSAL')
    norms = gradients.norm(dim=1)
    keep = norms > 0
    a = gradients[keep] / norms[keep, None]
    b = .9 * (gradients[keep] @ raw).clamp_min(0) / norms[keep] / radius
    if not len(b) or not torch.any(b > 0):
        return torch.zeros_like(raw), dict(status='ZERO_FEASIBLE')
    assert len(b) <= 2 and torch.isfinite(a).all() and torch.isfinite(b).all()
    def candidate(mu):
        coord = a @ geometry.v.T
        inverse = (coord / (geometry.values + RIDGE + mu)) @ geometry.v
        inverse += (a - coord @ geometry.v) / (RIDGE + mu)
        gram = a @ inverse.T
        for size in range(1, len(b) + 1):
            for subset in itertools.combinations(range(len(b)), size):
                ids = list(subset)
                matrix = gram[ids][:, ids]
                lam = torch.linalg.lstsq(matrix, b[ids, None], rcond=1e-12).solution[:, 0]
                if not torch.allclose(matrix @ lam, b[ids], atol=1e-9, rtol=1e-8):
                    continue
                value = lam @ inverse[ids]
                if lam.min() >= -1e-9 and torch.all(a @ value >= b - 1e-9):
                    return value, ids, lam
        return None
    answer = candidate(0.)
    if answer is None:
        return None, dict(status='INFEASIBLE_HALFSPACES')
    mu = 0.
    if answer[0].norm() > 1 + 1e-9:
        lo, hi = 0., 1.
        for _ in range(64):
            answer = candidate(hi)
            if answer is not None and answer[0].norm() <= 1 + 1e-10:
                break
            hi *= 2
        else:
            return None, dict(status='NO_FEASIBLE_TRUST_BALL_POINT')
        for _ in range(80):
            mid = (lo + hi) / 2
            point = candidate(mid)
            if point is None or point[0].norm() > 1:
                lo = mid
            else:
                hi, answer = mid, point
        mu = hi
    value, ids, lam = answer
    applied = geometry.v.T @ (geometry.values * (geometry.v @ value)) + RIDGE * value
    residual = applied + mu * value - lam @ a[ids]
    assert value.norm() <= 1 + 1e-8 and torch.all(a @ value >= b - 1e-8)
    assert residual.norm() < 1e-6 and abs(mu * float(value.square().sum() - 1)) < 1e-6
    return radius * value, dict(status='SOLVED', multiplier=mu, active=ids,
        KKT_residual=float(residual.norm()), required=b.tolist(), achieved=(a @ value).tolist())


def feasible(gain, required, tolerance=1e-10):
    return bool(torch.isfinite(gain).all() and torch.all(gain >= required - tolerance))


def selfcheck():
    geometry = Geometry(torch.eye(2))
    point, audit = solve(geometry, torch.eye(2), torch.ones(2))
    assert torch.allclose(point, torch.full((2,), .9, dtype=torch.float64), atol=1e-8)
    point, _ = solve(geometry, torch.tensor([[1., 0.], [-1., 0.]]), torch.tensor([1., 0.]))
    assert point is None
    point, _ = solve(geometry, torch.tensor([[1., 0.], [2., 0.]]), torch.tensor([1., 0.]))
    assert torch.allclose(point, torch.tensor([.9, 0.], dtype=torch.float64), atol=1e-8)
    point, _ = solve(geometry, torch.eye(2), -torch.ones(2))
    assert torch.count_nonzero(point) == 0
    geometry = Geometry(torch.diag(torch.tensor([.001, 1., .1])))
    g = torch.tensor([[1., 1., 0.], [0., 1., 1.]])
    raw = torch.tensor([.3, .4, .3])
    point, _ = solve(geometry, g, raw)
    assert point is not None and point.norm() <= raw.double().norm() + 1e-8
    assert torch.all(g.double() @ point >= .9 * (g.double() @ raw.double()) - 1e-8)
    assert not feasible(torch.tensor([1., -.1]), torch.zeros(2))
    return dict(status='PASS', conflicting_constraints=True, dependent_constraints=True,
                two_active_constraints=True, trust_ball=True, negative_group_rejected=True)


if __name__ == '__main__':
    print(selfcheck())
