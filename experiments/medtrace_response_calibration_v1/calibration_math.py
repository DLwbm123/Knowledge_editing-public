"""Separate observed logit curvature from the predicted directional response."""
import torch


def quadratic(reference, change):
    p = reference.double().softmax(-1)
    centered = change.double() - (p * change.double()).sum(-1, keepdim=True)
    return float(.5 * (p * centered.square()).sum(-1).mean())


def response_metrics(reference, candidate, tangent, scale):
    observed = candidate.double() - reference.double()
    predicted = tangent.double() * scale
    q_observed = quadratic(reference, observed)
    q_predicted = quadratic(reference, predicted)
    error = quadratic(reference, observed - predicted)
    return dict(observed_logit_quadratic=q_observed, JVP_quadratic=q_predicted,
                response_error_quadratic=error,
                relative_response_error=(error / q_observed) ** .5 if q_observed > 0 else None,
                logit_RMS=float(observed.square().mean().sqrt()),
                changed_logit_fraction=float((observed != 0).double().mean()))


def selfcheck():
    z = torch.tensor([[1., -1., .4], [.1, .8, -.3]], dtype=torch.float64)
    direction = torch.tensor([[.2, .1, -.3], [.4, -.5, .1]], dtype=torch.float64)
    f = lambda x: z + x * direction
    base, tangent = torch.autograd.functional.jvp(f, torch.tensor(0., dtype=torch.float64),
                                                 torch.tensor(1., dtype=torch.float64), strict=True)
    assert torch.equal(base, z) and torch.equal(tangent, direction)
    assert abs(quadratic(z, direction + 10) - quadratic(z, direction)) < 1e-13
    for scale in (.25, .5, 1.):
        m = response_metrics(z, f(scale), tangent, scale)
        assert m['response_error_quadratic'] < 1e-30
        assert abs(m['JVP_quadratic'] - scale ** 2 * quadratic(z, direction)) < 1e-14
    epsilon = 1e-3
    lp, lq = z.log_softmax(-1), f(epsilon).log_softmax(-1)
    kl = float((lp.exp() * (lp - lq)).sum(-1).mean())
    assert abs(kl / quadratic(z, epsilon * direction) - 1) < 1e-3
    return dict(status='PASS',directional_derivative=True,shift_invariance=True,quadratic_scaling=True)


if __name__ == '__main__':
    print(selfcheck())
