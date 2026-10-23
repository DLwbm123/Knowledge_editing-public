"""CPU convex correction with cumulative drift and explicit radius constraints.

SLSQP is accepted only with primal and nonnegative-multiplier KKT checks.
Failure means no numerically certified solution, never global infeasibility.
"""
import itertools
import numpy as np
from scipy.optimize import minimize, nnls


def solve(A, b, B, g, M, M0, z, C=None, d=None):
    A, b, B, g, M, M0, z = [np.asarray(x, dtype=np.float64)
                             for x in (A, b, B, g, M, M0, z)]
    B, M, M0 = [(x + x.T) / 2 for x in (B, M, M0)]
    assert A.shape[1] == len(z) == 2
    assert all(np.isfinite(x).all() for x in (A, b, B, g, M, M0, z))
    assert all(np.linalg.eigvalsh(x).min() > 0 for x in (B, M, M0))
    linear_A, linear_b = A.copy(), b.copy()
    if C is not None:
        C, d = np.asarray(C, dtype=np.float64), np.asarray(d, dtype=np.float64)
        assert C.shape == (5, 2) and d.shape == (5,)
        # All signs represent the five-value L1 ball exactly, without smoothing.
        signs = np.asarray(list(itertools.product((-1., 1.), repeat=5)))
        linear_A = np.concatenate((A, -signs @ C), axis=0)
        linear_b = np.concatenate((b, signs @ d - 7.5))

    def constraints(x):
        values = [*list(linear_A @ x - linear_b),
                  36. - x @ M @ x, 36. - (z + x) @ M0 @ (z + x)]
        if C is not None:
            values.append(2.25 - np.mean((d + C @ x) ** 2))
        return np.asarray(values)

    def jacobian(x):
        rows = [*list(linear_A), -2 * M @ x, -2 * M0 @ (z + x)]
        if C is not None:
            rows.append(-2 * C.T @ (d + C @ x) / len(d))
        return np.asarray(rows)

    def objective(x):
        return float(x @ B @ x + 2 * g @ x)

    def gradient(x):
        return 2 * (B @ x + g)

    # Deterministic numerical initialization, not parameter or seed search.
    result = minimize(objective, np.zeros(2), jac=gradient, method='SLSQP',
                      constraints=[dict(type='ineq', fun=constraints, jac=jacobian)],
                      options=dict(ftol=1e-10, maxiter=300, disp=False))
    x = result.x
    values = constraints(x)
    grad = gradient(x)
    active = values <= 1e-5
    multipliers = np.zeros(len(values))
    if active.any():
        try:
            multipliers[active], _ = nnls(jacobian(x)[active].T, grad, maxiter=1000)
        except (RuntimeError, np.linalg.LinAlgError) as error:
            return dict(status='NUMERICAL_KKT_FAILURE', delta=None,
                        error=repr(error), primal_min=float(values.min()),
                        scipy_success=bool(result.success),
                        scipy_message=str(result.message))
    stationarity = float(np.linalg.norm(grad - jacobian(x).T @ multipliers))
    complementarity = float(np.max(np.abs(values * multipliers)))
    tolerance = 1e-5 * (1 + float(np.linalg.norm(grad)))
    valid = (np.isfinite(x).all() and values.min() >= -1e-6 and
             stationarity <= tolerance and complementarity <= tolerance)
    return dict(status='KKT_VALID' if valid else 'NO_CERTIFIED_LOCAL_SOLUTION',
                delta=x.tolist() if valid else None, objective=objective(x),
                primal_min=float(values.min()), stationarity=stationarity,
                complementarity=complementarity, kkt_tolerance=tolerance,
                scipy_success=bool(result.success), scipy_message=str(result.message),
                iterations=int(result.nit))


def selfcheck():
    I = np.eye(2)
    A = np.tile([1., 0.], (5, 1))
    d = np.full(5, 2.)
    first = solve(A, np.array([.25, .1, .1, .1, .1]) - 1,
                  A.T @ A / 5 + .04 * I, A.T @ d / 5,
                  I, I, np.zeros(2), A, d)
    assert first['status'] == 'KKT_VALID', first
    assert first['delta'][0] < -.5
    second = solve([[1., 1.]], [7.], np.diag([.04, 4.04]), np.zeros(2),
                   I, I, np.zeros(2))
    assert second['status'] == 'KKT_VALID', second
    x = np.asarray(second['delta'])
    assert x.sum() >= 7 - 1e-6 and x @ x <= 36 + 1e-6
    impossible = solve([[1., 0.]], [7.], I, np.zeros(2), I, I, np.zeros(2))
    assert impossible['delta'] is None, impossible
    zero = solve(A, -np.ones(5), I, np.zeros(2), I, I, np.zeros(2), A, np.zeros(5))
    assert zero['status'] == 'KKT_VALID' and np.allclose(zero['delta'], 0), zero
    # Total radius binds independently of the correction radius.
    total = solve([[1., 0.]], [2.], I, np.zeros(2), I, I, np.array([5., 0.]))
    assert total['delta'] is None, total
    return dict(status='PASS', cumulative_drift=first, radius_boundary=second,
                impossible=impossible, zero=zero, total_radius=total,
                model_calls=0, note='Synthetic tests, no real-model efficacy claim')


if __name__ == '__main__':
    import json
    print(json.dumps(selfcheck(), indent=2))
