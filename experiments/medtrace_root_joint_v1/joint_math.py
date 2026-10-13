"""A three-multiplier convex root problem; certificates refer only to this model."""
import numpy as np
from scipy.optimize import minimize


def solve(rows, proposal, center, orthogonal_squared, radius, gradient,
          gain_at_proposal, required_gain, budget, scale):
    """Minimize root displacement with Fisher, operator-ball and affine-source bounds."""
    rows, proposal, center, gradient = [np.asarray(x, dtype=np.float64) for x in
                                        (rows, proposal, center, gradient)]
    assert rows.ndim == 2 and rows.shape[1] == len(proposal)
    assert proposal.shape == center.shape == gradient.shape
    assert all(np.isfinite(x).all() for x in (rows, proposal, center, gradient))
    assert min(radius, budget, scale) > 0 and orthogonal_squared >= 0
    ball = (radius**2 - orthogonal_squared) / scale**2
    if ball < -1e-8:
        return None, dict(status='CERTIFIED_LOCAL_INFEASIBLE', certificate='tail_distance_exceeds_ball')
    # u=(root-proposal)/scale makes all three constraint residuals dimensionless.
    a_matrix = rows * (scale / np.sqrt(budget))
    offset = rows @ proposal / np.sqrt(budget)
    displacement = (proposal-center) / scale
    source = -gradient*scale
    required = required_gain-gain_at_proposal
    normalization = max(np.linalg.norm(source), abs(required), 1e-12)
    source, required = source/normalization, required/normalization
    hessian = a_matrix.T @ a_matrix
    values, vectors = np.linalg.eigh(hessian)
    assert values.min() >= -1e-9*max(1., values.max())
    values = np.maximum(values, 0.)
    at_offset = vectors.T @ (a_matrix.T @ offset)
    displacement_e, source_e = vectors.T @ displacement, vectors.T @ source

    def evaluate(multipliers, regularizer=1.):
        lam, mu, nu = multipliers
        rhs = -2*lam*at_offset-2*mu*displacement_e+nu*source_e
        diagonal = regularizer+2*lam*values+2*mu
        if regularizer == 0:
            # A numerical null-space truncation cannot certify a global lower bound.
            if diagonal.min() <= 1e-10*max(1., diagonal.max()):
                return None
            u_e = rhs/diagonal
        else:
            u_e = rhs/diagonal
        u = vectors @ u_e
        residuals = np.array([np.linalg.norm(a_matrix@u+offset)**2-1.,
                              np.linalg.norm(u+displacement)**2-ball,
                              required-source@u])
        objective = .5*regularizer*(u@u)+multipliers@residuals
        return objective, residuals, u, rhs, diagonal

    def dual(x):
        objective, residuals, *_ = evaluate(x)
        return -objective, -residuals

    result = minimize(dual, np.zeros(3), jac=True, method='L-BFGS-B', bounds=[(0., None)]*3,
                      options=dict(maxiter=2000, maxls=50, ftol=1e-14, gtol=1e-10))
    objective, residuals, u, rhs, diagonal = evaluate(result.x)
    stationarity = float(np.linalg.norm(diagonal*(vectors.T@u)-rhs))
    complementarity = float(np.max(np.abs(result.x*residuals)))
    audit = dict(status='NUMERIC_UNRESOLVED', iterations=int(result.nit), solver_message=str(result.message),
                 residuals=residuals.tolist(), multipliers=result.x.tolist(),
                 stationarity=stationarity, complementarity=complementarity,
                 source_normalization=float(normalization), lower_norm_constrained=False)
    if (residuals.max() <= 1e-7 and stationarity <= 1e-7*max(1., np.linalg.norm(rhs))
            and complementarity <= 1e-6*max(1., abs(objective))):
        audit['status'] = 'LOCAL_KKT_PASS'
        return proposal+scale*u, audit
    if result.x.sum() > 0:
        weights = result.x/result.x.sum()
        bound = evaluate(weights, regularizer=0.)
        if bound is not None:
            minimum, _, _, linear, diagonal = bound
            # A strictly positive global minimum of a nonnegative constraint mixture
            # certifies infeasibility; a failed optimizer alone never does.
            if minimum > 1e-6*max(1., abs(weights[0]*(offset@offset-1.)),
                                   abs(weights[1]*(displacement@displacement-ball)), abs(weights[2]*required)):
                audit.update(status='CERTIFIED_LOCAL_INFEASIBLE', certificate='positive_constraint_mixture',
                             certificate_weights=weights.tolist(), certificate_lower_bound=float(minimum))
    return None, audit


def selfcheck():
    rows = np.eye(2)
    args = dict(rows=rows, proposal=np.array([2., 0.]), center=np.zeros(2), orthogonal_squared=0.,
                radius=2.1, gradient=np.array([0., -1.]), gain_at_proposal=0.,
                required_gain=.2, budget=1., scale=1.)
    root, audit = solve(**args)
    assert audit['status']=='LOCAL_KKT_PASS' and root is not None
    assert np.linalg.norm(root-np.array([np.sqrt(.96), .2])) < 1e-5
    assert root[1] >= .2-1e-7  # nearest Fisher-only point (1,0) misses this feasible source bound
    root, audit = solve(**dict(args, required_gain=2.))
    assert root is None and audit['status']=='CERTIFIED_LOCAL_INFEASIBLE'
    root, audit = solve(**dict(args, rows=np.zeros((2,2)), gradient=np.zeros(2), required_gain=-1.))
    assert audit['status']=='LOCAL_KKT_PASS' and np.allclose(root, args['proposal'])
    root, audit = solve(**dict(args, orthogonal_squared=5.))
    assert root is None and audit['status']=='CERTIFIED_LOCAL_INFEASIBLE'
    # A nonzero center exercises the affine offset of the before-state's operator.
    root, audit = solve(**dict(args, center=np.array([.5,0.]), orthogonal_squared=.01))
    assert audit['status']=='LOCAL_KKT_PASS' and root[1]>=.2-1e-7
    return dict(status='PASS', missed_feasible_direction=True, impossible_source_certificate=True,
                inactive_constraints=True, affine_operator_ball=True, no_norm_lower_bound_in_solver=True)


if __name__ == '__main__':
    import json
    print(json.dumps(selfcheck()))
