"""Fixed-objective root correction with a separate linearization anchor and trust ball."""
import numpy as np
from scipy.optimize import minimize


def solve(rows, proposal, center, orthogonal_squared, radius, gradient,
          anchor, anchor_gain, required_gain, budget, scale, trust_radius):
    rows, proposal, center, gradient, anchor = [np.asarray(x, dtype=np.float64) for x in
                                               (rows, proposal, center, gradient, anchor)]
    assert rows.ndim == 2 and rows.shape[1] == len(proposal)
    assert proposal.shape == center.shape == gradient.shape == anchor.shape
    assert all(np.isfinite(x).all() for x in (rows, proposal, center, gradient, anchor))
    assert min(radius, budget, scale, trust_radius) > 0 and orthogonal_squared >= 0
    ball = (radius**2-orthogonal_squared)/scale**2
    if ball < -1e-8:
        return None, dict(status='CERTIFIED_LOCAL_INFEASIBLE', certificate='tail_distance_exceeds_ball')
    matrix = rows*(scale/np.sqrt(budget))
    offset = rows@proposal/np.sqrt(budget)
    displacement = (proposal-center)/scale
    trust_displacement = (proposal-anchor)/scale
    source = -gradient*scale
    required = required_gain-anchor_gain+gradient@(proposal-anchor)
    normalization = max(np.linalg.norm(source), abs(required), 1e-12)
    source, required = source/normalization, required/normalization
    values, vectors = np.linalg.eigh(matrix.T@matrix)
    assert values.min() >= -1e-9*max(1., values.max())
    values = np.maximum(values, 0.)
    at_offset = vectors.T@(matrix.T@offset)
    displacement_e, source_e, trust_e = [vectors.T@x for x in (displacement, source, trust_displacement)]

    def evaluate(multipliers, regularizer=1.):
        lam, mu, nu, eta = multipliers
        rhs = -2*lam*at_offset-2*mu*displacement_e+nu*source_e-2*eta*trust_e
        diagonal = regularizer+2*lam*values+2*mu+2*eta
        if regularizer == 0 and diagonal.min() <= 1e-10*max(1., diagonal.max()):
            return None
        u = vectors@(rhs/diagonal)
        residuals = np.array([np.linalg.norm(matrix@u+offset)**2-1.,
                              np.linalg.norm(u+displacement)**2-ball,
                              required-source@u,
                              np.linalg.norm(u+trust_displacement)**2-(trust_radius/scale)**2])
        return .5*regularizer*(u@u)+multipliers@residuals, residuals, u, rhs, diagonal

    def dual(x):
        objective, residuals, *_ = evaluate(x)
        return -objective, -residuals

    result = minimize(dual, np.zeros(4), jac=True, method='L-BFGS-B', bounds=[(0., None)]*4,
                      options=dict(maxiter=2000, maxls=50, ftol=1e-14, gtol=1e-10))
    objective, residuals, u, rhs, diagonal = evaluate(result.x)
    stationarity = float(np.linalg.norm(diagonal*(vectors.T@u)-rhs))
    complementarity = float(np.max(np.abs(result.x*residuals)))
    audit = dict(status='NUMERIC_UNRESOLVED', iterations=int(result.nit), solver_message=str(result.message),
                 residuals=residuals.tolist(), multipliers=result.x.tolist(), stationarity=stationarity,
                 complementarity=complementarity, source_normalization=float(normalization),
                 trust_radius=float(trust_radius), objective_center_fixed=True)
    if (residuals.max() <= 1e-7 and stationarity <= 1e-7*max(1., np.linalg.norm(rhs))
            and complementarity <= 1e-6*max(1., abs(objective))):
        audit['status'] = 'LOCAL_KKT_PASS'
        return proposal+scale*u, audit
    if result.x.sum() > 0:
        weights = result.x/result.x.sum()
        bound = evaluate(weights, regularizer=0.)
        if bound is not None:
            minimum = bound[0]
            constants = np.array([offset@offset-1., displacement@displacement-ball, required,
                                  trust_displacement@trust_displacement-(trust_radius/scale)**2])
            if minimum > 1e-6*max(1., np.max(np.abs(weights*constants))):
                audit.update(status='CERTIFIED_LOCAL_INFEASIBLE', certificate='positive_constraint_mixture',
                             certificate_weights=weights.tolist(), certificate_lower_bound=float(minimum))
    return None, audit


def selfcheck():
    import joint_math
    base = dict(rows=np.eye(2), proposal=np.array([2., 0.]), center=np.zeros(2), orthogonal_squared=0.,
                radius=2.1, gradient=np.array([0., -1.]), gain_at_proposal=0., required_gain=.2, budget=1., scale=1.)
    expected, _ = joint_math.solve(**base)
    args = dict(base); args.pop('gain_at_proposal')
    root, audit = solve(**args, anchor=np.array([1., .1]), anchor_gain=.1, trust_radius=10.)
    assert audit['status']=='LOCAL_KKT_PASS' and np.linalg.norm(root-expected)<1e-5
    root, audit = solve(**dict(args, rows=np.zeros((2,2)), required_gain=-1., gradient=np.zeros(2)),
                        anchor=np.zeros(2), anchor_gain=0., trust_radius=.5)
    assert audit['status']=='LOCAL_KKT_PASS' and np.allclose(root, [.5,0.], atol=1e-6)
    root, audit = solve(**dict(args, required_gain=2.), anchor=np.zeros(2), anchor_gain=0., trust_radius=.5)
    assert root is None and audit['status']=='CERTIFIED_LOCAL_INFEASIBLE'
    # A curved true gain fails the first tangent model and passes after relinearization.
    anchor=expected.copy(); repairs=0
    for fraction in (.25,.125,.0625):
        gain=anchor[1]-.1*anchor[1]**2
        gradient=np.array([0., -(1.-.2*anchor[1])])
        root, audit=solve(**dict(args, gradient=gradient), anchor=anchor, anchor_gain=gain,
                          trust_radius=fraction*base['radius'])
        assert audit['status']=='LOCAL_KKT_PASS' and root is not None
        actual=root[1]-.1*root[1]**2
        assert actual>=gain and np.linalg.norm(root-anchor)<=fraction*base['radius']+1e-6
        repairs+=1;anchor=root
        if actual>=.2-1e-6:break
    assert repairs==2 and actual>=.2-1e-6 and np.linalg.norm(root)<=1.+1e-6
    return dict(status='PASS', fixed_objective_moving_anchor=True, inactive_trust_matches_parent=True,
                active_trust_ball=True, local_infeasibility_certificate=True, curved_gain_repair=True)


if __name__=='__main__':
    import json
    print(json.dumps(selfcheck()))
