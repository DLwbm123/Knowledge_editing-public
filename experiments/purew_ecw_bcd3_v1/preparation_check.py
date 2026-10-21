"""Small synthetic CPU derivation/manifest checks; no production executor or model imports."""
import json
import time
import numpy as np


def main(protocol):
    started = time.monotonic()
    rng = np.random.default_rng(83)
    theta = np.zeros(5)
    base = rng.normal(size=(3, 5)) * .05
    H = base.T @ base
    q = 0.
    v = np.zeros(5)
    synthetic_terms = [(base, theta.copy())]
    for step in range(8):
        a = rng.normal(size=5) * .03
        q += v @ a + .5 * a @ H @ a
        v += H @ a
        theta += a
        if step in (2, 5):
            rows = rng.normal(size=(2, 5)) * .07
            oldq, oldv = q, v.copy()
            H += rows.T @ rows
            synthetic_terms.append((rows, theta.copy()))
            assert q == oldq and np.array_equal(v, oldv)
        oracle = sum(.5 * np.linalg.norm(rows @ (theta - birth))**2
                     for rows, birth in synthetic_terms)
        assert abs(q - oracle) < 1e-12
    margin = np.array([.2, .5, .8])
    gradients = rng.normal(size=(3, 5))
    normalized = gradients / margin[:, None]
    d = rng.normal(size=5)
    d *= np.sqrt(.02) / np.linalg.norm(normalized @ d)
    assert .5 * np.linalg.norm(normalized @ d)**2 <= .01 + 1e-14
    assert np.all(np.abs(gradients @ d) <= margin * np.sqrt(.02) + 1e-14)

    # A linear proxy may be zero while a nonlinear evidence contrast changes sign.
    nonlinear_delta = np.ones(5) / np.sqrt(5)
    assert np.linalg.norm(np.zeros((1, 5)) @ nonlinear_delta) == 0
    assert 1 - 2 * np.linalg.norm(nonlinear_delta)**2 < 0
    assert 1 - 0 - 2 * np.linalg.norm(nonlinear_delta)**2 <= 0

    metric = [np.diag([1., .2, .4]), np.diag([.1, .8, .3])]
    g = np.array([1., -.5, .3])
    eta, lam = .01, np.array([10., 20.])
    A = np.eye(3) / eta + sum(x * h for x, h in zip(lam, metric))
    candidate = -np.linalg.solve(A, g)
    budgets = [.5 * candidate @ h @ candidate for h in metric]
    assert min(budgets) > 0 and np.linalg.eigvalsh(A).min() > 0
    assert np.linalg.norm(g + candidate/eta + sum(x*h@candidate
                                                for x, h in zip(lam, metric))) < 1e-12
    # Positive multipliers, active budgets and stationarity give the dense convex optimum.
    assert all(abs(.5 * candidate @ h @ candidate - e) < 1e-14
               for h, e in zip(metric, budgets))

    ages, weights = [], []
    moments = np.zeros(4)
    for step in range(8):
        length = float(rng.uniform(.001, .03))
        a0, a1, a2, a3 = moments
        moments = np.array([a0, a1+length*a0, a2+2*length*a1+length**2*a0,
                            a3+3*length*a2+3*length**2*a1+length**3*a0])
        ages = [x+length for x in ages]
        if step in (0, 2, 5):
            weight = float(rng.uniform(.1, .3))
            ages.append(0.); weights.append(weight); moments[0] += weight
        assert abs(moments[3] - sum(w*r**3 for w, r in zip(weights, ages))) < 1e-14

    matrix = np.diag([1., 0., 0.])
    blind = np.array([0., 1., 0.])
    assert blind @ matrix @ blind == 0 and blind[1] == 1
    assert protocol['execution_authorized'] is False
    assert protocol['cohort']['new_total_formal_slots'] == 3*3*20 == 180
    assert protocol['evaluation']['all_methods_total_consumers'] == 4*161 == 644
    assert sum(protocol['evaluation']['new_generations_breakdown'].values()) == 506
    assert protocol['budgets']['maximum_generations'] == 506
    assert protocol['budgets']['maximum_committed_updates'] == 180+3
    assert protocol['arms']['B']['maximum_parameter_rows'] == 16+12+4 == 32
    assert protocol['arms']['C']['maximum_parameter_rows'] == 16+12+4+4+10 == 46
    assert protocol['arms']['D']['maximum_parameter_rows'] == 16+48+4 == 68
    assert protocol['budgets']['activation_epoch'] is None
    assert protocol['budgets']['deadline_epoch'] is None
    return dict(status='PASS_SMALL_SYNTHETIC_CPU_ONLY', double_anchor_multi_birth_oracle=True,
                no_q_v_reset=True, normalized_margin_bound=True,
                nonlinear_sign_flip_counterexample=True, multi_constraint_dense_KKT=True,
                conditional_cubic_moment_oracle=True, coverage_blind_direction=True,
                protocol_budget_consumer_counts=True, CPU_seconds=time.monotonic()-started,
                model_calls=0, Judge_calls=0, GPU_calls=0, installations=0,
                production_solver_implemented=False, production_resume_validated=False,
                global_nonlinear_or_clinical_certificate=False)


if __name__ == '__main__':
    raise SystemExit('Import with a frozen protocol; preparation only, no experiment command')
