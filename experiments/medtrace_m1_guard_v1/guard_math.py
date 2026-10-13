"""Minimum operator correction within a candidate's fixed right-canonical tail."""
import math
import torch
import module_math as mm


def root_rows(state, cache):
    _, a = mm.factors(state)
    i, j = state['G1'].shape[1], state['G2'].shape[1]
    rows = []
    for x, y in zip(cache['x'].split(cache['token_counts']), cache['y'].split(cache['token_counts'])):
        xa = x @ a.T
        rows.append(torch.einsum('tsij,ajb,tb->sia', y.reshape(len(x), y.shape[1], i, j), state['G2'], xa).flatten(1))
    return torch.cat(rows).detach().cpu().double()


def ellipsoid(rows, vector, budget):
    assert budget > 0 and torch.isfinite(rows).all() and torch.isfinite(vector).all()
    rows, vector = rows.double(), vector.double()
    current = float((rows @ vector).square().sum())
    if current <= budget:
        return vector.clone(), dict(status='UNCHANGED', multiplier=0., penalty=current)
    h = rows.T @ rows
    eigenvalues, eigenvectors = torch.linalg.eigh(h)
    assert eigenvalues.min() >= -1e-10 * eigenvalues.max().clamp_min(1e-30)
    eigenvalues = eigenvalues.clamp_min(0.)
    coordinates = eigenvectors.T @ vector
    def point(mu):
        z = coordinates / (1. + mu * eigenvalues)
        return z, float((eigenvalues * z.square()).sum())
    lo, hi = 0., 1.
    for _ in range(64):
        if point(hi)[1] <= budget * (1. - 2e-5):
            break
        hi *= 2.
    else:
        raise FloatingPointError('No registered ellipsoid multiplier bracket')
    for _ in range(40):
        mid = (lo + hi) / 2.
        if point(mid)[1] > budget * (1. - 2e-5):
            lo = mid
        else:
            hi = mid
    z, penalty = point(hi)
    result = eigenvectors @ z
    residual = result - vector + hi * h @ result
    assert residual.norm() <= 1e-8 * vector.norm().clamp_min(1.)
    return result, dict(status='ROOT_PROJECTED', multiplier=hi, penalty=penalty,
                        correction_norm=float((result-vector).norm()), KKT_residual=float(residual.norm()))


def project(state, cache, budget):
    rows = root_rows(state, cache)
    root = state['G1'].detach().cpu().double().flatten()
    torch.testing.assert_close((rows @ root).square().sum(), mm.operator_penalty(state, cache).detach().cpu().double(), rtol=1e-4, atol=1e-8)
    value, audit = ellipsoid(rows, root, budget)
    out = {k: v.clone() for k, v in state.items()}
    out['G1'] = value.reshape_as(state['G1']).to(state['G1'])
    penalty = float(mm.operator_penalty(out, cache))
    assert penalty <= budget + 1e-9 + 1e-5 * budget
    audit['stored_penalty'] = penalty
    return out, audit


def source_moments_mixed_step(state, mixed, moments):
    result = {}
    for group, lr, moment in zip(mm.GROUPS, mm.LRS, moments['second_moments']):
        denom = math.sqrt(moment / (1 - .999 ** moments['step'])) + 1e-8
        for key in group:
            result[key] = state[key] - lr * mixed[key] / denom
    return result


def selfcheck():
    rows = torch.diag(torch.tensor([1., 2.], dtype=torch.float64))
    vector = torch.tensor([1., 1.], dtype=torch.float64)
    unchanged, a = ellipsoid(rows, vector, 6.)
    assert a['status'] == 'UNCHANGED' and torch.equal(unchanged, vector)
    value, a = ellipsoid(rows, vector, .5)
    assert a['status'] == 'ROOT_PROJECTED' and (rows @ value).square().sum() <= .5
    assert a['KKT_residual'] < 1e-9
    g = torch.Generator().manual_seed(20261012)
    shapes = ((1, 3, 2), (2, 4, 2), (2, 5, 2), (2, 6, 1))
    state = mm.canonicalize({k: torch.randn(s, generator=g, dtype=torch.float64) for k, s in zip(mm.KEYS, shapes)})[0]
    cache = dict(x=torch.randn(7, 30, generator=g, dtype=torch.float64),
                 y=torch.randn(7, 3, 12, generator=g, dtype=torch.float64), token_counts=[3, 4])
    projected, _ = project(state, cache, float(mm.operator_penalty(state, cache))*.5)
    b, a = mm.factors(state); bp, ap = mm.factors(projected)
    torch.testing.assert_close(((b @ a) - (bp @ ap)).norm(), (state['G1']-projected['G1']).norm(), rtol=1e-9, atol=1e-9)
    grads = {k: torch.randn(v.shape, generator=g, dtype=v.dtype) for k,v in state.items()}
    m = dict(step=0, second_moments=[0., 0.])
    one, clock = mm.canonical_rms_step(state, grads, m)
    two = source_moments_mixed_step(state, grads, clock)
    for key in state:
        torch.testing.assert_close(one[key], two[key])
    return dict(status='PASS', ellipsoid_KKT=True, operator_root_isometry=True,
                source_only_clock=True, full_TT_update_not_root_only_training=True)
