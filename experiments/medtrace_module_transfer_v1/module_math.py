"""Canonical TT steps, seven-objective projection, and operator Fisher penalties."""
import itertools
import math
import torch

KEYS = ('G1', 'G2', 'G3', 'G4')
GROUPS = (('G1', 'G2'), ('G3', 'G4'))
LRS = (1e-3, 1e-4)
ARMS = ('C', 'M1', 'M2', 'M3', 'M12', 'M123')


def factors(state):
    g1, g2, g3, g4 = (state[k] for k in KEYS)
    b = torch.einsum('ia,ajb->ijb', g1[0], g2).flatten(0, 1)
    a = torch.einsum('aib,bj->aij', g3, g4[:, :, 0]).flatten(1)
    return b, a


def canonical_frame(matrix):
    """A deterministic row-space frame, invariant to invertible row transforms."""
    q, r = torch.linalg.qr(matrix.T, mode='reduced')
    if torch.linalg.svdvals(r).min() <= 1e-12 * torch.linalg.norm(r):
        raise FloatingPointError('Non-root TT core lost row rank')
    basis, pivots = [], []
    for _ in range(matrix.shape[0]):
        residual = q if not basis else q - torch.stack(basis).T @ (torch.stack(basis) @ q)
        leverage = residual.square().sum(1)
        maximum = leverage.max()
        # Stable first-index tie breaking makes symmetric subspaces reproducible.
        pivot = int(torch.where(leverage >= maximum - 1e-11 * maximum)[0][0])
        vector = residual @ residual[pivot]
        vector /= vector.norm()
        if vector[pivot] < 0:
            vector = -vector
        basis.append(vector)
        pivots.append(pivot)
    frame = torch.stack(basis)
    torch.testing.assert_close(frame @ frame.T, torch.eye(len(frame), dtype=frame.dtype), rtol=1e-9, atol=1e-9)
    return frame, pivots


def canonicalize(state):
    """Right-canonicalize all three bonds; the root may be exactly zero."""
    result = {k: state[k].detach().cpu().double().clone() for k in KEYS}
    pivots = []
    for index in (3, 2, 1):
        key, left_key = KEYS[index], KEYS[index - 1]
        core = result[key]
        matrix = core.flatten(1)
        frame, chosen = canonical_frame(matrix)
        transfer = matrix @ frame.T
        result[left_key] = torch.einsum('aib,bc->aic', result[left_key], transfer)
        result[key] = frame.reshape_as(core)
        pivots.append(chosen)
    return {k: v.to(state[k]) for k, v in result.items()}, pivots


def canonical_rms_step(state, gradients, optimizer_state):
    """Only two scalar second moments survive changes of the canonical frame."""
    step = optimizer_state['step'] + 1
    moments, result = [], {}
    for index, (group, lr) in enumerate(zip(GROUPS, LRS)):
        count = sum(gradients[k].numel() for k in group)
        square = sum(float(gradients[k].double().square().sum()) for k in group) / count
        moment = .999 * optimizer_state['second_moments'][index] + .001 * square
        denominator = math.sqrt(moment / (1 - .999 ** step)) + 1e-8
        for key in group:
            result[key] = state[key] - (lr / denominator) * gradients[key]
        moments.append(moment)
    assert all(torch.isfinite(v).all() for v in result.values())
    return result, dict(step=step, second_moments=moments)


def project_seven(gradients, proposal):
    """Project a realized proposal onto every individual descent halfspace."""
    gradients, proposal = gradients.detach().cpu().double(), proposal.detach().cpu().double()
    assert gradients.ndim == 2 and gradients.shape[0] == 7 and gradients.shape[1] == proposal.numel()
    assert torch.isfinite(gradients).all() and torch.isfinite(proposal).all()
    norms = gradients.norm(dim=1)
    a = gradients[norms > 0] / norms[norms > 0, None]
    if not len(a) or proposal.norm() == 0:
        return proposal.clone(), dict(status='UNCHANGED', active=[], KKT_residual=0.)
    radius = float(proposal.norm())
    raw = proposal / radius
    gram, gains = a @ a.T, a @ raw
    if gains.min() >= -1e-10:
        return proposal.clone(), dict(status='ALREADY_FEASIBLE', active=[], KKT_residual=0.)
    for size in range(1, len(a) + 1):
        for subset in itertools.combinations(range(len(a)), size):
            ids = list(subset)
            matrix = gram[ids][:, ids]
            lam = torch.linalg.lstsq(matrix, -gains[ids, None], rcond=1e-12).solution[:, 0]
            value = raw + lam @ a[ids]
            if lam.min() < -1e-9 or (a @ value).min() < -1e-9:
                continue
            residual = float((matrix @ lam + gains[ids]).norm())
            if residual > 1e-8:
                continue
            assert value.norm() <= 1 + 1e-8
            return radius * value, dict(status='PROJECTED', active=ids, KKT_residual=residual,
                retained_norm=float(value.norm()), minimum_linear_gain=float((a @ value).min()))
    # Zero is feasible, but a failed numerical solve must not masquerade as one.
    raise FloatingPointError('Seven-objective projection failed KKT checks')


def operator_penalty(state, cache):
    """Full-vocabulary Fisher sketch applied to the finite composed TT operator."""
    b, a = factors(state)
    x, y, counts = cache['x'], cache['y'], cache['token_counts']
    xa = x @ a.T
    yb = (y.flatten(0, 1) @ b).reshape(y.shape[0], y.shape[1], b.shape[1])
    products = (xa[:, None] * yb).sum(-1)
    values = torch.stack([v.sum(0) for v in products.split(counts)])
    return values.square().sum()


def flags(arm):
    assert arm in ARMS
    return '1' in arm, '2' in arm, '3' in arm


def selfcheck():
    rng = torch.Generator().manual_seed(20261011)
    shapes = ((1, 6, 3), (3, 5, 2), (2, 7, 3), (3, 8, 1))
    original = {k: torch.randn(shape, generator=rng, dtype=torch.float64) for k, shape in zip(KEYS, shapes)}
    def gauge(state):
        value = {k: v.clone() for k, v in state.items()}
        for index in range(3):
            rank = shapes[index][-1]
            matrix = torch.randn(rank, rank, generator=rng, dtype=torch.float64)
            u, _, vh = torch.linalg.svd(matrix)
            matrix = u @ torch.diag(torch.linspace(.5, 2., rank, dtype=torch.float64)) @ vh
            value[KEYS[index]] = torch.einsum('aib,bc->aic', value[KEYS[index]], matrix)
            value[KEYS[index + 1]] = torch.einsum('ab,bic->aic', torch.linalg.inv(matrix), value[KEYS[index + 1]])
        return value
    for zero in (False, True):
        state = {k: v.clone() for k, v in original.items()}
        if zero:
            state['G1'].zero_()
        other = gauge(state)
        first, _ = canonicalize(state)
        second, _ = canonicalize(other)
        for key in KEYS:
            torch.testing.assert_close(first[key], second[key], rtol=1e-8, atol=1e-9)
        b, a = factors(state)
        bc, ac = factors(first)
        torch.testing.assert_close(b @ a, bc @ ac, rtol=1e-9, atol=1e-9)
        optimizer_states = [dict(step=0, second_moments=[0., 0.]) for _ in range(2)]
        target = torch.randn(b.shape[0], a.shape[1], generator=rng, dtype=torch.float64)
        for step in range(3):
            updated = []
            for index, current in enumerate((first, second)):
                current, _ = canonicalize(current)
                current = {k: v.requires_grad_() for k, v in current.items()}
                left, right = factors(current)
                loss = (left @ right - target).square().mean()
                grads = dict(zip(KEYS, torch.autograd.grad(loss, list(current.values()))))
                value, optimizer_states[index] = canonical_rms_step(current, grads, optimizer_states[index])
                updated.append(value)
            first, second = updated[0], gauge(updated[1])
            b1, a1 = factors(first); b2, a2 = factors(second)
            torch.testing.assert_close(b1 @ a1, b2 @ a2, rtol=1e-7, atol=1e-8)
            torch.testing.assert_close(torch.tensor(optimizer_states[0]['second_moments']), torch.tensor(optimizer_states[1]['second_moments']), rtol=1e-6, atol=1e-9)
    g = torch.tensor([[1., 0.], [-1., 0.], [0., 1.], [0., 1.], [0., 0.], [0., 2.], [1., 1.]])
    projected, audit = project_seven(g, torch.tensor([1., -1.]))
    torch.testing.assert_close(projected, torch.zeros(2, dtype=torch.float64), atol=1e-8, rtol=0.)
    projected, _ = project_seven(g, torch.tensor([1., 2.]))
    torch.testing.assert_close(projected, torch.tensor([0., 2.], dtype=torch.float64), atol=1e-8, rtol=0.)
    state = {k: v.clone().requires_grad_() for k, v in original.items()}
    b, a = factors(state)
    cache = dict(x=torch.randn(9, a.shape[1], generator=rng, dtype=torch.float64),
                 y=torch.randn(9, 4, b.shape[0], generator=rng, dtype=torch.float64), token_counts=[4, 5])
    penalty = operator_penalty(state, cache)
    matrices = [torch.einsum('tso,ti->soi', y, x) for x, y in zip(cache['x'].split([4, 5]), cache['y'].split([4, 5]))]
    dense = sum(torch.einsum('soi,oi->s', matrix, b @ a).square().sum() for matrix in matrices)
    torch.testing.assert_close(penalty, dense)
    torch.testing.assert_close(penalty, operator_penalty(gauge(state), cache), rtol=1e-8, atol=1e-8)
    first = torch.autograd.grad(penalty, list(state.values()), retain_graph=True)
    second = torch.autograd.grad(dense, list(state.values()))
    for x, y in zip(first, second):
        torch.testing.assert_close(x, y)
    return dict(status='PASS', zero_root=True, three_nonorthogonal_bonds=True, composed_operator_preserved=True,
                optimizer_scalar_state_invariance=True, conflicting_and_duplicate_objectives=True,
                zero_objective=True, operator_Fisher_value_and_gradient_match_dense=True)


if __name__ == '__main__':
    print(selfcheck())
