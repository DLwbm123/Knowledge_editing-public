"""Factorized response directions in the Frobenius metric of the actual operator."""
import math
import torch


class DegenerateCurvature(ValueError):
    pass


def gram(left, right):
    x, y = left['x'].double(), left['y'].double()
    u, v = right['x'].double(), right['y'].double()
    return torch.einsum('ts,tko,slo->kl', x @ u.T, y, v)


def energy(block):
    return float(gram(block, block).trace())


def concatenate(blocks):
    return dict(x=torch.cat([b['x'] for b in blocks]),
                y=torch.cat([b['y'] for b in blocks]),
                token_counts=[len(b['x']) for b in blocks])


def anchor(paths, scores, index, seed):
    """Four independent alias paths: two aliases per mutually exclusive class."""
    assert len(paths) == len(scores) == 4
    full = [dict(x=p['x'], y=p['full']) for p in paths]
    sizes = [b['y'].shape[1] for b in full]
    assert sizes == [8] * 4
    matrix = torch.cat([torch.cat([gram(a, b) for b in full], 1) for a in full], 0)
    matrix = (matrix + matrix.T) / 2
    values, vectors = torch.linalg.eigh(matrix)
    scale = float(values.max())
    assert math.isfinite(scale)
    if scale <= 0:
        raise DegenerateCurvature('No positive FULL_CTX response direction')
    assert float(values.min()) >= -1e-7 * scale
    keep = values > 1e-10 * scale
    assert keep.any()
    trace = float(matrix.trace())
    weights = torch.cat([torch.softmax(torch.tensor(scores[i:i+2], dtype=torch.float64), 0)
                         * sign for i, sign in ((0, 1), (2, -1))])
    med = dict(x=torch.cat([p['x'] for p in paths]),
               y=torch.cat([p['score'][:, None] * float(w) for p, w in zip(paths, weights)]))
    raw_trace = energy(med)
    assert math.isfinite(raw_trace)
    if raw_trace <= 1e-20:
        raise DegenerateCurvature('MED class-score difference has zero numerical operator rank')
    med['y'] *= math.sqrt(trace / raw_trace)
    rng = torch.Generator().manual_seed(seed + index)
    z = torch.randn(int(keep.sum()), generator=rng, dtype=torch.float64)
    z /= z.norm()
    coefficients = vectors[:, keep] @ (z / values[keep].sqrt())
    gen = dict(x=med['x'], y=torch.cat([
        torch.einsum('tso,s->to', b['y'].double(), coefficients[i*8:(i+1)*8])[:, None]
        for i, b in enumerate(full)]).float() * math.sqrt(trace))
    for b in (med, gen):
        assert abs(energy(b) / trace - 1) < 3e-5
    cosine = float(gram(med, gen)[0, 0] / trace)
    assert abs(cosine) <= 1 + 3e-5
    audit = dict(index=index, full_trace=trace, MED_trace=energy(med), GEN_trace=energy(gen),
                 full_rank=int(keep.sum()), MED_rank=1, GEN_rank=1,
                 MED_GEN_operator_cosine=cosine, eigen_relative_threshold=1e-10,
                 MED_unscaled_direction_square=raw_trace,
                 class_probabilities=torch.softmax(torch.stack([
                     torch.logsumexp(torch.tensor(scores[i:i+2], dtype=torch.float64), 0)
                     - math.log(2) for i in (0, 2)]), 0).tolist())
    return full, med, gen, audit


def penalty(state, cache, factors):
    b, a = factors(state)
    total = a.sum() * 0
    for part in cache['parts']:
        x, y = part['x'], part['y']
        xa = x @ a.T
        yb = (y.flatten(0, 1) @ b).reshape(y.shape[0], y.shape[1], b.shape[1])
        products = (xa[:, None] * yb).sum(-1)
        total = total + torch.stack([v.sum(0) for v in products.split(part['token_counts'])]).square().sum()
    return total


def selfcheck():
    rng = torch.Generator().manual_seed(132)
    paths = [dict(x=torch.randn(n, 7, generator=rng),
                  full=torch.randn(n, 8, 5, generator=rng),
                  score=torch.randn(n, 5, generator=rng)) for n in (3, 4, 2, 5)]
    full, med, gen, audit = anchor(paths, [-1., -2., -3., -2.5], 0, 132)
    for blocks in (full, [med], [gen]):
        cache = dict(parts=[concatenate(blocks)])
        left = torch.randn(5, 2, generator=rng, requires_grad=True)
        right = torch.randn(2, 7, generator=rng, requires_grad=True)
        state = dict(left=left, right=right)
        matrices = [torch.einsum('tso,ti->soi', v['y'], v['x']) for v in blocks]
        dense = sum(torch.einsum('soi,oi->s', m, left @ right).square().sum() for m in matrices)
        actual = penalty(state, cache, lambda s: (s['left'], s['right']))
        torch.testing.assert_close(actual, dense, rtol=3e-5, atol=1e-5)
        g = torch.autograd.grad(actual, (left, right), retain_graph=True)
        h = torch.autograd.grad(dense, (left, right), retain_graph=True)
        for x, y in zip(g, h):
            torch.testing.assert_close(x, y, rtol=3e-5, atol=1e-4)
        gauge = torch.tensor([[2., .3], [.2, 1.]])
        changed = penalty(dict(left=left @ gauge, right=torch.linalg.solve(gauge, right)),
                          cache, lambda s: (s['left'], s['right']))
        torch.testing.assert_close(actual, changed, rtol=3e-5, atol=1e-5)
        zero = penalty(dict(left=left * 0, right=right), cache, lambda s: (s['left'], s['right']))
        assert float(zero) == 0
    return dict(status='PASS', dense_value_and_gradients=True, gauge=True,
                cross_token_and_alias_terms=True, zero=True, anchor=audit)


if __name__ == '__main__':
    print(selfcheck())
