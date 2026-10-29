import json

g = torch.Generator().manual_seed(20261029)
max_error = 0.
for index in range(8):
    paths = [dict(x=torch.randn(n, 7, generator=g, dtype=torch.float64),
                  full=torch.randn(n, 8, 5, generator=g, dtype=torch.float64),
                  score=torch.randn(n, 5, generator=g, dtype=torch.float64))
             for n in (2, 3, 4, 5)]
    full, primary, audit = med_anchor(paths, [-1., -2., -3., -2.5], index)
    result, status = projected_direction(full, primary, audit['full_trace'], gram, energy, build)
    assert status == 'OK'
    fp = runtime_fp32(result, audit['full_trace'], gram)
    bound, status = mapped_block(full, primary, audit, gram, energy, build)
    assert status == 'OK' and bound['x'].dtype == torch.float32 and bound['y'].dtype == torch.float32
    torch.testing.assert_close(bound['y'], fp['y'])
    fleft = torch.randn(5, 2, generator=g, dtype=torch.float32, requires_grad=True)
    fright = torch.randn(2, 7, generator=g, dtype=torch.float32, requires_grad=True)
    fvalue = penalty(dict(left=fleft, right=fright), dict(parts=[fp]), lambda s: (s['left'], s['right']))
    fm = torch.einsum('tso,ti->soi', fp['y'].double(), fp['x'].double())
    fexpected = torch.einsum('soi,oi->s', fm, (fleft.double() @ fright.double())).square().sum()
    torch.testing.assert_close(fvalue.double(), fexpected, rtol=3e-5, atol=1e-4)
    fg = torch.autograd.grad(fvalue, (fleft, fright), retain_graph=True)
    fh = torch.autograd.grad(fexpected, (fleft, fright))
    for a, b in zip(fg, fh):
        torch.testing.assert_close(a, b, rtol=3e-5, atol=1e-4)
    merged = pack_mapped([fp, primary])
    fstate = dict(left=fleft, right=fright)
    fallback = {k: v.float() for k, v in primary.items()}
    separate = penalty(fstate, dict(parts=[fp, concatenate([fallback])]), lambda s: (s['left'], s['right']))
    packed = penalty(fstate, dict(parts=[merged]), lambda s: (s['left'], s['right']))
    torch.testing.assert_close(packed, separate, rtol=3e-5, atol=1e-4)
    target = torch.eye(2, dtype=torch.float64) * audit['full_trace'] / 2
    torch.testing.assert_close(gram(result, result), target, rtol=1e-9, atol=1e-9)
    probes = torch.cat([torch.einsum('tso,ti->soi', b['y'], b['x']).reshape(8, -1)
                        for b in full])
    p = torch.einsum('tso,ti->soi', primary['y'], primary['x']).reshape(-1)
    projected = probes - (probes @ p / p.square().sum())[:, None] * p
    dense = projected.T @ projected
    r = torch.einsum('tso,ti->soi', result['y'], result['x'])[1].reshape(-1)
    rayleigh = (r @ dense @ r) / r.square().sum()
    eigenmax = torch.linalg.eigvalsh(dense)[-1]
    torch.testing.assert_close(rayleigh, eigenmax, rtol=1e-9, atol=1e-9)
    max_error = max(max_error, abs(float(rayleigh - eigenmax)))
    left = torch.randn(5, 2, generator=g, dtype=torch.float64, requires_grad=True)
    right = torch.randn(2, 7, generator=g, dtype=torch.float64, requires_grad=True)
    actual = penalty(dict(left=left, right=right), dict(parts=[result]), lambda s: (s['left'], s['right']))
    matrices = torch.einsum('tso,ti->soi', result['y'], result['x'])
    expected = torch.einsum('soi,oi->s', matrices, left @ right).square().sum()
    torch.testing.assert_close(actual, expected, rtol=1e-9, atol=1e-9)
    for a, b in zip(torch.autograd.grad(actual, (left, right), retain_graph=True),
                    torch.autograd.grad(expected, (left, right))):
        torch.testing.assert_close(a, b, rtol=1e-9, atol=1e-9)
    zero = [dict(x=b['x'], y=torch.zeros_like(b['y'])) for b in full]
    try:
        projected_direction(zero, primary, audit['full_trace'], gram, energy, build)
    except ValueError:
        pass
    else:
        raise AssertionError('zero probe covariance accepted')
unit = torch.ones(1, 1, dtype=torch.float64)
primary = dict(x=unit, y=torch.tensor([[[1., 0., 0.]]], dtype=torch.float64))
equal = [dict(x=unit, y=torch.zeros(1, 8, 3, dtype=torch.float64)) for _ in range(4)]
equal[0]['y'][0, 0, 1] = 1
equal[0]['y'][0, 1, 2] = 1
assert projected_direction(equal, primary, 2., gram, energy, build)[1] == 'PROJECTED_TOP_TIE'
equal[0]['y'].zero_()
equal[0]['y'][0, 0, 0] = 1
assert projected_direction(equal, primary, 1., gram, energy, build)[1] == 'PROJECTED_DEGENERATE'
print(json.dumps(dict(status='PASS', cases=8, dense_principal=True, factor_gradients=True,
                      orthogonal_trace=True, invalid_zero_rejected=True, top_tie_fallback=True,
                      degenerate_fallback=True, float32_penalty_gradients=True,
                      mapped_fallback_packing=True, max_eigen_error=max_error)))
