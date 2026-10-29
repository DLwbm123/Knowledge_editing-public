import io
import json

rng = torch.Generator().manual_seed(20261029)
max_err = 0.0
for case in range(8):
    paths = [dict(x=torch.randn(n, 7, generator=rng, dtype=torch.float64),
                  full=torch.randn(n, 8, 5, generator=rng, dtype=torch.float64),
                  score=torch.randn(n, 5, generator=rng, dtype=torch.float64))
             for n in (2, 3, 4, 5)]
    _, primary, audit = med_anchor(paths, [-1., -2., -3., -2.5], case)
    secondary = dict(x=torch.randn(4, 7, generator=rng, dtype=torch.float64),
                     y=torch.randn(4, 1, 5, generator=rng, dtype=torch.float64))
    block = build(primary, secondary, audit['full_trace'], gram, energy)
    gram2 = gram(block, block)
    torch.testing.assert_close(gram2, torch.eye(2, dtype=torch.float64) * audit['full_trace'] / 2,
                               rtol=1e-9, atol=1e-9)
    left = torch.randn(5, 2, generator=rng, dtype=torch.float64, requires_grad=True)
    right = torch.randn(2, 7, generator=rng, dtype=torch.float64, requires_grad=True)
    state = dict(left=left, right=right)
    actual = penalty(state, dict(parts=[block]), lambda s: (s['left'], s['right']))
    matrices = torch.einsum('tso,ti->soi', block['y'], block['x'])
    dense = torch.einsum('soi,oi->s', matrices, left @ right).square().sum()
    torch.testing.assert_close(actual, dense, rtol=1e-9, atol=1e-9)
    g = torch.autograd.grad(actual, (left, right), retain_graph=True)
    h = torch.autograd.grad(dense, (left, right), retain_graph=True)
    for a, b in zip(g, h):
        torch.testing.assert_close(a, b, rtol=1e-9, atol=1e-9)
    gauge = torch.tensor([[2., .3], [.2, 1.]], dtype=torch.float64)
    transformed = penalty(dict(left=left @ gauge, right=torch.linalg.solve(gauge, right)),
                          dict(parts=[block]), lambda s: (s['left'], s['right']))
    torch.testing.assert_close(actual, transformed, rtol=1e-9, atol=1e-9)
    assert float(penalty(dict(left=left * 0, right=right), dict(parts=[block]),
                         lambda s: (s['left'], s['right']))) == 0
    buffer = io.BytesIO()
    torch.save(block, buffer)
    buffer.seek(0)
    loaded = torch.load(buffer, map_location='cpu', weights_only=True)
    torch.testing.assert_close(loaded['x'], block['x'])
    torch.testing.assert_close(loaded['y'], block['y'])
    assert loaded['token_counts'] == block['token_counts']
    max_err = max(max_err, abs(float(actual - dense)))
    for bad in (dict(x=primary['x'], y=primary['y'] * 2),
                dict(x=secondary['x'], y=secondary['y'] * 0)):
        try:
            build(primary, bad, audit['full_trace'], gram, energy)
        except ValueError:
            pass
        else:
            raise AssertionError('degenerate direction accepted')
print(json.dumps(dict(status='MECHANICAL_PASS', synthetic_cases=8,
                      value_and_both_gradients=True, orthogonality_and_trace=True,
                      gauge_and_zero=True, memory_serialization=True,
                      collinear_and_zero_rejected=True, max_value_error=max_err,
                      model_calls=0, training=0, judge_calls=0)))
