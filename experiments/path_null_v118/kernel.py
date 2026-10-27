"""A fixed activation-path annihilator, with no trainable scope classifier."""
import types
import torch


def basis(vectors):
    rows = torch.cat(vectors).double().cpu()
    assert rows.ndim == 2 and rows.shape[1] == 14336 and len(rows) <= 2048
    assert torch.isfinite(rows).all() and (rows.norm(dim=1) > 0).all()
    rows = rows / rows.norm(dim=1, keepdim=True)
    _, singular, right = torch.linalg.svd(rows, full_matrices=False)
    rank = int((singular > singular[0] * 1e-10).sum())
    assert 0 < rank <= 256, 'Do not truncate a large protected path'
    q = right[:rank].T.contiguous()
    error = float((rows - (rows @ q) @ q.T).norm(dim=1).max())
    assert error <= 1e-8 and torch.allclose(q.T @ q, torch.eye(rank, dtype=q.dtype), atol=1e-10, rtol=0)
    return q, dict(path_positions=len(rows), rank=rank, singular_values=singular.tolist(), maximum_relative_span_error=error,
                   relative_svd_cutoff=1e-10, basis_bytes=q.numel() * q.element_size())


def response(x, b, a, q, epsilon=1e-6):
    # Normalize before projection: normalizing a near-zero projected vector would undo protection.
    x = x.to(a.dtype)
    x = (x / (x.square().mean(-1, keepdim=True).sqrt() + epsilon)).double()
    projected = a.double()
    if q.shape[1]:
        projected = projected - (projected @ q) @ q.T
    return ((x @ projected.T) @ b.double().T).to(a.dtype)


def install(expert, q):
    assert q.ndim == 2 and q.shape[0] == expert.d_in and q.dtype == torch.float64
    assert not q.requires_grad and q.device == next(expert.parameters()).device
    def residual(self, activation):
        b, a = self.factors()
        return response(activation, b, a, q, self.epsilon)
    expert.residual = types.MethodType(residual, expert)


def map_norm(r, before, after, q):
    b0, a0 = r.mm.factors({k: v.double() for k, v in before.items()})
    b1, a1 = r.mm.factors({k: v.double() for k, v in after.items()})
    q = q.to(a0)
    if q.shape[1]:
        a0 = a0 - (a0 @ q) @ q.T
        a1 = a1 - (a1 @ q) @ q.T
    b = torch.cat((b1, b0), dim=1)
    a = torch.cat((a1, -a0), dim=0)
    value = ((b.T @ b) * (a @ a.T)).sum()
    assert value >= -1e-9
    return float(value.clamp_min(0).sqrt())


def selfcheck():
    rng = torch.Generator().manual_seed(11820261025)
    q = torch.linalg.qr(torch.randn(12, 3, generator=rng, dtype=torch.float64)).Q
    x = torch.randn(4, 3, generator=rng, dtype=torch.float64) @ q.T
    a = torch.randn(2, 12, generator=rng, dtype=torch.float64, requires_grad=True)
    b = torch.randn(5, 2, generator=rng, dtype=torch.float64, requires_grad=True)
    assert response(x, b, a, q).abs().max() < 1e-12
    empty = q[:, :0]
    normalized = x / (x.square().mean(-1, keepdim=True).sqrt() + 1e-6)
    assert torch.equal(response(x, b, a, empty), (normalized @ a.T) @ b.T)
    source = torch.randn(4, 12, generator=rng, dtype=torch.float64)
    assert response(source, b, a, q).norm() > 1e-3
    assert torch.autograd.gradcheck(lambda aa, bb: response(source, bb, aa, q), (a, b))
    for _ in range(3):
        loss = response(source, b, a, q).square().mean()
        ga, gb = torch.autograd.grad(loss, (a, b))
        a = (a - .01 * ga).detach().requires_grad_(True)
        b = (b - .01 * gb).detach().requires_grad_(True)
        assert response(x, b, a, q).abs().max() < 1e-12
    new_a = a.detach() + .01 * torch.randn(a.shape, generator=rng, dtype=a.dtype)
    new_b = b.detach() + .01 * torch.randn(b.shape, generator=rng, dtype=b.dtype)
    mock = types.SimpleNamespace(mm=types.SimpleNamespace(factors=lambda s: (s['b'], s['a'])))
    measured = map_norm(mock, dict(b=b.detach(), a=a.detach()), dict(b=new_b, a=new_a), q)
    dense = (new_b @ (new_a - (new_a @ q) @ q.T) - b @ (a - (a @ q) @ q.T)).norm()
    assert abs(measured - float(dense)) < 1e-10
    return dict(status='PASS', annihilation_after_multiple_updates=True,
                no_projection_control_exact=True, nonzero_source_response=True,
                double_autograd_gradcheck=True, effective_map_norm_dense_parity=True)


if __name__ == '__main__':
    print(selfcheck())
