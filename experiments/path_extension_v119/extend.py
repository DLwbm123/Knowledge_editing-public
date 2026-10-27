"""One fixed union of an existing path span and a uniform prompt-format path."""
import torch


def basis(old, vectors):
    assert old.ndim == 2 and old.dtype == torch.float64 and not old.requires_grad
    old = old.cpu()
    new = torch.cat(vectors).double().cpu()
    assert new.ndim == 2 and new.shape[1] == old.shape[0] and 0 < len(new) <= 2048
    assert torch.isfinite(old).all() and torch.isfinite(new).all() and (new.norm(dim=1) > 0).all()
    new = new / new.norm(dim=1, keepdim=True)
    rows = torch.cat((old.T, new))
    _, singular, right = torch.linalg.svd(rows, full_matrices=False)
    rank = int((singular > singular[0] * 1e-10).sum())
    assert old.shape[1] <= rank <= 512, 'Do not truncate the protected span'
    q = right[:rank].T.contiguous()
    old_error = float((old.T - (old.T @ q) @ q.T).norm(dim=1).max())
    new_error = float((new - (new @ q) @ q.T).norm(dim=1).max())
    assert max(old_error, new_error) <= 1e-8
    assert torch.allclose(q.T @ q, torch.eye(rank, dtype=q.dtype), atol=1e-10, rtol=0)
    return q, dict(old_rank=old.shape[1], new_rank=rank, new_path_positions=len(new),
        old_span_error=old_error, new_path_span_error=new_error,
        singular_values=singular.tolist(), relative_svd_cutoff=1e-10,
        basis_bytes=q.numel() * q.element_size())


def selfcheck():
    rng = torch.Generator().manual_seed(11920261025)
    old = torch.linalg.qr(torch.randn(12, 2, generator=rng, dtype=torch.float64)).Q
    new = torch.randn(2, 12, generator=rng, dtype=torch.float64)
    q, audit = basis(old, [new])
    assert audit['new_rank'] == 4 and audit['old_rank'] == 2
    assert (old.T - (old.T @ q) @ q.T).norm() < 1e-12
    assert (new - (new @ q) @ q.T).norm() < 1e-12
    source = torch.randn(12, generator=rng, dtype=torch.float64)
    assert (source - (source @ q) @ q.T).norm() > 1e-3
    return dict(status='PASS', both_spans_preserved=True, nonzero_complement=True)
