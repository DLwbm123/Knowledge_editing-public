"""A fixed response-gradient nullspace and a matched-write diagnostic."""
import torch

RANK_RTOL = 1e-6


def project(rows, delta):
    assert rows.ndim == 2 and delta.ndim == 1 and rows.shape[1] == delta.numel()
    assert torch.isfinite(rows).all() and torch.isfinite(delta).all()
    rows, delta = rows.double(), delta.double()
    norms = rows.norm(dim=1)
    unit = rows[norms > 0] / norms[norms > 0, None]
    if len(unit):
        _, values, right = torch.linalg.svd(unit, full_matrices=False)
        rank = int((values > RANK_RTOL * values[0]).sum())
        basis = right[:rank]
        result = delta - basis.T @ (basis @ delta)
        assert (basis @ result).norm() <= 1e-9 * delta.norm() + 1e-14
    else:
        values, basis, rank, result = torch.empty(0), rows[:0], 0, delta.clone()
    return result, dict(rank=rank, zero_rows=int((norms == 0).sum()),
        singular_values=values.tolist(), retained_parameter_norm=float(result.norm()),
        original_parameter_norm=float(delta.norm()),
        normalized_predicted_before=float((unit @ delta).norm()),
        normalized_predicted_after=float((unit @ result).norm()))


def selfcheck():
    rows = torch.tensor([[1., 0., 0.], [2., 0., 0.], [0., 0., 0.]])
    value, audit = project(rows, torch.tensor([3., 4., 5.]))
    assert audit['rank'] == 1 and audit['zero_rows'] == 1
    assert torch.allclose(value, torch.tensor([0., 4., 5.], dtype=torch.float64))
    value, _ = project(torch.eye(3), torch.ones(3))
    assert value.norm() < 1e-12
    value, audit = project(torch.zeros(2, 3), torch.ones(3))
    assert torch.equal(value, torch.ones(3, dtype=torch.float64)) and audit['rank'] == 0
    return dict(status='PASS', duplicate_and_zero_rows=True, full_rank=True,
                empty_constraint=True, RANK_RTOL=RANK_RTOL)
