"""Editing-time C/U/H directions and shared norms; no inference module."""
import math
import torch

ARMS = ('C', 'U', 'H')
NORM_RTOL = 5e-4
NORM_ATOL = 1e-6


def direction(gradient, arm, initial_row_basis):
    assert arm in ARMS and torch.isfinite(gradient).all()
    if arm == 'C':
        return gradient - (gradient @ initial_row_basis) @ initial_row_basis.T
    return gradient


def reference_step(projected_gradient):
    norm = float(projected_gradient.norm())
    assert math.isfinite(norm)
    return shared_step(projected_gradient, min(.01 * norm, .05))


def shared_step(gradient, target):
    assert math.isfinite(target) and 0 <= target <= .0501
    norm = float(gradient.norm())
    assert math.isfinite(norm)
    if target == 0:
        return torch.zeros_like(gradient)
    if norm == 0:
        raise ValueError('Positive shared norm with zero direction: stop, do not silently skip')
    delta = gradient * (-target / norm)
    assert torch.isfinite(delta).all()
    return delta


def apply_step(weight, delta, target):
    assert weight.dtype == torch.float32 or weight.dtype == torch.float64
    assert weight.shape == delta.shape and weight.dtype == delta.dtype and weight.device == delta.device
    updated = weight + delta
    actual = float((updated - weight).norm())
    if not math.isclose(actual, target, rel_tol=NORM_RTOL, abs_tol=NORM_ATOL):
        raise ValueError(f'Actual update norm {actual} differs from shared target {target}')
    assert torch.isfinite(updated).all()
    return updated, actual


def history(rows, current_owner, step):
    assert current_owner >= 1 and step >= 0
    past = [r for r in rows if r['owner'] < current_owner and r['audit_role'] in ('NATIVE', 'GFIT')]
    if current_owner == 1:
        assert not past
        return ()
    assert {r['owner'] for r in past} == set(range(1, current_owner))
    assert all(r['source_admission'] == 'PASS' for r in past)
    native = [r for r in past if r['audit_role'] == 'NATIVE']
    anchors = [r for r in past if r['audit_role'] == 'GFIT']
    assert len(native) == current_owner - 1 and len(anchors) == 2 * (current_owner - 1)
    native.sort(key=lambda r: (r['owner'], r['query_id']))
    anchors.sort(key=lambda r: (r['owner'], r['query_id']))
    return native[step % len(native)], anchors[step % len(anchors)]


def selfcheck():
    torch.manual_seed(19)
    for dtype in (torch.float64, torch.float32):
        base = torch.randn(3, 9, dtype=dtype) * .01
        q = torch.linalg.qr(base.T, mode='reduced').Q
        grad = torch.randn_like(base)
        c = direction(grad, 'C', q)
        assert float((c @ base.T).norm()) < (1e-12 if dtype == torch.float64 else 1e-7)
        proposal = reference_step(c)
        _, target = apply_step(base, proposal, float(proposal.norm()))
        for arm in ARMS:
            changed = grad + .5 * torch.randn_like(grad) if arm == 'H' else grad
            d = direction(changed, arm, q)
            updated, actual = apply_step(base, shared_step(d, target), target)
            assert math.isclose(actual, target, rel_tol=NORM_RTOL, abs_tol=NORM_ATOL)
            assert (d * (updated - base)).sum() < 0
        assert torch.equal(shared_step(grad, 0), torch.zeros_like(grad))
        try:
            shared_step(torch.zeros_like(grad), target)
        except ValueError:
            pass
        else:
            raise AssertionError('Zero direction must stop')
    rows = [dict(owner=o, query_id=f'{o}-{i}', audit_role='NATIVE' if i == 0 else 'GFIT', source_admission='PASS') for o in range(1, 4) for i in range(3)]
    rows += [dict(owner=1, query_id='never-use', audit_role='T2G')]
    assert history(rows, 1, 0) == ()
    selected = {r['query_id'] for s in range(12) for r in history(rows, 3, s)}
    assert selected == {f'{o}-{i}' for o in (1, 2) for i in range(3)}
    bad = [dict(r, source_admission='PENDING') if r['query_id'] == '1-1' else r for r in rows]
    try:
        history(bad, 3, 0)
    except AssertionError:
        pass
    else:
        raise AssertionError('Unadmitted history cannot train')
    return dict(status='PASS', test_scope='CPU_SYNTHETIC_ONLY', common_reference_norm=True, actual_write_norm_checked=True, fixed_full_projection=True, history_complete_deterministic=True, evaluation_roles_excluded=True, unadmitted_history_rejected=True, zero_direction_stops=True, real_VLM_admission='NOT_RUN')
