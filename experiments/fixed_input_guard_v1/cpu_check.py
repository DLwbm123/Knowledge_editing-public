"""Synthetic CPU checks only; run via a neutral Python stdin entry."""
import io
import json
import time
import torch
from guard import InputGuard, CAP_BYTES, ATOL, RTOL, select_inputs


def run():
    start = time.monotonic()
    torch.set_num_threads(2)
    torch.manual_seed(1616)
    def forbidden(*args, **kwargs):
        raise AssertionError('GPU initialization forbidden')
    torch.cuda.init = torch.cuda._lazy_init = forbidden
    result = {'scope': 'CPU_SYNTHETIC_ONLY', 'torch': torch.__version__,
              'tolerance': {'absolute': ATOL, 'relative': RTOL}, 'checks': {}}
    checks = result['checks']
    views = [torch.arange(101 * 128).reshape(101, 128).float() + i for i in range(5)]
    picked = select_inputs(views)
    assert picked.shape == (32, 128) and torch.equal(picked, select_inputs(views))
    assert torch.equal(picked[0], views[0][0]) and torch.equal(picked[6], views[0][-1])
    zero = InputGuard(torch.zeros(32, 128))
    zero.admit(torch.zeros(32, 128), successful=True)
    assert zero.basis().shape == (128, 0)
    extreme = InputGuard(torch.full((32, 128), 1e300, dtype=torch.float64))
    extreme.admit(torch.full((32, 128), 1e300, dtype=torch.float64), successful=True)
    assert torch.isfinite(extreme.sketch).all()
    assert int(extreme.counts[1]) == 0
    repeated = torch.randn(32, 128)
    duplicate = InputGuard(repeated)
    duplicate.admit(repeated, successful=True)
    assert int(duplicate.counts[1]) == 0
    checks['fixed_token_selection_zero_and_extreme_inputs'] = True
    guard = InputGuard(torch.randn(32, 128))
    initial_bytes = guard.check_storage()
    snapshots = []
    covariance = torch.zeros(128, 128, dtype=torch.float64)
    q0 = guard.base.double()
    base_original = guard.base.clone()
    for edit in range(1, 2001):
        x = torch.randn(1, 128)
        a = x.double() / x.double().norm()
        a -= (a @ q0) @ q0.T
        covariance += a.T @ a
        guard.admit(x, successful=True)
        assert guard.check_storage() == initial_bytes
        assert torch.equal(guard.base, base_original)
        if edit in (1, 146, 1000, 2000):
            stream = io.BytesIO()
            torch.save(guard.state(), stream)
            snapshots.append({'edits': edit, 'tensor_bytes': initial_bytes, 'checkpoint_bytes': stream.tell()})
    assert len({s['checkpoint_bytes'] for s in snapshots}) == 1
    eigen = torch.linalg.eigvalsh(covariance - guard.sketch.double().T @ guard.sketch.double())
    assert float(eigen.min()) > -0.003
    assert float(eigen.max()) <= float(guard.discarded) + 0.003
    checks['constant_storage_2000_edits'] = snapshots
    checks['fd_covariance_oracle'] = {'minimum': float(eigen.min()), 'maximum': float(eigen.max()),
                                      'discarded_bound': float(guard.discarded)}
    restored = InputGuard.restore(guard.state())
    native = torch.randn(16, 128)
    update = torch.randn_like(native) * .1
    w1, w2 = native.clone(), native.clone()
    r1, r2 = guard.write(w1, update), restored.write(w2, update)
    assert r1 == r2 and torch.equal(w1, w2) and r1['accepted']
    checks['restore_write_equivalence'] = r1
    rejected_state = guard.state()
    assert not guard.admit(torch.randn(32, 128), successful=False)
    assert all(torch.equal(rejected_state[k], guard.state()[k]) for k in rejected_state)
    checks['failed_edit_not_admitted'] = True
    # Explicitly exercise changing history subspaces over a write trajectory.
    trajectory = InputGuard(torch.randn(32, 128))
    weight = torch.randn(16, 128)
    max_ratio = 0.
    for _ in range(146):
        old, q = weight.clone(), trajectory.basis()
        receipt = trajectory.write(weight, torch.randn_like(weight) * .1)
        assert receipt['accepted']
        actual = weight.double() - old.double()
        independent = float((actual @ q).norm())
        assert independent <= ATOL + RTOL * float(actual.norm())
        max_ratio = max(max_ratio, independent / receipt['limit'])
        trajectory.admit(torch.randn(1, 128), successful=True)
    checks['146_native_fp32_writes'] = {'accepted': 146, 'max_residual_over_limit': max_ratio}
    checks['native_dtypes'] = {}
    aligned = InputGuard(torch.eye(128)[:32])
    for i in (32, 64):
        aligned.admit(torch.eye(128)[i:i+32], successful=True)
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        w = torch.randn(16, 128).to(dtype)
        old = w.clone()
        passed = aligned.write(w, torch.randn(16, 128) * .1)
        assert passed['accepted'] and not torch.equal(w, old)
        actual = w.double() - old.double()
        assert float((actual @ aligned.basis()).norm()) <= passed['limit']
        w = torch.randn(16, 128).to(dtype)
        old = w.clone()
        general = guard.write(w, torch.randn(16, 128) * .1)
        if dtype == torch.float32:
            assert general['accepted']
        else:
            assert not general['accepted'] and torch.equal(w, old)
        checks['native_dtypes'][str(dtype)] = {'aligned_nonzero_write': passed, 'general_subspace': general}
    old = w.clone()
    assert not guard.write(w, torch.zeros_like(w))['accepted']
    assert torch.equal(w, old)
    bad = torch.full_like(w.float(), float('nan'))
    try:
        guard.write(w, bad)
    except ValueError:
        pass
    else:
        raise AssertionError('Nonfinite update admitted')
    assert torch.equal(w, old)
    bad_state = guard.state()
    bad_state['history'][0, 0] = 100
    try:
        InputGuard.restore(bad_state)
    except ValueError:
        pass
    else:
        raise AssertionError('Corrupt checkpoint admitted')
    checks['invalid_zero_corrupt_rejection'] = True
    half = torch.ones(2, 128, dtype=torch.float16)
    old = half.clone()
    overflow = aligned.write(half, torch.full((2, 128), 1e10))
    assert not overflow['accepted'] and overflow['reason'] == 'NONFINITE_NATIVE'
    assert torch.equal(old, half)
    checks['native_overflow_no_mutation'] = True
    # Real input width; no model, data, tokenizer, or GPU involved.
    large = InputGuard(torch.eye(32, 14336))
    initial = large.check_storage()
    for _ in range(5):
        large.admit(torch.randn(32, 14336), successful=True)
    assert large.check_storage() == initial < CAP_BYTES
    stream = io.BytesIO()
    torch.save(large.state(), stream)
    assert stream.tell() < CAP_BYTES
    large_w = torch.randn(4096, 14336, dtype=torch.float16)
    # Coordinate-aligned native write proves the full shape can certify a nonzero write.
    coordinate = InputGuard(torch.eye(32, 14336))
    proposal = torch.zeros_like(large_w)
    proposal[:, 32] = .125
    full_receipt = coordinate.write(large_w, proposal)
    assert full_receipt['accepted'] and full_receipt['update_norm'] > 0
    proposal.normal_().mul_(.1)
    before = large_w.clone()
    general_half = large.write(large_w, proposal)
    assert not general_half['accepted'] and torch.equal(large_w, before)
    del before
    full_float = large.write(large_w.float(), proposal)
    assert full_float['accepted']
    checks['full_dimensions'] = {'shape': [4096, 14336], 'tensor_bytes': initial,
                                 'checkpoint_bytes': stream.tell(), 'native_fp16_write': full_receipt,
                                 'general_fp16_rejected': general_half, 'general_fp32_accepted': full_float,
                                 'sketch_compressed': float(large.discarded) > 0}
    assert checks['full_dimensions']['sketch_compressed']
    result.update(status='PASS', elapsed_seconds=time.monotonic()-start,
                  practical_low_precision_readiness='BLOCKED_GENERAL_SUBSPACE_QUANTIZATION',
                  experiment_started=False)
    return result


if __name__ == '__main__':
    print(json.dumps(run(), indent=2))
