"""Full-shape native-write acceptance and refusal checks, with no model calls."""
import io
import math
import time
import torch
from guard import InputGuard, ATOL, RTOL, CAP_BYTES, MAX_RELATIVE_DISTORTION


def verify(guard, before, after, proposal, receipt):
    q = guard.basis()
    if not receipt['accepted']:
        assert torch.equal(before, after), 'Rejected write changed native weight'
        return
    assert before.dtype == after.dtype and not torch.equal(before, after)
    # Independent post-return computation; do not trust the writer's certificate.
    actual = after.double()-before.double()
    intended = proposal.double()-(proposal.double() @ q) @ q.T
    norm = float(actual.norm())
    residual = float((actual @ q).norm())
    distortion = float((actual-intended).norm()/intended.norm())
    assert math.isfinite(residual) and norm > 0
    assert residual <= ATOL + RTOL * norm
    assert distortion <= MAX_RELATIVE_DISTORTION
    receipt['independent_residual'] = residual
    receipt['independent_distortion'] = distortion
    receipt['native_dtype_after'] = str(after.dtype)
    receipt['nonzero_native_elements'] = int(torch.count_nonzero(actual))


def check_refusals():
    torch.set_num_threads(2)
    aligned = InputGuard(torch.eye(128)[:32])
    before = torch.ones(4, 128, dtype=torch.bfloat16)
    weight = before.clone()
    proposal = torch.zeros(4, 128)
    proposal[:, 32:] = .005
    distortion = aligned.write(weight, proposal)
    assert distortion['reason'] == 'DIRECTION_DISTORTION_REJECTED'
    assert distortion['residual'] == 0 and distortion['update_norm'] > 0
    verify(aligned, before, weight, proposal, distortion)
    torch.manual_seed(73)
    dense = InputGuard(torch.randn(32, 14336))
    for _ in range(2):
        dense.admit(torch.randn(32, 14336), successful=True)
    before = torch.ones(4, 14336, dtype=torch.bfloat16)
    weight = before.clone()
    proposal = torch.randn(4, 14336)*.01
    coarse = dense.write(weight, proposal)
    assert not coarse['accepted'] and coarse['repair']['corrected_rows'] > 0
    verify(dense, before, weight, proposal, coarse)
    return dict(direction_distortion=distortion, no_fine_native_coordinates=coarse)


def run():
    start = time.monotonic()
    torch.set_num_threads(2)
    def forbidden(*args, **kwargs):
        raise AssertionError('No GPU initialization in CPU check')
    torch.cuda.init = torch.cuda._lazy_init = forbidden
    cases = []
    for seed, scale, ratio in ((1616, 1., .1), (2026, .02, .05)):
        torch.manual_seed(seed)
        guard = InputGuard(torch.randn(32, 14336))
        for _ in range(5):
            guard.admit(torch.randn(32, 14336), successful=True)
        assert guard.basis().shape == (14336, 96)
        assert float((guard.basis().abs() > 1e-8).float().mean()) > .999
        state = guard.state()
        for dtype in (torch.float16, torch.bfloat16):
            weight = (torch.randn(4096, 14336)*scale).to(dtype)
            before = weight.clone()
            proposal = torch.randn(4096, 14336)*(scale*ratio)
            started = time.monotonic()
            receipt = guard.write(weight, proposal)
            receipt.update(seed=seed, weight_scale=scale, update_ratio=ratio,
                           seconds=time.monotonic()-started)
            verify(guard, before, weight, proposal, receipt)
            assert receipt['accepted'], receipt
            assert receipt['repair']['corrected_rows'] > 0
            assert all(torch.equal(value, guard.state()[key]) for key, value in state.items())
            cases.append(receipt)
            print('Full-shape case passed:', seed, dtype, flush=True)
            del weight, before, proposal
        checkpoint = io.BytesIO()
        torch.save(guard.state(), checkpoint)
        assert guard.check_storage() == 12845096 and checkpoint.tell() < CAP_BYTES
    # Coarse lattice + tiny requested change must not be passed by inflating the update.
    torch.manual_seed(71)
    tiny = InputGuard(torch.randn(32, 128))
    tiny.admit(torch.randn(32, 128), successful=True)
    before = torch.ones(4, 128, dtype=torch.bfloat16)
    weight = before.clone()
    proposal = torch.randn(4, 128)*1e-5
    rejected = tiny.write(weight, proposal)
    assert not rejected['accepted']
    verify(tiny, before, weight, proposal, rejected)
    # Restore must reproduce repair decisions, and repair must be deterministic.
    torch.manual_seed(72)
    restored = InputGuard.restore(guard.state())
    before = torch.randn(8, 14336).to(torch.bfloat16)
    proposal = torch.randn(8, 14336)*.1
    a, b = before.clone(), before.clone()
    ra, rb = guard.write(a, proposal), restored.write(b, proposal)
    assert ra == rb and torch.equal(a, b) and ra['accepted']
    verify(restored, before, b, proposal, rb)
    return dict(status='PASS', scope='CPU_SYNTHETIC_NO_MODEL', shape=[4096, 14336],
                full_shape_cases=cases, tiny_update_rejected=rejected,
                restored_repair=rb, persistent_tensor_bytes=guard.check_storage(),
                checkpoint_bytes=checkpoint.tell(), elapsed_seconds=time.monotonic()-start,
                projection_atol=ATOL, projection_rtol=RTOL,
                max_relative_direction_distortion=MAX_RELATIVE_DISTORTION,
                additional_refusals=check_refusals(),
                experiment_started=False, real_model_readiness='NOT_VALIDATED')
