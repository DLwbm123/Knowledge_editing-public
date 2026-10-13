"""Training-only KL and matched proposal; no inference module or parameters."""
import io
import torch
import torch.nn.functional as F


def preservation_kl(current, teacher):
    current, teacher = current.float(), teacher.detach().float()
    if current.shape != teacher.shape or current.ndim != 2 or min(current.shape) <= 0 or not torch.isfinite(current).all() or not torch.isfinite(teacher).all():
        raise ValueError('Invalid fixed teacher/current logits')
    return F.kl_div(F.log_softmax(current, -1), F.softmax(teacher, -1), reduction='batchmean')


def proposal(master, gradient, target):
    if master.dtype != torch.float32 or gradient.dtype != torch.float32 or master.shape != gradient.shape:
        raise ValueError('Require matching FP32 master and gradient')
    norm = gradient.norm()
    if not torch.isfinite(master).all() or not torch.isfinite(gradient).all() or not torch.isfinite(norm) or norm <= 0:
        raise ValueError('Nonfinite or zero direction; no replacement')
    if not 0 < target <= .050001:
        raise ValueError('Invalid frozen C proposal norm')
    delta = -gradient * (target / norm)
    after = master + delta
    if not torch.isfinite(after).all():
        raise ValueError('Nonfinite proposal')
    return after, after.half(), delta.norm()


def reference(targets, owner, step):
    if targets['shape'] != [100, 160] or len(targets['proposal_norms']) != 16000 or not 1 <= owner <= 100 or not 1 <= step <= 160:
        raise ValueError('Corrupt or out-of-scope source clock')
    return targets['proposal_norms'][(owner - 1) * 160 + step - 1]


def selfcheck():
    torch.manual_seed(11)
    teacher = torch.randn(7, 13, requires_grad=True)
    current = teacher.detach().clone().requires_grad_(True)
    loss = preservation_kl(current, teacher); loss.backward()
    assert abs(float(loss.detach())) < 1e-6 and current.grad.abs().max() < 1e-6
    assert teacher.grad is None
    altered = (teacher.detach() + .4 * torch.randn_like(teacher)).requires_grad_(True)
    loss = preservation_kl(altered, teacher); loss.backward()
    assert float(loss.detach()) > 0 and altered.grad.norm() > 0
    master = torch.ones(3, 5); g = torch.randn_like(master)
    target = .01 * min(g.norm().item(), 5.)
    after, _, n = proposal(master, g, target)
    assert torch.allclose(after, master - .01 * g * min(1., 5. / g.norm().item())) and abs(n.item() - target) < 1e-7
    zero, half, _ = proposal(master, torch.full_like(master, 1e-4), 1e-5)
    assert torch.equal(half, master.half()) and not torch.equal(zero, master)
    initial = torch.randn(3, 5).half().float()
    def advance(value, count):
        native = None
        for _ in range(count):
            value, native, n = proposal(value, torch.randn_like(value), .002)
            assert abs(n.item() - .002) < 1e-8
        return value, native
    state = torch.get_rng_state()
    uninterrupted = advance(initial.clone(), 12)
    torch.set_rng_state(state)
    partial = advance(initial.clone(), 6)
    memory = io.BytesIO()
    torch.save(dict(master=partial[0], native=partial[1], rng=torch.get_rng_state(), step=6), memory)
    memory.seek(0); saved = torch.load(memory, weights_only=True)
    assert torch.equal(saved['native'], saved['master'].half())
    torch.set_rng_state(saved['rng']); resumed = advance(saved['master'], 12 - saved['step'])
    assert all(torch.equal(a, b) for a, b in zip(uninterrupted, resumed))
    targets = dict(shape=[100, 160], proposal_norms=[.002] * 16000)
    assert reference(targets, 100, 160) == .002
    for function in (lambda: proposal(master, torch.zeros_like(master), .01),
                     lambda: proposal(master, torch.full_like(master, float('nan')), .01),
                     lambda: proposal(master, torch.full_like(master, 1e30), .01),
                     lambda: preservation_kl(torch.empty(0, 3), torch.empty(0, 3)),
                     lambda: reference(targets, 101, 1)):
        try:
            function()
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid state accepted')
    return dict(status='PASS', teacher_detached=True, KL_moves_off_Base=True,
                matched_proposal=True, zero_KL_matches_SGD=True,
                small_native_step_can_round_to_zero=True, exact_toy_master_native_rng_resume=True,
                invalid_direction_and_clock_rejected=True, real_VLM=False,
                full_execution_recovery_checked=False, GPU_updates=0)
