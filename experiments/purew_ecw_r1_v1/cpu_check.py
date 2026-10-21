"""Disposable CPU check of R and the unchanged production J candidate caller."""
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
from unittest.mock import patch

payload = json.loads(sys.stdin.readline())
root = Path(tempfile.mkdtemp(prefix='e.', dir='/data/bmw/tmp'))
tools = root / 'private/tools'
tools.mkdir(parents=True)
parent = Path('/data/bmw/Knowledge_editing/outputs/purew-ecw-consistent3-v1-20261015/run/private')
for path in (parent / 'tools').glob('*.py'):
    shutil.copy2(path, tools / path.name)
for name, source in payload['files'].items():
    (tools / name).write_text(source)
os.environ.update(RUN_ROOT=str(root), ACTION='CPU_R_PREPARATION', ARM='J', CUDA_VISIBLE_DEVICES='', TMPDIR=str(root))
sys.path[:0] = [str(tools), str(parent / 'source')]
(root / 'private/FROZEN_PROTOCOL.json').write_text(json.dumps(payload['protocol']))
import numpy as np
import torch
import core
import legacy_core
from core import Memory, affine_fallback
from cpu_check_legacy import exact

torch.set_num_threads(4)
torch.manual_seed(310)
started = time.monotonic()
gpu_forbidden = lambda *a, **kw: (_ for _ in ()).throw(AssertionError('CPU preparation cannot call CUDA'))
torch.cuda.init = gpu_forbidden
torch.cuda._lazy_init = gpu_forbidden
torch.cuda.is_available = lambda: False

def rejected(self, gradient, values, gradients):
    return torch.zeros(self.n, dtype=torch.float64), dict(solver_reason='FINITE_DUAL_REJECT', proposal_norm=0., projected_KKT_residual=.001)


# A known five-active KKT solution, not a second copy of the enumeration algorithm.
memory = Memory(torch.zeros(9).half(), 'J')
rows = [torch.eye(9, dtype=torch.float64)[j] for j in range(5)]
target = torch.tensor([.001] * 5 + [.002] * 4, dtype=torch.float64)
g = -target / .01 - sum((j + 1) * a for j, a in enumerate(rows))
d, certificate = affine_fallback(memory, g, [.009] * 5, rows)
assert certificate['status'] == 'CERTIFIED_AFFINE_OPTIMUM_FULL_PROXY_FEASIBLE', certificate
assert float((d - target).abs().max()) < 1e-12
assert certificate['active_sets_checked'] <= 32
assert certificate['certificate']['primal_dual_gap'] <= 1e-7
assert all(k <= .01 for k in (.009 + torch.stack(rows) @ d).tolist())

# A nonzero birth displacement must enter the full quadratic, without resetting q/v.
memory.consolidate(torch.eye(9, dtype=torch.float64), 'Base')
memory.advance(torch.full((9,), .001, dtype=torch.float64))
birth_state = memory.state()
d, c = affine_fallback(memory, g, [.009] * 5, rows)
assert c['status'].startswith('CERTIFIED_AFFINE') and exact(birth_state, memory.state())
oracle = .5 * float((torch.full((9,), .001, dtype=torch.float64) + d).square().sum())
assert abs(c['certificate']['quadratic_predicted_risks']['Base'] - oracle) < 1e-14

# Duplicated/zero constraints require no ridge, no source deletion, and no lost inequalities.
duplicate = [rows[0], rows[0], rows[1], torch.zeros(9), torch.zeros(9)]
d, c = affine_fallback(memory, torch.tensor([-4., -2.] + [0.] * 7), [.009, .009, .008, 0., 0.], duplicate)
assert c['status'] == 'CERTIFIED_AFFINE_OPTIMUM_FULL_PROXY_FEASIBLE', c
assert abs(float(d[0]) - .001) < 1e-12 and abs(float(d[1]) - .002) < 1e-12
near = [rows[0], rows[0] + 1e-10 * rows[1], rows[1], torch.zeros(9), torch.zeros(9)]
d, c = affine_fallback(memory, torch.tensor([-4., -2.] + [0.] * 7), [.009, .009, .008, 0., 0.], near)
assert not c['infeasibility_proved'] and c['active_sets_checked'] <= 32
if c['status'].startswith('CERTIFIED_AFFINE'):
    assert all(k <= .01 for k in (torch.tensor([.009, .009, .008, 0., 0.], dtype=torch.float64) + torch.stack(near) @ d).tolist())

# An affine relaxation optimum that violates a quadratic must remain unaccepted.
blocked = Memory(torch.zeros(2).half(), 'J')
blocked.consolidate(torch.tensor([[100., 0.]], dtype=torch.float64), 'Base')
before = blocked.state()
d, c = affine_fallback(blocked, torch.tensor([-1., 0.]), [0.] * 5, [torch.zeros(2)] * 5)
assert c['status'] == 'NOT_CERTIFIED' and not torch.count_nonzero(d)
assert not c['infeasibility_proved'] and exact(before, blocked.state())
zero, c = affine_fallback(memory, torch.zeros(9), [0.] * 5, [torch.zeros(9)] * 5)
assert c['status'] == 'CERTIFIED_PROXY_NO_DESCENT' and not torch.count_nonzero(zero)
for invalid in ([float('nan')] + [0.] * 4, [.0101] + [0.] * 4):
    try:
        affine_fallback(memory, g, invalid, rows)
    except ValueError:
        pass
    else:
        raise AssertionError('Invalid current KL admitted')

# Original PASS is untouched. Rejected legacy proposals alone trigger the new path.
with patch.object(legacy_core.Memory, 'solve_current', return_value=(target, {'solver_reason': 'PASS'})):
    with patch.object(core, 'affine_fallback', side_effect=AssertionError('Fallback on PASS')):
        passed, event = memory.solve_current(g, [.009] * 5, rows)
        assert passed is target and event == {'solver_reason': 'PASS'}
with patch.object(legacy_core.Memory, 'solve_current', rejected):
    before = memory.state()
    d, e = memory.solve_current(g, [.009] * 5, rows)
    assert e['solver_reason'] == 'PASS' and e['legacy_solver']['solver_reason'] == 'FINITE_DUAL_REJECT'
    assert exact(before, memory.state())
    stream = io.BytesIO()
    torch.save(memory.state(), stream)
    stream.seek(0)
    restored = Memory.restore(torch.load(stream, weights_only=True))
    d2, e2 = restored.solve_current(g, [.009] * 5, rows)
    assert isinstance(restored, Memory) and torch.equal(d, d2) and e == e2

# Exercise a full column-tile crossing without a real model or GB-sized factor.
wide = Memory(torch.zeros(core.TILE + 3).half(), 'J')
wide_rows = [torch.zeros(wide.n, dtype=torch.float64) for _ in range(5)]
for i, a in enumerate(wide_rows):
    a[-i - 1] = 1.
wide_g = -sum(wide_rows)
d, c = affine_fallback(wide, wide_g, [.009] * 5, wide_rows)
assert c['status'].startswith('CERTIFIED_AFFINE') and d.device.type == 'cpu'
assert torch.allclose(d[-5:], torch.full((5,), .001, dtype=torch.float64), atol=1e-12, rtol=0)

# The real production attempt() retains candidate hard gates, rollback and counts.
import worker
import medical
assert worker.Memory is Memory
weight = torch.ones((4, 5), dtype=torch.float16, requires_grad=True)
master = weight.detach().float().clone()
risk = Memory(weight, 'J')
risk.consolidate(torch.ones((1, 20), dtype=torch.float64) * .01, 'Base')
worker.loss = lambda *args: (weight.float().square().mean(), float(weight.detach().float().square().mean()), 0.)
worker.backward = lambda value, w: torch.autograd.grad(value, w)[0].float()
worker.b.call = lambda *args: None
worker.b.counts = lambda: None
worker.b.logits = lambda *args: (torch.stack([weight.float().flatten(), -weight.float().flatten()], dim=1), torch.zeros(20, dtype=torch.long))
anchor = worker.b.logits()[0].detach().double().softmax(-1)
initial_native, initial_master, initial_risk = weight.detach().clone(), master.clone(), risk.state()
initial_count = worker.b.COUNTS.get('committed_updates', 0)
with patch.object(legacy_core.Memory, 'solve_current', rejected):
    with patch.object(medical, 'direct_KLs', return_value=[.02] * 5):
        unchanged, event = worker.attempt(None, weight, master, risk, [], [None] * 5, [anchor] * 5, 1, hard_removed=True)
    assert not event['accepted'] and len(event['candidates']) == 8
    assert any('CURRENT_REMOVED_HARD_KL' in c['reasons'] for c in event['candidates'])
    assert all(set(c['reasons']) <= {'CURRENT_REMOVED_HARD_KL', 'ZERO_NATIVE'} for c in event['candidates'])
    assert torch.equal(weight, initial_native) and torch.equal(unchanged, initial_master) and exact(initial_risk, risk.state())
    assert worker.b.COUNTS.get('committed_updates', 0) == initial_count
    with patch.object(medical, 'direct_KLs', return_value=[0.] * 5):
        master, accepted = worker.attempt(None, weight, master, risk, [], [None] * 5, [anchor] * 5, 2, hard_removed=True)
    assert accepted['accepted'] and accepted['actual_native_norm'] > 0
    assert worker.b.COUNTS['committed_updates'] == initial_count + 1
    assert torch.equal(weight, master.half())

print(json.dumps(dict(status='PASS', known_five_active_KKT=True, duplicate_and_zero_rows=True,
    near_degenerate_no_false_infeasibility=True, quadratic_violation_rejected=True,
    zero_descent_not_written=True, invalid_KL_rejected=True, legacy_PASS_unchanged=True,
    restored_fallback_next_step_exact=True, nonzero_birth_q_v_oracle=True, production_worker_import=True,
    production_native_accept_and_true_KL_rollback=True, rejected_write_not_counted=True,
    CPU_column_tile_dimension=core.TILE + 3, full_parameter_dimension_not_measured=58720256,
    real_VLM_admission=False, CUDA_calls=0, real_model_calls=0, Judge_calls=0,
    CPU_seconds=time.monotonic() - started, disposable_root=str(root))))
