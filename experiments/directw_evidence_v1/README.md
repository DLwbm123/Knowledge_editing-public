# DirectW-Evidence-v1

Latest: [self-review repairs and CPU-only validation](../../reports/directw_evidence_v1_review_fixes_20261001/REVIEW_FIXES.md).
The repaired source requires fresh native qualification; the historical smoke PASS
does not qualify this new editor path or authorize Pilot.

Stage A: [external review packet](../../reports/directw_evidence_v1_stage_a_20260930/REVIEW_PACKET.md).
Native entry points remain approval-gated; no old CP/LoRA editor is imported.

Reproduce the CPU tests with the existing Python environment containing PyTorch
2.6.0, NumPy and SciPy 1.16.0:

```sh
python - <<'PY'
import unittest
from experiments.directw_evidence_v1.tests import Checks
from experiments.directw_evidence_v1.phase_b_tests import PhaseBChecks
suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromTestCase(c)
                          for c in (Checks, PhaseBChecks))
raise SystemExit(not unittest.TextTestRunner().run(suite).wasSuccessful())
PY
```

The stdin entry keeps process argv neutral. No real model or paid service is
loaded. There are 34 CPU tests including 20 original requirement groups; their numerical operations are real
but their inputs and models are synthetic. `gate.cli` rejects native stages
before any loader. A later human-authorized native qualification may call the
generic editor on an ordinary clean model only after independently verified
phase approval and complete scientific/data/resource bindings.

Private run outputs go under this independent worktree's ignored `outputs/`.
Only code, draft configuration and deidentified aggregate reports are committed.

The editor requires separate functional and physical callbacks:
`Constraint.score(w)` differentiates; `Constraint.normal_score()` evaluates the
ordinary deployed model. `ProtectionGroup` likewise binds `logits/anchor/binding`
and `normal_logits/deployment_anchor/deployment_binding` separately. Use
`runtime.bind_inputs(..., arithmetic_dtype=torch.float32)` for the temporary
solver function and `runtime.bind_normal_inputs(...)` for physical evaluation.
For BF16 deployment, set `EditConfig.arithmetic_dtype=torch.float32`; do not cast
solver W back to BF16 in the functional callback. Score reductions and KL may
use FP32 on physical BF16 logits. `build_constraints` support tuples now include
the normal-score callback as their fourth field; visual pairs also require
`normal_plus_margin` and `normal_minus_margin`.

`gate.execution_bindings(config)` computes canonical config/protocol/budget
hashes without granting permission. The full configuration (excluding only
`bindings`) is included. Set `approved_protocol_digest` to the reviewed protocol
digest before computing config binding. A separately trusted human receipt must
match those hashes. Native smoke additionally requires bound `budget_limits`
(total native seconds, total GGN calls, maximum CG iterations), prior consumption,
and remaining per-attempt caps. Historical receipts are not upgraded or reused
for the repaired source. There is no authorized native-run configuration in this
repair delivery.
