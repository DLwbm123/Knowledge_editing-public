# CPU evidence and outstanding native qualification

Final executed result: 20 PASS, 0 FAIL, 0 SKIPPED (seed 20260930).
These counts cover the 20 numbered CPU requirement groups, not native checks.
There are 14 separately listed PENDING_NATIVE_CHECK admission items.

| Numerical check | Observed maximum error | Criterion |
|---|---:|---:|
| Exact GGN matvec vs explicit Jacobian | 1.11e-16 | atol 1e-8 |
| Symmetry | 1.11e-16 | atol 1e-8 |
| Reference KL gradient | 3.60e-17 | norm <1e-8; Fisher nonzero |
| KL finite difference at epsilon=3e-4 | 1.41e-5 | <1e-4 and decreasing truncation error |
| CG vs dense SPD solve | 1.54e-15 | atol 1e-8 |
| Slack QP vs independent SciPy primal | 4.13e-11 | atol 1e-6 and KKT <1e-7 |

Other groups verify rejection/status semantics, real synthetic forwards, fixed
prefix positions, original state keys, BF16 swallowed updates, clean-process
free generation, source/cohort role rules, temporal history, phase gates,
intention-to-edit denominators and dependency-aware cleanup. No native model
forward was used for any of these checks. Full per-group metrics and timings
are in CPU_TEST_RESULTS.json. Debugging rounds and resolved failures are in
TEST_DEVELOPMENT_HISTORY.json; full failed traces remain private.
