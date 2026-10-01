# DirectW-Evidence-v1 — self-review repair delivery

- `cpu_tests = PASS` (34 tests, real synthetic numerical calculations)
- `native_smoke_status = PENDING_REVALIDATION` for this repaired source
- `current_state = BLOCKED_NATIVE`
- `scientific_data_status = BLOCKED_DATA` for all five branches
- `additional_native_seconds = 0`, `additional_GGN_calls = 0`
- Original cumulative usage remains 144.393211 seconds and 46/48 GGN calls.
- `paid_judge_calls = 0`, `real_pilot_started = false`, `sequential_started = false`

The user authorized repairs of the three self-review findings. This delivery changes the independent `research/directw-evidence-v1` source and adds CPU regression evidence. It neither runs nor approves native model execution. The historical [B4 report](../directw_evidence_v1_phase_b_completion_20261001/PHASE_B_FINAL_REVIEW.md) and its source/receipt bindings are unchanged. That PASS remains evidence for the old dedicated mechanical harness, not native qualification of the repaired general editor.

## 1. FP32 solver and BF16 normal-forward acceptance

**Before:** the general `edit_one()` cloned the physical BF16 W as its solver point. The FP32 functional closure rejected it. Adding a cast at the call site allowed evaluation but bypassed `evaluate()`'s temporary `runtime.logits` substitution; a CPU reproduction returned ACCEPTED with zero ordinary forward calls.

**After:** differentiation and acceptance are explicit separate interfaces. `EditConfig.arithmetic_dtype=torch.float32` keeps the solver point, thresholds and drift arithmetic in FP32 while `MatrixRuntime.write()` retains physical BF16 deployment. `Constraint.normal_score()` and `ProtectionGroup.normal_logits/deployment_anchor/deployment_binding` are required for ordinary model evaluation. `MatrixRuntime.bind_normal_inputs()` freezes inputs for that physical path; no monkeypatching of the functional route remains. Functional and deployment protection anchors are separate, never silently substituted. The analytic categorical protection pullback is used in the FP32 solver, including nonzero protection terms away from the teacher. W_KEY geometry casts its fixed keys into solver arithmetic without changing the physical model.

Initial stopping, line-search merit and final acceptance all use deployed scores; a target satisfied only by the FP32 function cannot pass. Final protection is checked again. Missing physical callbacks fail before editing. Existing `build_constraints` callers must supply the additional physical support/endpoint callbacks; all in-repository callers have been updated.

**CPU evidence:** W_FT, W_EUCLIDEAN_QP, W_KEY_QP and W_FUNCTIONAL_QP each complete a real synthetic FP32 solve/BF16 transaction with measured ordinary BF16 callback calls, an actual accepted step and the selected-only state audit. Another test constructs a genuine FP32/BF16 output discrepancy and verifies that FP32-only target satisfaction is rejected. Existing visual endpoint/cross-constraint CPU tests remain passing. This is not a native VLM result or a medical evidence edit.

## 2. Actual execution configuration binding and cumulative caps

**Before:** the gate compared supplied digest strings with one another, without deriving them from the configuration consumed by the runner. A CPU reproduction changed the CG limit from 4 to 400, GGN allowance from 21 to 480 and damping from 100 to 200, while retaining the old receipt; admission still passed.

**After:** `execution_bindings(config)` derives canonical hashes of the entire configuration except its `bindings` self-reference, the actual protocol and the actual budget fields. Noncanonical/nonfinite JSON values are rejected. Admission compares these hashes to the separately trusted human receipt and verifies `approved_protocol_digest`. A copied receipt cannot admit changed damping, solver tolerance, call limits, prior consumption or other configuration fields.

For NATIVE_SMOKE the configuration must additionally include bound `budget_limits` (total native seconds, total GGN calls and maximum CG iterations), prior consumption and per-attempt remaining caps. Admission checks remaining time/calls against the cumulative totals and consistency of protocol/runner CG settings. Hash construction grants no permission. Source/model/dependency evidence remains subject to the existing independent content/runtime binding process; this repair does not pretend to re-hash large native checkpoints or authenticate a human merely from a JSON flag.

**CPU evidence:** changed CG/GGN/damping/tolerance/time/prior-consumption/protocol configurations are rejected before the test loader is invoked. Fresh *synthetic test-only* bindings also reject internally over-budget caps. Loader call count remains zero. No real approval file was created, upgraded or expanded. Historical run configs/receipts were preserved; they must not be silently rebound to this source.

## 3. Complete and verified exception rollback

**Before:** `edit_one.restore()` loaded only persistent `state_dict()` entries, then unconditionally set `rollback=true`. A nonpersistent buffer mutated from 0 to 2 survived an `EXCEPTION_ROLLED_BACK` result.

**After:** each edit snapshots all named buffers, including nonpersistent ones. The common `MatrixRuntime.restore_snapshot()` restores persistent state and buffers, checks selected-W equality, audits nonselected state, shape/dtype, buffers and hook counts, and only then reports success. `reset_single()` and the native mechanical harness use this same restoration path. An unrepairable registry/hook/state mutation yields `ROLLBACK_FAILED` and `rollback=false`, not a false success receipt. Known persistent state is restored even if the buffer registry itself has become invalid.

**CPU evidence:** a deliberate exception after a real BF16 candidate write mutates a nonpersistent buffer; both original W/persistent state and the buffer are restored to the edit-start values. The test checks actual tensor equality. A separate registry mutation fails restoration verification and must report ROLLBACK_FAILED. Existing ordinary-error and KeyboardInterrupt rollback tests also pass.

## Validation and remaining boundary

[CPU_REGRESSION_RESULTS.json](CPU_REGRESSION_RESULTS.json) records 34 passing tests (the original 30 plus four regression methods with multiple branch/adversarial subcases). Use the neutral stdin test command in the [package README](../../experiments/directw_evidence_v1/README.md). `SOURCE_BINDINGS.json` identifies the CPU-tested source relative to review parent `99488dc2ce53509b8caef58f342711f788e16fc6`.

The repaired general editor, explicit callbacks and revised native restoration call sites have **not** been exercised on the real VLM. No existing B4 PASS receipt is rewritten to cover these changes. The old budget has only two GGN calls left, which is insufficient for a complete native regression; further native execution needs an explicitly scoped and bound validation approval/budget. No fallback to CPU execution of a real large model, new Pilot approval, Judge call, automation or experiment continuation was created.

These repairs do not establish a useful deployed edit: the earlier BF16 rounding loss, slight mechanical score decrease, missing scientific data roles and exact-GGN scalability remain unresolved research constraints. The source is ready for review of the fixes; scientific experiments remain blocked.
