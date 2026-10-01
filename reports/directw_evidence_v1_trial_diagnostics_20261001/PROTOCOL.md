# E5: one-step native numerical decomposition

Frozen before native execution on2026-10-01 under the user's autonomous experimental-design delegation. E4 completed0/8 accepted, all backtracking failures before12 steps, with exact E3 first-three-step trajectories. E5 measures a mechanism; it does not enlarge steps or loosen acceptance.

## Design

All eight fixed development cases and their original references, order, Base, editable matrix, FP32 functional arithmetic and BF16 deployment are retained. Each case runs only its first QP step with the E3/E4 native-value gap and unchanged solver, target and protection settings. Record every candidate actually tried by the existing seven-factor line search until acceptance or exhaustion. This is an on-policy candidate sample, not an exhaustive response surface. Mechanical smoke rows remain separate. Initial BF16/FP32 scores must match historical controls within1e-6.

The optional diagnostics record additional no-gradient forwards on exactly the requested FP32 matrix and its physically written BF16-rounded matrix represented in FP32, plus a repeated normal BF16 forward. They must not change direction, merit, rejection/acceptance, restoration, or GGN call count. The actual original matrix is the only persistent edited parameter; no CP/LoRA state is loaded.

## Frozen numerical endpoints

For each attempted candidate let F0 be the pre-step full-FP32 functional score, Fq the FP32 score at the requested unrounded weight, Fr the FP32 score at the rounded deployed weight, Lq the affine FP32 prediction at the requested weight, and B0/Br the corresponding normal BF16 scores before/after deployment. Record:

1. affine predicted gain: Lq-F0;
2. local nonlinearity residual: Fq-Lq;
3. weight-rounding contribution in FP32 arithmetic: Fr-Fq;
4. deployment-arithmetic gain discrepancy: (Br-B0)-(Fr-F0).

Their sum must reconstruct Br-B0 (reported rounding residual tolerance1e-6). Also record affine prediction at the rounded weight; FP32 reference KL at both requested and rounded weights; the actual normal BF16 reference KL; requested/actual delta norms, zeroed fraction and native exact-repeat flag. Report all cases and attempts, ranges and per-case summaries. These are numerical decompositions, not independent causal effects or medical-efficacy estimates. No success threshold is invented for the relative sizes of these components. If opposing signs or severe arithmetic discrepancies explain rejected candidates, use that evidence to propose a distinct next hypothesis, not silently change this run.

Secondary: compare first-step attempt fields excluding new diagnostics to the complete E3/E4 control. Unexpected differences invalidate the assumption that instrumentation is passive and require diagnosis. One-step terminal failure is expected to remain a negative edit; intermediate improvement is not final acceptance. Accepted changed weights, if any, still require clean native reload/generated-token parity before counting success.

## Invariants and ceilings

Unchanged tau100, nu10, trust0.1, drift0.3, reference KL0.001, target=min(max(BaseBF16,BaseFP32)+0.1,-1e-4), CG16/rtol1e-4, tolerance1e-6, fixed seven backtrack factors, seed20261001, original layer21 down_proj, Judge0. Scientific case max_steps1; native regression remains one mechanical step with three exact-GGN probes. Diagnostic forwards are real model forwards and add runtime, but are not GGN matvecs.

New-run ceiling3600seconds/256GGN, eight cases, one step/case; storage20GiB, teacher cache2GiB and setup free-space requirement28GiB. Launch only pro5000 physicalGPU5 UUID GPU-4924bbd8-4082-3f4c-b4fb-78ca930122ca with66,000MiB free. Preserve prior952GGN and5437.3987826277735seconds. Expected time15–30minutes using prior one-step throughput plus additional forwards; this estimate may be wrong.

39 CPU tests passed, including exact equality of edit outcome, written weights, original attempt fields and GGN calls with diagnostics on/off in a real differentiable BF16 fixture. CPU coverage is synthetic. Native regression must exercise instrumentation first. Stop and restore on nonrepeatable native candidate forwards, original-state audit/rollback failure, comparison-integrity mismatch, native implementation error, failed required export parity, or time/GGN ceiling. No same-version blind retry. Keep all failures and cost in the ledger.

## Boundaries

Training-support diagnostics only, reused development cases, scope-unknown references and no independent calibration/evaluation. Verified visual pairs remain missing and scientific W_EVIDENCE admission remains BLOCKED_DATA. No invented medical negatives or clinical claims. Publish sanitized scalar/aggregate numerical evidence and source/config bindings; raw QA, images, tokens and native tensors remain private. Temporary edited matrices, if produced, are removed only after their final required consumer and parity receipt.
