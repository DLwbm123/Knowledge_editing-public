# Direct-W native-value QP ablation (E3)

Frozen before native execution, 2026-10-01. User authority: autonomous follow-up experiment design and execution on physical GPU5, granted in this chat on 2026-10-01. The E1/E2 historical control has already completed and been published with 0/8 accepted edits.

## Hypothesis and single change

The original local inequality is `J_FP32 d + xi >= target - score_FP32(W)`. Test `J_FP32 d + xi >= target - score_BF16_native(W)` instead, recomputing the native value at each accepted iterate. This translates the local affine value while retaining the same FP32 derivative, functional GGN and preservation gradient. It is a distinct algorithm ablation, not a claim that the original protocol was implemented incorrectly.

Hypothesis: removing the current-value mismatch improves the count of final accepted BF16 edits under unchanged protection and edit budgets. Null: no improvement. A zero accepted count will not be relabeled as success on the strength of intermediate gains.

## Design and control

Reuse exactly all eight previously source-hash-selected development training cases, same references and order, clean Base reset for each. Control is the frozen complete E1/E2 result (FINAL_RESULTS.json); it is historical rather than concurrently randomized. Initial BF16 and FP32 scores must each match the corresponding control within 1e-6 or the experiment stops as a comparison-integrity failure. Native-functional BF16 parity must be within 1e-3. Permanent mechanical smoke rows remain outside the eight-case denominator.

No target or KL acceptance relaxation, no added steps and no new medical data. Unchanged: original matrix at layer21 down_proj, BF16 deployed model, full FP32 temporary functional arithmetic, tau100, nu10, trust0.1, drift0.3, reference KL0.001, target min(max(Base_BF16_score, Base_FP32_score)+0.1,-1e-4), three edit steps, CG16/rtol1e-4, one constraint, seven fixed backtracking factors, content-score tolerance1e-6, seed20261001, greedy8-token export check and Judge0. The sole changed solver field is constraint_value_mode=native_value. New logging records functional/native local scores and QP right-hand sides.

## Endpoints and decision rule

Primary: final accepted edits / 8 (requiring normal BF16 target and KL checks and clean native reload/generated-token parity where weights change). Secondary: intermediate accepted-step gains (explicitly not final edit success), terminal statuses, reference KL, proposed-versus-written norms, rounding fraction, GGN and wall cost. Report every case and compare to its historical control. Improvement means a strictly higher accepted count with all acceptance and reload checks intact. This tiny development comparison supports no population-level efficacy claim or statistical significance test.

## Validation, budget and stop rules

38 CPU tests passed, including offset-invariance of native-value anchoring on a real differentiable BF16 fixture and rejection of the offset problem by the unchanged default branch. These are synthetic tests. Native regression is pending at protocol freeze and executes first on the two reserved rows, checking parity, exact-GGN symmetry/PSD/repeatability and the modified real editor path. Regression implementation or solver errors stop the experiment; a scientifically rejected edit remains a negative outcome.

Only pro5000 physical GPU5, UUID GPU-4924bbd8-4082-3f4c-b4fb-78ca930122ca; minimum 66,000 MiB available at launch. New-run ceiling: 14,400 seconds, 4,096 exact GGN calls including regression, eight cases, at most three editing steps per case. Cumulative prior time 1850.5913290828466 seconds and 304 GGN are retained. Expected runtime is approximately 25–40 minutes based on E1/E2; this is an estimate, not a result. Storage ceiling20GiB, temporary teacher cache2GiB; the setup requires 28GiB free.

Stop on timeout/call ceiling, implementation exception, failed rollback, control-parity failure, unexpected non-selected-state mutation, or failed required clean export validation. Preserve all receipts and failure costs. Do not change settings mid-run. Accepted matrices are exported only for the mandatory clean native consumer, then removed with receipts; rejected edits restore Base. No CP/LoRA state is used.

## Limitations

EXPLORATORY_TRAINING_ONLY: references remain scope-unknown and training cases are reused for development. No independent evaluation, clinical accuracy, verified visual evidence pairs or history-protection efficacy claim. W_EVIDENCE scientific admission remains BLOCKED_DATA. Original raw medical inputs and tensors remain private. Publish sanitized results and the full denominator when complete.
