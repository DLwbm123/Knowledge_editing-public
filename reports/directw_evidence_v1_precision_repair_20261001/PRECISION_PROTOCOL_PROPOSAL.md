# Proposed Phase-B precision contract revision — unapproved

This proposal is for the remaining mechanical smoke only. It grants no Pilot, sequential edit, paid Judge, additional input, increased CG iteration count, relaxed CG tolerance or new resource budget.

## Evidence requiring the decision

The old FP32-coordinate wrapper cast W back to BF16 before evaluating the model. Its Jacobian products therefore retained BF16 arithmetic. A complete FP32 functional calculation using the same original checkpoint values now passes all three numerical checks on the same native input: GGN symmetry error 5.2154e-8, finite-difference relative error 0.006012, and four-iteration CG residual 5.8399e-5. Physical deployment parameters remain BF16 and unchanged.

However, complete FP32 evaluation differs from normal BF16 evaluation by up to 0.434829 logits. The existing 0.001 same-function parity threshold does not pass for this cross-precision comparison. We cannot report that it does, add a detached offset, use a straight-through surrogate without disclosure, change the threshold after seeing the result, or call the FP32 operator an exact derivative of the rounded BF16 execution map.

## Proposed explicit semantics

1. Keep the deployed native model, selected original matrix, checkpoint values and final exported W in BF16. No adapter, additional deployment parameter or changed layer.
2. Name the temporary editor function `FP32_FULL_FUNCTIONAL`: the same network and original parameter values are evaluated in FP32. Only selected W is a differentiation variable; other FP32 parameter copies are frozen temporary arithmetic state. No output-offset correction or rank/curvature approximation.
3. Describe its curvature accurately as the **exact JVP/VJP GGN of the FP32 functional model**. Do not call it exact differentiation of the rounded BF16 execution map.
4. Keep BF16 normal-vs-BF16-functional parity at the original tolerance; retain the observed cross-precision discrepancy as a separate, non-PASS field. Derivative/finite-difference checks apply to the explicitly named FP32 editor function. Bind each fixed teacher/anchor to its arithmetic dtype and never interchange FP32/BF16 anchors.
5. Every physical candidate write, content-score/protection check, rollback, parameter/buffer audit and clean editor-free reload uses actual BF16 normal forward. No FP32 predicted improvement counts as deployed acceptance. A candidate disappearing under BF16 rounding is rejected; no effect score is reported.
6. Reuse Q-inverse solutions only for the same frozen operator and RHS, verifying actual residuals on reuse. This avoids duplicated solves in the one-/two-constraint mechanical tests without increasing the four CG iterations or hiding GGN calls. Preserve the protection linear term and explicit slack.
7. Total remaining caps after B1/B2/B3: 1724.796243 seconds and 21 full GGN calls; one input remains the same reserved row, at most two constraints, one edit step, eight unjudged generation tokens and zero Judge calls. Keep a cumulative ledger; never reset it.
8. Native PASS requires all remaining mechanical checks and this explicit precision contract to be accepted. Otherwise report the exact remaining blocker and stop. Scientific FIT and Pilot remain blocked independently.

Approval status: **NOT_APPROVED**. This is a reviewable proposed numerical model, not an already approved native result or a request to start training.
