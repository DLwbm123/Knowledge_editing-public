# E4: one finite step-budget ablation

Frozen before execution on2026-10-01 under the user's explicit delegation to design and run follow-up Direct-W experiments on pro5000 GPU5.

## Question and hypothesis

E3 native-value QP anchoring produced0/8 final accepted edits. Three cases reached the three-step ceiling with positive intermediate gains; case6 reached0.092556 of the required0.1. Test whether the uniform three-step budget, rather than an immediately rejected direction, limits final acceptance. Change only the scientific-case maximum edit steps from3 to12 (fourfold). Do not change the target, solver, model, data or protection limits.

Hypothesis: at least one additional final accepted case under the same BF16 target and protection checks. Null: final acceptance remains0/8. No further blind step-count expansion is planned if the null persists. This is adaptive exploratory development, not an independent confirmatory experiment.

## Fixed design

Reuse all eight source-hash-selected development cases, references and order from E3, each from clean Base; keep negative cases in the denominator. E3 is the historical control, not a randomized concurrent arm. Initial BF16 and FP32 scores must match E3 within1e-6 or stop. Compare first-three-step accepted/rejected trajectories and report unexpected discrepancies, then measure any benefit from extra steps. Cases stopped earlier by backtracking are not given a retry or relaxed rule.

Same original matrix layer21 down_proj; BF16 physical forward; full FP32 temporary functional arithmetic; native-value anchored QP gap with FP32 Jacobian; exact GGN and preservation first-order term; tau100, nu10, trust0.1, drift0.3, KL0.001, target=min(max(Base BF16,Base FP32)+0.1,-1e-4), CG16/rtol1e-4, tolerance1e-6, one constraint, seven fixed backtracking factors, seed20261001, greedy8-token required native export check, Judge0. Only max_edit_steps/max_steps changes to12. Mechanical regression remains one step on the two permanently reserved rows outside the scientific denominator.

Primary endpoint: final accepted cases/8, including mandatory clean native reload/generated-token parity for changed weights. Secondary: final statuses, accepted intermediate gains, first-three-step reproducibility, actual versus proposed norms and rounding, reference KL, GGN calls and wall time. Intermediate gains do not count as final successes. Report all cases and preserve rejected transaction rollbacks.

## Resources and validation

Run20261001E4; pro5000 physicalGPU5 UUID GPU-4924bbd8-4082-3f4c-b4fb-78ca930122ca only; launch requires66,000MiB free. New-run finite ceilings:4hours,4096 exact GGN calls including regression, eight cases, at most12 edits/case; storage20GiB and teacher cache2GiB, with28GiB free required during setup. Estimated runtime30–60minutes using E3 throughput; this is an estimate. Prior ledger retained:566GGN and3329.959251675755seconds.

No implementation change relative to E3 (published source6d7ebe5): reuse its38 passing CPU tests and completed native mechanical validation as prior evidence, while rerunning the native regression before this run's cases. The frozen configuration is validated against the human-delegation receipt and the unchanged E3 scientific fields. CPU tests are not rebranded as newly run. Any accepted matrix still requires the real clean-native consumer; no mock/native-PASS substitution.

Stop on native implementation/regression failure, timeout/call ceiling, initial control-parity failure, non-selected-state mutation, rollback failure or clean-export mismatch. Save failed-call consumption and artifacts; no automatic same-version repeated retries. Generated matrices are deleted only after the final required consumer and parity receipt, per existing lifecycle. No CP/LoRA adapters or optimizer state.

## Data and interpretation

Training-support development only. The same eight cases are reused; no formal evaluation/calibration mixing. References remain scope-unknown, verified visual pairs are missing and scientific W_EVIDENCE admission remains BLOCKED_DATA. No generated medical negatives, clinical-efficacy claim or paid Judge. Publish source/config bindings, all sanitized outcomes and the resource ledger; keep raw medical QA, tokens, weights and private logs private.
