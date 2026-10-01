# E6: full native FP32 precision control

Frozen before execution,2026-10-01; authorized by the user's delegated follow-up design/execution onGPU5. E5 measured substantial weight-rounding losses and six cases of candidate FP32/BF16 reference-KL disagreement. This is a precision-feasibility control, **not BF16 deployment success**.

## Hypothesis and intervention

Initialize using the same original native loader: language checkpoint loaded inBF16 and vision tower in its existing native loader precision, with no edited weights or adapters. Explicitly promote this initialized model and its image-input pipeline toFP32. Promotion preserves the initialized floating-point weight values; only the selected original layer21 down_proj matrix is optimized. Other parameter values remain frozen and audited against this run's Base. Default BF16 loading behavior is preserved by the code's default branch.

Test whether removing low-precision weight writes and low-precision native computation together allows at least one final accepted edit among the same eight development cases within three steps. Compare to E3's three-step native-value QP control0/8. These two precision effects are changed jointly and are not individually causally identified. Image features and Base scores can change because end-to-end arithmetic changes; this is not a matched absolute-logit-baseline comparison.

## Fixed scientific setup

Same eight original train cases, source-hash order, scope-unknown references, model checkpoint values and editable matrix. Clean Base reset per case. Same exact functional GGN, preservation first-order term, native-value QP gap, tau100, nu10, trust0.1, drift0.3, KL ceiling0.001, CG16/rtol1e-4, score tolerance1e-6, seven backtracking factors and seed20261001. Scientific maximum steps3 as in E3. Do not extend steps if the hypothesis fails.

The target remains relative gain0.1 from this precision's clean Base, capped at-1e-4: min(max(nativeFP32_Base_score,functionalFP32_Base_score)+0.1,-1e-4). Its absolute numerical value can differ from BF16 runs. FP32 functional and native teachers are independently bound to this run's Base/dtype. No previous BF16 teacher, optimizer or edit state is loaded. Raw answers and images are unchanged; no held-out evaluation or calibration data is used.

Primary: accepted FP32 native edits/8, requiring normal native target/KL acceptance, selected-matrix-only audits and clean FP32 native reload/logit parity<=1e-3, exact generated-token parity, zero hooks and no editor/adapter imports in the consumer. A positive result supports FP32 feasibility under these development constraints, not BF16 feasibility or medical effectiveness. Report all eight outcomes and paired development-case differences. Secondary: numerical trial decomposition, runtime/GGN/memory cost, requested-versus-written values and native/functional parity.

## Checks and resources

40 CPU tests passed; new test covers explicit precision selection, exact initialization-value preservation on a BF16 fixture promoted toFP32, and FP32 edited-weight/normal-functional diagnostic agreement. Native large-model FP32 remains pending before this run. Preliminary native regression uses the two permanent mechanical rows and the same modified loader/editor path; they stay outside the scientific denominator. Require actual same-precision native-functional parity, finite repeatable GGN and native candidate repeatability. Stop on runtime, repeatability, mutation, rollback or required export validation failures; no blind same-version retry.

Run20261001E6, pro5000 physicalGPU5 UUID GPU-4924bbd8-4082-3f4c-b4fb-78ca930122ca only. At least66,000MiB free at launch. New ceilings7200seconds/1024GGN, eight cases, three steps each; Judge0, storage20GiB, teacher cache2GiB,28GiB setup free space. Prior cumulative ledger1043GGN and6383.711352696759seconds retained. Expected duration30–60minutes, not guaranteed. Resident nativeFP32 model approximately28GiB; functional frozen tensors can share FP32 storage. A clean reload consumer temporarily adds another nativeFP32 model, so allow approximately60–68GiB combined peak; actual receipts decide feasibility.

Freeze protocol/config/source before launch. Diagnostic FP32 forward records are retained but do not alter acceptance. Old BF16 absolute-score parity is deliberately inapplicable (historical_control_scores removed); native/functional parity within the new FP32 regime remains mandatory. Legacy receipt field cross_precision_max_difference now represents native-versus-functional difference at identicalFP32 precision and must be interpreted with native_deployment_dtype.

## Lifecycle and limits

Export only the changed original FP32 matrix, run the clean nativeFP32 consumer then delete it after its last required parity check. Retain failure evidence and necessary unfinished consumers. All raw QA, images, generated tokens and tensors stay private; publish sanitized outcomes and source/config bindings. Original unknown medical scopes, missing verified visual pairs and lack of independent evaluation remain BLOCKED_DATA for scientific W_EVIDENCE claims. No medical negatives, CP/LoRA continuation, paid Judge or formal effectiveness claims.
