# Phase B completion review — approved precision contract

- `native_smoke_status = PASS`
- `scientific_data_status.W_FT = BLOCKED_DATA`
- `scientific_data_status.W_EUCLIDEAN_QP = BLOCKED_DATA`
- `scientific_data_status.W_KEY_QP = BLOCKED_DATA`
- `scientific_data_status.W_FUNCTIONAL_QP = BLOCKED_DATA`
- `scientific_data_status.W_EVIDENCE_QP = BLOCKED_DATA`
- `gpu_hours_used = 0.0401092252` (144.393211 cumulative native active seconds / 1800)
- `GGN_calls_used = 46 / 48` (B1/B2/B3/B4 = 0/17/10/19)
- `paid_judge_calls = 0`
- `real_pilot_started = false`
- `sequential_started = false`
- `current_state = WAITING_FOR_PILOT_REVIEW`

**Mechanical qualification passed under the explicitly approved FP32 editor / BF16 deployment contract. This is not an editing-effect result or Pilot permission.** No scientific edit, visual medical evidence edit or continuous editing experiment ran. All five branches independently remain blocked by data. The process exited, the GPU lease was removed, and no continuation was scheduled.

## Scope, approval and isolation

Repository `DLwbm123/Knowledge_editing-public`, branch `research/directw-evidence-v1`; Stage-A reviewed baseline `0be4446be7c472cb942b5c616dc04117a52096f3`; immediate source parent `d7abee52e7fa5a0a777515b26c3a0a673076f1fa`. Independent B4 run root, namespace and source snapshot preserve B1/B2/B3 and CP/LoRA. No adapter, optimizer state, old teacher or old model score was loaded.

The user's “继续” approved the previously presented [precision proposal](../directw_evidence_v1_precision_repair_20261001/PRECISION_PROTOCOL_PROPOSAL.md) for the remaining mechanical smoke. That historical proposal remains unchanged. This report records the subsequent human authorization, not an agent-created approval of Pilot. Original caps remain 0.5 GPU hours, 48 GGN calls, four CG iterations, at most two reserved mechanical inputs, two constraints, one edit step, eight unjudged generation tokens and zero Judge calls.

The actual host was pro5000, physical GPU 5, RTX PRO 5000 72GB, UUID ending `ca930122ca`. The prelaunch check found 72,830 MiB free. All files stayed on the approved `/data/bmw/` mount, which passed a write/read probe. The parent process command was neutral `python -`; the clean-reload child was also launched through stdin. Python 3.12.3, PyTorch 2.7.1+cu128, CUDA 12.8, transformers 4.51.3, eager attention and TF32 disabled are bound. Base + separately bound CLIP remain the original composition described in B0/B2; delayed-vision initialization warnings do not establish elementwise equivalence of the instantiated CLIP with the checkpoint's unused vision tensors.

`SOURCE_EXECUTION_BINDINGS.json` binds every executed namespace source file, model/dependency/data/config/protocol digests and the genuine human receipt reference. Source is unchanged after B4 execution. The private full manifest and raw medical payloads remain excluded from publication.

## A. Implementation status and native coverage

The missing nonpersistent-buffer binding was repaired in B2; it remains covered by CPU and native checks. Full FP32 temporary functional state now provides differentiation arithmetic; the physical original model remains BF16. Same-Q inverse reuse checks the current actual residual on every reuse. A categorical KL pullback computes `Jᵀ weights (p-q)`, preserving nonzero protection terms away from the teacher. The CPU regression agrees with ordinary KL autograd to 1e-12 on a nonstationary double-precision problem. At the identical FP32 teacher, the analytic residual is exactly zero; the measured ordinary autograd normalization roundoff norm was 1.270389e-6. This identity is not applied to a different teacher or later iterate.

**30 CPU tests passed.** Native checks below actually executed on the VLM. CPU tests are not substituted for them.

| Criterion | B4 evidence |
|---|---|
| Clean binding / exact path / alias | 7,566,219,264 physical parameters; `model.layers.21.mlp.down_proj.weight`, [4096,14336], BF16, single storage, no bias |
| Template / predictor / EOS / padding | Raw length 28, expanded 603, predictor 600, vocabulary 32000; independent score gather error 0 |
| Normal repeat / same-precision functional parity | Both max differences exactly 0 |
| Physical gradient / fixed BF16 teacher | Finite nonzero gradient; perturbed fixed-teacher gradient nonzero; no other physical parameter gradients |
| FP32 JVP / VJP / GGN | Executed, finite; PSD probe 0.081840515; symmetry error 5.21540642e-08; repeated difference 0 |
| Frozen four-step CG | Actual residual 5.83991469e-05, below 1e-4; reuse residuals rechecked |
| FP32 finite difference | Relative error 0.00601152219, below frozen 0.2 |
| One-/two-constraint QP | Both CONVERGED with explicit slack, protection linear term and frozen dual tolerance 1e-8 |
| BF16 write / state audit | Only selected original W changed; nonselected parameters, buffers and hooks audited |
| BF16 protection / rollback | KL 5.85726957e-06 <= 0.001; W restored bitwise; normal logits restored exactly |
| Native export / editor-free reload | Original BF16 matrix exported and replaced in clean native loader; no editor import/hooks/router/adapter/Tucker; logits max difference 0; ordinary generated token sequence identical |
| W_KEY | Physical down_proj input [1,603,14336], one call, hook removed; geometry only |
| Resource / projection | All listed mechanical microbenchmarks measured; extrapolation limitations below |

The second QP row is exactly half the first row, a deliberately redundant **mechanical** constraint. The two-row test does not establish independent visual evidence or general conditioning of arbitrary scientific constraint sets. Native W_EVIDENCE visual-pair execution remains NOT_RUN because no legal pair exists. One smoke input was used throughout; both reserved rows remain excluded from scientific FIT/calibration/evaluation denominators.

## B. Data blockers

The unchanged B0 audit found 29 original train/adaptation candidates in three image groups, two permanently reserved mechanical rows, and zero legally qualified scientific rows. See [data qualification](PHASE_B0_DATA_QUALIFICATION.md), [audit](DIRECTW_DATA_AUDIT_V2.json) and [branch eligibility](DIRECTW_METHOD_ELIGIBILITY_V2.json).

| Branch | EDIT | GEN | BG | NEAR | VIS pairs | Independent scientific sources | DEV_CAL | TEST_CONFIRM | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| W_FT | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | BLOCKED_DATA |
| W_EUCLIDEAN_QP | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | BLOCKED_DATA |
| W_KEY_QP | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | BLOCKED_DATA |
| W_FUNCTIONAL_QP | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | BLOCKED_DATA |
| W_EVIDENCE_QP | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | BLOCKED_DATA |

Corrected targets/permissions, same-fact paraphrase linkage, non-target scope, independent pair labels/incompatibility, accepted history and availability time, legal historical teachers, independent calibration and confirmation remain unestablished. Missing entries were not fabricated. No formal test/evaluation QA, old scores or generated medical counterexamples were used. Mechanical PASS does not repair these evidence gaps.

## C. Resources and budget

B4 used 69.189454 seconds and 19 complete GGN calls; cumulative use across every attempt is 144.393211 seconds (0.0401092252 GPU hours) and 46/48 calls. Remaining budget is 1655.606789 seconds and two calls, which does not authorize another stage. Child reload is included in the parent wall interval, not double-counted. No OOM occurred.

Peak parent GPU allocation was **49.560 GiB**. Peak parent host RSS was **17.285 GiB**; child reload GPU allocation 14.982 GiB and host RSS 17.285 GiB were measured separately. These are per-process peaks, not a measurement of simultaneous aggregate host RAM. Vector: 224 MiB FP32; selected snapshot: 112 MiB BF16; full original persistent CPU snapshot: 15,132,438,528 bytes. Full FP32 logits: 77,184,000 bytes. Activation increments and reserved GPU peaks are recorded per operation in JSON. B4 generated about 195 MB before lifecycle cleanup, within the 20 GiB cap.

| Operation | Wall seconds | Complete GGN calls | Peak allocated GPU GiB |
|---|---:|---:|---:|
| clean_model_load | 9.055573 | 0 | 14.659 |
| template_image_expansion | 0.363851 | 0 | 14.885 |
| Base_forward | 0.139836 | 0 | 14.940 |
| deterministic_repeat_forward | 0.061911 | 0 | 15.012 |
| functional_call_parity | 0.068560 | 0 | 15.084 |
| W_KEY_physical_input | 0.061596 | 0 | 15.172 |
| answer_score_backward | 0.492949 | 0 | 16.691 |
| protection_KL_backward_Base | 0.100166 | 0 | 16.686 |
| protection_KL_backward_perturbed | 0.092216 | 0 | 16.905 |
| solver_coordinate_answer_backward | 0.512098 | 0 | 44.719 |
| solver_coordinate_protection_backward | 0.488308 | 0 | 44.866 |
| analytic_categorical_KL_pullback | 0.483652 | 0 | 45.184 |
| solver_FrozenGGN_anchor | 0.390648 | 0 | 43.644 |
| native_JVP | 0.718122 | 0 | 44.562 |
| native_VJP | 0.493980 | 0 | 46.059 |
| exact_GGN_matvec | 1.127216 | 1 | 46.278 |
| repeated_exact_GGN_matvec | 1.127047 | 1 | 46.497 |
| symmetry_exact_GGN_matvec | 1.129289 | 1 | 46.716 |
| Q_inverse_CG | 7.962417 | 7 | 48.466 |
| Q_inverse_stationary_preservation_CG | 2.282323 | 2 | 48.028 |
| one_constraint_QP | 3.442702 | 3 | 48.466 |
| two_constraint_QP | 6.292148 | 4 | 49.560 |
| write_normal_forward_protection_rollback | 4.100212 | 0 | 47.684 |
| ordinary_unjudged_free_generation | 0.425445 | 0 | 46.767 |
| editor_free_clean_reload | 11.128667 | 0 | 4.692 |

One-/two-row QP timings include residual verification and KKT evaluation **after inverse precomputation**. They must not be advertised as fresh standalone arbitrary-QP costs. The shared precomputation took 7.962417 s for the constraint inverse and 2.282323 s for the zero-RHS preservation solve/probes. The sampled full GGN took **1.127216 s**. At the requested draft worst case of **948 calls / QP step**, GGN time alone is **1068.601 s / step**, about 17.81 minutes. A one-step edit has that same GGN-only lower bound; 8/12 one-step edits are 2.375/3.562 GPU hours. At the unapproved 20-step draft, one edit is **5.937 hours**, 8 edits **47.493 hours**, 12 edits **71.240 hours**. These extrapolations exclude other overhead and larger protection sets; they are neither observed edit runtimes nor approved Pilot budgets. `EXACT_GGN_SCALABILITY_BLOCKER` remains.

## D. Numerical limits and interpretation

The exact curvature claim is restricted to `FP32_FULL_FUNCTIONAL`: fixed native expanded inputs are promoted to FP32, frozen parameter/buffer copies are evaluated in FP32, and only selected W is differentiated. The prepared image/prefix embeddings stay fixed; no claim is made about differentiating image preprocessing. This operator is **not** the exact derivative of BF16 rounding. FP32-vs-BF16 logits still differ by **0.434828758**, above 0.001; `cross_precision_parity_passed=false` is retained. The approved contract separates this comparison from the unchanged BF16 normal/functional parity gate. No detached output offset, curvature approximation, extra CG iteration, damping retuning or threshold relaxation was used.

Requested delta norm was **3.51814242e-06**, actual represented BF16 delta norm **9.68575478e-08**. **99.996108%** of requested nonzero components disappeared under rounding. The original answer log score changed from -2.976303577 to -2.976473808, a **-0.000170230865 decrease**. These single-input mechanical diagnostics are not a knowledge-editing score.

The retained temporary transaction satisfied the predeclared mechanical conditions: finite nonzero represented write, protection KL <= 0.001, bounded original-W write, full state audit, exact rollback and clean reload parity. It did **not** establish a deployed scientific target-margin success, which was never the acceptance criterion for this mechanical transaction. Nearly complete rounding loss and the observed score decrease remain a material obstacle for any future scientific protocol. No post-result target/damping/trust-radius tuning or second optimization step was attempted.

## Stop boundary and review items

`current_state = WAITING_FOR_PILOT_REVIEW`. All native mechanical checks are complete under the revised contract, while scientific data and exact-GGN scalability remain blocked. External review must assess the cross-precision semantics, BF16 rounding, legal data-role evidence and a finite costed protocol before any new phase. No approved Pilot/Sequential file or automatic training task was created. Source, deidentified receipts, original negative results and this review are the public delivery; patient payloads, model tensors, raw generated tokens and private source paths stay private.

The temporary 112 MiB BF16 matrix export completed its final consumer (clean reload/generation). Under the existing checkpoint lifecycle, that single generated tensor file was then deleted; its deletion receipt is `CHECKPOINT_LIFECYCLE.json`. Original Base/CLIP, private native outputs/tokens/config/source and all prior runs are retained. There is no retained exported-matrix copy; recreating it would require a newly authorized bounded run.
