# Phase B final review

- `native_smoke_status = FAIL`
- `scientific_data_status.W_FT = BLOCKED_DATA`
- `scientific_data_status.W_EUCLIDEAN_QP = BLOCKED_DATA`
- `scientific_data_status.W_KEY_QP = BLOCKED_DATA`
- `scientific_data_status.W_FUNCTIONAL_QP = BLOCKED_DATA`
- `scientific_data_status.W_EVIDENCE_QP = BLOCKED_DATA`
- `gpu_hours_used = 0.0124834611` (44.940460 seconds across both attempts; original cap 0.5 hours / 1800 seconds)
- `paid_judge_calls = 0`
- `real_pilot_started = false`
- `sequential_started = false`
- `current_state = BLOCKED_NATIVE`

B0 qualification, the corrected role gate, 26 CPU regressions, two bounded native attempts, failure analysis and public review materials are complete. **Native qualification did not PASS.** No candidate edit, QP transaction, accepted export or editor-free reload was executed after the numerical hard stop. No performance claim is made.

## Baseline, permission and provenance

Reviewed baseline: `0be4446be7c472cb942b5c616dc04117a52096f3`, independent `research/directw-evidence-v1`, `DLwbm123/Knowledge_editing-public`. Parent history and Stage-A evidence were preserved. This work remains separate from CP/LoRA; no adapter, optimizer state, old teacher or Judge score was loaded. Public diffs extend the reviewed implementation without merging another method route.

The user's supplied next-stage prompt approved DATA QUALIFICATION and NATIVE_SMOKE only. The user then explicitly selected **pro5000**, approved the proposed finite mechanical budget, and authorized autonomous repairs (“如果有问题你自己解决一下”). The actual resource was physical GPU 5, UUID `GPU-4924bbd8-4082-3f4c-b4fb-78ca930122ca`, RTX PRO 5000 72GB. Fresh free-memory checks, a nonexclusive bounded lease, storage probe and neutral `python -` process checks preceded loading. Existing GPU jobs were left running. Both new native PIDs exited and the lease was removed.

Native environment: Python 3.12.3, PyTorch 2.7.1+cu128, CUDA 12.8, transformers 4.51.3. Attention was explicitly bound to **eager** before native execution; no backend retry or numerical approximation was used. Base, CLIP, tokenizer/config/processor and native source files were hashed by full file content on pro5000. The native source there differs from the source found on my-gpu; the pro5000 hashes and actual runtime are the bindings for these attempts. The loader uses the clean LLaVA-Med checkpoint plus its independently bound CLIP tower. Delayed CLIP initialization produced checkpoint vision-key warnings; the runtime census is reported, and no checkpoint tensor is claimed to have been compared elementwise with instantiated CLIP parameters.

Private source/medical payloads and genuine human authorization receipts remain under the local independent `outputs/directw_evidence_v1/20260930B1/` and `20260930B2/`, and remote `/data/bmw/Knowledge_editing/outputs/directw_evidence_v1/`. Only deidentified reports, source, protocol and resource configuration are public. `SOURCE_EXECUTION_BINDINGS.json` distinguishes each actually executed source snapshot from the final source review commit.

## A. Implementation blocker — repaired and natively rechecked

B1 stopped at strict `functional_call`: `state_dict()` omitted non-persistent `model.rotary_emb.inv_freq` and CLIP `...embeddings.position_ids`. Binding only persistent state was incomplete. The repair binds all named parameters and named buffers, freezes buffer copies, audits non-persistent buffer mutation, restores buffers on single reset, and rejects selected parameter/buffer storage aliases. A new CPU regression checks strict functional parity, JVP, buffer mutation/reset and alias refusal.

After explicit repair authorization, B2 used the same mechanical input and **remaining** original budget. Native normal/functional logits then matched exactly (max difference 0). All 26 CPU tests passed. There is no remaining observed implementation blocker at this point in the native path; downstream transaction/export paths remain unvalidated because numerical qualification stopped first.

## B. Data blockers — independent of native status

29 train/adaptation candidates / three image groups were source-audited. Two were permanently reserved as `SMOKE_MECHANICAL`, chosen by source identity hash without scores. They cannot enter scientific FIT, calibration, confirmation or Pilot denominators. The 27 other rows remain candidates with unresolved role evidence.

All six scientific role counts are zero. Corrected EDIT targets/permissions, verified GEN linkage, background scope/teacher, near `VERIFIED_OUT_OF_SCOPE`, accepted history and complete fact/time ledgers are absent. No visual pair satisfies all five conditions plus evidence reference; W_EVIDENCE is additionally `BLOCKED_DATA` and no native medical Evidence edit ran. The other four branches have their own common-role blockers, not merely a missing-pair failure. DEV_CAL and TEST_CONFIRM are `NOT_ESTABLISHED`. Formal test QA, old evaluation QA, model scores and medical negative synthesis were not accessed. See `PHASE_B0_DATA_QUALIFICATION.md` and the V2 audit/eligibility JSONs.

## C. Resource blockers and measured cost

Two attempts consumed **44.940460 GPU-active wall seconds / 0.0124834611 GPU hours**, and **17 / 48** permitted complete-GGN operator calls. No budget was reset. Peak allocated GPU memory was **21.2093 GiB**, and peak host RSS **16.8497 GiB**. These are observed process peaks, not reserved exclusive GPU capacity or projected Pilot peaks. One FP32 solver vector is 224 MiB; the selected BF16 snapshot is 112 MiB. Native logits `[1,603,32000]` occupy 38,592,000 BF16 bytes; a FP32 arithmetic copy occupies 77,184,000 bytes. Byte figures derive from actual measured shape/dtype.

Observed Base forward in B1 was 0.2409 s (repeat 0.1292 s). B2 forward/backward/JVP/VJP/GGN/CG measurements are in `NATIVE_RESOURCE_PROFILE.json`. A complete current-operator GGN matvec took **0.417434 s**. At the Stage-A draft worst case of **948 GGN calls / QP step**, GGN time alone is **395.728 s / step**. One-step eight-/12-edit projections are 0.8794 / 1.3191 GPU hours; at 20 steps/edit they become **17.5879 / 26.3818 GPU hours**, excluding all other overheads. This is a lower-bound cost calculation for the **currently unqualified operator**, not a Pilot run, qualified budget or performance result. The 20-step count is an unapproved worst-case draft assumption, not an executed edit. `EXACT_GGN_SCALABILITY_BLOCKER` is retained for that draft.

No OOM or resource-cap failure occurred. Completed one-/two-constraint QP, deployed write, clean reload and whole-edit timings are **NOT_MEASURED**; their totals cannot be replaced by static estimates. No resource budget for Pilot was approved.

## D. Numerical blockers — unresolved under frozen criteria

The ordinary native answer gradient was finite/nonzero, shape `[4096,14336]`, BF16, norm 28.6081. Other physical parameters accumulated no gradients. Fixed-anchor KL gradient norm was 5.66065e-6 at Base, and 0.00795475 after a controlled represented W perturbation of norm 0.00100040. Native JVP and VJP executed.

BF16 CG residual was **0.00597025**, exceeding frozen `rtol=1e-4` after the approved four-iteration cap. This triggered the predeclared FP32 temporary coordinate patch; deployment stayed BF16, native functional evaluation cast into original BF16, logits/protection and solver/dual arithmetic used FP32. FP32-coordinate CG residual was still **0.00529754**, so it also failed. These measurements do not distinguish iteration truncation from a BF16 numerical floor; a larger-iteration or broader precision experiment was not authorized or run.

GGN symmetry probe gave `u^T Fv = 0.00753784`, `v^T Fu = 0.00765991`, absolute difference **0.000122070**, above frozen 1e-5 scaled tolerance. PSD probe was positive (0.0800781) and repeated operator difference was exactly zero. Thus finite/nonnegative/repeatable execution is established, but symmetry qualification failed.

Native represented finite difference at epsilon 0.001 gave **-0.00858307**, versus autodiff **-0.80859375**, relative error **0.989385**, above the predeclared 0.2 mixed-precision tolerance. The represented plus/minus W displacements were recorded. BF16 rounding is a plausible contributor; these measurements do not establish that it is the sole cause. Native derivative numerical correctness cannot be declared PASS from the finite/nonzero-gradient check alone.

No CG tolerance was relaxed, no damping or mechanical target was retuned after native results, no input was replaced on performance, and no diagonal/K-FAC/LoRA or layer substitution was made. Continuing to QP or candidate editing after these failures would violate the native qualification hard stop.

## Native qualification checklist

| # | Required criterion | Observed result |
|---:|---|---|
| 1 | Clean content/runtime binding | Recorded; 7,566,219,264 actual parameters, 686 persistent state keys; external CLIP binding explicit |
| 2 | Exact editable matrix path/shape/BF16 | PASS |
| 3 | Selected storage alias/bias | PASS; single storage, no bias |
| 4 | Expanded template/predictor/mask/EOS | PASS; raw 28, expanded 603, predictor 600, vocab 32000 |
| 5 | Normal-forward repeatability | PASS; exact repeat |
| 6 | Functional-call parity | B1 FAIL, repaired B2 PASS; exact logits |
| 7 | Native gradient finite/nonzero | PASS; numerical derivative agreement separately FAIL |
| 8 | Fixed-anchor KL semantics | PASS for measured Base/perturbation mechanics |
| 9 | Native JVP support | PASS |
| 10 | Native VJP support | PASS |
| 11 | Complete GGN computational path | Executed; mathematical numerical qualification FAIL |
| 12 | GGN finite/PSD/symmetry/repeat probes | Finite/PSD/repeat passed; symmetry FAIL |
| 13 | CG/QP frozen convergence | CG FAIL; QP NOT_RUN |
| 14 | No silent approximation | PASS; exact computational JVP/VJP formula retained |
| 15 | Deployed candidate rounding | NOT_RUN |
| 16 | Bounded write only-selected state audit | NOT_RUN |
| 17 | Candidate rollback | NOT_RUN; final Base W bitwise restoration verified both attempts |
| 18 | Editor-free clean reload/generation | NOT_RUN |
| 19 | Peak GPU and host memory | MEASURED for completed operations; reload unmeasured |
| 20 | Per-matvec/projected cost | MEASURED partial lower bound; full QP/edit costs unmeasured |

The actual W_KEY input was `[1,603,14336]`, BF16, captured once at the physical down_proj input. Predictor geometry aligned, and the hook was removed. No W_KEY editing transaction was run.

## Stop and review boundary

Final state: **`BLOCKED_NATIVE`**. Scientific data additionally remains `BLOCKED_DATA` for every branch. Both native processes exited, no active continuation exists, and no Pilot/Sequential authorization was generated. The remaining numerical qualification and unexecuted native checks require a separately reviewed, explicitly bounded next action; this report neither broadens the four-iteration/call budget nor substitutes CPU evidence for them.
