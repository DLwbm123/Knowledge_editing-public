# DirectW-Evidence-v1: candidate method, stage A only

Only one original MLP output matrix is updated, W_t = W_(t-1) + DeltaW_t.
There are no adapters, CP/Tucker factors, experts, routers or inference editing
hooks. All other parameters and buffers must be identical before and after an
accepted edit. Inference uses an ordinary original-matrix load.

The proposed location is `model.layers.21.mlp.down_proj.weight`, the physical
location recorded in the old Stage17 method lock. A fresh read of the clean
checkpoint's safetensors headers confirms shape [4096,14336], BF16, and no
selected bias. This is **not** evidence of runtime JVP support, storage aliases,
or instantiated deployment precision. The matrix has 58,720,256 entries. The
checkpoint contains 7,566,219,264 tensor elements; an actual model parameter
census remains pending. New deployment parameters: zero. Temporary optimization
variables, full-vocabulary teachers, gradients and weight copies are nonzero.

## Function responsibilities

| Responsibility | Implementation |
|---|---|
| One native matrix / functional and normal forward / original input hook | `editor.MatrixRuntime` |
| Fixed answer token scoring and both visual endpoints | `editor.mean_answer_logprob`, `build_constraints` |
| Source/sample/token normalization and fixed anchor KL | `numerics.position_weights`, `protection_kl`, `editor.ProtectionGroup` |
| Exact output categorical GGN, frozen local point and RNG | `numerics.FrozenGGN`, `ggn_matvec` |
| Matrix-free Q solve / explicit slack dual / bounded hard reference | `numerics.cg`, `solve_step`, `hard_qp_reference` |
| True rounded-weight acceptance, finite backtracking and rollback | `editor.edit_one` |
| W_FT, Euclidean, actual-input key, functional, evidence branches | `editor.BRANCHES`, `edit_one` |
| True-input/gradient geometry | `editor.geometry`, `MatrixRuntime.capture_input` |
| Role, scope, teacher, leakage and temporal audit | `contracts.audit_data`, `validate_cache` |
| Fixed history capacity and accepted-history anchors | `contracts.HistoryMemory` |
| Native reconstruction and clean reload parity | `export.export_native`, `load_original_matrix`, `verify_clean_reload` |
| Trusted phase approval before loader / explicit state progression | `gate.require_external_approval`, `gated_load`, `advance_state` |
| Failure denominators, bytes and dependency-aware cleanup | `contracts.AttemptLedger`, `enforce_storage`, `cleanup` |
| CPU numerical regression and static review bundle | `tests`, `stage_a.build` |

The five branches use the same legal conventional pair supervision and protection
information. Only W_EVIDENCE_QP adds the cross-image score constraint. W_KEY_QP
uses inputs of the actual selected matrix, never an arbitrary residual stream.
W_FT minimizes squared fit violations, real anchored KL and a step-start update
penalty using plain gradient descent; no inherited optimizer is loaded.

## Acceptance and temporal semantics

Each solve freezes inputs, prefix, mask, probabilities and RNG. Curvature is
refreshed at each local iterate, while unrelated Base teachers stay fixed. An
accepted historical teacher is never silently replaced by a wrong Base teacher.
Source/fact strata and hash priority determine bounded historical retention;
explicit supersedes events are recorded. Eviction is a capacity decision, never
a selection using formal test results. Input, positions and teacher/cache bytes
are all capped. Keys/curvature must be refreshed for the current weight version.

The editor rotates a role-ordered constraint list with at most eight active QP
rows, and checks **all** legal fit constraints in each rounded-weight acceptance.
It uses normal model forwards through runtime-bound score closures, not predicted
QP scores. Each visual counterexample must name its own real KL protection group;
the two endpoint margins and cross-image constraint are separately checked.
Callers bind immutable tensor inputs with `MatrixRuntime.bind_inputs` and provide
already validated role/cache metadata. Native template expansion is not guessed.

An edit-start violation of a fixed protection budget is reported separately. An
unsatisfied terminal edit restores the edit-start state; exceptions and Ctrl-C
restore it as well. Every failed attempt is retained. A sequential call starts
from the previous accepted native matrix; `reset_single` restores the clean Base
before an independent single edit. There is no summation of independent patches.

## Scope of demonstrated evidence

The 20 CPU groups test actual synthetic PyTorch computations, including an
independent SciPy primal oracle and clean inference in a separate process. They
are implementation evidence only. No medical model task accuracy, native
generation parity, clinical grounding or sequential medical performance is
claimed. Real tokenizer, image expansion, backend and precision binding require
separately authorized NATIVE_SMOKE. The CLI fails closed for native phases; it
does not create a training dispatch or synthesize a trusted review receipt.
