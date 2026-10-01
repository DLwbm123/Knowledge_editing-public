# DirectW precision repair — native evidence, 2026-10-01

`current_state = BLOCKED_NATIVE` under the currently frozen BF16 consistency contract.
`three_numerical_checks = PASS_FOR_FP32_FUNCTIONAL_MODEL`.
`native_smoke_status = NOT_QUALIFIED`.
`scientific_data_status = BLOCKED_DATA` for all five branches.
`paid_judge_calls = 0`; `real_pilot_started = false`; `sequential_started = false`.

## What was actually repaired

The previous “FP32 coordinate” calculation immediately cast W to BF16 before forward. FP32 vector storage and dot products alone did not make its derivative calculation FP32. `MatrixRuntime.bind_inputs(..., arithmetic_dtype=torch.float32)` now provides a complete FP32 functional calculation with frozen temporary parameter/buffer copies and FP32 floating inputs. The original physical deployment model stays BF16. No adapter, trainable extra matrix, output offset, alternate layer or curvature approximation was added.

The original default binding is unchanged; the diagnostic backend is explicitly selected. It has not silently replaced the deployed BF16 function. New CPU regressions cover finite differences, GGN symmetry, preservation of physical BF16 state, the cast-induced derivative/finite-difference discrepancy and fresh residual verification for reused inverse solutions. All 29 CPU tests pass. These tests supplement, and do not replace, the native evidence below.

CG now records each recursive residual so that iteration truncation can be distinguished from the final true residual. Optional Q-inverse reuse re-evaluates `Q x - rhs` against the current operator and refuses stale/incorrect cached values. This is a computational reuse mechanism; it neither substitutes a curvature approximation nor grants extra CG iterations. It is CPU-tested and not yet used in a native QP.

## Native comparison on the same reserved mechanical input

Host: pro5000; GPU 5, UUID `GPU-4924bbd8-4082-3f4c-b4fb-78ca930122ca`. The same selected original matrix, native template, source row, checkpoint values and eager attention backend were used. TF32 was disabled for FP32 arithmetic. No medical sample or target was selected from model performance.

| Check | B2: FP32 coordinate with BF16 forward | B3: complete FP32 functional arithmetic | Frozen threshold |
|---|---:|---:|---:|
| CG actual relative residual | 0.00529754 | **0.0000583991** | <=0.0001 |
| CG iterations | 4 | **4** | <=4 |
| GGN symmetry absolute error | 0.000122070 | **0.0000000521541** | <=0.00001 scaled |
| Finite-difference relative error | 0.989385 | **0.00601152** | <=0.2 |
| Repeated operator max difference | 0 | **0** | Exact repeat |
| PSD quadratic probe | 0.0800781 | **0.0818391** | >=-0.000001 |

FP32 CG recursive residuals were 0.0703014, 0.00972417, 0.00109369 and 0.0000583007. The separately recomputed final residual was 0.0000583991. Thus four iterations suffice for this FP32 mechanical problem; the iteration cap and tolerance were not increased or relaxed.

The full-vocabulary exact computational JVP/VJP GGN remains matrix-free. Its symmetry/PSD/repeatability probes passed. The FP32 finite difference used the same epsilon 0.001: autodiff -0.802813 and central difference -0.797987. These are **derivatives of the named FP32 functional calculation**, not proof that the discontinuously rounded BF16 deployment map is differentiable.

## Why native admission still needs a precision decision

The high-precision calculation is internally consistent, but its logits differ from BF16 deployment: maximum **0.434829**, mean **0.0181278**. The maximum exceeds the existing 0.001 parity threshold. Argmax agreement over all sequence positions was 0.983416; the sole scored answer-predictor argmax happened to agree. These are mechanical parity observations, not medical accuracy or editing performance.

We preserve this failing comparison. No detached output correction, straight-through surrogate, post-hoc tolerance change or renamed PASS was used. The physical BF16 state was audited unchanged and selected W remained bitwise equal to Base.

There are two different questions:

- Is the FP32 functional network's numerical derivative/curvature/solver correct on the tested input? The measured checks now pass.
- Does that FP32 calculation satisfy the frozen BF16 same-function parity criterion? No.

A reviewed contract can explicitly use the FP32 function only as the editor's local numerical model and require all candidate acceptance in the actual BF16 deployment model. That is the concrete proposal in `PRECISION_PROTOCOL_PROPOSAL.md`; it is not currently represented as approved. No candidate write or QP was executed while that distinction was unresolved.

## Resources and preservation

B3 consumed **30.263297 seconds and 10 GGN calls**. Cumulative B1+B2+B3 consumption is **75.203757 seconds / 0.0208899324 GPU hours and 27 / 48 GGN calls**. Remaining original budget is **1724.796243 seconds and 21 GGN calls**; no budget was reset.

B3 peak GPU allocation was **47.4364 GiB**, peak host RSS about 17.08 GiB. The full FP32 matvec took **1.193766 seconds**, approximately 2.86 times the previous BF16-forward operator measurement. At the existing draft worst case of 948 calls/step, this is about 1131.69 seconds of GGN time per QP step, excluding other overheads. This is a cost projection of the diagnostic function, not a completed native QP or Pilot budget.

The B3 process exited and its lease was removed. No weights were written, no checkpoint was modified, no Judge was called, and no Pilot, sequential or scheduled continuation was started. B1/B2 reports remain untouched. Private medical payloads, actual authorization wording and raw logs remain in the independent RUN_ROOT; public JSON contains only deidentified measurements and source bindings.

Executed source hashes are in `SOURCE_EXECUTION_BINDINGS.json`. The final source additionally contains the CPU-tested inverse-reuse helper, which was not used by B3 and has no claimed native validation.
