# E6 final: native FP32 precision control

Run 20261001E6 completed all 8 intended development cases. **2/8 FP32 native edits were accepted**, versus 0/8 in the E3 BF16 three-step control. The other six were NOT_SATISFIED and rolled back. This supports precision feasibility under the frozen development protocol; it does not establish BF16 deployment, medical generalization, or isolated causality of write rounding versus forward arithmetic. Both precision effects changed together, including image features and clean Base scores.

| Case | Final status | Accepted steps | Best intermediate score gain | New GGN |
|---|---|---:|---:|---:|
| 0 | NOT_SATISFIED | 3 | 0.099202514 | 42 |
| 1 | ACCEPTED | 2 | 0.100000381 | 25 |
| 2 | ACCEPTED | 1 | 0.100268841 | 10 |
| 3 | NOT_SATISFIED | 3 | 0.099877864 | 42 |
| 4 | NOT_SATISFIED | 3 | 0.018540367 | 38 |
| 5 | NOT_SATISFIED | 3 | 0.098945618 | 40 |
| 6 | NOT_SATISFIED | 3 | 0.099996567 | 42 |
| 7 | NOT_SATISFIED | 3 | 0.099997520 | 41 |

Intermediate gains are diagnostics, not final successes. The target gain remains 0.1 with score tolerance 1e-6 and reference KL ceiling 0.001; no threshold was relaxed. Cases 6/7 narrowly missed but remain failures. All eight used torch.float32 native weights and had exact native/functional same-precision Base parity. Historical BF16 absolute-score parity is inapplicable, not a failed check.

The two accepted edits passed normal native target/protection checks and selected-original-matrix-only audits. Independent fresh native consumers reloaded each FP32 matrix: maximum logit difference 0, generated tokens identical, editor_imported=false. Their final reference KL receipts are -5.94446e-8 and -6.08034e-8, small floating-point roundoff around zero; these are preserved without claiming a negative mathematical KL. Each 234,882,664-byte temporary matrix was deleted after its final consumer. Every case and the final worker restored Base. PID 1296850 has exited; the live GPU5 check found no compute process and 72,830 MiB free.

Mechanical native regression passed finite, repeatable exact-GGN and native checks (symmetry error 5.83e-13, PSD probe 3.77e-9); its own one-step edit was NOT_SATISFIED and excluded from the denominator. Forty CPU tests preceded the run. No mechanical pass substitutes for accepted native export checks.

New consumption: 291 GGN and 2,138.571929 seconds (35.64 minutes), within 1,024 GGN / 7,200 seconds. Cumulative consumption including previous failures: **1,334 GGN / 8,522.283285 seconds**. Largest per-case PyTorch allocation was 47,675,453,952 bytes (44.40 GiB); this excludes the simultaneous fresh consumer process and is not a whole-GPU peak. Host peak RSS was 65,230,123,008 bytes. Judge calls: 0.

Source/config bindings and the frozen protocol are alongside this report; execution source was published in 10c106f84d34f99e14f749ac3d6c2686bc385081. Sanitized complete receipts are in FINAL_RESULTS.json. Private QA, generated tokens, images and tensors remain private.

Next test: return to the original BF16 path and compare fixed-seed unbiased stochastic weight rounding against E3 deterministic nearest rounding on the same eight cases and three-step budget. Reuse one random draw per step across all backtracking factors; no seed selection, retries for favorable draws or acceptance relaxation. Freeze this as a new protocol/RUN_ROOT before execution. Missing verified visual pairs, non-target scope labels and independent evaluation continue to block W_EVIDENCE scientific claims.
