# E9 paired final: stochastic8/8 versus nearest2/8

Both preregistered arms completed all8 new-QA cases. **Nearest BF16 rounding accepted2/8; stochastic BF16 rounding accepted8/8**, paired difference+0.75. The frozen criterion(B>=2 AND B>A) is met. All10 successful case-arm edits passed independent clean native BF16 reload with logits error0, exact generated-token equality and no editor imports. This supports the comparison on these eight QA targets sharing three previously used images; it does not establish image-disjoint, seed-stable or clinical generalization.

| Case | A: nearest | B: stochastic | B native score gain | B final reference KL |
|---|---|---|---:|---:|
| 0 | NOT_SATISFIED | ACCEPTED | 0.125384808 | 0.000000530 |
| 1 | ACCEPTED | ACCEPTED | 0.156061172 | 0.000032207 |
| 2 | NOT_SATISFIED | ACCEPTED | 0.102453947 | 0.000025128 |
| 3 | BACKTRACK_REJECTED | ACCEPTED | 0.112024307 | 0.000006555 |
| 4 | NOT_SATISFIED | ACCEPTED | 0.187514067 | 0.000002981 |
| 5 | ACCEPTED | ACCEPTED | 0.213625908 | 0.000028057 |
| 6 | BACKTRACK_REJECTED | ACCEPTED | 0.191894531 | 0.000002070 |
| 7 | BACKTRACK_REJECTED | ACCEPTED | 0.175294876 | 0.000026172 |

Six pairs favor B, none favor A, and two succeed in both. These are16 case-arm trials on8 distinct QA targets/3images, not16 independent medical examples. No outcome-dependent seed/sample selection occurred: seed20261002 was frozen before armA, retaining the seed that failed E8's replication criterion. E7/E8's prior5/8 versus3/8 and the failed4/8 replication criterion remain valid negative evidence about seed stability on the older QA pool.

B's initial native BF16/functional FP32 scores and thresholds matched A within1e-6 on all8. Same-precision parity0; every diagnostic native repeat exact. Every accepted intermediate step met actual native KL0.001 and edit/base drift0.3 limits. B used15 candidate trials:12 accepted steps and3 merit rejections (all case3). No thresholds were relaxed. A's six failures rolled back; all cases and both final workers restored Base. All10 temporary matrices were deleted after their final clean native consumers; each was117,442,152bytes. Native mechanical checks passed in both arms, while their own rejected edits remained outside scientific counts.

B consumed145GGN/1087.418348seconds (18.12minutes), peak per-case allocation58.100GiB. Pair consumed380GGN/2878.402223seconds (47.97minutes), within frozen2048GGN/14400seconds. Cumulative2129GGN/14555.745578seconds preserves all earlier failures and controls. B host peak RSS34987548672bytes. Per-process allocation is not a combined parent/consumer peak. Judge0. Live check: B PID1344840 exited, GPU5 free72830MiB, no compute processes/failure receipts/matrices/traceback. Other GPUs untouched.

Execution sourcebfe5c46; pair frozen200371a, B execution bindingb91eae0. E9A_FINAL_RESULTS.json, E9B_FINAL_RESULTS.json and PAIRED_RESULTS.json preserve full sanitized outcomes and denominators. Raw QA, images, generated tokens, logs and tensors stay private.

Next unresolved issue: preservation was measured on one reference QA per edit. A separately frozen diagnostic will replay the B algorithm/seed on the same cases and measure native pre/post changes on two additional, source-selected QA per edit, without feeding those values into optimization or acceptance. These probes have unknown medical scope and are not legal non-target labels; measured changes cannot be called clinical forgetting. Existing missing visual pairs, non-target scopes and independent evaluation remain BLOCKED_DATA.
