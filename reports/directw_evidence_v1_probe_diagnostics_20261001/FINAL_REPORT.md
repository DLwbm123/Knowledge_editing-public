# E10 final: exact replay, additional-QA distribution changes

All8/8 edits accepted; every complete numerical trajectory exactly matched E9B, including candidate scores/KL/drift/solver attempts, GGN counts and final scores. All8 independent clean native BF16 reloads had logits error0, identical generated tokens and no editor imports. All8 temporary117,442,152-byte matrices were deleted after final consumers. Every case/final worker restored Base. Mechanical replay/probes passed; its own rejected edit remained outside the denominator.

**The all-probe diagnostic hypothesis failed:** 2/16 case-probe slots exceeded KL0.001. Both are the same extra QA after different edits (cases3/6), with score improvements+0.121012419 and+0.034415841. This demonstrates distribution change beyond the single protected reference, not established harm or clinical forgetting. No probe score fell by more than0.1; the largest decrease was0.023121357. These16 slots reuse only3distinct QA, with unknown medical scope and previously used images, so absence of large score decreases is narrow evidence.

| Case | Probe role | Score change | Answer-position KL |
|---|---|---:|---:|
| 0 | same_image | 0.000054598 | 0.000002588 |
| 0 | third_image | -0.000009537 | 0.000000291 |
| 1 | same_image | -0.000230789 | 0.000007939 |
| 1 | third_image | -0.015638351 | 0.000001025 |
| 2 | same_image | 0.000000000 | 0.000000162 |
| 2 | third_image | -0.023121357 | 0.000657544 |
| 3 | same_image | 0.121012419 | 0.014059590 |
| 3 | third_image | -0.015664101 | 0.000000993 |
| 4 | same_image | 0.010297775 | 0.000004033 |
| 4 | third_image | -0.000033855 | 0.000000894 |
| 5 | same_image | 0.000120640 | 0.000003796 |
| 5 | third_image | 0.010410309 | 0.000000322 |
| 6 | same_image | 0.034415841 | 0.001564787 |
| 6 | third_image | 0.000001431 | 0.000000239 |
| 7 | same_image | -0.000339508 | 0.000011666 |
| 7 | third_image | -0.000021935 | 0.000000478 |

Every before/after probe was finite and post-edit repeats were exact. Probe measurements never entered edit acceptance. Existing reference KL/drift gates passed; the diagnostic crossings are retained without changing thresholds or relabeling the edits as failures. Same-image maximum KL0.014059590; third-image maximum KL0.000657544. Score means exclude EOS and use native BF16 forward logits; answer-position KL is computed in CPU FP32 from native log probabilities.

New cost145GGN/1119.903179seconds (18.665minutes); cumulative2274GGN/15675.648759seconds. Judge0, within7200seconds/1024GGN cap. Live workerPID1358793 exited, GPU5 free72830MiB, no failure receipts, remaining matrices or traceback. Original source/protocol3ddda17;43 CPU tests and actual native mechanical path checks passed. FINAL_RESULTS.json and PROBE_SUMMARY.json preserve sanitized complete receipts.

The next bounded diagnostic will cover the entire eligible original train QA pool instead of just the three unique source-hash-first probes, reusing the same8 E9B edits and exact replay checks. Existing KL crossings remain known, not retested as a new all-pass claim. The additional question is whether previously unprobed eligible QA show score decreases below-0.1; enumerate all values rather than selecting favorable probes. Unknown scopes and the same3images remain limitations. No invented medical negatives, paid Judge or CP/LoRA continuation. Raw QA, logits, tokens and images remain private; W_EVIDENCE admission remains BLOCKED_DATA.
