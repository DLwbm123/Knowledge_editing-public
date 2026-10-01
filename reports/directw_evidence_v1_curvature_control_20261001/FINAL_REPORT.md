# E12 final: curvature tradeoff, answer efficacy still unproven

The matched Euclidean arm accepted7/8 cases versus functional curvature8/8 (E11). Case3 was backtrack-rejected and restored Base. All initial-score checks passed; the functional mechanical regression exactly replayed. All7 accepted matrices passed independent native logit/short-token/full64-token-generation parity and were deleted after their final consumers. All cases and worker restored Base. No failures, matrices, lease or live worker remain.

| Metric | E11 functional | E12 Euclidean |
|---|---:|---:|
| Accepted / all cases | 8/8 | 7/8 |
| KL >0.001 / all199 probes | 29 | 26 |
| Score drops <-0.1 / all199 probes | 3 | 2 |
| KL >0.001 / common174 probes | 21 | 26 |
| Score drops <-0.1 / common174 probes | 2 | 2 |
| Mean post-edit lexical F1 / all8 | 0.110730446 | 0.107124677 |
| Exact matches / all8 | 0 | 0 |
| Changed generated sequences / all8 | 2 | 2 |
| Actual new GGN calls | 145 | 11 |
| Wall seconds | 1196.952264 | 906.319710 |

Common probes use the seven cases both methods accepted; do not count the Euclidean rollback's zero changes as successful protection. Functional curvature enabled one more numerical acceptance and had fewer KL crossings on the common accepted subset. It does not meet the frozen full-denominator dominance rule because full-pool crossings are higher; Euclidean does not dominate because acceptance and lexical F1 are lower. The declared outcome is a tradeoff, with no demonstrated general superiority. Timing is a sequential same-host observation, not a randomized throughput benchmark. Euclidean's11GGN calls are from the unchanged mechanical regression, while its scientific optimizer uses none.

All16 before/after generations reached EOS within64 tokens. Euclidean's two changed outputs were wording changes; private inspection did not establish correction of the reference answer. Exact match0/8 is not semantic accuracy0%: verbose answers can agree with a short reference. Small likelihood gains do not by themselves establish useful editing. Probe scopes are UNKNOWN, so score decreases/KL changes are not proof of clinical harm. Three reused images and train-support cases do not establish independent generalization. Judge0; missing fact replacements/roles/visual pairs remain BLOCKED_DATA.

New cost11GGN/906.319709792seconds; cumulative2430GGN/17778.920738058seconds.44CPU tests passed locally and remotely. Full sanitized metrics are in FINAL_RESULTS.json and PAIRED_SUMMARY.json; raw QA, images and tokens remain private.

Next: a bounded, read-only Base-model prompt-format control on the same eight targets, retaining their original references. Compare the original prompt to one fixed instruction requesting a short answer, and measure generated answers plus reference likelihood. This tests whether response-format mismatch contributes to low target likelihood before spending more editing/seed runs. It will not be counted as editing success, independent clinical evaluation or new-fact replacement. Freeze its exact prompt, metrics, success criterion and budget before execution.
