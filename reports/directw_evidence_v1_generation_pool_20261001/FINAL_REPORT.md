# E11 final: accepted edits do not establish answer improvement

Eight of eight edits accepted and exactly reproduced E9B's complete numerical trajectories. All eight independent clean reloads reproduced logits (maximum error zero), short tokens and full 64-token generation diagnostics exactly. All temporary matrices were deleted after the final consumers; all Base restorations passed. No worker remains and no failure receipt was found.

## Actual answer generation

Normalized exact match is 0/8 before and 0/8 after. Mean token F1 changes from 0.107124677 to 0.110730446. Two of eight token sequences changed; six stayed identical. Only one F1 increased. All16 before/after generations reached EOS without the64-token cap.

Private output inspection showed verbose sentences against short reference answers, including answers agreeing with the reference despite exact-match zero. Therefore0/8 exact match must not be called0% semantic accuracy. The two changed outputs were wording changes, without a demonstrated correction of the requested answer. No formal semantic grader was used (Judge0). These data support physical edit feasibility and modest target likelihood gains, not useful answer-correction efficacy.

## Full-pool side effects

Across199 case-probe slots/27 distinct QA on the same3 images,29 slots exceed KL0.001 and3 scores decrease by more than0.1. Worst mean answer-logprob change is -0.273224115; maximum KL is 0.015464095. Full per-case metrics are in FINAL_RESULTS.json and threshold crossings/per-role aggregates in SUMMARY.json. Scopes remain UNKNOWN; these are observed answer-distribution/score effects, not proven clinical harm or forgetting. Prior E10's narrower16-slot result remains intact.

| Role | Slots | KL >0.001 | Score change <-0.1 |
|---|---:|---:|---:|
| reference_image | 66 | 11 | 2 |
| same_image | 65 | 11 | 0 |
| third_image | 68 | 7 | 1 |

## Decision and cost

The outcome is not satisfactory evidence of useful editing or broad protection. Before expanding seeds/parameters, the next bounded experiment compares the existing Euclidean QP arm with functional curvature under the same8 cases, stochastic rounding seed, target/protection thresholds, three-step limit, and identical generation/full-pool diagnostics. E11 is the frozen functional control; the new arm must pass exact initial-score and mechanical replay checks. This isolates the added curvature term; no retrospective tuning or inflated accuracy claims. It does not fix the separate weakness of small likelihood gain as an efficacy objective.

New cost145GGN/1196.952263815seconds; cumulative2419GGN/16872.601025624seconds, within7200seconds/1024GGN. CPU44 local and remote tests passed. Raw medical QA/images/tokens and weights remain private; only sanitized receipts are published. The original train-support, three-image, unknown-scope and missing replacement-fact limitations remain BLOCKED_DATA for stronger scientific claims.
