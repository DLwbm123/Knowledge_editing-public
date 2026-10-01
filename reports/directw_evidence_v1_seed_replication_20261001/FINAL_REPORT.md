# E8 final: seed replication criterion not met

Run20261001E8 completed8/8 intended cases with **3/8 accepted native BF16 edits**, below the prespecified replication criterion4/8. The criterion failed; it was not relaxed. E7's5/8 does not establish seed stability. E8 used the identical algorithm and cases, changing only base rounding seed20261001 to20261002. This is a scientific negative replication result, not a runtime failure.

| Case | E8 final status | Accepted steps | Best intermediate gain | Last accepted reference KL | GGN |
|---|---|---:|---:|---:|---:|
| 0 | ACCEPTED | 2 | 0.111359954 | 0.000661065 | 26 |
| 1 | BACKTRACK_REJECTED | 1 | 0.041013718 | 0.000000236 | 26 |
| 2 | BACKTRACK_REJECTED | 2 | 0.093609810 | 0.000006770 | 41 |
| 3 | NOT_SATISFIED | 3 | 0.084469080 | 0.000651485 | 42 |
| 4 | BACKTRACK_REJECTED | 1 | 0.000029013 | 0.000005803 | 24 |
| 5 | BACKTRACK_REJECTED | 2 | 0.065768063 | 0.000174060 | 40 |
| 6 | ACCEPTED | 2 | 0.103959799 | 0.000742308 | 26 |
| 7 | ACCEPTED | 2 | 0.121882439 | 0.000003211 | 25 |

Intermediate gains of rejected/unsatisfied cases were rolled back, not counted as successes. All three successes0/6/7 were also E7 successes; E7-only successes2/5 failed here. Both seeds together yield8 accepted case-seed trials out of16 on **8 distinct cases**, not16 independent cases, and not a best-seed success rate. No seed was retried or selected.

All three accepted matrices passed independent clean native BF16 reload with logit error0, exact generated tokens and editor_imported=false. Their three117,442,152-byte temporary matrices were deleted after final consumers. Every case and final worker restored Base. Initial native BF16/functional FP32 control scores matched E3 within1e-6; same-precision parity0; every recorded normal-forward repeat was exact. All accepted intermediate steps passed fixed KL0.001 and drift0.3 limits. Native mechanical checks passed, while their own edit was BACKTRACK_REJECTED and excluded from the denominator.

There were55 scientific candidate trials,15 accepted intermediate steps and40 rejected trials. Primary rejection causes were30 protection-ceiling exceedances and10 lack of native merit improvement; no drift-first rejection. Full sanitized receipts preserve all outcomes. The frozen source/config bindings and reused42-test CPU receipt remain alongside this report. Source implementationbfe5c46, E8 protocolc97a08a.

New cost261GGN/1907.477476seconds (31.79minutes), within1024GGN/7200seconds. Cumulative1749GGN/11677.343349seconds includes all previous failures. Largest per-case PyTorch allocation58.118GiB; host peak RSS34987794432bytes. These are per-process allocation/RSS, not combined parent/consumer GPU peak. Judge0. Live check: PID1320879 exited, GPU5 free72830MiB, no compute processes, no failure receipts/matrices or log traceback. No other GPU/process was changed.

Next paired experiment is frozen on eight previously unused QA rows from the already audited original candidate pool: nearest rounding followed by stochastic rounding with the **less successful E8 seed20261002**, without method retuning. Both arms are fixed before the first run. Their images overlap the three old image groups, so this examines new-question transfer only; it cannot establish cross-image or clinical generalization. A check found zero image-and-question-disjoint candidates in this limited pool. The wider raw training file was inspected for metadata availability only and is not admitted into this experiment. Missing scope annotations, verified visual pairs and independent evaluation remain BLOCKED_DATA.
