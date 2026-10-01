# E7 final: BF16 stochastic weight rounding

Run20261001E7 completed8/8 intended development cases: **5/8 accepted native BF16 edits**, versus E3 deterministic-nearest0/8 on the same cases, solver and three-step budget. All five accepted edits passed an independent native BF16 reload with logit difference0, exact generated-token equality and no editor import. This supports feasibility on these repeatedly inspected training cases for the one frozen seed schedule. It does not establish seed robustness, independent evaluation, medical generalization or the missing visual-evidence branch.

| Case | Final status | Accepted steps | Best intermediate gain | Last accepted reference KL | GGN |
|---|---|---:|---:|---:|---:|
| 0 | ACCEPTED | 2 | 0.111404002 | 0.000650925 | 26 |
| 1 | BACKTRACK_REJECTED | 0 | 0 | — | 10 |
| 2 | ACCEPTED | 1 | 0.130644798 | 0.000635559 | 10 |
| 3 | BACKTRACK_REJECTED | 2 | 0.084349394 | 0.000746039 | 42 |
| 4 | BACKTRACK_REJECTED | 0 | 0 | — | 10 |
| 5 | ACCEPTED | 2 | 0.128131956 | 0.000218087 | 25 |
| 6 | ACCEPTED | 1 | 0.104725122 | 0.000000465 | 10 |
| 7 | ACCEPTED | 1 | 0.109321594 | 0.000002866 | 10 |

The frozen target gain0.1, KL ceiling0.001, drift limits0.3 and score tolerance1e-6 were unchanged. Rejected case3's intermediate gain is not a success: its final matrix was restored, like cases1/4. Of30 scientific candidate trials,21 were rejected:13 exceeded the protection ceiling and8 did not improve native merit. These reason counts use protection first, then drift, then merit; no rejected trial required drift as its primary reason. Fixed seeds followed20261001+1009*case_index+step, with one coupled random tensor across backtracking. No seeds were retried or selected.

All eight initial native BF16 and functional FP32 scores matched E3 within1e-6; native/functional same-precision parity was0. Every recorded native repeat was exact. Every accepted intermediate step met actual native protection/drift gates, single-matrix audits were enforced, and each case/final worker restored Base. Five temporary matrices of117,442,152bytes each were deleted after the last required clean consumer. No matrices or failure receipts remain. Worker PID1309549 exited; live GPU5 check found no compute process and72830MiB free. Log had no traceback.

The two permanently reserved mechanical rows passed actual finite, repeatable GGN/parity checks; their one-step edit was NOT_SATISFIED and remains outside the scientific denominator.42 CPU tests preceded the native run. Source was frozen inbfe5c46; full config/source hashes are in SOURCE_BINDINGS.json. Complete sanitized receipts are in FINAL_RESULTS.json; private QA, images, tokens and raw diagnostics remain private.

New cost154GGN/1247.582582seconds (20.79minutes); cumulative1488GGN/9769.865870seconds, preserving previous failed runs. Peak per-case allocation58.002GiB, within the frozen memory plan; this is one-process allocation, not combined parent/consumer GPU peak. Host peak RSS35008667648bytes. Judge0. No other GPU or process was changed.

Compared with E6's2/8 FP32 precision control, E7's5/8 must not be presented as a general claim that BF16 outperforms FP32: arithmetic, Base scores and rounding distributions differ. E7's properly matched control is E3. The next bounded test changes only the prespecified seed to20261002, retains all eight cases and evaluates all outcomes without choosing the better seed. W_EVIDENCE remains BLOCKED_DATA for missing verified visual pairs/scopes and independent evaluation.
