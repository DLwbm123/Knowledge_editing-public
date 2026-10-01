# E9A final: nearest-rounding control; paired experiment incomplete

20261001E9A completed8/8 intended new-QA cases with **2/8 accepted native BF16 edits** (cases1/5). Three cases were NOT_SATISFIED and three BACKTRACK_REJECTED; all six restored their starting matrices. E9B remains required under the frozen paired protocol. No comparison winner or transfer-hypothesis result is declared from armA alone.

| Case | Final status | Accepted steps | Best intermediate native gain | Last accepted reference KL | GGN |
|---|---|---:|---:|---:|---:|
| 0 | NOT_SATISFIED | 3 | 0.062412262 | 0.000001276 | 35 |
| 1 | ACCEPTED | 2 | 0.130336761 | 0.000021380 | 25 |
| 2 | NOT_SATISFIED | 3 | 0.062438488 | 0.000027661 | 43 |
| 3 | BACKTRACK_REJECTED | 0 | 0 | — | 9 |
| 4 | NOT_SATISFIED | 3 | 0.090619802 | 0.000032533 | 40 |
| 5 | ACCEPTED | 1 | 0.119877815 | 0.000028637 | 10 |
| 6 | BACKTRACK_REJECTED | 2 | 0.066902161 | 0.000000404 | 35 |
| 7 | BACKTRACK_REJECTED | 1 | 0.112387657 | 0.000030756 | 27 |

The target remains min(max(native BF16 Base,functional FP32 Base)+0.1,-1e-4), so gain over native Base alone exceeding0.1 is insufficient when FP32 Base is higher. Case7 remained below its exact target and was rolled back. Intermediate gains are not final successes. No score/protection/drift thresholds were changed.

Both successes passed independent clean native BF16 reload: logits error0, generated tokens equal, editor_imported=false. Each117,442,152-byte temporary matrix was deleted after its last required consumer. All cases and the final worker restored Base. Same-precision Base parity0 and every recorded native repeat exact; all accepted intermediate steps passed native KL0.001 and edit/base drift0.3 gates. Previous E3 score parity is inapplicable to new questions. ArmB will instead require initial native/functional scores matching these armA cases within1e-6.

51 scientific candidate trials comprised15 accepted intermediate steps and36 rejections:2 protection-ceiling failures and34 lack of native merit improvement. Native mechanical GGN/parity checks passed, but its edit was BACKTRACK_REJECTED and excluded from the denominator. The unchanged implementation retains the42 passing CPU-test evidence and sourcebfe5c46. Frozen pair/armA protocol was published in200371a before either arm ran. Exact original train annotation/selection checks are in SELECTION_AUDIT.json; these new QA share the same3 old image groups and do not establish cross-image or clinical generalization.

New cost235GGN/1790.983875seconds (29.85minutes); cumulative1984GGN/13468.327227seconds, retaining all earlier costs. Peak per-case PyTorch allocation57.881GiB; host peak RSS34923921408bytes. Allocation does not measure combined parent/consumer GPU peak. Judge0. Live check: PID1333578 exited, GPU5 free72830MiB, no compute processes, failure receipts, remaining matrices or traceback.

E9B must now execute the same frozen8 target/reference pairs with stochastic BF16 rounding at seed20261002, same model/solver/thresholds, and actual armA consumption carried into its ledger. The pair's criterion remains B>=2 accepted AND B>A; with A=2 this requires at least3 B successes, without changing the original criterion. E9B is not permitted to choose new samples or favorable seeds. Raw QA/images/tokens/configs/logs remain private. Missing role/scope/visual evidence and independent evaluation remain BLOCKED_DATA.
