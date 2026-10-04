# AlphaEdit Stage17: generation complete, semantic comparison pending

Status: **GENERATED_NOT_SCORED**. This is a generation-stage delivery, not a
completed five-method performance comparison. No Astra verdict has been
produced for this new run. Do not interpret text-match diagnostics below as
semantic Fix, generalization or locality, or compare them directly with the
historical Astra metrics of MedTRACE C_NO_H, BalancEdit, BELoRA or LoRA.

## Execution and coverage

The run started on 2026-10-03 at 22:33:23 and ended on 2026-10-04 at 02:28:56
(Asia/Shanghai): **3 h 55 min 32 s** total wall time. All three workers exited
with code 0, before the 8-hour deadline. The later read-only check found no
remaining GPU compute process. Source commit:
`aeab1e9` on `research/directw-evidence-v1`.

| Arm | Independent edits | Cumulative insertions | Prefix panels | Native reload parity |
|---|---:|---:|---|---:|
| Layer 21, sensitivity | 146/146 | 146/146 | 1, 50, 100, 146 | 150/150 |
| Layers 19–23, primary | 146/146 | 146/146 | 1, 50, 100, 146 | 150/150 |

Each arm generated 2,378 independent-panel rows and 3,355 sequential rows.
The latter comprises 3,213 unique-query prefix rows and 142 additional insertion
rows; four insertion-native outputs are already present in their prefix panels
and are reused. Total new edited-output rows: **11,466**. All 300 registered
fresh-process reloads reproduced exact target logits and native generation,
with no remaining hooks or AlphaEdit algorithm imports in the consumer.
This validates execution and persistence, not the medical correctness of answers.

Covariance statistics completed on 10,000 external Wikipedia articles and
4,920,417 tokens. All 1,512 Base queries were regenerated; **90/1,512 (5.95%)**
differed in raw continuation tokens from historical Base outputs. Token drift
is not automatically semantic error. Original cohort eligibility masks remain
unchanged, and the drift must be disclosed in the semantic comparison.

## Descriptive text-match diagnostics

The only normalization is lowercasing and collapsing whitespace. Punctuation,
additional explanation and synonymous wording still count as mismatches.
These are reproducible string comparisons, **not Astra judgments**. Denominators
are the 146 native edit targets, not all generalization/locality probes.

| Arm | Independent target text match | Match immediately after each sequence insertion | Target text match at final prefix 146 |
|---|---:|---:|---:|
| Base, newly generated | 0/146 (0%) | — | — |
| Layer 21 | 63/146 (43.15%) | 122/146 (83.56%) | 101/146 (69.18%) |
| Layers 19–23 | 55/146 (37.67%) | 144/146 (98.63%) | 112/146 (76.71%) |

The multilayer arm has more final-prefix exact text matches than the single
layer arm, while the single layer arm has more independent-edit matches.
Immediate-insertion counts and final-prefix counts refer to different model
states and must not be presented as a paired forgetting rate. These diagnostics
alone do not establish superiority over any historical baseline.

| Output condition | Layer 21 | Layers 19–23 |
|---|---:|---:|
| Independent output rows reaching the 1,024-token cap without EOS | 47/2,378 | 93/2,378 |
| Sequential output rows reaching the cap without EOS | 14/3,355 | 26/3,355 |
| Empty decoded sequential answers | 10/3,355 | 4/3,355 |

No independent output was empty. Cap and empty-answer observations are retained
without excluding cases or changing decoding. Semantic scoring must include
these outputs under the original protocol.

## Scope, costs and remaining work

Only language-model MLP down-projection weights changed. The vision encoder
and multimodal projector remained frozen. Projection statistics were text-only;
this does not establish preservation of visual capabilities. The projection
selects small positive eigenvalues below 0.02, not an exact mathematical nullspace:
selected ranks were 9,366 / 8,315 / 7,156 / 6,269 / 4,797 out of 14,336.

Measured worker residence times were 0.989 h for statistics/Base audit, 2.917 h
for the single-layer worker and 3.925 h for the multilayer worker, including
dependency waiting. These are not isolated GPU-kernel hours. Sum of case times:
single-layer independent 1.750 h and sequence 0.947 h; multilayer independent
2.310 h and sequence 1.386 h. Do not compare these directly with historical
training-only costs on another host.

Generated single weights, final sequence state and all five projection bases
were deleted after their registered consumers completed; zero corresponding
`.pt` files remain. Raw outputs/tokens, private bindings, configuration and small
receipts are retained remotely. Shared Base models and source data were not
deleted. Public artifacts contain only source and deidentified aggregates.

**Remaining work:** execute the frozen isolated Astra/high semantic queue,
validate complete verdict coverage, compute original eligible-denominator Fix,
generalization, locality and paired comparisons, then publish the final
five-method report. The queue was not started by the GPU launcher. The original
run window is now closed; this report neither starts a new queue nor silently
extends its budget. C_NO_H must remain labeled as the no-H method. Historical
LoRA sequence used its approved BF16 replacement whereas its single phase and
the new AlphaEdit run used FP16; immutable Astra snapshot, runtime epoch and
hardware also remain comparison limitations.

Reproduction: run `experiments/alphaedit_stage17/summarize.py` with
`CAMPAIGN_DIR` pointing to the private run directory. It reads receipts and
raw outputs without loading models or modifying experiment state. Aggregate
data are in `GENERATION_SUMMARY.json`; original design is in `PROTOCOL.md`.
