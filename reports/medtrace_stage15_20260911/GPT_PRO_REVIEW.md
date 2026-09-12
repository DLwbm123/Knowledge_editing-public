# MedTRACE Stage15 external evaluation

## Result summary

Both implemented branches completed all 200 independent edits. Native target output
match is 100% for both, and R0 retains high textual and cross-image generalization.
The primary frozen RC system does **not** transfer well to cross-image generalization:
it accepts only 2/164 cross-image probes, reducing success to 1.22%. Its 100% locality
is obtained by returning Base on every locality probe. This is a substantial scope
tradeoff, not evidence that the writer itself preserves all unrelated knowledge.

C_NO_H has better image-locality preservation than BalancEdit under R0, but the two
writers have similar editing/generalization success. C_FACT and C_EXTRA_QA have zero
legal H-supported requests and were not executed. No external validation or negative
performance conclusion about C_FACT follows from this run.

### Frozen system comparison

Percentages below are the predeclared normalized free-output match. For locality,
match means equality to the pre-edit Base output, **not medical correctness**. Each
listed row has one probe per supported edit, so edit-macro equals probe-micro.

| Metric | Supported edits | C_NO_H RC (primary) | BalancEdit RC (primary) | C_NO_H R0 (secondary) | BalancEdit R0 (secondary) |
|---|---:|---:|---:|---:|---:|
| Native reliability | 200 | 100.00% | 100.00% | 100.00% | 100.00% |
| Text generality | 200 | 97.50% | 98.00% | 97.50% | 98.00% |
| Image generality | 164 | 1.22% | 1.22% | 96.34% | 96.95% |
| Text locality | 200 | 100.00% | 100.00% | 100.00% | 100.00% |
| Image locality | 67 | 100.00% | 100.00% | 79.10% | 73.13% |
| 1-hop portability | 73 | 1.37% | 1.37% | 1.37% | 1.37% |
| 2-hop portability | 24 | 0.00% | 0.00% | 0.00% | 0.00% |
| 3-hop portability | 2 | 0.00% | 0.00% | 0.00% | 0.00% |

R0 activates 160/164 image-generality probes and 18/67 image-locality probes; RC
activates 2/164 and 0/67 respectively. Both reject all 200 text-locality probes.
All 200 native and text-generality probes remain ON in both systems. No threshold
was changed after these outcomes. R0's better tradeoff must not replace RC as the
predeclared primary result.

R0 image-locality favors C_NO_H by **5.97 percentage points**, paired edit-bootstrap
95% interval **[1.49, 11.94] pp** (67 edits). The image-generality difference is
−0.61 pp, interval [−2.44, 1.22] pp (164 edits). These are finite-cohort edit intervals,
not patient-level or multiple-comparison-adjusted population guarantees.

### Writer behavior without rejection

| FORCED_ON metric | Supported edits | C_NO_H | BalancEdit |
|---|---:|---:|---:|
| Native reliability | 200 | 100.00% | 100.00% |
| Text generality | 200 | 97.50% | 98.00% |
| Image generality | 164 | 98.17% | 99.39% |
| Text locality | 200 | 3.00% | 0.00% |
| Image locality | 67 | 10.45% | 0.00% |

C_NO_H reduces damage relative to BalancEdit, but absolute held-out preservation is
still poor when forced ON. Image-locality difference is +10.45 pp [2.99, 17.91];
text-locality difference is +3.00 pp [1.00, 5.50]. The reused **single training U QA**
has source-Judge correctness 99.5% versus 0% under forced ON, but this is an in-fit
diagnostic repeatedly measured over 200 edits, not 200 independent preservation
questions or held-out evidence. RC returns Base there and both score 100%.

### Portability and attack probes

Portability remains weak: only 1/73 one-hop exact matches, no two- or three-hop exact
matches. The semantic secondary one-hop scores are 9.59% for C_NO_H and 10.96% for
BalancEdit; two-hop scores are 4.17% and 0%; three-hop scores are 0% and 0%. Only two
three-hop examples are supported. High native success does not demonstrate multi-hop
medical knowledge propagation.

The 127 matched attack requests comprise five source attack types; these are not
portability hops. Exact-match rates are identical across the two writers in aggregate:
R0 125/127 (98.43%), RC 124/127 (97.64%). Under RC, pseudo-authoritative interference
is 21/23, irrelevant clinical details 28/28, vague qualifiers 27/27, symptom confusion
29/29, and misleading context 19/20. Individual per-edit outcomes can differ despite
equal aggregate rates. Per-type results remain in the standard table.

### Completion, uncertainty and cost

- Completed at **2026-09-12 03:10:47 Asia/Shanghai**, after approximately **7h45m**.
- 400 writer trajectories completed; zero recorded task failures. All six natural
  image/text routing replays passed. The checked text-only branch was OFF; no text-ON
  case arose in the bounded replay sample, so that case is not separately claimed verified.
- 1,057 distinct official probes were available. Missing images affect 169 probes:
  36 image-generality and 133 image-locality entries. They remain unsupported, not zero.
- Judge covered **2,426/2,426** distinct full-context tuples, all valid JSON; no old
  verdict was reused. Its native semantic score is 99% for both writers, although
  native normalized exact match is 100%. Two exact reference copies were rejected
  by the frozen semantic Judge. That disagreement was checked privately and retained;
  no semantic retry, relabeling or benchmark-target change was performed.
- Every Base native is initially-not-target under the declared literal normalization
  (0/200 exact matches); semantic Base target agreement is 3/200. The published
  BASE_NOT_TARGET strata use literal, not semantic, membership.
- The reported **15.30 GPU-process hours** is a conservative process-lifetime estimate,
  not busy-GPU time or FLOPs: both worker ends were recorded when the worker pair joined.
  Judge process lifetime was 580 seconds. Scoring and report generation added no training.
- Training-loop wall seconds, summed across independent edits: native CP 5,458.71;
  A2 3,944.11; CP-W0 14,781.68; C_NO_H continuation 21,654.70; BalancEdit 1,158.09.
  Initializer cost is counted once per edit. Corresponding optimizer steps: 27,000;
  16,000; 64,000; 64,000; 10,000. Low adapter parameter count is not equal compute.
- Frozen-queue `branches` counts in COVERAGE_AND_COST record preparation-time states;
  `completed` and `pending` are the final execution status (200+200, empty pending list).

This is an availability-selected 200-request subset of 260 resolved native requests,
not the complete 2,497-request release or a patient-independent clinical evaluation.
Public code/results are delivered separately from private QA, images and checkpoints.

## Original run protocol and provenance

Status: COMPUTE_COMPLETE. Frozen requests: 200; completed C_NO_H 200, BalancEdit 200.

C_FACT and C_EXTRA_QA have no legally verified out-of-edit-scope H support in the available source assembly. This stage does not establish external C_FACT validity and does not establish a C_FACT failure.

Reused 471 source-bound image references; downloaded only the three missing pinned author JSON indexes, no images. All 4490 MedMKEB training records were scanned once; all 131 resolved train-native rows intersect official evaluation image roles and were not used for H. One previously authorized SLAKE training QA is reused for U KL only.

Official native/alt, rephrase, image rephrase, locality, portability and exact-tuple-matched attack probes remain unchanged. Missing images are probe-specific unsupported cells. Targets are counterfactual benchmark targets, not clinical recommendations.

Main metric: deterministic free-generation output match, Unicode/case/whitespace normalization only. Locality is pre/post output agreement. Semantic source accuracy is separate; no zero-imputed Overall. This is release-aligned, NOT author execution parity or a paper-exact reproduction.

Three fixed deployment modes; RC_FIXED_OLD16 is primary, R0 predeclared secondary. No recalibration. Real text-only locality uses no image or visual tokens. Routes depend only on Base image/text features and legal native-only fit anchors.

New method-blind Judge judgments: 2426/2426; old semantic verdicts reused: 0. Snapshot 0499c3ac83fdef8810b907a23894ba91e95eddd8; distinct counterfactual target and source protocols frozen before student scoring.

Paired effects and fixed-seed edit-bootstrap intervals: PAIRED_EFFECTS.csv. Full/common support, initially-at-target strata and macro/micro denominators: MEDMKEB_STANDARD_RESULTS.csv. Training U diagnostics are not held-out locality.

Private: full questions/answers, source images, generation tokens, Judge mappings, checkpoints and environment paths. Public: source, protocol, counts, aggregate tables and this report. Patient identity and pretraining overlap UNKNOWN. No new algorithm, qualification gate, sealed-set access, or automatic next stage.

Sources: [MedMKEB pinned release](https://github.com/pkusixspace/MedMKEB/tree/d9f38639ec2285a0e9f541e22156ec14f87271d8) · [paper](https://ojs.aaai.org/index.php/AAAI/article/view/40705/44666) · [VLKEB evaluation source](https://github.com/VLKEB/VLKEB/blob/10951b7b3788928f578b73f07eac9e1eaa0316f3/easyeditor/evaluate/evaluate.py)
