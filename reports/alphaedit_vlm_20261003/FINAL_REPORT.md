# A1 AlphaEdit VLM baseline: completed

The bounded native BF16 run completed at **2026-10-03 22:05:51 Asia/Shanghai**,
about six minutes after launch. All three workers exited with code 0. At the
subsequent live inspection GPU 5, 6, 7 each showed 2 MiB idle usage and no compute
process. This was normal completion, not a stalled or unstarted experiment.

Both AlphaEdit variants generated the complete requested target, including EOS,
for all eight independent edits. Both also harmed preservation probes and failed
the preregistered joint criterion. This demonstrates BF16 direct-weight editing
feasibility for these original training answers; it does not establish protected
editing, new-fact learning, clinical correctness, or broad VLM generalization.

## Results

| Metric | Single layer 21 | Five layers 19–23 |
|---|---:|---:|
| Completed independent edits | 8/8 | 8/8 |
| Full target tokens exact, Base → edited | 0/8 → 8/8 | 0/8 → 8/8 |
| Mean lexical token F1, Base → edited | 0.1071 → 1.0000 | 0.1071 → 1.0000 |
| Mean target answer+EOS NLL, Base → edited | 7.2578 → 0.0158 | 7.2578 → 0.0545 |
| Probe answer-score drops below -0.1 | 25/199 | 26/199 |
| Probe KL above 0.001 | 68/199 | 72/199 |
| Worst probe score change | -2.6221 | -3.7863 |
| Maximum probe KL | 2.0449 | 2.6709 |
| Edits with zero probe drops below -0.1 | 0/8 | 0/8 |
| Independent native reload checks | 8/8 pass | 8/8 pass |
| Joint success criterion | FAIL | FAIL |
| Sum of per-case execution times | 192.11 s | 208.15 s |
| Worker duration including dependency wait/load | 345.11 s | 362.30 s |
| Peak parent-process CUDA allocation | 17.81 GiB | 17.23 GiB |

CUDA allocation excludes the concurrent clean-reload child and is not a
whole-device peak measurement. Shared statistics took 137.96 seconds and were
reused by both arms; do not double-count this cost or treat dependency waits as
active GPU compute. The prior Direct-W cost ledger remains separate and paused.

All cases reached the requested exact token sequence through ordinary native
greedy generation after physical weight writes. A fresh native loader reproduced
the target logits exactly and the full generated output identically, with no
editor import or hooks. The raw-token/metric receipts were checked against all
16 public case records. All Base restorations passed.

## Preservation breakdown

The 199 slots reuse 27 distinct image/question pairs across independent edits;
they are not 199 independent examples. Their semantic scope remains unaudited.

| Probe image relationship | Slots | Single drops / KL violations | Multi drops / KL violations |
|---|---:|---:|---:|
| Same image as target | 65 | 8 / 25 | 7 / 24 |
| Reference image | 66 | 12 / 18 | 10 / 22 |
| Third image | 68 | 5 / 25 | 9 / 26 |

Drops per edit, indices 0–7, were `[1,4,1,3,7,3,4,2]` for the single-layer arm
and `[2,4,1,2,10,2,2,3]` for the five-layer arm. No case had completely clean
preservation at the frozen -0.1 threshold. Five layers did not improve target
success on this set and did not establish a preservation advantage.

## Projection and interpretation

Statistics used 447 other English-QA SLAKE training images and 266,820 expanded
prefix tokens, excluding all three evaluation images and all answer tokens.
At threshold 0.02, the retained basis ranks for layers 19–23 were
`[13882,13849,13802,13810,13763]` out of 14,336 dimensions. Thus approximately
96–97% of directions were classed as low-eigenvalue directions under this
particular statistic. This is an approximate low-activation subspace, not an
exact nullspace: every measured minimum eigenvalue was positive. This diagnostic
helps characterize the limited protection; it is not a causal ablation proving
why the probe drops occurred. No threshold tuning followed these results.

The result rules out a blanket assertion that changing W cannot control free
generation in this VLM. It does not prove AlphaEdit dominates the earlier method:
the present baseline preserves AlphaEdit's own optimizer and has no Direct-W
trust/drift limit or preservation-based rejection, while the earlier experiments
used those constraints. E23 also used FP32. Here the single-layer actual update
norm reached 2.0843, illustrating a substantially different allowed update
regime. Compare efficacy and preservation jointly, not the 8/8 figure alone.

This remains a bounded VLM adaptation of the upstream method. The image input,
lookup position, context handling, KL reference, statistics corpus and chosen
layers differ from the original LLM benchmark. There is no sequential-edit
evaluation, and the eight targets were previously developed training examples.
Base exact mismatch does not by itself establish semantic or clinical error.

## Delivery and lifecycle

Source snapshot: `d555992`; upstream reference:
`b84624f44dfe8fc6cd9e41df916c44124a0c46dc`. CPU algebra/gradient/position checks
and reserved native latent-gradient qualification passed. The frozen protocol,
source bindings, aggregate summary, all sanitized case records, statistics
summary, artifact audit and deletion receipts accompany this report.

Every generated matrix was deleted only after edited generation, all probes and
independent reload completed. Shared projection tensors were deleted after both
arms completed. The artifact audit found zero remaining generated matrices or
projection tensors. Private configurations, source data, raw generated outputs,
logs and spectra remain on the server; original pretrained assets are preserved.
Published records contain no raw images, question/answer text or model weights.
No automatic retry, follow-on experiment or monitoring automation was started.
