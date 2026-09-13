# Qwen sidecar complete — 2026-09-13

All 2,378 frozen student output IDs have a Qwen verdict, with no duplicate or missing
IDs and matching packet configuration/snapshot. Completion: 06:02:17 UTC (14:02:17
Asia/Shanghai). This does not replace the primary Astra score or judge clinical truth.

|Astra|Qwen|Count|
|---|---|---:|
|Correct|Correct|1778|
|Incorrect|Incorrect|446|
|Incorrect|Correct|88|
|Correct|Incorrect|66|

Agreement = 2,224/2,378 = 93.52%; disagreements = 154 (6.48%). Astra judged 1,844
correct (77.54%); Qwen judged 1,866 correct (78.47%). These are decision rates across
the student-output packet, not an official task accuracy or Judge accuracy metric.
Neither judge is established as ground truth by agreement alone.

Same question/reference/candidate records and rubric; Astra/high uses fifty-record
contexts whereas Qwen3-32B-AWQ uses independent single-record contexts and non-thinking
generation. Thus this is not a pure model-only ablation. Qwen's first 550 decisions
used concurrency 1; the rest used concurrency 2 under the explicit resume amendment.
No accepted saved prefix was rescored. Fixed Qwen snapshot:
`0499c3ac83fdef8810b907a23894ba91e95eddd8`.

Successor elapsed time = 4,002.48 seconds, including model load; this excludes the
original 550-decision segment and preparation. Sampled board-memory peak = 22,376MiB.
Final raw JSONL, execution record, amendment and log were copied privately for later
paired analyses. No raw answers, question IDs, tokens or server details are published.
