# E11: actual answer generation and full eligible-pool diagnostics

Frozen before GPU execution on 2026-10-01 under the user instruction to continue the proposed experiments. Eight independent clean-Base edit cases exactly replay E9B, with the same stochastic BF16 rounding seed and edit settings. The prior E8 seed-replication failure and E10 KL crossings remain unchanged evidence.

## Measurements

1. Native greedy generation, before and after editing, up to 64 new tokens. Report paired normalized exact match, multiset token F1, output change, EOS and cap hits. Preserve all raw tokens, answers and references privately. Lexical metrics are descriptive and can miss semantically correct paraphrases; these remain original training answers, not validated replacement facts or independent medical evaluation.
2. All 199 eligible case-probe slots across 27 distinct original QA: 25/25/25/25/25/25/24/25 per case. Exclude that case's target/reference and permanent smoke IDs/question hashes. Separate same-image, reference-image and third-image roles, with scope UNKNOWN. Report score-change and KL distributions and crossings of -0.1 and 0.001 separately. Include every eligible QA; no favorable selection. Cases and probes reuse the same three images.
3. Exact E9B trajectory fingerprints and initial-score parity are mandatory. Diagnostics never enter optimization or acceptance. Retain native mechanical checks and independent clean reload. The latter must reproduce both original short generation/logits and the new full-answer diagnostic before deleting the temporary matrix.

## Limits and dependency order

Only physical GPU5. New ceiling 7,200 seconds and 1,024 GGN calls; inherited 15,675.648759132833 seconds and 2,274 GGN calls remain in the cumulative ledger. Require 66,000 MiB free GPU memory, estimate less than 512 MiB added prepared-probe GPU storage, 2 GiB teacher cache, 20 GiB outputs and 28 GiB free disk. Any mismatch/nonfinite/resource failure stops and is retained. Judge calls zero. Raw medical material stays private; source, protocol and sanitized results are public.

After completion and publication, use all outcomes to freeze a separate matched curvature comparison under fixed rounding and data. Do not silently alter this run. Independent image/scope/fact-replacement evidence remains unavailable and must not be synthesized.
