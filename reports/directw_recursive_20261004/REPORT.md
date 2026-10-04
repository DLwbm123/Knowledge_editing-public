# Language-W recursive pilot: no answer-level gain in eight edits

Both predefined arms completed eight cumulative edits and their final 123-query
panels. All final generation tokens were identical to Base and to each other.
Qwen semantic scoring confirms no target or generalization repair. This is a
negative result for this fixed single-layer, single-step adaptation; it does not
establish that full M-ORE is ineffective.

| Arm | Target T0 Fix | T1G Fix | T2G Fix | T1L / T2L retention |
|---|---:|---:|---:|---|
| Fresh Base | 0/8 | 0/29 | 0/32 | NA / NA |
| Recursive | 0/8 | 0/29 | 0/32 | NA / NA |
| Static P0 control | 0/8 | 0/29 | 0/32 | NA / NA |

The 32 T1G probes include three originally correct probes, so the Fix denominator
is 29. The 49 T1L and two T2L probes have no originally correct eligible probes;
retention is unsupported, not 100% and not zero. Each method and fresh Base are
correct on 3/123 total queries. Insertion-time target correctness is also 0/8 in
both arms. Original Base eligibility is retained; fresh Qwen labels differ from
those original labels on 0/123 queries. The eight edits belong to eight recorded
source groups, which does not establish full patient independence.

## Mechanics and interpretation

All 16 edits had finite nonzero gradients and actual FP16 weight writes. Their
per-edit write Frobenius norms range from about 0.000370 to 0.001294. Mean
insertion-target NLL before/after is 3.396241/3.394601 for recursive and
3.396088/3.394322 for static. These small loss changes did not alter any final
answer tokens. The recursive state changed and its first edit exactly matched
static as required. No noneditable parameter or buffer changed; no hook remained.
Each final W was reloaded in a fresh native-only process with zero target-logit
error and identical generation. This verifies actual language-W deployment,
not successful knowledge repair or a benefit from recursion.

The trial uses only language layer 21, rank 512, scale 2, learning rate 0.1,
ridge 2000, one gradient step per edit, and a frozen visual encoder/projector.
The combination produced no effective answer repair here. The experiment does
not separate which adaptation or hyperparameter limited its effect. No tuning,
extra arm, seed, expanded cohort or second experiment was automatically launched.

## Scoring, cost and operational repair

Qwen3-32B-AWQ revision `0499c3ac83fdef8810b907a23894ba91e95eddd8` scored all
new content at concurrency 32, with thinking disabled, temperature 0, AWQ Marlin
and complete constrained JSON. All 385 mapped occurrences reduce to 123 identical
query/reference/candidate-text combinations. These were scored once within this
pilot and mapped back, giving 123/123 valid, fully bound responses. This reuse
prevents identical answers from receiving different labels across arms; it does
not reuse prior-run verdicts. Source-answer agreement is not clinical validation.

Editing and native generation ran from 2026-10-04 22:56:17 to 23:07:01
Asia/Shanghai, taking 643.41 seconds. GPU 7 was idle when queried because this
stage had completed. Pending scoring was started after the status query on GPU 6.
Its initial startup failed before any verdict: multiprocessing spawn attempted
to reopen the neutral stdin entry as a file. The user authorized monitoring,
repair and continuation; clearing only the synthetic main-file metadata repaired
startup. The failed attempt and logs remain private, and no deadline was reset.

The successful scoring worker completed in 166.53 seconds, including 109.39
seconds of generation (67.46 unique records/minute). This includes an unwarmed
scoring pass and is not the earlier warmed throughput benchmark. Actual versions
were vLLM 0.10.2, Transformers 4.55.2 and Torch 2.8.0. Worker exit code was 0,
all record bindings/coverage passed, and semantic retries were zero. Process and
GPU process naming audits passed. The two-hour original wall limit was preserved.

## Storage and delivery boundary

After each arm's registered generation and clean-reload consumers completed,
its temporary resume state and native-consumer W were deleted: four generated
files, 547,364,858 bytes in total, with deletion receipts and no retained copy.
Reconstructing these weights requires rerunning the fixed pilot. Shared Base,
vision and judge weights, all earlier experiments, and first-route artifacts
remain separate. Private configuration, full outputs/tokens, failed-attempt logs,
judge mappings and raw verdicts remain on the authorized server. Only source,
protocol, sanitized aggregate results and receipts are published.

See [JSON results](RESULTS.json), [CSV results](RESULTS.csv) and
[protocol](PROTOCOL.md). The hourly monitor was requested to handle faults and
finish this bounded run; it stops after publication is verified.
