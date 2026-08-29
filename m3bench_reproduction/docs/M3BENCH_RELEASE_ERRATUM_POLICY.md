# M3Bench release erratum policy

`M3BENCH_RELEASE_ERRATUM__ONE_MISSING_GOLD_EXCLUDED_FROM_EVALUATION` preserves upstream M3Bench metadata unchanged. One released SLAKE QA record (`xmlab281_3`, global index 4467) has an empty gold answer: no value is imputed. Paper-level source and raw-generation totals remain 16,276; public-release evaluation eligibility is 16,275. Raw generation still includes all questions, but downstream judging and task construction must exclude this record and must never interpret null correctness as baseline-wrong.
