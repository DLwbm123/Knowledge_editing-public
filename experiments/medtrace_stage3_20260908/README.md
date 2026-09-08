# MedTRACE Stage3 frozen-system evaluation

Source release during execution, not a completed-result claim. Fixed systems: S1 = BE_ROUTE + P4-W1_KL_0.1; S0 = the same route + P4-W0_TASK_ONLY; B = native BalancEdit V4 adaptation. FORCED_ON is a paired diagnostic, not another method.

Track A retains the authoritative 179 T0 / 1,108-event / 2,496-probe catalog. Of 736 planned training groups, BE and S0 each support 45 (22 T0 anchors and 23 T2L groups), while S1 supports seven; the frozen common subset is seven. Unsupported groups remain in the ledger. These are not full V4 or paper-exact reproduction results. Formal evaluation inputs are excluded from training supervision.

Track B reuses the original 16 Stage2 experts without training, at prefixes 1/4/8/16. It executes the actually selected writer with target-free Base routing and no oracle substitution. This is viewed insertion replay, not causal online learning or sequential-179 evaluation.

Research branch: `medtrace-stage3-20260908`. Frozen runner/source commit: `22c397a`; scoring commit: `751608a`. The manifest was frozen before the scoring commit. Prior Stage2 public anchor: `74d2a337f7d2b830d58819f76c87058cef0c5f3b`. The 25 focused Stage3 checks passed in the existing Torch environment; historical Judge binding (7,780 tuples) and tokenizer preflight also passed. Consult the [status and reviewer report](../../reports/medtrace_stage3_20260908) for timestamped observations, not this source snapshot as live state.

Focused check: `python -m pytest -q tests/medtrace/test_stage3.py tests/medtrace/test_stage3_sources.py tests/medtrace/test_stage3_bank.py tests/medtrace/test_stage3_finalize.py` in the project-compatible environment.

This is an allowlisted dependency snapshot, not a complete research checkout or a data/model release. Authorized private V4 manifests, images, QA, runtime locks and checkpoints are required for real execution. No private Git history, raw answers, token IDs, teacher distributions, weights or private Judge mappings are published. Neutral entrypoints affect process-list display only, not access control or anonymity.

The detached GPU2/3 coordinator is bounded at 24 hours wall / 48 GPU-hours, with generation limited to the first 20 hours and time reserved for Judge and aggregation. No additional monitoring process is started for this release. Final public delivery will be checked when the user next requests progress after completion. No novelty, SOTA, clinical-safety or full-M3Bench claim.
