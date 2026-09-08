# MedTRACE Stage2 V2 public dependency snapshot

Stage2 is RUNNING, not complete. Active attempt: `medtrace_stage2_20260908_r02`. GPU2/3 run old DEV16 BalancEdit, followed by scoped Base-before generation/Judge and 16 newly source-qualified episodes (16 shared native-to-A2 initializations plus 80 fixed method trajectories). The first two old BE transforms were saved before a local post-save replay-key error; they are reused without another 50-step training. All failures and original clocks are retained. Stage1's 70 trajectories are reused, not retrained.

Research branch: `medtrace-stage2-20260908`. First two BE training states: `73bdb8f`; recovery/new execution: `3cde3ed`; source/readiness and guarded delivery release: `e1fd8fd`. Runtime imports reuse the existing Stage1 dependency snapshot. This folder is source-only, not a substitute for the full research worktree or private runtime/data locks.

See [Stage1 full-behavior audit](../../reports/medtrace_stage2_20260908/STAGE1_FULL_BEHAVIOR_ADDENDUM.md), [prospective amendment](../../reports/medtrace_stage2_20260908/STAGE2_AMENDMENT_V2.md) and [fixed methods](../../reports/medtrace_stage2_20260908/METHOD_CONFIG_LOCKS.json). Original T2G decline is a measured behavior tradeoff, not a discovered binding error.

Run the focused CPU check with `python -m pytest -q tests/medtrace/test_stage2.py` in the existing project-compatible Torch environment. It checks target-free routing records, queue dependencies, exact execution-reuse boundaries and zero-denominator aggregation. The source scanner also has `--self-test`. The neutral entrypoint changes process-list display only; it is not anonymity or access control.

This snapshot is not a complete research checkout or a data/model release. Original V4 runtime/source locks and authorized private manifests, images, QA, teacher distributions and checkpoints are required to execute real experiments; none are published. Do not run a public packaged directory in place of the complete research worktree. No private Git history is copied.

The detached remote coordinator is bounded at 24h wall/48 GPU-h and prioritizes Judge/closure after 20h. `closeout_stage2_local.py` is a one-shot bounded publication continuation on the already-authenticated local host, not a cron or recurring monitor. It cannot keep a sleeping/disconnected local host online: remote work remains independent, but publication may be delayed. Full QA, raw answers, tokens, states, teacher distributions and Judge mappings are withheld. No full M3Bench, sequential, novelty/SOTA or clinical-safety claim.
