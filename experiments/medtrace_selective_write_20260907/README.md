# Selective-write V1 public source snapshot

Current status: STAGE1_COMPLETE, 70/70 trajectories judged, all processes exited normally at 2026-09-08 01:23:58 Asia/Shanghai. Source 5981c54 uses stage-specific free-memory thresholds; nine CPU tests pass. See RUN_COMPLETION.json and RESULTS_SUMMARY_ZH.md in the reports. Timestamped process/pause/resume snapshots are historical, not current running state.

Reviewer-facing dependency snapshot from research branch `medtrace-selective-write-20260907`; neutral process entrypoint `5ef3a16`, checked at `8d3a344`; endpoint-resume implementation `c4397a7` (original training/cache binding `0c72cfe`, closure/scheduling `5106c14`).

See [the numerical result summary](../../reports/medtrace_selective_write_20260907/RESULTS_SUMMARY_ZH.md), [review](../../reports/medtrace_selective_write_20260907/GPT_PRO_REVIEW.md) and [protocol](../../reports/medtrace_selective_write_20260907/SELECTIVE_WRITE_PROTOCOL.json).

The selective-write source/test files reuse the established runtime, hook lifecycle, queue and Judge dependencies included here. The neutral entrypoint adds display-only main/run/job process names without renaming source or data directories. Historical pooling/verifier helpers are dependencies/reference code only; no new router is trained. Run CPU checks from this directory with `python -m unittest discover -s tests/medtrace -p test_selective_write.py -v` in the existing project-compatible Torch environment; nine tests pass.

This snapshot is not a complete research checkout or a data/model release. Original V4 runtime/source locks and authorized private manifests, images, QA, teacher distributions and checkpoints are required to execute real experiments; none are published. Do not run a public packaged directory in place of the complete research worktree. No private Git history is copied.

All 70 Stage-1 tasks and fixed Judge scoring are complete. Original results and clocks were preserved during two user-requested pauses. P4 calibrated W1 shows reduced collateral damage; W2's benefit is parameterization-dependent and its fit constraints are not universally satisfied. New-edit confirmation N=0. This is not a full M3Bench, TIME/LiveEdit/M-ORE reproduction, novelty, SOTA or clinical-safety claim.
