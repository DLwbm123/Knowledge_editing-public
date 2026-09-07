# Selective-write V1 public source snapshot

Reviewer-facing dependency snapshot from research branch `medtrace-selective-write-20260907`; resume implementation `c4397a7` (original training/cache binding `0c72cfe`, closure/scheduling `5106c14`).

See [the in-progress review](../../reports/medtrace_selective_write_20260907/GPT_PRO_REVIEW.md) and [protocol](../../reports/medtrace_selective_write_20260907/SELECTIVE_WRITE_PROTOCOL.json).

The five selective-write source/test files reuse the established runtime, hook lifecycle, queue and Judge dependencies included here. Historical pooling/verifier helpers are dependencies/reference code only; no new router is trained. Run CPU checks from this directory with `python -m unittest discover -s tests/medtrace -p test_selective_write.py -v` in the existing project-compatible Torch environment.

This snapshot is not a complete research checkout or a data/model release. Original V4 runtime/source locks and authorized private manifests, images, QA, teacher distributions and checkpoints are required to execute real experiments; none are published. Do not run a public packaged directory in place of the complete research worktree. No private Git history is copied.

Updated execution snapshot: 12 RAW_READY, 2 RUNNING, 56 PENDING. After the user's pause and explicit GPU2/3 authorization, e02 P4/W0 and L16/W0 resume endpoint generation from step320 with zero additional optimizer steps. Original results and clocks are preserved; paused time is excluded via a separate active-time overlay. Seven CPU tests pass. See USER_RESUME_STATUS.json, RESUME_AFTER_USER_PAUSE.md and LIVE_PROGRESS.json in the reports. The full 70-task/Judge run is not complete. New-edit confirmation N=0. No final method-effectiveness, full M3Bench, TIME/LiveEdit/M-ORE reproduction, novelty, SOTA or clinical-safety claim.
