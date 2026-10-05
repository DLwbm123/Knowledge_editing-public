# Running: startup verified

Run `20261005C1_8` started at 2026-10-05 11:42:53 Asia/Shanghai on GPU 7.
The original four-hour wall deadline is 15:42:53; this is a cap, not an ETA.
The detached workflow runs editing, native evaluation, Qwen scoring and report
aggregation in dependency order. An hourly monitor handles operational failures
and completes public delivery within the frozen scope and deadline.

Startup confirmed the native model loaded and Base generation progressed to
64/187 queries in the captured check, with 15,108 MiB allocated, live neutral
Python command lines and no failure receipt. This checkpoint establishes startup,
not completed native optimization, semantic scores or improvement.

CPU numerical checks passed locally and in the native runtime environment;
both editing and judge imports passed. The two arms are CrispEdit-Seq and matched
Adam on the first eight edits, with language layers 19–23, at most 25 steps per
edit, and frozen vision. The 187-query evaluation includes a separate 64-query
originally-correct holdout. See PROTOCOL.md for exact source/precision/data limits.
