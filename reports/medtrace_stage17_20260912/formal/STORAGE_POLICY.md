# Bounded storage preparation — 2026-09-13

The data filesystem reports 118G total, 90G used and 29G available. Finite storage
cannot retain an unlimited number of checkpoints; this is not a promise of unlimited
unattended execution. No existing artifact, model or environment has been deleted,
moved, quantized or overwritten by this change. The running Qwen sidecar is unchanged.

## Implemented, for a future clean and explicitly locked runtime

The single-BE admission check now budgets missing checkpoint files, not another full
copy of every checkpoint already on disk. Existing state still passes the normal
load/binding checks. Both dispatch admission and each edit retain an 8GiB free-space
allowance in addition to the estimated new checkpoint requirement. Low space stops
before a new edit; it does not delete evidence or fall back to the system partition.
The existing atomic temporary checkpoint write remains unchanged. This is a planning
guard, not a filesystem quota or an atomic reservation against other writers.

The active remote source lock is not silently changed. This patch must be included
in the next verified clean runtime commit and dispatch lock; no experiment is started
by this storage preparation. A regression check covers missing versus existing state.

## Proposed lifecycle — archive destination and deletion targets still need approval

- Keep active Base/vision weights and runtime once; do not create per-edit full-Base
  copies. Preserve each method's exact expert representation and precision. Changing
  BE to a low-rank approximation is not a storage-only optimization.
- Keep current/resume state, pending Judge inputs, outputs, tokens and provenance.
  Completed experts needed for sequential insertion replay must remain recoverable.
- Offload closed checkpoint groups to an approved private archive. Check transfer
  exit status and one small availability/readability check. Do not release the source
  until the exact source deletion is approved; no bulk hash is needed by default.
- Qwen weights plus its separate environment occupy about 27G. They remain active
  while the sidecar runs. Once it completes, offloading or removing this exact stack
  is a candidate only if future local Qwen judging is not needed and the user approves.
- The earlier pip download directory is about 3G; inventory its actual contents and
  dependencies before requesting removal. Do not infer that every download is disposable.
- The completed single-BE checkpoints total about 32G. They are important for possible
  sequential replay; do not delete them merely because scoring finished.

Use the GPU server as active working storage and another approved disk as archival
storage. Process methods in bounded batches when their checkpoint budget requires it;
never silently skip requested edits to fit the disk. Archive unavailability should
stop further large writes, not trigger deletion or re-training. GitHub is for approved
code/aggregates, not private checkpoints or raw evaluation materials.
