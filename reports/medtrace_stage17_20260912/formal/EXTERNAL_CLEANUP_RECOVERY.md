# External worker cleanup recovery

The hourly check on 2026-09-13 found LoRA single complete at 146/146, with
5,360.59 seconds recorded for its generation phase. The worker stopped before
sequential because a protected, pre-existing SFTP process denied inspection
of its file descriptors. No cleanup plan or deletion had occurred.

The repair makes the external controller restartable without rerunning a
completed phase. It preserves the original computation commit and bindings;
only the separate controller changes. A specific file-visibility failure now
leaves an explicit cleanup-blocked receipt and allows the next already
authorized mode to run within the recorded 40 GiB lifecycle budget. Other
cleanup or validation failures remain fatal. No inaccessible process is
ignored and no checkpoint is deleted without the existing safety checks.

The relay recognizes generation-complete/cleanup-pending as a waiting state.
It still refuses import until cleanup succeeds for both modes. Hourly
maintenance retries the blocked cleanup; unresolved visibility is a reported
blocker, not authorization for permanent baseline checkpoint retention. No
SFTP session or unrelated process is terminated.

Regression coverage verifies that a completed single phase is not retrained
when cleanup is blocked, sequential is dispatched once, and the final state
remains cleanup-pending rather than claiming full GPU handoff completion.
