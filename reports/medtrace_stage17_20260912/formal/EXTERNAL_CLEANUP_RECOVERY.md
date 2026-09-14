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

## Resolved on 2026-09-14

The authorized hourly maintenance passed the existing strict open-file safety
checks and deleted the 146 completed FP16 LoRA single adapter weight files:
16,537,639,584 bytes (approximately 15.4 GiB). No process was terminated and no
file-descriptor visibility check was bypassed. The cleanup receipt records no
weight backup; recreating these weights would require retraining.

The phase passed provenance and cleanup validation. One post-cleanup check
confirmed all 146 enumerated weight paths were absent. Raw generations, tokens,
configuration, bindings and training/completion receipts remain. The separate,
user-paused BF16 sequential edit-100 checkpoint remains available; its 764
saved prefix-panel outputs were unchanged. The historical blocked receipt was
retained as resolved evidence, and hourly monitoring no longer retries this
completed cleanup. BF16 computation and its relay remain paused until an explicit
user recovery request. This resolves checkpoint lifecycle for FP16 single only;
it does not complete the mixed-precision LoRA pair or overall experiment.
