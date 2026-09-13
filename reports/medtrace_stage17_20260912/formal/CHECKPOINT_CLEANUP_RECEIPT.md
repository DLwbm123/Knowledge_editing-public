# Completed targeted checkpoint cleanup — 2026-09-13

On the user-authorized my-gpu host, deleted exactly 200 Stage15 BalancEdit
`editor_state.pt` files after validating completed Stage15 result/training records,
Stage16 restored-generation results (133 BE results), and the lack of a Stage17
reuse dependency for these different editing requests. Stage16 Astra results remain
available locally. This does not authorize re-running any experiment.

- Deleted logical file size: 46,979,870,448 bytes (43.75 GiB).
- Pre-deletion NFS-reported allocated blocks: 93,962,240,000 bytes (87.51 GiB).
  This is not a measured storage-backend quota reduction; physical accounting may differ.
- The explicit-file plan and deletion journal are preserved privately on the source
  host and copied locally. Each path was checked for unchanged inode/size/mtime,
  symlinks, hardlinks, filesystem and open process references. Post-deletion exact
  target existence checks passed. No recursive directory removal or bulk hashing.
- Process-check limitation: a non-dumpable SFTP process exposed only standard
  descriptors 0/1/2 and no open transfer file; its cwd/maps were inaccessible.
- No backup was created; deleted weights cannot be restored directly. Rebuilding
  requires the original frozen training inputs/runtime. Published metrics remain valid,
  but historical checkpoint-based reruns must now recognize the deletion receipt.

Preserved: result JSON, outputs/tokens, Judge materials/verdicts, reports, training
records, router states, all own-method checkpoints, Base/vision/Judge weights,
datasets, running jobs and the rented server's Stage17 checkpoints needed for replay.
Other historical checkpoint groups remain unmodified until their dependencies can
be positively established. This is a targeted cleanup, not a claim that all project
storage has been fully audited or minimized. Large old feature/teacher caches were
not relabeled as checkpoints and deleted under this request.
