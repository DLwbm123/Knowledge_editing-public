# Stage16 source delta

Status: CPU decomposition executed; 133 exact author locality images recovered; GPU2 coverage runner prepared but **not started/runtime-validated** because the prelaunch memory check failed. See the [partial execution report](../../reports/medtrace_stage16_20260912/GPT_PRO_REVIEW.md).

This small source delta reuses the already published [Stage15 source](../medtrace_stage15_20260911) and its frozen model/runtime/Judge. No dependency installation, new model, optimizer, rank, writer loss or threshold is introduced. Do not replace the historical Stage15 outputs or run its worker again.

## Reproduction

Copy the Stage15 source directory to a separate working directory, then overlay this directory's `scripts/` and `tests/` folders. Use the existing compatible Python/model environments and privately configured approved data/runtime paths. The source-only public baseline redacts environment-specific paths; configure those locally using the same frozen model and Judge snapshots. Private raw QA, source images, weights and Judge mappings are deliberately not part of this release.

From that combined working directory:

```sh
python tests/test_stage16_contract.py
python scripts/medtrace/stage16.py --stage15-root "$STAGE15_RUN" --run-root "$STAGE16_RUN"
python scripts/medtrace/stage16_sources.py manifest --stage15-root "$STAGE15_RUN" --run-root "$STAGE16_RUN"
python scripts/medtrace/stage16_sources.py recover --manifest "$STAGE16_RUN/private/RECOVERY_MANIFEST.json" --destination "$STAGE16_RUN/private/recovered"
python scripts/medtrace/stage16_coverage.py prepare --stage15-root "$STAGE15_RUN" --run-root "$STAGE16_RUN" --recovered "$STAGE16_RUN/private/recovered"
python scripts/medtrace/stage16_coverage.py launch --run-root "$STAGE16_RUN"
```

These illustrate a **fresh reproduction**, not commands to rerun the prepared private campaign. The actual campaign has completed the first five stages and must resume at `launch` only, once GPU2 has enough memory. The resource guard requires 20 GiB free for the worker and later 34 GiB for the frozen Judge. The launcher uses neutral main/run/job process names and a detached coordinator; it creates no recurring monitor. A source failure remains unsupported rather than triggering a full download or substitute-image search.

The offline diagnostic script can read historical files without a GPU. It uses exact full generation/Judge context bindings and emits aggregate tables. B only loads existing checkpoints and generates restored probes. Its coordinator runs the frozen Judge after the student exits, then writes separate old/new/combined tables. Historical Judge entries are never rescored.

The appended interpretation in the report is human-readable analysis of the generated tables. In particular, post-hoc thresholds are diagnostic only. No RC_EXTCAL_V1, H/EXTRA pilot or confirmation cohort has been frozen or run; the current source/scope gap must be resolved first.

Sources/licenses: [MedMKEB](https://github.com/pkusixspace/MedMKEB/tree/d9f38639ec2285a0e9f541e22156ec14f87271d8), [PMC-VQA pinned author card](https://huggingface.co/datasets/RadGenome/PMC-VQA/blob/b56ae594f794867893143b337b4118a835794647/README.md). PMC-VQA is CC BY-SA; its source figures are described as CC0/CC BY. No source images or QA are redistributed here. GMAI gated access conditions are not automatically accepted.
