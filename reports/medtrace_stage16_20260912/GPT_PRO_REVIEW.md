# MedTRACE Stage16 — partial execution and prepared GPU2 continuation

Status: **A complete; B prepared but NOT started; C/D unsupported by current source/scope packages.** This is not a completed Stage16 GPU experiment.

## Executed

- Verified the Stage15 public baseline `0c51b959c80dcda205e14c2de92b7ae43c6e8dcf` against its immutable queue, 400 checkpoint paths, 2,514 completed writer-probes, actual inputs and 2,426 frozen full-context Judge identities. No old generation or scoring was repeated.
- Completed route/writer decomposition, all 1,354 ordered threshold breakpoints, separate positive/locality AUROCs, locality paired contingencies, source-component sensitivity, and portability-versus-Base tables. A used CPU only.
- Recovered **all 133 missing image-locality images** by exact author-archive member names. Read only **19,395,536 bytes** using HTTP ranges, including ZIP metadata and one already-bound identity control. Native ZIP CRC was checked; the existing image control matched its historical identity. No full archive was downloaded; original images/crops were preserved.
- Froze 133 existing edits × C_NO_H/BalancEdit = **266 new writer-probes**. Reuse the own-edit Stage15 writer and router; no training. Official questions and references are unchanged. Old-supported, newly restored and combined results will remain distinct.

## Not executed and why

| Branch | Verified state | Reason |
|---|---|---|
| B new locality generation/Judge | Prepared, **0 new outputs / 0 new judgments** | GPU2 changed from 40,445 MiB free during preparation to 4,788 MiB at launch. The existing worker guard requires 20,480 MiB; Judge later requires 34,816 MiB. The guard stopped before any project GPU process was started. |
| B missing image-generality | 36 probes remain unsupported | Exact GMAI source unavailable under current gated access. No terms accepted or access bypassed. |
| C new source/scoped support | Not frozen | 60 additional native records resolve in the existing assets, 56 without known Stage15 image-role overlap; these are only candidates, not a clean cohort. No newly verified H/G+ and independent fit/calibration/evaluation package is available. |
| D four-arm pilot/confirmation | Not run | Conditional legal-support requirements not met. No H loss removed, no FACT/EXTRA negative performance conclusion inferred. |

No automatic GPU polling or monitor was created. No other GPU or existing process was changed. Preparation and historical results are preserved for an explicitly requested continuation.

## Scientific reading of A

Old RC retains only 2/164 image-general successes for either writer, versus 158/164 or 159/164 at R0. Its 67/67 image-local output preservation is achieved by turning all those probes OFF. The writer and router effects must not be conflated.

Historical R0 locality preservation favors C_NO_H by 4 discordant pairs: +5.97 pp with the original bootstrap interval [1.49, 11.94] pp, but supplemental exact two-sided p=0.125. Source-semantic accuracy is a different endpoint (+2.99 pp; interval [0, 7.46] pp; exact p=0.5). These do not establish strong independent external FACT effectiveness.

The cached margins suggest that the old threshold is poorly transferred: image-general versus image-local AUROC is 0.981616 on 164/67 probes. A less restrictive **post-hoc diagnostic** region retains 152/164 image-general activations with 5/67 image-local activations. No point is adopted or renamed as a new primary; independent Base-only development data is still required.

Portability exact and semantic outcomes diverge. One-hop exact improves from 0/73 to 1/73, while semantic accuracy drops from Base 21/73 to 7/73 or 8/73. This is not evidence of reliable multi-hop transfer.

Known observed source connections produce 151 components across the 200 edits. These are source-image/preprocessed-image/article proxies, not 151 known patients. Unobserved near-duplicates and patient/study identity remain UNKNOWN. The single reused U-fit QA is not an independent 200-example U evaluation.

## Deliverables and boundaries

Public: Stage16 source, CPU contract check, frozen protocol, source/role/access audits, aggregate diagnostic tables, and this partial-status report. The coverage runner has passed CPU syntax/contract checks but **has not yet been GPU runtime-validated**. No new accuracy or GPU-completion claim is made.

Private: raw QA and images, outputs/tokens, per-item Judge/checkpoint bindings, patient information, model/checkpoint files, and environment paths. No sealed data was opened. Stage15 files were read-only. No threshold tuning, new algorithm, extra qualification gate, or next-stage run was performed.

Sources: [MedMKEB pinned release](https://github.com/pkusixspace/MedMKEB/tree/d9f38639ec2285a0e9f541e22156ec14f87271d8), [PMC-VQA author archive/license](https://huggingface.co/datasets/RadGenome/PMC-VQA/blob/b56ae594f794867893143b337b4118a835794647/README.md), [GMAI access and withheld TEST answers](https://huggingface.co/datasets/OpenGVLab/GMAI-MMBench).
