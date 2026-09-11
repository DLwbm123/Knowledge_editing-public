# Stage13R: frozen new-source evaluation

[Results and limitations](../../reports/medtrace_stage13r_20260911/GPT_PRO_REVIEW.md).

Seven new edits and21 writer branches completed; Judge214/214. The forced H-evaluation PairCorrect edit-macro gain from C_FACT over C_NO_H is85.71 percentage points, but fixed old RC makes the three deployed systems tie. C_FACT loses three same-answer challenge successes relative to the controls. This is a small source-image confirmation with unknown patient identity, not full-M3Bench or clinical validation.

New entrypoints are `scripts/medtrace/stage13r_sources.py`, `stage13r.py`, and `coordinate_stage13r.py`; the shared `stage11_worker.py` adds the fixed two-writer, step320, training-only branch. Other files reuse the already-published Stage12 dependency snapshot. No additional algorithm variants were run.

## Provenance and checks

- Source preparation:75022408936cda2ae20be4f93486b2178bb03ad4.
- Training launch:12da8f7b0f284740304c8401b540c7379a0c5af9.
- Generation/reporting and source-import completeness fix:677342c744e15e310a55ac7cc66585ec1b252d78.
- Two focused source/role/rank checks passed in the existing experiment environment. Check source: `tests/test_stage13r_sources.py`.
- Existing compatible runtime check: `PYTHONPATH=. python -m pytest -q tests/test_stage13r_sources.py`. No dataset, model download or GPU training is performed by these checks.

The curator reads the full authorized original training source, distinguishes never-executed drafts from actual development, preserves old protected identities, and creates prospective image roles before native Base-wrong eligibility. It does not revoke reservations because data are public. The original challenge importer cap was corrected within the frozen evaluation partition before evaluation generation; original files and the correction ledger are preserved. Reproducing the exact original run requires that recorded correction as well as the frozen input manifests.

## Reproducibility boundary

Source code and aggregate tables are supplied, not a standalone dataset distribution. Authorized private historical exposure ledgers, full raw training sources, frozen episode manifests, runtime/model/tokenizer locks and Judge ancestry are necessary to reproduce the exact cohort. Inherited path defaults are redacted to `/path/to/storage` and `/path/to/local`; configure actual authorized storage and GPU identity. Do not use this snapshot to reclassify sealed/evaluation-only material or overwrite completed runs.

The training process consumes only native/fit answer files. Evaluation starts in a separate process after training completes. This is a code-path boundary, not an OS-level access-control sandbox. Native CP -> A2 -> original CP-W0 initialization is retained and costed before the paired freeR4 continuations. BalancEdit uses its original frozen adaptation and different supervision budget, not a paper-exact claim.

Private QA, images, outputs/tokens, activations, checkpoints, weights, Judge mappings, credentials and private Git history are excluded. No automatic next stage is authorized.
