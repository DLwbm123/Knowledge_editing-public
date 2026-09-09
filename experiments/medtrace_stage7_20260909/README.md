# MedTRACE Stage7 APR source snapshot

Stage7 overrides the inherited Stage6 description below. [Stage7 results and limitations](../../reports/medtrace_stage7_20260909/GPT_PRO_REVIEW.md). Execution commit: `129d7c5b09567cb8f630d0bb6279477789439b63`; research branch: `medtrace-stage7-20260909`.

Stage7 entrypoints: `scripts/medtrace/stage7.py`, `scripts/medtrace/coordinate_stage7.py`; new writer: `methods/medtrace/anchor_repair.py`. Focused check: `PYTHONPATH=. python -m pytest -q tests/test_anchor_repair.py`. Use the existing compatible runtime and authorized private manifests/checkpoints; this is not a standalone data distribution.

All23 edits and Judge scoring completed. Finite anchor constraints passed, but original T2G regression remained. No additional SGD or automatic next experiment. Raw QA, images, tokens, activations, repair weights and Judge mappings remain private. The remaining files are unchanged shared dependencies from the published Stage6 snapshot.

## Inherited dependency provenance (Stage6, not Stage7 results)

[Results and limitations](../../reports/medtrace_stage6_20260909/GPT_PRO_REVIEW.md). Research branch: `medtrace-stage6-20260909`; result commit `6b47d46`. Actual remote inference/coordinator source: `55cbfb80c8bd53769202291bd876f89304677fc9`; matching local implementation through `6467aec`. These execution/history identities are distinct.

Entrypoints: `scripts/medtrace/stage6.py`, `stage6_worker.py`, `coordinate_stage6.py`. Focused check: `PYTHONPATH=. python -m pytest -q tests/test_stage6.py` in the existing compatible runtime. Shared dependencies reuse the published Stage5 snapshot, with no new training, dependencies, router or calibration algorithm.

The fixed old16 threshold is 0.7696741135364367. All eight real replay checks passed; Judge538/538, missing0. Source review covers127 new observations but only3 H observations across2 images; this is not broad independent hard-negative or clinical validation. New negative ON coverage is zero, so both costs and benefits of returning Base must be read together.

This source-only snapshot requires authorized private source-review inputs, runtime locks, data and existing checkpoints. It is not runnable standalone. Private QA, source-pair evidence, images, raw outputs/tokens, weights, credentials and private Git history are excluded. Shared dependencies retain path-default redactions to `/path/to/storage` and `/path/to/local`. No automatic next experiment is authorized.
