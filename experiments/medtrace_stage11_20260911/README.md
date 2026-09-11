# MedTRACE Stage11 joint fact writer source snapshot

[Stage11 results and limitations](../../reports/medtrace_stage11_20260911/GPT_PRO_REVIEW.md). Execution source2569831, research branch medtrace-stage11-20260910. New entrypoints stage11.py, stage11_worker.py, coordinate_stage11.py under scripts/medtrace. LowRankExpert and the shared F1 loss primitives are inherited unchanged. Original authorized private Stage9 data/checkpoints/runtime and Judge locks are required; this is not a standalone data package.

All45 final trajectories,90 fixed step endpoints and Judge1171/1171 completed. Provisional development recommendation: free rank4 improves joint fitting and held-out pair correctness but increases U damage; rank16 adds no pair benefit. CPU interface tests: 2 passed (test_joint_fact_writer.py and test_fact_contrast.py). No next experiment launched. Private assets and private Git history excluded. Historical descriptions below document inherited dependencies, not Stage11 results.

## Inherited Stage10 provenance

[Stage10 results and limitations](../../reports/medtrace_stage10_20260910/GPT_PRO_REVIEW.md). Execution source `752d8179e05aa8eff2266d2ef751f22edc3dfed3`, research branch `medtrace-stage10-20260910`. Entrypoints: `scripts/medtrace/stage10.py` and `coordinate_stage10.py`; loss `methods/medtrace/anatomical_evidence.py`; shared worker `scripts/medtrace/stage9.py` is updated for this stage.

10 supported trajectories and Judge387/387 complete. No anatomical-direction-specific benefit over randomized direction was demonstrated; PairCorrect remains1/5. Runtime checks: `PYTHONPATH=. python -m pytest -q tests/test_anatomical_evidence.py tests/test_fact_contrast.py` (2 passed in the existing experiment environment). Requires authorized private manifests, original checkpoints and local SLAKE assets; not a standalone data distribution. Official annotation mapping: https://huggingface.co/datasets/BoKelvin/SLAKE/resolve/main/mask.txt (author-linked dataset, CC-BY-4.0). No images or masks redistributed.

Remaining files are inherited published dependencies, not new variants. Historical descriptions below refer to their own stages, not Stage10 results.

## Inherited Stage9 dependency provenance

[Stage9 results and explicit coverage gaps](../../reports/medtrace_stage9_20260909/GPT_PRO_REVIEW.md). Successful launch source: `1a955e4fefe168dcc349e8f91e6b5ca6f3485fc5`; training implementation `ecf3921`; research result `e6610f4` on `medtrace-stage9-20260910`.

New entrypoints: `scripts/medtrace/stage9.py`, `scripts/medtrace/coordinate_stage9.py`; loss/source checks: `methods/medtrace/fact_contrast.py`. Direct check: `PYTHONPATH=. python -m pytest -q tests/test_fact_contrast.py` (passed). Existing runtime and authorized private Stage8 manifests/W0 checkpoints/data are required. All other source files are inherited published dependencies, not new experiment variants.

45 supported trajectories and Judge1620/1620 complete;24 planned trajectories unsupported. Native degradation remains; contrast showed no added H correctness over the same-supervision baseline. Auxiliary evaluation ranking/token-cost/deduplication coverage gaps are disclosed. Private QA/images/tokens/weights/activations/Judge mappings and credentials excluded. No next experiment launched.

## Inherited Stage8 dependency provenance

Stage8 supersedes the inherited dependency description below. [Stage8 complete results and limitations](../../reports/medtrace_stage8_20260909/GPT_PRO_REVIEW.md). Execution source: `85f8e9f616a780ab6a611b7a544884e41df0a0ee`; research result branch `medtrace-stage8-20260909`, commit `518e357`.

Entrypoints: `scripts/medtrace/stage8.py` and `scripts/medtrace/coordinate_stage8.py`; new solver `methods/medtrace/bounded_repair.py`. In the existing compatible runtime: `PYTHONPATH=. python -m pytest -q tests/test_bounded_repair.py tests/test_anchor_repair.py` (3 checks passed). Shared dependencies are unchanged published Stage7 files. Authorized private Stage7 activations, manifests, data and checkpoints are required; this is not a standalone dataset distribution. No new dependency installation is required.

23 edits complete; Judge1879/1879. Decision PARTIAL_TRADEOFF, not a generalization solution or clear TR superiority. No automatic next experiment. Raw QA/images/tokens/weights/activations/Judge mappings and credentials are excluded.

## Inherited Stage7 dependency provenance

Stage7 overrides the inherited Stage6 description below. [Stage7 results and limitations](../../reports/medtrace_stage7_20260909/GPT_PRO_REVIEW.md). Execution commit: `129d7c5b09567cb8f630d0bb6279477789439b63`; research branch: `medtrace-stage7-20260909`.

Stage7 entrypoints: `scripts/medtrace/stage7.py`, `scripts/medtrace/coordinate_stage7.py`; new writer: `methods/medtrace/anchor_repair.py`. Focused check: `PYTHONPATH=. python -m pytest -q tests/test_anchor_repair.py`. Use the existing compatible runtime and authorized private manifests/checkpoints; this is not a standalone data distribution.

All23 edits and Judge scoring completed. Finite anchor constraints passed, but original T2G regression remained. No additional SGD or automatic next experiment. Raw QA, images, tokens, activations, repair weights and Judge mappings remain private. The remaining files are unchanged shared dependencies from the published Stage6 snapshot.

## Inherited dependency provenance (Stage6, not Stage7 results)

[Results and limitations](../../reports/medtrace_stage6_20260909/GPT_PRO_REVIEW.md). Research branch: `medtrace-stage6-20260909`; result commit `6b47d46`. Actual remote inference/coordinator source: `55cbfb80c8bd53769202291bd876f89304677fc9`; matching local implementation through `6467aec`. These execution/history identities are distinct.

Entrypoints: `scripts/medtrace/stage6.py`, `stage6_worker.py`, `coordinate_stage6.py`. Focused check: `PYTHONPATH=. python -m pytest -q tests/test_stage6.py` in the existing compatible runtime. Shared dependencies reuse the published Stage5 snapshot, with no new training, dependencies, router or calibration algorithm.

The fixed old16 threshold is 0.7696741135364367. All eight real replay checks passed; Judge538/538, missing0. Source review covers127 new observations but only3 H observations across2 images; this is not broad independent hard-negative or clinical validation. New negative ON coverage is zero, so both costs and benefits of returning Base must be read together.

This source-only snapshot requires authorized private source-review inputs, runtime locks, data and existing checkpoints. It is not runnable standalone. Private QA, source-pair evidence, images, raw outputs/tokens, weights, credentials and private Git history are excluded. Shared dependencies retain path-default redactions to `/path/to/storage` and `/path/to/local`. No automatic next experiment is authorized.
