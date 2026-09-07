# Knowledge Editing for Medical Vision-Language Models

This repository is a public, source-only research snapshot of the `Knowledge_editing` project. It contains implementation code, tests, configuration files, and selected experiment reports. Datasets, model weights, checkpoints, raw generations, judge packets, and runtime outputs are intentionally excluded.

## Current experiment snapshot

The latest verified results are in the [2026-09-07 experiment closeout](reports/current_experiments_20260907/README.md): LoRA-Perf completed with `QUAL_VALIDATION_FAIL`; the MedTRACE execution-preservation campaign completed without improving the routing/execution tradeoff. The newer visual-verifier R1 campaign failed at startup and remains incomplete.

The corresponding source snapshots are `experiments/lora_perf_20260905/` and `experiments/medtrace_execution_preserving_20260906/`. Their checks and reproducibility limits are documented in the closeout report.

The earlier M3Bench editor paper-spec runtime is retained:

- `experiments/m3bench_editor_paperspec/m3bench_repro/editors/`: LoRA, GRACE, BalanceEdit, and BELoRA paper-spec implementations for the frozen LLaVA-Med runtime.
- `experiments/m3bench_editor_paperspec/scripts/`: setup, CPU-gate, smoke-input, and GPU-runtime entry points.
- `experiments/m3bench_editor_paperspec/tests/editor_paperspec/`: mechanics and state-roundtrip tests.

These implementations are independent paper-spec adaptations/reimplementations. They must not be described as unreleased author implementations. In particular, the packaged BELoRA implementation is explicitly labeled `BELORA_PAPER_SPEC_REIMPLEMENTATION_V1`.

## Experiment reports

- [M3Bench editor paper-spec final status (2026-08-28)](reports/M3BENCH_EDITOR_PAPERSPEC_FINAL_STATUS_20260828.md)
- [Current experiment status, LoRA-Perf and MedTRACE results (2026-09-07)](reports/current_experiments_20260907/README.md)
- [M3Bench V4 Foundation closure (2026-08-28)](reports/M3BENCH_V4_FOUNDATION_CLOSURE_20260828.md)
- [LiveEdit-Med router R1 oracle final results (2026-08-17)](reports/LIVEEDIT_MED_ROUTER_R1_ORACLE_FINAL_RESULTS_20260817.md)
- [LiveEdit-Med reproduction summary (2026-08-14)](reports/LIVEEDIT_MED_REPRODUCTION_SUMMARY_20260814.md)

The reports preserve their original scientific status labels. A runtime or smoke-gate PASS is not a claim that formal single-edit or 200-edit comparative evaluation has completed.

## Repository layout

- `methods/`, `src/`, `easyeditor/`, `KE/`: core and adapted knowledge-editing code.
- `scripts/`: experiment, evaluation, and audit entry points.
- `m3bench_reproduction/`: M3Bench reproduction utilities and protocol material.
- `experiments/`: dated implementation snapshots, including the earlier paper-spec runtime and the completed LoRA-Perf and MedTRACE campaigns.
- `tests/`: project tests.
- `reports/`: selected status and result reports.
- `third_party/`, `external/`: retained upstream/adapted sources with their own provenance and licensing terms.

## Minimal mechanics check

From the repository root:

```bash
PYTHONPATH=experiments/m3bench_editor_paperspec \
python -m unittest discover \
  -s experiments/m3bench_editor_paperspec/tests/editor_paperspec \
  -p 'test_*.py'
```

Full GPU experiments require separately obtained datasets, model checkpoints, and an environment matching the paths and versions documented in the corresponding report.

## Data and artifact policy

This public snapshot does not include:

- medical images or benchmark datasets;
- pretrained model weights or adapters;
- checkpoints, caches, logs, or raw `outputs/` trees;
- private/blind judge inputs, decisions, or temporary staging material;
- local archives, PDFs, IDE state, or machine-specific runtime files.

## Licensing

`VLKEB_LICENSE` applies to the VLKEB-derived portion of the codebase. Files under `third_party/` and `external/` remain subject to their respective upstream terms. No additional project-wide license is asserted by this snapshot.
