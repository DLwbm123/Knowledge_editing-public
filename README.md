# Knowledge Editing for Medical Vision-Language Models

This repository is a public, source-only research snapshot of the `Knowledge_editing` project. It contains implementation code, tests, configuration files, and selected experiment reports. Datasets, model weights, checkpoints, raw generations, judge packets, and runtime outputs are intentionally excluded.

## Current experiment snapshot

The latest MedTRACE release is [Stage15: MedMKEB external evaluation](reports/medtrace_stage15_20260911/GPT_PRO_REVIEW.md), with [source and reproducibility boundaries](experiments/medtrace_stage15_20260911/README.md). C_NO_H and BalancEdit each completed 200 independent edits; Judge2,426/2,426. Native output match is100% for both. R0 preserves roughly96% cross-image generality, but the primary frozen RC accepts only2/164 such probes and reduces success to1.22%. C_NO_H improves R0 image locality by5.97pp on67 supported edits. No legal H-supported requests were available, so C_FACT and C_EXTRA_QA are unsupported, not externally validated or failed. This is a release-aligned subset, not author-execution parity or a clinical evaluation.

The [Stage14 supervision-matched closeout](reports/medtrace_stage14_20260911/GPT_PRO_REVIEW.md) and its [source](experiments/medtrace_stage14_20260911/README.md) remain unchanged. Seven extra-QA controls completed; Judge1,416/1,416. Selected H facts outperform this generic extra-QA control on forced paired correction, while generic QA helps U correctness and FACT retains a same-answer challenge cost. Fixed old RC ties all four systems. That comparison is a post-publication supplement on the same new7 cohort, not a fresh blind confirmation.

The original [Stage13R new7 confirmation](reports/medtrace_stage13r_20260911/GPT_PRO_REVIEW.md) and [Stage13 N0 audit](reports/medtrace_stage13_20260911/GPT_PRO_REVIEW.md) are unchanged. Stage14 preserves the complete old15 and new7 historical panels and does not retrain prior methods.

### Earlier recovery snapshot

The earlier results are in the [visual-verifier R1 recovery report](reports/medtrace_visual_verifier_recovery_20260907T045800Z/GPT_PRO_REVIEW.md): 21/21 tasks and 42 verifier fits closed; none of M1–M3 passed the predeclared development retention signal. The [2026-09-07 experiment index](reports/current_experiments_20260907/README.md) preserves the earlier startup failure, LoRA-Perf `QUAL_VALIDATION_FAIL`, and the MedTRACE execution-preservation results.

The corresponding source snapshots are `experiments/lora_perf_20260905/`, `experiments/medtrace_execution_preserving_20260906/`, and [the visual-verifier recovery source](experiments/medtrace_visual_verifier_20260907T045800Z/README.md). Their checks and reproducibility limits are documented in the closeout reports.

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
