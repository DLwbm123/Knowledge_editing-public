# Exploratory 146-edit comparison with mixed judges

The user requested switching the remaining queue from Astra/high to Qwen3-32B-AWQ at concurrency 32. Accepted Astra scores are retained; the remaining inputs are scored once by Qwen. Per-record provenance is retained privately, and primary-panel judge counts are in HYBRID_SEMANTIC_RESULTS.json. Historical baselines and Base eligibility masks remain Astra-based. Differences below may reflect both the editing method and the judge; this is not a uniform-judge ranking.

New-record judge coverage: {"gpt-6-astra": 7150, "Qwen/Qwen3-32B-AWQ": 5828}. Values are edit-macro percentages; C_NO_H is the no-H version.

## single

| Method | T0 Fix | T1G Fix | T2G Fix | T2L retention |
|---|---:|---:|---:|---:|
| single_layer | 52.05 | 51.71 | 8.39 | 90.32 |
| multi_layer | 51.37 | 50.68 | 9.02 | 96.77 |
| C_NO_H | 100.00 | 100.00 | 91.95 | 48.39 |
| balancedit | 100.00 | 100.00 | 95.89 | 56.99 |
| belora | 80.14 | 79.11 | 67.29 | 71.51 |
| lora | 100.00 | 100.00 | 99.66 | 47.31 |

## sequential (prefix 146)

| Method | T0 Fix | T1G Fix | T2G Fix | T2L retention |
|---|---:|---:|---:|---:|
| single_layer | 78.08 | 75.68 | 23.40 | 51.08 |
| multi_layer | 86.30 | 85.45 | 29.28 | 55.91 |
| C_NO_H | 100.00 | 86.47 | 90.58 | 48.39 |
| balancedit | 100.00 | 86.47 | 94.86 | 63.44 |
| belora | 80.14 | 69.18 | 66.55 | 71.51 |
| lora | 0.00 | 0.34 | 0.34 | 0.00 |

## Limits

- User-authorized mid-queue judge switch: accepted Astra prefix retained; only the remaining inputs were judged by Qwen3-32B-AWQ
- Judge assignment follows queue order, not randomization; method/task comparisons may be confounded by grader changes
- Historical baselines and Base eligibility masks remain Astra-based; hybrid scores are exploratory, not a uniform-judge confirmatory comparison
- Paired confidence intervals quantify edit variation only and do not account for differences between judges
- C_NO_H has no H; not full C_FACTH
- Historical immutable model snapshot unavailable; new and old verdict epochs differ
- Historical hardware/runtime differ
- LoRA single FP16 and sequence BF16; original FP16 sequence failed edit17 step4
- One fixed edit order; no order robustness or full patient independence
- AlphaEdit VLM adaptation with bounded text-only projection statistics
- Source-answer agreement is not clinical validation

Do not rank solely by target Fix: generalization, locality, paired uncertainty and supported denominators are required. Generation costs and adverse outputs are recorded separately in GENERATION_REPORT.md.
