# CrispEdit native medical-VLM comparison

Completed eight sequential edits per arm. Both arms edit language layers 19–23, use 25 steps at most per edit and lr=5e-4, and retain the same native FP16 deployment. CrispEdit uses the pinned official ProjectedAdam with online K-FAC statistics; Adam is the matched unprojected control. Vision stays frozen. This is a declared VLM adaptation.

## Outcome

CrispEdit retained more edited targets than the matched Adam control (4/8 versus 1/8 after eight sequential edits), but this pilot does **not** demonstrate safe knowledge editing. Both arms initially answered each inserted target correctly (8/8); subsequent edits lost 4 targets with CrispEdit and 7 with Adam. In the separate medical holdout, fresh Base answered 61/64 correctly, while both edited models answered 0/64 correctly: retention of currently correct answers is 0/61 for both arms.

The holdout was selected using the original Base labels. Fresh evaluation with the fixed Qwen judge marked three of those 64 answers incorrect before editing, so 61 is the appropriate denominator for newly broken answers. All new Base and edited outputs were scored with the same Qwen setup. This is source-group separation, not verified patient independence.

CrispEdit also has lower text holdout NLL than Adam (2.9596 versus 13.0207), but both worsen from Base (1.8307). Its relative advantage does not offset the observed medical holdout failure. These results concern one eight-edit native medical-VLM adaptation; they neither reproduce nor refute the original paper's text-model benchmark claims. No tuning or follow-on run was performed.

| Arm | Task | Correct / eligible | Edit macro |
|---|---|---:|---:|
| base | T0 | 0/8 | 0.00% |
| base | T1G | 0/29 | 0.00% |
| base | T1L | NA (n=0) | NA |
| base | T2G | 0/32 | 0.00% |
| base | T2L | NA (n=0) | NA |
| base | INDEPENDENT_HOLDOUT | 61/64 | NA |
| crisp | T0 | 4/8 | 50.00% |
| crisp | T1G | 13/29 | 50.00% |
| crisp | T1L | NA (n=0) | NA |
| crisp | T2G | 8/32 | 25.00% |
| crisp | T2L | NA (n=0) | NA |
| crisp | INDEPENDENT_HOLDOUT | 0/64 | NA |
| adam | T0 | 1/8 | 12.50% |
| adam | T1G | 4/29 | 12.50% |
| adam | T1L | NA (n=0) | NA |
| adam | T2G | 4/32 | 12.50% |
| adam | T2L | NA (n=0) | NA |
| adam | INDEPENDENT_HOLDOUT | 0/64 | NA |

Zero-denominator locality panels are unsupported, not evidence of preservation. The separate 64-query holdout is selected before editing from originally correct, source-disjoint queries and never used for optimization or curvature statistics.

Current Base holdout correctness: 61/64. Fresh Base label changes across all queries: 3. Exact-token changes versus Base: {'crisp': 187, 'adam': 187}.

Generation/editing wall time: 1622.18 seconds. Qwen scored 455 unique inputs covering 577 occurrences; no prior-run verdict is reused.

crisp: insertion targets 8/8; successful insertions lost at final evaluation 4; newly broken current-Base-correct holdout answers 61; text holdout NLL 2.959586 (Base 1.830676). Fresh native reload: {'state': 'GENERATED_NOT_SCORED', 'edits': 8, 'queries': 187, 'native_reload_logit_error': 0.0, 'native_reload_tokens_identical': True}. Temporary-state deletion: {'files': 7, 'bytes': 13002364573, 'retained_copy': False, 'consumers': ['all_ordered_edits', 'insertion_and_final_generation', 'text_holdout', 'native_reload'], 'reconstruction': 'recompute frozen statistics and rerun fixed edits'}.
adam: insertion targets 8/8; successful insertions lost at final evaluation 7; newly broken current-Base-correct holdout answers 61; text holdout NLL 13.020692 (Base 1.830676). Fresh native reload: {'state': 'GENERATED_NOT_SCORED', 'edits': 8, 'queries': 187, 'native_reload_logit_error': 0.0, 'native_reload_tokens_identical': True}. Temporary-state deletion: {'files': 6, 'bytes': 4110433160, 'retained_copy': False, 'consumers': ['all_ordered_edits', 'insertion_and_final_generation', 'text_holdout', 'native_reload'], 'reconstruction': 'recompute frozen statistics and rerun fixed edits'}.

## Limitations

- Medical VLM adaptation, not a reproduction of original text-LLM benchmark numbers
- Eight developmental edits, one fixed order, no hyperparameter search
- Original Base eligibility is Astra-derived; new outputs uniformly Qwen-scored
- Upstream empirical CE-gradient covariance path, not an exact GGN estimator
- Independent holdout is disjoint by recorded source group, not guaranteed patient independence

Raw medical queries, answers, tokens, mappings and checkpoints are excluded from publication. See PROTOCOL.md for precision, data, optimizer and sequential-history adaptations; RESULTS.json contains complete deidentified per-step receipts. No follow-on tuning or additional experiment is authorized by these results.

## Execution and validation

- Started: 2026-10-05T11:42:53.196313+08:00; workflow completed: 2026-10-05T12:11:59.050947+08:00.
- Total workflow wall duration: 1745.85 seconds (29.10 minutes), including editing, generation, scoring preparation, scoring and aggregation. This is wall time, not measured GPU-active time.
- All four stage exit codes are zero. The existing report validator was rerun to verify input/output bindings, lock identity, verdict provenance and complete coverage of 577 mapped occurrences by 455 unique verdicts. No semantic rescoring or retries were performed.
- Both arms passed exact fresh-native reload checks. Noneditable parameters and buffers stayed frozen.
- Generated temporary checkpoint/cache files were deleted by the completed workflow after their consumers finished (17,112,797,733 bytes across both arms); shared pretrained models and earlier experiments were untouched.
- Public artifacts include protocol, source, aggregate JSON/CSV, per-step numerical receipts and stage execution receipts. Medical questions/answers/tokens, private mappings, model weights and unlicensed upstream source are excluded.

`RESULTS.csv` reports pooled fractions separately from the edit-macro averages in this report. Original T1L/T2L have zero eligible support and remain NA.
