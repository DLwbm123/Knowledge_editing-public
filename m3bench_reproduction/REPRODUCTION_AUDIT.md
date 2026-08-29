# M3Bench reproduction audit

## Frozen sources

| Source | Remote | Branch | Commit | State |
|---|---|---|---|---|
| M3Bench | `https://github.com/BioMed-AI-Lab-U-Michgan/M3Bench.git` | main | `03c6fda3813301dab3be5831fdc94b493c10afc9` | clean |
| GRACE | `https://github.com/Thartvigsen/GRACE.git` | main | `f674183f17a995d109e10ee6140d4c3e6d016115` | clean |
| BalancEdit | `https://github.com/donglgcn/BalancEdit.git` | MMOKVQA | `83749e52a1d27331d21cfec845b6089294730c2f` | clean |
| MEND | `https://github.com/eric-mitchell/mend.git` | main | `e04fdb9cc784188906feffeb171025872933a5a8` | clean |
| LLaVA-Med source | `https://github.com/microsoft/LLaVA-Med.git` | main | `30697ca50b5c29a8e955c99330b259776aef27b9` | clean |

The LLaVA-Med model checkpoint is locally cached at `/remote-home/wangbomin/hugging_cache/medical_vlms/llava_med_v1_5_mistral_7b`. Its config identifies `mistralai/Mistral-7B-Instruct-v0.2`, `LlavaMistralForCausalLM`, bf16 checkpoint weights, native image padding, and `transformers_version=4.36.2`.

## Release vs paper discrepancies

1. **T5:** the release defines `delta_ACC` and says reference experiments did not run it. The paper reports temporality on 324 edits from 440 curated longitudinal images spanning 43 diseases, with an earlier edit evaluated on the later study. `release.yaml` and `paper.yaml` therefore use different T5 definitions.
2. **T2G:** release variants are paraphrase, negation/uncertainty, numeric perturbation, and distractor insertion. The paper specifies paraphrase, synonym/abbreviation, question-form changes, and telegraphic clinical-note style. They are not interchangeable.
3. **T1G:** release constructs blur/crop/contrast/noise perturbations. The paper's same-semantic image generality also encompasses clinically matched images and patient variation. The release transform set is not claimed to equal the paper task.
4. **T0 and 200 sequential edits:** the public builder selects every baseline-wrong case. The paper's model-specific 200 IDs/order are not released. The original ordering is recorded as `UNKNOWN`; the later seed-42 reproduction manifest must not be named or represented as official.
5. **Scope gap:** `build_all_tasks.py` explicitly does not run VLM inference, edit models, metrics, or LLM APIs. It is a task constructor, not a paper experiment runner.

## Metric contract

Paper-mode reported scores use free autoregressive generation only. Teacher-forced loss is restricted to LoRA optimization and diagnostics. Locality only admits pre-edit-correct probes; generality only admits pre-edit-wrong probes. Both are macro-averaged by edit request. Unit tests enforce these exclusions and reject accidental micro averaging.

## Data audit (read-only)

`outputs/logs/data_audit.json` records the full result. All 642 SLAKE metadata image IDs and all 314 VQA-RAD metadata image IDs resolve and pass PIL integrity verification. VQA-RAD has 2,248 metadata QA pairs, all of which match the raw source by image, question, and answer. SLAKE has 14,028 metadata QA pairs and no duplicate metadata question IDs; 44 pairs differ textually from the downloaded raw JSON (typically typo/wording corrections such as `kidnyes` versus `kidneys`). These are recorded as metadata/source discrepancies and are not rewritten or silently normalized.

## Known unresolved provenance

The exact trained MEND editor checkpoint and exact edit-training split used in the paper are not present in the public M3Bench release. MEND is therefore `BLOCKED` for exact paper reproduction unless those artifacts are obtained.

## Gate A execution audit (2026-08-11)

Gate A ran on GPU 3 using the frozen local LLaVA-Med checkpoint and local CLIP tower. The historical `outputs/logs/llava_gate_a.json` is deliberately retained unchanged: it proves two byte-identical but empty decoded strings from the former adapter.

The root cause was confirmed as Branch A, a generation-result contract error. In the frozen Transformers 4.36.2 multimodal path, `LlavaMistralForCausalLM.generate` returns a continuation-only sequence, beginning with BOS and then the first generated token. The former adapter incorrectly decoded `output[0, input_ids.shape[1]:]`; for the 21-token input this removed the answer and retained only `[28705, 2]` (whitespace/EOS). The adapter now declares and uses an explicit `continuation_only` contract, preserves raw IDs, and decodes the complete returned sequence. No checkpoint, prompt template, package, upstream source, or logits/EOS behavior was changed.

Exact frozen upstream `external/LLaVA-Med/llava/eval/model_vqa.py` was run with `--conv-mode mistral_instruct` against the same VQA-RAD image/question. It returned the nonempty text, “According to the image, there are no regions of the brain that appear to be infarcted.” Evidence is retained under `outputs/logs/gate_a_forensics/`, including the upstream input/output, static code comparison, implementation audit, token-level diagnostics, and the preserved original Gate A record.

The prompt forensic record is `'[INST] <image>\\nAre regions of the brain infarcted? [/INST]'`; tokenized input length is 21 and it has exactly one image token at position 5. The first actual generated token is 6586 (`According`), with EOS at the end. First-step logits contain no NaN/Inf; EOS ranks 669 with probability 8.267e-7, and direct-forward top-1 equals the generated first token. Thus premature EOS/logit failure, image-token failure, prompt-format failure, and checkpoint/data failure are not supported by the evidence.

The model-specific fix is guarded by source-only tests for Mistral prompt mode, image-token cardinality, im-start/end behavior, gold-answer exclusion, continuation-only decoding, and the former slicing failure. The test command passed 7/7 in the frozen environment. `manifests/llava_gate_a_three_cases.json` and `outputs/logs/gate_a_forensics/llava_gate_a_three_case_validation.json` record three fixed VQA-RAD cases, each repeated three times with greedy decoding. Every answer was nonempty, image tensors were finite, each prompt had exactly one image token, and text plus raw IDs were exactly deterministic.

Final Gate A label: `LLAVA_GATE_A_PASS__CONTINUATION_DECODE_FIXED`. No full baseline, task construction, LoRA, GRACE, BalancEdit, MEND, or 200-edit experiment has been launched; those remain outside this audited Gate A repair.

## Base-question manifest hard stop (2026-08-11)

The canonical metadata expander was run before any new inference. Its audit has the expected 16,276 records (14,028 SLAKE, 2,248 VQA-RAD), no duplicate `(dataset, image_id, question_id)` keys, no missing/unreadable image, and exactly 44 retained SLAKE source-text mismatch flags. It nevertheless fails the required non-empty-answer invariant: SLAKE metadata row `xmlab281_3`, global index 4467, has an empty released `gold_answer`. Because M3Bench metadata is the benchmark source of truth, filling it from a downloaded SLAKE JSON or an inferred value would be an unauthorized data substitution. The final JSONL manifest was intentionally not emitted.

The phase therefore ends at `LLAVA_NEXT_STOP__BASE_MANIFEST_INVALID`. `outputs/gates/base_manifest_invalid.json` and the corresponding builder logs preserve the evidence. Gate B, preprocessing audit, worker allocation, full raw baseline, judging, task construction, LoRA, GRACE, BalancEdit, MEND, and every edit experiment were not run.
