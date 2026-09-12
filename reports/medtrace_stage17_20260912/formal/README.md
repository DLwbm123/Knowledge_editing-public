# Stage17 authorized continuation — Base Judge running

The user approved the Stage17-only scope in [AUTHORIZATION.json](AUTHORIZATION.json). This supersedes the earlier Stage17-A pending-authorization statements, not historical results or locks. GPU authorization remains physical GPU3 only. Sealed/heldout data, T5, license acceptance and clinical signoff are not expanded.

## Verified milestone

- The Base candidate input pool was frozen before the new Judge call. [BASE_PREPARATION.json](BASE_PREPARATION.json) records 148 preliminary T0 candidates, 2,465 unique Base inputs and 50 batches across supported T0–T4 structures. These are not the final Base-wrong N or trainable method denominators.
- Existing Base outputs passed original question/image identity, reconstructed native prompt-token and raw continuation-token decoding checks. Runtime/generation/reference/raw-output/Judge lineage is bound to each opaque ID. No Base generation or historical experiment was rerun; no bulk image rehash was performed.
- The legacy heldout *superset* was resolved with a governance-only identity projection of 16 target and 16 locality records. QA, references, outputs and verdicts from those records were not retained, exported or judged. Canonical image aliases and existing image identities were closed over the exclusion; qualification reservations were also preserved. This is not a release of heldout inputs.
- The isolated `gpt-6-astra`, high-reasoning, ephemeral CLI queue has started. The first 50-item batch passed strict format and ordered-ID coverage validation; batch 2 was observed running. No semantic results have been used to change selection or settings. The immutable vendor model snapshot remains unknown/null.
- Seven relevant CPU checks passed, including an actual macOS read-denial check for another batch. Runtime preflight additionally checked operator, project and memory denial, with only the current batch input readable. No tools or outside context are provided to the Judge, and model reasoning streams are not retained.

## Important unfinished work

The candidate pool is a Base-scoring superset, not permission to train every listed native. Before any student output, the final role/exposure join must include the Stage2/Stage5 episode evaluation/calibration/challenge roles and Stage14's 72-image evaluation list, in addition to the existing Stage13R partition. Project evaluation-only inputs must not become natives or auxiliary training data. Any role-ineligible candidates must be excluded independently of scores; extra Base scores remain unused and must not be rejudged.

The final Base-wrong cohorts, whole-queue H/U/G isolation, E_U/E_H/E_HG/E_HG_eval masks, formal single/sequential adapters and GPU3 training have **not** completed. Preliminary task counts are not a common N panel. Unknown exposure is not claimed as unseen confirmation. No effectiveness or performance claim is made here.

The detached local Judge queue stops on completion or failure, without automatic semantic retry, model fallback, next-stage launch, or recurring monitor. Keeping the local host awake and network connected is necessary. The GPU3 mechanical check from Stage17-A remains valid, but GPU3 student training has not started.

## Reproduction and private boundaries

Use the [Stage17 source overlay](../../../experiments/medtrace_stage17_20260912/README.md), with the Stage15 and Stage16 overlays as documented. The new prepare entry takes private `storage_root`, `run_root`, `official_source`, and `authorization` paths. The Judge entry takes private `bundle`, `repository`, and `cli` paths. Run both through the existing neutral entrypoint, keeping real paths in `JOB_ARGV`/environment, not visible process arguments. The prepare runner refuses an existing formal directory; the Judge refuses existing attempts/responses. Do not rerun either completed preparation or accepted batches.

No credentials, raw QA/images/references/model answers/tokens, private source mappings, process identifiers, checkpoint or weights are published. The CLI isolation options follow the [official non-interactive documentation](https://learn.chatgpt.com/docs/non-interactive-mode); passing a schema does not establish semantic accuracy or clinical certification.
