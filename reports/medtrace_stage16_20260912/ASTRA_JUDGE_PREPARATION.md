# Stage16: user-stopped Qwen Judge and prepared Astra materials

Status: **PREPARED_NOT_JUDGED**. No Astra calls, new training, generation, historical
rescoring, or experiment completion are claimed by this preparation.

On 2026-09-12 the user explicitly requested stopping the current Judge and preparing
materials and a prompt for GPT Astra with a correct output format. The specific Qwen
Judge and its waiting coordinator received SIGTERM; their absence was verified at
07:26:54 UTC. Other workloads were not stopped. The original run records the user
stop and retains all prior inputs, generated outputs, checkpoints and verdicts.

The original pending packet contains 326 unique tuples with zero reused verdicts.
It is split without rewriting questions, references or candidate answers into six
50-record batches and one 26-record batch. Only the existing opaque ID, question,
gold answer and candidate answer are exposed to the scoring task. Original
adjudication labels and full tuple mappings remain operator-only.

The semantic rubric is retained, but the target model, runtime, context batching
and response format are explicitly a new protocol:
`medtrace-stage16-astra-semantic-v1`, requested model `gpt-6-astra`.
No immutable Astra snapshot or actual execution configuration is asserted before
the eventual run. A blank execution-record template must be filled from real
operator evidence, not a model's self-identification.

## Output acceptance

- Per batch: one JSON object with exactly `batch_id` and `decisions`.
- Each decision: exactly `opaque_query_id` and JSON boolean `is_correct`.
- Required: matching batch, original ID order, complete coverage, no duplicate
  keys/IDs, no unknown IDs, no extra fields or Markdown, no coercion of null,
  strings or numbers to booleans.
- All batches are validated before publishing an independent Astra JSONL output;
  existing results are not overwritten. Source adjudication labels are restored
  by the operator's script, never inferred by the model.
- Tests use only synthetic records/verdicts and exercise malformed responses,
  source-text tampering, merge coverage and non-overwrite behavior. Passing these
  tests certifies structure/coverage, not semantic accuracy or model provenance.

## Scope and remaining work

The private delivery includes judge-only batch prompts, inputs, JSON Schemas,
operator instructions, a standard-library validator/merger and source backups.
Private QA, candidate answers and mappings are excluded from source control and
must not be published. Only a genuinely isolated, authorized cloud context may
receive the judge-only materials. Preparing this bundle does not upload it or
launch a new task, web session, model, or API request.

Do not overwrite the old Qwen output path or call the existing Stage16 report
entrypoint with Astra judgments: it would combine new semantic scores with
historical Qwen scores. The eventual Astra-scored restored subset needs a separately
labelled downstream report. Historical Qwen results are preserved and are not
rescored under this authorization. No method, threshold or new performance gate
is added; required human clinical review remains distinct from automated scoring.

Ponytail guidance kept the change to a standalone standard-library helper and one
focused test file, reusing existing fields and semantic rubric. OpenAI Docs
guidance informed explicit output-only instructions and context isolation; no
new API client, SDK, environment or authentication setup was introduced.

Sources: [Astra model guidance](https://developers.openai.com/api/docs/guides/latest-model?model=gpt-6-astra),
[Codex structured output](https://learn.chatgpt.com/docs/non-interactive-mode).
