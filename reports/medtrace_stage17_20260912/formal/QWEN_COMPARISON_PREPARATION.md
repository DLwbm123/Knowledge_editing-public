# Independent Qwen/Astra Judge agreement sidecar

2026-09-13 user authorization: run the Qwen Judge on the new 24GB server as
well, to compare its judgments with Astra. This does not replace, rescore or
mix the Stage17 primary Astra verdicts. Agreement is not clinical accuracy.

Status: **PREPARATION RUNNING, GPU JUDGING NOT YET VERIFIED**.

The same 2,378 student input/reference/raw-answer tuples and opaque source IDs
were projected from the frozen Astra inputs. No Astra decisions, method names,
training records or historical scores were copied into the Qwen packet. The
packet and new sidecar configuration are frozen before Qwen execution. A
synthetic trust-boundary test rejects packets containing a verdict field.

Target: RTX 4090 physical GPU0, 24,564MiB total and 24,082MiB free at preparation.
The actual project data filesystem is ext4 with 55GB free and passed a small
write/read probe. The model and vLLM are absent on this target. Only the missing
fixed Qwen snapshot is being fetched; existing medical model, environments,
training outputs and checkpoints remain untouched. No cleanup was performed.

Model: `Qwen/Qwen3-32B-AWQ`, revision
`0499c3ac83fdef8810b907a23894ba91e95eddd8`, matching the historical Qwen model.
The original host's copy was located, but the new host cannot reach Hugging Face
directly; the fixed revision is fetched through the HF mirror. Original license
and metadata are included. A separate environment pins vLLM 0.9.2 and
Transformers 4.53.2 (the historical versions); pip resolves vLLM's Torch 2.7.0
dependency without replacing the existing training or provider environment.

The detached preparation job installs dependencies and obtains the model, then
checks GPU UUID/free memory and starts the scorer without an SSH dependency.
An initial virtualenv creation error caused by using a symlink as the environment
root was corrected by placing the environment below that alias. Only that
preparation process was restarted, with partial model downloads retained; no
Qwen inference had started and no semantic judgments were repeated.

## Fixed 24GB lane and comparison limits

- Same semantic rubric as Stage17 Astra, same full question/reference/raw answer.
- Qwen non-thinking chat template; one record per context, maximum concurrency 1.
  Astra uses high reasoning and up to 50 records per context. Therefore this is
  a comparison of the two configured Judge pipelines, not a pure model-only test.
- AWQ/FP16, eager execution, memory utilization 0.90, chunked prefill with at most
  512 batched tokens, no prefix caching, seed 0 and temperature 0.
- Full-packet tokenizer preflight selects 1024/2048/4096 context; no truncation.
  Output cap is 256 tokens. Guided decoding permits exactly the two Boolean
  JSON responses carrying the one supplied ID. Raw outputs/tokens are retained.
- Write results in groups of 50, expose final JSONL only after complete coverage.
  No implicit semantic retries. Sample board memory every two seconds and record
  loading/total time. Initial free memory does not yet establish peak or feasibility.

Output namespace: `VERDICTS_QWEN.jsonl`, independent of `VERDICTS_ASTRA.jsonl`.
After both are available, compare exact common IDs, 2x2 decisions, agreement and
disagreements. Missing/failed judgments must not be counted as false or agreement.

At the preparation check, Astra had preserved 14/48 accepted batches (700
records) and stopped on batch 015 with a network response-body decoding error.
This sidecar preparation did not stop or restart Astra. Full comparison remains
pending; raw materials/identities/answers/weights are excluded from publication.
