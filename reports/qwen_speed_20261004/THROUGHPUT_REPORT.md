# Qwen3-32B-AWQ throughput on RTX PRO 5000 72GB

Both predefined arms completed at 2026-10-04T20:15:56+08:00. Concurrency 32 processed 312.75 records/minute versus 54.72 at concurrency 2, a 5.72x measured throughput ratio. Each arm processed the same 200 frozen records once; 200/200 responses in each arm passed the full JSON format check. This measures throughput, not semantic accuracy.

| Concurrency | Records/min | 200-record time (s) | Output tokens/s | Input tokens/s | Load (s) | Warm-up (s) | GPU memory after generation (MiB) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 54.72 | 219.32 | 79.93 | 277.02 | 25.48 | 58.36 | 54718 |
| 32 | 312.75 | 38.37 | 456.88 | 1583.43 | 25.71 | 4.70 | 55878 |

## Measurement scope

The timed interval covers the complete 200-request generate call after eight unmeasured warm-up records. Both arms have 60,755 input tokens and 17,530 output tokens, with mean input length 303.775 and maximum 618 tokens. The model context is 2,048 tokens. Full supplied IDs and JSON fields are retained in constrained outputs.

Both runs use the same GPU UUID, inputs, model snapshot, AWQ Marlin kernel, FP16 activations, output cap 256, temperature 0, seed 0, eager execution, chunked prefill, 4,096 batched-token budget, and 0.75 memory allocation fraction. Thinking and prefix caching are disabled. Only max_num_seqs changes (2 versus 32). Runtime: vLLM 0.10.2, PyTorch 2.8.0 with CUDA 12.8, Transformers 4.55.2.

The arms use fresh processes and run sequentially, 2 then 32. On-disk compilation caches persist; the first warm-up took longer. Loading and warm-up are excluded from the reported throughput. This is one pass per arm, without timing variance estimates. The memory column is an observation after generation, not a measured peak; it includes allocated KV cache.

At the measured concurrency-32 rate, 12,978 similarly sized records would take about 41.5 minutes of generation. This is a linear projection, not a completed full-corpus run or a speed guarantee. Historical measurements on another GPU used different versions and inputs, so they are not a hardware-only comparison.

## Setup, validation and delivery

Model preparation preserved the source snapshot. Slow PyPI downloads were switched to the measured faster TUNA mirror in this isolated job. Two startup issues were repaired: vLLM required a numeric GPU selector rather than a UUID, and Triton required Python development headers. The fixed launcher verifies the actual CUDA device UUID. Matching Ubuntu libpython3.12-dev headers were extracted into the isolated environment; CPATH points to its headers/usr/include/python3.12 and headers/usr/include directories. No system package installation was needed. Failed startup logs and receipts are retained privately; no failed startup produced a measured 200-record result.

Completion checks confirmed both arm statuses and coordinator completion, 200 outputs in input order per arm, token counts matching raw outputs, throughput arithmetic, and successful process-name audits. The measured results were not retried or selected by score. Shutdown emitted NCCL process-group and nanobind cleanup warnings after result writing; both subprocesses exited successfully and the coordinator reached COMPLETE. A live post-completion check found GPU 6 at 0% utilization and 2 MiB allocated.

The model and isolated environment remain available. No benchmark decision enters the formal Astra comparison. Published files contain source, protocol, setup receipts, aggregate results and this report; private questions, reference answers, model responses, generated decisions and model weights are not published.

See [protocol](PROTOCOL.md), [mirror acceleration](INSTALL_ACCELERATION.json), [startup repairs](STARTUP_REPAIR.json), [JSON results](RESULTS.json) and [CSV results](RESULTS.csv).

## Observed Astra workflow comparison

The active gpt-6-astra high run had completed 137 batches / 6,850 records by 2026-10-04T12:56:55.378155+00:00. The manifest confirms 50 records per completed batch. From first batch start to last completed batch, elapsed time was 10369.01 seconds: **39.64 records/minute**. Mean completed-batch time was 75.49 seconds. The 260-batch queue was still running; this is a throughput snapshot, not its final result.

Relative to that observed Astra workflow, Qwen concurrency 2 is 1.38x and concurrency 32 is 7.89x as fast. These are different inference workflows: Astra uses high reasoning and serial 50-record cloud batches with process/network overhead, while local Qwen uses no thinking and up to 32 concurrent single-record prompts. Qwen measured 200 uniformly selected records; Astra's snapshot covers its first 6,850 records. This comparison does not isolate hardware/model effects or establish comparable semantic grading accuracy. No formal judge was replaced or stopped.
