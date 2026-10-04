# Qwen3-32B-AWQ throughput on RTX PRO 5000 72GB

User request (2026-10-04): prepare the previously used open model and measure
scoring speed on pro5000. This is a bounded inference benchmark, not a replacement
for the active Astra semantic comparison. No GPU editing/training is rerun.

- Reuse Qwen/Qwen3-32B-AWQ revision
  `0499c3ac83fdef8810b907a23894ba91e95eddd8` from the existing my-gpu snapshot.
  Preserve its source, license and metadata. Copy only this snapshot to pro5000.
- New isolated Python environment with vLLM 0.10.2, Transformers 4.55.2,
  NumPy 2.2.6 and OpenAI client 1.99.9; do not modify existing
  experiment environments. vLLM's documented prebuilt CUDA 12.8 distribution
  supports the Blackwell runtime requirement. Runtime compatibility remains
  subject to actual model-load and generation checks.
- GPU 6 UUID is fixed in the private manifest. Recheck at least 57,000 MiB free
  before each arm; do not stop other workloads. Storage stays under /data/bmw.
- Select 200 evenly spaced records across all 12,978 frozen A2 Judge inputs,
  using indices `floor(i*(N-1)/199)` for i=0..199. No selection based on verdict,
  answer quality or performance. Each context contains one question, reference,
  candidate answer and opaque ID with the original source-agreement rubric.
- Run concurrency 2 then 32 on the same records in separate fresh processes.
  Each arm gets eight unmeasured warm-up inputs, followed by one measured pass
  over all 200 inputs. Repeat inputs are for throughput measurement only; their
  decisions never enter the formal score table. No automatic rerun by quality.
- Qwen thinking disabled; FP16 activations, AWQ Marlin backend, temperature 0,
  seed 0, 256 output-token cap. Constrained decoding retains the original full
  JSON response with supplied ID and Boolean; it does not substitute a shorter
  Boolean-only output. Validate every generated response.
- Prefix caching disabled, eager mode enabled, chunked prefill enabled,
  max_num_batched_tokens=4096 and memory utilization 0.75 for both arms. Choose
  a 2048/4096/8192 context from actual token lengths plus the full output budget.
  No input truncation. Each benchmark subprocess is bounded to 30 minutes.
- Report model load and warm-up time separately from measured records/minute,
  input/output tokens per second, actual package versions and board memory after
  generation. The latter is an observation, not a sampled whole-run peak.
- Historical 24GB measurements used vLLM 0.9.2, different kernel/runtime settings
  and inputs. This comparison is not a hardware-only or concurrency-only ablation
  against that historical run. The new two-arm comparison holds its input and
  engine settings fixed except concurrency.

Copy validation uses transport exit codes, one count/known-byte-size check and
normal model-index loading. No full model hash is calculated. Retain the prepared
model/environment for the user's use and small raw benchmark outputs privately;
publish only code, protocol, status and deidentified throughput results.

Sources: [vLLM GPU installation](https://docs.vllm.ai/en/v0.10.2/getting_started/installation/gpu.html),
[vLLM AWQ inference](https://docs.vllm.ai/en/v0.10.2/features/quantization/auto_awq.html).
