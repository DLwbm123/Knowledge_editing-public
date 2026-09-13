# Student Judge: bounded transport recovery

2026-09-13: first five batches (250 records) passed. Batch 006 failed after
about 327 seconds with `idle timeout waiting for SSE`, without a final response.
The original queue stopped and its responses and failure evidence are preserved.
This is a transport failure, not a semantic score or a proven proxy root cause.

The user authorized repair and continuation. The existing recovery runner now
permits this exact SSE-idle failure only with explicit authorization and no final
response, preserves the accepted prefix, and uses a separate write-once recovery
directory. The amendment is bound privately to the exact failed input and Judge
configuration. Four CPU/isolation/recovery tests passed.

Recovery changes only transport: explicit credential-free loopback HTTP/HTTPS
proxy, system-proxy discovery disabled, and a 900-second SSE idle timeout instead
of the documented 300-second default. Provider retries remain zero. No global
proxy setting, node, model, high reasoning effort, prompt, batch or JSON schema
was changed. Longer idle tolerance is not a claim of faster inference or a cure
for upstream stalls. No repeated semantic sampling is allowed.

Recovery batch 006 entered a fresh isolated session in a detached process.
Status: **RUNNING, NOT COMPLETE**. Accepted prefix: 5/48 batches. Remaining:
43 batches, 2,128 records. No GPU work or periodic monitor was started.

Implementation commit: `6c1bb8c`. Private QA, answers, tokens, mappings and raw
execution evidence are excluded from public release. The existing final merge
runs only after full format/coverage validation; no extra experiments follow.
