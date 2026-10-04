# MedTRACE execution and storage

Before scheduling experiments or deleting artifacts, read
`reports/medtrace_stage17_20260912/formal/CHECKPOINT_LIFECYCLE.md`.
Its user-approved lifecycle supersedes the earlier archive-by-default storage proposal.
Apply it to both the rented GPU server and my-gpu; shared storage is not unlimited.
Do not change frozen scientific protocols, stop active runs or delete historical
checkpoints merely by reading these instructions.

## Default judge for future scoring

User decision (2026-10-04): future scoring in this project defaults to
**Qwen3-32B-AWQ with concurrency 32**, using the validated local vLLM setup
documented in `reports/qwen_speed_20261004/THROUGHPUT_REPORT.md`.
Use the tested model revision, AWQ Marlin, thinking disabled, temperature 0,
and the complete structured response with supplied record IDs. Reuse the prepared
environment and model; check GPU identity and available memory before each run.

Record this judge choice when preparing each new scoring protocol and report the
model revision, prompt, runtime settings and provenance with its results.
Keep Astra-labelled and Qwen-labelled results distinguishable; this throughput
benchmark does not establish scoring equivalence. The already-running Astra
comparison continues under its original frozen protocol. This preference alone
does not launch a new run or authorize rescoring historical results.
