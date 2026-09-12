# Stage16 artifacts — supported coverage complete, Astra-scored extension

Start with the updated [GPT Pro review report](GPT_PRO_REVIEW.md) and [aggregate results](STAGE16_REVIEW_AGGREGATES.json). A is complete. B restored 133 official locality probes, completed 266 writer-probes using existing checkpoints, and obtained 326 accepted Astra judgments. C/D source/scope requirements remain unmet; no four-arm pilot, new threshold or independent confirmation is claimed.

The [Astra execution summary](ASTRA_EXECUTION_SUMMARY.json) and [protocol amendment](ASTRA_PROTOCOL_AMENDMENT.json) disclose the Judge change and rejected first attempt. Historical Qwen semantic scores and new Astra semantic scores remain separate. Only Judge-independent output preservation is combined across the old 67 and new 133 probes.

The original A diagnostic CSV files remain unchanged. Updated source-component sensitivity for the coverage extension is in the aggregate JSON, not a rewrite of historical tables.

`PROTOCOL_AND_METHOD_LOCK.json` preserves the original prelaunch lock; its Qwen-only clause is superseded for the new restored subset only by `ASTRA_PROTOCOL_AMENDMENT.json`. `COVERAGE_AND_COST.json` records current completion. H/source/role and calibration audits remain unchanged. Remote operational markers have not been relabelled; completion here is supported by actual output and scoring artifacts.

See [reproduction instructions](../../experiments/medtrace_stage16_20260912/README.md). Source-only release: no images, raw QA/answers/tokens, per-item mappings, credentials or checkpoint files. Prior preparation status remains in Git history. Publication verification is recorded separately after pushing, not implied by a pre-push status field.
