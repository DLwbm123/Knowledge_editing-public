# C_NO_H priority Astra scoring — 2026-09-15

The user requested scoring their method before the remaining baseline queue.
The completed C_NO_H single and sequential outputs passed the existing cohort,
Base-mask, generation, training-binding, prefix-role and query-coverage checks.
The priority packet contains 5,732 unique bound student judgments in 115 batches
of at most 50. It covers N=146 single edits and sequential prefixes 1/50/100/146,
with R0 primary, RC secondary and single FORCED_ON diagnostics. Exact bound Base
outputs on inactive routes reuse accepted Base correctness.

The detached scoring worker started at 15:35 China time. The first batch entered
an isolated fresh session using gpt-6-astra/high, the unchanged source-agreement
prompt and Judge lock, and the existing CLI version 0.154.0-alpha.6.2. All five
filesystem isolation probes passed. No tools, project context, memory or old
verdicts are available to the Judge. No semantic retry or model substitution is
enabled. This records startup, not completed scoring or a performance result.

The previous CPU-only overall follower was waiting for GPU completion and had
not prepared a Judge packet. Its replacement first waits for accepted priority
judgments, then for the original GPU campaign. It revalidates the complete Judge
execution, identical scientific bindings and Judge lock, excludes those IDs from
new requests, and incorporates the accepted decisions into the final report.
Changed or incomplete priority evidence stops the flow. GPU workers and the
LoRA pause were not changed.

After successful scoring, the priority worker generates a C_NO_H-only report
with the existing metric implementation and attempts sanitized publication under
`reports/medtrace_stage17_20260912/cnoh_priority_closeout`. Other-method results
and whole-campaign completion are not claimed. Full C_FACT remains unsupported
on this main cohort because H support is absent.

Validation: six campaign tests pass, including scoped-report coverage and refusal
to reuse incomplete, changed or non-Boolean priority verdicts. The actual priority
packet passed all preparation checks before launching. Neutral parent and child
command lines were checked. Private raw questions, answers, tokens, bindings,
runtime paths and credentials are excluded from this public report.
