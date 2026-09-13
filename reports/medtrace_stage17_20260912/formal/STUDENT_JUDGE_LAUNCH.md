# Stage17 first student Judge dispatch

2026-09-13: **STARTED, NOT COMPLETE**. The completed BalancEdit single-edit
main-T0 dispatch has 146 receipts and 2,378 edit-query outputs. Training is
not rerun. Other methods, task-specific dispatches and sequential runs remain
pending; this is not full Stage17 closeout.

The existing isolated Codex CLI runner now scores these 2,378 forced-expert
outputs in 48 batches using `gpt-6-astra`, reasoning `high`, the unchanged
`MEDTRACE_STAGE17_SOURCE_AGREEMENT_V1` prompt and strict Boolean JSON schema.
The exact Base Judge lock is retained; immutable backend snapshot is unknown.
User-config isolation remains enabled and system-proxy support is explicitly
enabled. This does not by itself certify a shared public egress IP with the app.

All 7,134 mode assignments remain mapped: 7,045 point to the 2,378 student
identities and 89 OFF-mode assignments reuse exactly bound, accepted Base
outputs/verdicts. Reuse is based on realized input, active expert and exact
output/tokens, never answer strings alone. No verdict is selected by score.

Preparation checked the frozen queue, complete edit/query coverage, training
receipts, runtime/generation bindings, Base masks and routed outputs. Four
CPU/isolation tests passed. The actual CLI version is `0.154.0-alpha.6.2`.
The detached neutral-named parent was alive with parent PID1; batch 001
entered a fresh session with all five isolation checks passing. No GPU is
required for this cloud scoring queue. No new monitor was created.

Runner/source commit: `bda0bee`. Raw QA, answers, tokens, identities,
checkpoints, private execution packets and credentials are not published.
Transport/format failure stops the queue without automatic semantic retries.
The last batch triggers the existing strict merge, not additional experiments.
