# MedTRACE FreshStart source snapshot

The `freshstart/` files here are the core experiment, metric, and report source from execution checkout `7a9b1f22dc410dd8322b5813aaee935c93c38a79`. They are copied without code changes. The private `judge_bridge.py` transport adapter is excluded because it binds a particular authenticated workstation and server; the shared Judge protocol lives in `scripts/medtrace/`.

The executed runtime used a fixed private state directory. Re-execution requires the original authorized data, model, environment, state layout, and Judge inputs, none of which are included here. This source snapshot documents the implementation and does not imply that the incomplete scoring or unavailable independent panels can be reconstructed from public files alone.

See the [result and coverage report](../../reports/medtrace_freshstart_20260922/FINAL_RESULTS_ZH.md).
