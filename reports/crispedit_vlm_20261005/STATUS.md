# Complete: scored comparison

Run `20261005C1_8` completed at 2026-10-05T12:11:59.050947+08:00 after 29.10 minutes of workflow wall time. Both arms completed all eight sequential edits and all 187 final queries. Qwen scored 455 unique inputs covering 577 occurrences; all stage exits and report binding/coverage checks passed.

CrispEdit retains 4/8 edited targets versus Adam 1/8. Both arms lose all 61 currently Base-correct independent medical holdout answers. The result does not demonstrate preservation; no follow-on tuning or experiment was launched.

See [REPORT.md](REPORT.md), [RESULTS.csv](RESULTS.csv) and [RESULTS.json](RESULTS.json). Code, protocol and sanitized results are delivered on `research/directw-evidence-v1` in the public repository. Raw medical inputs/outputs, private mappings and model weights are excluded.
