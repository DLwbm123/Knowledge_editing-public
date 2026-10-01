# GPU5 continuation and autonomous follow-up

## Latest state — 2026-10-01 E10 completion verified on user status request

E10 is COMPLETE, PID1358793 exited; GPU5 idle/free72830MiB. All8/8 accepted edits exactly replayed E9B numerical trajectories, all8 clean native consumers had logits error0/tokens equal/no editor import. All temporary matrices deleted after final consumers; all Base restorations passed, no traceback/failure receipts. New145GGN/1119.9031787399435seconds; cumulative2274GGN/15675.648759132833seconds; Judge0.

The diagnostic all-probe hypothesis FAILED: two of16 slots exceeded KL0.001 (same-image extra QA after cases3/6:0.014059590 and0.001564787). Both corresponding answer scores improved (+0.121012419,+0.034415841), so do not claim harm or clinical forgetting. No score drop exceeded0.1; worst drop-0.023121357.16 slots reuse only3distinct QA, all scopes UNKNOWN, all images previously used. Full audited report/receipts now in `reports/directw_evidence_v1_probe_diagnostics_20261001/FINAL_REPORT.md`, FINAL_RESULTS.json and PROBE_SUMMARY.json. Existing native edit acceptance remains valid; diagnostic violations were not suppressed.

No worker is currently running. Next hourly continuation should freeze and execute a full eligible-pool diagnostic under the existing user delegation: enumerate every original audited train/CANDIDATE_ONLY QA eligible per case (exclude target/reference/smoke IDs and question hashes), rather than choosing just the first QA per image role. Categorize same-image, protected-reference-image and third-image QA explicitly, with unknown scopes. Retain exact same8 E9B edit/reference pairs, algorithm/seed20261002/three steps, full trajectory fingerprint checks and clean consumers; use a fresh E11 RUN_ROOT and freeze finite wall/GGN/storage budget plus memory margin before launch. Existing probe loop supports this without source changes. Preserve all current negative observations: the new diagnostic is broader coverage, not a new claim that all KL<=0.001. Report distributions and any score changes below-0.1 on previously unprobed QA, with per-case slots and distinct QA denominators separate. Original scopes/visual pairs/independent evaluation remain BLOCKED_DATA; no clinical generalization or best-seed selection. If the full pool cannot fit budget/memory, freeze a justified smaller diagnostic before any execution, never silently truncate results.

The preceding E9 result remains stochastic8/8 versus nearest2/8 on8QA/3images, and E8 seed replication remained3/8 below its4/8 criterion. Do not erase those bounds. At next wake-up read actual run roots first in case another authorized continuation has already launched. No duplicate start; GPU5 only, other tasks untouched.

## User authority, 2026-10-01

The user explicitly assigned DirectW to pro5000 physical GPU5 and other experiments to GPUs6/7, requested hourly monitoring, authorized autonomous diagnosis/repair, and authorized designing and executing subsequent experiments once the current experiment completes. This supersedes prior manual-review stops for this exploration. It does not supply missing scientific data or permit fabricated medical counterexamples.

## Historical E2 continuation

Run `20261001E2`, PID at launch `1223994`, resumes E1's four unfinished independent cases. E1 and its four negative outcomes remain immutable. Resume code verifies identical model, selected matrix, data/selection, scientific protocol and solver settings, a contiguous terminal case prefix, restored Base and cumulative accounting before any model load. It re-executes the mechanical native regression, then starts at zero-based case index 4. No experiment hyperparameter or acceptance threshold changes.

Prior consumed time: 990.1430820168462 seconds, including E1 and earlier smoke runs. Prior GGN: 161. Remaining E1+E2 engineering allowance: 13,554 seconds and 3,981 GGN calls. These preserve the previously declared combined 4-hour/4,096-call exploratory ceiling; failed work is counted. GPU UUID: `GPU-4924bbd8-4082-3f4c-b4fb-78ca930122ca`. Prelaunch check requires at least 66,000 MiB free. The recorded previous maximum PyTorch allocation was about 57.78 GiB. No other process is killed or migrated.

The driver runs detached with neutral argv. It writes atomic status at phase/case boundaries and preserves exceptions, rollbacks and native checkpoint-consumer receipts. All 37 CPU tests passed, including rejection of changed scientific inputs, reset resource ledgers and malformed inherited case prefixes. See RESUME_CPU_TESTS.json and RESUME_BINDINGS.json.

## Hourly execution

The active Codex thread automation `directw-gpu5` checks this task hourly. Each wake-up must consult current user instructions, the latest status and live process/GPU evidence, not a stale PID or this launch snapshot. Diagnose actual errors and test minimal repairs before recovery. Preserve negative scientific results. Do not restart an active worker, reuse failed result records as successes, or reset the wall/GGN ledger. GPU6/7 are excluded.

At full completion, reconcile the complete eight-case denominator including inherited cases, publish sanitized metrics and the report, and inspect the distribution of physical BF16 score changes, rounding, solver residuals, protection KL and rejection causes. Mechanical regression PASS means the implemented mechanical checks passed; its rejected edit does not prove edit success or clean export qualification. An accepted scientific edit still needs native clean reload/generation parity.

## Subsequent experimental design

The next protocol must be written and bound before execution, based on all eight outcomes. Its initial aim is to distinguish BF16 deployment quantization and FP32/BF16 objective mismatch from insufficient optimization or protection pressure. Choose a small matched control or diagnostic that addresses the observed bottleneck; do not blindly expand cases or relax acceptance to manufacture success. Any algorithm, damping, precision or step-budget change belongs to a separately named run with a stated hypothesis and retained baseline, not a retroactive repair of E1/E2.

Use fixed source-verified training cases for developmental comparisons, identify overlap explicitly and retain all outcomes. Keep formal evaluation isolated and the permanently reserved smoke cases out of scientific counts. When visual evidence pairs or scope annotations remain unavailable, preserve BLOCKED_DATA for the unsupported scientific branches. Do not invent medical negatives or claim training-support outcomes generalize clinically. Judge remains zero.

The autonomous agent may implement and run this next bounded exploration under the user's explicit delegation. It must first freeze case selection, metric definitions, comparisons, stopping criteria, finite wall/GGN/storage ceilings and code/config bindings; run relevant CPU and native correctness checks; then launch only on GPU5. A failed hypothesis calls for reporting and a separately justified follow-up, not silent retuning. If no meaningful next experiment can run with available data, report that blocker instead of consuming GPU indefinitely.

## Publication and storage

Use the independent research/directw-evidence-v1 branch and a new remote RUN_ROOT per protocol or repair attempt. Do not load CP/LoRA adapters or optimizer state. Commit and push source, protocols, tests and sanitized result/report artifacts through the required GitHub proxy, verifying remote SHA and public access. Raw medical QA, images, generated tokens, checkpoints and credentials stay private. Delete only this task's generated matrices after their last required native consumer, retaining minimal receipts and resume evidence.
