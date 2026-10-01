# GPU5 continuation and autonomous follow-up

## Latest state — 2026-10-01 17:15 China-time follow-up

E6 completed all8 cases with **2/8 accepted native FP32 edits**,6NOT_SATISFIED with rollback. Both successful FP32 matrices passed independent clean native reload (logit difference0, generated tokens equal, no editor imports) and were deleted after the last required consumer. Every case/final worker restored Base. Native/functional FP32 parity0; old BF16 absolute-score parity is inapplicable. Final reference KL tiny negative values around-6e-8 are numerical roundoff, preserved as reported. E6 final report/results are published in `reports/directw_evidence_v1_fp32_control_20261001/FINAL_REPORT.md`, commitd1412aa. This is FP32 precision feasibility, not BF16 or medical generalization. Cumulative ledger now1334GGN/8522.28328455775seconds.

Current run **20261001E7**, launch PID1309549, is the BF16 stochastic-rounding ablation. Read `reports/directw_evidence_v1_stochastic_rounding_20261001/PROTOCOL.md` and source/config bindings (source commitbfe5c46). Revert to original BF16 native loader/image pipeline; same eight train cases and three steps as E3, initial BF16/FP32 scores must match E3 within1e-6. Only selected-matrix writes use adjacent-value stochastic rounding with fixed seed20261001+1009*case_index+step (mechanical index10000), one uniform tensor per step reused across backtracking. No favorable resampling/seed search. Same native-value QP, gain0.1, KL0.001, drift0.3 and all mandatory actual native acceptance/export checks. Stochastic weight rounding does not make nonlinear outputs unbiased. Missing verified visual data/scopes and independent evaluation remain BLOCKED_DATA.

42 CPU tests passed. Initial live check14seconds after launch: PID alive with neutral argv `/data/bmw/envs/s0/bin/python -`, onlyGPU5 UUID GPU-4924bbd8-4082-3f4c-b4fb-78ca930122ca, stateNATIVE_REGRESSION,0/8 scientific cases. Log present without traceback and no failure receipts; native regression result was still pending. Do not claim native validation passed from CPU tests or initial launch. New ceilings7200seconds/1024GGN/storage20GiB/cache2GiB; prior cumulative costs retained. Prelaunch required66000MiB free; observed initial loading use15548MiB is not peak. Other GPUs untouched.

Next heartbeat checks actual PID/status/failures/counters and receipts once, without duplicate launch. If complete, audit all8 outcomes including exact BF16 Base controls, stochastic seeds, requested/actual drift, native repeatability, rollback and every accepted matrix's clean consumer, then publish. A null result is retained and requires a separately justified bounded next protocol; do not repeat seeds until success or relax acceptance. Engineering failures require preserved evidence and diagnosis before recovery. The earlier E1–E6 protocols and outcomes remain immutable; E4 already rejected simple blind step-budget escalation. The historical E2 continuation below is retained only as context.

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
