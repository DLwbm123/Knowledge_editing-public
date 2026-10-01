# GPU5 continuation and autonomous follow-up

## Latest state — 2026-10-01 18:15 China-time follow-up

E7 completed8/8 with **5/8 accepted native BF16 edits**, against E3 nearest-rounding0/8. Accepted cases0,2,5,6,7 all passed independent clean native BF16 reload with logit difference0, exact generated-token equality and no editor imports. All five temporary matrices were deleted after final consumers. Cases1,3,4 were BACKTRACK_REJECTED and restored. All8/final Base restored, E3 initial BF16/FP32 control-score parity passed, every native repeat exact. No runtime failure, native mechanical checks PASS (mechanical edit NOT_SATISFIED, excluded).30 scientific trials:9 accepted intermediate steps and21 rejected (13 protection,8 no merit improvement). E7 final report/results published commitf0e3434 in `reports/directw_evidence_v1_stochastic_rounding_20261001/FINAL_REPORT.md`. Cost154GGN/1247.582582seconds; cumulative1488GGN/9769.865869862726seconds. Peak per-case allocation58.002GiB. These are train-support feasibility results for one seed; missing scientific data still BLOCKED_DATA.

Current run **20261001E8**, launch PID1320879, is a **prespecified seed replication**. Read `reports/directw_evidence_v1_seed_replication_20261001/PROTOCOL.md` and bindings, frozen commitc97a08a. No implementation change: E7 sourcebfe5c46 and42 passing CPU tests reused after source-binding equality check. Only base rounding seed changes to20261002. Case seed20261002+1009*case_index+local_step; mechanical index10000. Same8cases/3steps/native BF16/FP32functional solver/coupled draws/acceptance gates. Initial scores must match E3. Replication criterion frozen before launch: at least4/8 accepted BF16 edits. Report overlap with E7 and both seeds' outcomes; do not pick best seed, retry adverse draws or call16 case-seed trials16 independent cases.

Initial live check15seconds after launch: PID alive, neutral argv `/data/bmw/envs/s0/bin/python -`; GPU5 UUID GPU-4924bbd8-4082-3f4c-b4fb-78ca930122ca, stateNATIVE_REGRESSION,0/8 scientific cases. Log present/no traceback, no failure receipts; native validation remains pending at that snapshot. GPU used15548MiB/free57283MiB at loading, not peak. Prelaunch66000MiB minimum passed. New caps7200seconds/1024GGN/storage20GiB/cache2GiB, previous ledger retained, Judge0. GPU6/7 untouched.

Next heartbeat checks live process/status/receipts once; no duplicate start. If complete, audit all8, seeds, original-state invariants, E3 initial scores, every accepted export/clean consumer and deletion receipt, then publish both positive/negative results. Evaluate the frozen replication criterion without moving it. A null seed replication is a result, not an engineering fault. Any implementation failure requires preserved evidence and diagnosis before recovery. Further experiments need a separately justified frozen protocol using all prior results; no indefinite seed search or blind step expansion. E1–E7 protocols/results remain immutable. Historical E2 continuation below is context only.

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
