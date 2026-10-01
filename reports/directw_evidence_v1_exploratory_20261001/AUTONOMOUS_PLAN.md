# GPU5 continuation and autonomous follow-up

## Latest state — 2026-10-01 20:15 China-time follow-up

E9A completed8/8 new-QA nearest-rounding cases with **2/8 accepted native BF16 edits (indices1/5)**. Three NOT_SATISFIED and three BACKTRACK_REJECTED; all six rolled back. Both successful exports passed exact clean native reload/logit/token parity and were deleted after final consumers. All8/final Base restored, same-precision parity0, native repeats exact, no runtime failure. Native mechanical checks PASS while its own edit BACKTRACK_REJECTED/excluded.51 scientific trials:15 accepted intermediate steps,36 rejections(2protection/34merit). Published commit7667659, `reports/directw_evidence_v1_new_qa_pair_20261001/E9A_FINAL_REPORT.md`, public HTTP200 verified. New235GGN/1790.983875seconds; cumulative1984GGN/13468.327227218775seconds. Peak per-case allocation57.881GiB.

Current run **20261001E9B**, launch PID1344840, executes the **already frozen second arm** of the paired new-QA protocol. Read `reports/directw_evidence_v1_new_qa_pair_20261001/PROTOCOL.md`, E9B_PROTOCOL.json, E9B_SOURCE_BINDINGS.json and E9B_PREFLIGHT.json. Pair protocol was frozen in200371a before A; B execution binding publishedb91eae0 with A's actual cumulative ledger. Source unchangedbfe5c46/42 CPU tests reused; verified same selection digest, candidate/reference rows, source, model, solver and pair document hash. Only write-rounding mode differs from A. Base stochastic seed20261002, per-case20261002+1009*index+step; mechanical index10000, one draw per step reused across backtracking. All3-step/gain/KL/drift limits unchanged. B checks initial native/FP32 scores against A within1e-6 for each scientific case before editing. No edited A matrix or CP/LoRA state is loaded.

Initial16-second live check: PID alive with neutral argv `/data/bmw/envs/s0/bin/python -`; GPU5 UUID GPU-4924bbd8-4082-3f4c-b4fb-78ca930122ca only, stateNATIVE_REGRESSION,0/8 scientific cases. Log present/no traceback/failure receipts; native check pending at launch snapshot. Device used15548MiB/free57283MiB; prelaunch66000MiB free condition passed. /data ext4/dev/sdc1,17.1TiB available, probe passed. B cap7200seconds/1024GGN/storage20GiB/cache2GiB; pair cap14400seconds/2048GGN with A costs counted, Judge0. Other GPUs untouched.

### Next heartbeat: finalize the frozen E9 pair

Check actual live PID/status once; no duplicate start. When B completes, audit all8 outcomes: selected original matrix only, expected seeds, A/B initial-score parity, actual native target/protection/drift, rollback, independent clean reload/generated tokens and final-consumer deletion. Preserve runtime failures and scientific rejection distinctly. Publish B results AND full8-by2 paired table. Primary difference(B accepted-A accepted)/8; frozen hypothesis B>=2 AND B>A. With A=2, B must have>=3 successes; this is a consequence of the fixed criterion, not a new threshold. Report discordant pairs, actual per-case thresholds (max(nativeBase,FP32Base)+0.1), costs and all negative results. Gain>0.1 over nativeBase alone can still miss the exact target. Reconcile pair spending against its caps and keep cumulative history. The pair is incomplete until B audited/published.

Selection boundary: original29-row source-isolated pool; exclude prior edit/reference/smoke IDs and question hashes, leaving18 QA. Eight new target/reference pairs selected deterministically and both arms identical. Source-only18-row annotation matches and3readable images audited before A. All3 image groups overlap prior experiments; this is new-QA exploration, not image-disjoint or held-out medical evaluation. Wider raw train file was inspected for availability only, not admitted. E7=5/8 and E8=3/8 on oldQA, with E8 failing frozen4/8 seed-replication criterion; do not select best seed or erase that null. W_EVIDENCE/scientific roles remain BLOCKED_DATA.

Only after full pair review/publication may a separately justified bounded next experiment be designed using all positive and negative evidence. No indefinite seed search, favorable resampling or blind step expansion. Earlier E1–E9A protocols/results immutable; historical E2 continuation below is context only.

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
