# GPU5 continuation and autonomous follow-up

## Latest state — 2026-10-01 19:15 China-time follow-up

E8 completed8/8 with **3/8 accepted native BF16 edits; its prespecified >=4/8 replication criterion FAILED**. Successful cases0/6/7 passed exact clean native reload/generation parity and matrix deletion after consumers. E7-only successes2/5 failed here. All8/final Base restored, no runtime error, native regression PASS with its own mechanical edit BACKTRACK_REJECTED.55 scientific trials,15 accepted intermediate steps,40 rejected (30 protection,10 merit). E8 results/report published commitb36b554 in `reports/directw_evidence_v1_seed_replication_20261001/FINAL_REPORT.md`; anonymous HTTP200 verified. E7+E8 count8/16 accepted case-seed trials on8 distinct cases, not16 independent examples. New261GGN/1907.477476seconds; cumulative1749GGN/11677.34334912477seconds. Peak per-case allocation58.118GiB. Do not retune the failed replication criterion or search more seeds until favorable.

Current run **20261001E9A**, launch PID1333578, first arm of the **frozen two-arm new-QA comparison**, protocol commit200371a. Read `reports/directw_evidence_v1_new_qa_pair_20261001/PROTOCOL.md`, SELECTION_AUDIT.json and E9A bindings. From the original29-row audited manifest, exclude all previous edit/reference/smoke IDs and question hashes.18 eligible QA remain; existing hash round-robin selects8 target/reference pairs. Exact original train annotations for all18 matched and3 images were readable; order/disjointness assertions passed. These8 QA are new for this DirectW edit/reference sequence but all3 images overlap, so no cross-image or medical generalization claim. The wider train file was inspected for availability metadata only, not admitted. Scientific role/visual data still BLOCKED_DATA.

Both arms frozen BEFORE armA: **E9A nearest rounding, then E9B stochastic rounding at base seed20261002** (the less successful E8 seed), same8cases/refs/3steps/gain0.1/KL0.001/drift0.3. No code change; sourcebfe5c46/42 CPU checks reused after binding equality. Primary paired difference(E9B-E9A accepted)/8. Hypothesis requires E9B>=2 accepted AND E9B>E9A; otherwise fails. Preserve all16 case-arm outcomes on8 distinct newQA/3images. Complete armB irrespective of scientific armA rejection, without membership/seed/method retuning.

Initial19-second live check: PID1333578 alive, neutral argv `/data/bmw/envs/s0/bin/python -`; GPU5 UUID GPU-4924bbd8-4082-3f4c-b4fb-78ca930122ca only, stateNATIVE_REGRESSION,0/8 scientific cases. Log present/no traceback/failure receipt; validation still pending at snapshot. Prelaunch66000MiB free criterion passed; initial used15548MiB is not peak. /data mount ext4/dev/sdc1 with17.1TiB available, write/read probe passed. Each arm cap7200seconds/1024GGN/storage20GiB/cache2GiB; pair cap14400seconds/2048GGN; Judge0, prior ledger retained. Other GPUs untouched.

### Next heartbeat: finish the already frozen pair

First check actual PID/state once; do not duplicate. When E9A completes, audit/publish full8 outcomes including mechanical/normal-forward/rollback/clean consumers and all costs. The paired experiment remains incomplete until E9B. Prepare a fresh20261001E9B root from E9A private config with SAME candidate_rows/data manifest/source/selection/pair protocol, change only arm='E9B_STOCHASTIC', rounding_mode='stochastic_bf16', and attach historical_control_scores from E9A cases for initial native/FP32 parity1e-6. Keep seed20261002, all hyperparameters and pair membership unchanged. Record actual E9A cumulative time/GGN as B's prior ledger; B remains at most7200seconds/1024newGGN and the pair at most14400/2048. Rebind execution config/budget to the existing user's delegated authority and preserve the frozen pair protocol document hash. No CP/LoRA or edited armA matrix is loaded. Source identical unless an actual engineering fault requires documented diagnosis/repair. Write B bindings/test reference/launch report then run mechanical checks and8scientific cases onGPU5. On B completion publish final paired table and criterion outcome before separately designing further work. No new seed sweep or automatic expansion during the pair.

Earlier E1–E8 protocols/results and negative evidence remain immutable. Historical E2 continuation below is context only.

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
