# GPU5 continuation and autonomous follow-up

## Latest state — PAUSED BY USER, 2026-10-02

The user explicitly said “这个实验暂停，不要占用 gpu”. This overrides earlier autonomous continuation authorization. Do not start or resume any GPU work or successor experiment until the user explicitly authorizes it again. Hourly automation directw-gpu5 isPAUSED.

At the live stop check E23 was alreadyEXPERIMENT_COMPLETE8/8, PID1565973 absent, leaseabsent and no GPU compute process. No termination was needed. AllBase restored, no runtime error. Two actualfulltargetcorrections(cases3/4) passed independent nativeFP32cleanreload/fullgen; both matrices deletedafterfinalconsumers. Six othercasesBACKTRACK_REJECTED. Case4 has6probe scoredrops<-.1(worst-.9710361),case3none; fullFP32criterionFAIL and secondary<=3backtracksFAIL(6). These are FP32-only observations, not BF16deploysuccess or clinical/newfact/independentgeneralization. All original evidence stays preserved.

E23 consumed4372.422388298088seconds and11GGN. Retain cumulative5938GGN/51999.771838501794seconds in any authorized future continuation; never reset or hide old failed runs. Results/report are in reports/directw_evidence_v1_fp32_gradient_20261002/. The prior E20-E22 metadata erratum remains applicable; future protocols must be standalone and consistent with actualboundconfig, not blanket replacements. No next experiment is authorized while this explicit pause is in force.

## Ongoing research policy — explicit latest user authorization

The user requested long-running autonomous reasoning and hourly monitoring, continuing after each experiment. Set success criteria and bounded budgets before each new run. If all predeclared efficacy/protection criteria are met, freeze and run three fixed seeds or a bounded single-factor sensitivity set; report all seeds, mean/dispersion and failures, not the best. Otherwise diagnose the failure and test one key change per new protocol, preserving negative results and prior costs. Acceptance or lexical score alone is not medical efficacy. Default single-run ceiling2hours/1024GGN, three-seed batch6hours/3072GGN, plus prior cumulative ledger. No repeated approval is needed within this authorization. Respect user stops, actual missing data/permission and hard budgets; do not repeatedly launch an unchanged failed hypothesis. Keep hourly follow-up active and notify only meaningful results/errors/starts or required user action.

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
