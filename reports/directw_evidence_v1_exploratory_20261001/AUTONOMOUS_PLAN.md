# GPU5 continuation and autonomous follow-up

## Latest state — 2026-10-02 E15 delivered; E16 launched

E15 COMPLETE/public a4ec73f: 0/8 accepted, six NOT_SATISFIED after three accepted intermediate steps, one SOLVER_NOT_CONVERGED after one step and one BACKTRACK_REJECTED after one step. All rolled back/Base restored, no matrices/failures/worker remained. New 290 GGN/2426.387077790918 seconds; cumulative 2851 GGN/21306.461885862867 seconds. Seven cases avoided solver termination, but the complete numerical criterion failed. Case5 primal KKT1.2398e-5 exceeded1e-5 despite converged CG; case7 all second-step factors exceeded reference KL. The six capped cases continued improving yet margins remained negative and drift below0.3. Zero final probe changes reflect rollback, not useful protection.

E16 launched PID1414040, initial NATIVE_REGRESSION/neutral argv/logs/lease confirmed with no immediate failure. Fresh RUN_ROOT20261002E16; protocol reports/directw_evidence_v1_six_step_20261002/. Only scientific max_edit_steps3->6; same eight cases, original prompt, target-content-plus-EOS margin+0.01, functional curvature, stochastic BF16 seed20261002, QPdual_tol1e-6, CG1e-4/max16, trust0.1, drift0.3, referenceKL0.001,199probes,64token generation. Mechanical one-step/tol1e-8/exactE11 fingerprint and Base logprob/token controls retained. Exact source unchanged;47CPU checks reused, configuration delta verified. New7200seconds/1024GGN with the complete inherited ledger above. Inspect real process/STATUS/lease before continuation; don't duplicate a live run.

On completion audit first up-to-three numeric attempt receipts per case against E15 before causal claims; this is post-run comparison, not an implemented runtime guard. Keep all eight cases including known prior failures. Efficacy criterion unchanged: >=2exact full target-token sequences includingEOS after accepted physical edits, zeroextraQA score drops<-.1, all required native consumers pass. If all criteria pass, freeze3seed replication. Otherwise examine achieved margins, residuals/KL/drift and costs; no automatic additional step doubling or threshold relaxation. Report partials at hard-budget stops without resets. Publish complete sanitized results/report before next run; delete temporary matrices only after last required native consumer. Missing newfact/scope/visual/independent-image data remains BLOCKED_DATA, Judge0.

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
