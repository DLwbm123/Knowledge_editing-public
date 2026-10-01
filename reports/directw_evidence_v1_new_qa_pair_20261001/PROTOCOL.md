# E9 paired new-QA experiment: nearest versus stochastic rounding

Frozen before either arm executes,2026-10-01. Authority: user-delegated GPU5 follow-up design/execution. E7 yielded5/8 BF16 successes, E8 yielded3/8 and failed its frozen4/8 replication criterion. No new seed search or algorithm retuning follows. E9 tests the same method on new QA rows within the already audited source-isolated pool, with a matched nearest-rounding control.

## Selection and data boundaries

Start with the29-row original private candidate manifest from E8, including the two permanent smoke rows. Construct the set of all edit/reference rows selected for the previous eight-case experiment plus the two smoke rows. Exclude every row with an ID or question hash in that set; also retain only original train/CANDIDATE_ONLY, non-smoke rows. This leaves18 eligible QA rows from the same three image groups. Existing source-hash round-robin `select_cases(pool,8)` fixes eight target/reference pairs without reading model scores. References are selected only from this filtered pool, by the same deterministic first-distinct-image rule. Exclusion and order are frozen for both arms, with private rows and public selection digests.

All selected QA/reference content must match the original local train annotation exactly before launch, images must exist/read as images, and no formal evaluation QA or outcome is consulted. Reuse original provenance and source isolation; do not relabel missing scientific roles as admitted. Both arms use identical images/questions/answers/references. These are QA-disjoint from previous DirectW edit/reference/smoke rows but **not image-disjoint**: the three image groups overlap the old pool. Zero image-and-question-disjoint rows remain in the limited pool. The wider raw training file is not admitted. This is new-question exploratory development, not held-out evaluation, cross-image transfer or clinical generalization.

## Fixed paired comparison

Run order and membership frozen now:

1. `20261001E9A`: deterministic nearest BF16 writes, eight cases.
2. `20261001E9B`: stochastic BF16 writes, same eight cases, fixed E8 base seed20261002. Case seed20261002+1009*index+step; mechanical index10000; one draw per step reused across backtracking.

Execute both arms irrespective of scientific rejection in the first. The less successful E8 seed is retained rather than choosing the better E7 seed. Do not add seeds, select best-of-arm results or alter membership after observing E9A. If implementation/runtime fails, preserve evidence and diagnose before resuming; any repair that changes method or comparison must be reported explicitly.

Primary: paired difference in final accepted native BF16 edits `(E9B accepted - E9A accepted)/8`. Prespecified transfer hypothesis: E9B has at least2/8 accepted edits AND more accepted edits than E9A. Otherwise this criterion fails. This practical criterion is not statistical significance. Report the full eight-by-two outcome table, discordant pairs, actual target/KL/drift, candidate rejection reasons and all resource consumption. These are16 case-arm trials on8 distinct new QA targets sharing3 images. Do not pool these with E7/E8 as independent medical examples.

## Unchanged implementation and checks

Same initialized native BF16 model/image pipeline and single original layer21 down_proj matrix. FP32 full-functional GGN and Jacobian, preservation linear term, explicit slack, native-value QP RHS. tau100,nu10,trust0.1,edit/base drift0.3,three steps,CG16/rtol1e-4,seven backtracking factors,score tolerance1e-6,reference KL0.001. Relative target min(max(native Base,functional FP32 Base)+0.1,-1e-4). Clean Base resets per case, no CP/LoRA state, no Judge. Keep actual native acceptance, single-matrix audits and trial diagnostics. First-arm Base scores are new and not compared to E3's different questions. E9B must match E9A initial native/functional scores within1e-6 before each edit.

No implementation changes frombfe5c46; reuse42 passing CPU tests with exact source binding verification. Validate source-only deterministic selection/disjointness and annotation matches before launch. Each arm separately runs the excluded native mechanical pair first, and stops on nonfinite/nonrepeatable GGN, mutation, rollback or required consumer failures. Every accepted edit needs independent clean native BF16 reload: logits error<=1e-3, generated tokens exactly equal, no editor/adapter imports or hooks. Delete its temporary matrix after the last required consumer; retain private evidence and public receipts.

## Frozen budgets and continuation

Only pro5000 physicalGPU5 UUID GPU-4924bbd8-4082-3f4c-b4fb-78ca930122ca; GPU6/7 untouched. Sequential arms, never simultaneous. Each arm has new7200seconds/1024GGN caps; the entire pair has at most14400seconds/2048GGN. Begin with historical1749GGN/11677.34334912477seconds. E9B inherits actual cumulative E9A consumption; its execution binding is finalized with those receipts, without changing this scientific protocol. Judge0. Per-arm storage20GiB/cache2GiB, minimum28GiB setup free storage and66000MiB free GPU memory. Prior BF16 peak about58.12GiB supports margin; combined parent/consumer peak remains runtime-dependent. Expected each20–40minutes, not guaranteed.

Publish completed arm outcomes and final paired results; do not call the pair complete after armA. Hourly task resumes the frozen armB when armA is audited. All earlier negative results/budgets remain. Verified visual pairs, non-target scopes, fact timelines and independent calibration/evaluation remain BLOCKED_DATA; no fabricated medical negatives or effectiveness claim.
