# E8: prespecified BF16 seed replication

Frozen before execution,2026-10-01. User authority: delegated GPU5 diagnosis and follow-up experiment design/execution. E7's fixed seed schedule yielded5/8 accepted BF16 edits with exact clean consumer parity, against E3 nearest-rounding0/8. A single seed does not establish stability. E8 tests the identical algorithm on the same eight development cases using one additional prespecified seed schedule. It introduces no code change and reuses the frozen E7 implementation.

## Hypothesis and comparison

Intervention: base rounding seed20261002 instead of20261001. Per-case/step seed is20261002+1009*case_index+local_step; mechanical index10000. Same private device generator and one tensor per step shared across all backtracking factors. No seed sweep, outcome-dependent retries or selecting the better seed. Report E7 and E8 individually and as16 case-seed trials on8 distinct cases, never16 independent data cases.

Prespecified practical replication criterion: at least4/8 final accepted native BF16 edits. Below4/8 falsifies this criterion; preserve that result without retuning. This is an engineering criterion, not a statistical significance claim. Report exact overlap with E7 successes {0,2,5,6,7}, per-case gains/rejection reasons, requested/written displacement, final KL and compute cost. Do not choose the best result across seeds as the primary metric. The primary denominator includes all8 intended cases; implementation interruption means incomplete replication, not a smaller denominator.

E3 is the nearest-rounding control, E7 is the same-method first-seed reference. Cases, source-hash selection, reference assignment and permanent smoke exclusions are identical. Initial native BF16 and FP32 functional scores must match E3 within1e-6. Inputs are previously inspected train-support cases with unknown reference scopes; this is not independent calibration or formal evaluation.

## Frozen method and checks

Original native BF16 loader/image pipeline, same single original layer21 down_proj matrix, exact functional FP32 GGN, preservation first-order term, native-value QP RHS, explicit slack. tau100,nu10,trust0.1,edit/base drift0.3,three scientific steps,CG16/rtol1e-4,seven backtracking factors,score tolerance1e-6. Target min(max(native BF16 Base,functional FP32 Base)+0.1,-1e-4), native reference KL ceiling0.001. Clean Base reset per case; no CP/LoRA state. Scientific refusal remains a negative outcome.

No implementation changed since source commitbfe5c46. Reuse the42 passing CPU tests from E7, including unbiased grid rounding, fixed-seed real tiny-model edit/rejection, global-RNG isolation and mutation audit. Do not rerun unchanged tests just to label a new seed native-valid. E8 independently repeats actual native mechanical validation on the two excluded rows, with the new seed, before scientific cases. Native repeatability, parity, rollback and single-matrix invariants remain mandatory. A successful scientific edit additionally requires an independent clean native BF16 consumer with logits error<=1e-3, exact generated tokens and no editor/adapter imports/hooks; delete its temporary matrix only after that consumer.

## Resources and termination

RUN_ROOT20261001E8, pro5000 physicalGPU5 UUID GPU-4924bbd8-4082-3f4c-b4fb-78ca930122ca only. New finite caps7200seconds/1024GGN,8cases/3steps,Judge0,storage20GiB/cache2GiB. Carry prior1488GGN/9769.865869862726seconds. Require66000MiB free at launch and28GiB storage free at setup. E7 peak per-case allocation58.002GiB supports this margin; native clean consumer cost is part of the wall budget. Expected20–40minutes, not guaranteed.

Stop and preserve evidence on runtime, nonrepeatable forward, mutation, rollback or required consumer failure; no blind retry. Complete all8 despite scientific rejection, within finite caps. Publish all positive/negative results before any separately justified next protocol. Missing verified visual pairs, non-target scopes and independent evaluation remain BLOCKED_DATA; no medical-generalization claim or invented medical counterexamples.
