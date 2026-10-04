# Second-route language-W recursive pilot

Authorized on 2026-10-04: try the applicable parts of M-ORE for the independent
direct-W route; temporary low-rank optimization is allowed, with final merging
into language W and the visual modules frozen. This is a new bounded protocol,
not a resumption or replacement of the paused Direct-W E23 protocol, its ledger,
or MedTRACE. It does not establish the earlier missing visual-pair evidence.

## Fixed scope before native execution

Use the first eight tasks, in original order, from the frozen Stage17 146-edit
cohort. Run two continuous sequences from the same Base: `recursive`, then
`static`. Each sequence gets the same fixed orthogonal coordinates and P0.
Only language layer 21 MLP down_proj is writable. No evaluation query enters
optimization, initialization statistics, or hyperparameter selection. Run one
Base panel, an insertion-target generation after each write, and each arm's
complete final union of the eight registered query panels. Keep original
references, eligibility and event bindings; obsolete targets must use the
frozen active-target exclusions during later semantic aggregation.

This is a developmental eight-edit pilot, not an independent confirmation
set or the full 146-edit comparison. No independent-single campaign, automatic
retry, seed search, parameter tuning, extra arm or follow-on phase is included.
Wall cap: two hours on one GPU, including model load and clean reloads. Keep a
new run ledger; do not reset or reuse the previous campaign's spent budget.

## Adaptation

Source: [M-ORE, ICML 2026](https://arxiv.org/html/2605.20273), Eqs. 12–13.
This implementation is independent. The source repository's public main page
exposed README, LICENSE and .gitignore at inspection; no implementation was
available there for a source-level equivalence claim.

Rank 512; fixed row-orthogonal A; initial B=0; scale 2; learning rate 0.1;
ridge 2000; seed 20261004. P0=I/(1+ridge). One answer-plus-EOS NLL gradient
evaluation per insertion, without an optimizer iteration loop. FP32 A, B,
gradient and P; FP16 native backbone. A fixed loss multiplier 1024 is undone
on the gradient, to reduce FP16 backward underflow without changing the objective.

The differentiable forward uses the actual cast native weight
W=FP16(W0+2BA). Update B by -0.1 grad(B) P_previous, then merge again from W0.
Do not accumulate separately rounded delta-W writes. The recursion pools the
actual module inputs over all nonpadding native image-question prompt tokens
(excluding the target answer), recomputed after the write, and projects that
mean through A. Update P using Sherman–Morrison only after a finite committed
write. Static control keeps P=P0; otherwise its computation is identical.
Finite writes are retained regardless of quality or rounding to zero. Failure
stops the job and preserves the latest complete state; no automatic resumption.

Differences from paper configuration include one language layer, frozen vision
encoder/projector, native LLaVA-Med FP16 merging, this prompt pooling convention,
and the eight-edit medical cohort. These are declared adaptations, not a full
M-ORE reproduction. One editable module cannot test its multimodule decoupling
claim. Text/image token splitting is not substituted for module-wise history.

## Evidence and lifecycle

CPU checks compare recursive directions and state to dense inverse solutions,
including rejected proposals, zero keys and isolated modules; a tiny functional
model checks the low-rank gradient and native state-dict reload. Native execution
checks finite nonzero gradients, orthogonal A, noneditable parameter versions,
buffer equality and zero retained hooks. The two first writes must match exactly.
After each full sequence, a fresh process imports only the native runtime,
loads the saved W, and must reproduce target logits and full generation exactly.

Generation reuses the original Stage17 prompt/image binding checks, greedy
FP16 native decoding, EOS and 1024-token continuation cap. Preserve all generated
tokens privately. NLL is a mechanical/descriptive measurement, not semantic Fix,
Hit, locality or cross-image generalization. Future semantic scoring defaults
to the previously validated Qwen3-32B-AWQ concurrency-32 setup; scoring remains
pending until its own frozen input packet and receipts exist. No quality claim
may be based solely on CPU tests, gradient availability or NLL decrease.

Storage uses the existing pro5000 /data mount and pretrained models. Reserve
8 GiB, with less than 2 GiB expected temporary output. Keep only each arm's latest
resume state (A, B, P, W), one W for its native consumer and small receipts.
After final panel and clean reload complete, delete precisely those generated
state files and write a deletion receipt; keep failure resume states. No shared
model, historical checkpoint or first-route artifact is modified. Source,
protocol and sanitized results are published; private medical queries, answers,
tokens, paths and model weights are excluded.

## Completion scoring, 2026-10-04

After the user requested live status, generation completion was verified and
the pending scoring packet was frozen before any new verdicts. Reuse the validated
Qwen3-32B-AWQ revision `0499c3ac83fdef8810b907a23894ba91e95eddd8`, concurrency
32, AWQ Marlin, thinking off, temperature 0, seed 0 and full constrained JSON
responses. The existing Qwen worker is reused with only its protocol identifier
set to `DIRECTW_RECURSIVE8_QWEN32_20261004_V1`. Prompt and execution settings are
recorded in the scoring lock. No historical Astra verdict is reused as a new
Qwen verdict. Identical query/reference/candidate-text inputs within this pilot
are scored once and mapped back to every occurrence; all tokens remain saved.

The packet contains Base and both final 123-query panels plus 16 insertion-target
answers (385 occurrences). Primary panels retain the original Base eligibility
and exclude the eight inserted targets from locality. Their identity matches
the original active-target mapping restricted to the first eight edits. Report
fresh Base Qwen agreement and any changed Base labels alongside both arms;
original Astra-derived masks do not become Qwen-derived masks. No input truncation,
score-based retry or new edit is allowed. Scoring on GPU 6 uses the remaining
original wall deadline, with no overlap with the completed editing process.

The user subsequently requested hourly monitoring and repair/continuation when
problems occur. This authorizes recovery from operational faults within the
existing scope and deadline. The first scorer attempt failed before producing
any verdict because Python multiprocessing spawn tried to reopen `<stdin>`.
Its logs/status remain preserved. Removing the synthetic main-file metadata
repairs the neutral entry; no model, prompt, input, parameter or scoring rule is
changed, and the deadline is not reset. This is a startup repair, not a retry
selected by score. The hourly monitor ends after verified publication.
