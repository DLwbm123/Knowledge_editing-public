# CrispEdit native medical-VLM reproduction and matched control

Authorized on 2026-10-05 by the user's request to reproduce CrispEdit and run it
to see results. This is a separate bounded experiment. It does not resume the
old Direct-W E23 protocol or modify MedTRACE, AlphaEdit or the completed M-ORE
adaptation. The existing hourly monitoring and operational repair instruction
applies within the new scope and original deadline of this run.

## Upstream and what is reproduced

Paper: [CrispEdit, ICML 2026](https://arxiv.org/html/2602.15823v2).
Official repository: [zarifikram/CrispEdit](https://github.com/zarifikram/CrispEdit),
pinned commit `09035f16695998f3a71ec6006245d99e8cc648c8`.
The unmodified `easyeditor/models/crispedit/projected_adam.py` runs directly from
the private pinned upstream checkout. No upstream source is redistributed here;
the repository root did not provide a license file at inspection. Published code
is an independent native adapter plus checks against the pinned source.

Reproduce the K-FAC factorization, eigenvalue-product energy selection, strict
low-curvature mask, gradient projection before Adam, and momentum projection
when sequential caches change. The official code applies ordinary coordinatewise
Adam after projecting the gradient; this is not silently replaced with projection
of the complete Adam step. Its sequential cache option combines factors by valid
token counts. Follow that `edit_cache_style=sequential` option and Algorithm 2,
not the CLI's alternative default `mix` replay mode. Cache recalculation for
weight drift is disabled, matching the default flag. No extra clipping or trust
region is introduced; the YAML norm_constraint is not applied by this code path.

The Mistral configuration supplies layers 19–23 `mlp.down_proj`, 25 steps, lr=5e-4
and weight_decay=0. Energy threshold is 0.9, the effective CLI default which
overrides the YAML's 0.99. Adam betas=(0.9,0.999), eps=1e-8. Stop an edit early
only when its target NLL falls below the upstream 0.01 threshold. One native
request per sequential edit, in the original frozen order; no shuffle or search.

The upstream factor path actually uses outer products of observed next-token
CE gradients, not sampled-label Fisher or exact GGN matrix products. Reproduce
that empirical estimator and summed-CE normalization. Theoretical GGN motivation
must not be reported as an exact-GGN implementation result.

## Declared native adaptations

Use the existing official-native LLaVA-Med/Mistral 7B checkpoint, native image
preprocessing and question-only prompt bindings. Only the five language W matrices
are editable. Vision encoder/projector, tokenizer vocabulary and all other
parameters/buffers remain frozen. FP16 native forward/deployment with FP32 master
weights, optimizer states, curvature factors, eigensystems and loss. Differentiable
FP16 casts of the master weights make the optimized forward match deployment.
Fixed loss scaling 1024 is undone before projection/Adam. Both arms use identical
precision; upstream's BF16 CrispEdit versus FP32 no_crisp difference is not carried
into this matched comparison. No temporary low-rank parameterization is used.

For edit-history factors, native answer-predictor positions (including EOS) supply
the valid mask. Image and instruction token positions do not get invented target
labels. This differs from upstream's all-text prompt-plus-target cache. The model,
multimodal wrapper, precision, edit granularity/order and data adaptations mean
this is not a reproduction of the original text-LLM benchmark numbers.

## Fixed data and evaluation

- Use the first eight edits from the accepted Stage17 146-edit cohort, unchanged
  order, targets, 123-query panel union and original eligibility masks.
- Add the first 64 originally Base-correct queries in the original config order
  that are outside those 123 queries and disjoint from all eight edit source
  groups. There are 83 available candidates before selection. These 64 queries
  are evaluation-only; they never enter editing, curvature estimation, stopping
  or hyperparameter selection. This is source-group disjointness, not a claim of
  patient independence. Original T1L/T2L support may remain zero and must be NA.
- Capability statistics: first 1,000 documents of the already frozen Wikipedia
  shard, at most 512 tokens/document, batch one, no medical evaluation input.
  Corpus revision `97a0b052c326b45fb68593a14972d9eed884cd17`,
  `legacy-datasets/wikipedia` 20220301.en, same shard as AlphaEdit A2. This fixed
  prefix differs from upstream's sampled corpus selection. Documents 1,001–1,100
  are a separate text NLL holdout and never enter statistics or editing.
- Run one fresh Base evaluation, then CrispEdit-Seq, then matched ordinary Adam,
  restoring Base before each arm. Each arm performs eight cumulative edits,
  records all insertion target answers, and generates the complete final 187-query
  panel. No independent-single campaign or full 146-edit campaign is included.
- Native greedy generation, EOS and 1024-token maximum continuation; preserve
  complete raw answers/tokens privately, with no reference-length truncation.
- Report T0/T1G/T2G Fix, supported T1L/T2L retention, separate 64-query retention,
  newly broken current-Base-correct holdout answers, insertion-to-final losses,
  text holdout NLL, update/gradient/projection diagnostics and runtime. Original
  Astra eligibility stays frozen; also audit fresh Base labels with the new judge.

All new semantic decisions use the validated Qwen3-32B-AWQ revision
`0499c3ac83fdef8810b907a23894ba91e95eddd8`, concurrency 32, AWQ Marlin, thinking
off, temperature 0, seed 0, full constrained JSON, and the existing source-answer
agreement prompt. At most 577 mapped outputs; identical query/reference/answer
content is deduplicated within this run only. No prior-run decisions are reused,
inputs truncated or semantic scores retried to improve results.

## Verification, budget and recovery

Before launch: check the independent projection against a tiny dense Kronecker
oracle and exact upstream functions, the official Adam update and cache-reset
momentum, weighted history, eigenvalue ties and masked summed-CE factor estimates.
Native execution checks finite losses/gradients/weights, frozen noneditable
parameter versions/buffers and zero residual hooks. After each arm's final
generation, reload only the edited W in a fresh native process and require exact
target-logit and generated-token equality.

One GPU (pro5000 physical 7, UUID verified at launch) runs editing then judging
in separate environments/processes. Recheck at least 65,000 MiB free for editing
and 57,000 MiB for judging; never stop other jobs. Wall budget is four hours from
initial launch, including statistics, all arms, native reloads, grading and any
operational repair. No deadline or spent budget reset. No result-driven tuning,
additional layers, seeds, edits or follow-on experiment. Hourly monitoring may
repair operational faults and resume from saved state inside this boundary.

All bulk data and temporary state stay on the existing /data mount. Expected
generated-state peak is below 30 GiB, with at least 8 GiB free reserve. Keep only
one latest atomic resume state per active arm and a shared capability cache.
Resume retains FP32 master W, Adam state and online history; projection bases are
reconstructed. After an arm's insertion/final generation, text holdout and native
reload consumers finish, delete its exact generated resume/native-W files. The
capability cache's final consumer is the CrispEdit arm. Keep small receipts and
all private reconstruction/scoring evidence. Never delete shared pretrained models
or earlier experiments. Publish code, fixed protocol and sanitized aggregate
results to the existing GitHub branch through the proxy, verify remote commit
and anonymous report access, then stop this run's hourly monitor.
