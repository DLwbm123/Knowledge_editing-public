# AlphaEdit VLM versus the frozen Stage17 146-edit cohort

Frozen before A2 GPU execution. The user selected the existing **MedTRACE
C_NO_H (without H)**, BalancEdit, BELoRA and LoRA as comparison methods.
This does not authorize a full C_FACTH run or resume the paused Direct-W campaign.

## Comparison contract

- Keep all 146 original edits, their order, event/probe mappings, Base eligibility
  masks, references and active-target exclusions. No score-dependent exclusions.
- Reuse the completed historical four-method results. Their immutable Astra
  model snapshot was unavailable. Historical hardware/runtime/date and LoRA
  precision differences remain limitations, not silently matched conditions.
- AlphaEdit primary: layers 19–23. Layer 21 alone is a prespecified capacity
  sensitivity arm. Neither arm is selected retrospectively by its results.
- Each arm runs 146 independent edits from Base, then a distinct cumulative
  146-edit sequence. Evaluate prefixes 1, 50, 100 and 146. Sequential W and the
  official post-edit key cache accumulate continuously. Independent single
  checkpoints cannot substitute for cumulative editing.
- AlphaEdit uses Stage17 official-native LLaVA-Med loading, FP16, original
  deterministic image preprocessing, identical question-only prompt token IDs,
  greedy continuation generation, EOS stopping and max_new_tokens=1024. Full
  continuation tokens are retained. Answers never set a generation stop length.
- New Base generation is audited on all 1,512 unique queries. Prompt/image
  mismatch is a mechanical hard stop. Numerical output drift is reported while
  retaining the original accepted cohort and masks.

## AlphaEdit adaptation fixed in advance

The implementation adapts official AlphaEdit commit
`b84624f44dfe8fc6cd9e41df916c44124a0c46dc`; see the neighboring upstream license.
The edited position is the last native multimodal prompt token. Target latent
optimization uses answer-plus-EOS NLL and the approved U reference QA for KL;
the original cohort provides one distinct approved U QA. No evaluation probe
is used as an optimization constraint or selection criterion.

Hyperparameters: 25 latent objective evaluations, learning rate 0.1, norm penalty
0.5, KL coefficient 0.0625, clamp factor 0.75, projection eigenvalue threshold
0.02 and L2=10. The compact Woodbury solve is checked against the official dense
equation for empty and nonempty historical key caches. The cache appends keys
recomputed after all layers of the current edit have been written.

Projection statistics use external Wikipedia **text-only** activations through
the frozen VLM language backbone: the first 10,000 documents of the first
`legacy-datasets/wikipedia` 20220301.en Parquet shard at revision
`97a0b052c326b45fb68593a14972d9eed884cd17`, capped at 512 tokens/document. This
bounded corpus/version/context length differs from original AlphaEdit's
Wikipedia configuration; the result is a VLM adaptation, not a paper-exact
reproduction. Corpus source: [dataset](https://huggingface.co/datasets/legacy-datasets/wikipedia).
Its source license is CC BY-SA 3.0 / GFDL. Raw text stays private on the server.
No formal cohort medical images or QA are used for covariance estimation;
the earlier 447-image A1 projection is not reused.

## Validation and metrics

Before covariance accumulation, check original prompt/image bindings, finite
nonzero latent gradients at layers 21 and 23, frozen Base parameter versions,
and absence of residual hooks. No physical W change occurs in this gate.
Check projection finiteness and sampled orthogonality. Every independent edit
and every sequence prefix has a fresh native-process weight reload with exact
target logits and generated-token parity; its process imports no AlphaEdit
algebra or latent optimization. Other sequence insertions retain raw native
generation and state receipts.

Report the original Stage17 semantic Fix, generalization and locality metrics,
their original eligible denominators, paired differences and available source
cluster intervals, and costs. Use the existing Astra/high judging protocol;
never replace semantic judging with token exact match. Historical approved
verdicts may be reused only with their original bindings. New outputs require
judging; raw GPU generation completion is **GENERATED_NOT_SCORED**. At most
16,000 new verdict keys are authorized for this bounded comparison. No model
selection, new seeds, new orders, automatic scientific retries or follow-on run.

## Resources and lifecycle

Use pro5000 physical GPU 5 for statistics then Base audit, GPU 6 for layer-21
single→sequence, and GPU 7 for layers-19–23 single→sequence. The editing workers
wait for fixed projection readiness. Require matching GPU UUIDs and at least
52,000 MiB free before launch. The absolute wall cap is 8 hours and reservation
ceiling 24 GPU-hours; all dependent phases share this ledger. Mechanical setup
repairs before the first physical edit must be recorded, not hidden as a new run.

Keep at least 8 GiB free. Retain only the current generated W/state and fixed
projection (estimated generated-weight/projection peak below 9 GiB). Complete
all registered consumers before deleting single weights, final sequence state
or projection. Preserve active recovery state after failure, and keep private
outputs, tokens, bindings and small receipts for judging without weights.
An interrupted sequence is not automatically restarted. No permanent weight
archive, periodic monitor or unrelated experiment is created.

Historical LoRA single is FP16; sequence uses its approved BF16 replacement.
The original FP16 sequence failed at edit 17 step 4 and remains in the report.
The comparison must explicitly state that this is not a same-precision LoRA
comparison. C_NO_H is never labeled as the full H-enabled method.
