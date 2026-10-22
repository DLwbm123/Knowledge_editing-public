# Fixed medical development comparison

`run.py` reuses the existing frozen runtime, TT, generation, `protocol.brief_row`,
`path_v18` updates and `mechanism_v51/v51_cache.py`. Private staging supplies
`V52_ROOT`, `PARTITION`, `PARENT_RUN`, `DATA_ADMISSION.json`, the frozen protocol,
resource limits and the independent guard. Do not launch without the original
parent budget or real admission barrier. No acquisition or patient data are
packaged here.

`objective.selfcheck()` verifies analytic constraint gradients, causal answer
masking, exclusion of EOS and the zero-increment hinge. Real workers additionally
verify zero writer/Base generation and score equality, plus exact cached and
uncached CE/score gradients on all ten fixed teacher-forcing batches.

Both partitions must write `STAGE0_READY.json` before `STAGE1_GO.json` is created.
Every arm computes eight role CE gradients and four target-score gradients per
step; CE omits only the constraint terms. Identical initialization, 160 updates,
all data and both prompt contexts are fixed.

After both workers exit successfully, `evaluate.py validate ROOT` verifies the
96 bindings and exact update/backward receipts. `blind` prepares shuffled unique
outputs. Judge only that packet, freeze `BLIND_VERDICTS.json` with a
`VERDICTS_FROZEN.json` receipt, then run `summarize`. Do not inspect arm mappings
before verdicts freeze. Both-prompt paired repairs are evaluated only on heldout
same-pixel paraphrases, never relabeled as independent image/patient evidence.

The run improves the TT mainline through increment constraints. The CE comparator
is fixed and receives no search, structural change or extra tuning.
