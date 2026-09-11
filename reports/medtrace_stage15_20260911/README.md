# Stage15 — fixed MedMKEB external single edits

Runtime source commit: see the run's `private/CAMPAIGN_CONFIG.json`. Parent public result:
`42ebd8a67fd54d23e724a37d85da5b4f264a0196` (Stage14). No previous experiments are rerun.

The queue contains 200 requests selected from 260 native-image-resolved requests by the
predeclared 20260911 hash/stratified round robin. The pinned MedMKEB release is
`d9f38639ec2285a0e9f541e22156ec14f87271d8`: 4490 train, 2497 main, 721 attack rows.
Only these three missing small JSON indexes were downloaded. Existing image bundles
provide 471 exact original-reference bindings; no source images were downloaded.
There are 1057 available official probes and 169 probe-specific missing-image entries.

The complete MedMKEB train index was searched once. All 131 image-resolved train rows
overlap official evaluation image roles. No verified, scope-independent H was available;
`pred` was not silently promoted to a clinical gold label. GMAI access is gated and no
access terms were automatically accepted. Thus C_FACT and C_EXTRA_QA are unsupported
on this assembled external cohort, not failed. C_NO_H and BalancEdit run on the fixed
200 requests; one previously authorized original SLAKE train QA supplies U KL only.
This support limitation does not establish external validity for C_FACT.

The single entry `scripts/medtrace/stage15.py` uses the existing native CP initializer,
80-step A2, CP-W0 positive-only optimizer, freeR4 writer, full-vocabulary KL,
BalancEdit implementation and fixed Judge backend. The thin adapter adds no algorithm,
threshold or loss. 320 continuation steps, all original V1 hyperparameters and old16 RC
kappa remain fixed. Its no-image locality path uses `images=None`, zero image tokens,
and the same mean Base prompt-feature router for every writer. It never substitutes a
black or native image for text locality. The existing black routing anchor is unchanged.

Official probes and counterfactual targets remain verbatim. Four training text variants
are generated from native text alone, not the official rephrase. The new method-blind
Judge protocol distinguishes benchmark target adherence from source-answer agreement;
historical semantic verdicts are not reused across that protocol boundary.

The audited VLKEB source at `10951b7b3788928f578b73f07eac9e1eaa0316f3`,
`easyeditor/evaluate/evaluate.py`, uses shifted teacher-forced logits, last-target-length
alignment and a `labels != -100` top1 token mean. The MedMKEB release supplies no
medical-model execution adapter. Accordingly this run reports **PAPER_DEFINITION_OUTPUT_MATCH /
NOT_AUTHOR_EXECUTION_PARITY**, using deterministic free output, NFKC/case/whitespace
normalization only. Locality is pre/post agreement; source accuracy is a separate column.

Run CPU contract check: `python tests/test_stage15_contract.py`.
Run actions: `stage15.py lock|launch|worker|prepare-judge|judge|report --run-root RUN`.
The launch uses two independent GPU0/1 workers with neutral process names. It is detached
from SSH/Codex; 24h wall/48 GPU-process-hour ceilings reserve the last 3h for scoring and
closeout. No monitor or new experiment is scheduled. On completion, publish only the
source and the whitelisted aggregate public directory; retain full QA, images, raw
answers, Judge mappings, checkpoints and environment paths privately. Publication is
pending until a successful push and anonymous access check, and never triggers reruns.

Sources: [MedMKEB](https://github.com/pkusixspace/MedMKEB/tree/d9f38639ec2285a0e9f541e22156ec14f87271d8),
[paper](https://ojs.aaai.org/index.php/AAAI/article/view/40705/44666),
[VLKEB evaluation](https://github.com/VLKEB/VLKEB/blob/10951b7b3788928f578b73f07eac9e1eaa0316f3/easyeditor/evaluate/evaluate.py).
