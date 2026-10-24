# Competitor-relative increment ablation

V53 changes V52's absolute target score to target-minus-fixed-competitor score.
The competitor is the negative training answer, fixed before outputs. Same data,
TT initialization, optimizer, step count and gates; only the mainline writer is
trained. V52 Base and CE controls are reused with exact input bindings.

Use the V52 staging/independent-guard workflow with `V53_ROOT`. Real admission
checks all twelve CE/score batches before the `STAGE1_GO.json` barrier. Per worker:
160 updates, 2560 formal +72 admission backwards. The evaluator verifies 96
bindings including 64 control outputs; blind/freeze/summarize as in V52. Do not
inspect mappings before grading freezes. No independent confirmation claim.
