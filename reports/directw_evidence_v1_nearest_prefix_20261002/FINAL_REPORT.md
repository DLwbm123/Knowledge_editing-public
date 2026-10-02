# E20 nearest BF16 rounding: completed negative control

All eight cases completed. Numerical controls passed, but no edit was accepted and no complete target sequence was produced after an accepted edit. Five cases ended BACKTRACK_REJECTED and three reached the six-step cap. The predeclared prediction of at most three backtracking failures failed; E19 also had five. Nearest rounding does not solve the observed bottleneck under these fixed settings.

| Case | Terminal status | Last terminal-stage finding |
|---|---|---|
|0|BACKTRACK_REJECTED|All seven factors fail protection; smallest reference KL0.001931834|
|1|NOT_SATISFIED|Six steps, last accepted margin-4.0625|
|2|BACKTRACK_REJECTED|Only smallest factor passes protection; margin-10 is unchanged|
|3|BACKTRACK_REJECTED|Only smallest factor passes protection; margin-13.9375 is unchanged|
|4|BACKTRACK_REJECTED|Only smallest factor passes protection; margin-6.3125 is unchanged|
|5|NOT_SATISFIED|Six steps, last accepted margin-9.6875|
|6|NOT_SATISFIED|Six steps, last accepted margin-1.0625|
|7|BACKTRACK_REJECTED|All seven factors fail protection; smallest reference KL0.002826867|

For cases2/3/4 the smallest factors have reference KL0.0000253631/0.000000926895/0.0000883517, respectively, and satisfy the recorded protection/drift gates but do not improve the deployed minimum margin. Larger factors fail protection. This supports a physical-score plateau versus protection tradeoff; it does not identify which token causes the plateau or prove that quantization is the sole cause. Nearest writes often zero about99% of requested element changes, while the surviving matrix changes can still leave the minimum margin unchanged.

All historical initial logprob/FP32/native-prefix/Base-token controls, mechanical exact replay and independent prefix audits passed. All eight cases restored Base. Final generation and all199 probe slots are unchanged because edits rolled back; zero final probe damage is not evidence of successful protection of an edited model. There were no generated scientific matrices, leftover lease, live worker or recorded runtime errors at completion.

New cost519GGN and4285.285367141012seconds. Cumulative5397GGN and40405.21474880981seconds, including all earlier negative runs. SUMMARY.json and FINAL_RESULTS.json retain complete denominators, numerical controls and trial diagnostics. Raw QA/images/tokens/weights remain private.

The next bounded diagnostic will record intermediate per-target-position native margins during an exact replay of E20, with no optimizer or threshold change. This can determine whether the worst position switches across accepted intermediate updates and whether terminal plateaus affect every position or just the minimum. Until these measurements exist, token competition remains a hypothesis. No automatic extra steps, threshold relaxation, seed selection or repeated rounding sweep is justified. Original training-answer reinforcement on reused images remains distinct from medical fact editing; independent image/scope/visual evidence remains BLOCKED_DATA, Judge0.
