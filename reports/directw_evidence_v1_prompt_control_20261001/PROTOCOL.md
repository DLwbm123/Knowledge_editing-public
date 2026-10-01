# E13: read-only Base prompt-format control

Eight fixed E11/E12 original train targets, same3images, same references. Compare the original question with exactly this appended instruction:

> Answer with a single word or short phrase. Do not explain.

Use the same native BF16 Base, mistral_instruct template, greedy decoding and64-new-token cap. No optimizer, matrix write, adapter, checkpoint export or paid Judge. Original generated tokens must exactly replay the saved E11 Base outputs. All parameter version counters and the selected matrix must remain unchanged. Private raw outputs remain available for both arms, including on replay failure.

Measure native content-answer mean logprob excluding EOS, the already-frozen lexical exact match/token F1, token counts and EOS/cap hits. Support the format-alignment hypothesis only if at least4/8 generations shorten, mean paired reference logprob rises by at least1.0nat, and exact matches increase by at least2/8. If the combined criterion fails, retain the negative result and do not search many prompt variants. Truncated outputs remain in denominators.

This tests a response-format mechanism; it is neither weight editing nor new-fact replacement. A short reference can be correct inside a sentence even when exact match is zero. Meeting these diagnostic thresholds does not establish medical semantic accuracy, independent generalization or previous edit efficacy. If supported, separately freeze an aligned-prompt edit study with actual answer-target criteria before seed confirmation; otherwise analyze objective/constraint inadequacy.

Budget:3600newseconds,0GGN, inherited2430GGN/17778.920738057932seconds. GPU5 only with66000MiB prelaunch free requirement;20GiB output ceiling/28GiB free disk. No generated matrix lifecycle. Any native replay, parameter-preservation, finite-metric or time failure stops with receipts.45CPU tests pass, including mocked successful and failed replay with preserved raw evidence/ledger. Native model is never run on CPU.
