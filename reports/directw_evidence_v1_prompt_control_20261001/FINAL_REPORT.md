# E13 final: format affects likelihood, full hypothesis criterion fails

The read-only Base diagnostic completed8/8 paired cases. All original generations exactly replayed E11Base, every parameter-version check and selected-matrix comparison passed, and all16 outputs reached EOS without truncation. No weight was edited; no optimizer/checkpoint/Judge/GGN was used.

7/8 answers shortened and mean reference-answer logprob rose by 3.291502520nat, passing two of the three preregistered conditions. Exact matches remained0/8 to0/8, failing the required gain of at least2. The full format-alignment criterion therefore FAILED. Mean lexical F1 changed from0.107124677 to0.118402778. Full per-case effects, including the negative likelihood change, are retained. This shows a format-related influence on likelihood, not demonstrated answer correction or medical accuracy. We will not sweep further prompt variants.

Next causal test: restore the original prompt and keep the same8cases/rounding/protection/budgets, but replace the weak mean-logprob-plus0.1 target with minimum target-token logit margin over all content tokens and EOS. Greedy-answer exactness must be checked separately in native generation; the numerical margin alone is not sufficient. This tests whether the objective is too weak without relaxing protection or expanding the edit budget. It remains original train-answer reinforcement, not verified fact replacement.

New cost0GGN/23.620829069seconds; cumulative2430GGN/17802.541567989seconds.45CPU tests passed locally and remotely. PID1390189 exited, no failure receipt. Raw questions/references/outputs/tokens remain private. Original3image/train-support/unknown-scope limitations persist.
