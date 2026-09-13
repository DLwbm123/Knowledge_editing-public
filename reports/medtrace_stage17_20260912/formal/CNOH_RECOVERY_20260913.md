# Stage17 C_NO_H recovery at edit 95

The single worker completed 94/146 edits, then the native initializer stopped
with `base restoration or sampled guard failed`. The campaign and local
follower stopped on that failure. The user authorized repair and restart.

A fresh, unedited Base on the same rented GPU reproduced the discrepancy:
both the formal generator and the initializer's generator returned the same
52-token sequence, while the frozen historical Base sequence has 46 tokens.
Their first divergence is at zero-based token 41. Prompt IDs, attention mask
and source-image identity matched the frozen binding. The sampled Base guard
reported no changed parameters or trainable Base parameters. This establishes
that a mismatch already exists before editing; it does not identify which
hardware/library difference caused it, or establish semantic equivalence.

The bug was using the historical cache as the sole restoration reference.
The initializer now captures a same-runtime Base output before editing and
requires exact token/text restoration afterward, alongside the parameter
guard and native checkpoint reload check. It separately records whether that
pre-edit output matches the frozen Base cache. The extra pre-edit generation
precedes the existing RNG reset, preserving the initialization seed sequence.
Frozen Base answers, correctness masks, Judge inputs, method parameters and
the registered cohort remain unchanged. Historical-cache mismatch is explicit
in the private initialization receipt; it is not reported as exact parity.

An unfinished native checkpoint cannot now be reused without its successful
validation receipt. Edit 95's failed partial directory is retained as private
recovery evidence, and only that unfinished initialization is retried. The
completed 94 edits retain their original bindings and source commit; resumed
edits bind to the repaired runtime commit through an explicit recovery dispatch.
No Stage15/16 experiment or accepted judging is rerun.

Validation: the regression check accepts unchanged pre/post output while
reporting historical-cache drift, and rejects actual output or parameter
mutation. Two initializer checks and 22 Stage17 CPU checks pass. Real-GPU
recovery observations are recorded separately after deployment.

The original finite campaign will again wait for C_NO_H completion, reuse the
existing BalancEdit experts, and use the external LoRA gate before proceeding
to GRACE/BELoRA. The existing local follower is resumed for collection,
Astra/high judging and public aggregate reporting. The my-gpu LoRA process is
not restarted.
