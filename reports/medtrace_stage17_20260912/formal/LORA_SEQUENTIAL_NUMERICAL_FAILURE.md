# Frozen LoRA sequential numerical failure

LoRA single completed 146/146. The separate cumulative sequential trajectory
completed 16/146 edits and failed during edit 17 with a non-finite edit loss.
This is separate from the earlier checkpoint-cleanup visibility problem.

The last valid checkpoint (after edit 16) contains 192 finite adapter tensors,
with maximum absolute value 0.09489505. Its history exactly matches the first
16 registered edits. Loading that checkpoint into a fresh resident Base gives
finite inputs, 27 target tokens and initial edit-17 loss 23.57951927.

One diagnostic reproduction of the failed edit used the unchanged frozen
recipe and did not overwrite formal checkpoints. The first three losses were
23.57951927, 19.04458046 and 15.61536503; forward computation at step 4 produced
NaN. Leaf-module instrumentation first observed a non-finite output at language
layer 31 input normalization, in FP16. Adapter parameters remained finite and
no preceding non-finite adapter gradient was observed. This identifies a
reproducible forward numerical failure; it does not establish which earlier
operation first caused the invalid value.

The FP16 sequential branch is blocked, with its first 16 completed edits,
checkpoint and failed-attempt evidence preserved. It is not a completed
146-edit trajectory, and the missing outputs must not be treated as semantic
incorrect answers or silently dropped. The unchanged recipe is not retried
automatically. A separately labeled precision/stability variant requires
explicit authorization; no learning rate, step budget, seed, sample order,
precision, evaluation reference or accepted Judge result has been changed.

The rented-server C_NO_H experiment continues independently. The existing
queue's external LoRA dependency remains blocked until an approved disposition
of this numerical failure; no completion receipt is fabricated.
