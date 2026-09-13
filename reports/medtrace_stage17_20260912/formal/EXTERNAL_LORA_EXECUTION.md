# Stage17 LoRA assignment to a second GPU

The user authorized moving the pending LoRA-Perf-v1 single and sequential work
to an available A100 40GB GPU on 2026-09-13. The original C_NO_H process and
finite campaign parent remain running. At the original queue's LoRA phases,
new child processes wait for the assigned worker's accepted output instead of
training a second copy. If necessary, later phases wait at this dependency.

The external worker runs all 146 single edits, then the true cumulative
sequential trajectory with prefixes 1/50/100/146. It uses the frozen cohort,
method, generation, and seed contracts. Each mode performs the existing
save/load checks and dependency-based checkpoint cleanup on its own storage.
The cleanup mount is explicitly recorded in its private dispatch.

A finite Mac relay transfers only JSON/JSONL evidence after both modes and
cleanup complete. It verifies the assigned source commit, cohort, order,
method/runtime locks and completion receipts before importing the two phase
directories without overwriting existing results. The original follower then
collects these outputs with the remaining campaign for the existing Astra/high
judging and public aggregate report. No accepted judging is repeated.

GPU failures or transport failures stop this branch and retain recovery
evidence; a failure marker prevents the original queue from silently training
a replacement. The relay, like the existing follower, requires the Mac to
remain awake and connected. GPU computation is detached from the Mac session.

The existing target environment has Torch 2.6.0, Transformers 4.51.3 and PEFT
0.19.1. Pillow 10.4.0 and NumPy 1.26.4 differ from the rented server's 12.3.0
and 2.5.3. These actual versions are recorded rather than described as an
identical environment. Existing realized-input and token checks remain active;
cross-GPU numerical identity is not claimed. No package or model reinstall is
required. This document records execution preparation, not completed results.

Validation: 21 Stage17 CPU tests pass, including rejection of mismatched
external provenance and refusal to overwrite an existing phase. Real GPU
progress is reported separately from these checks.
