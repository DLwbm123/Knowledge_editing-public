# Next authorized execution order — 2026-09-13

User order: C_NO_H single and sequential; then BE sequential using its existing
single experts; delete checkpoints only after their final required consumer;
then LoRA-Perf-v1, GRACE and BELoRA, each following its own method state semantics.
No additional algorithm or performance qualification is introduced.

The new C_NO_H single adapter consumes the same frozen main T0 N=146 / E_U=146
ledger. It reuses the original native CP -> A2(80) -> CP-W0(320) -> free rank4
continuation(320) training helpers, now accepting the true Stage17 EditorRecord.
The former MedMKEB callers retain their defaults. Stage17 uses its locked ledger
seed for continuation and seed base 20260912 for native/A2 initialization.
The shared BE Base router is read from matched existing single states; BE is not
retrained, and C_NO_H never loads the BE writer into its active model. Router state
is saved separately for later bank construction.

The C_NO_H loss remains .5 native CE + .5 legal native-only fit CE + .01 U KL,
with the original optimizer/normalization and 73,728 FP32 parameters. Missing U
is rejected. Full training/input/runtime bindings precede per-edit checkpoints;
save/load and actual native routed-generation parity are checked. Base remains
frozen and routing runs with the residual hook off, no gold/edit identity passed.

Use a clean isolated runtime commit and existing training environment; keep source
data and models shared. No model download, dataset expansion, new clinical signoff,
old training rerun or preemptive checkpoint deletion. A 12GiB admission allowance
and per-step 8GiB free reserve protect working storage. Frozen tasks are not silently
discarded to fit the budget. Infrastructure failure stops with evidence.

This single adapter does not pretend to implement the sequential stage: that
consumer remains required next, before BE sequential and cleanup. The detached
worker exits after its 146 single trajectories. Judge material preparation can run
after generated outputs exist; it is not a prerequisite for the next GPU consumer.
