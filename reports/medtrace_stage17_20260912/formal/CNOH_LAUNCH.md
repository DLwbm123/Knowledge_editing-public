# C_NO_H single queue started — 2026-09-13

Runtime commit: `8bf319c1d37412393174de1ae2e2128872b8fc69`, verified clean on an
isolated remote worktree before launch. Source and the Qwen completion report were
pushed publicly before the dispatch. Existing training environment and Base/vision
weights are reused, with no installation/download or shared source overwrite.

One detached worker launched on the user-authorized rented RTX4090 GPU0, with
24,082MiB free at admission. Neutral parent command and GPU process names verified;
parent PID1 confirmed. The real data filesystem is ext4 on the data mount; a small
write/fsync/read probe passed. No existing checkpoint or Qwen artifact was deleted.

Observed state: RUNNING / INITIALIZING, edit 1 of 146, completed edits 0. GPU process
memory at observation: 17,162MiB. This is startup evidence, not completed C_NO_H
training or an estimate of its full peak/time. Model-loading vision-key warnings
were emitted by the existing frozen loader; the worker proceeded to initialization.

The single worker stops at completion or failure, preserving per-edit state. The
required C_NO_H sequential -> BE sequential -> dependency-approved cleanup -> other
baselines order remains registered, but is not falsely represented as an implemented
automatic multi-stage coordinator. No repeating monitor or new experiment is added.
