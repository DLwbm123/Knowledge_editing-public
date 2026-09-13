# Stage17: accepted Astra Base, frozen queues, first GPUHome dispatch

Status at 2026-09-13 10:24 Asia/Shanghai: **RUNNING, NOT COMPLETE**.

The new rented server's physical GPU0 is an RTX 4090. The old campus GPU1/3 servers are not this dispatch's target. The previous scheduled wake remains paused; no new monitor or semantic retry was created.

## Completed preparation

- Astra Base: 50/50 batches, 2,465/2,465 unique accepted verdicts. Recovery completed at 2026-09-13 01:52 UTC. The original first 40 accepted batches and failed transport evidence were preserved; no semantic resampling or historical Stage15/16 judging occurred.
- The existing model and vision snapshots match the frozen revisions. The old Stage16 checkout is preserved; Stage17 runs from a separate clean checkout. A project-local environment pins Torch 2.6.0, torchvision 0.21.0, Transformers 4.51.3, tokenizers 0.21.1, PEFT 0.19.1 and Accelerate 1.14.0. The preinstalled environment was not replaced.
- The migration had omitted 500 required T1G derived images. Exactly those existing images were copied (46,231,471 bytes), not regenerated. 779 required evaluation images passed a readability check. No model/dataset was downloaded again. LLaVA source cloning initially encountered an unnecessary LFS-data checkout error; code-only checkout was recovered without downloading its training corpus.
- Actual data mount: ext4, 118GB volume, about 87GB free after environment preparation. A small create/write/fsync/read probe passed. Large files remain on the data disk.
- [Cohort/support summary](GPUHOME_COHORT_SUMMARY.json): T0 N=146; the eight task-specific structures contain 594 unique natives in total, **not one shared N=594 stream**. Original amended T0 order and frozen task-specific orders are retained. No repeated complete native-input identities were found.
- Main T0: E_U=146; E_H=E_HG=E_HG_eval=0. Across all task-specific natives: E_U=594, E_H=2, E_HG=2, E_HG_eval=0. These are bounded support counts from 29 already role-isolated train QA on three source groups, not a claim that no other legal source could ever exist.
- A pre-execution draft incorrectly required a recognized H proposition family for U preservation. This extra restriction was removed before any student output; the draft is preserved privately/publicly as superseded metadata. No queue was selected by student performance. No source was clinically signed off by the agent.
- [One-DEV GPU migration check](GPUHOME_MIGRATION_SMOKE.json) passed all 12 checks in 19.1 seconds, including token parity with the old machine for that one DEV input. Peak allocated/reserved memory: 15,316,638,208 / 15,393,095,680 bytes. This is not a claim of all-input bitwise parity or full training peak.
- Eleven Stage17 CPU checks passed. No semantic performance qualification gate was added.

## Actually launched

Runtime source commit: `24119b055ee07550662fb440847c38fd4c2b4e0f`.

Run identifier: `medtrace_stage17_20260913_r01`. Start: 2026-09-13 02:23:28 UTC. Physical GPU0. Detached PID at startup: 1190, parent PID1. The visible Python command uses only neutral `job`/`main.py`/`run` names; private paths are supplied through the environment.

This first dispatch is **BalancEdit adaptation, single-edit main T0-146**, with all attached legal task probes and R0/RC/forced single diagnostics. The first native completed its frozen 50 training steps and entered generation; observed process memory was 15,494MiB. This is startup evidence, not campaign completion or semantic success.

The worker uses the existing 50-step full-linear editor, a native-only fit router anchor, exact realized input/Base bindings, immutable per-edit checkpoints and per-query output resume. It resets single-edit state and routes from Base without gold/true edit IDs. The first saved expert is read back and checked. Failure records stop automatic continuation; low scores do not cause resampling. No raw answers, tokens, source mappings, weights, credentials or host access details are published.

Storage requires staged dispatch: 594 full FP32 BE matrices alone would exceed the remaining disk; 146 matrices require about 34.3GB, plus reserved output space. Other task cohorts remain registered, not scientifically dropped. The current worker exits after this 146-edit dispatch; it does not pretend to schedule unimplemented branches.

## Still not launched/completed

C_NO_H's support is frozen but its Stage17 training adapter is not yet dispatched. C_FACT/EXTRA have only the two task-specific supported candidates and no independent H-eval PairCorrect panel. LoRA-Perf, GRACE, BELoRA, remaining task-specific BE edits, actual cumulative sequential systems, student Astra scoring, final comparisons and experiment closeout remain pending. T5, sealed/qualification sources and new clinical facts remain excluded.

The public preparation overlay may include portability/documentation-only follow-up commits; the exact running worker remains at the runtime source commit above. A running GPU job is not a completed experiment, and a publication receipt must not imply otherwise.
