# User-authorized two-GPU migration

On 2026-09-13 the user cancelled the delayed GPU2 restart and authorized immediate
migration of the existing LoRA experiment and part of the rented-server pending
queue onto my-gpu GPU0 and GPU1.

- GPU0 resumes the separate BF16 LoRA sequential variant: 45 edits were complete,
  edit 46 had a saved training/checkpoint receipt, and its evaluation resumes.
  The existing runtime commit, precision, order, seeds, profile and output
  bindings remain intact; only physical GPU selection changes. Historical FP16
  single results and the FP16 sequential numerical failure are preserved.
- GPU1 executes the still-unstarted GRACE single, GRACE sequential, BELoRA single,
  BELoRA sequential phases in that order. These methods have no weight dependency
  on LoRA or C_NO_H. Their original FP16 scientific recipes remain unchanged;
  BELoRA retains the disclosed 50-step effect-repaired implementation.
- The rented server continues its active C_NO_H single, then its dependent C_NO_H
  and BalanceEdit sequential work. It does not retrain externally assigned
  GRACE/BELoRA phases: future child entry calls wait for provenance-validated
  external imports. The active predecessor and queue parent are not stopped.

The existing external worker/relay is reused with a bounded GRACE/BELoRA phase
assignment. Their assignment, staging and completion markers are separate from
the original FP16 LoRA gate. Small JSON/JSONL outputs are relayed; weights remain
on their compute host and are deleted after registered consumers and validation,
subject to the existing mount/path/open-file safety checks. Cleanup visibility
blocks defer deletion, not trigger retraining. BF16 LoRA cannot silently satisfy
the original FP16 completion gate; that known scientific dependency remains.

The GPU1 checkpoint budget is 48 GiB plus the 8 GiB disk reserve. GRACE stores
small codebook values; BELoRA cumulative snapshots may total approximately 35.4
GiB of adapter tensors over 146 prefixes, plus its single snapshots and metadata.
Single and sequential consumers, save/load checks and registered prefix panels
remain required before cleanup. Shared Base/environment/data assets are reused.

The delayed GPU2 action is cancelled. The existing hourly monitor follows GPU0,
GPU1 and the rented server, preserving active resume state and reporting only
meaningful changes. No extra method, hyperparameter search or dataset expansion
is authorized by this migration.
