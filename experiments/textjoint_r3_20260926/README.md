# TextJoint-R3

Frozen review: `532b81be5b96848e5ccd9d46e8252d1155fe6aac`. New worktree and run; R2 is read-only evidence. This branch stacks on R2, without merging PR #1 or #2.

`setup.py` creates a one-time manifest with training/generation/target/hard boundaries at +9/+11/+12/+14 hours. Historical GPU residency and Judge submissions are carried forward. `audit.py` checks actual 80-step optimizer states and the unchanged data panels. `calibration_data.py` consumes separately reviewed private native-only paraphrases and isolated acquisition questions; no raw medical rows are released here.

`worker_r3.py` loads verified P/PS80 or trains only missing PS80 from the identical P-W0. It never interprets a `FINAL.pt` filename as proof of optimizer steps. `router_r3.py` freezes 21 CPU-calibrated rejection settings before formal generation. `budget.py` reserves concurrent GPU time atomically; `orchestrator_r3.py` runs finite dependency phases, alternates writer/GPU assignments, and adopts only matching live worker receipts after a controller restart. It does not retry failed jobs.

The local `scorer.py` uses the unchanged, isolated Judge runner with the existing explicit proxy. Set private `SCORER_STATE`, `SCORER_RUNTIME`, and `RUN_ROOT`; launch through a neutral entry. Failed requests remain missing, and existing scores are reused only for matching complete Judge payloads. The run preserves the hash of its actual deployed scorer entry separately.

`report_r3.py` separates standard retention, pressure retention, forced-on diagnostics, positive/negative routing, S and routing main effects, and their interaction. Edit bootstrap is primary; source clustering is sensitivity analysis. Missing bounds are not confidence intervals. All VERIFY results are exposed-panel extensions, never independent confirmation.

Run `selfcheck.py` in the existing environment with `RUN_ROOT` set; it checks rejection edge cases, missing paired bounds, and concurrent reservations. No new dependencies are required. Runtime data, checkpoints, per-question evidence and credentials remain private.
