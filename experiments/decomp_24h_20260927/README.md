# Bounded 24h CP/Tucker phase

Reference: `7dbf60f9cedb98d4ac52269567804f2d29b2aabc`.
This is a new phase, not a reset of R2–R4. Fresh setup requires `LEGACY_STORAGE_PREFIX` to identify the private source storage root; no personal source path is embedded in the public code. Runtime/model/data overlays stay private and read-only. Use `MODULE_PATHS.json` and `VERSION_LOCK.json` from the execution root to resolve imports; this source folder alone is not a runnable data release.

The finite controller admits resource-feasible paired blocks after the two-edit pilot closes training, verified save/load, generation and the fixed Judge. The target is 720 main units plus 96 additional CP-P teacher units. Seed 1 compares all seven methods on DEV24 and exposed REG24; extra seeds cover the four core methods. Sparse sequential prefixes are 12 and 24. Extra seeds and optional fixed-DEV4 structured continuation/rank8 are admitted only after prior paired blocks and scoring close. Further optional families remain deferred.

The hard limits are the original 24h wall deadline, 72 new GPU-hours on physical GPUs 5/6/7 and 6000 lifetime Judge attempt items. Restarting a process cannot reset any counter. GPU session leases, neutral entries, PID/cwd verification and shared ledgers prevent duplicate workers. `RUN_ROOT`, `PHYSICAL_GPU` and a live-verified `CUDA_VISIBLE_DEVICES` UUID are provided through the environment, never project-bearing visible launch arguments.

`storage.py` reserves atomic-write peaks and protects historical assets, pinned adapters, active readers and outstanding W0 consumers. Only rolling latest/previous optimizer state is retained. Final adapters keep normalized-input hook semantics and are not merged into Base. CPU diagnostics must use `CUDA_VISIBLE_DEVICES=''`; conversion forks RNG state only on its expert's device. The private run records the corrected CPU-test device-initialization deviation and charged cost.

Primary eligibility uses frozen historical Base masks; new-backend Base is a separate parity audit. Missing scores remain missing, empty denominators are not PASS, and these exposed panels do not establish independent confirmation. Reports separate observed cohort completeness from the full 24-edit target. Full-vocabulary teacher logits are bounded in memory and never published.

Run the CPU storage/serialization checks via `selfcheck.main()` within the existing environment. Actual model canaries and `audit.main()` additionally check gradients, algebra, runtime imports and support separation. `scorer.py` reuses the isolated historical Judge runner and quarantine mechanism; it requires the operator's existing local authenticated CLI and the same explicit proxy. Successful full-input-identical Judge decisions may be reused; failed requests cannot be retried by renaming them.

Publish only source and checked aggregate files from `run/public`. All private role tables, raw answers/tokens, per-question scores, checkpoints, model assets, credentials and process receipts remain private. A running phase is not a completed or published result.
