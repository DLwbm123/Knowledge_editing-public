# Frozen-C1 visual-verifier recovery source

This source snapshot contains the R1 recovery runner, finite coordinator, frozen verifier and the existing modules they import. Research execution is pinned separately in the linked [source manifest](../../reports/medtrace_visual_verifier_recovery_20260907T045800Z/EXECUTION_SOURCE_MANIFEST.json). The public repository does not contain the research Git history, private data, model weights, features, token outputs or Judge mappings.

Run the CPU fixtures in an existing compatible Python/Torch environment:

```sh
cd experiments/medtrace_visual_verifier_20260907T045800Z
CUDA_VISIBLE_DEVICES= PYTHONPATH=. python -m unittest discover -s tests/medtrace -p 'test_*verifier*.py' -v
```

The fixtures create synthetic inputs in temporary directories and never load a model. They cover the real prepare/start/preflight chain, concurrent starts and task claims, constant ON/OFF decisions and replay failure. The fixture's base reference is the current checkout HEAD so it can run in this separate public history; production commit and ancestor checks remain enabled.

Production execution requires the lawful frozen V4 inputs/runtime, original seven hard-supported edits and three seeds, source-matched C1 additional-step-800 checkpoints, original semantic Judge protocol and the authorized GPU UUIDs. Use the runner's `prepare`, then coordinator `start`/workers/Judge/finalize sequence; do not create or backdate a campaign start manually. A fresh execution in a different history needs an explicit new provenance record. The coordinator's paths are the verified original server environment, not portable model defaults.

All four route conditions use the same frozen executor. M2 and M3 add two supervised linear heads per edit/seed; neither is parameter-free routing. This release is development diagnosis, not a full MedTRACE/TIME or clinical validation.
