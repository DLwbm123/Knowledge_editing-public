# Stage18: full FACT/EXTRA smoke and source qualification

Latest: Astra reviewed 10 H relations (2 supported, 8 unresolved); GPU3 generated all 13 fresh Base outputs and isolated Astra judged them. Strict eligible pilot N is 0; no new student training was started. See [review and Base qualification](../../reports/medtrace_stage18_cfact_20260917/ASTRA_REVIEW_AND_BASE_ZH.md).

Two source-label smoke edits completed all three 320-update continuation branches. This is mechanical validation, not formal performance. See [smoke acceptance](../../reports/medtrace_stage18_cfact_20260917/SMOKE_ACCEPTANCE_ZH.md) and [data construction](../../reports/medtrace_stage18_cfact_20260917/DATA_GAP_AND_BUILD_ZH.md).

This source overlay reuses the frozen Stage15 runtime/methods, Stage16 overlay and Stage17 overlay. Assemble those in a separate directory using the Stage17 README, then overlay this directory's scripts and tests. Existing compatible model/environment and authorized private manifests are required; no model or patient data is distributed here.

CPU check: `PYTHONPATH=. python -m unittest discover -s tests -p test_stage18_cfact.py -v` (9 checks). Original core tests and lossless transfer test are inherited. Regression source identities are fictional. GPU smoke was executed from private implementation commit `0d54456c96742c1e8d32aef5976c4d55030e9077`; acceptance/pilot construction were subsequent read-only CPU work.

Logical helpers (private paths only via a neutral launcher/environment):

- `stage18_base.py` generates the frozen candidate Base-only packet; `stage18_base_judge.py` performs one isolated anonymous Astra batch. Both read `JOB_CONFIG`. `stage18_pilot.qualify` checks frozen input/verdict bindings and reports the intersection without upgrading unresolved reviews.
- `stage18_smoke.py` reads `JOB_CONFIG` and only accepts its frozen two-edit smoke scope. Do not rerun completed work or use it to bypass formal qualification.
- `python -m scripts.medtrace.stage18_acceptance RUN OUTPUT.json` verifies completed receipts, W0, losses, schedules, gradients and diagnostics.
- `python -m scripts.medtrace.stage18_pilot SOURCE_PRIVATE STAGE18_PRIVATE` builds the specific existing-source DEV candidates, consumes the private exposure audit and never dispatches training. It is a fixed, score-independent candidate recipe, not a general data downloader. Clinical/patient relationships remain unverified.

Private source-role ledger, image/QA review packet, raw answers/tokens and checkpoints are deliberately excluded. Published counters distinguish three constructed source packages from zero fully qualified launch-ready pilot edits. Stage17 reports are unchanged.
