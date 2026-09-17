# Stage18 existing-data DEV16

Current version: the user requested existing data only. A frozen source-only pool yielded 116 candidates; GPU3 generated 151 deduplicated fresh Base responses, all anonymously judged by Astra. After source-image and relation review, 16 Base-wrong edits qualified for an exploratory pilot. The GPU3 background four-method run has started; student training and scoring are not reported complete. See [current report](../../reports/medtrace_stage18_cfact_20260917/CURRENT_DATA_DEV_V2_ZH.md) and [counts/startup](../../reports/medtrace_stage18_cfact_20260917/CURRENT_DATA_DEV_V2.json).

Two earlier source-label smoke edits passed all three C continuation branches (320 updates each); H/G each contributed nonzero gradients at all 640 applicable updates. Those two smoke edits are mechanical validation, not the DEV16 outcome.

The new DEV uses C_NO_H, C_EXTRA, C_FACT and BalancEdit on the same frozen 16 edits, single and true expert-bank sequential prefixes 1/10/16. Native sources cover 13 images; H_eval has 7 distinct QA on 4 images, reused across 18 slots. Patient/study independence is unknown. Prior training exposure is permitted for this versioned DEV, while historical evaluation-only, calibration, sealed and reserved sources remain excluded from training. No patient-independent or confirmatory claim is supported.

This source overlay reuses the frozen Stage15 runtime/methods, Stage16 overlay and Stage17 overlay. Assemble those using the Stage17 README, then overlay these scripts and tests. A compatible existing model/environment and authorized private manifests are required. No model, images, private QA, raw answers/tokens, or checkpoints are distributed.

CPU validation: `PYTHONPATH=. python -m unittest discover -s tests -p test_stage18_cfact.py -v` (12 passing checks). The DEV implementation commit is `0e9e8458ac7dab5018930837047d549aa294d20b`; subsequent report commits do not alter its runtime.

- `stage18_existing.collect/build/freeze` derives a source-only candidate pool, then validates fresh Base/Judge and AI review bindings before freezing a maximum of 16 eligible edits. It downloads nothing.
- `stage18_base.py` generates Base; `stage18_base_judge.py` uses one isolated anonymous Astra batch, without semantic retries.
- `stage18_smoke.py` accepts only its original two-edit smoke or the explicitly qualified existing-data DEV schema. Training inputs exclude H_eval/U_eval.
- `stage18_dev.py` reuses shared-W0 C training and the existing BalancEdit implementation, checks saved states and Base parity, and generates single and full-prefix deployment outputs. Student scoring remains a separate consumer using the frozen Astra protocol.
- `stage18_acceptance.py` checks completed smoke receipts and actual H/G updates. V1 reports remain historical snapshots, superseded only for the explicitly versioned V2 cohort.

No historical experiments or datasets were modified. Checkpoints remain during active registered consumers; cleanup requires complete generation/recovery evidence and an explicit current-run file list under the approved lifecycle.
