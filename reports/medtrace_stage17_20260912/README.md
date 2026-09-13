# Stage17 progress

Current status: [authorized Base Judge recovery started](formal/RECOVERY_START.json). The first 40 batches (2,000 records) are preserved; 465 records are being recovered under the same configuration. Formal GPU1 training, final support masks/adapters and method comparisons are not complete. The [scheduled-wakeup report](formal/SCHEDULED_RESUME_REPORT.md) remains the historical failure record.

The following Stage17-A files are historical handoff records; their pending-authorization statements are superseded only by [the new approval](formal/AUTHORIZATION.json). Historical T0 N=179 is not a newly judged Stage17 N.

- [Cohort/support status](COHORT_AND_SUPPORT_SUMMARY.json): historical task counts, unknowns and exact missing role proofs.
- [Method lock](METHOD_LOCK.json), [evaluation contract](EVALUATION_CONTRACT.json), [execution scope](EXECUTION_SCOPE.json), [branch manifest](RUN_MANIFEST.json).
- [CPU checks](CPU_TEST_RESULT.json), [GPU3 real-model check](GPU_SMOKE.json), [Judge preparation only](JUDGE_PREPARATION.md).
- [Source and runnable commands](../../experiments/medtrace_stage17_20260912/README.md).
- [Publication receipt](PUBLICATION_RECEIPT.json).

The private metadata ledger and DEV outputs are not redistributed. No old Stage15/16 task, scoring, data download or training was repeated.
