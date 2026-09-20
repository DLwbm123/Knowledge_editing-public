# R1 evidence closure for Stage24E-R2A2

## Scope

R2-A2 is zero-GPU/zero-Judge CPU reconciliation and isolated integration. R1 remains an exploratory published endpoint record and is not promoted or re-scored.

## Recovered evidence

- Frozen plan: `reports/medtrace_stage24e_20260920/private/COMPILED_PLAN.json` (23 slots; 57 compiled reference events).
- Generation rows: 291 total; final prefix-19 rows: 237 (79 per arm).
- Existing score cache was used without semantic rejudging. Prefix-19 coverage is 197/237 matched and 40/237 unmatched; missing entries remain null.
- R1 receipt records 30 new Judge items and 92 exact cache reuses. A separate remote common snapshot reports 0 dispatched; the namespaces, request IDs and time scope are not present in the available local receipt, so the discrepancy is not recoverable here.

## Historical-unrecoverable or unavailable

- The compiled `ONLINE_EVENT_LOG.jsonl` is a schedule/compile artifact, not a runtime transaction log. No worker-side PREPARED/WRITTEN/EVALUATED/RELEASED trace, immutable parameter-version trace, actual teacher trace, mask-consumption trace, or release barrier was supplied.
- Raw request/response and cost records needed to map every logical score task to a physical request are unavailable in the checkout.
- Therefore no missing score is filled, no old event is reconstructed, and the original R1 totals remain preserved as published.

## Boundary

The missing historical fields limit conclusions about R1 protocol execution only. They do not make the new isolated CPU integration impossible; that integration is delivered separately and is not a real-model acceptance.
