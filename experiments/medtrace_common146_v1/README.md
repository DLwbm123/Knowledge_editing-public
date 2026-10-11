# Common original146 comparison

CPU-only admission, one common Astra medium epoch, and fixed-mask reporting.
There is no model-loading, training, or generation entrypoint in this directory.
Read `reports/medtrace_common146_v1_20261011/PREREG_ZH.md` first.

The operator supplies `RUN_ROOT` pointing to a new private run. `PLAN_CONFIG.json`
contains `TT_parent` and `baseline_parent`, referring to the completed post32 and
full-method-comparison-v2 runs. Historical parents are read-only. The new ledger
starts with zero GPU seconds/sessions and zero Judge attempts; the manifest caps
Judge attempts at 3146. This is separate from the online 2x2 ledger.

Reuse the existing deployment layout under `private/tools`:

| Local source | Deployed name |
|---|---|
| `admit.py` | `admit_common146.py` |
| `queue.py` | `common146_queue.py` |
| `report.py` | `common146_report.py` |
| `../medtrace_post32_closeout_v1/astra_queue.py` | `astra_queue.py` |
| `../medtrace_post32_closeout_v1/astra_report.py` | `astra_report.py` |
| `../medtrace_full_method_comparison_v2_20261003/judge_queue.py` | `astra_base_queue.py` |
| `../medtrace_full_method_comparison_v2_20261003/baseline_admit.py` | `baseline_admit.py` |
| `../medtrace_tt_u_continuation_v1/common.py` | `common.py` |
| `../medtrace_tt_u_continuation_v1/report.py` | `report.py` |

The existing source-agreement helpers are resolved through the historical
`private/source`. The baseline admission additionally needs the original
`legacy_stage17`, `baselines`, `cpu_gate`, `METHOD_LOCK.json`, and
`BASELINE_METHOD_LOCK_ADMISSION.json` under the new private root. Directory links
are read-only inputs; admission receipts and the new queue are written only in
the new root. No checkpoint is required.

Run `admit_common146.py` through a neutral entrypoint. It validates all frozen
raw consumer digests, original token/input identities, method/phase locks, and
duplicate slots. Identity digests here are required by the scoring protocol,
not bulk file-copy integrity checks. It must produce `public/ADMISSION.json`
with PASS before any score request. It prepares 17166 consumers/3146 payloads,
with no inherited scores. A failed admission is not an invitation to generate
replacement model outputs.

Use the existing Mac `scorer.py` with four disjoint `JUDGE_WORKER` values,
`JUDGE_QUEUE_FILE=common146_queue.py`, `JUDGE_EFFORT=medium`, and the preregistered
epoch. Each worker must have a new protected state directory, its own result
subdirectory, and all seven read-boundary checks passed before startup. Preserve
the existing shared-root isolation, neutral argv, 600-second request cap,
one-attempt policy, and validated-response-only recovery. The queue's outer lock
serializes all budget reservations.

`SELF_CHECK=1` runs the report's projection/missing-bound and inherited bootstrap
self-checks without scoring. Once the global queue is terminal, run the report
entrypoint. It audits every score against saved model/isolation evidence and
reproduces the historical TT25 panels separately before producing 75 new panels,
50 paired comparisons, insertion transitions, CSV, and Chinese report. Verify
the four scorers have exited before publication. Only anonymous public artifacts
are published; raw medical inputs, tokens, bindings, scores, and evidence stay
private. Historical failures and results are never overwritten.

## Authorized missing-only recovery

The subsequent user request authorizes one additional attempt for the original
200 missing payloads. Deploy `recovery_queue.py` as `common146_recovery_queue.py`.
It uses a separate `judge_common146_medium_recovery1` queue and
`Astra_common146_recovery_batches` ledger, copies only the exact missing inputs,
and retains the first-pass queue unchanged. Preserve the old manifest before
raising its cumulative Judge cap from3146 to3346. Use new isolated scorer state
and epoch `MEDTRACE_COMMON146_MEDIUM_RECOVERY1_20261011`.

After global recovery completion, run the same report with
`COMMON146_RECOVERY=1`. It verifies the exact200-key authorization, one additional
attempt each, unchanged records/bindings, and both passes' saved evidence before
merging accepted recovery scores. Outputs use the `RECOVERED_` prefix; original
reports stay immutable. No second recovery or successful-score rejudging is
implicit in this amendment.
