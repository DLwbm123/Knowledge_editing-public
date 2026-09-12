# Stage17 scheduled continuation: network-interrupted Base Judge

The one-shot task woke at 2026-09-13 01:00 Asia/Shanghai. GPU1 had 40,445 MiB free and its authorized UUID matched. No formal GPU training was started: the complete Base mask is unavailable, and remaining support/runner work is not complete. No old experiment, model generation, Judge batch, or mechanical GPU check was rerun.

## Verified interruption

The isolated Astra queue stopped at 00:00:45 Asia/Shanghai on batch 41 after a response-stream transport error (`network error: error decoding response body`). Exit code was 1, no final response file was produced, isolation checks passed, and there were no disallowed tool events. The generic runner error mentions both execution failure and tool events; the observed cause here is specifically the network transport error, not a tool-use violation. No quota or VRAM error was observed.

All 40 earlier accepted responses were rechecked with the existing strict schema/order validator: 2,000 records remain accepted. Batch 41 has 50 unresolved records; batches 42–50 were never attempted and contain 415 records. The total outstanding packet coverage is 465/2,465. Partial scores are not treated as a complete Base mask, and the failed batch is not assigned incorrect semantic labels.

The original attempts and responses are preserved. This wakeup made zero new Judge calls and no automatic retry. The recovery requested from the user is one same-input, same-configuration retry of the network-failed batch, followed by the previously unattempted batches. The 40 accepted batches must not be rejudged, the original failure must remain in the execution lineage, and no Judge/model/prompt selection or semantic resampling is permitted. See [JUDGE_INTERRUPTION.json](JUDGE_INTERRUPTION.json).

## Score-independent preparation completed

The new `stage17_roles.py` reuses existing canonical image identities and write-once helpers. It joins Stage2/Stage5 evaluation, calibration and challenge roles with Stage14's existing 72-image evaluation list and the existing Stage13R partition; no Base verdict or student output is read. Existing image-identity closure is applied without new image hashes. A runnable test covers transitive identity closure and order-preserving native exclusion.

The remote run completed successfully at source commit `53d9efee83a9fa1e78f60706a4140d79088df22b`. It protects 121 prior-project evaluation source groups and 41 reservation groups. Thirty-one T4G candidates were removed independently of scores (178 → 147 before Base filtering). T0 remains 148 preliminary candidates, not the final Base-wrong N. The retained structures require 2,433 unique Base inputs; 32 entries in the original scoring superset will be unused and are not rejudged. See [ROLE_BOUNDARY_JOIN.json](ROLE_BOUNDARY_JOIN.json).

From the existing 151-row source overlay, 91 rows retain an adaptation/train role; 29 rows across three image groups are isolated from the current structural candidate superset. These are source-level support candidates only, not 29 proven H/U/G supports. Proposition/relation checks and the final support masks remain pending. Exposure is not relabeled as independent confirmation.

## Remaining work and boundary

Recover complete Base scoring with explicit approval, finish the exposure/support relation ledger and final Base-derived cohorts, then complete and verify the formal single/sequential adapters before launching GPU1. Do not select an easier partial cohort based on which batches finished. The role-filter artifact must be applied before any student outputs. Preserve frozen methods, source roles, heldout/QUAL, clinical-review requirements and all old results.

The one-shot automation is paused after this wakeup; it will not repeatedly restart the queue. No persistent GPU reservation, background training or new monitor was created. Publication is a separate milestone: its success or failure does not authorize recomputation.
