# Rented-server exit handoff — 2026-09-15

Closeout update: GRACE final transfer has passed (146 native + 3213 panel outputs).
Both temporary 146-file recovery copies were strictly cleaned after validation.
The rented host is no longer a dependency. See GRACE_SEQUENTIAL_COMPLETION_20260915.md.
The initial snapshot and pending-state description below records the earlier handoff.

The user requested preserving the required material on the existing persistent
server before the 24 GiB server rental ends. A bounded snapshot of nine Stage17
run directories was transferred successfully. Its verified run payload contained
19,461 files / 1,078,027,880 bytes; ongoing GRACE outputs can increase these counts.

The snapshot includes generated outputs/tokens, frozen bindings, configurations,
training/completion/cleanup records, operational entries and relevant diagnostic
provenance. Existing persistent-server models, images, CPU gate and environment
are reused. Accepted Base/BE judge bindings, verdicts, locks and execution records
are also preserved privately. Referenced scientific source commits were verified
in the destination Git object database; source updates use an isolated checkout.

GRACE sequential was still running at transfer. All 146 saved states required by
the unchanged resume loop were preserved (578,859,276 bytes), and the latest state
loaded successfully on CPU. The verified final-panel snapshot contained 1153/1512
outputs. These states are a temporary recovery set, not a permanent checkpoint
archive. Completed methods' unneeded weights and generated router-bank tensors
were excluded; their existing output/route and validation records are retained.

A finite background process copies incremental GRACE outputs every five minutes
and validates the final 146 native plus 3213 panel records after real completion.
The checkpoint snapshot and transport copies become eligible for strict lifecycle
cleanup after that validation. The initial snapshot is complete; final output
transfer and snapshot cleanup remain pending while GRACE runs. If the source
becomes unavailable early, recovery material remains available; no GPU is silently
reallocated for continuation.

The BELoRA bridge, baseline relay and overall acceptance follower now target the
persistent server. A CPU-only coordinator marks readiness only after validating
real phase receipts and external-import bindings. This removes the rented host
from subsequent BELoRA/LoRA result-return and final judging dependencies. LoRA
BF16 remains paused and GPU0/1/2 remain unassigned. Active GRACE and BELoRA GPU
workers were not interrupted. The local computer must remain online for transfers
and the existing local judging follower.

The acceptance coordinator uses isolated source commit
`c07c004fc23f78c15d03e6d144d7ac44b37d35cd`; generation continues with its original
scientific commits. Targeted handoff and split-gate tests pass; deferred final-copy
validation compiles, and the first live incremental copy succeeded. An unrelated
legacy DEV16 test could not collect under system Python because Torch is absent;
that environment limit is not a failure of the handoff tests or GPU runtime.

Public material contains only orchestration code, tests and this aggregate report.
Private questions/answers/tokens, data paths, model or generated weights, judge
bindings and credentials are excluded from publication. Rental cancellation was
not performed, and this report does not claim completed generation or scoring.
