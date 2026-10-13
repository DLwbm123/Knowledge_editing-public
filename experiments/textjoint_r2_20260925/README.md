# TextJoint-R2

Frozen review base: `97baf0f61351cc06b3bfcda8ae0d79d441668a3e`.

The controller runs only B0, P and P+S at L30/rank4. Both P recipes save
80/160/320 along one trajectory; P+S starts from identical P-W0 and adds only
continuation protection. `supports_r2.py` preserves original U and chooses
isolated native-task-matched auxiliary examples, with explicit semantic-review
exclusions and real shortfalls. `coverage.py` freezes Base-correct pressure
panels before reading edited outcomes. Standard T2L stays separate.

`worker_r2.py` reuses the reviewed `worker_v3.py` and source overlay. Environment
variables select the private run root; neutral launchers keep sensitive paths
out of process arguments. The imported runtime, not just a nominal source tree,
is recorded for each job. No model, dataset, raw answer, per-item private score,
or credential belongs in this directory.

Execution order: two-edit B0/P and P+S canary → cross-GPU floor → full scoring
closure → DEV24 B0/P/P+S → global recipe/step freeze → VERIFY24 B0/P plus P+S
if selected → actual current-bank prefixes 12/24 → standard BalancEdit if
budget permits. Missing scores remain missing; failed Judge requests are never
retried. Empty DEV T1L cannot pass the engineering gate. With no qualified
candidate, P@320 is a predeclared exploratory verification fallback, not a win.

Official evaluation is restricted to the previously frozen Base-eligible
Fix/Retention masks; this reduces irrelevant generation without changing either
estimand. Full original questions and masks remain private. The supplementary
pressure panel is distributed across edits before candidate training. No
independent CONFIRM is claimed, including for the historically exposed VERIFY
edits. Source-disjoint pressure inputs do not make exposed edits independent.

The original start/deadlines and shared ledger survive restarts. Two GPUs are
charged cumulatively; 25% GPU budget is reserved from DEV expansion. The prior
704 Judge attempts remain charged against the existing 6000 cap. Controller
exceptions stop expansion and preserve checkpoints. The public report is
produced from receipts, not GPU idleness.

Checks: `RUN_ROOT=/tmp python report_r2.py` and Python compilation. Real canary
receipts supply model/gradient/conversion/reload checks; the self-check is not a
substitute for those experiments.
