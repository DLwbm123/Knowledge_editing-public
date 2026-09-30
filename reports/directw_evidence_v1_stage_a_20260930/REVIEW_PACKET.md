# DirectW-Evidence-v1 — stage A external review packet

**current_state = WAITING_FOR_EXTERNAL_REVIEW**

This packet requests review, not permission inferred from test success. Native
GPU forwards/backwards/edits: **0**. Native CPU model runs: **0**. Real-model
training started: **false**. Paid Judge calls: **0**. Automatic continuations: **0**.

## Code and isolation

Independent branch: `research/directw-evidence-v1`; parent commit:
`2c5bfc1537c553c2623a49eea023a49749f23eaf`. Code:
[`experiments/directw_evidence_v1`](../../experiments/directw_evidence_v1/).
The independent worktree has a separate ignored RUN_ROOT. No old code, adapters,
optimizer states, queues, links or checkpoints were altered. Package imports do
not enter legacy CP/LoRA runtime modules. `ISOLATION_AUDIT.json` records the scope.
`REVIEW_BINDINGS.json` binds source, installed dependencies, data-role manifest,
configuration, protocol and draft resource ledger. The full clean-model content
digest is intentionally null, so no native admission can pass.

Review the committed source with `git show --stat` and
`git diff 2c5bfc1537c553c2623a49eea023a49749f23eaf..HEAD -- experiments/directw_evidence_v1 reports/directw_evidence_v1_stage_a_20260930`.
No PR is merged. Any review-branch publication contains code and deidentified
aggregates only; the private role manifest and static metadata stay in RUN_ROOT.

## Implementation and numerical evidence

See [METHOD_SPEC.md](METHOD_SPEC.md) for function mapping and
[MATH_AND_APPROXIMATIONS.md](MATH_AND_APPROXIMATIONS.md) for the derivation.
Five candidate/control branches are implemented. Exact JVP/VJP curvature,
preservation linear term, explicit L2 slack, negative-end protection, native
weight writes, normal-forward checks, rollback, fixed history capacities and
Base+original-matrix export are executable on synthetic CPU fixtures.

[CPU_TEST_RESULTS.json](CPU_TEST_RESULTS.json) contains 20 requirement-group
results with actual seed, dtype, metrics, tolerances and timing. CPU PASS does
not mean native PASS. Tests include explicit Jacobian comparison, symmetry/PSD,
finite differences, a nonzero Fisher despite zero reference KL gradient, changing
operator rejection, independent SciPy primal comparison, both visual endpoints,
source-role rejection, rounded weights, native-key invariance, exception/Ctrl-C
rollback and a clean editor-free synthetic inference subprocess.

The recorded development history includes initial failures and their resolution.
One discovered merit bug mixed squared and unsquared units and was corrected.
A moderate-nu visual fixture retained slack and correctly failed hard acceptance;
the explicitly separate near-hard feasibility fixture uses nu=1e8. No medical
performance data or native threshold was selected using those fixtures.

## Model, data and access status

Static clean checkpoint headers confirm the proposed original down projection
at layer 21: 4096x14336, BF16, no bias. Actual instantiated model aliases,
compute precision, expanded prefixes and kernels remain pending. No adapter or
original tensor payload was loaded. See [BASE_BINDINGS.json](BASE_BINDINGS.json).

The legacy train overlay has 151 rows; its already source-isolated candidate list
contains **29 rows from 3 image source groups**. Only those train candidates'
metadata was read for this audit. Legacy source isolation alone does not prove
Direct-W edit/pair/near-protection semantics. Temporal availability and complete
exposure/fact-group information are unresolved. No certified incompatible-answer
visual pair was found; no medical negative was generated. No official eval QA
was imported into the new manifest, and no Base/Judge cache was reused.

| Branch | New verified fit/protection | Static readiness |
|---|---:|---|
| W_FT | 0 | BLOCKED_DATA |
| W_EUCLIDEAN_QP | 0 | BLOCKED_DATA |
| W_KEY_QP | 0 | BLOCKED_DATA; actual-input cache also native pending |
| W_FUNCTIONAL_QP | 0 | BLOCKED_DATA |
| W_EVIDENCE_QP | 0 qualified visual pairs | BLOCKED_DATA |

These zeros count **verified new-method eligibility**, not a claim that the
benchmark contains no edits. A legal independent DEV_CAL and TEST_CONFIRM have
not been established. Existing exposed/unknown panels cannot be renamed as
independent confirmation. See [DATA_AUDIT.json](DATA_AUDIT.json),
[ROLE_ACCESS_POLICY.json](ROLE_ACCESS_POLICY.json) and
[METHOD_DATA_ELIGIBILITY.json](METHOD_DATA_ELIGIBILITY.json).

## Resource feasibility and staged proposal

[RESOURCE_AND_STORAGE_ESTIMATE.json](RESOURCE_AND_STORAGE_ESTIMATE.json) reports
exact static matrix sizes and explicitly unmeasured native timing/peak memory.
One FP32 matrix is 224 MiB. A draft 8-row A plus 8 Q^-1A columns and 12 other
vectors totals **6.125 GiB**, before autodiff activations/full logits. The
implementation also retains CPU Base/reset and transaction snapshots. A three
protection-group, nine-solve, 32-CG-iteration worst-case draft can require roughly
**948 exact GGN calls per step**, making 20 steps a potentially expensive edit.
These are cost formulas, not a promise that a native run fits a GPU or time limit.

The read-only server inventory showed occupied GPU0–3. It neither established
availability nor granted a lease for the proposed later physical GPU5/6/7.
Authorized new GPU time and Judge calls remain zero. Storage proposals are 20 GiB
per RUN_ROOT and 2 GiB for teachers, with no per-edit full-backbone checkpoint.

[EXPERIMENT_PROTOCOL_DRAFT.yaml](EXPERIMENT_PROTOCOL_DRAFT.yaml) separates B
mechanical native smoke, C 8–12-edit pilot and D true sequential prefixes
1/4/8/12/24 with three proposed orders. Required thresholds, precision, cohort,
decode and budget fields remain null/zero. Later stages require explicit phase
authorization; smoke does not grant pilot. [NATIVE_PENDING_CHECKS.md](NATIVE_PENDING_CHECKS.md)
lists every native/data/resource gate still unresolved.

## Decisions requested from the external reviewer

1. Is the exact GGN + preservation-gradient + soft-QP formulation acceptable,
   including the fact that a converged slack solution may still be rejected?
2. Accept layer 21 as a first physical location, and choose bound native compute/
   deployment dtype and attention backend; no automatic layer search is planned.
3. Provide/confirm lawful per-role edits, near/background protection and visual
   pairs with independent labels and non-target scope, or approve only a data-ready
   component pilot later. Clarify temporal availability and supersedes policy.
4. Establish independent DEV_CAL/TEST_CONFIRM or explicitly limit conclusions
   to exploration/regression. Freeze thresholds by global authorized rules.
5. Approve or reject the exact-matvec memory/cost approach after authorized smoke
   measurement; freeze resource and whole paired-block scoring budgets.
6. If approving a next step, name the exact phase and bindings. The supplied
   [APPROVAL.template.json](APPROVAL.template.json) is **approved=false** with no
   phases, no reviewer authority and no GPU/Judge budget.

No approval is inferred or constructed. Work stops here for external review.
