# Appendix E1: matched Direct-LR4 supervision

This overlay runs after the parent decomp-24h phase closes. Copy the parent runtime Python modules, then overlay this directory's worker, training, bindings, report and controller modules. Reuse the existing environment; do not upgrade dependencies. `bootstrap.py` audits parent teacher bindings, creates a new run once, carries forward lifetime Judge attempts, and reserves complete E1 scoring capacity. Runtime/private inputs and weights are never published.

The parent W0 checkpoints were reclaimed. E1 therefore rebuilds Direct-W0 and forks **both M6 and M7** from that same state; the old parent M6 is not claimed as an initialization-matched reference. M7 uses the parent's S+G+D objective, optimizer, sampler, 80 updates and verified CP-M0 teacher. “SGD” names supervision, not the optimizer.

Frozen order: two REG24 canary edits, then the remaining REG24 edits, then real M6/M7 bank prefixes 12/24. Only GPU 5/6/7. A complete E1 pair reserves 1700 Judge attempts (1440 formal worst-case consumers plus 260 canary diagnostic reserve), within the inherited lifetime 6000 ceiling. The canary requires training/save/reload/generation/Judge closure before expansion. Existing failed keys remain missing; no retry.

E2, DEV24 expansion, E3 and E4 are **not implemented or admitted by this E1 controller**. They require the already-authorized plan's separate complete-pair budget audit. E1 completion is not completion of the entire appendix. No independent CONFIRM. The original clocks and actual consumption are preserved; user-authorized compute/storage quota waivers do not authorize extra paid Judge attempts.

Canary repair: some verified parent teachers retain legacy binding metadata. Import validation compares the exact teacher file digest, stored legacy binding, previously verified parent expected binding, and actual 80-step payload. It does not rewrite the parent or weaken checks for newly trained artifacts. `test_imported_teachers.py` checks all imported teachers and rejects an incorrect expected step. Existing M6 results and Direct-W0 are resumed without retraining; no Judge request is repeated.

## E2 REG24

`E2_PROTOCOL.json` freezes the admitted complete CP/Tucker FREE4–KEEP_STRUCT
pairs. Since the parent W0 assets were reclaimed, both branches are rebuilt
from the same new per-edit W0. Each receives 80 continuation updates; STRUCT
is expanded in memory before all primary inference. `e2.py` records residual,
predictor-logit and generation parity without adding Judge diagnostics.
`test_e2.py` checks nonzero CP/Tucker residuals, input gradients, bank dispatch
and the finite 52-job dependency queue. The two canary jobs must close before
remaining jobs can claim work. The paid lifetime ceiling is explicitly
6395, with 2880 reserved at the unchanged 3515-attempt baseline. This is an
E2 launch, not evidence that E2 or the full appendix has finished.

## Full E1 + E2 DEV24

`dev.py` freezes 78 jobs: 72 paired per-edit training jobs (144 continuation
branches) and six real bank12/24 jobs. Seed 20260927 and orders 1–24 are fixed.
Direct M6/M7, CP M1/M1_STRUCT and Tucker M4/M4_STRUCT each share their respective
rebuilt W0. Three edit1 canaries precede the remaining queue; inherited CP-M0
teachers are imported only after complete parent binding/hash verification.
The same locked supervision, coordinate-specific optimizer, 80 updates, router,
precision and structural expansion checks apply. No paid support diagnostics
are added. The user authorized a cumulative Judge ceiling of 7660; 4008 items
are reserved at the unchanged 3652-attempt baseline (6 × [268+132+268]). Original
clocks, all prior attempts and permanent failed keys remain unchanged. This is
a launch, not a completed DEV24 result or an authorization for further fees.

## E3 REG24 SVD gauge

E1/E2 REG24 and DEV24 have completed (see versioned reports). `E3_PROTOCOL.json`
freezes 52 jobs and 96 continuation branches, pairing rebuilt CP/Tucker RAW
with function-preserving SVDGAUGE at rank4. Separate method and W0 namespaces
preserve prior artifacts. `e3.py` uses CPU float64 thin QR and a full 4x4 SVD,
without a dense residual matrix or rank truncation. Every edit records step0
FP32/FP16/native-generation parity. Two canaries close before full admission.
2880 attempts are conservatively reserved from the existing 7660 ceiling at
3779 used. E4 is not admitted. Judge uses gpt-6-astra/high via ChatGPT-authenticated
Codex CLI: these limits count experiment attempt items, not money or purchased
API credits. Historical references to paid ceilings do not imply API billing.

## E4 rank capacity

User authorized immediate admission of the complete E4 block, raising the internal
attempt ceiling to 9539 (1879 above 7660). Reserve 2880 items without resetting
history. `activate_e4.py` automatically waits for existing E3 ownership release;
no same-device overlap. Four distinct E4 arms share CP/TK W0_E4 within each pair,
retain original rank4 factors and expand with random nonzero A/zero B columns.
`e4.py` checks function preservation and nonzero new-direction B gradients.
Every R8 step0 has GPU parity before its 80 updates; two canaries gate 52 jobs.
This is admission, not completed E4 evidence or purchase of Codex credits.
