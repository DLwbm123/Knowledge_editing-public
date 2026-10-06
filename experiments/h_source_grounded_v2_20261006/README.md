# H Source-Grounded V2 — private execution, public implementation

This is an additive V2 workflow based on PR #28. It never overwrites V1, changes the 146-edit inventory, starts training, runs GPU judges, or changes TT/router/loss code. See `reports/h_source_grounded_v2_20261006/H_BUILD_V2_REPORT_ZH.md` for actual execution results and limitations.

All modules use the Python standard library. Run the local regression checks without network/model calls:

```sh
python3 -m unittest discover -s experiments/h_source_grounded_v2_20261006 -p 'test_*.py'
```

## Private inputs and order

Use a private trusted macOS host with the official locally authenticated Codex CLI and `/usr/bin/sandbox-exec`. Do not put login credentials in CI. The controller takes paths via environment variables; never include medical data in process arguments. These commands are documentation, not a public automation job.

1. `audit.py`: `V1_ROOT` (frozen delivery), **new** `V2_ROOT`, `PANEL_BINDINGS` (private verified image-label/caption mapping). Reads V1 only; writes revised candidates, rule-impact records and two status axes.
2. `prepare.py`: `V2_ROOT`. Locks 12 uncovered edits and reconstructs reusable image-bound author QA units before real-source calls.
3. `runner.py`: `FIXTURE_ROOT`, `CLI_PATH`. Fictional fixture must pass before sending source text. The macOS filesystem boundary denies controller logs, previous contexts and project/memory files; model tools are disabled. Unknown outbound permission blocks that source only.
4. `pilot.py`: `V2_ROOT`, `CLI_PATH`, `PROMPT_ROOT` (V1 `prompt_1.txt` through `prompt_3.txt`), `PILOT_ORIGIN=V1_CACHE_REUSE`. Each source gets a cached blind extraction; construction and review use distinct fresh sessions. Review receives only the original edit, candidate QA/citations and raw source. No generator interpretation/verdict is passed.
5. Targeted retrieval: obtain original JATS XML, license metadata and publisher figure files into private `new_primary`; visually verify caption/whole-figure binding. Save the locked `ACQUISITION_PLAN`. `acquire.py` **qualifies already acquired files**, not a general web crawler: `V2_ROOT`, `ACQUISITION_PLAN`. Requires original test-paper ID exclusions and all 594 future-native image hashes. Paper/case/image groups fix FIT roles before construction. Limits: 40 papers/60 units.
6. Run `pilot.py` with `PILOT_ORIGIN=NEW_PRIMARY`; keep the same receipt/cache directory, selection, call budget and relationship ledger.
7. `finalize.py`: `V2_ROOT`, `AUDIT_ROOT` (complete distinct audit output), `V1_ROOT`, **new** `DELIVERY_ROOT`, optional `REMOTE_DELIVERY_ROOT` for canonical private image paths. Retains raw outcomes; generator/reviewer disagreements become UNKNOWN. Reviewed grounded never becomes verified. Existing EVAL stays frozen.

Invoke controller modules through stdin (e.g. `python3 - < module.py`) on this project so visible controller arguments remain neutral. Required environment paths are private. Run `ps` to inspect actual controller/CLI arguments; the runner records the CLI command and process line. Do not recursively invoke this runner inside a model session.

## Private and public boundaries

Private: original questions/answers, medical evidence, image/case/paper identifiers, hashes, source roles, exclusion lists, complete per-edit tables, model prompts/raw outputs/receipts, acquisition plans and all failures. Public: implementation/tests, V1 prompt templates already in this branch, generic schema, anonymous count tables and Chinese report. Public summaries expose inventory positions only, without patient/source identifiers or source quotes. No private medical fixture is bundled.

This execution used `gpt-6.1-sol` in both model roles: `SAME_MODEL_SEPARATE_CONTEXT`. CLI token usage is available; API invoices and immutable backend model snapshots are not. Public licensed text permission does not establish clinical correctness, patient independence, or any account retention guarantee. No zero-retention claim is made.
