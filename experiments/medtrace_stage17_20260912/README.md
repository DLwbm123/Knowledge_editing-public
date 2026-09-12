# Stage17-A: metadata and bounded mechanical handoff

This is a small source overlay, not a full formal Stage17 campaign. Read the [review](../../reports/medtrace_stage17_20260912/GPT_PRO_REVIEW.md), especially the remaining authorization, source-role and adapter work. No formal single/sequential CLI is claimed.

## Assemble

In a new working directory, copy the existing [Stage15 source](../medtrace_stage15_20260911), overlay the [Stage16 scripts/tests](../medtrace_stage16_20260912), then overlay this directory's scripts/tests. Reuse the compatible existing environment. No new dependency or model download is needed. The two editor contract tests included here fill test files absent from the Stage15 snapshot.

## CPU checks

From the assembled directory:

```sh
PYTHONPATH=. python -m pytest -q tests/test_stage17_contract.py tests/medtrace/test_core.py tests/medtrace/test_selective_write.py tests/medtrace/test_stage3_bank.py tests/medtrace/test_stage5_bank.py tests/editor_paperspec/test_mechanics.py tests/editor_paperspec/test_formal_state_contract.py
```

The live implementation passed 53 checks. These test routing/zero effects/masks/state lifecycles/support and denominator contracts, not benchmark accuracy. `stage17_contract.py` reuses the existing nearest-entry router, strips labels, limits prefix visibility, clears MedTRACE hooks for Base features, and rejects incomplete/mismatched cache bindings. Callers remain responsible for including **all** active hooks/wrappers in Base-only contexts; this is not a complete formal bank driver.

## Reproduce the metadata summary

```sh
python scripts/medtrace/stage17_inventory.py --snapshot "$PRIVATE_METADATA_LEDGER" --out "$NEW_COHORT_SUMMARY"
```

The private ledger is a JSON mapping from `cohort/runtime/methods/generation/dev/qual8/qual16/seq16/lora_profile/lora_qual` to `{path, content}` for already permitted metadata files. The collector does not open paths in that ledger. It never reads formal QA/images/answers or invokes a model. The output must not exist. It deliberately keeps new N/support memberships unknown until authorized Base judging and whole-queue role/exposure joins are completed. The published summary is not a frozen per-edit support ledger.

## GPU mechanical check (already completed; do not automatically rerun)

The existing neutral launcher passes this private logical argv through `JOB_ARGV`:

```sh
python scripts/medtrace/stage17_smoke.py --config "$PRIVATE_SMOKE_CONFIG" --out "$NEW_SMOKE_RESULT"
```

The visible detached command must use a neutral `main.py job` entrypoint, with private paths only in its environment, not process arguments. The private config contains `scope=DEV_NATIVE_ONLY_MECHANICAL`, `gpu="3"`, paths `cpu_gate/dev_manifest/dev_inputs`, and the full canonical `runtime_lock`. Set existing M3BENCH model/vision/official-source variables, authorized physical GPU3/UUID, and shared-storage temporary paths. Check the actual mount, small write/read probe and sufficient VRAM once before launching. The GPU result was produced from private source commit `31eec55224370f1ea7a27e0f65b48c0738704ea4`; subsequent additions are CPU inventory/reporting only.

This runner checks exactly the first already-exposed DEV16 native, never a formal probe or old Base answer file. It performs zero training/Judge calls, four canonical generations plus one one-token lifecycle probe, and records failures rather than selecting another sample. Outputs include private inputs/tokens and must not be published; only `GPU_SMOKE.json` is public-safe. One-sample mechanical memory is not a future training/bank estimate.

Judge prompt/schema/validation preparation is [documented separately](../../reports/medtrace_stage17_20260912/JUDGE_PREPARATION.md). Generic schema/validation functions are reused, but the old Stage16 prepare/merge protocol must not be mislabeled Stage17. No cloud Judge runner or formal packet has been launched.

No modifications to old experiment history, sealed inputs, evaluation-to-training roles, clinical signoffs, ranks/losses/threshold candidates, or legacy performance gates are authorized by these reproduction commands.
