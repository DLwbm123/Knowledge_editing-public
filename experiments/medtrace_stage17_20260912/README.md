# Stage17: preparation and first formal single-edit dispatch

This is a source overlay, not a completed Stage17 campaign. Read the [current launch report](../../reports/medtrace_stage17_20260912/formal/GPUHOME_LAUNCH.md): Astra Base and cohort/support preparation are complete; the first BalancEdit single-T0-146 dispatch runs on the new RTX 4090 GPU0. The true sequential driver and other method dispatches remain pending.

New logical commands, to be passed through the existing neutral launcher with private paths in its environment:

```sh
python -m scripts.medtrace.stage17_freeze --project "$PRIVATE_PROJECT" --run "$PRIVATE_RUN"
python scripts/medtrace/stage17_single.py smoke "$PRIVATE_DISPATCH"
python scripts/medtrace/stage17_single.py worker "$PRIVATE_DISPATCH"
```

Do not re-run completed preparation or the accepted Judge. The freeze command consumes the copied role join, source overlay, complete accepted Astra bindings and verdicts. The dispatch config binds project/run/cpu_gate, current GPU/UUID, clean source commit, frozen queue ID/N, method recipe and runtime lock. The first worker supports only `method=balancedit`, `mode=single_main_T0`; unsupported branches are not auto-launched. The smoke config is derived from that dispatch and requires the first already-exposed DEV record plus its earlier mechanical output. See the script for the exact private schema.

On the migrated server, missing dependencies were installed into a separate data-disk environment; existing environments and model snapshots were preserved. CPU checks for the current overlay: `PYTHONPATH=. python -m unittest discover -s tests -p 'test_stage17*.py'` (11 passed). The sections below describe earlier preparation milestones and are not current launch instructions.

The authorized continuation adds `stage17_prepare.py` (candidate Base packet assembly and cache binding), `stage17_judge.py` (one-pass isolated Astra execution), and `stage17_roles.py` (score-independent prior-evaluation source isolation). Generic schema/validator and task-specific source helpers are reused. Preparation and the role join are complete; the Judge stopped after 40 accepted batches when batch 41 failed at the transport layer. These are not instructions to duplicate accepted work or bypass its no-retry rule. All three logical commands take one private JSON config path through the existing neutral entrypoint. Run `python -m unittest tests.test_stage17_prepare tests.test_stage17_contract tests.test_stage17_roles` for the eight relevant checks; macOS is required for the real OS-isolation check. The remaining sections describe the original Stage17-A milestone.

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

The original Judge prompt/schema/validation preparation is [documented separately](../../reports/medtrace_stage17_20260912/JUDGE_PREPARATION.md). Generic schema/validation functions are reused, but the old Stage16 prepare/merge protocol must not be mislabeled Stage17. The later authorized runner and packet are described in the continuation linked above.

No modifications to old experiment history, sealed inputs, evaluation-to-training roles, clinical signoffs, or ranks/losses/threshold candidates are authorized by these reproduction commands. The approved Stage17-only performance-gate waiver does not modify any old experiment lock.
