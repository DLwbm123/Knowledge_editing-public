# Stage16 source delta and Astra coverage report

Status: A complete; B generated and Astra-scored on 133 restored official locality probes; C/D unsupported. Read the [current review report](../../reports/medtrace_stage16_20260912/GPT_PRO_REVIEW.md). No historical training, generation or Qwen scoring was repeated. This is coverage completion on the existing cohort, not new independent confirmation.

## Assemble the source

Copy the published [Stage15 source](../medtrace_stage15_20260911) into a separate working directory, then overlay this directory's `scripts/` and `tests/`. Use the existing compatible environments; no new dependency, model, optimizer, rank, loss or threshold is introduced. Configure approved private runtime paths locally. Do not rerun the completed campaign.

The original `stage16.py` performs offline A analysis, `stage16_sources.py` handles source recovery, and `stage16_coverage.py` preserves the checkpoint-only worker, neutral detached launch and explicitly authorized failed-worker/GPU continuation. These are provenance/reproduction code, not instructions to restart this completed run.

**Do not feed Astra verdicts to the old `stage16_coverage.report()` or overwrite the original Qwen `OUTPUT.jsonl`.** That entry point is the historical Qwen-only reporting path. Use the isolated report entry below for the actual amended protocol.

## Reproduce the aggregate report without GPU or Judge calls

The standard-library `stage16_astra_report.py` reads already completed private inputs. Supply an existing authorized bundle with:

```text
bundle/
  judge_only/batch_001.input.json ... batch_007.input.json
  operator/
    PACKET.jsonl
    SIDECAR.json
    JUDGE_LOCK.json
    MANIFEST.json
    EXECUTION_RECORD.json
    VERDICTS_ASTRA.jsonl
    responses/batch_001.json ... batch_007.json
    stage16_evidence/
      QUEUE.json
      A_BOUND_ROWS.json
      edits/eNNN/C_NO_H/RESULT.json
      edits/eNNN/BE/RESULT.json
```

From the assembled source directory:

```sh
python -m unittest discover -s tests -p 'test_astra_judge_bundle.py'
python tests/test_stage16_contract.py
python scripts/medtrace/stage16_astra_report.py --bundle "$JUDGE_BUNDLE" --output "$NEW_AGGREGATE_JSON"
```

`NEW_AGGREGATE_JSON` must not exist; the reporter refuses overwrite. It validates full structured input/output bindings against the original opaque IDs, source content, method pairing, fixed routing and accepted batch responses. It reuses the existing normalization, 2,000-replicate bootstrap and exact paired tests. It never invokes a model, trains a writer, modifies the private run, or creates mixed-Judge semantic aggregates. Python assertions must remain enabled; do not run with `-O`.

The published [aggregate JSON](../../reports/medtrace_stage16_20260912/STAGE16_REVIEW_AGGREGATES.json) contains 40 method/mode/score rows, 20 paired comparisons and 20 source-component sensitivity rows. `score=exact` with `score_definition=OUTPUT_PRESERVATION` means normalized pre/post output agreement, not matching a source reference; use `on_output_damaged` for output-change counts. `base_correct_damage` uses the corresponding reference-correctness score. Raw private inputs are excluded, so the public release permits code/statistical inspection but not independent raw-data replay without separately authorized inputs.

## Astra materials and acceptance

`astra_judge_bundle.py`, `astra_judge_prompt.md` and `astra_judge_operator.md` preserve the preparation and strict acceptance workflow. They do not launch Astra or upload data. A real isolated authorized model session is required for actual scoring. Existing batches are already complete and must not be scored again merely to reproduce this report.

Historical Qwen semantic results remain separate from new `gpt-6-astra / high` results. The original first Astra batch was excluded for unverified isolation and replaced by exactly one explicitly authorized rejudge; only the replacement is accepted. Model snapshot is unknown, not invented. Format validation is not clinical adjudication.

No source images, QA, answers, tokens, private execution IDs, per-item Judge mappings or checkpoints are redistributed. Sources/licenses remain those of [MedMKEB](https://github.com/pkusixspace/MedMKEB/tree/d9f38639ec2285a0e9f541e22156ec14f87271d8) and the [pinned PMC-VQA author card](https://huggingface.co/datasets/RadGenome/PMC-VQA/blob/b56ae594f794867893143b337b4118a835794647/README.md). Gated sources were not bypassed, and evaluation images were not repurposed for H training.
