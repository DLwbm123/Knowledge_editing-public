# Stage14: fixed supervision-matched closeout

[Results and limitations](../../reports/medtrace_stage14_20260911/GPT_PRO_REVIEW.md), [frozen method and claims](../../reports/medtrace_stage14_20260911/METHOD_V1_AND_CLAIMS.md).

Seven C_EXTRA_QA trajectories completed on the previously published new7 cohort; historical FACT/NO_H/BE outputs were reused. Judge1,416/1,416, including1,399 reused and17 new full context-bound tuples. FACT beats EXTRA on forced H-evaluation PairCorrect edit-macro by85.71pp, while generic extra QA improves U correctness and FACT still loses three same-answer challenge successes. Fixed old RC makes all four deployed systems tie. The extra-control comparison is a post-publication supplement, not unseen prospective or patient-independent confirmation.

Use `scripts/medtrace/stage14_sources.py` for the one-shot support overlay and `stage14.py` for explicit train/generate/prepare-judge/report actions. The shared `coordinate_stage13r.py` recognizes MEDTRACE_STAGE14 and runs its bounded GPU1-only background chain. The shared `stage11_worker.py` maps C_EXTRA_QA explicitly to rank4 and rejects unknown writer names; historical condition branches remain unchanged. Other dependencies are reused from the published Stage13R snapshot, not a fresh framework.

## Provenance and checks

- Public baseline:f22b56d6b9117d06784c4f355e30278c1397ba2e.
- Source preparation:1e56d476cdd03b5876fc47ff36d7288f4743939e.
- Training/generation execution:a3b85bc89406cc2c7ce2af0ed1e851244e4d945b.
- CPU-only report recovery:fb94ccd190ebc46bce351032b0dc59b53dae9be5.
- Research result:6456454d3feffa34c5a2efeb1b7aa3b37803608d.
- Two focused rank/source/paired-denominator checks passed: `PYTHONPATH=. python -m pytest -q tests/test_stage14_matched_support.py`. No data download or GPU experiment is performed by these checks. Earlier relevant Stage12/13R checks also passed before launch.

The curator is restricted to existing legal episode training images, not new-native discovery. All7 controls have SAME_H_IMAGE support (4 unique G QA/2 images). Old15 have no legal control under the combined evaluation-image exclusion; their original complete panels remain available. The different-image support subset is empty. Correct source answers, number of auxiliary CE calls and answer type are matched; information content, token count and FLOPs are not claimed equal. The historical Stage13R source and evaluation-import correction remain unchanged.

## Reproducibility boundary

Source code and aggregate tables are supplied, not a standalone dataset distribution. Authorized private historical exposure ledgers, full raw training sources, frozen episode manifests, runtime/model/tokenizer locks and Judge ancestry are necessary to reproduce the exact cohort. Inherited path defaults are redacted to `/path/to/storage` and `/path/to/local`; configure actual authorized storage and GPU identity. Do not use this snapshot to reclassify sealed/evaluation-only material or overwrite completed runs.

The training process consumes only native/fit answer files and the G overlay. Evaluation starts in a separate process after training completes. This is code-path separation, not an OS-level sandbox. Each control starts from its own original CP-W0 and inherits native/fit/U schedules; no native CP/A2/W0 or old writer is retrained. BalancEdit has its original different capacity and supervision budget, not a paper-exact or matched-H claim.

The original GPU phases completed in22.12 minutes. A final NFS xattr-copy failure was repaired with content-only copyfile; CPU reporting then completed without repeating training, generation or judging. Original failure logs and recovered status are preserved. This snapshot requires the existing compatible environment and private locks/lineage described above; it is not a self-contained public dataset release.

Private QA, images, outputs/tokens, activations, checkpoints, weights, Judge mappings, credentials and private Git history are excluded. No automatic next stage is authorized.
