# R first-source solver preparation

`core.py` provides a `Memory` subclass with the same `solve_current()` and restored-state interface used by the immutable J worker. It imports the frozen J core as `legacy_core`; it does not patch the historical J directory. The preparation checker stages that mapping in a disposable CPU-only directory and tests the actual `worker.attempt()` caller. No GPU executor, deployment or scoring launcher is provided here.

Run CPU preparation through a neutral stdin entry from the repository root:

```sh
python - <<'PY'
import runpy
runpy.run_path('experiments/purew_ecw_r1_v1/check.py', run_name='__main__')
PY
```

The remote interpreter has a neutral `-c` entry and receives sources through stdin. Failed checks and wall time remain in `.local_run/ecw_r1_preparation_20261015`; success writes the anonymous CPU report. This script creates no experiment RUN, installation, model/GPU allocation, Judge request or recurring monitor. A timeout is retained as uncertain CPU state, not automatically retried.

Frozen scope and uncompleted launch gates are in `reports/purew_ecw_r1_v1_20261015/PROTOCOL.json` and `THEORY_AND_SCOPE_ZH.md`. Do not deploy this adapter into the old J RUN, or substitute it into a live process.
