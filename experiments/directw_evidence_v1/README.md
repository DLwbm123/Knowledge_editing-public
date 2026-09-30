# DirectW-Evidence-v1 (stage A)

Start with the [external review packet](../../reports/directw_evidence_v1_stage_a_20260930/REVIEW_PACKET.md).
This package imports no native backbone loader or old CP/LoRA editor.

Reproduce the CPU tests with the existing Python environment containing PyTorch
2.6.0, NumPy and SciPy 1.16.0:

```sh
python - <<'PY'
from experiments.directw_evidence_v1.tests import main
raise SystemExit(main())
PY
```

The stdin entry keeps process argv neutral. No real model or paid service is
loaded. There are 20 requirement-group tests; their numerical operations are real
but their inputs and models are synthetic. `gate.cli` rejects native stages
before any loader. A later human-authorized native qualification may call the
generic editor on an ordinary clean model only after independently verified
phase approval and complete scientific/data/resource bindings.

Private run outputs go under this independent worktree's ignored `outputs/`.
Only code, draft configuration and deidentified aggregate reports are committed.
