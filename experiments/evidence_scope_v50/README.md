# Finite scope/evidence TT adapters

This package extends the existing private real-runtime framework; pretrained model/data are not bundled. All path arguments are environment variables so a neutral process entry can run without exposing project names in argv.

Local CPU checks:

```sh
python - <<'PY'
import runpy,sys
sys.path.insert(0,'experiments/evidence_scope_v50')
runpy.run_path('experiments/evidence_scope_v50/test_components.py',run_name='__main__')
PY
```

Real runner requires an admitted `V50_ROOT` containing DATA_ADMISSION, PROTOCOL_LOCK, RESOURCE_LIMITS, PARENT_ENV, CONTROL_OUTPUTS, MANIFEST, CPU_ADMISSION, black.png, tools and per-shard GO receipts; `PARENT_RUN` points to the unchanged inherited runtime. Launch through the existing resource guard and neutral entry. The runner refuses an expired deadline or missing fresh guard. `STAGE1_GO.json` is written only after source-only real smoke passes. No standalone command bypasses admission.

```python
# Contents of a neutral q.py entry, with environment already provisioned:
import os,runpy
runpy.run_path(os.environ['V50_ROOT']+'/tools/run.py',run_name='__main__')
```

Post-run (set RUN to completed private root in Python):

```python
import sys
from pathlib import Path
sys.path.insert(0,'experiments/evidence_scope_v50')
from evaluate import validate,blind,summarize
validate(Path(RUN)); blind(Path(RUN))
# Review only BLIND_PACKET; record C/I/U/A with reasons, then freeze verdicts.
# Do not inspect mapping to choose labels. No new generation or label retries.
summarize(Path(RUN))
```

A1 and A2 share the same W0 trained delta. Deterministic input-only gates select already generated Base/W0 full free outputs; this reuses prediction results, never truth labels. Forced/off diagnostic consumers select exact W0/Base outputs. B writers are independently trained from paired initialization; B_GATE and B_COND use the same frozen evidence gate. All new generated answers require full scoring; exact old tokens/text permit old verdict reuse.

Medical negatives, verified ROI, independent cases and larger source pools are not fabricated. See protocol/manifest admission and BLOCKED list in the report. Small deltas remain private until registered consumers are finished.
