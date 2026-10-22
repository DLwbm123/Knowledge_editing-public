# V51: canonical-coordinate diagnosis, paired FFN placement, boundary admission

The V50 terminal result stays unchanged. This finite study uses a separately authorized clock and storage ledger. It does not assume that a router must improve before testing layer placement.

CPU diagnostics (from the checkout root, inherited private runtime staging must be present):

```sh
python experiments/mechanism_v51/gauge.py
python experiments/mechanism_v51/test_components.py
```

`gauge.py` imports the existing `module_math.canonicalize`, uses full-size TT cores and small input probes, and never materializes a dense update matrix. `diagnose(state, module_math)` can also inspect a real trained raw proposal without model calls. The transported 4x4 matrix is a diagnostic identity, not an implemented/trained remedy.

Real execution requires the existing private runtime and an admitted root containing DATA_ADMISSION, CONTROL_OUTPUTS, PRIOR_LABELS, LOCALIZER_LOCKS, PROTOCOL_LOCK, CPU_ADMISSION, RESOURCE_LIMITS and live RESOURCE_GUARD_HEALTH. Use the established neutral launcher, which sets `V51_ROOT`, `PARTITION`, `PARENT_RUN` and GPU identity in the environment. The neutral entry executes:

```python
import os, runpy
runpy.run_path(os.environ['V51_ROOT']+'/tools/run.py', run_name='__main__')
```

It is not safe to run this entry without its registered guard. The source-only first-case smoke on both shards precedes STAGE1_GO. Fixed21 must reproduce the prior W0 outputs exactly. The selected-layer arm changes only the down-projection location, uses the already frozen four source layer choices, and remains forced on. Each arm has a real uncached/cached loss and gradient parity check. The cache prefix ends strictly before its edited layer.

After both workers complete, use the completed private root (not a public dataset):

```sh
python experiments/mechanism_v51/evaluate.py validate "$RUN"
python experiments/mechanism_v51/evaluate.py blind "$RUN"
# Read only BLIND_PACKET; save complete C/I/U/A judgments and freeze them.
python experiments/mechanism_v51/evaluate.py summarize "$RUN"
```

`blind` is write-once. Old labels are reused only for exact context/query/text/token matches. Source identities, images, tokens and individual medical answers remain private. Completed weights are retained until generation, reload checks, raw-step diagnosis and the registered successor decision are resolved.

Boundary inventory is metadata-only and does not admit a medical negative merely from an answer difference:

```sh
python experiments/mechanism_v51/admit_boundary.py "$REPO" "$EXISTING_MEDMKEB_ROOT" "$PRIVATE_OUTPUT"
```

The subsequent finite source/figure review has a private locked candidate list and explicit reasons. No source fact alone is treated as a complete boundary task pack or as patient-independent evidence.
