import json
import os
from pathlib import Path


ROOT = Path(os.environ["M3BENCH_ROOT"])
SIDECARS = (
    "foundation_evaluator_predictions_535.jsonl",
    "vqarad_evaluator_predictions_188.jsonl",
    "slake_evaluator_predictions_347.jsonl",
)
EXCLUDED = "m3bench-v2/SLAKE/xmlab444/xmlab444_8"


def test_excluded_record_absent_from_evaluator():
    for name in SIDECARS:
        path = ROOT / "gpt56_sol_judge/foundation_closure_v3/sidecars" / name
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
        assert EXCLUDED not in {row["record_id"] for row in rows}

