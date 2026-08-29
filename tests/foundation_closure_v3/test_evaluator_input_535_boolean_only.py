import json
import os
from collections import Counter
from pathlib import Path


ROOT = Path(os.environ["M3BENCH_ROOT"])
PATH = ROOT / "gpt56_sol_judge/foundation_closure_v3/sidecars/foundation_evaluator_predictions_535.jsonl"


def test_evaluator_input_535_boolean_only():
    rows = [json.loads(line) for line in PATH.read_text(encoding="utf-8").splitlines() if line]
    assert len(rows) == len({row["record_id"] for row in rows}) == 535
    assert all(type(row["is_correct"]) is bool for row in rows)
    assert Counter(row["dataset"] for row in rows) == {"SLAKE": 347, "VQA-RAD": 188}

