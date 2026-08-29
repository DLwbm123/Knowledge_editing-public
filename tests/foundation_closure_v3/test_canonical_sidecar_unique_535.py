import json
import os
from collections import Counter
from pathlib import Path


ROOT = Path(os.environ["M3BENCH_ROOT"])
PATH = ROOT / "gpt56_sol_judge/foundation_closure_v3/judgments/foundation_canonical_gpt56sol_v3_535.jsonl"
EXCLUDED = "m3bench-v2/SLAKE/xmlab444/xmlab444_8"


def test_canonical_sidecar_unique_535():
    rows = [json.loads(line) for line in PATH.read_text(encoding="utf-8").splitlines() if line]
    assert len(rows) == len({row["record_id"] for row in rows}) == 535
    assert all(type(row["is_correct"]) is bool for row in rows)
    assert EXCLUDED not in {row["record_id"] for row in rows}
    assert Counter(row["dataset"] for row in rows) == {"SLAKE": 347, "VQA-RAD": 188}
    assert Counter(row["is_correct"] for row in rows) == {False: 484, True: 51}

