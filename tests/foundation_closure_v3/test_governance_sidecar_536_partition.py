import json
import os
from collections import Counter
from pathlib import Path


ROOT = Path(os.environ["M3BENCH_ROOT"])
PATH = ROOT / "gpt56_sol_judge/foundation_closure_v3/manifests/foundation_governance_sidecar_536.jsonl"


def _rows():
    return [json.loads(line) for line in PATH.read_text(encoding="utf-8").splitlines() if line]


def test_governance_sidecar_536_partition():
    rows = _rows()
    assert len(rows) == len({row["record_id"] for row in rows}) == 536
    assert Counter(row["governance_action"] for row in rows) == {"keep": 535, "exclude": 1}
    excluded = [row for row in rows if row["governance_action"] == "exclude"]
    assert excluded[0]["record_id"] == "m3bench-v2/SLAKE/xmlab444/xmlab444_8"
    assert excluded[0]["score_status"] == "NA/excluded"
    assert excluded[0]["is_correct"] is None

