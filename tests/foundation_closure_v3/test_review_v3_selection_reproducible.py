import hashlib
import json
import os
from pathlib import Path


ROOT = Path(os.environ["M3BENCH_ROOT"])
STAGE = ROOT / "gpt56_sol_judge/foundation_closure_v3"


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_review_v3_selection_reproducible():
    path = STAGE / "manifests/review_v3_selection_535.jsonl"
    report = json.loads((STAGE / "reports/REVIEW_V3_SELECTION_REPORT.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    assert report["seed"] == 2026081906
    assert report["high_confidence_sample_size"] == 54
    assert len(rows) == len({row["record_id"] for row in rows}) == 203
    assert _sha256(path) == report["selection_sha256"] == "5ad548390d0139d73cacf7638d5b75ef161954bea81af189f4e36ae48f316a0c"

