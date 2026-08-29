import json
import os
from pathlib import Path


ROOT = Path(os.environ["M3BENCH_ROOT"])
STAGE = ROOT / "gpt56_sol_judge/foundation_closure_v3"


def test_review_v3_complete_coverage():
    selection = [json.loads(line) for line in (STAGE / "manifests/review_v3_selection_535.jsonl").read_text(encoding="utf-8").splitlines() if line]
    reviewed = [json.loads(line) for line in (STAGE / "judgments/foundation_review_v3.jsonl").read_text(encoding="utf-8").splitlines() if line]
    report = json.loads((STAGE / "reports/FOUNDATION_REVIEW_V3_REPORT.json").read_text(encoding="utf-8"))
    assert {row["record_id"] for row in selection} == {row["record_id"] for row in reviewed}
    assert len(reviewed) == len({row["record_id"] for row in reviewed}) == 203
    assert all(type(row["reviewed_verdict"]) is bool for row in reviewed)
    assert not any(row["dataset_issue"] for row in reviewed)
    assert report["coverage"] == 1.0
    assert report["missing"] == report["duplicate"] == report["unresolved"] == report["new_dataset_issue"] == 0

