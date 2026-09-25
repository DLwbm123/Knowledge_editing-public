"""Keep only final-prefix aggregate metrics; never export per-edit changes."""
import json
import os
from pathlib import Path


source = json.loads(Path(os.environ["PRIVATE_RESULTS_JSON"]).read_text())
public = {"scope": "aggregate_terminal_prefix_only", "independent_confirmation": False, "jobs": {}}
for name, job in source.items():
    summary = job["summary"]
    terminal = max((row["prefix"] for row in summary), default=None)
    rows = [row for row in summary if row["prefix"] == terminal]
    comparisons = job["comparisons"]
    assert all("changes" not in row for row in comparisons)
    public["jobs"][name] = {
        "status": job["status"],
        "job": job["job"],
        "terminal_prefix": terminal,
        "summary": rows,
        "comparisons": comparisons,
    }

forbidden = {"edit", "query_id", "question", "raw_answer", "image_path", "source_group", "changes"}
def check(value):
    if isinstance(value, dict):
        assert not forbidden.intersection(value)
        for child in value.values():
            check(child)
    elif isinstance(value, list):
        for child in value:
            check(child)
    elif isinstance(value, str):
        assert "/" not in value and "@" not in value


check(public)
Path(os.environ["PUBLIC_RESULTS_JSON"]).write_text(json.dumps(public, ensure_ascii=False, indent=2) + "\n")
