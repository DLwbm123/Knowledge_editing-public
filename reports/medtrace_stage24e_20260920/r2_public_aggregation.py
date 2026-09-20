"""Public, source-only R2 result aggregation; raw outputs remain private."""
import json
from pathlib import Path

def summarize(path: str) -> dict:
    rows = json.loads(Path(path).read_text())
    return {arm: {"items": v["items"], "correct": v["correct"], "accuracy": v["accuracy"]}
            for arm, v in rows["summary"].items()}

if __name__ == "__main__":
    print(json.dumps(summarize("R2_RESULTS.json"), ensure_ascii=False, indent=2))
