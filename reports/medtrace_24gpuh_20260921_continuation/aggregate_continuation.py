"""Recompute public coverage counts from PER_ARM_COVERAGE.json."""
import json
from pathlib import Path

def main():
    data = json.loads((Path(__file__).parent / "PER_ARM_COVERAGE.json").read_text())
    rows = list(data["arms"].values())
    outputs = sum(x["outputs"] for x in rows)
    scored = sum(x["exact_cache_scored"] for x in rows)
    missing = sum(x["missing_score"] for x in rows)
    assert outputs == scored + missing == 291
    print(json.dumps({"outputs": outputs, "exact_cache_scored": scored,
                      "missing_score": missing, "coverage": scored / outputs},
                     ensure_ascii=False, sort_keys=True))

if __name__ == "__main__":
    main()
