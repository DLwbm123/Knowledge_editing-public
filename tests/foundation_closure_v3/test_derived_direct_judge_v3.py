from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPT = Path(__file__).parents[2] / "scripts" / "foundation_derived_direct_judge_v3.py"
SPEC = importlib.util.spec_from_file_location("foundation_derived_direct_judge_v3", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def test_prepare_and_finalize_with_exact_cache_and_direct_verdict(tmp_path: Path) -> None:
    stage = tmp_path / "gpt56_sol_judge" / "foundation_closure_v3"
    artifacts = stage / "task_sets" / "local_535" / "artifacts"
    canonical = []
    for index in range(535):
        canonical.append({
            "record_id": f"record-{index}",
            "question": "cached question" if index == 0 else f"question {index}",
            "gold_answer": "cached gold" if index == 0 else f"gold {index}",
            "model_answer_raw": "cached answer" if index == 0 else f"answer {index}",
            "is_correct": index == 0,
            "judge_input_sha256": f"source-hash-{index}",
            "judge_reason": "canonical reason",
            "judge_confidence": "high",
        })
    manifest = [
        {"derived_probe_id": "probe-cache", "valid": True},
        {"derived_probe_id": "probe-direct", "valid": True},
    ]
    raw = [
        {
            "derived_probe_id": "probe-cache", "status": "success",
            "question": "cached question", "gold_answer": "cached gold",
            "model_answer_raw": "cached answer",
        },
        {
            "derived_probe_id": "probe-direct", "status": "success",
            "question": "new question", "gold_answer": "new gold",
            "model_answer_raw": "new answer",
        },
    ]
    write_jsonl(stage / "judgments" / "foundation_canonical_gpt56sol_v3_535.jsonl", canonical)
    write_jsonl(artifacts / "derived_probe_manifest.jsonl", manifest)
    write_jsonl(artifacts / "derived_probe_raw_predictions.jsonl", raw)

    MODULE.prepare(stage)
    packet = MODULE.read_jsonl(stage / "packets" / "derived_judge_blind_packet.jsonl")
    assert len(packet) == 1
    assert set(packet[0]) == {"blind_id", "question", "gold_answer", "model_answer"}
    report = json.loads((stage / "reports" / "DERIVED_JUDGE_PACKET_REPORT.json").read_text())
    assert report["exact_triple_cache_reuse_count"] == 1
    assert report["direct_judge_count"] == 1

    write_jsonl(
        stage / "judgments" / "derived_direct_decisions" / "part_001.jsonl",
        [{
            "blind_id": packet[0]["blind_id"], "is_correct": False,
            "confidence": "high", "reason": "direct reason", "probe_issue": False,
        }],
    )
    MODULE.finalize(stage)
    judged = {row["derived_probe_id"]: row for row in MODULE.read_jsonl(artifacts / "derived_probe_judged_predictions.jsonl")}
    assert judged["probe-cache"]["is_correct"] is True
    assert judged["probe-cache"]["judge_route"] == "gpt56_sol_exact_canonical_triple_reuse"
    assert judged["probe-direct"]["is_correct"] is False
    assert judged["probe-direct"]["judge_route"] == "gpt56_sol_direct_derived_v3"
