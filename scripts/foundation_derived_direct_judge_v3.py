#!/usr/bin/env python3
"""Prepare and finalize blind GPT-5.6 Sol judging for derived T1G/T2G probes.

This utility never produces semantic verdicts.  It may reuse a frozen verdict
only when question, gold answer, and raw model answer are byte-for-byte equal
to a canonical Foundation record.  Every other verdict must be supplied by the
current agent through blind decision shards.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


JUDGE_MODEL = "gpt-5.6-sol"
JUDGE_VERSION = "m3bench-gpt56sol-v1"
RUBRIC_SHA256 = "e6e58d7f6307740bf6f341066c0a74fe65683df0e6d4379a991ae82d0ea151a6"
SHARD_SIZE = 50


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def canonical_payload(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def encoded_jsonl(rows: Iterable[dict[str, Any]]) -> str:
    return "".join(canonical_payload(row) + "\n" for row in rows)


def write_frozen(path: Path, payload: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text(encoding="utf-8") != payload:
        raise RuntimeError(f"refusing to replace frozen artifact: {path}")
    path.write_text(payload, encoding="utf-8")
    return sha256(path)


def write_json(path: Path, value: Any) -> str:
    return write_frozen(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> str:
    return write_frozen(path, encoded_jsonl(rows))


def triple(row: dict[str, Any]) -> tuple[str, str, str]:
    return row["question"], row["gold_answer"], row["model_answer_raw"]


def input_hash(question: str, gold_answer: str, model_answer: str) -> str:
    return sha256_text(canonical_payload({
        "gold_answer": gold_answer,
        "model_answer": model_answer,
        "question": question,
        "rubric_sha256": RUBRIC_SHA256,
        "rubric_version": JUDGE_VERSION,
    }))


def stage_paths(stage: Path) -> dict[str, Path]:
    local = stage / "task_sets" / "local_535" / "artifacts"
    return {
        "canonical": stage / "judgments" / "foundation_canonical_gpt56sol_v3_535.jsonl",
        "manifest": local / "derived_probe_manifest.jsonl",
        "raw": local / "derived_probe_raw_predictions.jsonl",
        "output": local / "derived_probe_judged_predictions.jsonl",
        "hashes": local / "derived_probe_hashes.json",
        "packet": stage / "packets" / "derived_judge_blind_packet.jsonl",
        "mapping": stage / "packets" / "derived_judge_mapping.jsonl",
        "cache": stage / "judgments" / "derived_exact_triple_cache_reuse.jsonl",
        "shards": stage / "packets" / "derived_judge_shards",
        "decisions": stage / "judgments" / "derived_direct_decisions",
        "prepare_report": stage / "reports" / "DERIVED_JUDGE_PACKET_REPORT.json",
        "final_report": stage / "reports" / "DERIVED_JUDGE_REPORT.json",
    }


def validate_source(paths: dict[str, Path]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    canonical = read_jsonl(paths["canonical"])
    manifest = read_jsonl(paths["manifest"])
    raw = read_jsonl(paths["raw"])
    valid_ids = [row["derived_probe_id"] for row in manifest if row["valid"]]
    raw_ids = [row["derived_probe_id"] for row in raw]
    if len(canonical) != 535 or len({row["record_id"] for row in canonical}) != 535:
        raise RuntimeError("canonical Foundation census is not 535 unique records")
    if len(valid_ids) != len(set(valid_ids)) or len(raw_ids) != len(set(raw_ids)):
        raise RuntimeError("duplicate derived probe IDs")
    if set(valid_ids) != set(raw_ids) or any(row.get("status") != "success" for row in raw):
        raise RuntimeError("M3BENCH_V3_STOP__DERIVED_PROBE_BASELINE_INCOMPLETE")
    return canonical, manifest, raw


def prepare(stage: Path) -> None:
    paths = stage_paths(stage)
    canonical, manifest, raw = validate_source(paths)
    canonical_by_triple: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in canonical:
        canonical_by_triple[triple(row)].append(row)

    cached: list[dict[str, Any]] = []
    direct_raw: list[dict[str, Any]] = []
    for row in sorted(raw, key=lambda item: item["derived_probe_id"]):
        matches = canonical_by_triple.get(triple(row), [])
        if matches:
            verdicts = {match["is_correct"] for match in matches}
            if len(verdicts) != 1:
                raise RuntimeError("conflicting canonical verdicts for an exact input triple")
            source = sorted(matches, key=lambda item: item["record_id"])[0]
            cached.append({
                "derived_probe_id": row["derived_probe_id"],
                "exact_input_sha256": input_hash(*triple(row)),
                "is_correct": source["is_correct"],
                "source_record_ids": sorted(match["record_id"] for match in matches),
                "source_judge_input_sha256": source["judge_input_sha256"],
                "source_judge_reason": source["judge_reason"],
                "source_judge_confidence": source["judge_confidence"],
                "reuse_policy": "byte_exact_question_gold_model_answer_only",
            })
        else:
            direct_raw.append(row)

    packet: list[dict[str, Any]] = []
    mapping: list[dict[str, Any]] = []
    for ordinal, row in enumerate(direct_raw, start=1):
        blind_id = f"derived-v3-{ordinal:05d}"
        item = {
            "blind_id": blind_id,
            "question": row["question"],
            "gold_answer": row["gold_answer"],
            "model_answer": row["model_answer_raw"],
        }
        packet.append(item)
        mapping.append({
            "blind_id": blind_id,
            "derived_probe_id": row["derived_probe_id"],
            "judge_input_sha256": input_hash(row["question"], row["gold_answer"], row["model_answer_raw"]),
        })

    packet_sha = write_jsonl(paths["packet"], packet)
    mapping_sha = write_jsonl(paths["mapping"], mapping)
    cache_sha = write_jsonl(paths["cache"], cached)
    paths["shards"].mkdir(parents=True, exist_ok=True)
    expected_shards = set()
    shard_hashes: dict[str, str] = {}
    for start in range(0, len(packet), SHARD_SIZE):
        name = f"shard_{start // SHARD_SIZE + 1:03d}.jsonl"
        expected_shards.add(name)
        shard_hashes[name] = write_jsonl(paths["shards"] / name, packet[start:start + SHARD_SIZE])
    extras = {path.name for path in paths["shards"].glob("shard_*.jsonl")} - expected_shards
    if extras:
        raise RuntimeError(f"stale derived judge shards present: {sorted(extras)}")

    report = {
        "status": "READY__DERIVED_GPT56SOL_DIRECT_JUDGE",
        "canonical_count": len(canonical),
        "manifest_total": len(manifest),
        "manifest_valid": sum(row["valid"] for row in manifest),
        "raw_success_count": len(raw),
        "exact_triple_cache_reuse_count": len(cached),
        "direct_judge_count": len(packet),
        "shard_size_max": SHARD_SIZE,
        "shard_count": len(shard_hashes),
        "raw_sha256": sha256(paths["raw"]),
        "manifest_sha256": sha256(paths["manifest"]),
        "canonical_sha256": sha256(paths["canonical"]),
        "rubric_sha256": RUBRIC_SHA256,
        "packet_sha256": packet_sha,
        "mapping_sha256": mapping_sha,
        "cache_reuse_sha256": cache_sha,
        "shard_sha256": shard_hashes,
        "blind_packet_fields": ["blind_id", "question", "gold_answer", "model_answer"],
        "forbidden_blind_fields": ["derived_probe_id", "dataset", "family", "variant_type", "old_judge_label"],
    }
    write_json(paths["prepare_report"], report)
    print(canonical_payload(report))


def finalize(stage: Path) -> None:
    paths = stage_paths(stage)
    canonical, manifest, raw = validate_source(paths)
    del canonical, manifest
    packet = read_jsonl(paths["packet"])
    mapping = read_jsonl(paths["mapping"])
    cached = read_jsonl(paths["cache"])
    decisions: list[dict[str, Any]] = []
    for path in sorted(paths["decisions"].glob("part_*.jsonl")):
        decisions.extend(read_jsonl(path))

    expected = [row["blind_id"] for row in packet]
    actual = [row.get("blind_id") for row in decisions]
    if len(actual) != len(set(actual)):
        raise RuntimeError("duplicate direct derived-judge decision")
    if set(actual) != set(expected):
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        raise RuntimeError(f"M3BENCH_V3_STOP__DERIVED_JUDGE_INCOMPLETE missing={missing} extra={extra}")
    required = {"blind_id", "is_correct", "confidence", "reason", "probe_issue"}
    for row in decisions:
        if set(row) != required or type(row["is_correct"]) is not bool or type(row["probe_issue"]) is not bool:
            raise RuntimeError(f"invalid direct decision schema: {row.get('blind_id')}")
        if row["confidence"] not in {"high", "medium", "low"} or not str(row["reason"]).strip():
            raise RuntimeError(f"incomplete direct decision: {row['blind_id']}")
    issues = [row for row in decisions if row["probe_issue"]]
    if issues:
        write_json(stage / "reports" / "DERIVED_PROBE_ISSUES.json", {
            "status": "M3BENCH_V3_STOP__DERIVED_PROBE_SCHEMA_INVALID",
            "count": len(issues),
            "issues": issues,
        })
        raise RuntimeError("M3BENCH_V3_STOP__DERIVED_PROBE_SCHEMA_INVALID")

    packet_by_blind = {row["blind_id"]: row for row in packet}
    mapping_by_blind = {row["blind_id"]: row for row in mapping}
    direct_by_id: dict[str, dict[str, Any]] = {}
    for decision in decisions:
        blind_id = decision["blind_id"]
        packet_row = packet_by_blind[blind_id]
        map_row = mapping_by_blind[blind_id]
        direct_by_id[map_row["derived_probe_id"]] = {
            "is_correct": decision["is_correct"],
            "judge_correct": decision["is_correct"],
            "judge_reason": decision["reason"],
            "judge_confidence": decision["confidence"],
            "judge_model": JUDGE_MODEL,
            "judge_route": "gpt56_sol_direct_derived_v3",
            "judge_version": JUDGE_VERSION,
            "rubric_version": JUDGE_VERSION,
            "rubric_sha256": RUBRIC_SHA256,
            "judge_input_sha256": map_row["judge_input_sha256"],
            "judge_cache_key": map_row["judge_input_sha256"],
            "blind_id": blind_id,
            "blind_packet_sha256": sha256_text(canonical_payload(packet_row)),
            "exact_cache_source_record_ids": [],
        }

    cached_by_id: dict[str, dict[str, Any]] = {}
    for row in cached:
        cached_by_id[row["derived_probe_id"]] = {
            "is_correct": row["is_correct"],
            "judge_correct": row["is_correct"],
            "judge_reason": row["source_judge_reason"],
            "judge_confidence": row["source_judge_confidence"],
            "judge_model": JUDGE_MODEL,
            "judge_route": "gpt56_sol_exact_canonical_triple_reuse",
            "judge_version": JUDGE_VERSION,
            "rubric_version": JUDGE_VERSION,
            "rubric_sha256": RUBRIC_SHA256,
            "judge_input_sha256": row["exact_input_sha256"],
            "judge_cache_key": row["source_judge_input_sha256"],
            "blind_id": None,
            "blind_packet_sha256": None,
            "exact_cache_source_record_ids": row["source_record_ids"],
        }

    expected_ids = {row["derived_probe_id"] for row in raw}
    if set(direct_by_id) & set(cached_by_id) or set(direct_by_id) | set(cached_by_id) != expected_ids:
        raise RuntimeError("M3BENCH_V3_STOP__DERIVED_JUDGE_INCOMPLETE")
    judged = []
    for row in sorted(raw, key=lambda item: item["derived_probe_id"]):
        verdict = direct_by_id.get(row["derived_probe_id"]) or cached_by_id[row["derived_probe_id"]]
        judged.append({**row, **verdict})
    judged_sha = write_jsonl(paths["output"], judged)
    route_counts = Counter(row["judge_route"] for row in judged)
    report = {
        "status": "PASS__DERIVED_GPT56SOL_JUDGE_COMPLETE",
        "count": len(judged),
        "missing": 0,
        "duplicate": 0,
        "unresolved": 0,
        "probe_issue_count": 0,
        "verdict_counts": dict(Counter(str(row["is_correct"]).lower() for row in judged)),
        "route_counts": dict(route_counts),
        "raw_sha256": sha256(paths["raw"]),
        "judged_sha256": judged_sha,
        "packet_sha256": sha256(paths["packet"]),
        "mapping_sha256": sha256(paths["mapping"]),
        "rubric_sha256": RUBRIC_SHA256,
    }
    write_json(paths["hashes"], {
        "raw_inputs": [str(paths["raw"])],
        "raw_sha256": report["raw_sha256"],
        "judged_count": len(judged),
        "judged_sha256": judged_sha,
        "judge_version": JUDGE_VERSION,
        "route_counts": dict(route_counts),
    })
    write_json(paths["final_report"], report)
    print(canonical_payload(report))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "finalize"))
    parser.add_argument("--stage", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.stage.resolve())
    else:
        finalize(args.stage.resolve())


if __name__ == "__main__":
    main()
