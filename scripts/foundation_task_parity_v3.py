#!/usr/bin/env python3
"""Normalize public/local T0--T4 builders and freeze their formal parity audit."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from m3bench_foundation.judge import normalize_answer


TASKS = ("T0", "T1L", "T1G", "T2L", "T2G", "T3L", "T3G", "T4L", "T4G")
EXCLUDED = "m3bench-v2/SLAKE/xmlab444/xmlab444_8"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )
    return sha256(path)


def write_json(path: Path, value: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return sha256(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=Path, required=True)
    args = parser.parse_args()
    stage = args.stage.resolve()
    official_dir = stage / "task_sets" / "official_535"
    local_dir = stage / "task_sets" / "local_535"
    canonical = read_jsonl(stage / "judgments" / "foundation_canonical_gpt56sol_v3_535.jsonl")
    sequence = read_jsonl(local_dir / "artifacts" / "main_sequence_200.jsonl")
    derived_manifest = read_jsonl(local_dir / "artifacts" / "derived_probe_manifest.jsonl")
    derived_judged = read_jsonl(local_dir / "artifacts" / "derived_probe_judged_predictions.jsonl")
    if len(canonical) != 535 or len(sequence) != 200:
        raise RuntimeError("M3BENCH_V3_STOP__TASK_REBUILD_OR_PARITY_FAILURE:input_count")

    by_id = {row["record_id"]: row for row in canonical}
    by_diq = {(row["dataset"], row["image_id"], str(row["question_id"])): row for row in canonical}
    by_iq: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in canonical:
        by_iq[(row["image_id"], normalize_answer(row["question"]))].append(row)
    sequence_ids = [row["record_id"] for row in sequence]
    sequence_pos = {record_id: index for index, record_id in enumerate(sequence_ids, start=1)}
    derived_by_anchor_family: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in derived_manifest:
        derived_by_anchor_family[(row["source_record_id"], row["family"])].append(row)
    derived_judged_by_id = {row["derived_probe_id"]: row for row in derived_judged}

    def resolve(dataset: str | None, image_id: str, question: str | None = None, question_id: str | None = None) -> str:
        normalized_dataset = {"slake": "SLAKE", "vqarad": "VQA-RAD"}.get(str(dataset).casefold(), dataset)
        if question_id not in (None, ""):
            hit = by_diq.get((normalized_dataset, image_id, str(question_id)))
            if hit:
                return hit["record_id"]
        candidates = by_iq.get((image_id, normalize_answer(question or "")), [])
        if normalized_dataset:
            filtered = [row for row in candidates if row["dataset"] == normalized_dataset]
            if filtered:
                candidates = filtered
        if len(candidates) != 1:
            raise RuntimeError(f"cannot uniquely resolve public-builder row: {dataset}/{image_id}/{question_id}/{question}")
        return candidates[0]["record_id"]

    def public_anchor(task: str, row: dict[str, Any]) -> str:
        dataset = row.get("dataset")
        if task == "T0":
            return resolve(dataset, row["image_id"], row.get("question"), row.get("question_id"))
        if task == "T1L":
            return resolve(dataset, row["edit_image_id"], row.get("question"), row.get("question_id"))
        if task in {"T1G", "T2G"}:
            return resolve(dataset, row["image_id"], row.get("question"), row.get("question_id"))
        if task in {"T2L", "T3L", "T3G", "T4G"}:
            return resolve(dataset, row["edit_image_id"], row.get("edit_question"), row.get("edit_question_id"))
        if task == "T4L":
            return resolve(dataset, row["image_id"], row.get("edit_question"))
        raise ValueError(task)

    def public_probes(task: str, row: dict[str, Any], anchor_id: str) -> list[dict[str, Any]]:
        if task == "T0":
            return []
        if task in {"T1G", "T2G"}:
            probes = []
            for candidate in derived_by_anchor_family.get((anchor_id, task), []):
                if not candidate["valid"]:
                    continue
                judged = derived_judged_by_id.get(candidate["derived_probe_id"])
                if judged is None:
                    raise RuntimeError(f"missing derived judgment: {candidate['derived_probe_id']}")
                probes.append({"id": candidate["derived_probe_id"], "pre_is_correct": judged["is_correct"]})
            return probes
        if task == "T4L":
            cases = [{"image_id": row["image_id"], "question": row["eval_question"]}]
        else:
            cases = row.get("eval_cases", [])
        probes = []
        for case in cases:
            probe_id = resolve(None, case["image_id"], case.get("question"), case.get("question_id"))
            pre = by_id[probe_id]["is_correct"]
            if task == "T4G" and pre is True:
                continue
            probes.append({"id": probe_id, "pre_is_correct": pre})
        return probes

    public_normalized: list[dict[str, Any]] = []
    public_raw_hashes: dict[str, str] = {}
    for task in TASKS:
        path = official_dir / f"{task}_task_set.jsonl"
        public_raw_hashes[task] = sha256(path)
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in read_jsonl(path):
            anchor_id = public_anchor(task, row)
            if anchor_id in sequence_pos:
                grouped[anchor_id].append(row)
        for anchor_id in sequence_ids:
            probes: list[dict[str, Any]] = []
            for row in grouped.get(anchor_id, []):
                probes.extend(public_probes(task, row, anchor_id))
            unique = {(row["id"], row["pre_is_correct"]): row for row in probes}
            attempted = [row["derived_probe_id"] for row in derived_by_anchor_family.get((anchor_id, task), [])] if task in {"T1G", "T2G"} else [row["id"] for row in unique.values()]
            invalid = [row["derived_probe_id"] for row in derived_by_anchor_family.get((anchor_id, task), []) if not row["valid"]] if task in {"T1G", "T2G"} else []
            public_normalized.append({
                "task": task,
                "sequence_position": sequence_pos[anchor_id],
                "anchor_record_id": anchor_id,
                "anchor_pre_is_correct": False,
                "probes": sorted(unique.values(), key=lambda row: row["id"]),
                "attempted_probe_ids": sorted(attempted),
                "invalid_probe_ids": sorted(invalid),
                "public_raw_anchor_present": bool(grouped.get(anchor_id)),
            })
    public_norm_path = official_dir / "normalized_formal_task_membership.jsonl"
    public_norm_sha = write_jsonl(public_norm_path, public_normalized)

    local_normalized: list[dict[str, Any]] = []
    local_manifest_hashes: dict[str, str] = {}
    for task in TASKS:
        path = local_dir / "artifacts" / f"{task}_task_manifest.jsonl"
        local_manifest_hashes[task] = sha256(path)
        rows = read_jsonl(path)
        if len(rows) != 200:
            raise RuntimeError(f"local {task} does not contain 200 formal anchors")
        for row in rows:
            anchor_id = row["anchor"]["record_id"]
            probes = [{"id": probe["record_id_or_derived_probe_id"], "pre_is_correct": probe["pre_is_correct"]} for probe in row["probes"]]
            attempted = [item["derived_probe_id"] for item in derived_by_anchor_family.get((anchor_id, task), [])] if task in {"T1G", "T2G"} else [item["id"] for item in probes]
            invalid = [item["derived_probe_id"] for item in derived_by_anchor_family.get((anchor_id, task), []) if not item["valid"]] if task in {"T1G", "T2G"} else []
            local_normalized.append({
                "task": task,
                "sequence_position": row["sequence_position"],
                "anchor_record_id": anchor_id,
                "anchor_pre_is_correct": row["anchor_pre_correct"],
                "probes": sorted(probes, key=lambda item: item["id"]),
                "attempted_probe_ids": sorted(attempted),
                "invalid_probe_ids": sorted(invalid),
                "public_raw_anchor_present": None,
            })
    local_norm_path = local_dir / "normalized_formal_task_membership.jsonl"
    local_norm_sha = write_jsonl(local_norm_path, local_normalized)

    def key(row: dict[str, Any]) -> tuple[str, int]:
        return row["task"], row["sequence_position"]

    public_by_key, local_by_key = {key(row): row for row in public_normalized}, {key(row): row for row in local_normalized}
    mismatches: list[dict[str, Any]] = []
    for item_key in sorted(public_by_key):
        public, local = public_by_key[item_key], local_by_key.get(item_key)
        fields = ("anchor_record_id", "anchor_pre_is_correct", "probes", "attempted_probe_ids", "invalid_probe_ids")
        differences = {field: {"official": public[field], "local": None if local is None else local[field]} for field in fields if local is None or public[field] != local[field]}
        if differences:
            mismatches.append({"task": item_key[0], "sequence_position": item_key[1], "differences": differences})

    membership: list[dict[str, Any]] = []
    task_counts: dict[str, Any] = {}
    for task in TASKS:
        rows = [row for row in local_normalized if row["task"] == task]
        locality = task.endswith("L")
        defined = 0
        eligible_total = 0
        for row in rows:
            membership.append({
                "task": task, "sequence_position": row["sequence_position"], "role": "edit",
                "record_id_or_derived_probe_id": row["anchor_record_id"], "pre_is_correct": False,
                "eligible_for_denominator": task == "T0",
            })
            eligible = 0
            for probe in row["probes"]:
                is_eligible = bool(probe["pre_is_correct"]) if locality else not bool(probe["pre_is_correct"])
                eligible += int(is_eligible)
                membership.append({
                    "task": task, "sequence_position": row["sequence_position"], "role": "eval",
                    "record_id_or_derived_probe_id": probe["id"], "pre_is_correct": probe["pre_is_correct"],
                    "eligible_for_denominator": is_eligible,
                })
            if task == "T0" or eligible:
                defined += 1
            eligible_total += 1 if task == "T0" else eligible
        task_counts[task] = {
            "edit_count": len(rows),
            "probe_count": sum(len(row["probes"]) for row in rows),
            "defined_edit_count": defined,
            "null_edit_count": 0 if task == "T0" else len(rows) - defined,
            "eligible_denominator_count": eligible_total,
        }

    membership_path = stage / "manifests" / "foundation_task_membership_535.jsonl"
    membership_sha = write_jsonl(membership_path, membership)
    generated_paths = [
        local_dir / "artifacts" / "main_sequence_200.jsonl",
        public_norm_path,
        local_norm_path,
        membership_path,
    ] + [official_dir / f"{task}_task_set.jsonl" for task in TASKS] + [local_dir / "artifacts" / f"{task}_task_manifest.jsonl" for task in TASKS]
    excluded_occurrences = {str(path): path.read_text(encoding="utf-8").count(EXCLUDED) for path in generated_paths}
    excluded_total = sum(excluded_occurrences.values())

    historical_root = Path("/remote-home/wangbomin/worktrees/m3bench_foundation_v2_20260818/outputs/m3bench_llava_med_foundation_v3_20260818T110907Z/artifacts")
    historical_paths = [historical_root / f"{task}_task_manifest.jsonl" for task in TASKS]
    historical_available = all(path.is_file() for path in historical_paths)
    delta = {
        "status": "PASS__EXCLUDED_RECORD_ABSENT_FROM_TASK_DEPENDENCIES" if excluded_total == 0 else "FAIL",
        "excluded_record_id": EXCLUDED,
        "historical_task_sets_available": historical_available,
        "historical_state": "NOT_BUILT_BECAUSE_PRIOR_FOUNDATION_HARD_STOPPED" if not historical_available else "AVAILABLE",
        "audited_535_occurrences": excluded_total,
        "per_artifact_occurrences": excluded_occurrences,
        "edit_case_occurrences": sum(row["role"] == "edit" and row["record_id_or_derived_probe_id"] == EXCLUDED for row in membership),
        "eval_case_occurrences": sum(row["role"] == "eval" and row["record_id_or_derived_probe_id"] == EXCLUDED for row in membership),
        "denominator_occurrences": sum(row["eligible_for_denominator"] and row["record_id_or_derived_probe_id"] == EXCLUDED for row in membership),
        "formal_sequence_occurrences": sum(row["record_id"] == EXCLUDED for row in sequence),
    }
    delta_path = stage / "manifests" / "excluded_record_dependency_delta.json"
    write_json(delta_path, delta)
    (stage / "manifests" / "excluded_record_dependency_delta.md").write_text(
        "# Excluded Record Dependency Delta\n\n"
        f"Status: `{delta['status']}`\n\n"
        f"Historical task sets: `{delta['historical_state']}`\n\n"
        f"Audited 535 occurrences across edit/eval/order/denominator artifacts: `{excluded_total}`\n",
        encoding="utf-8",
    )

    report = {
        "status": "PASS__OFFICIAL_LOCAL_TASK_PARITY" if not mismatches and excluded_total == 0 else "M3BENCH_V3_STOP__TASK_REBUILD_OR_PARITY_FAILURE",
        "canonical_count": 535,
        "formal_sequence_count": 200,
        "official_commit": "03c6fda3813301dab3be5831fdc94b493c10afc9",
        "official_raw_hashes": public_raw_hashes,
        "local_manifest_hashes": local_manifest_hashes,
        "official_normalized_sha256": public_norm_sha,
        "local_normalized_sha256": local_norm_sha,
        "membership_sha256": membership_sha,
        "comparison_rows": len(public_normalized),
        "mismatch_count": len(mismatches),
        "mismatches": mismatches[:50],
        "task_counts": task_counts,
        "excluded_dependency": delta,
        "parity_policy": "public raw builder normalized to frozen formal sequence; T1G/T2G placeholders closed with the frozen derived-probe manifest; generality membership uses pre-wrong probes",
    }
    write_json(stage / "reports" / "TASK_REBUILD_PARITY_REPORT.json", report)
    (stage / "reports" / "TASK_REBUILD_PARITY_REPORT.md").write_text(
        "# T0-T4 Task Rebuild and Parity\n\n"
        f"Status: `{report['status']}`\n\n"
        f"Comparison rows: `{report['comparison_rows']}`\n\n"
        f"Mismatches: `{report['mismatch_count']}`\n\n"
        f"Excluded-record occurrences: `{excluded_total}`\n",
        encoding="utf-8",
    )
    print(json.dumps(report, sort_keys=True))
    if report["status"] != "PASS__OFFICIAL_LOCAL_TASK_PARITY":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
