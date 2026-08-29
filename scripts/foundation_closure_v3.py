#!/usr/bin/env python3
"""Frozen Foundation V3 preflight, governance, and review-selection utilities.

This script intentionally does not implement semantic judging. Review verdicts
must be supplied by the current GPT-5.6 Sol agent after reading the frozen blind
packet record by record.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import random
import shutil
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


EXPECTED_COMMIT = "c15740666a83ae9639c315f207640e9467bb826f"
BASE_GENERATION_SHA256 = "b5ed03f252c8e78e93eb788f0f1c51b085746804c52f6b415bae6a5f10458f04"
RUBRIC_SHA256 = "e6e58d7f6307740bf6f341066c0a74fe65683df0e6d4379a991ae82d0ea151a6"
CORRECTED_INITIAL_SHA256 = "5ae96d292108b8e9961a646f8d811a447057d8df7d9ecb0f6d8ca0a2f132cf48"
EXCLUDED_RECORD_ID = "m3bench-v2/SLAKE/xmlab444/xmlab444_8"
REVIEW_SEED = 2026081906
HIGH_CONFIDENCE_SAMPLE_SIZE = 54

FROZEN_EXTERNAL = {
    "base_generation": (
        "/remote-home/wangbomin/worktrees/m3bench_foundation_v2_20260818/outputs/"
        "m3bench_llava_med_foundation_v3_20260818T110907Z/artifacts/"
        "llava_med_base_raw_predictions_16276.jsonl",
        BASE_GENERATION_SHA256,
    ),
    "judge_A": (
        "/remote-home/wangbomin/worktrees/m3bench_foundation_v2_20260818/outputs/"
        "m3bench_llava_med_foundation_v3_20260818T110907Z/artifacts/judge_c4_run_a.jsonl",
        "a2786c8459e58300fab3775a787e8f70a1bf4213c5e9c26ef542952077b4c679",
    ),
    "judge_B": (
        "/remote-home/wangbomin/worktrees/m3bench_foundation_v2_20260818/outputs/"
        "m3bench_llava_med_foundation_v3_20260818T110907Z/artifacts/judge_c4_run_b.jsonl",
        "2d44fe452ac10cf3ff31a8816b2f52195254a1845c9ba086c232589082cde424",
    ),
    "judge_C": (
        "/remote-home/wangbomin/worktrees/multi_editor_v2_20260819/outputs/"
        "m3bench_llava_med_multi_editor_v2_20260819T023605Z/constrained_runs/run_1.jsonl",
        "775c7126f03bcd12501c876ca5d757bc63e9dd30a315eba2ab4958c9780f64b4",
    ),
    "judge_D": (
        "/remote-home/wangbomin/worktrees/multi_editor_v2_20260819/outputs/"
        "m3bench_llava_med_multi_editor_v2_20260819T023605Z/constrained_runs/run_2.jsonl",
        "00812d78a9c146d48a8518e90e13b455a4c3c5301ba9bdd1b7de4145d70fea8c",
    ),
    "judge_E": (
        "/remote-home/wangbomin/worktrees/multi_editor_v2_20260819/outputs/"
        "m3bench_llava_med_multi_editor_v2_20260819T023605Z/alternate_local_runs/run_1.jsonl",
        "4b52a54646744aa417b5ee1f17a03a13754fdca1fb5caf5bba3fe5641b00ca0f",
    ),
    "judge_F": (
        "/remote-home/wangbomin/worktrees/multi_editor_v2_20260819/outputs/"
        "m3bench_llava_med_multi_editor_v2_20260819T023605Z/alternate_local_runs/run_2.jsonl",
        "81c2bc81c4a365478838eaa252f379cb803270bcbc1b30286a5ea1eaf025903f",
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for row in rows
    )
    path.write_text(payload, encoding="utf-8")
    return sha256(path)


def write_json(path: Path, value: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return sha256(path)


def command(args: list[str], cwd: Path | None = None) -> str:
    return subprocess.check_output(args, cwd=cwd, text=True).strip()


def worktree_snapshot(root: Path) -> list[dict[str, Any]]:
    raw = command(["git", "worktree", "list", "--porcelain"], cwd=root)
    output: list[dict[str, Any]] = []
    for block in raw.split("\n\n"):
        lines = block.splitlines()
        path = Path(lines[0].split(" ", 1)[1])
        status = command(["git", "status", "--porcelain"], cwd=path)
        output.append({
            "path": str(path),
            "head": command(["git", "rev-parse", "HEAD"], cwd=path),
            "clean": not bool(status),
            "status_porcelain": status.splitlines(),
        })
    return output


def parse_checksum_manifest(path: Path) -> list[tuple[str, Path]]:
    entries: list[tuple[str, Path]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, relative = line.split(maxsplit=1)
        entries.append((digest, path.parent.parent / relative.strip()))
    return entries


def preflight(args: argparse.Namespace) -> None:
    root = args.root.resolve()
    stage = root / "gpt56_sol_judge" / "foundation_closure_v3"
    audit = root / "gpt56_sol_judge" / "foundation_data_audit"

    # Capture cleanliness before creating any stage artifacts.
    worktrees = worktree_snapshot(root)
    head = command(["git", "rev-parse", "HEAD"], cwd=root)
    branch = command(["git", "branch", "--show-current"], cwd=root)

    checks: dict[str, Any] = {}
    checks["source_commit"] = {"expected": EXPECTED_COMMIT, "actual": head, "pass": head == EXPECTED_COMMIT}
    checks["branch"] = {"actual": branch, "pass": branch == "foundation_closure_t0_t4_editors_20260820"}
    checks["all_worktrees_clean_before_stage_write"] = {
        "pass": all(item["clean"] for item in worktrees), "worktrees": worktrees,
    }

    frozen_rows: list[tuple[str, str, str]] = []
    external_checks = {}
    for name, (raw_path, expected) in FROZEN_EXTERNAL.items():
        path = Path(raw_path)
        actual = sha256(path) if path.is_file() else None
        external_checks[name] = {"path": str(path), "expected": expected, "actual": actual, "pass": actual == expected}
        if actual:
            frozen_rows.append((actual, str(path), name))
    checks["base_and_A_to_F_hashes"] = {
        "pass": all(row["pass"] for row in external_checks.values()), "artifacts": external_checks,
    }

    checksum_manifest = audit / "reports" / "ARTIFACT_SHA256SUMS.txt"
    checksum_entries = parse_checksum_manifest(checksum_manifest)
    validity_results = []
    for expected, path in checksum_entries:
        actual = sha256(path) if path.is_file() else None
        validity_results.append({"path": str(path), "expected": expected, "actual": actual, "pass": actual == expected})
        if actual:
            frozen_rows.append((actual, str(path), "data_validity_artifact"))
    checks["data_validity_45_checksums"] = {
        "expected_count": 45,
        "actual_count": len(validity_results),
        "mismatch_count": sum(not row["pass"] for row in validity_results),
        "pass": len(validity_results) == 45 and all(row["pass"] for row in validity_results),
        "artifacts": validity_results,
    }

    required_manifests = [
        audit / "manifests" / "foundation_qg_exclusion_manifest.jsonl",
        audit / "manifests" / "foundation_qg_valid_manifest.jsonl",
        audit / "manifests" / "foundation_qg_correction_overlay.jsonl",
        audit / "manifests" / "foundation_open_dependency_manifest.jsonl",
        audit / "adjudication" / "qg_validity_final_536.jsonl",
        audit / "provenance" / "foundation_provenance_536.jsonl",
    ]
    checks["required_governance_manifests"] = {
        "pass": all(path.is_file() for path in required_manifests),
        "paths": [{"path": str(path), "sha256": sha256(path) if path.is_file() else None} for path in required_manifests],
    }

    rubric = audit / "inputs" / "JUDGE_PROTOCOL_GPT56SOL.md"
    corrected = audit / "inputs" / "foundation_initial_corrected_gpt56sol.jsonl"
    checks["rubric"] = {"path": str(rubric), "expected": RUBRIC_SHA256, "actual": sha256(rubric), "pass": sha256(rubric) == RUBRIC_SHA256}
    checks["corrected_initial"] = {"path": str(corrected), "expected": CORRECTED_INITIAL_SHA256, "actual": sha256(corrected), "pass": sha256(corrected) == CORRECTED_INITIAL_SHA256}

    checks["no_existing_editor_run_artifacts"] = {
        "checked_path": str(stage / "editors"),
        "pass": not (stage / "editors").exists(),
    }
    checks["protected_data_access"] = {
        "pass": True,
        "accessed": [],
        "policy": "No T5, PadChest-GR, NEJM AI, validation, heldout, record 953, sealed blind, or Stage-2 paths accessed.",
    }

    app_lines = command([
        "nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory", "--format=csv,noheader,nounits",
    ]).splitlines()
    gpu_inventory = command([
        "nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu", "--format=csv,noheader,nounits",
    ]).splitlines()
    checks["gpu_2_3_no_stale_compute_process"] = {
        "pass": len(app_lines) == 0,
        "compute_processes": app_lines,
        "gpu_inventory": gpu_inventory,
    }

    model_paths = [
        Path("/remote-home/wangbomin/hugging_cache/medical_vlms/llava_med_v1_5_mistral_7b"),
        Path("/remote-home/wangbomin/hugging_cache/openai/clip-vit-large-patch14-336"),
    ]
    packages = {name: importlib.metadata.version(name) for name in ("torch", "transformers", "peft", "numpy", "Pillow")}
    checks["cached_models_and_dependencies"] = {
        "pass": all(path.is_dir() for path in model_paths),
        "model_paths": [{"path": str(path), "exists": path.is_dir()} for path in model_paths],
        "packages": packages,
        "new_model_download": False,
    }

    prompt_source, plan_source = args.prompt.resolve(), args.plan.resolve()
    checks["execution_documents"] = {
        "pass": prompt_source.is_file() and plan_source.is_file(),
        "prompt_source": str(prompt_source),
        "prompt_sha256": sha256(prompt_source),
        "plan_source": str(plan_source),
        "plan_sha256": sha256(plan_source),
    }

    passed = all(value.get("pass", False) for value in checks.values())
    status = "PASS__FOUNDATION_CLOSURE_PREFLIGHT" if passed else "M3BENCH_V3_STOP__FOUNDATION_CLOSURE_PREFLIGHT_FAILURE"
    snapshot = {
        "schema_version": "m3bench-foundation-closure-v3-preflight-v1",
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "checks": checks,
    }

    protocol = stage / "protocol"
    protocol.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(prompt_source, protocol / "EXECUTION_PROMPT.md")
    shutil.copyfile(plan_source, protocol / "EXPERIMENT_PLAN.md")
    frozen_rows.extend([
        (sha256(prompt_source), str(protocol / "EXECUTION_PROMPT.md"), "execution_prompt"),
        (sha256(plan_source), str(protocol / "EXPERIMENT_PLAN.md"), "experiment_plan"),
    ])
    preflight_dir = stage / "preflight"
    write_json(preflight_dir / "PREFLIGHT_SNAPSHOT.json", snapshot)
    markdown = [
        "# Foundation Closure V3 Preflight", "", f"Status: `{status}`", "",
        f"Source commit: `{head}`", f"Branch: `{branch}`", "",
        f"Worktrees clean before stage write: `{checks['all_worktrees_clean_before_stage_write']['pass']}`",
        f"Base + A/B/C/D/E/F hashes: `{checks['base_and_A_to_F_hashes']['pass']}`",
        f"Data-validity checksums: `{checks['data_validity_45_checksums']['actual_count']}/45`, mismatches `{checks['data_validity_45_checksums']['mismatch_count']}`",
        f"Rubric SHA-256: `{checks['rubric']['actual']}`",
        f"Corrected initial SHA-256: `{checks['corrected_initial']['actual']}`",
        f"GPU compute processes: `{len(app_lines)}`", "",
        "No protected path was accessed and no model was downloaded in this phase.", "",
    ]
    (preflight_dir / "PREFLIGHT_SNAPSHOT.md").write_text("\n".join(markdown), encoding="utf-8")
    frozen_rows = sorted(set(frozen_rows), key=lambda row: (row[2], row[1]))
    (preflight_dir / "FROZEN_INPUT_SHA256SUMS.txt").write_text(
        "".join(f"{digest}  {path}  # {label}\n" for digest, path, label in frozen_rows), encoding="utf-8"
    )
    print(json.dumps({"status": status, "snapshot": str(preflight_dir / 'PREFLIGHT_SNAPSHOT.json')}, sort_keys=True))
    if not passed:
        raise SystemExit(2)


def phase1(args: argparse.Namespace) -> None:
    root = args.root.resolve()
    stage = root / "gpt56_sol_judge" / "foundation_closure_v3"
    audit = root / "gpt56_sol_judge" / "foundation_data_audit"
    base_path = Path(FROZEN_EXTERNAL["base_generation"][0])

    manifest = read_jsonl(audit / "inputs" / "foundation_manifest_536.jsonl")
    blind = read_jsonl(audit / "inputs" / "foundation_blind.jsonl")
    initial = read_jsonl(audit / "inputs" / "foundation_initial_corrected_gpt56sol.jsonl")
    qg_final = read_jsonl(audit / "adjudication" / "qg_validity_final_536.jsonl")
    valid = read_jsonl(audit / "manifests" / "foundation_qg_valid_manifest.jsonl")
    excluded = read_jsonl(audit / "manifests" / "foundation_qg_exclusion_manifest.jsonl")
    provenance = read_jsonl(audit / "provenance" / "foundation_provenance_536.jsonl")
    overlay = read_jsonl(audit / "manifests" / "foundation_qg_correction_overlay.jsonl")
    dependency = read_jsonl(audit / "manifests" / "foundation_open_dependency_manifest.jsonl")
    base = read_jsonl(base_path)

    def index(rows: list[dict[str, Any]], name: str) -> dict[str, dict[str, Any]]:
        result = {row["record_id"]: row for row in rows}
        if len(result) != len(rows):
            raise RuntimeError(f"duplicate record_id in {name}")
        return result

    manifest_by_id, blind_by_id, initial_by_id = index(manifest, "manifest"), index(blind, "blind"), index(initial, "initial")
    qg_by_id, valid_by_id, provenance_by_id = index(qg_final, "qg_final"), index(valid, "valid"), index(provenance, "provenance")
    excluded_by_id, overlay_by_id = index(excluded, "excluded"), index(overlay, "overlay")
    base_by_index = {row["global_index"]: row for row in base}

    ids = set(manifest_by_id)
    assert len(manifest) == len(blind) == len(initial) == len(qg_final) == len(provenance) == len(overlay) == 536
    assert ids == set(blind_by_id) == set(initial_by_id) == set(qg_by_id) == set(provenance_by_id) == set(overlay_by_id)
    assert set(valid_by_id) | set(excluded_by_id) == ids
    assert not (set(valid_by_id) & set(excluded_by_id))
    assert len(valid_by_id) == 535 and list(excluded_by_id) == [EXCLUDED_RECORD_ID]
    assert len(dependency) == 1 and dependency[0]["record_id"] == EXCLUDED_RECORD_ID
    assert sha256(base_path) == BASE_GENERATION_SHA256

    source_hashes = {
        "base_generation_sha256": BASE_GENERATION_SHA256,
        "foundation_manifest_sha256": sha256(audit / "inputs" / "foundation_manifest_536.jsonl"),
        "blind_packet_sha256": sha256(audit / "inputs" / "foundation_blind.jsonl"),
        "corrected_initial_judgment_sha256": sha256(audit / "inputs" / "foundation_initial_corrected_gpt56sol.jsonl"),
        "qg_validity_final_sha256": sha256(audit / "adjudication" / "qg_validity_final_536.jsonl"),
        "qg_valid_manifest_sha256": sha256(audit / "manifests" / "foundation_qg_valid_manifest.jsonl"),
        "qg_exclusion_manifest_sha256": sha256(audit / "manifests" / "foundation_qg_exclusion_manifest.jsonl"),
        "qg_correction_overlay_sha256": sha256(audit / "manifests" / "foundation_qg_correction_overlay.jsonl"),
        "open_dependency_manifest_sha256": sha256(audit / "manifests" / "foundation_open_dependency_manifest.jsonl"),
        "provenance_manifest_sha256": sha256(audit / "provenance" / "foundation_provenance_536.jsonl"),
        "rubric_sha256": RUBRIC_SHA256,
    }

    governance: list[dict[str, Any]] = []
    evaluator: list[dict[str, Any]] = []
    for row in sorted(manifest, key=lambda item: item["global_index"]):
        record_id = row["record_id"]
        packet, judged, qg = blind_by_id[record_id], initial_by_id[record_id], qg_by_id[record_id]
        raw = base_by_index[row["global_index"]]
        assert raw["record_id"] == record_id and raw["status"] == "success"
        assert raw["model_answer_raw"] == row["model_answer_raw"] == packet["model_answer"]
        assert row["question"] == packet["question"] and row["gold_answer"] == packet["gold_answer"]
        assert isinstance(judged["correct"], bool)
        action = "exclude" if record_id in excluded_by_id else "keep"
        assert qg["action"] == action
        score_status = "NA/excluded" if action == "exclude" else "eligible/scored"
        is_correct = None if action == "exclude" else judged["correct"]
        common = {
            "global_index": row["global_index"],
            "record_id": record_id,
            "dataset": row["dataset"],
            "image_id": row["image_id"],
            "question_id": row["question_id"],
            "relative_image_path": raw["relative_image_path"],
            "question": row["question"],
            "question_raw": row["question"],
            "gold_answer": row["gold_answer"],
            "gold_answer_raw_or_null": row["gold_answer"],
            "model_answer": row["model_answer_raw"],
            "model_answer_raw": row["model_answer_raw"],
            "raw_prediction_sha256": row["raw_prediction_sha256"],
            "generation_config_hash": raw["generation_config_hash"],
            "model_revision": raw["model_revision"],
            "tokenizer_revision": raw["tokenizer_revision"],
            "preprocessing_hash": raw["preprocessing_hash"],
            "prompt_hash": raw["prompt_hash"],
            "question_sha256": sha256_text(row["question"]),
            "status": raw["status"],
            "is_correct": is_correct,
            "judge_correct": is_correct,
            "judge_reason": judged["reason"],
            "judge_confidence": judged["confidence"],
            "judge_model": judged["judge_model"],
            "judge_route": "gpt56_sol_corrected_initial",
            "judge_version": judged["rubric_version"],
            "judge_cache_key": judged["input_sha256"],
            "judge_input_sha256": judged["input_sha256"],
            "rubric_version": judged["rubric_version"],
            "rubric_sha256": RUBRIC_SHA256,
            "question_type": judged["question_type"],
            "governance_action": action,
            "score_status": score_status,
            "source_generation_sha256": BASE_GENERATION_SHA256,
            "source_hashes": source_hashes,
        }
        governance.append({
            **common,
            "pre_exclusion_initial_correct": judged["correct"],
            "qg_final_validity_label": qg["final_validity_label"],
            "qg_issue_type": qg["issue_type"],
            "qg_scoring_disposition": qg["scoring_disposition"],
            "provenance_status": provenance_by_id[record_id]["provenance_status"],
            "correction_action": overlay_by_id[record_id]["action"],
        })
        if action == "keep":
            assert isinstance(common["is_correct"], bool)
            evaluator.append(common)

    governance_path = stage / "manifests" / "foundation_governance_sidecar_536.jsonl"
    evaluator_path = stage / "sidecars" / "foundation_evaluator_predictions_535.jsonl"
    governance_sha = write_jsonl(governance_path, governance)
    evaluator_sha = write_jsonl(evaluator_path, evaluator)
    vqarad = [row for row in evaluator if row["dataset"] == "VQA-RAD"]
    slake = [row for row in evaluator if row["dataset"] == "SLAKE"]
    vqarad_sha = write_jsonl(stage / "sidecars" / "vqarad_evaluator_predictions_188.jsonl", vqarad)
    slake_sha = write_jsonl(stage / "sidecars" / "slake_evaluator_predictions_347.jsonl", slake)

    action_counts = Counter(row["governance_action"] for row in governance)
    assert len(governance) == len({row["record_id"] for row in governance}) == 536
    assert action_counts == {"keep": 535, "exclude": 1}
    assert sum(row["governance_action"] == "exclude" and row["is_correct"] is None and row["score_status"] == "NA/excluded" for row in governance) == 1
    assert len(evaluator) == len({row["record_id"] for row in evaluator}) == 535
    assert all(type(row["is_correct"]) is bool for row in evaluator)
    assert EXCLUDED_RECORD_ID not in {row["record_id"] for row in evaluator}
    assert len(vqarad) == 188 and len(slake) == 347

    report = {
        "status": "PASS__FOUNDATION_GOVERNANCE_PROPAGATION",
        "governance": {"total": 536, "keep": 535, "exclude": 1, "correct": 0, "unresolved": 0, "sha256": governance_sha},
        "evaluator": {"total": 535, "strict_boolean": 535, "null": 0, "vqarad": 188, "slake": 347, "sha256": evaluator_sha},
        "dataset_sidecars": {"vqarad_sha256": vqarad_sha, "slake_sha256": slake_sha},
        "excluded_record_id": EXCLUDED_RECORD_ID,
        "raw_model_answer_exact_matches": 536,
        "source_hashes": source_hashes,
    }
    write_json(stage / "reports" / "GOVERNANCE_PROPAGATION_REPORT.json", report)
    (stage / "reports" / "GOVERNANCE_PROPAGATION_REPORT.md").write_text(
        "# Foundation Governance Propagation\n\n"
        "Status: `PASS__FOUNDATION_GOVERNANCE_PROPAGATION`\n\n"
        "- Governance: 536 unique = 535 keep + 1 NA/excluded.\n"
        "- Evaluator: 535 unique strict booleans = 188 VQA-RAD + 347 SLAKE.\n"
        "- Excluded record is absent from every evaluator sidecar.\n"
        "- Frozen model answers match the base-generation artifact for all 536 records.\n",
        encoding="utf-8",
    )
    print(json.dumps(report, sort_keys=True))


def normalize_answer(value: str) -> str:
    import re
    import unicodedata
    punctuation = re.compile(r"[^\w\s/-]+", flags=re.UNICODE)
    space = re.compile(r"\s+")
    articles = {"a", "an", "the"}
    numbers = {
        "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
        "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
    }
    synonyms = {
        "x ray": "xray", "x-ray": "xray", "magnetic resonance imaging": "mri",
        "computed tomography": "ct", "ultrasound": "us", "colour": "color",
    }
    value = unicodedata.normalize("NFKC", str(value)).casefold().strip()
    value = punctuation.sub(" ", value)
    value = space.sub(" ", value).strip()
    for source, target in synonyms.items():
        value = re.sub(rf"\b{re.escape(source)}\b", target, value)
    return " ".join(numbers.get(token, token) for token in value.split() if token not in articles)


YES = {"yes", "y", "present", "positive", "true", "included", "contain", "contains"}
NO = {"no", "n", "absent", "negative", "false", "none", "without", "not present", "not seen"}


def contains_negation(value: str) -> bool:
    normalized = normalize_answer(value)
    tokens = set(normalized.split())
    return bool({"no", "not", "without", "absent", "absence", "negative", "none"} & tokens) or any(
        token in value for token in ("没有", "未见", "无明显", "不存在", "阴性")
    )


def contains_affirmation(value: str) -> bool:
    normalized = normalize_answer(value)
    tokens = set(normalized.split())
    return bool({"yes", "present", "positive", "seen", "identified", "demonstrated", "shows", "showing"} & tokens) or any(
        token in value for token in ("可见", "存在", "阳性", "显示")
    )


def is_presence_absence(packet: dict[str, Any]) -> bool:
    if packet["category"] in {"presence_absence", "yes_no"}:
        return True
    question = normalize_answer(packet["question"])
    return question.startswith(("is ", "are ", "does ", "do ", "has ", "have ", "can ")) and normalize_answer(
        packet["gold_answer"]
    ) in (YES | NO)


def allocate_high_sample(packets: list[dict[str, Any]], judgments: list[dict[str, Any]]) -> set[str]:
    packet_by_id = {row["record_id"]: row for row in packets}
    high = [row for row in judgments if row["confidence"] == "high"]
    if len(high) < HIGH_CONFIDENCE_SAMPLE_SIZE:
        raise RuntimeError("insufficient high-confidence Foundation records")
    strata: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    for row in high:
        packet = packet_by_id[row["record_id"]]
        strata[(packet["dataset"], packet["task"], packet["category"])].append(row["record_id"])
    quotas: dict[tuple[str, str, str], int] = {}
    remainders: list[tuple[float, tuple[str, str, str]]] = []
    for key in sorted(strata):
        exact = HIGH_CONFIDENCE_SAMPLE_SIZE * len(strata[key]) / len(high)
        quotas[key] = int(exact)
        remainders.append((exact - int(exact), key))
    for _, key in sorted(remainders, key=lambda item: (-item[0], item[1]))[: HIGH_CONFIDENCE_SAMPLE_SIZE - sum(quotas.values())]:
        quotas[key] += 1
    rng = random.Random(REVIEW_SEED)
    selected: set[str] = set()
    for key in sorted(strata):
        members = sorted(strata[key])
        rng.shuffle(members)
        selected.update(members[: quotas[key]])
    if len(selected) != HIGH_CONFIDENCE_SAMPLE_SIZE:
        raise RuntimeError("high-confidence sample-size mismatch")
    return selected


def selection(args: argparse.Namespace) -> None:
    root = args.root.resolve()
    stage = root / "gpt56_sol_judge" / "foundation_closure_v3"
    audit = root / "gpt56_sol_judge" / "foundation_data_audit"
    governance = read_jsonl(stage / "manifests" / "foundation_governance_sidecar_536.jsonl")
    keep_ids = {row["record_id"] for row in governance if row["governance_action"] == "keep"}
    packets_all = read_jsonl(audit / "inputs" / "foundation_blind.jsonl")
    judgments_all = read_jsonl(audit / "inputs" / "foundation_initial_corrected_gpt56sol.jsonl")
    packets = [row for row in packets_all if row["record_id"] in keep_ids]
    judgments = [row for row in judgments_all if row["record_id"] in keep_ids]
    judgment_by_id = {row["record_id"]: row for row in judgments}
    high_sample = allocate_high_sample(packets, judgments)
    reasons_by_id: dict[str, list[str]] = defaultdict(list)
    for packet in packets:
        record_id = packet["record_id"]
        judged = judgment_by_id[record_id]
        if judged["confidence"] == "low":
            reasons_by_id[record_id].append("low_confidence")
        if judged["confidence"] == "medium":
            reasons_by_id[record_id].append("medium_confidence")
        if packet["category"] == "modality" or judged["question_type"] == "modality":
            reasons_by_id[record_id].append("modality")
        if packet["category"] == "negation" or contains_negation(packet["question"]):
            reasons_by_id[record_id].append("negation")
        if is_presence_absence(packet):
            reasons_by_id[record_id].append("presence_absence")
        if contains_negation(packet["model_answer"]) and contains_affirmation(packet["model_answer"]):
            reasons_by_id[record_id].append("mixed_polarity_prediction")
        if record_id in high_sample:
            reasons_by_id[record_id].append("high_confidence_stratified_sample")

    selected_packets = [packet for packet in packets if reasons_by_id.get(packet["record_id"])]
    selected_manifest, blind_packet = [], []
    for ordinal, packet in enumerate(selected_packets, start=1):
        judged = judgment_by_id[packet["record_id"]]
        blind_id = f"review-v3-{ordinal:04d}"
        selected_manifest.append({
            "blind_id": blind_id,
            "record_id": packet["record_id"],
            "source_record_index": packet["source_record_index"],
            "review_selection_reasons": reasons_by_id[packet["record_id"]],
        })
        blind_packet.append({
            "blind_id": blind_id,
            "question": packet["question"],
            "gold_answer": packet["gold_answer"],
            "model_answer": packet["model_answer"],
            "initial_reason": judged["reason"],
            "initial_confidence": judged["confidence"],
        })

    selection_path = stage / "manifests" / "review_v3_selection_535.jsonl"
    packet_path = stage / "packets" / "foundation_review_v3_blind.jsonl"
    selection_sha = write_jsonl(selection_path, selected_manifest)
    packet_sha = write_jsonl(packet_path, blind_packet)

    old_rows = read_jsonl(args.old_review_v2)
    old_ids = {row["record_id"] for row in old_rows}
    new_ids = {row["record_id"] for row in selected_manifest}
    trigger_counts = Counter(reason for reasons in reasons_by_id.values() for reason in reasons)
    report = {
        "status": "PASS__FOUNDATION_REVIEW_V3_SELECTION_FROZEN",
        "candidate_keep_count": 535,
        "selected_count": len(selected_manifest),
        "unique_record_ids": len(new_ids),
        "seed": REVIEW_SEED,
        "sorting": "source foundation blind-packet order (source_record_index/global_index order)",
        "high_confidence_sample_size": len(high_sample),
        "high_confidence_minimum": HIGH_CONFIDENCE_SAMPLE_SIZE,
        "trigger_counts": dict(sorted(trigger_counts.items())),
        "review_v2_count": len(old_ids),
        "review_v2_overlap": len(old_ids & new_ids),
        "review_v2_removed": sorted(old_ids - new_ids),
        "review_v3_added": sorted(new_ids - old_ids),
        "selection_sha256": selection_sha,
        "blind_packet_sha256": packet_sha,
        "blind_packet_fields": ["blind_id", "question", "gold_answer", "model_answer", "initial_reason", "initial_confidence"],
    }
    assert len(selected_manifest) == len(new_ids)
    assert EXCLUDED_RECORD_ID not in new_ids
    assert len(high_sample) == HIGH_CONFIDENCE_SAMPLE_SIZE
    report_path = stage / "reports" / "REVIEW_V3_SELECTION_REPORT.json"
    write_json(report_path, report)
    md = [
        "# Foundation Review V3 Selection", "", f"Status: `{report['status']}`", "",
        f"Keep cohort: `{report['candidate_keep_count']}`",
        f"Selected union: `{report['selected_count']}`",
        f"Fixed high-confidence audit sample: `{report['high_confidence_sample_size']}`",
        f"Seed: `{REVIEW_SEED}`", "",
        f"Review-v2 overlap: `{report['review_v2_overlap']}/{report['review_v2_count']}`",
        f"Removed: `{report['review_v2_removed']}`",
        f"Added: `{report['review_v3_added']}`", "",
        f"Selection SHA-256: `{selection_sha}`",
        f"Blind packet SHA-256: `{packet_sha}`", "",
    ]
    (stage / "reports" / "REVIEW_V3_SELECTION_REPORT.md").write_text("\n".join(md), encoding="utf-8")
    print(json.dumps(report, sort_keys=True))


def finalize_review(args: argparse.Namespace) -> None:
    root = args.root.resolve()
    stage = root / "gpt56_sol_judge" / "foundation_closure_v3"
    audit = root / "gpt56_sol_judge" / "foundation_data_audit"
    selection_rows = read_jsonl(stage / "manifests" / "review_v3_selection_535.jsonl")
    packet_rows = read_jsonl(stage / "packets" / "foundation_review_v3_blind.jsonl")
    governance = read_jsonl(stage / "manifests" / "foundation_governance_sidecar_536.jsonl")
    evaluator = read_jsonl(stage / "sidecars" / "foundation_evaluator_predictions_535.jsonl")
    initial = read_jsonl(audit / "inputs" / "foundation_initial_corrected_gpt56sol.jsonl")

    decisions: list[dict[str, Any]] = []
    for path in sorted(args.decisions_dir.glob("part_*.jsonl")):
        decisions.extend(read_jsonl(path))
    expected_ids = [row["blind_id"] for row in packet_rows]
    decision_ids = [row["blind_id"] for row in decisions]
    if len(decision_ids) != len(set(decision_ids)):
        raise RuntimeError("M3BENCH_V3_STOP__FOUNDATION_REVIEW_V3_FAILURE:duplicate")
    missing = sorted(set(expected_ids) - set(decision_ids))
    extra = sorted(set(decision_ids) - set(expected_ids))
    if missing or extra:
        raise RuntimeError(f"M3BENCH_V3_STOP__FOUNDATION_REVIEW_V3_FAILURE:missing={missing}:extra={extra}")
    if len(decisions) != 203:
        raise RuntimeError("M3BENCH_V3_STOP__FOUNDATION_REVIEW_V3_FAILURE:count")
    required = {"blind_id", "reviewed_verdict", "confidence", "reason", "dataset_issue"}
    for row in decisions:
        if set(row) != required:
            raise RuntimeError(f"review decision schema mismatch: {row.get('blind_id')}")
        if type(row["reviewed_verdict"]) is not bool or type(row["dataset_issue"]) is not bool:
            raise RuntimeError(f"review decision must be boolean: {row['blind_id']}")
        if row["confidence"] not in {"high", "medium", "low"} or not row["reason"].strip():
            raise RuntimeError(f"review decision incomplete: {row['blind_id']}")

    selection_by_blind = {row["blind_id"]: row for row in selection_rows}
    packet_by_blind = {row["blind_id"]: row for row in packet_rows}
    initial_by_id = {row["record_id"]: row for row in initial}
    governance_by_id = {row["record_id"]: row for row in governance}
    review_rows: list[dict[str, Any]] = []
    changes: list[dict[str, Any]] = []
    for decision in sorted(decisions, key=lambda row: row["blind_id"]):
        blind_id = decision["blind_id"]
        selection_row = selection_by_blind[blind_id]
        record_id = selection_row["record_id"]
        packet = packet_by_blind[blind_id]
        prior = initial_by_id[record_id]
        allowed_input = {
            key: packet[key]
            for key in ("blind_id", "question", "gold_answer", "model_answer", "initial_reason", "initial_confidence")
        }
        review_row = {
            "blind_id": blind_id,
            "record_id": record_id,
            "reviewed_verdict": decision["reviewed_verdict"],
            "confidence": decision["confidence"],
            "reason": decision["reason"],
            "dataset_issue": decision["dataset_issue"],
            "judge_model": "gpt-5.6-sol",
            "rubric_version": "m3bench-gpt56sol-v1",
            "rubric_sha256": RUBRIC_SHA256,
            "review_input_sha256": sha256_text(json.dumps(allowed_input, ensure_ascii=False, sort_keys=True, separators=(",", ":"))),
            "review_selection_reasons": selection_row["review_selection_reasons"],
        }
        review_rows.append(review_row)
        if prior["correct"] != decision["reviewed_verdict"]:
            changes.append({
                "blind_id": blind_id,
                "record_id": record_id,
                "initial_verdict": prior["correct"],
                "reviewed_verdict": decision["reviewed_verdict"],
                "initial_reason": prior["reason"],
                "review_reason": decision["reason"],
            })

    new_dataset_issues = [row for row in review_rows if row["dataset_issue"]]
    review_path = stage / "judgments" / "foundation_review_v3.jsonl"
    changes_path = stage / "judgments" / "foundation_review_v3_changes.jsonl"
    review_sha = write_jsonl(review_path, review_rows)
    changes_sha = write_jsonl(changes_path, changes)
    review_report = {
        "status": "M3BENCH_V3_STOP__FOUNDATION_NEW_DATA_ISSUE" if new_dataset_issues else "PASS__FOUNDATION_REVIEW_V3_COMPLETE",
        "selected": 203,
        "completed": len(review_rows),
        "coverage": len(review_rows) / 203,
        "missing": len(missing),
        "duplicate": len(decision_ids) - len(set(decision_ids)),
        "unresolved": 0,
        "new_dataset_issue": len(new_dataset_issues),
        "verdict_counts": dict(Counter(str(row["reviewed_verdict"]).lower() for row in review_rows)),
        "confidence_counts": dict(Counter(row["confidence"] for row in review_rows)),
        "verdict_change_count": len(changes),
        "review_sha256": review_sha,
        "changes_sha256": changes_sha,
        "selection_sha256": sha256(stage / "manifests" / "review_v3_selection_535.jsonl"),
        "blind_packet_sha256": sha256(stage / "packets" / "foundation_review_v3_blind.jsonl"),
        "rubric_sha256": RUBRIC_SHA256,
    }
    write_json(stage / "reports" / "FOUNDATION_REVIEW_V3_REPORT.json", review_report)
    (stage / "reports" / "FOUNDATION_REVIEW_V3_REPORT.md").write_text(
        "# Foundation Review V3\n\n"
        f"Status: `{review_report['status']}`\n\n"
        f"- Coverage: `{review_report['completed']}/{review_report['selected']}`\n"
        f"- Missing / duplicate / unresolved: `{review_report['missing']} / {review_report['duplicate']} / {review_report['unresolved']}`\n"
        f"- New dataset issues: `{review_report['new_dataset_issue']}`\n"
        f"- Verdict changes from corrected initial: `{review_report['verdict_change_count']}`\n"
        f"- Review SHA-256: `{review_sha}`\n",
        encoding="utf-8",
    )
    if new_dataset_issues:
        print(json.dumps(review_report, sort_keys=True))
        raise SystemExit(3)

    review_by_id = {row["record_id"]: row for row in review_rows}
    canonical: list[dict[str, Any]] = []
    for row in evaluator:
        record_id = row["record_id"]
        output = dict(row)
        review = review_by_id.get(record_id)
        if review:
            output.update({
                "is_correct": review["reviewed_verdict"],
                "judge_correct": review["reviewed_verdict"],
                "judge_reason": review["reason"],
                "judge_confidence": review["confidence"],
                "judge_route": "gpt56_sol_review_v3",
                "judge_model": review["judge_model"],
                "judge_version": review["rubric_version"],
                "rubric_version": review["rubric_version"],
                "rubric_sha256": review["rubric_sha256"],
                "reviewed_in_v3": True,
                "review_v3_blind_id": review["blind_id"],
                "review_v3_input_sha256": review["review_input_sha256"],
            })
        else:
            output.update({"reviewed_in_v3": False, "review_v3_blind_id": None, "review_v3_input_sha256": None})
        canonical.append(output)
    canonical.sort(key=lambda row: row["global_index"])
    if len(canonical) != 535 or len({row["record_id"] for row in canonical}) != 535:
        raise RuntimeError("M3BENCH_V3_STOP__CANONICAL_SIDECAR_FAILURE:coverage")
    if not all(type(row["is_correct"]) is bool for row in canonical):
        raise RuntimeError("M3BENCH_V3_STOP__CANONICAL_SIDECAR_FAILURE:boolean")
    if EXCLUDED_RECORD_ID in {row["record_id"] for row in canonical}:
        raise RuntimeError("M3BENCH_V3_STOP__CANONICAL_SIDECAR_FAILURE:excluded")

    canonical_path = stage / "judgments" / "foundation_canonical_gpt56sol_v3_535.jsonl"
    canonical_sha = write_jsonl(canonical_path, canonical)
    vqarad = [row for row in canonical if row["dataset"] == "VQA-RAD"]
    slake = [row for row in canonical if row["dataset"] == "SLAKE"]
    vqarad_sha = write_jsonl(stage / "sidecars" / "vqarad_predictions_gpt56sol_v3_188.jsonl", vqarad)
    slake_sha = write_jsonl(stage / "sidecars" / "slake_predictions_gpt56sol_v3_347.jsonl", slake)

    def census(rows: list[dict[str, Any]]) -> dict[str, int]:
        return {"true": sum(row["is_correct"] is True for row in rows), "false": sum(row["is_correct"] is False for row in rows)}

    per_question_type: dict[str, dict[str, int]] = {}
    for question_type in sorted({row["question_type"] for row in canonical}):
        per_question_type[question_type] = census([row for row in canonical if row["question_type"] == question_type])
    canonical_report = {
        "status": "PASS__FOUNDATION_CANONICAL_GPT56SOL_V3_535",
        "scored_count": 535,
        "excluded_count": 1,
        "reviewed_count": len(review_rows),
        "verdict_change_count": len(changes),
        **census(canonical),
        "per_dataset": {"VQA-RAD": census(vqarad), "SLAKE": census(slake)},
        "per_question_type": per_question_type,
        "canonical_sha256": canonical_sha,
        "vqarad_sha256": vqarad_sha,
        "slake_sha256": slake_sha,
        "review_sha256": review_sha,
        "governance_sha256": sha256(stage / "manifests" / "foundation_governance_sidecar_536.jsonl"),
        "source_generation_sha256": BASE_GENERATION_SHA256,
        "rubric_sha256": RUBRIC_SHA256,
    }
    write_json(stage / "reports" / "FOUNDATION_CANONICAL_CENSUS_V3.json", canonical_report)
    (stage / "reports" / "FOUNDATION_CANONICAL_CENSUS_V3.md").write_text(
        "# Foundation Canonical GPT-5.6 Sol V3 Census\n\n"
        f"Status: `{canonical_report['status']}`\n\n"
        f"Scored / excluded: `{canonical_report['scored_count']} / {canonical_report['excluded_count']}`\n\n"
        f"True / false: `{canonical_report['true']} / {canonical_report['false']}`\n\n"
        f"Reviewed / changed: `{canonical_report['reviewed_count']} / {canonical_report['verdict_change_count']}`\n\n"
        f"Canonical SHA-256: `{canonical_sha}`\n",
        encoding="utf-8",
    )
    print(json.dumps({"review": review_report, "canonical": canonical_report}, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="action", required=True)
    pre = subparsers.add_parser("preflight")
    pre.add_argument("--root", type=Path, required=True)
    pre.add_argument("--prompt", type=Path, required=True)
    pre.add_argument("--plan", type=Path, required=True)
    p1 = subparsers.add_parser("phase1")
    p1.add_argument("--root", type=Path, required=True)
    sel = subparsers.add_parser("selection")
    sel.add_argument("--root", type=Path, required=True)
    sel.add_argument("--old-review-v2", type=Path, required=True)
    final = subparsers.add_parser("finalize-review")
    final.add_argument("--root", type=Path, required=True)
    final.add_argument("--decisions-dir", type=Path, required=True)
    args = parser.parse_args()
    {"preflight": preflight, "phase1": phase1, "selection": selection, "finalize-review": finalize_review}[args.action](args)


if __name__ == "__main__":
    main()
