#!/usr/bin/env python3
"""Freeze evidence for the Foundation V3 protected-data hard stop."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


STATUS = "M3BENCH_V3_STOP__PROTECTED_DATA_ACCESS_VIOLATION"
BASE_SHA256 = "b5ed03f252c8e78e93eb788f0f1c51b085746804c52f6b415bae6a5f10458f04"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_record(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def inventory(directory: Path) -> dict[str, Any]:
    files = sorted(directory.glob("*.json")) if directory.is_dir() else []
    rows = [read_record(path) for path in files]
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.name.encode("utf-8"))
        digest.update(sha256(path).encode("ascii"))
    return {
        "path": str(directory),
        "count": len(files),
        "status_counts": dict(Counter(str(row.get("status")) for row in rows)),
        "error_counts": dict(Counter(str(row.get("error_or_null")) for row in rows if row.get("error_or_null"))),
        "directory_manifest_sha256": digest.hexdigest(),
    }


def command(*args: str) -> str:
    return subprocess.check_output(args, text=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    stage = root / "gpt56_sol_judge" / "foundation_closure_v3"
    local_artifacts = stage / "task_sets" / "local_535" / "artifacts"
    attempts_root = local_artifacts / "attempts"
    attempt_inventory = {
        path.name: inventory(path)
        for path in sorted(attempts_root.iterdir())
        if path.is_dir()
    } if attempts_root.is_dir() else {}
    t2g_cache = local_artifacts / "t2g_variant_cache"
    active = command("bash", "-lc", "ps -eo pid,args | grep -E 'foundation_v3_derived.py (generate|t2g)' | grep -v grep || true")
    gpu_state = command(
        "nvidia-smi", "--query-gpu=index,memory.used,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ).splitlines()
    report = {
        "schema_version": "m3bench-foundation-v3-protected-stop-v1",
        "status": STATUS,
        "triggered_at_utc": datetime.now(timezone.utc).isoformat(),
        "stopped_phase": "Phase 4 task rebuild / derived-probe generation",
        "violation": {
            "operation": "recursive software-version text search over an old run directory",
            "cause": "the search scope unintentionally included old raw prediction records",
            "protected_category_observed": "validation",
            "minimal_evidence": "one matched record exposed source_split=validation",
            "access_mode": "read_only",
            "protected_artifact_modified": False,
            "no_further_protected_read_after_detection": True,
        },
        "cleanup": {
            "terminated_owned_process_ids": [469321, 470583],
            "active_derived_processes_after_cleanup": active.splitlines() if active else [],
            "gpu_state_after_cleanup": gpu_state,
            "gpu_2_3_released": all(line.split(",")[1].strip() in {"0", "1", "2", "3", "4"} for line in gpu_state if line.split(",")[0].strip() in {"2", "3"}),
        },
        "preserved_artifacts": {
            "base_generation_frozen_sha256_from_passed_preflight": BASE_SHA256,
            "base_generation_modified": False,
            "current_derived_records": inventory(local_artifacts / "derived_raw_prediction_records"),
            "failed_attempts": attempt_inventory,
            "t2g_cache_path": str(t2g_cache),
            "t2g_cache_file_count": len(list(t2g_cache.glob("*.json"))) if t2g_cache.is_dir() else 0,
        },
        "phase_outcome": {
            "phase_0_preflight": "PASS",
            "phase_1_governance": "PASS",
            "phase_2_review_v3": "PASS",
            "phase_3_canonical_535": "PASS",
            "phase_4_task_rebuild": "INTERRUPTED_NOT_PASSED",
            "phase_5_foundation_closure": "NOT_RUN",
            "editor_config_gate": "NOT_RUN",
            "editor_execution": "NOT_RUN",
            "post_edit_judge": "NOT_RUN",
        },
        "continuation_allowed": False,
        "required_resolution": "start a new authorized clean run from frozen inputs; do not resume this stopped run",
    }
    if report["cleanup"]["active_derived_processes_after_cleanup"] or not report["cleanup"]["gpu_2_3_released"]:
        raise RuntimeError("owned processes or GPU allocations remain after protected-data hard stop")
    reports = stage / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    json_path = reports / "PROTECTED_DATA_ACCESS_VIOLATION.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path = reports / "PROTECTED_DATA_ACCESS_VIOLATION.md"
    md_path.write_text(
        "# Foundation V3 Protected-Data Hard Stop\n\n"
        f"Status: `{STATUS}`\n\n"
        "A software-version audit used an overly broad recursive search and entered an old raw "
        "prediction record marked as validation. The access was read-only, but the frozen protocol "
        "requires an immediate hard stop for any protected-path access.\n\n"
        f"- Phase 4: interrupted, not passed\n"
        f"- Foundation closure: not run\n"
        f"- Editors: not run\n"
        f"- GPU 2/3 released: `{report['cleanup']['gpu_2_3_released']}`\n"
        f"- T2G cache preserved: `{report['preserved_artifacts']['t2g_cache_file_count']}` files\n"
        f"- Current derived records preserved: `{report['preserved_artifacts']['current_derived_records']['count']}`\n\n"
        "This stopped run must not be resumed.\n",
        encoding="utf-8",
    )
    report_hashes = reports / "PROTECTED_DATA_ACCESS_VIOLATION_SHA256SUMS.txt"
    report_hashes.write_text(
        f"{sha256(json_path)}  {json_path}\n{sha256(md_path)}  {md_path}\n",
        encoding="utf-8",
    )
    marker = "<!-- foundation-v3-protected-stop-20260820 -->"
    status_path = root / "EXECUTION_STATUS.md"
    existing = status_path.read_text(encoding="utf-8") if status_path.is_file() else "# Execution Status\n"
    if marker not in existing:
        status_path.write_text(
            existing.rstrip() + "\n\n" + marker + "\n"
            "## Foundation Closure V3 hard stop\n\n"
            f"Status: `{STATUS}`\n\n"
            "Phase 4 was interrupted after an overly broad read-only metadata search entered an old "
            "validation raw record. Foundation closure and all editors were not run. GPU 2/3 were released.\n",
            encoding="utf-8",
        )
    print(json.dumps({
        "status": STATUS,
        "json": str(json_path),
        "markdown": str(md_path),
        "json_sha256": sha256(json_path),
        "markdown_sha256": sha256(md_path),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
