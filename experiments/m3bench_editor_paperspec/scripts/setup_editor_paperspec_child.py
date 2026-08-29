#!/usr/bin/env python3
"""Freeze the additive M3Bench editor child-run E0 evidence boundary.

This script intentionally touches only the new child run root.  The parent
Foundation run and all historical 0444 artifacts are read-only inputs.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


PARENT_RUN = Path(
    "/remote-home/wangbomin/Knowledge_editing/outputs/"
    "m3bench_foundation_t0_t4_editors_clean_v4/20260827T111255Z"
)
CHILD_WORKTREE = Path(
    "/remote-home/wangbomin/worktrees/"
    "m3bench_editor_paperspec_runtime_v1_20260828T032447Z"
)
CHILD_RUN = Path(
    "/remote-home/wangbomin/Knowledge_editing/outputs/"
    "m3bench_editor_paperspec_runtime_v1/20260828T032447Z"
)
PARENT_COMMIT = "5854df1dd902ec039f5cf1bd6eff4f8f061c917e"
AUTH_SHA256 = "a11b744aa937a233efa8c3f2dd4b0ee8ac836c25fdaf10dd4fd24b1f9fb53481"
OFFICIAL_COMMIT = "03c6fda3813301dab3be5831fdc94b493c10afc9"
EXCLUDED_RECORD = "m3bench-v2/SLAKE/xmlab444/xmlab444_8"

AUTH_PATH = CHILD_RUN / "governance/M3BENCH_V4_OPERATOR_AUTHORIZATION_EDITOR_PAPER_SPEC_AND_COHORTS_20260828.json"
PROMPT_PATH = CHILD_RUN / "governance/CODEX_PROMPT_M3BENCH_EDITOR_PAPER_SPEC_RUNTIME_AND_SMOKE.md"
PLAN_PATH = CHILD_RUN / "governance/M3BENCH_EDITOR_UNBLOCK_AND_SMOKE_EXPERIMENT_PLAN.md"
OFFICIAL_REPO = Path(
    "/remote-home/wangbomin/m3bench_reproduction/external/M3Bench-release-03c6fda"
)

SOURCE_REPOS = {
    "M3Bench public release": (OFFICIAL_REPO, OFFICIAL_COMMIT),
    "LLaVA-Med": (
        Path("/remote-home/wangbomin/m3bench_reproduction/external/LLaVA-Med"),
        "30697ca50b5c29a8e955c99330b259776aef27b9",
    ),
    "GRACE": (
        Path("/remote-home/wangbomin/m3bench_reproduction/external/GRACE"),
        "f674183f17a995d109e10ee6140d4c3e6d016115",
    ),
    "BalanceEdit": (
        Path("/remote-home/wangbomin/m3bench_reproduction/external/BalancEdit"),
        "83749e52a1d27331d21cfec845b6089294730c2f",
    ),
    "PEFT": (
        Path("/remote-home/wangbomin/m3bench_reproduction/external/peft-v0.19.1"),
        "ba6a19060d6ab54a87538a6e77e3e4d5a907375b",
    ),
}

COHORT_HASHES = {
    "cohorts/PROPOSED_COHORT_SELECTION_RULE.json":
        "e65852ea622c7b65f7900ac8c559011a2f46cb0d7eda2a20d1025ec20e1d866c",
    "cohorts/PROPOSED_SMOKE_COHORT_8.jsonl":
        "c773711eba42dbef2bf497e09caae54cfd198b9384a5028970d36bbeb7bfc4b9",
    "cohorts/PROPOSED_SEQUENTIAL_MINISTREAM_4.jsonl":
        "414c212473c20e8f1b00aec5ecccc5fe7b3d1b1661a69177dc4fafca286af0b0",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run(*args: str, cwd: Path | None = None) -> str:
    return subprocess.check_output(args, cwd=cwd, text=True).strip()


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise RuntimeError(f"invalid JSONL at {path}:{line_number}") from exc
    return rows


def write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def copy_read_only(source: Path, destination: Path) -> dict:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # /remote-home does not support the source filesystem's extended attrs;
    # copy the bytes only and prove identity with SHA-256 below.
    if destination.exists():
        os.chmod(destination, 0o644)
    shutil.copyfile(source, destination)
    os.chmod(destination, 0o444)
    source_hash = sha256(source)
    destination_hash = sha256(destination)
    if source_hash != destination_hash:
        raise RuntimeError(f"copy hash mismatch: {source} -> {destination}")
    return {
        "source": str(source),
        "destination": str(destination),
        "sha256": source_hash,
        "size_bytes": source.stat().st_size,
        "destination_mode": "0444",
    }


def main() -> None:
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    if run("git", "rev-parse", "HEAD", cwd=CHILD_WORKTREE) != PARENT_COMMIT:
        raise RuntimeError("child worktree is not rooted at the authorized parent commit")
    if run("git", "status", "--porcelain", cwd=CHILD_WORKTREE):
        raise RuntimeError("child worktree must be clean before E0 carry-forward")
    if sha256(AUTH_PATH) != AUTH_SHA256:
        raise RuntimeError("operator authorization SHA-256 mismatch")
    authorization = json.loads(AUTH_PATH.read_text(encoding="utf-8"))
    if authorization.get("approval_status") != "APPROVED" or authorization.get("approved_by") != "wangbomin":
        raise RuntimeError("operator authorization fields do not approve this run")
    if authorization.get("parent_worktree_commit") != PARENT_COMMIT:
        raise RuntimeError("authorization parent commit mismatch")
    if not (PARENT_RUN / "foundation_v4/M3BENCH_V4_PASS__FOUNDATION_CLOSURE_T0_T4").is_file():
        raise RuntimeError("Foundation PASS marker is missing")

    source_repo_audit = {}
    for name, (repo_path, expected_head) in SOURCE_REPOS.items():
        actual_head = run("git", "rev-parse", "HEAD", cwd=repo_path)
        status = run("git", "status", "--porcelain", cwd=repo_path)
        if actual_head != expected_head or status:
            raise RuntimeError(f"source lock mismatch or dirty checkout: {name}")
        source_repo_audit[name] = {
            "path": str(repo_path),
            "expected_head": expected_head,
            "actual_head": actual_head,
            "clean": True,
        }

    parent_allowlist = [
        "foundation_v4/M3BENCH_V4_PASS__FOUNDATION_CLOSURE_T0_T4",
        "foundation_v4/FOUNDATION_V4_CLOSURE_REPORT.md",
        "foundation_v4/closure/FOUNDATION_V4_CLOSURE_INPUT_MANIFEST.json",
        "foundation_v4/runtime/llava_med_generation_frozen.json",
    ]
    editor_gate = PARENT_RUN / "foundation_v4/editor_gate"
    parent_allowlist.extend(
        str(path.relative_to(PARENT_RUN))
        for path in sorted(editor_gate.rglob("*"))
        if path.is_file()
    )
    judge_audit = PARENT_RUN / "foundation_v4/judge_audit"
    parent_allowlist.extend(
        str(path.relative_to(PARENT_RUN))
        for path in sorted(judge_audit.iterdir())
        if path.is_file()
    )

    copies = []
    for relative in parent_allowlist:
        copies.append(copy_read_only(PARENT_RUN / relative, CHILD_RUN / "carry_forward" / relative))

    cohort_source_root = editor_gate
    for relative, expected_hash in COHORT_HASHES.items():
        actual_hash = sha256(cohort_source_root / relative)
        if actual_hash != expected_hash:
            raise RuntimeError(f"cohort SHA-256 mismatch: {relative}")

    smoke_path = cohort_source_root / "cohorts/PROPOSED_SMOKE_COHORT_8.jsonl"
    mini_path = cohort_source_root / "cohorts/PROPOSED_SEQUENTIAL_MINISTREAM_4.jsonl"
    smoke = load_jsonl(smoke_path)
    mini = load_jsonl(mini_path)
    smoke_ids = [row["record_id"] for row in smoke]
    mini_ids = [row["record_id"] for row in mini]
    dataset_counts = Counter(row["dataset"] for row in smoke)
    invariants = {
        "smoke_count_8": len(smoke) == 8,
        "smoke_vqa_rad_count_4": dataset_counts["VQA-RAD"] == 4,
        "smoke_slake_count_4": dataset_counts["SLAKE"] == 4,
        "smoke_duplicate_record_ids_0": len(smoke_ids) == len(set(smoke_ids)),
        "excluded_record_absent": EXCLUDED_RECORD not in smoke_ids and EXCLUDED_RECORD not in mini_ids,
        "mini_count_4": len(mini) == 4,
        "mini_duplicate_record_ids_0": len(mini_ids) == len(set(mini_ids)),
        "mini_subset_of_smoke": set(mini_ids).issubset(set(smoke_ids)),
        "mini_ordered_by_formal_sequence": [row["formal_sequence_position"] for row in mini]
        == sorted(row["formal_sequence_position"] for row in mini),
    }
    if not all(invariants.values()):
        raise RuntimeError(f"cohort invariant failure: {invariants}")

    pending_source = cohort_source_root / "cohorts/SMOKE_COHORT_PENDING_OPERATOR_APPROVAL"
    pending_copy = CHILD_RUN / "carry_forward/foundation_v4/editor_gate/cohorts/SMOKE_COHORT_PENDING_OPERATOR_APPROVAL"
    if not pending_copy.is_file() or sha256(pending_source) != sha256(pending_copy):
        raise RuntimeError("historical cohort pending marker was not preserved")

    approval = {
        "schema_version": "m3bench-v4-editor-smoke-cohort-additive-approval-v1",
        "status": "APPROVED_BY_EXACT_OPERATOR_AUTHORIZATION",
        "approved_at_utc": authorization["approved_at_utc"],
        "frozen_at_utc": now,
        "authorization_path": str(AUTH_PATH),
        "authorization_sha256": AUTH_SHA256,
        "historical_pending_marker_preserved": str(pending_copy),
        "artifacts": {
            relative: {
                "source": str(cohort_source_root / relative),
                "sha256": expected_hash,
            }
            for relative, expected_hash in COHORT_HASHES.items()
        },
        "invariants": invariants,
        "record_ids": smoke_ids,
        "mini_stream_record_ids": mini_ids,
        "excluded_record": EXCLUDED_RECORD,
        "formal_editor_output_observed_before_selection": False,
    }
    write_json(CHILD_RUN / "OPERATOR_APPROVED_SMOKE_COHORT.json", approval)

    parent_audit = {
        "schema_version": "m3bench-v4-editor-child-parent-artifact-audit-v1",
        "status": "PASS",
        "created_at_utc": now,
        "parent_run": str(PARENT_RUN),
        "parent_commit": PARENT_COMMIT,
        "allowlist_file_count": len(copies),
        "copies": copies,
        "source_repositories": source_repo_audit,
        "parent_modified": False,
    }
    write_json(CHILD_RUN / "PARENT_ARTIFACT_HASH_AUDIT.json", parent_audit)

    access_policy = {
        "schema_version": "m3bench-v4-editor-child-access-policy-v1",
        "status": "ACTIVE_FAIL_CLOSED",
        "created_at_utc": now,
        "allowed_write_roots": [str(CHILD_WORKTREE), str(CHILD_RUN)],
        "parent_read_policy": "exact files enumerated in PARENT_ARTIFACT_HASH_AUDIT.json only",
        "allowed_data_roots": [
            "/remote-home/wangbomin/DataP/knowledge_editing/data/m3bench/VQA-RAD/images",
            "/remote-home/wangbomin/DataP/knowledge_editing/data/m3bench/SLAKE/imgs",
        ],
        "allowed_model_roots": [
            "/remote-home/wangbomin/hugging_cache/medical_vlms/llava_med_v1_5_mistral_7b",
            "/remote-home/wangbomin/hugging_cache/openai/clip-vit-large-patch14-336",
        ],
        "allowed_external_source_roots": [str(path) for path, _ in SOURCE_REPOS.values()],
        "prohibited": [
            "validation raw records",
            "heldout data",
            "record 953",
            "sealed blind data",
            "Stage-2",
            "T5",
            "PadChest-GR",
            "formal single-edit",
            "formal 200-edit sequential",
            "post-edit full Judge",
        ],
        "recursive_parent_or_stopped_run_scan": False,
    }
    write_json(CHILD_RUN / "ACCESS_POLICY.json", access_policy)

    implementation_audit = f"""# M3Bench Editor Paper-Spec Implementation Audit

Status: `PASS__READ_ONLY_IMPLEMENTATION_AUDIT_COMPLETE`

Classification: `M3BENCH_PAPER_SPEC_INDEPENDENT_REIMPLEMENTATION_V1`

The implementations in this child run are independent paper-spec adaptations/reimplementations. They are not the unreleased author runtime. If author code becomes available later, it must be evaluated in a separate run and must not overwrite these results.

## Public benchmark release

- Repository: `https://github.com/BioMed-AI-Lab-U-Michgan/M3Bench.git`
- Frozen checkout: `{OFFICIAL_REPO}`
- Commit: `{OFFICIAL_COMMIT}`
- Checkout: clean
- Finding: `AUTHOR_EDITOR_RUNTIME_NOT_PRESENT_IN_PUBLIC_RELEASE`

The public release contains benchmark metadata, task builders, task definitions, and reporting assets. It does not contain executable LoRA, GRACE, BalanceEdit, or BELoRA editor runtime code.

## Locked implementation sources

| Source | Commit | Local path | Status |
|---|---|---|---|
| LLaVA-Med | `30697ca50b5c29a8e955c99330b259776aef27b9` | `/remote-home/wangbomin/m3bench_reproduction/external/LLaVA-Med` | clean, Foundation-compatible runtime source |
| GRACE | `f674183f17a995d109e10ee6140d4c3e6d016115` | `/remote-home/wangbomin/m3bench_reproduction/external/GRACE` | clean; source retrieval is Euclidean and requires transparent cosine adaptation |
| BalanceEdit | `83749e52a1d27331d21cfec845b6089294730c2f` | `/remote-home/wangbomin/m3bench_reproduction/external/BalancEdit` | clean; released MiniGPT-4 config/runtime requires LLaVA-Med adapter |
| PEFT | `ba6a19060d6ab54a87538a6e77e3e4d5a907375b` | `/remote-home/wangbomin/m3bench_reproduction/external/peft-v0.19.1` | clean; tag-compatible PEFT 0.19.1 mechanics |

No locked third-party source checkout will be modified. Adaptations are implemented in the child project and recorded by child git diff/commit.

## Existing frozen method evidence

The exact existing `METHOD_CONFIG_LOCK`, `METHOD_PROVENANCE_LOCK`, `METHOD_STATE_CONTRACT`, and unit-test artifacts for LoRA, GRACE, BalanceEdit, and BELoRA were copied byte-for-byte from the parent `foundation_v4/editor_gate` into the child carry-forward area and are enumerated in `PARENT_ARTIFACT_HASH_AUDIT.json`.

Current evidence boundary:

- LoRA: CPU tiny-Llama PEFT mechanics passed; real multimodal runtime missing.
- GRACE: source mechanics passed; source Euclidean behavior conflicts with authorized primary cosine distance; LLaVA-Med adapter missing.
- BalanceEdit: radius formula and inclusive boundary passed; LLaVA-Med key/module/input adapter missing.
- BELoRA: no author implementation was released; authorized independent BalanceEdit-routing plus per-edit-LoRA implementation is missing.

## Components required before GPU smoke

1. One shared Foundation-compatible LLaVA-Med editor runtime.
2. Real-model module inventory and exact edit-target lock.
3. Target-answer-token-only multimodal loss and masking tests.
4. Method-local save/load/reset and base-integrity contracts.
5. LoRA all-LM-MLP adapter, GRACE cosine adapter, BalanceEdit LLaVA-Med adapter, and BELoRA paper-spec implementation.
6. CPU/synthetic and integration tests before any real GPU preflight.

No editor output, formal single-edit run, formal 200-edit run, or post-edit Judge output existed when this audit was frozen.
"""
    (CHILD_RUN / "EDITOR_PAPERSPEC_IMPLEMENTATION_AUDIT.md").write_text(
        implementation_audit, encoding="utf-8"
    )

    run_manifest = {
        "schema_version": "m3bench-v4-editor-paperspec-child-run-v1",
        "run_id": "20260828T032447Z",
        "created_at_utc": "2026-08-28T03:24:47Z",
        "e0_frozen_at_utc": now,
        "status": "E0_PASS__READY_FOR_IMPLEMENTATION",
        "classification": "M3BENCH_PAPER_SPEC_INDEPENDENT_REIMPLEMENTATION_V1",
        "branch": "m3bench_editor_paperspec_runtime_v1_20260828T032447Z",
        "worktree": str(CHILD_WORKTREE),
        "run_root": str(CHILD_RUN),
        "parent_run": str(PARENT_RUN),
        "parent_commit": PARENT_COMMIT,
        "authorization_sha256": AUTH_SHA256,
        "prompt_sha256": sha256(PROMPT_PATH),
        "plan_sha256": sha256(PLAN_PATH),
        "approved_cohort_artifact": str(CHILD_RUN / "OPERATOR_APPROVED_SMOKE_COHORT.json"),
        "foundation_preserved": True,
        "formal_editor_experiments_authorized": False,
        "human_audit_approved": False,
        "author_belora_implementation_available": False,
        "next_gate": "IMPLEMENTATION_AND_CPU_TESTS",
    }
    write_json(CHILD_RUN / "RUN_MANIFEST.json", run_manifest)

    output_files = [
        CHILD_RUN / "RUN_MANIFEST.json",
        CHILD_RUN / "ACCESS_POLICY.json",
        CHILD_RUN / "PARENT_ARTIFACT_HASH_AUDIT.json",
        CHILD_RUN / "OPERATOR_APPROVED_SMOKE_COHORT.json",
        CHILD_RUN / "EDITOR_PAPERSPEC_IMPLEMENTATION_AUDIT.md",
    ]
    for path in output_files:
        os.chmod(path, 0o444)
    print(json.dumps({"status": "PASS", "outputs": [str(p) for p in output_files]}, indent=2))


if __name__ == "__main__":
    main()
