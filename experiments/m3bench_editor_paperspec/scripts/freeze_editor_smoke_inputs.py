#!/usr/bin/env python3
"""Freeze the exact approved editor smoke rows plus their official T2G rephrases."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


RUN = Path(
    "/remote-home/wangbomin/Knowledge_editing/outputs/"
    "m3bench_editor_paperspec_runtime_v1/20260828T032447Z"
)
SMOKE = RUN / (
    "carry_forward/foundation_v4/editor_gate/cohorts/PROPOSED_SMOKE_COHORT_8.jsonl"
)
MINI = RUN / (
    "carry_forward/foundation_v4/editor_gate/cohorts/PROPOSED_SEQUENTIAL_MINISTREAM_4.jsonl"
)
T2G = Path(
    "/remote-home/wangbomin/Knowledge_editing/outputs/"
    "m3bench_foundation_t0_t4_editors_clean_v4/20260827T111255Z/"
    "foundation_v4/t2g_gpt56sol_generator/20260827T133313Z/final/"
    "t2g_derived_probe_manifest_800.jsonl"
)
IMAGE_ROOTS = {
    "VQA-RAD": Path("/remote-home/wangbomin/DataP/knowledge_editing/data/m3bench/VQA-RAD/images"),
    "SLAKE": Path("/remote-home/wangbomin/DataP/knowledge_editing/data/m3bench/SLAKE/imgs"),
}
EXPECTED_SMOKE_SHA = "c773711eba42dbef2bf497e09caae54cfd198b9384a5028970d36bbeb7bfc4b9"
EXPECTED_MINI_SHA = "414c212473c20e8f1b00aec5ecccc5fe7b3d1b1661a69177dc4fafca286af0b0"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def canonical_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, obj: object) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    if sha256(SMOKE) != EXPECTED_SMOKE_SHA or sha256(MINI) != EXPECTED_MINI_SHA:
        raise RuntimeError("approved cohort hash mismatch")
    smoke_rows = load_jsonl(SMOKE)
    mini_rows = load_jsonl(MINI)
    t2g_rows = load_jsonl(T2G)
    paraphrases: dict[str, dict] = {}
    for row in t2g_rows:
        if row.get("variant_type") == "paraphrase" and row.get("valid") is True:
            record_id = row["source_record_id"]
            if record_id in paraphrases:
                raise RuntimeError(f"duplicate valid paraphrase: {record_id}")
            paraphrases[record_id] = row

    frozen_rows = []
    for index, row in enumerate(smoke_rows):
        record_id = row["record_id"]
        rephrase = paraphrases.get(record_id)
        if rephrase is None:
            raise RuntimeError(f"missing official paraphrase for {record_id}")
        if rephrase["original_question"] != row["question"]:
            raise RuntimeError(f"original-question mismatch for {record_id}")
        if rephrase["target_reference"] != row["gold_answer"]:
            raise RuntimeError(f"target-reference mismatch for {record_id}")
        image_path = IMAGE_ROOTS[row["dataset"]] / row["relative_image_path"]
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        frozen_rows.append(
            {
                **row,
                "smoke_position": index + 1,
                "image_path": str(image_path),
                "image_sha256": sha256(image_path),
                "official_rephrase": rephrase["prompt"],
                "official_rephrase_probe_id": rephrase["derived_probe_id"],
                "official_rephrase_sha256": rephrase["candidate_question_sha256"],
                "official_rephrase_source_manifest": str(T2G),
                "official_rephrase_source_record_sha256": rephrase["generator_record_sha256"],
            }
        )

    ids = [row["record_id"] for row in frozen_rows]
    mini_ids = [row["record_id"] for row in mini_rows]
    if len(frozen_rows) != 8 or len(ids) != len(set(ids)):
        raise RuntimeError("smoke input cardinality/uniqueness failure")
    if not set(mini_ids).issubset(set(ids)):
        raise RuntimeError("mini stream is not a subset of smoke")

    out = RUN / "inputs/APPROVED_SMOKE_RECORDS_WITH_REPHRASES.jsonl"
    canonical_jsonl(out, frozen_rows)
    manifest = {
        "schema_version": "m3bench-v4-editor-smoke-input-freeze-v1",
        "status": "PASS",
        "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "smoke_source": str(SMOKE),
        "smoke_source_sha256": sha256(SMOKE),
        "mini_source": str(MINI),
        "mini_source_sha256": sha256(MINI),
        "t2g_source": str(T2G),
        "t2g_source_sha256": sha256(T2G),
        "frozen_records": str(out),
        "frozen_records_sha256": sha256(out),
        "record_count": len(frozen_rows),
        "mini_stream_record_ids": mini_ids,
        "protected_data_accessed": False,
    }
    manifest_path = RUN / "inputs/SMOKE_INPUT_MANIFEST.json"
    write_json(manifest_path, manifest)

    amendment = {
        "schema_version": "m3bench-v4-editor-access-policy-additive-amendment-v1",
        "status": "ACTIVE",
        "reason": "BalanceEdit and BELoRA authorization requires the frozen official question rephrase for each approved smoke record.",
        "authorization_basis": "paper_spec_conventions.balancedit.positive_anchor",
        "new_exact_read_allowlist": [str(T2G)],
        "source_sha256": sha256(T2G),
        "derived_output": str(out),
        "derived_output_sha256": sha256(out),
        "parent_artifact_modified": False,
        "protected_path_scope_expanded": False,
    }
    amendment_path = RUN / "ACCESS_POLICY_AMENDMENT_T2G_REPHRASES.json"
    write_json(amendment_path, amendment)
    for path in (out, manifest_path, amendment_path):
        os.chmod(path, 0o444)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
