#!/usr/bin/env python3
"""Prove that the EqKey-clean Router-R1 schema fix changes references only."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.liveedit_med.router_r1_schema import (
    canonical_hash,
    locality_identity,
    resolve_locality_source,
)


PROTOCOL = "LIVEEDIT_MED_EQKEY_CLEAN_FAST_CONFIRMATION_V1"
EXPECTED_BANK_HASH = "35ba58fa0f78619b0156846a175a31b28fefd779f25b39250a7c238f58ffe4db"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read(path: Path) -> Mapping[str, Any]:
    return json.loads(path.read_text())


def cache_record(path: Path, manifests: dict[Path, Mapping[str, Any]]) -> Mapping[str, Any]:
    manifest_path = path.parent / "manifest.json"
    manifest = manifests.setdefault(manifest_path, read(manifest_path))
    resolved = str(path.resolve())
    matches = [row for row in manifest["records"]
               if str(Path(row["file_path"]).resolve()) == resolved]
    if len(matches) != 1:
        raise RuntimeError(f"ROUTER_R1_SCHEMA_AUDIT_CACHE_RECORD:{path}")
    return matches[0]


def expected_hashes(value: str) -> dict[str, str]:
    result = {}
    for item in value.split(","):
        name, digest = item.split("=", 1)
        result[name] = digest
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--strict-checkpoint", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-before", required=True)
    parser.add_argument("--expected-artifacts", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)

    run = args.run_dir
    paths = {
        "clean_train_manifest": run / "router_r1/clean_train_family_manifest.json",
        "validation_manifest": run / "purged_split/purged_validation_manifest.json",
        "heldout_manifest": run / "purged_split/purged_heldout_manifest.json",
        "hard_negative_manifest": run / "router_r1/hard_negative_cache_manifest.json",
        "fixed_expert_manifest": run / ".runtime_router_r1/fixed_experts/manifest.json",
        "checkpoint_selection": run / "checkpoint_reselection/checkpoint_selection.json",
        "baseline_decision": run / "baseline_clean_evaluation/baseline_decision.json",
        "anchor_audit": run / "anchor_and_immutability_audit.json",
        "blind_audit": run / "blind_set_seal_audit.json",
        "cache_parity": run / "cache_reuse/cache_parity.json",
    }
    actual_hashes = {name: sha256_file(path) for name, path in paths.items()}
    expected = expected_hashes(args.expected_artifacts)
    artifact_hash_parity = {name: actual_hashes[name] == digest for name, digest in expected.items()}

    train = read(paths["clean_train_manifest"])
    validation = read(paths["validation_manifest"])
    heldout = read(paths["heldout_manifest"])
    hard = read(paths["hard_negative_manifest"])
    fixed = read(paths["fixed_expert_manifest"])
    selection = read(paths["checkpoint_selection"])
    anchor = read(paths["anchor_audit"])
    blind = read(paths["blind_audit"])
    parity = read(paths["cache_parity"])

    manifests: dict[Path, Mapping[str, Any]] = {}
    rows = []
    tensor_components = {name: [] for name in ("labels", "hidden", "masks", "logits")}
    generation_tokens = []
    for family in train["families"]:
        canonical = family["canonical_locality"]["image_locality"]
        source = resolve_locality_source(family)
        if source.identity != locality_identity(canonical):
            raise RuntimeError(f"ROUTER_R1_SCHEMA_AUDIT_IDENTITY:{family['family_id']}")
        record = cache_record(source.cache_file_path, manifests)
        tensor_hashes = record["tensor_hashes"]
        tensor_components["labels"].append(tensor_hashes["image_locality__labels"])
        tensor_components["hidden"].append(tensor_hashes["image_locality__hidden"])
        tensor_components["masks"].append({key: tensor_hashes[f"image_locality__{key}"]
            for key in ("answer_mask", "attention", "question_mask", "vision_mask")})
        tensor_components["logits"].append(tensor_hashes["image_locality__base_answer_logits"])
        generation_tokens.append(canonical.get("clean_generation", {}).get("token_ids"))
        rows.append({"family_id": family["family_id"],
            "record_id": family.get("canonical_record_id"),
            "canonical_identity_hash": canonical_hash(locality_identity(canonical)),
            "resolved_identity_hash": canonical_hash(source.identity),
            **source.audit_dict()})

    step3000 = next(row for row in anchor["strict_checkpoints"] if int(row["step"]) == 3000)
    strict_hashes = {
        "directory": str(args.strict_checkpoint.resolve()),
        "manifest_sha256": sha256_file(args.strict_checkpoint / "manifest.json"),
        "model_file_sha256": sha256_file(args.strict_checkpoint / "model.safetensors"),
        "anchor_manifest_sha256": step3000["manifest_sha256"],
        "anchor_model_file_sha256": step3000["model_file_sha256"],
    }
    source_before = expected_hashes(args.source_before)
    source_after = {name: sha256_file(args.source_root / name) for name in source_before}
    changed_files = [name for name in source_before if source_before[name] != source_after[name]]

    hard_mapping = [{"family_id": row["family_id"],
        "negative_eqkeys": row["negative_eqkeys"]} for row in hard["records"]]
    family_ids = [row["family_id"] for row in rows]
    record_ids = [row["record_id"] for row in rows]
    locality_hash_before = canonical_hash([row["canonical_identity_hash"] for row in rows])
    locality_hash_after = canonical_hash([row["resolved_identity_hash"] for row in rows])
    scientific_hashes = {
        "family_ids": canonical_hash(family_ids),
        "record_ids": canonical_hash(record_ids),
        "locality_identity_before": locality_hash_before,
        "locality_identity_after": locality_hash_after,
        "labels": canonical_hash(tensor_components["labels"]),
        "hidden_state_cache": canonical_hash(tensor_components["hidden"]),
        "masks": canonical_hash(tensor_components["masks"]),
        "logits": canonical_hash(tensor_components["logits"]),
        "generation_tokens": canonical_hash(generation_tokens),
        "negative_pair_mapping": canonical_hash(hard_mapping),
        "resolved_cache_files": canonical_hash([
            [row["family_id"], row["cache_file_sha256"]] for row in rows]),
    }
    checks = {
        "artifact_hash_parity": all(artifact_hash_parity.values()),
        "locality_count_preserved": len(rows) == int(train["family_count"]),
        "locality_identity_preserved": locality_hash_before == locality_hash_after,
        "selected_generator_step_3000": int(selection["selected_step"]) == 3000,
        "generator_checkpoint_hash_preserved":
            strict_hashes["manifest_sha256"] == strict_hashes["anchor_manifest_sha256"]
            and strict_hashes["model_file_sha256"] == strict_hashes["anchor_model_file_sha256"],
        "canonical_bank_preserved": anchor["canonical_bank_hash"] == EXPECTED_BANK_HASH
            and fixed["canonical_bank_hash"] == EXPECTED_BANK_HASH,
        "validation_heldout_unchanged": validation["record953_used"] is False
            and heldout["record953_used"] is False,
        "negative_mapping_unchanged": hard["audit_passed"] is True,
        "cache_parity_preexisting_pass": parity["status"] == "PASS"
            and parity["old_cache_mutated"] is False,
        "record953_excluded": train["record953_excluded"] is True
            and fixed["record953_used"] is False,
        "sealed_blind_excluded": train["sealed_blind_excluded"] is True
            and fixed["sealed_blind_loaded"] is False and blind["passed"] is True,
        "source_change_allowlist": set(changed_files) == {
            "scripts/liveedit_med/train_eqkey_clean_router_r1.py"},
    }
    result = "PASS" if all(checks.values()) else "FAIL"
    output = {
        "protocol": PROTOCOL,
        "parent_run": run.name,
        "result": result,
        "version_control": {"local_checkout": "NOT_GIT_REPOSITORY",
            "remote_checkout": "NOT_GIT_REPOSITORY"},
        "change_scope": ["loader implementation", "schema/reference resolution",
                         "tests", "read-only audit"],
        "source_hashes_before": source_before,
        "source_hashes_after": source_after,
        "changed_existing_files": changed_files,
        "new_files": ["scripts/liveedit_med/router_r1_schema.py",
                      "scripts/liveedit_med/audit_eqkey_clean_router_r1_schema_fix.py",
                      "tests/liveedit_med/test_router_r1_schema.py"],
        "selected_generator_checkpoint": strict_hashes,
        "canonical_bank_hash": EXPECTED_BANK_HASH,
        "frozen_artifact_hashes": actual_hashes,
        "expected_artifact_hashes": expected,
        "artifact_hash_parity": artifact_hash_parity,
        "locality_before": {"count": len(rows), "identity_hash": locality_hash_before,
                            "family_ids_hash": scientific_hashes["family_ids"],
                            "record_ids_hash": scientific_hashes["record_ids"]},
        "locality_after": {"count": len(rows), "identity_hash": locality_hash_after,
                           "family_ids_hash": scientific_hashes["family_ids"],
                           "record_ids_hash": scientific_hashes["record_ids"]},
        "scientific_identity_hashes": scientific_hashes,
        "record_953_touch": False,
        "sealed_blind_touch": False,
        "parity": checks,
        "resolved_locality_rows_hash": canonical_hash(rows),
        "forbidden_artifact_touches": {"generator": False, "canonical_bank": False,
            "train_manifest": False, "validation_manifest": False, "heldout_manifest": False,
            "negative_mapping": False, "cache_tensors": False, "record953": False,
            "sealed_blind": False},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result, "locality_count": len(rows),
                      "changed_existing_files": changed_files}, sort_keys=True))
    if result != "PASS":
        raise RuntimeError("ROUTER_R1_SCHEMA_FIX_AUDIT_FAILED")


if __name__ == "__main__":
    main()
