#!/usr/bin/env python3
"""Audit EqKey families and construct a non-overwriting clean fast lane."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from methods.liveedit_med.eqkey_clean_fast import (POSITIVE_ROLES, PROTOCOL, allocate_purged,
    build_families, canonical_hash, future_family_split)
from methods.liveedit_med.router_r1 import (EXPECTED_BANK_HASH, EXPECTED_BLIND_SEALED_HASH,
    EXPECTED_BLIND_SELECTION_HASH)
from methods.liveedit_med.serialization import tensor_hashes
from scripts.engram.run_engram_v2_stage0_generation_audit import bank_manifest


EXPECTED_STEPS = (500, 1000, 1500, 2000, 2500, 3000, 3200)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def recursive_values(value: Any, keys: set[str]) -> set[str]:
    result: set[str] = set()
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key) in keys and isinstance(child, (str, int)):
                result.add(str(child))
            result.update(recursive_values(child, keys))
    elif isinstance(value, list):
        for child in value:
            result.update(recursive_values(child, keys))
    return result


def checkpoint_dir(training: Path, step: int) -> Path:
    return training / f"checkpoint_{step:04d}"


def audit_checkpoints(strict_run: Path) -> list[dict[str, Any]]:
    results = []
    for step in EXPECTED_STEPS:
        directory = checkpoint_dir(strict_run / "training", step)
        manifest_path, tensor_path = directory / "manifest.json", directory / "model.safetensors"
        if not manifest_path.is_file() or not tensor_path.is_file():
            raise RuntimeError(f"EQKEY_FAST_CONFIRMATION_ANCHOR_MISMATCH:checkpoint_{step}")
        manifest = json.loads(manifest_path.read_text())
        if int(manifest.get("step", -1)) != step:
            raise RuntimeError(f"EQKEY_FAST_CONFIRMATION_ANCHOR_MISMATCH:checkpoint_step_{step}")
        state = load_file(str(tensor_path), device="cpu")
        actual_hashes = tensor_hashes(state)
        if actual_hashes != manifest.get("tensor_hashes"):
            raise RuntimeError(f"EQKEY_FAST_CONFIRMATION_ANCHOR_MISMATCH:checkpoint_hash_{step}")
        results.append({
            "step": step,
            "directory": str(directory.resolve()),
            "manifest_sha256": sha256_file(manifest_path),
            "model_file_sha256": sha256_file(tensor_path),
            "tensor_hash_set": canonical_hash(actual_hashes),
            "tensor_count": len(actual_hashes),
            "verified": True,
        })
        del state
    return results


def cache_manifests(blocked_run: Path) -> list[Path]:
    result = []
    expected = {"train": 512, "validation": 64, "heldout": 64}
    for split, count in expected.items():
        paths = sorted((blocked_run / ".runtime_cache").glob(f"{split}_*/manifest.json"))
        manifests = [json.loads(path.read_text()) for path in paths]
        if not paths or sum(int(row.get("count", -1)) for row in manifests) != count:
            raise RuntimeError(f"EQKEY_FAST_CONFIRMATION_ANCHOR_MISMATCH:cache_count_{split}")
        if any(row.get("split") != split for row in manifests):
            raise RuntimeError(f"EQKEY_FAST_CONFIRMATION_ANCHOR_MISMATCH:cache_split_{split}")
        result.extend(paths)
    return result


def read_cache(blocked_run: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    ledger, cache_index, manifest_rows = [], [], []
    image_hashes: dict[str, str] = {}
    seen_records: set[tuple[str, str]] = set()
    for manifest_path in cache_manifests(blocked_run):
        manifest = json.loads(manifest_path.read_text())
        manifest_rows.append({"path": str(manifest_path.resolve()), "sha256": sha256_file(manifest_path),
                              "split": manifest["split"], "count": manifest["count"]})
        for record in manifest["records"]:
            split, record_id = str(manifest["split"]), str(record["record_id"])
            owner = (split, record_id)
            if owner in seen_records:
                raise RuntimeError(f"EQKEY_FAST_CONFIRMATION_ANCHOR_MISMATCH:duplicate_record:{owner}")
            seen_records.add(owner)
            cache_path = manifest_path.parent / record["file"]
            if not cache_path.is_file():
                raise RuntimeError(f"EQKEY_FAST_CONFIRMATION_ANCHOR_MISMATCH:missing_cache:{cache_path}")
            actual_cache_hash = sha256_file(cache_path)
            if actual_cache_hash != record["file_sha256"]:
                raise RuntimeError(f"EQKEY_FAST_CONFIRMATION_ANCHOR_MISMATCH:cache_hash:{owner}")
            cached_inputs = [{key: item.get(key) for key in ("category", "role", "image", "prompt", "target",
                              "eqkey", "source_hash", "processed_pixel_tensor_sha256", "clean_generation")}
                             for item in record["inputs"]]
            cache_index.append({"source_split": split, "record_id": record_id,
                "source_ordinal": int(record["ordinal"]), "cache_file_path": str(cache_path.resolve()),
                "cache_file_sha256": actual_cache_hash, "tensor_hashes": record["tensor_hashes"],
                "inputs": cached_inputs})
            for item in record["inputs"]:
                role = str(item["category"])
                if role not in POSITIVE_ROLES:
                    continue
                image_path = str(item["image"])
                if image_path not in image_hashes:
                    image_hashes[image_path] = sha256_file(Path(image_path))
                prompt = str(item["prompt"])
                ledger.append({
                    "source_split": split,
                    "source_ordinal": int(record["ordinal"]),
                    "record_id": record_id,
                    "selection_hash": str(record["selection_hash"]),
                    "role": role,
                    "eqkey": str(item["eqkey"]),
                    "target": str(item["target"]),
                    "image_path": image_path,
                    "raw_image_sha256": image_hashes[image_path],
                    "processed_pixel_tensor_sha256": str(item.get("processed_pixel_tensor_sha256", "")),
                    "question": prompt,
                    "question_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                    "source_hash": str(item.get("source_hash", "")),
                    "cache_file_path": str(cache_path.resolve()),
                    "cache_file_sha256": actual_cache_hash,
                    "tensor_hashes": record["tensor_hashes"],
                    "clean_generation": item.get("clean_generation"),
                })
    counts = Counter(row["source_split"] for row in cache_index)
    if counts != {"train": 512, "validation": 64, "heldout": 64}:
        raise RuntimeError(f"EQKEY_FAST_CONFIRMATION_ANCHOR_MISMATCH:cache_records:{dict(counts)}")
    if len(ledger) != 2560:
        raise RuntimeError(f"EQKEY_FAST_CONFIRMATION_ANCHOR_MISMATCH:positive_inputs:{len(ledger)}")
    return ledger, cache_index, manifest_rows


def write_family_outputs(out: Path, ledger: list[dict[str, Any]], families: list[dict[str, Any]]) -> None:
    audit = out / "eqkey_family_audit"
    audit.mkdir(parents=True, exist_ok=False)
    fields = [name for name in ledger[0] if name not in ("tensor_hashes", "clean_generation")]
    with (audit / "positive_input_ledger.csv").open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(ledger)
    occurrences: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in ledger:
        occurrences[row["eqkey"]].append(row)
    duplicate_rows = []
    for eqkey, rows in sorted(occurrences.items()):
        owners = sorted({(row["source_split"], row["record_id"]) for row in rows})
        if len(owners) < 2:
            continue
        duplicate_rows.append({
            "eqkey": eqkey, "owner_count": len(owners),
            "owners": json.dumps(owners),
            "splits": json.dumps(sorted({row["source_split"] for row in rows})),
            "roles": json.dumps(sorted({row["role"] for row in rows})),
            "normalized_targets": json.dumps(sorted({str(row["target"]).casefold().strip() for row in rows})),
        })
    with (audit / "eqkey_duplicate_groups.csv").open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(duplicate_rows[0]))
        writer.writeheader(); writer.writerows(duplicate_rows)
    write_jsonl(audit / "family_components.jsonl", families)
    conflicts = [row for row in families if row["target_conflict"]]
    write_json(audit / "target_conflicts.json", {"count": len(conflicts), "families": conflicts})
    seen = [row for row in families if row["touches_original_train"]]
    write_json(audit / "generator_seen_families.json", {
        "count": len(seen), "family_ids": [row["family_id"] for row in seen],
        "propagation_rule": "ANY_ORIGINAL_TRAIN_MEMBER_MAKES_COMPLETE_FAMILY_GENERATOR_SEEN",
    })
    component_sizes = Counter(row["component_size"] for row in families)
    (audit / "EQKEY_FAMILY_AUDIT_REPORT.md").write_text(
        "# EqKey Family Audit\n\n"
        f"- Positive inputs: **{len(ledger)}**\n"
        f"- Unique EqKeys: **{len(occurrences)}**\n"
        f"- Duplicate EqKey groups: **{len(duplicate_rows)}**\n"
        f"- Connected families: **{len(families)}**\n"
        f"- Generator-seen families: **{len(seen)}**\n"
        f"- Generator-unseen families: **{len(families)-len(seen)}**\n"
        f"- Target-conflict families: **{len(conflicts)}**\n"
        f"- Component-size distribution: `{dict(sorted(component_sizes.items()))}`\n"
        "- Record 953 used for fitting/selection: **No**\n"
        "- Sealed blind edited outputs opened: **No**\n"
    )


def compact_family(family: Mapping[str, Any]) -> dict[str, Any]:
    return dict(family)


def write_split_outputs(out: Path, allocation: Mapping[str, Any], families: list[dict[str, Any]]) -> None:
    directory = out / "purged_split"; directory.mkdir(parents=True, exist_ok=False)
    selected_ids = {family["family_id"] for family in allocation["validation"] + allocation["heldout"]}
    if len(selected_ids) != len(allocation["validation"]) + len(allocation["heldout"]):
        raise RuntimeError("EQKEY_FAST_CONFIRMATION_INVALID_ENGINEERING_RUN:family_cross_split")
    report = {key: value for key, value in allocation.items() if key not in ("validation", "heldout")}
    report.update({
        "total_family_count": len(families),
        "generator_seen_family_count": sum(row["touches_original_train"] for row in families),
        "generator_unseen_family_count": sum(not row["touches_original_train"] for row in families),
        "validation_family_count": len(allocation["validation"]),
        "heldout_family_count": len(allocation["heldout"]),
        "no_family_crosses_new_splits": True,
        "no_generator_seen_family_in_evaluation": all(not row["touches_original_train"]
            for row in allocation["validation"] + allocation["heldout"]),
    })
    write_json(directory / "eligibility_report.json", report)
    for split in ("validation", "heldout"):
        rows = allocation[split]
        write_json(directory / f"purged_{split}_manifest.json", {
            "protocol": PROTOCOL, "label": allocation["label"], "split": split,
            "family_count": len(rows), "role_counts": allocation[f"{split}_role_counts"],
            "family_ids": [row["family_id"] for row in rows],
            "families": [compact_family(row) for row in rows],
            "record953_used": False, "sealed_blind_edited_outputs_loaded": False,
        })
    write_json(directory / "family_level_future_split_proposal.json", future_family_split(families))
    (directory / "PURGED_SPLIT_REPORT.md").write_text(
        "# Purged Split Report\n\n"
        f"Primary data label: `{allocation['label']}`\n\n"
        f"- Eligible clean unseen families: **{allocation['eligible_family_count']}**\n"
        f"- Purged validation families: **{len(allocation['validation'])}**\n"
        f"- Purged validation roles: `{allocation['validation_role_counts']}`\n"
        f"- Purged held-out families: **{len(allocation['heldout'])}**\n"
        f"- Purged held-out roles: `{allocation['heldout_role_counts']}`\n"
        "- Generator-seen families in evaluation: **0**\n"
        "- Families crossing new splits: **0**\n"
        "- Record 953 used: **No**\n"
        "- Sealed blind edited outputs opened: **No**\n"
    )


def write_cache_index(out: Path, cache_rows: list[dict[str, Any]], families: list[dict[str, Any]],
                      allocation: Mapping[str, Any], manifest_rows: list[dict[str, Any]]) -> None:
    directory = out / "cache_reuse"; directory.mkdir(parents=True, exist_ok=False)
    family_split = {family["family_id"]: "clean_train" for family in families if family["touches_original_train"]}
    for split in ("validation", "heldout"):
        family_split.update({family["family_id"]: f"purged_{split}" for family in allocation[split]})
    indexed = []
    for family in families:
        if family["family_id"] not in family_split:
            continue
        for view in family["canonical_views"]:
            indexed.append({
                "protocol": PROTOCOL, "family_id": family["family_id"],
                "canonical_edit_id": family["canonical_record_id"], "source_edit_id": view["record_id"],
                "eqkey": view["eqkey"], "old_cache_path": view["cache_file_path"],
                "old_cache_sha256": view["cache_file_sha256"], "tensor_hashes": view["tensor_hashes"],
                "new_split": family_split[family["family_id"]], "role": view["role"],
                "read_only_reference": True,
            })
    write_json(directory / "old_cache_manifest.json", {
        "protocol": PROTOCOL, "source_manifests": manifest_rows,
        "cache_record_count": len(cache_rows), "cache_files_hash_verified": True,
        "mutated_or_copied": False,
    })
    write_jsonl(directory / "new_cache_index.jsonl", indexed)
    (directory / "CACHE_REUSE_REPORT.md").write_text(
        "# Cache Reuse Report\n\n"
        f"- Hash-verified cache records: **{len(cache_rows)}**\n"
        f"- Canonical read-only references: **{len(indexed)}**\n"
        "- Large tensors copied: **No**\n"
        "- Old cache modified: **No**\n"
        "- Deterministic cached-vs-online parity: **Pending model-stage audit**\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict-run", type=Path, required=True)
    parser.add_argument("--blocked-run", type=Path, required=True)
    parser.add_argument("--blind-selection-manifest", type=Path, required=True)
    parser.add_argument("--blind-sealed-manifest", type=Path, required=True)
    parser.add_argument("--record953-input-manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(args.out_dir)
    args.out_dir.mkdir(parents=True)

    blocked_failure = json.loads((args.blocked_run / "supervisor_failure.json").read_text())
    blocked_anchor = json.loads((args.blocked_run / "anchor_and_freeze_audit.json").read_text())
    focused = json.loads((args.blocked_run / "focused_test_report.json").read_text())
    blind_selection = json.loads(args.blind_selection_manifest.read_text())
    blind_sealed = json.loads(args.blind_sealed_manifest.read_text())
    record953 = json.loads(args.record953_input_manifest.read_text())
    checks = {
        "blocked_status": blocked_failure.get("status") == "ROUTER_ADAPTATION_INVALID_ENGINEERING_RUN",
        "blocked_eqkey_reason": "positive_eqkey_duplicate" in blocked_failure.get("error", ""),
        "blocked_focused_29_of_29": focused.get("passed") == 29 and focused.get("failed") == 0,
        "blocked_anchor_pass": blocked_anchor.get("status") == "PASS",
        "canonical_bank": bank_manifest().get("sha256") == EXPECTED_BANK_HASH,
        "blind_selection_internal_hash": blind_selection.get("selection_manifest_hash") == EXPECTED_BLIND_SELECTION_HASH,
        "blind_sealed_internal_hash": blind_sealed.get("manifest_hash") == EXPECTED_BLIND_SEALED_HASH,
        "blind_selection_file_hash": sha256_file(args.blind_selection_manifest) == "f947edf27b7fd38e5bb3c2837d8cb783a6b1fe614b75b2d39afba87b356670d2",
        "blind_sealed_file_hash": sha256_file(args.blind_sealed_manifest) == "26d850a935df5159df0c9a0d61be9289ee9c00815b36fd44ce81b242eb3624f7",
        "blind_edited_checkpoint_not_loaded": blind_selection.get("edited_checkpoint_loaded") is False,
    }
    if not all(checks.values()):
        raise RuntimeError(f"EQKEY_FAST_CONFIRMATION_ANCHOR_MISMATCH:{checks}")
    checkpoints = audit_checkpoints(args.strict_run)
    ledger, cache_rows, manifest_rows = read_cache(args.blocked_run)
    blind_record_ids = recursive_values(blind_selection.get("selected", []), {"record_id"})
    blind_eqkeys = recursive_values(blind_selection, {"router_input_equivalence_key", "equivalence_key"})
    record953_eqkeys = recursive_values(record953, {"router_input_equivalence_key", "equivalence_key"})
    families = build_families(ledger, blind_record_ids=blind_record_ids,
                              blind_eqkeys=blind_eqkeys, record953_eqkeys=record953_eqkeys)
    cache_by_owner = {(row["source_split"], row["record_id"]): row for row in cache_rows}
    for family in families:
        native_view = next(row for row in family["canonical_views"]
                           if row["eqkey"] == family["canonical_native_eqkey"])
        owner = cache_by_owner[(native_view["source_split"], native_view["record_id"])]
        family["canonical_locality"] = {
            item["category"]: item for item in owner["inputs"]
            if item["category"] in ("image_locality", "text_locality")
        }
    allocation = allocate_purged(families)

    anchor = {
        "protocol": PROTOCOL, "status": "PASS", "checks": checks,
        "strict_run": str(args.strict_run.resolve()), "blocked_run": str(args.blocked_run.resolve()),
        "strict_checkpoints": checkpoints, "canonical_bank_hash": EXPECTED_BANK_HASH,
        "blocked_failure_snapshot_sha256": sha256_file(args.blocked_run / "supervisor_failure.json"),
        "blocked_cache_manifests": manifest_rows, "blocked_cache_file_count": len(cache_rows),
        "record953_manifest_sha256": sha256_file(args.record953_input_manifest),
        "record953_edited_for_selection": False,
        "blind_selection_manifest_sha256": sha256_file(args.blind_selection_manifest),
        "blind_sealed_manifest_sha256": sha256_file(args.blind_sealed_manifest),
        "blind_outcomes_enumerated": False, "blind_edited_checkpoint_loaded": False,
    }
    write_json(args.out_dir / "anchor_and_immutability_audit.json", anchor)
    write_json(args.out_dir / "blind_set_seal_audit.json", {
        "protocol": PROTOCOL, "selection_manifest_hash": EXPECTED_BLIND_SELECTION_HASH,
        "sealed_manifest_hash": EXPECTED_BLIND_SEALED_HASH, "edited_checkpoint_loaded": False,
        "outcomes_enumerated": False, "passed": True,
    })
    write_jsonl(args.out_dir / "state_and_bank_hash_ledger.jsonl", [{
        "stage": "eqkey_family_audit", "canonical_bank_hash": EXPECTED_BANK_HASH,
        "canonical_bank_unchanged": True, "old_cache_mutated": False,
        "record953_loaded_for_model_evaluation": False, "sealed_blind_edited_checkpoint_loaded": False,
    }])
    write_family_outputs(args.out_dir, ledger, families)
    write_split_outputs(args.out_dir, allocation, families)
    write_cache_index(args.out_dir, cache_rows, families, allocation, manifest_rows)
    write_json(args.out_dir / "run_manifest.json", {
        "protocol": PROTOCOL, "status": "PURGED_SPLIT_COMPLETE__MODEL_EVALUATION_PENDING",
        "purged_split_label": allocation["label"], "record953_used": False,
        "blind_set_opened_with_edited_model": False, "blind_evaluation_permitted": False,
        "stage2_permitted": False,
    })
    print(json.dumps({
        "status": "EQKEY_FAMILY_AND_PURGED_SPLIT_COMPLETE", "family_count": len(families),
        "generator_seen_families": sum(row["touches_original_train"] for row in families),
        "eligible_clean_families": allocation["eligible_family_count"],
        "label": allocation["label"], "validation_families": len(allocation["validation"]),
        "heldout_families": len(allocation["heldout"]),
        "validation_roles": allocation["validation_role_counts"],
        "heldout_roles": allocation["heldout_role_counts"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
