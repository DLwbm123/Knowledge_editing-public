#!/usr/bin/env python3
"""Plan and cache EqKey-clean Router-R1 family hard negatives."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parents[2]
for item in (ROOT, ROOT / "scripts"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from methods.liveedit_med.eqkey_clean_fast import PROTOCOL as EQKEY_PROTOCOL
from methods.liveedit_med.llavamed_adapter import resolve_layer21_block
from methods.liveedit_med.router_r1 import EXPECTED_BANK_HASH, NEGATIVE_CATEGORIES
from methods.liveedit_med.serialization import tensor_hashes
from scripts.engram.run_engram_v2_stage0_generation_audit import bank_manifest
from scripts.engram.run_record953_routed_banked_lora_v1_1 import visible_input_audit
from scripts.engram.equivalence_aware_router_utils import router_input_equivalence_key
from scripts.liveedit_med.cache_router_r1 import capture, sha256_file, tensor_prefix
from scripts.liveedit_med.evaluate_eqkey_clean_checkpoint import sample
from scripts.liveedit_med.evaluate_eqkey_clean_safety import candidate_sample, native_view
from scripts.liveedit_med.evaluate_posthoc_validation_checkpoint import load_clean_model
from scripts.liveedit_med.evaluate_router_r1_checkpoint import variant


PROTOCOL = "LIVEEDIT_MED_EQKEY_CLEAN_ROUTER_R1_V1"


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode()).hexdigest()


def read_families(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def clean_train(families: list[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return sorted((row for row in families if row["touches_original_train"]
                   and not row["target_conflict"] and not row["touches_record953"]
                   and not row["touches_sealed_blind"]),
                  key=lambda row: (row["allocation_hash"], row["family_id"]))


def native_features(family: Mapping[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    view = native_view(family)
    tensors = load_file(view["cache_file_path"], device="cpu")
    cached = variant(tensors, view["role"], torch.device("cpu"))
    vision = F.normalize(cached["vision"].float().mean(1), dim=1)[0]
    question = F.normalize(cached["question"].float().mean(1), dim=1)[0]
    return vision, question


def nearest_order(target: str, families: list[Mapping[str, Any]],
                  features: Mapping[str, torch.Tensor]) -> list[Mapping[str, Any]]:
    current = features[target]
    return sorted((row for row in families if row["family_id"] != target),
                  key=lambda row: (-float(torch.dot(current, features[row["family_id"]]).item()),
                                   canonical_hash([target, row["family_id"]]), row["family_id"]))


def stable_order(target: str, families: list[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return sorted((row for row in families if row["family_id"] != target),
                  key=lambda row: (canonical_hash([target, row["family_id"]]), row["family_id"]))


@torch.inference_mode()
def plan(args: argparse.Namespace) -> None:
    if args.out.exists():
        raise FileExistsError(args.out)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.physical_gpu):
        raise RuntimeError("EQKEY_CLEAN_ROUTER_R1_GPU_VISIBILITY_MISMATCH")
    families = read_families(args.family_components)
    train = clean_train(families)
    validation_ids = set(json.loads(args.validation_manifest.read_text())["family_ids"])
    heldout_ids = set(json.loads(args.heldout_manifest.read_text())["family_ids"])
    train_ids = {row["family_id"] for row in train}
    if train_ids & validation_ids or train_ids & heldout_ids or validation_ids & heldout_ids:
        raise RuntimeError("EQKEY_CLEAN_ROUTER_R1_FAMILY_SPLIT_COLLISION")
    model, _bank = load_clean_model(args.physical_gpu)
    if bank_manifest()["sha256"] != EXPECTED_BANK_HASH:
        raise RuntimeError("EQKEY_CLEAN_ROUTER_R1_BANK_MISMATCH")
    features = {row["family_id"]: native_features(row) for row in train}
    visual = {key: value[0] for key, value in features.items()}
    text = {key: value[1] for key, value in features.items()}
    joint = {key: F.normalize(torch.cat(value), dim=0) for key, value in features.items()}
    positive_eqkeys = {eqkey for row in families for eqkey in row["positive_eqkeys"]}
    used: set[str] = set()
    image_audits: dict[str, dict[str, Any]] = {}

    def mixed_audit(current: Mapping[str, Any]) -> dict[str, Any]:
        image_path = str(current["image"])
        if image_path not in image_audits:
            image_audits[image_path] = visible_input_audit(model, {
                "question": current["prompt"], "image_path": image_path,
                "image_sha256": sha256_file(Path(image_path))})
        base = image_audits[image_path]
        question = str(current["prompt"])
        prompt = model._conversation_prompt(question, None)
        input_ids = model.tokenizer_image_token(prompt, model.llava_tokenizer,
            model.IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(model.lm_device)
        ids = [int(value) for value in input_ids[0].tolist()]
        mask = [1] * len(ids)
        result = dict(base)
        result.update({
            "router_input_equivalence_key": router_input_equivalence_key(
                base["processed_pixel_tensor_sha256"], base["image_sizes_equivalence_field"], ids, mask),
            "normalized_original_question": " ".join(question.casefold().split()),
            "question_token_ids": [int(value) for value in model.llava_tokenizer(
                question, add_special_tokens=False).input_ids],
            "rendered_canonical_routing_prompt": prompt,
            "routing_input_ids": ids, "attention_mask": mask,
        })
        return result

    rows = []
    for complete, family in enumerate(train, 1):
        family_id = family["family_id"]
        orders = {
            "same_image_different_question": stable_order(family_id, train),
            "same_question_different_image": stable_order(family_id, train),
            "visual_nearest": nearest_order(family_id, train, visual),
            "text_nearest": nearest_order(family_id, train, text),
            "joint_near_miss": nearest_order(family_id, train, joint),
        }
        negatives = []
        for category in NEGATIVE_CATEGORIES:
            selected = None
            candidates = [row for row in orders[category]
                          if row["canonical_target_normalized"] != family["canonical_target_normalized"]]
            for index, left in enumerate(candidates):
                right = candidates[(index + 1) % len(candidates)] if category == "joint_near_miss" else None
                current = candidate_sample(category, family, left, right)
                audit = mixed_audit(current)
                eqkey = audit["router_input_equivalence_key"]
                if eqkey not in positive_eqkeys and eqkey not in used:
                    selected = {"category": category, "eqkey": eqkey, "sample": current,
                        "source_family_ids": [left["family_id"]]
                            if right is None else [left["family_id"], right["family_id"]],
                        "visible_input_audit": audit}
                    used.add(eqkey)
                    break
            if selected is None:
                raise RuntimeError(f"EQKEY_CLEAN_ROUTER_R1_NEGATIVE_EXHAUSTED:{family_id}:{category}")
            negatives.append(selected)
        rows.append({"family_id": family_id, "canonical_record_id": family["canonical_record_id"],
                     "negatives": negatives})
        if complete % 20 == 0 or complete == len(train):
            print(json.dumps({"event": "eqkey_r1_negative_plan", "complete": complete,
                              "total": len(train)}), flush=True)
    payload = {"protocol": PROTOCOL, "eqkey_protocol": EQKEY_PROTOCOL,
        "family_count": len(train), "negative_count": len(used), "rows": rows,
        "positive_eqkey_count": len(positive_eqkeys), "positive_negative_disjoint": not bool(positive_eqkeys & used),
        "negative_negative_unique": len(used) == len(train) * len(NEGATIVE_CATEGORIES),
        "train_validation_family_disjoint": not bool(train_ids & validation_ids),
        "train_heldout_family_disjoint": not bool(train_ids & heldout_ids),
        "validation_heldout_family_disjoint": not bool(validation_ids & heldout_ids),
        "clean_s0_only": True, "record953_used": False, "sealed_blind_loaded": False,
        "canonical_bank_unchanged": bank_manifest()["sha256"] == EXPECTED_BANK_HASH}
    required = ("positive_negative_disjoint", "negative_negative_unique",
                "train_validation_family_disjoint", "train_heldout_family_disjoint",
                "validation_heldout_family_disjoint", "canonical_bank_unchanged")
    if not all(payload[key] for key in required):
        raise RuntimeError("EQKEY_CLEAN_ROUTER_R1_PLAN_AUDIT_FAILURE")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    if args.train_manifest_out is not None:
        if args.train_manifest_out.exists():
            raise FileExistsError(args.train_manifest_out)
        args.train_manifest_out.parent.mkdir(parents=True, exist_ok=True)
        args.train_manifest_out.write_text(json.dumps({
            "protocol": PROTOCOL, "eqkey_protocol": EQKEY_PROTOCOL,
            "family_count": len(train), "family_ids": [row["family_id"] for row in train],
            "families": train, "canonical_expert_identity_per_family": True,
            "positive_eqkeys_deduplicated": True, "target_conflicts_excluded": True,
            "record953_excluded": True, "sealed_blind_excluded": True,
        }, indent=2, sort_keys=True) + "\n")


@torch.inference_mode()
def worker(args: argparse.Namespace) -> None:
    if args.out.exists():
        raise FileExistsError(args.out)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.physical_gpu):
        raise RuntimeError("EQKEY_CLEAN_ROUTER_R1_GPU_VISIBILITY_MISMATCH")
    plan_data = json.loads(args.plan.read_text())
    model, _bank = load_clean_model(args.physical_gpu)
    if bank_manifest()["sha256"] != EXPECTED_BANK_HASH:
        raise RuntimeError("EQKEY_CLEAN_ROUTER_R1_BANK_MISMATCH")
    _name, block = resolve_layer21_block(model)
    args.out.mkdir(parents=True)
    rows = []
    assigned = [(index, row) for index, row in enumerate(plan_data["rows"])
                if index % args.worker_count == args.worker_index]
    for complete, (ordinal, row) in enumerate(assigned, 1):
        tensors: dict[str, torch.Tensor] = {}
        for negative in row["negatives"]:
            current = negative["sample"]
            image_hash = sha256_file(Path(current["image"]))
            audit = visible_input_audit(model, {"question": current["prompt"],
                "image_path": current["image"], "image_sha256": image_hash})
            if audit["router_input_equivalence_key"] != negative["eqkey"]:
                raise RuntimeError("EQKEY_CLEAN_ROUTER_R1_PLAN_CAPTURE_DRIFT")
            tensor_prefix(tensors, negative["category"], capture(model, block, current))
        path = args.out / f"family_{row['family_id']}.safetensors"
        save_file(tensors, str(path))
        loaded = load_file(str(path), device="cpu")
        if loaded.keys() != tensors.keys() or any(not torch.equal(loaded[key], tensors[key]) for key in tensors):
            raise RuntimeError("EQKEY_CLEAN_ROUTER_R1_CACHE_RELOAD_FAILURE")
        rows.append({"ordinal": ordinal, "family_id": row["family_id"], "file_path": str(path.resolve()),
                     "file_sha256": sha256_file(path), "tensor_hashes": tensor_hashes(tensors),
                     "negative_eqkeys": {item["category"]: item["eqkey"] for item in row["negatives"]}})
        if complete % 10 == 0 or complete == len(assigned):
            print(json.dumps({"event": "eqkey_r1_negative_cache", "worker": args.worker_index,
                              "complete": complete, "total": len(assigned)}), flush=True)
    (args.out / "manifest.json").write_text(json.dumps({"protocol": PROTOCOL,
        "worker_index": args.worker_index, "worker_count": args.worker_count,
        "count": len(rows), "records": rows, "clean_s0_only": True,
        "record953_used": False, "sealed_blind_loaded": False}, indent=2, sort_keys=True) + "\n")


def finalize(args: argparse.Namespace) -> None:
    if any((args.out_dir / name).exists() for name in
           ("hard_negative_ledger.csv", "full_eqkey_role_audit.json",
            "hard_negative_cache_manifest.json")):
        raise FileExistsError(args.out_dir)
    plan_data = json.loads(args.plan.read_text())
    cached = []
    for path in args.shard:
        value = json.loads(path.read_text())
        if value.get("protocol") != PROTOCOL:
            raise RuntimeError("EQKEY_CLEAN_ROUTER_R1_CACHE_PROTOCOL_MISMATCH")
        cached.extend(value["records"])
    cached.sort(key=lambda row: row["ordinal"])
    if len(cached) != plan_data["family_count"] or len({row["family_id"] for row in cached}) != len(cached):
        raise RuntimeError("EQKEY_CLEAN_ROUTER_R1_CACHE_COUNT_MISMATCH")
    by_family = {row["family_id"]: row for row in cached}
    ledger = []
    for row in plan_data["rows"]:
        cache = by_family[row["family_id"]]
        for negative in row["negatives"]:
            ledger.append({"family_id": row["family_id"], "category": negative["category"],
                "eqkey": negative["eqkey"], "source_family_ids": json.dumps(negative["source_family_ids"]),
                "file_path": cache["file_path"], "file_sha256": cache["file_sha256"]})
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "hard_negative_ledger.csv").open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ledger[0])); writer.writeheader(); writer.writerows(ledger)
    audit = {key: plan_data[key] for key in ("protocol", "family_count", "negative_count",
        "positive_eqkey_count", "positive_negative_disjoint", "negative_negative_unique",
        "train_validation_family_disjoint", "train_heldout_family_disjoint",
        "validation_heldout_family_disjoint", "record953_used", "sealed_blind_loaded",
        "canonical_bank_unchanged")}
    audit["cache_files_verified"] = all(sha256_file(Path(row["file_path"])) == row["file_sha256"] for row in cached)
    (args.out_dir / "full_eqkey_role_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    (args.out_dir / "hard_negative_cache_manifest.json").write_text(json.dumps({
        "protocol": PROTOCOL, "count": len(cached), "records": cached,
        "plan_sha256": sha256_file(args.plan), "audit_passed": all(value for key, value in audit.items()
            if key.endswith("disjoint") or key.endswith("unique") or key.endswith("unchanged")
            or key.endswith("verified")), "clean_s0_only": True,
    }, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(); sub = parser.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--family-components", type=Path, required=True)
    p.add_argument("--validation-manifest", type=Path, required=True)
    p.add_argument("--heldout-manifest", type=Path, required=True)
    p.add_argument("--physical-gpu", type=int, required=True); p.add_argument("--out", type=Path, required=True)
    p.add_argument("--train-manifest-out", type=Path)
    p = sub.add_parser("worker")
    p.add_argument("--plan", type=Path, required=True); p.add_argument("--physical-gpu", type=int, required=True)
    p.add_argument("--worker-index", type=int, required=True); p.add_argument("--worker-count", type=int, required=True)
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("finalize")
    p.add_argument("--plan", type=Path, required=True); p.add_argument("--shard", type=Path, action="append", required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    {"plan": plan, "worker": worker, "finalize": finalize}[args.mode](args)


if __name__ == "__main__":
    main()
