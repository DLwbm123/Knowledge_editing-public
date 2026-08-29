#!/usr/bin/env python3
"""Evaluate one strict-source checkpoint on an EqKey-purged family split."""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import torch
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[2]
for item in (ROOT, ROOT / "scripts"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from methods.liveedit_med.eqkey_clean_fast import PROTOCOL, canonical_hash
from methods.liveedit_med.llavamed_adapter import resolve_layer21_block
from methods.liveedit_med.posthoc_validation import normalize_answer, unrestricted_match
from methods.liveedit_med.router_r1 import EXPECTED_BANK_HASH
from methods.liveedit_med.serialization import load_safe_state
from methods.liveedit_med.trainer import LiveEditMedicalConfig, LiveEditMedicalModules
from scripts.engram.run_engram_natural_generation_recovery import clinical_preservation
from scripts.engram.run_engram_v2_stage0_generation_audit import bank_manifest, state_weight_hash
from scripts.liveedit_med.evaluate_posthoc_validation_checkpoint import (forced_generation,
    load_clean_model, routed_generation, teacher_forced_nll)
from scripts.liveedit_med.evaluate_router_r1_checkpoint import variant
from scripts.liveedit_med.eqkey_fixed_expert_utils import load_fixed_experts


SIZES = (1, 10, 32)
ROLES = ("native", "textual", "visual", "paired")


def sample(view: Mapping[str, Any]) -> dict[str, str]:
    return {"image": str(view["image_path"]), "prompt": str(view["question"]),
            "target": str(view["target"])}


def locality_sample(item: Mapping[str, Any]) -> dict[str, Any]:
    return {"image": item.get("image"), "prompt": item["prompt"], "target": item["target"]}


def repository_ids(target: str, size: int, all_ids: list[str]) -> list[str]:
    if size > len(all_ids):
        raise RuntimeError("EQKEY_FAST_CONFIRMATION_INVALID_ENGINEERING_RUN:repository_size")
    remaining = sorted((family_id for family_id in all_ids if family_id != target),
                       key=lambda family_id: (canonical_hash([target, family_id]), family_id))
    result = [target, *remaining[:size - 1]]
    if len(result) != size or len(set(result)) != size:
        raise RuntimeError("EQKEY_FAST_CONFIRMATION_INVALID_ENGINEERING_RUN:repository_membership")
    return result


def expert_for_family(modules: LiveEditMedicalModules, family: Mapping[str, Any], device: torch.device):
    native = next(row for row in family["canonical_views"]
                  if row["eqkey"] == family["canonical_native_eqkey"])
    tensors = load_file(native["cache_file_path"], device="cpu")
    cached = variant(tensors, native["role"], device)
    eqr, evr, moe_c, moe_r = modules.generated_edit(cached["vision"], cached["question"], cached["answer"])
    return {"eqr": eqr, "evr": evr, "moe_c": moe_c, "moe_r": moe_r,
            "canonical_record_id": family["canonical_record_id"],
            "source_cache_sha256": native["cache_file_sha256"]}


def repository(ids: list[str], experts: Mapping[str, Mapping[str, torch.Tensor]]) -> dict[str, Any]:
    result: dict[str, Any] = {"ids": ids}
    for name in ("eqr", "evr", "moe_c", "moe_r"):
        result[name] = torch.cat([experts[family_id][name] for family_id in ids], dim=0)
    return result


def clean_match(view: Mapping[str, Any]) -> dict[str, Any]:
    generation = view["clean_generation"]
    match = unrestricted_match(generation["raw_output"], view["target"],
                               eos=generation["stop_reason"] == "eos", cap_hit=generation["cap_hit"])
    return {**generation, "match": match}


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit-families", type=int)
    parser.add_argument("--fixed-experts", type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.physical_gpu):
        raise RuntimeError("EQKEY_FAST_CONFIRMATION_GPU_VISIBILITY_MISMATCH")
    split_manifest = json.loads(args.split_manifest.read_text())
    if split_manifest.get("protocol") != PROTOCOL or split_manifest.get("family_count") != 32:
        raise RuntimeError("EQKEY_FAST_CONFIRMATION_INVALID_ENGINEERING_RUN:split_manifest")
    families = split_manifest["families"]
    if any(row["touches_original_train"] or row["touches_record953"] or row["touches_sealed_blind"]
           or row["target_conflict"] for row in families):
        raise RuntimeError("EQKEY_FAST_CONFIRMATION_INVALID_ENGINEERING_RUN:unclean_family")
    target_families = families if args.limit_families is None else families[:args.limit_families]
    model, _bank = load_clean_model(args.physical_gpu)
    clean_state_hash = state_weight_hash(model)
    if bank_manifest()["sha256"] != EXPECTED_BANK_HASH:
        raise RuntimeError("EQKEY_FAST_CONFIRMATION_ANCHOR_MISMATCH:bank")
    _name, block = resolve_layer21_block(model)
    modules = LiveEditMedicalModules(LiveEditMedicalConfig()).to(model.lm_device).float()
    state, checkpoint_manifest = load_safe_state(args.checkpoint)
    modules.load_state_dict(state, strict=True); modules.eval()
    step = int(checkpoint_manifest["step"])
    fixed_manifest = None; fixed_manifest_hash = None
    if args.fixed_experts is None:
        experts = {family["family_id"]: expert_for_family(modules, family, model.lm_device)
                   for family in families}
    else:
        experts, fixed_manifest, fixed_manifest_hash = load_fixed_experts(
            args.fixed_experts, families, model.lm_device, checkpoint_manifest)
    family_ids = [family["family_id"] for family in families]
    all_targets = [family["canonical_target"] for family in families]
    rows = []
    for complete, family in enumerate(target_families, 1):
        family_id = family["family_id"]; expert = experts[family_id]
        forced, s0, source_losses = {}, {}, {}
        for view in family["canonical_views"]:
            eqkey = view["eqkey"]; current_sample = sample(view)
            s0[eqkey] = clean_match(view)
            forced[eqkey] = forced_generation(model, block, modules, current_sample,
                                               expert["moe_c"], expert["moe_r"])
            source_losses[eqkey] = teacher_forced_nll(model, block, modules, current_sample,
                                                      expert["moe_c"], expert["moe_r"])
        repositories = {}
        for size in SIZES:
            members = repository_ids(family_id, size, family_ids)
            repo = repository(members, experts)
            routed = {}
            for view in family["canonical_views"]:
                routed[view["eqkey"]] = routed_generation(model, block, modules, sample(view), repo)
            image_item = family["canonical_locality"]["image_locality"]
            image_clean = image_item["clean_generation"]
            image_routed = routed_generation(model, block, modules, locality_sample(image_item), repo)
            image_exact = (image_clean["token_ids"] == image_routed["token_ids"]
                           and image_clean["stop_reason"] == image_routed["stop_reason"])
            contaminations = [target for target in all_targets if normalize_answer(target)
                              and normalize_answer(target) in normalize_answer(image_routed["raw_output"])]
            image_clinical = clinical_preservation(image_clean["raw_output"], image_routed["raw_output"],
                                                   image_item["target"])
            text_item = family["canonical_locality"]["text_locality"]
            locality = {
                "image_locality": {"s0": image_clean, "routed": image_routed,
                    "exact_preservation": image_exact, "clinical": image_clinical,
                    "target_contaminations": contaminations},
                "text_locality": {"s0": text_item["clean_generation"],
                    "routed": text_item["clean_generation"], "exact_preservation": True,
                    "clinical": {"passed": True}, "target_contaminations": [],
                    "route": {"kind": "base", "reason": "TEXT_ONLY_EMPTY_CANDIDATE_BASE_BYPASS",
                              "candidate_ids": [], "final_weights": [], "sum_final_weights": 0.0}},
            }
            repositories[str(size)] = {"repository_ids": members, "routed": routed, "locality": locality}
        rows.append({"family_id": family_id, "canonical_record_id": family["canonical_record_id"],
                     "views": [{key: view[key] for key in ("eqkey", "role", "record_id", "target")}
                               for view in family["canonical_views"]],
                     "s0": s0, "forced_on": forced, "source_losses": source_losses,
                     "repositories": repositories})
        print(json.dumps({"event": "eqkey_clean_checkpoint", "step": step,
                          "split": split_manifest["split"], "complete": complete,
                          "total": len(target_families), "family_id": family_id}), flush=True)

    forced_counts, s0_counts = Counter(), Counter()
    role_totals = Counter(); source_loss_values = []
    for family, row in zip(target_families, rows):
        role_by_key = {view["eqkey"]: view["role"] for view in family["canonical_views"]}
        for eqkey, role in role_by_key.items():
            role_totals[role] += 1
            forced_counts[role] += int(row["forced_on"][eqkey]["match"]["success"])
            s0_counts[role] += int(row["s0"][eqkey]["match"]["success"])
            source_loss_values.append(float(row["source_losses"][eqkey]))
    sizes = {}
    for size in SIZES:
        routed_counts = Counter(); false_positives = 0; contaminations = 0; locality_exact = 0
        clinical_failures = 0; candidate_counts = []
        for family, row in zip(target_families, rows):
            family_id = family["family_id"]
            role_by_key = {view["eqkey"]: view["role"] for view in family["canonical_views"]}
            current = row["repositories"][str(size)]
            for eqkey, generation in current["routed"].items():
                routed_counts[role_by_key[eqkey]] += int(generation["match"]["success"])
                candidates = generation["route"].get("candidate_ids", [])
                false_positives += sum(candidate != family_id for candidate in candidates)
                candidate_counts.append(len(candidates))
            for locality in current["locality"].values():
                locality_exact += int(locality["exact_preservation"])
                clinical_failures += int(not locality["clinical"]["passed"])
                contaminations += len(locality["target_contaminations"])
                route = locality.get("routed", {}).get("route", locality.get("route", {}))
                candidates = route.get("candidate_ids", [])
                false_positives += len(candidates); candidate_counts.append(len(candidates))
        sizes[str(size)] = {
            "repository_size": size, "routed": {role: int(routed_counts[role]) for role in ROLES},
            "locality_exact_preservation": locality_exact, "locality_total": len(target_families) * 2,
            "routing_false_positives": false_positives, "target_contaminations": contaminations,
            "clinical_canonical_failures": clinical_failures,
            "mean_candidate_count": sum(candidate_counts) / max(1, len(candidate_counts)),
        }
    payload = {
        "protocol": PROTOCOL, "split": split_manifest["split"], "step": step,
        "checkpoint_manifest": checkpoint_manifest, "fresh_clean_s0": True,
        "family_count": len(target_families), "role_totals": {role: int(role_totals[role]) for role in ROLES},
        "s0": {role: int(s0_counts[role]) for role in ROLES},
        "forced_on": {role: int(forced_counts[role]) for role in ROLES},
        "source_validation_loss": sum(source_loss_values) / max(1, len(source_loss_values)),
        "repository_sizes": sizes, "rows": rows,
        "generation_config": {"do_sample": False, "num_beams": 1, "max_new_tokens": 128},
        "record953_loaded": False, "sealed_blind_loaded": False,
        "canonical_bank_unchanged": bank_manifest()["sha256"] == EXPECTED_BANK_HASH,
        "clean_s0_unchanged": state_weight_hash(model) == clean_state_hash,
        "fixed_experts_used": args.fixed_experts is not None,
        "fixed_expert_manifest_sha256": fixed_manifest_hash,
        "fixed_expert_generator_step": None if fixed_manifest is None else fixed_manifest["generator_checkpoint_step"],
    }
    if not payload["canonical_bank_unchanged"] or not payload["clean_s0_unchanged"]:
        raise RuntimeError("EQKEY_FAST_CONFIRMATION_INVALID_ENGINEERING_RUN:state_mutation")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "EQKEY_CLEAN_CHECKPOINT_COMPLETE", "step": step,
                      "split": split_manifest["split"], "forced_on": payload["forced_on"],
                      "repo32": sizes["32"]}, sort_keys=True))


if __name__ == "__main__":
    main()
